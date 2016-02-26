package nodes.learning

import nodes.util.VectorSplitter
import workflow.LabelEstimator

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.stats._

import org.apache.spark.rdd.RDD
import org.apache.spark.HashPartitioner

import edu.berkeley.cs.amplab.mlmatrix.util.{Utils => MLMatrixUtils}

import nodes.stats.StandardScaler
import pipelines.Logging
import utils.{MatrixUtils, Stats}

/**
 * Train a block-coordinate descent model using least squares
 *
 * @param blockSize size of blocks to use
 * @param numIter number of passes of co-ordinate descent to run
 * @param lambda regularization parameter
 */

object BinaryBlockCoordinateDescent extends Logging {

  // Use BCD to solve W = (X.t * * X) + \lambda) \ (X.t *  Y)
  def trainSingleClassLS(
      blockSize: Int,
      numIter: Int,
      lambda: Double,
      numFeatures: Int,
      trainingFeatureBlocks: Seq[RDD[DenseVector[Double]]],
      labels: RDD[Double]): Seq[DenseVector[Double]] = {

    val labelsMat = labels.mapPartitions { iter =>
      Iterator.single(DenseVector(iter.toArray))
    }

    var residual = labelsMat.map { l =>
      DenseVector.zeros[Double](l.size)
    }.cache()

    // Initialize model to blockSize. This will be resized if its different
    // inside the solve loop
    val numBlocks = math.ceil(numFeatures / blockSize).toInt
    val model = (0 until numBlocks).map { block =>
      DenseVector.zeros[Double](blockSize)
    }.toArray

    val treeBranchingFactor = labels.context.getConf.getInt(
      "spark.mlmatrix.treeBranchingFactor", 8).toInt
    val depth = math.max(math.ceil(math.log(labelsMat.partitions.size) /
        math.log(treeBranchingFactor)).toInt, 1)
    val aTaCache = new Array[DenseMatrix[Double]](numBlocks)
    val blockShuffler = new Random(1235231L)
    val nTrain = labels.count

    (0 until numIter).foreach { pass =>
      val inOrder = (0 until numBlocks).toIndexedSeq
      val blockOrder = blockShuffler.shuffle(inOrder)

      blockOrder.foreach { block =>
        val aPart = trainingFeatureBlocks(block)
        val aPartMat = aPart.mapPartitions { iter =>
          Iterator.single(MatrixUtils.rowsToMatrix(iter))
        }.cache()
        // Compute X.t * X if this the first iteration
        // Note that wOld is always zero in first pass, so no need to
        // subtract X * wOld here
        val modelBC = labels.context.broadcast(model(block))
        val (aTaBlock, aTbBlock) = if (pass == 0) {
          val covarsComputed = MLMatrixUtils.treeReduce(
            aPartMat.zip(labelsMat.zip(residual)).map { x =>
              (x._1.t * x._1, x._1.t * (x._2._1 - x._2._2))
            }, addMatrixVecPair, depth=depth)
          aTaCache(block) = covarsComputed._1
          model(block) = DenseVector.zeros[Double](covarsComputed._1.rows)
          covarsComputed
        } else {
          val aTbBlockComputed = MLMatrixUtils.treeReduce(
            aPartMat.zip(labelsMat.zip(residual)).map { x =>
              val featPart = x._1
              val labelPart = x._2._1
              val resPart = x._2._2
           
              // Remove (X * wOld) from the residual
              // Then compute (Y - residual)
              // This overall becomes Y - residual + X * wOld
              val resUpdated = featPart * modelBC.value
              resUpdated -= resPart
              resUpdated += labelPart
           
              // Compute X.t * (Y - residual)
              val aTbPart = featPart.t * (resUpdated)
              aTbPart 
            }, (a: DenseVector[Double], b: DenseVector[Double]) => a += b, depth = depth)
          (aTaCache(block), aTbBlockComputed)
        }

        val newModel = ((aTaBlock :/ nTrain.toDouble) + (DenseMatrix.eye[Double](aTaBlock.rows) * lambda)) \ (aTbBlock :/ nTrain.toDouble)
        val newModelBC = labels.context.broadcast(newModel)

        model(block) = newModel

        // Update the residual by adding (X * wNew) and subtracting (X * wOld)
        val newResidual = aPartMat.zip(residual).map { part =>
          val diffModel = if (pass == 0) {
            newModelBC.value
          } else {
            newModelBC.value - modelBC.value
          }
          part._2 += (part._1 * diffModel)
          part._2
        }.cache().setName("residual")
        newResidual.count
        residual.unpersist()
        residual = newResidual

        aPartMat.unpersist()
      }
    }
    model
  }

  def addMatrixVecPair(
    a: (DenseMatrix[Double], DenseVector[Double]),
    b: (DenseMatrix[Double], DenseVector[Double])) = {

    a._1 += b._1
    a._2 += b._2
    a
  }
}
