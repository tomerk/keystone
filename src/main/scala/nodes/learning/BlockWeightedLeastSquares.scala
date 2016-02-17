package nodes.learning

import nodes.util.VectorSplitter
import workflow.{WeightedNode, LabelEstimator}

import scala.collection.mutable.ArrayBuffer

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.stats._

import org.apache.spark.rdd.RDD
import org.apache.spark.HashPartitioner

import edu.berkeley.cs.amplab.mlmatrix.{RowPartition, NormalEquations, BlockCoordinateDescent, RowPartitionedMatrix}
import edu.berkeley.cs.amplab.mlmatrix.util.{Utils => MLMatrixUtils}

import nodes.stats.StandardScaler
import pipelines.Logging
import utils.{MatrixUtils, Stats}

// Utility class that holds statistics related to each block we solve
// Used to cache information across many passes
case class BlockStatistics(popCov: DenseMatrix[Double], popMean: DenseVector[Double],
  jointMean: DenseMatrix[Double], jointMeanRDD: RDD[DenseVector[Double]])

/**
 * Train a weighted block-coordinate descent model using least squares
 *
 * @param blockSize size of blocks to use
 * @param numIter number of passes of co-ordinate descent to run
 * @param lambda regularization parameter
 * @param mixtureWeight how much should positive samples be weighted
 */
class BlockWeightedLeastSquaresEstimator(
    blockSize: Int,
    numIter: Int,
    lambda: Double,
    mixtureWeight: Double,
    numFeaturesOpt: Option[Int] = None)
  extends LabelEstimator[DenseVector[Double], DenseVector[Double], DenseVector[Double]] with WeightedNode {

  override val weight = (3*numIter)+1

  /**
   * Fit a weighted least squares model using blocks of features provided.
   * 
   * NOTE: This function makes multiple passes over the training data. Caching
   *
   * @trainingFeatures and @trainingLabels before calling this function is recommended.
   * @param trainingFeatures Blocks of training data RDDs
   * @param trainingLabels training labels RDD
   * @returns A BlockLinearMapper that contains the model, intercept
   */
  def fit(
      trainingFeatures: Seq[RDD[DenseVector[Double]]],
      trainingLabels: RDD[DenseVector[Double]]): BlockLinearMapper[DenseVector[Double]] = {
    BlockWeightedLeastSquaresEstimator.trainWithL2(
      trainingFeatures,
      trainingLabels,
      blockSize,
      numIter,
      lambda,
      mixtureWeight)
  }

  /**
   * Split features into appropriate blocks and fit a weighted least squares model.
   *
   * NOTE: This function makes multiple passes over the training data. Caching
   *
   * @trainingFeatures and @trainingLabels before calling this function is recommended.
   * @param trainingFeatures training data RDD
   * @param trainingLabels training labels RDD
   * @returns A BlockLinearMapper that contains the model, intercept
   */
  override def fit(
      trainingFeatures: RDD[DenseVector[Double]],
      trainingLabels: RDD[DenseVector[Double]]): BlockLinearMapper[DenseVector[Double]] = {
    val trainingFeaturesSplit = new VectorSplitter(blockSize, numFeaturesOpt).apply(trainingFeatures)
    fit(trainingFeaturesSplit, trainingLabels)
  }
}

object BlockWeightedLeastSquaresEstimator extends Logging {

  /**
   * Returns a weighted block-coordinate descent model using least squares
   * NOTE: This function assumes that the trainingFeatures have been partitioned by
   * their class index. i.e. each partition of training data contains data for a single class
   *
   * NOTE: This function makes multiple passes over the training data.
   * Caching @trainingFeatures and @trainingLabels before calling this function is recommended.
   * 
   * @param trainingFeatures Blocks of training data RDDs
   * @param trainingLabels training labels RDD
   * @param lambda regularization parameter
   * @param mixtureWeight how much should positive samples be weighted
   * @param numIter number of passes of co-ordinate descent to run
   */
  def trainWithL2(
      trainingFeatures: Seq[RDD[DenseVector[Double]]],
      trainingLabels: RDD[DenseVector[Double]],
      blockSize: Int,
      numIter: Int,
      lambda: Double,
      mixtureWeight: Double): BlockLinearMapper[DenseVector[Double]] = {
    val sc = trainingFeatures.head.context

    val reshuffleData = {
      // Check if all examples in a partition are of the same class
      val sameClasses = trainingLabels.mapPartitions { iter =>
        Iterator.single(iter.map(label => label.toArray.indexOf(label.max)).toSeq.distinct.length == 1)
      }.collect()
      if (sameClasses.forall(x => x)) {
        val localClassIdxs = trainingLabels.mapPartitions { iter =>
          Iterator.single(iter.map(label => label.toArray.indexOf(label.max)).toSeq.distinct.head)
        }.collect()
        localClassIdxs.distinct.size != localClassIdxs.size
      } else {
        true
      }
    }

    val (features, labels) = if (reshuffleData) {
      logWarning("Partitions do not contain elements of the same class. Re-shuffling")
      groupByClasses(trainingFeatures, trainingLabels)
    } else {
      (trainingFeatures, trainingLabels)
    }

    val classIdxs = labels.mapPartitions { iter =>
      if (iter.hasNext) {
        Iterator.single(iter.map(label => label.toArray.indexOf(label.max)).toSeq.distinct.head)
      } else {
        Iterator.empty
      }
    }.cache().setName("classIdxs")

    val nTrain = labels.count
    val nClasses = labels.first.length.toInt

    val labelsMat = labels.mapPartitions { part =>
      MatrixUtils.rowsToMatrixIter(part)
    }

    val jointLabelMeanData = labelsMat.zip(classIdxs).map { x =>
      (x._2, 2*mixtureWeight + (2*(1.0-mixtureWeight) * x._1.rows/nTrain.toDouble) - 1)
    }.collect()

    // The jointLabelMean should be nClasses * 1
    val jointLabelMean = DenseVector.zeros[Double](nClasses)
    jointLabelMeanData.foreach { x =>
      jointLabelMean(x._1) = x._2
    }

    // Initialize models to zero here. Each model is a (W, b)
    val models = trainingFeatures.map { block =>
      // NOTE: We initialize this to blockSize but we adjust this in the first epoch of the solve.
      DenseMatrix.zeros[Double](blockSize, nClasses)
    }.toArray

    val numBlocks = models.length

    // Initialize residual to labels - jointLabelMean
    var residual = labelsMat.map { mat =>
      mat(*, ::) :- jointLabelMean
    }.cache().setName("residual")

    var residualMean = MatrixUtils.computeMean(residual)

    @transient val blockStats: Array[Option[BlockStatistics]] = (0 until numBlocks).map { blk =>
      None
    }.toArray

    (0 until numIter).foreach { pass =>
      var blockIdx = 0
       // TODO: Figure out if this should be shuffled ? rnd.shuffle((0 until numBlocks).toList)
      val randomBlocks = (0 until numBlocks).toList
      while (blockIdx < numBlocks) {
        val block = randomBlocks(blockIdx)
        logInfo(s"Running pass $pass block $block")
        val blockFeatures = features(block)

        val blockFeaturesMat = blockFeatures.mapPartitions { part =>
          MatrixUtils.rowsToMatrixIter(part)
        }.cache().setName("blockFeaturesMat")

        val treeBranchingFactor = sc.getConf.getInt("spark.mlmatrix.treeBranchingFactor", 2).toInt
        val depth = math.max(math.ceil(math.log(blockFeaturesMat.partitions.size) /
          math.log(treeBranchingFactor)).toInt, 1)

        val (popCov, popXTR, jointMeansRDD, popMean) = if (pass == 0) {
          // Step 1: Calculate population mean, covariance
          // TODO: This expects blockFeatures to be cached as this does one pass ??
          val blockPopMean = new StandardScaler(normalizeStdDev=false).fit(blockFeatures).mean

          // This is numClasses x blockSize -- So keep a RDD version of it that we can zip with each
          // partition and also a local version of it.
          val blockJointMeansRDD = blockFeaturesMat.map { mat =>
            mean(mat(::, *)).toDenseVector * mixtureWeight + blockPopMean * (1.0 -
              mixtureWeight)
          }.cache().setName("jointMeans")
          val blockJointMeans = DenseMatrix.zeros[Double](nClasses, blockPopMean.length)
          val blockJointMeansData = blockJointMeansRDD.zip(classIdxs).collect()
          blockJointMeansData.foreach { x =>
            blockJointMeans(x._2, ::) := x._1.t
          }

          val aTaResidual = MLMatrixUtils.treeReduce(blockFeaturesMat.zip(residual).map { part =>
            (part._1.t * part._1, part._1.t * part._2)
          }, addPairMatrices, depth=depth)

          val blockPopCov = (aTaResidual._1 / nTrain.toDouble) - (blockPopMean * blockPopMean.t)

          blockStats(block) =
            Some(BlockStatistics(blockPopCov, blockPopMean, blockJointMeans, blockJointMeansRDD))

          (blockPopCov, aTaResidual._2 / (nTrain.toDouble), blockJointMeansRDD, blockPopMean)
        } else {
          val aTResidual = MLMatrixUtils.treeReduce(blockFeaturesMat.zip(residual).map { part =>
            part._1.t * part._2
          }, (a: DenseMatrix[Double], b: DenseMatrix[Double]) => a += b, depth=depth)

          val blockStat = blockStats(block).get 
          (blockStat.popCov, aTResidual / (nTrain.toDouble), blockStat.jointMeanRDD,
            blockStat.popMean)
        }

        if (pass == 0) {
          models(block) = DenseMatrix.zeros[Double](popCov.rows, nClasses)
        }

        val popCovBC = sc.broadcast(popCov)
        val popMeanBC = sc.broadcast(popMean)
        val popXTRBC = sc.broadcast(popXTR)
        val modelBC = sc.broadcast(models(block))

        val modelsThisPass = blockFeaturesMat.zip(residual.zip(jointMeansRDD.zip(classIdxs))).map {
            case (featuresLocal, secondPart) =>
          val (classJointMean, classIdx) = secondPart._2
          val resLocal = secondPart._1(::, classIdx)
          // compute the number of examples in this class
          val numPosEx = featuresLocal.rows
          // compute the mean and covariance of the features in this class
          val classMean = mean(featuresLocal(::, *)).toDenseVector
          val classFeatures_ZM = featuresLocal(*, ::) :- classMean
          val classCov = (classFeatures_ZM.t * classFeatures_ZM) /= numPosEx.toDouble
          val classXTR = (featuresLocal.t * resLocal) /= numPosEx.toDouble
       
          val popCovMat = popCovBC.value
          val popMeanVec = popMeanBC.value
          val popXTRMat = popXTRBC.value

          val meanDiff = classMean - popMeanVec

          val jointXTX = popCovMat * (1.0 - mixtureWeight) +
            classCov * mixtureWeight +
            meanDiff * meanDiff.t * (1.0 - mixtureWeight) * mixtureWeight

          val meanMixtureWt: Double = (residualMean(classIdx) * (1.0 - mixtureWeight) +
            mixtureWeight * mean(resLocal))

          val jointXTR = popXTRMat(::, classIdx) * (1.0 - mixtureWeight) +
            classXTR.toDenseVector * mixtureWeight -
            classJointMean * meanMixtureWt

          val numDims = jointXTX.cols

          val W = (jointXTX + (DenseMatrix.eye[Double](numDims) :* lambda) ) \ (jointXTR -
            modelBC.value(::, classIdx) * lambda)

          (classIdx, W.toDenseMatrix.t)
        }.collect()

        val localFullModel = DenseMatrix.zeros[Double](models(block).rows, models(block).cols)
        modelsThisPass.foreach { m =>
          localFullModel(::, m._1) := m._2.toDenseVector
        }

        // So newXj is oldXj + localFullModel
        models(block) += localFullModel
        val localFullModelBC = sc.broadcast(localFullModel)

        val newResidual = blockFeaturesMat.zip(residual).map { part =>
          part._2 -= part._1 * localFullModelBC.value 
          part._2
        }.cache().setName("residual")

        newResidual.count
        residual.unpersist()
        residual = newResidual

        residualMean = MatrixUtils.computeMean(residual)

        popCovBC.unpersist()
        popMeanBC.unpersist()
        popXTRBC.unpersist()
        modelBC.unpersist()
        localFullModelBC.unpersist()

        blockFeaturesMat.unpersist()

        sc.parallelize(0 until sc.getExecutorMemoryStatus.size,
          sc.getExecutorMemoryStatus.size).foreach { x =>
          System.gc()
        }

        blockIdx = blockIdx + 1
      }
    }

    // Takes the local models stacks them vertically to get a full model
    val finalFullModel = DenseMatrix.vertcat(models:_*)
    val jointMeansCombined = DenseMatrix.horzcat(blockStats.map(_.get.jointMean):_*)

    val finalB = jointLabelMean - sum(jointMeansCombined.t :* finalFullModel, Axis._0).toDenseVector
    new BlockLinearMapper(models, blockSize, Some(finalB))
  }

  def addPairMatrices(
      a: (DenseMatrix[Double], DenseMatrix[Double]),
      b: (DenseMatrix[Double], DenseMatrix[Double]))
    : (DenseMatrix[Double], DenseMatrix[Double]) = {

    a._1 += b._1
    a._2 += b._2
    a
  }

  def groupByClasses(
      features: Seq[RDD[DenseVector[Double]]],
      labels: RDD[DenseVector[Double]])
    : (Seq[RDD[DenseVector[Double]]], RDD[DenseVector[Double]]) = {

    val nClasses = labels.first.length.toInt

    // NOTE(shivaram): We use two facts here
    // a. That the hashCode of an integer is the value itself
    // b. That the HashPartitioner in Spark works by computing (k mod nClasses)
    // This ensures that we get a single class per partition.
    val hp = new HashPartitioner(nClasses)
    val n = labels.partitions.length.toLong

    // Associate a unique id with each item
    // as Spark does not gaurantee two groupBy's will come out in the same order.
    val shuffledLabels = labels.mapPartitionsWithIndex { case (k, iter) =>
      iter.zipWithIndex.map { case (item, i) =>
        val classIdx = argmax(item)
        (classIdx, (i * n + k, item))
      }
    }.partitionBy(hp).values.mapPartitions { part =>
      part.toArray.sortBy(_._1).map(_._2).iterator
    }

    val shuffledFeatures = features.map { featureRDD =>
      featureRDD.zip(labels).mapPartitionsWithIndex { case (k, iter) =>
        iter.zipWithIndex.map { case (item, i) =>
          val classIdx = argmax(item._2)
          (classIdx, (i * n + k, item._1))
        }
      }.partitionBy(hp).values.mapPartitions { part =>
        part.toArray.sortBy(_._1).map(_._2).iterator
      }
    }

    (shuffledFeatures, shuffledLabels)
  }

}
