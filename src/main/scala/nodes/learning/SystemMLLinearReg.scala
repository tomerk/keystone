package nodes.learning

import breeze.linalg.{*, DenseVector, DenseMatrix, Vector}
import nodes.stats.StandardScalerModel
import org.apache.spark.api.java.{JavaPairRDD, JavaSparkContext}
import org.apache.spark.rdd.RDD
import org.apache.sysml.api.MLContext
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.matrix.data.{MatrixCell, MatrixIndexes}
import utils.MatrixUtils
import workflow.{Transformer, LabelEstimator}

/**
 * Created by tomerk11 on 2/22/16.
 */
class SystemMLLinearReg[T <: Vector[Double]](scriptLocation: String, numFeatures: Int, numIters: Double, blockSize: Int = 1024) extends LabelEstimator[T, Boolean, Boolean] {
  /**
   * A LabelEstimator estimator is an estimator which expects labeled data.
   *
   * @param data   Input data.
   * @param labels Input labels.
   * @return A [[Transformer]] which can be called on new data.
   */
  override def fit(data: RDD[T], labels: RDD[Boolean]): BooleanLinearMapper[T] = {
    val featuresToMatrixCell = data.zipWithIndex().flatMap {
      x => x._1.activeIterator.map {
        case (col, value) => (new MatrixIndexes(x._2 + 1, col + 1), new MatrixCell(value))
      }
    }

    val labelsToMatrixCell = labels.map(i => if (i) 1 else -1).zipWithIndex().map {
      x => (new MatrixIndexes(x._2 + 1, 1), new MatrixCell(x._1))
    }

    val sc = data.sparkContext
    val ml = new MLContext(sc)

    val numRows = data.count()
    val numCols = numFeatures

    val mc = new MatrixCharacteristics(numRows, numCols, blockSize, blockSize)
    val labelsMC = new MatrixCharacteristics(numRows, 1, blockSize, 1)

    val featuresMatrix = RDDConverterUtils.binaryCellToBinaryBlock(
      new JavaSparkContext(sc),
      new JavaPairRDD(featuresToMatrixCell),
      mc,
      false).cache()

    val labelsMatrix = RDDConverterUtils.binaryCellToBinaryBlock(
      new JavaSparkContext(sc),
      new JavaPairRDD(labelsToMatrixCell),
      labelsMC,
      false).cache()

    ml.reset()
    ml.setConfig("defaultblocksize", s"$blockSize")
    ml.registerInput("X", featuresMatrix, mc)
    ml.registerInput("y", labelsMatrix, labelsMC)
    ml.registerOutput("beta_out")

    val nargs = Map(
      "X" -> " ",
      "Y" -> " ",
      "B" -> " ",
      "reg" -> "0",
      "tol" -> "0",
      "icpt" -> "0",
      "maxi" -> s"$numIters")
    val outputBlocks = ml.execute(scriptLocation, nargs).getBinaryBlockedRDD("beta_out").rdd.collect()

    val maxR = {
      val maxRBlock = outputBlocks.maxBy(_._1.getRowIndex)
      (maxRBlock._1.getRowIndex - 1) * blockSize + maxRBlock._2.getMaxRow
    }.toInt

    val maxC = {
      val maxCBlock = outputBlocks.maxBy(_._1.getColumnIndex)
      (maxCBlock._1.getColumnIndex - 1) * blockSize + maxCBlock._2.getMaxColumn
    }.toInt

    val matOut = DenseMatrix.zeros[Double](maxR, maxC)
    outputBlocks.foreach {
      case (mIndex, mBlock) =>
        val rInit = (mIndex.getRowIndex.toInt - 1) * blockSize
        val cInit = (mIndex.getColumnIndex.toInt - 1) * blockSize

        for (i <- 0 until mBlock.getMaxRow) {
          for (j <- 0 until mBlock.getMaxColumn) {
            val r = rInit + i
            val c = cInit + j
            matOut(r, c) = mBlock.getValue(i, j)
          }
        }
    }

    BooleanLinearMapper(matOut)
  }
}

/**
 * Computes A * x + b i.e. a linear map of data using a trained model.
 *
 * @param x trained model
 * @param bOpt optional intercept to add
 * @param featureScaler optional scaler to apply to data before applying the model
 */
case class BooleanLinearMapper[T <: Vector[Double]](
                                              x: DenseMatrix[Double],
                                              bOpt: Option[DenseVector[Double]] = None,
                                              featureScaler: Option[StandardScalerModel] = None)
  extends Transformer[T, Boolean] {

  /**
   * Apply a linear model to an input.
   *
   * @param in Input.
   * @return Output.
   */
  def apply(in: T): Boolean = {
    val scaled = featureScaler.map(_.apply(in match {
      case dense: DenseVector[Double] => dense
      case _ => in.toDenseVector
    })).getOrElse(in)

    val out = x.t * scaled
    val weights = bOpt.map { b =>
      out :+= b
    }.getOrElse(out)

    weights(0) > 0
  }

  /**
   * Apply a linear model to a collection of inputs.
   *
   * @param in Collection of A's.
   * @return Collection of B's.
   */
  override def apply(in: RDD[T]): RDD[Boolean] = {
    val modelBroadcast = in.context.broadcast(x)
    val bBroadcast = in.context.broadcast(bOpt)

    val inScaled = featureScaler.map(_.apply(in.map {
      case dense: DenseVector[Double] => dense
      case notDense => notDense.toDenseVector
    })).getOrElse(in)

    inScaled.mapPartitions(rows => {
      val mat = MatrixUtils.rowsToMatrix(rows) * modelBroadcast.value
      val out = bBroadcast.value.map { b =>
        mat(*, ::) :+= b
        mat
      }.getOrElse(mat)

      MatrixUtils.matrixToRowArray(out).iterator
    }).map(x => x(0) > 0)
  }
}
