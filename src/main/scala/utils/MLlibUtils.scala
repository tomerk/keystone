package utils

import breeze.linalg.{SparseVector, DenseMatrix, DenseVector}
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization.{SquaredL2Updater, LogisticGradient, LBFGS}
import org.apache.spark.mllib.regression.{LabeledPoint, GeneralizedLinearAlgorithm}
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD

/**
 * Provides conversions between MLlib vectors & matrices, and Breeze vectors & matrices
 */
object MLlibUtils {

  /** Convert an MLlib vector to a Breeze dense vector */
  def mllibVectorToDenseBreeze(vector: org.apache.spark.mllib.linalg.Vector): DenseVector[Double] = {
    vector match {
      case dense: org.apache.spark.mllib.linalg.DenseVector => new DenseVector[Double](dense.values)
      case _ => new DenseVector[Double](vector.toArray)
    }
  }

  /** Convert an MLlib matrix to a Breeze dense matrix */
  def mllibMatrixToDenseBreeze(matrix: org.apache.spark.mllib.linalg.Matrix): DenseMatrix[Double] = {
    matrix match {
      case dense: org.apache.spark.mllib.linalg.DenseMatrix => {
        if (!dense.isTransposed) {
          new DenseMatrix[Double](dense.numRows, dense.numCols, dense.values)
        } else {
          val breezeMatrix = new DenseMatrix[Double](dense.numRows, dense.numCols, dense.values)
          breezeMatrix.t
        }
      }

      case _ => new DenseMatrix[Double](matrix.numRows, matrix.numCols, matrix.toArray)
    }
  }

  /** Convert a Breeze vector to an MLlib vector, maintaining underlying data structure (sparse vs dense) */
  def breezeVectorToMLlib(breezeVector: breeze.linalg.Vector[Double]): org.apache.spark.mllib.linalg.Vector = {
    breezeVector match {
      case v: DenseVector[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new org.apache.spark.mllib.linalg.DenseVector(v.data)
        } else {
          new org.apache.spark.mllib.linalg.DenseVector(v.toArray)  // Can't use underlying array directly, so make a new one
        }
      case v: SparseVector[Double] =>
        if (v.index.length == v.used) {
          new org.apache.spark.mllib.linalg.SparseVector(v.length, v.index, v.data)
        } else {
          new org.apache.spark.mllib.linalg.SparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: breeze.linalg.Vector[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }

}

/**
 * Train a classification model for Multinomial/Binary Logistic Regression using
 * Limited-memory BFGS. Standard feature scaling and L2 regularization are used by default.
 * NOTE: Labels used in Logistic Regression should be {0, 1, ..., k - 1}
 * for k classes multi-label classification problem.
 */
class LogisticRegressionWithLBFGS
    extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {

  override val optimizer = new LBFGS(new LogisticGradient, new SquaredL2Updater)

  override protected val validators = List(multiLabelValidator)

  private def multiLabelValidator: RDD[LabeledPoint] => Boolean = { data =>
    if (numOfLinearPredictor > 1) {
      DataValidators.multiLabelValidator(numOfLinearPredictor + 1)(data)
    } else {
      DataValidators.binaryLabelValidator(data)
    }
  }

  /**
   * :: Experimental ::
   * Set the number of possible outcomes for k classes classification problem in
   * Multinomial Logistic Regression.
   * By default, it is binary logistic regression so k will be set to 2.
   */
  @Experimental
  def setNumClasses(numClasses: Int): this.type = {
    require(numClasses > 1)
    numOfLinearPredictor = numClasses - 1
    if (numClasses > 2) {
      optimizer.setGradient(new LogisticGradient(numClasses))
    }
    this
  }

  override protected def createModel(weights: Vector, intercept: Double) = {
    if (numOfLinearPredictor == 1) {
      new LogisticRegressionModel(weights, intercept)
    } else {
      new LogisticRegressionModel(weights, intercept, numFeatures, numOfLinearPredictor + 1)
    }
  }
}

