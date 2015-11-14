package nodes.learning

import workflow.LabelEstimator

import scala.collection.mutable.ArrayBuffer

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.stats._

import org.apache.spark.rdd.RDD

import edu.berkeley.cs.amplab.mlmatrix.{RowPartition, NormalEquations, BlockCoordinateDescent, RowPartitionedMatrix}
import nodes.stats.{StandardScaler, StandardScalerModel}

import pipelines.Logging
import utils.{MatrixUtils, Stats}

/**
 * Train a weighted block-coordinate descent model using least squares
 *
 * @param blockSize size of blocks to use
 * @param numIter number of passes of co-ordinate descent to run
 * @param lambda regularization parameter
 * @param mixtureWeight how much should positive samples be weighted
 */
class LocalLeastSquaresEstimator(lambda: Double)
  extends LabelEstimator[DenseVector[Double], DenseVector[Double], DenseVector[Double]] {

  override def fit(
      trainingFeatures: RDD[DenseVector[Double]],
      trainingLabels: RDD[DenseVector[Double]]): LinearMapper = {
    LocalLeastSquaresEstimator.trainWithL2(trainingFeatures, trainingLabels, lambda)
  }
}

object LocalLeastSquaresEstimator {
  /**
   * Learns a linear model (OLS) based on training features and training labels.
   * Works well when the number of features >> number of examples.
   *
   * @param trainingFeatures Training features.
   * @param trainingLabels Training labels.
   * @return
   */
  def trainWithL2(
      trainingFeatures: RDD[DenseVector[Double]],
      trainingLabels: RDD[DenseVector[Double]],
      lambda: Double) = {

    val A_parts = trainingFeatures.mapPartitions { x =>
      Iterator.single(MatrixUtils.rowsToMatrix(x))
    }.collect()
    val b_parts = trainingLabels.mapPartitions { x =>
      Iterator.single(MatrixUtils.rowsToMatrix(x))
    }.collect()

    val A_local = DenseMatrix.vertcat(A_parts:_*)
    val b_local = DenseMatrix.vertcat(b_parts:_*)

    val featuresMean = mean(A_local(::, *)).toDenseVector
    val labelsMean = mean(b_local(::, *)).toDenseVector

    val A_zm = A_local(*, ::) - featuresMean
    val b_zm = b_local(*, ::) - labelsMean

    val AAt = A_zm * A_zm.t 
    val model = A_zm.t * ( (AAt + (DenseMatrix.eye[Double](AAt.rows) :* lambda)) \ b_zm )
    LinearMapper(model, Some(labelsMean), Some(new StandardScalerModel(featuresMean, None)))
  }

}
