package nodes.learning

import workflow.LabelEstimator

import scala.collection.mutable.ArrayBuffer

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.stats._

import org.apache.spark.rdd.RDD

import edu.berkeley.cs.amplab.mlmatrix.{RowPartition, NormalEquations, BlockCoordinateDescent, RowPartitionedMatrix}
import nodes.stats.StandardScaler

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

    val labelScaler = new StandardScaler(normalizeStdDev = false).fit(trainingLabels)
    val featureScaler = new StandardScaler(normalizeStdDev = false).fit(trainingFeatures)

    val A_parts = featureScaler.apply(trainingFeatures).mapPartitions { x =>
      Iterator.single(MatrixUtils.rowsToMatrix(x))
    }.collect()
    val b_parts = labelScaler.apply(trainingLabels).mapPartitions { x =>
      Iterator.single(MatrixUtils.rowsToMatrix(x))
    }.collect()

    val A_local = DenseMatrix.vertcat(A_parts:_*)
    val b_local = DenseMatrix.vertcat(b_parts:_*)
    
    val AAt = A_local * A_local.t 
    val model = A_local.t * ( (AAt + (DenseMatrix.eye[Double](AAt.rows) :* lambda)) \ b_local )
    LinearMapper(model, Some(labelScaler.mean), Some(featureScaler)) 
  }

}
