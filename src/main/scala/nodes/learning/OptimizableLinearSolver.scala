package nodes.learning

import org.apache.spark.rdd.RDD
import workflow.{Transformer, LabelEstimator, OptimizableLabelEstimator}

import breeze.linalg._

import scala.reflect._

/**
 * Created by tomerk11 on 1/27/16.
 */
class OptimizableLinearSolver[T <: Vector[Double]: ClassTag](
    lambda: Double = 0,
    cpuWeight: Double = 0.2,
    memWeight: Double = 0.0833,
    networkWeight: Double = 8.33)
  extends OptimizableLabelEstimator[T, DenseVector[Double], DenseVector[Double]] {
  override val default: LabelEstimator[T, DenseVector[Double], DenseVector[Double]] = {
    if (classTag[T].runtimeClass == classOf[SparseVector[Double]]) {
      new SparseLBFGSwithL2(new LeastSquaresSparseGradient(), fitIntercept = true).asInstanceOf[LabelEstimator[T, DenseVector[Double], DenseVector[Double]]]
    } else {
      null
    }
  }

  override def optimize(sample: RDD[T], sampleLabels: RDD[DenseVector[Double]], numPerPartition: Map[Int, Int]): LabelEstimator[T, DenseVector[Double], DenseVector[Double]] = ???
}