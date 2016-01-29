package nodes.learning

import org.apache.spark.rdd.RDD
import workflow.{Transformer, LabelEstimator, OptimizableLabelEstimator}

import breeze.linalg._

import scala.reflect._


class OptimizableLeastSquaresSolver[T <: Vector[Double]: ClassTag](
    lambda: Double = 0,
    clusterProfile: ClusterProfile = ClusterProfile(
      numMachines = 16,
      cpuWeight = 0.2,
      memWeight = 0.0833,
      networkWeight = 8.33))

  extends OptimizableLabelEstimator[T, DenseVector[Double], DenseVector[Double]] {
  override val default: LabelEstimator[T, DenseVector[Double], DenseVector[Double]] = {
    if (classTag[T].runtimeClass == classOf[SparseVector[Double]]) {
      new SparseLBFGSwithL2(new LeastSquaresSparseGradient(), regParam = lambda, fitIntercept = true)
    } else {
      new DenseLBFGSwithL2(new LeastSquaresDenseGradient(), regParam = lambda, fitIntercept = true)
    }
  }

  override def optimize(
    sample: RDD[T],
    sampleLabels: RDD[DenseVector[Double]],
    numPerPartition: Map[Int, Int])
  : LabelEstimator[T, DenseVector[Double], DenseVector[Double]] = {
    val n = numPerPartition.values.sum
    val d = sample.first().length
    val k = sampleLabels.first().length
    val sparsity = sample.map(x => x.activeSize / x.length).sum() / n

    val dataProfile = DataProfile(n = n, d = d, k = k, sparsity = sparsity)
    // FixME
    // TODO: Implement stuff
    null
  }
}
