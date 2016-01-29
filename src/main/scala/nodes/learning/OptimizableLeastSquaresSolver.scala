package nodes.learning

import breeze.linalg._
import org.apache.spark.rdd.RDD
import workflow.{LabelEstimator, OptimizableLabelEstimator}

import scala.reflect._


class OptimizableLeastSquaresSolver[T <: Vector[Double]: ClassTag](
    lambda: Double = 0,
    clusterProfile: ClusterProfile = ClusterProfile(
      numMachines = 16,
      cpuWeight = 0.2,
      memWeight = 0.0833,
      networkWeight = 8.33))
  extends OptimizableLabelEstimator[T, DenseVector[Double], DenseVector[Double]] {

  val options: Seq[SolverWithCostModel[T]] = Seq(
    LeastSquaresSparseLBFGSwithL2(regParam = lambda, numIterations = 20),
    LeastSquaresDenseLBFGSwithL2(regParam = lambda, numIterations = 20),
    new BlockLeastSquaresEstimator[T](1000, 3, lambda = lambda),
    LinearMapEstimator(Some(lambda))
  )

  override val default: LabelEstimator[T, DenseVector[Double], DenseVector[Double]] = options.head

  override def optimize(
    sample: RDD[T],
    sampleLabels: RDD[DenseVector[Double]],
    numPerPartition: Map[Int, Int])
  : LabelEstimator[T, DenseVector[Double], DenseVector[Double]] = {
    val n = numPerPartition.values.map(_.toLong).sum
    val d = sample.first().length
    val k = sampleLabels.first().length
    val sparsity = sample.map(x => x.activeSize / x.length).sum() / sample.count()

    val dataProfile = DataProfile(n = n, d = d, k = k, sparsity = sparsity)

    options.minBy(_.cost(dataProfile, clusterProfile))
  }
}
