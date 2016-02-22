package nodes.learning

import breeze.linalg._
import org.apache.spark.rdd.RDD
import pipelines.Logging
import workflow.{WeightedNode, LabelEstimator, OptimizableLabelEstimator}

import scala.reflect._


class OptimizableLeastSquaresSolver[T <: Vector[Double]: ClassTag](
                                                                    lambda: Double = 0,
                                                                    numMachines: Option[Int] = None,
                                                                    cpuWeight: Double = 0.2,
                                                                    memWeight: Double = 0.0833,
                                                                    networkWeight: Double = 8.33)
  extends OptimizableLabelEstimator[T, DenseVector[Double], DenseVector[Double]] with WeightedNode with Logging {

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

    val realNumMachines = numMachines.getOrElse {
      if (sample.sparkContext.getExecutorStorageStatus.length == 1) {
        1
      } else {
        sample.sparkContext.getExecutorStorageStatus.length - 1
      }
    }

    logInfo(s"Optimizable Param n is $n")
    logInfo(s"Optimizable Param d is $d")
    logInfo(s"Optimizable Param k is $k")
    logInfo(s"Optimizable Param sparsity is $sparsity")
    logInfo(s"Optimizable Param numMachines is $realNumMachines")
    options.minBy(_.cost(n, d, k, sparsity, realNumMachines, cpuWeight, memWeight, networkWeight))
  }

  override val weight: Int = 21 // The weight of the default lbfgs node
}
