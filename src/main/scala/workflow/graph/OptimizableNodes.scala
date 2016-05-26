package workflow.graph

import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Represents a node-level optimizable transformer and its optimization rules
 */
abstract class OptimizableTransformer[A, B : ClassTag] extends Transformer[A, B] {
  val default: Transformer[A, B]
  override protected def singleTransform(a: A): B = {
    default.singleTransform(Seq(new DatumExpression(a))).asInstanceOf[B]
  }
  override protected def batchTransform(data: RDD[A]): RDD[B] = {
    default.batchTransform(Seq(new DatasetExpression(data))).asInstanceOf[RDD[B]]
  }

  def optimize(sample: RDD[A], numPerPartition: Map[Int, Int]): Pipeline[A, B]
}

/**
 * Represents a node-level optimizable Estimator and its optimization rules
 */
abstract class OptimizableEstimator[A, B] extends Estimator[A, B] {
  val default: Estimator[A, B]

  // By default should just fit using whatever the default is.
  // Due to some crazy Scala compiler shenanigans we need to do this roundabout fitRDDs call thing.
  override protected def fitRDD(data: RDD[A]): Transformer[A, B] = {
    default.fitRDDs(Seq(new DatasetExpression(data))).asInstanceOf[Transformer[A, B]]
  }

  def optimize(sample: RDD[A], numPerPartition: Map[Int, Int]): RDD[A] => Pipeline[A, B]
}

/**
 * Represents a node-level optimizable LabelEstimator and its optimization rules
 */
abstract class OptimizableLabelEstimator[A, B, L] extends LabelEstimator[A, B, L] {
  val default: LabelEstimator[A, B, L]

  // By default should just fit using whatever the default is.
  // Due to some crazy Scala compiler shenanigans we need to do this roundabout fitRDDs call thing.
  override protected def fitRDDs(data: RDD[A], labels: RDD[L]): Transformer[A, B] = {
    default.fitRDDs(Seq(new DatasetExpression(data), new DatasetExpression(labels))).asInstanceOf[Transformer[A, B]]
  }

  def optimize(sample: RDD[A], sampleLabels: RDD[L], numPerPartition: Map[Int, Int]): (RDD[A], RDD[L]) => Pipeline[A, B]
}
