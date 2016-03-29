package workflow

import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Created by tomerk11 on 3/29/16.
 */
class ModelSelector[T : ClassTag, L](evaluator: RDD[(L, T)] => Double) extends EstimatorNode {
  override private[workflow] def fitRDDs(dependencies: Seq[RDD[_]]): TransformerNode = {
    val data = dependencies.grouped(2).map {
      case Seq(a: RDD[T], b: RDD[L]) => b.zip(a)
    }.toSeq

    val bestModel = data.indices.maxBy { i =>
      evaluator(data(i))
    }

    SelectedModel(bestModel)
  }
}

case class SelectedModel(index: Int) extends TransformerNode {
  override private[workflow] def transform(dataDependencies: Seq[_]): Any = dataDependencies(index)
  override private[workflow] def transformRDD(dataDependencies: Seq[RDD[_]]): RDD[_] = dataDependencies(index)
}