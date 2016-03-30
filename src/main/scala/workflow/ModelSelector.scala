package workflow

import org.apache.spark.rdd.RDD
import pipelines.Logging

import scala.reflect.ClassTag

/**
 * Created by tomerk11 on 3/29/16.
 */
case class ModelSelector[T : ClassTag, L](evaluator: (RDD[T], RDD[L]) => Double) extends EstimatorNode with Logging {
  override private[workflow] def fitRDDs(dependencies: Iterator[RDD[_]]): TransformerNode = {
    val inputs = dependencies.grouped(2).map {
      case Seq(a: RDD[T], b: RDD[L]) => (a, b)
    }.zipWithIndex

    val choices = for (((data, labels), i) <- inputs) yield {
      val selector = SelectedModel(i)
      val evaluation = evaluator(data, labels)
      logInfo(s"Evaluation for option $i is $evaluation")
      (selector, evaluation)
    }

    val finalChoices = choices.toSeq

    val decision = finalChoices.maxBy(_._2)
    logInfo(s"Chose option ${decision._1.index} with evaluation ${decision._2}")
    decision._1
  }
}

case class SelectedModel(index: Int) extends TransformerNode {
  override private[workflow] def transform(dataDependencies: Iterator[_]): Any = dataDependencies.drop(index).next()
  override private[workflow] def transformRDD(dataDependencies: Iterator[RDD[_]]): RDD[_] = dataDependencies.drop(index).next()
}