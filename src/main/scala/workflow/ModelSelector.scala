package workflow

import org.apache.spark.rdd.RDD
import pipelines.Logging

import scala.reflect.ClassTag

/**
 * An EstimatorNode that tunes between several branches using training data and labels and an evaluator.
 * Outputs a [[SelectedModel]] that chooses for test time whichever branch had the highest evaluation on
 * the train data.
 *
 * @param evaluator The validation metric to use to evaluate the data and labels
 * @tparam T The type of the training data
 * @tparam L The type of the training labels
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

/**
 * A TransformerNode that takes multiple branches as input, and only outputs the one it has been selected to.
 * @param index The index of the branch to use
 */
case class SelectedModel(index: Int) extends TransformerNode {
  override private[workflow] def transform(dataDependencies: Iterator[_]): Any = dataDependencies.drop(index).next()
  override private[workflow] def transformRDD(dataDependencies: Iterator[RDD[_]]): RDD[_] = {
    dataDependencies.drop(index).next()
  }
}