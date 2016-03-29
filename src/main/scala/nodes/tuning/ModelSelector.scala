package nodes.tuning

import org.apache.spark.rdd.RDD
import workflow.{WeightedNode, Transformer, LabelEstimator}

// FIXME: This is WAYYYY inefficient, because it has to zip all the different choices together for each time it evaluates
class ModelSelector[T, L](numChoices: Int, evaluator: RDD[(L, T)] => Double ) extends LabelEstimator[Seq[T], T, L] with WeightedNode {
  override protected def fit(data: RDD[Seq[T]], labels: RDD[L]): Transformer[Seq[T], T] = {
    val choices = (0 until numChoices).map(i => IndexSelector[T](i))
    choices.maxBy { selector =>
      evaluator(labels.zip(selector(data)))
    }
  }

  override val weight: Int = numChoices
}

case class IndexSelector[T](index: Int) extends Transformer[Seq[T], T] {
  override def apply(in: Seq[T]): T = in(index)
}
