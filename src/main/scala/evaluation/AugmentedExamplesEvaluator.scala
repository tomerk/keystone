package evaluation

import breeze.linalg._
import breeze.numerics._
import org.apache.spark.rdd.RDD
import nodes.util.{MaxClassifier}

object AugmentedExamplesEvaluator {

  def averagePolicy(preds: Array[DenseVector[Double]]): DenseVector[Double] = {
    preds.reduce(_ + _) :/ preds.size.toDouble
  }

  // * Let s(k) be the ordering of patch k.
  // * Let s(k)[i] be an integer, so that the worst item is ranked 1 and the best item is ranked n.
  // For i in images,
  //  For k in patches,
  //   score[i] += s(k)[i]
  def bordaPolicy(preds: Array[DenseVector[Double]]): DenseVector[Double] = {
    val ranks = preds.map { vec =>
      val sortedPreds = vec.toArray.zipWithIndex.sortBy(_._1).map(_._2)
      val rank = DenseVector(sortedPreds.zipWithIndex.sortBy(_._1).map(x => x._2.toDouble))
      rank
    }
    ranks.reduceLeft(_ + _)
  }

  def apply(
      names: RDD[Int],
      predicted: RDD[DenseVector[Double]],
      actualLabels: RDD[Int],
      numClasses: Int): MulticlassMetrics = {

    // associate a name with each predicted, actual
    val namedPreds = names.zip(predicted.zip(actualLabels))
    // group by name to get all the predicted values for a name
    val groupedPreds = namedPreds.groupByKey(names.partitions.length).map { case (group, iter) =>
      val predActuals = iter.toArray // this is a array of tuples
      val predsForName = predActuals.map(_._1)
      assert(predActuals.map(_._2).distinct.size == 1)
      val actualForName: Int = predActuals.map(_._2).head

      (predsForName, actualForName)
    }.cache()

    // Averaging policy
    val finalPred = groupedPreds.map(x => (averagePolicy(x._1), x._2) )
    val finalPredictedLabels = MaxClassifier(finalPred.map(_._1))
    val finalActualLabels = finalPred.map(_._2)

    val ret = MulticlassClassifierEvaluator(finalPredictedLabels, finalActualLabels, numClasses)
    groupedPreds.unpersist()
    ret
  }
}
