package nodes.learning

import breeze.linalg.{DenseMatrix, DenseVector, Vector}
import org.apache.spark.mllib.classification.{LogisticRegressionModel => MLlibLRM, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, NaiveBayes}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import utils.MLlibUtils.breezeVectorToMLlib
import workflow.{Transformer, LabelEstimator}

import scala.reflect.ClassTag

/**
 * A Multinomial Naive Bayes model that transforms feature vectors to vectors containing
 * the log posterior probabilities of the different classes
 *
 * @param labels list of class labels, ranging from 0 to (C - 1) inclusive
 * @param pi log of class priors, whose dimension is C, number of labels
 * @param theta log of class conditional probabilities, whose dimension is C-by-D,
 *              where D is number of features
 */
class LogisticRegressionModel[T <: Vector[Double]](
    val model: MLlibLRM) extends Transformer[T, Double] {

  /**
   * Transforms a feature vector to a vector containing the log(posterior probabilities) of the different classes
   * according to this naive bayes model.

   * @param in The input feature vector
   * @return Log-posterior probabilites of the classes for the input features
   */
  override def apply(in: T): Double = {
    model.predict(breezeVectorToMLlib(in))
  }
}

/**
 * A LabelEstimator which learns a multinomial naive bayes model from training data.
 * Outputs a Transformer that maps features to vectors containing the log-posterior-probabilities
 * of the various classes according to the learned model.
 *
 * @param lambda The lambda parameter to use for the naive bayes model
 */
case class LogisticRegressionSGDEstimator[T <: Vector[Double] : ClassTag](numIterations: Int, stepSize: Double)
    extends LabelEstimator[T, Double, Int] {
  override def fit(in: RDD[T], labels: RDD[Int]): LogisticRegressionModel[T] = {
    val labeledPoints = labels.zip(in).map(x => LabeledPoint(x._1, breezeVectorToMLlib(x._2)))
    val model = LogisticRegressionWithSGD.train(labeledPoints, numIterations, stepSize)

    new LogisticRegressionModel(model)
  }
}

case class LogisticRegressionLBFGSEstimator[T <: Vector[Double] : ClassTag](numClasses: Int = 2, numIters: Int = 100, convergenceTol: Double = 1E-4)
    extends LabelEstimator[T, Double, Int] {
  override def fit(in: RDD[T], labels: RDD[Int]): LogisticRegressionModel[T] = {
    val labeledPoints = labels.zip(in).map(x => LabeledPoint(x._1, breezeVectorToMLlib(x._2)))
    val trainer = new LogisticRegressionWithLBFGS().setNumClasses(numClasses)
    trainer.setValidateData(false).optimizer.setConvergenceTol(convergenceTol).setNumIterations(numIters)
    val model = trainer.run(labeledPoints)

    new LogisticRegressionModel(model)
  }
}
