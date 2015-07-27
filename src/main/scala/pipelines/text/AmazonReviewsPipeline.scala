package pipelines.text

import evaluation.{BinaryClassifierEvaluator, MulticlassClassifierEvaluator}
import loaders.{LabeledData, AmazonReviewsDataLoader, NewsgroupsDataLoader}
import nodes.learning.{LogisticRegressionSGDEstimator, NaiveBayesEstimator}
import nodes.nlp._
import nodes.stats.TermFrequency
import nodes.util.{Cacher, CommonSparseFeatures, MaxClassifier}
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser
import workflow.Optimizer

object AmazonReviewsPipeline extends Logging {
  val appName = "AmazonReviewsPipeline"

  def run(sc: SparkContext, conf: AmazonReviewsConfig) {

    val amazonData = AmazonReviewsDataLoader(sc, conf.dataLocation, conf.threshold).labeledData.repartition(conf.numParts).cache().randomSplit(Array(0.8, 0.2), 1l)
    val trainData = LabeledData(amazonData(0))
    val testData = LabeledData(amazonData(1))

    val training = trainData.data.cache()
    val labels = trainData.labels.cache()

    // Build the classifier estimator
    logInfo("Training classifier")
    val predictorPipeline = Trim andThen LowerCase() andThen
        Tokenizer() andThen
        NGramsFeaturizer(1 to conf.nGrams) andThen
        TermFrequency(x => 1) andThen
        (CommonSparseFeatures(conf.commonFeatures), training) andThen new Cacher() andThen
        (LogisticRegressionSGDEstimator(20, 0.5, 1.0), training, labels)


    val predictor = Optimizer.execute(predictorPipeline)
    logInfo("\n" + predictor.toDOTString)

    // Evaluate the classifier
    logInfo("Evaluating classifier")

    val testLabels = testData.labels
    val testResults = predictor(testData.data)
    val eval = BinaryClassifierEvaluator(testResults.map(_ > 0), testLabels.map(_ > 0))

    logInfo("\n" + eval.summary())
  }

  case class AmazonReviewsConfig(
    dataLocation: String = "",
    threshold: Double = 3.5,
    nGrams: Int = 2,
    commonFeatures: Int = 100000,
    numParts: Int = 512)

  def parse(args: Array[String]): AmazonReviewsConfig = new OptionParser[AmazonReviewsConfig](appName) {
    head(appName, "0.1")
    opt[String]("dataLocation") required() action { (x,c) => c.copy(dataLocation=x) }
    opt[Double]("threshold") action { (x,c) => c.copy(threshold=x)}
    opt[Int]("nGrams") action { (x,c) => c.copy(nGrams=x) }
    opt[Int]("commonFeatures") action { (x,c) => c.copy(commonFeatures=x) }
    opt[Int]("numParts") action { (x,c) => c.copy(numParts=x) }
  }.parse(args, AmazonReviewsConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   * @param args
   */
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[2]") // This is a fallback if things aren't set via spark submit.

    val sc = new SparkContext(conf)

    val appConfig = parse(args)
    run(sc, appConfig)

    sc.stop()
  }

}