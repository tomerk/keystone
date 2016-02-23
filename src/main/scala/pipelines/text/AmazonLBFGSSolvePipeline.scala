package pipelines.text

import evaluation.BinaryClassifierEvaluator
import loaders.{AmazonReviewsDataLoader, LabeledData}
import nodes.learning.{LeastSquaresSparseGradient, SparseLBFGSwithL2, LogisticRegressionLBFGSEstimator}
import nodes.nlp._
import nodes.stats.TermFrequency
import nodes.util.{MaxClassifier, CommonSparseFeatures}
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser

object AmazonLBFGSSolvePipeline extends Logging {
  val appName = "AmazonReviewsPipeline"

  def run(sc: SparkContext, conf: AmazonReviewsConfig) {

    logInfo("PIPELINE TIMING: Started training the classifier")
    val trainData = LabeledData(AmazonReviewsDataLoader(sc, conf.trainLocation, conf.threshold).labeledData.repartition(conf.numParts).cache())

    val training = trainData.data
    val labels = trainData.labels

    // Build the classifier estimator
    logInfo("Training classifier")
    val featurizer = Trim andThen LowerCase() andThen
        Tokenizer() andThen
        NGramsFeaturizer(1 to conf.nGrams) andThen
        TermFrequency(x => 1) andThen
        (CommonSparseFeatures(conf.commonFeatures), training)

    val featurizedTrainData = featurizer.apply(training).cache()
    featurizedTrainData.count()

    val solveStartTime = System.currentTimeMillis()
    val model = new SparseLBFGSwithL2(new LeastSquaresSparseGradient(2), false, 2, numIterations=20).fit(featurizedTrainData, labels) andThen
        MaxClassifier
    val solveEndTime  = System.currentTimeMillis()

    logInfo(s"PIPELINE TIMING: Finished Solve in ${solveEndTime - solveStartTime} ms")

    // Evaluate the classifier
    logInfo("PIPELINE TIMING: Evaluating the classifier")

    val trainResults = model(featurizedTrainData)
    val eval = BinaryClassifierEvaluator(trainResults.map(_ > 0), labels.map(_ > 0))

    logInfo("\n" + eval.summary())
    logInfo("TRAIN Error is " + (100d * (1.0 - eval.accuracy)) + "%")

    logInfo("PIPELINE TIMING: Finished evaluating the classifier")
  }

  case class AmazonReviewsConfig(
    trainLocation: String = "",
    testLocation: String = "",
    threshold: Double = 3.5,
    nGrams: Int = 2,
    commonFeatures: Int = 100000,
    numParts: Int = 512)

  def parse(args: Array[String]): AmazonReviewsConfig = new OptionParser[AmazonReviewsConfig](appName) {
    head(appName, "0.1")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
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

    // NOTE: ONLY APPLICABLE IF YOU CAN DONE COPY-DIR
    conf.remove("spark.jars")
    conf.setIfMissing("spark.master", "local[8]") // This is a fallback if things aren't set via spark submit.

    val sc = new SparkContext(conf)

    val appConfig = parse(args)
    run(sc, appConfig)

    sc.stop()
  }

}
