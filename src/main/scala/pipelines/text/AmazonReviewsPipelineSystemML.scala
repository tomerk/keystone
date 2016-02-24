package pipelines.text

import breeze.linalg.{DenseVector, SparseVector}
import evaluation.BinaryClassifierEvaluator
import loaders.{AmazonReviewsDataLoader, LabeledData}
import nodes.learning.{LinearMapEstimator, SystemMLLinearReg}
import nodes.nlp._
import nodes.stats.TermFrequency
import nodes.util.CommonSparseFeatures
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser

object AmazonReviewsPipelineSystemML extends Logging {
  val appName = "AmazonReviewsPipeline SystemML Solve"

  def run(sc: SparkContext, conf: AmazonReviewsConfig) {
    logInfo("PIPELINE TIMING: Started training the classifier")
    val trainData = LabeledData(AmazonReviewsDataLoader(sc, conf.trainLocation, conf.threshold).labeledData.repartition(conf.numParts).cache())

    trainData.data.count()
    val training = trainData.data
    val labels = trainData.labels.map(_ > 0)

    // Build the classifier estimator
    logInfo("Training classifier")
    val featurizer = Trim andThen LowerCase() andThen
      Tokenizer() andThen
      NGramsFeaturizer(1 to conf.nGrams) andThen
      TermFrequency(x => 1) andThen
      (CommonSparseFeatures(conf.commonFeatures), training)

    val featurizedTrainData = featurizer.apply(training).cache()
    featurizedTrainData.count()

    logInfo("Starting Solve")
    val solveStartTime = System.currentTimeMillis()
    val solver = new SystemMLLinearReg[SparseVector[Double]](conf.scriptLocation, conf.commonFeatures, conf.numIters)
    val model = solver.fit(featurizedTrainData, labels)
    val solveEndTime  = System.currentTimeMillis()

    logInfo(s"PIPELINE TIMING: Finished Solve in ${solveEndTime - solveStartTime} ms")

    // Evaluate the classifier
    logInfo("PIPELINE TIMING: Evaluating the classifier")

    val vecLabels = labels.map(i => if (i) DenseVector(1.0) else DenseVector(-1.0))
    val loss = LinearMapEstimator.computeCostItemAtATime(featurizedTrainData.map(_.toDenseVector), vecLabels, 0, model.x, model.bOpt)
    logInfo(s"PIPELINE TIMING: Least squares loss was $loss")

    val trainResults = model(featurizedTrainData)
    val eval = BinaryClassifierEvaluator(trainResults, labels)

    logInfo("\n" + eval.summary())
    logInfo("TRAIN Error is " + (100d * (1.0 - eval.accuracy)) + "%")

    logInfo("PIPELINE TIMING: Finished evaluating the classifier")

  }

  case class AmazonReviewsConfig(
    trainLocation: String = "",
    testLocation: String = "",
    scriptLocation: String = "",
    bOutLocation: String = "",
    threshold: Double = 3.5,
    nGrams: Int = 2,
    commonFeatures: Int = 100000,
    numIters: Int = 20,
    numParts: Int = 512)

  def parse(args: Array[String]): AmazonReviewsConfig = new OptionParser[AmazonReviewsConfig](appName) {
    head(appName, "0.1")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[String]("scriptLocation") required() action { (x,c) => c.copy(scriptLocation=x) }
    opt[String]("bOutLocation") required() action { (x,c) => c.copy(bOutLocation=x) }
    opt[Double]("threshold") action { (x,c) => c.copy(threshold=x)}
    opt[Int]("nGrams") action { (x,c) => c.copy(nGrams=x) }
    opt[Int]("commonFeatures") action { (x,c) => c.copy(commonFeatures=x) }
    opt[Int]("numIters") action { (x,c) => c.copy(numParts=x) }
    opt[Int]("numParts") action { (x,c) => c.copy(numParts=x) }
  }.parse(args, AmazonReviewsConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   *
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
