package pipelines.speech

import breeze.stats.distributions.{CauchyDistribution, RandBasis, ThreadLocalRandomGenerator}
import evaluation.MulticlassClassifierEvaluator
import loaders.TimitFeaturesDataLoader
import nodes.learning.LogisticRegressionLBFGSEstimator
import nodes.stats.{CosineRandomFeatures, StandardScaler}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.{SparkConf, SparkContext}
import pipelines._
import pipelines.text.AmazonReviewsPipeline._
import scopt.OptionParser
import workflow.{Transformer, Optimizer}


object LogRegressionTimitPipeline extends Logging {
  val appName = "VW Timit Featurization"

  case class TimitConfig(
    trainDataLocation: String = "",
    trainLabelsLocation: String = "",
    testDataLocation: String = "",
    testLabelsLocation: String = "",
    numParts: Int = 512,
    numCosines: Int = 50,
    gamma: Double = 0.05555,
    rfType: Distributions.Value = Distributions.Gaussian,
    lambda: Double = 0.0,
    numEpochs: Int = 5,
    checkpointDir: Option[String] = None)

  def run(sc: SparkContext, conf: TimitConfig) {

    // Set the constants
    val seed = 123L
    val random = new java.util.Random(seed)
    val randomSource = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(random.nextLong())))

    val numCosineFeatures = 4096
    val numCosineBatches = conf.numCosines
    val colsPerBatch = numCosineFeatures + 1

    // Load the data
    val timitFeaturesData = TimitFeaturesDataLoader(
      sc,
      conf.trainDataLocation,
      conf.trainLabelsLocation,
      conf.testDataLocation,
      conf.testLabelsLocation,
      conf.numParts)

    // Build the pipeline
    val trainDataAndLabels = timitFeaturesData.train.labels.zip(timitFeaturesData.train.data).cache()
    val trainData = trainDataAndLabels.map(_._2)
    val trainLabels = trainDataAndLabels.map(_._1)

    val predictorPipeline = (if (conf.rfType == Distributions.Cauchy) {
      // TODO: Once https://github.com/scalanlp/breeze/issues/398 is released,
      // use a RandBasis for cauchy
      CosineRandomFeatures(
        TimitFeaturesDataLoader.timitDimension,
        numCosineFeatures * numCosineBatches,
        conf.gamma,
        new CauchyDistribution(0, 1),
        randomSource.uniform)
    } else {
      CosineRandomFeatures(
        TimitFeaturesDataLoader.timitDimension,
        numCosineFeatures * numCosineBatches,
        conf.gamma,
        randomSource.gaussian,
        randomSource.uniform)
    }) andThen
    (LogisticRegressionLBFGSEstimator(numClasses = TimitFeaturesDataLoader.numClasses), trainData, trainLabels) andThen
    Transformer(_.toInt)

    val predictor = Optimizer.execute(predictorPipeline)
    logInfo("\n" + predictor.toDOTString)

    val testDataAndLabels = timitFeaturesData.test.labels.zip(timitFeaturesData.test.data).cache()
    val testData = testDataAndLabels.map(_._2)
    val testLabels = testDataAndLabels.map(_._1)

    val evaluator = MulticlassClassifierEvaluator(predictor(testData), testLabels,
      TimitFeaturesDataLoader.numClasses)
    logInfo("TEST Error is " + (100d * evaluator.totalError) + "%")


  }

  object Distributions extends Enumeration {
    type Distributions = Value
    val Gaussian, Cauchy = Value
  }

  def parse(args: Array[String]): TimitConfig = new OptionParser[TimitConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainDataLocation") required() action { (x,c) => c.copy(trainDataLocation=x) }
    opt[String]("trainLabelsLocation") required() action { (x,c) => c.copy(trainLabelsLocation=x) }
    opt[String]("testDataLocation") required() action { (x,c) => c.copy(testDataLocation=x) }
    opt[String]("testLabelsLocation") required() action { (x,c) => c.copy(testLabelsLocation=x) }
    opt[String]("checkpointDir") action { (x,c) => c.copy(checkpointDir=Some(x)) }
    opt[Int]("numParts") action { (x,c) => c.copy(numParts=x) }
    opt[Int]("numCosines") action { (x,c) => c.copy(numCosines=x) }
    opt[Int]("numEpochs") action { (x,c) => c.copy(numEpochs=x) }
    opt[Double]("gamma") action { (x,c) => c.copy(gamma=x) }
    opt[Double]("lambda") action { (x,c) => c.copy(lambda=x) }
    opt("rfType")(scopt.Read.reads(Distributions withName _)) action { (x,c) => c.copy(rfType = x)}
  }.parse(args, TimitConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   * @param args
   */
  def main(args: Array[String]) = {
    val appConfig = parse(args)

    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[2]")

    val sc = new SparkContext(conf)
    run(sc, appConfig)

    sc.stop()
  }

}
