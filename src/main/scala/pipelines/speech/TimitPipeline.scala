package pipelines.speech

import breeze.stats.distributions.{CauchyDistribution, RandBasis, ThreadLocalRandomGenerator}
import breeze.linalg.DenseVector
import org.apache.commons.math3.random.MersenneTwister
import scopt.OptionParser

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

import evaluation.MulticlassClassifierEvaluator
import loaders.TimitFeaturesDataLoader
import nodes.learning.{BlockLinearMapper, BlockLeastSquaresEstimator}
import nodes.stats.{CosineRandomFeatures, StandardScaler}
import nodes.util.{ClassLabelIndicatorsFromIntLabels, MaxClassifier}

import pipelines._
import workflow.Optimizer


object TimitPipeline extends Logging {
  val appName = "Timit"

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
    logInfo("PIPELINE TIMING: Started training the classifier")

    conf.checkpointDir.foreach(_ => sc.setCheckpointDir(_))

    Thread.sleep(5000)

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
    val trainData = timitFeaturesData.train.data.cache().setName("trainRaw")
    trainData.count()

    val batchFeaturizer = (0 until numCosineBatches).map { batch =>
      val featurizer = if (conf.rfType == Distributions.Cauchy) {
        // TODO: Once https://github.com/scalanlp/breeze/issues/398 is released,
        // use a RandBasis for cauchy
        CosineRandomFeatures(
          TimitFeaturesDataLoader.timitDimension,
          numCosineFeatures,
          conf.gamma,
          new CauchyDistribution(0, 1),
          randomSource.uniform)
      } else {
        CosineRandomFeatures(
          TimitFeaturesDataLoader.timitDimension,
          numCosineFeatures,
          conf.gamma,
          randomSource.gaussian,
          randomSource.uniform)
      }
      Optimizer.execute(featurizer.andThen(new StandardScaler(), trainData))
    }

    val trainingBatches = batchFeaturizer.map { x =>
      x.apply(trainData)//.cache().setName("Training Block")
    }

    val labels = ClassLabelIndicatorsFromIntLabels(TimitFeaturesDataLoader.numClasses).apply(
      timitFeaturesData.train.labels
    ).cache().setName("trainLabels")

    // Train the model
    val blockLinearMapper = new BlockLeastSquaresEstimator(
      numCosineFeatures, conf.numEpochs, conf.lambda).fit(trainingBatches, labels)

    logInfo("PIPELINE TIMING: Finished training the classifier")


    logInfo("PIPELINE TIMING: Evaluating the classifier")

    val testData = timitFeaturesData.test.data.cache().setName("testRaw")
    val numTest = testData.count()

    val testBatches = batchFeaturizer.map { case x =>
      val rdd = x.apply(testData)
      rdd//.cache().setName("testFeatures")
    }

    val actual = timitFeaturesData.test.labels.cache().setName("actual")


    // Calculate test error
    val evaluator = MulticlassClassifierEvaluator(MaxClassifier(blockLinearMapper.apply(testBatches)), actual,
      TimitFeaturesDataLoader.numClasses)
    logInfo("TEST Error is " + (100d * evaluator.totalError) + "%")
    logInfo("\n" + evaluator.summary((0 until 147).map(_.toString).toArray))

    logInfo("PIPELINE TIMING: Finished evaluating the classifier")
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
