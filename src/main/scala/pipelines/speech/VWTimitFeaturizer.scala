package pipelines.speech

import breeze.linalg.DenseVector
import breeze.stats.distributions.{CauchyDistribution, RandBasis, ThreadLocalRandomGenerator}
import evaluation.MulticlassClassifierEvaluator
import loaders.TimitFeaturesDataLoader
import nodes.learning.BlockLeastSquaresEstimator
import nodes.stats.{CosineRandomFeatures, StandardScaler}
import nodes.util.{ClassLabelIndicatorsFromIntLabels, MaxClassifier}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import pipelines._
import scopt.OptionParser


object VWTimitFeaturizer extends Logging {
  val appName = "VW Timit Featurization"

  case class TimitConfig(
    trainDataLocation: String = "",
    trainLabelsLocation: String = "",
    testDataLocation: String = "",
    testLabelsLocation: String = "",
    testOutLocation: String = "",
    trainOutLocation: String = "",
    numCores: Int = 512,
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
    val trainData = timitFeaturesData.train.data.cache().setName("trainRaw")

    val featurizer = if (conf.rfType == Distributions.Cauchy) {
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
    } andThen (new StandardScaler(), trainData)

    val vwTrainingFeatures = featurizer.apply(trainData)
    val vwTrainData = timitFeaturesData.train.labels.zip(vwTrainingFeatures).map {
      case (label, features) =>
        val stringBuilder = new StringBuilder()
        // also make sure to attach the label as a tag so we can keep ground truth next to predictions
        stringBuilder.append(label + 1).append(" '").append(label + 1).append(" |")
        (0 until features.length).foreach { i =>
          stringBuilder
              .append(" ")
              .append(i)
              .append(":")
              .append(features(i))
        }
        stringBuilder.toString()
    }

    vwTrainData.coalesce(conf.numCores).mapPartitions(i => scala.util.Random.shuffle(i)).saveAsTextFile(conf.trainOutLocation, classOf[GzipCodec])

    val vwTestFeatures = featurizer.apply(timitFeaturesData.test.data)
    val vwTestData = timitFeaturesData.test.labels.zip(vwTestFeatures).map {
      case (label, features) =>
        val stringBuilder = new StringBuilder()
        // also make sure to attach the label as a tag so we can keep ground truth next to predictions
        stringBuilder.append(label + 1).append(" '").append(label + 1).append(" |")
        (0 until features.length).foreach { i =>
          stringBuilder
              .append(" ")
              .append(i)
              .append(":")
              .append(features(i))
        }
        stringBuilder.toString()
    }

    vwTestData.coalesce(conf.numCores).saveAsTextFile(conf.testOutLocation, classOf[GzipCodec])
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
    opt[String]("trainOutLocation") required() action { (x,c) => c.copy(trainOutLocation=x) }
    opt[String]("testOutLocation") required() action { (x,c) => c.copy(testOutLocation=x) }
    opt[String]("checkpointDir") action { (x,c) => c.copy(checkpointDir=Some(x)) }
    opt[Int]("numCores") required() action { (x,c) => c.copy(numCores=x) }
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
