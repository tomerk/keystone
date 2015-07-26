package pipelines.text

import evaluation.BinaryClassifierEvaluator
import loaders.{LabeledData, AmazonReviewsDataLoader}
import nodes.learning.NaiveBayesEstimator
import nodes.nlp._
import nodes.stats.TermFrequency
import nodes.util.{CommonSparseFeatures, MaxClassifier}
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser
import workflow.Optimizer

object AmazonReviewsVWFeaturizer extends Logging {
  val appName = "AmazonReviewsPipeline"

  def run(sc: SparkContext, conf: AmazonReviewsConfig) {

    val amazonData = AmazonReviewsDataLoader(sc, conf.dataLocation, conf.threshold).labeledData.cache().randomSplit(Array(0.8, 0.2), 1l)
    val trainData = LabeledData(amazonData(0))
    val testData = LabeledData(amazonData(1))

    val training = trainData.data.cache()
    val labels = trainData.labels.cache()

    // Build the featurizer
    logInfo("Training classifier")
    val featurizer = Trim andThen LowerCase() andThen
        Tokenizer() andThen
        NGramsFeaturizer(1 to conf.nGrams) andThen
        TermFrequency(x => 1) andThen
        (CommonSparseFeatures(conf.commonFeatures), training)

    val vwTrainingFeatures = featurizer.apply(trainData.data)
    val vwTrainData = trainData.labels.zip(vwTrainingFeatures).map {
      case (label, features) =>
        val stringBuilder = new StringBuilder()
        // also make sure to attach the label as a tag so we can keep ground truth next to predictions
        stringBuilder.append(label + 1).append(" '").append(label + 1).append(" |")
        features.activeIterator.foreach { case (index, feature) =>
          stringBuilder
              .append(" ")
              .append(index)
              .append(":")
              .append(feature)
        }
        stringBuilder.toString()
    }

    vwTrainData.saveAsTextFile(conf.trainOutLocation, classOf[GzipCodec])

    val vwTestFeatures = featurizer.apply(testData.data)
    val vwTestData = testData.labels.zip(vwTestFeatures).map {
      case (label, features) =>
        val stringBuilder = new StringBuilder()
        // also make sure to attach the label as a tag so we can keep ground truth next to predictions
        stringBuilder.append(label + 1).append(" '").append(label + 1).append(" |")
        features.activeIterator.foreach { case (index, feature) =>
          stringBuilder
              .append(" ")
              .append(index)
              .append(":")
              .append(feature)
        }
        stringBuilder.toString()
    }

    vwTestData.saveAsTextFile(conf.testOutLocation, classOf[GzipCodec])
  }

  case class AmazonReviewsConfig(
    dataLocation: String = "",
    trainOutLocation: String = "",
    testOutLocation: String = "",
    threshold: Double = 3.5,
    nGrams: Int = 2,
    commonFeatures: Int = 100000)

  def parse(args: Array[String]): AmazonReviewsConfig = new OptionParser[AmazonReviewsConfig](appName) {
    head(appName, "0.1")
    opt[String]("dataLocation") required() action { (x,c) => c.copy(dataLocation=x) }
    opt[String]("trainOutLocation") required() action { (x,c) => c.copy(trainOutLocation=x) }
    opt[String]("testOutLocation") required() action { (x,c) => c.copy(testOutLocation=x) }
    opt[Double]("threshold") action { (x,c) => c.copy(threshold=x)}
    opt[Int]("nGrams") action { (x,c) => c.copy(nGrams=x) }
    opt[Int]("commonFeatures") action { (x,c) => c.copy(commonFeatures=x) }
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