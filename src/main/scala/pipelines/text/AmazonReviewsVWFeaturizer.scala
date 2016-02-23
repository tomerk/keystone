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

    logInfo("PIPELINE TIMING: Started featurizing training data")
    val trainData = LabeledData(AmazonReviewsDataLoader(sc, conf.trainLocation, conf.threshold).labeledData.repartition(conf.numParts).cache())
    val training = trainData.data
    val labels = trainData.labels

    training.count()
    // Build the featurizer
    logInfo("Training featurizer")
    val featurizer = Trim andThen LowerCase() andThen
        Tokenizer() andThen
        NGramsFeaturizer(1 to conf.nGrams) andThen
        TermFrequency(x => 1) andThen
        (CommonSparseFeatures(conf.commonFeatures), training)

    val vwTrainingFeatures = featurizer.apply(training).cache()
    vwTrainingFeatures.count()

    logInfo("PIPELINE TIMING: Finished featurizing training data")

    val startConversionTime = System.currentTimeMillis()
    val vwTrainData = labels.zip(vwTrainingFeatures).map {
      case (label, features) =>
        val target = if (label > 0) 1 else -1
        val stringBuilder = new StringBuilder()
        // also make sure to attach the label as a tag so we can keep ground truth next to predictions
        stringBuilder.append(target).append(" '").append(target).append(" |")
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
    val endConversionTime = System.currentTimeMillis()
    logInfo(s"PIPELINE TIMING: Finished System Conversion And Transfer in ${endConversionTime - startConversionTime} ms")


    /*logInfo("PIPELINE TIMING: Started featurizing test data")
    val testData = LabeledData(AmazonReviewsDataLoader(sc, conf.testLocation, conf.threshold).labeledData.repartition(conf.numParts).cache())
    val vwTestFeatures = featurizer.apply(testData.data)
    val vwTestData = testData.labels.zip(vwTestFeatures).map {
      case (label, features) =>
        val target = if (label > 0) 1 else -1
        val stringBuilder = new StringBuilder()
        // also make sure to attach the label as a tag so we can keep ground truth next to predictions
        stringBuilder.append(target).append(" '").append(target).append(" |")
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
    logInfo("PIPELINE TIMING: Finished featurizing test data")*/
  }

  case class AmazonReviewsConfig(
    trainLocation: String = "",
    testLocation: String = "",
    trainOutLocation: String = "",
    testOutLocation: String = "",
    threshold: Double = 3.5,
    nGrams: Int = 2,
    commonFeatures: Int = 100000,
    numParts: Int = 512)

  def parse(args: Array[String]): AmazonReviewsConfig = new OptionParser[AmazonReviewsConfig](appName) {
    head(appName, "0.1")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[String]("trainOutLocation") required() action { (x,c) => c.copy(trainOutLocation=x) }
    opt[String]("testOutLocation") required() action { (x,c) => c.copy(testOutLocation=x) }
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

    conf.setIfMissing("spark.master", "local[2]") // This is a fallback if things aren't set via spark submit.

    val sc = new SparkContext(conf)

    val appConfig = parse(args)
    run(sc, appConfig)

    sc.stop()
  }

}