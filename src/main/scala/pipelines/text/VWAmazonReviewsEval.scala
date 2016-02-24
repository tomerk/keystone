package pipelines.text

import breeze.linalg.norm
import evaluation.{BinaryClassifierEvaluator, MulticlassClassifierEvaluator}
import loaders.TimitFeaturesDataLoader
import org.apache.spark.{SparkConf, SparkContext}
import pipelines._
import pipelines.text.AmazonReviewsVWPreprocessor._
import scopt.OptionParser


object VWAmazonReviewsEval extends Logging {
  val appName = "VW Amazon Eval"

  case class TimitConfig(
    dataLocation: String = "",
    vwLocation: String = "",
    modelLocation: String = "")

  def run(sc: SparkContext, conf: TimitConfig) {
    val data = sc.textFile(conf.dataLocation)
    val predictedData = data.pipe(s"${conf.vwLocation} -i ${conf.modelLocation} -t -p /dev/stdout --quiet").cache()
    val predicted = predictedData.map(_.split(" ")(0).toDouble)
    val actual = predictedData.map(_.split(" ")(1).toDouble)

    logInfo(s"PIPELINE TIMING: has training points ${predictedData.count()}")
    logInfo(s"PIPELINE TIMING: has training points ${predictedData.take(200).mkString("\n")}")

    val eval = BinaryClassifierEvaluator(predicted.map(_ > 0), actual.map(_ > 0))


    logInfo("\n" + eval.summary())
    logInfo("TRAIN Error is " + (100d * (1.0 - eval.accuracy)) + "%")

    val cost = predicted.zip(actual).map { part =>
      val axb = part._1
      val labels = part._2
      val out = axb - labels
      out * out
    }.reduce(_ + _)

    val loss = cost/(2.0*predictedData.count().toDouble)
    logInfo(s"PIPELINE TIMING: Least squares loss was $loss")
  }

  def parse(args: Array[String]): TimitConfig = new OptionParser[TimitConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("dataLocation") required() action { (x,c) => c.copy(dataLocation=x) }
    opt[String]("vwLocation") required() action { (x,c) => c.copy(vwLocation=x) }
    opt[String]("modelLocation") required() action { (x,c) => c.copy(modelLocation=x) }
  }.parse(args, TimitConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
 *
   * @param args
   */
  def main(args: Array[String]) = {
    val appConfig = parse(args)

    val conf = new SparkConf().setAppName(appName)
    // NOTE: ONLY APPLICABLE IF YOU CAN DONE COPY-DIR
    conf.remove("spark.jars")

    conf.setIfMissing("spark.master", "local[2]")

    val sc = new SparkContext(conf)
    run(sc, appConfig)

    sc.stop()
  }

}
