package pipelines.speech

import evaluation.MulticlassClassifierEvaluator
import loaders.TimitFeaturesDataLoader
import org.apache.spark.{SparkConf, SparkContext}
import pipelines._
import scopt.OptionParser


object VWTimitEval extends Logging {
  val appName = "Timit"

  case class TimitConfig(
    dataLocation: String = "",
    vwLocation: String = "",
    modelLocation: String = "")

  def run(sc: SparkContext, conf: TimitConfig) {
    val data = sc.textFile(conf.dataLocation)
    val predictedData = data.pipe(s"${conf.vwLocation} -i ${conf.modelLocation} -t -p /dev/stdout --quiet")
    val predicted = predictedData.map(_.split(" ")(0).toDouble.toInt - 1)
    val actual = predictedData.map(_.split(" ")(1).toDouble.toInt - 1)
    val evaluator = MulticlassClassifierEvaluator(predicted, actual,
      TimitFeaturesDataLoader.numClasses)
    logInfo("TEST Error is " + (100d * evaluator.totalError) + "%")
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
