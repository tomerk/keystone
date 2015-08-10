package pipelines.speech

import evaluation.MulticlassClassifierEvaluator
import loaders.TimitFeaturesDataLoader
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import pipelines._
import scopt.OptionParser


object LogisticRegressionTimitSolverEval extends Logging {
  val appName = "LR Timit Solver & Eval"

  case class TimitConfig(
    trainDataLocation: String = "",
    testDataLocation: String = "",
    numIters: Int = 20
  )

  def run(sc: SparkContext, conf: TimitConfig) {
    logInfo("PIPELINE TIMING: Starting LR solve")

    val trainData = sc.objectFile[LabeledPoint](conf.trainDataLocation)

    val trainer = new LogisticRegressionWithLBFGS().setNumClasses(TimitFeaturesDataLoader.numClasses)
    trainer.setValidateData(false).optimizer.setNumIterations(conf.numIters)
    val model = trainer.run(trainData.cache())

    logInfo("PIPELINE TIMING: Finished LR solve")

    logInfo("PIPELINE TIMING: Starting eval")

    val testData = sc.objectFile[LabeledPoint](conf.testDataLocation)
    val predicted = model.predict(testData.map(_.features)).map(_.toInt)
    val actual = testData.map(_.label.toInt)
    val evaluator = MulticlassClassifierEvaluator(predicted, actual,
      TimitFeaturesDataLoader.numClasses)
    logInfo("TEST Error is " + (100d * evaluator.totalError) + "%")

    logInfo("PIPELINE TIMING: Finished eval")
  }

  def parse(args: Array[String]): TimitConfig = new OptionParser[TimitConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainDataLocation") required() action { (x,c) => c.copy(trainDataLocation=x) }
    opt[String]("testDataLocation") required() action { (x,c) => c.copy(testDataLocation=x) }
    opt[Int]("numIters") action { (x,c) => c.copy(numIters=x) }
  }.parse(args, TimitConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
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
