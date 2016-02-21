package pipelines.text

import loaders.{AmazonReviewsDataLoader, LabeledData}
import nodes.learning.LogisticRegressionEstimator
import nodes.nlp._
import nodes.stats.TermFrequency
import nodes.util.CommonSparseFeatures
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser
import workflow.AutoCacheRule.GreedyCache
import workflow.{NodeOptimizationRule, AutoCacheRule, EquivalentNodeMergeRule, Optimizer}

object AmazonReviewsPipelineAutomaticOptimization extends Logging {
  val appName = "AmazonReviewsPipelineAutomaticOptimization"

  def run(sc: SparkContext, conf: AmazonReviewsConfig) {
    val amazonTrainData = AmazonReviewsDataLoader(sc, conf.trainLocation, conf.threshold).labeledData
    val trainData = LabeledData(amazonTrainData.repartition(conf.numParts).cache())

    val training = trainData.data
    val labels = trainData.labels

    training.count()

    // Build the classifier estimator
    val predictor = Trim andThen
        LowerCase() andThen
        Tokenizer() andThen
        NGramsFeaturizer(1 to conf.nGrams) andThen
        TermFrequency(x => 1) andThen
        (CommonSparseFeatures(conf.commonFeatures), training) andThen
        (LogisticRegressionEstimator(numClasses = 2, numIters = conf.numIters), training, labels)

    val wholePipelineOptimizer = new Optimizer {
      protected val batches: Seq[Batch] =
        Batch("DAG Optimization", FixedPoint(100), EquivalentNodeMergeRule) ::
          Batch("Auto Cache", Once, new AutoCacheRule(GreedyCache())) ::
          Nil
    }

    logInfo("PIPELINE TIMING: Starting whole pipeline optimization")
    val wholeOptimizedPipeline = wholePipelineOptimizer.execute(predictor)
    logInfo(wholeOptimizedPipeline.toDOTString)
    logInfo("PIPELINE TIMING: Finished whole pipeline optimization")

    val fullOptimizer = new Optimizer {
      protected val batches: Seq[Batch] =
        Batch("DAG Optimization", FixedPoint(100), EquivalentNodeMergeRule) ::
          Batch("Node Level Optimization", Once, new NodeOptimizationRule) ::
          Batch("Auto Cache", Once, new AutoCacheRule(GreedyCache())) ::
          Nil
    }

    logInfo("PIPELINE TIMING: Starting ALL optimization")
    val allOptimized = fullOptimizer.execute(predictor)
    logInfo(allOptimized.toDOTString)
    logInfo("PIPELINE TIMING: Finished ALL optimization")
  }

  case class AmazonReviewsConfig(
    trainLocation: String = "",
    testLocation: String = "",
    threshold: Double = 3.5,
    nGrams: Int = 2,
    commonFeatures: Int = 100000,
    numIters: Int = 20,
    numParts: Int = 512)

  def parse(args: Array[String]): AmazonReviewsConfig = new OptionParser[AmazonReviewsConfig](appName) {
    head(appName, "0.1")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
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
