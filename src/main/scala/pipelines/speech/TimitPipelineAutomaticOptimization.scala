package pipelines.speech

import breeze.linalg.DenseVector
import breeze.stats.distributions.{CauchyDistribution, RandBasis, ThreadLocalRandomGenerator}
import evaluation.MulticlassClassifierEvaluator
import loaders.{LabeledData, TimitFeaturesDataLoader}
import nodes.learning.{OptimizableLeastSquaresSolver, BlockLeastSquaresEstimator}
import nodes.stats.CosineRandomFeatures
import nodes.util.{ClassLabelIndicatorsFromIntLabels, MaxClassifier, VectorCombiner}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.{SparkConf, SparkContext}
import pipelines._
import scopt.OptionParser
import workflow.AutoCacheRule.GreedyCache
import workflow._


object TimitPipelineAutomaticOptimization extends Logging {
  val appName = "TimitPipelineAutomaticOptimization"

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

    conf.checkpointDir.foreach(_ => sc.setCheckpointDir(_))

    Thread.sleep(5000)

    // Set the constants
    val seed = 123L
    val random = new java.util.Random(seed)
    val randomSource = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(random.nextLong())))

    val numCosineFeatures = 1024
    val numCosineBatches = conf.numCosines
    val colsPerBatch = numCosineFeatures + 1

    // Load the data
    val timitFeaturesData = LabeledData(TimitFeaturesDataLoader(
      sc,
      conf.trainDataLocation,
      conf.trainLabelsLocation,
      conf.testDataLocation,
      conf.testLabelsLocation,
      conf.numParts).train.labeledData.repartition(conf.numParts).cache())

    // Build the pipeline
    val trainData = timitFeaturesData.data.setName("trainRaw")
    trainData.count()

    val labels = ClassLabelIndicatorsFromIntLabels(TimitFeaturesDataLoader.numClasses).apply(
      timitFeaturesData.labels
    ).cache().setName("trainLabels")

    // Train the model
    val featurizer = Pipeline.gather {
      Seq.fill(numCosineBatches) {
        if (conf.rfType == Distributions.Cauchy) {
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
      }
    } andThen VectorCombiner()

    val predictor = featurizer andThen
        (new OptimizableLeastSquaresSolver[DenseVector[Double]](conf.lambda),
            trainData, labels) andThen MaxClassifier

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
 *
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
