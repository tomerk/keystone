package pipelines.speech

import breeze.linalg.DenseVector
import breeze.stats.distributions.{CauchyDistribution, RandBasis, ThreadLocalRandomGenerator}
import evaluation.{BinaryClassifierEvaluator, MulticlassClassifierEvaluator}
import loaders.TimitFeaturesDataLoader
import nodes.learning._
import nodes.stats.CosineRandomFeatures
import nodes.util.{ClassLabelIndicatorsFromIntLabels, MaxClassifier}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.api.java.{JavaPairRDD, JavaSparkContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.sysml.api.MLContext
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.matrix.data.{MatrixCell, MatrixIndexes}
import pipelines._
import scopt.OptionParser


object SystemMLSolveBinaryTimitPipeline extends Logging {
  val appName = "SystemML Solve Binary TIMIT Pipeline"

  case class TimitConfig(
    trainDataLocation: String = "",
    trainLabelsLocation: String = "",
    testDataLocation: String = "",
    testLabelsLocation: String = "",
    scriptLocation: String = "",
    bOutLocation: String = "",
    numParts: Int = 512,
    numCosines: Int = 50,
    gamma: Double = 0.05555,
    rfType: Distributions.Value = Distributions.Gaussian,
    lambda: Double = 0.0,
    numEpochs: Int = 5,
    checkpointDir: Option[String] = None)

  def run(sc: SparkContext, conf: TimitConfig) {
    logInfo("PIPELINE TIMING: Started training the classifier")

    // Set the constants
    val seed = 123L
    val random = new java.util.Random(seed)
    val randomSource = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(random.nextLong())))

    val numCosineFeatures = 1024
    val numCosineBatches = conf.numCosines

    // Load the data
    val timitFeaturesData = TimitFeaturesDataLoader(
      sc,
      conf.trainDataLocation,
      conf.trainLabelsLocation,
      conf.testDataLocation,
      conf.testLabelsLocation,
      conf.numParts)

    // Build the pipeline
    val trainDataAndLabels = timitFeaturesData.train.labeledData.repartition(conf.numParts).cache()
    val trainData = trainDataAndLabels.map(_._2)
    val labels = trainDataAndLabels.map(_._1 == 0)

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
    }

    val featurizedTrainData = featurizer(trainData).cache()
    featurizedTrainData.count()

    logInfo("PIPELINE TIMING: Starting the Solve")
    val solveStartTime = System.currentTimeMillis()
    val solver = new SystemMLLinearReg[DenseVector[Double]](conf.scriptLocation, numCosineFeatures * numCosineBatches, conf.numEpochs)
    val model = solver.fit(featurizedTrainData, labels)

    val solveEndTime  = System.currentTimeMillis()

    logInfo(s"PIPELINE TIMING: Finished Solve in ${solveEndTime - solveStartTime} ms")
    logInfo("PIPELINE TIMING: Finished training the classifier")

    // Evaluate the classifier
    logInfo("PIPELINE TIMING: Evaluating the classifier")

    val vecLabels = labels.map(i => if (i) DenseVector(1.0) else DenseVector(-1.0))
    val loss = LinearMapEstimator.computeCostItemAtATime(featurizedTrainData.map(_.toDenseVector), vecLabels, 0, model.x, model.featureScaler.map(_.mean), model.bOpt)
    logInfo(s"PIPELINE TIMING: Least squares loss was $loss")

    val trainResults = model(featurizedTrainData)
    val eval = BinaryClassifierEvaluator(trainResults, labels)
    logInfo("TRAIN Error is " + (100d * (1.0 - eval.accuracy)) + "%")

    logInfo("\n" + eval.summary())
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
    opt[String]("scriptLocation") required() action { (x,c) => c.copy(scriptLocation=x) }
    opt[String]("bOutLocation") required() action { (x,c) => c.copy(bOutLocation=x) }
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
