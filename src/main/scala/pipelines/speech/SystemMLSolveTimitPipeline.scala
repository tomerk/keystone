package pipelines.speech

import breeze.stats.distributions.{CauchyDistribution, RandBasis, ThreadLocalRandomGenerator}
import evaluation.MulticlassClassifierEvaluator
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


object SystemMLSolveTimitPipeline extends Logging {
  val appName = "LBFGS Solve TIMIT Pipeline"

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
    val trainLabels = ClassLabelIndicatorsFromIntLabels(TimitFeaturesDataLoader.numClasses).apply(trainDataAndLabels.map(_._1))

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
    val featuresToMatrixCell = featurizedTrainData.zipWithIndex().flatMap {
      x => x._1.activeIterator.map {
        case (col, value) => (new MatrixIndexes(x._2 + 1, col + 1), new MatrixCell(value))
      }
    }

    val labelsToMatrixCell = trainDataAndLabels.map(i => if (i._1 > 0) 1 else -1).zipWithIndex().map {
      x => (new MatrixIndexes(x._2 + 1, 1), new MatrixCell(x._1))
    }

    val ml = new MLContext(sc)

    val numRows = trainDataAndLabels.count()
    val numCols = numCosineBatches * numCosineFeatures
    val numRowsPerBlock = 1000
    val numColsPerBlock = 1000

    val mc = new MatrixCharacteristics(numRows, numCols, numRowsPerBlock, numColsPerBlock)
    val labelsMC = new MatrixCharacteristics(numRows, 1, numRowsPerBlock, 1)

    val featuresMatrix = RDDConverterUtils.binaryCellToBinaryBlock(
      new JavaSparkContext(sc),
      new JavaPairRDD(featuresToMatrixCell),
      mc,
      false)

    val labelsMatrix = RDDConverterUtils.binaryCellToBinaryBlock(
      new JavaSparkContext(sc),
      new JavaPairRDD(labelsToMatrixCell),
      labelsMC,
      false)

    ml.reset()
    ml.registerInput("X", featuresMatrix, mc)
    ml.registerInput("y", labelsMatrix, labelsMC)

    val nargs = Map(
      "X" -> " ",
      "Y" -> " ",
      "B" -> conf.bOutLocation,
      "reg" -> "0",
      "tol" -> "0",
      "maxi" -> s"${conf.numEpochs}")
    val outputs = ml.execute(conf.scriptLocation, nargs)

    val solveEndTime  = System.currentTimeMillis()

    logInfo(s"PIPELINE TIMING: Finished Solve in ${solveEndTime - solveStartTime} ms")
    logInfo("PIPELINE TIMING: Finished training the classifier")

    /*
    val loss = LinearMapEstimator.computeCost(featurizedTrainData, trainLabels, conf.lambda, model.x, model.bOpt)
    logInfo(s"PIPELINE TIMING: Least squares loss was $loss")

    logInfo("PIPELINE TIMING: Evaluating the classifier")
    val evaluator = MulticlassClassifierEvaluator(MaxClassifier(model(featurizedTrainData)), trainDataAndLabels.map(_._1),
      TimitFeaturesDataLoader.numClasses)
    logInfo("TRAIN Error is " + (100d * evaluator.totalError) + "%")
    logInfo("\n" + evaluator.summary((0 until 147).map(_.toString).toArray))

    logInfo("PIPELINE TIMING: Finished evaluating the classifier")
    */
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
