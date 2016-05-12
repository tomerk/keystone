package pipelines.images.voc

import java.io.File

import breeze.linalg._
import breeze.stats._
import evaluation.MeanAveragePrecisionEvaluator
import loaders.{VOCDataPath, VOCLabelPath, VOCLoader}
import nodes.images.external.{FisherVector, SIFTExtractor}
import nodes.images._
import nodes.learning._
import nodes.stats.{ColumnSampler, NormalizeRows, SignedHellingerMapper}
import nodes.util.{Cacher, ClassLabelIndicatorsFromIntArrayLabels, FloatToDouble, MatrixVectorizer}
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser
import utils.Image
import workflow.Pipeline

object VOCSIFTFisher extends Serializable with Logging {
  val appName = "VOCSIFTFisher"

  def run(sc: SparkContext, conf: SIFTFisherConfig): Pipeline[Image, DenseVector[Double]] =  {

    // Load the data and extract training labels.
    val parsedRDD = VOCLoader(
      sc,
      VOCDataPath(conf.trainLocation, "VOCdevkit/VOC2007/JPEGImages/", Some(1)),
      VOCLabelPath(conf.labelPath)).repartition(conf.numParts).cache()

    val labelGrabber = MultiLabelExtractor andThen new Cacher // Slight modification of label grabber to make sure to get int array for train labels at the end for eval
    val trainActuals = labelGrabber(parsedRDD) // newly added code to return a promise of int array train labels.
    val trainingLabels = ClassLabelIndicatorsFromIntArrayLabels(VOCLoader.NUM_CLASSES).apply(labelGrabber(parsedRDD)) // Previously an RDD[Labels] now a Promise[RDD[Labels]]
    val trainingData = MultiLabeledImageExtractor(parsedRDD) // Previously an RDD[Image], now a Promise[RDD[Image]]
    val numTrainingImages = parsedRDD.count().toInt // Previously `trainingData.count().toInt`, can no longer do because trainingData is now a promise
    val numPCASamplesPerImage = conf.numPcaSamples / numTrainingImages
    val numGMMSamplesPerImage = conf.numGmmSamples / numTrainingImages

    // Part 1: Scale and convert images to grayscale & Extract Sifts.
    val siftExtractor = PixelScaler andThen
        GrayScaler andThen
        new Cacher andThen
        new SIFTExtractor(scaleStep = conf.scaleStep)

    // Part 1a: If necessary, perform PCA on samples of the SIFT features, or load a PCA matrix from disk.
    // Part 2: Compute dimensionality-reduced PCA features.
    val pcaFeaturizer = (conf.pcaFile match {
      case Some(fname) =>
        siftExtractor andThen new BatchPCATransformer(convert(csvread(new File(fname)), Float).t)
      case None =>
        val columnSampler = ColumnSampler(numPCASamplesPerImage) // A column sampling transformer
        val pcaTrainData = columnSampler(siftExtractor(trainingData)) // A promise of column-sampled sift data
        val pca = ColumnPCAEstimator(conf.descDim).fit(pcaTrainData) // A pipeline that does the pca transform on Sift features
        siftExtractor andThen pca // Pipeline that does raw data -> sift features then applies PCA
    }) andThen new Cacher

    // Part 2a: If necessary, compute a GMM based on the dimensionality-reduced features, or load from disk.
    // Part 3: Compute Fisher Vectors and signed-square-root normalization.
    val fisherFeaturizer = (conf.gmmMeanFile match {
      case Some(f) =>
        val gmm = new GaussianMixtureModel(
          csvread(new File(conf.gmmMeanFile.get)),
          csvread(new File(conf.gmmVarFile.get)),
          csvread(new File(conf.gmmWtsFile.get)).toDenseVector)
        pcaFeaturizer andThen FisherVector(gmm)
      case None =>
        val columnSampler = ColumnSampler(numGMMSamplesPerImage) // A column sampling transformer
        val fisherVecTrainData = columnSampler(pcaFeaturizer(trainingData)) // A promise of column-sampled, pca-transformed sifts
        val fisherVector = GMMFisherVectorEstimator(conf.vocabSize).fit(fisherVecTrainData) // A pipeline that gets fisher vectors from pca-transformed sifts
        pcaFeaturizer andThen fisherVector // Pipeline that does raw data -> fisher vectors
    }) andThen
        FloatToDouble andThen
        MatrixVectorizer andThen
        NormalizeRows andThen
        SignedHellingerMapper andThen
        NormalizeRows andThen
        new Cacher

    // Part 4: Fit a linear model to the data.
    val predictor = fisherFeaturizer andThen
        (new BlockLeastSquaresEstimator(4096, 1, conf.lambda, Some(2 * conf.descDim * conf.vocabSize)),
        trainingData,
        trainingLabels)

    // Now featurize and apply the model to train & test data.
    val trainPredictions = predictor(trainingData) // W/ previous APIs, this would not take any advantage of all the explicitly specified caching! Now is a promise of train predictions.
    val testParsedRDD = VOCLoader(
      sc,
      VOCDataPath(conf.testLocation, "VOCdevkit/VOC2007/JPEGImages/", Some(1)),
      VOCLabelPath(conf.labelPath)).repartition(conf.numParts)

    val testData = MultiLabeledImageExtractor(testParsedRDD) // Previously returned raw test data. Now returns a promise of raw test data.

    val testActuals = labelGrabber(testParsedRDD) // Previously returned test labels. Now a promise of test labels.
    logInfo("Test Cached RDD has: " + testParsedRDD.count) // Previously `logInfo("Test Cached RDD has: " + testData.count)`. Changed because testData is now a promis

    val testPredictions = predictor(testData) // Previously returned raw test predictions. Now returns a promise of raw test data.
    Pipeline.session(Seq(trainPredictions, trainActuals, testPredictions, testActuals), DefaultOptimizer) // Newly added code to specify session

    val trainMap = MeanAveragePrecisionEvaluator(trainActuals.get(), trainPredictions.get(), VOCLoader.NUM_CLASSES) // Newly added eval of train data
    logInfo(s"TRAIN APs are: ${trainMap.toArray.mkString(",")}")
    logInfo(s"TRAIN MAP is: ${mean(trainMap)}")

    val testMap = MeanAveragePrecisionEvaluator(testActuals.get(), testPredictions.get(), VOCLoader.NUM_CLASSES) // test eval, modified to thunk promises
    logInfo(s"TEST APs are: ${testMap.toArray.mkString(",")}")
    logInfo(s"TEST MAP is: ${mean(testMap)}")

    predictor
  }

  case class SIFTFisherConfig(
    trainLocation: String = "",
    testLocation: String = "",
    labelPath: String = "",
    numParts: Int = 496,
    lambda: Double = 0.5,
    descDim: Int = 80,
    vocabSize: Int = 256,
    scaleStep: Int = 0,
    pcaFile: Option[String] = None,
    gmmMeanFile: Option[String]= None,
    gmmVarFile: Option[String] = None,
    gmmWtsFile: Option[String] = None,
    numPcaSamples: Int = 1e6.toInt,
    numGmmSamples: Int = 1e6.toInt)

  def parse(args: Array[String]): SIFTFisherConfig = new OptionParser[SIFTFisherConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[String]("labelPath") required() action { (x,c) => c.copy(labelPath=x) }
    opt[Int]("numParts") action { (x,c) => c.copy(numParts=x) }
    opt[Double]("lambda") action { (x,c) => c.copy(lambda=x) }
    opt[Int]("descDim") action { (x,c) => c.copy(descDim=x) }
    opt[Int]("vocabSize") action { (x,c) => c.copy(vocabSize=x) }
    opt[Int]("scaleStep") action { (x,c) => c.copy(scaleStep=x) }
    opt[String]("pcaFile") action { (x,c) => c.copy(pcaFile=Some(x)) }
    opt[String]("gmmMeanFile") action { (x,c) => c.copy(gmmMeanFile=Some(x)) }
    opt[String]("gmmVarFile") action { (x,c) => c.copy(gmmVarFile=Some(x)) }
    opt[String]("gmmWtsFile") action { (x,c) => c.copy(gmmWtsFile=Some(x)) }
    opt[Int]("numPcaSamples") action { (x,c) => c.copy(numPcaSamples=x) }
    opt[Int]("numGmmSamples") action { (x,c) => c.copy(numGmmSamples=x) }
  }.parse(args, SIFTFisherConfig()).get

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
