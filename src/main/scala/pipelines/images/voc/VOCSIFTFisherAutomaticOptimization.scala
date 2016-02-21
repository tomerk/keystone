package pipelines.images.voc

import java.io.File

import breeze.linalg._
import breeze.stats._
import evaluation.MeanAveragePrecisionEvaluator
import loaders.{VOCDataPath, VOCLabelPath, VOCLoader}
import nodes.images.external.{FisherVector, GMMFisherVectorEstimator, SIFTExtractor}
import nodes.images.{GrayScaler, MultiLabelExtractor, MultiLabeledImageExtractor, PixelScaler}
import nodes.learning._
import nodes.stats.{ColumnSampler, NormalizeRows, SignedHellingerMapper}
import nodes.util.{ClassLabelIndicatorsFromIntArrayLabels, FloatToDouble, MatrixVectorizer}
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import pipelines.speech.TimitPipelineAutomaticOptimization._
import scopt.OptionParser
import workflow.AutoCacheRule.GreedyCache
import workflow.{NodeOptimizationRule, AutoCacheRule, EquivalentNodeMergeRule, Optimizer}

object VOCSIFTFisherAutomaticOptimization extends Serializable with Logging {
  val appName = "VOCSIFTFisher"

  def run(sc: SparkContext, conf: SIFTFisherConfig) {

    // Load the data and extract training labels.
    val parsedRDD = VOCLoader(
      sc,
      VOCDataPath(conf.trainLocation, "VOCdevkit/VOC2007/JPEGImages/", Some(1)),
      VOCLabelPath(conf.labelPath)).repartition(conf.numParts).cache()

    val labelGrabber = MultiLabelExtractor andThen
      ClassLabelIndicatorsFromIntArrayLabels(VOCLoader.NUM_CLASSES)

    val trainingLabels = labelGrabber(parsedRDD, None)
    val trainingData = MultiLabeledImageExtractor(parsedRDD)
    val numTrainingImages = trainingData.count().toInt
    val numPCASamplesPerImage = conf.numPcaSamples / numTrainingImages
    val numGMMSamplesPerImage = conf.numGmmSamples / numTrainingImages

    // Part 1: Scale and convert images to grayscale & Extract Sifts.
    val siftExtractor = PixelScaler andThen
        GrayScaler andThen
        new SIFTExtractor(scaleStep = conf.scaleStep)

    // Part 1a: If necessary, perform PCA on samples of the SIFT features, or load a PCA matrix from disk.
    // Part 2: Compute dimensionality-reduced PCA features.
    val pcaFeaturizer = (conf.pcaFile match {
      case Some(fname) =>
        siftExtractor andThen new BatchPCATransformer(convert(csvread(new File(fname)), Float).t)
      case None =>
        val pca = siftExtractor andThen
            ColumnSampler(numPCASamplesPerImage) andThen
            (ColumnPCAEstimator(conf.descDim), trainingData)

        siftExtractor andThen pca.fittedTransformer
    })

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
        val fisherVector = pcaFeaturizer andThen
            ColumnSampler(numGMMSamplesPerImage) andThen
            (GMMFisherVectorEstimator(conf.vocabSize), trainingData)
        pcaFeaturizer andThen fisherVector.fittedTransformer
    }) andThen
        FloatToDouble andThen
        MatrixVectorizer andThen
        NormalizeRows andThen
        SignedHellingerMapper andThen
        NormalizeRows

    // Part 4: Fit a linear model to the data.
    val predictor = fisherFeaturizer andThen
        (new OptimizableLeastSquaresSolver[DenseVector[Double]](conf.lambda),
        trainingData,
        trainingLabels)

    val wholePipelineOptimizer = new Optimizer {
      protected val batches: Seq[Batch] =
        Batch("DAG Optimization", FixedPoint(100), EquivalentNodeMergeRule) ::
          Batch("Auto Cache", Once, new AutoCacheRule(GreedyCache())) ::
          Nil
    }

    logInfo("PIPELINE TIMING: Starting whole pipeline optimization")
    wholePipelineOptimizer.execute(predictor)
    logInfo("PIPELINE TIMING: Finished whole pipeline optimization")

    val fullOptimizer = new Optimizer {
      protected val batches: Seq[Batch] =
        Batch("DAG Optimization", FixedPoint(100), EquivalentNodeMergeRule) ::
          Batch("Node Level Optimization", Once, new NodeOptimizationRule) ::
          Batch("Auto Cache", Once, new AutoCacheRule(GreedyCache())) ::
          Nil
    }

    logInfo("PIPELINE TIMING: Starting ALL optimization")
    fullOptimizer.execute(predictor)
    logInfo("PIPELINE TIMING: Finished ALL optimization")
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
