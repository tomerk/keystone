package pipelines.images.cifar

import breeze.linalg._
import breeze.numerics._
import evaluation.MulticlassClassifierEvaluator
import loaders.CifarLoader
import nodes.images._
import nodes.learning.{BlockLeastSquaresEstimator, ZCAWhitener, ZCAWhitenerEstimator}
import nodes.stats.{Sampler, StandardScaler}
import nodes.util.{Cacher, ClassLabelIndicatorsFromIntLabels, MaxClassifier}
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser
import utils.{Image, MatrixUtils, Stats}
import workflow.Pipeline


object RandomPatchCifar extends Serializable with Logging {
  val appName = "RandomPatchCifar"

  def run(sc: SparkContext, conf: RandomCifarConfig): Pipeline[Image, Int] = {
    //Set up some constants.
    val numClasses = 10
    val imageSize = 32
    val numChannels = 3
    val whitenerSize = 100000

    // Load up training data, and optionally sample.
    val trainData = conf.sampleFrac match {
      case Some(f) => CifarLoader(sc, conf.trainLocation).sample(false, f).cache
      case None => CifarLoader(sc, conf.trainLocation).cache
    }
    val trainImages = ImageExtractor(trainData)

    val labelExtractor = LabelExtractor andThen
      ClassLabelIndicatorsFromIntLabels(numClasses) andThen
      new Cacher[DenseVector[Double]]

    val trainLabels = labelExtractor(trainData)
    // Previously this was effectively an inlined estimator fit.
    // Because trainImages is now a promise, the code that accesses trainImages
    // directly must be moved into an Estimator.
    val patchConvolverEstimator = new Estimator[Image, Image] {
      override protected def fitRDD(data: RDD[Image]): Transformer[Image, Image] = {
        val patchExtractor = new Windower(conf.patchSteps, conf.patchSize)
          .andThen(ImageVectorizer.apply)
          .andThen(new Sampler(whitenerSize))
        val baseFilters = patchExtractor(data)
        val baseFilterMat = Stats.normalizeRows(MatrixUtils.rowsToMatrix(baseFilters), 10.0)
        val whitener = new ZCAWhitenerEstimator(eps=conf.whiteningEpsilon).fitSingle(baseFilterMat)

        //Normalize them.
        val sampleFilters = MatrixUtils.sampleRows(baseFilterMat, conf.numFilters)
        val unnormFilters = whitener(sampleFilters)
        val unnormSq = pow(unnormFilters, 2.0)
        val twoNorms = sqrt(sum(unnormSq(*, ::)))
        val filters = (unnormFilters(::, *) / (twoNorms + 1e-10)) * whitener.whitener.t
        new Convolver(filters, imageSize, imageSize, numChannels, Some(whitener), true)
      }
    }

    val predictionPipeline =
        new patchConvolverEstimator.fit(trainImages) andThen
        SymmetricRectifier(alpha=conf.alpha) andThen
        new Pooler(conf.poolStride, conf.poolSize, identity, _.sum) andThen
        ImageVectorizer andThen
        new Cacher[DenseVector[Double]] andThen
        (new StandardScaler, trainImages) andThen
        (new BlockLeastSquaresEstimator(4096, 1, conf.lambda.getOrElse(0.0)), trainImages,
          trainLabels) andThen
        MaxClassifier andThen
        new Cacher[Int]

    // Calculate training error.
    val trainPredictions = predictionPipeline(trainImages)
    val trainIntLabels = LabelExtractor(trainData)
    Pipeline.session(Seq(trainPredictions, trainIntLabels, testPredictions, testLabels), DefaultOptimizer) // Add the session call

    // Do testing.
    val testData = CifarLoader(sc, conf.testLocation)
    val testImages = ImageExtractor(testData)
    val testPredictions = predictionPipeline(testImages)
    val testLabels = LabelExtractor(testData)
    val trainEval = MulticlassClassifierEvaluator(
      trainPredictions.get(), trainIntLabels.get(), numClasses)
    val testEval = MulticlassClassifierEvaluator(testPredictions.get(), testLabels.get(), numClasses)

    logInfo(s"Training error is: ${trainEval.totalError}")
    logInfo(s"Test error is: ${testEval.totalError}")

    predictionPipeline
  }

  case class RandomCifarConfig(
      trainLocation: String = "",
      testLocation: String = "",
      numFilters: Int = 100,
      whiteningEpsilon: Double = 0.1,
      patchSize: Int = 6,
      patchSteps: Int = 1,
      poolSize: Int = 14,
      poolStride: Int = 13,
      alpha: Double = 0.25,
      lambda: Option[Double] = None,
      sampleFrac: Option[Double] = None)

  def parse(args: Array[String]): RandomCifarConfig = new OptionParser[RandomCifarConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Double]("whiteningEpsilon") required() action { (x,c) => c.copy(whiteningEpsilon=x) }
    opt[Int]("numFilters") action { (x,c) => c.copy(numFilters=x) }
    opt[Int]("patchSize") action { (x,c) => c.copy(patchSize=x) }
    opt[Int]("patchSteps") action { (x,c) => c.copy(patchSteps=x) }
    opt[Int]("poolSize") action { (x,c) => c.copy(poolSize=x) }
    opt[Double]("alpha") action { (x,c) => c.copy(alpha=x) }
    opt[Double]("lambda") action { (x,c) => c.copy(lambda=Some(x)) }
    opt[Double]("sampleFrac") action { (x,c) => c.copy(sampleFrac=Some(x)) }
  }.parse(args, RandomCifarConfig()).get

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
