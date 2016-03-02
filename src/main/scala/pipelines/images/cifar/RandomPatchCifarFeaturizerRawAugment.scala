package pipelines.images.cifar

import scala.reflect.ClassTag

import breeze.linalg._
import breeze.numerics._
import evaluation.{AugmentedExamplesEvaluator, MulticlassClassifierEvaluator}
import loaders.CifarLoader
import nodes.images._
import nodes.learning.{BlockLeastSquaresEstimator, ZCAWhitener, ZCAWhitenerEstimator}
import nodes.stats.{StandardScaler, Sampler}
import nodes.util.{Cacher, ClassLabelIndicatorsFromIntLabels, MaxClassifier}
import pipelines.FunctionNode

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import pipelines.Logging
import scopt.OptionParser
import utils.{MatrixUtils, Stats, Image, ImageUtils}

object RandomPatchCifarFeaturizerRawAugment extends Serializable with Logging {
  val appName = "RandomPatchCifarFeaturizerRawAugment"

  class LabelAugmenter[T: ClassTag](mult: Int) extends FunctionNode[RDD[T], RDD[T]] {
    def apply(in: RDD[T]) = in.flatMap(x => Seq.fill(mult)(x))
  }

  def run(sc: SparkContext, conf: RandomCifarFeaturizerConfig) {
    //Set up some constants.
    val numClasses = 10
    val numChannels = 3
    val whitenerSize = 100000
    val numRandomPatchesAugment = conf.numRandomPatchesAugment
    val augmentRandomPatchSize = 24

    // Load up training data, and optionally sample.
    val trainData = CifarLoader(sc, conf.trainLocation).cache()
    // Augment data here
    val randomFlipper = new RandomFlips(numRandomPatchesAugment, augmentRandomPatchSize,
      testData=false, centerOnly=false)

    val trainImages = ImageExtractor.andThen(new Cacher[Image](Some("trainImages"))).apply(trainData)
    val trainImagesAugmented = randomFlipper(trainImages)

    val trainImageIds = trainImages.zipWithIndex.map(x => x._2.toInt)
    val trainImageIdsAugmented = new LabelAugmenter(numRandomPatchesAugment).apply(trainImageIds)

    val patchExtractor = new Windower(conf.patchSteps, conf.patchSize)
      .andThen(ImageVectorizer.apply)
      .andThen(new Sampler(whitenerSize * numRandomPatchesAugment))

    val (filters, whitener): (DenseMatrix[Double], ZCAWhitener) = {
        val baseFilters = patchExtractor(trainImagesAugmented)
        val baseFilterMat = Stats.normalizeRows(MatrixUtils.rowsToMatrix(baseFilters), 10.0)
        val whitener = new ZCAWhitenerEstimator(1e-1).fitSingle(baseFilterMat)

        //Normalize them.
        val sampleFilters = MatrixUtils.sampleRows(baseFilterMat, conf.numFilters)
        val unnormFilters = whitener(sampleFilters)
        val unnormSq = pow(unnormFilters, 2.0)
        val twoNorms = sqrt(sum(unnormSq(*, ::)))

        ((unnormFilters(::, *) / (twoNorms + 1e-10)) * whitener.whitener.t, whitener)
    }

    val unscaledFeaturizer = new Convolver(filters, augmentRandomPatchSize, augmentRandomPatchSize, numChannels, Some(whitener), true)
        .andThen(SymmetricRectifier(alpha=conf.alpha))
        .andThen(new Pooler(conf.poolStride, conf.poolSize, identity, Pooler.sumVector))
        .andThen(ImageVectorizer)
        .andThen(new Cacher[DenseVector[Double]](Some("features")))

    val unscaledTrainFeats = unscaledFeaturizer(trainImagesAugmented)
    val scalerModel = new StandardScaler().fit(unscaledTrainFeats)

    val labelExtractorVectorizer = LabelExtractor andThen ClassLabelIndicatorsFromIntLabels(numClasses)

    val trainLabels = new LabelAugmenter(numRandomPatchesAugment).apply(LabelExtractor(trainData)).cache()
    val trainLabelsVect = new LabelAugmenter(numRandomPatchesAugment).apply(labelExtractorVectorizer(trainData)).cache()

    val trainFeatures = scalerModel(unscaledTrainFeats).map(x => DenseVector(MatrixUtils.shuffleArray(x.toArray)))

    // Do testing.
    val testData = CifarLoader(sc, conf.testLocation)
    val testImages = ImageExtractor(testData)

    val numTestAugment = 10 // 4 corners, center and flips of each of the 5
    val testImagesAugmented = new RandomFlips(numRandomPatchesAugment, augmentRandomPatchSize, true, false).apply(testImages)
    val testLabelsAugmented = new LabelAugmenter(numTestAugment).apply(LabelExtractor(testData))
    val testImageIds = testImages.zipWithIndex.map(x => x._2.toInt)
    val testImageIdsAugmented = new LabelAugmenter(numTestAugment).apply(testImageIds)

    val testFeaturesAugmented = scalerModel(unscaledFeaturizer(testImagesAugmented)).map(x => DenseVector(MatrixUtils.shuffleArray(x.toArray)))

    val testImagesCenterOnly = new RandomFlips(numRandomPatchesAugment, augmentRandomPatchSize, true, true).apply(testImages)
    val testLabelsCenterOnly = new LabelAugmenter(1).apply(LabelExtractor(testData))
    val testImageIdsCenterOnly = new LabelAugmenter(1).apply(testImageIds)
    val testFeaturesCenterOnly = scalerModel(unscaledFeaturizer(testImagesCenterOnly)).map(x => DenseVector(MatrixUtils.shuffleArray(x.toArray)))

    val numFeatures = conf.numFilters * 2 * 4 // 4 pools, 2 for symm rectifier

    val model = new BlockLeastSquaresEstimator(4096, 1,
      conf.lambda.getOrElse(0.0), useIntercept=false).fit(trainFeatures, trainLabelsVect, Some(numFeatures))

    // Lets do two tests here 
    model.applyAndEvaluate(testFeaturesAugmented,
      (predictions: RDD[DenseVector[Double]]) => {
        val testEval = AugmentedExamplesEvaluator(testImageIdsAugmented, predictions, testLabelsAugmented, numClasses)
        logInfo(s"Center+4 corners+Flip Test error is: ${testEval.totalError}")
      })

    model.applyAndEvaluate(testFeaturesCenterOnly,
      (predictions: RDD[DenseVector[Double]]) => {
        val testEval = AugmentedExamplesEvaluator(testImageIdsCenterOnly, predictions, testLabelsCenterOnly, numClasses)
        logInfo(s"Center Only Test error is: ${testEval.totalError}")
      })
  }

  case class RandomCifarFeaturizerConfig(
      trainLocation: String = "",
      testLocation: String = "",
      numFilters: Int = 100,
      patchSize: Int = 6,
      patchSteps: Int = 1,
      poolSize: Int = 10,
      poolStride: Int = 9,
      alpha: Double = 0.25,
      lambda: Option[Double] = None,
      numRandomPatchesAugment: Int = 10)

  def parse(args: Array[String]): RandomCifarFeaturizerConfig = new OptionParser[RandomCifarFeaturizerConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Int]("numFilters") action { (x,c) => c.copy(numFilters=x) }
    opt[Int]("patchSize") action { (x,c) => c.copy(patchSize=x) }
    opt[Int]("patchSteps") action { (x,c) => c.copy(patchSteps=x) }
    opt[Int]("poolSize") action { (x,c) => c.copy(poolSize=x) }
    opt[Int]("numRandomPatchesAugment") action { (x,c) => c.copy(numRandomPatchesAugment=x) }
    opt[Double]("alpha") action { (x,c) => c.copy(alpha=x) }
    opt[Double]("lambda") action { (x,c) => c.copy(lambda=Some(x)) }
  }.parse(args, RandomCifarFeaturizerConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   * @param args
   */
  def main(args: Array[String]) = {
    val appConfig = parse(args)

    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[2]")
    // NOTE: ONLY APPLICABLE IF YOU CAN DONE COPY-DIR
    conf.remove("spark.jars")
    val sc = new SparkContext(conf)
    run(sc, appConfig)

    sc.stop()
  }
}
