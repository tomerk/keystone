package pipelines.text

import loaders.{AmazonReviewsDataLoader, LabeledData}
import nodes.nlp._
import nodes.stats.TermFrequency
import nodes.util.CommonSparseFeatures
import org.apache.spark.api.java.{JavaPairRDD, JavaSparkContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.sysml.api.MLContext
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.matrix.data.{MatrixCell, MatrixIndexes}
import pipelines.Logging
import scopt.OptionParser

object AmazonReviewsPipelineSystemML extends Logging {
  val appName = "AmazonReviewsPipeline"

  def run(sc: SparkContext, conf: AmazonReviewsConfig) {
    val amazonTrainData = AmazonReviewsDataLoader(sc, conf.trainLocation, conf.threshold).labeledData
    val trainData = LabeledData(amazonTrainData.repartition(conf.numParts).cache())

    val training = trainData.data
    val labels = trainData.labels

    // Build the classifier estimator
    val featurizer = Trim andThen
        LowerCase() andThen
        Tokenizer() andThen
        NGramsFeaturizer(1 to conf.nGrams) andThen
        TermFrequency(x => 1) andThen
        (CommonSparseFeatures(conf.commonFeatures), training)

    val features = featurizer(trainData.data)

    val featuresToMatrixCell = features.zipWithIndex().flatMap {
      x => x._1.activeIterator.map {
        case (col, value) => (new MatrixIndexes(x._2 + 1, col + 1), new MatrixCell(value))
      }
    }

    val labelsToMatrixCell = trainData.labels.map(i => if (i > 0) 1 else -1).zipWithIndex().map {
      x => (new MatrixIndexes(x._2 + 1, 1), new MatrixCell(x._1))
    }

    val ml = new MLContext(sc)

    val numRows = training.count()
    val numCols = conf.commonFeatures
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

    val nargs = Map("X" -> " ", "Y" -> " ", "B" -> "/Users/tomerk11/Desktop/bOut.mtx")
    val outputs = ml.execute("/Users/tomerk11/Development/incubator-systemml/scripts/algorithms/LinearRegCG.dml", nargs)

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
