import breeze.linalg.DenseVector
import evaluation.MulticlassClassifierEvaluator
import loaders.{CsvDataLoader, NewsgroupsDataLoader, LabeledData}
import nodes.util.{MaxClassifier, ClassLabelIndicatorsFromIntLabels}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import workflow.Pipeline

import scala.collection.mutable

/**
 * Created by tomerk11 on 10/28/15.
 */
package object ampcamp {
  val newsgroupsClasses = NewsgroupsDataLoader.classes

  private val newsgroupsDataMap = mutable.Map.empty[String, LabeledData[Int, String]]
  def loadNewsgroupsData(sc: SparkContext, dataDir: String): (RDD[String], RDD[Int]) = {
    val trainLabeledData = newsgroupsDataMap.getOrElseUpdate(dataDir, LabeledData(NewsgroupsDataLoader(sc, dataDir).labeledData.cache()))
    (trainLabeledData.data, trainLabeledData.labels)
  }

  // Define load & eval methods here
  def evalNewsgroupsPipeline(pipeline: Pipeline[String, Int], sc: SparkContext, dataDir: String): Unit = {
    val (testData, testLabels) = loadNewsgroupsData(sc, dataDir)
    val testResults = pipeline(testData)
    val testEval = MulticlassClassifierEvaluator(testResults, testLabels, newsgroupsClasses.length)

    val fmt = "%2.3f"
    System.out.println(testEval.pprintConfusionMatrix(newsgroupsClasses))
    System.out.println(s"Total Accuracy: ${fmt.format(testEval.totalAccuracy)}")
  }

  private val mnistNumPartitions = 10
  private val mnistDataMap = mutable.Map.empty[String, LabeledData[Int, DenseVector[Double]]]
  val mnistClasses = Array("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

  def loadMnistData(sc: SparkContext, dataDir: String): (RDD[DenseVector[Double]], RDD[DenseVector[Double]]) = {
    val trainLabeledData = mnistDataMap.getOrElseUpdate(dataDir, {
      LabeledData(
        CsvDataLoader(sc, dataDir, mnistNumPartitions)
        // The pipeline expects 0-indexed class labels, but the labels in the file are 1-indexed
        .map(x => (x(0).toInt - 1, x(1 until x.length)))
            .cache())
    })
    (trainLabeledData.data, ClassLabelIndicatorsFromIntLabels(mnistClasses.length).apply(trainLabeledData.labels))
  }

  def evalMnistPipeline(pipeline: Pipeline[DenseVector[Double], Int], sc: SparkContext, dataDir: String): Unit = {
    val (testData, testLabels) = loadMnistData(sc, dataDir)
    val testResults = pipeline(testData)
    val testEval = MulticlassClassifierEvaluator(testResults, MaxClassifier(testLabels), mnistClasses.length)

    val fmt = "%2.3f"
    System.out.println(testEval.pprintConfusionMatrix(mnistClasses))
    System.out.println(s"Total Accuracy: ${fmt.format(testEval.totalAccuracy)}")
  }
}

/*
./bin/spark-shell --master local[2] --jars /Users/tomerk11/Development/keystone/target/scala-2.10/keystoneml-assembly-NAPSHOT.jar --driver-java-options "-Xmx2400m" --driver-class-path /Users/tomerk11/Development/keystone/target/scala-2.10/keystoneml-assembly-0.3.0-SNAPSHOT.jar
 */

/*
import ampcamp._
import nodes.learning._
import nodes.nlp._
import nodes.stats._

val (trainData, trainLabels) = loadNewsgroupsData(sc, "/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train")

val numExamples = trainData.count

val pipeline = {
  Tokenizer("[\\s]+") andThen
  TermFrequency(x => 1) andThen
  (CommonSparseFeatures(100000), trainData) andThen
  (LogisticRegressionEstimator(newsgroupsClasses.length, regParam = 0, numIters = 10), trainData, trainLabels)
}

val pipeline = {
  LowerCase() andThen
  Tokenizer("[\\s]+") andThen
  TermFrequency(x => 1) andThen
  (CommonSparseFeatures(100000), trainData) andThen
  (LogisticRegressionEstimator(newsgroupsClasses.length, regParam = 0, numIters = 10), trainData, trainLabels)
}

val pipeline = {
  LowerCase() andThen
  Tokenizer("[\\s]+") andThen
  TermFrequency(x => x) andThen
  (IDFCommonSparseFeatures(x => math.log(numExamples/x), 100000), trainData) andThen
  (LogisticRegressionEstimator(newsgroupsClasses.length, regParam = 0, numIters = 10), trainData, trainLabels)
}

evalNewsgroupsPipeline(pipeline, sc, "/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test")

 */

/*
import ampcamp._
import nodes.learning._
import nodes.stats._
import nodes.util._
import breeze.linalg.DenseVector
import workflow._

val (trainData, trainLabels) = loadMnistData(sc, "/Users/tomerk11/Desktop/mnist/train-mnist-dense-with-labels.data")

val mnistImageSize = 784

val pipeline = {
  Identity() andThen
  (LinearMapEstimator(lambda = Some(1.0)), trainData, trainLabels) andThen
  MaxClassifier
}

val pipeline = {
  RandomSignNode(mnistImageSize) andThen
  PaddedFFT() andThen
  LinearRectifier(0.0) andThen
  (LinearMapEstimator(lambda = Some(1.0)), trainData, trainLabels) andThen
  MaxClassifier
}

val pipeline = {
  Pipeline.gather {
    Seq.fill(8) {
      RandomSignNode(mnistImageSize) andThen
      PaddedFFT() andThen
      LinearRectifier(0.0)
    }
  } andThen
  VectorCombiner() andThen
  (LinearMapEstimator(lambda = Some(1.0)), trainData, trainLabels) andThen
  MaxClassifier
}


evalMnistPipeline(pipeline, sc, "/Users/tomerk11/Desktop/mnist/test-mnist-dense-with-labels.data")

 */
