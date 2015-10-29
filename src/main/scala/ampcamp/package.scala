import evaluation.MulticlassClassifierEvaluator
import loaders.{NewsgroupsDataLoader, LabeledData}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import workflow.Pipeline

import scala.collection.mutable

/**
 * Created by tomerk11 on 10/28/15.
 */
package object ampcamp {
  val numMnistClasses = 10
  val newsgroupsClasses = NewsgroupsDataLoader.classes
  val mnistImageSize = 784

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
    System.out.println(s"Total Accuracy: ${fmt.format(testEval.totalAccuracy)}")
    System.out.println(testEval.pprintConfusionMatrix(newsgroupsClasses))
  }
}

/*
./bin/spark-shell --master local[2] --jars /Users/tomerk11/Development/keystone/target/scala-2.10/keystoneml-assembly-0.3.0-SNAPSHOT.jar
 */

/*
import ampcamp._
import nodes.learning._
import nodes.nlp._
import nodes.stats._

val (trainData, trainLabels) = loadNewsgroupsData(sc, "/Users/tomerk11/Downloads/20news-bydate/20news-bydate-train")

val numExamples = trainData.count

val pipeline = {
    LowerCase() andThen
    Tokenizer("[\\s]+") andThen
    TermFrequency(x => x) andThen
    (IDFCommonSparseFeatures(x => math.log(numExamples/x), 100000), trainData) andThen
    (LogisticRegressionEstimator(newsgroupsClasses.length, regParam = 0, numIters = 10), trainData, trainLabels)
}

evalNewsgroupsPipeline(pipeline, sc, "/Users/tomerk11/Downloads/20news-bydate/20news-bydate-test")

 */
