package pipelines.text

import java.io.{File, FileOutputStream, ObjectOutputStream}
import java.util.regex.Pattern
import scala.collection.JavaConverters._

import breeze.linalg.SparseVector
import evaluation.MulticlassClassifierEvaluator
import loaders.NewsgroupsDataLoader
import net.greypanther.javaadvent.regex.factories._
import nodes.learning.NaiveBayesEstimator
import nodes.nlp._
import nodes.stats.TermFrequency
import nodes.util.{CommonSparseFeatures, MaxClassifier}
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser
import workflow.Pipeline

import scala.reflect.io.Directory

case class Result(index: Long, matches: Boolean, elapsed: Long, length: Int, containsED: Boolean, wordCount: Int)

object TextTuning extends Logging {
  val appName = "NewsgroupsPipeline"

  def run(sc: SparkContext, conf: NewsgroupsConfig): Pipeline[String, Int] = {

    println("Halp?")
    val text = sc.textFile("/Users/tomerk11/Desktop/newsgroups-whole-files-to-line/*").repartition(4).zipWithIndex()
    //val text = sc.wholeTextFiles("/Users/tomerk11/Desktop/texteth").map(_._2)

    logInfo("Starting to load")
    text.cache()
    logInfo(s"There are ${text.count()} documents.")
    logInfo(s"There are ${text.map(_._1.length).mean()} characters per document.")
    logInfo(text.count().toString)
    //logInfo(text.takeSample(false, 20)(0))

    logInfo("Bout to sorts!")
    //text.map(_.sortBy(x => - x)).count()
    logInfo("Maybe doness")

    //val regexp = "\\s*([^\\s.!?]*)\\s+[a-z]*\\s+([a-z]*\\s+)?([a-z]*\\s+)?([a-z]*\\s+)?([^\\s.!?]+ed)\\s"//"(\\s+[^.!?]*[.!?])"
    // For Super slow, just add the ed and .* at the ends: val regexp = ".*[A-Ta-t][ \t\n\r]+([A-Za-z]+ed)[ \t\n\r]+.*"
    val regexp = "[ \t\n\r]*([A-Za-z]+)[ \t\n\r]+([A-Za-z]+)[ \t\n\r]+([A-Za-z]+ed)[ \t\n\r]+"
//    val regexp = ".*[Aa]lice.*"//".*[A-Ta-t]([ \t\n\r]+[A-Za-z]+)?([ \t\n\r]+[A-Za-z]+)?[ \t\n\r]+([A-Za-z]+ed).*"//"(\\s+[^.!?]*[.!?])"

    // TODO WARNME: REGEXES may not be threadsafe
    val factories = Seq[(String, Unit=>RegexFactory)](
      ("ComBasistechTclRegexFactory", _ => new ComBasistechTclRegexFactory),
      ("DkBricsAutomatonRegexFactory", _ => new DkBricsAutomatonRegexFactory),
      ("JRegexFactory", _ => new JRegexFactory),
      ("OroRegexFactory", _ => new OroRegexFactory),
      ("JavaUtilPatternRegexFactory", _ => new JavaUtilPatternRegexFactory)

      // The following don't support find all matches
      //("OrgApacheRegexpRegexFactory", _ => new OrgApacheRegexpRegexFactory),
      //("ComStevesoftPatRegexFactory", _ => new ComStevesoftPatRegexFactory),
      //("KmyRegexUtilRegexFactory", _ => new KmyRegexUtilRegexFactory),
      //("GnuRegexpReRegexFactory", _ => new GnuRegexpReRegexFactory)
    )
    val regexes = factories.map(x => (x._1, x._2().create(regexp)))
    logInfo("PRepeth")


    val doc = text.first()
    factories.foreach { case (libName, factory) =>
      val startedTime = System.currentTimeMillis()

      val matcher = factory().create(regexp)
      val matches = matcher.getMatches(doc._1, Array(0))

      val endTime = System.currentTimeMillis()
      logInfo(s"Finished $libName in ${endTime - startedTime} ms")

    }


    val dir = Directory.makeTemp("Regex_traces", null, new File("/Users/tomerk11/Desktop"))
    logInfo(s"Storing traces in ${dir.toString()}")

    factories.foreach { case (libName, factory) =>
      val startedTime = System.currentTimeMillis()

      val traceRDD = text.mapPartitions(it => {
        val matcher = factory().create(regexp)
        it.map{ x =>
          val containsED = x._1.contains("ed")
          val wordCount = x._1.split("[ \t\n\r]+").length

          val length = x._1.length
          val startTime = System.nanoTime()

          val matched = matcher.getMatches(x._1, Array(0))

          val endTime = System.nanoTime()

          val elapsed = endTime - startTime

          Result(x._2, matched.asScala.nonEmpty, elapsed, length, containsED, wordCount)
        }
      })

      val trace = traceRDD.collect.sortBy(_.index)

      val fos = new FileOutputStream(s"${dir.toString()}/$libName.trace")
      val oos = new ObjectOutputStream(fos)
      oos.writeObject(trace)
      oos.close()

      val endTime = System.currentTimeMillis()
      logInfo(s"Finished $libName in ${endTime - startedTime} ms")

    }

    /*
    val doc = text.first()
    val splitDoc = doc.split("[\n!.?,\":;()\']+")
    logInfo(s"Got this many sentences: ${splitDoc.length}")
    regexes.foreach { case (libName, matcher) =>
      val startTime = System.currentTimeMillis()

      var counts = 0
      for (sent <- splitDoc) {
        if (matcher.containsMatch(sent)) {
          counts += 1
        }
      }
      val endTime = System.currentTimeMillis()
      logInfo(s"Finished $libName in ${endTime - startTime} ms, $counts")

    }
*/
    /*(0 until 1).foreach { _ =>
      val startTime = System.currentTimeMillis()
      for (sent <- splitDoc) {
        val matcher = pattern.matcher(sent)
        //logInfo("Triessss")
        while (matcher.find()) {
          //logInfo("GOT A MATCH!!!")
          val y = (s"${matcher.group(0)}, ${matcher.group(2)}")
          //val y = matcher.group(0)
        }
      }
      val endTime = System.currentTimeMillis()
      logInfo(s"Finished in ${endTime - startTime}")

    }*/

    //logInfo(s"Maybe doness NAXT? ${text.filter(_.matches(".*<a.*?\\shref\\s*=\\s*([\\\"\\']*)(.*?)([\\\"\\'\\s].*?>|>).*")).count()}")

    //NGramsFeaturizer(1 to 5).apply(text.map(_.split("[\\p{Punct}\\s]+").toSeq)).count()
    logInfo("Still not doness")

    import Ordering.Implicits._
    //NGramsFeaturizer(1 to 5).apply(text.map(_.split("[\\p{Punct}\\s]+").toSeq)).map(_.sorted).count()
    logInfo("Okay now doness")

    null

/*    val trainData = NewsgroupsDataLoader(sc, conf.trainLocation)
    val numClasses = NewsgroupsDataLoader.classes.length

    // Build the classifier estimator
    logInfo("Training classifier")
    val predictor = Trim andThen
        LowerCase() andThen
        Tokenizer() andThen
        NGramsFeaturizer(1 to conf.nGrams) andThen
        TermFrequency(x => 1) andThen
        (CommonSparseFeatures[Seq[String]](conf.commonFeatures), trainData.data) andThen
        (NaiveBayesEstimator[SparseVector[Double]](numClasses), trainData.data, trainData.labels) andThen
        MaxClassifier

    // Evaluate the classifier
    logInfo("Evaluating classifier")

    val testData = NewsgroupsDataLoader(sc, conf.testLocation)
    val testLabels = testData.labels
    val testResults = predictor(testData.data)
    val eval = MulticlassClassifierEvaluator(testResults, testLabels, numClasses)

    logInfo("\n" + eval.summary(NewsgroupsDataLoader.classes))

    predictor
    */
  }

  case class NewsgroupsConfig(
    trainLocation: String = "",
    testLocation: String = "",
    nGrams: Int = 2,
    commonFeatures: Int = 100000)

  def parse(args: Array[String]): NewsgroupsConfig = new OptionParser[NewsgroupsConfig](appName) {
    head(appName, "0.1")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Int]("nGrams") action { (x,c) => c.copy(nGrams=x) }
    opt[Int]("commonFeatures") action { (x,c) => c.copy(commonFeatures=x) }
  }.parse(args, NewsgroupsConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   *
   * @param args
   */
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[4]") // This is a fallback if things aren't set via spark submit.

    val sc = new SparkContext(conf)

    val appConfig = parse(args)
    run(sc, appConfig)

    sc.stop()
  }

}
