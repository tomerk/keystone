package pipelines.text

import java.io.{File, FileInputStream, ObjectInputStream}

import pipelines.Logging

import scala.reflect.io.Directory
import breeze.linalg._
import breeze.stats._
import breeze.plot._

/**
 * Analyze the traces
 */
object TraceAnalysis extends Logging {
  def main(args: Array[String]): Unit = {
    //val dirPath = "/Users/tomerk11/Desktop/Regex_traces4436180565598422827.tmp" // 500 partitions switching often
    //val dirPath = "/Users/tomerk11/Desktop/Regex_traces1790765659707268971.tmp" // Normal 500 partitions
    //val dirPath = "/Users/tomerk11/Desktop/Regex_traces4777448709145877024.tmp" // 4 partitions switching often
    //val dirPath = args.head // 4 partitions switching often
    val dirPath = "/Users/tomerk11/Desktop/Regex_traces3839215267253318110.tmp"
    val dir = new Directory(new File(dirPath))
    val traceFiles = dir.files.filter(_.path.endsWith(".trace"))
    val traces = traceFiles.toList.map { file =>
      val fis = new FileInputStream(file.toString())
      val ois = new ObjectInputStream(fis)

      val trace = ois.readObject().asInstanceOf[Array[Result]]
      ois.close()
      (file.name, trace)
    }

    traces.foreach {
      case (name, rawTrace) =>
        val trace = DenseVector(rawTrace.map(_.elapsed.toDouble / 1000000.0))
        logInfo(s"\n$name")
        logInfo(s"length: ${trace.length}")
        logInfo(s"sum: ${sum(trace)}")
        logInfo(s"mean: ${mean(trace)}")
        logInfo(s"meanOfSample: ${mean(trace(1 to 1000))}")
        logInfo(s"variance: ${variance(trace)}")

        val f = Figure()
        val p = f.subplot(0)
        val indices = DenseVector((0 until trace.length).map(_.toDouble):_*)
        p += plot(indices, trace)
        //p += plot(DenseVector((0 until trace.length).map(_.toDouble).sliding(5).map(_.head).toArray), DenseVector(rawTrace.map(_.elapsed.toDouble / 1000000.0).sliding(5).map(x => median(DenseVector(x))).toArray))
        p.xlabel = "Document Index"
        p.ylabel = "Time in milliseconds"
        p.title = name
        f.saveas(s"/Users/tomerk11/Desktop/trace_visualizations/500parts/$name.png")
    }

    traces.indices.foreach { i =>
      ((i + 1) until traces.length).foreach { j =>
        val firstTrace = DenseVector(traces(i)._2.map(_.elapsed.toDouble / 1000000.0))//.grouped(5).map(_.sum).toArray)
        val secondTrace = DenseVector(traces(j)._2.map(_.elapsed.toDouble / 1000000.0))//.grouped(5).map(_.sum).toArray)

        logInfo(s"\nCompare: ${traces(i)._1}, ${traces(j)._1}")
        val firstFaster = (firstTrace :<= secondTrace).map(x => if (x) 1.0 else 0.0)
        val minOfBoth = min(firstTrace, secondTrace)

        logInfo(s"mean of first at start: ${mean(firstTrace(0 until 50))}")
        logInfo(s"mean of second at start: ${mean(secondTrace(0 until 50))}")
        logInfo(s"mean of min at start: ${mean(minOfBoth(0 until 50))}")
        logInfo(s"percent of time first is equal to or faster at start: ${mean(firstFaster(0 until 50))}")

        logInfo(s"mean of first at start: ${mean(firstTrace(0 until 200))}")
        logInfo(s"mean of second at start: ${mean(secondTrace(0 until 200))}")
        logInfo(s"mean of min at start: ${mean(minOfBoth(0 until 200))}")
        logInfo(s"percent of time first is equal to or faster at start: ${mean(firstFaster(0 until 200))}")

        logInfo(s"mean of first: ${mean(firstTrace)}")
        logInfo(s"mean of second: ${mean(secondTrace)}")
        logInfo(s"mean of min: ${mean(minOfBoth)}")
        logInfo(s"percent of time first is equal to or faster: ${mean(firstFaster)}")

      }
    }
  }

}