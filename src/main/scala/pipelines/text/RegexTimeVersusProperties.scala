package pipelines.text

import java.awt.Color
import java.io.{File, FileInputStream, ObjectInputStream}

import breeze.linalg._
import breeze.plot._
import breeze.stats._
import pipelines.Logging

import scala.reflect.io.Directory

/**
 * Analyze the traces
 */
object RegexTimeVersusProperties extends Logging {
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

    val f = Figure()
    val p = f.subplot(0)
    val colors = traces.zip(Array(java.awt.Color.WHITE, java.awt.Color.GREEN, java.awt.Color.BLUE, java.awt.Color.BLACK, java.awt.Color.RED, java.awt.Color.ORANGE, java.awt.Color.MAGENTA, java.awt.Color.YELLOW)).map {
      case (tr, color) =>
        val cCopy = new Color(color.getRed, color.getGreen, color.getBlue, 128)
        (tr._1, cCopy)
    }.toMap

    logInfo(colors.toString())

    traces.foreach {
      case (name, rawTrace) =>
        val trace = DenseVector(rawTrace.filter(x => x.matches).map(_.elapsed.toDouble / 1000000.0))
        logInfo(s"\n$name")
        logInfo(s"length: ${trace.length}")
        logInfo(s"sum: ${sum(trace)}")
        logInfo(s"mean: ${mean(trace)}")
        //logInfo(s"meanOfSample: ${mean(trace(1 to 15000))}")
        logInfo(s"variance: ${variance(trace)}")

        val indices = DenseVector((0 until trace.length).map(_.toDouble):_*)
        val containsED = DenseVector(rawTrace.map(x => if (x.containsED) 1.0 else 1000.0):_*)
        val matchesRegex = DenseVector(rawTrace.map(x => if (x.matches) 1000.0 else -1000.0):_*)
        val lengths = DenseVector(rawTrace.filter(x => x.matches).map(x => x.length.toDouble/10.0):_*)

        logInfo(s"${lengths.length}, ${trace.length}")
        //val scatt = scatter(matchesRegex, trace, {_ => 40}, name = name, colors = {_ => colors(name)})
        //p += scatt
        //p += plot(DenseVector((0 until trace.length).map(_.toDouble).sliding(5).map(_.head).toArray), DenseVector(rawTrace.map(_.elapsed.toDouble / 1000000.0).sliding(5).map(x => median(DenseVector(x))).toArray))
        val fa = Figure()
        val pa = fa.subplot(0)
        pa += scatter(lengths, trace, {_ => 40}, name = name, colors = {_ => java.awt.Color.GREEN})

        val trace2 = DenseVector(rawTrace.filter(x => !x.matches).map(_.elapsed.toDouble / 1000000.0))
        val lengths2 = DenseVector(rawTrace.filter(x => !x.matches).map(x => x.length.toDouble/10.0):_*)
        pa += scatter(lengths2, trace2, {_ => 40}, name = name, colors = {_ => java.awt.Color.RED})

        pa.xlabel = "Document Index"
        pa.ylabel = "Time in milliseconds"
        pa.title = name
        // FIXME: LEGEND IS BUGGY AND SAYS WRONG COLOR
        fa.saveas(s"/Users/tomerk11/Desktop/trace_visualizations/500parts/$name.png")


    }

    p.xlabel = "Document Index"
    p.ylabel = "Time in milliseconds"
    p.title = "blah"
    // FIXME: LEGEND IS BUGGY AND SAYS WRONG COLOR
    //p.legend = (true)
    //f.saveas(s"/Users/tomerk11/Desktop/trace_visualizations/500parts/blah.png")

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
