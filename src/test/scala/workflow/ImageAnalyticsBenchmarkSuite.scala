package workflow

import java.io.File

import breeze.linalg._
import breeze.stats.distributions.{Rand, RandBasis}
import org.apache.spark.{HashPartitioner, SparkContext}
import org.scalatest.FunSuite
import pipelines.Logging
import utils.{ChannelMajorArrayVectorizedImage, ImageMetadata, _}

case class Slice(subject: Int, sliceId: Int, slice: DenseMatrix[Double])
case class SliceChunk(subject: Int, firstNotPad: Int, slices: Map[Int, DenseMatrix[Double]])

// NOTE not using a "operate on each pixel as a tuple" because it is a super straw man in this case due to how
// the slices are already stored.

case class Overlap(originPatch: (Int, Int), data: DenseMatrix[Double])

object Overlap extends Serializable {
  def overlaps(patchId: (Int, Int), patch: DenseMatrix[Double], overlap: Int): Iterator[((Int, Int), Overlap)] = {
    // All overlapping patches: left, right, bottom, top, top-left, top-right, bottom-left, bottom-right
    Iterator("l", "r", "b", "t", "tl", "tr", "bl", "br").map {
      case "l" =>
        val overlapData = patch(::, 0 until overlap)
        val o = Overlap(patchId, overlapData.copy)
        ((patchId._1 - 1, patchId._2), o)
      case "r" =>
        val overlapData = patch(::, patch.cols - overlap until patch.cols)
        val o = Overlap(patchId, overlapData.copy)
        ((patchId._1 + 1, patchId._2), o)
      case "b" =>
        val overlapData = patch(patch.rows - overlap until patch.rows, ::)
        val o = Overlap(patchId, overlapData.copy)
        ((patchId._1, patchId._2 + 1), o)
      case "t" =>
        val overlapData = patch(0 until overlap, ::)
        val o = Overlap(patchId, overlapData.copy)
        ((patchId._1, patchId._2 - 1), o)
      case "tl" =>
        val overlapData = patch(0 until overlap, 0 until overlap)
        val o = Overlap(patchId, overlapData.copy)
        ((patchId._1 - 1, patchId._2 - 1), o)
      case "tr" =>
        val overlapData = patch(0 until overlap, patch.cols - overlap until patch.cols)
        val o = Overlap(patchId, overlapData.copy)
        ((patchId._1 + 1, patchId._2 - 1), o)
      case "bl" =>
        val overlapData = patch(patch.rows - overlap until patch.rows, 0 until overlap)
        val o = Overlap(patchId, overlapData.copy)
        ((patchId._1 - 1, patchId._2 + 1), o)
      case "br" =>
        val overlapData = patch(patch.rows - overlap until patch.rows, patch.cols - overlap until patch.cols)
        val o = Overlap(patchId, overlapData.copy)
        ((patchId._1 + 1, patchId._2 + 1), o)
    }
  }

  def buildChunkWithOverlapsThenApplyOp(patchId: (Int, Int), patch: DenseMatrix[Double], overlaps: TraversableOnce[Overlap], overlap: Int): ((Int, Int), DenseMatrix[Double]) = {
    //println(overlaps.size)//.map(_.originPatch).toList)
    (patchId, patch)

    val matToDoOp = DenseMatrix.zeros[Double](patch.rows + overlap * 2, patch.cols + overlap * 2)
    matToDoOp(overlap until (patch.rows + overlap), overlap until (patch.cols + overlap)) := patch

    overlaps.foreach {
      case Overlap(pid, data) if pid == (patchId._1 - 1, patchId._2) =>
        matToDoOp(overlap until (patch.rows + overlap), 0 until overlap) := data
      case Overlap(pid, data) if pid == (patchId._1 + 1, patchId._2) =>
        matToDoOp(overlap until (patch.rows + overlap), (patch.cols + overlap) until (patch.cols + overlap * 2)) := data
      case Overlap(pid, data) if pid == (patchId._1, patchId._2 - 1) =>
        matToDoOp(0 until overlap, overlap until (patch.cols + overlap)) := data
      case Overlap(pid, data) if pid == (patchId._1, patchId._2 + 1) =>
        matToDoOp((patch.rows + overlap) until (patch.rows + overlap * 2), overlap until (patch.cols + overlap)) := data
      case Overlap(pid, data) if pid == (patchId._1 - 1, patchId._2 - 1) =>
        matToDoOp(0 until overlap, 0 until overlap) := data
      case Overlap(pid, data) if pid == (patchId._1 + 1, patchId._2 - 1) =>
        matToDoOp(0 until overlap, (patch.cols + overlap) until (patch.cols + overlap * 2)) := data
      case Overlap(pid, data) if pid == (patchId._1 - 1, patchId._2 + 1) =>
        matToDoOp((patch.rows + overlap) until (patch.rows + overlap * 2), 0 until overlap) := data
      case Overlap(pid, data) if pid == (patchId._1 + 1, patchId._2 + 1) =>
        matToDoOp((patch.rows + overlap) until (patch.rows + overlap * 2), (patch.cols + overlap) until (patch.cols + overlap * 2)) := data
    }

    var row = overlap
    while (row < patch.rows + overlap) {
      var col = overlap
      while (col < patch.cols + overlap) {
        //val mx = max(matToDoOp((row - overlap) to (row + overlap), (col - overlap) to (col + overlap)))

        //if (matToDoOp(row, col) > 0) {
        var mx = 0.0
        var x = row - overlap
        while (x <= row + overlap) {
          var y = col - overlap
          while (y <= col + overlap) {
            mx = math.max(mx, matToDoOp(x, y))
            y += 1
          }
          x += 1
        }
        // FIXME: For some reason the following line changes how much data gets written out....
          //matToDoOp(row, col) = mx//max(matToDoOp((row - overlap) to (row + overlap), (col - overlap) to (col + overlap)))
        //}

        col += 1
      }
      row += 1
    }

    (patchId, matToDoOp(overlap until (patch.rows + overlap), overlap until (patch.cols + overlap)))

  }

  def applyOpToChunkWithOverlaps(patch: ((Int, Int), DenseMatrix[Double]), overlaps: Iterable[((Int, Int), Overlap)], overlap: Int): ((Int, Int), DenseMatrix[Double]) = {
    null
  }
}

class ImageAnalyticsBenchmarkSuite extends FunSuite with PipelineContext with Logging with Serializable {

  test("Simulated Sky Source Detect") {
    sc = new SparkContext("local[4]", "test")
    val sky = csvread(new File("/Users/tomerk11/Desktop/simulatedsky.csv"), separator = ' ')

    val skyRDD = sc.parallelize(Seq(sky))

    val patchSize = 400
    val repeatsX = 4
    val repeatsY = 3

    val numPatchesX = sky.cols / patchSize
    val numPatchesY = sky.rows / patchSize

    val partitioner = new HashPartitioner(48)
    val patchRDD = skyRDD.flatMap(sky => (0 until repeatsX * numPatchesX).iterator.flatMap(patchIdX => (0 until repeatsY * numPatchesY).iterator.map {
      patchIdY =>
        val patchData = DenseMatrix.zeros[Double](patchSize, patchSize)
        var patchX = 0
        while (patchX < patchSize) {
          var patchY = 0
          while (patchY < patchSize) {
            val skyX = patchX + (patchIdX % numPatchesX) * patchSize
            val skyY = patchY + (patchIdY % numPatchesY) * patchSize
            patchData(patchY, patchX) = sky(skyY, skyX)
            patchY += 1
          }
          patchX += 1
        }

        ((patchIdX, patchIdY), patchData)
    })).repartitionAndSortWithinPartitions(partitioner).cache()

    patchRDD.count()


    var curIterRDD = patchRDD

    (0 until 10).foreach { i =>
      curIterRDD = patchRDD
      logInfo(s"Iter ${i}")
      val prevRDD = curIterRDD
      val overlapRDD = curIterRDD.flatMap(x => Overlap.overlaps(x._1, x._2, overlap = 10)).repartitionAndSortWithinPartitions(partitioner)
      def zipFunc(itOne: Iterator[((Int, Int), DenseMatrix[Double])], itTwo: Iterator[((Int, Int), Overlap)]): Iterator[((Int, Int), DenseMatrix[Double])] = {
        import scala.math.Ordering.Implicits._
        var it = itTwo
        itOne.map {
          case (patchId, patch) =>
            val (overlaps, itnext) = it.dropWhile(_._1 < patchId).span(_._1 == patchId)
            it = itnext
            Overlap.buildChunkWithOverlapsThenApplyOp(patchId, patch, overlaps.map(_._2).toList, overlap = 10)
        }
      }
      curIterRDD = curIterRDD.zipPartitions(overlapRDD)(zipFunc)//.cache()
      curIterRDD.count()

      //prevRDD.unpersist()
    }
    logInfo("Done")

/*
    (0 until 10).foreach { i =>
      logInfo(s"Iter ${i}")
      val prevRDD = curIterRDD
      val overlapRDD = curIterRDD.flatMap(x => Overlap.overlaps(x._1, x._2, overlap = 1))
      curIterRDD = curIterRDD.cogroup(overlapRDD, partitioner = partitioner).filter(_._2._1.nonEmpty).map(x => Overlap.buildChunkWithOverlapsThenApplyOp(x._1, x._2._1.head, x._2._2, overlap = 1)).cache()
      curIterRDD.count()

      prevRDD.unpersist()
    }
    logInfo("Done")
*/
    System.in.read()

  }

  def generateVolumes(numSubjects: Int, slicesPerSubject: Int, sliceCols: Int, sliceRows: Int): Seq[Slice] = {
    val rand = RandBasis.mt0.gaussian
    (0 until numSubjects).flatMap { subject =>
      (0 until slicesPerSubject).map(sliceId =>
        Slice(subject, sliceId, DenseMatrix.rand[Double](sliceRows, sliceCols, rand))
      )
    }
  }

  test("Operate on padded chunks of slices, know for sure funcs are reduceable") {
    sc = new SparkContext("local[4]", "test")

    val chunkSize = 32
    val padding = Conf.sliceWindow / 2
    require(chunkSize > padding)

    logInfo("now starting")

    val localData = generateVolumes(Conf.numSubjects, Conf.slicesPerSubject, Conf.sliceRows, Conf.sliceCols)

    val data = sc.parallelize(localData, Conf.numPartitions).cache()
    data.count()

    val start = System.nanoTime()

    // Scale the data by the standard deviation (using a fake scaling)
    val pixelStdev = data.flatMap(_.slice.data).stdev()
    val scaledData = data.map(slice => Slice(slice.subject, slice.sliceId, slice.slice.map(_ + pixelStdev)))

    // Convert the data into padded slice chunks
    val chunkedData = scaledData.flatMap { slice =>
      val trueChunkStart = (slice.sliceId / chunkSize) * chunkSize
      if ((slice.sliceId % chunkSize < padding) && (trueChunkStart > 0)) {
        Seq(((slice.subject, trueChunkStart), (slice.sliceId, slice.slice)), ((slice.subject, trueChunkStart - chunkSize), (slice.sliceId, slice.slice)))
      } else if ((slice.sliceId % chunkSize >= (chunkSize - padding)) && (trueChunkStart + chunkSize < Conf.slicesPerSubject)) {
        Seq(((slice.subject, trueChunkStart), (slice.sliceId, slice.slice)), ((slice.subject, trueChunkStart + chunkSize), (slice.sliceId, slice.slice)))
      } else {
        Seq(((slice.subject, trueChunkStart), (slice.sliceId, slice.slice)))
      }
    }.groupByKey().map(x => SliceChunk(subject = x._1._1, firstNotPad = x._1._2, slices = x._2.toMap))

    val slidingData = chunkedData.flatMap {chunk =>
      (chunk.firstNotPad until chunk.firstNotPad + chunkSize).iterator.map { sliceId =>
        val sliceWindow = (math.max(0, sliceId - padding) to math.min(Conf.slicesPerSubject - 1, sliceId + padding)).map(chunk.slices)
        ((chunk.subject, sliceId), sliceWindow)
      }
    }

    // Do the sliding sum stencil op using a naive (but not overly straw man) strategy:
    // First sum across all slice indexes into a single sliceRowWindow by sliceColWindow matrix,
    // Then do the sliding window sum in the remaining dimensions
    val denoisedSlices = slidingData.map { sliceGroup =>
      val sumAcrossSliceIndexes = sliceGroup._2.reduce(_ + _)
      val newSlice = DenseMatrix.zeros[Double](Conf.sliceRows, Conf.sliceCols)
      var row = 0
      while (row < Conf.sliceRows) {
        var col = 0
        while (col < Conf.sliceCols) {
          var pixelVal: Double = 0
          var rowWindow = math.max(0, row - (Conf.sliceRowWindow / 2))
          val maxRowWindow = math.min(Conf.sliceRows - 1, row + (Conf.sliceRowWindow / 2))
          while (rowWindow <= maxRowWindow) {
            var colWindow = math.max(0, col - (Conf.sliceColWindow / 2))
            val maxColWindow = math.min(Conf.sliceCols - 1, row + (Conf.sliceColWindow / 2))
            while (colWindow <= maxColWindow) {
              pixelVal += sumAcrossSliceIndexes(rowWindow, colWindow)
              colWindow += 1
            }
            rowWindow += 1
          }

          newSlice(row, col) = pixelVal
          col += 1
        }
        row += 1
      }

      Slice(sliceGroup._1._1, sliceGroup._1._2, newSlice)
    }

    // Group each all subject's slices for each slice index, and merge them
    val result = denoisedSlices.map(slice => (slice.sliceId, slice.slice)).reduceByKeyLocally(_ + _)

    val elapsed = System.nanoTime - start

    logInfo(s"${result(0)(0, 0)} result (0, 0, 0)")
    logInfo(s"${result(0)(4, 0)} result (0, 4, 0)")
    logInfo(s"${result(2)(0, 4)} result (2, 0, 4)")
    logInfo(s"${result(0)(4, 4)} result (0, 4, 4)")

    logInfo(s"${elapsed / 10e8} seconds")
  }

  test("Operate on padded chunks of slices, don't know if funcs are reduceable") {
    sc = new SparkContext("local[4]", "test")

    val chunkSize = 16
    val padding = Conf.sliceWindow / 2
    require(chunkSize > Conf.sliceWindow)

    logInfo("now starting")

    val localData = generateVolumes(Conf.numSubjects, Conf.slicesPerSubject, Conf.sliceRows, Conf.sliceCols)

    val data = sc.parallelize(localData, Conf.numPartitions).cache()
    data.count()

    val start = System.nanoTime()

    // Scale the data by the standard deviation (using a fake scaling)
    val pixelStdev = data.flatMap(_.slice.data).stdev()
    val scaledData = data.map(slice => Slice(slice.subject, slice.sliceId, slice.slice.map(_ + pixelStdev)))

    // Convert the data into padded slice chunks
    val chunkedData = scaledData.flatMap { slice =>
      val trueChunkStart = (slice.sliceId / chunkSize) * chunkSize
      if ((slice.sliceId % chunkSize < padding) && (trueChunkStart > 0)) {
        Seq(((slice.subject, trueChunkStart), (slice.sliceId, slice.slice)), ((slice.subject, trueChunkStart - chunkSize), (slice.sliceId, slice.slice)))
      } else if ((slice.sliceId % chunkSize >= (chunkSize - padding)) && (trueChunkStart + chunkSize < Conf.slicesPerSubject)) {
        Seq(((slice.subject, trueChunkStart), (slice.sliceId, slice.slice)), ((slice.subject, trueChunkStart + chunkSize), (slice.sliceId, slice.slice)))
      } else {
        Seq(((slice.subject, trueChunkStart), (slice.sliceId, slice.slice)))
      }
    }.groupByKey().map(x => SliceChunk(subject = x._1._1, firstNotPad = x._1._2, slices = x._2.toMap))

    val slidingData = chunkedData.flatMap {chunk =>
      (chunk.firstNotPad until chunk.firstNotPad + chunkSize).iterator.map { sliceId =>
        val sliceWindow = (math.max(0, sliceId - padding) to math.min(Conf.slicesPerSubject - 1, sliceId + padding)).map(chunk.slices)
        ((chunk.subject, sliceId), sliceWindow)
      }
    }

    // Do the sliding sum stencil op using a naive (but not overly straw man) strategy:
    // First sum across all slice indexes into a single sliceRowWindow by sliceColWindow matrix,
    // Then do the sliding window sum in the remaining dimensions
    val denoisedSlices = slidingData.map { sliceGroup =>
      val sumAcrossSliceIndexes = sliceGroup._2.reduce(_ + _)
      val newSlice = DenseMatrix.zeros[Double](Conf.sliceRows, Conf.sliceCols)
      var row = 0
      while (row < Conf.sliceRows) {
        var col = 0
        while (col < Conf.sliceCols) {
          var pixelVal: Double = 0
          var rowWindow = math.max(0, row - (Conf.sliceRowWindow / 2))
          val maxRowWindow = math.min(Conf.sliceRows - 1, row + (Conf.sliceRowWindow / 2))
          while (rowWindow <= maxRowWindow) {
            var colWindow = math.max(0, col - (Conf.sliceColWindow / 2))
            val maxColWindow = math.min(Conf.sliceCols - 1, row + (Conf.sliceColWindow / 2))
            while (colWindow <= maxColWindow) {
              pixelVal += sumAcrossSliceIndexes(rowWindow, colWindow)
              colWindow += 1
            }
            rowWindow += 1
          }

          newSlice(row, col) = pixelVal
          col += 1
        }
        row += 1
      }

      Slice(sliceGroup._1._1, sliceGroup._1._2, newSlice)
    }

    // Group each all subject's slices for each slice index, and merge them
    // Not using reduceByKey because this is to mimic a 'fully in memory' udf
    val averageSubject = denoisedSlices.groupBy(_.sliceId).mapValues(_.map(_.slice).reduce(_ + _))
    val result = averageSubject.collectAsMap()

    val elapsed = System.nanoTime - start

    logInfo(s"${result(0)(0, 0)} result (0, 0, 0)")
    logInfo(s"${result(0)(4, 0)} result (0, 4, 0)")
    logInfo(s"${result(2)(0, 4)} result (2, 0, 4)")
    logInfo(s"${result(0)(4, 4)} result (0, 4, 4)")

    logInfo(s"${elapsed / 10e8} seconds")
  }

  test("Operate on slices if you know ops are reducable") {
    sc = new SparkContext("local[4]", "test")

    val localData = generateVolumes(Conf.numSubjects, Conf.slicesPerSubject, Conf.sliceRows, Conf.sliceCols)

    val data = sc.parallelize(localData, Conf.numPartitions).cache()
    data.count()

    val start = System.nanoTime()

    // Scale the data by the standard deviation (using a fake scaling)
    val pixelStdev = data.flatMap(_.slice.data).stdev()
    val scaledData = data.map(slice => Slice(slice.subject, slice.sliceId, slice.slice.map(_ + pixelStdev)))

    // Group all the slices into {sliceWindow}-width sliding window groups (for the 3d stencil op)
    val shuffledData = scaledData.flatMap { slice =>
      val minSlice = math.max(0, slice.sliceId - (Conf.sliceWindow / 2))
      val maxSlice = math.min(Conf.slicesPerSubject - 1, slice.sliceId + (Conf.sliceWindow / 2))
      (minSlice to maxSlice).map(targetSlice => ((slice.subject, targetSlice), slice.slice))
    }.reduceByKey(_ + _)

    // Do the sliding sum stencil op using a naive (but not overly straw man) strategy:
    // First sum across all slice indexes into a single sliceRowWindow by sliceColWindow matrix,
    // Then do the sliding window sum in the remaining dimensions
    val denoisedSlices = shuffledData.map { sliceGroup =>
      val sumAcrossSliceIndexes = sliceGroup._2
      val newSlice = DenseMatrix.zeros[Double](Conf.sliceRows, Conf.sliceCols)
      var row = 0
      while (row < Conf.sliceRows) {
        var col = 0
        while (col < Conf.sliceCols) {
          var pixelVal: Double = 0
          var rowWindow = math.max(0, row - (Conf.sliceRowWindow / 2))
          val maxRowWindow = math.min(Conf.sliceRows - 1, row + (Conf.sliceRowWindow / 2))
          while (rowWindow <= maxRowWindow) {
            var colWindow = math.max(0, col - (Conf.sliceColWindow / 2))
            val maxColWindow = math.min(Conf.sliceCols - 1, row + (Conf.sliceColWindow / 2))
            while (colWindow <= maxColWindow) {
              pixelVal += sumAcrossSliceIndexes(rowWindow, colWindow)
              colWindow += 1
            }
            rowWindow += 1
          }

          newSlice(row, col) = pixelVal
          col += 1
        }
        row += 1
      }

      Slice(sliceGroup._1._1, sliceGroup._1._2, newSlice)
    }

    // Group each all subject's slices for each slice index, and merge them
    val result = denoisedSlices.map(slice => (slice.sliceId, slice.slice)).reduceByKeyLocally(_ + _)

    val elapsed = System.nanoTime - start

    logInfo(s"${result(0)(0, 0)} result (0, 0, 0)")
    logInfo(s"${result(0)(4, 0)} result (0, 4, 0)")
    logInfo(s"${result(2)(0, 4)} result (2, 0, 4)")
    logInfo(s"${result(0)(4, 4)} result (0, 4, 4)")

    logInfo(s"${elapsed / 10e8} seconds")
  }

  test("Operate on slices, don't know if funcs are reduceable") {
    sc = new SparkContext("local[4]", "test")

    logInfo("now starting")


    val localData = generateVolumes(Conf.numSubjects, Conf.slicesPerSubject, Conf.sliceRows, Conf.sliceCols)

    val data = sc.parallelize(localData, Conf.numPartitions).cache()
    data.count()

    val start = System.nanoTime()

    // Scale the data by the standard deviation (using a fake scaling)
    val pixelStdev = data.flatMap(_.slice.data).stdev()
    val scaledData = data.map(slice => Slice(slice.subject, slice.sliceId, slice.slice.map(_ + pixelStdev)))

    // Group all the slices into {sliceWindow}-width sliding window groups (for the 3d stencil op)
    val shuffledData = scaledData.flatMap { slice =>
      val minSlice = math.max(0, slice.sliceId - (Conf.sliceWindow / 2))
      val maxSlice = math.min(Conf.slicesPerSubject - 1, slice.sliceId + (Conf.sliceWindow / 2))
      (minSlice to maxSlice).map(targetSlice => ((slice.subject, targetSlice), slice.slice))
    }.groupByKey()

    // Do the sliding sum stencil op using a naive (but not overly straw man) strategy:
    // First sum across all slice indexes into a single sliceRowWindow by sliceColWindow matrix,
    // Then do the sliding window sum in the remaining dimensions
    val denoisedSlices = shuffledData.map { sliceGroup =>
      val sumAcrossSliceIndexes = sliceGroup._2.reduce(_ + _)
      val newSlice = DenseMatrix.zeros[Double](Conf.sliceRows, Conf.sliceCols)
      var row = 0
      while (row < Conf.sliceRows) {
        var col = 0
        while (col < Conf.sliceCols) {
          var pixelVal: Double = 0
          var rowWindow = math.max(0, row - (Conf.sliceRowWindow / 2))
          val maxRowWindow = math.min(Conf.sliceRows - 1, row + (Conf.sliceRowWindow / 2))
          while (rowWindow <= maxRowWindow) {
            var colWindow = math.max(0, col - (Conf.sliceColWindow / 2))
            val maxColWindow = math.min(Conf.sliceCols - 1, row + (Conf.sliceColWindow / 2))
            while (colWindow <= maxColWindow) {
              pixelVal += sumAcrossSliceIndexes(rowWindow, colWindow)
              colWindow += 1
            }
            rowWindow += 1
          }

          newSlice(row, col) = pixelVal
          col += 1
        }
        row += 1
      }

      Slice(sliceGroup._1._1, sliceGroup._1._2, newSlice)
    }

    // Group each all subject's slices for each slice index, and merge them
    // Not using reduceByKey because this is to mimic a 'fully in memory' udf
    val averageSubject = denoisedSlices.groupBy(_.sliceId).mapValues(_.map(_.slice).reduce(_ + _))
    val result = averageSubject.collectAsMap()

    val elapsed = System.nanoTime - start

    logInfo(s"${result(0)(0, 0)} result (0, 0, 0)")
    logInfo(s"${result(0)(4, 0)} result (0, 4, 0)")
    logInfo(s"${result(2)(0, 4)} result (2, 0, 4)")
    logInfo(s"${result(0)(4, 4)} result (0, 4, 4)")

    logInfo(s"${elapsed / 10e8} seconds")
  }

}

object Conf extends Serializable {
  val numPartitions = 16

  val numSubjects = 8
  val slicesPerSubject = 256
  val sliceCols = 128
  val sliceRows = 128

  // All the sliding window dimensions must be odd :(
  val sliceWindow = 11
  val sliceRowWindow = 3
  val sliceColWindow = 3
}

trait DataChunkTrait extends Serializable {
  def apply(pos: Int*): Double
  def map(f: Double => Double): DataChunkTrait
}

class DataChunk(posFunc: Seq[Int] => Int, data: Array[Double]) extends DataChunkTrait {
  def apply(pos: Int*): Double = data(posFunc(pos))
  def map(f: Double => Double): DataChunkTrait = {
    val newData = Array[Double](data.length)
    var x = 0
    while (x < newData.length) {
      newData(x) = f(data(x))
      x = x + 1
    }

    new DataChunk(posFunc, newData)
  }
}


/**
 *  Generate as:
 Map[subid, Array[Matrices]|]

simple toy example:
normalize voxel values (based off global mean & std dev)
‘denoise’ (3d sliding window average)
‘model fit’ (merge each volume across all subjects)
 Desired finished format: Array[DenseMatrix] for all subjects

 **/