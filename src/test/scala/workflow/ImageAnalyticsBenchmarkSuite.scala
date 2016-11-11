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
    })).partitionBy(new HashPartitioner(12)).cache()

    println(s"Size: ${patchRDD.count()}")

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