package workflow

import java.io.File

import breeze.linalg._
import breeze.stats.distributions.{Rand, RandBasis}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{HashPartitioner, SparkContext}
import org.scalatest.FunSuite
import pipelines.Logging
import utils.{ChannelMajorArrayVectorizedImage, ImageMetadata, _}


// NOTE not using a "operate on each pixel as a tuple" because it is a super straw man in this case due to how
// the slices are already stored.

case class Overlap(originPatch: (Int, Int), data: DenseMatrix[Double])

// -take patch id, in all dims
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
      var col = 0
      while (col < patch.cols + overlap * 2) {
        //val mx = max(matToDoOp((row - overlap) to (row + overlap), (col - overlap) to (col + overlap)))

        //if (matToDoOp(row, col) > 0) {
        var mx = 0.0
        var x = row - overlap
        while (x <= row + overlap) {
          mx = math.max(mx, matToDoOp(x, col))
          x += 1
        }
        // FIXME: For some reason the following line changes how much data gets written out....
        matToDoOp(row, col) = mx//patch(row-overlap, col-overlap) = mx//max(matToDoOp((row - overlap) to (row + overlap), (col - overlap) to (col + overlap)))//
        //}

        col += 1
      }
      row += 1
    }

    row = 0
    while (row < patch.rows + overlap * 2) {
      var col = overlap
      while (col < patch.cols + overlap) {
        //val mx = max(matToDoOp((row - overlap) to (row + overlap), (col - overlap) to (col + overlap)))

        //if (matToDoOp(row, col) > 0) {
        var mx = 0.0
        var y = col - overlap
        while (y <= col + overlap) {
          mx = math.max(mx, matToDoOp(row, y))
          y += 1
        }
        // FIXME: For some reason the following line changes how much data gets written out....
        matToDoOp(row, col) = mx//max(matToDoOp((row - overlap) to (row + overlap), (col - overlap) to (col + overlap)))
        //}

        col += 1
      }
      row += 1
    }

    /*
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
*/
    (patchId, matToDoOp(overlap until (patch.rows + overlap), overlap until (patch.cols + overlap)))

  }

}

class SkySourceDetectSuite extends FunSuite with PipelineContext with Logging with Serializable {

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
      //curIterRDD = patchRDD
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
      curIterRDD = curIterRDD.zipPartitions(overlapRDD)(zipFunc).persist(StorageLevel.MEMORY_ONLY)
      curIterRDD.count()

      prevRDD.unpersist()
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

}
