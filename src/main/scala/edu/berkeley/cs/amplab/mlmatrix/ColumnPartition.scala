package edu.berkeley.cs.amplab.mlmatrix

import java.util.concurrent.ThreadLocalRandom
import scala.reflect.ClassTag

import breeze.linalg._

import com.github.fommil.netlib.LAPACK.{getInstance=>lapack}
import org.netlib.util.intW
import org.netlib.util.doubleW

import org.apache.spark.{SparkContext, SparkException}
import org.apache.spark.rdd.RDD

/** Note: [[breeze.linalg.DenseMatrix]] by default uses column-major layout. */
case class ColumnPartition(mat: DenseMatrix[Double]) extends Serializable
case class ColumnPartitionInfo(
                                  partitionId: Int, // RDD partition this block is in
                                  blockId: Int, // BlockId goes from 0 to numBlocks
                                  startCol: Long) extends Serializable

class ColumnPartitionedMatrix(
                                 val rdd: RDD[ColumnPartition],
                                 rows: Option[Long] = None,
                                 cols: Option[Long] = None) extends DistributedMatrix(rows, cols) with Logging {

  // Map from partitionId to ColumnPartitionInfo
  // Each RDD partition can have multiple ColumnPartition
  @transient var partitionInfo_ : Map[Int, Array[ColumnPartitionInfo]] = null

  override def getDim() = {
    val dims = rdd.map { lm =>
      (lm.mat.rows.toLong, lm.mat.cols.toLong)
    }.reduce { case(a, b) =>
      (a._1, a._2 + b._2)
    }
    dims
  }

  private def calculatePartitionInfo() {
    // Partition information sorted by (partitionId, matrixInPartition)
    val colsPerPartition = rdd.mapPartitionsWithIndex { case (part, iter) =>
      if (iter.isEmpty) {
        Iterator()
      } else {
        iter.zipWithIndex.map(x => (part, x._2, x._1.mat.cols.toLong))
      }
    }.collect().sortBy(x => (x._1, x._2))

    // TODO(shivaram): Test this and make it simpler ?
    val blocksPerPartition = colsPerPartition.groupBy(x => x._1).mapValues(_.length)

    val partitionBlockStart = new collection.mutable.HashMap[Int, Int]
    partitionBlockStart.put(0, 0)
    (1 until rdd.partitions.size).foreach { p =>
      partitionBlockStart(p) =
          blocksPerPartition.getOrElse(p - 1, 0) + partitionBlockStart(p - 1)
    }

    val colsWithblockIds = colsPerPartition.map { x =>
      (x._1, partitionBlockStart(x._1) + x._2, x._3)
    }

    val cumulativeSum = colsWithblockIds.scanLeft(0L){ case (x1, x2) =>
      x1 + x2._3
    }.dropRight(1)

    partitionInfo_ = colsWithblockIds.map(x => (x._1, x._2)).zip(
      cumulativeSum).map(x => ColumnPartitionInfo(x._1._1, x._1._2, x._2)).groupBy(x => x.partitionId)
  }

  def getPartitionInfo = {
    if (partitionInfo_ == null) {
      calculatePartitionInfo()
    }
    partitionInfo_
  }

  override def +(other: Double) = {
    new ColumnPartitionedMatrix(rdd.map { lm =>
      ColumnPartition(lm.mat :+ other)
    }, rows, cols)
  }

  override def *(other: Double) = {
    new ColumnPartitionedMatrix(rdd.map { lm =>
      ColumnPartition(lm.mat :* other)
    }, rows, cols)
  }

  override def mapElements(f: Double => Double) = {
    new ColumnPartitionedMatrix(rdd.map { lm =>
      ColumnPartition(
        new DenseMatrix[Double](lm.mat.rows, lm.mat.cols, lm.mat.data.map(f)))
    }, rows, cols)
  }

  override def aggregateElements[U: ClassTag](zeroValue: U)(seqOp: (U, Double) => U, combOp: (U, U) => U): U = {
    rdd.map { part =>
      part.mat.data.aggregate(zeroValue)(seqOp, combOp)
    }.reduce(combOp)
  }

  override def reduceRowElements(f: (Double, Double) => Double): DistributedMatrix = ???

  override def reduceColElements(f: (Double, Double) => Double): DistributedMatrix = ???

  override def +(other: DistributedMatrix) = {
    other match {
      case otherBlocked: ColumnPartitionedMatrix =>
        if (this.dim == other.dim) {
          // Check if matrices share same partitioner and can be zipped
          if (rdd.partitions.size == otherBlocked.rdd.partitions.size) {
            new ColumnPartitionedMatrix(rdd.zip(otherBlocked.rdd).map { case (lm, otherLM) =>
              ColumnPartition(lm.mat :+ otherLM.mat)
            }, rows, cols)
          } else {
            throw new SparkException(
              "Cannot add matrices with unequal partitions")
          }
        } else {
          throw new IllegalArgumentException("Cannot add matrices of unequal size")
        }
      case _ =>
        throw new IllegalArgumentException("Cannot add matrices of different types")
    }
  }

  override def apply(rowRange: Range, colRange: ::.type) = {
    new ColumnPartitionedMatrix(rdd.map { lm =>
      ColumnPartition(lm.mat(rowRange, ::))
    })
  }

  override def apply(rowRange: ::.type, colRange: Range) = {
    this.apply(Range(0, numRows().toInt), colRange)
  }

  override def apply(rowRange: Range, colRange: Range) = {
    // TODO: Make this a class member
    val partitionBroadcast = rdd.sparkContext.broadcast(getPartitionInfo)

    // First filter partitions which have columns in this index, then select them
    ColumnPartitionedMatrix.fromMatrix(rdd.mapPartitionsWithIndex { case (part, iter) =>
      if (partitionBroadcast.value.contains(part)) {
        val startCols = partitionBroadcast.value(part).sortBy(x => x.blockId).map(x => x.startCol)
        iter.zip(startCols.iterator).flatMap { case (lm, sc) =>
          // TODO: Handle Longs vs. Ints correctly here
          val matRange = sc.toInt until (sc.toInt + lm.mat.cols)
          if (matRange.contains(colRange.start) || colRange.contains(sc.toInt)) {
            // The end row is min of number of rows in this partition
            // and number of rows left to read
            val start = (math.max(rowRange.start - sc, 0)).toInt
            val end = (math.min(rowRange.end - sc, lm.mat.rows)).toInt
            Iterator(lm.mat(rowRange, start until end))
          } else {
            Iterator()
          }
        }
      } else {
        Iterator()
      }
    })
  }

  override def cache() = {
    rdd.cache()
    this
  }

  // TODO: This is terribly inefficient if we have more partitions.
  // Make this more efficient
  override def collect(): DenseMatrix[Double] = {
    val parts = rdd.map(x => x.mat).collect()
    parts.reduceLeftOption((a,b) => DenseMatrix.horzcat(a, b)).getOrElse(new DenseMatrix[Double](0, 0))
  }

  def qrR(): DenseMatrix[Double] = ???

  // Estimate the condition number of the matrix
  // Optionally pass in a R that correspondings to the R matrix obtained
  // by a QR decomposition
  def condEst(rOpt: Option[DenseMatrix[Double]] = None): Double = {
    val R = rOpt match {
      case None => qrR()
      case Some(rMat) => rMat
    }
    val n = R.rows
    val work = new Array[Double](3*n)
    val iwork = new Array[Int](n)
    val rcond = new doubleW(0)
    val info = new intW(0)
    lapack.dtrcon("1", "U", "n", n, R.data, n, rcond, work, iwork, info)
    1/(rcond.`val`)
  }

  // Apply a function to each partition of the matrix
  def mapPartitions(f: DenseMatrix[Double] => DenseMatrix[Double]) = {
    // TODO: This can be efficient if we don't change num rows per partition
    ColumnPartitionedMatrix.fromMatrix(rdd.map { lm =>
      f(lm.mat)
    })
  }
}

object ColumnPartitionedMatrix {

  // Convert an RDD[DenseMatrix[Double]] to an RDD[RowPartition]
  def fromMatrix(matrixRDD: RDD[DenseMatrix[Double]]): ColumnPartitionedMatrix = {
    new ColumnPartitionedMatrix(matrixRDD.map(mat => ColumnPartition(mat)))
  }

  def fromArray(matrixRDD: RDD[Array[Double]]): ColumnPartitionedMatrix = {
    fromMatrix(arrayToMatrix(matrixRDD))
  }

  def fromArray(
                   matrixRDD: RDD[Array[Double]],
                   rowsPerPartition: Seq[Int],
                   cols: Int): RowPartitionedMatrix = {
    new RowPartitionedMatrix(
      arrayToMatrix(matrixRDD, rowsPerPartition, cols).map(mat => RowPartition(mat)),
      Some(rowsPerPartition.sum), Some(cols))
  }

  /**
   * Note, this assumes that matrixRDD is a matrix of *columns*.
   * @param matrixRDD
   * @param colsPerPartition
   * @param rows
   * @return
   */
  def arrayToMatrix(
                       matrixRDD: RDD[Array[Double]],
                       colsPerPartition: Seq[Int],
                       rows: Int) = {
    val cBroadcast = matrixRDD.context.broadcast(colsPerPartition)
    val data = matrixRDD.mapPartitionsWithIndex { case (part, iter) =>
      val cols = cBroadcast.value(part)
      val matData = new Array[Double](rows * cols)

      var idx = 0
      while (iter.hasNext) {
        val arr = iter.next()
        var subidx = 0
        while(subidx < arr.size) {
          matData(idx) = arr(subidx)
          subidx += 1
          idx += 1
        }
      }

      Iterator(new DenseMatrix[Double](rows, cols, matData.toArray))
    }
    data
  }

  /**
   * Note, this assumes that matrixRDD is a matrixo of *columns*.
   * @param matrixRDD
   * @return
   */
  def arrayToMatrix(matrixRDD: RDD[Array[Double]]): RDD[DenseMatrix[Double]] = {
    val rowsColsPerPartition = matrixRDD.mapPartitionsWithIndex { case (part, iter) =>
      if (iter.hasNext) {
        val nRows = iter.next().size
        Iterator((part, nRows, 1 + iter.size))
      } else {
        Iterator((part, 0, 0))
      }
    }.collect().sortBy(x => (x._1, x._2, x._3)).map(x => (x._1, (x._2, x._3))).toMap

    val rcBroadcast = matrixRDD.context.broadcast(rowsColsPerPartition)

    val data = matrixRDD.mapPartitionsWithIndex { case (part, iter) =>
      val (rows, cols) = rcBroadcast.value(part)
      val matData = new Array[Double](rows * cols)

      var idx = 0
      while (iter.hasNext) {
        val arr = iter.next()
        var subidx = 0
        while(subidx < arr.size) {
          matData(idx) = arr(subidx)
          subidx += 1
          idx += 1
        }
      }

      Iterator(new DenseMatrix[Double](rows, cols, matData.toArray))
    }
    data
  }

  def createRandom(sc: SparkContext,
                   numRows: Int,
                   numCols: Int,
                   numParts: Int,
                   cache: Boolean = true): ColumnPartitionedMatrix = {
    val colsPerPart = numCols / numParts
    val matrixParts = sc.parallelize(1 to numParts, numParts).mapPartitions { part =>
      val data = new Array[Double](numRows * colsPerPart)
      var i = 0
      while (i < numRows * colsPerPart) {
        data(i) = ThreadLocalRandom.current().nextDouble()
        i = i + 1
      }
      val mat = new DenseMatrix[Double](numRows, colsPerPart, data)
      Iterator(mat)
    }
    if (cache) {
      matrixParts.cache()
    }
    ColumnPartitionedMatrix.fromMatrix(matrixParts)
  }

}