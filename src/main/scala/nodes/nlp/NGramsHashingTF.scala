package nodes.nlp

import java.lang.Integer._
import java.lang.Integer.{ rotateLeft => rotl }

import breeze.linalg.SparseVector
import org.apache.spark.rdd.RDD
import workflow.Transformer

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

case class HashingTFNode[T <: Seq[Any]](numFeatures: Int) extends Transformer[T, SparseVector[Double]] {
  def nonNegativeMod(x: Int, mod: Int): Int = {
    val rawMod = x % mod
    rawMod + (if (rawMod < 0) mod else 0)
  }

  def apply(document: T): SparseVector[Double] = {
    val termFrequencies = mutable.HashMap.empty[Int, Double]
    document.foreach { term =>
      val i = nonNegativeMod(term.##, numFeatures)
      termFrequencies.put(i, termFrequencies.getOrElse(i, 0.0) + 1.0)
    }

    SparseVector(numFeatures)(termFrequencies.toSeq:_*)
  }
}

case class NGramsHashingTF(orders: Seq[Int], numFeatures: Int)
    extends Transformer[Seq[String], SparseVector[Double]] {

  private[this] final val minOrder = orders.min
  private[this] final val maxOrder = orders.max

  require(minOrder >= 1, s"minimum order is not >= 1, found $minOrder")
  orders.sliding(2).foreach {
    case xs if xs.length > 1 => require(xs(0) == xs(1) - 1,
      s"orders are not consecutive; contains ${xs(0)} and ${xs(1)}")
    case _ =>
  }

  final val seqSeed = "Seq".hashCode

  /** Mix in a block of data into an intermediate hash value. */
  final def mix(hash: Int, data: Int): Int = {
    var h = mixLast(hash, data)
    h = rotl(h, 13)
    h * 5 + 0xe6546b64
  }

  /** May optionally be used as the last mixing step. Is a little bit faster than mix,
    *  as it does no further mixing of the resulting hash. For the last element this is not
    *  necessary as the hash is thoroughly mixed during finalization anyway. */
  final def mixLast(hash: Int, data: Int): Int = {
    var k = data

    k *= 0xcc9e2d51
    k = rotl(k, 15)
    k *= 0x1b873593

    hash ^ k
  }

  /** Finalize a hash to incorporate the length and make sure all bits avalanche. */
  final def finalizeHash(hash: Int, length: Int): Int = avalanche(hash ^ length)

  /** Force all bits of the hash to avalanche. Used for finalizing the hash. */
  private final def avalanche(hash: Int): Int = {
    var h = hash

    h ^= h >>> 16
    h *= 0x85ebca6b
    h ^= h >>> 13
    h *= 0xc2b2ae35
    h ^= h >>> 16

    h
  }

  def nonNegativeMod(x: Int, mod: Int): Int = {
    val rawMod = x % mod
    rawMod + (if (rawMod < 0) mod else 0)
  }

  def apply(line: Seq[String]): SparseVector[Double] = {
    val hashes = new Array[Integer](line.length)
    var i = 0
    while (i < line.length) {
      hashes(i) = line(i).##
      i += 1
    }

    var j = 0
    var order = 0
    val termFrequencies = mutable.HashMap.empty[Int, Double]
    i = 0
    /*while (i + minOrder <= line.length) {
      ngramBuf.clear()

      j = i
      while (j < i + minOrder) {
        ngramBuf += line(j)
        j += 1
      }
      ngramsBuf += ngramBuf.clone()

      order = minOrder + 1
      while (order <= maxOrder && i + order <= line.length) {
        ngramBuf += line(i + order - 1)
        ngramsBuf += ngramBuf.clone()
        order += 1
      }
      i += 1
    }*/

    SparseVector(numFeatures)(termFrequencies.toSeq:_*)
  }

}
