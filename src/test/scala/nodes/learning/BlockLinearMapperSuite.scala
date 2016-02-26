package nodes.learning

import java.io._

import breeze.linalg._
import breeze.stats.distributions.Rand
import scala.collection.mutable.ArrayBuffer

import org.scalatest.FunSuite

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import pipelines._
import nodes.util.VectorSplitter
import utils.{Stats, MatrixUtils, TestUtils}

class BlockLinearMapperSuite extends FunSuite with LocalSparkContext with Logging {

  def loadMatrixRDDs(aMatFile: String, bMatFile: String, numParts: Int, sc: SparkContext) = {
    val aMat = csvread(new File(TestUtils.getTestResourceFileName("aMat.csv")))
    val bMat = csvread(new File(TestUtils.getTestResourceFileName("bMat.csv")))

    val fullARDD = sc.parallelize(MatrixUtils.matrixToRowArray(aMat), numParts).cache()
    val bRDD = sc.parallelize(MatrixUtils.matrixToRowArray(bMat), numParts).cache()
    (fullARDD, bRDD)
  }

  test("BlockLinearMapper transformation") {
    sc = new SparkContext("local", "test")

    val inDims = 1000
    val outDims = 100
    val numChunks = 5
    val numPerChunk = inDims/numChunks

    val mat = DenseMatrix.rand(inDims, outDims, Rand.gaussian)
    val vec = DenseVector.rand(inDims, Rand.gaussian)
    val intercept = DenseVector.rand(outDims, Rand.gaussian)

    val splitVec = (0 until numChunks).map(i => vec((numPerChunk*i) until (numPerChunk*i + numPerChunk)))
    val splitMat = (0 until numChunks).map(i => mat((numPerChunk*i) until (numPerChunk*i + numPerChunk), ::))

    val linearMapper = new LinearMapper(mat, Some(intercept))
    val blockLinearMapper = new BlockLinearMapper(splitMat, numPerChunk, Some(intercept))

    val linearOut = linearMapper(vec)

    // Test with intercept
    assert(Stats.aboutEq(blockLinearMapper(vec), linearOut, 1e-4))

    // Test the apply and evaluate call
    val blmOuts = new ArrayBuffer[RDD[DenseVector[Double]]]
    val splitVecRDDs = splitVec.map { vec =>
      sc.parallelize(Seq(vec), 1)
    }
    blockLinearMapper.applyAndEvaluate(splitVecRDDs,
      (predictedValues: RDD[DenseVector[Double]]) => {
        blmOuts += predictedValues
        ()
      }
    )

    // The last blmOut should match the linear mapper's output
    assert(Stats.aboutEq(blmOuts.last.collect()(0), linearOut, 1e-4))
  }

  test("Binary BCD should match LinearMapper") {
    sc = new SparkContext("local", "test")
    val blockSize = 4
    val numIter = 10
    val lambda = 0.1
    val numParts = 3

    val (fullARDD, bRDD) = loadMatrixRDDs("aMat.csv", "bMat.csv", numParts, sc)

    val nTrain = fullARDD.count.toDouble

    val vectorSplitter = new VectorSplitter(blockSize)
    val featureBlocks = vectorSplitter.apply(fullARDD)
    val firstClassLabels = bRDD.map(x => x(0))

    val model = BinaryBlockCoordinateDescent.trainSingleClassLS(blockSize,
      numIter, lambda, featureBlocks, firstClassLabels)

    val finalFullModel = DenseVector.vertcat(model:_*)

    val aMat = csvread(new File(TestUtils.getTestResourceFileName("aMat.csv")))
    val bMat = csvread(new File(TestUtils.getTestResourceFileName("bMat.csv")))

    val localModel =
      ((((aMat.t * aMat):/nTrain) + (DenseMatrix.eye[Double](aMat.cols) * lambda))) \
        ((aMat.t * bMat(::, 0)) :/nTrain)

    val aTaxb = (aMat.t * (aMat * finalFullModel - bMat(::, 0)))
    val gradient = ( aTaxb :/ nTrain) + (finalFullModel :* lambda)
    println("norm gradient " + norm(gradient))
    assert(norm(gradient) < 1e-2)
  }
}
