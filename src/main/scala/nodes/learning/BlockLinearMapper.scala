package nodes.learning

import breeze.linalg._
import edu.berkeley.cs.amplab.mlmatrix.{RowPartition, NormalEquations, BlockCoordinateDescent, RowPartitionedMatrix}
import nodes.stats.{StandardScalerModel, StandardScaler}
import org.apache.spark.rdd.RDD
import nodes.util.{VectorSplitter, Identity}
import utils.{MatrixUtils, Stats}
import workflow.{Transformer, LabelEstimator}


/**
 * Transformer that applies a linear model to an input.
 * Different from [[LinearMapper]] in that the matrix representing the transformation
 * is split into a seq.
 * @param xs  The chunks of the matrix representing the linear model
 * @param blockSize blockSize to split data before applying transformations
 * @param bOpt optional intercept term to be added
 * @param featureScalersOpt optional seq of transformers to be applied before transformation
 */
class BlockLinearMapper[T <: Vector[Double]](
    val xs: Seq[DenseMatrix[Double]],
    val blockSize: Int,
    val bOpt: Option[DenseVector[Double]] = None,
    val featureScalersOpt: Option[Seq[Transformer[DenseVector[Double], DenseVector[Double]]]] = None)
  extends Transformer[T, DenseVector[Double]] {

  // Use identity nodes if we don't need to do scaling
  val featureScalers = featureScalersOpt.getOrElse(
    Seq.fill(xs.length)(new Identity[DenseVector[Double]]))
  val vectorSplitter = new VectorSplitter[T](blockSize)

  /**
   * Applies the linear model to feature vectors large enough to have been split into several RDDs.
   * @param in RDD of vectors to apply the model to
   * @return the output vectors
   */
  override def apply(in: RDD[T]): RDD[DenseVector[Double]] = {
    apply(vectorSplitter(in))
  }

  /**
   * Applies the linear model to feature vectors large enough to have been split into several RDDs.
   * @param ins RDD of vectors to apply the model to, split into same size as model blocks
   * @return the output vectors
   */
  def apply(in: Seq[RDD[DenseVector[Double]]]): RDD[DenseVector[Double]] = {
    val res = in.zip(xs.zip(featureScalers)).map {
      case (rdd, xScaler) => {
        val (x, scaler) = xScaler
        val modelBroadcast = rdd.context.broadcast(x)
        scaler(rdd).mapPartitions(rows => {
          if (!rows.isEmpty) {
            Iterator.single(MatrixUtils.rowsToMatrix(rows) * modelBroadcast.value)
          } else {
            Iterator.empty
          }
        })
      }
    }

    val matOut = res.reduceLeft((sum, next) => sum.zip(next).map(c => c._1 + c._2))

    // Add the intercept here
    val bBroadcast = matOut.context.broadcast(bOpt)
    val matOutWithIntercept = matOut.map { mat =>
      bOpt.map { b =>
        mat(*, ::) :+= b
        mat
      }.getOrElse(mat)
    }

    matOutWithIntercept.flatMap(x => MatrixUtils.matrixToRowArray(x))
  }

  override def apply(in: T): DenseVector[Double] = {
    val res = vectorSplitter.splitVector(in).zip(xs.zip(featureScalers)).map {
      case (in, xScaler) => {
        xScaler._1.t * xScaler._2(in)
      }
    }

    val out = res.reduceLeft((sum, next) => sum + next)
    bOpt.map { b =>
      out += b
      out
    }.getOrElse(out)
  }

  /**
   * Applies the linear model to feature vectors. After processing chunk i of every vector, applies
   * @param evaluator to the intermediate output vector.
   * @param in input RDD
   */
  def applyAndEvaluate(in: RDD[T], evaluator: (RDD[DenseVector[Double]]) => Unit) {
    applyAndEvaluate(vectorSplitter(in), evaluator)
  }

  /**
   * Applies the linear model to feature vectors. After processing chunk i of every vector, applies
   * @param evaluator to the intermediate output vector.
   * @param in sequence of input RDD chunks
   */
  def applyAndEvaluate(
      in: Seq[RDD[DenseVector[Double]]],
      evaluator: (RDD[DenseVector[Double]]) => Unit) {
    val res = in.zip(xs.zip(featureScalers)).map {
      case (rdd, xScaler) => {
        val modelBroadcast = rdd.context.broadcast(xScaler._1)
        xScaler._2(rdd).mapPartitions(rows => {
          val out = MatrixUtils.rowsToMatrix(rows) * modelBroadcast.value
          Iterator.single(out)
        })
      }
    }

    var prev: Option[RDD[DenseMatrix[Double]]] = None
    for (next <- res) {
      val sum = prev match {
        case Some(prevVal) => prevVal.zip(next).map(c => c._1 + c._2).cache()
        case None => next.cache()
      }

      // NOTE: We should only add the intercept once. So do it right before
      // we call the evaluator but don't cache this
      val sumAndIntercept = sum.map { mat =>
        bOpt.map { b =>
          mat(*, ::) :+= b
          mat
        }.getOrElse(mat)
      }
      evaluator.apply(sumAndIntercept.flatMap(x => MatrixUtils.matrixToRowArray(x)))
      prev.map(_.unpersist())
      prev = Some(sum)
    }
    prev.map(_.unpersist())
  }
}

object BlockLeastSquaresEstimator {

  def computeCost(
      trainingFeatures: Seq[RDD[DenseVector[Double]]],
      trainingLabels: RDD[DenseVector[Double]],
      lambda: Double,
      xs: Seq[DenseMatrix[Double]],
      bOpt: Option[DenseVector[Double]]): Double = {

    val nTrain = trainingLabels.count

    val res = trainingFeatures.zip(xs).map {
      case (rdd, x) => {
        val modelBroadcast = rdd.context.broadcast(x)
        rdd.mapPartitions(rows => {
          if (!rows.isEmpty) {
            Iterator.single(MatrixUtils.rowsToMatrix(rows) * modelBroadcast.value)
          } else {
            Iterator.empty
          }
        })
      }
    }

    val matOut = res.reduceLeft((sum, next) => sum.zip(next).map(c => c._1 + c._2))

    // Add the intercept here
    val bBroadcast = matOut.context.broadcast(bOpt)
    val matOutWithIntercept = matOut.map { mat =>
      bOpt.map { b =>
        mat(*, ::) :+= b
        mat
      }.getOrElse(mat)
    }

    val axb = matOutWithIntercept.flatMap(x => MatrixUtils.matrixToRowArray(x))

    val cost = axb.zip(trainingLabels).map { part =>
      val axb = part._1
      val labels = part._2
      val out = axb - labels
      math.pow(norm(out), 2)
    }.reduce(_ + _)

    if (lambda == 0) {
      cost/(2.0*nTrain.toDouble)
    } else {
      val wNorm = xs.map(part => math.pow(norm(part.toDenseVector), 2)).reduce(_+_)
      cost/(2.0*nTrain.toDouble) + lambda/2.0 * wNorm
    }

  }

}

/**
 * Fits a least squares model using block coordinate descent with provided
 * training features and labels
 * @param blockSize size of block to use in the solver
 * @param numIter number of iterations of solver to run
 * @param lambda L2-regularization to use
 */
class BlockLeastSquaresEstimator[T <: Vector[Double]](blockSize: Int, numIter: Int, lambda: Double = 0.0, numFeaturesOpt: Option[Int] = None)
  extends SolverWithCostModel[T] {

  /**
   * Fit a model using blocks of features and labels provided.
   *
   * @param trainingFeatures feature blocks to use in RDDs.
   * @param trainingLabels RDD of labels to use.
   */
  def fit(
      trainingFeatures: Seq[RDD[DenseVector[Double]]],
      trainingLabels: RDD[DenseVector[Double]]): BlockLinearMapper[T] = {
    val labelScaler = new StandardScaler(normalizeStdDev = false).fit(trainingLabels)
    // Find out numRows, numCols once
    val b = RowPartitionedMatrix.fromArray(
      labelScaler.apply(trainingLabels).map(_.toArray)).cache()
    val numRows = Some(b.numRows())
    val numCols = Some(blockSize.toLong)

    // NOTE: This will cause trainingFeatures to be evaluated twice
    // which might not be optimal if its not cached ?
    val featureScalers = trainingFeatures.map { rdd =>
      new StandardScaler(normalizeStdDev = false).fit(rdd)
    }

    val A = trainingFeatures.zip(featureScalers).map { case (rdd, scaler) =>
      new RowPartitionedMatrix(scaler.apply(rdd).mapPartitions { rows =>
        Iterator.single(MatrixUtils.rowsToMatrix(rows))
      }.map(RowPartition), numRows, numCols)
    }

    val bcd = new BlockCoordinateDescent()
    val models = if (numIter > 1) {
      bcd.solveLeastSquaresWithL2(
        A, b, Array(lambda), numIter, new NormalEquations()).transpose
    } else {
      bcd.solveOnePassL2(A.iterator, b, Array(lambda), new NormalEquations()).toSeq.transpose
    }

    new BlockLinearMapper(models.head, blockSize, Some(labelScaler.mean), Some(featureScalers))
  }

  /**
   * Fit a model after splitting training data into appropriate blocks.
   *
   * @param trainingFeatures training data to use in one RDD.
   * @param trainingLabels labels for training data in a RDD.
   */
  override def fit(
      trainingFeatures: RDD[T],
      trainingLabels: RDD[DenseVector[Double]]): BlockLinearMapper[T] = {
    val vectorSplitter = new VectorSplitter[T](blockSize, numFeaturesOpt)
    val featureBlocks = vectorSplitter.apply(trainingFeatures)
    fit(featureBlocks, trainingLabels)
  }

  def fit(
      trainingFeatures: RDD[T],
      trainingLabels: RDD[DenseVector[Double]],
      numFeaturesOpt: Option[Int]): BlockLinearMapper[T] = {
    val vectorSplitter = new VectorSplitter[T](blockSize, numFeaturesOpt)
    val featureBlocks = vectorSplitter.apply(trainingFeatures)
    fit(featureBlocks, trainingLabels)
  }

  override def cost(dataProfile: DataProfile, clusterProfile: ClusterProfile): Double = {
    (dataProfile, clusterProfile) match {
      case (DataProfile(n, d, k, sparsity), ClusterProfile(numMachines, cpuWeight, memWeight, networkWeight)) =>
        val flops = n.toDouble * d * (blockSize + k) / numMachines
        val bytesScanned = n.toDouble * d / numMachines + (d.toDouble * k)
        val network = 2.0 * (d.toDouble * (blockSize + k)) * math.log(numMachines)

        numIter * (math.max(cpuWeight * flops, memWeight * bytesScanned) + networkWeight * network)
    }
  }
}
