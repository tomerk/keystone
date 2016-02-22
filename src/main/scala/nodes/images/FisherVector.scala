package nodes.images

import breeze.linalg._
import breeze.numerics._
import breeze.stats.mean
import nodes.learning.{GaussianMixtureModelEstimator, GaussianMixtureModel}
import org.apache.spark.rdd.RDD
import utils.MatrixUtils
import workflow.{OptimizableEstimator, Estimator, Transformer}

/**
 * Abstract interface for Fisher Vector.
 */
trait FisherVectorInterface extends Transformer[DenseMatrix[Float], DenseMatrix[Float]]

/**
 * Implements a fisher vector.
 *
 * @param gmm A trained Gaussian Mixture Model
 */
case class FisherVector(gmm: GaussianMixtureModel)
    extends FisherVectorInterface {

  private val gmmMeans = gmm.means
  private val gmmVars = gmm.variances
  private val gmmWeights = gmm.weights

  /**
   *
   * @param in  matrix of size numSiftDimensions by numSiftDescriptors
   * @return  The output value
   */
  override def apply(in: DenseMatrix[Float]): DenseMatrix[Float] = {
    val nDesc = in.cols.toDouble

    // Get the fisher vector posterior assignments
    val x = convert(in, Double)
    val q = gmm.apply(x.t) // numSiftDescriptors x K

    /* here is the Fisher Vector in all of its beauty.  This is directly
    from the FV survey by Sanchez et al: */
    val s0 = mean(q, Axis._0).toDenseVector // 1 x K, but really K x 1 because it's a dense vector
    val s1 = (x * q) :/= nDesc // D x K
    val s2 = ((x :* x) * q) :/= nDesc // D x K

    val fv1 = (s1 - gmmMeans * diag(s0)) :/ (sqrt(gmmVars) * diag(sqrt(gmmWeights)))
    val fv2 = (s2 - (gmmMeans * 2.0 :* s1) + (((gmmMeans :* gmmMeans) - gmmVars)*diag(s0))) :/
        (gmmVars * diag(sqrt(gmmWeights :* 2.0)))

    // concatenate the two fv terms
    convert(DenseMatrix.horzcat(fv1, fv2), Float)
  }
}

case class GMMFisherVectorEstimator(k: Int) extends Estimator[DenseMatrix[Float], DenseMatrix[Float]] {
  protected def fit(data: RDD[DenseMatrix[Float]]): FisherVector = {
    val gmmTrainingData = data.flatMap(x => convert(MatrixUtils.matrixToColArray(x), Double))
    val gmmEst = new GaussianMixtureModelEstimator(k)
    val gmm = gmmEst.fit(gmmTrainingData)
    FisherVector(gmm)
  }
}

case class OptimizableGMMFisherVectorEstimator(k: Int) extends OptimizableEstimator[DenseMatrix[Float], DenseMatrix[Float]] {
  val default = GMMFisherVectorEstimator(k)

  override def optimize(sample: RDD[DenseMatrix[Float]], numPerPartition: Map[Int, Int]): Estimator[DenseMatrix[Float], DenseMatrix[Float]] = {
    if (k > 16) {
      nodes.images.external.GMMFisherVectorEstimator(k)
    } else {
      GMMFisherVectorEstimator(k)
    }
  }
}