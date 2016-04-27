package nodes.images.external

import breeze.linalg._
import nodes.images.SIFTExtractorInterface
import org.apache.spark.rdd.RDD
import utils._
import utils.external.VLFeat
import workflow.Transformer

/**
  * Performs pooling on images using native code.
  *
  */
class NativePooler(val poolStride: Int, val poolSize: Int, val maxVal: Double = 0.0, val alpha: Double = 0.25, val meta: ImageMetadata)
  extends Transformer[RowMajorArrayVectorizedImage, Image] {
  @transient lazy val extLib = new external.NativePooler()

  def apply(in: RowMajorArrayVectorizedImage): Image = {
    val rawDescDataShort = extLib.pool(meta.xDim, meta.yDim, meta.numChannels, poolStride, poolSize, maxVal, alpha, in.toArray)

    new RowMajorArrayVectorizedImage(rawDescDataShort, ImageMetadata(2, 2, meta.numChannels))
  }
}

object NativePooler {
  def apply(poolStride: Int, poolSize: Int, maxVal: Double = 0.0, alpha: Double = 0.25, meta: ImageMetadata) = {
    new NativePooler(poolStride, poolSize, maxVal, alpha, meta)
  }
}
