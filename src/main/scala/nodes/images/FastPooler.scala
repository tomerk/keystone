package nodes.images

import breeze.linalg.DenseVector
import pipelines._
import utils._
import workflow.Transformer

/**
  * This node takes an image and performs pooling on regions of the image.
  *
  * Divides images into fixed size pools, but when fed with images of various
  * sizes may produce a varying number of pools.
  *
  * Assumes sum pooling and identity transformation.
  *
  * NOTE: By default strides start from poolSize/2.
  *
  * @param stride x and y stride to get regions of the image
  * @param poolSize size of the patch to perform pooling on
  */
class FastPooler(
              stride: Int,
              poolSize: Int,
              maxVal: Double,
              alpha: Double,
              imageMeta: ImageMetadata)
  extends Transformer[RowMajorArrayVectorizedImage, Image] {

  val strideStart = poolSize / 2

  //FIXME: this currently assumes x and y dimensions are identical.
  /*
  val poolStarts = (strideStart until imageMeta.xDim by stride)
  val poolRanges = poolStarts.map(x => (x-poolSize/2, math.min(x + poolSize / 2, imageMeta.xDim))).zipWithIndex

  def determineOutputBuckets(coord: Int): Seq[Int] = {
    poolRanges.filter{ case ((b,e),i) => coord >= b && coord < e}.map(_._2)
  }
  */
  val xDim = imageMeta.xDim
  val yDim = imageMeta.yDim
  val numChannels = imageMeta.numChannels

  val numPoolsX = math.ceil((xDim - strideStart).toDouble / stride).toInt
  val numPoolsY = math.ceil((yDim - strideStart).toDouble / stride).toInt
  val outmeta = ImageMetadata(numPoolsX, numPoolsY, numChannels*2)


  def apply(image: RowMajorArrayVectorizedImage): RowMajorArrayVectorizedImage = {
    val outputImage = RowMajorArrayVectorizedImage(Array.fill(numPoolsX*numPoolsY*numChannels*2)(0.0), outmeta)

    var x, y, c, xp, yp = 0
    while (c < numChannels) {
      y = 0
      while (y < yDim) {
        val ypool = y / poolSize
        x = 0
        while(x < xDim) {
          //Determine which pools x and y belong in.
          val xpool = x / poolSize
          val pix = image.get(x,y,c)
          val upval = math.max(maxVal, pix-alpha)
          val downval = math.max(maxVal, -pix-alpha)

          outputImage.put(xpool,ypool, c, outputImage.get(xpool,ypool,c)+upval)
          outputImage.put(xpool,ypool, c+1, outputImage.get(xpool,ypool,c+1)+downval)

          x+=1
        }
        y+=1
      }
      c+=2
    }

    outputImage
  }
}