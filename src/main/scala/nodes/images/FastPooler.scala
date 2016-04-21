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
              poolStride: Int,
              poolSize: Int,
              maxVal: Double=0.0,
              alpha: Double=0.0,
              imageMeta: ImageMetadata)
  extends Transformer[Image, Image] {

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

  val numPoolsX = math.ceil((xDim - strideStart).toDouble / poolStride).toInt
  val numPoolsY = math.ceil((yDim - strideStart).toDouble / poolStride).toInt
  val outmeta = ImageMetadata(numPoolsX, numPoolsY, numChannels*2)


  def apply(image: Image): Image = {
    val outputImage = RowMajorArrayVectorizedImage(Array.fill(numPoolsX*numPoolsY*numChannels*2)(0.0), outmeta)

    var x, y, c, xp, yp = 0
    while (c < numChannels) {
      y = 0
      while (y < yDim) {
        val ypool = y / poolStride

        val yPoolStart = ypool*poolStride
        val yDoubleCount = (poolSize > poolStride && y > poolStride && y - yPoolStart < poolSize-poolStride)

        x = 0
        while(x < xDim) {
          //Determine which pools x and y belong in.
          val xpool = x / poolStride
          val xPoolStart = xpool*poolStride
          val xDoubleCount = (poolSize > poolStride && x > poolStride & x - xPoolStart < poolSize-poolStride)

          val pix = image.get(x,y,c)
          val upval = math.max(maxVal, pix-alpha)
          val downval = math.max(maxVal, -pix - alpha)

          outputImage.put(xpool,ypool, c, outputImage.get(xpool,ypool,c)+upval)
          outputImage.put(xpool,ypool, c+1, outputImage.get(xpool,ypool,c+1)+downval)

          //In the event that 2*poolStride > poolSize > poolStride, we must do border handling.
          if (xDoubleCount) {
            outputImage.put(xpool-1,ypool, c, outputImage.get(xpool-1, ypool, c)+upval)
            outputImage.put(xpool-1,ypool, c+1, outputImage.get(xpool-1, ypool, c+1)+downval)

            if (yDoubleCount) {
              outputImage.put(xpool-1,ypool-1, c, outputImage.get(xpool-1, ypool-1, c)+upval)
              outputImage.put(xpool-1,ypool-1, c+1, outputImage.get(xpool-1, ypool-1, c+1)+downval)
            }
          }

          if (yDoubleCount) {
            outputImage.put(xpool,ypool-1, c+1, outputImage.get(xpool, ypool-1, c+1)+upval)
            outputImage.put(xpool,ypool-1, c+1, outputImage.get(xpool, ypool-1, c+1)+downval)
          }

          x+=1
        }
        y+=1
      }
      c+=2
    }

    outputImage
  }
}