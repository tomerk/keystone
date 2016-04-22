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
  * @param poolStride x and y stride to get regions of the image
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

  val xPoolStarts = (strideStart until imageMeta.xDim by poolStride)
  val xPoolRanges = xPoolStarts.map(x => (x-poolSize/2, math.min(x + poolSize / 2, imageMeta.xDim))).zipWithIndex
  val yPoolStarts = (strideStart until imageMeta.yDim by poolStride)
  val yPoolRanges = yPoolStarts.map(y => (y-poolSize/2, math.min(y + poolSize / 2, imageMeta.yDim))).zipWithIndex

  def determineOutputBuckets(coord: Int, ranges: Seq[((Int,Int), Int)]): Seq[Int] = {
    ranges.filter{ case ((b,e),i) => coord >= b && coord < e}.map(_._2)
  }
  val xPools = (0 until imageMeta.xDim).map(c => determineOutputBuckets(c, xPoolRanges)).toArray.map(_.head)
  val yPools = (0 until imageMeta.yDim).map(c => determineOutputBuckets(c, yPoolRanges)).toArray.map(_.head)
  println(xPools.mkString(","))
  println(yPools.mkString(","))


  val xDim = imageMeta.xDim
  val yDim = imageMeta.yDim
  val numChannels = imageMeta.numChannels

  val numPoolsX = math.ceil((xDim - strideStart).toDouble / poolStride).toInt
  val numPoolsY = math.ceil((yDim - strideStart).toDouble / poolStride).toInt
  val outmeta = ImageMetadata(numPoolsX, numPoolsY, numChannels*2)




  def apply(image: Image): Image = {
    val outputImage = RowMajorArrayVectorizedImage(Array.fill(numPoolsX*numPoolsY*numChannels*2)(0.0), outmeta)

    var x, y, c, xp, yp, xPool, yPool = 0
    var i = 0
    while (c < numChannels) {
      y = 0
      while (y < yDim) {
        x = 0
        yPool = yPools(y)
        while(x < xDim) {

          //Do symmetric rectification
          val pix = image.get(x,y,c)
          val upval = math.max(maxVal, pix-alpha)
          val downval = math.max(maxVal, -pix - alpha)

//          //Put the pixel in all appropriate pools
//          yp = 0
//          while (yp < yPools(y).length) {
//            yPool = yPools(y)(yp)
//
//            xp = 0
//            while (xp < xPools(x).length) {
//              xPool = xPools(x)(xp)
//              outputImage.put(xPool,yPool,2*c, outputImage.get(xPool,yPool,2*c)+upval)
//              outputImage.put(xPool,yPool,2*c+1, outputImage.get(xPool,yPool,2*c+1)+downval)
//              i+=1
//
//              xp+=1
//            }
//            yp+=1
//          }
          xPool = xPools(x)

          outputImage.put(xPool,yPool,2*c, outputImage.get(xPool,yPool,2*c)+upval)
          outputImage.put(xPool,yPool,2*c+1, outputImage.get(xPool,yPool,2*c+1)+downval)

          x+=1
        }
        y+=1
      }
      c+=1
    }

    outputImage
  }
}