package nodes.images

import breeze.linalg._
import breeze.math.Complex
import breeze.signal.{fft, ifft}
import nodes.learning.ZCAWhitener
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import pipelines._
import utils.{ChannelMajorArrayVectorizedImage, ImageMetadata, _}
import workflow.Transformer
import utils.Image

object FFTConvolver extends Logging {
  /**
   * User-friendly constructor interface for the Conovler.
   *
   * @param filters An array of images with which we convolve each input image. These images should *not* be pre-whitened.
   * @param flipFilters Should the filters be flipped before convolution is applied (used for comparability to MATLAB's
   *                    convnd function.)
   */
  def apply(filters: Array[Image],
            flipFilters: Boolean = false): FFTConvolver = {

    //If we are told to flip the filters, invert their indexes.
    val filterImages = if (flipFilters) {
      filters.map(ImageUtils.flipImage)
    } else filters

    //Pack the filter array into a dense matrix of the right format.
    val packedFilters = Convolver.packFilters(filterImages)

    FFTConvolver(packedFilters, filters.head.metadata.xDim, filters.head.metadata.numChannels)
  }
}


/**
 * An FFT-based convolution
 */
case class FFTConvolver(filters: DenseMatrix[Double], filterSize: Int, numChannels: Int) extends Transformer[Image, Image] {

  // TODO FIXME: This is a buggy unpacking of filters!!!
  val imgFilters = filters.t
    .toArray
    .grouped(filterSize*filterSize*numChannels)
    .map(x => new RowMajorArrayVectorizedImage(x, ImageMetadata(filterSize, filterSize, numChannels)))
    .toArray

  def apply(data: Image): Image = {
    convolve2dFft(data, imgFilters)
  }

  def convolve2dFfts(x: DenseMatrix[Complex], m: DenseMatrix[Complex], origRows: Int, origCols: Int) = {
    //This code based (loosely) on the MATLAB Code here:
    //Assumes you've already computed the fft of x and m.
    //http://www.mathworks.com/matlabcentral/fileexchange/31012-2-d-convolution-using-the-fft

    //Step 1: Pure fucking magic.
    val res = ifft(x :* m).map(_.real)

    //Step 2: the output we care about is in the bottom right corner.
    val startr = origRows - 1
    val startc = origCols - 1
    res(startr until x.rows, startc until x.cols).copy
  }

  def getChannelMatrix(in: Image, c: Int): DenseMatrix[Double] = {
    val dat = (0 until in.metadata.yDim).flatMap(y => (0 until in.metadata.xDim).map(x => in.get(x, y, c))).toArray
    new DenseMatrix[Double](in.metadata.xDim, in.metadata.yDim, dat)
  }

  def addChannelMatrix(in: Image, c: Int, m: DenseMatrix[Double]) = {
    var x = 0
    while (x < in.metadata.xDim) {
      var y = 0
      while (y < in.metadata.yDim) {
        in.put(x, y, c, in.get(x, y, c) + m(x, y))
        y += 1
      }
      x += 1
    }
  }

  def padMat(m: DenseMatrix[Double], nrows: Int, ncols: Int): DenseMatrix[Double] = {
    val res = DenseMatrix.zeros[Double](nrows, ncols)
    res(0 until m.rows, 0 until m.cols) := m
    res
  }

  /**
   * Convolves an n-dimensional image with a k-dimensional
   *
   * @param x
   * @param m
   * @return
   */
  def convolve2dFft(x: Image, m: Array[_ <: Image]): Image = {
    val mx = x.metadata.xDim - m.head.metadata.xDim + 1
    val my = x.metadata.yDim - m.head.metadata.yDim + 1
    val chans = x.metadata.numChannels

    val ressize = mx * my * m.length

    val res = new RowMajorArrayVectorizedImage(Array.fill(ressize)(0.0), ImageMetadata(mx, my, m.length))

    val start = System.currentTimeMillis
    val fftXs = (0 until chans).map(c => fft(getChannelMatrix(x, c)))
    val fftMs = (0 until m.length).map(f => (0 until chans).map(c => fft(padMat(getChannelMatrix(m(f), c), x.metadata.xDim, x.metadata.yDim)))).toArray

    //logInfo(s"Length of Xs: ${fftXs.length}, Length of each m: ${fftMs.first.length}, Total ms: ${fftMs.length}")
    var c = 0
    while (c < chans) {
      var f = 0
      while (f < m.length) {
        val convBlock = convolve2dFfts(fftXs(c), fftMs(f)(c), m(f).metadata.xDim, m(f).metadata.yDim)
        addChannelMatrix(res, f, convBlock) //todo - this could be vectorized.
        f += 1
      }
      c += 1
    }
    res
  }
}

object LoopConvolver {
  /**
   * User-friendly constructor interface for the Conovler.
   *
   * @param filters An array of images with which we convolve each input image. These images should *not* be pre-whitened.
   * @param flipFilters Should the filters be flipped before convolution is applied (used for comparability to MATLAB's
   *                    convnd function.)
   */
  def apply(filters: Array[Image],
            flipFilters: Boolean = false): LoopConvolver = {

    //If we are told to flip the filters, invert their indexes.
    val filterImages = if (flipFilters) {
      filters.map(ImageUtils.flipImage)
    } else filters

    //Pack the filter array into a dense matrix of the right format.
    val packedFilters = Convolver.packFilters(filterImages)

    LoopConvolver(packedFilters)
  }
}

/**
 * A zero-padded convolution
 */
case class LoopConvolver(filters: DenseMatrix[Double]) extends Transformer[Image, Image] {

  val convolutions = filters.t

  def apply(data: Image): Image = {
    val imgWidth = data.metadata.xDim
    val imgHeight = data.metadata.yDim
    val imgChannels = data.metadata.numChannels

    val convSize = math.sqrt(filters.cols/imgChannels).toInt
    val numConvolutions = convolutions.cols

    val resWidth = imgWidth - convSize + 1
    val resHeight = imgHeight - convSize + 1

    val convRes = DenseMatrix.zeros[Double](resWidth*resHeight, numConvolutions)
    var x, y, channel, convNum, convX, convY, cell = 0
    while (convNum < numConvolutions) {
      convX = 0
      while (convX < convSize) {
        convY = 0
        while (convY < convSize) {
          channel = 0
          while (channel < imgChannels) {
            val kernelPixel = convolutions(channel+convX*imgChannels+convY*imgChannels*convSize, convNum)
            y = 0
            while (y < resHeight) {
              x = 0
              while (x < resWidth) {
                val imgPixel = data.get(x + convX, y + convY, channel)
                convRes(y*resWidth+x, convNum) += kernelPixel*imgPixel
                x += 1
              }
              y += 1
            }
            channel += 1
          }
          convY += 1
        }
        convX += 1
      }
      convNum += 1
    }

    val res = new RowMajorArrayVectorizedImage(
      convRes.data,
      ImageMetadata(resWidth, resHeight, numConvolutions))

    res
  }
}

/**
 * Convolves images with a bank of convolution filters. Convolution filters must be square.
 * Used for using the same label for all patches from an image.
 * TODO: Look into using Breeze's convolve
 *
 * @param filters Bank of convolution filters to apply - each filter is an array in row-major order.
 * @param imgWidth Width of images in pixels.
 * @param imgHeight Height of images in pixels.
 */
class Convolver(
    filters: DenseMatrix[Double],
    imgWidth: Int,
    imgHeight: Int,
    imgChannels: Int,
    whitener: Option[ZCAWhitener] = None,
    normalizePatches: Boolean = true,
    varConstant: Double = 10.0)
  extends Transformer[Image, Image] {

  val convSize = math.sqrt(filters.cols/imgChannels).toInt
  val convolutions = filters.t

  val resWidth = imgWidth - convSize + 1
  val resHeight = imgHeight - convSize + 1

  override def apply(in: RDD[Image]): RDD[Image] = {
    in.mapPartitions(Convolver.convolvePartitions(_, resWidth, resHeight, imgChannels, convSize,
      normalizePatches, whitener, convolutions, varConstant))
  }

  def apply(in: Image): Image = {
    var patchMat = new DenseMatrix[Double](resWidth*resHeight, convSize*convSize*imgChannels)
    Convolver.convolve(in, patchMat, resWidth, resHeight,
      imgChannels, convSize, normalizePatches, whitener, convolutions)
  }
}

object Convolver {
  /**
    * User-friendly constructor interface for the Conovler.
    *
    * @param filters An array of images with which we convolve each input image. These images should *not* be pre-whitened.
    * @param imgInfo Metadata of a typical image we will be convolving. All images must have the same size/shape.
    * @param whitener Whitener to be applied to both the input images and the filters before convolving.
    * @param normalizePatches Should the patches be normalized before convolution?
    * @param varConstant Constant to be used in scaling.
    * @param flipFilters Should the filters be flipped before convolution is applied (used for comparability to MATLAB's
    *                    convnd function.)
    */
  def apply(filters: Array[Image],
           imgInfo: ImageMetadata,
           whitener: Option[ZCAWhitener] = None,
           normalizePatches: Boolean = true,
           varConstant: Double = 10.0,
           flipFilters: Boolean = false) = {

    //If we are told to flip the filters, invert their indexes.
    val filterImages = if (flipFilters) {
      filters.map(ImageUtils.flipImage)
    } else filters

    //Pack the filter array into a dense matrix of the right format.
    val packedFilters = packFilters(filterImages)

    //If the whitener is not empty, construct a new one:
    val whitenedFilterMat = whitener match {
      case Some(x) => x.apply(packedFilters) * x.whitener.t
      case None => packedFilters
    }

    new Convolver(
      whitenedFilterMat,
      imgInfo.xDim,
      imgInfo.yDim,
      imgInfo.numChannels,
      whitener,
      normalizePatches,
      varConstant)
  }

  /**
    * Given an array of filters, packs the filters into a DenseMatrix[Double] which has the following form:
    * for a row i, column c+y*numChannels+x*numChannels*yDim corresponds to the pixel value at (x,y,c) in image i of
    * the filters array.
    *
    * @param filters Array of filters.
    * @return DenseMatrix of filters, as described above.
    */
  def packFilters(filters: Array[Image]): DenseMatrix[Double] = {
    val (xDim, yDim, numChannels) = (filters(0).metadata.xDim, filters(0).metadata.yDim, filters(0).metadata.numChannels)
    val filterSize = xDim*yDim*numChannels
    val res = DenseMatrix.zeros[Double](filters.length, filterSize)

    var i,x,y,c = 0
    while(i < filters.length) {
      x = 0
      while(x < xDim) {
        y = 0
        while(y < yDim) {
          c = 0
          while (c < numChannels) {
            val rc = c + x*numChannels + y*numChannels*xDim
            res(i, rc) = filters(i).get(x,y,c)

            c+=1
          }
          y+=1
        }
        x+=1
      }
      i+=1
    }

    res
  }


  def convolve(img: Image,
      patchMat: DenseMatrix[Double],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      convSize: Int,
      normalizePatches: Boolean,
      whitener: Option[ZCAWhitener],
      convolutions: DenseMatrix[Double],
      varConstant: Double = 10.0): Image = {

    val imgMat = makePatches(img, patchMat, resWidth, resHeight, imgChannels, convSize,
      normalizePatches, whitener, varConstant)

    val convRes: DenseMatrix[Double] = imgMat * convolutions

    val res = new RowMajorArrayVectorizedImage(
      convRes.toArray,
      ImageMetadata(resWidth, resHeight, convolutions.cols))

    res
  }

  /**
   * This function takes an image and generates a matrix of all of its patches. Patches are expected to have indexes
   * of the form: c + x*numChannels + y*numChannels*xDim
   *
   * @param img
   * @return
   */
  def makePatches(img: Image,
      patchMat: DenseMatrix[Double],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      convSize: Int,
      normalizePatches: Boolean,
      whitener: Option[ZCAWhitener],
      varConstant: Double): DenseMatrix[Double] = {
    var x,y,chan,pox,poy,py,px = 0

    poy = 0
    while (poy < convSize) {
      pox = 0
      while (pox < convSize) {
        y = 0
        while (y < resHeight) {
          x = 0
          while (x < resWidth) {
            chan = 0
            while (chan < imgChannels) {
              px = chan + pox*imgChannels + poy*imgChannels*convSize
              py = x + y*resWidth

              patchMat(py, px) = img.get(x+pox, y+poy, chan)

              chan+=1
            }
            x+=1
          }
          y+=1
        }
        pox+=1
      }
      poy+=1
    }

    val patchMatN = if(normalizePatches) Stats.normalizeRows(patchMat, varConstant) else patchMat

    val res = whitener match {
      case None => patchMatN
      case Some(whiteness) => patchMatN(*, ::) - whiteness.means
    }

    res
  }

  def convolvePartitions(
      imgs: Iterator[Image],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      convSize: Int,
      normalizePatches: Boolean,
      whitener: Option[ZCAWhitener],
      convolutions: DenseMatrix[Double],
      varConstant: Double): Iterator[Image] = {

    var patchMat = new DenseMatrix[Double](resWidth*resHeight, convSize*convSize*imgChannels)
    imgs.map(convolve(_, patchMat, resWidth, resHeight, imgChannels, convSize, normalizePatches,
      whitener, convolutions, varConstant))

  }
}
