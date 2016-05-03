package nodes.images

import breeze.linalg.{DenseVector, sum}
import nodes._
import nodes.images.external.NativePooler
import org.scalatest.FunSuite
import pipelines.Logging
import utils.{ChannelMajorArrayVectorizedImage, ImageMetadata, RowMajorArrayVectorizedImage}

class PoolingSuite extends FunSuite with Logging {

  test("pooling") {
    val imgArr =
      (0 until 4).flatMap { x =>
        (0 until 4).flatMap { y =>
          (0 until 1).map { c =>
            (c + x * 1 + y * 4 * 1).toDouble
          }
        }
      }.toArray

    val image = new ChannelMajorArrayVectorizedImage(imgArr, ImageMetadata(4, 4, 1))
    val pooling = new Pooler(2, 2, x => x, x => x.max)

    val poolImage = pooling(image)

    assert(poolImage.get(0, 0, 0) === 5.0)
    assert(poolImage.get(0, 1, 0) === 7.0)
    assert(poolImage.get(1, 0, 0) === 13.0)
    assert(poolImage.get(1, 1, 0) === 15.0)
  }

  test("pooling odd") {
    val hogImgSize = 14
    val convSizes = List(1, 2, 3, 4, 6, 8)
    convSizes.foreach { convSize =>
      val convResSize = hogImgSize - convSize + 1

      val imgArr =
        (0 until convResSize).flatMap { x =>
          (0 until convResSize).flatMap { y =>
            (0 until 1000).map { c =>
              (c + x * 1 + y * 4 * 1).toDouble
            }
          }
        }.toArray

      val image = new ChannelMajorArrayVectorizedImage(
        imgArr, ImageMetadata(convResSize, convResSize, 1000))

      val poolSizeReqd = math.ceil(convResSize / 2.0).toInt

      // We want poolSize to be even !!
      val poolSize = (math.ceil(poolSizeReqd / 2.0) * 2).toInt
      // overlap as little as possible
      val poolStride = convResSize - poolSize


      println(s"VALUES: $convSize $convResSize $poolSizeReqd $poolSize $poolStride")

      def summ(x: DenseVector[Double]): Double = sum(x)

      val pooling = new Pooler(poolStride, poolSize, identity, summ)
      val poolImage = pooling(image)
    }
  }

  test("SymmetricRectifier/Pooler and FastPooler should be the same") {
    val (x, y, c) = (24-6+1, 24-6+1, 10)

    val baseImage = utils.TestUtils.genRowMajorArrayVectorizedImage(x,y,c)

    val pipe = SymmetricRectifier(alpha=0.25) andThen new Pooler(9, 10, identity, Pooler.sumVector)

    val goodOutput = pipe(baseImage)

    val badOutput = new FastPooler(9, 10, 0.0, 0.25, ImageMetadata(x,y,c)).apply(baseImage)

    for(x <- 0 until goodOutput.metadata.xDim;
        y <- 0 until goodOutput.metadata.yDim;
        c <- 0 until goodOutput.metadata.numChannels) {
      assert(goodOutput.get(x,y,c) == badOutput.get(x,y,c),
        s"Mismatch at ($x,$y,$c): good: ${goodOutput.get(x,y,c)}, bad: ${badOutput.get(x,y,c)}")
    }

  }

  test("some pooling configs") {
    for(
      config <- Array((9,9,18), (9, 10, 19), (10,10,20))
    ) {
      val (patchStride, patchSize, imsize) = config
      val im = ImageMetadata(imsize, imsize, 20)
      val pooler = new FastPooler(patchStride, patchSize, 0.0, 0.25, im)
      logInfo(s"$patchStride $patchSize $imsize ${pooler.xPools.mkString(",")}")
    }
  }

  test("SymmetricRectifier/Pooler should work with multiple channels") {
    val (x,y,numChannels) = (19, 19, 10)

    val baseImage = utils.TestUtils.genRowMajorArrayVectorizedImage(x,y,numChannels)

    val channelImages = (0 until numChannels).map { channel =>
      val channelImage = RowMajorArrayVectorizedImage(new Array(x * y), ImageMetadata(x,y,1))
      for (xPos <- 0 until x;
           yPos <- 0 until y) {
        channelImage.put(xPos, yPos, 0, baseImage.get(xPos, yPos, channel))
      }
      channelImage
    }

    val pipe = SymmetricRectifier(alpha=0.25, maxVal = -0.5) andThen new Pooler(9, 10, identity, Pooler.sumVector)

    val fullImage = pipe(baseImage)
    val pooledChannels = channelImages.map(image => pipe(image))

    for(x <- 0 until fullImage.metadata.xDim;
        y <- 0 until fullImage.metadata.yDim;
        c <- 0 until numChannels;
        r <- 0 until 2) {
      assert(fullImage.get(x,y,c + numChannels*r) == pooledChannels(c).get(x,y,r),
        s"Mismatch at ($x,$y,$c,$r): good: ${fullImage.get(x,y,c + numChannels*r)}, " +
          s"bad: ${pooledChannels(c).get(x,y,r)}")
    }
  }

  test("NativePooler should work with multiple channels") {
    val (x,y,numChannels) = (19, 19, 10)

    val baseImage = utils.TestUtils.genRowMajorArrayVectorizedImage(x,y,numChannels)

    val channelImages = (0 until numChannels).map { channel =>
      val channelImage = RowMajorArrayVectorizedImage(new Array(x * y), ImageMetadata(x,y,1))
      for (xPos <- 0 until x;
           yPos <- 0 until y) {
        channelImage.put(xPos, yPos, 0, baseImage.get(xPos, yPos, channel))
      }
      channelImage
    }

    val pipe = new NativePooler(9, 10, 0.0, 0.25)

    val fullImage = pipe(baseImage)
    val pooledChannels = channelImages.map(image => pipe(image))

    for(x <- 0 until fullImage.metadata.xDim;
        y <- 0 until fullImage.metadata.yDim;
        c <- 0 until numChannels;
        r <- 0 until 2) {
      assert(fullImage.get(x,y,c + numChannels*r) == pooledChannels(c).get(x,y,r),
        s"Mismatch at ($x,$y,$c,$r): good: ${fullImage.get(x,y,c + numChannels*r)}, " +
          s"bad: ${pooledChannels(c).get(x,y,r)}")
    }
  }

  test("SymmetricRectifier/Pooler and NativePooler should be the same") {
    val (x,y,c) = (19, 19, 10)

    val baseImage = utils.TestUtils.genRowMajorArrayVectorizedImage(x,y,c)

    val pipe = SymmetricRectifier(alpha=0.25) andThen new Pooler(9, 10, identity, Pooler.sumVector)

    val goodOutput = pipe(baseImage)

    val badOutput = new NativePooler(9, 10, 0.0, 0.25).apply(baseImage)

    for(x <- 0 until goodOutput.metadata.xDim;
        y <- 0 until goodOutput.metadata.yDim;
        c <- 0 until goodOutput.metadata.numChannels) {
      assert(goodOutput.get(x,y,c) == badOutput.get(x,y,c),
        s"Mismatch at ($x,$y,$c): good: ${goodOutput.get(x,y,c)}, bad: ${badOutput.get(x,y,c)}")
    }
  }
}
