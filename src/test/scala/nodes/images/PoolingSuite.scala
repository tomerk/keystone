package nodes.images

import breeze.linalg.{DenseVector, sum}
import nodes._
import org.scalatest.FunSuite
import pipelines.Logging
import utils.{ChannelMajorArrayVectorizedImage, ImageMetadata}

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
}
