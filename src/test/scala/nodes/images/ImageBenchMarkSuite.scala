package nodes.images

import breeze.linalg._
import breeze.stats._
import nodes.images.external.NativePooler
import nodes.learning.ZCAWhitener
import org.apache.spark.SparkContext
import org.scalatest.FunSuite
import pipelines.{LocalSparkContext, Logging}
import pipelines.images.cifar.RandomPatchCifarRawAugmentLazy.RandomCifarFeaturizerConfig
import utils._
import utils.TestUtils._
import workflow.{Pipeline, Transformer}

import scala.util.Random

class ImageBenchMarkSuite extends FunSuite with Logging with LocalSparkContext {

  // TODO(shivaram): Uncomment this after we figure out its resource usage (4G for sbt ?)

  val tests = Array(
    IbUtils.TestParam("AugmentedCifar512", (24,24,3), 6, 512, 13, 14))
  //   TestParam("Cifar1000", (32,32,3), 6, 1000, 13, 14),
  //   TestParam("Cifar10000", (32,32,3), 6, 10000, 13, 14),
  //   TestParam("ImageNet", (256,256,3), 6, 100, (256-5)/2, (256-5)/2),
  //   TestParam("SolarFlares", (256,256,12), 6, 100, (256-5)/12, (256-5)/12),
  //   TestParam("ConvolvedSolarFlares", (251,251,100), 6, 100, 251/12, 251/12),
  //   TestParam("SolarFlares2", (256,256,12), 5, 1024, (256-4)/12, (256-4)/12)
  // )
  // 
  // def getImages(t: TestParam) = Array[VectorizedImage](
  //   genRowMajorArrayVectorizedImage(t.size._1, t.size._2, t.size._3),
  //   genColumnMajorArrayVectorizedImage(t.size._1, t.size._2, t.size._3),
  //   genChannelMajorArrayVectorizedImage(t.size._1, t.size._2, t.size._3)
  // )
  // 
  // test("Reverse map") {
  //   var img = genColumnMajorArrayVectorizedImage(7, 13, 17)
  //   val item = for(
  //     z <- 0 until img.metadata.numChannels;
  //     x <- 0 until img.metadata.xDim;
  //     y <- 0 until img.metadata.yDim
  //   ) yield img.get(x,y,z)
  //   assert(item.toArray === img.iter.map(_.v).toArray, "Column Major Array iterator does not produce the right order.")
  // 
  //   val img2 = genRowMajorArrayVectorizedImage(7, 13, 17)
  //   val item2 = for(
  //     z <- 0 until img2.metadata.numChannels;
  //     y <- 0 until img2.metadata.yDim;
  //     x <- 0 until img2.metadata.xDim
  //   ) yield img2.get(x,y,z)
  // 
  //   assert(item2.toArray === img2.iter.map(_.v).toArray, "Row Major Array iterator does not produce the right order.")
  // 
  //   val img3 = genChannelMajorArrayVectorizedImage(7, 13, 17)
  //   val item3 = for(
  //     y <- 0 until img3.metadata.yDim;
  //     x <- 0 until img3.metadata.xDim;
  //     z <- 0 until img3.metadata.numChannels
  //   ) yield img3.get(x,y,z)
  // 
  //   assert(item3.toArray === img3.iter.map(_.v).toArray, "Channel Major Array iterator does not produce the right order.")
  // }
  // 
  // 
  // test("Iteration Benchmarks") {
  //   def iterTimes(x: VectorizedImage) = {
  //     var sum1 = 0.0
  //     val data = (0 until x.metadata.xDim*x.metadata.yDim*x.metadata.numChannels).map(x.getInVector).toArray
  // 
  //     val start1 = System.nanoTime()
  //     var i = 0
  // 
  //     while (i < data.length) {
  //       sum1+=data(i)
  //       i+=1
  //     }
  //     val t1 = System.nanoTime() - start1
  // 
  // 
  //     var sum2 = 0.0
  //     val start2 = System.nanoTime()
  //     val iter = x.iter
  //     while (iter.hasNext) {
  //       sum2 += iter.next.v
  //     }
  //     val t2 = System.nanoTime() - start2
  // 
  //     val start3 = System.nanoTime()
  //     val tot = data.sum
  //     val t3 = System.nanoTime() - start3
  // 
  //     (t1, t2, t3, sum1, sum2, tot)
  //   }
  // 
  //   for (
  //     iter <- 1 to 10;
  //     t <- tests;
  //     i <- getImages(t)
  //   ) {
  //     val (t1, t2, t3, a, b, c) = iterTimes(i)
  //     val slowdown = t2.toDouble/t1
  //     val istr = '"' + i.toString + '"'
  //     logInfo(s"${t.name},$istr,$t1,$t2,$t3,$slowdown,${t2.toDouble/t3},$a,$b,$c")
  //   }
  //   //Iteration just going through the data.
  // 
  // }
  // 
   test("Convolution Benchmarks") {
     def convTime(x: Image, t: IbUtils.TestParam) = {
       val whiteSize = t.kernelSize*t.kernelSize*t.size._3
       val whitener = new ZCAWhitener(DenseMatrix.rand[Double](whiteSize, whiteSize), DenseVector.rand[Double](whiteSize))
       val filters = DenseMatrix.rand[Double](t.numKernels, t.kernelSize*t.kernelSize*t.size._3)
       val conv = new Convolver(filters, x.metadata.xDim, x.metadata.yDim, x.metadata.numChannels, whitener = Some(whitener), normalizePatches = true)

       val start = System.nanoTime
       val res = conv(x)
       val elapsed = System.nanoTime - start

       elapsed
     }

     val res = for(
       iter <- 1 to 100;
       t <- tests
     ) yield {
       val img = genChannelMajorArrayVectorizedImage(t.size._1, t.size._2, t.size._3) //Standard grayScale format.

       val flops = (t.size._1.toLong-t.kernelSize+1)*(t.size._2-t.kernelSize+1)*
         t.size._3*t.kernelSize*t.kernelSize*
         t.numKernels

       val t1 = convTime(img, t)

       logDebug(s"${t.name},$t1,$flops,${2.0*flops.toDouble/t1}")
       (t.name, t1, flops, (2.0*flops.toDouble/t1))
     }
     val groups = res.groupBy(_._1)


     logInfo(Seq("name","max(flops)","median(flops)","stddev(flops)").mkString(","))
     groups.foreach { case (name, values) =>
       val flops = DenseVector(values.map(_._4):_*)
       val times = DenseVector(values.map(_._2.toDouble):_*)
       val maxf = max(flops)
       val medf = median(flops)
       val stddevf = stddev(flops)
       val medt = median(times)
       logInfo(f"$name,$maxf%2.3f,$medf%2.3f,$stddevf%2.3f,$medt")
     }
   }

  test("Pooler benchmark") {
    def poolTime(x: RowMajorArrayVectorizedImage, pooler: Pipeline[Image,Image]) = {

      val start = System.nanoTime
      val res = pooler(x)
      val elapsed = System.nanoTime - start

      (elapsed, res)
    }
    //Thread.sleep(10000)

    val conf = RandomCifarFeaturizerConfig()
    val pooler = new FastPooler(conf.poolStride, conf.poolSize, 0.0, conf.alpha, ImageMetadata(24-6+1, 24-6+1, 512))
    val pooler2 = new NativePooler(conf.poolStride, conf.poolSize, 0.0, conf.alpha)
    val pooler3 = SymmetricRectifier(0.0, conf.alpha) andThen new Pooler(conf.poolStride, conf.poolSize, identity, Pooler.sumVector)

    val poolers: Array[(String, Pipeline[Image, Image])] = Array(("fast",pooler), ("native", pooler2), ("standard", pooler3))

    val res = for(
      p <- poolers;
      iter <- 1 to 100;
      t <- tests
    ) yield {
      val convOutputSizeX = t.size._1 - t.kernelSize + 1
      val convOutputSizeY = t.size._2 - t.kernelSize + 1

      val img = genRowMajorArrayVectorizedImage(convOutputSizeX, convOutputSizeY, t.numKernels)


      val gbs = (convOutputSizeX*convOutputSizeY*t.numKernels*8.0)// + (2*2*t.numKernels*8.0)


      //val pooler = new Pooler(conf.poolStride, conf.poolSize, identity, Pooler.sumVector)
      val (t1, res) = poolTime(img, p._2)
      //logInfo(s"$t1, $gbs, ${res.get(1,1,1)}, ${gbs.toDouble/t1}")

      (t.name, p._1, t1, gbs, gbs.toDouble/t1)
    }

    val groups = res.groupBy(x => (x._1,x._2))

    logInfo(Seq("name","max(gbs)","median(gbs)","stddev(gbs)").mkString(","))
    groups.foreach { case (name, values) =>
      val gBytesPerSec = DenseVector(values.map(_._5):_*)
      val times = DenseVector(values.map(_._3.toDouble):_*)
      val maxgbs = max(gBytesPerSec)
      val medgbs = median(gBytesPerSec)
      val stddevgbs = stddev(gBytesPerSec)
      val medtimes = median(times)

      logInfo(f"$name,$maxgbs%2.3f,$medgbs%2.3f,$stddevgbs%2.3f,$medtimes")
    }
  }

  test("spark benchmark") {
    def logStats(s: DenseVector[Double], name: String) = {
      val meant = mean(s)
      val medt = median(s)
      val sdt = stddev(s)

      logInfo(f"$name Avg:$meant%2.3f, Med: $medt%2.3f, Stdev: $sdt%2.3f")
    }

    sc = new SparkContext("local[8]", "test")

    val parts = sc.parallelize(0 until 8)
    val t = tests(0)

    val filters = DenseMatrix.rand[Double](t.numKernels, t.kernelSize*t.kernelSize*t.size._3)
    val conv = new Convolver(filters, t.size._1, t.size._2, t.size._3, normalizePatches = true)


    val imgRdd = parts.flatMap(x => {
      (0 until 10).map(i => genRowMajorArrayVectorizedImage(t.size._1, t.size._2, t.kernelSize))
    })

    val convolvedImageRdd = imgRdd.map(i => IbUtils.timeIt(conv.apply: Image => Image, i)).cache()
    val convTimes = convolvedImageRdd.collect().map(_._1.toDouble)


    val convTimeVec = DenseVector(convTimes:_*)
    logStats(convTimeVec, "conv")


    val img = convolvedImageRdd.first._2

    val conf = RandomCifarFeaturizerConfig()

    val pooler = new FastPooler(conf.poolStride, conf.poolSize, 0.0, conf.alpha, img.metadata)
    val pooledImageRdd = convolvedImageRdd.map(i => IbUtils.timeIt(pooler.apply, i._2)).cache()

    val poolTimes = pooledImageRdd.collect().map(_._1.toDouble)

    val poolTimeVec = DenseVector(poolTimes:_*)

    logStats(poolTimeVec, "pool")

  }
}


object IbUtils extends Serializable {
  def timeIt[I,O](f: I => O, in: I): (Double, O) = {
    val start = System.nanoTime
    val out = f(in)
    val t = System.nanoTime - start
    (t, out)
  }

  case class TestParam(name: String, size: (Int, Int, Int), kernelSize: Int, numKernels: Int, poolSize: Int, poolStride: Int)


}