package nodes.learning

import java.util.Comparator

import breeze.linalg._
import breeze.stats.distributions.{Bernoulli, Uniform}
import com.jwetherell.algorithms.sorts.MergeSort.SPACE_TYPE
import com.jwetherell.algorithms.sorts.{HeapSort, InsertionSort, MergeSort, QuickSort}
import com.jwetherell.algorithms.sorts.QuickSort.PIVOT_TYPE
import edu.berkeley.cs.amplab.mlmatrix.RowPartitionedMatrix
import net.greypanther.javaadvent.regex.factories.DkBricsAutomatonRegexFactory
import nodes.stats.StandardScaler
import org.apache.spark.{SparkContext, rdd}
import org.scalatest.FunSuite
import pipelines.Logging
import utils.{MatrixUtils, Stats, TestUtils}
import workflow.{PipelineContext, ZipfDistribution}

class LBFGSSuite extends FunSuite with PipelineContext with Logging {
  test("Solve a dense linear system (fit intercept)") {
    sc = new SparkContext("local", "test")

    // Create the data.
    val A = TestUtils.createRandomMatrix(sc, 128, 5, 4)
    val x = DenseMatrix((5.0, 4.0, 3.0, 2.0, -1.0), (3.0, -1.0, 2.0, -2.0, 1.0))
    val dataMean = DenseVector(1.0, 0.0, 1.0, 2.0, 0.0)
    val extraBias = DenseVector(3.0, 4.0)

    val initialAary = A.rdd.flatMap(part => MatrixUtils.matrixToRowArray(part.mat).toIterator)
    val meanScaler = new StandardScaler(normalizeStdDev = false).fit(initialAary)
    val Aary = meanScaler.apply(initialAary).map(_ + dataMean)
    val bary = Aary.map(a => (x * (a - dataMean)) + extraBias)

    val mapper = new DenseLBFGSwithL2(new LeastSquaresDenseGradient(), fitIntercept = true).fit(Aary, bary)

    val trueResult = MatrixUtils.rowsToMatrix(bary.collect())
    val solverResult = MatrixUtils.rowsToMatrix(mapper(Aary).collect())

    assert(Stats.aboutEq(trueResult, solverResult, 1e-5), "Results from the solve must match the hand-created model.")
    assert(Stats.aboutEq(mapper.x, x.t, 1e-5), "Model weights from the solve must match the hand-created model.")
    assert(Stats.aboutEq(mapper.bOpt.get, extraBias, 1e-5), "Learned intercept must match the hand-created model.")
    assert(Stats.aboutEq(mapper.featureScaler.get.mean, dataMean, 1e-5),
      "Learned intercept must match the hand-created model.")

  }

  test("Solve a dense linear system (no fit intercept)") {
    sc = new SparkContext("local", "test")

    // Create the data.
    val A = TestUtils.createRandomMatrix(sc, 128, 5, 4)
    val x = DenseMatrix((5.0, 4.0, 3.0, 2.0, -1.0), (3.0, -1.0, 2.0, -2.0, 1.0))
    val b = A.mapPartitions(part => part * x.t)

    val Aary = A.rdd.flatMap(part => MatrixUtils.matrixToRowArray(part.mat).toIterator)
    val bary = b.rdd.flatMap(part => MatrixUtils.matrixToRowArray(part.mat).toIterator)

    val mapper = new DenseLBFGSwithL2(new LeastSquaresDenseGradient(), fitIntercept = false).fit(Aary, bary)

    val trueResult = MatrixUtils.rowsToMatrix(bary.collect())
    val solverResult = MatrixUtils.rowsToMatrix(mapper(Aary).collect())

    assert(Stats.aboutEq(trueResult, solverResult, 1e-5), "Results from the solve must match the hand-created model.")
    assert(Stats.aboutEq(mapper.x, x.t, 1e-5), "Model weights from the solve must match the hand-created model.")
    assert(mapper.bOpt.isEmpty, "Not supposed to have learned an intercept.")
    assert(mapper.featureScaler.isEmpty, "Not supposed to have learned an intercept.")
  }

  test("Solve a sparse linear system (fit intercept)") {
    sc = new SparkContext("local", "test")

    // Create the data.
    val A = TestUtils.createRandomMatrix(sc, 128, 5, 4)
    val x = DenseMatrix((5.0, 4.0, 3.0, 2.0, -1.0), (3.0, -1.0, 2.0, -2.0, 1.0))
    val dataMean = DenseVector(1.0, 0.0, 1.0, 2.0, 0.0)
    val extraBias = DenseVector(3.0, 4.0)
    val b = A.mapPartitions(part => part * x.t)

    val Aary = A.rdd.flatMap(part => MatrixUtils.matrixToRowArray(part.mat).toIterator)
      .map(x => SparseVector((x + dataMean).toArray))
    val bary = b.rdd.flatMap(part => MatrixUtils.matrixToRowArray(part.mat).toIterator).map(_ + extraBias)

    val mapper = new SparseLBFGSwithL2(new LeastSquaresSparseGradient(), fitIntercept = true).fit(Aary, bary)

    val trueResult = MatrixUtils.rowsToMatrix(bary.collect())
    val solverResult = MatrixUtils.rowsToMatrix(mapper(Aary).collect())

    assert(Stats.aboutEq(trueResult, solverResult, 1e-3), "Results from the solve must match the hand-created model.")
    assert(Stats.aboutEq(mapper.x, x.t, 1e-3), "Model weights from the solve must match the hand-created model.")

    val trueBias = extraBias - (x * dataMean)
    assert(Stats.aboutEq(mapper.bOpt.get, trueBias, 1e-3), "Learned intercept must match the hand-created model.")
  }

  test("Solve a sparse linear system (no fit intercept)") {
    sc = new SparkContext("local", "test")

    // Create the data.
    val A = TestUtils.createRandomMatrix(sc, 128, 5, 4)
    val x = DenseMatrix((5.0, 4.0, 3.0, 2.0, -1.0), (3.0, -1.0, 2.0, -2.0, 1.0))
    val b = A.mapPartitions(part => part * x.t)

    val Aary = A.rdd.flatMap(part => MatrixUtils.matrixToRowArray(part.mat).toIterator)
      .map(x => SparseVector(x.toArray))
    val bary = b.rdd.flatMap(part => MatrixUtils.matrixToRowArray(part.mat).toIterator)

    val mapper = new SparseLBFGSwithL2(new LeastSquaresSparseGradient(), fitIntercept = false).fit(Aary, bary)

    val trueResult = MatrixUtils.rowsToMatrix(bary.collect())
    val solverResult = MatrixUtils.rowsToMatrix(mapper(Aary).collect())

    assert(Stats.aboutEq(trueResult, solverResult, 1e-4), "Results from the solve must match the hand-created model.")
    assert(Stats.aboutEq(mapper.x, x.t, 1e-4), "Model weights from the solve must match the hand-created model.")
    assert(mapper.bOpt.isEmpty, "Not supposed to have learned an intercept.")
  }


  test("Solve a sparse vs dense linear system (no fit intercept)") {

    val regexp = ".*[A-Ta-t]([ \t\n\r]+[A-Za-z]+)?([ \t\n\r]+[A-Za-z]+)?[ \t\n\r]+([A-Za-z]+ed).*"
    //    val regexp = ".*[Aa]lice.*"//".*[A-Ta-t]([ \t\n\r]+[A-Za-z]+)?([ \t\n\r]+[A-Za-z]+)?[ \t\n\r]+([A-Za-z]+ed).*"//"(\\s+[^.!?]*[.!?])"

    val dk = new DkBricsAutomatonRegexFactory().create(regexp)

    dk.containsMatch("hi")




    logInfo("PRepeth")

    sc = new SparkContext("local", "test")

    // Create the data.
    val A = TestUtils.createRandomMatrix(sc, 10000, 1000, 1)
    val x = DenseMatrix.rand[Double](3, 1000)
    val b = A.mapPartitions(part => part * x.t)

    val Aary = A.rdd.flatMap(part => MatrixUtils.matrixToRowArray(part.mat).toIterator)
      .map(x => {
        val out = SparseVector.zeros[Double](1000)
        val inds = util.Random.shuffle((0 until 1000).toList).take(1000).sorted
        for (i <- inds) {
          out(i) = x(i)
        }
        out
      }).cache()
      //.map(x => SparseVector(x.toArray)).cache()
    val bary = b.rdd.flatMap(part => MatrixUtils.matrixToRowArray(part.mat).toIterator).cache()

    Aary.count()
    bary.count()

    logInfo("Start fitting dense")
    val denseMapper = new DenseLBFGSwithL2(new LeastSquaresDenseGradient(), fitIntercept = false, numIterations = 10).fit(Aary, bary)
    logInfo("Done fitting dense")

    logInfo("Start fitting sparse")
    val mapper = new SparseLBFGSwithL2(new LeastSquaresSparseGradient(), fitIntercept = false, numIterations = 10).fit(Aary, bary)
    logInfo("Done fitting sparse")

    val trueResult = MatrixUtils.rowsToMatrix(bary.collect())
    val solverResult = MatrixUtils.rowsToMatrix(mapper(Aary).collect())

  }

  test("GenMe YAY!") {
    //sc = new SparkContext("local[4]", "test")
    //val partData = sc.parallelize(Seq.fill(1)(1.0), 1)

    //val totalWeight = partData.sum

    val nGrid = Seq(131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456)
    val elementsInDistribution = Seq(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, -1)
    val newDrawForB = Seq(0, 1)

    case class Datum(a: Double, b: Double) extends Comparable[Datum] {
      override def compareTo(o: Datum): Int = {
        val cmp = a.compare(o.a)
        if (cmp == 0) {
          b.compare(o.b)
        } else {
          cmp
        }
      }
    }

    println("n, elementsInDistribution, newDrawForB, algorithm, timeInMs")
    nGrid.foreach { n =>
      elementsInDistribution.filter(_ < n).foreach { uniqueElementsOrAll =>
        val uniqueElements = if (uniqueElementsOrAll < 0) n else uniqueElementsOrAll

        newDrawForB.filter{
          drawAgain => if (drawAgain == 0) {
            true
          } else {
            (uniqueElements.toLong * uniqueElements) <= n
          }
        }.foreach { drawAgain =>
          val dist = new Uniform(0.0, uniqueElements.toDouble)
          val data = (0 until n).map { _ =>
            val draw = math.floor(dist.draw())
            if (drawAgain > 0) {
              Datum(draw, math.floor(dist.draw()))
            } else {
              Datum(draw, draw)
            }
          }

          {
            val unsorted = data.toArray
            val start = System.currentTimeMillis()
            QuickSort.sort(PIVOT_TYPE.RANDOM, unsorted)
            val end = System.currentTimeMillis()
            val elapsed = end - start
            println(s"$n, $uniqueElements, $drawAgain, WetherellQuickSort, $elapsed")
          }

          {
            val unsorted = data.toArray
            val start = System.currentTimeMillis()
            HeapSort.sort(unsorted)
            val end = System.currentTimeMillis()
            val elapsed = end - start
            println(s"$n, $uniqueElements, $drawAgain, WetherellHeapSort, $elapsed")
          }

          {
            val unsorted = data.toArray
            val start = System.currentTimeMillis()
            MergeSort.sort(SPACE_TYPE.NOT_IN_PLACE, unsorted)
            val end = System.currentTimeMillis()
            val elapsed = end - start
            println(s"$n, $uniqueElements, $drawAgain, WetherellMergeSort, $elapsed")
          }


          {
            val unsorted = data.toArray
            val start = System.currentTimeMillis()
            util.Sorting.quickSort(unsorted)
            val end = System.currentTimeMillis()
            val elapsed = end - start
            println(s"$n, $uniqueElements, $drawAgain, ScalaUtilQuickSort, $elapsed")
          }

          {
            val unsorted = data.toArray
            val cmp = new Comparator[Datum] {
              override def compare(o1: Datum, o2: Datum): Int = o1.compareTo(o2)
            }
            val start = System.currentTimeMillis()
            java.util.Arrays.sort(unsorted, cmp)
            val end = System.currentTimeMillis()
            val elapsed = end - start
            println(s"$n, $uniqueElements, $drawAgain, JavaDefaultTimSort, $elapsed")
          }
        }
      }
    }
  }


}

