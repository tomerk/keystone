package workflow

import breeze.linalg.{max, DenseVector, DenseMatrix}
import org.apache.spark.rdd.RDD

case class Profile(ns: Long, mem: Long) {
  def +(p: Profile) = Profile(this.ns + p.ns, this.mem + p.mem)
}

case class SampleProfile(scale: Long, profile: Profile)

class AutoCacheRule(
    profileScales: Seq[Long] = Seq(5000, 10000),
    numProfileTrials: Int = 1
  ) extends Rule {

  private def numPerPartition[T](rdd: RDD[T]): Map[Int, Int] = {
    rdd.mapPartitionsWithIndex {
      case (id, partition) => Iterator.single((id, partition.length))
    }.collect().toMap
  }

  def generalizeProfiles(newScale: Long, sampleProfiles: Seq[SampleProfile]): Profile = {
    def getModel(inp: Iterable[(Long, String, Long)]): Double => Double = {
      val observations = inp.toArray

      //Pack a data matrix with observations
      val X = DenseMatrix.ones[Double](observations.length, 2)
      observations.zipWithIndex.foreach(o => X(o._2, 0) = o._1._1.toDouble)
      val y = DenseVector(observations.map(_._3.toDouble))
      val model = max(X \ y, 0.0)

      //A function to apply the model.
      def res(x: Double): Double = DenseVector(x, 1.0).t * model

      res
    }

    val samples = sampleProfiles.flatMap { case SampleProfile(scale, value) =>
      Array(
        (scale, "memory", value.mem),
        (scale, "time", value.ns)
      )}.groupBy(a => a._2)

    val models = samples.mapValues(getModel)

    Profile(models("time").apply(newScale).toLong, models("memory").apply(newScale).toLong)
  }

  def profileInstructions(
      instructions: Seq[Instruction],
      scales: Seq[Long],
      numTrials: Int
    ): Map[Int, Profile] = {
    val instructionsToProfile = instructions.indices.toSet -- WorkflowUtils.getChildren(Pipeline.SOURCE, instructions)

    val registers = scala.collection.mutable.Map[Int, Any]()
    val numPerPartitionPerNode = scala.collection.mutable.Map[Int, Map[Int, Int]]()
    val profiles = scala.collection.mutable.Map[Int, Profile]()

    val sortedScales = scales.sorted
    for ((instruction, i) <- instructions.zipWithIndex if instructionsToProfile.contains(i)) {
      instruction match {
        case SourceNode(rdd) =>
          val npp = numPerPartition(rdd)
          numPerPartitionPerNode(i) = npp

          val totalCount = npp.values.map(_.toLong).sum

          val sampleProfiles = for (
            (scale, scaleIndex) <- sortedScales.zipWithIndex;
            trial <- 1 to numTrials
          ) yield {
            // Calculate the necessary number of items per partition to maintain the same partition distribution,
            // while only having scale items instead of totalCount items.
            // Can't use mapValues because that isn't serializable
            val scaledNumPerPartition = npp.toSeq.map(x => (x._1, ((scale.toDouble / totalCount) * x._2).toInt)).toMap

            // Profile sample timing
            val start = System.nanoTime()
            // Construct a sample containing only scale items, but w/ the same relative partition distribution
            val sample = rdd.mapPartitionsWithIndex {
              case (pid, partition) => partition.take(scaledNumPerPartition(pid))
            }.cache()
            sample.count()
            val duration = System.nanoTime() - start

            // Profile sample memory
            val memSize = sample.context.getRDDStorageInfo.filter(_.id == sample.id).map(_.memSize).head

            // If this sample was computed using the final and largest scale, add it to the registers
            if ((scaleIndex == (sortedScales.length - 1)) && (trial == numTrials)) {
              registers(i) = sample
            } else {
              sample.unpersist()
            }

            SampleProfile(scale, Profile(duration, memSize))
          }

          profiles(i) = generalizeProfiles(totalCount, sampleProfiles)

        case transformer: TransformerNode =>
          registers(i) = transformer

        case estimator: EstimatorNode =>
          registers(i) = estimator

        case TransformerApplyNode(tIndex, inputIndices) =>
          // We assume that all input rdds to this transformer have equal, zippable partitioning
          val npp = numPerPartitionPerNode(inputIndices.head)
          numPerPartitionPerNode(i) = npp
          val totalCount = npp.values.map(_.toLong).sum

          val transformer = registers(tIndex).asInstanceOf[TransformerNode]
          val inputs = inputIndices.map(x => registers(x).asInstanceOf[RDD[_]])

          val sampleProfiles = for (
            (scale, scaleIndex) <- sortedScales.zipWithIndex;
            trial <- 1 to numTrials
          ) yield {
            // Calculate the necessary number of items per partition to maintain the same partition distribution,
            // while only having scale items instead of totalCount items.
            // Can't use mapValues because that isn't serializable
            val scaledNumPerPartition = npp.toSeq.map(x => (x._1, ((scale.toDouble / totalCount) * x._2).toInt)).toMap

            // Sample the inputs. Samples containing only scale items, but w/ the same relative partition distribution
            // NOTE: Assumes all inputs have equal, zippable partition counts
            val sampledInputs = inputs.map(_.mapPartitionsWithIndex {
              case (pid, partition) => partition.take(scaledNumPerPartition(pid))
            })
            sampledInputs.foreach(_.count())

            // Profile sample timing
            val start = System.nanoTime()
            // Construct a
            val sample = transformer.transformRDD(sampledInputs).cache()
            sample.count()
            val duration = System.nanoTime() - start

            // Profile sample memory
            val memSize = sample.context.getRDDStorageInfo.filter(_.id == sample.id).map(_.memSize).head

            // If this sample was computed using the final and largest scale, add it to the registers
            if ((scaleIndex == (sortedScales.length - 1)) && (trial == numTrials)) {
              registers(i) = sample
            } else {
              sample.unpersist()
            }

            SampleProfile(scale, Profile(duration, memSize))
          }

          profiles(i) = generalizeProfiles(totalCount, sampleProfiles)

        case EstimatorFitNode(estIndex, inputIndices) =>
          val estimator = registers(estIndex).asInstanceOf[EstimatorNode]
          val inputs = inputIndices.map(x => registers(x).asInstanceOf[RDD[_]])
          registers(i) = estimator.fitRDDs(inputs)
      }
    }

    profiles.toMap
  }

  override def apply[A, B](plan: Pipeline[A, B]): Pipeline[A, B] = {
    val instructions = WorkflowUtils.pipelineToInstructions(plan)

    val profiles = profileInstructions(instructions, profileScales, numProfileTrials)

    null
  }
}
