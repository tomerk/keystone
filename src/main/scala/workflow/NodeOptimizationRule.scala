package workflow

import org.apache.spark.rdd.RDD
import pipelines.Logging

/**
 * Node-level optimization, such as selecting a Linear Solver
 *
 * @param samplePerPartition The number per partition of the RDD to use for operations
 */
class NodeOptimizationRule(samplePerPartition: Int = 3, seed: Long = 0) extends Rule with Logging {
  override def apply[A, B](plan: Pipeline[A, B]): Pipeline[A, B] = {
    logInfo("Optimization Timing: Started NodeOptimizationRule")
    val instructions = WorkflowUtils.pipelineToInstructions(plan)

    // First, figure out which instructions we should actually execute.
    // Take the set of all instructions that are parents of an optimizable node transform or fit instruction.
    // And, keep those that do not depend on the runtime input of a fit pipeline
    val instructionsToExecute = instructions.zipWithIndex.map {
      case (TransformerApplyNode(tIndex, _), index) =>
        instructions(tIndex) match {
          case _: OptimizableTransformer[_, _] => WorkflowUtils.getParents(index, instructions) + index
          case _ => Set[Int]()
        }

      case (EstimatorFitNode(estIndex, _), index) =>
        instructions(estIndex) match {
          case _: OptimizableEstimator[_, _] => WorkflowUtils.getParents(index, instructions) + index
          case _: OptimizableLabelEstimator[_, _, _] => WorkflowUtils.getParents(index, instructions) + index
          case _ => Set[Int]()
        }

      case _ => Set[Int]()
    }.reduce(_ union _) -- WorkflowUtils.getChildren(Pipeline.SOURCE, instructions) - Pipeline.SOURCE

    // Execute the minimal amount necessary of the pipeline on sampled nodes, and optimize the optimizable nodes
    val optimizedInstructions = instructions.toArray
    val registers = new Array[InstructionOutput](instructions.length)
    val numPerPartitionPerNode = Array.fill[Option[Map[Int, Int]]](instructions.length)(None)

    // Copy this value to avoid serializing the rule
    val spp = samplePerPartition

    for ((instruction, index) <- instructions.zipWithIndex) {
      if (instructionsToExecute.contains(index)) {
        instruction match {
          case SourceNode(rdd) => {
            val sampledRDD = rdd.mapPartitions(_.take(spp))
            registers(index) = RDDOutput(sampledRDD)
            numPerPartitionPerNode(index) = Some(WorkflowUtils.numPerPartition(rdd))
          }

          case TransformerApplyNode(tIndex, inputIndices) => {
            val numPerPartition = numPerPartitionPerNode(inputIndices.head)
            val inputs = inputIndices.map(registers).collect {
              case RDDOutput(rdd) => rdd.cache()
            }
            inputs.foreach(_.count())

            // Optimize the transformer if possible
            val transformer = registers(tIndex) match {
              case TransformerOutput(ot: OptimizableTransformer[a, b]) =>
                ot.optimize(inputs.head.asInstanceOf[RDD[a]], numPerPartition.get)
              case TransformerOutput(t) => t
              case _ => throw new ClassCastException("TransformerApplyNode dep wasn't pointing at a transformer")
            }

            registers(index) = RDDOutput(transformer.transformRDD(inputs))
            optimizedInstructions(tIndex) = transformer
            numPerPartitionPerNode(index) = numPerPartition
          }

          case EstimatorFitNode(estIndex, inputIndices) => {
            val numPerPartition = numPerPartitionPerNode(inputIndices.head)
            val inputs = inputIndices.map(registers).collect {
              case RDDOutput(rdd) => rdd.cache()
            }
            inputs.foreach(_.count())

            // Optimize the estimator if possible
            val estimator = registers(estIndex) match {
              case EstimatorOutput(oe: OptimizableEstimator[a, b]) =>
                oe.optimize(inputs.head.asInstanceOf[RDD[a]], numPerPartition.get)
              case EstimatorOutput(oe: OptimizableLabelEstimator[a, b, l]) =>
                oe.optimize(inputs(0).asInstanceOf[RDD[a]], inputs(1).asInstanceOf[RDD[l]], numPerPartition.get)
              case EstimatorOutput(e) => e
              case _ => throw new ClassCastException("EstimatorFitNode dep wasn't pointing at an estimator")
            }

            registers(index) = TransformerOutput(estimator.fitRDDs(inputs))
            optimizedInstructions(estIndex) = estimator
            numPerPartitionPerNode(index) = numPerPartition
          }

          case node: Instruction => {
            val deps = node.getDependencies.map(registers)
            registers(index) = node.execute(deps)
          }
        }
      }
    }

    // Unpersist anything that may still be cached
    registers.foreach {
      case RDDOutput(rdd) => rdd.unpersist()
      case _ => Unit
    }

    val outPipe = WorkflowUtils.instructionsToPipeline[A, B](optimizedInstructions)
    logInfo("Optimization Timing: Finished NodeOptimizationRule")
    outPipe
  }
}
