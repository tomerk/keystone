package workflow

import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * A Pipeline takes data as input (single item or an RDD), and outputs some transformation
 * of that data. Internally, a Pipeline is represented by a DAG of [[Node]]s with two types of
 * edges: One representing the data flow of RDDs through the DAG (for both transformations & for
 * [[Estimator]] fitting), and one representing which [[Transformer]]s are output by which
 * Estimators.
 *
 * @tparam A type of the data this Pipeline expects as input
 * @tparam B type of the data this Pipeline outputs
 */
trait Pipeline[A, B] {
  /**
   * The nodes found in this Pipeline (index is the id)
   */
  private[workflow] val nodes: Seq[Node]

  /**
   * Edges representing data flow through the DAG.
   * (An edge points towards a node whose output data this node takes as input data)
   * The indices of these edges map to the node indices.
   */
  private[workflow] val dataDeps: Seq[Seq[Int]]

  /**
   * Edges pointing towards the Estimators a given Delegating Transformer is fit by.
   * (Only non-empty for Delegating Transformer nodes)
   * The indices of these edges map to the node indices.
   */
  private[workflow] val fitDeps: Seq[Option[Int]]

  /**
   * Index of the last node in the DAG before the transformed data is returned.
   */
  private[workflow] val sink: Int

  def apply(in: A): B
  def apply(in: A, optimizer: Option[Optimizer]): B

  def apply(in: RDD[A]): RDD[B]
  def apply(in: RDD[A], optimizer: Option[Optimizer]): RDD[B]

  /**
   * Chains a pipeline onto the end of this one, producing a new pipeline.
   *
   * @param next the pipeline to chain
   */
  final def andThen[C](next: Pipeline[B, C]): Pipeline[A, C] = {
    val nodes = this.nodes ++ next.nodes
    val dataDeps = this.dataDeps ++ next.dataDeps.map(_.map {
      x => if (x == Pipeline.SOURCE) this.sink else x + this.nodes.size
    })
    val fitDeps = this.fitDeps ++ next.fitDeps.map(_.map {
      x => if (x == Pipeline.SOURCE) this.sink else x + this.nodes.size
    })
    val sink = next.sink + this.nodes.size

    Pipeline(nodes, dataDeps, fitDeps, sink)
  }

  final def andThen[C](est: Estimator[B, C], data: RDD[A]): PipelineWithFittedTransformer[A, B, C] = {
    val transformerLabel = est.label + ".fit"

    val newNodes = this.nodes :+ est :+ SourceNode(data) :+ new DelegatingTransformerNode(transformerLabel)
    val newDataDeps = this.dataDeps.map(_.map {
      x => if (x == Pipeline.SOURCE) this.nodes.size + 1 else x
    }) :+ Seq(this.sink) :+ Seq() :+ Seq(Pipeline.SOURCE)
    val newFitDeps = this.fitDeps :+ None :+ None :+ Some(this.nodes.size)
    val newSink = newNodes.size - 1

    val fittedTransformer = Pipeline[B, C](newNodes, newDataDeps, newFitDeps, newSink)
    val totalOut = this andThen fittedTransformer
    new PipelineWithFittedTransformer(
      totalOut.nodes,
      totalOut.dataDeps,
      totalOut.fitDeps,
      totalOut.sink,
      fittedTransformer)
  }

  final def andThen[C, L](est: LabelEstimator[B, C, L], data: RDD[A], labels: RDD[L]):
  PipelineWithFittedTransformer[A, B, C] = {
    val transformerLabel = est.label + ".fit"

    val newNodes = this.nodes :+
        est :+
        SourceNode(data) :+
        SourceNode(labels) :+
        new DelegatingTransformerNode(transformerLabel)
    val newDataDeps = this.dataDeps.map(_.map {
      x => if (x == Pipeline.SOURCE) this.nodes.size + 1 else x
    }) :+ Seq(this.sink, this.nodes.size + 2) :+ Seq() :+ Seq() :+ Seq(Pipeline.SOURCE)
    val newFitDeps = this.fitDeps :+ None :+ None :+ None :+ Some(this.nodes.size)
    val newSink = newNodes.size - 1

    val fittedTransformer = Pipeline[B, C](newNodes, newDataDeps, newFitDeps, newSink)
    val totalOut = this andThen fittedTransformer
    new PipelineWithFittedTransformer(
      totalOut.nodes,
      totalOut.dataDeps,
      totalOut.fitDeps,
      totalOut.sink,
      fittedTransformer)
  }


  /**
   * @return A graphviz dot representation of this pipeline
   */
  final def toDOTString: String = {
    val nodeLabels: Seq[String] = "-1 [label=\"In\" shape=\"Msquare\"]" +: nodes.zipWithIndex.map {
      case (data: SourceNode, id)  => s"$id [label=${'"' + data.label + '"'} shape=${"\"box\""} style=${"\"filled\""}]"
      case (transformer: TransformerNode, id) => s"$id [label=${'"' + transformer.label + '"'}]"
      case (delTransformer: DelegatingTransformerNode, id) => s"$id [label=${'"' + delTransformer.label + '"'}]"
      case (estimator: EstimatorNode, id) => s"$id [label=${'"' + estimator.label + '"'} shape=${"\"box\""}]"
    } :+ s"${nodes.size} [label=${"\"Out\""} shape=${"\"Msquare\""}]"

    val dataEdges: Seq[String] = dataDeps.zipWithIndex.flatMap {
      case (deps, id) => deps.map(x => s"$x -> $id")
    } :+ s"$sink -> ${nodes.size}"

    val fitEdges: Seq[String] = fitDeps.zipWithIndex.flatMap {
      case (deps, id) => deps.map(x => s"$x -> $id [dir=${"\"none\""} style=${"\"dashed\""}]")
    }

    val ranks = fitDeps.zipWithIndex.flatMap {
      case (deps, id) => deps.map(x => s"{rank=same; $x $id}")
    }

    val lines = nodeLabels ++ dataEdges ++ fitEdges ++ ranks
    lines.mkString("digraph pipeline {\n  rankdir=LR;\n  ", "\n  ", "\n}")
  }

  final private[workflow] def planEquals(pipeline: Pipeline[A, B]): Boolean = {
    this.eq(pipeline) || (
        (nodes == pipeline.nodes) &&
        (dataDeps == pipeline.dataDeps) &&
        (fitDeps == pipeline.fitDeps) &&
        (sink == pipeline.sink))
  }
}

object Pipeline {
  val SOURCE: Int = -1

  /**
   * Constructs the Identity pipeline
   * @tparam T The type of input to take
   */
  def apply[T](): Pipeline[T, T] = new ConcretePipeline(Seq(), Seq(), Seq(), SOURCE)

  /**
   * Constructs a Pipeline with the given DAG specification
   */
  private[workflow] def apply[A, B](
    nodes: Seq[Node],
    dataDeps: Seq[Seq[Int]],
    fitDeps: Seq[Option[Int]],
    sink: Int): Pipeline[A, B] = {
    new ConcretePipeline(nodes, dataDeps, fitDeps, sink)
  }

  /**
   * Produces a pipeline that when given an input,
   * combines the outputs of all its branches when executed on that input into a single Seq (in order)
   * @param branches The pipelines whose outputs should be combined into a Seq
   */
  def gather[A, B : ClassTag](branches: Seq[Pipeline[A, B]]): Pipeline[A, Seq[B]] = {
    // attach a value per branch to offset all existing node ids by.
    val branchesWithNodeOffsets = branches.scanLeft(0)(_ + _.nodes.size).zip(branches)

    val newNodes = branches.map(_.nodes).reduceLeft(_ ++ _) :+ new GatherTransformer[B]

    val newDataDeps = branchesWithNodeOffsets.map { case (offset, branch) =>
      val dataDeps = branch.dataDeps
      dataDeps.map(_.map(x => if (x == Pipeline.SOURCE) Pipeline.SOURCE else x + offset))
    }.reduceLeft(_ ++ _) :+  branchesWithNodeOffsets.map { case (offset, branch) =>
      val sink = branch.sink
      if (sink == Pipeline.SOURCE) Pipeline.SOURCE else sink + offset
    }

    val newFitDeps = branchesWithNodeOffsets.map { case (offset, branch) =>
      val fitDeps = branch.fitDeps
      fitDeps.map(_.map(x => if (x == Pipeline.SOURCE) Pipeline.SOURCE else x + offset))
    }.reduceLeft(_ ++ _) :+  None

    val newSink = newNodes.size - 1
    Pipeline(newNodes, newDataDeps, newFitDeps, newSink)
  }

  /**
   * A method used for tuning between several pipeline options on validation data.
   *
   * It takes multiple pipelines, validation data and labels, and an evaluation metric,
   * and returns a newpipeline that evaluates all of the inputs on the validation data, then applies
   * the best-scoring pipeline to test data.
   *
   * @param branches The sequence of pipelines to tune between.
   * @param data The validation data to pass into each branch
   * @param labels The labels for the validation data
   * @param evaluator The evaluation metric to use on the validation data
   * @tparam A The input type of the pipelines
   * @tparam B The output type of the pipelines
   * @tparam L The type of the validation labels
   * @return
   */
  def tune[A, B : ClassTag, L](
      branches: Seq[Pipeline[A, B]],
      data: RDD[A],
      labels: RDD[L],
      evaluator: (RDD[B], RDD[L]) => Double
    ): Pipeline[A, B] = {
    // attach a value per branch to offset all existing node ids by.
    val branchesWithNodeOffsets = branches.scanLeft(0)(_ + _.nodes.size).zip(branches)
    val numNodesInBranches = branches.map(_.nodes.size).sum

    // The new nodes consist of:
    // - all of the branches together twice, once for passing into a ModelSelector at train time, and one to pass into
    //   the tuned decision transformer at test time.
    // - A ModelSelector created with the evaluator, to select the optimal branch
    // - A source node for the train eval data
    // - A source node for the train eval labels
    // - A delegating transformer to apply the tuning decision
    val newNodes = branches.map(_.nodes).reduceLeft(_ ++ _) ++
      branches.map(_.nodes).reduceLeft(_ ++ _) :+
      new ModelSelector[B, L](evaluator) :+
      SourceNode(data) :+
      SourceNode(labels) :+
      new DelegatingTransformerNode("TunedModel")

    val delegatingTransformerIndex = newNodes.size - 1
    val evalLabelsIndex = newNodes.size - 2
    val evalDataIndex = newNodes.size - 3
    val modelSelectorIndex = newNodes.size - 4

    // When updating the data dependencies:
    // - The first copy of the branches must be connected to the train eval data
    // - The second copy of the branches must remain connected to the source
    // - The model selector must be connected to the first set of branches (being fed by train eval data),
    //   and to the train eval labels
    // - The train eval data and labels require no data dependencies
    // - The delegating transformer needs to be connected to the second set of branches, being fed by
    //   Pipeline.Source
    val newDataDeps = branchesWithNodeOffsets.map { case (offset, branch) =>
      val dataDeps = branch.dataDeps
      dataDeps.map(_.map(x => if (x == Pipeline.SOURCE) evalDataIndex else x + offset))
    }.reduceLeft(_ ++ _) ++
      branchesWithNodeOffsets.map { case (offset, branch) =>
        val dataDeps = branch.dataDeps
        dataDeps.map(_.map(x => if (x == Pipeline.SOURCE) Pipeline.SOURCE else x + offset + numNodesInBranches))
      }.reduceLeft(_ ++ _) :+
      branchesWithNodeOffsets.flatMap { case (offset, branch) =>
        val sink = branch.sink
        val branchIndex = if (sink == Pipeline.SOURCE) Pipeline.SOURCE else sink + offset
        Seq(branchIndex, evalLabelsIndex)
      } :+
      Seq() :+
      Seq() :+
      branchesWithNodeOffsets.map { case (offset, branch) =>
        val sink = branch.sink
        if (sink == Pipeline.SOURCE) Pipeline.SOURCE else sink + offset + numNodesInBranches
      }

    // The fit dependencies must be updated for both sets of branches given the new node ids.
    // The only new fit dependency to be added is from the delegating transformer to the ModelSelector
    val newFitDeps = branchesWithNodeOffsets.map { case (offset, branch) =>
      val fitDeps = branch.fitDeps
      fitDeps.map(_.map(x => if (x == Pipeline.SOURCE) Pipeline.SOURCE else x + offset))
    }.reduceLeft(_ ++ _) ++
      branchesWithNodeOffsets.map { case (offset, branch) =>
        val fitDeps = branch.fitDeps
        fitDeps.map(_.map(x => if (x == Pipeline.SOURCE) Pipeline.SOURCE else x + offset + numNodesInBranches))
      }.reduceLeft(_ ++ _) :+
      None :+
      None :+
      None :+
      Some(modelSelectorIndex)

    // The output of the delegating transformer becomes the new sink
    val newSink = delegatingTransformerIndex
    Pipeline(newNodes, newDataDeps, newFitDeps, newSink)
  }
}

class PipelineWithFittedTransformer[A, B, C] private[workflow] (
    nodes: Seq[Node],
    dataDeps: Seq[Seq[Int]],
    fitDeps: Seq[Option[Int]],
    sink: Int,
    val fittedTransformer: Pipeline[B, C])
    extends ConcretePipeline[A, C](nodes, dataDeps, fitDeps, sink)
