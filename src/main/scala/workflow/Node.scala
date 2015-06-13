package workflow

import org.apache.spark.rdd.RDD

sealed trait Node extends Serializable

abstract class EstimatorNode[T <: TransformerNode[_]] extends Node {
  def fit(dependencies: Seq[RDD[_]]): T
}

abstract class TransformerNode[T] extends Node {
  def transform(dataDependencies: Seq[_], fitDependencies: Seq[TransformerNode[_]]): T
  def transformRDD(dataDependencies: Seq[RDD[_]], fitDependencies: Seq[TransformerNode[_]]): RDD[T]
  def partialApply(fitDependencies: Seq[TransformerNode[_]]): TransformerNode[T]
}

class DataNode[T](rdd: RDD[T]) extends Node {
  def get(): RDD[T] = rdd
}

case class Pipeline[A, B](
  nodes: Seq[Node],
  dataDeps: Seq[Seq[Int]],
  fitDeps: Seq[Seq[Int]],
  sources: Seq[Int],
  sink: Int) {

  validate()

  val SOURCE: Int = -1

  private def validate(): Unit = {
    /*
    TODO: require that
    - nodes.size = dataDeps.size == fitDeps.size
    - there is a sink
    - there is a data path from sink to in
    - there are no fit paths from sink to in
    - there are no floating nodes that sink doesn’t depend on
    - only data nodes may be sources (no deps)
    - data nodes must have no deps
    - estimators may not have fit deps, must have data deps
    - transformers must have data deps, may or may not have fit deps
    - fit deps may only point to estimators
     */

    /*
    If we add classtags and some other complexity to nodes could also validate:
    sink output type matches pipeline output type
     */
  }

  private def fitEstimator(node: Int): TransformerNode[_] = nodes(node) match {
    case _: DataNode[_] =>
      throw new RuntimeException("Pipeline DAG error: Cannot have a fit dependency on a DataNode")
    case _: TransformerNode[_] =>
      throw new RuntimeException("Pipeline DAG error: Cannot have a data dependency on a Transformer")
    case estimator: EstimatorNode[_] =>
      val nodeDataDeps = dataDeps(node).map(x => rddDataEval(x, null))
      estimator.fit(nodeDataDeps)
  }

  private def singleDataEval(node: Int, in: A): Any = {
    if (node == SOURCE) {
      in
    } else {
      nodes(node) match {
        case transformer: TransformerNode[_] =>
          val nodeFitDeps = fitDeps(node).map(fitEstimator)
          val nodeDataDeps = dataDeps(node).map(x => singleDataEval(x, in))
          transformer.transform(nodeDataDeps, nodeFitDeps)
        case _: DataNode[_] =>
          throw new RuntimeException("Pipeline DAG error: came across an RDD data dependency when trying to do a single item apply")
        case _: EstimatorNode[_] =>
          throw new RuntimeException("Pipeline DAG error: Cannot have a data dependency on an Estimator")
      }
    }
  }

  private def rddDataEval(node: Int, in: RDD[A]): RDD[_] = {
    if (node == SOURCE) {
      in
    } else {
      nodes(node) match {
        case dataNode: DataNode[_] => dataNode.get()
        case transformer: TransformerNode[_] =>
          val nodeFitDeps = fitDeps(node).map(fitEstimator)
          val nodeDataDeps = dataDeps(node).map(x => rddDataEval(x, in))
          transformer.transformRDD(nodeDataDeps, nodeFitDeps)
        case _: EstimatorNode[_] =>
          throw new RuntimeException("Pipeline DAG error: Cannot have a data dependency on an Estimator")
      }
    }
  }

  def apply(in: A): B = singleDataEval(sink, in).asInstanceOf[B]

  def apply(in: RDD[A]): RDD[B] = rddDataEval(sink, in).asInstanceOf[RDD[B]]

  def toDOT(): String = {
    val nodeLabels: Seq[String] = "-1 [label='In' shape='Msquare']" +: nodes.zipWithIndex.map {
      case (data: DataNode[_], id)  => s"$id [label='${data.get().toString()}' shape='box' style='filled']"
      case (transformer: TransformerNode[_], id) => s"$id [label='${transformer.getClass.getSimpleName}']"
      case (estimator: EstimatorNode[_], id) => s"$id [label='${estimator.getClass.getSimpleName}' shape='diamond']"
    } :+ s"${nodes.size} [label='Out' shape='Msquare']"

    val dataEdges: Seq[String] = dataDeps.zipWithIndex.flatMap {
      case (deps, id) => deps.map(x => s"$x -> $id")
    } :+ s"$sink -> ${nodes.size}"

    val fitEdges: Seq[String] = fitDeps.zipWithIndex.flatMap {
      case (deps, id) => deps.map(x => s"$x -> $id [dir='none' style='dashed']")
    }

    val lines = nodeLabels ++ dataEdges ++ fitEdges
    lines.mkString("digraph pipeline {\n  rankdir=LR;\n  ", "\n  ", "\n}")
  }
}

object Pipeline {

  implicit class PipelineDSL[A, B](val pipeline: Pipeline[A, B]) {
    def andThen[C](next: Pipeline[B, C]): Pipeline[A, C] = {
      val nodes = pipeline.nodes ++ next.nodes
      val dataDeps = pipeline.fitDeps ++ next.fitDeps.map(_.map {
        x => if (x == next.SOURCE) pipeline.sink else x + pipeline.nodes.size
      })
      val fitDeps = pipeline.fitDeps ++ next.fitDeps.map(_.map {
        x => if (x == next.SOURCE) pipeline.sink else x + pipeline.nodes.size
      })
      val sources = pipeline.sources ++ next.sources.map(_ + pipeline.nodes.size)
      val sink = next.sink

      Pipeline(nodes, dataDeps, fitDeps, sources, sink)
    }
  }

}