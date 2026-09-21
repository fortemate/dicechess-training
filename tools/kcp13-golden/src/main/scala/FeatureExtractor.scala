package dicechess.training.golden

import dicechess.engine.domain.{Color, GameState}
import dicechess.engine.search.{KcpFeatures, RichFeatures}

/** One schema's answer: the engine's column layout and the engine's extractor for it. */
final case class FeatureExtractor(columns: Seq[String], extract: (GameState, Color) => Array[Float])

/** The schemas every released engine this tool can resolve publishes.
  *
  * `-Dengine.version` exists so the same probes can be replayed against several released engines, and that promise
  * only holds if the sources compile against the oldest of them. `KcpMobilityFeatures` and
  * `KcpMobilityPawnsFeatures` arrived in engine 0.9.3; naming them here would make a replay against an earlier
  * release fail to compile rather than run, so they live in `src/main/scala-mobility/Schemas.scala`, which the build
  * adds only when the selected engine actually publishes them. See `Schemas`, which has one variant per tier.
  */
object BaseSchemas:

  val byId: Map[String, FeatureExtractor] = Map(
    "kcp-13"    -> FeatureExtractor(KcpFeatures.columnNames, (s, c) => KcpFeatures.extract(s, c)),
    "rich-9-v1" -> FeatureExtractor(RichFeatures.columnNames, (s, c) => RichFeatures.extract(s, c))
  )
