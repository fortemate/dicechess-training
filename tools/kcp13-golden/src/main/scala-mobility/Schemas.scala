package dicechess.training.golden

import dicechess.engine.search.{KcpMobilityFeatures, KcpMobilityPawnsFeatures}

/** Every schema an engine of 0.9.3 or later publishes. Compiled only for those engines — see [[BaseSchemas]]. */
object Schemas:

  val byId: Map[String, FeatureExtractor] = BaseSchemas.byId ++ Map(
    "kcp-mobility-27-v1" -> FeatureExtractor(
      KcpMobilityFeatures.columnNames,
      (s, c) => KcpMobilityFeatures.extract(s, c)
    ),
    "kcp-mobility-pawns-31-v1" -> FeatureExtractor(
      KcpMobilityPawnsFeatures.columnNames,
      (s, c) => KcpMobilityPawnsFeatures.extract(s, c)
    )
  )
