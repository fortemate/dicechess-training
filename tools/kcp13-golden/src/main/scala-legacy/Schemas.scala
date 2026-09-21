package dicechess.training.golden

/** The schemas available when replaying against an engine older than 0.9.3 — see [[BaseSchemas]].
  *
  * Asking this build for a mobility schema fails with the same message as asking it for a schema that does not
  * exist at all, which is the honest answer: the selected engine does not publish that extractor, and this tool
  * never invents a feature layout the engine has not stated.
  */
object Schemas:

  val byId: Map[String, FeatureExtractor] = BaseSchemas.byId
