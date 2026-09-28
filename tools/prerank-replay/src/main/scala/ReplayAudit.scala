import dicechess.engine.domain.*
import dicechess.engine.movegen.Dfen
import dicechess.engine.search.{RichFeatures, TurnGenerator, KingCaptureProbability}
import com.google.gson.{Gson, JsonParser}
import com.google.gson.stream.JsonReader
import java.nio.file.{Files, Path}
import java.nio.charset.StandardCharsets.UTF_8
import java.security.MessageDigest
import scala.jdk.CollectionConverters.*

/** Outputs may contain corpus data: keep them with the private input. */
object ReplayAudit:
  def uci(m: Move): String =
    m.fromSquare.toNotation + m.toSquare.toNotation + m.promotionPieceType.map(_.asNotation).getOrElse("")
  def parseRoot(fen: String, dice: String): GameState =
    val fields = fen.split("\\s+")
    require(fields.length == 4 || fields.length == 6, "expected four or six FEN fields")
    val padded = if fields.length == 4 then fen + " 0 1" else fen
    FenParser.parse(padded + " " + dice).fold(e => throw IllegalArgumentException(e), identity)
  // Retain ALL EP targets. Clocks survive in witnesses, but the pinned feature/evaluator
  // contracts do not use them, so they are excluded from the position-overlap key.
  def positionKey(s: GameState): String = FenParser.serialize(s).split(" ").take(4).mkString(" ")
  def sha(s: String): String =
    MessageDigest.getInstance("SHA-256").digest(s.getBytes(UTF_8)).map("%02x".format(_)).mkString
  case class Entry(moves: String, state: GameState)
  def entries(root: GameState): List[Entry] =
    TurnGenerator.generateAllLegalTurnPaths(root).map { path =>
      Entry(path.map(uci).mkString(" "), path.foldLeft(root)((s,m) => s.makeMove(m)).endTurn())
    }
  def signature(s: GameState, mover: Color): List[Double] =
    RichFeatures.extract(s,mover).map(_.toDouble).toList ++ List(
      KingCaptureProbability.kingCaptureProbability(s,mover),
      KingCaptureProbability.kingCaptureProbability(s,mover.opponent))

  def main(args: Array[String]): Unit =
    require(args.length == 3, "usage: ReplayAudit <groups.json> <report.json> <states.jsonl>")
    val counters = scala.collection.mutable.LinkedHashMap.empty[String, Long].withDefaultValue(0L)
    def add(k: String, n: Long = 1): Unit = counters(k) += n
    List("groups", "candidates", "legal_paths", "illegal_paths", "feature_mismatches",
      "result_mismatches", "side_mismatches", "missing_normalized_candidates", "extra_normalized_candidates",
      "normalized_collision_keys", "lost_full_ep_states", "signature_distinct_collisions",
      "candidate_ep_reconstruction_differences", "terminal_candidates", "terminal_target_mismatches").foreach(add(_,0))
    val witnesses = new java.util.ArrayList[Object]()
    val collisionExamples = new java.util.ArrayList[Object]()
    val gson = new Gson()
    val started = System.nanoTime()
    val reader = new JsonReader(Files.newBufferedReader(Path.of(args(0)), UTF_8))
    val states = Files.newBufferedWriter(Path.of(args(2)), UTF_8)
    try
      reader.beginArray()
      while reader.hasNext do
        val g = JsonParser.parseReader(reader).getAsJsonObject
        add("groups")
        val root = parseRoot(g.get("root_fen").getAsString, g.get("dice").getAsString)
        val mover = root.activeColor
        if g.get("side").getAsString != (if mover.isWhite then "w" else "b") then add("side_mismatches")
        val all = entries(root)
        add("legal_paths", all.size)
        val byMoves = all.map(e => e.moves -> e).toMap
        val byNormalized = all.groupBy(e => Dfen.normalizedFen(e.state))
        val candidates = g.getAsJsonArray("candidates").asScala.toList.map(_.getAsJsonObject)
        val storedKeys = candidates.map(_.get("result_fen").getAsString).toSet
        add("missing_normalized_candidates", (byNormalized.keySet -- storedKeys).size)
        add("extra_normalized_candidates", (storedKeys -- byNormalized.keySet).size)
        val hashes = new java.util.ArrayList[String]()
        candidates.foreach { c =>
          add("candidates")
          val path = c.getAsJsonArray("moves").asScala.map(_.getAsString).mkString(" ")
          byMoves.get(path) match
            case None => add("illegal_paths")
            case Some(e) =>
              val after = e.state
              hashes.add(sha(positionKey(after)))
              if Dfen.normalizedFen(after) != c.get("result_fen").getAsString then add("result_mismatches")
              if positionKey(after) != c.get("result_fen").getAsString then add("candidate_ep_reconstruction_differences")
              val actual = RichFeatures.extract(after,mover)
              val expected = c.getAsJsonArray("features").asScala.map(_.getAsDouble).toArray
              if actual.length != expected.length || actual.zip(expected).exists((a,b) => math.abs(a-b)>1e-6) then add("feature_mismatches")
              val enemy = if mover.isWhite then after.blackPieces else after.whitePieces
              val terminal = (after.kings & enemy).isEmpty
              if terminal then add("terminal_candidates")
              if terminal != (c.get("target").getAsDouble == Int.MaxValue.toDouble) then add("terminal_target_mismatches")
        }
        byNormalized.toList.sortBy(_._1).foreach { (key, same) =>
          val distinct = same.groupBy(e => positionKey(e.state)).toList.sortBy(_._1).map(_._2.head)
          if distinct.size > 1 then
            add("normalized_collision_keys")
            add("lost_full_ep_states",distinct.size-1)
            if collisionExamples.size < 8 then
              val example = new java.util.LinkedHashMap[String,Object]()
              example.put("group_id",g.get("group_id").getAsString)
              example.put("root_fen",g.get("root_fen").getAsString)
              example.put("dice",g.get("dice").getAsString)
              example.put("normalized",key)
              example.put("paths",distinct.map(_.moves).asJava)
              example.put("full_states",distinct.map(e => FenParser.serialize(e.state)).asJava)
              collisionExamples.add(example)
            val signatures = distinct.map(e => signature(e.state,mover))
            if signatures.distinct.size > 1 then
              add("signature_distinct_collisions")
              if witnesses.size < 8 then
                val w = new java.util.LinkedHashMap[String,Object]()
                w.put("group_id",g.get("group_id").getAsString)
                w.put("root_fen",g.get("root_fen").getAsString)
                w.put("dice",g.get("dice").getAsString)
                w.put("normalized",key)
                w.put("paths",distinct.map(_.moves).asJava)
                w.put("full_states",distinct.map(e => FenParser.serialize(e.state)).asJava)
                w.put("rich9_and_kcp",signatures.map(_.map(Double.box).asJava).asJava)
                witnesses.add(w)
        }
        val record = new java.util.LinkedHashMap[String,Object]()
        record.put("group_id",g.get("group_id").getAsString)
        record.put("game_id",g.get("game_id").getAsString)
        record.put("root_state_sha256",sha(positionKey(root)))
        record.put("candidate_state_sha256",hashes)
        states.write(gson.toJson(record)); states.newLine()
        if counters("groups") % 100 == 0 then System.err.println(s"audited ${counters("groups")} groups")
      reader.endArray()
    finally
      reader.close()
      states.close()
    val report = new java.util.LinkedHashMap[String,Object]()
    report.put("schema","prerank-state-audit-v1")
    report.put("engine_version",classOf[GameState].getPackage.getImplementationVersion)
    report.put("input_sha256",MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(Path.of(args(0)))).map("%02x".format(_)).mkString)
    report.put("counts",counters.map((k,v) => k -> Long.box(v)).toMap.asJava)
    report.put("witnesses",witnesses)
    report.put("ep_collision_examples",collisionExamples)
    report.put("elapsed_seconds",Double.box((System.nanoTime()-started)/1e9))
    report.put("target_parity","terminal sentinel only; nonterminal teacher requires private evaluator replay")
    Files.writeString(Path.of(args(1)),gson.toJson(report)+"\n",UTF_8)
    println(gson.toJson(counters.map((k,v) => k -> Long.box(v)).toMap.asJava))
