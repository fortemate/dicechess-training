import dicechess.engine.domain.*
import dicechess.engine.movegen.Dfen
import dicechess.engine.search.RichFeatures
import java.nio.file.Files
import java.nio.charset.StandardCharsets.UTF_8
import java.io.{ByteArrayOutputStream, PrintStream}
import com.google.gson.JsonParser
import scala.jdk.CollectionConverters.*

class ReplayAuditSuite extends munit.FunSuite:
  private val InputFile = "groups.json"
  private val CountsField = "counts"
  private val IllegalPathsField = "illegal_paths"
  private val ExtraCandidatesField = "extra_normalized_candidates"
  private val PreviousReport = "previous report"
  private val MissingCandidatesField = "missing_normalized_candidates"
  private val FeatureMismatchesField = "feature_mismatches"
  private val TerminalMismatchesField = "terminal_target_mismatches"
  private val GroupsField = "groups"
  private val CandidatesField = "candidates"
  private val GroupIdField = "group_id"
  private val ReportFile = "report.json"
  private val StatesFile = "states.jsonl"
  private val SparseRootFen = "7k/8/8/8/8/8/P7/K7 w - -"
  private val CollisionRootFen = "7k/8/8/8/r3p1K1/8/3P4/5Q2 w - -"
  private def auditCounts(prefix: String, raw: String): com.google.gson.JsonObject =
    val dir = Files.createTempDirectory(prefix)
    val input = dir.resolve(InputFile)
    val output = dir.resolve(ReportFile)
    val states = dir.resolve(StatesFile)
    Files.writeString(input,raw)
    ReplayAudit.main(Array(input.toString,output.toString,states.toString))
    JsonParser.parseString(Files.readString(output)).getAsJsonObject.getAsJsonObject(CountsField)

  test("completed replay retains delayed en-passant lost by normalization") {
    val root = ReplayAudit.parseRoot("7k/8/8/1p6/8/8/PK6/8 w - -", "BBP")
    val e = ReplayAudit.entries(root).find(_.moves == "a2a4").get
    assertEquals(e.state.activeColor, Color.Black)
    assertEquals(ReplayAudit.positionKey(e.state), "7k/8/8/1p6/P7/8/1K6/8 b - a3")
    assertEquals(Dfen.normalizedFen(e.state), "7k/8/8/1p6/P7/8/1K6/8 b - -")
    val lossy = FenParser.parse(Dfen.normalizedFen(e.state) + " 0 1").toOption.get
    val exactPaths = ReplayAudit.entries(e.state.withDicePool(List(1,1,1))).map(_.moves).toSet
    val normalizedPaths = ReplayAudit.entries(lossy.withDicePool(List(1,1,1))).map(_.moves).toSet
    assert(exactPaths != normalizedPaths)
    assert(exactPaths.contains("b5b4 b4a3 a3b2"))
    assert(!normalizedPaths.contains("b5b4 b4a3 a3b2"))
  }
  test("features use the root mover for both colors; captured king stays missing") {
    for (fen,dice,moves) <- List(
      (SparseRootFen, "BBP", "a2a4"),
      ("k7/p7/8/8/8/8/8/7K b - -", "BBP", "a7a5"),
      ("7k/7R/8/8/8/8/8/K7 w - -", "BBR", "h7h8")
    ) do
      val root = ReplayAudit.parseRoot(fen,dice)
      val after = ReplayAudit.entries(root).find(_.moves == moves).get.state
      assertEquals(after.activeColor,root.activeColor.opponent)
      val own = RichFeatures.extract(after,root.activeColor)
      val other = RichFeatures.extract(after,after.activeColor)
      assertEquals(own(5),-other(5))
      assert(own(5) > 0)
      if moves == "h7h8" then assert((after.kings & after.blackPieces).isEmpty)
  }
  test("audit counts incomplete paths and missing alternatives") {
    val counts = auditCounts("replay-synthetic-", """[{"group_id":"synthetic","game_id":"synthetic","root_fen":"7k/8/8/8/8/8/P7/K7 w - -","dice":"PPP","side":"w","candidates":[{"moves":["a2a3"],"result_fen":"bad","features":[0],"target":0}]}]""")
    assertEquals(counts.get(IllegalPathsField).getAsLong,1L)
    assert(counts.get(MissingCandidatesField).getAsLong > 0)
    assertEquals(counts.get(ExtraCandidatesField).getAsLong,1L)
  }

  test("audit detects corrupted feature values and terminal sentinel on a legal path") {
    val counts = auditCounts("replay-synthetic-corrupt-", """[{"group_id":"synthetic","game_id":"synthetic","root_fen":"7k/8/8/8/8/8/P7/K7 w - -","dice":"BBP","side":"w","candidates":[{"moves":["a2a4"],"result_fen":"bad","features":[0],"target":2147483647}]}]""")
    assertEquals(counts.get(IllegalPathsField).getAsLong,0L)
    assertEquals(counts.get(FeatureMismatchesField).getAsLong,1L)
    assertEquals(counts.get("result_mismatches").getAsLong,1L)
    assertEquals(counts.get("candidate_ep_reconstruction_differences").getAsLong,0L)
    assertEquals(counts.get(TerminalMismatchesField).getAsLong,1L)
  }

  test("different complete legal paths can share a normalized key but retain different EP") {
    val root = ReplayAudit.parseRoot("7k/8/8/1p6/8/8/P7/2Q4K w - -", "PQQ")
    val all = ReplayAudit.entries(root).map(e => e.moves -> e.state).toMap
    val crossed = all("a2a4 c1a3 a3c5")
    val retained = all("a2a4 c1c3 c3c5")
    assertEquals(Dfen.normalizedFen(crossed),Dfen.normalizedFen(retained))
    assert(ReplayAudit.positionKey(crossed) != ReplayAudit.positionKey(retained))
  }

  test("reject aliases before opening output writers and preserve input bytes") {
    val dir = Files.createTempDirectory("replay-alias-")
    val input = dir.resolve("input.json")
    Files.writeString(input,"[]")
    val report = dir.resolve(ReportFile)
    val states = dir.resolve(StatesFile)
    val symlink = Files.createSymbolicLink(dir.resolve("alias.json"), input)
    val hardlink = Files.createLink(dir.resolve("hard.json"), input)
    val parentAlias = Files.createSymbolicLink(dir.resolve("parent"),dir)
    for paths <- List(
      List(input,report,input), List(input,input,states), List(input,report,report),
      List(input,report,symlink), List(input,hardlink,states),
      List(input,report,parentAlias.resolve(ReportFile))
    ) do
      intercept[IllegalArgumentException](ReplayAudit.main(paths.map(_.toString).toArray))
      assertEquals(Files.readString(input),"[]")
      assert(!Files.exists(report))
      assert(!Files.exists(states))
  }
  test("streaming input digest matches SHA-256 across multiple buffers and empty files") {
    val input = Files.createTempFile("replay-sha-", ".json")
    for value <- List("", "abc", "0123456789" * 20000) do
      Files.writeString(input,value)
      assertEquals(ReplayAudit.fileSha(input),ReplayAudit.sha(value))
    assertEquals(ReplayAudit.sha("abc"),"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
  }

  test("empty input finishes a complete report and an empty states stream") {
    val dir = Files.createTempDirectory("replay-empty-")
    val input = dir.resolve(InputFile)
    val report = dir.resolve(ReportFile)
    val states = dir.resolve(StatesFile)
    val raw = "[ ]\n"
    Files.writeString(input, raw)
    ReplayAudit.main(Array(input.toString, report.toString, states.toString))
    val result = JsonParser.parseString(Files.readString(report)).getAsJsonObject
    assertEquals(result.get("schema").getAsString, "prerank-state-audit-v1")
    assertEquals(result.get("input_sha256").getAsString, ReplayAudit.sha(raw))
    val counts = result.getAsJsonObject(CountsField)
    val names = Set(GroupsField, CandidatesField, "legal_paths", IllegalPathsField, FeatureMismatchesField,
      "result_mismatches", "side_mismatches", MissingCandidatesField, ExtraCandidatesField,
      "normalized_collision_keys", "lost_full_ep_states", "signature_distinct_collisions",
      "candidate_ep_reconstruction_differences", "terminal_candidates", TerminalMismatchesField)
    assertEquals(counts.keySet().asScala.toSet, names)
    names.foreach(name => assertEquals(counts.get(name).getAsLong, 0L))
    assertEquals(result.getAsJsonArray("witnesses").size(), 0)
    assertEquals(result.getAsJsonArray("ep_collision_examples").size(), 0)
    val elapsed = result.get("elapsed_seconds").getAsDouble
    assert(java.lang.Double.isFinite(elapsed) && elapsed > 0)
    assertEquals(Files.readString(states), "")
    assertEquals(Files.readString(input), raw)
  }

  test("empty candidate group records every missing alternative and still writes group identity") {
    val dir = Files.createTempDirectory("replay-empty-candidates-")
    val input = dir.resolve(InputFile)
    val report = dir.resolve(ReportFile)
    val states = dir.resolve(StatesFile)
    val fen = SparseRootFen
    Files.writeString(input, s"""[{"group_id":"empty-candidates","game_id":"sample","root_fen":"$fen","dice":"BBP","side":"b","candidates":[]}]""")
    ReplayAudit.main(Array(input.toString, report.toString, states.toString))
    val result = JsonParser.parseString(Files.readString(report)).getAsJsonObject
    val counts = result.getAsJsonObject(CountsField)
    assertEquals(counts.get(GroupsField).getAsLong, 1L)
    assertEquals(counts.get(CandidatesField).getAsLong, 0L)
    assertEquals(counts.get("side_mismatches").getAsLong, 1L)
    assertEquals(counts.get("legal_paths").getAsLong, 2L)
    assertEquals(counts.get(MissingCandidatesField).getAsLong, 2L)
    assertEquals(counts.get(ExtraCandidatesField).getAsLong, 0L)
    val rows = Files.readAllLines(states).asScala.toList
    assertEquals(rows.size, 1)
    val row = JsonParser.parseString(rows.head).getAsJsonObject
    assertEquals(row.get(GroupIdField).getAsString, "empty-candidates")
    assertEquals(row.get("game_id").getAsString, "sample")
    assertEquals(row.get("root_state_sha256").getAsString, ReplayAudit.sha(fen))
    assertEquals(row.getAsJsonArray("candidate_state_sha256").size(), 0)
  }

  test("illegal paths do not read fields used only for legal candidates") {
    val dir = Files.createTempDirectory("replay-illegal-lazy-")
    val input = dir.resolve(InputFile)
    val report = dir.resolve(ReportFile)
    val states = dir.resolve(StatesFile)
    val raw = """[{"group_id":"illegal","game_id":"sample","root_fen":"7k/8/8/8/8/8/P7/K7 w - -","dice":"BBP","side":"w","candidates":[{"moves":["not-a-move"],"result_fen":"bad"}]}]"""
    Files.writeString(input, raw)
    ReplayAudit.main(Array(input.toString, report.toString, states.toString))
    val result = JsonParser.parseString(Files.readString(report)).getAsJsonObject
    val counts = result.getAsJsonObject(CountsField)
    assertEquals(counts.get(CandidatesField).getAsLong, 1L)
    assertEquals(counts.get(IllegalPathsField).getAsLong, 1L)
    assertEquals(counts.get(FeatureMismatchesField).getAsLong, 0L)
    assertEquals(counts.get(TerminalMismatchesField).getAsLong, 0L)
    assertEquals(counts.get(ExtraCandidatesField).getAsLong, 1L)
    val row = JsonParser.parseString(Files.readString(states).trim).getAsJsonObject
    assertEquals(row.getAsJsonArray("candidate_state_sha256").size(), 0)
    assertEquals(Files.readString(input), raw)
  }

  test("wrong top-level JSON preserves an existing report but truncates opened states") {
    val dir = Files.createTempDirectory("replay-input-error-")
    val input = dir.resolve(InputFile)
    val report = dir.resolve(ReportFile)
    val states = dir.resolve(StatesFile)
    Files.writeString(input, "{}")
    Files.writeString(report, PreviousReport)
    Files.writeString(states, "previous states")
    intercept[IllegalStateException](ReplayAudit.main(Array(input.toString, report.toString, states.toString)))
    assertEquals(Files.readString(report), PreviousReport)
    assertEquals(Files.readString(states), "")
    assertEquals(Files.readString(input), "{}")
  }

  test("a later invalid group leaves the already completed state record and old report") {
    val dir = Files.createTempDirectory("replay-partial-error-")
    val input = dir.resolve(InputFile)
    val report = dir.resolve(ReportFile)
    val states = dir.resolve(StatesFile)
    val raw = """[{"group_id":"first","game_id":"sample","root_fen":"7k/8/8/8/8/8/P7/K7 w - -","dice":"BBP","side":"w","candidates":[]},{"root_fen":"bad","dice":"BBP"}]"""
    Files.writeString(input, raw)
    Files.writeString(report, PreviousReport)
    intercept[IllegalArgumentException](ReplayAudit.main(Array(input.toString, report.toString, states.toString)))
    assertEquals(Files.readString(report), PreviousReport)
    val rows = Files.readAllLines(states).asScala.toList
    assertEquals(rows.size, 1)
    assertEquals(JsonParser.parseString(rows.head).getAsJsonObject.get(GroupIdField).getAsString, "first")
    assertEquals(Files.readString(input), raw)
  }

  test("trailing JSON is not consumed after the completed root array") {
    val dir = Files.createTempDirectory("replay-trailing-contract-")
    val input = dir.resolve(InputFile)
    val report = dir.resolve(ReportFile)
    val states = dir.resolve(StatesFile)
    val raw = "[] {}"
    Files.writeString(input, raw)
    ReplayAudit.main(Array(input.toString, report.toString, states.toString))
    val result = JsonParser.parseString(Files.readString(report)).getAsJsonObject
    assertEquals(result.get("input_sha256").getAsString, ReplayAudit.sha(raw))
    assertEquals(result.getAsJsonObject(CountsField).get(GroupsField).getAsLong, 0L)
    assertEquals(Files.readString(states), "")
    assertEquals(Files.readString(input), raw)
  }

  test("capturable EP remains part of the normalized key") {
    val root = ReplayAudit.parseRoot(CollisionRootFen, "PQQ")
    val all = ReplayAudit.entries(root).map(e => e.moves -> e.state).toMap
    val crossed = all("d2d4 f1d3 d3b5")
    val retained = all("d2d4 f1f5 f5b5")
    assertEquals(Dfen.normalizedFen(crossed), "7k/8/8/1Q6/r2Pp1K1/8/8/8 b - -")
    assertEquals(Dfen.normalizedFen(retained), "7k/8/8/1Q6/r2Pp1K1/8/8/8 b - d3")
    assert(ReplayAudit.positionKey(crossed) != ReplayAudit.positionKey(retained))
    assertEquals(ReplayAudit.signature(crossed, Color.White).size, 11)
    assertEquals(ReplayAudit.signature(retained, Color.White).size, 11)
  }

  test("collision examples are capped without inventing signature witnesses") {
    val dir = Files.createTempDirectory("replay-collision-cap-")
    val input = dir.resolve(InputFile)
    val report = dir.resolve(ReportFile)
    val states = dir.resolve(StatesFile)
    val raw = (0 until 12).map(index => s"""{"group_id":"collision-$index","game_id":"sample","root_fen":"$CollisionRootFen","dice":"PQQ","side":"w","candidates":[]}""").mkString("[", ",", "]")
    Files.writeString(input, raw)
    ReplayAudit.main(Array(input.toString, report.toString, states.toString))
    val result = JsonParser.parseString(Files.readString(report)).getAsJsonObject
    val counts = result.getAsJsonObject(CountsField)
    assertEquals(counts.get("normalized_collision_keys").getAsLong, 12L)
    assertEquals(counts.get("signature_distinct_collisions").getAsLong, 0L)
    val examples = result.getAsJsonArray("ep_collision_examples").asScala.toList.map(_.getAsJsonObject)
    val witnesses = result.getAsJsonArray("witnesses").asScala.toList.map(_.getAsJsonObject)
    assertEquals(examples.size, 8)
    assertEquals(examples.map(_.get(GroupIdField).getAsString), (0 until 8).map(i => s"collision-$i").toList)
    assertEquals(witnesses.size, 0)
    assert(examples.forall(!_.has("rich9_and_kcp")))
    assert(examples.forall(_.getAsJsonArray("full_states").size() > 1))
    assertEquals(Files.readString(input), raw)
  }

  test("progress is emitted exactly after the hundredth completed group") {
    val dir = Files.createTempDirectory("replay-progress-")
    val input = dir.resolve(InputFile)
    val report = dir.resolve(ReportFile)
    val states = dir.resolve(StatesFile)
    val group = s"""{"group_id":"progress","game_id":"authored","root_fen":"$SparseRootFen","dice":"BBP","side":"w","candidates":[]}"""
    Files.writeString(input, List.fill(100)(group).mkString("[", ",", "]"))
    val captured = new ByteArrayOutputStream()
    val stream = new PrintStream(captured, true, UTF_8)
    val previous = System.err
    System.setErr(stream)
    try ReplayAudit.main(Array(input.toString, report.toString, states.toString))
    finally
      System.setErr(previous)
      stream.close()
    assertEquals(captured.toString(UTF_8), "audited 100 groups" + System.lineSeparator())
    assertEquals(Files.readAllLines(states).size(), 100)
    val counts = JsonParser.parseString(Files.readString(report)).getAsJsonObject.getAsJsonObject(CountsField)
    assertEquals(counts.get(GroupsField).getAsLong, 100L)
  }
