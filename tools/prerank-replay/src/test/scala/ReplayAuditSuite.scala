import dicechess.engine.domain.*
import dicechess.engine.movegen.Dfen
import dicechess.engine.search.RichFeatures
import java.nio.file.Files
import com.google.gson.JsonParser

class ReplayAuditSuite extends munit.FunSuite:
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
      ("7k/8/8/8/8/8/P7/K7 w - -", "BBP", "a2a4"),
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
    val dir = Files.createTempDirectory("replay-synthetic-")
    val input = dir.resolve("groups.json")
    val output = dir.resolve("report.json")
    val states = dir.resolve("states.jsonl")
    Files.writeString(input,"""[{"group_id":"synthetic","game_id":"synthetic","root_fen":"7k/8/8/8/8/8/P7/K7 w - -","dice":"PPP","side":"w","candidates":[{"moves":["a2a3"],"result_fen":"bad","features":[0],"target":0}]}]""")
    ReplayAudit.main(Array(input.toString,output.toString,states.toString))
    val counts = JsonParser.parseString(Files.readString(output)).getAsJsonObject.getAsJsonObject("counts")
    assertEquals(counts.get("illegal_paths").getAsLong,1L)
    assert(counts.get("missing_normalized_candidates").getAsLong > 0)
    assertEquals(counts.get("extra_normalized_candidates").getAsLong,1L)
  }

  test("audit detects corrupted feature values and terminal sentinel on a legal path") {
    val dir = Files.createTempDirectory("replay-synthetic-corrupt-")
    val input = dir.resolve("groups.json")
    val output = dir.resolve("report.json")
    val states = dir.resolve("states.jsonl")
    Files.writeString(input,"""[{"group_id":"synthetic","game_id":"synthetic","root_fen":"7k/8/8/8/8/8/P7/K7 w - -","dice":"BBP","side":"w","candidates":[{"moves":["a2a4"],"result_fen":"bad","features":[0],"target":2147483647}]}]""")
    ReplayAudit.main(Array(input.toString,output.toString,states.toString))
    val counts = JsonParser.parseString(Files.readString(output)).getAsJsonObject.getAsJsonObject("counts")
    assertEquals(counts.get("illegal_paths").getAsLong,0L)
    assertEquals(counts.get("feature_mismatches").getAsLong,1L)
    assertEquals(counts.get("result_mismatches").getAsLong,1L)
    assertEquals(counts.get("terminal_target_mismatches").getAsLong,1L)
  }

  test("different complete legal paths can share a normalized key but retain different EP") {
    val root = ReplayAudit.parseRoot("7k/8/8/1p6/8/8/P7/2Q4K w - -", "PQQ")
    val all = ReplayAudit.entries(root).map(e => e.moves -> e.state).toMap
    val crossed = all("a2a4 c1a3 a3c5")
    val retained = all("a2a4 c1c3 c3c5")
    assertEquals(Dfen.normalizedFen(crossed),Dfen.normalizedFen(retained))
    assert(ReplayAudit.positionKey(crossed) != ReplayAudit.positionKey(retained))
  }
