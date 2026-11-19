import dicechess.engine.domain.*
import dicechess.engine.search.RichFeatures
import com.google.gson.{Gson, JsonParser}
import java.nio.file.Files
import java.nio.charset.StandardCharsets.UTF_8
import scala.jdk.CollectionConverters.*
class ExperimentalFeaturesSuite extends munit.FunSuite:
  private val SparseAfterstate = "7k/8/8/8/8/8/P7/K7 b - - 0 1"
  def features(fen: String, mover: Color = Color.White) = ExperimentalFeatures.extract(ExperimentalFeatures.parse(fen),mover)
  test("pawn attacks use diagonals and friendly defenders protect attacked non-kings") {
    val f = features("7k/8/8/3p4/4N3/3P4/8/K7 b - - 0 1")
    assertEquals(f.b.drop(9).toList,List(1f,0f,0f,0f))
    val undefended = features("7k/8/8/3p4/4N3/8/8/K7 b - - 0 1")
    assertEquals(undefended.b.drop(9).toList,List(1f,1f,0f,0f))
    val push = features("7k/8/8/3p4/3N4/8/8/K7 b - - 0 1")
    assertEquals(push.b.drop(9).toList,List(0f,0f,0f,0f))
  }
  test("color swap and vertical reflection preserve root-relative features") {
    val white = features("7k/8/8/3p4/4N3/3P4/8/K7 b - - 0 1")
    val black = features("k7/8/3p4/4n3/3P4/8/8/7K w - - 0 1",Color.Black)
    assertEquals(white.a.toList,black.a.toList)
    assertEquals(white.b.toList,black.b.toList)
    assertEquals(white.c.map(_.toList).toList,black.c.map(_.toList).toList)
  }
  test("king buckets, category halves and square IDs are exact") {
    val f = features(SparseAfterstate)
    assertEquals(f.c.map(_.toList).toList,List(List(8),List(((16+15)*10)*64+8)))
  }
  test("terminal has no invented king anchor") {
    val f = features("8/8/8/8/8/8/P7/K7 b - - 0 1")
    assert(f.terminal)
    assert(f.c.forall(_.isEmpty))
    intercept[IllegalArgumentException](features("8/8/8/8/8/8/P7/8 b - - 0 1"))
  }

  test("several attackers and defenders count a piece once") {
    val f = features("7k/8/8/3p1p2/4N3/3P1P2/8/K7 b - - 0 1")
    assertEquals(f.b.drop(9).toList, List(1f, 0f, 0f, 0f))
  }

  test("safety counts follow the explicit mover rather than active color") {
    val f = features("7k/8/8/3p1p2/4N3/3P1P2/8/K7 b - - 0 1", Color.Black)
    assertEquals(f.b.drop(9).toList, List(0f, 0f, 1f, 0f))
  }

  test("sliding blockers and their defenders are reflected in both color counts") {
    val blocked = features("3r3k/8/3p4/8/3R4/8/8/K7 b - - 0 1")
    assertEquals(blocked.b.drop(9).toList, List(0f, 0f, 1f, 0f))
    val open = features("3r3k/8/8/8/3R4/8/8/K7 b - - 0 1")
    assertEquals(open.b.drop(9).toList, List(1f, 1f, 1f, 1f))
  }

  test("a pinned friendly piece still supplies geometric defence") {
    val f = features("4r2k/8/8/8/2p5/3NR3/8/4K3 b - - 0 1")
    assertEquals(f.b.drop(9).toList, List(2f, 1f, 1f, 1f))
  }

  test("two bare kings are nonterminal with no piece encodings or attack counts") {
    val f = features("7k/8/8/8/8/8/8/K7 b - - 0 1")
    assertEquals(f.terminal, false)
    assertEquals(f.a.length, 9)
    assertEquals(f.b.length, 13)
    assertEquals(f.b.take(9).toList, f.a.toList)
    assertEquals(f.b.drop(9).toList, List(0f, 0f, 0f, 0f))
    assertEquals(f.c.map(_.toList).toList, List(Nil, Nil))
    assert(f.a.forall(java.lang.Float.isFinite))
  }

  test("black terminal positions retain safety counts without invented anchors") {
    val f = features("7k/7p/8/8/7R/8/8/8 w - - 0 1", Color.Black)
    assertEquals(f.terminal, true)
    assertEquals(f.b.drop(9).toList, List(1f, 0f, 0f, 0f))
    assertEquals(f.c.map(_.toList).toList, List(Nil, Nil))
  }

  test("the white mover needs its own king before any feature extraction") {
    val error = intercept[IllegalArgumentException] {
      features("7k/8/8/8/8/8/P7/8 b - - 0 1")
    }
    assertEquals(error.getMessage, "requirement failed: invalid king cardinality")
  }

  test("the black mover needs its own king before any feature extraction") {
    val error = intercept[IllegalArgumentException] {
      features("8/8/8/8/8/8/P7/K7 w - - 0 1", Color.Black)
    }
    assertEquals(error.getMessage, "requirement failed: invalid king cardinality")
  }

  test("returned arrays are independent and the input mailbox is unchanged") {
    val state = ExperimentalFeatures.parse(SparseAfterstate)
    val mailbox = (0 until 64).map(i => state.mailbox(Square.fromIndex(i)))
    val first = ExperimentalFeatures.extract(state, Color.White)
    val a = first.a.toList
    val b = first.b.toList
    val c = first.c.map(_.toList).toList
    first.a(0) = Float.NaN
    assertEquals(first.b.toList, b)
    first.b(0) = Float.NaN
    first.c(0)(0) = -1
    val again = ExperimentalFeatures.extract(state, Color.White)
    assertEquals(again.a.toList, a)
    assertEquals(again.b.toList, b)
    assertEquals(again.c.map(_.toList).toList, c)
    assertEquals((0 until 64).map(i => state.mailbox(Square.fromIndex(i))), mailbox)
  }

  test("CLI extraction retains grouped terminal and nonterminal records and manifest") {
    val root = "7k/8/8/8/8/8/P7/K7 w - - 0 1"
    val terminal = "8/8/8/8/8/8/P7/K7 b - - 0 1"
    def candidate(fen: String, target: Int): java.util.Map[String, Object] =
      Map[String, Object](
        "result_fen" -> fen,
        "features" -> RichFeatures.extract(ExperimentalFeatures.parse(fen), Color.White),
        "target" -> Int.box(target),
        "moves" -> List.empty[String].asJava
      ).asJava
    val inputRows = List(Map[String, Object](
      "group_id" -> "authored-group",
      "root_fen" -> root,
      "side" -> "w",
      "candidates" -> List(candidate(SparseAfterstate, 0), candidate(terminal, Int.MaxValue)).asJava
    ).asJava).asJava
    val directory = Files.createTempDirectory("experimental-features-contract-")
    val input = directory.resolve("groups.json")
    val output = directory.resolve("new-output")
    val raw = new Gson().toJson(inputRows)
    Files.writeString(input, raw, UTF_8)
    ExperimentalFeatures.main(Array(input.toString, output.toString))
    val lines = Files.readAllLines(output.resolve("features.jsonl"), UTF_8).asScala.toList
    assertEquals(lines.size, 1)
    val group = JsonParser.parseString(lines.head).getAsJsonObject
    assertEquals(group.get("group_id").getAsString, "authored-group")
    val rows = group.getAsJsonArray("rows").asScala.toList.map(_.getAsJsonObject)
    assertEquals(rows.size, 2)
    assertEquals(rows.head.get("terminal").getAsBoolean, false)
    assertEquals(rows(1).get("terminal").getAsBoolean, true)
    assertEquals(rows.head.get("target").getAsLong, 0L)
    assertEquals(rows(1).get("target").getAsLong, Int.MaxValue.toLong)
    assertEquals(rows.head.getAsJsonArray("c").asScala.map(_.getAsJsonArray.size()).toList, List(1, 1))
    assertEquals(rows(1).getAsJsonArray("c").asScala.map(_.getAsJsonArray.size()).toList, List(0, 0))
    val manifest = JsonParser.parseString(Files.readString(output.resolve("manifest.json"), UTF_8)).getAsJsonObject
    assertEquals(manifest.get("schema").getAsString, "experimental-prerank-features-v1")
    assertEquals(manifest.get("engine_version").getAsString, "0.14.0")
    assertEquals(manifest.get("groups").getAsLong, 1L)
    assertEquals(manifest.get("candidates").getAsLong, 2L)
    assertEquals(manifest.get("source_sha256").getAsString, ExperimentalFeatures.sha(input))
    assertEquals(manifest.get("features_sha256").getAsString, ExperimentalFeatures.sha(output.resolve("features.jsonl")))
    assertEquals(Files.readString(input, UTF_8), raw)
  }
