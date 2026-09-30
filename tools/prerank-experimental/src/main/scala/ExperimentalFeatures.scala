import dicechess.engine.domain.*
import dicechess.engine.movegen.MoveGenerator
import dicechess.engine.search.RichFeatures
import com.google.gson.{Gson, JsonParser}
import com.google.gson.stream.{JsonReader, JsonToken}
import java.nio.file.{Files, Path, StandardOpenOption}
import java.nio.charset.StandardCharsets.UTF_8
import java.security.MessageDigest
import scala.jdk.CollectionConverters.*

/** Experimental features only. Does not label data or admit it for external transfer. */
object ExperimentalFeatures:
  val Schema = "experimental-prerank-features-v1"
  val SafetyColumns = List("own_attacked", "own_attacked_undefended",
    "opponent_attacked", "opponent_attacked_undefended")
  case class Features(a: Array[Float], b: Array[Float], c: Array[Array[Int]], terminal: Boolean)
  def parse(fen: String): GameState =
    require(fen.split("\\s+").length == 6, "six-field FEN required")
    FenParser.parse(fen).fold(e => throw IllegalArgumentException(e), identity)
  def extract(s: GameState, mover: Color): Features =
    val pieces = (0 until 64).flatMap { i =>
      val p = s.mailbox(Square.fromIndex(i))
      if p.isEmpty then None else Some((i,p))
    }
    val ownKings = pieces.filter((_,p) => p.pieceType == PieceType.King && p.color == mover)
    val enemyKings = pieces.filter((_,p) => p.pieceType == PieceType.King && p.color == mover.opponent)
    require(ownKings.size == 1 && enemyKings.size <= 1, "invalid king cardinality")
    val terminal = enemyKings.isEmpty
    val a = RichFeatures.extract(s,mover)
    def counts(color: Color): Array[Float] =
      val nonKings = pieces.filter((_,p) => p.color == color && p.pieceType != PieceType.King)
      val attacked = nonKings.filter((i,_) => !MoveGenerator.allAttackers(s,Square.fromIndex(i),color.opponent).isEmpty)
      Array(attacked.size.toFloat, attacked.count((i,_) => MoveGenerator.allAttackers(s,Square.fromIndex(i),color).isEmpty).toFloat)
    val b = a ++ counts(mover) ++ counts(mover.opponent)
    def orient(i: Int): Int = if mover.isWhite then i else i ^ 56
    val c = if terminal then Array(Array.emptyIntArray, Array.emptyIntArray) else
      val kings = List(ownKings.head._1,enemyKings.head._1).map(orient)
      kings.zipWithIndex.map { (king,anchor) =>
        val bucket = (king % 8)/2 + 4*((king/8)/2)
        pieces.filter((_,p) => p.pieceType != PieceType.King).map { (i,p) =>
          val category = (if p.color == mover then 0 else 5) + p.pieceType.diceValue - 1
          ((anchor*16+bucket)*10+category)*64+orient(i)
        }.sorted.toArray
      }.toArray
    Features(a,b,c,terminal)
  def sha(path: Path): String =
    val d = MessageDigest.getInstance("SHA-256")
    val in = Files.newInputStream(path)
    try
      val buffer = new Array[Byte](65536)
      var n = in.read(buffer)
      while n >= 0 do
        d.update(buffer,0,n)
        n = in.read(buffer)
    finally in.close()
    d.digest().map("%02x".format(_)).mkString
  def main(args: Array[String]): Unit =
    require(args.length == 2, "usage: ExperimentalFeatures <groups.json> <new-output-directory>")
    val input = Path.of(args(0))
    val output = Path.of(args(1))
    Files.createDirectory(output)
    val inputHash = sha(input)
    val gson = new Gson()
    val reader = new JsonReader(Files.newBufferedReader(input,UTF_8))
    val writer = Files.newBufferedWriter(output.resolve("features.jsonl"),UTF_8,StandardOpenOption.CREATE_NEW)
    var groups = 0L
    var candidates = 0L
    try
      reader.beginArray()
      while reader.hasNext do
        val g = JsonParser.parseReader(reader).getAsJsonObject
        val root = parse(g.get("root_fen").getAsString)
        val mover = root.activeColor
        require(g.get("side").getAsString == (if mover.isWhite then "w" else "b"), "root side mismatch")
        val rows = g.getAsJsonArray("candidates").asScala.map { item =>
          val c = item.getAsJsonObject
          val s = parse(c.get("result_fen").getAsString)
          require(s.activeColor == mover.opponent,"afterstate side mismatch")
          val f = extract(s,mover)
          val expected = c.getAsJsonArray("features").asScala.map(_.getAsDouble).toArray
          require(expected.length == 9 && f.a.zip(expected).forall((a,b) => math.abs(a-b)<=1e-6),"rich9 mismatch")
          require(f.terminal == (c.get("target").getAsLong == Int.MaxValue.toLong),"terminal mismatch")
          Map[String,Object]("a" -> f.a, "b" -> f.b, "c" -> f.c,
            "terminal" -> Boolean.box(f.terminal), "target" -> c.get("target"),
            "moves" -> c.get("moves")).asJava
        }.toList
        require(rows.nonEmpty,"empty group")
        writer.write(gson.toJson(Map[String,Object]("group_id" -> g.get("group_id"),
          "rows" -> rows.asJava).asJava)); writer.newLine()
        groups += 1; candidates += rows.size
      reader.endArray()
      require(reader.peek() == JsonToken.END_DOCUMENT,"trailing data")
    finally
      reader.close(); writer.close()
    require(sha(input) == inputHash,"input changed during extraction")
    val manifest = Map[String,Object]("schema" -> Schema, "engine_version" -> "0.14.0",
      "source_sha256" -> inputHash, "features_sha256" -> sha(output.resolve("features.jsonl")),
      "columns_a" -> RichFeatures.columnNames.asJava,
      "columns_b" -> (RichFeatures.columnNames ++ SafetyColumns).asJava,
      "groups" -> Long.box(groups), "candidates" -> Long.box(candidates))
    Files.writeString(output.resolve("manifest.json"),gson.toJson(manifest.asJava)+"\n",UTF_8,StandardOpenOption.CREATE_NEW)
