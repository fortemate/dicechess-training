package dicechess.training.golden

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

import dicechess.engine.domain.FenParser
import dicechess.engine.search.KcpFeatures

/** Replays `probes.tsv` through the engine's [[KcpFeatures]] — the single source of truth for the `kcp-13` feature
  * vector the evaluation service serves — and writes the resulting vectors as a deterministic JSON golden corpus.
  *
  * The Python side of the training repository never reimplements these features; it loads this file and checks
  * invariants against it (see `dicechess_training.contracts.kcp13`). Extraction latency is printed to stdout, not
  * written to the fixture, so regenerating against another engine version yields a byte-comparable file.
  *
  * Usage: `sbt -Dengine.version=0.9.2 "run tests/fixtures/kcp13/probes.tsv tests/fixtures/kcp13/golden-engine-0.9.2.json"`
  */
object Kcp13Golden:

  private val Schema      = "kcp-13"
  private val Perspective = "side-to-move"
  private val Header      = List("id", "fen", "tags", "note")

  final case class Probe(id: String, fen: String, tags: String, note: String)

  /** Mirrors the evaluator: a 4-field FEN gets neutral clocks appended; anything else is whitespace-normalised. */
  def padFen(fen: String): String =
    val parts = fen.trim.split("\\s+").filter(_.nonEmpty)
    if parts.length == 4 then (parts :+ "0" :+ "1").mkString(" ") else parts.mkString(" ")

  def readProbes(path: Path): List[Probe] =
    val lines = Files.readAllLines(path, UTF_8).asScala.toList.filter(_.nonEmpty)
    require(lines.nonEmpty && lines.head.split("\t").toList == Header, s"$path: expected header ${Header.mkString("\t")}")
    lines.tail.map { line =>
      val fields = line.split("\t", -1)
      require(fields.length == Header.length, s"$path: expected ${Header.length} fields, got ${fields.length}: $line")
      Probe(fields(0), fields(1), fields(2), fields(3))
    }

  private def jsonString(value: String): String =
    "\"" + value.flatMap {
      case '"'         => "\\\""
      case '\\'        => "\\\\"
      case c if c < ' ' => f"\\u${c.toInt}%04x" // control characters are never legal raw JSON
      case c           => c.toString
    } + "\""

  private def jsonStrings(values: Seq[String]): String = values.map(jsonString).mkString("[", ", ", "]")

  def main(args: Array[String]): Unit =
    val (input, output) = args match
      case Array(in, out) => (Path.of(in), Path.of(out))
      case _              => sys.error("usage: Kcp13Golden <probes.tsv> <golden.json>")
    val engineVersion = sys.props.getOrElse("engine.version", sys.error("-Dengine.version is required"))
    val columns       = KcpFeatures.columnNames
    val probes        = readProbes(input)
    val ids           = probes.map(_.id)
    require(ids.distinct == ids, s"duplicate probe ids: ${ids.diff(ids.distinct).mkString(", ")}")

    println(f"${"probe"}%-28s ${"side"}%4s ${"median µs"}%10s ${"p95 µs"}%8s")
    val entries = probes.map { probe =>
      val state = FenParser.parse(padFen(probe.fen)).fold(err => sys.error(s"${probe.id}: $err"), identity)
      val side  = if state.activeColor.isWhite then "w" else "b"
      val features = KcpFeatures.extract(state, state.activeColor)
      require(features.length == columns.length, s"${probe.id}: ${features.length} features, expected ${columns.length}")
      require(features.forall(f => !f.isNaN && !f.isInfinite), s"${probe.id}: non-finite feature")

      var i = 0
      while i < 20 do { KcpFeatures.extract(state, state.activeColor); i += 1 }
      val samples = Array.fill(50) {
        val t0 = System.nanoTime()
        KcpFeatures.extract(state, state.activeColor)
        (System.nanoTime() - t0) / 1000.0
      }.sorted
      val median = samples(samples.length / 2)
      val p95    = samples(((samples.length - 1) * 0.95).round.toInt)
      println(f"${probe.id}%-28s $side%4s $median%10.1f $p95%8.1f")

      val tags = probe.tags.split("\\s+").toList.filter(_.nonEmpty)
      s"""    {
         |      "id": ${jsonString(probe.id)},
         |      "fen": ${jsonString(probe.fen)},
         |      "side": ${jsonString(side)},
         |      "tags": ${jsonStrings(tags)},
         |      "note": ${jsonString(probe.note)},
         |      "features": ${features.map(f => java.lang.Float.toString(f)).mkString("[", ", ", "]")}
         |    }""".stripMargin
    }

    val json =
      s"""{
         |  "schema": ${jsonString(Schema)},
         |  "perspective": ${jsonString(Perspective)},
         |  "engineVersion": ${jsonString(engineVersion)},
         |  "engineArtifact": ${jsonString(s"com.fortemate:dicechess-engine_3:$engineVersion")},
         |  "generator": "tools/kcp13-golden (dicechess.training.golden.Kcp13Golden)",
         |  "columns": ${jsonStrings(columns)},
         |  "probes": [
         |${entries.mkString(",\n")}
         |  ]
         |}
         |""".stripMargin
    Files.createDirectories(output.toAbsolutePath.getParent)
    Files.writeString(output, json, UTF_8)
    println(s"wrote ${probes.length} probes x ${columns.length} features to $output (engine $engineVersion)")
