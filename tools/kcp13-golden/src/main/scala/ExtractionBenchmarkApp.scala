package dicechess.training.golden

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.time.Instant
import java.util.Locale
import scala.jdk.CollectionConverters.*

import dicechess.engine.domain.{Color, FenParser, GameState}
import dicechess.engine.search.{KcpFeatures, KcpMobilityFeatures, KcpMobilityPawnsFeatures}

/** Reproducible feature extraction latency benchmark across candidate schemas.
  *
  * Measures latency (warmup = 20, samples = 50) for each probe position across
  * S0 (kcp-13), S1 (kcp-mobility-27-v1), and S2 (kcp-mobility-pawns-31-v1).
  * Records runtime environment provenance and writes a structured JSON artifact.
  *
  * Usage:
  *   sbt "runMain dicechess.training.golden.ExtractionBenchmarkApp tests/fixtures/kcp13/probes.tsv tests/fixtures/benchmark/extraction-cost-0.9.3.json"
  */
object ExtractionBenchmarkApp:

  private val Schema = "playground-extraction-benchmark-v1"

  private case class SchemaConfig(
      key: String,
      schemaId: String,
      featureCount: Int,
      extract: (GameState, Color) => Array[Float]
  )

  private def resolvePath(raw: String): Path =
    val p = Path.of(raw)
    if Files.exists(p) then p
    else
      val alt = Path.of("../..").resolve(p)
      if Files.exists(alt) then alt else p

  def main(args: Array[String]): Unit =
    val probesPath = if args.length > 0 then resolvePath(args(0)) else resolvePath("tests/fixtures/kcp13/probes.tsv")
    val outputPath = if args.length > 1 then Path.of(args(1)) else resolvePath("tests/fixtures/benchmark/extraction-cost-0.9.3.json")
    val engineVersion = sys.props.getOrElse("engine.version", "0.9.3")

    val schemas = List(
      SchemaConfig("S0", "kcp-13", 13, (s, c) => KcpFeatures.extract(s, c)),
      SchemaConfig("S1", "kcp-mobility-27-v1", 27, (s, c) => KcpMobilityFeatures.extract(s, c)),
      SchemaConfig("S2", "kcp-mobility-pawns-31-v1", 31, (s, c) => KcpMobilityPawnsFeatures.extract(s, c))
    )

    val probes = Kcp13Golden.readProbes(probesPath)
    println(s"Benchmarking extraction cost on ${probes.length} probes across ${schemas.length} schemas...")

    val resultsBuilder = new java.lang.StringBuilder()
    resultsBuilder.append("{\n")
    resultsBuilder.append(s"""  "schema": "$Schema",\n""")
    resultsBuilder.append(s"""  "engine_version": "$engineVersion",\n""")
    resultsBuilder.append(s"""  "engine_artifact": "com.fortemate:dicechess-engine_3:$engineVersion",\n""")
    resultsBuilder.append(s"""  "timestamp": "${Instant.now()}",\n""")
    resultsBuilder.append("  \"runtime\": {\n")
    resultsBuilder.append(s"""    "java_version": "${sys.props.getOrElse("java.version", "unknown")}",\n""")
    resultsBuilder.append(s"""    "java_vendor": "${sys.props.getOrElse("java.vendor", "unknown")}",\n""")
    resultsBuilder.append(s"""    "os_name": "${sys.props.getOrElse("os.name", "unknown")}",\n""")
    resultsBuilder.append(s"""    "os_arch": "${sys.props.getOrElse("os.arch", "unknown")}"\n""")
    resultsBuilder.append("  },\n")
    resultsBuilder.append("  \"config\": {\n")
    resultsBuilder.append("    \"warmup_iterations\": 20,\n")
    resultsBuilder.append("    \"sample_iterations\": 50\n")
    resultsBuilder.append("  },\n")
    resultsBuilder.append("  \"probes\": {\n")

    val probeJsonEntries = probes.zipWithIndex.map { case (probe, pIdx) =>
      val state = FenParser.parse(Kcp13Golden.padFen(probe.fen)).fold(err => sys.error(s"${probe.id}: $err"), identity)
      val side = state.activeColor

      val schemaEntries = schemas.map { sch =>
        // Warmup
        var w = 0
        while w < 20 do
          sch.extract(state, side)
          w += 1

        // Timed samples
        val samples = Array.fill(50) {
          val t0 = System.nanoTime()
          sch.extract(state, side)
          (System.nanoTime() - t0) / 1000.0 // microseconds
        }.sorted

        val median = samples(samples.length / 2)
        val p95 = samples(((samples.length - 1) * 0.95).round.toInt)
        val min = samples.head
        val max = samples.last

        s"""      "${sch.key}": {\n""" +
        s"""        "schema_id": "${sch.schemaId}",\n""" +
        s"""        "median_us": ${String.format(Locale.ROOT, "%.2f", median)},\n""" +
        s"""        "p95_us": ${String.format(Locale.ROOT, "%.2f", p95)},\n""" +
        s"""        "min_us": ${String.format(Locale.ROOT, "%.2f", min)},\n""" +
        s"""        "max_us": ${String.format(Locale.ROOT, "%.2f", max)}\n""" +
        s"""      }"""
      }

      s"""    "${probe.id}": {\n""" +
      s"""      "fen": "${probe.fen}",\n""" +
      s"""      "schemas": {\n${schemaEntries.mkString(",\n")}\n      }\n""" +
      s"""    }"""
    }

    resultsBuilder.append(probeJsonEntries.mkString(",\n"))
    resultsBuilder.append("\n  }\n}\n")

    Files.createDirectories(outputPath.toAbsolutePath.getParent)
    Files.writeString(outputPath, resultsBuilder.toString, UTF_8)
    println(s"Wrote extraction benchmark results to: $outputPath")
