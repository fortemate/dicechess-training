package dicechess.training.latency

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.util.Collections
import scala.jdk.CollectionConverters.*

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}
import dicechess.engine.domain.*
import dicechess.engine.search.{Evaluator, RichFeatures, TurnGenerator}

/** What a learned pre-ranker costs at the seam it would be served in.
  *
  * The seam is the reason this number decides anything. The pre-rank pass sees **every** legal
  * turn and is paid in full *before* the search consults its deadline, because its output is what
  * the anytime fallback plays. So its cost is not amortised over a search that might finish early
  * — it is subtracted from the turn budget up front, on every turn, at the full branching factor.
  *
  * Three costs are measured on the same real positions, so they can be compared rather than
  * quoted side by side:
  *
  *   - `material` — the ordering the engine ships and the one this would replace. Its own
  *     `ExpectimaxSearch.materialBatch` is `private[search]`, so the same arithmetic is called
  *     through the public `Evaluator.evaluateMaterial`, once per candidate. That is what the
  *     engine's pre-ranker does; what it is not is the identical method, so treat this as the
  *     cost of a material pass rather than as that method's own number.
  *   - `features` — `RichFeatures.extract` per candidate, which material does not pay at all.
  *   - `inference` — one ONNX Runtime call on the whole batch.
  *
  * The learned ranker's true added cost is `features + inference - material`, and reporting the
  * parts separately says where it went if it is ever too much.
  *
  * One caveat the numbers cannot carry. This is the cost of replacing the *material* pre-ranker.
  * A bot that already sets `PRE_RANK_WITH_MODEL=true` — `dexus-atlas-1` does — already pays the
  * feature extraction and an inference of its own, so for that bot the incremental cost of
  * swapping in this model is the difference between two inferences, which is close to nothing.
  * The table below is the pessimistic case.
  *
  * The candidate counts here are **raw legal turn paths**, not the deduplicated decisions the
  * training corpus holds: 78% of paths are transpositions, so a root whose corpus group has 2,420
  * candidates can hand this pass 17,000 states. Whether the engine deduplicates before
  * pre-ranking is not something this benchmark can see, so it measures the pessimistic reading.
  *
  * Threads are pinned to one. The pass runs inside a turn that is already using the machine, and
  * the bots this serves run on small Cloud Run containers; a benchmark that lets ONNX Runtime
  * spread over every core measures a machine nobody serves on. The unpinned number is reported
  * beside it so the difference is visible rather than assumed.
  *
  * ```text
  * sbt "run <model.onnx> <roots.tsv> [repeats]"
  * ```
  */
object PreRankLatency:

  private val Warmup  = 5
  //: Upper bounds of the reported buckets. The last is open: without it every root above it —
  //: including the widest, which the summary line quotes — would be counted nowhere, and the
  //: bucket sizes would not add up to the roots measured.
  private val Buckets = List(8, 16, 32, 64, 128, 256, 512, 1024, 4096, Int.MaxValue)

  /** `learned` is the whole extract-and-score pass timed as one thing, and it is what `added`
    * is computed from. `features` and `inference` are its parts, kept for diagnosis only: their
    * minima can fall in different repetitions, so their sum can be lower than any pass that
    * actually happened and must not be used as a total.
    */
  final case class Sample(
      candidates: Int,
      material: Double,
      features: Double,
      inference: Double,
      learned: Double
  ):
    def added: Double = learned - material

  def main(args: Array[String]): Unit =
    val (modelPath, rootsPath, repeats) = args match
      case Array(model, roots)          => (Path.of(model), Path.of(roots), 20)
      case Array(model, roots, count)   => (Path.of(model), Path.of(roots), count.toInt)
      case _ => sys.error("usage: PreRankLatency <model.onnx> <roots.tsv> [repeats]")
    // Without this the timing loop never runs and `Long.MaxValue` is reported as a duration.
    require(repeats > 0, s"repeats must be positive, got $repeats")

    val lines = rootLines(rootsPath)
    require(lines.nonEmpty, "the roots file has no rows")
    println(s"[latency] ${lines.size} roots")

    val single = measure(modelPath, lines, repeats, threads = 1)
    report("one thread", single, repeats)
    val many = measure(modelPath, lines, repeats, threads = 0)
    report("unpinned", many, repeats)

    Files.writeString(Path.of("prerank-latency.json"), json(single, many, repeats), UTF_8)
    println("[latency] wrote prerank-latency.json")

  /** The roots, as lines. Candidate states are built one root at a time and thrown away after it
    * is measured: holding every root's candidates at once needs gigabytes — 438,000 states for
    * 576 roots — and a benchmark that cannot run in the memory the bots are given is measuring
    * the wrong machine. */
  private def rootLines(rootsPath: Path): List[String] =
    Files.readAllLines(rootsPath, UTF_8).asScala.toList.filter(_.nonEmpty).tail

  private def candidatesOf(line: String): Option[(Array[GameState], Color)] =
    val fields = line.split("\t", -1)
    val padded = fields(1).trim.split("\\s+").filter(_.nonEmpty)
    val dfen   = s"${(padded :+ "0" :+ "1").mkString(" ")} ${fields(2)}"
    FenParser.parse(dfen).toOption.flatMap { state =>
      val paths = TurnGenerator.generateAllLegalTurnPaths(state)
      if paths.isEmpty then None
      else
        val states = paths.map(path => path.foldLeft(state)((s, m) => s.makeMove(m)).endTurn())
        Some((states.toArray, state.activeColor))
    }

  private def measure(
      modelPath: Path,
      lines: List[String],
      repeats: Int,
      threads: Int
  ): List[Sample] =
    val environment = OrtEnvironment.getEnvironment
    val options     = new OrtSession.SessionOptions()
    if threads > 0 then
      options.setIntraOpNumThreads(threads)
      options.setInterOpNumThreads(threads)
    val session = environment.createSession(modelPath.toString, options)
    try
      // Warm up the JIT and the session on the first few roots, so the first measured one is
      // not paying for everything that only happens once.
      lines.iterator.take(Warmup).flatMap(candidatesOf(_).iterator).foreach { (states, colour) =>
        val _ = materialPass(states, colour)
        score(environment, session, extract(states, colour))
      }

      // An iterator, not a list: each root's candidate states are built, measured and become
      // garbage before the next root is read. `List.flatMap` would materialise all of them and
      // put the benchmark back over the memory the serving containers have.
      lines.iterator.flatMap(candidatesOf(_).iterator).map { (states, colour) =>
        var material  = Long.MaxValue
        var features  = Long.MaxValue
        var inference = Long.MaxValue
        var learned   = Long.MaxValue
        (1 to repeats).foreach { _ =>
          val t0 = System.nanoTime()
          val _  = materialPass(states, colour)
          val t1 = System.nanoTime()
          val matrix = extract(states, colour)
          val t2     = System.nanoTime()
          score(environment, session, matrix)
          val t3 = System.nanoTime()
          // The minimum, not the mean: every sample is the same work, so anything above the
          // floor is the machine doing something else. A mean here measures the laptop.
          material = math.min(material, t1 - t0)
          features = math.min(features, t2 - t1)
          inference = math.min(inference, t3 - t2)
          learned = math.min(learned, t3 - t1)
        }
        Sample(
          states.length,
          material / 1000.0,
          features / 1000.0,
          inference / 1000.0,
          learned / 1000.0
        )
      }.toList
    finally
      session.close()
      options.close()

  private def materialPass(states: Array[GameState], colour: Color): Array[Int] =
    states.map(state => Evaluator.evaluateMaterial(state, colour))

  private def extract(states: Array[GameState], colour: Color): Array[Array[Float]] =
    states.map(state => RichFeatures.extract(state, colour))

  private def score(
      environment: OrtEnvironment,
      session: OrtSession,
      matrix: Array[Array[Float]]
  ): Unit =
    val tensor = OnnxTensor.createTensor(environment, matrix)
    try
      val result = session.run(Collections.singletonMap("input", tensor))
      result.close()
    finally tensor.close()

  private def report(label: String, samples: List[Sample], repeats: Int): Unit =
    println("=" * 86)
    println(s"Pre-rank pass, $label, best of $repeats — microseconds")
    println("=" * 86)
    println(f"${"candidates"}%12s ${"roots"}%7s ${"material"}%10s ${"features"}%10s ${"model"}%10s ${"added"}%10s ${"per cand"}%9s")
    Buckets.foreach { upper =>
      val lower = Buckets.takeWhile(_ < upper).lastOption.getOrElse(0)
      val group = samples.filter(s => s.candidates > lower && s.candidates <= upper)
      if group.nonEmpty then
        val n     = group.map(_.candidates.toDouble).sum / group.size
        val mat   = median(group.map(_.material))
        val feat  = median(group.map(_.features))
        val inf   = median(group.map(_.inference))
        val added = median(group.map(_.added))
        val label = if upper == Int.MaxValue then s"> $lower" else s"<= $upper"
        println(f"$label%12s ${group.size}%7d $mat%10.1f $feat%10.1f $inf%10.1f $added%10.1f ${added / n}%9.2f")
    }
    println(f"${"measured"}%12s ${samples.size}%7d roots in total")
    val worst = samples.maxBy(_.candidates)
    val added = worst.added
    // `added` is microseconds and the budget is 2000 ms, so the share is added / 20_000 — the
    // division already yields percent and must not be multiplied by a hundred again.
    println(
      f"widest root: ${worst.candidates} candidates, added ${added / 1000}%.2f ms " +
        f"(${added / 20000}%.2f%% of a 2000 ms turn budget)"
    )
    val typical = samples.sortBy(_.candidates)(using Ordering[Int])(samples.length / 2)
    val cost    = typical.added
    println(
      f"median root: ${typical.candidates} candidates, added ${cost / 1000}%.3f ms " +
        f"(${cost / 20000}%.3f%% of that budget)"
    )

  private def median(values: List[Double]): Double =
    val sorted = values.sorted
    sorted(sorted.length / 2)

  private def json(single: List[Sample], many: List[Sample], repeats: Int): String =
    def block(label: String, samples: List[Sample]): String =
      val rows = samples
        .map(s =>
          s"""    {"candidates": ${s.candidates}, "material_us": ${s.material}, """ +
            s""""features_us": ${s.features}, "inference_us": ${s.inference}, "learned_us": ${s.learned}}"""
        )
        .mkString(",\n")
      s"""  "$label": [
         |$rows
         |  ]""".stripMargin
    s"""{
       |  "benchmark": "prerank-latency-v1",
       |  "repeats": $repeats,
       |  "statistic": "minimum over repeats, per root",
       |  "engine": "${sys.props.getOrElse("engine.version", "unknown")}",
       |${block("one_thread", single)},
       |${block("unpinned", many)}
       |}
       |""".stripMargin
