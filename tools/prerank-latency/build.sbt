// Latency of the pre-ranking seam: what a learned ranker adds to a turn, per candidate and per root.
//
// Pinned to engine 0.12.0 rather than parameterised. `ExpectimaxSearch.materialBatch` — the
// ordering this replaces, and the only honest baseline for its cost — became public in engine
// PR #256, and the artifact under test declares `engineCompatibility >=0.12.0`. Measuring the
// seam against an engine that does not have the seam would measure something else.
val engineVersion = "0.12.0"

ThisBuild / scalaVersion := "3.9.0"

lazy val root = (project in file("."))
  .settings(
    name := "prerank-latency",
    libraryDependencies ++= Seq(
      "com.fortemate"           %% "dicechess-engine" % engineVersion,
      // `optional` in the engine's POM, so it does not arrive transitively. The version is the
      // engine's own, because the point is to measure the runtime the engine would use.
      "com.microsoft.onnxruntime" % "onnxruntime"     % "1.29.0"
    ),
    fork := true,
    // The pre-rank pass runs inside a turn that is already using the machine. A benchmark that
    // lets the JVM and ONNX Runtime spread over every core measures a machine nobody serves on.
    // A heap the serving containers would actually give it. The reported numbers come from a
    // one-vCPU, 512 MB container with -Xmx384m, and a benchmark whose default is 2 GB would
    // measure different garbage collection from the environment its own report describes. The
    // pass streams roots, so this is ample; see docs/prerank/latency-v1.md for the container
    // command that produced the published table.
    javaOptions ++= Seq("-Xmx384m", s"-Dengine.version=$engineVersion")
  )
