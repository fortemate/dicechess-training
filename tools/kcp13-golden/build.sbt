// Golden-vector generator for the `kcp-13` serving contract (see docs/decisions/0001-*.md).
// The engine version is a parameter so the same probes can be replayed against several released
// engines: `sbt -Dengine.version=0.9.2 "run <probes.tsv> <golden.json>"`.
val engineVersion = sys.props.getOrElse("engine.version", "0.9.3")

// `KcpMobilityFeatures` and `KcpMobilityPawnsFeatures` first appear in engine 0.9.3. Sources that name them are
// therefore kept out of the always-compiled tree and added back only for engines that publish them, because the
// parameter above is worth nothing if selecting an older engine turns into a compile error about a class that
// engine never had. The enrichment producer and the extraction benchmark exist to serve those schemas, so they
// live in the same tier; `Kcp13Golden` itself compiles against every resolvable engine.
val mobilityFloor = (0, 9, 3)

def versionTriple(version: String): (Int, Int, Int) = {
  val parts = version.split("[.\\-+]").flatMap(part => scala.util.Try(part.toInt).toOption)
  (parts.lift(0).getOrElse(0), parts.lift(1).getOrElse(0), parts.lift(2).getOrElse(0))
}

val hasMobilitySchemas = Ordering[(Int, Int, Int)].gteq(versionTriple(engineVersion), mobilityFloor)
val schemaTier         = if (hasMobilitySchemas) "scala-mobility" else "scala-legacy"

ThisBuild / scalaVersion := "3.9.0"

lazy val root = (project in file("."))
  .settings(
    name                                  := "kcp13-golden",
    Compile / unmanagedSourceDirectories += (Compile / sourceDirectory).value / schemaTier,
    Test / unmanagedSourceDirectories ++= {
      if (hasMobilitySchemas) Seq((Test / sourceDirectory).value / "scala-mobility") else Nil
    },
    libraryDependencies ++= Seq(
      "com.fortemate"     %% "dicechess-engine" % engineVersion,
      "org.apache.parquet" % "parquet-avro"     % "1.14.4",
      "org.apache.hadoop"  % "hadoop-client"    % "3.4.1",
      "org.slf4j"          % "slf4j-nop"        % "2.0.16",
      "org.scalameta"     %% "munit"            % "1.3.0" % Test
    ),
    fork                := true,
    javaOptions         += s"-Dengine.version=$engineVersion"
  )
