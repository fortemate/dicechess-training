// Golden-vector generator for the `kcp-13` serving contract (see docs/decisions/0001-*.md).
// The engine version is a parameter so the same probes can be replayed against several released
// engines: `sbt -Dengine.version=0.9.2 "run <probes.tsv> <golden.json>"`.
val engineVersion = sys.props.getOrElse("engine.version", "0.9.3")

ThisBuild / scalaVersion := "3.9.0"

lazy val root = (project in file("."))
  .settings(
    name                := "kcp13-golden",
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
