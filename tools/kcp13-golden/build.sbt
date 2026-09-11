// Golden-vector generator for the `kcp-13` serving contract (see docs/decisions/0001-*.md).
// The engine version is a parameter so the same probes can be replayed against several released
// engines: `sbt -Dengine.version=0.9.2 "run <probes.tsv> <golden.json>"`.
val engineVersion = sys.props.getOrElse("engine.version", sys.error("pass -Dengine.version=<released dicechess-engine version>"))

ThisBuild / scalaVersion := "3.9.0"

lazy val root = (project in file("."))
  .settings(
    name                := "kcp13-golden",
    libraryDependencies += "com.fortemate" %% "dicechess-engine" % engineVersion,
    fork                := true,
    javaOptions         += s"-Dengine.version=$engineVersion"
  )
