ThisBuild / scalaVersion := "3.9.0"
libraryDependencies ++= Seq(
  "com.fortemate" %% "dicechess-engine" % "0.12.0",
  "com.google.code.gson" % "gson" % "2.13.2",
  "org.scalameta" %% "munit" % "1.3.0" % Test
)
fork := true
javaOptions ++= Seq("-Xmx2g", "-XX:ActiveProcessorCount=2")
