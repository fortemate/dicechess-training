package dicechess.training.golden

import java.nio.file.{Files, Path}
import java.util.ArrayList
import scala.jdk.CollectionConverters.*
import org.apache.avro.Schema
import org.apache.avro.generic.{GenericData, GenericRecord}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path as HPath}
import org.apache.parquet.avro.{AvroParquetReader, AvroParquetWriter}
import org.apache.parquet.hadoop.ParquetFileWriter.Mode
import org.apache.parquet.hadoop.metadata.CompressionCodecName
import dicechess.engine.domain.{Color, FenParser}
import dicechess.engine.search.{KcpFeatures, KcpMobilityFeatures, KcpMobilityPawnsFeatures}

object EnrichShardsApp:

  private val BaseFields = List(
    ("game_id", "string"),
    ("ply", "int"),
    ("fen", "string"),
    ("dice", "string"),
    ("side", "string"),
    ("result", "float")
  )

  final case class RawRow(
    gameId: String,
    ply: Int,
    fen: String,
    dice: String,
    side: String,
    result: Float
  )

  final case class EnrichedRow(
    raw: RawRow,
    features: Array[Float]
  )

  def padFen(fen: String): String =
    val parts = fen.trim.split("\\s+").filter(_.nonEmpty)
    if parts.length == 4 then (parts :+ "0" :+ "1").mkString(" ") else parts.mkString(" ")

  def buildAvroSchema(featureColumns: List[String]): Schema =
    val fieldsJson = BaseFields.map { case (name, tpe) =>
      s"""{"name": "$name", "type": "$tpe"}"""
    } ++ featureColumns.map { col =>
      s"""{"name": "$col", "type": "float"}"""
    }
    val json =
      s"""{
         |  "type": "record",
         |  "name": "EnrichedPosition",
         |  "namespace": "dicechess.training",
         |  "fields": [
         |    ${fieldsJson.mkString(",\n    ")}
         |  ]
         |}""".stripMargin
    new Schema.Parser().parse(json)

  private def resolveExtractor(schemaId: String): (List[String], (dicechess.engine.domain.GameState, Color) => Array[Float]) =
    schemaId match
      case "kcp-13" =>
        (KcpFeatures.columnNames, (s: dicechess.engine.domain.GameState, c: Color) => KcpFeatures.extract(s, c))
      case "kcp-mobility-27-v1" =>
        (KcpMobilityFeatures.columnNames, (s: dicechess.engine.domain.GameState, c: Color) => KcpMobilityFeatures.extract(s, c))
      case "kcp-mobility-pawns-31-v1" =>
        (KcpMobilityPawnsFeatures.columnNames, (s: dicechess.engine.domain.GameState, c: Color) => KcpMobilityPawnsFeatures.extract(s, c))
      case other =>
        sys.error(s"unsupported schema '$other' (expected 'kcp-13', 'kcp-mobility-27-v1', or 'kcp-mobility-pawns-31-v1')")

  private def writeBatch(
      chunk: ArrayList[RawRow],
      extractor: (dicechess.engine.domain.GameState, Color) => Array[Float],
      columns: List[String],
      schema: Schema,
      writer: org.apache.parquet.hadoop.ParquetWriter[GenericRecord]
  ): Long =
    if chunk.isEmpty then 0L
    else
      val rawList = new ArrayList(chunk)
      chunk.clear()
      val enriched = rawList.parallelStream().map { raw =>
        val state = FenParser.parse(padFen(raw.fen)).fold(
          err => sys.error(s"FEN error in game ${raw.gameId} ply ${raw.ply}: $err"),
          identity
        )
        val color = raw.side match
          case "w" => Color.White
          case "b" => Color.Black
          case other => sys.error(s"Invalid side '$other' in game ${raw.gameId} ply ${raw.ply}")
        require(state.activeColor == color, s"side/FEN mismatch in game ${raw.gameId} ply ${raw.ply}")
        EnrichedRow(raw, extractor(state, color))
      }.iterator()

      var count = 0L
      while enriched.hasNext do
        val row = enriched.next()
        val outRecord = new GenericData.Record(schema)
        outRecord.put("game_id", row.raw.gameId)
        outRecord.put("ply", row.raw.ply)
        outRecord.put("fen", row.raw.fen)
        outRecord.put("dice", row.raw.dice)
        outRecord.put("side", row.raw.side)
        outRecord.put("result", row.raw.result)
        var i = 0
        while i < columns.length do
          outRecord.put(columns(i), row.features(i))
          i += 1
        writer.write(outRecord)
        count += 1L
      count

  final case class EnrichContext(
      schemaId: String,
      engineVersion: String,
      columns: List[String],
      extractor: (dicechess.engine.domain.GameState, Color) => Array[Float],
      schema: Schema,
      conf: Configuration,
      chunkSize: Int
  )

  private def enrichShard(
      shardPath: Path,
      outFile: Path,
      ctx: EnrichContext
  ): Long =
    val tmpFile = outFile.resolveSibling(s".${outFile.getFileName}.tmp")
    Files.deleteIfExists(tmpFile)
    val inHPath  = new HPath(shardPath.toAbsolutePath.toString)
    val tmpHPath = new HPath(tmpFile.toAbsolutePath.toString)

    val extraMeta: Map[String, String] = Map(
      "feature_schema"            -> ctx.schemaId,
      "engine_version"            -> ctx.engineVersion,
      "ruleset"                   -> "standard-dicechess-v1",
      "perspective"               -> "side-to-move",
      "dicechess_training_schema" -> "v0-enriched",
      "columns"                   -> ctx.columns.mkString(",")
    )

    val reader = AvroParquetReader.builder[GenericRecord](inHPath).withConf(ctx.conf).build()
    val writer = AvroParquetWriter.builder[GenericRecord](tmpHPath)
      .withSchema(ctx.schema)
      .withConf(ctx.conf)
      .withCompressionCodec(CompressionCodecName.SNAPPY)
      .withWriteMode(Mode.OVERWRITE)
      .withExtraMetaData(extraMeta.asJava)
      .build()

    var shardRows = 0L
    var succeeded = false
    try
      var record = reader.read()
      val chunk = new ArrayList[RawRow](ctx.chunkSize)
      while record != null do
        chunk.add(RawRow(
          record.get("game_id").toString,
          record.get("ply").asInstanceOf[java.lang.Integer].intValue(),
          record.get("fen").toString,
          record.get("dice").toString,
          record.get("side").toString,
          record.get("result").asInstanceOf[java.lang.Float].floatValue()
        ))
        if chunk.size() >= ctx.chunkSize then
          shardRows += writeBatch(chunk, ctx.extractor, ctx.columns, ctx.schema, writer)
        record = reader.read()

      shardRows += writeBatch(chunk, ctx.extractor, ctx.columns, ctx.schema, writer)
      succeeded = true
    finally
      reader.close()
      writer.close()
      if succeeded then
        Files.move(
          tmpFile,
          outFile,
          java.nio.file.StandardCopyOption.ATOMIC_MOVE,
          java.nio.file.StandardCopyOption.REPLACE_EXISTING
        )
      else
        Files.deleteIfExists(tmpFile)

    println(f"  ${shardPath.getFileName}%-32s -> ${outFile.getFileName}%-32s ($shardRows%,d rows)")
    shardRows

  def main(args: Array[String]): Unit =
    val (inputDir, outputDir, schemaId) = args match
      case Array(in, out, s) => (Path.of(in), Path.of(out), s)
      case _                 => sys.error("usage: EnrichShardsApp <inputDir> <outputDir> <schemaId>")

    val engineVersion = sys.props.getOrElse("engine.version", "0.9.3")
    val (columns, extractor) = resolveExtractor(schemaId)
    val conf = new Configuration()
    val schema = buildAvroSchema(columns)
    Files.createDirectories(outputDir)

    val shardPaths = Files.list(inputDir).iterator().asScala
      .filter(_.toString.endsWith(".parquet"))
      .toList
      .sortBy(_.getFileName.toString)

    require(shardPaths.nonEmpty, s"no .parquet shards found in $inputDir")
    println(s"Enriching ${shardPaths.size} shards with $schemaId (engine $engineVersion) to $outputDir [parallel]")

    val startTime = System.currentTimeMillis()
    val chunkSize = 2048
    val ctx = EnrichContext(schemaId, engineVersion, columns, extractor, schema, conf, chunkSize)

    var totalRows = 0L
    shardPaths.foreach { shardPath =>
      val outFile = outputDir.resolve(shardPath.getFileName)
      totalRows += enrichShard(shardPath, outFile, ctx)
    }

    val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
    println(f"Done: enriched $totalRows%,d rows with $schemaId in $elapsed%.2f s (${totalRows / math.max(0.01, elapsed)}%.1f rows/s)")
