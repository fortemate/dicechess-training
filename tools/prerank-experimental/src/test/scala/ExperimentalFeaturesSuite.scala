import dicechess.engine.domain.*
class ExperimentalFeaturesSuite extends munit.FunSuite:
  def features(fen: String, mover: Color = Color.White) = ExperimentalFeatures.extract(ExperimentalFeatures.parse(fen),mover)
  test("pawn attacks use diagonals and friendly defenders protect attacked non-kings") {
    val f = features("7k/8/8/3p4/4N3/3P4/8/K7 b - - 0 1")
    assertEquals(f.b.drop(9).toList,List(1f,0f,0f,0f))
    val undefended = features("7k/8/8/3p4/4N3/8/8/K7 b - - 0 1")
    assertEquals(undefended.b.drop(9).toList,List(1f,1f,0f,0f))
    val push = features("7k/8/8/3p4/3N4/8/8/K7 b - - 0 1")
    assertEquals(push.b.drop(9).toList,List(0f,0f,0f,0f))
  }
  test("color swap and vertical reflection preserve root-relative features") {
    val white = features("7k/8/8/3p4/4N3/3P4/8/K7 b - - 0 1")
    val black = features("k7/8/3p4/4n3/3P4/8/8/7K w - - 0 1",Color.Black)
    assertEquals(white.a.toList,black.a.toList)
    assertEquals(white.b.toList,black.b.toList)
    assertEquals(white.c.map(_.toList).toList,black.c.map(_.toList).toList)
  }
  test("king buckets, category halves and square IDs are exact") {
    val f = features("7k/8/8/8/8/8/P7/K7 b - - 0 1")
    assertEquals(f.c.map(_.toList).toList,List(List(8),List(((16+15)*10)*64+8)))
  }
  test("terminal has no invented king anchor") {
    val f = features("8/8/8/8/8/8/P7/K7 b - - 0 1")
    assert(f.terminal)
    assert(f.c.forall(_.isEmpty))
    intercept[IllegalArgumentException](features("8/8/8/8/8/8/P7/8 b - - 0 1"))
  }
