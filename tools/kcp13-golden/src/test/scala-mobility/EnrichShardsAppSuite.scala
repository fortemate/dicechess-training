package dicechess.training.golden

import java.util.ArrayList
import java.util.concurrent.{Callable, CountDownLatch, ForkJoinPool, TimeUnit}
import java.util.concurrent.atomic.AtomicInteger
import scala.jdk.CollectionConverters.*

class EnrichShardsAppSuite extends munit.FunSuite:
  import EnrichShardsApp.*

  private def rows(count: Int): ArrayList[RawRow] =
    new ArrayList((0 until count).map { i =>
      RawRow(s"game-$i", i, "4k3/8/8/8/8/8/8/4K3 w - -", "PPP", "w", (i % 2).toFloat)
    }.asJava)

  test("batch extraction overlaps work and retains exact encounter order") {
    val input = rows(128)
    val pool = new ForkJoinPool(2)
    val entered = new CountDownLatch(2)
    val active = new AtomicInteger()
    val peak = new AtomicInteger()
    try
      val result = pool.submit(new Callable[java.util.List[EnrichedRow]]:
        override def call(): java.util.List[EnrichedRow] =
          enrichRows(input, (_, _) => {
            val current = active.incrementAndGet()
            peak.accumulateAndGet(current, (a, b) => math.max(a, b))
            entered.countDown()
            try
              require(entered.await(5, TimeUnit.SECONDS), "extraction did not overlap")
              Array(1.0f)
            finally active.decrementAndGet()
          })
      ).get(20, TimeUnit.SECONDS)
      assert(peak.get() >= 2)
      assertEquals(result.asScala.map(_.raw).toList, input.asScala.toList)
      assert(result.asScala.forall(_.features.sameElements(Array(1.0f))))
    finally
      pool.shutdownNow()
      assert(pool.awaitTermination(10, TimeUnit.SECONDS))
  }

  test("invalid side and extractor failures are propagated before returning a batch") {
    val input = rows(2)
    input.set(1, input.get(1).copy(side = "b"))
    val mismatch = intercept[IllegalArgumentException] {
      enrichRows(input, (_, _) => Array(1.0f))
    }
    assert(mismatch.getMessage.contains("side/FEN mismatch"))
    val failure = intercept[IllegalStateException] {
      enrichRows(rows(2), (_, _) => throw new IllegalStateException("extractor failure"))
    }
    assert(failure.getMessage.contains("extractor failure"))
  }
