package com.cupet.fallingball.signal

import kotlin.math.abs

/**
 * Order statistics used by the tick pipeline: gap estimation, shift estimation and
 * contrast measurement all need medians and percentiles of small-to-medium arrays.
 *
 * These run per frame, so they use quickselect and sort in place rather than allocating.
 * Every entry point documents that it reorders its input.
 */
object RobustStats {

    /**
     * Median of `a[0 until n]`. **Reorders [a] in place.** Even-length arrays average the
     * two central elements, matching numpy.
     */
    fun median(a: FloatArray, n: Int): Float {
        if (n <= 0) return 0f
        if (n == 1) return a[0]
        val mid = n / 2
        val hi = select(a, n, mid)
        return if (n % 2 == 1) hi else (hi + maxOfFirst(a, mid)) * 0.5f
    }

    /**
     * Linear-interpolated percentile, matching `numpy.percentile` default. [q] in [0, 1].
     * **Reorders [a] in place.**
     */
    fun percentile(a: FloatArray, n: Int, q: Float): Float {
        if (n <= 0) return 0f
        if (n == 1) return a[0]
        sortRange(a, n)
        val pos = q.coerceIn(0f, 1f) * (n - 1)
        val lo = pos.toInt()
        val hi = (lo + 1).coerceAtMost(n - 1)
        val frac = pos - lo
        return a[lo] + (a[hi] - a[lo]) * frac
    }

    /**
     * Median of the [k] largest entries of `a[0 until n]`. **Reorders [a] in place.**
     *
     * This is the robust way to ask "how strong is a typical member of the dominant
     * population at the top of this distribution". A fixed *percentile* cannot answer that
     * when the number of irrelevant small values varies: with 140 peaks of which 37 are
     * scale lines the 75th percentile lands just inside the tick band, but drop three
     * lines and it falls just outside, collapsing any threshold derived from it. Looking
     * at a fixed *count* from the top is independent of how much noise sits below, while
     * the median still shrugs off up to `k/2` outliers.
     */
    fun medianOfTopK(a: FloatArray, n: Int, k: Int): Float {
        if (n <= 0) return 0f
        val kk = k.coerceAtMost(n)
        sortRange(a, n)
        // Ascending, so the top kk entries are a[n-kk until n]; their median sits halfway.
        return if (kk % 2 == 1) {
            a[n - 1 - kk / 2]
        } else {
            (a[n - kk / 2] + a[n - 1 - kk / 2]) * 0.5f
        }
    }

    /**
     * Median absolute deviation. **Reorders [a] and uses [scratch]** (must hold n floats).
     */
    fun mad(a: FloatArray, n: Int, scratch: FloatArray): Float {
        if (n <= 0) return 0f
        val m = median(a, n)
        for (i in 0 until n) scratch[i] = abs(a[i] - m)
        return median(scratch, n)
    }

    /** Coefficient of variation (stddev / mean) of `a[0 until n]`. Does not reorder. */
    fun coefficientOfVariation(a: FloatArray, n: Int): Float {
        if (n < 2) return 0f
        var sum = 0.0
        for (i in 0 until n) sum += a[i]
        val mean = sum / n
        if (abs(mean) < 1e-9) return Float.MAX_VALUE
        var sq = 0.0
        for (i in 0 until n) {
            val d = a[i] - mean
            sq += d * d
        }
        return (Math.sqrt(sq / (n - 1)) / abs(mean)).toFloat()
    }

    /**
     * Quickselect: partially orders [a] so that index [k] holds the value it would have
     * after a full sort, and returns it.
     */
    private fun select(a: FloatArray, n: Int, k: Int): Float {
        var lo = 0
        var hi = n - 1
        while (lo < hi) {
            val pivot = a[(lo + hi) ushr 1]
            var i = lo
            var j = hi
            while (i <= j) {
                while (a[i] < pivot) i++
                while (a[j] > pivot) j--
                if (i <= j) {
                    val t = a[i]; a[i] = a[j]; a[j] = t
                    i++; j--
                }
            }
            if (k <= j) hi = j else if (k >= i) lo = i else break
        }
        return a[k]
    }

    /** Largest value in `a[0 until end]` -- valid after [select] partitioned around end. */
    private fun maxOfFirst(a: FloatArray, end: Int): Float {
        var m = Float.NEGATIVE_INFINITY
        for (i in 0 until end) if (a[i] > m) m = a[i]
        return m
    }

    private fun sortRange(a: FloatArray, n: Int) {
        java.util.Arrays.sort(a, 0, n)
    }
}
