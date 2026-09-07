package com.cupet.fallingball.signal

/**
 * Savitzky-Golay smoothing with window 7, polynomial order 2 -- the filter the reference
 * algorithm uses before picking out scale lines.
 *
 * It is tempting to treat this as a plain FIR convolution with `[-2,3,6,7,6,3,-2]/21`,
 * and for the interior that is exactly right. But scipy's default `mode="interp"` does
 * *not* convolve the first and last three samples: it fits a quadratic to the edge window
 * and evaluates it. Convolving there instead produces visibly wrong values (verified: for
 * one test signal scipy gives 2.34 where convolution gives 6.35), which would put phantom
 * peaks at both ends of every frame's signal.
 *
 * Both cases fall out of the same hat matrix `H = A(AᵀA)⁻¹Aᵀ` for `A = [1, t, t²]`,
 * `t = -3..3`: row 3 is the interior kernel, rows 0..2 produce the leading samples from
 * `x[0..6]`, and rows 4..6 (the mirrors of rows 2..0) produce the trailing samples from
 * `x[n-7..n-1]`. The exact rational coefficients are baked in below.
 */
object SavitzkyGolay {

    const val WINDOW = 7
    private const val HALF = WINDOW / 2

    // Rows of the hat matrix, divisor folded in. Kept as Double so the accumulation in
    // smooth() stays exact enough to match scipy within the test tolerance.
    private val ROW0 = doubleArrayOf(32.0, 15.0, 3.0, -4.0, -6.0, -3.0, 5.0).divBy(42.0)
    private val ROW1 = doubleArrayOf(5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0).divBy(14.0)
    private val ROW2 = doubleArrayOf(1.0, 3.0, 4.0, 4.0, 3.0, 1.0, -2.0).divBy(14.0)
    private val KERNEL = doubleArrayOf(-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0).divBy(21.0)
    private val ROW4 = ROW2.reversedArray()
    private val ROW5 = ROW1.reversedArray()
    private val ROW6 = ROW0.reversedArray()

    private fun DoubleArray.divBy(d: Double): DoubleArray {
        for (i in indices) this[i] /= d
        return this
    }

    /**
     * Smooths `src[0 until n]` into `dst[0 until n]`. [src] and [dst] must not alias.
     *
     * @return false when `n < WINDOW`, in which case [dst] is left untouched. scipy raises
     *   here; a signal that short is not a usable tube profile, so callers should treat it
     *   as an invalid frame rather than smoothing something degenerate.
     */
    fun smooth(src: FloatArray, n: Int, dst: FloatArray): Boolean {
        if (n < WINDOW) return false

        // Leading edge: quadratic fit over x[0..6].
        dst[0] = dot(ROW0, src, 0).toFloat()
        dst[1] = dot(ROW1, src, 0).toFloat()
        dst[2] = dot(ROW2, src, 0).toFloat()

        // Interior: sliding convolution.
        for (i in HALF until n - HALF) {
            dst[i] = dot(KERNEL, src, i - HALF).toFloat()
        }

        // Trailing edge: quadratic fit over x[n-7..n-1].
        val tail = n - WINDOW
        dst[n - 3] = dot(ROW4, src, tail).toFloat()
        dst[n - 2] = dot(ROW5, src, tail).toFloat()
        dst[n - 1] = dot(ROW6, src, tail).toFloat()

        return true
    }

    private fun dot(row: DoubleArray, src: FloatArray, offset: Int): Double {
        var acc = 0.0
        for (j in 0 until WINDOW) acc += row[j] * src[offset + j]
        return acc
    }
}
