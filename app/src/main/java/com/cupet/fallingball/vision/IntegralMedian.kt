package com.cupet.fallingball.vision

import java.util.Arrays

/**
 * Exact counting-sort medians for the TinyHeatmapNet preprocessing contract.
 *
 * Every median in that contract is taken over integral data: ROI samples are uint8 levels,
 * and the lag difference subtracts two flattened uint8 images. A comparison sort therefore
 * produces no more precision than a histogram, while costing far more: a full sort of the
 * 155,648-sample ROI measures about 3 ms per call on a JVM and the frame path needs five
 * such medians. These helpers are bit-exact replacements, not approximations.
 *
 * Callers own the histogram buffers so the per-frame path stays allocation-free.
 */
internal object IntegralMedian {
    const val MAX_BYTE_LEVEL = 255
    const val BYTE_LEVELS = MAX_BYTE_LEVEL + 1
    const val DIFFERENCE_LEVELS = 2 * MAX_BYTE_LEVEL + 1

    /** Median of `count` uint8 samples starting at [offset]. */
    fun ofBytes(source: ByteArray, offset: Int, count: Int, histogram: IntArray): Float {
        require(count > 0) { "A median needs at least one sample" }
        require(histogram.size >= BYTE_LEVELS) { "A uint8 histogram needs $BYTE_LEVELS bins" }
        require(offset >= 0 && offset + count <= source.size) { "Median range out of bounds" }
        Arrays.fill(histogram, 0, BYTE_LEVELS, 0)
        for (index in offset until offset + count) {
            histogram[source[index].toInt() and 0xff]++
        }
        return select(histogram, BYTE_LEVELS, count, valueOffset = 0)
    }

    /**
     * Median of integral differences spanning -255..255.
     *
     * Values are shifted into a 511-bin histogram. Non-integral or out-of-range input would
     * silently distort the frozen contract, so it is rejected rather than clamped.
     */
    fun ofDifferences(values: FloatArray, count: Int, histogram: IntArray): Float {
        require(count > 0) { "A median needs at least one sample" }
        require(histogram.size >= DIFFERENCE_LEVELS) {
            "A difference histogram needs $DIFFERENCE_LEVELS bins"
        }
        require(count <= values.size) { "Median range out of bounds" }
        Arrays.fill(histogram, 0, DIFFERENCE_LEVELS, 0)
        for (index in 0 until count) {
            val value = values[index]
            val level = value.toInt()
            require(level.toFloat() == value) { "A difference median requires integral input" }
            require(level in -MAX_BYTE_LEVEL..MAX_BYTE_LEVEL) {
                "Difference $level outside -$MAX_BYTE_LEVEL..$MAX_BYTE_LEVEL"
            }
            histogram[level + MAX_BYTE_LEVEL]++
        }
        return select(histogram, DIFFERENCE_LEVELS, count, valueOffset = -MAX_BYTE_LEVEL)
    }

    /**
     * Median of the difference of two uint8 planes, taken straight from the bytes.
     *
     * The caller used to materialise `minuend - subtrahend` into a `FloatArray` first,
     * which cost a full pass of float subtracts plus a 4-byte store and load per pixel,
     * and then forced [ofDifferences] to re-derive integrality on *every* sample with a
     * `toInt()`, an int-to-float conversion, a float compare and a range check.  Both
     * operands are uint8 levels, so their difference is an integer in `-255..255` by
     * construction and none of that work was ever necessary.
     *
     * The result is identical to the float path's, including the even/odd averaging:
     * the histogram and [select] are the same code.
     */
    fun ofByteDifferences(
        minuend: ByteArray,
        subtrahend: ByteArray,
        count: Int,
        histogram: IntArray
    ): Float {
        require(count > 0) { "A median needs at least one sample" }
        require(histogram.size >= DIFFERENCE_LEVELS) {
            "A difference histogram needs $DIFFERENCE_LEVELS bins"
        }
        require(count <= minuend.size && count <= subtrahend.size) {
            "Median range out of bounds"
        }
        Arrays.fill(histogram, 0, DIFFERENCE_LEVELS, 0)
        for (index in 0 until count) {
            val level = (minuend[index].toInt() and 0xff) - (subtrahend[index].toInt() and 0xff)
            histogram[level + MAX_BYTE_LEVEL]++
        }
        return select(histogram, DIFFERENCE_LEVELS, count, valueOffset = -MAX_BYTE_LEVEL)
    }

    /**
     * Median of `count` uint8 samples, returned as **twice** the median.
     *
     * An even sample count makes the median a half-integer, which is why the frame path used
     * to drag a `Float` through every pixel.  Doubling keeps it an exact `Int`, so `flatten`
     * can clamp and truncate with integer arithmetic instead of float compare + float-to-int
     * conversion per pixel.  Same histogram, same traversal as [ofBytes].
     */
    fun ofBytesDoubled(source: ByteArray, offset: Int, count: Int, histogram: IntArray): Int {
        require(count > 0) { "A median needs at least one sample" }
        require(histogram.size >= BYTE_LEVELS) { "A uint8 histogram needs $BYTE_LEVELS bins" }
        require(offset >= 0 && offset + count <= source.size) { "Median range out of bounds" }
        Arrays.fill(histogram, 0, BYTE_LEVELS, 0)
        for (index in offset until offset + count) {
            histogram[source[index].toInt() and 0xff]++
        }
        return selectDoubled(histogram, BYTE_LEVELS, count, valueOffset = 0)
    }

    /**
     * Median of the difference of two uint8 planes, returned as **twice** the median.
     *
     * The motion channels quantise `4 * residual`, and `2 * residual` is exactly
     * `2 * (a - b) - median2`, so the whole motion block stays integral until the final
     * `/255f`.  See [ofByteDifferences] for why no integrality check is needed here.
     */
    fun ofByteDifferencesDoubled(
        minuend: ByteArray,
        subtrahend: ByteArray,
        count: Int,
        histogram: IntArray
    ): Int {
        require(count > 0) { "A median needs at least one sample" }
        require(histogram.size >= DIFFERENCE_LEVELS) {
            "A difference histogram needs $DIFFERENCE_LEVELS bins"
        }
        require(count <= minuend.size && count <= subtrahend.size) {
            "Median range out of bounds"
        }
        Arrays.fill(histogram, 0, DIFFERENCE_LEVELS, 0)
        for (index in 0 until count) {
            val level = (minuend[index].toInt() and 0xff) - (subtrahend[index].toInt() and 0xff)
            histogram[level + MAX_BYTE_LEVEL]++
        }
        return selectDoubled(histogram, DIFFERENCE_LEVELS, count, valueOffset = -MAX_BYTE_LEVEL)
    }

    /** Same traversal as [select], returning twice the median as an exact integer. */
    private fun selectDoubled(
        histogram: IntArray,
        levels: Int,
        count: Int,
        valueOffset: Int
    ): Int {
        val half = count / 2
        var seen = 0
        if (count % 2 == 1) {
            for (level in 0 until levels) {
                seen += histogram[level]
                if (seen > half) return 2 * (level + valueOffset)
            }
        } else {
            var lower = Int.MIN_VALUE
            for (level in 0 until levels) {
                seen += histogram[level]
                if (lower == Int.MIN_VALUE && seen >= half) lower = level
                if (seen >= half + 1) {
                    return (lower + valueOffset) + (level + valueOffset)
                }
            }
        }
        error("A histogram median must consume every sample")
    }

    /** Mirrors the even/odd averaging of the comparison sort this replaces. */
    private fun select(histogram: IntArray, levels: Int, count: Int, valueOffset: Int): Float {
        val half = count / 2
        var seen = 0
        if (count % 2 == 1) {
            for (level in 0 until levels) {
                seen += histogram[level]
                if (seen > half) return (level + valueOffset).toFloat()
            }
        } else {
            var lower = Int.MIN_VALUE
            for (level in 0 until levels) {
                seen += histogram[level]
                if (lower == Int.MIN_VALUE && seen >= half) lower = level
                if (seen >= half + 1) {
                    return ((lower + valueOffset) + (level + valueOffset)) * 0.5f
                }
            }
        }
        error("A histogram median must consume every sample")
    }
}
