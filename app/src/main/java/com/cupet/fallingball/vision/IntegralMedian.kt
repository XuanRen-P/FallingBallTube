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
