package com.cupet.fallingball.signal

/**
 * Kotlin port of the parts of `scipy.signal.find_peaks` / `peak_widths` that the tick
 * and ball detectors rely on.
 *
 * The reference algorithm was tuned against scipy's exact semantics, so this reproduces
 * them rather than reimplementing "peak finding" from scratch. Three details are load
 * bearing, and each is pinned by a golden vector captured from real scipy:
 *
 *  1. A flat top counts as one peak reported at its *midpoint*, rounded down:
 *     `(left + right) / 2` in integer arithmetic.
 *  2. The prominence walk steps *over* samples of equal height (`x[i] <= top`), but only
 *     updates the base when a sample is *strictly* lower (`x[i] < min`). Getting either
 *     comparison wrong changes both the prominence and the reported base, and only on
 *     signals that contain ties -- which is why the golden vectors include integer-
 *     quantised noise.
 *  3. A plateau touching the last sample is never a peak (the scan bound is `n - 1`).
 *
 * All routines are allocation-free apart from growing the supplied [PeakSet].
 */
object PeakFinder {

    /**
     * Fills [out] with the local maxima of `x[0 until n]`, in ascending index order.
     * Only [PeakSet.indices] and [PeakSet.count] are written.
     */
    fun findLocalMaxima(x: FloatArray, n: Int, out: PeakSet) {
        out.reset()
        if (n < 3) return

        // Mirrors scipy's _local_maxima_1d. Note iMax = n - 1: a plateau that runs into
        // the final sample has no strictly-lower right edge and so is not a peak.
        val iMax = n - 1
        var i = 1
        while (i < iMax) {
            if (x[i - 1] < x[i]) {
                var iAhead = i + 1
                while (iAhead < iMax && x[iAhead] == x[i]) iAhead++
                if (x[iAhead] < x[i]) {
                    out.add((i + iAhead - 1) / 2)
                    i = iAhead
                }
            }
            i++
        }
    }

    /**
     * Computes prominence and the left/right bases for every peak already in [peaks].
     * Equivalent to `scipy.signal.peak_prominences(x, peaks)` with `wlen=None`.
     *
     * Worst case is O(n) per peak. That is fine at our sizes (n ~ 1200, a few hundred
     * peaks) and introducing `wlen` to bound it would change the results, breaking
     * agreement with the golden vectors.
     */
    fun computeProminences(x: FloatArray, n: Int, peaks: PeakSet) {
        for (p in 0 until peaks.count) {
            val peak = peaks.indices[p]
            val top = x[peak]

            var i = peak
            var leftMin = top
            var leftBase = peak
            while (i >= 0 && x[i] <= top) {
                if (x[i] < leftMin) {
                    leftMin = x[i]
                    leftBase = i
                }
                i--
            }

            i = peak
            var rightMin = top
            var rightBase = peak
            while (i < n && x[i] <= top) {
                if (x[i] < rightMin) {
                    rightMin = x[i]
                    rightBase = i
                }
                i++
            }

            peaks.prominences[p] = top - if (leftMin > rightMin) leftMin else rightMin
            peaks.leftBases[p] = leftBase
            peaks.rightBases[p] = rightBase
        }
    }

    /**
     * Computes widths at [relHeight] of each peak's prominence, with linear interpolation
     * at the crossing points. Equivalent to `scipy.signal.peak_widths(x, peaks, relHeight)`.
     *
     * Requires [computeProminences] to have run first.
     */
    fun computeWidths(x: FloatArray, peaks: PeakSet, relHeight: Float = 0.5f) {
        for (p in 0 until peaks.count) {
            val peak = peaks.indices[p]
            val iMin = peaks.leftBases[p]
            val iMax = peaks.rightBases[p]
            val height = x[peak] - peaks.prominences[p] * relHeight

            var i = peak
            while (iMin < i && height < x[i]) i--
            var leftIp = i.toFloat()
            if (x[i] < height) {
                // Interpolate between the sample below the line and the one above it.
                leftIp += (height - x[i]) / (x[i + 1] - x[i])
            }

            i = peak
            while (i < iMax && height < x[i]) i++
            var rightIp = i.toFloat()
            if (x[i] < height) {
                rightIp -= (height - x[i]) / (x[i - 1] - x[i])
            }

            peaks.widthHeights[p] = height
            peaks.leftIps[p] = leftIp
            peaks.rightIps[p] = rightIp
            peaks.widths[p] = rightIp - leftIp
        }
    }

    /** Convenience: local maxima + prominences + widths in one pass. */
    fun findPeaks(x: FloatArray, n: Int, out: PeakSet, relHeight: Float = 0.5f) {
        findLocalMaxima(x, n, out)
        computeProminences(x, n, out)
        computeWidths(x, out, relHeight)
    }
}
