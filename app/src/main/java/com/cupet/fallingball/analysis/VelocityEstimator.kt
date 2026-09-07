package com.cupet.fallingball.analysis

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.sqrt

/**
 * Summary of a drop.
 *
 * @param terminalVelocityMmPerS null when the robust trajectory fit is not reliable enough
 *   for a physical result.
 * @param r2 goodness of the straight-line fit over the retained robust inliers.
 *
 * The persisted `plateau*` field names remain for FBR compatibility. Since analysis method
 * v4 they describe robust fit membership and its covered range, not a contiguous plateau.
 */
class VelocitySummary(
    val terminalVelocityMmPerS: Float?,
    val plateauFound: Boolean,
    val plateauStartMm: Float,
    val plateauEndMm: Float,
    val plateauDurationMs: Long,
    val r2: Float,
    val velocityStdDev: Float,
    val sampleCount: Int,
    val validSampleCount: Int,
    val meanVelocityMmPerS: Float?
) {
    /**
     * Too few samples for a trustworthy robust trajectory fit.
     *
     * Worth surfacing rather than hiding: the bias runs one way (terminal velocity reads
     * low), and viscosity is inversely proportional to it, so a sparsely sampled run gives a
     * confidently wrong answer instead of an obviously bad one.
     */
    val underSampled: Boolean
        get() = validSampleCount < VelocityEstimator.MIN_TRUSTED_SAMPLES
}

data class VelocityAnalysis(
    val summary: VelocitySummary,
    val usedInTerminalFit: BooleanArray,
    val excludedAsOutlier: BooleanArray,
    val quality: VelocityAnalysisQuality
)

data class VelocityAnalysisQuality(
    val effectiveSampleRateFps: Float,
    val velocityWindowMs: Long,
    val usablePositionSampleCount: Int,
    val velocitySampleCount: Int,
    val outlierSampleCount: Int,
    val trackDiscontinuity: Boolean = false,
    val terminalFitRelativeDisagreement: Float? = null,
    val terminalFitCheckAttempted: Boolean = false
) {
    val lowSampleRate: Boolean
        get() = effectiveSampleRateFps > 0f &&
            effectiveSampleRateFps < VelocityEstimator.LOW_SAMPLE_RATE_FPS

    val excessiveOutliers: Boolean
        get() {
            val observed = usablePositionSampleCount + outlierSampleCount
            return observed > 0 &&
                outlierSampleCount.toFloat() / observed > VelocityEstimator.MAX_OUTLIER_FRACTION
        }

    val insufficientVelocitySamples: Boolean
        get() = velocitySampleCount < VelocityEstimator.MIN_VELOCITY_SAMPLES

    val unstableTerminalFit: Boolean
        get() = lowSampleRate && terminalFitCheckAttempted &&
            terminalFitRelativeDisagreement?.let {
                it.isFinite() &&
                    it > VelocityEstimator.MAX_TERMINAL_FIT_RELATIVE_DISAGREEMENT
            } == true

    val terminalFitCheckUnavailable: Boolean
        get() = lowSampleRate && terminalFitCheckAttempted &&
            terminalFitRelativeDisagreement?.isFinite() != true
}

/**
 * Turns a position/time series into velocities.
 *
 * Two choices worth stating, because the obvious alternatives are both wrong here:
 *
 *  - **Instantaneous velocity comes from a regression over a time window, not from the
 *    difference between adjacent samples.** At 10-30 fps a finite difference divides a few
 *    pixels of position noise by a few tens of milliseconds and amplifies it into a useless
 *    curve. The window is specified in *time* rather than sample count so it adapts as the
 *    frame rate changes.
 *
 *  - **Terminal velocity comes from regressing position over the robustly retained samples,
 *    not from averaging instantaneous velocities.** Fitting position avoids feeding the
 *    differentiated noise back into the final result.
 */
object VelocityEstimator {
    /** Increment whenever persisted raw trajectories must be reclassified or refitted. */
    const val ANALYSIS_METHOD_VERSION = 4

    const val ROBUST_FIT_RESIDUAL_MM = 2f
    const val ROBUST_VELOCITY_MAD_K = 4f
    const val ROBUST_RANSAC_ITERATIONS = 300
    private const val ROBUST_MAD_SCALE = 1.4826
    private const val ROBUST_TAIL_MARGIN_SAMPLES = 5
    private const val MIN_ROBUST_FIT_FRACTION = 0.5f


    /** Half-width of the regression window. */
    const val WINDOW_MS = 100L

    /**
     * Keep one real neighbour inside the window at the slowest cadence accepted by the
     * trajectory gate. Detector cadence can fall sharply under thermal load shedding and
     * can stretch model steps further, so the old 300 ms cap produced zero velocities even
     * for a valid track. A 2.5 s cap matches FallTrackingGate's maximum accepted gap while
     * still rejecting multi-second, effectively disconnected samples.
     */
    const val MAX_WINDOW_MS = 2_500L

    /** A falling ball cannot make a brief upward excursion of more than one graduation. */
    const val MAX_TRANSIENT_UPWARD_JUMP_TICKS = 1f

    private const val TRANSIENT_RETURN_TOLERANCE_TICKS = 0.5f
    private const val MAX_TRANSIENT_RESET_SAMPLES = 3
    private const val MAX_TRANSIENT_RESET_MS = 300L

    /**
     * Below this many usable samples the robust fit is too weak to trust, so the result is
     * flagged. Roughly one second at 30 fps.
     */
    const val MIN_TRUSTED_SAMPLES = 10
    const val MIN_VELOCITY_SAMPLES = 3

    const val LOW_SAMPLE_RATE_FPS = 10f
    const val MAX_OUTLIER_FRACTION = 0.15f
    const val MAX_TERMINAL_FIT_RELATIVE_DISAGREEMENT = 0.10f
    private const val MAX_THEIL_SEN_SAMPLES = 512

    /** Fraction of the run used when no plateau qualifies. */
    private const val FALLBACK_TAIL_FRAC = 0.4f

    /** Fills [BallSample.velocityMmPerS] for every sample that has a position. */
    fun computeVelocities(samples: List<BallSample>) {
        computeVelocities(samples, BooleanArray(samples.size))
    }

    private fun computeVelocities(samples: List<BallSample>, excluded: BooleanArray): Long {
        val windowNs = velocityWindowNs(samples, excluded)
        for (i in samples.indices) {
            val centre = samples[i]
            if (centre.positionMm == null || excluded[i]) {
                centre.velocityMmPerS = null
                continue
            }
            var lo = i
            while (lo > 0 && centre.tNs - samples[lo - 1].tNs <= windowNs) lo--
            var hi = i
            while (hi < samples.size - 1 && samples[hi + 1].tNs - centre.tNs <= windowNs) hi++
            centre.velocityMmPerS = slope(samples, lo, hi, excluded)
        }
        return windowNs
    }

    /**
     * Keeps the established 100 ms half-window at normal cadence and expands it only when
     * valid ball positions arrive too slowly to place a neighbour inside that window.
     */
    private fun velocityWindowNs(samples: List<BallSample>, excluded: BooleanArray): Long {
        val validTimes = samples.asSequence()
            .filterIndexed { index, sample -> sample.positionMm != null && !excluded[index] }
            .map { it.tNs }
            .toList()
        if (validTimes.size < 2) return WINDOW_MS * 1_000_000L

        val gaps = LongArray(validTimes.size - 1)
        var count = 0
        for (index in 1 until validTimes.size) {
            val gap = validTimes[index] - validTimes[index - 1]
            if (gap > 0L) gaps[count++] = gap
        }
        if (count == 0) return WINDOW_MS * 1_000_000L
        gaps.sort(0, count)
        val medianGapNs = if (count and 1 == 1) {
            gaps[count / 2].toDouble()
        } else {
            (gaps[count / 2 - 1] + gaps[count / 2]) / 2.0
        }
        val baseNs = WINDOW_MS * 1_000_000L
        if (medianGapNs <= baseNs) return baseNs
        return ceil(medianGapNs * 1.5)
            .toLong()
            .coerceIn(baseNs, MAX_WINDOW_MS * 1_000_000L)
    }

    /** Computes velocities, the robust-fit summary, and exact regression membership. */
    fun analyze(samples: List<BallSample>): VelocityAnalysis {
        val trajectory = classifyTrajectory(samples)
        val robustFit = robustFit(samples, trajectory.excludedAsOutlier)
        val excluded = robustFit.excludedAsOutlier
        val windowNs = computeVelocities(samples, excluded)
        val quality = analysisQuality(
            samples,
            excluded,
            windowNs,
            trajectory.trackDiscontinuity
        )
        val observedCount = samples.count { it.positionMm != null }
        val transientOutlierCount = trajectory.excludedAsOutlier.count { it }
        val excessiveTransientOutliers = observedCount > 0 &&
            transientOutlierCount.toFloat() / observedCount > MAX_OUTLIER_FRACTION
        return summarizeWithMembership(
            samples = samples,
            robustFit = robustFit,
            quality = quality,
            excessiveTransientOutliers = excessiveTransientOutliers
        )
    }

    fun summarize(samples: List<BallSample>): VelocitySummary =
        analyze(samples).summary

    private fun summarizeWithMembership(
        samples: List<BallSample>,
        robustFit: RobustFitClassification,
        quality: VelocityAnalysisQuality,
        excessiveTransientOutliers: Boolean
    ): VelocityAnalysis {
        val fitMembership = robustFit.fitMembership.copyOf()
        val excluded = robustFit.excludedAsOutlier
        val indexedUsable = samples.withIndex().filter {
            it.value.positionMm != null && !excluded[it.index]
        }
        val detectedSampleCount = samples.count { it.positionMm != null }
        val usable = indexedUsable.map { it.value }
        val fitted = samples.indices.filter { fitMembership[it] }.map { samples[it] }
        val fitFraction = if (robustFit.validWindowSampleCount > 0) {
            fitted.size.toFloat() / robustFit.validWindowSampleCount
        } else {
            0f
        }
        val fittedVelocity = if (fitted.size >= 2) slope(fitted, 0, fitted.lastIndex) else null
        val fitUsable = fittedVelocity?.let { it.isFinite() && it > 0f } == true &&
            fitFraction >= MIN_ROBUST_FIT_FRACTION &&
            !quality.insufficientVelocitySamples &&
            !quality.trackDiscontinuity &&
            !excessiveTransientOutliers
        val mean = if (usable.size >= 2) slope(usable, 0, usable.lastIndex) else null

        if (!fitUsable) {
            fitMembership.fill(false)
            val fallback = usable.ifEmpty { samples.filter { it.positionMm != null } }
            val tailStart = if (fallback.size >= 2) {
                ((fallback.size * (1f - FALLBACK_TAIL_FRAC)).toInt())
                    .coerceIn(0, fallback.size - 2)
            } else {
                0
            }
            return VelocityAnalysis(
                summary = VelocitySummary(
                    terminalVelocityMmPerS = null,
                    plateauFound = false,
                    plateauStartMm = fallback.getOrNull(tailStart)?.positionMm ?: 0f,
                    plateauEndMm = fallback.lastOrNull()?.positionMm ?: 0f,
                    plateauDurationMs = if (fallback.size >= 2) {
                        (fallback.last().tNs - fallback[tailStart].tNs) / 1_000_000L
                    } else {
                        0L
                    },
                    r2 = 0f,
                    velocityStdDev = if (fallback.size >= 2) {
                        velocityStdDev(fallback, tailStart, fallback.lastIndex)
                    } else {
                        0f
                    },
                    sampleCount = samples.size,
                    validSampleCount = detectedSampleCount,
                    meanVelocityMmPerS = mean
                ),
                usedInTerminalFit = fitMembership,
                excludedAsOutlier = excluded,
                quality = quality
            )
        }

        val v = requireNotNull(fittedVelocity)
        val terminalQuality = quality.copy(
            terminalFitCheckAttempted = quality.lowSampleRate,
            terminalFitRelativeDisagreement = terminalFitRelativeDisagreement(
                terminalVelocity = v,
                usable = usable,
                lowSampleRate = quality.lowSampleRate
            )
        )
        return VelocityAnalysis(
            summary = VelocitySummary(
                terminalVelocityMmPerS = v,
                plateauFound = true,
                plateauStartMm = fitted.first().positionMm!!,
                plateauEndMm = fitted.last().positionMm!!,
                plateauDurationMs = (fitted.last().tNs - fitted.first().tNs) / 1_000_000L,
                r2 = rSquared(fitted, 0, fitted.lastIndex),
                velocityStdDev = velocityStdDev(fitted, 0, fitted.lastIndex),
                sampleCount = samples.size,
                validSampleCount = detectedSampleCount,
                meanVelocityMmPerS = mean
            ),
            usedInTerminalFit = fitMembership,
            excludedAsOutlier = excluded,
            quality = terminalQuality
        )
    }

    private data class RobustFitClassification(
        val fitMembership: BooleanArray,
        val excludedAsOutlier: BooleanArray,
        val validWindowSampleCount: Int
    )

    private data class RegressionLine(val slope: Double, val intercept: Double)

    private fun robustFit(
        samples: List<BallSample>,
        transientOutliers: BooleanArray
    ): RobustFitClassification {
        val fitMembership = BooleanArray(samples.size)
        val excluded = BooleanArray(samples.size)
        val observed = samples.withIndex().filter { it.value.positionMm != null }
        if (observed.size < 5) {
            transientOutliers.copyInto(excluded)
            return RobustFitClassification(fitMembership, excluded, observed.size)
        }

        val allPositions = DoubleArray(observed.size) {
            requireNotNull(observed[it].value.positionMm).toDouble()
        }
        val validEnd = findFallEnd(allPositions)
        if (validEnd < 5) {
            return RobustFitClassification(fitMembership, excluded, validEnd)
        }
        val firstTimestamp = observed.first().value.tNs
        val times = DoubleArray(validEnd) {
            (observed[it].value.tNs - firstTimestamp) / 1_000_000_000.0
        }
        val positions = allPositions.copyOf(validEnd)
        val ransacInliers = ransacInliers(times, positions)
        val instantaneousVelocity = nonUniformGradient(positions, times)
        val velocityMedian = median(instantaneousVelocity)
        val velocityMad = median(
            DoubleArray(instantaneousVelocity.size) {
                abs(instantaneousVelocity[it] - velocityMedian)
            }
        )
        val velocityThreshold = ROBUST_VELOCITY_MAD_K * ROBUST_MAD_SCALE * velocityMad

        for (localIndex in 0 until validEnd) {
            val sourceIndex = observed[localIndex].index
            val velocityOutlier = velocityMad > 1e-6 &&
                abs(instantaneousVelocity[localIndex] - velocityMedian) > velocityThreshold
            val isOutlier = transientOutliers[sourceIndex] ||
                !ransacInliers[localIndex] || velocityOutlier
            excluded[sourceIndex] = isOutlier
            fitMembership[sourceIndex] = !isOutlier
        }
        return RobustFitClassification(fitMembership, excluded, validEnd)
    }

    private fun findFallEnd(positions: DoubleArray): Int {
        if (positions.size < 10) return positions.size
        val filtered = DoubleArray(positions.size) { index ->
            val window = DoubleArray(5) { offset ->
                positions[(index + offset - 2).coerceIn(0, positions.lastIndex)]
            }
            median(window)
        }
        val minimum = filtered.minOrNull() ?: return positions.size
        val maximum = filtered.maxOrNull() ?: return positions.size
        val span = maximum - minimum
        if (span <= 1e-6) return positions.size
        val startLevel = minimum + 0.30 * span
        val dropTolerance = 0.03 * span
        var cut = positions.size
        var started = false
        for (index in 1 until filtered.size) {
            if (!started) {
                if (filtered[index] >= startLevel) started = true
                continue
            }
            if (filtered[index] < filtered[index - 1] - dropTolerance) {
                cut = index
                break
            }
        }
        return (cut - ROBUST_TAIL_MARGIN_SAMPLES).coerceIn(1, positions.size)
    }

    private fun ransacInliers(times: DoubleArray, positions: DoubleArray): BooleanArray {
        val random = SplitMix64(12_345uL)
        var bestInliers: BooleanArray? = null
        var bestCount = -1
        repeat(ROBUST_RANSAC_ITERATIONS) {
            val left = random.nextIndex(times.size)
            val right = random.nextIndex(times.size)
            val elapsed = times[right] - times[left]
            if (abs(elapsed) < 1e-12) return@repeat
            val slope = (positions[right] - positions[left]) / elapsed
            val intercept = positions[left] - slope * times[left]
            val inliers = BooleanArray(times.size) {
                abs(positions[it] - (slope * times[it] + intercept)) < ROBUST_FIT_RESIDUAL_MM
            }
            val count = inliers.count { it }
            if (count > bestCount) {
                bestCount = count
                bestInliers = inliers
            }
        }
        val initial = bestInliers?.takeIf { it.count { included -> included } >= 2 }
            ?: return BooleanArray(times.size) { true }
        val line = regressionLine(times, positions, initial) ?: return initial
        val refined = BooleanArray(times.size) {
            abs(positions[it] - (line.slope * times[it] + line.intercept)) <
                ROBUST_FIT_RESIDUAL_MM
        }
        return if (refined.count { it } >= 2) refined else initial
    }

    private fun regressionLine(
        times: DoubleArray,
        positions: DoubleArray,
        included: BooleanArray
    ): RegressionLine? {
        var count = 0
        var sumTime = 0.0
        var sumPosition = 0.0
        for (index in times.indices) {
            if (!included[index]) continue
            count++
            sumTime += times[index]
            sumPosition += positions[index]
        }
        if (count < 2) return null
        val meanTime = sumTime / count
        val meanPosition = sumPosition / count
        var covariance = 0.0
        var variance = 0.0
        for (index in times.indices) {
            if (!included[index]) continue
            val centeredTime = times[index] - meanTime
            covariance += centeredTime * (positions[index] - meanPosition)
            variance += centeredTime * centeredTime
        }
        if (variance < 1e-12) return null
        val slope = covariance / variance
        return RegressionLine(slope, meanPosition - slope * meanTime)
    }

    private fun nonUniformGradient(values: DoubleArray, times: DoubleArray): DoubleArray {
        if (values.size == 1) return doubleArrayOf(0.0)
        val gradient = DoubleArray(values.size)
        gradient[0] = (values[1] - values[0]) / (times[1] - times[0])
        gradient[values.lastIndex] =
            (values.last() - values[values.lastIndex - 1]) /
                (times.last() - times[times.lastIndex - 1])
        for (index in 1 until values.lastIndex) {
            val before = times[index] - times[index - 1]
            val after = times[index + 1] - times[index]
            val leftWeight = -after / (before * (before + after))
            val centerWeight = (after - before) / (before * after)
            val rightWeight = before / (after * (before + after))
            gradient[index] = leftWeight * values[index - 1] +
                centerWeight * values[index] + rightWeight * values[index + 1]
        }
        return gradient
    }

    private fun median(values: DoubleArray): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.copyOf().apply { sort() }
        val middle = sorted.size / 2
        return if (sorted.size and 1 == 1) {
            sorted[middle]
        } else {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        }
    }

    private class SplitMix64(seed: ULong) {
        private var state = seed

        fun nextIndex(bound: Int): Int {
            state += 0x9E3779B97F4A7C15uL
            var value = state
            value = (value xor (value shr 30)) * 0xBF58476D1CE4E5B9uL
            value = (value xor (value shr 27)) * 0x94D049BB133111EBuL
            value = value xor (value shr 31)
            return (value % bound.toULong()).toInt()
        }
    }

    /**
     * A terminal estimate can remain available for diagnostics without being safe to feed
     * into the inverse velocity-to-viscosity calculation.
     */
    fun isReliableTerminalEstimate(
        plateauFound: Boolean,
        terminalVelocityMmPerS: Float?,
        validSampleCount: Int,
        quality: VelocityAnalysisQuality?
    ): Boolean {
        val checkedQuality = quality ?: return false
        val terminalFitVerified = !checkedQuality.lowSampleRate ||
            (checkedQuality.terminalFitCheckAttempted &&
                checkedQuality.terminalFitRelativeDisagreement?.let {
                    it.isFinite() && it <= MAX_TERMINAL_FIT_RELATIVE_DISAGREEMENT
                } == true)
        // analyze() already rejects excessive transient tracking jumps by clearing the fit.
        // RANSAC/MAD may legitimately exclude more than 15% of an acceleration phase.
        return plateauFound &&
            terminalVelocityMmPerS?.let { it.isFinite() && it > 0f } == true &&
            validSampleCount >= MIN_TRUSTED_SAMPLES &&
            !checkedQuality.insufficientVelocitySamples &&
            !checkedQuality.trackDiscontinuity &&
            terminalFitVerified
    }

    private fun terminalFitRelativeDisagreement(
        terminalVelocity: Float?,
        usable: List<BallSample>,
        lowSampleRate: Boolean
    ): Float? {
        if (!lowSampleRate || terminalVelocity == null || !terminalVelocity.isFinite()) {
            return null
        }
        val robustSlope = theilSenSlope(usable) ?: return null
        if (!robustSlope.isFinite() || abs(robustSlope) < 1e-6f) return null
        return abs(terminalVelocity - robustSlope) / abs(robustSlope)
    }

    /** Median of pairwise slopes across the whole track; resistant to endpoint detections. */
    private fun theilSenSlope(samples: List<BallSample>): Float? {
        val robustSamples = if (samples.size <= MAX_THEIL_SEN_SAMPLES) {
            samples
        } else {
            List(MAX_THEIL_SEN_SAMPLES) { index ->
                val sourceIndex =
                    index.toLong() * samples.lastIndex / (MAX_THEIL_SEN_SAMPLES - 1)
                samples[sourceIndex.toInt()]
            }
        }
        val slopes = ArrayList<Double>()
        for (left in 0 until robustSamples.lastIndex) {
            val leftPosition = robustSamples[left].positionMm ?: continue
            for (right in left + 1..robustSamples.lastIndex) {
                val rightPosition = robustSamples[right].positionMm ?: continue
                val elapsedSeconds =
                    (robustSamples[right].tNs - robustSamples[left].tNs) / 1_000_000_000.0
                if (elapsedSeconds <= 0.0) continue
                val pairSlope = (rightPosition - leftPosition) / elapsedSeconds
                if (pairSlope.isFinite()) slopes += pairSlope
            }
        }
        if (slopes.isEmpty()) return null
        slopes.sort()
        val middle = slopes.size / 2
        val median = if (slopes.size and 1 == 1) {
            slopes[middle]
        } else {
            (slopes[middle - 1] + slopes[middle]) / 2.0
        }
        return median.toFloat()
    }

    private fun analysisQuality(
        samples: List<BallSample>,
        excluded: BooleanArray,
        windowNs: Long,
        trackDiscontinuity: Boolean
    ): VelocityAnalysisQuality {
        val usableIndices = samples.indices.filter {
            samples[it].positionMm != null && !excluded[it]
        }
        // Sampling cadence describes detector/camera performance, so it must not fall merely
        // because trajectory outliers were excluded later. Figure-two-style resets otherwise
        // turn a normal 10.9 fps recording into a false low-frame-rate warning.
        val detectedIndices = samples.indices.filter {
            samples[it].positionMm != null
        }
        val spanSeconds = if (detectedIndices.size < 2) {
            0.0
        } else {
            (samples[detectedIndices.last()].tNs - samples[detectedIndices.first()].tNs) /
                1_000_000_000.0
        }
        val sampleRate = if (spanSeconds > 0.0) {
            ((detectedIndices.size - 1) / spanSeconds).toFloat()
        } else {
            0f
        }
        return VelocityAnalysisQuality(
            effectiveSampleRateFps = sampleRate,
            velocityWindowMs = (windowNs + 999_999L) / 1_000_000L,
            usablePositionSampleCount = usableIndices.size,
            velocitySampleCount = samples.indices.count {
                !excluded[it] && samples[it].velocityMmPerS != null
            },
            outlierSampleCount = excluded.count { it },
            trackDiscontinuity = trackDiscontinuity
        )
    }

    /**
     * Finds short jumps back toward the tube entrance that promptly return to the established
     * downward track. Sustained changes are deliberately retained: they indicate a track
     * switch or a genuinely failed measurement, not an isolated sample that can be ignored.
     */
    private data class TrajectoryClassification(
        val excludedAsOutlier: BooleanArray,
        val trackDiscontinuity: Boolean
    )

    private fun classifyTrajectory(samples: List<BallSample>): TrajectoryClassification {
        val excluded = BooleanArray(samples.size)
        var trustedIndex = samples.indexOfFirst { it.positionTicks != null }
        if (trustedIndex < 0) return TrajectoryClassification(excluded, false)
        var trackDiscontinuity = false

        var index = trustedIndex + 1
        while (index < samples.size) {
            val position = samples[index].positionTicks
            if (position == null) {
                index++
                continue
            }
            val trustedPosition = samples[trustedIndex].positionTicks
            if (trustedPosition == null ||
                position >= trustedPosition - MAX_TRANSIENT_UPWARD_JUMP_TICKS
            ) {
                trustedIndex = index
                index++
                continue
            }

            var returnIndex = -1
            var candidateIndex = index + 1
            var validCandidates = 0
            val excursionStartNs = samples[index].tNs
            val maxExcursionNs = MAX_TRANSIENT_RESET_MS * 1_000_000L
            while (
                candidateIndex <= samples.lastIndex &&
                validCandidates < MAX_TRANSIENT_RESET_SAMPLES &&
                samples[candidateIndex].tNs - excursionStartNs <= maxExcursionNs
            ) {
                val candidate = samples[candidateIndex].positionTicks
                if (candidate == null) {
                    candidateIndex++
                    continue
                }
                validCandidates++
                if (candidate >= trustedPosition - TRANSIENT_RETURN_TOLERANCE_TICKS) {
                    returnIndex = candidateIndex
                    break
                }
                candidateIndex++
            }
            if (returnIndex < 0) {
                trackDiscontinuity = true
                trustedIndex = index
                index++
                continue
            }
            for (outlierIndex in index until returnIndex) excluded[outlierIndex] = true
            trustedIndex = returnIndex
            index = returnIndex + 1
        }
        return TrajectoryClassification(excluded, trackDiscontinuity)
    }

    /** Least-squares slope of position against time, in mm/s. */
    private fun slope(
        s: List<BallSample>,
        lo: Int,
        hi: Int,
        excluded: BooleanArray? = null
    ): Float? {
        var n = 0
        var sx = 0.0
        var sy = 0.0
        var sxx = 0.0
        var sxy = 0.0
        val t0 = s[lo].tNs
        for (i in lo..hi) {
            if (excluded?.get(i) == true) continue
            val mm = s[i].positionMm ?: continue
            val t = (s[i].tNs - t0) / 1_000_000_000.0
            n++
            sx += t
            sy += mm
            sxx += t * t
            sxy += t * mm
        }
        if (n < 2) return null
        val denom = n * sxx - sx * sx
        if (abs(denom) < 1e-12) return null
        return ((n * sxy - sx * sy) / denom).toFloat()
    }

    private fun rSquared(s: List<BallSample>, lo: Int, hi: Int): Float {
        val k = slope(s, lo, hi) ?: return 0f
        val t0 = s[lo].tNs
        var n = 0
        var sy = 0.0
        for (i in lo..hi) {
            s[i].positionMm?.let { sy += it; n++ }
        }
        if (n < 2) return 0f
        val meanY = sy / n

        // Intercept from the fitted slope through the centroid.
        var st = 0.0
        for (i in lo..hi) {
            if (s[i].positionMm != null) st += (s[i].tNs - t0) / 1_000_000_000.0
        }
        val meanT = st / n
        val intercept = meanY - k * meanT

        var ssRes = 0.0
        var ssTot = 0.0
        for (i in lo..hi) {
            val mm = s[i].positionMm ?: continue
            val t = (s[i].tNs - t0) / 1_000_000_000.0
            val pred = intercept + k * t
            ssRes += (mm - pred) * (mm - pred)
            ssTot += (mm - meanY) * (mm - meanY)
        }
        if (ssTot < 1e-12) return 0f
        return (1.0 - ssRes / ssTot).toFloat().coerceIn(0f, 1f)
    }

    private fun velocityStdDev(s: List<BallSample>, lo: Int, hi: Int): Float {
        var n = 0
        var sum = 0.0
        for (i in lo..hi) s[i].velocityMmPerS?.let { sum += it; n++ }
        if (n < 2) return 0f
        val mean = sum / n
        var sq = 0.0
        for (i in lo..hi) s[i].velocityMmPerS?.let { sq += (it - mean) * (it - mean) }
        return sqrt(sq / (n - 1)).toFloat()
    }

}
