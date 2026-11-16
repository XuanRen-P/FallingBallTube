package com.cupet.fallingball.camera

import com.cupet.fallingball.analysis.FallFrameObservation
import com.cupet.fallingball.analysis.FallTrackingGate
import kotlin.math.abs

/**
 * Acceptance rule for the TinyHeatmapNet whole-sequence decode.
 *
 * The decode optimises a global path score and has no falling-ball prior, so a persistent
 * static peak (tube end shadow, meniscus) can win the path outright: one expensive jump off
 * the ball is repaid by a tiny per-frame saving across the rest of the sequence. Measured on
 * device, that collapsed real 1029-1048 px descents to 12-206 px and made the
 * terminal-velocity fit unusable while live tracking looked correct.
 *
 * Axial travel is the quantity the fit consumes, so it is the right basis for comparison.
 */
internal object FinalDecodeAcceptance {
    /**
     * Least share of the causal trajectory's travel an accepted decode may retain.
     *
     * Half still lets the decode drop genuinely absent frames and trim spurious endpoints,
     * while rejecting a collapse onto a stationary feature.
     */
    const val MIN_TRAVEL_FRACTION = 0.5f

    /**
     * A decoded path may contain small detector noise, but it must not repeatedly traverse a
     * large portion of the tube in both directions. This is expressed as a fraction of the
     * path range so it remains independent of camera resolution.
     */
    const val MAX_BACKWARD_TRAVEL_FRACTION = 0.25f

    /**
     * Whether the decoded trajectory should replace the causal one.
     *
     * With no causal baseline, the decoded observations must independently confirm coherent
     * downward motion. This prevents an all-static Viterbi path from becoming a measurement.
     */
    fun acceptsDecode(
        causalTravelPx: Float,
        decodedTravelPx: Float,
        decodedFallConfirmed: Boolean,
        decodedBackwardTravelPx: Float? = null,
        decodedMaxBackwardStepTicks: Float? = null
    ): Boolean {
        if (!decodedTravelPx.isFinite()) return false
        decodedMaxBackwardStepTicks?.let { maxStep ->
            if (!maxStep.isFinite() || maxStep > MAX_ACCEPTED_LOCAL_BACKWARD_STEP_TICKS) {
                return false
            }
        }
        decodedBackwardTravelPx?.let { backward ->
            if (!backward.isFinite() || backward < 0f) return false
            if (decodedTravelPx <= 0f ||
                backward > decodedTravelPx * MAX_BACKWARD_TRAVEL_FRACTION
            ) {
                return false
            }
        }
        if (!causalTravelPx.isFinite() || causalTravelPx <= 0f) return decodedFallConfirmed
        return decodedTravelPx >= causalTravelPx * MIN_TRAVEL_FRACTION
    }

    /**
     * Extra travel caused by reversals in a chronological axial sequence.
     *
     * For a monotonic path this is zero apart from measurement noise. A path that jumps from a
     * bottom feature to the ball and then back to the bottom accumulates a large total variation
     * while its end-to-end range remains bounded; their difference is the unwanted backtracking.
     */
    fun backwardTravelPx(axialPx: List<Float>): Float {
        val finite = axialPx.filter(Float::isFinite)
        if (finite.size < 2) return 0f
        var totalVariation = 0f
        for (index in 1 until finite.size) {
            totalVariation += kotlin.math.abs(finite[index] - finite[index - 1])
        }
        val range = finite.maxOrNull()!! - finite.minOrNull()!!
        return (totalVariation - range).coerceAtLeast(0f)
    }

    fun confirmsFall(observations: List<FallFrameObservation>): Boolean {
        if (observations.size < MIN_CONFIRMATION_OBSERVATIONS) return false
        return confirmsStrictFall(observations) || confirmsSustainedFall(observations)
    }

    /** The original 4-8 point cadence gate, kept for normal/fast falls. */
    private fun confirmsStrictFall(observations: List<FallFrameObservation>): Boolean {
        val gate = FallTrackingGate()
        val intervalsMs = observations.zipWithNext { previous, current ->
            ((current.timestampNs - previous.timestampNs) / NANOS_PER_MILLISECOND)
                .takeIf { it > 0L }
        }.filterNotNull().sorted()
        intervalsMs.getOrNull(intervalsMs.size / 2)?.let(gate::setExpectedObservationIntervalMs)
        return observations.any { observation ->
            gate.observe(observation, observation.timestampNs).acceptedObservations.isNotEmpty()
        }
    }

    /**
     * Slow-ball fallback.
     *
     * The strict gate needs 0.05 ruler ticks between consecutive observations. At 60 fps and
     * 10 mm per tick that is 30 mm/s, and the 8-point total-travel rule is even stricter
     * (~43 mm/s). A 1-2.5 mm ball in castor oil falls below that, so a correct final Viterbi
     * path can never pass the strict gate even though it is a clean, monotonic descent.
     *
     * This confirmation works on the whole final sequence instead of a short window: it
     * requires a statistically coherent downward trend over at least a second, and rejects
     * static/noise paths because the robust slope must be positive and the inliers must
     * actually travel. It is only used by [acceptsDecode] when there is no causal baseline,
     * i.e. exactly the case the strict gate was rejecting.
     */
    private fun confirmsSustainedFall(observations: List<FallFrameObservation>): Boolean =
        selectSustainedFall(observations) != null

    /**
     * Returns the temporally ordered inlier subset of a coherent slow fall, or null.
     *
     * The subset is what the final Viterbi path should have contained. It lets the
     * session use the causal selector output directly when the whole-sequence decode
     * collapsed onto a reflection, and it lets a noisy global path be trimmed instead
     * of rejected outright.
     */
    fun selectSustainedFall(
        observations: List<FallFrameObservation>
    ): List<FallFrameObservation>? {
        val valid = observations.asSequence()
            .filter { it.rulerValid && it.positionTicks?.isFinite() == true }
            .distinctBy { it.timestampNs }
            .sortedBy { it.timestampNs }
            .toList()
        if (valid.size < MIN_SUSTAINED_OBSERVATIONS) return null
        val firstNs = valid.first().timestampNs
        val times = DoubleArray(valid.size) {
            (valid[it].timestampNs - firstNs) / NANOS_PER_SECOND
        }
        if (times.last() < MIN_SUSTAINED_DURATION_S) return null
        val positions = DoubleArray(valid.size) {
            requireNotNull(valid[it].positionTicks).toDouble()
        }
        val slope = robustSlope(times, positions) ?: return null
        if (slope < MIN_SUSTAINED_SLOPE_TICKS_PER_SECOND) return null
        val intercept = median(DoubleArray(positions.size) {
            positions[it] - slope * times[it]
        })
        val residuals = DoubleArray(positions.size) {
            abs(positions[it] - (intercept + slope * times[it]))
        }
        val residualMedian = median(residuals)
        val residualMad = median(DoubleArray(residuals.size) {
            abs(residuals[it] - residualMedian)
        })
        val tolerance = maxOf(
            MAX_SUSTAINED_RESIDUAL_TICKS,
            RESIDUAL_MAD_MULTIPLIER * residualMad
        ).coerceAtMost(MAX_SUSTAINED_RESIDUAL_TOLERANCE_TICKS)

        val inliers = ArrayList<FallFrameObservation>(positions.size)
        var previousInlierPosition: Double? = null
        for (index in positions.indices) {
            if (residuals[index] > tolerance) continue
            if (previousInlierPosition != null &&
                positions[index] - previousInlierPosition!! < -MAX_SUSTAINED_UPWARD_STEP_TICKS
            ) {
                return null
            }
            previousInlierPosition = positions[index]
            inliers.add(valid[index])
        }
        if (inliers.size < MIN_SUSTAINED_OBSERVATIONS) return null
        if (inliers.size.toDouble() / positions.size < MIN_SUSTAINED_INLIER_FRACTION) {
            return null
        }
        val inlierDurationSec =
            (inliers.last().timestampNs - inliers.first().timestampNs) / NANOS_PER_SECOND
        if (inlierDurationSec < MIN_SUSTAINED_DURATION_S) return null
        val travel = requireNotNull(inliers.last().positionTicks).toDouble() -
            requireNotNull(inliers.first().positionTicks).toDouble()
        if (travel < MIN_SUSTAINED_TRAVEL_TICKS) return null
        return inliers
    }

    /**
     * Largest single backward step in a chronological position sequence, in ruler ticks.
     * A whole-sequence decode that jumps down to a static bottom feature and back can still
     * keep a small total backward-travel fraction; this local check rejects that case.
     */
    fun maxBackwardStepTicks(positions: List<Float?>): Float {
        var previous: Float? = null
        var maxBackward = 0f
        for (position in positions) {
            if (position == null || !position.isFinite()) continue
            if (previous != null) {
                val step = position - previous!!
                if (step < 0f) maxBackward = maxOf(maxBackward, -step)
            }
            previous = position
        }
        return maxBackward
    }

    /** Median pairwise slope, capped in sample count so long sequences stay cheap. */
    private fun robustSlope(times: DoubleArray, positions: DoubleArray): Double? {
        if (times.size < 2) return null
        val indices = if (times.size <= MAX_THEIL_SEN_POINTS) {
            IntArray(times.size) { it }
        } else {
            IntArray(MAX_THEIL_SEN_POINTS) {
                it * (times.size - 1) / (MAX_THEIL_SEN_POINTS - 1)
            }.distinct().toIntArray()
        }
        val slopes = ArrayList<Double>(MAX_THEIL_SEN_PAIRS.coerceAtMost(
            indices.size * (indices.size - 1) / 2
        ))
        outer@ for (left in indices.indices) {
            for (right in left + 1 until indices.size) {
                val elapsed = times[indices[right]] - times[indices[left]]
                if (elapsed <= 1e-9) continue
                slopes.add(
                    (positions[indices[right]] - positions[indices[left]]) / elapsed
                )
                if (slopes.size >= MAX_THEIL_SEN_PAIRS) break@outer
            }
        }
        if (slopes.isEmpty()) return null
        slopes.sort()
        return slopes[slopes.size / 2]
    }

    private fun median(values: DoubleArray): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.copyOf()
        sorted.sort()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else {
            sorted[middle]
        }
    }

    private const val MIN_CONFIRMATION_OBSERVATIONS = 4
    private const val NANOS_PER_MILLISECOND = 1_000_000L
    private const val NANOS_PER_SECOND = 1_000_000_000.0
    private const val MAX_ACCEPTED_LOCAL_BACKWARD_STEP_TICKS = 1.0f
    private const val MIN_SUSTAINED_OBSERVATIONS = 10
    private const val MIN_SUSTAINED_DURATION_S = 1.0
    private const val MIN_SUSTAINED_SLOPE_TICKS_PER_SECOND = 0.5
    private const val MIN_SUSTAINED_TRAVEL_TICKS = 2.0
    private const val MAX_SUSTAINED_RESIDUAL_TICKS = 0.5
    private const val MAX_SUSTAINED_RESIDUAL_TOLERANCE_TICKS = 1.5
    private const val RESIDUAL_MAD_MULTIPLIER = 3.5
    private const val MIN_SUSTAINED_INLIER_FRACTION = 0.65
    private const val MAX_SUSTAINED_UPWARD_STEP_TICKS = 0.75
    private const val MAX_THEIL_SEN_POINTS = 64
    private const val MAX_THEIL_SEN_PAIRS = 4_096
}
