package com.cupet.fallingball.camera

import com.cupet.fallingball.analysis.FallFrameObservation
import com.cupet.fallingball.analysis.FallTrackingGate

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
        decodedBackwardTravelPx: Float? = null
    ): Boolean {
        if (!decodedTravelPx.isFinite()) return false
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

    private const val MIN_CONFIRMATION_OBSERVATIONS = 4
    private const val NANOS_PER_MILLISECOND = 1_000_000L
}
