package com.cupet.fallingball.analysis

import com.cupet.fallingball.vision.BallEndReason
import com.cupet.fallingball.vision.BallDetectionSource
import com.cupet.fallingball.vision.RulerConfig
import com.cupet.fallingball.vision.WallContactState
import java.util.concurrent.atomic.AtomicBoolean

/**
 * One recorded sample of the falling ball.
 *
 * @param tNs frame timestamp from `ImageProxy.imageInfo.timestamp`. Always the capture
 *   clock, never the time the callback happened to run -- processing jitter would otherwise
 *   turn straight into velocity noise.
 * @param positionMm null when the ruler was not trustworthy for this frame. Left empty
 *   rather than filled with a best guess: a wrong millimetre value is worse than none,
 *   because it silently corrupts the velocity fit.
 * @param shiftPx registration shift for this frame, recorded so an unusable run can be
 *   diagnosed afterwards as "the phone was moving too much".
 */
class BallSample(
    val tNs: Long,
    val axialPx: Float,
    val positionMm: Float?,
    val rulerValid: Boolean,
    val extrapolated: Boolean,
    val shiftPx: Float,
    var velocityMmPerS: Float? = null,
    val positionTicks: Float? = null,
    val lateralPx: Float? = null,
    val lateralMm: Float? = null,
    val normalizedLateral: Float? = null,
    val wallClearanceMm: Float? = null,
    val wallContactState: WallContactState = WallContactState.UNKNOWN,
    val detectionSource: BallDetectionSource = BallDetectionSource.CONVGRU,
    val detectionConfidence: Float? = null,
    var usedInTerminalFit: Boolean = false,
    var excludedAsOutlier: Boolean = false
)

/** Small always-on counter set retained with a completed measurement. */
data class MeasurementPipelineDiagnostics(
    val schedulerSkipCount: Int = 0,
    val selectorRejectionCount: Int? = null,
    val gateRejectionCount: Int = 0
)

/**
 * A single drop, from the moment tracking was armed until it ended.
 *
 * Samples are appended on the analysis thread. UI diagnostics may read the count while a run
 * is active, so all mutable state is protected by this session's monitor and callers receive
 * snapshots rather than the backing list.
 */
class FallSession(
    val startNs: Long,
    val config: RulerConfig,
    val resolutionLabel: String,
    val runId: String = java.util.UUID.randomUUID().toString(),
    val diagnosticsEnabled: Boolean = false
) {
    private val completionDelivered = AtomicBoolean(false)
    private val mutableSamples = ArrayList<BallSample>(128)
    @get:Synchronized
    val samples: List<BallSample>
        get() = mutableSamples.toList()
    @get:Synchronized
    val sampleCount: Int
        get() = mutableSamples.size
    @Volatile
    var endReason: BallEndReason? = null
        private set
    @Volatile
    var summary: VelocitySummary? = null
    @Volatile
    var analysisQuality: VelocityAnalysisQuality? = null
        private set
    @Volatile
    var pipelineDiagnostics: MeasurementPipelineDiagnostics? = null
        private set

    val isActive get() = endReason == null
    fun acceptsCaptureTimestamp(timestampNs: Long): Boolean =
        isActive && timestampNs >= startNs

    @get:Synchronized
    val validSampleCount get() = mutableSamples.count { it.rulerValid && it.positionMm != null }

    @Synchronized
    fun add(sample: BallSample) {
        if (
            acceptsCaptureTimestamp(sample.tNs) &&
            (mutableSamples.isEmpty() || sample.tNs > mutableSamples.last().tNs)
        ) {
            mutableSamples.add(sample)
        }
    }

    /**
     * Replaces causal preview samples with the completed TinyHeatmapNet sequence result.
     * This is intentionally available only while the run is still active; [finish] then
     * computes the terminal velocity fit from the final Viterbi path.
     */
    @Synchronized
    fun replaceSamplesForFinalDecode(finalSamples: List<BallSample>) {
        check(isActive) { "Cannot replace samples after a session has finished" }
        require(finalSamples.zipWithNext().all { (a, b) -> b.tNs > a.tNs }) {
            "Final samples must be strictly ordered by capture timestamp"
        }
        mutableSamples.clear()
        mutableSamples.addAll(finalSamples)
    }

    /** Returns true only for the caller that performed the one-time transition. */
    @Synchronized
    fun finish(reason: BallEndReason): Boolean {
        if (isActive) {
            val analysis = VelocityEstimator.analyze(mutableSamples)
            summary = analysis.summary
            analysisQuality = analysis.quality
            mutableSamples.indices.forEach { index ->
                mutableSamples[index].usedInTerminalFit = analysis.usedInTerminalFit[index]
                mutableSamples[index].excludedAsOutlier = analysis.excludedAsOutlier[index]
            }
            // Publish completion only after every derived field and sample flag is final.
            endReason = reason
            return true
        }
        return false
    }

    /** Exactly one lifecycle path may hand this completed session to persistence/UI. */
    fun claimCompletionDelivery(): Boolean = completionDelivered.compareAndSet(false, true)

    fun setPipelineDiagnostics(value: MeasurementPipelineDiagnostics) {
        pipelineDiagnostics = value
    }

    /** Mean sampling rate actually achieved, for the export header. */
    @Synchronized
    fun meanFps(): Float {
        if (mutableSamples.size < 2) return 0f
        val span = (mutableSamples.last().tNs - mutableSamples.first().tNs) / 1_000_000_000.0
        return if (span <= 0.0) 0f else ((mutableSamples.size - 1) / span).toFloat()
    }

    /** Largest registration shift seen, as a shake indicator. */
    @Synchronized
    fun peakShiftPx(): Float =
        mutableSamples.maxOfOrNull { kotlin.math.abs(it.shiftPx) } ?: 0f

    /** Warn only when large motion persists; one corrected registration jump is not shake. */
    @Synchronized
    fun hasExcessiveRegistrationShake(): Boolean =
        RegistrationShakeAssessment.shouldWarn(mutableSamples.map { it.shiftPx })
}
