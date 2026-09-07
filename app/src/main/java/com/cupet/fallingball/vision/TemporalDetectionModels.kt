package com.cupet.fallingball.vision

/** Shared candidate output used by the current TinyHeatmapNet preview path and diagnostics. */
data class TemporalBallPeak(
    val roiX: Float,
    val roiY: Float,
    val score: Float
)

data class TemporalBallDetection(
    val roiX: Float,
    val roiY: Float,
    val score: Float,
    val inferenceMs: Float,
    val alternativePeaks: List<TemporalBallPeak> = emptyList(),
    /** Hardware and stage timings captured for the exact frame that produced this detection. */
    val runtime: TinyHeatmapRuntimeDiagnostics = TinyHeatmapRuntimeDiagnostics()
)

/** Thresholds shared by the causal preview selector and its regression tests. */
object TemporalDetectionThresholds {
    const val PRESENCE_THRESHOLD = 0.23f
    const val TRACK_RECOVERY_THRESHOLD = 0.12f
}
