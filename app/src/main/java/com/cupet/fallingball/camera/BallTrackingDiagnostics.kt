package com.cupet.fallingball.camera

import com.cupet.fallingball.analysis.FallTrackingStatus
import com.cupet.fallingball.vision.BallDetectionSource
import com.cupet.fallingball.vision.TinyHeatmapRuntimeDiagnostics
import com.cupet.fallingball.vision.TinyHeatmapThroughputStats
import com.cupet.fallingball.vision.TemporalSequenceStatus

enum class TemporalPipelineStage {
    UNAVAILABLE,
    WAITING_FOR_TUBE,
    WARMING_UP,
    RUNNING,
    ERROR
}

data class BallTrackingDiagnostics(
    val stage: TemporalPipelineStage,
    val sequence: TemporalSequenceStatus,
    val modelInputFps: Float,
    val score: Float?,
    val threshold: Float,
    val inferenceMs: Float?,
    val resultAgeMs: Long?,
    val selectedSource: BallDetectionSource,
    val gate: FallTrackingStatus,
    val recordedSamples: Int,
    /** Backend and per-stage timing for the latest completed parent inference. */
    val runtime: TinyHeatmapRuntimeDiagnostics = TinyHeatmapRuntimeDiagnostics(),
    /** Camera-to-worker accounting; this is the effective model rate, not camera FPS. */
    val throughput: TinyHeatmapThroughputStats = TinyHeatmapThroughputStats()
)
