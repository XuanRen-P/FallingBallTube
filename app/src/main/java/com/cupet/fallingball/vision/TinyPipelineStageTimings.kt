package com.cupet.fallingball.vision

/** Broad ownership of a measured stage in the camera-to-result path. */
enum class TinyPipelinePhase {
    CAMERA,
    WORKER,
    END_TO_END,
    FINALIZATION,
    RECORD_BUILD
}

/**
 * Stable names for the low-overhead TinyHeatmapNet timing trace.
 *
 * Aggregate stages intentionally overlap their children and are excluded from leaf sums and
 * hotspot ranking. All other stages are mutually exclusive at the point where they are measured.
 */
enum class TinyPipelineStage(
    val jsonKey: String,
    val phase: TinyPipelinePhase,
    val aggregate: Boolean = false
) {
    CAMERA_ANALYZER_TOTAL("camera_analyzer_total_ms", TinyPipelinePhase.CAMERA, aggregate = true),
    FRAME_SETUP("frame_setup_ms", TinyPipelinePhase.CAMERA),
    Y_PLANE_DIAGNOSTIC_CAPTURE("y_plane_diagnostic_capture_ms", TinyPipelinePhase.CAMERA),
    TRACK_AND_POLICY("track_and_policy_ms", TinyPipelinePhase.CAMERA),
    READ_SCALE_TOTAL("read_scale_total_ms", TinyPipelinePhase.CAMERA, aggregate = true),
    SCALE_SAMPLE("scale_sample_ms", TinyPipelinePhase.CAMERA),
    TICK_DETECTION("tick_detection_ms", TinyPipelinePhase.CAMERA),
    RULER_UPDATE("ruler_update_ms", TinyPipelinePhase.CAMERA),
    DENSE_DIAGNOSTICS("dense_diagnostics_ms", TinyPipelinePhase.CAMERA),
    RESULT_SELECTION("result_selection_ms", TinyPipelinePhase.CAMERA),
    GATE_AND_SAMPLE_RECORD("gate_and_sample_record_ms", TinyPipelinePhase.CAMERA),
    ROI_GEOMETRY("roi_geometry_ms", TinyPipelinePhase.CAMERA),
    ROI_SAMPLE("roi_sample_ms", TinyPipelinePhase.CAMERA),
    RAW_ROI_COPY("raw_roi_copy_ms", TinyPipelinePhase.CAMERA),
    ROI_STABILIZATION("roi_stabilization_ms", TinyPipelinePhase.CAMERA),
    STABLE_ROI_COPY("stable_roi_copy_ms", TinyPipelinePhase.CAMERA),
    HISTORY_SNAPSHOT("history_snapshot_ms", TinyPipelinePhase.CAMERA),
    ENQUEUE("enqueue_ms", TinyPipelinePhase.CAMERA),
    OBB_FRAME_SNAPSHOT_ENQUEUE("obb_frame_snapshot_enqueue_ms", TinyPipelinePhase.CAMERA),
    CAMERA_RESULT_PUBLISH("camera_result_publish_ms", TinyPipelinePhase.CAMERA),

    WORKER_TOTAL("worker_total_ms", TinyPipelinePhase.WORKER, aggregate = true),
    QUEUE_WAIT("queue_wait_ms", TinyPipelinePhase.WORKER),
    GENERATION_RESET("generation_reset_ms", TinyPipelinePhase.WORKER),
    PRIME_HISTORY("prime_history_ms", TinyPipelinePhase.WORKER),
    DETECTOR_INPUT_COPY("detector_input_copy_ms", TinyPipelinePhase.WORKER),
    PARENT_PREPARE("parent_prepare_ms", TinyPipelinePhase.WORKER),
    CURRENT_FLATTEN("current_flatten_ms", TinyPipelinePhase.WORKER),
    SPATIAL_FEATURES("spatial_features_ms", TinyPipelinePhase.WORKER),
    HISTORY_ALIGNMENT("history_alignment_ms", TinyPipelinePhase.WORKER),
    MOTION_FEATURES("motion_features_ms", TinyPipelinePhase.WORKER),
    PARENT_INFERENCE("parent_inference_ms", TinyPipelinePhase.WORKER),
    SIGMOID_NMS_TOPK("sigmoid_nms_topk_ms", TinyPipelinePhase.WORKER),
    PATCH_EXTRACTION("patch_extraction_ms", TinyPipelinePhase.WORKER),
    RERANKER_INFERENCE("reranker_inference_ms", TinyPipelinePhase.WORKER),
    CANDIDATE_MAPPING("candidate_mapping_ms", TinyPipelinePhase.WORKER),
    DETECTOR_HISTORY_APPEND("detector_history_append_ms", TinyPipelinePhase.WORKER),
    SEQUENCE_BOOKKEEPING("sequence_bookkeeping_ms", TinyPipelinePhase.WORKER),
    DIAGNOSTIC_CAPTURE("diagnostic_capture_ms", TinyPipelinePhase.WORKER),
    RESULT_BUILD("result_build_ms", TinyPipelinePhase.WORKER),
    RESULT_PUBLISH("result_publish_ms", TinyPipelinePhase.WORKER),

    // ImageProxy timestamps are not guaranteed to share System.nanoTime()'s clock domain.
    // Measure from analyzer callback entry instead so the interval is always meaningful.
    ANALYZER_TO_PUBLISH("analyzer_to_publish_ms", TinyPipelinePhase.END_TO_END, aggregate = true),

    FINALIZATION_TOTAL("finalization_total_ms", TinyPipelinePhase.FINALIZATION, aggregate = true),
    WORKER_DRAIN("worker_drain_ms", TinyPipelinePhase.FINALIZATION),
    FINAL_INPUT_BUILD("final_input_build_ms", TinyPipelinePhase.FINALIZATION),
    FINAL_VITERBI("final_viterbi_ms", TinyPipelinePhase.FINALIZATION),
    FINAL_TRAJECTORY("final_trajectory_ms", TinyPipelinePhase.FINALIZATION),
    FINAL_ACCEPTANCE("final_acceptance_ms", TinyPipelinePhase.FINALIZATION),
    VELOCITY_ANALYSIS("velocity_analysis_ms", TinyPipelinePhase.FINALIZATION),

    RECORD_BUILD_TOTAL("record_build_total_ms", TinyPipelinePhase.RECORD_BUILD, aggregate = true),
    RECORD_SAMPLE_COPY("record_sample_copy_ms", TinyPipelinePhase.RECORD_BUILD),
    VISCOSITY_CALCULATION("viscosity_calculation_ms", TinyPipelinePhase.RECORD_BUILD)
}

/** Immutable timing snapshot. Missing stages are represented internally, never as a fake zero. */
class TinyPipelineStageTimings internal constructor(
    private val nanoseconds: LongArray
) {
    init {
        require(nanoseconds.size == TinyPipelineStage.entries.size)
    }

    val isEmpty: Boolean
        get() = nanoseconds.none { it >= 0L }

    fun milliseconds(stage: TinyPipelineStage): Float? {
        val value = nanoseconds[stage.ordinal]
        return value.takeIf { it >= 0L }?.div(NANOS_PER_MILLISECOND)
    }

    fun measuredStages(phase: TinyPipelinePhase? = null): Map<String, Float> = buildMap {
        TinyPipelineStage.entries.forEach { stage ->
            if (phase == null || stage.phase == phase) {
                milliseconds(stage)?.let { put(stage.jsonKey, it) }
            }
        }
    }

    fun forEachMeasured(action: (TinyPipelineStage, Float) -> Unit) {
        TinyPipelineStage.entries.forEach { stage ->
            milliseconds(stage)?.let { action(stage, it) }
        }
    }

    fun measuredLeafTotalMs(phase: TinyPipelinePhase): Float =
        TinyPipelineStage.entries.asSequence()
            .filter { it.phase == phase && !it.aggregate }
            .mapNotNull(::milliseconds)
            .sum()

    internal fun copyNanoseconds(): LongArray = nanoseconds.copyOf()

    companion object {
        private const val NANOS_PER_MILLISECOND = 1_000_000f
        private val EMPTY_VALUES = LongArray(TinyPipelineStage.entries.size) { UNMEASURED }
        val EMPTY = TinyPipelineStageTimings(EMPTY_VALUES)
        internal const val UNMEASURED = -1L
    }
}

/** Mutable single-owner recorder. The disabled singleton makes release builds a no-op. */
class TinyPipelineStageTimer private constructor(
    private val enabled: Boolean,
    private val nanoseconds: LongArray
) {
    /** Avoids even the clock read when detailed diagnostics are disabled. */
    fun start(): Long = if (enabled) System.nanoTime() else TinyPipelineStageTimings.UNMEASURED

    fun recordSince(stage: TinyPipelineStage, startedNs: Long) {
        if (!enabled) return
        recordElapsed(stage, startedNs, System.nanoTime())
    }

    fun recordElapsed(stage: TinyPipelineStage, startedNs: Long, completedNs: Long) {
        if (!enabled || completedNs < startedNs) return
        addNanoseconds(stage, completedNs - startedNs)
    }

    fun addNanoseconds(stage: TinyPipelineStage, durationNs: Long) {
        if (!enabled || durationNs < 0L) return
        val index = stage.ordinal
        val previous = nanoseconds[index]
        nanoseconds[index] = if (previous == TinyPipelineStageTimings.UNMEASURED) {
            durationNs
        } else {
            (previous + durationNs).coerceAtLeast(previous)
        }
    }

    fun snapshot(): TinyPipelineStageTimings = if (enabled) {
        TinyPipelineStageTimings(nanoseconds.copyOf())
    } else {
        TinyPipelineStageTimings.EMPTY
    }

    companion object {
        private val DISABLED = TinyPipelineStageTimer(
            enabled = false,
            nanoseconds = LongArray(0)
        )

        fun enabled(): TinyPipelineStageTimer = TinyPipelineStageTimer(
            enabled = true,
            nanoseconds = LongArray(TinyPipelineStage.entries.size) {
                TinyPipelineStageTimings.UNMEASURED
            }
        )

        fun disabled(): TinyPipelineStageTimer = DISABLED

        fun from(snapshot: TinyPipelineStageTimings): TinyPipelineStageTimer =
            if (snapshot.isEmpty) {
                disabled()
            } else {
                TinyPipelineStageTimer(enabled = true, nanoseconds = snapshot.copyNanoseconds())
            }

        fun create(enabled: Boolean): TinyPipelineStageTimer = if (enabled) enabled() else disabled()
    }
}
