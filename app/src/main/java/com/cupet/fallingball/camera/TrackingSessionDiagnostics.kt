package com.cupet.fallingball.camera

import com.cupet.fallingball.vision.TinyHeatmapThroughputStats
import com.cupet.fallingball.vision.TinyPipelinePhase
import com.cupet.fallingball.vision.TinyPipelineStage
import java.io.File
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.ceil
import org.json.JSONObject

internal enum class TemporalScheduleOutcome {
    SUBMITTED,
    NO_TUBE,
    MODEL_UNAVAILABLE,
    YOLO_SERIALIZED,
    TEMPORAL_COOLDOWN,
    TEMPORAL_BUSY,
    ROI_SAMPLE_FAILED
}

/**
 * Always-on bounded counters for teaching-quality evidence. Unlike debug capture, this stores
 * no images, ROIs, timestamps, or per-frame events.
 */
internal class LightweightTrackingDiagnosticAccumulator {
    private val schedulerSkips = AtomicInteger()
    private val selectorObservations = AtomicInteger()
    private val selectorRejections = AtomicInteger()
    private val gateRejections = AtomicInteger()

    fun recordSchedule(outcome: TemporalScheduleOutcome) {
        if (outcome != TemporalScheduleOutcome.SUBMITTED) schedulerSkips.incrementAndGet()
    }

    fun recordSelector(outcome: TemporalSelectionOutcome) {
        selectorObservations.incrementAndGet()
        if (outcome != TemporalSelectionOutcome.SELECTED) selectorRejections.incrementAndGet()
    }

    fun recordGate(rejection: com.cupet.fallingball.analysis.FallTrackingRejection) {
        if (rejection != com.cupet.fallingball.analysis.FallTrackingRejection.NONE) {
            gateRejections.incrementAndGet()
        }
    }

    fun snapshot() = com.cupet.fallingball.analysis.MeasurementPipelineDiagnostics(
        schedulerSkipCount = schedulerSkips.get(),
        selectorRejectionCount = selectorRejections.get()
            .takeIf { selectorObservations.get() > 0 },
        gateRejectionCount = gateRejections.get()
    )
}

internal enum class TemporalResetReason {
    PIPELINE_RESET,
    PROFILE_CHANGED,
    NO_TUBE,
    CAMERA_DISCONTINUITY,
    MODEL_STEP_ORDER,
    ROI_SAMPLE_FAILED
}

internal fun isCurrentTemporalResultGeneration(
    resultGeneration: Long,
    analysisGeneration: Long,
    activeGeneration: Long
): Boolean = resultGeneration == analysisGeneration && resultGeneration == activeGeneration

internal fun isTrackingSessionTimestampEligible(
    trackingSessionActive: Boolean,
    sessionStartTimestampNs: Long,
    timestampNs: Long
): Boolean = !trackingSessionActive || timestampNs >= sessionStartTimestampNs

internal enum class TrackingDiagnosticOutcome {
    MODEL_ERROR,
    MODEL_NOT_RUNNING,
    WARMUP_INCOMPLETE,
    HEATMAP_BELOW_RECOVERY,
    HEATMAP_BELOW_PRESENCE,
    CLASSICAL_NO_COHERENT_TRACK,
    SELECTOR_NO_COHERENT_TRACK,
    GATE_UNCONFIRMED,
    DENSE_TRACKER_FAILED,
    SHORT_TRAJECTORY,
    NORMAL_TRAJECTORY
}

internal enum class TrackingScoreBand {
    BELOW_RECOVERY,
    RECOVERY_TO_PRESENCE,
    AT_OR_ABOVE_PRESENCE;

    companion object {
        fun from(score: Float): TrackingScoreBand = when {
            score < CLASSICAL_RECOVERY_THRESHOLD -> BELOW_RECOVERY
            score < CLASSICAL_PRESENCE_THRESHOLD -> RECOVERY_TO_PRESENCE
            else -> AT_OR_ABOVE_PRESENCE
        }
    }
}

internal data class DiagnosticDistribution(
    val count: Int = 0,
    val p50Ms: Float = 0f,
    val p95Ms: Float = 0f,
    val maximumMs: Float = 0f
)

internal data class TrackingDiagnosticSnapshot(
    val modelAvailable: Boolean = true,
    val modelError: Boolean = false,
    val modelInputCount: Int = 0,
    val maximumFramesSeen: Int = 0,
    val maximumScore: Float? = null,
    val classicalEvidenceCount: Int = 0,
    val classicalOutputCount: Int = 0,
    val selectionCount: Int = 0,
    val acceptedObservationCount: Int = 0,
    val convgruSampleCount: Int = 0,
    val localSampleCount: Int = 0,
    val denseOutputEnabledFrames: Int = 0,
    val denseProposalCount: Int = 0,
    val denseAcceptedCount: Int = 0,
    val schedulerSkips: Map<TemporalScheduleOutcome, Int> = emptyMap(),
    val scoreBands: Map<TrackingScoreBand, Int> = emptyMap(),
    val selectorOutcomes: Map<TemporalSelectionOutcome, Int> = emptyMap(),
    val gateRejections: Map<com.cupet.fallingball.analysis.FallTrackingRejection, Int> =
        emptyMap(),
    val resetCount: Int = 0,
    val cameraIntervalsMs: List<Float> = emptyList(),
    val modelIntervalsMs: List<Float> = emptyList(),
    val inferenceTimesMs: List<Float> = emptyList(),
    val preprocessTimesMs: List<Float> = emptyList(),
    val parentInferenceTimesMs: List<Float> = emptyList(),
    val postprocessTimesMs: List<Float> = emptyList(),
    val rerankerTimesMs: List<Float> = emptyList(),
    val pipelineStageTimesMs: Map<TinyPipelineStage, List<Float>> = emptyMap(),
    val backendNames: Set<String> = emptySet(),
    val computeUnits: Set<String> = emptySet(),
    val placementVerifiedStates: Set<Boolean> = emptySet(),
    val placementSummaries: Set<String> = emptySet(),
    val tinyThroughput: TinyHeatmapThroughputStats = TinyHeatmapThroughputStats(),
    val captureCostsMs: List<Float> = emptyList(),
    val droppedEventCount: Int = 0,
    val roiTruncated: Boolean = false
) {
    val sampleCount: Int get() = convgruSampleCount + localSampleCount
}

internal data class TrackingDiagnosticSummary(
    val primary: TrackingDiagnosticOutcome,
    val contributingReasons: List<String>,
    val cameraCadence: DiagnosticDistribution,
    val modelCadence: DiagnosticDistribution,
    val inferenceLatency: DiagnosticDistribution,
    val captureCost: DiagnosticDistribution,
    val preprocessLatency: DiagnosticDistribution = DiagnosticDistribution(),
    val parentInferenceLatency: DiagnosticDistribution = DiagnosticDistribution(),
    val postprocessLatency: DiagnosticDistribution = DiagnosticDistribution(),
    val rerankerLatency: DiagnosticDistribution = DiagnosticDistribution(),
    val pipelineStageLatency: Map<TinyPipelineStage, DiagnosticDistribution> = emptyMap(),
    val pipelineLeafP50TotalMs: Map<TinyPipelinePhase, Float> = emptyMap()
)

internal object TrackingDiagnosticClassifier {
    private const val RECOVERY_THRESHOLD = CLASSICAL_RECOVERY_THRESHOLD
    private const val PRESENCE_THRESHOLD = CLASSICAL_PRESENCE_THRESHOLD
    private const val SEQUENCE_LENGTH = 8
    private const val MAX_SHORT_TRAJECTORY_SAMPLES = 40

    fun classify(snapshot: TrackingDiagnosticSnapshot): TrackingDiagnosticSummary {
        val primary = when {
            snapshot.modelError -> TrackingDiagnosticOutcome.MODEL_ERROR
            !snapshot.modelAvailable || snapshot.modelInputCount == 0 ->
                TrackingDiagnosticOutcome.MODEL_NOT_RUNNING
            snapshot.maximumFramesSeen < SEQUENCE_LENGTH ->
                TrackingDiagnosticOutcome.WARMUP_INCOMPLETE
            snapshot.maximumScore == null || snapshot.maximumScore < RECOVERY_THRESHOLD ->
                TrackingDiagnosticOutcome.HEATMAP_BELOW_RECOVERY
            snapshot.maximumScore < PRESENCE_THRESHOLD ->
                TrackingDiagnosticOutcome.HEATMAP_BELOW_PRESENCE
            snapshot.classicalEvidenceCount > 0 && snapshot.classicalOutputCount == 0 ->
                TrackingDiagnosticOutcome.CLASSICAL_NO_COHERENT_TRACK
            snapshot.classicalOutputCount > 0 && snapshot.acceptedObservationCount == 0 ->
                TrackingDiagnosticOutcome.GATE_UNCONFIRMED
            snapshot.selectionCount == 0 && snapshot.acceptedObservationCount == 0 ->
                TrackingDiagnosticOutcome.SELECTOR_NO_COHERENT_TRACK
            snapshot.acceptedObservationCount == 0 ->
                TrackingDiagnosticOutcome.GATE_UNCONFIRMED
            snapshot.sampleCount <= MAX_SHORT_TRAJECTORY_SAMPLES &&
                snapshot.convgruSampleCount > 0 &&
                snapshot.localSampleCount == 0 &&
                snapshot.denseOutputEnabledFrames > 0 ->
                TrackingDiagnosticOutcome.DENSE_TRACKER_FAILED
            snapshot.sampleCount <= MAX_SHORT_TRAJECTORY_SAMPLES ->
                TrackingDiagnosticOutcome.SHORT_TRAJECTORY
            else -> TrackingDiagnosticOutcome.NORMAL_TRAJECTORY
        }
        val reasons = buildList {
            TemporalScheduleOutcome.entries.forEach { outcome ->
                val count = snapshot.schedulerSkips[outcome] ?: 0
                if (outcome != TemporalScheduleOutcome.SUBMITTED && count > 0) {
                    add("${outcome.name}:$count")
                }
            }
            if (snapshot.resetCount > 0) add("HIDDEN_STATE_RESETS:${snapshot.resetCount}")
            if (snapshot.roiTruncated) add("ROI_TRUNCATED")
            if (snapshot.droppedEventCount > 0) {
                add("DROPPED_EVENTS:${snapshot.droppedEventCount}")
            }
        }
        val pipelineStageLatency = snapshot.pipelineStageTimesMs.mapValues { (_, values) ->
            distribution(values)
        }
        val pipelineLeafP50TotalMs = TinyPipelinePhase.entries.associateWith { phase ->
            TinyPipelineStage.entries.asSequence()
                .filter { it.phase == phase && !it.aggregate }
                .mapNotNull { pipelineStageLatency[it] }
                .sumOf { it.p50Ms.toDouble() }
                .toFloat()
        }
        return TrackingDiagnosticSummary(
            primary = primary,
            contributingReasons = reasons,
            cameraCadence = distribution(snapshot.cameraIntervalsMs),
            modelCadence = distribution(snapshot.modelIntervalsMs),
            inferenceLatency = distribution(snapshot.inferenceTimesMs),
            captureCost = distribution(snapshot.captureCostsMs),
            preprocessLatency = distribution(snapshot.preprocessTimesMs),
            parentInferenceLatency = distribution(snapshot.parentInferenceTimesMs),
            postprocessLatency = distribution(snapshot.postprocessTimesMs),
            rerankerLatency = distribution(snapshot.rerankerTimesMs),
            pipelineStageLatency = pipelineStageLatency,
            pipelineLeafP50TotalMs = pipelineLeafP50TotalMs
        )
    }

    private fun distribution(values: List<Float>): DiagnosticDistribution {
        val sorted = values.filter { it.isFinite() && it >= 0f }.sorted()
        if (sorted.isEmpty()) return DiagnosticDistribution()
        fun percentile(fraction: Float): Float {
            val rank = ceil(sorted.size * fraction).toInt().coerceIn(1, sorted.size)
            return sorted[rank - 1]
        }
        return DiagnosticDistribution(
            count = sorted.size,
            p50Ms = percentile(0.50f),
            p95Ms = percentile(0.95f),
            maximumMs = sorted.last()
        )
    }
}

internal class FixedPreRollBuffer<T>(private val capacity: Int) {
    private val values = ArrayDeque<T>(capacity)

    init {
        require(capacity > 0) { "Pre-roll capacity must be positive" }
    }

    @Synchronized
    fun add(value: T) {
        if (values.size == capacity) values.removeFirst()
        values.addLast(value)
    }

    @Synchronized
    fun snapshot(): List<T> = values.toList()

    @Synchronized
    fun clear() {
        values.clear()
    }
}

internal object TrackingDiagnosticRoiBudget {
    const val PER_SESSION_BYTES = 96L * 1024L * 1024L
    const val TOTAL_BYTES = 384L * 1024L * 1024L

    fun availableForSession(existingBytes: Long): Long = minOf(
        PER_SESSION_BYTES,
        (TOTAL_BYTES - existingBytes.coerceAtLeast(0L)).coerceAtLeast(0L)
    )

    fun canAppend(bytesWritten: Long, nextBytes: Long, sessionBudgetBytes: Long): Boolean =
        bytesWritten >= 0L && nextBytes >= 0L && sessionBudgetBytes >= 0L &&
            bytesWritten <= sessionBudgetBytes - nextBytes
}

internal data class TrackingDiagnosticRoiRetentionResult(
    val existingBytes: Long,
    val evictedRunIds: List<String>
)

/** Keeps enough bounded storage available for the next complete debug ROI recording. */
internal object TrackingDiagnosticRoiRetention {
    fun prepareForNextSession(
        root: File,
        totalBytes: Long = TrackingDiagnosticRoiBudget.TOTAL_BYTES,
        perSessionBytes: Long = TrackingDiagnosticRoiBudget.PER_SESSION_BYTES,
        protectedRunIds: Set<String> = emptySet()
    ): TrackingDiagnosticRoiRetentionResult {
        require(totalBytes >= 0L) { "Total ROI budget must be non-negative" }
        require(perSessionBytes >= 0L) { "Per-session ROI budget must be non-negative" }
        if (!root.isDirectory) {
            return TrackingDiagnosticRoiRetentionResult(0L, emptyList())
        }

        var existingBytes = roiBytes(root)
        val reservedTarget = (totalBytes - perSessionBytes).coerceAtLeast(0L)
        if (existingBytes <= reservedTarget) {
            return TrackingDiagnosticRoiRetentionResult(existingBytes, emptyList())
        }

        val evicted = ArrayList<String>()
        val candidates = root.listFiles { file -> file.isDirectory }
            .orEmpty()
            .filter { it.name !in protectedRunIds }
            .mapNotNull(::evictionCandidate)
            .sortedWith(compareBy<EvictionCandidate> { it.finishedAt }.thenBy { it.runId })
        for (candidate in candidates) {
            if (existingBytes <= reservedTarget) break
            if (!candidate.directory.deleteRecursively()) continue
            existingBytes -= candidate.roiBytes
            evicted += candidate.runId
        }
        return TrackingDiagnosticRoiRetentionResult(existingBytes, evicted)
    }

    private data class EvictionCandidate(
        val runId: String,
        val directory: java.io.File,
        val finishedAt: Long,
        val roiBytes: Long
    )

    private fun evictionCandidate(directory: java.io.File): EvictionCandidate? {
        val roiBytes = roiBytes(directory)
        if (roiBytes <= 0L) return null
        val metadata = runCatching {
            JSONObject(directory.resolve("manifest.json").readText())
        }.getOrNull()
        val finishedAt = metadata?.optLong("finished_at_epoch_ms", directory.lastModified())
            ?: directory.lastModified()
        return EvictionCandidate(directory.name, directory, finishedAt, roiBytes)
    }

    private fun roiBytes(root: java.io.File): Long = root.walkTopDown()
        .filter { it.isFile && it.name in ROI_FILES }
        .sumOf(java.io.File::length)

    private val ROI_FILES = setOf("raw_rois.bin", "stable_rois.bin")
}

internal class TrackingDiagnosticRoiQuota(
    existingBytes: Long,
    private val totalBytes: Long = TrackingDiagnosticRoiBudget.TOTAL_BYTES
) {
    private val claimedBytes = AtomicLong(existingBytes.coerceAtLeast(0L))

    fun tryClaim(byteCount: Long): Boolean {
        if (byteCount < 0L) return false
        while (true) {
            val current = claimedBytes.get()
            if (current > totalBytes - byteCount) return false
            if (claimedBytes.compareAndSet(current, current + byteCount)) return true
        }
    }

    val usedBytes: Long get() = claimedBytes.get()
}

/** Preserves FIFO order with already-submitted detector work without blocking its caller. */
internal class TrackingSessionFinalizationQueue(private val executor: Executor) {
    fun submit(action: () -> Unit): Boolean = runCatching {
        executor.execute(action)
    }.isSuccess
}

private const val CLASSICAL_RECOVERY_THRESHOLD = 0.12f
private const val CLASSICAL_PRESENCE_THRESHOLD = 0.18f
