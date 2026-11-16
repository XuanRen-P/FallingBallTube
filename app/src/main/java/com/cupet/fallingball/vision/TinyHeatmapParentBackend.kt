package com.cupet.fallingball.vision

import kotlin.math.ceil

/** The two tensors produced by the fixed-shape TinyHeatmapNet parent graph. */
internal data class TinyHeatmapParentOutputs(
    val heatmap: FloatArray,
    val offset: FloatArray
)

/** A bounded, thread-safe summary of parent inference latency. */
data class TinyHeatmapLatencyStats(
    val sampleCount: Int = 0,
    val p50Ms: Float = 0f,
    val p95Ms: Float = 0f,
    val lastMs: Float = 0f,
    val maximumMs: Float = 0f
)

/**
 * The timing contract shared by every TinyHeatmapNet parent implementation.
 *
 * Keeping this as a value object means the camera/decoder code can report what actually ran
 * without knowing anything about NCNN, ONNX Runtime, Vulkan, or a future QNN adapter.
 */
data class TinyHeatmapRuntimeDiagnostics(
    val backendName: String = "unknown",
    val computeUnit: String = "unknown",
    /** True only when the backend has explicit evidence for the reported placement. */
    val placementVerified: Boolean = false,
    /** Machine-readable placement evidence, rather than a requested-provider label. */
    val placementSummary: String = "unverified",
    val preprocessMs: Float = 0f,
    val parentInferenceMs: Float = 0f,
    val postprocessMs: Float = 0f,
    val rerankerMs: Float = 0f,
    /** Parent + candidate decode/rerank time, excluding input construction. */
    val totalMs: Float = 0f,
    /**
     * Fraction of lag-1/lag-3 motion channels that located their exact source frame.
     *
     * Channels 5..8 of the causal parent are differences against the frames
     * `sourceFrameIndex - 1` and `- 3`. CameraX drops frames, and a lag whose exact
     * source frame is missing falls back to "no motion" — which silently changes what
     * the parent sees rather than merely slowing it down. A value below 1.0 therefore
     * means the deployed model is consuming a degraded causal input, so this is
     * reported as a first-class metric instead of being inferred from the drop rate.
     */
    val motionHistoryHitRate: Float = 1f,
    val parentLatencyStats: TinyHeatmapLatencyStats = TinyHeatmapLatencyStats()
) {
    /** The exact production GPU claim gate used by logs and tests. */
    val verifiedGpu: Boolean
        get() = backendName == "ncnn_vulkan" && computeUnit == "GPU" && placementVerified
}

/** Counts the work submitted to the latest-only temporal worker. */
data class TinyHeatmapThroughputStats(
    val submittedCount: Int = 0,
    val completedCount: Int = 0,
    val droppedCount: Int = 0,
    val failedCount: Int = 0,
    val effectiveFps: Float = 0f,
    val elapsedMs: Float = 0f
) {
    val inFlightCount: Int
        get() = (submittedCount - completedCount - droppedCount - failedCount).coerceAtLeast(0)
}

/**
 * One explicit benchmark result.  The requested path and the path that actually ran are kept
 * separate so an ORT/NNAPI request cannot accidentally be reported as a verified NPU result.
 */
internal data class TinyHeatmapBenchmarkResult(
    val backendName: String,
    val computeUnit: String,
    val providerPlacementVerified: Boolean,
    val samplesMs: FloatArray,
    val assetName: String? = null,
    val placementSummary: String = "unverified"
)

/**
 * Thread-safe counters for the camera-to-worker boundary.
 *
 * A completed frame rate is calculated from successful completion timestamps, not from the
 * number of camera callbacks. This prevents a latest-only queue from reporting a fake FPS when
 * it is silently discarding stale work.
 */
internal class TinyHeatmapThroughputTracker {
    private var submittedCount = 0
    private var completedCount = 0
    private var droppedCount = 0
    private var failedCount = 0
    private var firstCompletedNs = 0L
    private var lastCompletedNs = 0L

    @Synchronized
    fun recordSubmitted() {
        submittedCount += 1
    }

    @Synchronized
    fun recordCompleted(completedAtNs: Long) {
        completedCount += 1
        if (completedAtNs > 0L) {
            if (firstCompletedNs == 0L) firstCompletedNs = completedAtNs
            lastCompletedNs = maxOf(lastCompletedNs, completedAtNs)
        }
    }

    @Synchronized
    fun recordDropped() {
        droppedCount += 1
    }

    @Synchronized
    fun recordFailed() {
        failedCount += 1
    }

    @Synchronized
    fun snapshot(): TinyHeatmapThroughputStats {
        val elapsedNs = if (firstCompletedNs > 0L && lastCompletedNs > firstCompletedNs) {
            lastCompletedNs - firstCompletedNs
        } else {
            0L
        }
        val elapsedMs = elapsedNs / 1_000_000f
        val effectiveFps = if (completedCount >= 2 && elapsedNs > 0L) {
            ((completedCount - 1).toDouble() * 1_000_000_000.0 / elapsedNs)
                .toFloat()
                .takeIf { it.isFinite() && it >= 0f }
                ?: 0f
        } else {
            0f
        }
        return TinyHeatmapThroughputStats(
            submittedCount = submittedCount,
            completedCount = completedCount,
            droppedCount = droppedCount,
            failedCount = failedCount,
            effectiveFps = effectiveFps,
            elapsedMs = elapsedMs
        )
    }

    @Synchronized
    fun reset() {
        submittedCount = 0
        completedCount = 0
        droppedCount = 0
        failedCount = 0
        firstCompletedNs = 0L
        lastCompletedNs = 0L
    }
}

/**
 * Keeps only a recent bounded window. A window is preferable to an unbounded list here: thermal
 * throttling changes the useful answer over the lifetime of a camera session, while an ever
 * growing list would retain memory for the whole app process.
 */
internal class TinyHeatmapLatencyAccumulator(private val capacity: Int = DEFAULT_CAPACITY) {
    private val samples = FloatArray(capacity)
    private var stored = 0
    private var next = 0

    init {
        require(capacity > 0) { "Latency window capacity must be positive" }
    }

    @Synchronized
    fun record(milliseconds: Float) {
        if (!milliseconds.isFinite() || milliseconds < 0f) return
        samples[next] = milliseconds
        next = (next + 1) % capacity
        stored = (stored + 1).coerceAtMost(capacity)
    }

    @Synchronized
    fun snapshot(): TinyHeatmapLatencyStats {
        if (stored == 0) return TinyHeatmapLatencyStats()
        val ordered = FloatArray(stored)
        for (index in 0 until stored) {
            // The physical order is irrelevant for percentiles; this also works for a partially
            // filled ring before the first wrap.
            ordered[index] = samples[index]
        }
        ordered.sort()
        return TinyHeatmapLatencyStats(
            sampleCount = stored,
            p50Ms = ordered[nearestRankIndex(stored, 0.50f)],
            p95Ms = ordered[nearestRankIndex(stored, 0.95f)],
            lastMs = samples[(next - 1 + capacity) % capacity],
            maximumMs = ordered.last()
        )
    }

    @Synchronized
    fun reset() {
        stored = 0
        next = 0
    }

    private fun nearestRankIndex(size: Int, fraction: Float): Int =
        (ceil(size * fraction).toInt() - 1).coerceIn(0, size - 1)

    private companion object {
        const val DEFAULT_CAPACITY = 120
    }
}

/**
 * Small seam between the detector/decoder and a hardware-specific parent implementation.
 *
 * Implementations own session creation, one-time warm-up, output buffers and timing. Callers do
 * not need to know whether the graph is running through NCNN Vulkan, an NPU provider, or the
 * conservative ORT CPU fallback.
 */
internal interface TinyHeatmapParentBackend : AutoCloseable {
    /** Stable machine-readable name written to diagnostics and benchmark output. */
    val backendName: String

    /** Human-readable compute unit, for example `GPU`, `NPU`, or `CPU`. */
    val computeUnit: String

    /** True only when the implementation has evidence for the reported placement. */
    val placementVerified: Boolean

    /** Details suitable for logs and persisted diagnostics. */
    val placementSummary: String

    /** Recent inference-only latency; warm-up calls are deliberately excluded. */
    val latencyStats: TinyHeatmapLatencyStats

    /** Performs any compilation/cache warm-up. Safe to call more than once. */
    fun warmUp()

    /** Runs one fixed-shape inference and returns buffers owned by this backend. */
    fun infer(inputValues: FloatArray): TinyHeatmapParentOutputs
}
