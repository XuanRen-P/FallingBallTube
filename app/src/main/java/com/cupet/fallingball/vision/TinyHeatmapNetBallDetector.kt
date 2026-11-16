package com.cupet.fallingball.vision

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.ArrayDeque
import java.util.EnumSet
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

data class TinyHeatmapFrameCandidate(
    val roiX: Float,
    val roiY: Float,
    val sourceX: Float,
    val sourceY: Float,
    val parentScore: Float,
    val logit: Float
)

data class TinyHeatmapFrameCandidates(
    val candidates: List<TinyHeatmapFrameCandidate>,
    val inferenceMs: Float,
    val framesSeen: Int,
    val resetByGap: Boolean,
    /** Stable backend identifier, e.g. `ncnn_vulkan` or `ort_cpu`. */
    val backendName: String = "unknown",
    val computeUnit: String = "unknown",
    val placementVerified: Boolean = false,
    val placementSummary: String = "unverified",
    /** Time spent in the parent graph only; excludes candidate decode and reranking. */
    val parentInferenceMs: Float = 0f,
    /** Candidate NMS, patch extraction and reranker time. */
    val postprocessMs: Float = 0f,
    val rerankerMs: Float = 0f,
    /** Time spent constructing the five parent input channels. */
    val preprocessMs: Float = 0f,
    val parentLatencyStats: TinyHeatmapLatencyStats = TinyHeatmapLatencyStats(),
    /**
     * Fraction of lag-1/lag-3 motion channels that found their exact source frame in the
     * history snapshot. A dropped capture makes its channel fall back to "no motion", so a
     * rate below 1.0 means the parent is silently seeing a degraded causal input.
     */
    val motionHistoryHitRate: Float = 1f,
    val pipelineTimings: TinyPipelineStageTimings = TinyPipelineStageTimings.EMPTY
) {
    val runtime: TinyHeatmapRuntimeDiagnostics
        get() = TinyHeatmapRuntimeDiagnostics(
            backendName = backendName,
            computeUnit = computeUnit,
            placementVerified = placementVerified,
            placementSummary = placementSummary,
            preprocessMs = preprocessMs,
            parentInferenceMs = parentInferenceMs,
            postprocessMs = postprocessMs,
            rerankerMs = rerankerMs,
            totalMs = inferenceMs,
            motionHistoryHitRate = motionHistoryHitRate,
            parentLatencyStats = parentLatencyStats
        )
}

/**
 * Android implementation of the frozen PC TinyHeatmapNet pipeline.
 *
 * The model, feature construction, candidate NMS, patch sampling, and reranker contract are
 * deliberately explicit here. This keeps the mobile implementation numerically auditable
 * against the PC reference instead of hiding preprocessing in a platform-specific graph.
 */
class TinyHeatmapNetBallDetector(
    context: Context,
    /** Debug benchmarks can disable hardware selection to measure a CPU configuration directly. */
    private val allowHardwareParent: Boolean = true,
    private val cpuInferenceThreads: Int = inferenceThreadCount(),
    /**
     * Whether to re-score the parent's top-5 peaks with the candidate reranker.
     *
     * **Off by default, and the evidence says it should stay off.**
     *
     * The reranker costs `patch_extraction` 4.34 ms + `reranker_inference` 1.08 ms
     * per frame (measured on an SM8650 device), which is roughly what the causal
     * motion channels cost to materialise for the 9-channel parent. Dropping it
     * is therefore what keeps end-to-end frame rate neutral after the parent
     * grew from 5 to 9 input channels.
     *
     * Two independent measurements say it buys nothing:
     *
     * 1. **Its ordering is the parent's ordering.** Over 20,518 multi-candidate
     *    frames spanning 14 clips, 3 tube diameters and both recording batches,
     *    the 5-way ranking by `reranker_logit` is identical to the ranking by
     *    `parent_score` in 100.00% of frames, and the argmax differs in 0 frames
     *    (95% upper bound on the disagreement rate < 0.015%).
     * 2. **Its calibration does not move the answer.** Replacing every
     *    `reranker_logit` with `logit(parent_score)` — leaving the ordering
     *    untouched and changing only the confidence that reaches the fall gate —
     *    reproduced the terminal velocity to 0.0 mm/s on every clip of the
     *    tube-held-out fold, with identical fit sample counts and decode paths.
     *
     * Scope limit, stated plainly: every frame in that evidence is glycerin. Castor
     * oil was not measured, on the argument that a ball visible in a single frame
     * gives the parent heatmap an easier peak than glycerin's motion-only signal
     * does. If castor regressions ever appear, this flag is the first thing to
     * turn back on — the graph and the assets still ship either way.
     *
     * See `dev_tools/ball_video_lab/outputs/unified_v2_20260912/reranker_ablation.json`.
     */
    private val candidateReranker: Boolean = false
) : AutoCloseable {
    private data class HistoryFrame(
        val bytes: ByteArray,
        val geometry: TubeRoiGeometry,
        val sourceFrameIndex: Int
    )

    private data class Candidate(
        val x: Float,
        val y: Float,
        val baseScore: Float,
        val logit: Float
    )

    private val appContext = context.applicationContext
    private val environment = OrtEnvironment.getEnvironment()
    /**
     * The parent is selected lazily on the temporal worker, not while CameraController is being
     * constructed on the main thread. Vulkan instance creation and NCNN graph loading can take
     * hundreds of milliseconds on a cold process; doing that work in a constructor would turn a
     * faster inference path into startup jank. The first selection is sticky for this detector:
     * once hardware creation fails we use the CPU fallback rather than retrying on every frame.
     * The BatchNorm graph remains benchmark-only until it passes the full quality gate.
     */
    private var parentBackend: TinyHeatmapParentBackend? = null
    private var qnnParentAttempted = false
    private var hardwareParentAttempted = false
    /** One batched call is substantially cheaper than five scalar ORT calls on a camera frame. */
    private var batchedRerankerSession: OrtSession? = if (candidateReranker) {
        createOptionalSession(appContext, RERANKER_BATCH_MODEL_FILE)
    } else {
        null
    }
    /** Kept as a compatibility fallback for old APK assets or a batch-shape runtime failure. */
    private val rerankerSession = if (candidateReranker) {
        createSession(appContext, RERANKER_MODEL_FILE)
    } else {
        null
    }
    private val history = ArrayDeque<HistoryFrame>(HISTORY_CAPACITY)
    /** Diagnostic counters for how often the causal history actually had the exact lag frame. */
    private var motionHistoryHits = 0L
    private var motionHistoryLookups = 0L
    private val rawBytes = ByteArray(PIXELS)
    private val flatBytes = ByteArray(PIXELS)
    private val historyFlatBytes = ByteArray(PIXELS)
    /** Exact O(n) median support: ROI samples are uint8 and motion residuals are integral. */
    private val byteHistogram = IntArray(IntegralMedian.BYTE_LEVELS)
    private val differenceHistogram = IntArray(IntegralMedian.DIFFERENCE_LEVELS)
    /** Deferred motion channels: one median-centred residual per lag, expanded on demand. */
    private val motionResidual = Array(MOTION_LAGS.size) { FloatArray(PIXELS) }
    private val input = FloatArray(PARENT_CHANNELS * PIXELS)
    private val patch = FloatArray(INPUT_CHANNELS * PATCH_SIZE * PATCH_SIZE)
    private val batchedPatches = FloatArray(TOP_K * INPUT_CHANNELS * PATCH_SIZE * PATCH_SIZE)
    private val batchedScores = FloatArray(TOP_K)
    /** Reused candidate decode scratch; inference is serialized by [inferCandidates]. */
    private val heatmapScores = FloatArray(HEATMAP_PIXELS)
    private val selectedIndices = IntArray(TOP_K)
    private val selectedScores = FloatArray(TOP_K)
    private val parentInputBuffer = ByteBuffer.allocateDirect(
        PARENT_CHANNELS * PIXELS * Float.SIZE_BYTES
    ).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val patchInputBuffer = ByteBuffer.allocateDirect(
        INPUT_CHANNELS * PATCH_SIZE * PATCH_SIZE * Float.SIZE_BYTES
    ).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val scoreInputBuffer = ByteBuffer.allocateDirect(Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val batchedPatchInputBuffer = ByteBuffer.allocateDirect(
        TOP_K * INPUT_CHANNELS * PATCH_SIZE * PATCH_SIZE * Float.SIZE_BYTES
    ).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val batchedScoreInputBuffer = ByteBuffer.allocateDirect(TOP_K * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val candidates = ArrayList<Candidate>(TOP_K)
    private var frameCount = 0
    private var previousTimestampNs = 0L
    private var lastParentBackendName = "unknown"
    private var lastParentComputeUnit = "unknown"
    private var lastParentPlacementVerified = false
    private var lastParentPlacementSummary = "unverified"
    private var lastParentInferenceMs = 0f
    private var lastParentLatencyStats = TinyHeatmapLatencyStats()
    private var lastRerankerMs = 0f
    private var closed = false

    private lateinit var rawMat: org.opencv.core.Mat
    private lateinit var flatFloatMat: org.opencv.core.Mat
    private lateinit var blur2Mat: org.opencv.core.Mat
    private lateinit var blur5Mat: org.opencv.core.Mat
    private lateinit var historyMat: org.opencv.core.Mat
    private lateinit var alignedHistoryMat: org.opencv.core.Mat
    private lateinit var historyGeometryMat: org.opencv.core.Mat
    private lateinit var currentGeometryMat: org.opencv.core.Mat
    private lateinit var inverseGeometryMat: org.opencv.core.Mat
    private lateinit var historyToCurrentMat: org.opencv.core.Mat
    private lateinit var gemmZeroMat: org.opencv.core.Mat
    private val blur2 = FloatArray(PIXELS)
    private val blur5 = FloatArray(PIXELS)

    init {
        OpenCvRuntime.ensureLoaded()
        rawMat = org.opencv.core.Mat(ROI_HEIGHT, ROI_WIDTH, org.opencv.core.CvType.CV_8UC1)
        flatFloatMat = org.opencv.core.Mat(
            ROI_HEIGHT,
            ROI_WIDTH,
            org.opencv.core.CvType.CV_32FC1
        )
        blur2Mat = flatFloatMat.clone()
        blur5Mat = flatFloatMat.clone()
        historyMat = rawMat.clone()
        alignedHistoryMat = rawMat.clone()
        historyGeometryMat = org.opencv.core.Mat(3, 3, org.opencv.core.CvType.CV_64FC1)
        currentGeometryMat = org.opencv.core.Mat(3, 3, org.opencv.core.CvType.CV_64FC1)
        inverseGeometryMat = org.opencv.core.Mat(3, 3, org.opencv.core.CvType.CV_64FC1)
        historyToCurrentMat = org.opencv.core.Mat(3, 3, org.opencv.core.CvType.CV_64FC1)
        gemmZeroMat = org.opencv.core.Mat()
    }

    @Synchronized
    fun reset() {
        check(!closed) { "TinyHeatmapNet detector is closed" }
        history.clear()
        frameCount = 0
        previousTimestampNs = 0L
        motionHistoryHits = 0L
        motionHistoryLookups = 0L
    }

    @Synchronized
    fun benchmarkParent(iterations: Int = 8): FloatArray {
        return benchmarkParentResult(iterations).samplesMs
    }

    @Synchronized
    internal fun benchmarkParentResult(iterations: Int = 8): TinyHeatmapBenchmarkResult {
        check(!closed) { "TinyHeatmapNet detector is closed" }
        require(iterations in 1..100) { "Benchmark iterations must be in 1..100" }
        input.fill(0.5f)
        repeat(2) { runParent() }
        val samples = FloatArray(iterations) {
            val started = System.nanoTime()
            runParent()
            (System.nanoTime() - started) / 1_000_000f
        }
        val runtime = parentRuntimeDiagnostics()
        return TinyHeatmapBenchmarkResult(
            backendName = runtime.backendName,
            computeUnit = runtime.computeUnit,
            providerPlacementVerified = runtime.placementVerified,
            samplesMs = samples,
            assetName = PARENT_MODEL_FILE,
            placementSummary = runtime.placementSummary
        )
    }

    @Synchronized
    fun benchmarkAsset(
        assetName: String,
        useNnapi: Boolean,
        iterations: Int = 8
    ): FloatArray {
        return benchmarkAssetResult(assetName, useNnapi, iterations).samplesMs
    }

    @Synchronized
    internal fun benchmarkAssetResult(
        assetName: String,
        useNnapi: Boolean,
        iterations: Int = 8
    ): TinyHeatmapBenchmarkResult {
        check(!closed) { "TinyHeatmapNet detector is closed" }
        require(iterations in 1..100) { "Benchmark iterations must be in 1..100" }
        val session = createSession(appContext, assetName, useNnapi)
        val inputChannels = parentInputChannels(session)
        return try {
            repeat(2) { runBenchmarkSession(session, inputChannels) }
            val samples = FloatArray(iterations) {
                val started = System.nanoTime()
                runBenchmarkSession(session, inputChannels)
                (System.nanoTime() - started) / 1_000_000f
            }
            TinyHeatmapBenchmarkResult(
                // ORT's Java API does not expose per-node placement after session creation.  The
                // NNAPI label therefore says "requested" rather than claiming that Qualcomm's
                // GPU/NPU actually executed the graph.
                backendName = if (useNnapi) "ort_nnapi_requested" else "ort_cpu",
                computeUnit = if (useNnapi) "NNAPI(requested)" else "CPU",
                providerPlacementVerified = !useNnapi,
                samplesMs = samples,
                assetName = assetName,
                placementSummary = if (useNnapi) {
                    "nnapi_requested_unverified"
                } else {
                    "ort_cpu_full_graph"
                }
            )
        } finally {
            session.close()
        }
    }

    /**
     * Measures [buildInput] on a synthetic ROI, without the parent.
     *
     * Preprocessing is roughly two thirds of a frame and `parent_inference` cannot see it, so
     * an optimisation needs its own number.  History is primed for every iteration with a
     * *different* homography, which forces the real lag path (invert + warp + difference
     * median) instead of the "geometry unchanged" fast path.  The produced tensor is
     * checksummed and logged: a bit-exact refactor must keep the same checksum.
     */
    @Synchronized
    internal fun benchmarkPreprocessResult(iterations: Int = 8): TinyHeatmapBenchmarkResult {
        check(!closed) { "TinyHeatmapNet detector is closed" }
        require(iterations in 1..100) { "Benchmark iterations must be in 1..100" }
        val current = benchmarkGeometry(0.0)
        val historyGeometry = benchmarkGeometry(1.0)
        val timer = TinyPipelineStageTimer.enabled()
        val warmups = 2
        repeat(warmups) {
            primeBenchmarkHistory(historyGeometry)
            runPreprocessIteration(current, timer)
        }
        val samples = FloatArray(iterations) {
            primeBenchmarkHistory(historyGeometry)
            val started = System.nanoTime()
            runPreprocessIteration(current, timer)
            (System.nanoTime() - started) / 1_000_000f
        }
        val runs = iterations + warmups
        val stages = timer.snapshot().measuredStages()
            .entries
            .joinToString(" ") { (name, totalMs) -> "$name=${"%.2f".format(totalMs / runs)}" }
        Log.i(TAG, "preprocess checksum=" + inputChecksumHex() + " stages(ms/frame): " + stages)
        return TinyHeatmapBenchmarkResult(
            backendName = "preprocess_cpu",
            computeUnit = "CPU",
            providerPlacementVerified = true,
            samplesMs = samples,
            assetName = PARENT_MODEL_FILE,
            placementSummary = "preprocess_only"
        )
    }

    /** Unit homography, optionally translated so the history warp cannot take the fast path. */
    private fun benchmarkGeometry(xShift: Double): TubeRoiGeometry {
        val matrix = DoubleArray(9).also {
            it[0] = 1.0
            it[2] = xShift
            it[4] = 1.0
            it[8] = 1.0
        }
        return TubeRoiGeometry(
            revision = 1L,
            sourceQuad = FloatArray(8),
            sourceToRoi = matrix,
            roiToSource = matrix.copyOf()
        )
    }

    private fun primeBenchmarkHistory(geometry: TubeRoiGeometry) {
        history.clear()
        for (lag in MOTION_LAGS) {
            val bytes = ByteArray(PIXELS)
            for (index in 0 until PIXELS) {
                bytes[index] = ((index * 37 + lag * 11) and 0xff).toByte()
            }
            history.addLast(HistoryFrame(bytes, geometry, -lag))
        }
    }

    private fun runPreprocessIteration(
        geometry: TubeRoiGeometry,
        timer: TinyPipelineStageTimer = TinyPipelineStageTimer.disabled()
    ) {
        for (index in 0 until PIXELS) {
            rawBytes[index] = (((index * 31) ushr 3) and 0xff).toByte()
        }
        buildInput(rawBytes, geometry, 0, timer)
    }

    /** FNV-1a over the raw float bits of the produced 9-channel tensor. */
    private fun inputChecksumHex(): String {
        var hash = -0x340d631b
        for (value in input) {
            hash = (hash xor value.toRawBits()) * 0x01000193
        }
        return Integer.toHexString(hash)
    }

    /** Backend identity after the most recent parent run, including a hardware fallback. */
    @Synchronized
    internal fun parentRuntimeDiagnostics(): TinyHeatmapRuntimeDiagnostics = TinyHeatmapRuntimeDiagnostics(
        backendName = lastParentBackendName,
        computeUnit = lastParentComputeUnit,
        placementVerified = lastParentPlacementVerified,
        placementSummary = lastParentPlacementSummary,
        parentInferenceMs = lastParentInferenceMs,
        parentLatencyStats = lastParentLatencyStats
    )

    /**
     * Replaces causal image history before a sparse worker inference. The camera thread
     * stabilizes every capture, while the worker may only infer the newest capture; supplying
     * this snapshot keeps lag-1/lag-3 channels on the same source-frame semantics as PC.
     */
    @Synchronized
    fun primeHistory(frames: List<TinyHeatmapHistoryFrame>) {
        check(!closed) { "TinyHeatmapNet detector is closed" }
        history.clear()
        val recent = frames.takeLast(HISTORY_CAPACITY)
        recent.forEach { frame ->
            require(frame.bytes.size >= PIXELS) {
                "Expected $PIXELS history bytes, got ${frame.bytes.size}"
            }
            history.addLast(
                HistoryFrame(
                    bytes = frame.bytes.copyOf(PIXELS),
                    geometry = frame.geometry,
                    sourceFrameIndex = frame.sourceFrameIndex
                )
            )
        }
        previousTimestampNs = recent.lastOrNull()?.timestampNs ?: 0L
    }

    @Synchronized
    fun inferCandidates(
        inputBytes: ByteBuffer,
        timestampNs: Long,
        geometry: TubeRoiGeometry,
        stableOutput: ByteBuffer,
        sourceFrameIndex: Int,
        stageTimer: TinyPipelineStageTimer = TinyPipelineStageTimer.disabled()
    ): TinyHeatmapFrameCandidates {
        check(!closed) { "TinyHeatmapNet detector is closed" }
        require(inputBytes.remaining() >= PIXELS) { "Expected $PIXELS grayscale bytes" }
        require(stableOutput.capacity() >= PIXELS) { "Stable output requires $PIXELS bytes" }
        val resetByGap = previousTimestampNs != 0L && (
            timestampNs <= previousTimestampNs ||
                timestampNs - previousTimestampNs > MAX_HISTORY_GAP_NS
            )
        if (resetByGap) reset()
        val inputCopyStarted = stageTimer.start()
        inputBytes.duplicate().get(rawBytes, 0, PIXELS)
        stableOutput.clear()
        stableOutput.put(rawBytes)
        stableOutput.flip()
        stageTimer.recordSince(TinyPipelineStage.DETECTOR_INPUT_COPY, inputCopyStarted)

        val preprocessStarted = System.nanoTime()
        buildInput(rawBytes, geometry, sourceFrameIndex, stageTimer)
        val preprocessMs = (System.nanoTime() - preprocessStarted) / 1_000_000f
        // Warm-up is deliberately outside the measured inference interval. The first frame may
        // still wait for it, but a one-time shader/session compile must not masquerade as the
        // steady-state parent latency shown in diagnostics.
        val prepareStarted = stageTimer.start()
        ensureParentReady()
        stageTimer.recordSince(TinyPipelineStage.PARENT_PREPARE, prepareStarted)
        val started = System.nanoTime()
        val parentStarted = stageTimer.start()
        val parentOutputs = runParent()
        stageTimer.recordSince(TinyPipelineStage.PARENT_INFERENCE, parentStarted)
        val candidates = decodeCandidates(parentOutputs.first, parentOutputs.second, stageTimer)
        val inferenceMs = (System.nanoTime() - started) / 1_000_000f
        val postprocessMs = (inferenceMs - lastParentInferenceMs).coerceAtLeast(0f)

        val historyAppendStarted = stageTimer.start()
        history.addLast(HistoryFrame(rawBytes.copyOf(), geometry, sourceFrameIndex))
        while (history.size > HISTORY_CAPACITY) history.removeFirst()
        stageTimer.recordSince(TinyPipelineStage.DETECTOR_HISTORY_APPEND, historyAppendStarted)
        frameCount += 1
        previousTimestampNs = timestampNs
        val mappingStarted = stageTimer.start()
        val mappedCandidates = candidates.map { candidate ->
            val source = transformPoint(
                candidate.x,
                candidate.y,
                geometry.roiToSource
            )
            TinyHeatmapFrameCandidate(
                roiX = candidate.x,
                roiY = candidate.y,
                sourceX = source.first,
                sourceY = source.second,
                parentScore = candidate.baseScore,
                logit = candidate.logit
            )
        }
        stageTimer.recordSince(TinyPipelineStage.CANDIDATE_MAPPING, mappingStarted)
        return TinyHeatmapFrameCandidates(
            candidates = mappedCandidates,
            inferenceMs = inferenceMs,
            framesSeen = frameCount,
            resetByGap = resetByGap,
            backendName = lastParentBackendName,
            computeUnit = lastParentComputeUnit,
            placementVerified = lastParentPlacementVerified,
            placementSummary = lastParentPlacementSummary,
            parentInferenceMs = lastParentInferenceMs,
            postprocessMs = postprocessMs.coerceAtLeast(0f),
            rerankerMs = lastRerankerMs,
            preprocessMs = preprocessMs,
            parentLatencyStats = lastParentLatencyStats,
            motionHistoryHitRate = if (motionHistoryLookups == 0L) {
                1f
            } else {
                motionHistoryHits.toFloat() / motionHistoryLookups.toFloat()
            },
            pipelineTimings = stageTimer.snapshot()
        )
    }

    private fun buildInput(
        current: ByteArray,
        geometry: TubeRoiGeometry,
        sourceFrameIndex: Int,
        stageTimer: TinyPipelineStageTimer
    ) {
        val flattenStarted = stageTimer.start()
        flatten(current, flatBytes)
        stageTimer.recordSince(TinyPipelineStage.CURRENT_FLATTEN, flattenStarted)
        val spatialStarted = stageTimer.start()
        // uint8 -> float32 is exact, so a bulk put plus the native convertTo replaces the
        // 622 KB float plane this loop used to materialise from Kotlin on every frame.
        rawMat.put(0, 0, flatBytes)
        rawMat.convertTo(flatFloatMat, org.opencv.core.CvType.CV_32F)
        org.opencv.imgproc.Imgproc.GaussianBlur(
            flatFloatMat,
            blur2Mat,
            org.opencv.core.Size(0.0, 0.0),
            2.0
        )
        org.opencv.imgproc.Imgproc.GaussianBlur(
            flatFloatMat,
            blur5Mat,
            org.opencv.core.Size(0.0, 0.0),
            5.0
        )
        blur2Mat.get(0, 0, blur2)
        blur5Mat.get(0, 0, blur5)
        for (index in 0 until PIXELS) {
            val flat = (flatBytes[index].toInt() and 0xff).toFloat()
            input[index] = flat / 255f
            input[PIXELS + index] = uint8Normalized((blur2[index] - flat) * 4f)
            input[2 * PIXELS + index] = uint8Normalized((flat - blur2[index]) * 4f)
            input[3 * PIXELS + index] = uint8Normalized((blur5[index] - flat) * 4f)
            input[4 * PIXELS + index] = uint8Normalized((flat - blur5[index]) * 4f)
        }
        stageTimer.recordSince(TinyPipelineStage.SPATIAL_FEATURES, spatialStarted)

        // Channels 0..4 are the single-frame spatial block.  Channels 5..8 are the causal
        // motion block and are now consumed by the parent itself, so they must be written into
        // `input` here as well as kept as a residual for extractPatch/sampleMotion.
        // uint8Normalized(residual * 4f) is exactly the mapping PC applies in
        // preprocess_image, so the eager and lazy forms stay bit-identical.
        for (channelPair in 0 until MOTION_LAGS.size) {
            val lag = MOTION_LAGS[channelPair]
            // PC reads the exact source frame (frame_index - lag). CameraX may drop frames,
            // so ordinal lookup among processed frames would silently change motion channels.
            val historyFrame = history.firstOrNull {
                it.sourceFrameIndex == sourceFrameIndex - lag
            }
            motionHistoryLookups += 1
            if (historyFrame != null) motionHistoryHits += 1
            val alignmentStarted = stageTimer.start()
            val aligned = if (historyFrame == null) {
                current
            } else {
                alignHistory(historyFrame, geometry)
            }
            stageTimer.recordSince(TinyPipelineStage.HISTORY_ALIGNMENT, alignmentStarted)
            val motionStarted = stageTimer.start()
            // Only the flattened bytes are needed: both operands are uint8 levels, so their
            // difference is an exact integer and the float plane is pure overhead.
            flatten(aligned, historyFlatBytes)
            val median2 = IntegralMedian.ofByteDifferencesDoubled(
                flatBytes,
                historyFlatBytes,
                PIXELS,
                differenceHistogram
            )
            val positiveBase = (MOTION_CHANNEL_START + channelPair * 2) * PIXELS
            val negativeBase = positiveBase + PIXELS
            // `motionResidual` is read only by sampleMotion -> extractPatch -> the reranker.
            // With the reranker disabled it would be a dead 622 KB store per lag per frame,
            // so the hot loop is specialised rather than carrying a null check per pixel.
            val residual = if (candidateReranker) motionResidual[channelPair] else null
            // `residual * 4` is exactly `2 * value2`, so the channel mapping stays integral
            // until the final divide and matches uint8Normalized(residual * 4f) bit for bit.
            if (residual != null) {
                for (index in 0 until PIXELS) {
                    val value2 = 2 * ((flatBytes[index].toInt() and 0xff) -
                        (historyFlatBytes[index].toInt() and 0xff)) - median2
                    residual[index] = value2 * 0.5f
                    val scaled = value2 * 2
                    input[positiveBase + index] = scaledToUnit(scaled)
                    input[negativeBase + index] = scaledToUnit(-scaled)
                }
            } else {
                for (index in 0 until PIXELS) {
                    val value2 = 2 * ((flatBytes[index].toInt() and 0xff) -
                        (historyFlatBytes[index].toInt() and 0xff)) - median2
                    val scaled = value2 * 2
                    input[positiveBase + index] = scaledToUnit(scaled)
                    input[negativeBase + index] = scaledToUnit(-scaled)
                }
            }
            stageTimer.recordSince(TinyPipelineStage.MOTION_FEATURES, motionStarted)
        }
    }

    /**
     * Reproduces channel [channel] of the deferred motion block at one pixel.
     *
     * Channels alternate positive/negative per lag, exactly as the eager loop wrote them.
     */
    private fun sampleMotion(channel: Int, index: Int): Float {
        val motionChannel = channel - MOTION_CHANNEL_START
        val residual = motionResidual[motionChannel / 2][index]
        return if (motionChannel % 2 == 0) {
            uint8Normalized(residual * 4f)
        } else {
            uint8Normalized(-residual * 4f)
        }
    }

    /**
     * Row-median flattened uint8 plane: `clamp(byte - rowMedian + globalMedian)`.
     *
     * An even sample count makes the medians half-integers, so they are carried **doubled**
     * and the per-pixel clamp becomes integer arithmetic.  `floor(value2 / 2)` equals the
     * previous `coerceIn(0f, 255f).toInt()` because the clamped value is non-negative, so the
     * produced plane is bit-identical to the float version.
     */
    private fun flatten(source: ByteArray, destination: ByteArray) {
        val globalMedian2 = IntegralMedian.ofBytesDoubled(source, 0, PIXELS, byteHistogram)
        for (row in 0 until ROI_HEIGHT) {
            val offset = row * ROI_WIDTH
            val rowMedian2 = IntegralMedian.ofBytesDoubled(source, offset, ROI_WIDTH, byteHistogram)
            val bias = globalMedian2 - rowMedian2
            for (column in 0 until ROI_WIDTH) {
                val value2 = 2 * (source[offset + column].toInt() and 0xff) + bias
                destination[offset + column] = (value2.coerceIn(0, 510) shr 1).toByte()
            }
        }
    }

    private fun alignHistory(frame: HistoryFrame, current: TubeRoiGeometry): ByteArray {
        if (frame.geometry.sourceToRoi.contentEquals(current.sourceToRoi)) return frame.bytes
        historyMat.put(0, 0, frame.bytes)
        historyGeometryMat.put(0, 0, *frame.geometry.sourceToRoi)
        currentGeometryMat.put(0, 0, *current.sourceToRoi)
        org.opencv.core.Core.invert(historyGeometryMat, inverseGeometryMat)
        org.opencv.core.Core.gemm(
            currentGeometryMat,
            inverseGeometryMat,
            1.0,
            gemmZeroMat,
            0.0,
            historyToCurrentMat
        )
        org.opencv.imgproc.Imgproc.warpPerspective(
            historyMat,
            alignedHistoryMat,
            historyToCurrentMat,
            org.opencv.core.Size(ROI_WIDTH.toDouble(), ROI_HEIGHT.toDouble()),
            org.opencv.imgproc.Imgproc.INTER_LINEAR,
            org.opencv.core.Core.BORDER_REFLECT_101
        )
        alignedHistoryMat.get(0, 0, historyFlatBytes)
        return historyFlatBytes
    }

    private fun runParent(): Pair<FloatArray, FloatArray> {
        val backend = parentBackend ?: createPreferredParent().also { parentBackend = it }
        updateBackendIdentity(backend)
        val started = System.nanoTime()
        return try {
            val outputs = backend.infer(input)
            lastParentInferenceMs = (System.nanoTime() - started) / 1_000_000f
            lastParentBackendName = backend.backendName
            lastParentComputeUnit = backend.computeUnit
            lastParentLatencyStats = backend.latencyStats
            outputs.heatmap to outputs.offset
        } catch (error: Throwable) {
            if (backend.computeUnit != "CPU") {
                Log.e(
                    TAG,
                    "TinyHeatmapNet parent backend=${backend.backendName} failed; " +
                        "falling back to FP32 CPU",
                    error
                )
                parentBackend = null
                runCatching { backend.close() }.onFailure { closeError ->
                    Log.w(TAG, "Unable to close failed ${backend.backendName} backend", closeError)
                }
                runParent()
            } else {
                throw error
            }
        }
    }

    /** Creates and warms the selected backend before a frame enters the measured interval. */
    private fun ensureParentReady() {
        val backend = parentBackend ?: createPreferredParent().also { parentBackend = it }
        updateBackendIdentity(backend)
        try {
            backend.warmUp()
        } catch (error: Throwable) {
            if (backend.computeUnit == "CPU") throw error
            Log.e(
                TAG,
                "TinyHeatmapNet parent warm-up failed on ${backend.backendName}; " +
                    "falling back to FP32 CPU",
                error
            )
            parentBackend = null
            runCatching { backend.close() }.onFailure { closeError ->
                Log.w(TAG, "Unable to close failed ${backend.backendName} backend", closeError)
            }
            createCpuParent().also {
                parentBackend = it
                it.warmUp()
            }
        }
    }

    /**
     * Benchmarks one ORT session on a synthetic 0.5f input.
     *
     * [channels] must come from the graph under test, not from the production parent: the
     * shipped parent is 9-channel while the retained candidate assets are still 5-channel.
     * Feeding the production buffer to a 5-channel graph is exactly how the harness broke in
     * 2026-11 after the parent switched to 9 channels.
     */
    private fun runBenchmarkSession(session: OrtSession, channels: Int) {
        val elements = channels * PIXELS
        require(elements <= parentInputBuffer.capacity()) {
            "Benchmark input of $channels channels does not fit the parent buffer"
        }
        parentInputBuffer.clear()
        repeat(elements) { parentInputBuffer.put(0.5f) }
        parentInputBuffer.flip()
        val tensor = OnnxTensor.createTensor(
            environment,
            parentInputBuffer,
            longArrayOf(1, channels.toLong(), ROI_HEIGHT.toLong(), ROI_WIDTH.toLong())
        )
        tensor.use { image ->
            session.run(mapOf(PARENT_INPUT to image)).close()
        }
    }

    /** Channel count declared by [session]'s input, so legacy 5-channel assets stay benchable. */
    private fun parentInputChannels(session: OrtSession): Int {
        val info = session.inputInfo[PARENT_INPUT]?.info as? TensorInfo
        return channelsFromShape(info?.shape)
    }

    private fun decodeCandidates(
        heatmap: FloatArray,
        offset: FloatArray,
        stageTimer: TinyPipelineStageTimer
    ): List<Candidate> {
        val decodeStarted = stageTimer.start()
        candidates.clear()
        lastRerankerMs = 0f
        val scores = heatmapScores
        for (index in heatmap.indices) scores[index] = sigmoid(heatmap[index])
        selectHeatmapLocalMaxima(
            scores, HEATMAP_WIDTH, HEATMAP_HEIGHT, NMS_RADIUS, selectedIndices, selectedScores
        )
        val pending = ArrayList<Candidate>(TOP_K)
        for (slot in selectedIndices.indices) {
            val index = selectedIndices[slot]
            val resolvedIndex = if (index >= 0) index else firstUnusedIndex(selectedIndices)
            if (resolvedIndex < 0) continue
            val peakY = resolvedIndex / HEATMAP_WIDTH
            val peakX = resolvedIndex % HEATMAP_WIDTH
            val x = (peakX + offset[resolvedIndex]) * OUTPUT_STRIDE
            val y = (peakY + offset[HEATMAP_PIXELS + resolvedIndex]) * OUTPUT_STRIDE
            // PC keeps masked top-k entries, then clamps their -1 score to 1e-5 before
            // the reranker logit conversion. Every fixed top-k candidate therefore passes
            // through the same patch and CNN path on Android.
            val baseScore = selectedScores[slot].coerceAtLeast(1e-5f)
            val logit = probabilityToLogit(baseScore)
            pending += Candidate(x, y, baseScore, logit)
        }
        stageTimer.recordSince(TinyPipelineStage.SIGMOID_NMS_TOPK, decodeStarted)
        rerankCandidates(pending, stageTimer)
        return candidates
    }

    private fun firstUnusedIndex(selectedIndices: IntArray): Int =
        (0 until HEATMAP_PIXELS).firstOrNull { it !in selectedIndices } ?: -1

    private fun rerankCandidates(
        pending: List<Candidate>,
        stageTimer: TinyPipelineStageTimer
    ) {
        if (pending.isEmpty()) return
        // With the reranker disabled, `decodeCandidates` already left `logit` at
        // `probabilityToLogit(baseScore)`, so `sigmoid(logit) == baseScore == parentScore`.
        // That is exactly the state the ablation replayed, so leaving the list untouched
        // here is the measured configuration, not an unmeasured shortcut.
        if (!candidateReranker) {
            candidates += pending
            return
        }
        val batchSession = batchedRerankerSession
        if (batchSession != null && pending.size == TOP_K) {
            val started = System.nanoTime()
            try {
                val patchStarted = stageTimer.start()
                for ((slot, candidate) in pending.withIndex()) {
                    batchedScores[slot] = candidate.baseScore
                    extractPatch(
                        candidate.x,
                        candidate.y,
                        destination = batchedPatches,
                        destinationOffset = slot * PATCH_VALUES
                    )
                }
                stageTimer.recordSince(TinyPipelineStage.PATCH_EXTRACTION, patchStarted)
                val rerankerStarted = stageTimer.start()
                batchedPatchInputBuffer.clear()
                batchedPatchInputBuffer.put(batchedPatches)
                batchedPatchInputBuffer.flip()
                batchedScoreInputBuffer.clear()
                batchedScoreInputBuffer.put(batchedScores)
                batchedScoreInputBuffer.flip()
                val patchTensor = OnnxTensor.createTensor(
                    environment,
                    batchedPatchInputBuffer,
                    RERANKER_BATCH_PATCH_SHAPE
                )
                val scoreTensor = OnnxTensor.createTensor(
                    environment,
                    batchedScoreInputBuffer,
                    RERANKER_BATCH_SCORE_SHAPE
                )
                val logits = patchTensor.use { patchValue ->
                    scoreTensor.use { scoreValue ->
                        batchSession.run(
                            mapOf(
                                RERANKER_PATCH_INPUT to patchValue,
                                RERANKER_SCORE_INPUT to scoreValue
                            )
                        ).use { result ->
                            val output = (result[0] as? OnnxTensor)?.floatBuffer
                                ?: error("Batched candidate reranker did not return logits")
                            require(output.remaining() >= pending.size) {
                                "Batched candidate reranker returned ${output.remaining()} logits"
                            }
                            FloatArray(pending.size) { output.get(it) }
                        }
                    }
                }
                pending.forEachIndexed { index, candidate ->
                    candidates += candidate.copy(logit = logits[index])
                }
                stageTimer.recordSince(TinyPipelineStage.RERANKER_INFERENCE, rerankerStarted)
                lastRerankerMs = (System.nanoTime() - started) / 1_000_000f
                return
            } catch (error: Throwable) {
                // A vendor/runtime may reject the fixed batch even though the scalar graph is
                // valid. Disable only the optional fast path and preserve the exact old path.
                Log.w(TAG, "Batched candidate reranker failed; falling back to scalar graph", error)
                runCatching { batchSession.close() }
                batchedRerankerSession = null
            }
        }

        val started = System.nanoTime()
        pending.forEach { candidate -> rerankScalar(candidate, stageTimer) }
        lastRerankerMs = (System.nanoTime() - started) / 1_000_000f
    }

    private fun rerankScalar(candidate: Candidate, stageTimer: TinyPipelineStageTimer) {
        val session = rerankerSession ?: error("Scalar reranker reached with the reranker disabled")
        val patchStarted = stageTimer.start()
        extractPatch(candidate.x, candidate.y)
        stageTimer.recordSince(TinyPipelineStage.PATCH_EXTRACTION, patchStarted)
        val rerankerStarted = stageTimer.start()
        patchInputBuffer.clear()
        patchInputBuffer.put(patch)
        patchInputBuffer.flip()
        scoreInputBuffer.clear()
        scoreInputBuffer.put(candidate.baseScore)
        scoreInputBuffer.flip()
        val patchTensor = OnnxTensor.createTensor(
            environment,
            patchInputBuffer,
            RERANKER_PATCH_SHAPE
        )
        val scoreTensor = OnnxTensor.createTensor(
            environment,
            scoreInputBuffer,
            RERANKER_SCORE_SHAPE
        )
        val logit = patchTensor.use { patchValue ->
            scoreTensor.use { scoreValue ->
                session.run(
                    mapOf(
                        RERANKER_PATCH_INPUT to patchValue,
                        RERANKER_SCORE_INPUT to scoreValue
                    )
                ).use { result ->
                    (result[0] as OnnxTensor).floatBuffer.get(0)
                }
            }
        }
        candidates.add(candidate.copy(logit = logit))
        stageTimer.recordSince(TinyPipelineStage.RERANKER_INFERENCE, rerankerStarted)
    }

    private fun extractPatch(
        centerX: Float,
        centerY: Float,
        destination: FloatArray = patch,
        destinationOffset: Int = 0
    ) {
        val radius = PATCH_SIZE / 2
        var outputIndex = 0
        for (channel in 0 until INPUT_CHANNELS) {
            val channelOffset = channel * PIXELS
            for (dy in -radius..radius) {
                val y = centerY + dy
                val y0 = floor(y.toDouble()).toInt()
                val y1 = y0 + 1
                val fy = y - y0
                for (dx in -radius..radius) {
                    val x = centerX + dx
                    val x0 = floor(x.toDouble()).toInt()
                    val x1 = x0 + 1
                    val fx = x - x0
                    val top = sample(channelOffset, x0, y0) * (1f - fx) +
                        sample(channelOffset, x1, y0) * fx
                    val bottom = sample(channelOffset, x0, y1) * (1f - fx) +
                        sample(channelOffset, x1, y1) * fx
                    destination[destinationOffset + outputIndex++] = top * (1f - fy) + bottom * fy
                }
            }
        }
    }

    private fun sample(channelOffset: Int, x: Int, y: Int): Float {
        val clampedX = x.coerceIn(0, ROI_WIDTH - 1)
        val clampedY = y.coerceIn(0, ROI_HEIGHT - 1)
        val index = clampedY * ROI_WIDTH + clampedX
        val channel = channelOffset / PIXELS
        return if (channel < MOTION_CHANNEL_START) {
            input[channelOffset + index]
        } else {
            sampleMotion(channel, index)
        }
    }

    private fun transformPoint(x: Float, y: Float, matrix: DoubleArray): Pair<Float, Float> {
        require(matrix.size == 9) { "A homography must contain nine values" }
        val denominator = matrix[6] * x + matrix[7] * y + matrix[8]
        if (!denominator.isFinite() || kotlin.math.abs(denominator) < 1e-12) {
            return Float.NaN to Float.NaN
        }
        return (
            (matrix[0] * x + matrix[1] * y + matrix[2]) / denominator
            ).toFloat() to (
            (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator
            ).toFloat()
    }

    /**
     * [uint8Normalized] for a motion channel that is already an exact integer.
     *
     * Keeps the motion block in integer arithmetic; the result is identical because
     * `coerceIn(0f, 255f).toInt()` on an integral value is just a clamp.
     */
    private fun scaledToUnit(scaled: Int): Float =
        scaled.coerceIn(0, 255).toFloat() / 255f

    /**
     * Exact median of the lag difference, which is integral because both flattened operands
     * are whole uint8 levels.
     */
    private fun probabilityToLogit(probability: Float): Float {
        val clipped = probability.coerceIn(1e-5f, 1f - 1e-5f)
        return kotlin.math.ln(clipped / (1f - clipped))
    }

    private fun sigmoid(value: Float): Float =
        (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()

    private fun uint8Normalized(value: Float): Float =
        value.coerceIn(0f, 255f).toInt() / 255f

    private fun createCpuParent(): TinyHeatmapParentBackend =
        OrtTinyHeatmapParentBackend(
            environment = environment,
            modelBytes = appContext.assets.open(PARENT_MODEL_FILE).use { it.readBytes() },
            inputShape = parentInputShape,
            heatmapSize = HEATMAP_PIXELS,
            offsetSize = HEATMAP_PIXELS * 2,
            threads = cpuInferenceThreads.coerceAtLeast(1),
            backendName = "ort_cpu",
            computeUnit = "CPU",
            placementVerified = true,
            placementSummary = "ort_cpu_full_graph"
        ).also {
            Log.i(TAG, "TinyHeatmapNet parent is running on FP32 CPU")
        }

    /**
     * Selects the production backend once. A failed hardware construction is remembered so a
     * device without a usable HTP/Vulkan stack does not repeatedly allocate/destroy accelerator
     * state on subsequent frames.
     */
    private fun createPreferredParent(): TinyHeatmapParentBackend {
        if (!allowHardwareParent) return createCpuParent()
        if (!qnnParentAttempted) {
            qnnParentAttempted = true
            createQnnParent(appContext)?.let { return it }
        }
        if (!hardwareParentAttempted) {
            hardwareParentAttempted = true
            createNcnnParent(appContext)?.let { return it }
        }
        return createCpuParent()
    }

    private fun createQnnParent(context: Context): TinyHeatmapParentBackend? =
        runCatching {
            QnnContextTinyHeatmapParentBackend.create(context)
        }
            .onSuccess { backend ->
                if (backend != null) {
                    Log.i(
                        TAG,
                        "TinyHeatmapNet parent is running on QNN HTP placement=" +
                            backend.placementSummary
                    )
                }
            }
            .onFailure {
                Log.w(TAG, "QNN HTP parent unavailable; trying next backend", it)
            }
            .getOrNull()

    private fun createNcnnParent(context: Context): TinyHeatmapParentBackend? {
        // The shipped ncnn asset is a 5-channel graph.  The parent input is now 9 channels,
        // so this backend cannot serve it; skip it explicitly instead of handing NCNN a
        // tensor whose channel count its graph cannot accept.  The chain stays
        // QNN HTP -> ORT CPU, which is what production already ran.
        if (PARENT_CHANNELS != NCNN_PARENT_CHANNELS) {
            Log.i(
                TAG,
                "NCNN parent skipped: shipped asset is $NCNN_PARENT_CHANNELS-channel, " +
                    "parent is $PARENT_CHANNELS-channel"
            )
            return null
        }
        return runCatching {
            NcnnTinyHeatmapNet(
                context = context,
                useVulkan = true,
                paramAsset = PRODUCTION_NCNN_PARAM_FILE,
                modelAsset = PRODUCTION_NCNN_MODEL_FILE
            )
        }
            .onSuccess {
                Log.i(
                    TAG,
                    "TinyHeatmapNet parent is running on NCNN Vulkan placement=" +
                        it.placementSummary
                )
            }
            .onFailure {
                Log.w(TAG, "NCNN Vulkan parent unavailable; using FP32 CPU fallback", it)
            }
            .getOrNull()
    }

    private fun updateBackendIdentity(backend: TinyHeatmapParentBackend) {
        lastParentBackendName = backend.backendName
        lastParentComputeUnit = backend.computeUnit
        lastParentPlacementVerified = backend.placementVerified
        lastParentPlacementSummary = backend.placementSummary
    }

    private fun createSession(
        context: Context,
        assetName: String,
        useNnapi: Boolean = false
    ): OrtSession {
        val modelBytes = context.assets.open(assetName).use { it.readBytes() }
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(cpuInferenceThreads.coerceAtLeast(1))
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            if (useNnapi) {
                setIntraOpNumThreads(1)
                addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED))
            }
        }
        return try {
            environment.createSession(modelBytes, options)
        } finally {
            options.close()
        }
    }

    private fun createOptionalSession(context: Context, assetName: String): OrtSession? =
        runCatching { createSession(context, assetName) }
            .onFailure {
                Log.w(TAG, "Optional batched reranker asset unavailable: $assetName", it)
            }
            .getOrNull()

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        runCatching { parentBackend?.close() }.onFailure {
            Log.w(TAG, "Unable to close TinyHeatmapNet parent backend", it)
        }
        parentBackend = null
        runCatching { batchedRerankerSession?.close() }.onFailure {
            Log.w(TAG, "Unable to close batched candidate reranker session", it)
        }
        batchedRerankerSession = null
        runCatching { rerankerSession?.close() }.onFailure {
            Log.w(TAG, "Unable to close candidate reranker session", it)
        }
        rawMat.release()
        flatFloatMat.release()
        blur2Mat.release()
        blur5Mat.release()
        historyMat.release()
        alignedHistoryMat.release()
        historyGeometryMat.release()
        currentGeometryMat.release()
        inverseGeometryMat.release()
        historyToCurrentMat.release()
        gemmZeroMat.release()
    }

    companion object {
        private const val TAG = "TinyHeatmapNet"
        private const val PARENT_MODEL_FILE = "tinyheatmapnet_causal9_heads.onnx"
        private const val PRODUCTION_NCNN_PARAM_FILE = "tinyheatmapnet_final.ncnn.param"
        private const val PRODUCTION_NCNN_MODEL_FILE = "tinyheatmapnet_final.ncnn.bin"
        private const val RERANKER_MODEL_FILE = "candidate_reranker_final.onnx"
        private const val RERANKER_BATCH_MODEL_FILE = "candidate_reranker_batched.onnx"
        private const val ROI_WIDTH = TubeRoiSampler.OUTPUT_WIDTH
        private const val ROI_HEIGHT = TubeRoiSampler.OUTPUT_HEIGHT
        private const val PIXELS = ROI_WIDTH * ROI_HEIGHT
        private const val INPUT_CHANNELS = 9

        /**
         * The parent now consumes the whole 9-channel causal block: 5 spatial channels plus
         * 2 lags x 2 signs of aligned frame difference. Model: `tinyheatmapnet_causal9.onnx`
         * (ORT/NNAPI CPU fallback) and `tinyheatmapnet_causal9_qnn_context.bin` (QNN HTP FP16,
         * compiled by AI Hub, 145/145 layers on NPU).
         */
        private const val PARENT_CHANNELS = 9
        /** Channel count baked into the shipped ncnn asset, used to skip it for a 9-channel parent. */
        private const val NCNN_PARENT_CHANNELS = 5
        private const val MOTION_CHANNEL_START = 5
        private val MOTION_LAGS = intArrayOf(1, 3)
        private const val OUTPUT_STRIDE = 2f
        private const val HEATMAP_WIDTH = ROI_WIDTH / 2
        private const val HEATMAP_HEIGHT = ROI_HEIGHT / 2
        private const val HEATMAP_PIXELS = HEATMAP_WIDTH * HEATMAP_HEIGHT
        private const val TOP_K = 5
        private const val NMS_RADIUS = 2
        private const val PATCH_SIZE = 17
        private const val PATCH_VALUES = INPUT_CHANNELS * PATCH_SIZE * PATCH_SIZE
        private val RERANKER_BATCH_PATCH_SHAPE = longArrayOf(
            TOP_K.toLong(),
            INPUT_CHANNELS.toLong(),
            PATCH_SIZE.toLong(),
            PATCH_SIZE.toLong()
        )
        private val RERANKER_BATCH_SCORE_SHAPE = longArrayOf(TOP_K.toLong())
        /**
         * The camera thread stabilises every capture, so three entries are enough while the
         * analyzer runs on every frame. The analyzer can skip frames under load, and one skip
         * would drop lag-1 for the next capture; six entries keep both lags reachable through
         * a short burst of skipped captures at ~1 MB of retained ROI bytes.
         */
        private const val HISTORY_CAPACITY = 6
        /**
         * Threads for the single ROI inference.
         *
         * This deliberately stays conservative. The app runs a second ONNX session for YOLO
         * plus the camera analysis thread, so a larger pool competes for the same cores and
         * ONNX Runtime's spinning thread pool makes that contention expensive. Raising this
         * measured neutral on a desktop, so there is no evidence it helps the phone; it must
         * not be widened without an on-device measurement.
         */
        private const val MAX_INFERENCE_THREADS = 4

        internal fun inferenceThreadCount(
            availableProcessors: Int = Runtime.getRuntime().availableProcessors(),
            fastCores: Int = CpuAffinity.fastCoreCount
        ): Int {
            // Three pipelines share the big cluster: the camera analyzer (ROI stabilisation,
            // scale sampling), the tube detector session, and this parent session.  On a
            // four-fast-core phone every one of them claiming `cores / 2` oversubscribes the
            // cluster and they thrash each other: measured on a Dimensity 1000+ (2026-11-15)
            // the parent ran 287 ms inside the pipeline against 44 ms in isolation.  Reserve
            // cores for the camera phase instead of taking half of everything.  Devices with
            // six or more fast cores keep the old four-thread cap.
            val cap = if (fastCores in 1..MAX_INFERENCE_THREADS) {
                (fastCores - RESERVED_FAST_CORES).coerceAtLeast(1)
            } else {
                MAX_INFERENCE_THREADS
            }
            return (availableProcessors / 2).coerceIn(1, cap)
        }
        private const val RESERVED_FAST_CORES = 2
        private const val MAX_HISTORY_GAP_NS = 700_000_000L
        // Derived from PARENT_CHANNELS on purpose: when the parent moved to 9 causal channels
        // this stayed hard-coded at 5 and broke every CPU fallback (and every benchmark) with
        // "Expected 778240 parent input values, got 1400832".  Never hard-code the channel
        // count here again.  Internal so the contract test can lock it to the shipped ONNX.
        internal val parentInputShape =
            longArrayOf(1, PARENT_CHANNELS.toLong(), ROI_HEIGHT.toLong(), ROI_WIDTH.toLong())

        /**
         * Channel count implied by a model input shape.
         *
         * Benchmark sessions must match the graph they load: the shipped parent is 9-channel
         * while the retained candidate assets are 5-channel.  Unknown/incomplete shapes fall
         * back to the legacy count so a malformed report cannot silently allocate the wrong
         * tensor.
         */
        internal fun channelsFromShape(shape: LongArray?): Int =
            shape?.getOrNull(1)?.toInt()?.takeIf { it > 0 } ?: NCNN_PARENT_CHANNELS

        private val RERANKER_PATCH_SHAPE = longArrayOf(1, 9, PATCH_SIZE.toLong(), PATCH_SIZE.toLong())
        private val RERANKER_SCORE_SHAPE = longArrayOf(1)
        private const val PARENT_INPUT = "image"
        private const val RERANKER_PATCH_INPUT = "patches"
        private const val RERANKER_SCORE_INPUT = "base_scores"
    }
}
