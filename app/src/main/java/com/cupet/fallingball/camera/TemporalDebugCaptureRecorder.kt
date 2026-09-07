package com.cupet.fallingball.camera

import android.content.Context
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import com.cupet.fallingball.vision.AnalysisFrame
import com.cupet.fallingball.analysis.FallFrameObservation
import com.cupet.fallingball.analysis.FallTrackingStatus
import com.cupet.fallingball.analysis.FallTrackingUpdate
import com.cupet.fallingball.vision.BallEndReason
import com.cupet.fallingball.vision.BufferTransform
import com.cupet.fallingball.vision.ClassicalHeatmapDiagnostics
import com.cupet.fallingball.vision.ClassicalHeatmapRejection
import com.cupet.fallingball.vision.DenseTrackedBall
import com.cupet.fallingball.vision.DenseTrackerDiagnostics
import com.cupet.fallingball.vision.RoiStabilizationTransform
import com.cupet.fallingball.vision.RulerConfig
import com.cupet.fallingball.vision.RulerSnapshot
import com.cupet.fallingball.vision.TemporalBallDetection
import com.cupet.fallingball.vision.TemporalBallPeak
import com.cupet.fallingball.vision.TemporalRoiSequenceBuffer
import com.cupet.fallingball.vision.TinyHeatmapRuntimeDiagnostics
import com.cupet.fallingball.vision.TinyHeatmapThroughputStats
import com.cupet.fallingball.vision.TinyPipelinePhase
import com.cupet.fallingball.vision.TinyPipelineStage
import com.cupet.fallingball.vision.TinyPipelineStageTimings
import com.cupet.fallingball.vision.TubeAxis
import com.cupet.fallingball.vision.TubeRoiSampler
import java.io.BufferedOutputStream
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.nio.ByteBuffer
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import org.json.JSONArray
import org.json.JSONObject

/** Debug-build evidence captured at the exact CameraX-to-ball-detector production boundary. */
internal class TemporalDebugCaptureRecorder(private val context: Context) : AutoCloseable {
    private enum class CaptureMode { EXTERNAL, TRACKING_SESSION }
    private enum class CaptureDetail { FULL, TIMING_ONLY }

    internal data class BufferedStep(
        val token: Long,
        val raw: ByteArray,
        val stable: ByteArray,
        val timestampNs: Long,
        val axis: TubeAxis,
        val ruler: RulerSnapshot,
        val transform: BufferTransform,
        val stabilization: RoiStabilizationTransform,
        val geometryCaptureTimestampNs: Long,
        val geometryRevision: Long,
        val temporalGeneration: Long,
        val captureCostMs: Float,
        @Volatile var detection: DetectionEvidence? = null
    )

    internal data class DetectionEvidence(
        val framesSeen: Int,
        val detection: TemporalBallDetection?,
        val generation: Long,
        val cameraTimestampNs: Long,
        val inferenceMs: Float,
        val queueMs: Float,
        val queuedAtNs: Long,
        val startedAtNs: Long,
        val completedAtNs: Long,
        val runtime: TinyHeatmapRuntimeDiagnostics = TinyHeatmapRuntimeDiagnostics(),
        val throughput: TinyHeatmapThroughputStats = TinyHeatmapThroughputStats(),
        val classicalDiagnostics: ClassicalHeatmapDiagnostics? = null,
        val pipelineTimings: TinyPipelineStageTimings = TinyPipelineStageTimings.EMPTY
    )

    val diagnosticsEnabled: Boolean =
        context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0
    private val packageInfo = runCatching {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.packageManager.getPackageInfo(
                context.packageName,
                PackageManager.PackageInfoFlags.of(0L)
            )
        } else {
            @Suppress("DEPRECATION")
            context.packageManager.getPackageInfo(context.packageName, 0)
        }
    }.getOrNull()
    private val appVersionName = packageInfo?.versionName ?: "unknown"
    @Suppress("DEPRECATION")
    private val appVersionCode = packageInfo?.let {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) it.longVersionCode
        else it.versionCode.toLong()
    } ?: -1L
    private val activeRef = AtomicReference<CaptureSession?>(null)
    private val modeRef = AtomicReference<CaptureMode?>(null)
    private val detailRef = AtomicReference<CaptureDetail?>(null)
    private val finishingRunId = AtomicReference<String?>(null)
    private val trackingStartTimestampNs = AtomicLong(Long.MIN_VALUE)
    private val activeStepIds = ConcurrentHashMap<Long, Long>()
    private val nextToken = AtomicLong(1L)
    private val preRoll = FixedPreRollBuffer<BufferedStep>(8)
    private val finalizer = Executors.newSingleThreadExecutor()
    private val closed = AtomicBoolean(false)
    private var trackingRoiQuota: TrackingDiagnosticRoiQuota? = null

    val isActive: Boolean get() = activeRef.get() != null
    val capturesDenseDiagnostics: Boolean get() = detailRef.get() == CaptureDetail.FULL

    /** Existing bounded runner control. It keeps the legacy fixed directory contract. */
    @Synchronized
    fun start(): Boolean = startExternal(CaptureDetail.FULL)

    /** Low-overhead timing mode: keep events, but skip Y-plane/ROI payloads and dense probes. */
    @Synchronized
    fun startTimingOnly(): Boolean = startExternal(CaptureDetail.TIMING_ONLY)

    private fun startExternal(detail: CaptureDetail): Boolean {
        if (!diagnosticsEnabled || closed.get() || activeRef.get() != null) return false
        return runCatching {
            val session = CaptureSession(
                directory = context.filesDir.resolve(EXTERNAL_OUTPUT_DIR),
                resetDirectory = true,
                roiBudgetBytes = TrackingDiagnosticRoiBudget.PER_SESSION_BYTES,
                globalRoiQuota = null,
                appVersionName = appVersionName,
                appVersionCode = appVersionCode,
                capturePayloads = detail == CaptureDetail.FULL
            )
            activeStepIds.clear()
            finishingRunId.set(null)
            trackingStartTimestampNs.set(Long.MIN_VALUE)
            modeRef.set(CaptureMode.EXTERNAL)
            detailRef.set(detail)
            activeRef.set(session)
        }.onFailure { Log.e(TAG, "Unable to start external temporal capture", it) }.isSuccess
    }

    @Synchronized
    fun stop(): Boolean {
        if (modeRef.get() != CaptureMode.EXTERNAL) return false
        val capture = activeRef.getAndSet(null) ?: return false
        modeRef.set(null)
        detailRef.set(null)
        activeStepIds.clear()
        finishingRunId.set(null)
        trackingStartTimestampNs.set(Long.MIN_VALUE)
        capture.close()
        return true
    }

    /** Starts one automatically retained package without resetting causal detector state. */
    @Synchronized
    fun startTrackingSession(
        runId: String,
        startTimestampNs: Long,
        config: RulerConfig,
        resolutionLabel: String,
        generation: Long,
        modelAvailable: Boolean
    ): Boolean {
        if (!diagnosticsEnabled) return true
        if (closed.get()) return false
        if (finishingRunId.get() != null) return false
        val existing = activeRef.get()
        if (existing != null) {
            if (modeRef.get() != CaptureMode.EXTERNAL) return false
            existing.recordSessionStart(
                runId,
                startTimestampNs,
                config,
                resolutionLabel,
                generation,
                modelAvailable,
                preRollCount = 0
            )
            return true
        }
        return runCatching {
            val root = context.filesDir.resolve(TRACKING_OUTPUT_DIR)
            require(root.exists() || root.mkdirs()) { "Cannot create diagnostics root" }
            val directory = root.resolve(runId)
            require(!directory.exists()) { "Diagnostic run already exists" }
            val retention = TrackingDiagnosticRoiRetention.prepareForNextSession(
                root = root,
                protectedRunIds = setOfNotNull(finishingRunId.get())
            )
            if (retention.evictedRunIds.isNotEmpty()) {
                Log.i(
                    TAG,
                    "Evicted oldest ROI diagnostics: " +
                        retention.evictedRunIds.joinToString(",")
                )
            }
            val quota = TrackingDiagnosticRoiQuota(retention.existingBytes)
                .also { trackingRoiQuota = it }
            val capture = CaptureSession(
                directory,
                resetDirectory = false,
                roiBudgetBytes = TrackingDiagnosticRoiBudget.PER_SESSION_BYTES,
                globalRoiQuota = quota,
                appVersionName = appVersionName,
                appVersionCode = appVersionCode
            )
            activeStepIds.clear()
            val buffered = preRoll.snapshot()
            capture.recordSessionStart(
                runId,
                startTimestampNs,
                config,
                resolutionLabel,
                generation,
                modelAvailable,
                preRollCount = buffered.size
            )
            buffered.forEach { step ->
                activeStepIds[step.token] = capture.recordInput(step, preRoll = true)
            }
            trackingStartTimestampNs.set(startTimestampNs)
            modeRef.set(CaptureMode.TRACKING_SESSION)
            detailRef.set(CaptureDetail.FULL)
            activeRef.set(capture)
        }.onFailure { Log.e(TAG, "Unable to start tracking diagnostics", it) }.isSuccess
    }

    @Synchronized
    fun finishTrackingSession(
        runId: String,
        endReason: BallEndReason,
        sampleCount: Int,
        convgruSampleCount: Int,
        localSampleCount: Int,
        modelError: Boolean,
        endTimestampNs: Long,
        throughput: TinyHeatmapThroughputStats,
        onFinalized: () -> Unit
    ) {
        if (!diagnosticsEnabled) {
            onFinalized()
            return
        }
        val capture = activeRef.get()
        if (capture == null) {
            onFinalized()
            return
        }
        capture.recordSessionEnd(
            runId,
            endReason,
            sampleCount,
            convgruSampleCount,
            localSampleCount,
            modelError,
            endTimestampNs,
            throughput
        )
        if (modeRef.get() != CaptureMode.TRACKING_SESSION) {
            if (!capture.runAfterPending(onFinalized)) onFinalized()
            return
        }
        activeRef.set(null)
        modeRef.set(null)
        detailRef.set(null)
        trackingStartTimestampNs.set(Long.MIN_VALUE)
        activeStepIds.clear()
        runCatching {
            finalizer.execute {
                try {
                    capture.close()
                } finally {
                    finishingRunId.compareAndSet(runId, null)
                    onFinalized()
                }
            }
        }.onFailure {
            try {
                capture.close()
            } finally {
                finishingRunId.compareAndSet(runId, null)
                onFinalized()
            }
        }
    }

    /** Stops post-finish camera frames joining this run while in-flight inference can drain. */
    fun beginTrackingSessionFinish(runId: String) {
        if (modeRef.get() == CaptureMode.TRACKING_SESSION && activeRef.get() != null) {
            finishingRunId.compareAndSet(null, runId)
        }
    }

    private fun acceptsTrackingEvent(): Boolean =
        modeRef.get() != CaptureMode.TRACKING_SESSION || finishingRunId.get() == null

    private fun acceptsSessionTimestamp(timestampNs: Long): Boolean =
        isTrackingSessionTimestampEligible(
            trackingSessionActive = modeRef.get() == CaptureMode.TRACKING_SESSION,
            sessionStartTimestampNs = trackingStartTimestampNs.get(),
            timestampNs = timestampNs
        )

    fun recordSchedule(
        timestampNs: Long,
        outcome: TemporalScheduleOutcome,
        ruler: RulerSnapshot,
        geometryAgeNs: Long,
        generation: Long,
        loadLevel: AnalysisLoadLevel,
        thermal: DeviceThermalSnapshot?,
        yoloBusy: Boolean,
        temporalBusy: Boolean,
        cooldownRemainingNs: Long
    ) {
        if (!acceptsTrackingEvent() || !acceptsSessionTimestamp(timestampNs)) return
        activeRef.get()?.recordSchedule(
            timestampNs,
            outcome,
            ruler,
            geometryAgeNs,
            generation,
            loadLevel,
            thermal,
            yoloBusy,
            temporalBusy,
            cooldownRemainingNs
        )
    }

    fun recordAnalyzerTiming(timestampNs: Long, timings: TinyPipelineStageTimings) {
        if (!acceptsTrackingEvent() || !acceptsSessionTimestamp(timestampNs)) return
        activeRef.get()?.recordAnalyzerTiming(timestampNs, timings)
    }

    fun recordFinalizationTimings(runId: String, timings: TinyPipelineStageTimings) {
        if (!acceptsTrackingEvent()) return
        activeRef.get()?.recordFinalizationTimings(runId, timings)
    }

    /** Accepts the one post-session record build while ordinary late frame events stay blocked. */
    fun recordRecordBuildTimings(runId: String, timings: TinyPipelineStageTimings) {
        val mode = modeRef.get() ?: return
        if (mode == CaptureMode.TRACKING_SESSION && finishingRunId.get() != runId) return
        activeRef.get()?.recordRecordBuildTimings(runId, timings)
    }

    @Synchronized
    fun recordReset(
        reason: TemporalResetReason,
        timestampNs: Long?,
        previousGeneration: Long,
        nextGeneration: Long
    ) {
        preRoll.clear()
        if (!acceptsTrackingEvent()) return
        activeRef.get()?.recordReset(reason, timestampNs, previousGeneration, nextGeneration)
    }

    fun captureYPlaneOnce(image: AnalysisFrame, transform: BufferTransform, timestampNs: Long) {
        if (!acceptsTrackingEvent() || !acceptsSessionTimestamp(timestampNs)) return
        activeRef.get()?.captureYPlaneOnce(image, transform, timestampNs)
    }

    @Synchronized
    fun recordInput(
        raw: ByteBuffer,
        stable: ByteBuffer,
        timestampNs: Long,
        axis: TubeAxis,
        ruler: RulerSnapshot,
        transform: BufferTransform,
        stabilization: RoiStabilizationTransform,
        geometryCaptureTimestampNs: Long,
        geometryRevision: Long,
        temporalGeneration: Long
    ): Long? {
        if (!diagnosticsEnabled || closed.get() || !acceptsTrackingEvent()) return null
        val startedNs = System.nanoTime()
        val capturePayloads = detailRef.get() != CaptureDetail.TIMING_ONLY
        val rawBytes = if (capturePayloads) copyRemaining(raw) else EMPTY_BYTES
        val stableBytes = if (capturePayloads) copyRemaining(stable) else EMPTY_BYTES
        val expectedBytes = TubeRoiSampler.OUTPUT_WIDTH * TubeRoiSampler.OUTPUT_HEIGHT
        if (capturePayloads) {
            require(rawBytes.size == expectedBytes && stableBytes.size == expectedBytes)
        }
        val token = nextToken.getAndIncrement()
        val step = BufferedStep(
            token,
            rawBytes,
            stableBytes,
            timestampNs,
            axis,
            ruler,
            transform,
            stabilization,
            geometryCaptureTimestampNs,
            geometryRevision,
            temporalGeneration,
            (System.nanoTime() - startedNs) / 1_000_000f
        )
        preRoll.add(step)
        activeRef.get()?.let { capture ->
            activeStepIds[token] = capture.recordInput(
                step,
                preRoll = !acceptsSessionTimestamp(timestampNs)
            )
        }
        return token
    }

    @Synchronized
    fun recordDetection(
        stepToken: Long?,
        framesSeen: Int,
        detection: TemporalBallDetection?,
        generation: Long,
        cameraTimestampNs: Long,
        inferenceMs: Float,
        queueMs: Float,
        queuedAtNs: Long,
        startedAtNs: Long,
        completedAtNs: Long,
        runtime: TinyHeatmapRuntimeDiagnostics = TinyHeatmapRuntimeDiagnostics(),
        throughput: TinyHeatmapThroughputStats = TinyHeatmapThroughputStats(),
        classicalDiagnostics: ClassicalHeatmapDiagnostics? = null,
        pipelineTimings: TinyPipelineStageTimings = TinyPipelineStageTimings.EMPTY
    ) {
        if (stepToken == null) return
        val evidence = DetectionEvidence(
            framesSeen,
            detection,
            generation,
            cameraTimestampNs,
            inferenceMs,
            queueMs,
            queuedAtNs,
            startedAtNs,
            completedAtNs,
            runtime,
            throughput,
            classicalDiagnostics,
            pipelineTimings
        )
        preRoll.snapshot().firstOrNull { it.token == stepToken }?.detection = evidence
        val localId = activeStepIds[stepToken] ?: return
        activeRef.get()?.recordDetection(localId, evidence)
    }

    @Synchronized
    fun recordInferenceError(
        stepToken: Long?,
        generation: Long,
        queuedAtNs: Long,
        startedAtNs: Long,
        completedAtNs: Long,
        message: String
    ) {
        if (stepToken == null) return
        val localId = activeStepIds[stepToken] ?: return
        activeRef.get()?.recordInferenceError(
            localId,
            generation,
            queuedAtNs,
            startedAtNs,
            completedAtNs,
            message
        )
    }

    fun recordSelection(
        stepToken: Long?,
        primary: MappedTemporalPeak,
        alternatives: List<MappedTemporalPeak>,
        selected: MappedTemporalPeak?,
        threshold: Float,
        gate: FallTrackingStatus,
        resultAgeMs: Float,
        diagnostics: TemporalSelectionDiagnostics? = null
    ) {
        if (!acceptsTrackingEvent()) return
        val localId = stepToken?.let(activeStepIds::get) ?: return
        activeRef.get()?.recordSelection(
            localId,
            primary,
            alternatives,
            selected,
            threshold,
            gate,
            resultAgeMs,
            diagnostics
        )
    }

    fun recordGate(
        stepToken: Long?,
        observation: FallFrameObservation?,
        before: FallTrackingStatus,
        update: FallTrackingUpdate,
        after: FallTrackingStatus
    ) {
        if (!acceptsTrackingEvent()) return
        activeRef.get()?.recordGate(
            stepToken?.let(activeStepIds::get),
            observation,
            before,
            update,
            after
        )
    }

    fun recordDense(
        timestampNs: Long,
        outputEnabled: Boolean,
        motionSeedAvailable: Boolean = outputEnabled,
        diagnostics: DenseTrackerDiagnostics,
        proposal: DenseTrackedBall?,
        accepted: Boolean
    ) {
        if (!acceptsTrackingEvent() || !acceptsSessionTimestamp(timestampNs)) return
        activeRef.get()?.recordDense(
            timestampNs,
            outputEnabled,
            motionSeedAvailable,
            diagnostics,
            proposal,
            accepted
        )
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        activeRef.getAndSet(null)?.close()
        modeRef.set(null)
        detailRef.set(null)
        finishingRunId.set(null)
        trackingStartTimestampNs.set(Long.MIN_VALUE)
        activeStepIds.clear()
        finalizer.shutdown()
        if (!finalizer.awaitTermination(CLOSE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
            finalizer.shutdownNow()
        }
    }

    internal class CaptureSession(
        private val directory: File,
        resetDirectory: Boolean,
        private val roiBudgetBytes: Long,
        private val globalRoiQuota: TrackingDiagnosticRoiQuota?,
        private val appVersionName: String,
        private val appVersionCode: Long,
        private val capturePayloads: Boolean = true
    ) : AutoCloseable {
        private val accepting = AtomicBoolean(true)
        private val yPlaneCaptured = AtomicBoolean(false)
        private val nextStepId = AtomicLong(1L)
        private val detectionStepIds = ConcurrentHashMap.newKeySet<Long>()
        private val selectedStepIds = ConcurrentHashMap.newKeySet<Long>()
        private val preRollStepIds = ConcurrentHashMap.newKeySet<Long>()
        private val executor = Executors.newSingleThreadExecutor()
        private val droppedEvents = AtomicInteger(0)
        private val rawOutput: BufferedOutputStream
        private val stableOutput: BufferedOutputStream
        private val events: BufferedWriter
        private val startedAtEpochMs = System.currentTimeMillis()
        private val skipCounts = linkedMapOf<TemporalScheduleOutcome, Int>()
        private val cameraIntervalsMs = ArrayList<Float>()
        private val modelIntervalsMs = ArrayList<Float>()
        private val inferenceTimesMs = ArrayList<Float>()
        private val preprocessTimesMs = ArrayList<Float>()
        private val parentInferenceTimesMs = ArrayList<Float>()
        private val postprocessTimesMs = ArrayList<Float>()
        private val rerankerTimesMs = ArrayList<Float>()
        private val pipelineStageTimesMs = linkedMapOf<TinyPipelineStage, ArrayList<Float>>()
        private val captureCostsMs = ArrayList<Float>()
        private val backendNames = linkedSetOf<String>()
        private val computeUnits = linkedSetOf<String>()
        private val placementVerifiedStates = linkedSetOf<Boolean>()
        private val placementSummaries = linkedSetOf<String>()
        private val scoreBands = linkedMapOf<TrackingScoreBand, Int>()
        private val selectorOutcomes = linkedMapOf<TemporalSelectionOutcome, Int>()
        private val gateRejections = linkedMapOf<com.cupet.fallingball.analysis.FallTrackingRejection, Int>()
        private val classicalRejections = linkedMapOf<ClassicalHeatmapRejection, Int>()
        private var lastCameraTimestampNs = 0L
        private var lastModelTimestampNs = 0L
        private var inputCount = 0
        private var sessionInputCount = 0
        private var roiInputCount = 0
        private var detectionCount = 0
        private var classicalEvidenceCount = 0
        private var classicalOutputCount = 0
        private var selectionEventCount = 0
        private var selectedCount = 0
        private var gateCount = 0
        private var acceptedObservationCount = 0
        private var denseEventCount = 0
        private var denseOutputEnabledFrames = 0
        private var denseProposalCount = 0
        private var denseAcceptedCount = 0
        private var resetCount = 0
        private var yPlaneByteCount = 0
        private var roiBytesWritten = 0L
        private var maximumFramesSeen = 0
        private var maximumScore: Float? = null
        private var modelAvailable = true
        private var modelError = false
        private var sampleCount = 0
        private var convgruSampleCount = 0
        private var localSampleCount = 0
        private var activeRunId: String? = null
        private var endReason: String? = null
        private var sessionEndTimestampNs: Long? = null
        private var roiTruncated = false
        private var sessionStartTimestampNs: Long? = null
        private var sessionStartResolution: String? = null
        private var sessionTickSpacingMm: Float? = null
        private var sessionExpectedTickCount: Int? = null
        private var sessionGeneration: Long? = null
        private var sessionEnded = false
        private var modelErrorEventCount = 0
        private var lastThroughput = TinyHeatmapThroughputStats()

        init {
            require(directory.exists() || directory.mkdirs()) {
                "Cannot create temporal capture directory"
            }
            if (resetDirectory) OUTPUT_FILES.forEach { name -> File(directory, name).delete() }
            rawOutput = BufferedOutputStream(FileOutputStream(File(directory, RAW_FILE)))
            stableOutput = BufferedOutputStream(FileOutputStream(File(directory, STABLE_FILE)))
            events = BufferedWriter(
                OutputStreamWriter(FileOutputStream(File(directory, EVENTS_FILE)), Charsets.UTF_8)
            )
            File(directory, Y_PLANE_FILE).writeBytes(ByteArray(0))
            writeStatus("running")
        }

        fun recordSessionStart(
            runId: String,
            startTimestampNs: Long,
            config: RulerConfig,
            resolutionLabel: String,
            generation: Long,
            available: Boolean,
            preRollCount: Int
        ) = submit {
            activeRunId = runId
            sessionStartTimestampNs = startTimestampNs
            sessionStartResolution = resolutionLabel
            sessionTickSpacingMm = config.tickSpacingMm
            sessionExpectedTickCount = config.expectedTickCount
            sessionGeneration = generation
            modelAvailable = available
            appendEvent(
                JSONObject()
                    .put("event", "session_start")
                    .put("run_id", runId)
                    .put("camera_timestamp_ns", startTimestampNs)
                    .put("wall_clock_epoch_ms", System.currentTimeMillis())
                    .put("app_version_name", appVersionName)
                    .put("app_version_code", appVersionCode)
                    .put("resolution", resolutionLabel)
                    .put("tick_spacing_mm", config.tickSpacingMm)
                    .put("expected_tick_count", config.expectedTickCount)
                    .put("temporal_generation", generation)
                    .put("pre_roll_count", preRollCount)
                    .put("model", MODEL_FILE)
                    .put("roi_width", TubeRoiSampler.OUTPUT_WIDTH)
                    .put("roi_height", TubeRoiSampler.OUTPUT_HEIGHT)
                    .put("sequence_length", 1)
                    .put("warmup_frames", CLASSICAL_WARMUP_FRAMES)
                    .put("presence_threshold", CLASSICAL_PRESENCE_THRESHOLD)
                    .put("recovery_threshold", CLASSICAL_RECOVERY_THRESHOLD)
                    .put("maximum_frame_gap_ms", TemporalRoiSequenceBuffer.MAX_FRAME_GAP_NS / 1_000_000L)
                    .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
                    .put("sdk", Build.VERSION.SDK_INT)
            )
        }

        fun recordSessionEnd(
            runId: String,
            reason: BallEndReason,
            samples: Int,
            convgruSamples: Int,
            localSamples: Int,
            hasModelError: Boolean,
            endTimestampNs: Long? = null,
            throughput: TinyHeatmapThroughputStats? = null
        ) = submit {
            sampleCount = samples
            convgruSampleCount = convgruSamples
            localSampleCount = localSamples
            modelError = modelError || hasModelError
            throughput?.let { lastThroughput = it }
            endReason = reason.name
            sessionEndTimestampNs = endTimestampNs
            sessionEnded = true
            appendEvent(
                JSONObject()
                    .put("event", "session_end")
                    .put("run_id", runId)
                    .put("camera_timestamp_ns", endTimestampNs ?: JSONObject.NULL)
                    .put("wall_clock_epoch_ms", System.currentTimeMillis())
                    .put("end_reason", reason.name)
                    .put("sample_count", samples)
                    .put("primary_detector_sample_count", convgruSamples)
                    .put("convgru_sample_count", convgruSamples)
                    .put("local_sample_count", localSamples)
                    .put("model_error", hasModelError)
                    .put("throughput", throughputJson(lastThroughput))
            )
        }

        fun recordSchedule(
            timestampNs: Long,
            outcome: TemporalScheduleOutcome,
            ruler: RulerSnapshot,
            geometryAgeNs: Long,
            generation: Long,
            loadLevel: AnalysisLoadLevel,
            thermal: DeviceThermalSnapshot?,
            yoloBusy: Boolean,
            temporalBusy: Boolean,
            cooldownRemainingNs: Long
        ) = submit {
            if (lastCameraTimestampNs > 0L && timestampNs > lastCameraTimestampNs) {
                cameraIntervalsMs += (timestampNs - lastCameraTimestampNs) / 1_000_000f
            }
            lastCameraTimestampNs = timestampNs
            if (outcome != TemporalScheduleOutcome.SUBMITTED) {
                skipCounts[outcome] = (skipCounts[outcome] ?: 0) + 1
            }
            appendEvent(
                JSONObject()
                    .put("event", "schedule")
                    .put("camera_timestamp_ns", timestampNs)
                    .put("wall_clock_epoch_ms", System.currentTimeMillis())
                    .put("outcome", outcome.name)
                    .put("tube_available", outcome != TemporalScheduleOutcome.NO_TUBE)
                    .put("ruler_state", ruler.state.name)
                    .put("ruler_usable", ruler.isUsable)
                    .put("geometry_age_ms", geometryAgeNs.coerceAtLeast(0L) / 1_000_000f)
                    .put("temporal_generation", generation)
                    .put("load_level", loadLevel.name)
                    .put("thermal_signal", thermal?.signal?.name ?: JSONObject.NULL)
                    .put("battery_temperature_c", thermal?.batteryTemperatureC ?: JSONObject.NULL)
                    .put("yolo_busy", yoloBusy)
                    .put("temporal_busy", temporalBusy)
                    .put("cooldown_remaining_ms", cooldownRemainingNs / 1_000_000f)
            )
        }

        fun recordAnalyzerTiming(timestampNs: Long, timings: TinyPipelineStageTimings) = submit {
            accumulatePipelineTimings(timings, setOf(TinyPipelinePhase.CAMERA))
            appendEvent(
                JSONObject()
                    .put("event", "analyzer_timing")
                    .put("camera_timestamp_ns", timestampNs)
                    .put("pipeline_stages_ms", pipelineTimingsJson(timings))
            )
        }

        fun recordFinalizationTimings(runId: String, timings: TinyPipelineStageTimings) = submit {
            accumulatePipelineTimings(timings, setOf(TinyPipelinePhase.FINALIZATION))
            appendEvent(
                JSONObject()
                    .put("event", "finalization_timing")
                    .put("run_id", runId)
                    .put("pipeline_stages_ms", pipelineTimingsJson(timings))
            )
        }

        fun recordRecordBuildTimings(runId: String, timings: TinyPipelineStageTimings) = submit {
            accumulatePipelineTimings(timings, setOf(TinyPipelinePhase.RECORD_BUILD))
            appendEvent(
                JSONObject()
                    .put("event", "record_build_timing")
                    .put("run_id", runId)
                    .put("pipeline_stages_ms", pipelineTimingsJson(timings))
            )
        }

        fun recordReset(
            reason: TemporalResetReason,
            timestampNs: Long?,
            previousGeneration: Long,
            nextGeneration: Long
        ) = submit {
            resetCount += 1
            appendEvent(
                JSONObject()
                    .put("event", "hidden_reset")
                    .put("camera_timestamp_ns", timestampNs ?: JSONObject.NULL)
                    .put("wall_clock_epoch_ms", System.currentTimeMillis())
                    .put("reason", reason.name)
                    .put("previous_generation", previousGeneration)
                    .put("next_generation", nextGeneration)
            )
        }

        fun captureYPlaneOnce(image: AnalysisFrame, transform: BufferTransform, timestampNs: Long) {
            if (!capturePayloads || !accepting.get() ||
                !yPlaneCaptured.compareAndSet(false, true)
            ) return
            val captureStartedNs = System.nanoTime()
            val plane = image.planes[0]
            val source = plane.buffer.duplicate()
            val originalPosition = source.position()
            val originalLimit = source.limit()
            val capacity = source.capacity()
            source.clear()
            val bytes = ByteArray(capacity)
            source.get(bytes)
            val captureCostMs = (System.nanoTime() - captureStartedNs) / 1_000_000f
            val event = JSONObject()
                .put("event", "y_plane")
                .put("camera_timestamp_ns", timestampNs)
                .put("wall_clock_epoch_ms", System.currentTimeMillis())
                .put("image_width", image.width)
                .put("image_height", image.height)
                .put("rotation_degrees", image.imageInfo.rotationDegrees)
                .put("buffer_position", originalPosition)
                .put("buffer_limit", originalLimit)
                .put("buffer_capacity", capacity)
                .put("row_stride", plane.rowStride)
                .put("pixel_stride", plane.pixelStride)
                .put("capture_cost_ms", captureCostMs)
                .put("transform", transformJson(transform))
            submit {
                File(directory, Y_PLANE_FILE).writeBytes(bytes)
                yPlaneByteCount = bytes.size
                captureCostsMs += captureCostMs
                appendEvent(event)
            }
        }

        fun recordInput(step: BufferedStep, preRoll: Boolean): Long {
            val stepId = nextStepId.getAndIncrement()
            if (preRoll) preRollStepIds.add(stepId)
            submit {
                val expectedBytes = TubeRoiSampler.OUTPUT_WIDTH * TubeRoiSampler.OUTPUT_HEIGHT
                val modelIntervalMs = if (lastModelTimestampNs > 0L &&
                    step.timestampNs > lastModelTimestampNs
                ) {
                    (step.timestampNs - lastModelTimestampNs) / 1_000_000f
                } else {
                    null
                }
                val fitsSession = TrackingDiagnosticRoiBudget.canAppend(
                    roiBytesWritten,
                    expectedBytes * 2L,
                    roiBudgetBytes
                )
                val captureRoi = capturePayloads && fitsSession &&
                    (globalRoiQuota?.tryClaim(expectedBytes * 2L) ?: true)
                val roiOffset = if (captureRoi) roiInputCount.toLong() * expectedBytes else -1L
                if (captureRoi) {
                    rawOutput.write(step.raw)
                    stableOutput.write(step.stable)
                    roiBytesWritten += expectedBytes * 2L
                    roiInputCount += 1
                } else if (capturePayloads) {
                    roiTruncated = true
                }
                if (!preRoll && modelIntervalMs != null) modelIntervalsMs += modelIntervalMs
                lastModelTimestampNs = step.timestampNs
                inputCount += 1
                if (!preRoll) {
                    sessionInputCount += 1
                    captureCostsMs += step.captureCostMs
                }
                appendEvent(
                    JSONObject()
                        .put("event", "input")
                        .put("step_id", stepId)
                        .put("step_token", step.token)
                        .put("pre_roll", preRoll)
                        .put("camera_timestamp_ns", step.timestampNs)
                        .put("geometry_capture_timestamp_ns", step.geometryCaptureTimestampNs)
                        .put("geometry_revision", step.geometryRevision)
                        .put("temporal_generation", step.temporalGeneration)
                        .put("model_interval_ms", modelIntervalMs ?: JSONObject.NULL)
                        .put("geometry_age_ns", (step.timestampNs - step.geometryCaptureTimestampNs).coerceAtLeast(0L))
                        .put("wall_clock_epoch_ms", System.currentTimeMillis())
                        .put("roi_captured", captureRoi)
                        .put("raw_offset_bytes", if (captureRoi) roiOffset else JSONObject.NULL)
                        .put("stable_offset_bytes", if (captureRoi) roiOffset else JSONObject.NULL)
                        .put("capture_cost_ms", step.captureCostMs)
                        .put("axis", axisJson(step.axis))
                        .put("ruler", rulerJson(step.ruler))
                        .put("transform", transformJson(step.transform))
                        .put("stabilization", stabilizationJson(step.stabilization))
                )
                step.detection?.let { recordDetectionNow(stepId, it) }
            }
            return stepId
        }

        fun recordDetection(stepId: Long, evidence: DetectionEvidence) = submit {
            recordDetectionNow(stepId, evidence)
        }

        private fun recordDetectionNow(stepId: Long, evidence: DetectionEvidence) {
            if (!detectionStepIds.add(stepId)) return
            lastThroughput = evidence.throughput
            if (stepId !in preRollStepIds) {
                maximumFramesSeen = maxOf(maximumFramesSeen, evidence.framesSeen)
                val diagnosticScore = evidence.classicalDiagnostics?.strongestCandidateScore
                    ?: evidence.detection?.score
                diagnosticScore?.let { score ->
                    scoreBands[TrackingScoreBand.from(score)] =
                        (scoreBands[TrackingScoreBand.from(score)] ?: 0) + 1
                    maximumScore = maximumScore?.let { maxOf(it, score) } ?: score
                }
                evidence.classicalDiagnostics?.rejection?.let { rejection ->
                    classicalEvidenceCount += 1
                    if (evidence.detection != null) classicalOutputCount += 1
                    classicalRejections[rejection] = (classicalRejections[rejection] ?: 0) + 1
                }
                if (evidence.inferenceMs.isFinite() && evidence.inferenceMs >= 0f) {
                    inferenceTimesMs += evidence.inferenceMs
                }
                if (evidence.runtime.parentInferenceMs.isFinite() &&
                    evidence.runtime.parentInferenceMs >= 0f
                ) {
                    parentInferenceTimesMs += evidence.runtime.parentInferenceMs
                }
                if (evidence.runtime.preprocessMs.isFinite() && evidence.runtime.preprocessMs >= 0f) {
                    preprocessTimesMs += evidence.runtime.preprocessMs
                }
                if (evidence.runtime.postprocessMs.isFinite() &&
                    evidence.runtime.postprocessMs >= 0f
                ) {
                    postprocessTimesMs += evidence.runtime.postprocessMs
                }
                if (evidence.runtime.rerankerMs.isFinite() && evidence.runtime.rerankerMs >= 0f) {
                    rerankerTimesMs += evidence.runtime.rerankerMs
                }
                if (evidence.runtime.backendName != "unknown") {
                    backendNames += evidence.runtime.backendName
                }
                if (evidence.runtime.computeUnit != "unknown") {
                    computeUnits += evidence.runtime.computeUnit
                }
                placementVerifiedStates += evidence.runtime.placementVerified
                if (evidence.runtime.placementSummary.isNotBlank()) {
                    placementSummaries += evidence.runtime.placementSummary
                }
                accumulatePipelineTimings(
                    evidence.pipelineTimings,
                    setOf(TinyPipelinePhase.WORKER, TinyPipelinePhase.END_TO_END)
                )
            }
            appendEvent(
                JSONObject()
                    .put("event", "detection")
                    .put("step_id", stepId)
                    .put("frames_seen", evidence.framesSeen)
                    .put("temporal_generation", evidence.generation)
                    .put("camera_timestamp_ns", evidence.cameraTimestampNs)
                    .put("inference_ms", evidence.inferenceMs)
                    .put("queue_ms", evidence.queueMs)
                    .put("queued_at_elapsed_realtime_ns", evidence.queuedAtNs)
                    .put("started_at_elapsed_realtime_ns", evidence.startedAtNs)
                    .put("completed_at_elapsed_realtime_ns", evidence.completedAtNs)
                    .put("backend", evidence.runtime.backendName)
                    .put("compute_unit", evidence.runtime.computeUnit)
                    .put("placement_verified", evidence.runtime.placementVerified)
                    .put("placement_summary", evidence.runtime.placementSummary)
                    .put("preprocess_ms", evidence.runtime.preprocessMs)
                    .put("parent_inference_ms", evidence.runtime.parentInferenceMs)
                    .put("postprocess_ms", evidence.runtime.postprocessMs)
                    .put("reranker_ms", evidence.runtime.rerankerMs)
                    .put("parent_latency", latencyStatsJson(evidence.runtime.parentLatencyStats))
                    .put("throughput", throughputJson(evidence.throughput))
                    .put("pipeline_stages_ms", pipelineTimingsJson(evidence.pipelineTimings))
                    .put(
                        "end_to_end_ms",
                        ((evidence.completedAtNs - evidence.queuedAtNs).coerceAtLeast(0L)) /
                            1_000_000f
                    )
                    .put("detection", detectionJson(evidence.detection))
                    .put("classical", classicalDiagnosticsJson(evidence.classicalDiagnostics))
            )
            detectionCount += 1
        }

        fun recordInferenceError(
            stepId: Long,
            generation: Long,
            queuedAtNs: Long,
            startedAtNs: Long,
            completedAtNs: Long,
            message: String
        ) = submit {
            if (stepId !in preRollStepIds) modelError = true
            modelErrorEventCount += 1
            appendEvent(
                JSONObject()
                    .put("event", "model_error")
                    .put("step_id", stepId)
                    .put("temporal_generation", generation)
                    .put("queued_at_elapsed_realtime_ns", queuedAtNs)
                    .put("started_at_elapsed_realtime_ns", startedAtNs)
                    .put("completed_at_elapsed_realtime_ns", completedAtNs)
                    .put("error", message.take(512))
            )
        }

        fun recordSelection(
            stepId: Long,
            primary: MappedTemporalPeak,
            alternatives: List<MappedTemporalPeak>,
            selected: MappedTemporalPeak?,
            threshold: Float,
            gate: FallTrackingStatus,
            resultAgeMs: Float,
            diagnostics: TemporalSelectionDiagnostics?
        ) {
            if (!selectedStepIds.add(stepId)) return
            submit {
                appendEvent(
                    JSONObject()
                        .put("event", "selection")
                        .put("step_id", stepId)
                        .put("primary", mappedPeakJson(primary))
                        .put("alternatives", JSONArray().also { values ->
                            alternatives.forEach { values.put(mappedPeakJson(it)) }
                        })
                        .put("selected", selected?.let(::mappedPeakJson) ?: JSONObject.NULL)
                        .put("threshold", threshold)
                        .put("result_age_ms", resultAgeMs)
                        .put("gate", gateJson(gate))
                        .put("diagnostics", selectionDiagnosticsJson(diagnostics))
                )
                selectionEventCount += 1
                diagnostics?.let {
                    if (stepId !in preRollStepIds) {
                        selectorOutcomes[it.outcome] = (selectorOutcomes[it.outcome] ?: 0) + 1
                    }
                }
                if (selected != null && stepId !in preRollStepIds) selectedCount += 1
            }
        }

        fun recordGate(
            stepId: Long?,
            observation: FallFrameObservation?,
            before: FallTrackingStatus,
            update: FallTrackingUpdate,
            after: FallTrackingStatus
        ) = submit {
            appendEvent(
                JSONObject()
                    .put("event", "gate")
                    .put("step_id", stepId ?: JSONObject.NULL)
                    .put("observation", observationJson(observation))
                    .put("before", gateJson(before))
                    .put("accepted_count", update.acceptedObservations.size)
                    .put("end_reason", update.endReason?.name ?: JSONObject.NULL)
                    .put("after", gateJson(after))
            )
            gateCount += 1
            if (stepId == null || stepId !in preRollStepIds) {
                gateRejections[after.lastRejection] =
                    (gateRejections[after.lastRejection] ?: 0) + 1
                acceptedObservationCount += update.acceptedObservations.size
            }
        }

        fun recordDense(
            timestampNs: Long,
            outputEnabled: Boolean,
            motionSeedAvailable: Boolean = outputEnabled,
            diagnostics: DenseTrackerDiagnostics,
            proposal: DenseTrackedBall?,
            accepted: Boolean
        ) = submit {
            appendEvent(
                JSONObject()
                    .put("event", "dense")
                    .put("camera_timestamp_ns", timestampNs)
                    .put("output_enabled", outputEnabled)
                    .put("motion_seed_available", motionSeedAvailable)
                    .put("background_ready_bands", diagnostics.backgroundReadyBands)
                    .put("candidate_count", diagnostics.candidateCount)
                    .put("prediction_available", diagnostics.predictionAvailable)
                    .put("plausible_candidate_count", diagnostics.plausibleCandidateCount)
                    .put("predicted_ticks", diagnostics.predictedTicks ?: JSONObject.NULL)
                    .put("closest_candidate_error_ticks", diagnostics.closestCandidateErrorTicks ?: JSONObject.NULL)
                    .put("closest_candidate_ticks", diagnostics.closestCandidateTicks ?: JSONObject.NULL)
                    .put("startup_track_count", diagnostics.startupTrackCount)
                    .put("ready_startup_track_count", diagnostics.readyStartupTrackCount)
                    .put("unanchored_acquisition_ready", diagnostics.unanchoredAcquisitionReady)
                    .put("proposal", denseProposalJson(proposal))
                    .put("accepted", accepted)
                    .put("decision", when {
                        proposal == null -> "NONE"
                        accepted -> "COMMIT"
                        else -> "DISCARD"
                    })
            )
            denseEventCount += 1
            if (outputEnabled) denseOutputEnabledFrames += 1
            if (proposal != null) denseProposalCount += 1
            if (accepted) denseAcceptedCount += 1
        }

        private fun submit(block: () -> Unit): Boolean {
            if (!accepting.get()) return false
            return runCatching {
                executor.execute {
                    runCatching(block).onFailure {
                        droppedEvents.incrementAndGet()
                        Log.e(TAG, "Tracking diagnostic event failed", it)
                    }
                }
            }
                .onFailure { droppedEvents.incrementAndGet() }
                .isSuccess
        }

        fun runAfterPending(action: () -> Unit): Boolean = submit(action)

        private fun appendEvent(value: JSONObject) {
            events.append(value.toString()).append('\n')
        }

        override fun close() {
            if (!accepting.compareAndSet(true, false)) return
            executor.shutdown()
            if (!executor.awaitTermination(CLOSE_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                executor.shutdownNow()
                executor.awaitTermination(CLOSE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            }
            rawOutput.flush()
            stableOutput.flush()
            events.flush()
            rawOutput.close()
            stableOutput.close()
            events.close()
            val snapshot = TrackingDiagnosticSnapshot(
                modelAvailable = modelAvailable,
                modelError = modelError,
                modelInputCount = sessionInputCount,
                maximumFramesSeen = maximumFramesSeen,
                maximumScore = maximumScore,
                classicalEvidenceCount = classicalEvidenceCount,
                classicalOutputCount = classicalOutputCount,
                selectionCount = selectedCount,
                acceptedObservationCount = acceptedObservationCount,
                convgruSampleCount = convgruSampleCount,
                localSampleCount = localSampleCount,
                denseOutputEnabledFrames = denseOutputEnabledFrames,
                denseProposalCount = denseProposalCount,
                denseAcceptedCount = denseAcceptedCount,
                schedulerSkips = skipCounts,
                scoreBands = scoreBands,
                selectorOutcomes = selectorOutcomes,
                gateRejections = gateRejections,
                resetCount = resetCount,
                cameraIntervalsMs = cameraIntervalsMs,
                modelIntervalsMs = modelIntervalsMs,
                inferenceTimesMs = inferenceTimesMs,
                preprocessTimesMs = preprocessTimesMs,
                parentInferenceTimesMs = parentInferenceTimesMs,
                postprocessTimesMs = postprocessTimesMs,
                rerankerTimesMs = rerankerTimesMs,
                pipelineStageTimesMs = pipelineStageTimesMs.mapValues { it.value.toList() },
                backendNames = backendNames,
                computeUnits = computeUnits,
                placementVerifiedStates = placementVerifiedStates,
                placementSummaries = placementSummaries,
                tinyThroughput = lastThroughput,
                captureCostsMs = captureCostsMs,
                droppedEventCount = droppedEvents.get(),
                roiTruncated = roiTruncated
            )
            val classified = TrackingDiagnosticClassifier.classify(snapshot)
            writeJsonAtomically(File(directory, SUMMARY_FILE), summaryJson(snapshot, classified))
            val manifest = JSONObject()
                .put("format_version", FORMAT_VERSION)
                .put("state", "complete")
                .put("app_version_name", appVersionName)
                .put("app_version_code", appVersionCode)
                .put("run_id", activeRunId ?: JSONObject.NULL)
                .put("started_at_epoch_ms", startedAtEpochMs)
                .put("finished_at_epoch_ms", System.currentTimeMillis())
                .put("termination_status", when {
                    activeRunId == null -> "capture_only"
                    sessionEnded -> "complete"
                    else -> "abnormal"
                })
                .put("session_start_timestamp_ns", sessionStartTimestampNs ?: JSONObject.NULL)
                .put("session_end_timestamp_ns", sessionEndTimestampNs ?: JSONObject.NULL)
                .put("session_resolution", sessionStartResolution ?: JSONObject.NULL)
                .put("session_tick_spacing_mm", sessionTickSpacingMm ?: JSONObject.NULL)
                .put("session_expected_tick_count", sessionExpectedTickCount ?: JSONObject.NULL)
                .put("session_temporal_generation", sessionGeneration ?: JSONObject.NULL)
                .put("detector", MODEL_FILE)
                .put("sequence_length", 1)
                .put("width", TubeRoiSampler.OUTPUT_WIDTH)
                .put("height", TubeRoiSampler.OUTPUT_HEIGHT)
                .put("input_count", inputCount)
                .put("session_input_count", sessionInputCount)
                .put("pre_roll_input_count", preRollStepIds.size)
                .put("roi_input_count", roiInputCount)
                .put("detection_count", detectionCount)
                .put("classical_output_count", classicalOutputCount)
                .put("selection_count", selectionEventCount)
                .put("gate_count", gateCount)
                .put("dense_count", denseEventCount)
                .put("model_error_event_count", modelErrorEventCount)
                .put("backend_names", JSONArray(backendNames.toList()))
                .put("compute_units", JSONArray(computeUnits.toList()))
                .put("placement_verified_states", JSONArray(placementVerifiedStates.toList()))
                .put("placement_summaries", JSONArray(placementSummaries.toList()))
                .put("tiny_throughput", throughputJson(lastThroughput))
                .put("reset_count", resetCount)
                .put("sample_count", sampleCount)
                .put("end_reason", endReason ?: JSONObject.NULL)
                .put("roi_truncated", roiTruncated)
                .put("capture_payloads", capturePayloads)
                .put("raw_roi_byte_count", File(directory, RAW_FILE).length())
                .put("stable_roi_byte_count", File(directory, STABLE_FILE).length())
                .put("y_plane_byte_count", yPlaneByteCount)
                .put("raw_rois_sha256", sha256(File(directory, RAW_FILE)))
                .put("stable_rois_sha256", sha256(File(directory, STABLE_FILE)))
                .put("events_sha256", sha256(File(directory, EVENTS_FILE)))
                .put("summary_sha256", sha256(File(directory, SUMMARY_FILE)))
            File(directory, Y_PLANE_FILE).takeIf(File::isFile)?.let {
                manifest.put("y_plane_sha256", sha256(it))
            }
            writeJsonAtomically(File(directory, MANIFEST_FILE), manifest)
            writeStatus("complete")
        }

        private fun summaryJson(
            snapshot: TrackingDiagnosticSnapshot,
            summary: TrackingDiagnosticSummary
        ) = JSONObject()
            .put("format_version", FORMAT_VERSION)
            .put("run_id", activeRunId ?: JSONObject.NULL)
            .put("primary_outcome", summary.primary.name)
            .put("contributing_reasons", JSONArray(summary.contributingReasons))
            .put("model_error_event_count", modelErrorEventCount)
            .put("model_input_count", snapshot.modelInputCount)
            .put("maximum_frames_seen", snapshot.maximumFramesSeen)
            .put("maximum_score", snapshot.maximumScore ?: JSONObject.NULL)
            .put("classical_evidence_count", snapshot.classicalEvidenceCount)
            .put("classical_output_count", snapshot.classicalOutputCount)
            .put("selected_count", selectedCount)
            .put("accepted_observation_count", snapshot.acceptedObservationCount)
            .put("sample_count", sampleCount)
            .put("primary_detector_sample_count", snapshot.convgruSampleCount)
            .put("convgru_sample_count", snapshot.convgruSampleCount)
            .put("local_sample_count", snapshot.localSampleCount)
            .put("dense_output_enabled_frames", snapshot.denseOutputEnabledFrames)
            .put("dense_proposal_count", snapshot.denseProposalCount)
            .put("dense_accepted_count", snapshot.denseAcceptedCount)
            .put("scheduler_skips", JSONObject().also { value ->
                TemporalScheduleOutcome.entries.forEach { outcome ->
                    if (outcome != TemporalScheduleOutcome.SUBMITTED) {
                        value.put(outcome.name, snapshot.schedulerSkips[outcome] ?: 0)
                    }
                }
            })
            .put("hidden_state_reset_count", snapshot.resetCount)
            .put("camera_cadence_ms", distributionJson(summary.cameraCadence))
            .put("model_cadence_ms", distributionJson(summary.modelCadence))
            .put("inference_latency_ms", distributionJson(summary.inferenceLatency))
            .put("preprocess_latency_ms", distributionJson(summary.preprocessLatency))
            .put("parent_inference_latency_ms", distributionJson(summary.parentInferenceLatency))
            .put("postprocess_latency_ms", distributionJson(summary.postprocessLatency))
            .put("reranker_latency_ms", distributionJson(summary.rerankerLatency))
            .put("pipeline_stage_latency_ms", JSONObject().also { stages ->
                TinyPipelineStage.entries.forEach { stage ->
                    summary.pipelineStageLatency[stage]?.let { distribution ->
                        stages.put(stage.jsonKey, distributionJson(distribution))
                    }
                }
            })
            .put("pipeline_leaf_p50_total_ms", JSONObject().also { phases ->
                TinyPipelinePhase.entries.forEach { phase ->
                    phases.put(
                        phase.name.lowercase(),
                        summary.pipelineLeafP50TotalMs[phase] ?: 0f
                    )
                }
            })
            .put(
                "pipeline_hotspots",
                pipelineHotspotsJson(
                    summary,
                    setOf(TinyPipelinePhase.CAMERA, TinyPipelinePhase.WORKER)
                )
            )
            .put(
                "finalization_hotspots",
                pipelineHotspotsJson(summary, setOf(TinyPipelinePhase.FINALIZATION))
            )
            .put(
                "record_build_hotspots",
                pipelineHotspotsJson(summary, setOf(TinyPipelinePhase.RECORD_BUILD))
            )
            .put("backend_names", JSONArray(snapshot.backendNames.toList()))
            .put("compute_units", JSONArray(snapshot.computeUnits.toList()))
            .put("placement_verified_states", JSONArray(snapshot.placementVerifiedStates.toList()))
            .put("placement_summaries", JSONArray(snapshot.placementSummaries.toList()))
            .put("tiny_throughput", throughputJson(snapshot.tinyThroughput))
            .put("capture_cost_ms", distributionJson(summary.captureCost))
            .put("dropped_event_count", snapshot.droppedEventCount)
            .put("roi_truncated", snapshot.roiTruncated)
            .put("capture_payloads", capturePayloads)
            .put("score_bands", JSONObject().also { value ->
                TrackingScoreBand.entries.forEach { band ->
                    value.put(band.name, snapshot.scoreBands[band] ?: 0)
                }
            })
            .put("selector_outcomes", JSONObject().also { value ->
                TemporalSelectionOutcome.entries.forEach { outcome ->
                    value.put(outcome.name, snapshot.selectorOutcomes[outcome] ?: 0)
                }
            })
            .put("gate_rejections", JSONObject().also { value ->
                com.cupet.fallingball.analysis.FallTrackingRejection.entries.forEach { rejection ->
                    value.put(rejection.name, snapshot.gateRejections[rejection] ?: 0)
                }
            })
            .put("classical_rejections", JSONObject().also { value ->
                ClassicalHeatmapRejection.entries.forEach { rejection ->
                    value.put(rejection.name, classicalRejections[rejection] ?: 0)
                }
            })

        private fun writeStatus(state: String) {
            writeJsonAtomically(
                File(directory, STATUS_FILE),
                JSONObject()
                    .put("format_version", FORMAT_VERSION)
                    .put("state", state)
                    .put("updated_at_epoch_ms", System.currentTimeMillis())
            )
        }

        private fun writeJsonAtomically(target: File, value: JSONObject) {
            val temporary = File(directory, "${target.name}.tmp")
            temporary.writeText(value.toString(2), Charsets.UTF_8)
            require(!target.exists() || target.delete()) { "Cannot replace ${target.name}" }
            require(temporary.renameTo(target)) { "Cannot publish ${target.name}" }
        }

        private fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { stream ->
                val buffer = ByteArray(1024 * 1024)
                while (true) {
                    val count = stream.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        private fun axisJson(axis: TubeAxis) = JSONObject()
            .put("ox", axis.ox).put("oy", axis.oy)
            .put("ux", axis.ux).put("uy", axis.uy)
            .put("vx", axis.vx).put("vy", axis.vy)
            .put("length_px", axis.lengthPx).put("width_px", axis.widthPx)
            .put("tilt_rad", axis.tiltRad)

        private fun rulerJson(ruler: RulerSnapshot) = JSONObject()
            .put("state", ruler.state.name)
            .put("usable", ruler.isUsable)
            .put("count", ruler.count)
            .put("median_gap_px", ruler.medianGapPx)
            .put("shift_px", ruler.shiftPx)
            .put("invalid_reason", ruler.invalidReason?.name ?: JSONObject.NULL)

        private fun transformJson(transform: BufferTransform) = JSONObject()
            .put("crop_left", transform.cropLeft).put("crop_top", transform.cropTop)
            .put("crop_width", transform.cropW).put("crop_height", transform.cropH)
            .put("display_width", transform.displayW).put("display_height", transform.displayH)

        private fun stabilizationJson(value: RoiStabilizationTransform) = JSONObject()
            .put("x", value.sourceToStabilizedX)
            .put("y", value.sourceToStabilizedY)
            .put("score", value.score)

        private fun detectionJson(detection: TemporalBallDetection?): Any {
            if (detection == null) return JSONObject.NULL
            return JSONObject()
                .put("x", detection.roiX).put("y", detection.roiY)
                .put("score", detection.score).put("inference_ms", detection.inferenceMs)
                .put("runtime", runtimeJson(detection.runtime))
                .put("alternatives", JSONArray().also { values ->
                    detection.alternativePeaks.forEach { values.put(peakJson(it)) }
                })
        }

        private fun runtimeJson(value: TinyHeatmapRuntimeDiagnostics) = JSONObject()
            .put("backend", value.backendName)
            .put("compute_unit", value.computeUnit)
            .put("placement_verified", value.placementVerified)
            .put("placement_summary", value.placementSummary)
            .put("verified_gpu", value.verifiedGpu)
            .put("preprocess_ms", value.preprocessMs)
            .put("parent_inference_ms", value.parentInferenceMs)
            .put("postprocess_ms", value.postprocessMs)
            .put("reranker_ms", value.rerankerMs)
            .put("total_ms", value.totalMs)
            .put("parent_latency", latencyStatsJson(value.parentLatencyStats))

        private fun classicalDiagnosticsJson(value: ClassicalHeatmapDiagnostics?): Any {
            if (value == null) return JSONObject.NULL
            return JSONObject()
                .put("raw_peak_score", value.rawPeakScore)
                .put("strongest_candidate_score", value.strongestCandidateScore ?: JSONObject.NULL)
                .put("tiny_peak_score", value.tinyPeakScore ?: JSONObject.NULL)
                .put("recovery_peak_score", value.recoveryPeakScore ?: JSONObject.NULL)
                .put("normal_candidate_count", value.normalCandidateCount)
                .put("tiny_candidate_count", value.tinyCandidateCount)
                .put("normal_startup_samples", value.normalStartupSamples)
                .put("tiny_startup_samples", value.tinyStartupSamples)
                .put("phase", value.phase.name)
                .put("rejection", value.rejection.name)
                .put("tracking_from_tiny", value.trackingFromTiny)
        }

        private fun peakJson(peak: TemporalBallPeak) = JSONObject()
            .put("x", peak.roiX).put("y", peak.roiY).put("score", peak.score)

        private fun mappedPeakJson(peak: MappedTemporalPeak) = JSONObject()
            .put("x", peak.peak.roiX).put("y", peak.peak.roiY)
            .put("score", peak.peak.score).put("axial_px", peak.axialPx)
            .put("axial_max_px", peak.axialMaxPx)
            .put("position_ticks", peak.positionTicks ?: JSONObject.NULL)
            .put("lateral_fraction", peak.trajectoryLateralFraction ?: JSONObject.NULL)

        private fun gateJson(gate: FallTrackingStatus) = JSONObject()
            .put("phase", gate.phase.name).put("pending", gate.pendingObservations)
            .put("allowed_gap_ms", gate.allowedGapMs)
            .put("loss_timeout_ms", gate.lossTimeoutMs)
            .put("rejection", gate.lastRejection.name)
            .put("motion_seeded", gate.motionSeeded)

        private fun observationJson(observation: FallFrameObservation?): Any {
            if (observation == null) return JSONObject.NULL
            return JSONObject()
                .put("timestamp_ns", observation.timestampNs)
                .put("axial_px", observation.axialPx)
                .put("position_ticks", observation.positionTicks ?: JSONObject.NULL)
                .put("progress_fraction", observation.progressFraction)
                .put("lateral_fraction", observation.trajectoryLateralFraction ?: JSONObject.NULL)
                .put("ruler_valid", observation.rulerValid)
                .put("source", observation.source.name)
                .put("confidence", observation.detectionConfidence ?: JSONObject.NULL)
        }

        private fun selectionDiagnosticsJson(value: TemporalSelectionDiagnostics?): Any {
            if (value == null) return JSONObject.NULL
            return JSONObject()
                .put("candidate_count", value.candidateCount)
                .put("valid_candidate_count", value.validCandidateCount)
                .put("startup_track_count", value.startupTrackCount)
                .put("ready_startup_track_count", value.readyStartupTrackCount)
                .put("downward_motion_formed", value.readyStartupTrackCount > 0)
                .put("outcome", value.outcome.name)
        }

        private fun denseProposalJson(value: DenseTrackedBall?): Any {
            if (value == null) return JSONObject.NULL
            return JSONObject()
                .put("timestamp_ns", value.timestampNs).put("axial_px", value.axialPx)
                .put("position_ticks", value.positionTicks).put("lateral_px", value.lateralPx)
                .put("lateral_fraction", value.lateralFraction).put("confidence", value.confidence)
        }

        private fun latencyStatsJson(value: com.cupet.fallingball.vision.TinyHeatmapLatencyStats) =
            JSONObject()
                .put("sample_count", value.sampleCount)
                .put("p50_ms", value.p50Ms)
                .put("p95_ms", value.p95Ms)
                .put("last_ms", value.lastMs)
                .put("max_ms", value.maximumMs)

        private fun throughputJson(value: TinyHeatmapThroughputStats) = JSONObject()
            .put("submitted", value.submittedCount)
            .put("completed", value.completedCount)
            .put("dropped", value.droppedCount)
            .put("failed", value.failedCount)
            .put("in_flight", value.inFlightCount)
            .put("effective_fps", value.effectiveFps)
            .put("elapsed_ms", value.elapsedMs)
            .put(
                "drop_ratio",
                if (value.submittedCount > 0) {
                    value.droppedCount.toDouble() / value.submittedCount
                } else {
                    0.0
                }
            )

        private fun accumulatePipelineTimings(
            timings: TinyPipelineStageTimings,
            phases: Set<TinyPipelinePhase>
        ) {
            timings.forEachMeasured { stage, milliseconds ->
                if (stage.phase in phases && milliseconds.isFinite() && milliseconds >= 0f) {
                    pipelineStageTimesMs.getOrPut(stage, ::ArrayList).add(milliseconds)
                }
            }
        }

        private fun pipelineTimingsJson(value: TinyPipelineStageTimings) = JSONObject().also {
            value.forEachMeasured { stage, milliseconds -> it.put(stage.jsonKey, milliseconds) }
        }

        private fun pipelineHotspotsJson(
            summary: TrackingDiagnosticSummary,
            phases: Set<TinyPipelinePhase>
        ) = JSONArray().also { hotspots ->
            summary.pipelineStageLatency.entries
                .asSequence()
                .filter { (stage, distribution) ->
                    stage.phase in phases && !stage.aggregate && distribution.count > 0
                }
                .sortedByDescending { it.value.p50Ms }
                .forEach { (stage, distribution) ->
                    hotspots.put(
                        JSONObject()
                            .put("stage", stage.jsonKey)
                            .put("phase", stage.phase.name)
                            .put("count", distribution.count)
                            .put("p50_ms", distribution.p50Ms)
                            .put("p95_ms", distribution.p95Ms)
                            .put("max_ms", distribution.maximumMs)
                    )
                }
        }

        private fun distributionJson(value: DiagnosticDistribution) = JSONObject()
            .put("count", value.count).put("p50", value.p50Ms)
            .put("p95", value.p95Ms).put("max", value.maximumMs)
    }

    private fun copyRemaining(buffer: ByteBuffer): ByteArray {
        val source = buffer.duplicate()
        return ByteArray(source.remaining()).also(source::get)
    }

    private companion object {
        private const val TAG = "TemporalDiagnostics"
        private const val EXTERNAL_OUTPUT_DIR = "debug_temporal_capture"
        private const val TRACKING_OUTPUT_DIR = "diagnostics/tracking_sessions"
        private const val FORMAT_VERSION = 5
        private const val CLOSE_TIMEOUT_SECONDS = 10L
        private const val MODEL_FILE = "classical_heatmap_c"
        private const val CLASSICAL_WARMUP_FRAMES = 8
        private val EMPTY_BYTES = ByteArray(0)
        private const val CLASSICAL_PRESENCE_THRESHOLD = 0.18f
        private const val CLASSICAL_RECOVERY_THRESHOLD = 0.12f
        private const val RAW_FILE = "raw_rois.bin"
        private const val STABLE_FILE = "stable_rois.bin"
        private const val Y_PLANE_FILE = "y_plane_first.bin"
        private const val EVENTS_FILE = "events.jsonl"
        private const val SUMMARY_FILE = "summary.json"
        private const val MANIFEST_FILE = "manifest.json"
        private const val STATUS_FILE = "status.json"
        private val OUTPUT_FILES = listOf(
            RAW_FILE, STABLE_FILE, Y_PLANE_FILE, EVENTS_FILE,
            SUMMARY_FILE, MANIFEST_FILE, STATUS_FILE
        )
    }
}
