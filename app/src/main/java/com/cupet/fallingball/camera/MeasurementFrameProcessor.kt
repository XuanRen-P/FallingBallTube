package com.cupet.fallingball.camera

import android.content.Context
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.util.Log
import android.util.Range
import android.util.Size
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.CameraState
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import com.cupet.fallingball.vision.AnalysisFrame
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.Observer
import androidx.lifecycle.Lifecycle
import com.cupet.fallingball.analysis.BallSample
import com.cupet.fallingball.analysis.FallFrameObservation
import com.cupet.fallingball.analysis.FallSession
import com.cupet.fallingball.analysis.FallTrackingGate
import com.cupet.fallingball.analysis.VelocityEstimator
import com.cupet.fallingball.ui.AnalysisResolution
import com.cupet.fallingball.vision.AxialSampler
import com.cupet.fallingball.vision.BallEndReason
import com.cupet.fallingball.vision.BufferTransform
import com.cupet.fallingball.vision.DetectionRoi
import com.cupet.fallingball.vision.InputRegion
import com.cupet.fallingball.vision.ModelProfile
import com.cupet.fallingball.vision.ObbDetector
import com.cupet.fallingball.vision.RegisteredBallResult
import com.cupet.fallingball.vision.RulerConfig
import com.cupet.fallingball.vision.RulerSnapshot
import com.cupet.fallingball.vision.RulerState
import com.cupet.fallingball.vision.ScaleReading
import com.cupet.fallingball.vision.TickDetector
import com.cupet.fallingball.vision.TickRuler
import com.cupet.fallingball.vision.TemporalBallDetection
import com.cupet.fallingball.vision.TemporalBallPeak
import com.cupet.fallingball.vision.TemporalRoiSequenceBuffer
import com.cupet.fallingball.vision.TrackMode
import com.cupet.fallingball.vision.TubeRoiSampler
import com.cupet.fallingball.vision.TubeRoiGeometry
import com.cupet.fallingball.vision.BallGeometry
import com.cupet.fallingball.vision.CausalRoiStabilizer
import com.cupet.fallingball.vision.TinyHeatmapNetBallDetector
import com.cupet.fallingball.vision.NcnnTinyHeatmapNet
import com.cupet.fallingball.vision.TinyHeatmapFrameCandidate
import com.cupet.fallingball.vision.TinyHeatmapHistoryFrame
import com.cupet.fallingball.vision.TinyHeatmapInputHistory
import com.cupet.fallingball.vision.TinyHeatmapSequenceCandidate
import com.cupet.fallingball.vision.TinyHeatmapSequenceDecoder
import com.cupet.fallingball.vision.TinyHeatmapSequenceFrame
import com.cupet.fallingball.vision.TinyHeatmapRuntimeDiagnostics
import com.cupet.fallingball.vision.TinyHeatmapThroughputStats
import com.cupet.fallingball.vision.TinyHeatmapThroughputTracker
import com.cupet.fallingball.vision.TinyHeatmapBenchmarkResult
import com.cupet.fallingball.vision.TinyPipelineStage
import com.cupet.fallingball.vision.TinyPipelineStageTimer
import com.cupet.fallingball.vision.TinyPipelineStageTimings
import com.cupet.fallingball.vision.DenseBallAnchor
import com.cupet.fallingball.vision.DenseRegisteredBallTracker
import com.cupet.fallingball.vision.RoiStabilizationTransform
import com.cupet.fallingball.vision.TubeBandSampler
import com.cupet.fallingball.vision.TubeAxis
import com.cupet.fallingball.vision.TubeFocusLockGate
import com.cupet.fallingball.vision.TubeTrackResult
import com.cupet.fallingball.vision.TubeTracker
import com.cupet.fallingball.vision.YuvTensorPreprocessor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.roundToInt

/** Model-frame numbering follows received camera captures; capture timestamps remain separate. */
internal fun nextTinyModelFrameIndex(previousIndex: Int): Int =
    (previousIndex + 1).coerceAtLeast(0)

/**
 * Camera-frame numbering for the whole-sequence decoder.
 *
 * [TinyHeatmapSequenceDecoder] divides candidate motion by the number of *camera* frames
 * between two model inputs, matching the PC dataset's `source_frame_index`. Latest-only
 * scheduling drops camera frames, so this cannot reuse the contiguous model-frame index:
 * that would report every gap as one frame and inflate decoded velocity by the current
 * drop ratio. Model-frame numbering stays contiguous for the lag-1/lag-3 history lookup,
 * which matches consecutive stored frames rather than true elapsed time.
 */
internal fun nextTinyCaptureFrameIndex(
    previousIndex: Int,
    previousTimestampNs: Long,
    timestampNs: Long,
    framePeriodNs: Long
): Int {
    require(framePeriodNs > 0L) { "A camera frame period must be positive" }
    if (previousTimestampNs <= 0L || timestampNs <= previousTimestampNs) {
        return (previousIndex + 1).coerceAtLeast(0)
    }
    val elapsedFrames = ((timestampNs - previousTimestampNs).toDouble() / framePeriodNs)
        .roundToInt()
        .coerceAtLeast(1)
    return previousIndex + elapsedFrames
}

/**
 * @param detector shared with the activity and owned by the caller — [shutdown] does not close it.
 */
open class MeasurementFrameProcessor(
    private val context: Context,
    private val cameraLifecycleOwner: LifecycleOwner? = null,
    private val cameraPreviewView: PreviewView? = null,
    detector: ObbDetector,
    private val onResult: (TubeTrackResult) -> Unit = {},
    private val onStabilizationInfo: (Boolean) -> Unit = {},
    private val onSessionRecordReady: (FallSession, TinyPipelineStageTimer) -> Unit = { _, _ -> },
    private val onSessionFinished: (FallSession) -> Unit = {},
    private val offline: Boolean = false
) {
    constructor(context: Context, detector: ObbDetector, offline: Boolean) :
        this(context, null, null, detector, offline = offline)

    private val previewView: PreviewView get() = requireNotNull(cameraPreviewView)
    private val lifecycleOwner: LifecycleOwner get() = requireNotNull(cameraLifecycleOwner)
    var cameraControls: CameraControlsController? = null
    private var tubeAutofocusEnabled = true

    fun setTubeAutofocusEnabled(enabled: Boolean) {
        tubeAutofocusEnabled = enabled
        releaseTubeFocus()
    }

    fun rebindCamera() {
        check(!isSessionActive)
        provider?.let(::bindUseCases)
    }
    private var offlinePrepared = false
    private var offlineResolutionLabel = ""
    private var offlineSourceIndex = -1
    private var offlineLastTimestampNs = Long.MIN_VALUE
    private var offlineSawRuler = false
    private var offlineSawTube = false
    private var offlineProcessedFrames = 0
    private var offlineInferenceFrames = 0
    private var offlineRunId = java.util.UUID.randomUUID().toString()
    /**
     * The tube is a static object in an imported experiment.  Running the 640px OBB graph on
     * every video frame needlessly serialises a second large ONNX inference with the ball model.
     * Keep a short refresh cadence so a moved/occluded tube is still reacquired, while reusing
     * the last tracked geometry between refreshes.  Ball inference remains every frame.
     */
    private var offlineObbRefreshPolicy = OfflineTubeRefreshPolicy()
    private val offlineRawRoi = ByteArray(TubeRoiSampler.OUTPUT_WIDTH * TubeRoiSampler.OUTPUT_HEIGHT)

    data class OfflineFrameStatus(
        val processedFrames: Int,
        val inferenceFrames: Int,
        val sawTube: Boolean,
        val sawRuler: Boolean,
        val ballInferenceCompleted: Boolean,
        val rulerValid: Boolean
    )

    /** Called once on the offline worker; its frame loop owns all mutable analysis state. */
    fun configureOffline(
        config: RulerConfig,
        defaults: com.cupet.fallingball.analysis.MeasurementDefaults,
        resolutionLabel: String,
        runId: String,
        /** Zero is available to device benchmarks comparing the previous every-frame OBB path. */
        obbRefreshIntervalNs: Long = 250_000_000L
    ) {
        check(offline && !offlinePrepared && !closed.get())
        rulerConfig = config
        tickRuler.updateConfig(config)
        sphereRadiusMm = defaults.viscometryParameters.sphereRadiusMm
        tubeInnerDiameterMm = defaults.viscometryParameters.tubeInnerDiameterMm
        offlineResolutionLabel = resolutionLabel
        offlineRunId = runId
        offlineObbRefreshPolicy = OfflineTubeRefreshPolicy(obbRefreshIntervalNs)
        desiredRunning.set(true)
        offlinePrepared = true
    }

    /** Takes ownership of the frame. No latest-only queues or wall-clock cooldowns apply. */
    fun processOfflineFrame(frame: AnalysisFrame): OfflineFrameStatus {
        try {
            check(offlinePrepared && !closed.get())
            require(frame.imageInfo.timestamp > offlineLastTimestampNs) {
                "Video frame timestamps must be strictly increasing"
            }
            require(frame.sourceFrameIndex > offlineSourceIndex) {
                "Video frame indices must be strictly increasing"
            }
            offlineLastTimestampNs = frame.imageInfo.timestamp
            offlineSourceIndex = frame.sourceFrameIndex
            val priorInferences = offlineInferenceFrames
            analyzeFrame(frame)
            offlineProcessedFrames++
            return OfflineFrameStatus(
                offlineProcessedFrames, offlineInferenceFrames, offlineSawTube, offlineSawRuler,
                offlineInferenceFrames > priorInferences, offlineSawRuler
            )
        } finally {
            frame.close()
        }
    }

    fun finishOffline(): FallSession {
        check(offlinePrepared && !closed.get())
        check(offlineSawTube) { "所选片段中未识别到试管" }
        check(offlineSawRuler) { "刻度未能稳定标定，请保留清晰的投球前画面" }
        val completed = checkNotNull(session) { "未能建立投球前的标定与背景" }
        if (completed.isActive) finishSession(BallEndReason.MANUAL)
        check(completed.validSampleCount > 0) { "未检测到有效的小球下落轨迹" }
        return completed
    }

    /** Offline cancellation never finalizes or saves a partial measurement. */
    fun closeOffline() {
        check(offline)
        if (!closed.compareAndSet(false, true)) return
        desiredRunning.set(false)
        session = null
        analysisExecutor.shutdown()
        obbExecutor.shutdown()
        temporalExecutor.shutdown()
        benchmarkExecutor.shutdown()
        try { temporalBallDetector.close() } finally {
            try { temporalDebugCaptureRecorder.close() } finally { diagnosticsRecorder.close() }
        }
    }
    companion object {
        private const val TAG = "CameraController"
        private const val PERFORMANCE_TAG = "FallingBallPerf"
        private const val TRACK_TRACE_TAG = "FallingBallTrack"
        private const val TINY_BENCHMARK_TAG = "TinyHeatmapBenchmark"
        private val EMPTY_FLAGS = BooleanArray(0)

        /**
         * Beyond this, geometry is too old to place the sampling band on a handheld shot:
         * the band is only a quarter of the bore wide, so a drift of a dozen pixels pushes
         * it onto the tube wall and the profile stops containing graduations.
         */
        private const val TARGET_ANALYSIS_FPS = 30
        private const val FRAME_PERIOD_NS = 1_000_000_000L / TARGET_ANALYSIS_FPS
        private const val CLASSICAL_DEFAULT_PERIOD_MS = 100L
        private const val CLASSICAL_TUBE_LOSS_RESET_NS = 2_000_000_000L

        private const val PERFORMANCE_LOG_INTERVAL_NS = 2_000_000_000L
        private val DURABLE_BALL_SOURCES = setOf(
            com.cupet.fallingball.vision.BallDetectionSource.CONVGRU,
            com.cupet.fallingball.vision.BallDetectionSource.LOCAL_TRACKER,
            com.cupet.fallingball.vision.BallDetectionSource.TINY_HEATMAP
        )

    }

    /** Applies new cylinder parameters; drops any existing tick lock. */
    fun setRulerConfig(config: RulerConfig) {
        if (closed.get()) return
        val changed = config != rulerConfig
        rulerConfig = config
        if (changed) {
            analysisExecutor.execute {
                tickRuler.updateConfig(config)
            }
        }
    }

    /** Arms ball tracking and starts a fresh recording. */
    fun startSession(): FallSession {
        check(!closed.get()) { "Camera controller is closed" }
        // ImageProxy timestamps are the capture clock used by every observation. Using
        // System.nanoTime() here can be off by the device's boot/suspend offset and silently
        // drops the first frames in FallSession.add(). Exclude the last frame already observed
        // by starting just after the latest capture timestamp; the next frame is the first one
        // eligible for this run.
        val latestCaptureNs = frameEpoch.latestCaptureTimestampNs
        check(latestCaptureNs > 0L) { "Camera has not produced a capture timestamp" }
        val s = FallSession(
            startNs = latestCaptureNs + 1L,
            config = rulerConfig,
            resolutionLabel = resolutionRef.get().label,
            diagnosticsEnabled = temporalDebugCaptureRecorder.diagnosticsEnabled
        )
        lightweightTrackingDiagnostics = LightweightTrackingDiagnosticAccumulator()
        check(
            temporalDebugCaptureRecorder.startTrackingSession(
                runId = s.runId,
                startTimestampNs = s.startNs,
                config = s.config,
                resolutionLabel = s.resolutionLabel,
                generation = temporalGeneration.get(),
                modelAvailable = temporalBallEnabled
            )
        ) { "Unable to create tracking diagnostics" }
        sessionRefOrdinal = null
        fallTrackingGate.reset()
        denseBallTracker.resetTrajectory()
        // The detector and its candidate history are owned by analysisExecutor. Wait for the
        // reset barrier before publishing the new session, otherwise an in-flight frame can
        // append to the new run after the UI thread clears the old ArrayList.
        analysisExecutor.submit { resetTinySequenceState() }
            .get(5L, TimeUnit.SECONDS)
        temporalBallEnabled = true
        temporalBallError = false
        temporalBallRef.set(null)
        session = s
        return s
    }

    /** Ends the current run at the user's request. */
    fun stopSession() {
        if (closed.get()) return
        runCatching {
            analysisExecutor.execute { finishSession(BallEndReason.MANUAL) }
        }.onFailure {
            Log.w(TAG, "Unable to queue session stop", it)
        }
    }

    val isSessionActive: Boolean get() = session?.isActive == true
    val isRunning: Boolean get() = desiredRunning.get()

    private val analysisExecutor = Executors.newSingleThreadExecutor()

    /**
     * Inference runs here, off the analysis thread, so the 1D scale path keeps full frame
     * rate while the model runs as often as the device can manage. See the handshake in
     * the analyzer.
     */
    private val obbExecutor = Executors.newSingleThreadExecutor()
    private val obbBusy = AtomicBoolean(false)
    private val detector = detector
    private val temporalExecutor = Executors.newSingleThreadExecutor()
    private val temporalFinalizationQueue = TrackingSessionFinalizationQueue(temporalExecutor)
    /** Debug-only model benchmarks must not queue behind the live temporal worker. */
    private val benchmarkExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "FallingBall-tiny-benchmark")
    }
    private val temporalBusy = AtomicBoolean(false)
    /** TinyHeatmapNet is causal for preview; the same candidate records are decoded at finish. */
    private val temporalBallDetector = TinyHeatmapNetBallDetector(context)
    private val tinyRoiStabilizer = CausalRoiStabilizer()
    @Volatile private var temporalBallEnabled = true
    @Volatile private var temporalBallError = false
    private val tubeRoiSampler = TubeRoiSampler()
    private val temporalSequenceBuffer = TemporalRoiSequenceBuffer()
    private val tubeRoiBuffer = ByteBuffer.allocateDirect(
        TubeRoiSampler.OUTPUT_WIDTH * TubeRoiSampler.OUTPUT_HEIGHT
    ).order(ByteOrder.nativeOrder())
    private data class TemporalFrameContext(
        val timestampNs: Long,
        val axis: TubeAxis,
        val ruler: RulerSnapshot,
        val axialMaxPx: Float,
        val geometryCaptureTimestampNs: Long = 0L,
        val geometryRevision: Long = 0L,
        val debugCaptureStepId: Long? = null,
        val stabilization: RoiStabilizationTransform = RoiStabilizationTransform(
            sourceToStabilizedX = 0f,
            sourceToStabilizedY = 0f,
            score = 1f
        )
    )
    private data class ScaleFrameResult(
        val reading: ScaleReading,
        val temporalContext: TemporalFrameContext
    )
    private data class PublishedTemporalBall(
        val detection: TemporalBallDetection?,
        val context: TemporalFrameContext,
        val framesSeen: Int,
        val backgroundReady: Boolean,
        val generation: Long
    )
    /** Immutable camera snapshot handed to the single TinyHeatmapNet worker. */
    private data class PendingTemporalFrame(
        val rawRoi: ByteArray,
        val stableRoi: ByteArray,
        val geometry: TubeRoiGeometry,
        val context: TemporalFrameContext,
        val transform: BufferTransform,
        val sourceFrameIndex: Int,
        val captureFrameIndex: Int,
        val historyFrames: List<TinyHeatmapHistoryFrame>,
        val generation: Long,
        val workToken: Long,
        val queuedAtNs: Long,
        val captureTimestampNs: Long,
        val frameCallbackStartedAtNs: Long,
        val pipelineTimings: TinyPipelineStageTimings,
        val sessionEligible: Boolean
    )
    private val temporalBallRef = AtomicReference<PublishedTemporalBall?>(null)
    /** One-slot latest-only queue prevents model latency from becoming camera latency. */
    private val pendingTemporalFrame = AtomicReference<PendingTemporalFrame?>(null)
    private val temporalWorkerScheduled = AtomicBoolean(false)
    private val ballDiagnosticsRef = AtomicReference<BallTrackingDiagnostics?>(null)
    private val temporalGeneration = AtomicLong(0L)
    @Volatile private var analysisTemporalGeneration = 0L
    private var temporalWorkerGeneration = Long.MIN_VALUE
    private var temporalWorkerDiagnosticOrdinal = 0
    /** Start of the current unavailable-tube interval; owned by the analysis executor. */
    private var classicalNoTubeSinceNs = 0L
    private val temporalResultUsage = TemporalResultUsageTracker()
    private val temporalDebugCaptureRecorder = TemporalDebugCaptureRecorder(context)
    private var lightweightTrackingDiagnostics: LightweightTrackingDiagnosticAccumulator? = null
    private val stabilizedRoiBuffer = ByteBuffer.allocateDirect(
        TubeRoiSampler.OUTPUT_WIDTH * TubeRoiSampler.OUTPUT_HEIGHT
    ).order(ByteOrder.nativeOrder())
    /** Owned by the camera analysis executor; worker snapshots contain immutable byte arrays. */
    // Must match TinyHeatmapNetBallDetector.HISTORY_CAPACITY so every retained capture
    // can actually reach the worker; see the note on skipped captures there.
    private val tinyInputHistory = TinyHeatmapInputHistory(capacity = 6)
    /** Counts successful model work separately from camera callbacks and queue replacements. */
    private val tinyThroughput = TinyHeatmapThroughputTracker()
    private val tinyRuntimeRef = AtomicReference(TinyHeatmapRuntimeDiagnostics())
    private data class TinySequenceFrameRecord(
        val sourceFrameIndex: Int,
        val captureFrameIndex: Int,
        val timestampNs: Long,
        val context: TemporalFrameContext,
        val candidates: List<TinyHeatmapSequenceCandidate>
    )
    private val tinySequenceFrames = ArrayList<TinySequenceFrameRecord>(2048)
    private val causalFallbackObservations = ArrayList<FallFrameObservation>(2048)
    private val causalFallbackLock = Any()
    private var tinySourceFrameIndex = -1
    private var tinyCaptureFrameIndex = -1
    private var tinyLastCaptureTimestampNs = 0L
    private val tinySequenceLock = Any()

    private fun resetTinySequenceState() {
        synchronized(tinySequenceLock) {
            tinySequenceFrames.clear()
            synchronized(causalFallbackLock) { causalFallbackObservations.clear() }
            tinySourceFrameIndex = -1
            tinyCaptureFrameIndex = -1
            tinyLastCaptureTimestampNs = 0L
        }
        tinyInputHistory.reset()
        tinyRoiStabilizer.reset()
        temporalDetectionSelector.reset()
        pendingTemporalFrame.getAndSet(null)
        temporalWorkerScheduled.set(false)
        temporalBusy.set(false)
        runCatching {
            temporalExecutor.submit {
                temporalBallDetector.reset()
                temporalWorkerGeneration = Long.MIN_VALUE
                temporalWorkerDiagnosticOrdinal = 0
            }.get(5L, TimeUnit.SECONDS)
        }.onFailure {
            if (!closed.get()) Log.w(TAG, "Unable to reset TinyHeatmapNet worker", it)
        }
        // The barrier above waits for any in-flight native inference. Start the next generation
        // with an independent throughput window, otherwise a previous camera session would make
        // the effective FPS look better (or worse) than the current one.
        tinyThroughput.reset()
        tinyRuntimeRef.set(TinyHeatmapRuntimeDiagnostics())
    }
    private val tracker = TubeTracker(defaultTransitionMs = 0L)
    private val obbRefreshPolicy = ObbRefreshPolicy()
    private val loadProfileRef = AtomicReference(AnalysisLoadProfile.NORMAL)
    private val obbCompletionThrottle = InferenceCompletionThrottle()
    private val thermalSnapshotRef = AtomicReference<DeviceThermalSnapshot?>(null)
    private val diagnosticsRecorder = RuntimeDiagnosticsRecorder(context)
    private val thermalMonitor = DeviceThermalMonitor(context) { snapshot ->
        if (closed.get()) return@DeviceThermalMonitor
        val previous = loadProfileRef.getAndSet(snapshot.profile)
        thermalSnapshotRef.set(snapshot)
        tracker.setLossTimeoutMs(snapshot.profile.trackerLossTimeoutMs)
        fallTrackingGate.setExpectedObservationIntervalMs(
            CLASSICAL_DEFAULT_PERIOD_MS
        )
        if (previous.level != snapshot.profile.level) {
            Log.i(
                PERFORMANCE_TAG,
                "thermal level=${snapshot.profile.level} signal=${snapshot.signal} " +
                    "batteryC=${snapshot.batteryTemperatureC ?: "unknown"} " +
                    "yoloLimit=${snapshot.profile.yoloFpsLimit}fps " +
                    "yoloCooldown=${snapshot.profile.yoloCooldownNs / 1_000_000L}ms"
            )
        }
    }
    private val busy = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)
    private val desiredRunning = AtomicBoolean(false)
    /** Binding clocks and publication generations change atomically with published geometry. */
    private val frameEpoch = FrameProcessingEpoch()
    private val resolutionRef = AtomicReference(AnalysisResolution.P720)
    private val tubeFocusLockGate = TubeFocusLockGate()
    private var boundCamera: Camera? = null
    private var nativeCamera: XiaomiCamera2LensBackend? = null
    private var nativeCloseFuture = CompletableFuture.completedFuture(Unit)
    private var nativePausedForLifecycle = false
    private var lifecycleObserverAdded = false
    private val nativeLifecycleObserver = object : DefaultLifecycleObserver {
        override fun onStop(owner: LifecycleOwner) {
            if (nativeCamera == null && cameraControls?.selectedNativeLens == null) return
            nativePausedForLifecycle = desiredRunning.get()
            frameEpoch.replaceBinding { tracker.reset() }
            releaseTubeFocus()
            cameraControls?.unbind()
            closeNativeCamera()
            resetAnalysisState()
        }
        override fun onStart(owner: LifecycleOwner) {
            if (!nativePausedForLifecycle) return
            nativePausedForLifecycle = false
            if (!closed.get() && desiredRunning.get()) provider?.let(::bindUseCases)
        }
    }
    private var focusGeneration = 0L
    private data class ModelSelection(val profile: ModelProfile, val generation: Long)
    private val modelSelectionRef =
        AtomicReference(ModelSelection(ModelProfile.DEFAULT, generation = 0L))
    private var provider: ProcessCameraProvider? = null
    // One max-sized native-order tensor supports every profile without reallocating.
    private val modelInput: FloatBuffer =
        ByteBuffer.allocateDirect(
            3 * ModelProfile.ACCURATE_640.inputSize * ModelProfile.ACCURATE_640.inputSize * 4
        )
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
    private val tensorPreprocessor = YuvTensorPreprocessor()
    private val obbFrameSnapshotBuffer = YuvFrameSnapshotBuffer()
    private var pendingRebind = false
    private data class PendingUiResult(
        val result: TubeTrackResult,
        val workToken: Long
    )
    private val pendingUiResult = AtomicReference<PendingUiResult?>(null)
    private val uiResultScheduled = AtomicBoolean(false)
    private val shutdownFuture = CompletableFuture<Unit>()

    // 1D scale-line path. Also confined to analysisExecutor.
    private val axialSampler = AxialSampler()
    private val tickDetector = TickDetector()
    private val tickRuler = TickRuler(RulerConfig.DEFAULT)
    private val denseBandSampler = TubeBandSampler()
    private val denseBallTracker = DenseRegisteredBallTracker(denseBandSampler.lateralFractions)
    private val fallTrackingGate = FallTrackingGate()
    private val temporalDetectionSelector = TemporalDetectionSelector()
    private var axialSignal = FloatArray(2048)
    private var tickAxialScratch = FloatArray(256)
    private var denseBandSignals = Array(denseBandSampler.bandCount) { FloatArray(2048) }

    private val tubeGeometryConsumerCursor = TubeGeometryConsumerCursor()

    @Volatile private var session: FallSession? = null
    @Volatile private var rulerConfig = RulerConfig.DEFAULT
    @Volatile private var sphereRadiusMm: Double? = null
    @Volatile private var tubeInnerDiameterMm: Double? = null

    /** Graduation the current run measures from, so exported distances start at zero. */
    private var sessionRefOrdinal: Int? = null

    private var sampleFps = 0f
    private var lastSampleNs = 0L
    private val cameraRate = CameraRateDiagnostics()
    private var temporalStepFps = 0f
    private var lastTemporalStepNs = 0L
    private var lastPerformanceLogElapsedNs = 0L
    private var analysisBallSource = com.cupet.fallingball.vision.BallDetectionSource.NONE

    val isBallTrackingReady: Boolean
        get() = !closed.get() && desiredRunning.get() &&
            temporalBallEnabled &&
            ballDiagnosticsRef.get()?.stage == TemporalPipelineStage.RUNNING

    fun ballTrackingDiagnostics(): BallTrackingDiagnostics? = ballDiagnosticsRef.get()

    val isTemporalDebugCaptureActive: Boolean
        get() = temporalDebugCaptureRecorder.isActive

    fun startTemporalDebugCapture(): Boolean = temporalDebugCaptureRecorder.start()

    fun startPipelineTimingCapture(): Boolean = temporalDebugCaptureRecorder.startTimingOnly()

    fun stopTemporalDebugCapture(): Boolean = temporalDebugCaptureRecorder.stop()

    fun benchmarkTinyParent() {
        if (closed.get()) return
        pause()
        runCatching {
            benchmarkExecutor.execute {
                awaitTemporalWorkerDrainForBenchmark()
                runCatching {
                    TinyHeatmapNetBallDetector(context).use { detector ->
                        detector.benchmarkParentResult(iterations = 12)
                    }
                }
                    .onSuccess { result -> logTinyBenchmark(result, "production_parent") }
                    .onFailure { failure ->
                        Log.e(TINY_BENCHMARK_TAG, "status=error", failure)
                    }
            }
        }.onFailure { failure ->
            Log.e(TINY_BENCHMARK_TAG, "status=error unable to schedule", failure)
        }
    }

    fun benchmarkTinyPreprocess() {
        if (closed.get()) return
        pause()
        benchmarkExecutor.execute {
            awaitTemporalWorkerDrainForBenchmark()
            runCatching {
                TinyHeatmapNetBallDetector(context).use { detector ->
                    detector.benchmarkPreprocessResult(iterations = 12)
                }
            }
                .onSuccess { result -> logTinyBenchmark(result, "preprocess") }
                .onFailure { failure ->
                    Log.e(TINY_BENCHMARK_TAG, "status=error backend=preprocess", failure)
                }
        }
    }

    fun benchmarkTinyCandidate() {
        if (closed.get()) return
        pause()
        benchmarkExecutor.execute {
            awaitTemporalWorkerDrainForBenchmark()
            val candidates = listOf(
                "sep_fp32_nnapi" to ("tinyheatmapnet_sep_qnn.onnx" to true),
                "production_fp32_nnapi" to ("tinyheatmapnet_final.onnx" to true),
                "sep_fp32_cpu" to ("tinyheatmapnet_sep_qnn.onnx" to false)
            )
            candidates.forEach { (label, config) ->
                runCatching {
                    temporalBallDetector.benchmarkAssetResult(config.first, config.second)
                }.onSuccess { result -> logTinyBenchmark(result, label) }
                    .onFailure { failure ->
                    Log.e(
                        TINY_BENCHMARK_TAG,
                        "status=error backend=$label",
                        failure
                    )
                }
            }
        }
    }

    fun benchmarkTinyNcnn() {
        if (closed.get()) return
        pause()
        benchmarkExecutor.execute {
            awaitTemporalWorkerDrainForBenchmark()
            listOf(
                "ncnn_cpu" to false,
                "ncnn_vulkan" to true
            ).forEach { (label, useVulkan) ->
                runCatching {
                    NcnnTinyHeatmapNet(
                        context = context,
                        useVulkan = useVulkan,
                        // This is an explicit diagnostic path.  If the graph passes the static
                        // capability preflight, measure it, but keep placementVerified=false
                        // until NCNN exposes actual runtime layer evidence.
                        requireVerifiedPlacement = false
                    ).use { it.benchmarkResult() }
                }.onSuccess { result -> logTinyBenchmark(result, label) }
                    .onFailure { failure ->
                    Log.e(TINY_BENCHMARK_TAG, "status=error backend=$label", failure)
                }
            }
        }
    }

    /**
     * Measures CPU thread counts without hardware selection or the live camera competing for the
     * same cores. Each result uses a fresh detector so the session options are unambiguous.
     */
    fun benchmarkTinyCpuThreads() {
        if (closed.get()) return
        pause()
        benchmarkExecutor.execute {
            awaitTemporalWorkerDrainForBenchmark()
            listOf(1, 2, 3, 4).forEach { threads ->
                runCatching {
                    TinyHeatmapNetBallDetector(
                        context = context,
                        allowHardwareParent = false,
                        cpuInferenceThreads = threads
                    ).use { detector ->
                        detector.benchmarkParentResult(iterations = 12)
                    }
                }.onSuccess { result ->
                    logTinyBenchmark(result, "production_parent_cpu_threads_$threads")
                }.onFailure { failure ->
                    Log.e(
                        TINY_BENCHMARK_TAG,
                        "status=error backend=production_parent_cpu_threads_$threads",
                        failure
                    )
                }
            }
        }
    }

    private fun awaitTemporalWorkerDrainForBenchmark() {
        runCatching {
            temporalExecutor.submit { }.get(10L, TimeUnit.SECONDS)
        }.onFailure { failure ->
            Log.e(TINY_BENCHMARK_TAG, "status=error unable to drain temporal worker", failure)
        }
    }

    private fun logTinyBenchmark(result: TinyHeatmapBenchmarkResult, requestedPath: String) {
        val samples = result.samplesMs
        val sorted = samples.sorted()
        val p50 = sorted[sorted.size / 2]
        val p95Index = (kotlin.math.ceil(sorted.size * 0.95).toInt() - 1)
            .coerceIn(0, sorted.lastIndex)
        val mean = samples.average()
        Log.i(
            TINY_BENCHMARK_TAG,
            String.format(
                Locale.US,
                    "status=ok requested=%s backend=%s computeUnit=%s " +
                    "placementVerified=%s placement=%s asset=%s iterations=%d p50Ms=%.3f p95Ms=%.3f " +
                    "meanMs=%.3f samplesMs=%s",
                requestedPath,
                result.backendName,
                result.computeUnit,
                result.providerPlacementVerified,
                result.placementSummary,
                result.assetName ?: "none",
                samples.size,
                p50,
                sorted[p95Index],
                mean,
                samples.joinToString(",") { "%.3f".format(Locale.US, it) }
            )
        )
    }

    fun setResolution(resolution: AnalysisResolution) {
        if (closed.get()) return
        if (resolutionRef.get() == resolution) return
        val shouldRebind = desiredRunning.get() && (boundCamera != null || nativeCamera != null)
        resolutionRef.set(resolution)
        // All registration/background coordinates are pixels and cannot cross resolutions.
        relock()
        // A parameter change while history has paused CameraX must not silently
        // turn the camera back on. The remembered value is applied by the next [resume].
        if (shouldRebind) provider?.let { bindUseCases(it) }
    }

    fun setBallGeometry(
        sphereRadiusMm: Double?,
        tubeInnerDiameterMm: Double?,
        sphereMaterial: String? = null
    ) {
        if (closed.get()) return
        this.sphereRadiusMm = sphereRadiusMm
        this.tubeInnerDiameterMm = tubeInnerDiameterMm
        // TinyHeatmapNet is a single frozen model for all validated appearances. Geometry
        // changes still invalidate the candidate sequence so no run crosses coordinate systems.
        val previousGeneration = temporalGeneration.get()
        val nextGeneration = temporalGeneration.incrementAndGet()
        temporalBallRef.set(null)
        ballDiagnosticsRef.set(null)
        temporalDebugCaptureRecorder.recordReset(
            TemporalResetReason.PROFILE_CHANGED,
            timestampNs = null,
            previousGeneration = previousGeneration,
            nextGeneration = nextGeneration
        )
        runCatching {
            analysisExecutor.execute {
                // A newer configuration supersedes this queued reset. Its own reset is already
                // queued behind this one on the same executor.
                if (temporalGeneration.get() != nextGeneration) return@execute
                temporalSequenceBuffer.reset()
                resetTinySequenceState()
                classicalNoTubeSinceNs = 0L
                temporalResultUsage.reset()
                denseBallTracker.resetTrajectory()
                fallTrackingGate.reset()
                sessionRefOrdinal = null
                analysisTemporalGeneration = nextGeneration
                temporalStepFps = 0f
                lastTemporalStepNs = 0L
            }
        }.onFailure {
            if (!closed.get()) Log.w(TAG, "Unable to queue classical detector profile reset", it)
        }
    }

    /** CameraX may recreate the device across a lifecycle stop, invalidating its AF lock. */
    fun refreshTubeFocus() {
        if (closed.get()) return
        releaseTubeFocus()
    }

    /**
     * Changes only the YOLO tensor size; CameraX capture resolution stays untouched so the
     * native-resolution ruler path keeps its measurement precision.
     */
    @Synchronized
    fun setModelProfile(profile: ModelProfile): Long {
        val current = modelSelectionRef.get()
        if (closed.get()) return current.generation
        if (isSessionActive || current.profile == profile) return current.generation

        val next = ModelSelection(profile, current.generation + 1L)
        modelSelectionRef.set(next)
        frameEpoch.invalidateWork { tracker.resetForProfile(next.generation) }
        obbRefreshPolicy.reset()
        releaseTubeFocus()
        resetAnalysisState()
        return next.generation
    }

    fun relock() {
        if (closed.get()) return
        finishSession(BallEndReason.MANUAL)
        releaseTubeFocus()
        frameEpoch.invalidateWork { tracker.reset() }
        obbRefreshPolicy.reset()
        session = null
        resetAnalysisState()
    }

    /**
     * Tick/ruler state is owned by [analysisExecutor]. Queue resets on that same executor
     * instead of mutating it from the UI thread.
     */
    private fun resetAnalysisState() {
        // The UI must stop accepting a run immediately; the owned detector state itself is
        // reset in-order on the analysis executor below.
        temporalBallRef.set(null)
        ballDiagnosticsRef.set(null)
        val previousTemporalGeneration = temporalGeneration.get()
        val nextTemporalGeneration = temporalGeneration.incrementAndGet()
        temporalDebugCaptureRecorder.recordReset(
            TemporalResetReason.PIPELINE_RESET,
            timestampNs = null,
            previousGeneration = previousTemporalGeneration,
            nextGeneration = nextTemporalGeneration
        )
        analysisExecutor.execute {
            if (nextTemporalGeneration != temporalGeneration.get()) return@execute
            tickRuler.reset()
            denseBallTracker.reset()
            fallTrackingGate.reset()
            sessionRefOrdinal = null
            tubeGeometryConsumerCursor.reset()
            temporalSequenceBuffer.reset()
            resetTinySequenceState()
            classicalNoTubeSinceNs = 0L
            analysisTemporalGeneration = nextTemporalGeneration
            temporalResultUsage.reset()
            temporalStepFps = 0f
            lastTemporalStepNs = 0L
            cameraRate.reset()
        }
    }

    /** Pause camera analysis while the activity leaves the live camera screen. */
    fun pause() {
        if (closed.get()) return
        desiredRunning.set(false)
        nativePausedForLifecycle = false
        frameEpoch.replaceBinding { tracker.reset() }
        releaseTubeFocus()
        cameraControls?.unbind()
        closeNativeCamera()
        obbRefreshPolicy.reset()
        boundCamera = null
        resetAnalysisState()
        try {
            provider?.unbindAll()
        } catch (_: Throwable) {
        }
    }

    /** Clear tube registration and the empty-tube background without binding the camera. */
    fun clearAnalysisReferences() {
        if (closed.get()) return
        releaseTubeFocus()
        frameEpoch.invalidateWork { tracker.reset() }
        obbRefreshPolicy.reset()
        resetAnalysisState()
    }

    /**
     * Resume camera analysis after pause.
     *
     * History intentionally requests a reset: the phone may have moved while CameraX was
     * unbound, so neither the old tube registration nor the old empty-tube background is a
     * valid reference after returning.
     */
    fun resume(resetAnalysis: Boolean = false) {
        if (closed.get()) return
        desiredRunning.set(true)
        if (resetAnalysis) {
            releaseTubeFocus()
            frameEpoch.invalidateWork { tracker.reset() }
            obbRefreshPolicy.reset()
            resetAnalysisState()
        }
        provider?.let { bindUseCases(it) }
    }

    fun start() {
        if (closed.get()) return
        desiredRunning.set(true)
        if (!lifecycleObserverAdded) {
            lifecycleObserverAdded = true
            lifecycleOwner.lifecycle.addObserver(nativeLifecycleObserver)
        }
        thermalMonitor.start()
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            if (closed.get()) return@addListener
            val cameraProvider = runCatching { future.get() }
                .onFailure { Log.e(TAG, "Unable to obtain camera provider", it) }
                .getOrNull() ?: return@addListener
            if (closed.get()) return@addListener
            provider = cameraProvider
            if (desiredRunning.get()) bindUseCases(cameraProvider)
        }, ContextCompat.getMainExecutor(context))
    }

    fun shutdown(): CompletableFuture<Unit> {
        if (!closed.compareAndSet(false, true)) return shutdownFuture
        desiredRunning.set(false)
        nativePausedForLifecycle = false
        cameraLifecycleOwner?.lifecycle?.removeObserver(nativeLifecycleObserver)
        frameEpoch.invalidateWork()
        temporalGeneration.incrementAndGet()
        pendingUiResult.set(null)
        // Deliver a final active measurement before the Activity closes its persistence queue.
        finishSession(BallEndReason.MANUAL, deliverImmediately = true)
        frameEpoch.replaceBinding()
        runCatching { thermalMonitor.close() }
            .onFailure { Log.w(TAG, "Unable to stop thermal monitor", it) }
        releaseTubeFocus()
        boundCamera = null
        val cameraProvider = provider
        cameraControls?.unbind()
        closeNativeCamera()
        provider = null
        pendingRebind = false
        runCatching { cameraProvider?.unbindAll() }
            .onFailure { Log.w(TAG, "Unable to unbind camera during shutdown", it) }
        analysisExecutor.shutdown()
        obbExecutor.shutdown()
        temporalExecutor.shutdown()
        benchmarkExecutor.shutdown()
        Thread({
            // Native workers can take time while thermally throttled. Keep all waits and
            // ONNX teardown away from the Activity main thread, and isolate each cleanup stage
            // so one vendor/runtime failure cannot prevent the remaining resources closing.
            fun cleanup(label: String, action: () -> Unit) {
                runCatching(action).onFailure {
                    Log.w(TAG, "Camera cleanup failed at $label", it)
                }
            }

            cleanup("analysis executor") { shutdownExecutor(analysisExecutor, "analysis") }
            cleanup("YOLO executor") { shutdownExecutor(obbExecutor, "YOLO") }
            cleanup("temporal executor") { shutdownExecutor(temporalExecutor, "TinyHeatmapNet") }
            cleanup("benchmark executor") {
                shutdownExecutor(benchmarkExecutor, "TinyHeatmapNet benchmark")
            }
            cleanup("TinyHeatmapNet detector") { temporalBallDetector.close() }
            cleanup("temporal capture") { temporalDebugCaptureRecorder.close() }
            cleanup("diagnostics recorder") { diagnosticsRecorder.close() }
            shutdownFuture.complete(Unit)
        }, "FallingBall-camera-cleanup").start()
        return shutdownFuture
    }

    private fun bindUseCases(cameraProvider: ProcessCameraProvider) {
        if (closed.get() || !desiredRunning.get()) return
        val changeToken = frameEpoch.replaceBinding { tracker.reset() }
        obbRefreshPolicy.reset()
        resetAnalysisState()
        // ViewPort is only valid after PreviewView is laid out. Rebind once when ready so
        // Preview + ImageAnalysis share the same FOV (eliminates overlay offset).
        if (previewView.width == 0 || previewView.height == 0 || previewView.viewPort == null) {
            if (!pendingRebind) {
                pendingRebind = true
                previewView.post {
                    pendingRebind = false
                    if (closed.get() || !desiredRunning.get()) return@post
                    provider?.let { bindUseCases(it) }
                }
            }
            // Fall through and bind without ViewPort so preview still starts.
        }

        releaseTubeFocus()
        cameraControls?.unbind()
        val previousCameraX = boundCamera
        closeNativeCamera()
        boundCamera = null
        cameraProvider.unbindAll()
        val resolution = resolutionRef.get()

        if (!nativeCloseFuture.isDone) {
            nativeCloseFuture.whenComplete { _, _ ->
                ContextCompat.getMainExecutor(context).execute {
                    if (!closed.get() && desiredRunning.get() && frameEpoch.isCurrentBinding(changeToken)) {
                        bindSelectedCamera(cameraProvider, resolution, previousCameraX)
                    }
                }
            }
            return
        }
        bindSelectedCamera(cameraProvider, resolution, previousCameraX)
    }

    private fun bindSelectedCamera(cameraProvider: ProcessCameraProvider, resolution: AnalysisResolution, previousCameraX: Camera?) {
        cameraControls?.selectedNativeLens?.let { lens ->
            bindNativeCamera(cameraProvider, resolution, lens, previousCameraX)
            return
        }

        try {
            bindInternal(
                cameraProvider,
                resolution,
                enableStab = true,
                requestTargetFps = true
            )
            if (!closed.get()) onStabilizationInfo(true)
        } catch (stabilizedError: Throwable) {
            Log.w(TAG, "bind with stabilization failed, retry plain", stabilizedError)
            releaseTubeFocus()
            boundCamera = null
            cameraProvider.unbindAll()
            try {
                bindInternal(
                    cameraProvider,
                    resolution,
                    enableStab = false,
                    requestTargetFps = true
                )
                if (!closed.get()) onStabilizationInfo(false)
            } catch (targetFpsError: Throwable) {
                Log.w(TAG, "bind with 30 fps request failed, retry camera default", targetFpsError)
                releaseTubeFocus()
                boundCamera = null
                cameraProvider.unbindAll()
                bindInternal(
                    cameraProvider,
                    resolution,
                    enableStab = false,
                    requestTargetFps = false
                )
                if (!closed.get()) onStabilizationInfo(false)
            }
        }
    }

    private fun closeNativeCamera(): XiaomiCamera2LensBackend? {
        val previous = nativeCamera
        nativeCamera = null
        previous?.close()
        if (previous != null) nativeCloseFuture = previous.closedFuture
        return previous
    }

    private fun bindNativeCamera(cameraProvider: ProcessCameraProvider, resolution: AnalysisResolution,
        lens: XiaomiCamera2Lens, previousCameraX: Camera?) {
        val token = frameEpoch.replaceBinding { tracker.reset() }
        obbRefreshPolicy.reset()
        resetAnalysisState()
        lateinit var backend: XiaomiCamera2LensBackend
        try {
            backend = XiaomiCamera2LensBackend(context, lens, previewView, Size(resolution.width, resolution.height),
                onFrame = { frame ->
                    if (closed.get() || !desiredRunning.get() || nativeCamera !== backend || !frameEpoch.isCurrentBinding(token) ||
                        !busy.compareAndSet(false, true)) {
                        frame.close()
                    } else {
                        try {
                            analysisExecutor.execute {
                                if (closed.get() || nativeCamera !== backend || !frameEpoch.isCurrentBinding(token)) {
                                    busy.set(false)
                                    frame.close()
                                } else analyzeFrame(frame, token)
                            }
                        } catch (_: java.util.concurrent.RejectedExecutionException) {
                            busy.set(false)
                            frame.close()
                        }
                    }
                },
                onReady = {
                    if (nativeCamera === backend && frameEpoch.isCurrentBinding(token)) {
                        cameraControls?.bindNative(cameraProvider, backend)
                        onStabilizationInfo(true)
                    }
                },
                onFocusDistance = { distance -> cameraControls?.observeNativeFocus(lens.id, distance) },
                onFailure = { failure ->
                    if (nativeCamera === backend && frameEpoch.isCurrentBinding(token)) {
                        Log.e(TAG, "Native telephoto binding failed; returning to main camera", failure)
                        // Stop admitting frames immediately, but preserve the capture clock
                        // and trajectory until the existing session finalizer freezes them.
                        val failedSession = session?.takeIf { it.isActive }
                        frameEpoch.invalidateWork()
                        cameraControls?.unbind()
                        closeNativeCamera()
                        val recover = {
                            if (!closed.get() && desiredRunning.get() && frameEpoch.isCurrentBinding(token)) {
                                cameraControls?.nativeBindingFailed(lens.id)
                                bindUseCases(cameraProvider)
                            }
                        }
                        if (failedSession != null) {
                            analysisExecutor.execute {
                                if (session === failedSession) finishSession(BallEndReason.BALL_LOST)
                                // Record construction was queued by finishSession before this
                                // callback; no main-camera sample can enter the old run.
                                ContextCompat.getMainExecutor(context).execute(recover)
                            }
                        } else recover()
                    }
                })
            nativeCamera = backend
            val start = {
                if (!closed.get() && desiredRunning.get() && nativeCamera === backend && frameEpoch.isCurrentBinding(token)) {
                    if (lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) backend.start()
                    else nativePausedForLifecycle = true
                }
            }
            // CameraX closes asynchronously. Wait for its device before opening the auxiliary
            // lens, otherwise Xiaomi may reject the new camera as already in use.
            if (previousCameraX == null || previousCameraX.cameraInfo.cameraState.value?.type == CameraState.Type.CLOSED) {
                start()
            } else {
                val states = previousCameraX.cameraInfo.cameraState
                var observer: Observer<CameraState>? = null
                observer = Observer { state ->
                    if (state.type == CameraState.Type.CLOSED) {
                        observer?.let(states::removeObserver)
                        start()
                    }
                }
                states.observeForever(observer)
                previewView.postDelayed({ observer?.let(states::removeObserver); start() }, 3_000)
            }
        } catch (failure: Exception) {
            Log.e(TAG, "Unable to initialize native telephoto", failure)
            cameraControls?.nativeBindingFailed(lens.id)
            bindUseCases(cameraProvider)
        }
    }

    private fun bindInternal(
        cameraProvider: ProcessCameraProvider,
        resolution: AnalysisResolution,
        enableStab: Boolean,
        requestTargetFps: Boolean
    ) {
        if (closed.get() || !desiredRunning.get()) return
        // Each fallback creates a distinct ImageAnalysis instance. Invalidate callbacks from
        // the failed attempt before registering the next analyzer.
        val bindingRegistrationToken = frameEpoch.replaceBinding { tracker.reset() }
        obbRefreshPolicy.reset()
        resetAnalysisState()
        val analysis = buildAnalysis(
            resolution,
            enableStab,
            requestTargetFps,
            bindingRegistrationToken
        )
        val preview = buildPreview(enableStab)
        val viewPort = previewView.viewPort
        if (viewPort != null) {
            val group = UseCaseGroup.Builder()
                .setViewPort(viewPort)
                .addUseCase(preview)
                .addUseCase(analysis)
                .build()
            boundCamera = cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraControls?.selector ?: CameraSelector.DEFAULT_BACK_CAMERA,
                group
            )
            Log.d(TAG, "Bound with shared ViewPort")
        } else {
            boundCamera = cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraControls?.selector ?: CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
            Log.w(TAG, "bound without ViewPort (layout not ready)")
        }
        boundCamera?.let { cameraControls?.bind(cameraProvider, it) }
    }

    private fun buildPreview(enableStab: Boolean): Preview {
        val builder = Preview.Builder()
        cameraControls?.attachCaptureCallback(builder)
        if (enableStab) enableStabilization(builder)
        return builder.build().also { it.surfaceProvider = previewView.surfaceProvider }
    }

    private fun buildAnalysis(
        resolution: AnalysisResolution,
        enableStab: Boolean,
        requestTargetFps: Boolean,
        bindingRegistrationToken: Long
    ): ImageAnalysis {
        val analysisSize = Size(resolution.width, resolution.height)
        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    analysisSize,
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()

        val builder = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
        if (requestTargetFps) requestAnalysisFrameRate(builder)
        if (enableStab) enableStabilization(builder)
        observeAppliedFrameRate(builder)

        val analysis = builder.build()
        analysis.setAnalyzer(analysisExecutor) { proxy ->
            val image = runCatching { AnalysisFrame.fromCamera(proxy) }.getOrElse {
                proxy.close()
                Log.w(TAG, "Ignoring malformed camera frame", it)
                return@setAnalyzer
            }
            if (closed.get() || !frameEpoch.isCurrentBinding(bindingRegistrationToken)) {
                image.close()
                return@setAnalyzer
            }
            if (!busy.compareAndSet(false, true)) {
                image.close()
                return@setAnalyzer
            }
            val frameStartNs = System.nanoTime()
            cameraRate.onFrameStart(image.imageInfo.timestamp, frameStartNs)
            try {
                analyzeFrame(image, bindingRegistrationToken)
            } finally {
                val frameEndNs = System.nanoTime()
                cameraRate.onFrameEnd(frameEndNs - frameStartNs, frameEndNs)
            }
        }
        return analysis
    }

    /** Both sources enter here. Live scheduling stays asynchronous; offline work is serial. */
    private fun analyzeFrame(image: AnalysisFrame, bindingRegistrationToken: Long? = null) {
            val analyzerTimer = TinyPipelineStageTimer.create(
                temporalDebugCaptureRecorder.isActive
            )
            val analyzerStartedNs = analyzerTimer.start()
            val frameSetupStartedNs = analyzerTimer.start()
            var analyzerTimestampNs: Long? = null
            try {
                if (closed.get()) return
                val tsNs = image.imageInfo.timestamp
                // Recheck admission under the publication lock: a bind may have changed after
                // the analyzer's cheap preflight check, including to a different capture clock.
                val workToken = frameEpoch.enterFrame(bindingRegistrationToken, tsNs) ?: return
                val rot = image.imageInfo.rotationDegrees
                val crop = image.cropRect
                // With shared ViewPort, cropRect is the FOV that matches the preview.
                val xf = BufferTransform.of(
                    rot, crop.left, crop.top,
                    crop.width().coerceAtLeast(1), crop.height().coerceAtLeast(1)
                )
                val displayW = xf.displayW
                val displayH = xf.displayH

                // Centre ROI in display coordinates. Everything outside is never sampled
                // into the tensor, so the model cannot fire on bench clutter at the edges.
                val roiX = (displayW * DetectionRoi.LEFT).roundToInt()
                    .coerceIn(0, (displayW - 1).coerceAtLeast(0))
                val roiY = (displayH * DetectionRoi.TOP).roundToInt()
                    .coerceIn(0, (displayH - 1).coerceAtLeast(0))
                val roiW = (displayW * DetectionRoi.WIDTH).roundToInt()
                    .coerceIn(1, displayW - roiX)
                val roiH = (displayH * DetectionRoi.HEIGHT).roundToInt()
                    .coerceIn(1, displayH - roiY)

                analyzerTimestampNs = tsNs
                analyzerTimer.recordSince(TinyPipelineStage.FRAME_SETUP, frameSetupStartedNs)
                val yPlaneCaptureStartedNs = analyzerTimer.start()
                temporalDebugCaptureRecorder.captureYPlaneOnce(image, xf, tsNs)
                analyzerTimer.recordSince(
                    TinyPipelineStage.Y_PLANE_DIAGNOSTIC_CAPTURE,
                    yPlaneCaptureStartedNs
                )
                analysisBallSource = com.cupet.fallingball.vision.BallDetectionSource.NONE

                val trackStartedNs = analyzerTimer.start()
                val modelSelection = modelSelectionRef.get()
                val profile = modelSelection.profile
                val generation = modelSelection.generation
                val loadProfile = loadProfileRef.get()
                if (offline && shouldRefreshOfflineObb(tsNs, loadProfile.maxTrackedAxisAgeNs)) {
                    submitObbDetection(
                        image, tsNs, displayW, displayH, roiX, roiY, roiW, roiH,
                        profile, generation, workToken
                    )
                }
                val trackerState = tracker.advance(tsNs)
                val trackedResult = trackerState.result
                    ?.takeIf { it.profileGeneration == generation }
                val trackedAxis = trackerState.axis.takeIf { trackedResult != null }
                // Tube geometry is static, so re-detecting it is pure overhead once it is
                // locked.  On a device where one OBB inference costs hundreds of milliseconds,
                // the fixed 200 ms interval keeps a whole core busy and starves the ball
                // pipeline (Dimensity 1000+: 341 ms OBB at 4 fps tiny throughput).  Never
                // refresh faster than two measured inference periods; fast devices
                // (~84 ms) keep their profile interval unchanged.
                val measuredObbMs = trackerState.result
                    ?.inferenceMs
                    ?.takeIf { it.isFinite() && it > 0f }
                    ?: 0f
                val obbIntervalNs = maxOf(
                    loadProfile.yoloIntervalNs,
                    (2f * measuredObbMs * 1_000_000f).toLong()
                )
                val obbDecision = obbRefreshPolicy.decide(
                    frameTimestampNs = tsNs,
                    lastDetectionTimestampNs = trackerState.detectionTimestampNs,
                    workerBusy = obbBusy.get() ||
                        !obbCompletionThrottle.canStart(
                            System.nanoTime(),
                            loadProfile.yoloCooldownNs
                        ),
                    hasTrackedAxis = trackedAxis != null,
                    refreshIntervalNs = obbIntervalNs,
                    maxTrackedAxisAgeNs = loadProfile.maxTrackedAxisAgeNs
                )
                analyzerTimer.recordSince(TinyPipelineStage.TRACK_AND_POLICY, trackStartedNs)

                var result = trackedResult ?: TubeTrackResult(
                    mode = TrackMode.SEARCHING,
                    frameWidth = displayW,
                    frameHeight = displayH,
                    modelInputSize = profile.inputSize,
                    profileGeneration = generation
                )
                if (obbDecision.trackedAxisUsable && trackedAxis != null) {
                    if (offline) offlineSawTube = true
                    classicalNoTubeSinceNs = 0L
                    var ballSubmitted = false
                    val readScaleStartedNs = analyzerTimer.start()
                    val scaleFrame = readScale(
                        image = image,
                        xf = xf,
                        track = result,
                        axis = trackedAxis,
                        tsNs = tsNs,
                        stageTimer = analyzerTimer,
                        onTemporalContextReady = { temporalContext ->
                            ballSubmitted = true
                            submitTemporalBall(
                                image,
                                xf,
                                trackedAxis,
                                temporalContext,
                                workToken,
                                tsNs,
                                analyzerStartedNs,
                                analyzerTimer
                            )
                        }
                    )
                    analyzerTimer.recordSince(TinyPipelineStage.READ_SCALE_TOTAL, readScaleStartedNs)
                    if (scaleFrame != null) {
                        result = result.copy(scale = scaleFrame.reading)
                        if (offline && scaleFrame.reading.state == RulerState.VALID) {
                            offlineSawRuler = true
                            if (session == null && temporalBallRef.get()?.backgroundReady == true) {
                                lightweightTrackingDiagnostics = LightweightTrackingDiagnosticAccumulator()
                                session = FallSession(tsNs + 1L, rulerConfig, offlineResolutionLabel, offlineRunId)
                                fallTrackingGate.reset()
                                sessionRefOrdinal = null
                            }
                        }
                    }
                    if (!ballSubmitted) {
                        submitTemporalBall(
                            image,
                            xf,
                            trackedAxis,
                            TemporalFrameContext(
                                timestampNs = tsNs,
                                axis = trackedAxis,
                                ruler = RulerSnapshot.EMPTY,
                                axialMaxPx = trackedAxis.lengthPx,
                                geometryCaptureTimestampNs = result.geometryCaptureTimestampNs,
                                geometryRevision = result.geometryRevision
                            ),
                            workToken,
                            tsNs,
                            analyzerStartedNs,
                            analyzerTimer
                        )
                    }
                } else {
                    temporalDebugCaptureRecorder.recordSchedule(
                        timestampNs = tsNs,
                        outcome = TemporalScheduleOutcome.NO_TUBE,
                        ruler = RulerSnapshot.EMPTY,
                        geometryAgeNs = (tsNs - result.geometryCaptureTimestampNs)
                            .coerceAtLeast(0L),
                        generation = analysisTemporalGeneration,
                        loadLevel = loadProfile.level,
                        thermal = thermalSnapshotRef.get(),
                        yoloBusy = obbBusy.get(),
                        temporalBusy = temporalBusy.get(),
                        cooldownRemainingNs = 0L
                    )
                    if (classicalNoTubeSinceNs == 0L) {
                        classicalNoTubeSinceNs = tsNs
                    }
                    if (tsNs - classicalNoTubeSinceNs >= CLASSICAL_TUBE_LOSS_RESET_NS) {
                        // A brief YOLO miss is normal on the phone. Preserve heatmap-C's
                        // causal background until the tube has genuinely been unavailable
                        // for a sustained interval, then start a new coordinate system.
                        invalidateTemporalSequence(TemporalResetReason.NO_TUBE, tsNs)
                        classicalNoTubeSinceNs = tsNs
                    }
                    advanceTrackingWithoutObservation(tsNs)
                }

                // Only a raw plane copy remains on the high-rate path. YOLO resize, colour
                // conversion and inference all run on the low-rate worker.
                if (!offline && obbDecision.submitDetection) {
                    val obbSnapshotStartedNs = analyzerTimer.start()
                    submitObbDetection(
                        image = image,
                        timestampNs = tsNs,
                        displayW = displayW,
                        displayH = displayH,
                        roiX = roiX,
                        roiY = roiY,
                        roiW = roiW,
                        roiH = roiH,
                        profile = profile,
                        generation = generation,
                        workToken = workToken
                    )
                    analyzerTimer.recordSince(
                        TinyPipelineStage.OBB_FRAME_SNAPSHOT_ENQUEUE,
                        obbSnapshotStartedNs
                    )
                }
                val resultPublishStartedNs = analyzerTimer.start()
                publishBallDiagnostics(tsNs, obbDecision.trackedAxisUsable)
                result = result.copy(fps = sampleFps(tsNs))
                logRuntimeDiagnostics(result)
                publishUiResult(result, workToken)
                analyzerTimer.recordSince(
                    TinyPipelineStage.CAMERA_RESULT_PUBLISH,
                    resultPublishStartedNs
                )
            } catch (t: Throwable) {
                Log.e(TAG, "analyze failed", t)
                if (offline) throw t
            } finally {
                analyzerTimer.recordSince(TinyPipelineStage.CAMERA_ANALYZER_TOTAL, analyzerStartedNs)
                analyzerTimestampNs?.let { timestampNs ->
                    temporalDebugCaptureRecorder.recordAnalyzerTiming(
                        timestampNs,
                        analyzerTimer.snapshot()
                    )
                }
                busy.set(false)
                image.close()
            }
    }

    private fun shouldRefreshOfflineObb(timestampNs: Long, maxTrackedAxisAgeNs: Long): Boolean {
        val state = tracker.stateSnapshot()
        return offlineObbRefreshPolicy.shouldRefresh(
            timestampNs = timestampNs,
            hasTrackedAxis = state.axis != null,
            lastDetectionTimestampNs = state.detectionTimestampNs,
            maxTrackedAxisAgeNs = maxTrackedAxisAgeNs
        )
    }

    /** Coalesces camera-rate callbacks so the main thread never queues stale frame results. */
    private fun publishUiResult(result: TubeTrackResult, workToken: Long) {
        if (closed.get() || !frameEpoch.isCurrentWork(workToken)) return
        if (offline) return
        pendingUiResult.set(PendingUiResult(result, workToken))
        scheduleUiResultDrain()
    }

    private fun scheduleUiResultDrain() {
        if (!uiResultScheduled.compareAndSet(false, true)) return
        ContextCompat.getMainExecutor(context).execute {
            try {
                val pending = pendingUiResult.getAndSet(null)
                if (pending != null && !closed.get() &&
                    frameEpoch.isCurrentWork(pending.workToken)
                ) {
                    lockFocusOnTubeIfNeeded(pending.result)
                    onResult(pending.result)
                }
            } finally {
                uiResultScheduled.set(false)
                if (pendingUiResult.get() != null && !closed.get()) scheduleUiResultDrain()
            }
        }
    }

    /**
     * Focus once on the detected tube and keep that lens position until registration is reset.
     *
     * The detector can update at several frames per second, but autofocus must not: repeatedly
     * submitting metering actions is what causes visible focus breathing and small changes in
     * the apparent spacing of the graduation lines.
     */
    private fun lockFocusOnTubeIfNeeded(result: TubeTrackResult) {
        if (!tubeAutofocusEnabled) return
        nativeCamera?.let { native ->
            val target = tubeFocusLockGate.takeRequest(result) ?: return
            val requestGeneration = focusGeneration
            val future = native.autofocus(target.xFraction, target.yFraction)
            future.addListener({
                if (!closed.get() && requestGeneration == focusGeneration) {
                    tubeFocusLockGate.onRequestCompleted(runCatching { future.get() }.getOrDefault(false))
                }
            }, ContextCompat.getMainExecutor(context))
            return
        }
        val camera = boundCamera ?: return
        if (previewView.width <= 0 || previewView.height <= 0) return
        val target = tubeFocusLockGate.takeRequest(result) ?: return
        val requestGeneration = focusGeneration

        val point = previewView.meteringPointFactory.createPoint(
            (if (cameraControls?.state?.let { state ->
                state.cameras.firstOrNull { it.id == state.selectedCameraId }?.isFront
            } == true) 1f - target.xFraction else target.xFraction) * previewView.width,
            target.yFraction * previewView.height,
            0.18f
        )
        val action = FocusMeteringAction.Builder(
            point,
            FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE
        )
            .disableAutoCancel()
            .build()
        val request = runCatching {
            camera.cameraControl.startFocusAndMetering(action)
        }.onFailure {
            Log.w(TAG, "Unable to lock focus on tube", it)
        }.getOrElse {
            if (requestGeneration == focusGeneration) {
                tubeFocusLockGate.onRequestCompleted(success = false)
            }
            return
        }
        request.addListener(
            {
                if (closed.get()) return@addListener
                if (requestGeneration != focusGeneration) return@addListener
                val successful = runCatching { request.get().isFocusSuccessful }
                    .getOrDefault(false)
                tubeFocusLockGate.onRequestCompleted(successful)
            },
            ContextCompat.getMainExecutor(context)
        )
    }

    private fun releaseTubeFocus() {
        focusGeneration += 1L
        tubeFocusLockGate.reset()
        nativeCamera?.cancelAutofocus()
        runCatching {
            boundCamera?.cameraControl?.cancelFocusAndMetering()
        }.onFailure {
            Log.d(TAG, "Unable to release focus lock", it)
        }
    }

    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun requestAnalysisFrameRate(builder: ImageAnalysis.Builder) {
        try {
            Camera2Interop.Extender(builder).setCaptureRequestOption(
                CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                Range(TARGET_ANALYSIS_FPS, TARGET_ANALYSIS_FPS)
            )
        } catch (t: Throwable) {
            Log.w(TAG, "30 fps analysis request not applied: ${t.message}")
        }
    }

    /**
     * Records the frame rate the pipeline actually runs at, not the one we asked for.
     *
     * Stabilization and vendor overrides can rewrite `CONTROL_AE_TARGET_FPS_RANGE`, and a 20 fps
     * report that turns out to be our own 49 ms analyzer holding the ImageProxy is a completely
     * different problem from a sensor stuck at 20 fps. The 1 Hz diagnostics line prints both.
     */
    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun observeAppliedFrameRate(builder: ImageAnalysis.Builder) {
        try {
            Camera2Interop.Extender(builder).setSessionCaptureCallback(
                object : CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureCompleted(
                        session: CameraCaptureSession,
                        request: CaptureRequest,
                        result: TotalCaptureResult
                    ) {
                        val range = result.get(CaptureResult.CONTROL_AE_TARGET_FPS_RANGE)
                        cameraRate.onCaptureResult(
                            range?.lower,
                            range?.upper,
                            result.get(CaptureResult.SENSOR_FRAME_DURATION)
                        )
                    }
                }
            )
        } catch (t: Throwable) {
            Log.w(TAG, "applied frame rate probe not attached: ${t.message}")
        }
    }

    @androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
    private fun <T> enableStabilization(builder: androidx.camera.core.ExtendableBuilder<T>) {
        try {
            val ext = Camera2Interop.Extender(builder)
            ext.setCaptureRequestOption(
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON
            )
            ext.setCaptureRequestOption(
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_ON
            )
        } catch (t: Throwable) {
            Log.w(TAG, "stabilization options not applied: ${t.message}")
        }
    }

    /**
     * Full YOLOv8-style preprocessing on the ViewPort crop:
     * YUV_420_888 -> bilinear resize -> selectable square RGB letterbox -> normalize -> NCHW.
     *
     * CameraX planes are copied into reusable direct buffers before the image is closed.
     * Resize, colour conversion and ONNX execution then stay on [obbExecutor].
     *
     * @param roiX/roiY/roiW/roiH  detection ROI within the crop, in display coordinates.
     *   Only this sub-rectangle is sampled, and the letterbox is computed from its size —
     *   [ObbDetector.letterboxParams] must be given the same `roiW`/`roiH` to invert it.
     */
    @Suppress("LongParameterList")
    private fun submitObbDetection(
        image: AnalysisFrame,
        timestampNs: Long,
        displayW: Int,
        displayH: Int,
        roiX: Int,
        roiY: Int,
        roiW: Int,
        roiH: Int,
        profile: ModelProfile,
        generation: Long,
        workToken: Long
    ) {
        if (!obbBusy.compareAndSet(false, true)) return
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val crop = image.cropRect
        try {
            val frame = obbFrameSnapshotBuffer.copyFrom(
                YuvTensorPreprocessor.Frame(
                    y = yPlane.buffer,
                    u = uPlane.buffer,
                    v = vPlane.buffer,
                    width = image.width,
                    height = image.height,
                    yRowStride = yPlane.rowStride,
                    yPixelStride = yPlane.pixelStride,
                    uRowStride = uPlane.rowStride,
                    uPixelStride = uPlane.pixelStride,
                    vRowStride = vPlane.rowStride,
                    vPixelStride = vPlane.pixelStride
                )
            )
            val geometry = YuvTensorPreprocessor.Geometry(
                rotationDegrees = image.imageInfo.rotationDegrees,
                cropLeft = crop.left,
                cropTop = crop.top,
                cropWidth = crop.width(),
                cropHeight = crop.height(),
                roiX = roiX,
                roiY = roiY,
                roiWidth = roiW,
                roiHeight = roiH,
                modelSize = profile.inputSize
            )
            val region = InputRegion(
                roiW = roiW,
                roiH = roiH,
                offsetX = roiX,
                offsetY = roiY,
                frameW = displayW,
                frameH = displayH
            )
            obbRefreshPolicy.onDetectionSubmitted(timestampNs)
            val enqueuedNs = System.nanoTime()
            val detectFrame = Runnable {
                try {
                    val workerStartedNs = System.nanoTime()
                    val queueMs = (workerStartedNs - enqueuedNs) / 1_000_000f
                    val preprocessStartedNs = System.nanoTime()
                    val preprocessResult = tensorPreprocessor.preprocess(
                        frame = frame,
                        geometry = geometry,
                        output = modelInput
                    )
                    val preprocessMs =
                        (System.nanoTime() - preprocessStartedNs) / 1_000_000f
                    val detection = detector.detect(modelInput, region, profile)
                    val timings = detection.timings.copy(
                        queueMs = queueMs,
                        preprocessMs = preprocessMs,
                        totalMs = preprocessMs + queueMs + detection.timings.totalMs,
                        nativePreprocess = preprocessResult.nativeUsed
                    )
                    frameEpoch.publishIfCurrent(workToken) {
                        if (!closed.get()) {
                            tracker.onDetection(
                                detection.observation,
                                timestampNs,
                                displayW,
                                displayH,
                                timings.totalMs,
                                detection.inputSize,
                                generation,
                                timings
                            )
                        }
                    }
                } catch (t: Throwable) {
                    Log.e(TAG, "obb inference failed", t)
                    if (offline) throw t
                } finally {
                    obbCompletionThrottle.onCompleted(System.nanoTime())
                    obbBusy.set(false)
                }
            }
            if (offline) detectFrame.run() else obbExecutor.execute(detectFrame)
        } catch (t: Throwable) {
            obbBusy.set(false)
            throw t
        }
    }

    /**
     * Builds the 1D axial profile from the Y plane and runs single-frame tick/ball
     * detection. Independent of the tensor path: it samples at native resolution, because
     * a downsampled square tensor smears scale lines together.
     *
     * @return null when there is no usable tube geometry or the profile is unreadable
     */
    private fun readScale(
        image: AnalysisFrame,
        xf: BufferTransform,
        track: TubeTrackResult,
        axis: TubeAxis,
        tsNs: Long,
        stageTimer: TinyPipelineStageTimer,
        onTemporalContextReady: (TemporalFrameContext) -> Unit
    ): ScaleFrameResult? {
        val geometryDelta = tubeGeometryConsumerCursor.advance(axis)

        if (axialSignal.size < axis.lengthPx.toInt() + 1) {
            axialSignal = FloatArray(axis.lengthPx.toInt() + 64)
        }
        val scaleSampleStartedNs = stageTimer.start()
        val n = axialSampler.sample(image, xf, axis, axialSignal)
        stageTimer.recordSince(TinyPipelineStage.SCALE_SAMPLE, scaleSampleStartedNs)
        if (n == 0) return null

        val offset = axialSampler.startOffsetPx
        val config = rulerConfig
        val expectedVisibleCount = config.expectedVisibleCount.takeIf { config.hasExpectedCount } ?: 0f
        val tickDetectionStartedNs = stageTimer.start()
        val detected = tickDetector.detect(
            axialSignal,
            n,
            axis.widthPx,
            expectedVisibleCount
        )
        stageTimer.recordSince(TinyPipelineStage.TICK_DETECTION, tickDetectionStartedNs)
        val r = tickDetector.result
        val bandHalf = axis.widthPx * AxialSampler.BAND_FRAC * 0.5f

        // Feed the ruler even on a failed detection so its dropout clock keeps running; an
        // empty frame is just another rejected one.
        val tickCount = if (detected) r.count else 0
        if (detected) {
            ensureAxialScratch(r.count)
            for (i in 0 until r.count) tickAxialScratch[i] = offset + r.positions[i]
        }
        val rulerStartedNs = stageTimer.start()
        val snap = tickRuler.onFrame(
            ticksIn = tickAxialScratch,
            tickCount = tickCount,
            interpolated = if (detected) r.interpolated else EMPTY_FLAGS,
            medianGapPx = if (detected) r.medianGapPx else 0f,
            axisLengthPx = axis.lengthPx,
            tiltRad = axis.tiltRad,
            fullyInFrame = isFullyInFrame(track),
            signalLen = if (detected) n else 0,
            coarseShiftPx = geometryDelta.coarseShiftPx,
            geometryChanged = geometryDelta.geometryChanged,
            timestampNs = tsNs,
            geometryScale = geometryDelta.geometryScale
        )
        stageTimer.recordSince(TinyPipelineStage.RULER_UPDATE, rulerStartedNs)
        // Dense band tracking is a disabled diagnostic path in production. Sampling all
        // 17 bands (and three luma probes per sample) here only blocks the camera analyzer;
        // retain the path while an explicit debug capture is active, but do not pay for it
        // during normal TinyHeatmapNet operation.
        //
        // "an explicit debug capture is active" used to mean `capturesDenseDiagnostics`,
        // which is `detailRef == FULL` -- and `startTrackingSession`, which runs on *every*
        // measurement, sets exactly that. So the sampler ran on every frame of every
        // measurement, and so did `denseBallTracker.propose`, even though `propose` returns
        // null unconditionally when `outputEnabled` is false (DenseRegisteredBallTracker
        // line 254) and that flag has been hard-coded false here since TinyHeatmapNet became
        // the sole detector. Nothing downstream ever read the result.
        //
        // Measured cost on an SM8650: `dense_diagnostics_ms` p50 24.2 ms per frame -- the
        // largest single stage in the pipeline, nearly four times the parent inference.
        // Gating the sampling on the same flag that gates the output makes the scaffold
        // genuinely dormant rather than merely ignored. Set the flag true to restore the
        // whole path for tracker debugging.
        val denseOutputEnabled = false
        val denseSamplingEnabled =
            denseOutputEnabled && temporalDebugCaptureRecorder.capturesDenseDiagnostics
        val denseSamplingStartedNs = stageTimer.start()
        val denseSignalLength = if (denseSamplingEnabled) {
            ensureDenseBandCapacity(axis.lengthPx.toInt() + 64)
            denseBandSampler.sample(
                image = image,
                transform = xf,
                axis = axis,
                output = denseBandSignals
            )
        } else {
            0
        }
        stageTimer.recordSince(TinyPipelineStage.DENSE_DIAGNOSTICS, denseSamplingStartedNs)

        // PC heatmap-C publishes the current ROI result on the current video frame. Run the
        // detector after this frame's ruler snapshot exists but before reading its published
        // result, so detection, geometry, ruler coordinates and persistence share one timestamp.
        val temporalContext = TemporalFrameContext(
            timestampNs = tsNs,
            axis = axis,
            ruler = snap,
            axialMaxPx = offset + n,
            geometryCaptureTimestampNs = track.geometryCaptureTimestampNs,
            geometryRevision = track.geometryRevision
        )
        onTemporalContextReady(temporalContext)

        // TickDetector still locates a wide dark trough solely to mask it out of the tick
        // candidates. That protects ruler stability without exposing a legacy 1D ball result.
        val resultSelectionStartedNs = stageTimer.start()
        val publishedTemporal = temporalBallRef.get()?.takeIf {
            isCurrentTemporalResultGeneration(
                resultGeneration = it.generation,
                analysisGeneration = analysisTemporalGeneration,
                activeGeneration = temporalGeneration.get()
            ) && tsNs >= it.context.timestampNs
        }
        val previewMaxAgeMs = TemporalResultDisplayPolicy.maxAgeMs(
            temporalStepFps,
            publishedTemporal?.detection?.inferenceMs
        )
        val usage = publishedTemporal?.let {
            temporalResultUsage.resolve(
                frameTimestampNs = tsNs,
                resultTimestampNs = it.context.timestampNs,
                previewMaxAgeMs = previewMaxAgeMs
            )
        }
        val acceptedPublishedDetection = publishedTemporal?.detection?.takeIf {
            usage?.preview == true || usage?.record == true
        }?.let { detection ->
            selectTemporalDetection(detection, publishedTemporal.context)
        }
        val displayTemporalDetection = acceptedPublishedDetection?.takeIf {
            usage?.preview == true
        }
        val temporalPoint = displayTemporalDetection?.let {
            val published = requireNotNull(publishedTemporal)
            temporalPoint(it, published.context, axis)
        }
        val recordingTemporal = publishedTemporal?.takeIf { usage?.record == true }
        val recordingDetection = acceptedPublishedDetection?.takeIf { recordingTemporal != null }
        val capturedTemporalPoint = recordingDetection?.let {
            temporalPoint(it, requireNotNull(recordingTemporal).context, recordingTemporal.context.axis)
        }
        stageTimer.recordSince(TinyPipelineStage.RESULT_SELECTION, resultSelectionStartedNs)
        val denseAnchorCandidate = if (
            recordingTemporal != null && recordingDetection != null && capturedTemporalPoint != null
        ) {
            val anchorTicks = recordingTemporal.context.ruler.temporalPositionTicks(
                capturedTemporalPoint.axialPx,
                rulerConfig.tickSpacingMm
            )
            val anchorLateral = capturedTemporalPoint.lateralPx /
                (recordingTemporal.context.axis.widthPx * 0.5f)
            if (anchorTicks != null && anchorLateral.isFinite()) {
                DenseBallAnchor(
                    timestampNs = recordingTemporal.context.timestampNs,
                    positionTicks = anchorTicks,
                    lateralFraction = anchorLateral,
                    confidence = recordingDetection.score
                )
            } else {
                null
            }
        } else {
            null
        }
        var denseAnchor: DenseBallAnchor? = null
        val temporalBall = if (temporalPoint != null) {
            RegisteredBallResult(
                ballAxialPx = temporalPoint.axialPx,
                source = com.cupet.fallingball.vision.BallDetectionSource.TINY_HEATMAP,
                confidence = requireNotNull(displayTemporalDetection).score,
                backgroundReady = publishedTemporal?.backgroundReady == true
            )
        } else {
            RegisteredBallResult(
                ballAxialPx = null,
                source = com.cupet.fallingball.vision.BallDetectionSource.NONE,
                confidence = 0f,
                backgroundReady = false
            )
        }
        if (recordingTemporal != null) {
            val recordingBall = if (
                recordingDetection != null && capturedTemporalPoint != null
            ) {
                RegisteredBallResult(
                    ballAxialPx = capturedTemporalPoint.axialPx,
                    source = com.cupet.fallingball.vision.BallDetectionSource.TINY_HEATMAP,
                    confidence = recordingDetection.score,
                    backgroundReady = recordingTemporal.backgroundReady
                )
            } else {
                RegisteredBallResult(
                    ballAxialPx = null,
                    source = com.cupet.fallingball.vision.BallDetectionSource.NONE,
                    confidence = 0f,
                    backgroundReady = false
                )
            }
            val capturedWallContact = if (capturedTemporalPoint != null) {
                wallContactFor(capturedTemporalPoint, recordingTemporal.context.ruler)
            } else {
                wallContactFor(null, recordingTemporal.context.ruler)
            }
            val gateStartedNs = stageTimer.start()
            val temporalAccepted = recordObservation(
                detection = recordingBall,
                axialMaxPx = recordingTemporal.context.axialMaxPx,
                snap = recordingTemporal.context.ruler,
                timestampNs = recordingTemporal.context.timestampNs,
                ballPoint = capturedTemporalPoint,
                wallContact = capturedWallContact,
                tubeWidthPx = recordingTemporal.context.axis.widthPx,
                debugCaptureStepId = recordingTemporal.context.debugCaptureStepId
            )
            stageTimer.recordSince(TinyPipelineStage.GATE_AND_SAMPLE_RECORD, gateStartedNs)
            denseAnchor = denseAnchorCandidate.takeIf { temporalAccepted }
            analysisBallSource = recordingBall.source
        }

        val denseGateStatus = fallTrackingGate.status()
        val denseMotionSeedAvailable = denseGateStatus.hasUsableMotionTrajectory
        // `denseOutputEnabled` is declared above, next to the sampler it now also gates.
        val denseDiagnosticsStartedNs = stageTimer.start()
        val denseDetection = if (denseSamplingEnabled && denseSignalLength > 0) {
            denseBallTracker.propose(
                signals = denseBandSignals,
                signalLength = denseSignalLength,
                signalOffsetPx = denseBandSampler.startOffsetPx,
                ruler = snap,
                tickSpacingMm = rulerConfig.tickSpacingMm,
                tubeWidthPx = axis.widthPx,
                timestampNs = tsNs,
                learnBackground = session?.isActive != true ||
                    fallTrackingGate.allowsBackgroundLearning,
                outputEnabled = denseOutputEnabled,
                anchor = denseAnchor.takeIf { denseMotionSeedAvailable },
                committedTrackRetentionMs = denseGateStatus.lossTimeoutMs
            )
        } else {
            null
        }
        val densePoint = denseDetection?.let {
            com.cupet.fallingball.vision.TubeBallPoint(
                x = axis.pointX(it.axialPx, it.lateralPx),
                y = axis.pointY(it.axialPx, it.lateralPx),
                axialPx = it.axialPx,
                lateralPx = it.lateralPx
            )
        }
        val denseBall = denseDetection?.let {
            RegisteredBallResult(
                ballAxialPx = it.axialPx,
                source = com.cupet.fallingball.vision.BallDetectionSource.LOCAL_TRACKER,
                confidence = it.confidence,
                backgroundReady = true
            )
        }
        var denseAccepted = false
        if (denseDetection != null && denseBall != null && densePoint != null) {
            val denseWallContact = wallContactFor(densePoint, snap)
            val gateStartedNs = stageTimer.start()
            denseAccepted = recordObservation(
                detection = denseBall,
                axialMaxPx = offset + n,
                snap = snap,
                timestampNs = tsNs,
                ballPoint = densePoint,
                wallContact = denseWallContact,
                tubeWidthPx = axis.widthPx
            )
            stageTimer.recordSince(TinyPipelineStage.GATE_AND_SAMPLE_RECORD, gateStartedNs)
            if (denseAccepted) {
                check(denseBallTracker.commit(denseDetection)) {
                    "Accepted dense proposal was no longer current"
                }
                analysisBallSource = denseBall.source
            } else {
                denseBallTracker.discard(denseDetection)
            }
        } else if (recordingTemporal == null && !temporalBallEnabled) {
            val gateStartedNs = stageTimer.start()
            recordObservation(
                detection = RegisteredBallResult(
                    ballAxialPx = null,
                    source = com.cupet.fallingball.vision.BallDetectionSource.NONE,
                    confidence = 0f,
                    backgroundReady = false
                ),
                axialMaxPx = offset + n,
                snap = snap,
                timestampNs = tsNs,
                ballPoint = null,
                wallContact = wallContactFor(null, snap),
                tubeWidthPx = axis.widthPx
            )
            stageTimer.recordSince(TinyPipelineStage.GATE_AND_SAMPLE_RECORD, gateStartedNs)
            analysisBallSource = com.cupet.fallingball.vision.BallDetectionSource.NONE
        }
        if (session?.isActive == true) {
            temporalDebugCaptureRecorder.recordDense(
                timestampNs = tsNs,
                outputEnabled = denseOutputEnabled,
                motionSeedAvailable = denseMotionSeedAvailable,
                diagnostics = denseBallTracker.lastDiagnostics,
                proposal = denseDetection,
                accepted = denseAccepted
            )
        }
        stageTimer.recordSince(TinyPipelineStage.DENSE_DIAGNOSTICS, denseDiagnosticsStartedNs)

        val displayPoint = densePoint.takeIf { denseAccepted } ?: temporalPoint
        val ball = denseBall.takeIf { denseAccepted } ?: temporalBall
        val ballAxial = ball.ballAxialPx
        val wallContact = wallContactFor(displayPoint, snap)

        val visibleRuler = snap.clippedToAxis(axis.lengthPx)
        return ScaleFrameResult(
            reading = ScaleReading(
                axis = axis,
                axialPx = if (visibleRuler.isUsable) visibleRuler.axialPx else FloatArray(0),
                ordinals = if (visibleRuler.isUsable) visibleRuler.ordinals else IntArray(0),
                ballAxialPx = ballAxial,
                ballCenter = displayPoint,
                ballConfidence = ball.confidence,
                ballSource = ball.source,
                wallContact = wallContact,
                medianGapPx = if (snap.isUsable) snap.medianGapPx else 0f,
                bandHalfWidthPx = bandHalf,
                state = snap.state,
                invalidReason = snap.invalidReason
            ),
            temporalContext = temporalContext
        )
    }

    private fun temporalPoint(
        detection: TemporalBallDetection,
        context: TemporalFrameContext,
        displayAxis: TubeAxis
    ) = BallGeometry.canonicalToDisplay(
        roiX = context.stabilization.stabilizedToSourceX(detection.roiX),
        roiY = context.stabilization.stabilizedToSourceY(detection.roiY),
        roiWidth = TubeRoiSampler.OUTPUT_WIDTH,
        roiHeight = TubeRoiSampler.OUTPUT_HEIGHT,
        axis = displayAxis
    )

    private fun selectTemporalDetection(
        detection: TemporalBallDetection,
        context: TemporalFrameContext
    ): TemporalBallDetection? {
        val primary = mapTemporalPeak(
            TemporalBallPeak(detection.roiX, detection.roiY, detection.score),
            context
        )
        val alternatives = detection.alternativePeaks.map { mapTemporalPeak(it, context) }
        val selected = temporalDetectionSelector.select(
            primary = primary,
            alternatives = alternatives,
            gate = fallTrackingGate,
            timestampNs = context.timestampNs
        ) ?: return null
        return detection.copy(
            roiX = selected.peak.roiX,
            roiY = selected.peak.roiY,
            score = selected.peak.score,
            alternativePeaks = emptyList()
        )
    }

    private fun mapTemporalPeak(
        peak: TemporalBallPeak,
        context: TemporalFrameContext
    ): MappedTemporalPeak {
        val point = BallGeometry.canonicalToDisplay(
            roiX = context.stabilization.stabilizedToSourceX(peak.roiX),
            roiY = context.stabilization.stabilizedToSourceY(peak.roiY),
            roiWidth = TubeRoiSampler.OUTPUT_WIDTH,
            roiHeight = TubeRoiSampler.OUTPUT_HEIGHT,
            axis = context.axis
        )
        val halfWidth = (context.axis.widthPx * 0.5f).takeIf { it.isFinite() && it > 0f }
        val lateralFraction = halfWidth?.let { point.lateralPx / it }
        return MappedTemporalPeak(
            peak = peak,
            axialPx = point.axialPx,
            axialMaxPx = context.axialMaxPx,
            positionTicks = context.ruler.temporalPositionTicks(
                point.axialPx,
                rulerConfig.tickSpacingMm
            ),
            trajectoryLateralFraction = lateralFraction?.takeIf { it.isFinite() }
        )
    }

    private fun wallContactFor(
        point: com.cupet.fallingball.vision.TubeBallPoint?,
        snap: RulerSnapshot
    ): com.cupet.fallingball.vision.WallContactResult {
        if (point == null) {
            return com.cupet.fallingball.vision.WallContactResult(
                com.cupet.fallingball.vision.WallContactState.UNKNOWN,
                null,
                null
            )
        }
        val pixelsPerMm = pixelsPerMm(snap)
        return BallGeometry.classifyWallContact(
            lateralPx = point.lateralPx,
            tubeInnerWidthPx = tubeInnerDiameterMm?.let { diameter ->
                pixelsPerMm?.times(diameter.toFloat())
            },
            sphereRadiusPx = sphereRadiusMm?.let { radius ->
                pixelsPerMm?.times(radius.toFloat())
            },
            pixelsPerMm = pixelsPerMm,
            contactTolerancePx = pixelsPerMm
                ?.times(BallGeometry.CONTACT_TOLERANCE_MM)
                ?: 2f
        )
    }

    private fun pixelsPerMm(snap: RulerSnapshot): Float? =
        if (snap.isUsable && snap.medianGapPx > 0f && rulerConfig.tickSpacingMm > 0f) {
            snap.medianGapPx / rulerConfig.tickSpacingMm
        } else {
            null
        }

    private fun submitTemporalBall(
        image: AnalysisFrame,
        transform: BufferTransform,
        axis: TubeAxis,
        context: TemporalFrameContext,
        workToken: Long,
        captureTimestampNs: Long,
        frameCallbackStartedAtNs: Long,
        stageTimer: TinyPipelineStageTimer
    ) {
        if (!temporalBallEnabled) {
            check(!offline) { "The ball detector is unavailable" }
            recordTemporalSchedule(context, TemporalScheduleOutcome.MODEL_UNAVAILABLE, 0L)
            return
        }

        // Keep camera cadence bookkeeping on the analysis executor, but never hold this
        // executor while the TinyHeatmapNet worker runs. The one-slot queue below deliberately
        // drops stale work so model latency cannot turn into an ever-growing camera backlog.
        temporalSequenceBuffer.observe(context.timestampNs)
        updateTemporalStepFps(context.timestampNs)
        val geometryStartedNs = stageTimer.start()
        val geometry = tubeRoiSampler.geometry(axis, context.geometryRevision)
        stageTimer.recordSince(TinyPipelineStage.ROI_GEOMETRY, geometryStartedNs)
        val roiSampleStartedNs = stageTimer.start()
        val sampled = tubeRoiSampler.sample(image, transform, geometry, tubeRoiBuffer)
        stageTimer.recordSince(TinyPipelineStage.ROI_SAMPLE, roiSampleStartedNs)
        if (!sampled) {
            check(!offline) { "Unable to sample the video tube at its original resolution" }
            recordTemporalSchedule(context, TemporalScheduleOutcome.ROI_SAMPLE_FAILED, 0L)
            invalidateTemporalSequence(TemporalResetReason.ROI_SAMPLE_FAILED, context.timestampNs)
            return
        }
        val rawCopyStartedNs = stageTimer.start()
        // The raw buffer is consumed before the next offline frame. Stabilized buffers below
        // must remain separate because TinyHeatmapInputHistory retains preceding frames.
        val rawRoi = if (offline) offlineRawRoi else
            ByteArray(TubeRoiSampler.OUTPUT_WIDTH * TubeRoiSampler.OUTPUT_HEIGHT)
        tubeRoiBuffer.duplicate().apply {
            position(0)
            limit(rawRoi.size)
        }.get(rawRoi)
        stageTimer.recordSince(TinyPipelineStage.RAW_ROI_COPY, rawCopyStartedNs)
        val sourceFrameIndex = if (offline) offlineSourceIndex else nextTinySourceFrameIndex()
        val captureFrameIndex = if (offline) offlineSourceIndex else nextTinyCaptureFrameIndex(context.timestampNs)
        // Stabilize every camera capture before latest-only scheduling drops work. The model
        // worker still runs only the newest frame, but its causal motion history remains the
        // exact preceding camera frames used by the PC preprocessing contract.
        val stabilizationStartedNs = stageTimer.start()
        val stabilization = tinyRoiStabilizer.observe(
            ByteBuffer.wrap(rawRoi),
            stabilizedRoiBuffer,
            context.timestampNs
        )
        stageTimer.recordSince(TinyPipelineStage.ROI_STABILIZATION, stabilizationStartedNs)
        val stableCopyStartedNs = stageTimer.start()
        val stableRoi = ByteArray(rawRoi.size)
        stabilizedRoiBuffer.duplicate().apply {
            position(0)
            limit(stableRoi.size)
        }.get(stableRoi)
        stageTimer.recordSince(TinyPipelineStage.STABLE_ROI_COPY, stableCopyStartedNs)
        val stableGeometry = geometry.withStabilization(stabilization)
        val stableContext = context.copy(stabilization = stabilization)
        val historyStartedNs = stageTimer.start()
        val historyFrames = tinyInputHistory.appendAndSnapshot(
            TinyHeatmapHistoryFrame(
                bytes = stableRoi,
                geometry = stableGeometry,
                timestampNs = context.timestampNs,
                sourceFrameIndex = sourceFrameIndex
            )
        )
        stageTimer.recordSince(TinyPipelineStage.HISTORY_SNAPSHOT, historyStartedNs)
        val enqueueStartedNs = stageTimer.start()
        val generation = analysisTemporalGeneration
        val sessionEligible = session?.acceptsCaptureTimestamp(context.timestampNs) == true
        val queuedAtNs = System.nanoTime()
        stageTimer.recordSince(TinyPipelineStage.ENQUEUE, enqueueStartedNs)
        val pending = PendingTemporalFrame(
            rawRoi = rawRoi,
            stableRoi = stableRoi,
            geometry = stableGeometry,
            context = stableContext,
            transform = transform,
            sourceFrameIndex = sourceFrameIndex,
            captureFrameIndex = captureFrameIndex,
            historyFrames = historyFrames,
            generation = generation,
            workToken = workToken,
            queuedAtNs = queuedAtNs,
            captureTimestampNs = captureTimestampNs,
            frameCallbackStartedAtNs = frameCallbackStartedAtNs,
            pipelineTimings = stageTimer.snapshot(),
            sessionEligible = sessionEligible
        )
        if (offline) {
            tinyThroughput.recordSubmitted()
            processTemporalFrame(pending)
            offlineInferenceFrames++
            return
        }
        val replaced = pendingTemporalFrame.getAndSet(pending)
        tinyThroughput.recordSubmitted()
        if (replaced != null) {
            // The replaced frame never reaches the native worker. Count it explicitly so a
            // benchmark cannot mistake camera-callback rate for useful model throughput.
            tinyThroughput.recordDropped()
        }
        temporalBusy.set(true)
        recordTemporalSchedule(
            context,
            if (replaced == null) TemporalScheduleOutcome.SUBMITTED
            else TemporalScheduleOutcome.TEMPORAL_BUSY,
            0L
        )
        scheduleTemporalWorker()
    }

    private fun nextTinySourceFrameIndex(): Int =
        synchronized(tinySequenceLock) {
            tinySourceFrameIndex = nextTinyModelFrameIndex(tinySourceFrameIndex)
            tinySourceFrameIndex
        }

    /** Advances camera-frame numbering used by the final whole-sequence decode. */
    private fun nextTinyCaptureFrameIndex(timestampNs: Long): Int =
        synchronized(tinySequenceLock) {
            tinyCaptureFrameIndex = nextTinyCaptureFrameIndex(
                previousIndex = tinyCaptureFrameIndex,
                previousTimestampNs = tinyLastCaptureTimestampNs,
                timestampNs = timestampNs,
                framePeriodNs = FRAME_PERIOD_NS
            )
            tinyLastCaptureTimestampNs = timestampNs
            tinyCaptureFrameIndex
        }

    private fun scheduleTemporalWorker() {
        if (!temporalWorkerScheduled.compareAndSet(false, true)) return
        runCatching {
            temporalExecutor.execute(::drainTemporalFrames)
        }.onFailure {
            temporalWorkerScheduled.set(false)
            if (pendingTemporalFrame.getAndSet(null) != null) {
                tinyThroughput.recordDropped()
            }
            temporalBusy.set(false)
            if (!closed.get()) Log.w(TAG, "Unable to schedule TinyHeatmapNet worker", it)
        }
    }

    private fun drainTemporalFrames() {
        while (true) {
            val pending = pendingTemporalFrame.getAndSet(null)
            if (pending != null) {
                processTemporalFrame(pending)
                continue
            }
            temporalWorkerScheduled.set(false)
            if (pendingTemporalFrame.get() == null ||
                !temporalWorkerScheduled.compareAndSet(false, true)
            ) {
                temporalBusy.set(false)
                return
            }
        }
    }

    private fun processTemporalFrame(pending: PendingTemporalFrame) {
        if (pending.generation != temporalGeneration.get()) {
            tinyThroughput.recordDropped()
            return
        }
        val stageTimer = TinyPipelineStageTimer.from(pending.pipelineTimings)
        val workerStartedAtNs = System.nanoTime()
        stageTimer.addNanoseconds(
            TinyPipelineStage.QUEUE_WAIT,
            (workerStartedAtNs - pending.queuedAtNs).coerceAtLeast(0L)
        )
        if (temporalWorkerGeneration != pending.generation) {
            val resetStartedNs = stageTimer.start()
            temporalBallDetector.reset()
            synchronized(tinySequenceLock) { tinySequenceFrames.clear() }
            temporalWorkerGeneration = pending.generation
            temporalWorkerDiagnosticOrdinal = 0
            stageTimer.recordSince(TinyPipelineStage.GENERATION_RESET, resetStartedNs)
        }
        var captureStepId: Long? = null
        var inferenceStartedAtNs = 0L
        try {
            val primeStartedNs = stageTimer.start()
            temporalBallDetector.primeHistory(pending.historyFrames)
            stageTimer.recordSince(TinyPipelineStage.PRIME_HISTORY, primeStartedNs)
            inferenceStartedAtNs = System.nanoTime()
            val tiny = temporalBallDetector.inferCandidates(
                ByteBuffer.wrap(pending.stableRoi),
                pending.context.timestampNs,
                pending.geometry,
                ByteBuffer.wrap(pending.stableRoi),
                pending.sourceFrameIndex,
                stageTimer
            )
            val inferenceCompletedAtNs = System.nanoTime()
            tinyRuntimeRef.set(tiny.runtime)
            val inferenceMs = (inferenceCompletedAtNs - inferenceStartedAtNs) / 1_000_000f
            val sequenceStartedNs = stageTimer.start()
            temporalSequenceBuffer.append(
                ByteBuffer.wrap(pending.stableRoi),
                pending.context.timestampNs
            )
            val sequenceCandidates = tiny.candidates.map {
                TinyHeatmapSequenceCandidate(
                    sourceX = it.sourceX,
                    sourceY = it.sourceY,
                    logit = it.logit
                )
            }
            if (pending.sessionEligible) {
                synchronized(tinySequenceLock) {
                    check(!offline || tinySequenceFrames.size < 100_000) {
                        "所选片段超出测量记录容量，请选择更短的片段"
                    }
                    tinySequenceFrames += TinySequenceFrameRecord(
                        sourceFrameIndex = pending.sourceFrameIndex,
                        captureFrameIndex = pending.captureFrameIndex,
                        timestampNs = pending.context.timestampNs,
                        context = pending.context,
                        candidates = sequenceCandidates
                    )
                }
            }
            stageTimer.recordSince(TinyPipelineStage.SEQUENCE_BOOKKEEPING, sequenceStartedNs)
            val diagnosticCaptureStartedNs = stageTimer.start()
            captureStepId = temporalDebugCaptureRecorder.recordInput(
                raw = ByteBuffer.wrap(pending.rawRoi),
                stable = ByteBuffer.wrap(pending.stableRoi),
                timestampNs = pending.context.timestampNs,
                axis = pending.context.axis,
                ruler = pending.context.ruler,
                transform = pending.transform,
                stabilization = pending.context.stabilization,
                geometryCaptureTimestampNs = pending.context.geometryCaptureTimestampNs,
                geometryRevision = pending.context.geometryRevision,
                temporalGeneration = pending.generation
            )
            stageTimer.recordSince(TinyPipelineStage.DIAGNOSTIC_CAPTURE, diagnosticCaptureStartedNs)
            val resultBuildStartedNs = stageTimer.start()
            val rankedCandidates = tiny.candidates.sortedWith(
                compareByDescending<TinyHeatmapFrameCandidate> { it.logit }
                    .thenByDescending { it.parentScore }
            )
            val preview = rankedCandidates.firstOrNull()
            val detection = preview?.let {
                TemporalBallDetection(
                    roiX = it.roiX,
                    roiY = it.roiY,
                    score = sigmoid(it.logit),
                    inferenceMs = inferenceMs,
                    alternativePeaks = rankedCandidates.drop(1).map { alternative ->
                        TemporalBallPeak(
                            roiX = alternative.roiX,
                            roiY = alternative.roiY,
                            score = sigmoid(alternative.logit)
                        )
                    },
                    runtime = tiny.runtime
                )
            }
            stageTimer.recordSince(TinyPipelineStage.RESULT_BUILD, resultBuildStartedNs)
            temporalWorkerDiagnosticOrdinal += 1
            val publishStartedNs = stageTimer.start()
            frameEpoch.publishIfCurrent(pending.workToken) {
                if (!closed.get() && temporalGeneration.get() == pending.generation) {
                    temporalBallRef.set(
                        PublishedTemporalBall(
                            detection = detection,
                            context = pending.context.copy(debugCaptureStepId = captureStepId),
                            framesSeen = tiny.framesSeen,
                            backgroundReady = true,
                            generation = pending.generation
                        )
                    )
                }
            }
            stageTimer.recordSince(TinyPipelineStage.RESULT_PUBLISH, publishStartedNs)
            val completedAtNs = System.nanoTime()
            stageTimer.recordElapsed(
                TinyPipelineStage.WORKER_TOTAL,
                workerStartedAtNs,
                completedAtNs
            )
            stageTimer.recordElapsed(
                TinyPipelineStage.ANALYZER_TO_PUBLISH,
                pending.frameCallbackStartedAtNs,
                completedAtNs
            )
            tinyThroughput.recordCompleted(completedAtNs)
            temporalDebugCaptureRecorder.recordDetection(
                stepToken = captureStepId,
                framesSeen = tiny.framesSeen,
                detection = detection,
                generation = pending.generation,
                cameraTimestampNs = pending.captureTimestampNs,
                inferenceMs = inferenceMs,
                queueMs = (workerStartedAtNs - pending.queuedAtNs).coerceAtLeast(0L) /
                    1_000_000f,
                queuedAtNs = pending.queuedAtNs,
                startedAtNs = inferenceStartedAtNs,
                completedAtNs = completedAtNs,
                runtime = tiny.runtime,
                throughput = tinyThroughput.snapshot(),
                classicalDiagnostics = null,
                pipelineTimings = stageTimer.snapshot()
            )
        } catch (error: Throwable) {
            tinyThroughput.recordFailed()
            Log.e(TAG, "TinyHeatmapNet detection failed", error)
            val completedAtNs = System.nanoTime()
            temporalDebugCaptureRecorder.recordInferenceError(
                stepToken = captureStepId,
                generation = pending.generation,
                queuedAtNs = pending.queuedAtNs,
                startedAtNs = inferenceStartedAtNs,
                completedAtNs = completedAtNs,
                message = error.message ?: error.javaClass.simpleName
            )
            temporalBallError = true
            temporalBallEnabled = false
            temporalBallRef.set(null)
            if (offline) throw error
        }
    }

    private fun recordTemporalSchedule(
        context: TemporalFrameContext,
        outcome: TemporalScheduleOutcome,
        cooldownRemainingNs: Long
    ) {
        if (session?.acceptsCaptureTimestamp(context.timestampNs) == true) {
            lightweightTrackingDiagnostics?.recordSchedule(outcome)
        }
        temporalDebugCaptureRecorder.recordSchedule(
            timestampNs = context.timestampNs,
            outcome = outcome,
            ruler = context.ruler,
            geometryAgeNs = (context.timestampNs - context.geometryCaptureTimestampNs)
                .coerceAtLeast(0L),
            generation = analysisTemporalGeneration,
            loadLevel = loadProfileRef.get().level,
            thermal = thermalSnapshotRef.get(),
            yoloBusy = obbBusy.get(),
            temporalBusy = temporalBusy.get(),
            cooldownRemainingNs = cooldownRemainingNs
        )
    }

    /** Invalidates both buffered input and any result still running for the old window. */
    private fun invalidateTemporalSequence(reason: TemporalResetReason, timestampNs: Long?) {
        temporalSequenceBuffer.reset()
        invalidateTemporalWorkerState(reason, timestampNs)
    }

    /** Discards detector state without erasing camera-cadence diagnostics. */
    private fun invalidateTemporalWorkerState(reason: TemporalResetReason, timestampNs: Long?) {
        temporalBallRef.set(null)
        temporalDetectionSelector.reset()
        val previousGeneration = temporalGeneration.get()
        analysisTemporalGeneration = temporalGeneration.incrementAndGet()
        temporalDebugCaptureRecorder.recordReset(
            reason,
            timestampNs,
            previousGeneration,
            analysisTemporalGeneration
        )
        temporalResultUsage.reset()
        temporalStepFps = 0f
        lastTemporalStepNs = 0L
    }

    private fun publishBallDiagnostics(timestampNs: Long, trackedAxisUsable: Boolean) {
        val sequence = temporalSequenceBuffer.status()
        val published = temporalBallRef.get()?.takeIf {
            isCurrentTemporalResultGeneration(
                resultGeneration = it.generation,
                analysisGeneration = analysisTemporalGeneration,
                activeGeneration = temporalGeneration.get()
            )
        }
        val ageMs = published?.takeIf { timestampNs >= it.context.timestampNs }
            ?.let { (timestampNs - it.context.timestampNs) / 1_000_000L }
        val stage = when {
            temporalBallError -> TemporalPipelineStage.ERROR
            !temporalBallEnabled -> TemporalPipelineStage.UNAVAILABLE
            !trackedAxisUsable -> TemporalPipelineStage.WAITING_FOR_TUBE
            published == null || !published.backgroundReady ->
                TemporalPipelineStage.WARMING_UP
            else -> TemporalPipelineStage.RUNNING
        }
        val runtime = published?.detection?.runtime ?: tinyRuntimeRef.get()
        val throughput = tinyThroughput.snapshot()
        ballDiagnosticsRef.set(
            BallTrackingDiagnostics(
                stage = stage,
                sequence = sequence,
                modelInputFps = temporalStepFps,
                score = published?.detection?.score,
                threshold = TinyHeatmapSequenceDecoder.CANDIDATE_THRESHOLD,
                inferenceMs = published?.detection?.inferenceMs,
                resultAgeMs = ageMs,
                selectedSource = analysisBallSource,
                gate = fallTrackingGate.status(),
                recordedSamples = session?.sampleCount ?: 0,
                runtime = runtime,
                throughput = throughput
            )
        )
    }

    private fun updateTemporalStepFps(timestampNs: Long) {
        val previousTimestampNs = lastTemporalStepNs
        if (lastTemporalStepNs != 0L && timestampNs > lastTemporalStepNs) {
            val seconds = (timestampNs - lastTemporalStepNs) / 1_000_000_000.0
            val instantaneous = (1.0 / seconds).toFloat()
            temporalStepFps = if (temporalStepFps <= 0f) {
                instantaneous
            } else {
                temporalStepFps * 0.8f + instantaneous * 0.2f
            }
        }
        lastTemporalStepNs = timestampNs
        val observedPeriodMs = temporalStepFps.takeIf { it.isFinite() && it > 0f }
            ?.let { (1_000f / it).roundToInt().toLong() }
            ?: CLASSICAL_DEFAULT_PERIOD_MS
        val expectedPeriodMs = if (offline && previousTimestampNs != 0L) {
            // Offline cadence follows adjacent video PTS, including variable-rate intervals.
            ((timestampNs - previousTimestampNs) / 1_000_000L).coerceAtLeast(1L)
        } else {
            observedPeriodMs.coerceIn(33L, 250L)
        }
        fallTrackingGate.setExpectedObservationIntervalMs(expectedPeriodMs)
    }

    private fun logRuntimeDiagnostics(result: TubeTrackResult) {
        val nowNs = System.nanoTime()
        if (lastPerformanceLogElapsedNs != 0L &&
            nowNs - lastPerformanceLogElapsedNs < PERFORMANCE_LOG_INTERVAL_NS
        ) {
            return
        }
        lastPerformanceLogElapsedNs = nowNs
        val thermal = thermalSnapshotRef.get()
        val load = loadProfileRef.get()
        val ball = ballDiagnosticsRef.get()
        val pipeline = result.pipelineTimings
        val message = String.format(
                Locale.ROOT,
                "load=%s thermal=%s tempC=%s camera=%.1ffps yolo=%.1ffps/%dpx " +
                    "yoloMs=%.1f pre=%.1f ort=%.1f post=%.1f queue=%.1f " +
                    "classical=%.1ffps tinyBackend=%s/%s verified=%s gpu=%s placement=%s tinyFps=%.1f " +
                    "tinyWork=%d/%d/%d/%d tinyMs=%s parentP50=%s parentP95=%s " +
                    "tinyPre=%s tinyParent=%s tinyPost=%s tinyRerank=%s motionHit=%.2f " +
                    "score=%s threshold=%.2f inferMs=%s ageMs=%s " +
                    "ruler=%s gate=%s reject=%s recorded=%d cam=%s",
                load.level,
                thermal?.signal ?: ThermalSignal.NORMAL,
                thermal?.batteryTemperatureC?.let { "%.1f".format(Locale.ROOT, it) } ?: "unknown",
                result.fps,
                result.inferenceFps,
                result.modelInputSize,
                result.inferenceMs,
                pipeline?.preprocessMs ?: 0f,
                pipeline?.inferenceRunMs ?: 0f,
                pipeline?.postprocessMs ?: 0f,
                pipeline?.queueMs ?: 0f,
                ball?.modelInputFps ?: 0f,
                ball?.runtime?.backendName ?: "unknown",
                ball?.runtime?.computeUnit ?: "unknown",
                ball?.runtime?.placementVerified ?: false,
                ball?.runtime?.verifiedGpu ?: false,
                ball?.runtime?.placementSummary ?: "unverified",
                ball?.throughput?.effectiveFps ?: 0f,
                ball?.throughput?.submittedCount ?: 0,
                ball?.throughput?.completedCount ?: 0,
                ball?.throughput?.droppedCount ?: 0,
                ball?.throughput?.failedCount ?: 0,
                ball?.runtime?.totalMs?.let { "%.1f".format(Locale.ROOT, it) } ?: "none",
                ball?.runtime?.parentLatencyStats?.p50Ms
                    ?.let { "%.1f".format(Locale.ROOT, it) } ?: "none",
                ball?.runtime?.parentLatencyStats?.p95Ms
                    ?.let { "%.1f".format(Locale.ROOT, it) } ?: "none",
                ball?.runtime?.preprocessMs
                    ?.let { "%.1f".format(Locale.ROOT, it) } ?: "none",
                ball?.runtime?.parentInferenceMs
                    ?.let { "%.1f".format(Locale.ROOT, it) } ?: "none",
                ball?.runtime?.postprocessMs
                    ?.let { "%.1f".format(Locale.ROOT, it) } ?: "none",
                ball?.runtime?.rerankerMs
                    ?.let { "%.1f".format(Locale.ROOT, it) } ?: "none",
                ball?.runtime?.motionHistoryHitRate ?: 1f,
                ball?.score?.let { "%.3f".format(Locale.ROOT, it) } ?: "none",
                ball?.threshold ?: TinyHeatmapSequenceDecoder.CANDIDATE_THRESHOLD,
                ball?.inferenceMs?.let { "%.1f".format(Locale.ROOT, it) } ?: "none",
                ball?.resultAgeMs?.toString() ?: "none",
                result.scale?.state ?: "none",
                ball?.gate?.phase ?: "none",
                ball?.gate?.lastRejection ?: "none",
                ball?.recordedSamples ?: 0,
                cameraRate.describe()
            )
        Log.i(PERFORMANCE_TAG, message)
        diagnosticsRecorder.record(message)
    }

    private fun shutdownExecutor(
        executor: java.util.concurrent.ExecutorService,
        label: String
    ) {
        executor.shutdown()
        var forced = false
        while (!executor.isTerminated) {
            val waitSeconds = if (forced) 5L else 2L
            val terminated = try {
                executor.awaitTermination(waitSeconds, TimeUnit.SECONDS)
            } catch (_: InterruptedException) {
                Log.w(TAG, "$label executor cleanup interrupted; continuing safely")
                false
            }
            if (terminated) break
            if (!forced) {
                executor.shutdownNow()
                forced = true
            } else {
                // ORT/native inference may ignore Java interruption. Never close its session
                // concurrently; this background cleanup thread can safely keep waiting.
                Log.w(TAG, "$label executor still active; delaying native resource close")
            }
        }
    }

    /** Advances the loss watchdog even when YOLO geometry disappears entirely. */
    private fun advanceTrackingWithoutObservation(timestampNs: Long) {
        val active = session ?: return
        if (!active.acceptsCaptureTimestamp(timestampNs)) return
        val before = fallTrackingGate.status()
        val update = fallTrackingGate.observe(null, timestampNs)
        lightweightTrackingDiagnostics?.recordGate(fallTrackingGate.status().lastRejection)
        temporalDebugCaptureRecorder.recordGate(
            stepToken = null,
            observation = null,
            before = before,
            update = update,
            after = fallTrackingGate.status()
        )
        update.endReason?.let(::finishSession)
    }

    /**
     * Passes detector observations through the fall-start gate before recording.
     *
     * Confirmation frames are buffered in ruler coordinates and restored only after coherent
     * downward motion is established. Therefore an unconfirmed candidate cannot add samples,
     * start the lost-ball timer, or satisfy the bottom condition.
     */
    private fun recordObservation(
        detection: RegisteredBallResult,
        axialMaxPx: Float,
        snap: RulerSnapshot,
        timestampNs: Long,
        ballPoint: com.cupet.fallingball.vision.TubeBallPoint?,
        wallContact: com.cupet.fallingball.vision.WallContactResult,
        tubeWidthPx: Float,
        debugCaptureStepId: Long? = null
    ): Boolean {
        val active = session ?: return false
        if (!active.acceptsCaptureTimestamp(timestampNs)) return false

        // A durable measurement is always two-dimensional and image-derived. Legacy axial
        // candidates remain diagnostic-only; classical anchors and camera-rate local residuals
        // are both real observations and retain their source in persistence.
        val ballAxial = detection.ballAxialPx?.takeIf {
            detection.source in DURABLE_BALL_SOURCES &&
                ballPoint != null && BallGeometry.isInsideTubeAxialRange(it, axialMaxPx)
        }
        val absoluteTicks = ballAxial?.let {
            snap.temporalPositionTicks(it, active.config.tickSpacingMm)
        }
        val rulerOk = absoluteTicks != null
        val absoluteMm = absoluteTicks?.times(active.config.tickSpacingMm)
        val observation = ballAxial?.let {
            FallFrameObservation(
                timestampNs = timestampNs,
                axialPx = it,
                absolutePositionMm = absoluteMm,
                referenceOrdinal = if (rulerOk) snap.ordinalAbove(it) else null,
                rulerValid = rulerOk,
                shiftPx = snap.shiftPx,
                source = detection.source,
                positionTicks = absoluteTicks,
                progressFraction = if (axialMaxPx > 0f) it / axialMaxPx else 0f,
                lateralPx = ballPoint?.lateralPx,
                lateralMm = wallContact.lateralMm,
                normalizedLateral = wallContact.normalizedLateral,
                wallClearanceMm = wallContact.clearanceMm,
                wallContactState = wallContact.state,
                trajectoryLateralFraction = ballPoint?.lateralPx?.let { lateral ->
                    tubeWidthPx.takeIf { it.isFinite() && it > 0f }
                        ?.let { width -> lateral / (width * 0.5f) }
                },
                detectionConfidence = detection.confidence.takeIf { it.isFinite() }
            )
        }
        if (observation != null &&
            observation.rulerValid &&
            observation.positionTicks != null &&
            observation.source == com.cupet.fallingball.vision.BallDetectionSource.TINY_HEATMAP
        ) {
            synchronized(causalFallbackLock) {
                val last = causalFallbackObservations.lastOrNull()
                if (last?.timestampNs == observation.timestampNs) {
                    causalFallbackObservations[causalFallbackObservations.lastIndex] = observation
                } else {
                    causalFallbackObservations.add(observation)
                }
            }
        }
        val gateBefore = fallTrackingGate.status()
        val update = fallTrackingGate.observe(observation, timestampNs)
        lightweightTrackingDiagnostics?.recordGate(fallTrackingGate.status().lastRejection)
        val currentObservationAccepted = observation != null &&
            update.acceptedObservations.any { it === observation }
        temporalDebugCaptureRecorder.recordGate(
            stepToken = debugCaptureStepId,
            observation = observation,
            before = gateBefore,
            update = update,
            after = fallTrackingGate.status()
        )
        val debugBuild = context.applicationInfo.flags and
            android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0
        if (debugBuild) {
            val gateStatus = fallTrackingGate.status()
            Log.i(
                TRACK_TRACE_TAG,
                "trace tsNs=$timestampNs source=${detection.source} " +
                    "score=${detection.confidence} axial=${observation?.axialPx} " +
                    "ticks=${observation?.positionTicks} progress=${observation?.progressFraction} " +
                    "lateral=${observation?.trajectoryLateralFraction} ruler=${snap.state} " +
                    "shift=${snap.shiftPx} phase=${gateStatus.phase} " +
                    "pending=${gateStatus.pendingObservations} reject=${gateStatus.lastRejection} " +
                    "accepted=${update.acceptedObservations.size} end=${update.endReason}"
            )
        }

        if (update.acceptedObservations.isNotEmpty()) {
            val referenceOrdinal = sessionRefOrdinal
                ?: update.acceptedObservations.firstNotNullOfOrNull { it.referenceOrdinal }
                    .also { sessionRefOrdinal = it }
            for (accepted in update.acceptedObservations) {
                val relativeTicks = if (
                    accepted.rulerValid &&
                    accepted.positionTicks != null &&
                    referenceOrdinal != null
                ) {
                    accepted.positionTicks - referenceOrdinal
                } else {
                    null
                }
                val mm = relativeTicks?.times(active.config.tickSpacingMm)
                active.add(
                    BallSample(
                        tNs = accepted.timestampNs,
                        axialPx = accepted.axialPx,
                        positionMm = mm,
                        rulerValid = accepted.rulerValid,
                        extrapolated = false,
                        shiftPx = accepted.shiftPx,
                        positionTicks = relativeTicks,
                        lateralPx = accepted.lateralPx,
                        lateralMm = accepted.lateralMm,
                        normalizedLateral = accepted.normalizedLateral,
                        wallClearanceMm = accepted.wallClearanceMm,
                        wallContactState = accepted.wallContactState,
                        detectionSource = accepted.source,
                        detectionConfidence = accepted.detectionConfidence
                    )
                )
            }
        }

        update.endReason?.let(::finishSession)
        return currentObservationAccepted
    }

    private fun finishSession(
        reason: BallEndReason,
        deliverImmediately: Boolean = false
    ) {
        val active = session ?: return
        val finalizationTimer = TinyPipelineStageTimer.create(
            temporalDebugCaptureRecorder.isActive
        )
        val finalizationStartedNs = finalizationTimer.start()
        lightweightTrackingDiagnostics?.snapshot()?.let(active::setPipelineDiagnostics)
        val wasActive = active.isActive
        if (wasActive) {
            // The model now runs asynchronously. Drain work submitted before the stop request
            // so the final Viterbi pass sees the same complete candidate prefix as before.
            val drainStartedNs = finalizationTimer.start()
            if (!offline) awaitTemporalWorkerDrain()
            finalizationTimer.recordSince(TinyPipelineStage.WORKER_DRAIN, drainStartedNs)
            applyFinalTinyHeatmapDecode(active, finalizationTimer)
            applyCausalSustainedFallbackIfNeeded(active, finalizationTimer)
        }
        val velocityStartedNs = finalizationTimer.start()
        val newlyFinished = active.finish(reason)
        if (offline) return
        if (wasActive) {
            finalizationTimer.recordSince(TinyPipelineStage.VELOCITY_ANALYSIS, velocityStartedNs)
            finalizationTimer.recordSince(
                TinyPipelineStage.FINALIZATION_TOTAL,
                finalizationStartedNs
            )
            temporalDebugCaptureRecorder.recordFinalizationTimings(
                active.runId,
                finalizationTimer.snapshot()
            )
        }
        if (!newlyFinished && !deliverImmediately) return
        if (newlyFinished) temporalDebugCaptureRecorder.beginTrackingSessionFinish(active.runId)
        val sessionEndTimestampNs = frameEpoch.latestCaptureTimestampNs.coerceAtLeast(active.startNs)
        // Freeze throughput while still on the serial analysis executor. The main-thread record
        // build can run after later preview-only frames have already reached the model worker.
        val finalThroughput = tinyThroughput.snapshot()
        val samples = active.samples
        val complete = Runnable {
            // A normal finish and Activity shutdown can race. The winner owns persistence,
            // diagnostics closure, and the final UI callback as one indivisible hand-off.
            if (!active.claimCompletionDelivery()) return@Runnable
            val recordTimer = TinyPipelineStageTimer.create(
                temporalDebugCaptureRecorder.isActive
            )
            runCatching { onSessionRecordReady(active, recordTimer) }
                .onFailure { Log.e(TAG, "Unable to build completed measurement record", it) }
            temporalDebugCaptureRecorder.recordRecordBuildTimings(
                active.runId,
                recordTimer.snapshot()
            )
            val finalizeDiagnostics = {
                temporalDebugCaptureRecorder.finishTrackingSession(
                    runId = active.runId,
                    endReason = reason,
                    sampleCount = samples.size,
                    convgruSampleCount = samples.count {
                        it.detectionSource ==
                            com.cupet.fallingball.vision.BallDetectionSource.TINY_HEATMAP
                    },
                    localSampleCount = samples.count {
                        it.detectionSource ==
                            com.cupet.fallingball.vision.BallDetectionSource.LOCAL_TRACKER
                    },
                    modelError = temporalBallError,
                    endTimestampNs = sessionEndTimestampNs,
                    throughput = finalThroughput,
                    onFinalized = {
                        if (deliverImmediately) onSessionFinished(active)
                        else ContextCompat.getMainExecutor(context).execute {
                            onSessionFinished(active)
                        }
                    }
                )
            }
            if (!temporalFinalizationQueue.submit(finalizeDiagnostics)) {
                // The normal path queues behind the single temporal worker. This fallback is
                // only for an executor already shutting down.
                finalizeDiagnostics()
            }
        }
        if (deliverImmediately) complete.run()
        else ContextCompat.getMainExecutor(context).execute(complete)
    }

    private fun awaitTemporalWorkerDrain() {
        runCatching {
            temporalExecutor.submit { }.get(10L, TimeUnit.SECONDS)
        }.onFailure {
            if (!closed.get()) Log.w(TAG, "Unable to drain TinyHeatmapNet worker before finish", it)
        }
    }

    /** Converts the buffered candidate tensors into the exact PC whole-sequence result. */
    private fun applyFinalTinyHeatmapDecode(
        active: FallSession,
        stageTimer: TinyPipelineStageTimer
    ) {
        val inputBuildStartedNs = stageTimer.start()
        val records = synchronized(tinySequenceLock) { tinySequenceFrames.toList() }
        if (records.isEmpty()) {
            stageTimer.recordSince(TinyPipelineStage.FINAL_INPUT_BUILD, inputBuildStartedNs)
            return
        }
        // Diameter only selects the frozen PC fallback velocity. An omitted diameter must not
        // discard the complete TinyHeatmapNet candidate sequence or the final measurement.
        val diameterMm = TinyHeatmapSequenceDecoder.diameterMmOrNeutral(sphereRadiusMm)
        val frames = records.map {
            TinyHeatmapSequenceFrame(
                // Camera-frame numbering: the decoder's velocity terms are per camera frame,
                // so dropped frames must widen the gap instead of looking like one step.
                sourceFrameIndex = it.captureFrameIndex,
                candidates = it.candidates
            )
        }
        stageTimer.recordSince(TinyPipelineStage.FINAL_INPUT_BUILD, inputBuildStartedNs)
        val viterbiStartedNs = stageTimer.start()
        val result = TinyHeatmapSequenceDecoder.decode(frames, diameterMm)
        stageTimer.recordSince(TinyPipelineStage.FINAL_VITERBI, viterbiStartedNs)
        val trajectoryStartedNs = stageTimer.start()
        val firstReference = sessionRefOrdinal ?: records.indices.firstNotNullOfOrNull { index ->
            if (!result.predictedPresent[index]) return@firstNotNullOfOrNull null
            val record = records[index]
            val candidate = record.candidates[result.selectedCandidate[index]]
            val point = sourcePoint(candidate, record.context.axis)
            record.context.ruler.temporalPositionTicks(point.axialPx, rulerConfig.tickSpacingMm)
                ?.toInt()
        }
        var referenceOrdinal = firstReference
        val finalSamples = ArrayList<BallSample>(records.size)
        val finalObservations = ArrayList<FallFrameObservation>(records.size)
        for (index in records.indices) {
            if (!result.predictedPresent[index]) continue
            val record = records[index]
            val candidate = record.candidates[result.selectedCandidate[index]]
            val point = sourcePoint(candidate, record.context.axis)
            if (!BallGeometry.isInsideTubeAxialRange(point.axialPx, record.context.axialMaxPx)) {
                continue
            }
            val ticks = record.context.ruler.temporalPositionTicks(
                point.axialPx,
                rulerConfig.tickSpacingMm
            )
            if (referenceOrdinal == null && ticks != null) referenceOrdinal = ticks.toInt()
            val relativeTicks = if (ticks != null && referenceOrdinal != null) {
                ticks - referenceOrdinal
            } else {
                null
            }
            val wallContact = wallContactFor(point, record.context.ruler)
            finalSamples += BallSample(
                tNs = record.timestampNs,
                axialPx = point.axialPx,
                positionMm = relativeTicks?.times(rulerConfig.tickSpacingMm),
                rulerValid = ticks != null,
                extrapolated = false,
                shiftPx = record.context.ruler.shiftPx,
                positionTicks = relativeTicks,
                lateralPx = point.lateralPx,
                lateralMm = wallContact.lateralMm,
                normalizedLateral = wallContact.normalizedLateral,
                wallClearanceMm = wallContact.clearanceMm,
                wallContactState = wallContact.state,
                detectionSource = com.cupet.fallingball.vision.BallDetectionSource.TINY_HEATMAP,
                detectionConfidence = sigmoid(candidate.logit)
            )
            finalObservations += FallFrameObservation(
                timestampNs = record.timestampNs,
                axialPx = point.axialPx,
                absolutePositionMm = ticks?.times(rulerConfig.tickSpacingMm),
                referenceOrdinal = referenceOrdinal,
                rulerValid = ticks != null,
                shiftPx = record.context.ruler.shiftPx,
                source = com.cupet.fallingball.vision.BallDetectionSource.TINY_HEATMAP,
                positionTicks = ticks,
                progressFraction = if (record.context.axialMaxPx > 0f) {
                    point.axialPx / record.context.axialMaxPx
                } else {
                    0f
                },
                lateralPx = point.lateralPx,
                lateralMm = wallContact.lateralMm,
                normalizedLateral = wallContact.normalizedLateral,
                wallClearanceMm = wallContact.clearanceMm,
                wallContactState = wallContact.state,
                trajectoryLateralFraction = (point.lateralPx /
                    (record.context.axis.widthPx * 0.5f))
                    .takeIf { it.isFinite() },
                detectionConfidence = sigmoid(candidate.logit)
            )
        }
        stageTimer.recordSince(TinyPipelineStage.FINAL_TRAJECTORY, trajectoryStartedNs)
        // The whole-sequence decode normally refines the causal trajectory, but it optimises a
        // global path score with no notion of "the ball must fall". A bright static feature
        // (tube end shadow, meniscus) that scores well on many frames can win the path
        // outright: one expensive jump off the ball is repaid by a tiny per-frame saving for
        // the rest of the sequence. Measured on device, that collapsed real 1029-1048 px
        // descents to 12-206 px of travel and made the terminal-velocity fit unusable.
        // Travel is exactly what that fit consumes, so keep the decode only when it retains a
        // credible fraction of the causal trajectory's travel.
        val acceptanceStartedNs = stageTimer.start()
        val causalTravelPx = axialTravelPx(active.samples)
        val decodedTravelPx = axialTravelPx(finalSamples)
        val decodedBackwardTravelPx = FinalDecodeAcceptance.backwardTravelPx(
            finalSamples.map { it.axialPx }
        )
        val decodedMaxBackwardStepTicks = FinalDecodeAcceptance.maxBackwardStepTicks(
            finalSamples.map { it.positionTicks }
        )
        val decodedFallConfirmed = FinalDecodeAcceptance.confirmsFall(finalObservations)
        val accepted = FinalDecodeAcceptance.acceptsDecode(
                causalTravelPx,
                decodedTravelPx,
                decodedFallConfirmed,
                decodedBackwardTravelPx,
                decodedMaxBackwardStepTicks
            )
        stageTimer.recordSince(TinyPipelineStage.FINAL_ACCEPTANCE, acceptanceStartedNs)
        if (!accepted) {
            Log.w(
                TAG,
                "Final TinyHeatmapNet decode rejected: causalTravel=${causalTravelPx}px " +
                    "decodedTravel=${decodedTravelPx}px backwardTravel=" +
                    "${decodedBackwardTravelPx}px confirmed=$decodedFallConfirmed " +
                    "(frames=${records.size} present=${finalSamples.size}); keeping causal samples"
            )
            return
        }

        active.replaceSamplesForFinalDecode(finalSamples)
        Log.i(
            TAG,
            "Final TinyHeatmapNet Viterbi decoded frames=${records.size} " +
                "present=${finalSamples.size} velocity=${result.expectedVerticalVelocity}"
        )
    }

    /**
     * Last deterministic fallback before [FallSession.finish].
     *
     * The strict gate and the whole-sequence decode can both reject a slow, clean causal
     * selection: the gate needs at least 0.05 ruler ticks between consecutive frames and the
     * Viterbi decode can win with a static reflection. The selector buffer contains every
     * selected TinyHeatmap observation, so a robust whole-sequence fit can recover those
     * runs without loosening the live gate for fast balls or no-ball controls.
     */
    private fun applyCausalSustainedFallbackIfNeeded(
        active: FallSession,
        stageTimer: TinyPipelineStageTimer
    ) {
        val startedNs = stageTimer.start()
        try {
            val currentAnalysis = VelocityEstimator.analyze(active.samples)
            val currentReliable = VelocityEstimator.isReliableTerminalEstimate(
                currentAnalysis.summary.plateauFound,
                currentAnalysis.summary.terminalVelocityMmPerS,
                currentAnalysis.summary.validSampleCount,
                currentAnalysis.quality
            )
            if (currentReliable) return

            val observations = synchronized(causalFallbackLock) {
                causalFallbackObservations.toList()
            }
            val selected = FinalDecodeAcceptance.selectSustainedFall(observations) ?: return
            val fallbackSamples = samplesFromCausalFallback(
                selected,
                active.config.tickSpacingMm
            )
            if (fallbackSamples.size < 2) return

            val fallbackAnalysis = VelocityEstimator.analyze(fallbackSamples)
            val fallbackReliable = VelocityEstimator.isReliableTerminalEstimate(
                fallbackAnalysis.summary.plateauFound,
                fallbackAnalysis.summary.terminalVelocityMmPerS,
                fallbackAnalysis.summary.validSampleCount,
                fallbackAnalysis.quality
            )
            if (!fallbackReliable ||
                fallbackAnalysis.summary.terminalVelocityMmPerS == null ||
                fallbackSamples.zipWithNext().any { (a, b) -> b.tNs <= a.tNs }
            ) {
                return
            }
            active.replaceSamplesForFinalDecode(fallbackSamples)
            Log.i(
                TAG,
                "Causal sustained fallback replaced samples count=${fallbackSamples.size} " +
                    "velocity=${fallbackAnalysis.summary.terminalVelocityMmPerS}"
            )
        } finally {
            stageTimer.recordSince(TinyPipelineStage.FINAL_ACCEPTANCE, startedNs)
        }
    }

    private fun samplesFromCausalFallback(
        observations: List<FallFrameObservation>,
        tickSpacingMm: Float
    ): List<BallSample> {
        if (observations.isEmpty()) return emptyList()
        val reference = observations.first().positionTicks?.toInt() ?: return emptyList()
        return observations.map { observation ->
            val relativeTicks = observation.positionTicks?.minus(reference)
            BallSample(
                tNs = observation.timestampNs,
                axialPx = observation.axialPx,
                positionMm = relativeTicks?.times(tickSpacingMm),
                rulerValid = observation.rulerValid,
                extrapolated = false,
                shiftPx = observation.shiftPx,
                positionTicks = relativeTicks,
                lateralPx = observation.lateralPx,
                lateralMm = observation.lateralMm,
                normalizedLateral = observation.normalizedLateral,
                wallClearanceMm = observation.wallClearanceMm,
                wallContactState = observation.wallContactState,
                detectionSource = observation.source,
                detectionConfidence = observation.detectionConfidence
            )
        }
    }

    /**
     * Axial travel covered by a sample list, in pixels, or zero when it cannot be measured.
     *
     * Only used to compare the whole-sequence decode against the causal trajectory it would
     * replace; it never feeds a measurement.
     */
    private fun axialTravelPx(samples: List<BallSample>): Float {
        if (samples.size < 2) return 0f
        var minimum = Float.MAX_VALUE
        var maximum = -Float.MAX_VALUE
        for (sample in samples) {
            val axial = sample.axialPx
            if (!axial.isFinite()) continue
            if (axial < minimum) minimum = axial
            if (axial > maximum) maximum = axial
        }
        if (minimum > maximum) return 0f
        return maximum - minimum
    }


    private fun sourcePoint(
        candidate: TinyHeatmapSequenceCandidate,
        axis: TubeAxis
    ): com.cupet.fallingball.vision.TubeBallPoint {
        val dx = candidate.sourceX - axis.ox
        val dy = candidate.sourceY - axis.oy
        val axial = dx * axis.ux + dy * axis.uy
        val lateral = dx * axis.vx + dy * axis.vy
        return com.cupet.fallingball.vision.TubeBallPoint(
            x = axis.pointX(axial, lateral),
            y = axis.pointY(axial, lateral),
            axialPx = axial,
            lateralPx = lateral
        )
    }

private fun sigmoid(logit: Float): Float =
        (1.0 / (1.0 + kotlin.math.exp(-logit.toDouble()))).toFloat()

    private fun sampleFps(tsNs: Long): Float {
        if (lastSampleNs != 0L) {
            val dt = (tsNs - lastSampleNs) / 1_000_000_000.0
            if (dt > 1e-4) {
                val inst = (1.0 / dt).toFloat()
                sampleFps = if (sampleFps <= 0f) inst else sampleFps * 0.85f + inst * 0.15f
            }
        }
        lastSampleNs = tsNs
        return sampleFps
    }

    /**
     * Whether the whole tube is comfortably inside the frame. The tick-count gate compares
     * against the user's total graduation count, which only means anything if nothing is
     * cropped away.
     */
    private fun isFullyInFrame(track: TubeTrackResult): Boolean {
        val corners = track.corners ?: return false
        val margin = 4f
        val w = track.frameWidth.toFloat()
        val h = track.frameHeight.toFloat()
        for (c in corners) {
            if (c.first < margin || c.first > w - margin) return false
            if (c.second < margin || c.second > h - margin) return false
        }
        return true
    }

    private fun ensureAxialScratch(n: Int) {
        if (tickAxialScratch.size < n) {
            var cap = tickAxialScratch.size
            while (cap < n) cap *= 2
            tickAxialScratch = FloatArray(cap)
        }
    }

    private fun ensureDenseBandCapacity(n: Int) {
        if (denseBandSignals.all { it.size >= n }) return
        var cap = denseBandSignals.firstOrNull()?.size?.coerceAtLeast(1) ?: 1
        while (cap < n) cap *= 2
        denseBandSignals = Array(denseBandSampler.bandCount) { FloatArray(cap) }
    }

}

private fun TubeRoiGeometry.withStabilization(
    transform: RoiStabilizationTransform
): TubeRoiGeometry {
    val stableFromRaw = doubleArrayOf(
        1.0, 0.0, transform.sourceToStabilizedX.toDouble(),
        0.0, 1.0, transform.sourceToStabilizedY.toDouble(),
        0.0, 0.0, 1.0
    )
    val rawFromStable = doubleArrayOf(
        1.0, 0.0, -transform.sourceToStabilizedX.toDouble(),
        0.0, 1.0, -transform.sourceToStabilizedY.toDouble(),
        0.0, 0.0, 1.0
    )
    return copy(
        sourceToRoi = multiply3x3(stableFromRaw, sourceToRoi),
        roiToSource = multiply3x3(roiToSource, rawFromStable)
    )
}

private fun multiply3x3(left: DoubleArray, right: DoubleArray): DoubleArray {
    require(left.size == 9 && right.size == 9)
    return DoubleArray(9) { index ->
        val row = index / 3
        val column = index % 3
        left[row * 3] * right[column] +
            left[row * 3 + 1] * right[3 + column] +
            left[row * 3 + 2] * right[6 + column]
    }
}
