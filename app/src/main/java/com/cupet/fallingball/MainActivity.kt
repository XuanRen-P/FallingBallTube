package com.cupet.fallingball

import android.Manifest
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updateLayoutParams
import com.cupet.fallingball.analysis.FallSession
import com.cupet.fallingball.analysis.MeasurementHistoryStore
import com.cupet.fallingball.analysis.MeasurementDefaults
import com.cupet.fallingball.analysis.MeasurementRecord
import com.cupet.fallingball.learning.QuestionWorkScheduler
import com.cupet.fallingball.camera.CameraController
import com.cupet.fallingball.camera.CameraControlsController
import com.cupet.fallingball.video.VideoAnalysisActivity
import com.cupet.fallingball.video.VideoAnalysisWork
import com.cupet.fallingball.camera.TemporalPipelineStage
import com.cupet.fallingball.databinding.ActivityMainBinding
import com.cupet.fallingball.ui.AnalysisResolution
import com.cupet.fallingball.ui.DeviceLevelMonitor
import com.cupet.fallingball.ui.ModelProfileStore
import com.cupet.fallingball.ui.RulerConfigStore
import com.cupet.fallingball.vision.BallEndReason
import com.cupet.fallingball.vision.CpuAffinity
import com.cupet.fallingball.vision.InvalidReason
import com.cupet.fallingball.vision.ModelProfile
import com.cupet.fallingball.vision.ObbDetector
import com.cupet.fallingball.vision.RulerConfig
import com.cupet.fallingball.vision.RulerState
import com.cupet.fallingball.vision.TinyPipelineStageTimer
import com.cupet.fallingball.vision.TrackMode
import com.cupet.fallingball.vision.TubeTrackResult
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private companion object {
        const val CAMERA_SHUTDOWN_TIMEOUT_SECONDS = 12L
    }

    private lateinit var binding: ActivityMainBinding
    private var cameraController: CameraController? = null
    private lateinit var cameraOptics: CameraControlsController
    private var measurementControlsLocked = false
    private var videoPageOpen = false
    private var videoWorkKnown = false
    private var videoWorkRunning = false
    private var stabilizationEnabled = false
    private var currentResolution = AnalysisResolution.P720
    private var currentModelProfile = ModelProfile.DEFAULT
    private var currentModelGeneration = 0L
    private var destroying = false

    private val ioExecutor = Executors.newSingleThreadExecutor()

    private var rulerConfig = RulerConfig.DEFAULT
    private var measurementDefaults = MeasurementDefaults()
    private var activeMeasurementDefaults = MeasurementDefaults()
    private var lastSession: FallSession? = null
    private var rulerLocked = false
    private var historyLaunchPending = false
    private val pendingRecordSaves = ConcurrentHashMap<String, MeasurementRecord>()
    private lateinit var levelMonitor: DeviceLevelMonitor
    private lateinit var measurementHistoryStore: MeasurementHistoryStore

    private val historyLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        historyLaunchPending = false
        binding.btnHistory.isEnabled = true
        rulerLocked = false
        binding.overlayView.updateResult(null)
        binding.statusText.setText(R.string.status_searching)
        refreshMeta(null)
        if (!videoWorkRunning) cameraController?.resume(resetAnalysis = true)
        startLevelIndicator()
    }

    private val videoLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        videoPageOpen = false
        restoreCameraAfterVideo()
    }

    /** One tube ONNX session shared by camera mode; owned and closed here. */
    private val obbDetectorLazy = lazy { ObbDetector(this, currentModelProfile) }
    private val obbDetector: ObbDetector get() = obbDetectorLazy.value

    // Camera permission
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else {
            Toast.makeText(this, R.string.permission_required, Toast.LENGTH_LONG).show()
            binding.statusText.setText(R.string.permission_required)
        }
    }

    // Lifecycle
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Threads spawned later inherit this mask, so pin before camera/inference start.
        CpuAffinity.pinProcessToFastCores()

        applyWindowInsets()
        rulerConfig = RulerConfigStore.load(this)
        measurementDefaults = RulerConfigStore.loadMeasurementDefaults(this)
        currentResolution = RulerConfigStore.loadAnalysisResolution(this)
        currentModelProfile = ModelProfileStore.load(this)
        measurementHistoryStore = MeasurementHistoryStore(
            filesDir.resolve("measurement_history")
        )
        levelMonitor = DeviceLevelMonitor(this) { reading ->
            binding.levelIndicator.setReading(reading)
        }
        setupCameraControls()
        setupMeasurementControls()
        VideoAnalysisWork.observe(this).observe(this) { infos ->
            val wasRunning = videoWorkRunning
            videoWorkKnown = true
            videoWorkRunning = VideoAnalysisWork.isRunning(infos)
            if (videoWorkRunning) {
                cameraController?.pause()
                levelMonitor.stop()
                binding.overlayView.updateResult(null)
                binding.statusText.setText(R.string.video_phase_analyzing)
            } else if (wasRunning || cameraController == null) {
                restoreCameraAfterVideo()
            }
            updateCameraControlAvailability()
        }
        handleDebugTestAction(intent)

        // Prompt on the very first launch only. After that the remembered values are used,
        // and the 参数 button is there to change them.
        if (!RulerConfigStore.isConfigured(this)) {
            showParameterDialog()
        }
    }

    private fun setupMeasurementControls() {
        binding.btnTrack.setOnClickListener { toggleTracking() }
        binding.btnConfig.setOnClickListener { showParameterDialog() }
        binding.btnHistory.setOnClickListener { openHistory() }
        binding.btnVideoAnalysis.setOnClickListener {
            if (measurementControlsLocked || videoPageOpen) return@setOnClickListener
            videoPageOpen = true
            cameraController?.pause()
            levelMonitor.stop()
            videoLauncher.launch(Intent(this, VideoAnalysisActivity::class.java))
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleDebugTestAction(intent)
    }

    private fun handleDebugTestAction(sourceIntent: Intent?) {
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) return
        val rawAction = sourceIntent?.getStringExtra(DebugTestControl.EXTRA_ACTION)
        val action = DebugTestControl.parse(rawAction)
        if (action == null) {
            if (rawAction != null) {
                android.util.Log.w(
                    DebugTestControl.LOG_TAG,
                    "ignored unknown action=$rawAction"
                )
            }
            return
        }
        sourceIntent?.removeExtra(DebugTestControl.EXTRA_ACTION)
        binding.root.post {
            if (destroying || isFinishing || isDestroyed) return@post
            when (action) {
                DebugTestAction.RELOCK -> binding.btnRelock.performClick()
                DebugTestAction.START_TRACKING -> {
                    if (cameraController?.isSessionActive != true) toggleTracking()
                }
                DebugTestAction.STOP_TRACKING -> {
                    if (cameraController?.isSessionActive == true) toggleTracking()
                }
                DebugTestAction.START_TEMPORAL_CAPTURE -> {
                    cameraController?.startTemporalDebugCapture()
                }
                DebugTestAction.START_PIPELINE_TIMING_CAPTURE -> {
                    cameraController?.startPipelineTimingCapture()
                }
                DebugTestAction.STOP_TEMPORAL_CAPTURE -> {
                    cameraController?.stopTemporalDebugCapture()
                }
                DebugTestAction.BENCHMARK_TINY_PARENT -> {
                    cameraController?.benchmarkTinyParent()
                }
                DebugTestAction.BENCHMARK_TINY_PREPROCESS -> {
                    cameraController?.benchmarkTinyPreprocess()
                }
                DebugTestAction.BENCHMARK_TINY_CPU_THREADS -> {
                    cameraController?.benchmarkTinyCpuThreads()
                }
                DebugTestAction.BENCHMARK_TINY_CANDIDATE -> {
                    cameraController?.benchmarkTinyCandidate()
                }
                DebugTestAction.BENCHMARK_TINY_NCNN -> {
                    cameraController?.benchmarkTinyNcnn()
                }
            }
            android.util.Log.i(
                DebugTestControl.LOG_TAG,
                "action=${action.wireValue} active=" +
                    (cameraController?.isSessionActive == true) +
                    " capture=${cameraController?.isTemporalDebugCaptureActive == true}"
            )
        }
    }

    private fun showParameterDialog() {
        RulerConfigStore.prompt(
            this,
            rulerConfig,
            currentModelProfile,
            currentAnalysisResolution = currentResolution,
            currentDefaults = measurementDefaults,
            cameraSettingsEnabled = cameraController?.isSessionActive != true,
            cancellable = true
        ) { config, modelProfile, analysisResolution, defaults ->
            applyConfig(
                config,
                modelProfile,
                analysisResolution,
                defaults
            )
        }
    }

    private fun applyConfig(
        config: RulerConfig,
        modelProfile: ModelProfile,
        analysisResolution: AnalysisResolution,
        defaults: MeasurementDefaults
    ) {
        rulerConfig = config
        measurementDefaults = defaults
        cameraController?.setRulerConfig(config)
        cameraController?.setBallGeometry(
            defaults.viscometryParameters.sphereRadiusMm,
            defaults.viscometryParameters.tubeInnerDiameterMm,
            defaults.sphereMaterial
        )
        val modelChanged = modelProfile != currentModelProfile
        val resolutionChanged = analysisResolution != currentResolution
        if (modelChanged) {
            currentModelProfile = modelProfile
            ModelProfileStore.save(this, currentModelProfile)
            cameraController?.let {
                currentModelGeneration = it.setModelProfile(currentModelProfile)
            }
        }
        if (resolutionChanged) {
            currentResolution = analysisResolution
            cameraController?.setResolution(currentResolution)
        }
        if (modelChanged || resolutionChanged) {
            rulerLocked = false
            binding.overlayView.updateResult(null)
            binding.statusText.setText(R.string.status_searching)
        }
        refreshMeta(null)
    }

    private fun toggleTracking() {
        val controller = cameraController ?: return
        if (controller.isSessionActive) {
            controller.stopSession()
            return
        }
        if (measurementControlsLocked || videoWorkRunning || cameraOptics.state.applying ||
            !cameraOptics.state.bound) return
        // Without a locked ruler every sample would come back with no millimetre value, so
        // the run would produce a position curve in pixels and no velocity at all.
        if (!rulerLocked) {
            Toast.makeText(this, R.string.toast_need_ruler, Toast.LENGTH_SHORT).show()
            return
        }
        if (!controller.isBallTrackingReady) {
            Toast.makeText(this, R.string.toast_need_ball_background, Toast.LENGTH_SHORT).show()
            return
        }
        controller.setRulerConfig(rulerConfig)
        activeMeasurementDefaults = measurementDefaults
        val startedSession = runCatching { controller.startSession() }.getOrElse {
            Toast.makeText(
                this,
                R.string.toast_diagnostics_start_failed,
                Toast.LENGTH_LONG
            ).show()
            return
        }
        lastSession = startedSession
        measurementControlsLocked = true
        updateCameraControlAvailability()
        binding.btnTrack.setText(R.string.btn_stop_track)
        binding.btnHistory.isEnabled = false
        binding.btnConfig.isEnabled = false
        binding.btnRelock.isEnabled = false
        binding.root.keepScreenOn = true
    }

    private fun persistFinishedSession(
        session: FallSession,
        stageTimer: TinyPipelineStageTimer
    ) {
        lastSession = session
        val record = if (session.sampleCount > 0 || session.diagnosticsEnabled) {
            MeasurementRecord.fromSession(
                session = session,
                recordedAtEpochMs = System.currentTimeMillis(),
                id = session.runId,
                viscometryParameters = activeMeasurementDefaults.viscometryParameters,
                metadata = activeMeasurementDefaults.metadataSnapshot(),
                stageTimer = stageTimer
            )
        } else {
            null
        }
        if (record != null) saveMeasurementRecord(record)
    }

    private fun onSessionFinished(session: FallSession) {
        lastSession = session
        if (destroying) return
        measurementControlsLocked = false
        updateCameraControlAvailability()
        binding.btnTrack.setText(R.string.btn_start_track)
        binding.btnRelock.isEnabled = true
        binding.root.keepScreenOn = false
        binding.btnHistory.isEnabled = true
        binding.btnConfig.isEnabled = true
        showSummary(session)
    }

    private fun saveMeasurementRecord(record: MeasurementRecord) {
        pendingRecordSaves[record.id] = record
        val saveTask = Runnable {
            val result = runCatching {
                measurementHistoryStore.save(record)
            }
            if (result.isSuccess) pendingRecordSaves.remove(record.id)
            runOnUiThread {
                if (destroying || isFinishing || isDestroyed) return@runOnUiThread
                result.fold(
                    onSuccess = {
                        if (record.samples.isNotEmpty()) {
                            QuestionWorkScheduler.enqueue(this, record.id)
                        }
                        Toast.makeText(
                            this,
                            if (record.samples.isEmpty()) {
                                R.string.history_save_log_only
                            } else {
                                R.string.history_save_ok
                            },
                            Toast.LENGTH_SHORT
                        ).show()
                    },
                    onFailure = { error ->
                        // If a queued history-open retry has already saved it, this error is
                        // stale and no longer needs user action.
                        if (!pendingRecordSaves.containsKey(record.id)) return@fold
                        MaterialAlertDialogBuilder(this)
                            .setTitle(R.string.history_save_retry_title)
                            .setMessage(
                                getString(
                                    R.string.history_save_failed,
                                    error.message ?: ""
                                )
                            )
                            .setPositiveButton(R.string.history_save_retry) { _, _ ->
                                saveMeasurementRecord(record)
                            }
                            .setNegativeButton(android.R.string.cancel, null)
                            .show()
                    }
                )
            }
        }
        runCatching { ioExecutor.execute(saveTask) }.onFailure {
            // A finish callback racing Activity teardown must still get a durable write.
            Thread(saveTask, "FallingBall-record-save").start()
        }
    }

    private fun showSummary(session: FallSession) {
        val s = session.summary
        val quality = session.analysisQuality
        val sb = StringBuilder()

        if (quality?.terminalFitCheckUnavailable == true) {
            sb.appendLine(getString(R.string.summary_terminal_fit_check_unavailable))
        } else if (quality?.unstableTerminalFit == true) {
            sb.appendLine(
                getString(
                    R.string.summary_unstable_terminal_fit,
                    (quality.terminalFitRelativeDisagreement ?: 0f) * 100f
                )
            )
        }
        if (s?.plateauFound == true && s.terminalVelocityMmPerS != null) {
            sb.appendLine(getString(R.string.summary_terminal, s.terminalVelocityMmPerS))
            sb.appendLine(
                getString(
                    R.string.summary_plateau,
                    s.plateauStartMm, s.plateauEndMm, s.plateauDurationMs
                )
            )
            sb.appendLine(getString(R.string.summary_fit, s.r2))
        } else if (quality?.trackDiscontinuity == true) {
            sb.appendLine(getString(R.string.summary_track_discontinuity))
        } else if (quality?.excessiveOutliers == true) {
            sb.appendLine(getString(R.string.summary_too_many_outliers))
        } else {
            // Deliberately not substituting the mean here: a mean over an accelerating run is
            // not a terminal velocity, and presenting it as one would be misleading.
            sb.appendLine(getString(R.string.summary_no_plateau))
            s?.meanVelocityMmPerS?.let {
                sb.appendLine(getString(R.string.summary_mean, it))
            }
        }
        sb.appendLine(
            getString(
                R.string.summary_samples,
                session.sampleCount,
                quality?.usablePositionSampleCount ?: session.validSampleCount,
                quality?.effectiveSampleRateFps ?: session.meanFps()
            )
        )
        appendAnalysisQuality(sb, quality, s?.plateauFound == true)
        if (s?.underSampled == true) sb.appendLine(getString(R.string.summary_under_sampled))
        val shake = session.peakShiftPx()
        sb.appendLine(getString(R.string.summary_shake, shake))
        if (session.hasExcessiveRegistrationShake()) {
            sb.appendLine(getString(R.string.summary_shake_warning))
        }
        sb.append(getString(R.string.summary_end_reason, endReasonText(session.endReason)))

        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.summary_title)
            .setMessage(sb.toString())
            .setPositiveButton(R.string.btn_history) { _, _ ->
                openHistory()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun appendAnalysisQuality(
        target: StringBuilder,
        quality: com.cupet.fallingball.analysis.VelocityAnalysisQuality?,
        robustFitFound: Boolean
    ) {
        if (quality == null) return
        if (quality.lowSampleRate) {
            target.appendLine(
                getString(
                    if (quality.insufficientVelocitySamples) {
                        R.string.summary_sample_rate_too_low
                    } else {
                        R.string.summary_low_sample_rate
                    },
                    quality.effectiveSampleRateFps,
                    quality.velocityWindowMs
                )
            )
        }
        if (quality.outlierSampleCount > 0) {
            target.appendLine(
                getString(R.string.summary_outliers_excluded, quality.outlierSampleCount)
            )
        }
        if (quality.excessiveOutliers && !robustFitFound) {
            target.appendLine(getString(R.string.summary_outliers_excessive))
        }
        if (quality.trackDiscontinuity) {
            target.appendLine(getString(R.string.summary_track_discontinuity_warning))
        }
    }

    private fun endReasonText(reason: BallEndReason?): String = getString(
        when (reason) {
            BallEndReason.BALL_LOST -> R.string.end_ball_lost
            BallEndReason.REACHED_BOTTOM -> R.string.end_reached_bottom
            else -> R.string.end_manual
        }
    )

    /**
     * Keep the camera preview full-bleed but push the info/control panels clear of the
     * status bar, navigation bar and display cutout (punch-hole / notch).
     *
     * Required since targetSdk 35: Android 15 enforces edge-to-edge, so the panels would
     * otherwise be drawn underneath the system bars and the front camera.
     */
    private fun applyWindowInsets() {
        val topLp = binding.topPanel.layoutParams as ViewGroup.MarginLayoutParams
        val bottomLp = binding.bottomPanel.layoutParams as ViewGroup.MarginLayoutParams
        val baseTop = topLp.topMargin
        val baseBottom = bottomLp.bottomMargin
        val baseSide = topLp.leftMargin

        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { _, windowInsets ->
            val bars = windowInsets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            binding.topPanel.updateLayoutParams<ViewGroup.MarginLayoutParams> {
                topMargin = baseTop + bars.top
                leftMargin = baseSide + bars.left
                rightMargin = baseSide + bars.right
            }
            binding.bottomPanel.updateLayoutParams<ViewGroup.MarginLayoutParams> {
                bottomMargin = baseBottom + bars.bottom
                leftMargin = baseSide + bars.left
                rightMargin = baseSide + bars.right
            }
            // Panels own the insets; the preview underneath stays edge-to-edge.
            WindowInsetsCompat.CONSUMED
        }
    }

    // Camera mode setup
    private fun setupCameraControls() {
        cameraOptics = CameraControlsController(
            context = this,
            lifecycleOwner = this,
            onCameraSelected = {
                try {
                    cameraController?.rebindCamera()
                } catch (failure: Exception) {
                    // selectCamera restores its previous selector before this posted recovery.
                    binding.root.post {
                        runCatching { cameraController?.rebindCamera() }
                    }
                    throw failure
                }
            },
            onBeforeOpticsChange = {
                if (measurementControlsLocked || videoWorkRunning) false else {
                    invalidateOpticalCalibration()
                    true
                }
            },
            onAutofocusModeChanged = { cameraController?.setTubeAutofocusEnabled(it) },
            onOpticsApplied = { invalidateOpticalCalibration() },
            onError = { Toast.makeText(this, it, Toast.LENGTH_LONG).show() }
        )
        binding.cameraOpticsView.attach(cameraOptics)
        cameraOptics.onStateChanged = {
            binding.cameraOpticsView.render(it)
            binding.overlayView.mirrorHorizontally = it.cameras
                .firstOrNull { lens -> lens.id == it.selectedCameraId }?.isFront == true
            binding.btnTrack.isEnabled = cameraController?.isSessionActive == true ||
                (!measurementControlsLocked && !videoWorkRunning && it.bound && !it.applying)
        }
        binding.btnRelock.setOnClickListener {
            rulerLocked = false
            cameraController?.relock()
            binding.overlayView.updateResult(null)
            binding.statusText.setText(R.string.status_searching)
            refreshMeta(null)
        }

        ensurePermissionAndStart()
    }

    // Camera
    private fun ensurePermissionAndStart() {
        if (!videoWorkKnown || videoWorkRunning || videoPageOpen || historyLaunchPending) return
        val ok = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        if (ok) startCamera() else permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        if (!videoWorkKnown || videoWorkRunning || videoPageOpen || historyLaunchPending) return
        if (cameraController != null) {
            cameraController?.let {
                it.setResolution(currentResolution)
                currentModelGeneration = it.setModelProfile(currentModelProfile)
            }
            return
        }
        cameraController = CameraController(
            context = this,
            lifecycleOwner = this,
            previewView = binding.previewView,
            detector = obbDetector,
            onResult = { result -> onTrackResult(result) },
            onStabilizationInfo = { enabled ->
                stabilizationEnabled = enabled
                refreshMeta(null)
                if (!enabled) {
                    Toast.makeText(
                        this, R.string.stabilization_unsupported, Toast.LENGTH_LONG
                    ).show()
                }
            },
            onSessionRecordReady = { session, stageTimer ->
                persistFinishedSession(session, stageTimer)
            },
            onSessionFinished = { session -> onSessionFinished(session) }
        ).also {
            it.cameraControls = cameraOptics
            it.start()
            it.setResolution(currentResolution)
            currentModelGeneration = it.setModelProfile(currentModelProfile)
            it.setRulerConfig(rulerConfig)
            it.setBallGeometry(
                measurementDefaults.viscometryParameters.sphereRadiusMm,
                measurementDefaults.viscometryParameters.tubeInnerDiameterMm,
                measurementDefaults.sphereMaterial
            )
        }
    }

    private fun onTrackResult(result: TubeTrackResult) {
        if (result.profileGeneration != currentModelGeneration) return
        binding.overlayView.updateResult(result)
        rulerLocked = result.scale?.state == RulerState.VALID
        binding.statusText.text = statusText(result)
        refreshMeta(result)
    }

    /**
     * Status line priority: what the user has to act on first. A tracking run outranks the
     * ruler state, and a specific reason for rejection outranks a generic "unstable", since
     * only the reason tells them what to change.
     */
    private fun statusText(result: TubeTrackResult): String {
        val controller = cameraController
        if (controller != null && controller.isSessionActive) {
            val recorded = lastSession?.sampleCount ?: 0
            return if (recorded == 0) getString(R.string.status_track_armed)
            else getString(R.string.status_track_running, recorded)
        }

        val scale = result.scale
        if (result.mode == TrackMode.SEARCHING && scale == null) {
            return getString(R.string.status_searching)
        }
        return when (scale?.state) {
            RulerState.VALID -> {
                when (controller?.ballTrackingDiagnostics()?.stage) {
                    TemporalPipelineStage.UNAVAILABLE ->
                        getString(R.string.status_ball_detector_unavailable)
                    TemporalPipelineStage.ERROR ->
                        getString(R.string.status_ball_detector_error)
                    TemporalPipelineStage.WAITING_FOR_TUBE,
                    TemporalPipelineStage.WARMING_UP ->
                        getString(R.string.status_ball_detector_warming)
                    TemporalPipelineStage.RUNNING -> getString(R.string.status_ruler_ok, scale.count)
                    null -> getString(R.string.status_ball_detector_warming)
                }
            }
            RulerState.DEGRADED ->
                scale.invalidReason?.let { getString(reasonRes(it)) }
                    ?: getString(R.string.status_ruler_degraded)
            RulerState.LOST -> getString(R.string.status_ruler_failed)
            RulerState.ACQUIRING ->
                scale.invalidReason?.let { getString(reasonRes(it)) }
                    ?: getString(R.string.status_ruler_acquiring)
            null -> getString(R.string.status_locked)
        }
    }

    private fun reasonRes(reason: InvalidReason): Int = when (reason) {
        InvalidReason.SHORT_SIGNAL -> R.string.reason_short_signal
        InvalidReason.GAP_TOO_SMALL -> R.string.reason_gap_too_small
        InvalidReason.TILT -> R.string.reason_tilt
        InvalidReason.NOT_FULLY_IN_FRAME -> R.string.reason_not_fully_in_frame
        InvalidReason.COUNT -> R.string.reason_count
        InvalidReason.GAP_JUMP -> R.string.reason_gap_jump
        InvalidReason.GEOM_INCONSISTENT -> R.string.reason_geom_inconsistent
        InvalidReason.IRREGULAR -> R.string.reason_irregular
        InvalidReason.SHIFT -> R.string.reason_shift
    }

    private fun refreshMeta(result: TubeTrackResult?) {
        val stab = getString(
            if (stabilizationEnabled) R.string.stabilization_on else R.string.stabilization_off
        )
        val sampleFps = result?.fps
            ?.let { getString(R.string.meta_rate, it) }
            ?: getString(R.string.meta_rate_empty)
        val boxFps = result?.inferenceFps
            ?.takeIf { it > 0f }
            ?.let { getString(R.string.meta_rate, it) }
            ?: getString(R.string.meta_rate_empty)
        val inferenceMs = result?.inferenceMs
            ?.takeIf { it > 0f }
            ?.let { getString(R.string.meta_latency, it) }
            ?: getString(R.string.meta_latency_empty)
        val conf = result?.confidence
            ?.takeIf { it > 0f }
            ?.let { getString(R.string.meta_confidence, it) }
            ?: getString(R.string.meta_confidence_empty)
        val size = if (result != null && result.frameWidth > 0) {
            getString(R.string.meta_frame_size, result.frameWidth, result.frameHeight)
        } else {
            currentResolution.label
        }
        val modelSize = result?.modelInputSize ?: currentModelProfile.inputSize
        binding.metaText.text = getString(
            R.string.meta_camera_compact,
            stab,
            size,
            sampleFps,
            modelSize,
            boxFps,
            inferenceMs,
            conf
        )
    }

    private fun openHistory() {
        val controller = cameraController
        if (controller?.isSessionActive == true || historyLaunchPending) return
        historyLaunchPending = true
        binding.btnHistory.isEnabled = false
        levelMonitor.stop()
        controller?.pause()
        rulerLocked = false
        // The automatic record save uses this same single-thread executor. Queueing the
        // launch behind it guarantees the just-completed run is visible on the first list.
        // A record retained after an earlier I/O failure gets one more save attempt first.
        ioExecutor.execute {
            for (record in pendingRecordSaves.values.toList()) {
                if (runCatching { measurementHistoryStore.save(record) }.isSuccess) {
                    pendingRecordSaves.remove(record.id)
                }
            }
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                historyLauncher.launch(Intent(this, MeasurementHistoryActivity::class.java))
            }
        }
    }

    private fun invalidateOpticalCalibration() {
        rulerLocked = false
        cameraController?.clearAnalysisReferences()
        binding.overlayView.updateResult(null)
        binding.statusText.setText(R.string.status_searching)
    }

    private fun updateCameraControlAvailability() {
        cameraOptics.setLocked(measurementControlsLocked || videoWorkRunning)
        binding.btnVideoAnalysis.isEnabled = !measurementControlsLocked
        binding.btnRelock.isEnabled = !measurementControlsLocked && !videoWorkRunning
        binding.btnConfig.isEnabled = !measurementControlsLocked && !videoWorkRunning
    }

    private fun restoreCameraAfterVideo() {
        if (destroying || measurementControlsLocked || !videoWorkKnown || videoWorkRunning ||
            videoPageOpen || historyLaunchPending) return
        if (cameraController == null) ensurePermissionAndStart()
        else if (cameraController?.isRunning != true) cameraController?.resume(resetAnalysis = true)
        startLevelIndicator()
    }

    override fun onResume() {
        super.onResume()
        restoreCameraAfterVideo()
        if (!measurementControlsLocked) cameraController?.refreshTubeFocus()
        startLevelIndicator()
    }

    override fun onPause() {
        levelMonitor.stop()
        super.onPause()
    }

    private fun startLevelIndicator() {
        if (videoWorkRunning || videoPageOpen || historyLaunchPending) return
        binding.levelIndicator.visibility =
            if (levelMonitor.start()) View.VISIBLE else View.GONE
    }

    override fun onDestroy() {
        destroying = true
        levelMonitor.stop()
        val cameraShutdown = cameraController?.shutdown()
        cameraController = null
        ioExecutor.shutdown()
        val detectorToClose = if (obbDetectorLazy.isInitialized()) obbDetectorLazy.value else null
        val detectorClosed = AtomicBoolean(false)
        val closeDetector = {
            if (detectorClosed.compareAndSet(false, true)) {
                runCatching { detectorToClose?.close() }.onFailure {
                    android.util.Log.w("MainActivity", "Unable to close YOLO detector", it)
                }
            }
        }
        Thread({
            var forcedIoShutdown = false
            while (!ioExecutor.isTerminated) {
                val waitSeconds = if (forcedIoShutdown) 5L else 3L
                val terminated = try {
                    ioExecutor.awaitTermination(waitSeconds, TimeUnit.SECONDS)
                } catch (_: InterruptedException) {
                    android.util.Log.w(
                        "MainActivity",
                        "I/O executor cleanup interrupted; continuing safely"
                    )
                    false
                }
                if (terminated) break
                if (!forcedIoShutdown) {
                    ioExecutor.shutdownNow()
                    forcedIoShutdown = true
                } else {
                    // Native inference may still be in progress. Do not close the
                    // shared detector until that task has actually left the executor.
                    android.util.Log.w(
                        "MainActivity",
                        "I/O executor still active; delaying YOLO resource close"
                    )
                }
            }
            if (cameraShutdown == null) {
                closeDetector()
                return@Thread
            }
            val cameraFinished = runCatching {
                cameraShutdown.get(CAMERA_SHUTDOWN_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                true
            }.onFailure {
                android.util.Log.w(
                    "MainActivity",
                    "Camera cleanup exceeded ${CAMERA_SHUTDOWN_TIMEOUT_SECONDS}s; " +
                        "YOLO will close when camera cleanup completes",
                    it
                )
            }.getOrDefault(false)
            if (cameraFinished) {
                closeDetector()
            } else {
                cameraShutdown.whenComplete { _, _ -> closeDetector() }
            }
        }, "FallingBall-main-cleanup").start()
        super.onDestroy()
    }
}
