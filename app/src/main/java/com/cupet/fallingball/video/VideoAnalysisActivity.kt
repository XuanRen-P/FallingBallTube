package com.cupet.fallingball.video

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.MediaController
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import android.widget.VideoView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import androidx.work.WorkInfo
import com.cupet.fallingball.MeasurementHistoryActivity
import com.cupet.fallingball.R
import com.cupet.fallingball.analysis.MeasurementDefaults
import com.cupet.fallingball.ui.RulerConfigStore
import com.cupet.fallingball.ui.UiColors
import com.cupet.fallingball.vision.ModelProfile
import com.cupet.fallingball.vision.RulerConfig
import com.google.android.material.button.MaterialButton
import com.google.android.material.checkbox.MaterialCheckBox
import com.google.android.material.slider.RangeSlider
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.UUID

internal fun shouldRestoreVideoTaskSnapshot(
    taskId: UUID,
    displayedTaskId: UUID?,
    selectedNewVideo: Boolean,
    state: WorkInfo.State
): Boolean = taskId != displayedTaskId && (!selectedNewVideo || !state.isFinished)

class VideoAnalysisActivity : AppCompatActivity() {
    private lateinit var preview: VideoView
    private lateinit var previewFrame: FrameLayout
    private lateinit var nameText: TextView
    private lateinit var formatText: TextView
    private lateinit var parameterText: TextView
    private lateinit var statusText: TextView
    private lateinit var progressText: TextView
    private lateinit var range: RangeSlider
    private lateinit var startInput: TextInputEditText
    private lateinit var endInput: TextInputEditText
    private lateinit var normalSpeed: MaterialCheckBox
    private lateinit var progress: ProgressBar
    private lateinit var importButton: MaterialButton
    private lateinit var parameterButton: MaterialButton
    private lateinit var startButton: MaterialButton
    private lateinit var cancelButton: MaterialButton
    private lateinit var resultButton: MaterialButton
    private var metadata: VideoMetadata? = null
    private var config = RulerConfig.DEFAULT
    private var defaults = MeasurementDefaults()
    private var clipStartUs = 0L
    private var clipEndUs = 0L
    private var running = false
    private var loading = false
    private var submitting = false
    private var updatingRange = false
    private var activeRecordId: String? = null
    private var loadGeneration = 0
    private var selectedNewVideo = false
    private var latestWork: WorkInfo? = null
    private var openResultWhenComplete = false
    private var displayedTaskId: UUID? = null

    private val notifications = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        enqueueAnalysis()
    }
    private val chooseVideo = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        try {
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            selectedNewVideo = true
            displayedTaskId = null
            activeRecordId = null
            normalSpeed.isChecked = false
            loadVideo(uri.toString())
        } catch (failure: SecurityException) {
            showMessage(getString(R.string.video_permission_failed))
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        config = RulerConfigStore.load(this)
        defaults = RulerConfigStore.loadMeasurementDefaults(this)
        setContentView(buildView())
        updateParameterText()
        val savedDraft = savedInstanceState?.getString(STATE_REQUEST)
            ?.let { runCatching { VideoAnalysisRequest.fromJson(it) }.getOrNull() }
        val savedRequest = savedDraft ?: VideoAnalysisWork.lastRequest(this)
        selectedNewVideo = savedInstanceState?.getBoolean(STATE_NEW_VIDEO) ?: false
        normalSpeed.isChecked = savedInstanceState?.getBoolean(STATE_NORMAL_SPEED) ?: false
        openResultWhenComplete = savedInstanceState?.getBoolean(STATE_OPEN_RESULT) ?: false
        displayedTaskId = if (savedDraft != null) {
            savedInstanceState?.getString(STATE_DISPLAYED_TASK_ID)
                ?.let { runCatching { UUID.fromString(it) }.getOrNull() }
        } else if (!selectedNewVideo) VideoAnalysisWork.lastId(this) else null
        savedRequest?.let { restoreRequest(it) }
        VideoAnalysisWork.observe(this).observe(this) { infos ->
            running = VideoAnalysisWork.isRunning(infos)
            val expectedId = VideoAnalysisWork.lastId(this)
            latestWork = infos.firstOrNull { !it.state.isFinished }
                ?: infos.firstOrNull { it.id == expectedId }
                ?: infos.firstOrNull()
            latestWork?.let { work ->
                if (shouldRestoreVideoTaskSnapshot(work.id, displayedTaskId, selectedNewVideo, work.state)) {
                    VideoAnalysisWork.requestForTask(this, work.id)?.let { request ->
                        displayedTaskId = work.id
                        selectedNewVideo = false
                        restoreRequest(request)
                    }
                }
            }
            renderWork(latestWork)
            updateEnabled()
        }
    }

    override fun onPause() {
        preview.pause()
        super.onPause()
    }

    override fun onDestroy() {
        preview.stopPlayback()
        super.onDestroy()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        currentRequest()?.let { outState.putString(STATE_REQUEST, it.toJson()) }
        outState.putBoolean(STATE_NORMAL_SPEED, normalSpeed.isChecked)
        outState.putBoolean(STATE_NEW_VIDEO, selectedNewVideo)
        outState.putBoolean(STATE_OPEN_RESULT, openResultWhenComplete)
        outState.putString(STATE_DISPLAYED_TASK_ID, displayedTaskId?.toString())
        super.onSaveInstanceState(outState)
    }

    private fun buildView(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(UiColors.PAGE_BACKGROUND)
        }
        val toolbar = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            setBackgroundColor(UiColors.SURFACE)
            setPadding(dp(8), dp(4), dp(12), dp(4))
        }
        toolbar.addView(ImageButton(this).apply {
            setImageResource(R.drawable.ic_learning_back)
            setColorFilter(UiColors.ON_SURFACE_VARIANT)
            setBackgroundResource(R.drawable.bg_learning_icon_button)
            contentDescription = getString(R.string.history_back)
            setOnClickListener { finish() }
        }, LinearLayout.LayoutParams(dp(48), dp(48)))
        toolbar.addView(text(getString(R.string.video_title), 20f), LinearLayout.LayoutParams(0, -2, 1f))
        root.addView(toolbar, LinearLayout.LayoutParams(-1, dp(60)))
        val body = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(12), dp(16), dp(20))
        }
        nameText = text(getString(R.string.video_none_selected), 16f)
        formatText = text("", 13f, UiColors.ON_SURFACE_VARIANT)
        importButton = command(R.string.video_import, android.R.drawable.ic_menu_upload) {
            chooseVideo.launch(arrayOf("video/*"))
        }
        body.addView(importButton, rowLayout())
        body.addView(nameText, rowLayout())
        body.addView(formatText, rowLayout())
        previewFrame = FrameLayout(this).apply { setBackgroundColor(android.graphics.Color.BLACK) }
        preview = VideoView(this)
        previewFrame.addView(preview, FrameLayout.LayoutParams(-1, -1, Gravity.CENTER))
        val controller = MediaController(this)
        controller.setAnchorView(previewFrame)
        preview.setMediaController(controller)
        preview.setOnErrorListener { _, _, _ ->
            showMessage(getString(R.string.video_preview_failed))
            true
        }
        preview.setOnPreparedListener { player ->
            player.isLooping = false
            preview.seekTo((clipStartUs / 1_000L).coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
        }
        body.addView(previewFrame, LinearLayout.LayoutParams(-1, dp(220)).apply {
            topMargin = dp(10)
            bottomMargin = dp(10)
        })
        previewFrame.visibility = View.GONE
        range = RangeSlider(this).apply {
            valueFrom = 0f
            valueTo = 1f
            values = listOf(0f, 1f)
            setLabelFormatter { value -> formatSeconds((value * 1_000_000).toLong()) }
            addOnChangeListener { _, _, fromUser ->
                if (fromUser && !updatingRange) {
                    val video = metadata ?: return@addOnChangeListener
                    clipStartUs = (values[0].toDouble() * 1_000_000.0).toLong().coerceIn(0, video.durationUs - 1)
                    clipEndUs = if (values[1] == valueTo) video.durationUs
                    else (values[1].toDouble() * 1_000_000.0).toLong().coerceAtMost(video.durationUs)
                    updateRangeInputs()
                    preview.seekTo((clipStartUs / 1_000L).toInt())
                    updateEnabled()
                }
            }
        }
        body.addView(range, rowLayout())
        val times = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        fun timeInput(label: Int): TextInputEditText {
            val layout = TextInputLayout(this).apply { hint = getString(label) }
            val input = TextInputEditText(layout.context).apply {
                inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
                setSingleLine(true)
                setOnFocusChangeListener { _, focused -> if (!focused) applyTimeInputs() }
                setOnEditorActionListener { _, _, _ -> applyTimeInputs(); false }
            }
            layout.addView(input, LinearLayout.LayoutParams(-1, -2))
            times.addView(layout, LinearLayout.LayoutParams(0, -2, 1f).apply {
                marginEnd = dp(6)
            })
            return input
        }
        startInput = timeInput(R.string.video_start_seconds)
        endInput = timeInput(R.string.video_end_seconds)
        body.addView(times, rowLayout())
        parameterText = text("", 14f, UiColors.ON_SURFACE_VARIANT)
        body.addView(parameterText, rowLayout())
        parameterButton = command(R.string.video_parameters, android.R.drawable.ic_menu_manage) {
            RulerConfigStore.prompt(this, config, ModelProfile.ACCURATE_640,
                RulerConfigStore.loadAnalysisResolution(this), defaults,
                cameraSettingsEnabled = false, cancellable = true,
                cameraSettingsVisible = false, persistChanges = false
            ) { ruler, _, _, snapshot ->
                config = ruler
                defaults = snapshot
                updateParameterText()
            }
        }
        body.addView(parameterButton, rowLayout())
        normalSpeed = MaterialCheckBox(this).apply {
            setText(R.string.video_normal_speed)
            setOnCheckedChangeListener { _, _ -> updateEnabled() }
        }
        body.addView(normalSpeed, rowLayout())
        startButton = command(R.string.video_start_analysis, android.R.drawable.ic_media_play) {
            if (!applyTimeInputs()) return@command
            preview.pause()
            if (Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(this,
                    Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
            } else enqueueAnalysis()
        }
        body.addView(startButton, rowLayout())
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 100 }
        body.addView(progress, LinearLayout.LayoutParams(-1, dp(6)).apply { topMargin = dp(16) })
        statusText = text("", 15f)
        progressText = text("", 13f, UiColors.ON_SURFACE_VARIANT)
        body.addView(statusText, rowLayout())
        body.addView(progressText, rowLayout())
        cancelButton = command(android.R.string.cancel, android.R.drawable.ic_menu_close_clear_cancel) {
            VideoAnalysisWork.cancel(this)
            cancelButton.isEnabled = false
        }
        body.addView(cancelButton, rowLayout())
        resultButton = command(R.string.video_open_result, android.R.drawable.ic_menu_view) {
            openResult()
        }
        body.addView(resultButton, rowLayout())
        root.addView(ScrollView(this).apply { addView(body) }, LinearLayout.LayoutParams(-1, 0, 1f))
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        return root
    }

    private fun restoreRequest(request: VideoAnalysisRequest) {
        config = request.rulerConfig
        defaults = request.defaults
        updateParameterText()
        loadVideo(request.uri, request.clipStartUs, request.clipEndUs)
    }

    private fun loadVideo(uri: String, startUs: Long = 0L, endUs: Long? = null) {
        val generation = ++loadGeneration
        loading = true
        updateEnabled()
        lifecycleScope.launch {
            val loaded = withContext(Dispatchers.IO) {
                runCatching { VideoFrameDecoder(applicationContext).metadata(android.net.Uri.parse(uri)) }
            }
            if (generation != loadGeneration) return@launch
            loading = false
            loaded.fold(onSuccess = { video ->
                metadata = video
                clipStartUs = startUs.coerceIn(0L, video.durationUs - 1L)
                clipEndUs = (endUs ?: video.durationUs).coerceIn(clipStartUs + 1L, video.durationUs)
                nameText.text = video.videoName
                formatText.text = getString(R.string.video_format, video.width, video.height,
                    formatSeconds(video.durationUs), video.nominalFrameRate?.let {
                        String.format(Locale.getDefault(), "%.2f fps", it)
                    }.orEmpty())
                previewFrame.visibility = View.VISIBLE
                val availableWidth = resources.displayMetrics.widthPixels - dp(32)
                previewFrame.layoutParams.height = (availableWidth.toDouble() * video.displayHeight /
                    video.displayWidth).toInt().coerceIn(dp(160), dp(320))
                previewFrame.requestLayout()
                preview.setVideoURI(android.net.Uri.parse(uri))
                updateRange()
            }, onFailure = { failure ->
                metadata = null
                statusText.text = failure.message ?: getString(R.string.video_load_failed)
                statusText.setTextColor(UiColors.ERROR)
                previewFrame.visibility = View.GONE
            })
            updateEnabled()
        }
    }

    private fun updateRange() {
        val video = metadata ?: return
        updatingRange = true
        range.values = listOf(range.valueFrom, range.valueTo)
        range.valueTo = (video.durationUs / 1_000_000.0).toFloat().coerceAtLeast(0.001f)
        range.values = listOf((clipStartUs / 1_000_000.0).toFloat().coerceAtMost(range.valueTo),
            (clipEndUs / 1_000_000.0).toFloat().coerceAtMost(range.valueTo))
        updatingRange = false
        updateRangeInputs()
    }

    private fun updateRangeInputs() {
        startInput.setText(String.format(Locale.ROOT, "%.3f", clipStartUs / 1_000_000.0))
        endInput.setText(String.format(Locale.ROOT, "%.3f", clipEndUs / 1_000_000.0))
        startInput.error = null
        endInput.error = null
    }

    private fun applyTimeInputs(): Boolean {
        if (updatingRange || running) return false
        val video = metadata ?: return false
        val start = startInput.text.toString().toDoubleOrNull()
        val end = endInput.text.toString().toDoubleOrNull()
        if (start == null || !start.isFinite() || start < 0 || start >= video.durationUs / 1_000_000.0) {
            startInput.error = getString(R.string.video_invalid_start)
            return false
        }
        if (end == null || !end.isFinite() || end <= start || end * 1_000_000 > video.durationUs + 500L) {
            endInput.error = getString(R.string.video_invalid_end)
            return false
        }
        if (startInput.text.toString() != String.format(Locale.ROOT, "%.3f", clipStartUs / 1_000_000.0)) {
            clipStartUs = (start * 1_000_000.0).toLong()
        }
        if (endInput.text.toString() != String.format(Locale.ROOT, "%.3f", clipEndUs / 1_000_000.0)) {
            clipEndUs = (end * 1_000_000.0).toLong().coerceAtMost(video.durationUs)
        }
        if (clipEndUs <= clipStartUs) {
            endInput.error = getString(R.string.video_invalid_end)
            return false
        }
        updateRange()
        return true
    }

    private fun currentRequest(): VideoAnalysisRequest? {
        val video = metadata ?: return null
        if (clipEndUs <= clipStartUs) return null
        return VideoAnalysisRequest(video.uri, video.videoName, clipStartUs, clipEndUs,
            config, defaults, System.currentTimeMillis())
    }

    private fun enqueueAnalysis() {
        if (running || submitting || !normalSpeed.isChecked) return
        val request = currentRequest() ?: return
        submitting = true
        selectedNewVideo = false
        openResultWhenComplete = true
        updateEnabled()
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { VideoAnalysisWork.enqueue(applicationContext, request) }
            }
            submitting = false
            result.onFailure { statusText.text = it.message ?: getString(R.string.video_start_failed) }
            updateEnabled()
        }
    }

    private fun renderWork(work: WorkInfo?) {
        if (selectedNewVideo && !running) {
            statusText.text = ""
            progressText.text = ""
            progress.progress = 0
            activeRecordId = null
            return
        }
        activeRecordId = null
        statusText.setTextColor(UiColors.ON_SURFACE)
        if (work == null) return
        when (work.state) {
            WorkInfo.State.SUCCEEDED -> {
                activeRecordId = work.outputData.getString(VideoAnalysisWorker.KEY_RECORD_ID)
                progress.isIndeterminate = false
                progress.progress = 100
                statusText.setText(R.string.video_complete)
                progressText.text = ""
                if (openResultWhenComplete) {
                    openResultWhenComplete = false
                    resultButton.post { if (!isFinishing && !isDestroyed) openResult() }
                }
            }
            WorkInfo.State.FAILED -> {
                statusText.text = work.outputData.getString(VideoAnalysisWorker.KEY_ERROR)
                    ?: getString(R.string.video_analysis_failed)
                statusText.setTextColor(UiColors.ERROR)
                progress.isIndeterminate = false
            }
            WorkInfo.State.CANCELLED -> {
                statusText.setText(R.string.video_cancelled)
                progress.isIndeterminate = false
            }
            WorkInfo.State.ENQUEUED, WorkInfo.State.BLOCKED -> {
                statusText.setText(if (work.runAttemptCount > 0) R.string.video_restarting else R.string.video_queued)
                progress.isIndeterminate = true
            }
            WorkInfo.State.RUNNING -> {
                openResultWhenComplete = true
                val data = work.progress
                statusText.setText(when (data.getString(VideoAnalysisWorker.KEY_PHASE)) {
                    VideoAnalysisWorker.PHASE_CALIBRATING -> R.string.video_phase_calibrating
                    VideoAnalysisWorker.PHASE_ANALYZING -> R.string.video_phase_analyzing
                    VideoAnalysisWorker.PHASE_FINALIZING -> R.string.video_phase_finalizing
                    VideoAnalysisWorker.PHASE_SAVING -> R.string.video_phase_saving
                    else -> R.string.video_phase_initializing
                })
                val percent = data.getInt(VideoAnalysisWorker.KEY_PERCENT, 0)
                progress.isIndeterminate = percent == 0
                progress.progress = percent
                progressText.text = getString(R.string.video_progress,
                    formatSeconds(data.getLong(VideoAnalysisWorker.KEY_PROCESSED_US, 0)),
                    data.getLong(VideoAnalysisWorker.KEY_FRAME_COUNT, 0),
                    data.getLong(VideoAnalysisWorker.KEY_INFERRED_COUNT, 0), percent) +
                    data.getDouble(VideoAnalysisWorker.KEY_PROCESSING_FPS, 0.0).takeIf { it > 0.0 }?.let { fps ->
                        "\n" + getString(R.string.video_processing_speed, fps) +
                            if (data.getBoolean(VideoAnalysisWorker.KEY_NPU_ACTIVE, false))
                                getString(R.string.video_npu_active) else ""
                    }.orEmpty()
            }
        }
    }

    private fun updateEnabled() {
        val editable = !running && !submitting && !loading
        importButton.isEnabled = editable
        parameterButton.isEnabled = editable
        range.isEnabled = editable && metadata != null
        startInput.isEnabled = editable && metadata != null
        endInput.isEnabled = editable && metadata != null
        normalSpeed.isEnabled = editable
        startButton.isEnabled = editable && metadata != null && normalSpeed.isChecked && clipEndUs > clipStartUs
        startButton.setText(if (latestWork?.state == WorkInfo.State.FAILED ||
            latestWork?.state == WorkInfo.State.CANCELLED) R.string.video_retry else R.string.video_start_analysis)
        cancelButton.visibility = if (running) View.VISIBLE else View.GONE
        cancelButton.isEnabled = running
        resultButton.visibility = if (activeRecordId != null) View.VISIBLE else View.GONE
        progress.visibility = if (latestWork != null && !selectedNewVideo) View.VISIBLE else View.GONE
    }

    private fun updateParameterText() {
        parameterText.text = getString(R.string.video_parameter_summary,
            config.tickSpacingMm.toString(), defaults.liquidName.ifBlank { getString(R.string.video_unset) },
            defaults.sphereMaterial.ifBlank { getString(R.string.video_unset) })
    }

    private fun openResult() {
        activeRecordId?.let { id ->
            startActivity(Intent(this, MeasurementHistoryActivity::class.java)
                .putExtra(MeasurementHistoryActivity.EXTRA_RECORD_ID, id)
                .putExtra(MeasurementHistoryActivity.EXTRA_DETAIL_ONLY, true))
        }
    }

    private fun text(value: String, size: Float, color: Int = UiColors.ON_SURFACE) = TextView(this).apply {
        text = value
        textSize = size
        setTextColor(color)
    }

    private fun command(label: Int, icon: Int, action: () -> Unit) = MaterialButton(this).apply {
        setText(label)
        setIconResource(icon)
        cornerRadius = dp(6)
        setOnClickListener { action() }
    }

    private fun rowLayout() = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
        ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(6) }

    private fun formatSeconds(timeUs: Long): String = String.format(Locale.getDefault(),
        "%02d:%06.3f", timeUs / 60_000_000L, (timeUs % 60_000_000L) / 1_000_000.0)

    private fun showMessage(message: String) = Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()

    private companion object {
        const val STATE_REQUEST = "video_request"
        const val STATE_NORMAL_SPEED = "normal_speed"
        const val STATE_NEW_VIDEO = "new_video"
        const val STATE_OPEN_RESULT = "open_result"
        const val STATE_DISPLAYED_TASK_ID = "displayed_task_id"
    }
}
