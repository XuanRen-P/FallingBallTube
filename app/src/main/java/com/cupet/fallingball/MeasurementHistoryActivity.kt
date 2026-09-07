package com.cupet.fallingball

import android.graphics.Color
import android.graphics.Typeface
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ListView
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.cupet.fallingball.analysis.CsvExporter
import com.cupet.fallingball.analysis.MeasurementHistoryStore
import com.cupet.fallingball.analysis.MeasurementMetadata
import com.cupet.fallingball.analysis.MeasurementRecord
import com.cupet.fallingball.analysis.MeasurementRecordSummary
import com.cupet.fallingball.analysis.MeasurementSource
import com.cupet.fallingball.analysis.RecordedBallSample
import com.cupet.fallingball.analysis.ViscometryParameters
import com.cupet.fallingball.analysis.ViscosityWarning
import com.cupet.fallingball.ui.DisplacementTimeChartView
import com.cupet.fallingball.ui.UiColors
import com.cupet.fallingball.learning.DataAssistantActivity
import com.cupet.fallingball.learning.LearningServiceFactory
import com.cupet.fallingball.learning.QuestionWorkScheduler
import com.cupet.fallingball.vision.BallEndReason
import com.cupet.fallingball.vision.RulerConfig
import com.cupet.fallingball.vision.WallContactState
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

internal fun lineAlignedViewportHeight(
    maxHeightPx: Int,
    verticalPaddingPx: Int,
    lineHeightPx: Int
): Int {
    if (maxHeightPx <= 0 || lineHeightPx <= 0) return maxHeightPx.coerceAtLeast(0)
    val padding = verticalPaddingPx.coerceIn(0, maxHeightPx)
    val fullLines = ((maxHeightPx - padding) / lineHeightPx).coerceAtLeast(1)
    return (padding + fullLines * lineHeightPx).coerceAtMost(maxHeightPx)
}

internal data class HistoryRowPresentation(
    val enabled: Boolean,
    val subdued: Boolean,
    val statusText: String,
    val detailText: String
)

internal fun isHistoryEntryEnabled(sampleCount: Int): Boolean = sampleCount > 0

internal fun historyRowPresentation(
    sampleCount: Int,
    logOnlyStatusText: String,
    noTrackDetailText: String,
    measuredStatusText: String,
    measuredDetailText: String
): HistoryRowPresentation = if (!isHistoryEntryEnabled(sampleCount)) {
    HistoryRowPresentation(
        enabled = false,
        subdued = true,
        statusText = logOnlyStatusText,
        detailText = noTrackDetailText
    )
} else {
    HistoryRowPresentation(
        enabled = true,
        subdued = false,
        statusText = measuredStatusText,
        detailText = measuredDetailText
    )
}

/**
 * Full-screen history browser. Camera ownership remains in [MainActivity]; that activity
 * unbinds CameraX before launch and rebuilds all analysis references when this screen exits.
 */
class MeasurementHistoryActivity : AppCompatActivity() {

    private enum class Page { LIST, DETAIL, EDIT }

    private val ioExecutor = Executors.newSingleThreadExecutor()
    private lateinit var store: MeasurementHistoryStore
    private lateinit var content: FrameLayout
    private lateinit var title: TextView
    private lateinit var progress: ProgressBar
    private var page = Page.LIST
    private var currentRecord: MeasurementRecord? = null
    private var pendingExportId: String? = null
    private val displayDateFormat = SimpleDateFormat("MM-dd HH:mm:ss", Locale.getDefault())
    private var listFirstVisiblePosition = 0
    private var listTopOffset = 0
    private var detailOnly = false

    private val exportLauncher = registerForActivityResult(
        ActivityResultContracts.CreateDocument("text/csv")
    ) { uri ->
        val recordId = pendingExportId
        pendingExportId = null
        if (uri != null && recordId != null) writeCsv(uri, recordId)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        pendingExportId = savedInstanceState?.getString(STATE_PENDING_EXPORT_ID)
        detailOnly = intent.getBooleanExtra(EXTRA_DETAIL_ONLY, false)
        store = MeasurementHistoryStore(filesDir.resolve("measurement_history"))
        setContentView(buildShell())
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() = navigateBack()
        })
        val requestedRecordId = intent.getStringExtra(EXTRA_RECORD_ID)
        if (detailOnly && requestedRecordId != null) loadDetail(requestedRecordId) else showList()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString(STATE_PENDING_EXPORT_ID, pendingExportId)
        super.onSaveInstanceState(outState)
    }

    private fun buildShell(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(UiColors.PAGE_BACKGROUND)
        }
        val toolbar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(8), dp(4), dp(12), dp(4))
            setBackgroundColor(UiColors.SURFACE)
        }
        val back = ImageButton(this).apply {
            id = R.id.history_back_button
            setImageResource(R.drawable.ic_learning_back)
            setColorFilter(UiColors.ON_SURFACE_VARIANT)
            setBackgroundResource(R.drawable.bg_learning_icon_button)
            contentDescription = getString(R.string.history_back)
            minimumWidth = dp(48)
            minimumHeight = dp(48)
            setOnClickListener { navigateBack() }
        }
        title = TextView(this).apply {
            textSize = 20f
            setTextColor(UiColors.ON_SURFACE)
            setTypeface(typeface, Typeface.BOLD)
            gravity = Gravity.CENTER_VERTICAL
        }
        toolbar.addView(back, LinearLayout.LayoutParams(dp(52), dp(52)))
        toolbar.addView(
            title,
            LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f)
        )
        content = FrameLayout(this)
        root.addView(toolbar, LinearLayout.LayoutParams(-1, dp(60)))
        root.addView(content, LinearLayout.LayoutParams(-1, 0, 1f))
        progress = ProgressBar(this).apply { visibility = View.GONE }
        content.addView(
            progress,
            FrameLayout.LayoutParams(dp(48), dp(48), Gravity.CENTER)
        )
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        return root
    }

    private fun showList() {
        page = Page.LIST
        currentRecord = null
        title.setText(R.string.history_title)
        showLoading(true)
        ioExecutor.execute {
            val result = runCatching { store.scanSummaries() }
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                showLoading(false)
                result.fold(
                    onSuccess = { scan ->
                        renderList(scan.summaries, scan.corruptedFileNames)
                    },
                    onFailure = { showError(R.string.history_load_failed) }
                )
            }
        }
    }

    private fun renderList(
        summaries: List<MeasurementRecordSummary>,
        corruptedFileNames: List<String> = emptyList()
    ) {
        content.removeAllViews()
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        if (corruptedFileNames.isNotEmpty()) {
            val visibleNames = corruptedFileNames.take(3).joinToString(", ")
            root.addView(TextView(this).apply {
                text = getString(
                    R.string.history_corrupted_records,
                    corruptedFileNames.size,
                    visibleNames
                )
                setTextColor(UiColors.ON_ERROR_CONTAINER)
                textSize = 13f
                setPadding(dp(12), dp(10), dp(12), dp(10))
                setBackgroundColor(UiColors.ERROR_CONTAINER)
            }, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        }
        if (summaries.isEmpty()) {
            root.addView(TextView(this).apply {
                setText(R.string.history_empty)
                setTextColor(UiColors.ON_SURFACE_VARIANT)
                textSize = 16f
                gravity = Gravity.CENTER
            }, LinearLayout.LayoutParams(-1, 0, 1f))
            content.addView(root, FrameLayout.LayoutParams(-1, -1))
            return
        }
        title.text = getString(R.string.history_title_count, summaries.size)
        val list = ListView(this).apply {
            dividerHeight = dp(1)
            adapter = SummaryAdapter(summaries)
            setOnItemClickListener { _, _, position, _ ->
                if (!isHistoryEntryEnabled(summaries[position].sampleCount)) {
                    return@setOnItemClickListener
                }
                listFirstVisiblePosition = firstVisiblePosition
                listTopOffset = getChildAt(0)?.top ?: 0
                startActivity(
                    Intent(this@MeasurementHistoryActivity, DataAssistantActivity::class.java)
                        .putExtra(DataAssistantActivity.EXTRA_MEASUREMENT_ID, summaries[position].id)
                )
            }
        }
        root.addView(list, LinearLayout.LayoutParams(-1, 0, 1f))
        content.addView(root, FrameLayout.LayoutParams(-1, -1))
        val restoredPosition = listFirstVisiblePosition.coerceIn(0, summaries.lastIndex)
        list.setSelectionFromTop(restoredPosition, listTopOffset)
    }

    private fun loadDetail(id: String) {
        page = Page.DETAIL
        showLoading(true)
        ioExecutor.execute {
            val result = runCatching { store.load(id) }
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                showLoading(false)
                result.fold(
                    onSuccess = { record ->
                        if (record == null) {
                            Toast.makeText(
                                this,
                                R.string.history_load_failed,
                                Toast.LENGTH_LONG
                            ).show()
                            showList()
                        }
                        else renderDetail(record)
                    },
                    onFailure = {
                        Toast.makeText(
                            this,
                            R.string.history_load_failed,
                            Toast.LENGTH_LONG
                        ).show()
                        showList()
                    }
                )
            }
        }
    }

    private fun renderDetail(record: MeasurementRecord) {
        page = Page.DETAIL
        currentRecord = record
        title.text = recordTitle(record)
        content.removeAllViews()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }
        val overviewScroll = ScrollView(this).apply {
            isVerticalScrollBarEnabled = true
            isScrollbarFadingEnabled = false
            scrollBarStyle = View.SCROLLBARS_INSIDE_INSET
        }
        overviewScroll.addView(buildResultOverview(record))
        val overviewHeight = if (record.sustainedWallContact) dp(270) else dp(230)
        root.addView(overviewScroll, LinearLayout.LayoutParams(-1, overviewHeight))

        val toggle = MaterialButtonToggleGroup(this).apply {
            isSingleSelection = true
            isSelectionRequired = true
        }
        val chartButton = outlinedButton(getString(R.string.history_chart)).apply {
            id = View.generateViewId()
        }
        val dataButton = outlinedButton(getString(R.string.history_frame_data)).apply {
            id = View.generateViewId()
        }
        toggle.addView(chartButton, LinearLayout.LayoutParams(0, dp(48), 1f))
        toggle.addView(dataButton, LinearLayout.LayoutParams(0, dp(48), 1f))
        root.addView(toggle, LinearLayout.LayoutParams(-1, dp(48)))

        val analysisPanel = FrameLayout(this)
        root.addView(analysisPanel, LinearLayout.LayoutParams(-1, 0, 1f))
        val selection = TextView(this).apply {
            setTextColor(UiColors.ON_SURFACE_VARIANT)
            textSize = 13f
            setPadding(dp(4), dp(5), dp(4), dp(5))
            text = if (record.summary?.plateauFound == true) {
                getString(R.string.history_tap_point)
            } else {
                getString(R.string.history_no_terminal_interval)
            }
        }
        root.addView(selection, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))

        fun showChart() {
            analysisPanel.removeAllViews()
            analysisPanel.addView(DisplacementTimeChartView(this).apply {
                setRecord(record)
                onSelectionChanged = { sample ->
                    selection.text = sample?.let(::selectedSampleText)
                        ?: getString(R.string.history_tap_point)
                }
            }, FrameLayout.LayoutParams(-1, -1))
            selection.text = if (record.summary?.plateauFound == true) {
                getString(R.string.history_tap_point)
            } else {
                getString(R.string.history_no_terminal_interval)
            }
        }

        fun showFrames() {
            analysisPanel.removeAllViews()
            val frameRoot = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
            }
            frameRoot.addView(buildFrameHeader(), LinearLayout.LayoutParams(-1, dp(38)))
            frameRoot.addView(ListView(this).apply {
                dividerHeight = dp(1)
                adapter = SampleAdapter(record)
            }, LinearLayout.LayoutParams(-1, 0, 1f))
            analysisPanel.addView(frameRoot, FrameLayout.LayoutParams(-1, -1))
            selection.setText(R.string.history_frame_legend)
        }
        toggle.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            chartButton.setTextColor(
                if (checkedId == chartButton.id) UiColors.PRIMARY else UiColors.ON_SURFACE_VARIANT
            )
            dataButton.setTextColor(
                if (checkedId == dataButton.id) UiColors.PRIMARY else UiColors.ON_SURFACE_VARIANT
            )
            if (checkedId == chartButton.id) showChart() else showFrames()
        }
        toggle.check(chartButton.id)

        val actions = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        val edit = outlinedButton(getString(R.string.history_edit)).apply {
            setOnClickListener { renderEdit(record) }
        }
        val export = outlinedButton(getString(R.string.btn_export_csv)).apply {
            setOnClickListener { beginExport(record) }
        }
        val delete = outlinedButton(getString(R.string.history_delete)).apply {
            setTextColor(UiColors.ERROR)
            setOnClickListener { confirmDelete(record) }
        }
        actions.addView(edit, LinearLayout.LayoutParams(0, dp(50), 1f))
        actions.addView(export, LinearLayout.LayoutParams(0, dp(50), 1f))
        actions.addView(delete, LinearLayout.LayoutParams(0, dp(50), 1f))
        root.addView(actions, LinearLayout.LayoutParams(-1, dp(54)))
        content.addView(root, FrameLayout.LayoutParams(-1, -1))
    }

    private fun renderEdit(record: MeasurementRecord) {
        page = Page.EDIT
        title.setText(R.string.history_edit_record)
        content.removeAllViews()
        val scroll = ScrollView(this)
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(12), dp(18), dp(24))
        }
        scroll.addView(form)
        val name = editField(form, R.string.history_record_name, record.metadata.recordName)
        val liquid = editField(form, R.string.config_liquid_name, record.metadata.liquidName)
        val material =
            editField(form, R.string.config_sphere_material, record.metadata.sphereMaterial)
        val temperature = editField(
            form,
            R.string.config_temperature,
            optionalText(record.metadata.temperatureC),
            signedDecimal = true
        )
        val spacing = editField(
            form,
            R.string.config_spacing_mm,
            optionalText(record.config.tickSpacingMm.toDouble()),
            numeric = true
        )
        val parameters = record.viscometryParameters
        val sphereDiameter = editField(
            form,
            R.string.config_sphere_diameter,
            optionalText(parameters.sphereDiameterMm),
            numeric = true
        )
        val diameter = editField(
            form,
            R.string.config_tube_diameter,
            optionalText(parameters.tubeInnerDiameterMm),
            numeric = true
        )
        val liquidDensity = editField(
            form,
            R.string.config_liquid_density,
            optionalText(parameters.liquidDensityGPerCm3),
            numeric = true
        )
        val sphereDensity = editField(
            form,
            R.string.config_sphere_density,
            optionalText(parameters.sphereDensityGPerCm3),
            numeric = true
        )
        val notes = editField(
            form,
            R.string.history_notes,
            record.metadata.notes,
            multiline = true
        )
        val hint = TextView(this).apply {
            setText(R.string.history_recalculate_hint)
            setTextColor(UiColors.ON_SURFACE_VARIANT)
            textSize = 13f
            setPadding(0, dp(6), 0, dp(10))
        }
        form.addView(hint)
        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val cancel = outlinedButton(getString(android.R.string.cancel)).apply {
            setOnClickListener { renderDetail(record) }
        }
        val save = MaterialButton(this).apply {
            setText(R.string.history_save_changes)
            setOnClickListener {
                saveEdits(
                    record = record,
                    saveButton = this,
                    name = name,
                    liquid = liquid,
                    material = material,
                    temperature = temperature,
                    spacing = spacing,
                    sphereDiameter = sphereDiameter,
                    tubeDiameter = diameter,
                    liquidDensity = liquidDensity,
                    sphereDensity = sphereDensity,
                    notes = notes
                )
            }
        }
        buttons.addView(cancel, LinearLayout.LayoutParams(0, dp(52), 1f))
        buttons.addView(save, LinearLayout.LayoutParams(0, dp(52), 1f))
        form.addView(buttons)
        content.addView(scroll, FrameLayout.LayoutParams(-1, -1))
    }

    @Suppress("LongParameterList")
    private fun saveEdits(
        record: MeasurementRecord,
        saveButton: MaterialButton,
        name: EditText,
        liquid: EditText,
        material: EditText,
        temperature: EditText,
        spacing: EditText,
        sphereDiameter: EditText,
        tubeDiameter: EditText,
        liquidDensity: EditText,
        sphereDensity: EditText,
        notes: EditText
    ) {
        fun optional(input: EditText, positive: Boolean = true): Double? {
            val value = input.text.toString().trim()
            if (value.isEmpty()) return null
            val parsed = value.toDoubleOrNull()
                ?: throw IllegalArgumentException(getString(R.string.history_invalid_number))
            if (!parsed.isFinite()) {
                throw IllegalArgumentException(getString(R.string.history_invalid_number))
            }
            if (positive && parsed <= 0.0) {
                throw IllegalArgumentException(getString(R.string.config_optional_positive))
            }
            return parsed
        }
        val updated = try {
            val spacingValue = optional(spacing)
                ?: throw IllegalArgumentException(getString(R.string.config_invalid_spacing))
            if (spacingValue > RulerConfig.MAX_SPACING_MM) {
                throw IllegalArgumentException(getString(R.string.config_invalid_spacing))
            }
            val newParameters = ViscometryParameters(
                sphereRadiusMm = optional(sphereDiameter)?.div(2.0),
                tubeInnerDiameterMm = optional(tubeDiameter),
                liquidDensityGPerCm3 = optional(liquidDensity),
                sphereDensityGPerCm3 = optional(sphereDensity)
            )
            if (newParameters.sphereRadiusMm != null &&
                newParameters.tubeInnerDiameterMm != null &&
                newParameters.tubeInnerDiameterMm <= newParameters.sphereRadiusMm * 2.0
            ) {
                throw IllegalArgumentException(getString(R.string.config_tube_too_small))
            }
            if (newParameters.liquidDensityGPerCm3 != null &&
                newParameters.sphereDensityGPerCm3 != null &&
                newParameters.sphereDensityGPerCm3 <= newParameters.liquidDensityGPerCm3
            ) {
                throw IllegalArgumentException(getString(R.string.config_sphere_not_denser))
            }
            record.recalculate(
                config = RulerConfig(
                    spacingValue.toFloat(),
                    record.config.expectedTickCount
                ),
                viscometryParameters = newParameters,
                metadata = MeasurementMetadata(
                    recordName = name.text.toString().trim(),
                    liquidName = liquid.text.toString().trim(),
                    sphereMaterial = material.text.toString().trim(),
                    temperatureC = optional(temperature, positive = false),
                    notes = notes.text.toString().trim()
                )
            )
        } catch (error: IllegalArgumentException) {
            Toast.makeText(this, error.message, Toast.LENGTH_LONG).show()
            return
        }
        saveButton.isEnabled = false
        ioExecutor.execute {
            val result = runCatching { store.update(updated) }
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                saveButton.isEnabled = true
                result.fold(
                    onSuccess = {
                        QuestionWorkScheduler.enqueue(this, updated.id, replace = true)
                        Toast.makeText(
                            this,
                            R.string.history_update_ok,
                            Toast.LENGTH_SHORT
                        ).show()
                        renderDetail(updated)
                    },
                    onFailure = {
                        Toast.makeText(
                            this,
                            R.string.history_update_failed,
                            Toast.LENGTH_LONG
                        ).show()
                    }
                )
            }
        }
    }

    private fun editField(
        parent: LinearLayout,
        hintRes: Int,
        value: String,
        numeric: Boolean = false,
        signedDecimal: Boolean = false,
        multiline: Boolean = false
    ): TextInputEditText {
        val layout = TextInputLayout(this).apply {
            hint = getString(hintRes)
            setPadding(0, 0, 0, dp(6))
        }
        val input = TextInputEditText(layout.context).apply {
            setText(value)
            inputType = when {
                multiline -> InputType.TYPE_CLASS_TEXT or
                    InputType.TYPE_TEXT_FLAG_MULTI_LINE or
                    InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
                signedDecimal -> InputType.TYPE_CLASS_NUMBER or
                    InputType.TYPE_NUMBER_FLAG_DECIMAL or
                    InputType.TYPE_NUMBER_FLAG_SIGNED
                numeric -> InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
                else -> InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            }
            if (multiline) {
                minLines = 3
                gravity = Gravity.TOP
            }
        }
        layout.addView(
            input,
            LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT)
        )
        parent.addView(layout, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        return input
    }

    private fun beginExport(record: MeasurementRecord) {
        pendingExportId = record.id
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
            .format(Date(record.recordedAtEpochMs))
        exportLauncher.launch("fall_$stamp.csv")
    }

    private fun writeCsv(uri: Uri, recordId: String) {
        ioExecutor.execute {
            val result = runCatching {
                val record = store.load(recordId) ?: error("Measurement record unavailable")
                val iso = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US)
                    .format(Date(record.recordedAtEpochMs))
                contentResolver.openOutputStream(uri)?.use {
                    CsvExporter.writeTo(it, record, iso)
                } ?: error("openOutputStream returned null")
            }
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                Toast.makeText(
                    this,
                    if (result.isSuccess) R.string.toast_export_ok
                    else R.string.toast_export_failed_short,
                    if (result.isSuccess) Toast.LENGTH_SHORT else Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    private fun confirmDelete(record: MeasurementRecord) {
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.history_delete_title)
            .setMessage(R.string.history_delete_message)
            .setPositiveButton(R.string.history_delete) { _, _ -> delete(record) }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun delete(record: MeasurementRecord) {
        showLoading(true)
        QuestionWorkScheduler.cancel(this, record.id)
        ioExecutor.execute {
            val deleted = runCatching {
                val measurementDeleted = store.delete(record.id)
                if (measurementDeleted) {
                    val learningDeleted = LearningServiceFactory.get(this)
                        .store.deleteMeasurement(record.id)
                    if (!learningDeleted) {
                        check(runCatching { store.save(record) }.isSuccess) {
                            "Learning data delete failed and measurement rollback failed"
                        }
                        error("Learning data delete failed; measurement restored")
                    }
                }
                measurementDeleted
            }.getOrDefault(false)
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                showLoading(false)
                Toast.makeText(
                    this,
                    if (deleted) R.string.history_deleted else R.string.history_delete_failed,
                    Toast.LENGTH_SHORT
                ).show()
                if (deleted) {
                    if (detailOnly) finish() else showList()
                } else {
                    renderDetail(record)
                }
            }
        }
    }

    private fun buildResultOverview(record: MeasurementRecord): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(8), dp(6), dp(8), dp(12))
        }
        addOverviewText(
            root,
            getString(
                R.string.history_recorded_at,
                displayDateFormat.format(Date(record.recordedAtEpochMs))
            ),
            sizeSp = 12f,
            color = UiColors.ON_SURFACE_MUTED
        )
        addOverviewText(
            root,
            sourceText(record.source),
            sizeSp = 12f,
            color = UiColors.ON_SURFACE_MUTED
        )
        record.videoSource?.let { video ->
            addOverviewText(
                root,
                getString(
                    R.string.history_video_source_details,
                    video.sourceName,
                    video.width,
                    video.height,
                    video.rotationDegrees,
                    video.nominalFrameRate?.let {
                        getString(R.string.history_video_frame_rate, it)
                    } ?: getString(R.string.history_video_frame_rate_unknown),
                    video.durationUs / 1_000_000.0,
                    video.clipStartUs / 1_000_000.0,
                    video.clipEndUs / 1_000_000.0,
                    video.decodedFrameCount,
                    video.processedFrameCount,
                    video.inferenceFrameCount
                ),
                sizeSp = 12f,
                color = UiColors.ON_SURFACE_VARIANT,
                topPaddingDp = 4
            )
        }

        val viscosity = record.viscosityResult.takeIf { record.hasReliableTerminalEstimate }
        if (viscosity != null) {
            val wallLimited = ViscosityWarning.OFF_AXIS_OR_WALL_CONTACT in viscosity.warnings
            addOverviewText(
                root,
                getString(
                    if (wallLimited) {
                        R.string.history_final_viscosity_reference_label
                    } else {
                        R.string.history_final_viscosity_label
                    },
                    wallCorrectionMethodText(viscosity.wallCorrectionModel)
                ),
                sizeSp = 13f,
                color = if (wallLimited) UiColors.WARNING else UiColors.SUCCESS,
                bold = true,
                topPaddingDp = 8
            )
            addOverviewText(
                root,
                getString(
                    R.string.history_final_viscosity_value,
                    viscosity.wallCorrectedStokesPaS
                ),
                sizeSp = 30f,
                color = if (wallLimited) UiColors.WARNING else UiColors.ON_PRIMARY_CONTAINER,
                bold = true
            )
            addOverviewText(
                root,
                getString(
                    R.string.history_uncorrected_stokes,
                    viscosity.uncorrectedStokesPaS,
                    viscosity.wallCorrectionFactor,
                    viscosity.diameterRatio,
                    viscosity.reynoldsNumber
                ),
                sizeSp = 12f,
                color = UiColors.ON_SURFACE_VARIANT
            )
        } else {
            addOverviewText(
                root,
                when {
                    !record.hasReliableTerminalEstimate ->
                        getString(R.string.history_viscosity_no_reliable_fit)
                    !record.viscometryParameters.isComplete ->
                        getString(R.string.history_parameters_pending)
                    else -> getString(R.string.history_viscosity_invalid)
                },
                sizeSp = 14f,
                color = UiColors.WARNING,
                bold = true,
                topPaddingDp = 8
            )
        }

        val summary = record.summary
        val reliableSpeed = summary?.terminalVelocityMmPerS?.takeIf {
            record.hasReliableTerminalEstimate
        }
        addOverviewText(
            root,
            getString(
                if (reliableSpeed != null) {
                    R.string.history_fitted_velocity_label
                } else {
                    R.string.history_diagnostic_velocity_label
                }
            ),
            sizeSp = 12f,
            color = UiColors.ON_SURFACE_VARIANT,
            topPaddingDp = 10
        )
        addOverviewText(
            root,
            summary?.terminalVelocityMmPerS?.let {
                getString(R.string.history_fitted_velocity_value, it)
            } ?: getString(R.string.history_velocity_unavailable),
            sizeSp = if (viscosity == null) 28f else 22f,
            color = UiColors.ON_PRIMARY_CONTAINER,
            bold = true
        )
        if (summary?.plateauFound == true) {
            addOverviewText(
                root,
                getString(
                    R.string.history_fit_interval,
                    summary.plateauStartMm,
                    summary.plateauEndMm,
                    summary.plateauDurationMs
                ),
                sizeSp = 12f,
                color = UiColors.ON_SURFACE_VARIANT
            )
        }

        val effectiveFps = record.analysisQuality?.effectiveSampleRateFps ?: record.meanFps
        addOverviewText(
            root,
            getString(
                R.string.history_measurement_meta,
                record.samples.size,
                effectiveFps,
                endReasonText(record.endReason)
            ),
            sizeSp = 12f,
                color = UiColors.ON_SURFACE_VARIANT,
            topPaddingDp = 8
        )

        val wallAssessment = wallAssessmentText(record)
        addOverviewText(
            root,
            wallAssessment.first,
            sizeSp = 13f,
            color = wallAssessment.second,
            bold = record.sustainedWallContact,
            topPaddingDp = 8
        )

        val qualityWarnings = qualityWarnings(record)
        if (qualityWarnings.isNotEmpty()) {
            addOverviewText(
                root,
                qualityWarnings.joinToString("\n"),
                sizeSp = 12f,
                color = UiColors.WARNING,
                topPaddingDp = 8
            )
        }
        if (record.metadata.notes.isNotBlank()) {
            addOverviewText(
                root,
                getString(R.string.history_notes_value, record.metadata.notes),
                sizeSp = 12f,
                color = UiColors.ON_SURFACE_VARIANT,
                topPaddingDp = 8
            )
        }
        return root
    }

    private fun sourceText(source: MeasurementSource): String = getString(
        when (source) {
            MeasurementSource.UNKNOWN -> R.string.history_source_unknown
            MeasurementSource.LIVE_CAMERA -> R.string.history_source_live
            MeasurementSource.IMPORTED_VIDEO -> R.string.history_source_video
        }
    )

    private fun addOverviewText(
        parent: LinearLayout,
        value: CharSequence,
        sizeSp: Float,
        color: Int,
        bold: Boolean = false,
        topPaddingDp: Int = 0
    ) {
        parent.addView(TextView(this).apply {
            text = value
            textSize = sizeSp
            setTextColor(color)
            if (bold) setTypeface(typeface, Typeface.BOLD)
            setLineSpacing(0f, 1.08f)
            setPadding(0, dp(topPaddingDp), 0, 0)
        }, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
    }

    private fun wallCorrectionMethodText(model: String): String = getString(
        when (model) {
            "LADENBURG" -> R.string.history_method_ladenburg
            "FAXEN" -> R.string.history_method_faxen
            "HABERMAN_SAYRE" -> R.string.history_method_haberman_sayre
            else -> R.string.history_method_wall_correction
        }
    )

    private fun wallAssessmentText(record: MeasurementRecord): Pair<String, Int> {
        val averageOffset = getString(
            R.string.history_average_center_offset,
            record.averageCenterOffsetMm?.let(::format) ?: "—"
        )
        if (record.wallContactKnownSampleCount == 0) {
            return buildString {
                append(getString(R.string.history_wall_unknown))
                append('\n').append(averageOffset)
            } to UiColors.ON_SURFACE_VARIANT
        }
        if (record.sustainedWallContact) {
            return buildString {
                append(
                    getString(
                        R.string.history_wall_high_risk,
                        record.wallContactSampleCount,
                        record.wallContactKnownSampleCount
                    )
                )
                append('\n').append(averageOffset)
                append('\n').append(getString(R.string.history_wall_correction_limit))
            } to UiColors.WARNING
        }
        if (record.wallContactSampleCount > 0) {
            return buildString {
                append(
                    getString(
                        R.string.history_wall_isolated,
                        record.wallContactSampleCount,
                        record.wallContactKnownSampleCount
                    )
                )
                append('\n').append(averageOffset)
            } to UiColors.WARNING
        }
        return buildString {
            append(getString(R.string.history_wall_clear, record.wallContactKnownSampleCount))
            append('\n').append(averageOffset)
        } to UiColors.SUCCESS
    }

    private fun qualityWarnings(record: MeasurementRecord): List<String> = buildList {
        val quality = record.analysisQuality
        when {
            quality?.terminalFitCheckUnavailable == true ->
                add(getString(R.string.summary_terminal_fit_check_unavailable))
            quality?.unstableTerminalFit == true -> add(
                getString(
                    R.string.summary_unstable_terminal_fit,
                    (quality.terminalFitRelativeDisagreement ?: 0f) * 100f
                )
            )
        }
        if (quality?.lowSampleRate == true) {
            add(
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
        if (quality != null && quality.outlierSampleCount > 0) {
            add(getString(R.string.summary_outliers_excluded, quality.outlierSampleCount))
        }
        if (quality?.trackDiscontinuity == true) {
            add(getString(R.string.summary_track_discontinuity_warning))
        }
        if (record.hasExcessiveRegistrationShake) {
            add(getString(R.string.summary_shake_warning))
        }
    }

    private fun buildFrameHeader(): View = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        setPadding(dp(6), 0, dp(8), 0)
        setBackgroundColor(UiColors.SURFACE_CONTAINER)
        addView(frameHeaderCell(getString(R.string.history_column_index), Gravity.CENTER),
            LinearLayout.LayoutParams(dp(42), -1))
        addView(frameHeaderCell(getString(R.string.history_column_time), Gravity.CENTER),
            LinearLayout.LayoutParams(dp(76), -1))
        addView(frameHeaderCell(getString(R.string.history_column_position), Gravity.END),
            LinearLayout.LayoutParams(0, -1, 1f))
        addView(frameHeaderCell(getString(R.string.history_column_velocity), Gravity.END),
            LinearLayout.LayoutParams(dp(92), -1))
    }

    private fun frameHeaderCell(value: String, cellGravity: Int) = TextView(this).apply {
        text = value
        textSize = 11f
        gravity = cellGravity or Gravity.CENTER_VERTICAL
        setTextColor(UiColors.ON_SURFACE_VARIANT)
        setTypeface(typeface, Typeface.BOLD)
        setPadding(dp(4), 0, dp(4), 0)
    }

    private fun selectedSampleText(sample: RecordedBallSample): String = getString(
        R.string.history_selected_point,
        sample.elapsedMs / 1000f,
        sample.positionMm?.let { format(it) } ?: "—",
        sample.lateralMm?.let { format(it) } ?: "—",
        wallContactText(sample.wallContactState),
        sample.velocityMmPerS?.let { format(it) } ?: "—",
        if (sample.excludedAsOutlier) {
            getString(R.string.history_outlier)
        } else if (sample.usedInTerminalFit) {
            getString(R.string.history_fit_yes)
        } else {
            getString(R.string.history_fit_no)
        }
    )

    private fun wallContactText(state: WallContactState): String = getString(
        when (state) {
            WallContactState.UNKNOWN -> R.string.wall_state_unknown
            WallContactState.CLEAR -> R.string.wall_state_clear
            WallContactState.LEFT_WALL -> R.string.wall_state_left
            WallContactState.RIGHT_WALL -> R.string.wall_state_right
        }
    )

    private fun recordTitle(record: MeasurementRecord): String =
        record.metadata.recordName.ifBlank {
            displayDateFormat.format(Date(record.recordedAtEpochMs))
        }

    private fun endReasonText(reason: BallEndReason): String = getString(
        when (reason) {
            BallEndReason.BALL_LOST -> R.string.end_ball_lost
            BallEndReason.REACHED_BOTTOM -> R.string.end_reached_bottom
            BallEndReason.MANUAL -> R.string.end_manual
        }
    )

    private fun navigateBack() {
        when (page) {
            Page.LIST -> finish()
            Page.DETAIL -> if (detailOnly) finish() else showList()
            Page.EDIT -> currentRecord?.let(::renderDetail) ?: showList()
        }
    }

    private fun showLoading(show: Boolean) {
        if (show) {
            content.removeAllViews()
            progress = ProgressBar(this)
            content.addView(progress, FrameLayout.LayoutParams(dp(48), dp(48), Gravity.CENTER))
        } else {
            progress.visibility = View.GONE
        }
    }

    private fun showError(message: Int) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        if (page == Page.LIST) renderList(emptyList())
    }

    private fun outlinedButton(label: String) = MaterialButton(
        this,
        null,
        com.google.android.material.R.attr.materialButtonOutlinedStyle
    ).apply {
        text = label
        minWidth = 0
        insetTop = 0
        insetBottom = 0
    }

    private fun optionalText(value: Double?): String = value?.let {
        String.format(Locale.ROOT, "%.6f", it).trimEnd('0').trimEnd('.')
    }.orEmpty()

    private fun format(value: Float) = String.format(Locale.ROOT, "%.3f", value)

    private fun dp(value: Int) = (value * resources.displayMetrics.density + 0.5f).toInt()

    override fun onDestroy() {
        ioExecutor.shutdown()
        super.onDestroy()
    }

    private inner class SummaryAdapter(
        private val items: List<MeasurementRecordSummary>
    ) : BaseAdapter() {
        override fun getCount() = items.size
        override fun getItem(position: Int) = items[position]
        override fun getItemId(position: Int) = position.toLong()
        override fun areAllItemsEnabled() = false
        override fun isEnabled(position: Int) = presentation(items[position]).enabled

        override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
            val view = (convertView as? TextView) ?: TextView(this@MeasurementHistoryActivity).apply {
                setPadding(dp(18), dp(14), dp(18), dp(14))
                setTextColor(UiColors.ON_SURFACE)
                textSize = 15f
                minHeight = dp(88)
            }
            val item = items[position]
            val name = item.metadata.recordName.ifBlank {
                displayDateFormat.format(Date(item.recordedAtEpochMs))
            }
            val presentation = presentation(item)
            view.setTextColor(if (presentation.subdued) UiColors.ON_SURFACE_MUTED else UiColors.ON_SURFACE)
            view.text = getString(
                R.string.history_list_item,
                name,
                presentation.statusText,
                "${sourceText(item.source)} · ${presentation.detailText}"
            )
            return view
        }

        private fun presentation(item: MeasurementRecordSummary): HistoryRowPresentation {
            val velocity = if (item.plateauFound && item.terminalVelocityMmPerS != null) {
                getString(R.string.history_item_terminal, item.terminalVelocityMmPerS)
            } else {
                getString(R.string.history_item_no_plateau)
            }
            val viscosity = when {
                item.plateauFound && item.viscosityResult != null -> getString(
                    R.string.history_item_viscosity,
                    item.viscosityResult.wallCorrectedStokesPaS
                )
                item.plateauFound && item.parametersComplete ->
                    getString(R.string.history_viscosity_invalid)
                else -> ""
            }
            val measuredStatus = if (viscosity.isBlank()) {
                velocity
            } else {
                "$velocity · $viscosity"
            }
            return historyRowPresentation(
                sampleCount = item.sampleCount,
                logOnlyStatusText = getString(R.string.history_item_log_only),
                noTrackDetailText = getString(R.string.history_item_no_track),
                measuredStatusText = measuredStatus,
                measuredDetailText = getString(R.string.history_item_samples, item.sampleCount)
            )
        }
    }

    private inner class SampleAdapter(
        private val record: MeasurementRecord
    ) : BaseAdapter() {
        override fun getCount() = record.samples.size
        override fun getItem(position: Int) = record.samples[position]
        override fun getItemId(position: Int) = position.toLong()

        override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
            return ((convertView as? FrameSampleRow) ?: FrameSampleRow()).apply {
                val sample = record.samples[position]
                bind(position, sample, record.centerOffsetMm(sample))
            }
        }
    }

    private inner class FrameSampleRow : LinearLayout(this@MeasurementHistoryActivity) {
        private val indexText = numericCell(Gravity.CENTER)
        private val timeText = numericCell(Gravity.CENTER)
        private val positionText = numericCell(Gravity.END)
        private val velocityText = numericCell(Gravity.END)
        private val secondaryText = TextView(context).apply {
            textSize = 11f
            setTextColor(UiColors.ON_SURFACE_MUTED)
            maxLines = 1
        }
        private val statusText = TextView(context).apply {
            textSize = 11f
            gravity = Gravity.START
            maxLines = 1
        }

        init {
            orientation = VERTICAL
            setPadding(dp(6), dp(7), dp(8), dp(7))
            minimumHeight = dp(84)
            val primary = LinearLayout(context).apply {
                orientation = HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }
            primary.addView(indexText, LayoutParams(dp(42), dp(24)))
            primary.addView(timeText, LayoutParams(dp(76), dp(24)))
            primary.addView(positionText, LayoutParams(0, dp(24), 1f))
            primary.addView(velocityText, LayoutParams(dp(92), dp(24)))
            addView(primary, LayoutParams(-1, dp(24)))

            val secondary = LinearLayout(context).apply {
                orientation = VERTICAL
                setPadding(dp(46), dp(3), 0, 0)
            }
            secondary.addView(secondaryText, LayoutParams(-1, dp(20)))
            secondary.addView(statusText, LayoutParams(-1, dp(20)))
            addView(secondary, LayoutParams(-1, dp(43)))
        }

        fun bind(index: Int, sample: RecordedBallSample, centerOffsetMm: Float?) {
            indexText.text = String.format(Locale.ROOT, "%03d", index + 1)
            timeText.text = String.format(Locale.ROOT, "%.3f s", sample.elapsedMs / 1000f)
            positionText.text = sample.positionMm?.let {
                String.format(Locale.ROOT, "%.2f mm", it)
            } ?: "—"
            velocityText.text = sample.velocityMmPerS?.let {
                String.format(Locale.ROOT, "%.2f mm/s", it)
            } ?: "—"
            secondaryText.text = getString(
                R.string.history_frame_secondary,
                sample.positionTicks?.let { String.format(Locale.ROOT, "%.3f", it) } ?: "—",
                centerOffsetMm?.let { String.format(Locale.ROOT, "%.3f", it) } ?: "—"
            )

            val states = buildList {
                add(
                    when {
                        sample.excludedAsOutlier -> getString(R.string.history_outlier)
                        sample.usedInTerminalFit -> getString(R.string.history_fit_yes)
                        else -> getString(R.string.history_fit_no)
                    }
                )
                if (!sample.rulerValid) add(getString(R.string.history_ruler_invalid))
                if (sample.wallContactState == WallContactState.LEFT_WALL ||
                    sample.wallContactState == WallContactState.RIGHT_WALL
                ) {
                    add(wallContactText(sample.wallContactState))
                }
            }
            statusText.text = states.joinToString(" · ")
            val stateColor = when {
                sample.excludedAsOutlier || !sample.rulerValid -> UiColors.ERROR
                sample.wallContactState == WallContactState.LEFT_WALL ||
                    sample.wallContactState == WallContactState.RIGHT_WALL -> UiColors.WARNING
                sample.usedInTerminalFit -> UiColors.SUCCESS
                else -> UiColors.ON_SURFACE_MUTED
            }
            statusText.setTextColor(stateColor)
            positionText.setTextColor(stateColor)
            setBackgroundColor(
                if (index % 2 == 0) UiColors.SURFACE else UiColors.SURFACE_TINT
            )
            contentDescription = getString(
                R.string.history_frame_accessibility,
                index + 1,
                timeText.text,
                positionText.text,
                velocityText.text,
                statusText.text
            )
        }

        private fun numericCell(cellGravity: Int) = TextView(context).apply {
            textSize = 12f
            gravity = cellGravity or Gravity.CENTER_VERTICAL
            setTextColor(UiColors.ON_SURFACE_VARIANT)
            typeface = Typeface.MONOSPACE
            setPadding(dp(4), 0, dp(4), 0)
            maxLines = 1
        }
    }

    companion object {
        const val STATE_PENDING_EXPORT_ID = "pending_export_id"
        const val EXTRA_RECORD_ID = "record_id"
        const val EXTRA_DETAIL_ONLY = "detail_only"
    }
}
