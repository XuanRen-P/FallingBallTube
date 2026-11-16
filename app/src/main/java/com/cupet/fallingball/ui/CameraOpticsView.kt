package com.cupet.fallingball.ui

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.drawable.ColorDrawable
import android.graphics.drawable.GradientDrawable
import android.os.Handler
import android.os.Looper
import android.util.AttributeSet
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.HorizontalScrollView
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.PopupMenu
import android.widget.SeekBar
import android.widget.TextView
import androidx.appcompat.widget.AppCompatSeekBar
import androidx.appcompat.widget.AppCompatToggleButton
import androidx.core.content.ContextCompat
import com.cupet.fallingball.R
import com.cupet.fallingball.camera.CameraControlsController
import com.cupet.fallingball.camera.CameraOpticsPolicy
import com.cupet.fallingball.camera.CameraOpticsState
import java.util.Locale
import kotlin.math.abs

/**
 * Optics bar for the camera page.
 *
 * The always-visible bar is a single flat [BAR_HEIGHT_DP] row carrying the four controls an
 * operator touches while framing the tube: camera switch, zoom presets, automatic/manual focus and
 * the expand toggle. Every control in it is square or shorter than it is wide, so the focus toggle
 * reads as a circle instead of a tall oval. The focus ruler and the continuous zoom slider live in
 * a [TUNING_HEIGHT_DP] panel that opens on demand - two flat rows, each keeping its slider at full
 * width - closes by itself after [AUTO_COLLAPSE_MS] without touch, and closes as soon as the
 * measurement locks the optics. Collapsed height is therefore 40dp instead of the previous 152dp
 * three-row stack (focus row + preset row + slider row), which returned most of the preview to the
 * tube without removing any control.
 *
 * The bar keeps reporting the current optics while collapsed: the confirmed preset is highlighted,
 * a confirmed ratio between presets is appended to the preset row, and the focus toggle shows the
 * active AF/MF mode. All state and camera behavior still live in CameraControlsController.
 */
class CameraOpticsView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : LinearLayout(context, attrs) {
    private val accent = ContextCompat.getColor(context, R.color.camera_accent)
    private val primary = ContextCompat.getColor(context, R.color.camera_text_primary)
    private val muted = Color.argb(155, 255, 255, 255)
    private val pendingZoom = Color.argb(200, Color.red(accent), Color.green(accent), Color.blue(accent))
    private val handler = Handler(Looper.getMainLooper())
    private var controls: CameraControlsController? = null
    private var rendering = false
    private var trackingZoom = false
    private var expanded = false
    private var state = CameraOpticsState()
    private var renderedPresets = emptyList<Float>()
    private var activeChip: View? = null
    private val auto = AppCompatToggleButton(context).apply {
        id = R.id.camera_optics_focus_toggle
        textOn = "A"
        textOff = "A"
        text = "A"
        textSize = 22f
        minWidth = 0
        minimumWidth = 0
        minHeight = 0
        minimumHeight = 0
        setPadding(0, 0, 0, 0)
        isChecked = true
        contentDescription = context.getString(R.string.camera_optics_auto)
        tooltipText = contentDescription
        setOnCheckedChangeListener { _, checked ->
            if (!rendering) {
                controls?.setAutomaticFocus(checked)
                render(controls?.state ?: state)
            }
        }
    }
    private val camera = ImageButton(context).apply {
        setImageResource(android.R.drawable.ic_menu_camera)
        imageTintList = ColorStateList.valueOf(primary)
        setBackgroundColor(Color.TRANSPARENT)
        setPadding(dp(10), dp(10), dp(10), dp(10))
        contentDescription = context.getString(R.string.camera_optics_select)
        tooltipText = contentDescription
        setOnClickListener { showCameraMenu() }
    }
    private val expand = ImageButton(context).apply {
        id = R.id.camera_optics_expand
        setImageResource(R.drawable.ic_camera_optics_chevron)
        setBackgroundColor(Color.TRANSPARENT)
        setPadding(dp(12), dp(12), dp(12), dp(12))
        contentDescription = context.getString(R.string.camera_optics_expand)
        tooltipText = contentDescription
        setOnClickListener { setExpanded(!expanded) }
    }
    private val presets = LinearLayout(context).apply {
        orientation = HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL or Gravity.CENTER_HORIZONTAL
    }
    private val presetScroll = HorizontalScrollView(context).apply {
        id = R.id.camera_optics_presets
        isHorizontalScrollBarEnabled = true
        isFillViewport = true
        addView(presets, ViewGroup.LayoutParams(LayoutParams.WRAP_CONTENT, dp(BAR_HEIGHT_DP)))
    }
    private val currentChip = label("", 12f).apply {
        id = R.id.camera_optics_current
        minimumWidth = dp(CHIP_WIDTH_DP)
        setPadding(dp(6), 0, dp(6), 0)
        isClickable = false
        isFocusable = false
        background = chipBackground(selected = true)
        setTextColor(accent)
    }
    private val focusLabel = label("MF 50", 11f)
    private val focus = FocusRuler(context).apply {
        id = R.id.camera_optics_focus_ruler
        max = 100
        progress = 50
        contentDescription = context.getString(R.string.camera_optics_manual)
        setOnSeekBarChangeListener(sliderListener { controls?.setFocusPosition(it) })
    }
    private val zoomLabel = label("1x", 11f).apply { maxLines = 2 }
    private val zoom = AppCompatSeekBar(context).apply {
        id = R.id.camera_optics_zoom_slider
        max = 1000
        progressTintList = ColorStateList.valueOf(accent)
        thumbTintList = ColorStateList.valueOf(accent)
        contentDescription = context.getString(R.string.camera_optics_zoom)
        setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                if (fromUser && !rendering) controls?.setZoomRatio(
                    CameraOpticsPolicy.zoomRatio(progress, state.minimumZoomRatio, state.maximumZoomRatio)
                )
            }
            override fun onStartTrackingTouch(seekBar: SeekBar) { trackingZoom = true }
            override fun onStopTrackingTouch(seekBar: SeekBar) {
                trackingZoom = false
                render(state)
            }
        })
    }
    private val tuningPanel = LinearLayout(context).apply {
        id = R.id.camera_optics_tuning
        orientation = VERTICAL
        visibility = GONE
    }
    private val bar = LinearLayout(context).apply {
        id = R.id.camera_optics_bar
        orientation = HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
    }
    private val collapse = Runnable { setExpanded(false) }

    init {
        orientation = VERTICAL
        // Two flat rows, each keeping its slider at full width: 近 [ruler] 远 MF62 / [slider] 3.2x
        val focusRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        focusRow.addView(
            label(context.getString(R.string.camera_optics_near), 10f),
            LayoutParams(dp(NEAR_FAR_WIDTH_DP), LayoutParams.MATCH_PARENT)
        )
        focusRow.addView(focus, LayoutParams(0, LayoutParams.MATCH_PARENT, 1f))
        focusRow.addView(
            label(context.getString(R.string.camera_optics_far), 10f),
            LayoutParams(dp(NEAR_FAR_WIDTH_DP), LayoutParams.MATCH_PARENT)
        )
        focusRow.addView(focusLabel, LayoutParams(dp(FOCUS_VALUE_WIDTH_DP), LayoutParams.MATCH_PARENT))
        tuningPanel.addView(focusRow, LayoutParams(LayoutParams.MATCH_PARENT, dp(TUNING_ROW_HEIGHT_DP)))

        val zoomRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        zoomRow.addView(zoom, LayoutParams(0, LayoutParams.MATCH_PARENT, 1f))
        zoomRow.addView(zoomLabel, LayoutParams(dp(ZOOM_VALUE_WIDTH_DP), LayoutParams.MATCH_PARENT))
        tuningPanel.addView(zoomRow, LayoutParams(LayoutParams.MATCH_PARENT, dp(TUNING_ROW_HEIGHT_DP)))

        // Square controls in a flat row: the focus toggle has to read as a circle, not an oval.
        bar.addView(camera, LayoutParams(dp(ICON_SIZE_DP), dp(ICON_SIZE_DP)))
        bar.addView(presetScroll, LayoutParams(0, dp(BAR_HEIGHT_DP), 1f))
        bar.addView(auto, LayoutParams(dp(ICON_SIZE_DP), dp(ICON_SIZE_DP)))
        bar.addView(expand, LayoutParams(dp(ICON_SIZE_DP), dp(ICON_SIZE_DP)))

        // The tuning panel opens upwards so the bar keeps its distance to the measurement buttons.
        addView(tuningPanel, LayoutParams(LayoutParams.MATCH_PARENT, dp(TUNING_HEIGHT_DP)))
        addView(bar, LayoutParams(LayoutParams.MATCH_PARENT, dp(BAR_HEIGHT_DP)))
        setExpanded(false)
        render(state)
    }

    fun attach(controller: CameraControlsController) {
        controls = controller
        render(controller.state)
    }

    /** MainActivity forwards controller state here so it can also guard the measurement button. */
    fun render(newState: CameraOpticsState) {
        state = newState
        rendering = true
        val enabled = state.bound && !state.locked
        val zoomEnabled = (state.bound || state.requestedZoomRatio != null) && !state.locked
        if (state.locked && expanded) setExpanded(false)
        renderFocus(enabled)
        renderCamera(enabled)
        renderZoom(zoomEnabled)
        expand.isEnabled = !state.locked
        expand.alpha = if (expand.isEnabled) 1f else 0.45f
        rendering = false
    }

    private fun renderFocus(enabled: Boolean) {
        val manual = state.manualFocusSupported
        val automatic = manual && state.automaticFocus
        val toggleLabel = when {
            !manual -> "AF"
            automatic -> "A"
            else -> "MF"
        }
        auto.textOn = toggleLabel
        auto.textOff = toggleLabel
        auto.text = toggleLabel
        auto.setTextSize(if (toggleLabel == "A") 22f else 15f)
        auto.isChecked = automatic
        auto.isEnabled = enabled && manual
        auto.setTextColor(if (automatic) Color.BLACK else primary)
        auto.background = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(if (automatic) accent else Color.argb(60, 255, 255, 255))
        }
        auto.alpha = if (auto.isEnabled) 1f else 0.45f
        auto.contentDescription = context.getString(if (manual) R.string.camera_optics_auto else R.string.camera_optics_fixed)
        auto.tooltipText = auto.contentDescription
        focus.isEnabled = enabled && manual && !state.automaticFocus
        focus.alpha = if (focus.isEnabled) 1f else 0.4f
        focus.progress = state.focusPosition
        focusLabel.text = when {
            !manual && state.bound -> context.getString(R.string.camera_optics_fixed)
            state.automaticFocus -> "AF"
            else -> "MF ${state.focusPosition}"
        }
    }

    private fun renderCamera(enabled: Boolean) {
        camera.isEnabled = enabled && state.cameras.size > 1 && !state.applying
        camera.alpha = if (camera.isEnabled) 1f else 0.45f
        camera.tooltipText = state.cameras.firstOrNull { it.id == state.selectedCameraId }?.label
            ?: context.getString(R.string.camera_optics_select)
    }

    private fun renderZoom(zoomEnabled: Boolean) {
        zoom.isEnabled = zoomEnabled && state.minimumZoomRatio < state.maximumZoomRatio
        zoom.alpha = if (zoom.isEnabled) 1f else 0.4f
        if (!trackingZoom) zoom.progress = CameraOpticsPolicy.zoomPosition(
            state.requestedZoomRatio ?: state.zoomRatio, state.minimumZoomRatio, state.maximumZoomRatio
        )
        zoomLabel.text = state.requestedZoomRatio?.let {
            "${formatRatio(state.zoomRatio)}\n${formatRatio(it)} 调整中…"
        } ?: formatRatio(state.zoomRatio)

        val ratios = CameraOpticsPolicy.zoomPresets(state.minimumZoomRatio, state.maximumZoomRatio)
        if (ratios != renderedPresets) {
            renderedPresets = ratios
            presets.removeAllViews()
            for (ratio in ratios) {
                presets.addView(presetChip(ratio), LayoutParams(dp(CHIP_WIDTH_DP), dp(BAR_HEIGHT_DP)))
            }
        }
        val requested = state.requestedZoomRatio
        var active: View? = null
        for (index in ratios.indices) {
            val chip = presets.getChildAt(index) as TextView
            val confirmed = abs(ratios[index] - state.zoomRatio) < 0.015f
            val pending = requested != null && abs(ratios[index] - requested) < 0.015f
            chip.isEnabled = zoomEnabled
            chip.alpha = if (zoomEnabled) 1f else 0.45f
            setChipSelected(chip, confirmed)
            chip.setTextColor(
                when {
                    confirmed -> accent
                    pending -> pendingZoom
                    else -> primary
                }
            )
            if (confirmed) active = chip
        }
        active = renderCurrentChip(ratios, zoomEnabled, active)
        activeChip = active
        ensureActiveChipVisible()
    }

    /**
     * A confirmed ratio between two presets has no chip of its own, so the row shows it as a
     * trailing read-only chip. Without it the collapsed bar could not report the zoom that the
     * slider had landed on.
     */
    private fun renderCurrentChip(ratios: List<Float>, zoomEnabled: Boolean, active: View?): View? {
        val isPreset = ratios.any { abs(it - state.zoomRatio) < 0.015f }
        if (ratios.isEmpty() || isPreset) {
            if (currentChip.parent != null) presets.removeView(currentChip)
            return active
        }
        val ratio = formatRatio(state.zoomRatio)
        currentChip.text = if (state.requestedZoomRatio != null) "$ratio…" else ratio
        currentChip.contentDescription = context.getString(R.string.camera_optics_current_zoom, ratio)
        currentChip.alpha = if (zoomEnabled) 1f else 0.45f
        if (presets.indexOfChild(currentChip) != presets.childCount - 1 ||
            currentChip.layoutParams?.height != dp(BAR_HEIGHT_DP)
        ) {
            presets.removeView(currentChip)
            presets.addView(
                currentChip,
                LayoutParams(LayoutParams.WRAP_CONTENT, dp(BAR_HEIGHT_DP)).apply {
                    minimumWidth = dp(CHIP_WIDTH_DP)
                }
            )
        }
        return currentChip
    }

    private fun presetChip(ratio: Float): TextView = label(formatRatio(ratio), 12f).apply {
        minimumWidth = dp(CHIP_WIDTH_DP)
        setPadding(dp(6), 0, dp(6), 0)
        isClickable = true
        isFocusable = true
        contentDescription = "${context.getString(R.string.camera_optics_zoom)} ${formatRatio(ratio)}"
        tooltipText = contentDescription
        setOnClickListener { controls?.setZoomRatio(ratio) }
    }

    private fun setChipSelected(chip: TextView, selected: Boolean) {
        if (chip.isSelected == selected) return
        chip.isSelected = selected
        chip.background = chipBackground(selected)
    }

    private fun chipBackground(selected: Boolean): GradientDrawable = GradientDrawable().apply {
        cornerRadius = dp(6).toFloat()
        setColor(
            if (selected) Color.argb(56, Color.red(accent), Color.green(accent), Color.blue(accent))
            else Color.TRANSPARENT
        )
    }

    private fun setExpanded(open: Boolean) {
        expanded = open
        tuningPanel.visibility = if (open) VISIBLE else GONE
        expand.rotation = if (open) 180f else 0f
        expand.contentDescription = context.getString(
            if (open) R.string.camera_optics_collapse else R.string.camera_optics_expand
        )
        expand.tooltipText = expand.contentDescription
        handler.removeCallbacks(collapse)
        if (open) handler.postDelayed(collapse, AUTO_COLLAPSE_MS)
    }

    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        // Any touch inside the optics keeps the fine-tune panel open; let go for
        // AUTO_COLLAPSE_MS and the preview is handed back to the tube.
        if (expanded) {
            handler.removeCallbacks(collapse)
            handler.postDelayed(collapse, AUTO_COLLAPSE_MS)
        }
        return super.dispatchTouchEvent(event)
    }

    override fun onLayout(changed: Boolean, left: Int, top: Int, right: Int, bottom: Int) {
        super.onLayout(changed, left, top, right, bottom)
        ensureActiveChipVisible()
    }

    override fun onDetachedFromWindow() {
        handler.removeCallbacks(collapse)
        super.onDetachedFromWindow()
    }

    /** Scrolls the preset row just enough to keep the confirmed ratio on screen. */
    private fun ensureActiveChipVisible() {
        val chip = activeChip ?: return
        val viewport = presetScroll.width - presetScroll.paddingLeft - presetScroll.paddingRight
        if (viewport <= 0 || presets.width <= viewport || chip.parent == null) return
        val scrollX = presetScroll.scrollX
        val target = when {
            chip.left < scrollX -> chip.left
            chip.right > scrollX + viewport -> chip.right - viewport
            else -> return
        }
        presetScroll.scrollTo(target, 0)
    }

    private fun showCameraMenu() {
        PopupMenu(context, camera).apply {
            state.cameras.forEachIndexed { index, lens ->
                menu.add(0, index, index, lens.label).apply {
                    isCheckable = true
                    isChecked = lens.id == state.selectedCameraId
                }
            }
            setOnMenuItemClickListener { item ->
                state.cameras.getOrNull(item.itemId)?.let { controls?.selectCamera(it.id) }
                true
            }
            show()
        }
    }

    private fun sliderListener(change: (Int) -> Unit) = object : SeekBar.OnSeekBarChangeListener {
        override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
            if (fromUser && !rendering) change(progress)
        }
        override fun onStartTrackingTouch(seekBar: SeekBar) = Unit
        override fun onStopTrackingTouch(seekBar: SeekBar) = Unit
    }

    private fun label(value: String, size: Float) = TextView(context).apply {
        text = value
        textSize = size
        setTextColor(muted)
        gravity = Gravity.CENTER
        maxLines = 1
        includeFontPadding = false
    }

    private fun formatRatio(ratio: Float): String =
        if (abs(ratio - ratio.toInt()) < 0.005f) "${ratio.toInt()}x"
        else String.format(Locale.ROOT, "%.1fx", ratio)

    private fun dp(value: Int) = (value * resources.displayMetrics.density + 0.5f).toInt()

    private class FocusRuler(context: Context) : AppCompatSeekBar(context) {
        private val density = resources.displayMetrics.density
        private val ticks = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            strokeWidth = 1.5f * density
        }

        init {
            splitTrack = false
            setPadding((6 * density).toInt(), 0, (6 * density).toInt(), 0)
            progressDrawable = ColorDrawable(Color.TRANSPARENT)
            thumb = GradientDrawable().apply {
                setColor(Color.rgb(244, 99, 106))
                setSize((3 * density).toInt(), (28 * density).toInt())
            }
        }

        override fun onDraw(canvas: Canvas) {
            val left = paddingLeft.toFloat()
            val usable = width - paddingLeft - paddingRight
            val middle = height / 2f
            for (index in 0..30) {
                val x = left + usable * index / 30f
                val halfHeight = if (index % 5 == 0) 10f * density else 6f * density
                ticks.alpha = if (index % 5 == 0) 230 else 150
                canvas.drawLine(x, middle - halfHeight, x, middle + halfHeight, ticks)
            }
            super.onDraw(canvas)
        }
    }

    companion object {
        /** Height of the always-visible optics bar; the row is deliberately flat. */
        private const val BAR_HEIGHT_DP = 40

        /** Height of one fine-tune row; the panel stacks the focus ruler above the zoom slider. */
        private const val TUNING_ROW_HEIGHT_DP = 40

        /** Height added while the tuning panel is open. */
        private const val TUNING_HEIGHT_DP = TUNING_ROW_HEIGHT_DP * 2

        /** Touch width of a zoom preset chip; the flat row supplies 40dp of height. */
        private const val CHIP_WIDTH_DP = 44

        /** Side of the square camera switch, focus toggle and expand toggle. */
        private const val ICON_SIZE_DP = 40

        private const val NEAR_FAR_WIDTH_DP = 20
        private const val FOCUS_VALUE_WIDTH_DP = 52
        private const val ZOOM_VALUE_WIDTH_DP = 88

        /** Idle time after which the tuning panel closes itself again. */
        private const val AUTO_COLLAPSE_MS = 5_000L
    }
}
