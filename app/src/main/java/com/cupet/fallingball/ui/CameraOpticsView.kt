package com.cupet.fallingball.ui

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.drawable.ColorDrawable
import android.graphics.drawable.GradientDrawable
import android.util.AttributeSet
import android.view.Gravity
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

/** Compact preview controls; all state and camera behavior live in CameraControlsController. */
class CameraOpticsView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : LinearLayout(context, attrs) {
    private val accent = ContextCompat.getColor(context, R.color.camera_accent)
    private val primary = ContextCompat.getColor(context, R.color.camera_text_primary)
    private val muted = Color.argb(155, 255, 255, 255)
    private var controls: CameraControlsController? = null
    private var rendering = false
    private var trackingZoom = false
    private var state = CameraOpticsState()
    private var renderedPresets = emptyList<Float>()
    private val auto = AppCompatToggleButton(context).apply {
        textOn = "A"
        textOff = "A"
        text = "A"
        textSize = 24f
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
    private val focusLabel = label("MF 50", 11f)
    private val focus = FocusRuler(context).apply {
        max = 100
        progress = 50
        contentDescription = context.getString(R.string.camera_optics_manual)
        setOnSeekBarChangeListener(sliderListener { controls?.setFocusPosition(it) })
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
    private val presets = LinearLayout(context).apply {
        orientation = HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
    }
    private val zoomLabel = label("1x", 11f).apply { maxLines = 2 }
    private val zoom = AppCompatSeekBar(context).apply {
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

    init {
        orientation = VERTICAL
        val focusRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        focusRow.addView(auto, LayoutParams(dp(44), dp(44)).apply { marginEnd = dp(8) })
        val ruler = LinearLayout(context).apply { orientation = VERTICAL }
        val rulerLabels = LinearLayout(context).apply { orientation = HORIZONTAL }
        rulerLabels.addView(label(context.getString(R.string.camera_optics_near), 10f), LayoutParams(dp(24), dp(16)))
        rulerLabels.addView(focusLabel, LayoutParams(0, dp(16), 1f))
        rulerLabels.addView(label(context.getString(R.string.camera_optics_far), 10f), LayoutParams(dp(24), dp(16)))
        ruler.addView(rulerLabels, LayoutParams(LayoutParams.MATCH_PARENT, dp(16)))
        ruler.addView(focus, LayoutParams(LayoutParams.MATCH_PARENT, dp(36)))
        focusRow.addView(ruler, LayoutParams(0, dp(52), 1f))
        addView(focusRow, LayoutParams(LayoutParams.MATCH_PARENT, dp(56)))

        val scroll = HorizontalScrollView(context).apply {
            isHorizontalScrollBarEnabled = true
            isFillViewport = true
            addView(presets, ViewGroup.LayoutParams(LayoutParams.WRAP_CONTENT, dp(48)))
        }
        // Give the five common presets the full width. Camera selection and the confirmed
        // value share the slider row so 3.2x/5x stay visible without horizontal scrolling.
        addView(scroll, LayoutParams(LayoutParams.MATCH_PARENT, dp(48)))
        val sliderRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        sliderRow.addView(camera, LayoutParams(dp(48), dp(48)))
        sliderRow.addView(zoom, LayoutParams(0, dp(48), 1f))
        sliderRow.addView(zoomLabel, LayoutParams(dp(96), dp(48)))
        addView(sliderRow, LayoutParams(LayoutParams.MATCH_PARENT, dp(48)))
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
        auto.isChecked = state.automaticFocus
        auto.isEnabled = enabled && state.manualFocusSupported
        auto.setTextColor(if (state.automaticFocus) Color.BLACK else primary)
        auto.background = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(if (newState.automaticFocus) accent else Color.argb(60, 255, 255, 255))
        }
        auto.alpha = if (auto.isEnabled) 1f else 0.45f
        auto.contentDescription = context.getString(if (state.manualFocusSupported) R.string.camera_optics_auto else R.string.camera_optics_fixed)
        auto.tooltipText = auto.contentDescription
        focus.isEnabled = enabled && state.manualFocusSupported && !state.automaticFocus
        focus.alpha = if (focus.isEnabled) 1f else 0.4f
        focus.progress = state.focusPosition
        focusLabel.text = when {
            !state.manualFocusSupported && state.bound -> context.getString(R.string.camera_optics_fixed)
            state.automaticFocus -> "AF"
            else -> "MF ${state.focusPosition}"
        }
        camera.isEnabled = enabled && state.cameras.size > 1 && !state.applying
        camera.alpha = if (camera.isEnabled) 1f else 0.45f
        camera.tooltipText = state.cameras.firstOrNull { it.id == state.selectedCameraId }?.label
            ?: context.getString(R.string.camera_optics_select)
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
                presets.addView(label(formatRatio(ratio), 12f).apply {
                    minWidth = dp(48)
                    minimumWidth = dp(48)
                    setPadding(dp(6), 0, dp(6), 0)
                    isClickable = true
                    isFocusable = true
                    contentDescription = "${context.getString(R.string.camera_optics_zoom)} ${formatRatio(ratio)}"
                    tooltipText = contentDescription
                    setOnClickListener { controls?.setZoomRatio(ratio) }
                }, LayoutParams(dp(48), dp(48)))
            }
        }
        for (index in ratios.indices) {
            (presets.getChildAt(index) as TextView).apply {
                isEnabled = zoomEnabled
                alpha = if (zoomEnabled) 1f else 0.45f
                setTextColor(if (abs(ratios[index] - state.zoomRatio) < 0.015f) accent else primary)
            }
        }
        rendering = false
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
}
