package com.cupet.fallingball.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.util.TypedValue
import android.view.MotionEvent
import android.view.View
import com.cupet.fallingball.analysis.MeasurementRecord
import com.cupet.fallingball.analysis.RecordedBallSample
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Lightweight, dependency-free displacement/time chart.
 *
 * It deliberately draws only measured positions. Missing ruler frames break the polyline and
 * receive a marker on the lower edge, so the graph cannot imply data that was never observed.
 */
class DisplacementTimeChartView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    var onSelectionChanged: ((RecordedBallSample?) -> Unit)? = null

    private val density = resources.displayMetrics.density
    private val white = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = UiColors.CHART_POINT
        strokeWidth = 1.5f * density
        style = Paint.Style.STROKE
    }
    private val green = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = UiColors.CHART_LINE
        strokeWidth = 2f * density
        style = Paint.Style.STROKE
    }
    private val whiteFill = Paint(white).apply { style = Paint.Style.FILL }
    private val greenFill = Paint(green).apply { style = Paint.Style.FILL }
    private val grid = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = UiColors.CHART_GRID
        strokeWidth = density
    }
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = UiColors.ON_SURFACE_VARIANT
        textSize = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_SP,
            11f,
            resources.displayMetrics
        )
    }
    private val selectedPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = UiColors.WARNING
        strokeWidth = 1.5f * density
    }
    private val fitQuality = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = UiColors.ON_SURFACE
        textSize = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_SP,
            12f,
            resources.displayMetrics
        )
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val plot = RectF()
    private val sharedRenderer = DisplacementChartRenderer(resources)
    private val useSharedRenderer = true
    private var record: MeasurementRecord? = null
    private var selectedIndex = -1

    fun setRecord(value: MeasurementRecord) {
        record = value
        selectedIndex = -1
        onSelectionChanged?.invoke(null)
        contentDescription = "位移—时间图，共 ${value.samples.size} 个采样点"
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (useSharedRenderer) {
            sharedRenderer.draw(canvas, width, height, record, selectedIndex)
            plot.set(sharedRenderer.plotBounds)
            return
        }
        canvas.drawColor(UiColors.SURFACE_TINT)
        val samples = record?.samples.orEmpty()
        if (samples.isEmpty()) {
            canvas.drawText("没有可绘制的数据", 20f * density, 36f * density, label)
            return
        }
        val positions = samples.mapNotNull { it.positionMm }
        if (positions.isEmpty()) {
            canvas.drawText("没有有效位移数据", 20f * density, 36f * density, label)
            drawMissingMarkers(canvas, samples, 0f, max(1f, samples.last().elapsedMs))
            return
        }

        val left = 50f * density
        val top = 18f * density
        val right = width - 14f * density
        val bottom = height - 32f * density
        if (right <= left || bottom <= top) return
        plot.set(left, top, right, bottom)

        val tMin = samples.first().elapsedMs
        val tMax = max(tMin + 1f, samples.last().elapsedMs)
        var yMin = positions.minOrNull() ?: 0f
        var yMax = positions.maxOrNull() ?: 1f
        val yPad = max(0.5f, (yMax - yMin) * 0.08f)
        yMin -= yPad
        yMax += yPad

        fun x(sample: RecordedBallSample) =
            plot.left + (sample.elapsedMs - tMin) / (tMax - tMin) * plot.width()
        fun y(position: Float) =
            plot.bottom - (position - yMin) / (yMax - yMin) * plot.height()

        for (i in 0..4) {
            val fraction = i / 4f
            val gy = plot.top + fraction * plot.height()
            val gx = plot.left + fraction * plot.width()
            canvas.drawLine(plot.left, gy, plot.right, gy, grid)
            canvas.drawLine(gx, plot.top, gx, plot.bottom, grid)
        }
        canvas.drawRect(plot, white)
        canvas.drawText(
            String.format(java.util.Locale.ROOT, "%.1f", yMax),
            4f * density,
            plot.top + 4f * density,
            label
        )
        canvas.drawText(
            String.format(java.util.Locale.ROOT, "%.1f", yMin),
            4f * density,
            plot.bottom,
            label
        )
        canvas.drawText("位移 mm", 4f * density, 14f * density, label)
        canvas.drawText("0", plot.left, height - 8f * density, label)
        canvas.drawText(
            String.format(java.util.Locale.ROOT, "%.2f s", (tMax - tMin) / 1000f),
            plot.right - 38f * density,
            height - 8f * density,
            label
        )
        for (index in 1 until samples.size) {
            val previous = samples[index - 1]
            val current = samples[index]
            val py = previous.positionMm
            val cy = current.positionMm
            if (py == null || cy == null) continue
            if (previous.excludedAsOutlier || current.excludedAsOutlier) continue
            val paint = if (
                previous.usedInTerminalFit && current.usedInTerminalFit
            ) green else white
            canvas.drawLine(x(previous), y(py), x(current), y(cy), paint)
        }
        for (sample in samples) {
            val position = sample.positionMm ?: continue
            val fill = if (sample.usedInTerminalFit) greenFill else whiteFill
            canvas.drawCircle(x(sample), y(position), 2.5f * density, fill)
        }
        drawFitLine(canvas, samples, tMin, tMax, yMin, yMax)
        drawMissingMarkers(canvas, samples, tMin, tMax)
        record?.summary?.takeIf { it.plateauFound && it.r2.isFinite() }?.let { summary ->
            canvas.drawText(
                String.format(java.util.Locale.ROOT, "R² = %.4f", summary.r2),
                plot.left + 8f * density,
                plot.top + 18f * density,
                fitQuality
            )
        }

        samples.getOrNull(selectedIndex)?.let { sample ->
            val px = x(sample)
            canvas.drawLine(px, plot.top, px, plot.bottom, selectedPaint)
            sample.positionMm?.let { position ->
                canvas.drawCircle(px, y(position), 5f * density, selectedPaint)
            }
        }
    }

    private fun drawFitLine(
        canvas: Canvas,
        samples: List<RecordedBallSample>,
        tMin: Float,
        tMax: Float,
        yMin: Float,
        yMax: Float
    ) {
        val fitted = samples.filter { it.usedInTerminalFit && it.positionMm != null }
        if (fitted.size < 2) return
        val meanT = fitted.sumOf { it.elapsedMs.toDouble() } / fitted.size
        val meanY = fitted.sumOf { requireNotNull(it.positionMm).toDouble() } / fitted.size
        var covariance = 0.0
        var variance = 0.0
        for (sample in fitted) {
            val dt = sample.elapsedMs - meanT
            covariance += dt * (requireNotNull(sample.positionMm) - meanY)
            variance += dt * dt
        }
        if (variance <= 0.0) return
        val slope = covariance / variance
        val intercept = meanY - slope * meanT
        val start = fitted.first().elapsedMs
        val end = fitted.last().elapsedMs
        fun x(t: Float) = plot.left + (t - tMin) / (tMax - tMin) * plot.width()
        fun y(value: Double) =
            plot.bottom - ((value - yMin) / (yMax - yMin)).toFloat() * plot.height()
        canvas.drawLine(
            x(start),
            y(intercept + slope * start),
            x(end),
            y(intercept + slope * end),
            green
        )
    }

    private fun drawMissingMarkers(
        canvas: Canvas,
        samples: List<RecordedBallSample>,
        tMin: Float,
        tMax: Float
    ) {
        if (plot.width() <= 0f) {
            plot.set(20f * density, 20f * density, width - 20f * density, height - 20f * density)
        }
        for (sample in samples) {
            if (sample.positionMm != null) continue
            val x = plot.left +
                (sample.elapsedMs - tMin) / max(1f, tMax - tMin) * plot.width()
            val y = plot.bottom - 4f * density
            canvas.drawLine(x - 3f * density, y - 3f * density, x + 3f * density, y + 3f * density, white)
            canvas.drawLine(x - 3f * density, y + 3f * density, x + 3f * density, y - 3f * density, white)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.actionMasked == MotionEvent.ACTION_UP) {
            parent?.requestDisallowInterceptTouchEvent(false)
            performClick()
            return true
        }
        if (event.actionMasked == MotionEvent.ACTION_CANCEL) {
            parent?.requestDisallowInterceptTouchEvent(false)
            return true
        }
        if (event.actionMasked != MotionEvent.ACTION_DOWN &&
            event.actionMasked != MotionEvent.ACTION_MOVE
        ) {
            return super.onTouchEvent(event)
        }
        val samples = record?.samples.orEmpty()
        if (samples.isEmpty() || plot.width() <= 0f) return false
        val tMin = samples.first().elapsedMs
        val tMax = max(tMin + 1f, samples.last().elapsedMs)
        val target = tMin +
            ((event.x - plot.left) / plot.width()).coerceIn(0f, 1f) * (tMax - tMin)
        var bestIndex = 0
        var bestDistance = Float.MAX_VALUE
        samples.forEachIndexed { index, sample ->
            val distance = abs(sample.elapsedMs - target)
            if (distance < bestDistance) {
                bestDistance = distance
                bestIndex = index
            }
        }
        selectedIndex = min(samples.lastIndex, max(0, bestIndex))
        onSelectionChanged?.invoke(samples[selectedIndex])
        invalidate()
        parent?.requestDisallowInterceptTouchEvent(true)
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }
}
