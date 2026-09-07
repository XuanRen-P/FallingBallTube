package com.cupet.fallingball.ui

import android.content.res.Resources
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.TypedValue
import com.cupet.fallingball.analysis.MeasurementRecord
import com.cupet.fallingball.analysis.RecordedBallSample
import kotlin.math.max

/** Shared displacement chart renderer used by the detail UI and AI chart snapshots. */
class DisplacementChartRenderer(resources: Resources) {
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
            TypedValue.COMPLEX_UNIT_SP, 11f, resources.displayMetrics
        )
    }
    private val yAxisValue = Paint(label).apply { textAlign = Paint.Align.RIGHT }
    private val xAxisValue = Paint(label).apply { textAlign = Paint.Align.CENTER }
    private val xAxisEndValue = Paint(label).apply { textAlign = Paint.Align.RIGHT }
    private val excludedPoint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = UiColors.CHART_WARNING
        strokeWidth = 1.8f * density
        style = Paint.Style.STROKE
    }
    private val selectedPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = UiColors.WARNING
        strokeWidth = 1.5f * density
    }
    private val fitQuality = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = UiColors.ON_SURFACE
        textSize = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_SP, 12f, resources.displayMetrics
        )
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    val plotBounds = RectF()

    fun draw(
        canvas: Canvas,
        width: Int,
        height: Int,
        record: MeasurementRecord?,
        selectedIndex: Int = -1,
        backgroundColor: Int = UiColors.SURFACE_TINT
    ) {
        canvas.drawColor(backgroundColor)
        val samples = record?.samples.orEmpty()
        if (samples.isEmpty()) {
            canvas.drawText("没有可绘制的数据", 20f * density, 36f * density, label)
            return
        }
        val left = 50f * density
        val top = 18f * density
        val right = width - 14f * density
        val bottom = height - 32f * density
        if (right <= left || bottom <= top) return
        plotBounds.set(left, top, right, bottom)
        val positions = samples.mapNotNull { it.positionMm }
        if (positions.isEmpty()) {
            canvas.drawText("没有有效位移数据", 20f * density, 36f * density, label)
            drawMissingMarkers(canvas, samples, 0f, max(1f, samples.last().elapsedMs))
            return
        }

        val tMin = samples.first().elapsedMs
        val tMax = max(tMin + 1f, samples.last().elapsedMs)
        var yMin = positions.minOrNull() ?: 0f
        var yMax = positions.maxOrNull() ?: 1f
        val yPad = max(0.5f, (yMax - yMin) * 0.08f)
        yMin -= yPad
        yMax += yPad
        fun x(sample: RecordedBallSample) = plotBounds.left +
            (sample.elapsedMs - tMin) / (tMax - tMin) * plotBounds.width()
        fun y(position: Float) = plotBounds.bottom -
            (position - yMin) / (yMax - yMin) * plotBounds.height()

        for (index in 0..4) {
            val fraction = index / 4f
            val gy = plotBounds.top + fraction * plotBounds.height()
            val gx = plotBounds.left + fraction * plotBounds.width()
            canvas.drawLine(plotBounds.left, gy, plotBounds.right, gy, grid)
            canvas.drawLine(gx, plotBounds.top, gx, plotBounds.bottom, grid)

            val yValue = yMax - fraction * (yMax - yMin)
            canvas.drawText(
                format("%.1f", yValue),
                plotBounds.left - 6f * density,
                centeredTextBaseline(gy, yAxisValue),
                yAxisValue
            )

            val elapsedSeconds = (tMax - tMin) * fraction / 1_000f
            val xText = when (index) {
                0 -> "0"
                4 -> format("%.2f s", elapsedSeconds)
                else -> format("%.2f", elapsedSeconds)
            }
            val xPaint = when (index) {
                0 -> label
                4 -> xAxisEndValue
                else -> xAxisValue
            }
            canvas.drawText(xText, gx, height - 8f * density, xPaint)
        }
        canvas.drawRect(plotBounds, white)
        canvas.drawText("位移 mm", plotBounds.left, 14f * density, label)
        for (index in 1 until samples.size) {
            val previous = samples[index - 1]
            val current = samples[index]
            val py = previous.positionMm
            val cy = current.positionMm
            if (py == null || cy == null || previous.excludedAsOutlier || current.excludedAsOutlier) {
                continue
            }
            canvas.drawLine(
                x(previous), y(py), x(current), y(cy),
                if (previous.usedInTerminalFit && current.usedInTerminalFit) green else white
            )
        }
        samples.forEach { sample ->
            sample.positionMm?.let { position ->
                val cx = x(sample)
                val cy = y(position)
                if (sample.excludedAsOutlier) {
                    val radius = 3.6f * density
                    val cross = 2.2f * density
                    canvas.drawCircle(cx, cy, radius, excludedPoint)
                    canvas.drawLine(cx - cross, cy - cross, cx + cross, cy + cross, excludedPoint)
                    canvas.drawLine(cx - cross, cy + cross, cx + cross, cy - cross, excludedPoint)
                } else {
                    canvas.drawCircle(
                        cx, cy, 2.5f * density,
                        if (sample.usedInTerminalFit) greenFill else whiteFill
                    )
                }
            }
        }
        drawFitLine(canvas, samples, tMin, tMax, yMin, yMax)
        drawMissingMarkers(canvas, samples, tMin, tMax)
        record?.summary?.takeIf { it.plateauFound && it.r2.isFinite() }?.let { summary ->
            canvas.drawText(
                format("R² = %.4f", summary.r2),
                plotBounds.left + 8f * density,
                plotBounds.top + 18f * density,
                fitQuality
            )
        }
        samples.getOrNull(selectedIndex)?.let { sample ->
            val px = x(sample)
            canvas.drawLine(px, plotBounds.top, px, plotBounds.bottom, selectedPaint)
            sample.positionMm?.let { canvas.drawCircle(px, y(it), 5f * density, selectedPaint) }
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
        fitted.forEach { sample ->
            val dt = sample.elapsedMs - meanT
            covariance += dt * (requireNotNull(sample.positionMm) - meanY)
            variance += dt * dt
        }
        if (variance <= 0.0) return
        val slope = covariance / variance
        val intercept = meanY - slope * meanT
        fun x(time: Float) = plotBounds.left +
            (time - tMin) / (tMax - tMin) * plotBounds.width()
        fun y(value: Double) = plotBounds.bottom -
            ((value - yMin) / (yMax - yMin)).toFloat() * plotBounds.height()
        canvas.drawLine(
            x(fitted.first().elapsedMs), y(intercept + slope * fitted.first().elapsedMs),
            x(fitted.last().elapsedMs), y(intercept + slope * fitted.last().elapsedMs), green
        )
    }

    private fun drawMissingMarkers(
        canvas: Canvas,
        samples: List<RecordedBallSample>,
        tMin: Float,
        tMax: Float
    ) {
        if (plotBounds.width() <= 0f) return
        samples.filter { it.positionMm == null }.forEach { sample ->
            val x = plotBounds.left +
                (sample.elapsedMs - tMin) / max(1f, tMax - tMin) * plotBounds.width()
            val y = plotBounds.bottom - 4f * density
            canvas.drawLine(x - 3f * density, y - 3f * density, x + 3f * density, y + 3f * density, white)
            canvas.drawLine(x - 3f * density, y + 3f * density, x + 3f * density, y - 3f * density, white)
        }
    }

    private fun format(pattern: String, value: Float): String =
        String.format(java.util.Locale.ROOT, pattern, value)

    private fun centeredTextBaseline(centerY: Float, paint: Paint): Float =
        centerY - (paint.fontMetrics.ascent + paint.fontMetrics.descent) / 2f
}
