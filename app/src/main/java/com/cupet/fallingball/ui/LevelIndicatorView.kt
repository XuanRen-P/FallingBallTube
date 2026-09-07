package com.cupet.fallingball.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.cupet.fallingball.R
import com.cupet.fallingball.vision.CameraLevelReading
import java.util.Locale
import kotlin.math.abs

/**
 * Minimal side scale for camera elevation.
 *
 * Static 15-degree ticks stay translucent white. The live tick is green inside the
 * accepted +/-10-degree range and white otherwise. Tick length grows continuously from
 * 0 to 90 degrees, matching the visual language requested for the camera screen.
 */
class LevelIndicatorView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val density = resources.displayMetrics.density
    private val scaledDensity = density * resources.configuration.fontScale
    private val tickPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x88FFFFFF.toInt()
        strokeWidth = 1f * density
        strokeCap = Paint.Cap.ROUND
    }
    private val livePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 3f * density
        strokeCap = Paint.Cap.ROUND
    }
    private val warningPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        textSize = 13f * scaledDensity
        typeface = android.graphics.Typeface.DEFAULT_BOLD
    }
    private val warningBackgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xAA000000.toInt()
        style = Paint.Style.FILL
    }
    private val warningBounds = RectF()

    private var reading: CameraLevelReading? = null

    fun setReading(value: CameraLevelReading?) {
        reading = value
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width <= 0 || height <= 0) return

        val right = width - 18f * density
        val centreY = height * 0.5f
        val halfRange = minOf(height * 0.30f, 150f * density)
        val minLength = 12f * density
        val maxLength = 44f * density

        for (degrees in -90..90 step 15) {
            val fraction = abs(degrees) / 90f
            val length = minLength + (maxLength - minLength) * fraction
            val y = centreY - degrees / 90f * halfRange
            canvas.drawLine(right - length, y, right, y, tickPaint)
        }

        val current = reading ?: return
        val y = centreY -
            (current.angleDegrees.coerceIn(-90f, 90f) / 90f) * halfRange
        val length = minLength + (maxLength - minLength) * current.lineLengthFraction
        livePaint.color = if (current.isLevel) LEVEL_GREEN else 0xFFFFFFFF.toInt()
        canvas.drawLine(right - length, y, right, y, livePaint)

        if (!current.isLevel) drawWarning(canvas, current.angleDegrees, right, y, maxLength)
    }

    private fun drawWarning(
        canvas: Canvas,
        angleDegrees: Float,
        scaleRight: Float,
        markerY: Float,
        maxTickLength: Float
    ) {
        val angle = String.format(Locale.US, "%+.0f°", angleDegrees)
        val warning = context.getString(R.string.level_warning)
        val warningWidth = warningPaint.measureText(warning)
        val angleWidth = warningPaint.measureText(angle)
        val textWidth = maxOf(warningWidth, angleWidth)
        val lineHeight = warningPaint.fontSpacing
        val paddingX = 8f * density
        val paddingY = 6f * density
        val textX = (scaleRight - maxTickLength - 12f * density - textWidth)
            .coerceAtLeast(12f * density)
        val firstBaseline = markerY.coerceIn(
            30f * density,
            height - lineHeight - 18f * density
        )
        warningBounds.set(
            textX - paddingX,
            firstBaseline + warningPaint.ascent() - paddingY,
            textX + textWidth + paddingX,
            firstBaseline + lineHeight + warningPaint.descent() + paddingY
        )
        canvas.drawRoundRect(warningBounds, 8f * density, 8f * density, warningBackgroundPaint)
        canvas.drawText(warning, textX, firstBaseline, warningPaint)
        canvas.drawText(angle, textX + (textWidth - angleWidth), firstBaseline + lineHeight, warningPaint)
    }

    private companion object {
        const val LEVEL_GREEN = 0xFF00E5A8.toInt()
    }
}
