package com.cupet.fallingball.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.cupet.fallingball.R
import java.util.Locale
import kotlin.math.min

/**
 * Compact schematic of a falling-ball experiment for history rows: the tube cross-section
 * width follows the tube inner diameter (10..50 mm) and the ball radius follows the sphere
 * diameter (1..5 mm). Values outside the documented ranges clamp to the nearest bound, so
 * extreme records stay legible instead of disappearing.
 *
 * Colors are keyword-matched from the record metadata (sphere material / liquid name) and
 * fall back to theme colors when the text is unknown or empty.
 */
class HistoryConditionView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var sphereDiameterMm: Double? = null
    private var tubeDiameterMm: Double? = null
    private var sphereMaterial: String? = null
    private var liquidName: String? = null
    private val density = resources.displayMetrics.density

    private val tubeFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        alpha = 34
    }
    private val tubePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.2f * density
        color = UiColors.OUTLINE
    }
    private val ballPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = UiColors.SECONDARY
    }
    private val ballOutlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1f * density
        color = UiColors.SECONDARY
    }
    private val fallHintPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1f * density
        color = UiColors.ON_SURFACE_DISABLED
    }

    fun setConditions(
        sphereDiameterMm: Double?,
        tubeDiameterMm: Double?,
        sphereMaterial: String? = null,
        liquidName: String? = null
    ) {
        this.sphereDiameterMm = sphereDiameterMm
        this.tubeDiameterMm = tubeDiameterMm
        this.sphereMaterial = sphereMaterial
        this.liquidName = liquidName

        val sphere = sphereDiameterMm
        val tube = tubeDiameterMm
        ballPaint.color = HistoryConditionPalette.materialColor(sphereMaterial)
        tubePaint.color = HistoryConditionPalette.liquidColor(liquidName)
        tubeFillPaint.color = HistoryConditionPalette.liquidColor(liquidName)

        contentDescription = if (
            sphere == null || tube == null || sphere <= 0.0 || tube <= 0.0
        ) {
            null
        } else {
            val parts = mutableListOf(
                context.getString(R.string.history_conditions_description, sphere, tube)
            )
            sphereMaterial?.takeIf { it.isNotBlank() }?.let {
                parts += context.getString(R.string.history_conditions_material, it)
            }
            liquidName?.takeIf { it.isNotBlank() }?.let {
                parts += context.getString(R.string.history_conditions_liquid, it)
            }
            parts.joinToString("，")
        }
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val sphere = sphereDiameterMm ?: return
        val tube = tubeDiameterMm ?: return
        if (sphere <= 0.0 || tube <= 0.0) return

        val centerX = width / 2f
        val tubeHeight = min(dp(40f), height - dp(2f))
        if (tubeHeight <= 0f) return
        val top = (height - tubeHeight) / 2f
        val bottom = top + tubeHeight
        val tubeWidth = dp(historyTubeWidthDp(tube))
        val rect = RectF(centerX - tubeWidth / 2f, top, centerX + tubeWidth / 2f, bottom)
        val corner = dp(3f)
        canvas.drawRoundRect(rect, corner, corner, tubeFillPaint)
        canvas.drawRoundRect(rect, corner, corner, tubePaint)

        val radius = dp(historyBallRadiusDp(sphere))
        val ballCenterY = top + tubeHeight * 0.32f
        canvas.drawCircle(centerX, ballCenterY, radius, ballPaint)
        HistoryConditionPalette.materialOutlineColor(sphereMaterial)?.let { outline ->
            ballOutlinePaint.color = outline
            canvas.drawCircle(centerX, ballCenterY, radius, ballOutlinePaint)
        }

        // Two short dashes above the ball: a cheap "falling" cue without extra labels.
        val hintTop = top + tubeHeight * 0.06f
        val hintBottom = ballCenterY - radius - dp(1.5f)
        if (hintBottom > hintTop) {
            canvas.drawLine(centerX, hintTop, centerX, hintBottom, fallHintPaint)
        }
    }

    private fun dp(value: Float) = value * density
}

/** Keyword-matched, muted colors for the history condition badge. */
internal object HistoryConditionPalette {
    val MATERIAL_STEEL: Int = 0xFF4E5E66.toInt()
    val MATERIAL_CERAMIC: Int = 0xFFB9BEC2.toInt()
    val MATERIAL_CERAMIC_OUTLINE: Int = 0xFF000000.toInt()
    val MATERIAL_COPPER: Int = 0xFFA9704D.toInt()
    val MATERIAL_DEFAULT: Int = UiColors.SECONDARY

    val LIQUID_CASTOR: Int = 0xFFB08A2E.toInt()
    val LIQUID_GLYCEROL: Int = 0xFF6F9A92.toInt()
    val LIQUID_SILICONE: Int = 0xFF8C7FA8.toInt()
    val LIQUID_WATER: Int = 0xFF5F8FB8.toInt()
    val LIQUID_DEFAULT: Int = UiColors.OUTLINE

    fun materialColor(name: String?): Int {
        val text = name.orEmpty().lowercase(Locale.ROOT)
        return when {
            text.contains("陶瓷") || text.contains("ceramic") -> MATERIAL_CERAMIC
            text.contains("铜") || text.contains("copper") -> MATERIAL_COPPER
            text.contains("钢") || text.contains("steel") ||
                text.contains("铁") || text.contains("iron") -> MATERIAL_STEEL
            else -> MATERIAL_DEFAULT
        }
    }

    /** Ceramic balls are painted gray with a black outline; other materials have no outline. */
    fun materialOutlineColor(name: String?): Int? {
        val text = name.orEmpty().lowercase(Locale.ROOT)
        return if (text.contains("陶瓷") || text.contains("ceramic")) {
            MATERIAL_CERAMIC_OUTLINE
        } else {
            null
        }
    }

    fun liquidColor(name: String?): Int {
        val text = name.orEmpty().lowercase(Locale.ROOT)
        return when {
            text.contains("蓖麻") || text.contains("castor") -> LIQUID_CASTOR
            text.contains("甘油") || text.contains("丙三醇") || text.contains("glycer") -> LIQUID_GLYCEROL
            text.contains("硅油") || text.contains("silicone") || text.contains("pdms") -> LIQUID_SILICONE
            text.contains("水") || text.contains("water") -> LIQUID_WATER
            else -> LIQUID_DEFAULT
        }
    }
}

/** Pure scaling used by [HistoryConditionView]; testable without Android graphics. */
internal object HistoryConditionScale {
    const val MIN_SPHERE_DIAMETER_MM = 1.0
    const val MAX_SPHERE_DIAMETER_MM = 5.0
    const val MIN_TUBE_INNER_DIAMETER_MM = 10.0
    const val MAX_TUBE_INNER_DIAMETER_MM = 50.0

    fun fraction(value: Double?, min: Double, max: Double): Float {
        if (value == null || value <= 0.0 || max <= min) return 0f
        return ((value - min) / (max - min)).coerceIn(0.0, 1.0).toFloat()
    }

    fun sphereFraction(sphereDiameterMm: Double?): Float =
        fraction(sphereDiameterMm, MIN_SPHERE_DIAMETER_MM, MAX_SPHERE_DIAMETER_MM)

    fun tubeFraction(tubeInnerDiameterMm: Double?): Float =
        fraction(tubeInnerDiameterMm, MIN_TUBE_INNER_DIAMETER_MM, MAX_TUBE_INNER_DIAMETER_MM)
}

private fun lerp(start: Float, stop: Float, fraction: Float) = start + (stop - start) * fraction

/** Drawn tube width in dp, clamped to the 10..50 mm range. */
internal fun historyTubeWidthDp(tubeInnerDiameterMm: Double?): Float =
    lerp(12f, 32f, HistoryConditionScale.tubeFraction(tubeInnerDiameterMm))

/** Drawn ball radius in dp, clamped to the 1..5 mm range. */
internal fun historyBallRadiusDp(sphereDiameterMm: Double?): Float =
    lerp(1.5f, 6f, HistoryConditionScale.sphereFraction(sphereDiameterMm))
