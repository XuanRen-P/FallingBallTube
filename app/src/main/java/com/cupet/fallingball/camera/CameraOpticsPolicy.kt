package com.cupet.fallingball.camera

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.roundToInt

/** Lens distances use diopters; the UI deliberately runs from near (0) to infinity (100). */
object CameraOpticsPolicy {
    fun supportsManualFocus(
        minimumFocusDistance: Float?,
        supportsAfOff: Boolean,
        supportsDistanceRequest: Boolean
    ): Boolean = minimumFocusDistance != null && minimumFocusDistance.isFinite() &&
        minimumFocusDistance > 0f && supportsAfOff && supportsDistanceRequest

    fun focusDistance(position: Int, minimumFocusDistance: Float): Float {
        require(minimumFocusDistance.isFinite() && minimumFocusDistance > 0f)
        return minimumFocusDistance * (1f - position.coerceIn(0, 100) / 100f)
    }

    fun focusPosition(distance: Float?, minimumFocusDistance: Float, remembered: Int?): Int {
        if (distance != null && distance.isFinite() && distance >= 0f &&
            minimumFocusDistance.isFinite() && minimumFocusDistance > 0f
        ) {
            return ((1f - distance / minimumFocusDistance) * 100f).roundToInt().coerceIn(0, 100)
        }
        return (remembered ?: 50).coerceIn(0, 100)
    }

    fun zoomPresets(minimum: Float, maximum: Float): List<Float> {
        if (!validZoomRange(minimum, maximum)) return emptyList()
        return listOf(0.6f, 1f, 2f, 3.2f, 5f, 10f)
            .filter { it >= minimum && it <= maximum }
    }

    /** A logarithmic slider gives each focal-length doubling the same amount of travel. */
    fun zoomRatio(position: Int, minimum: Float, maximum: Float): Float {
        require(validZoomRange(minimum, maximum))
        return exp(ln(minimum.toDouble()) + position.coerceIn(0, 1000) / 1000.0 *
            ln((maximum / minimum).toDouble())).toFloat().coerceIn(minimum, maximum)
    }

    fun zoomPosition(ratio: Float, minimum: Float, maximum: Float): Int {
        require(validZoomRange(minimum, maximum))
        if (minimum == maximum) return 0
        return (ln((ratio.coerceIn(minimum, maximum) / minimum).toDouble()) /
            ln((maximum / minimum).toDouble()) * 1000).roundToInt().coerceIn(0, 1000)
    }

    private fun validZoomRange(minimum: Float, maximum: Float): Boolean =
        minimum.isFinite() && maximum.isFinite() && minimum > 0f && maximum >= minimum
}
