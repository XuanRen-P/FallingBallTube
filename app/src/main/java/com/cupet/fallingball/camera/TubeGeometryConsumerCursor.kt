package com.cupet.fallingball.camera

import com.cupet.fallingball.vision.TubeAxis

internal data class TubeGeometryDelta(
    val geometryChanged: Boolean,
    val coarseShiftPx: Float,
    val geometryScale: Float
)

/** Tracks how the tube coordinate frame changes between analysis-frame consumers. */
internal class TubeGeometryConsumerCursor {
    private var previousAxis: TubeAxis? = null

    fun reset() {
        previousAxis = null
    }

    fun advance(axis: TubeAxis): TubeGeometryDelta {
        val oldAxis = previousAxis
        val geometryChanged = oldAxis == null || oldAxis != axis
        val coarseShift = if (geometryChanged) {
            oldAxis?.let { axis.axialOf(it.ox, it.oy) } ?: 0f
        } else {
            0f
        }
        val geometryScale = if (geometryChanged && oldAxis != null) {
            (axis.ux * oldAxis.ux + axis.uy * oldAxis.uy)
                .takeIf { it.isFinite() && it > 0f } ?: 1f
        } else {
            1f
        }
        previousAxis = axis
        return TubeGeometryDelta(
            geometryChanged = geometryChanged,
            coarseShiftPx = coarseShift,
            geometryScale = geometryScale
        )
    }
}
