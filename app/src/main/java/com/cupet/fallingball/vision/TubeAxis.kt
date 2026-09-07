package com.cupet.fallingball.vision

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * The tube's own coordinate frame, derived from the detected OBB.
 *
 * Scale lines run perpendicular to the tube, so the 1D profile has to be built by
 * marching along the tube's long axis and averaging across its short axis. An
 * axis-aligned strip (as in the PC reference, which assumes a vertical tube) smears the
 * lines together as soon as the tube tilts.
 *
 * @param ox,oy   axis origin: the centre of the "top" end cap, in display coordinates
 * @param ux,uy   unit vector along the tube, always pointing in +y, i.e. the direction a
 *                ball falls
 * @param vx,vy   unit vector across the tube (left normal of u)
 * @param lengthPx  long-side length
 * @param widthPx   short-side length
 * @param tiltRad   signed angle away from vertical; 0 means perfectly upright
 */
data class TubeAxis(
    val ox: Float,
    val oy: Float,
    val ux: Float,
    val uy: Float,
    val vx: Float,
    val vy: Float,
    val lengthPx: Float,
    val widthPx: Float,
    val tiltRad: Float
) {

    /** Display-space point at axial distance [s] and lateral offset [t]. */
    fun pointX(s: Float, t: Float): Float = ox + ux * s + vx * t

    fun pointY(s: Float, t: Float): Float = oy + uy * s + vy * t

    /** Projects a display-space point onto the axis, returning axial distance from origin. */
    fun axialOf(x: Float, y: Float): Float = (x - ox) * ux + (y - oy) * uy

    companion object {
        /**
         * Beyond this tilt the "average across the short axis" premise breaks down: the
         * band starts spanning several scale lines at once and the profile washes out.
         */
        const val MAX_TILT_RAD = 0.436f // 25 degrees

        /**
         * Derives the axis from four OBB corners, or null if the shape is unusable.
         *
         * Deliberately ignores corner *ordering*. `ObbDetector` normalises its angle into
         * `[0, pi)`, so which physical corner lands at index 0 flips as the tube rotates
         * through vertical, and `TubeTracker.smoothCorners` then re-matches corners by
         * nearest neighbour, which can permute them again. Anything that indexes corners
         * positionally would silently swap the tube's ends. So: take the longer of the two
         * adjacent edges as the axis, and fix the sign by requiring +y.
         */
        fun fromCorners(corners: List<Pair<Float, Float>>?): TubeAxis? {
            if (corners == null || corners.size != 4) return null

            val e01 = dist(corners[0], corners[1])
            val e12 = dist(corners[1], corners[2])

            // Longer adjacent edge is the tube axis; the other one is its width.
            val a: Pair<Float, Float>
            val b: Pair<Float, Float>
            val widthPx: Float
            if (e01 >= e12) {
                a = corners[0]; b = corners[1]; widthPx = e12
            } else {
                a = corners[1]; b = corners[2]; widthPx = e01
            }

            val len = dist(a, b)
            if (len < MIN_LENGTH_PX || widthPx < MIN_WIDTH_PX) return null

            var ux = (b.first - a.first) / len
            var uy = (b.second - a.second) / len
            // Orient down-frame so axial distance increases as the ball falls.
            if (uy < 0f) {
                ux = -ux
                uy = -uy
            }

            val tilt = atan2(ux, uy)
            if (abs(tilt) > MAX_TILT_RAD) return null

            // Origin from the centroid rather than a corner: immune to any permutation.
            var cx = 0f
            var cy = 0f
            for (c in corners) {
                cx += c.first
                cy += c.second
            }
            cx /= 4f
            cy /= 4f

            return TubeAxis(
                ox = cx - ux * len * 0.5f,
                oy = cy - uy * len * 0.5f,
                ux = ux,
                uy = uy,
                vx = -uy,
                vy = ux,
                lengthPx = len,
                widthPx = widthPx,
                tiltRad = tilt
            )
        }

        private const val MIN_LENGTH_PX = 32f
        private const val MIN_WIDTH_PX = 4f

        private fun dist(p: Pair<Float, Float>, q: Pair<Float, Float>): Float =
            hypot(q.first - p.first, q.second - p.second)
    }
}
