package com.cupet.fallingball.vision

import com.cupet.fallingball.vision.AnalysisFrame
import java.nio.ByteBuffer
import kotlin.math.floor

/**
 * Collapses the tube into a 1D luminance profile along its axis.
 *
 * Reads the Y plane directly at native sensor resolution. Two reasons this cannot reuse
 * the square model tensor:
 *
 *  - Scale lines are only a few pixels wide. Downsampling and letterboxing the camera
 *    frame for YOLO makes the tube occupy fewer pixels and smears adjacent lines together.
 *  - `YUV_420_888`'s Y plane *is* BT.601 luma, the same quantity `cv2.cvtColor(BGR2GRAY)`
 *    produces. No colour conversion is needed, which makes this path far cheaper than the
 *    tensor path despite running at full resolution.
 *
 * Sampling walks the tube's own axis rather than image rows, so a tilted tube still
 * produces sharp line features. Roughly 13k bilinear taps per frame -- about 3% of what
 * the tensor path costs.
 */
class AxialSampler {

    companion object {
        /** Trim off each end of the tube, matching the reference implementation. */
        const val AXIAL_TRIM = 0.05f

        /** Width of the averaging band, as a fraction of tube width. */
        const val BAND_FRAC = 0.25f

        /** One lateral tap roughly every 2 px, clamped to keep the inner loop bounded. */
        private const val LATERAL_STEP_PX = 2f
        private const val MAX_HALF_TAPS = 5

        /**
         * Above this fraction of samples landing outside the visible crop the profile is
         * built from pixels the user cannot see, so the frame is rejected outright rather
         * than silently clamped.
         */
        private const val MAX_OUT_OF_CROP = 0.02f
    }

    /** Axial offset (in px from the tube's top end) that `out[0]` corresponds to. */
    var startOffsetPx: Float = 0f
        private set

    /**
     * Fills `out` with the mean luminance across the tube at each axial step and returns
     * the sample count, or 0 when the frame is unusable. `out` must hold at least
     * `axis.lengthPx` entries.
     */
    fun sample(image: AnalysisFrame, xf: BufferTransform, axis: TubeAxis, out: FloatArray): Int {
        val s0 = axis.lengthPx * AXIAL_TRIM
        val n = floor(axis.lengthPx * (1f - 2f * AXIAL_TRIM)).toInt()
            .coerceAtMost(out.size)
        if (n < TickDetector.MIN_SIGNAL_LEN) return 0
        startOffsetPx = s0

        val plane = image.planes[0]
        val buf = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val capacity = buf.capacity()
        val bufW = image.width
        val bufH = image.height

        val halfBand = axis.widthPx * BAND_FRAC * 0.5f
        val taps = (halfBand / LATERAL_STEP_PX).toInt().coerceIn(1, MAX_HALF_TAPS)
        val tapStep = halfBand / taps
        val tapCount = 2 * taps + 1

        var outside = 0
        val outsideLimit = (n * tapCount * MAX_OUT_OF_CROP).toInt().coerceAtLeast(8)

        for (i in 0 until n) {
            val s = s0 + i
            val px = axis.ox + axis.ux * s
            val py = axis.oy + axis.uy * s
            var acc = 0f
            for (j in -taps..taps) {
                val t = j * tapStep
                val lx = px + axis.vx * t
                val ly = py + axis.vy * t
                val bx = xf.bufX(lx, ly)
                val by = xf.bufY(lx, ly)
                if (!xf.isInsideCrop(bx, by)) {
                    outside++
                    if (outside > outsideLimit) return 0
                }
                acc += bilinear(buf, capacity, rowStride, pixelStride, bufW, bufH, bx, by)
            }
            out[i] = acc / tapCount
        }
        return n
    }

    /**
     * Bilinear tap on the Y plane. Nearest-neighbour is not good enough: on a tilted tube
     * the taps land on sub-pixel positions, and snapping them makes scale lines flicker in
     * and out as the tube rotates by a fraction of a degree, which destroys the temporal
     * tick memory downstream.
     */
    private fun bilinear(
        buf: ByteBuffer, capacity: Int,
        rowStride: Int, pixelStride: Int,
        bufW: Int, bufH: Int,
        x: Float, y: Float
    ): Float {
        val x0 = floor(x).toInt()
        val y0 = floor(y).toInt()
        val fx = x - x0
        val fy = y - y0

        val v00 = at(buf, capacity, rowStride, pixelStride, bufW, bufH, x0, y0)
        val v10 = at(buf, capacity, rowStride, pixelStride, bufW, bufH, x0 + 1, y0)
        val v01 = at(buf, capacity, rowStride, pixelStride, bufW, bufH, x0, y0 + 1)
        val v11 = at(buf, capacity, rowStride, pixelStride, bufW, bufH, x0 + 1, y0 + 1)

        val top = v00 + (v10 - v00) * fx
        val bot = v01 + (v11 - v01) * fx
        return top + (bot - top) * fy
    }

    private fun at(
        buf: ByteBuffer, capacity: Int,
        rowStride: Int, pixelStride: Int,
        bufW: Int, bufH: Int,
        x: Int, y: Int
    ): Float {
        val xx = if (x < 0) 0 else if (x >= bufW) bufW - 1 else x
        val yy = if (y < 0) 0 else if (y >= bufH) bufH - 1 else y
        val idx = yy * rowStride + xx * pixelStride
        if (idx < 0 || idx >= capacity) return 0f
        return (buf.get(idx).toInt() and 0xFF).toFloat()
    }
}
