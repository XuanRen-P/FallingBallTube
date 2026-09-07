package com.cupet.fallingball.vision

import com.cupet.fallingball.vision.AnalysisFrame
import java.nio.ByteBuffer
import kotlin.math.floor
import kotlin.math.roundToInt

/**
 * Samples narrow axial profiles across the complete YOLO tube width.
 *
 * The yellow ruler band remains unchanged. These profiles are a separate, camera-rate 2D
 * ball path: each profile has a known lateral coordinate, including both tube walls.
 */
class TubeBandSampler(
    val bandCount: Int = DEFAULT_BAND_COUNT,
    private val lateralExtent: Float = DEFAULT_LATERAL_EXTENT
) {
    val lateralFractions = FloatArray(bandCount) { index ->
        -lateralExtent + 2f * lateralExtent * index / (bandCount - 1f)
    }

    var startOffsetPx: Float = 0f
        private set

    init {
        require(bandCount >= 3 && bandCount % 2 == 1)
        require(lateralExtent in 0.5f..1f)
    }

    /** Returns the profile length, or zero when the current geometry is not safely visible. */
    fun sample(
        image: AnalysisFrame,
        transform: BufferTransform,
        axis: TubeAxis,
        output: Array<FloatArray>
    ): Int {
        require(output.size == bandCount)
        val offset = axis.lengthPx * AxialSampler.AXIAL_TRIM
        val count = floor(axis.lengthPx * (1f - 2f * AxialSampler.AXIAL_TRIM)).toInt()
        if (count < TickDetector.MIN_SIGNAL_LEN || output.any { it.size < count }) return 0
        startOffsetPx = offset

        val plane = image.planes[0]
        val buffer = plane.buffer
        val capacity = buffer.capacity()
        val halfWidth = axis.widthPx * 0.5f
        val lateralProbePx = (axis.widthPx / bandCount).coerceIn(1f, 3f)
        var outside = 0
        val sampleTotal = count * bandCount * LATERAL_PROBES
        val outsideLimit = (sampleTotal * MAX_OUT_OF_CROP).roundToInt().coerceAtLeast(12)

        for (axialIndex in 0 until count) {
            val axial = offset + axialIndex
            val centreX = axis.ox + axis.ux * axial
            val centreY = axis.oy + axis.uy * axial
            for (bandIndex in 0 until bandCount) {
                val lateral = lateralFractions[bandIndex] * halfWidth
                var darkest = 255f
                for (probe in -1..1) {
                    val probeLateral = lateral + probe * lateralProbePx
                    val displayX = centreX + axis.vx * probeLateral
                    val displayY = centreY + axis.vy * probeLateral
                    val bufferX = transform.bufX(displayX, displayY)
                    val bufferY = transform.bufY(displayX, displayY)
                    if (!transform.isInsideCrop(bufferX, bufferY)) {
                        outside++
                        if (outside > outsideLimit) return 0
                    }
                    darkest = minOf(
                        darkest,
                        nearest(
                            buffer,
                            capacity,
                            plane.rowStride,
                            plane.pixelStride,
                            image.width,
                            image.height,
                            bufferX,
                            bufferY
                        )
                    )
                }
                output[bandIndex][axialIndex] = darkest
            }
        }
        return count
    }

    private fun nearest(
        buffer: ByteBuffer,
        capacity: Int,
        rowStride: Int,
        pixelStride: Int,
        width: Int,
        height: Int,
        x: Float,
        y: Float
    ): Float {
        val xx = floor(x + 0.5f).toInt().coerceIn(0, width - 1)
        val yy = floor(y + 0.5f).toInt().coerceIn(0, height - 1)
        val index = yy * rowStride + xx * pixelStride
        return if (index in 0 until capacity) {
            (buffer.get(index).toInt() and 0xff).toFloat()
        } else {
            0f
        }
    }

    companion object {
        const val DEFAULT_BAND_COUNT = 17
        const val DEFAULT_LATERAL_EXTENT = 0.95f
        private const val LATERAL_PROBES = 3
        private const val MAX_OUT_OF_CROP = 0.02f
    }
}
