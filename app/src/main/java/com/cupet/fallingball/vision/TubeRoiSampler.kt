package com.cupet.fallingball.vision

import com.cupet.fallingball.vision.AnalysisFrame
import java.nio.ByteBuffer
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.floor

data class TubeRoiGeometry(
    val revision: Long,
    /** top-left, top-right, bottom-right, bottom-left in logical display coordinates. */
    val sourceQuad: FloatArray,
    val sourceToRoi: DoubleArray,
    val roiToSource: DoubleArray
)

/** Samples the camera Y plane into the same 152 x 1024 tube coordinates used by PC heatmap-C. */
class TubeRoiSampler(
    val outputWidth: Int = OUTPUT_WIDTH,
    val outputHeight: Int = OUTPUT_HEIGHT
) {
    init {
        require(outputWidth > 1 && outputHeight > 1)
    }

    fun sample(
        image: AnalysisFrame,
        transform: BufferTransform,
        axis: TubeAxis,
        output: ByteBuffer
    ): Boolean = sample(image, transform, geometry(axis, 0L), output)

    fun sample(
        image: AnalysisFrame,
        transform: BufferTransform,
        geometry: TubeRoiGeometry,
        output: ByteBuffer
    ): Boolean {
        val required = outputWidth * outputHeight
        require(output.capacity() >= required) { "ROI buffer requires $required bytes" }
        val plane = image.planes[0]
        val source = plane.buffer
        if (!source.isDirect || !output.isDirect || plane.pixelStride != 1) {
            return sampleFallback(image, transform, geometry.sourceQuad, output)
        }

        OpenCvRuntime.ensureLoaded()
        val sourceMat = Mat(
            image.height,
            image.width,
            CvType.CV_8UC1,
            source,
            plane.rowStride.toLong()
        )
        val outputMat = Mat(outputHeight, outputWidth, CvType.CV_8UC1, output)
        val sourcePoints = MatOfPoint2f(*Array(4) { index ->
            val logicalX = geometry.sourceQuad[index * 2]
            val logicalY = geometry.sourceQuad[index * 2 + 1]
            Point(
                transform.bufX(logicalX, logicalY).toDouble(),
                transform.bufY(logicalX, logicalY).toDouble()
            )
        })
        val targetPoints = canonicalTargetPoints()
        val matrix = Imgproc.getPerspectiveTransform(sourcePoints, targetPoints)
        try {
            Imgproc.warpPerspective(
                sourceMat,
                outputMat,
                matrix,
                Size(outputWidth.toDouble(), outputHeight.toDouble()),
                Imgproc.INTER_LINEAR,
                Core.BORDER_REPLICATE
            )
            output.position(0)
            output.limit(required)
            return true
        } finally {
            matrix.release()
            targetPoints.release()
            sourcePoints.release()
            outputMat.release()
            sourceMat.release()
        }
    }

    fun geometry(axis: TubeAxis, revision: Long): TubeRoiGeometry {
        OpenCvRuntime.ensureLoaded()
        val quad = sourceQuad(axis)
        val sourcePoints = MatOfPoint2f(*Array(4) { index ->
            Point(quad[index * 2].toDouble(), quad[index * 2 + 1].toDouble())
        })
        val targetPoints = canonicalTargetPoints()
        val sourceToRoiMat = Imgproc.getPerspectiveTransform(sourcePoints, targetPoints)
        val roiToSourceMat = Imgproc.getPerspectiveTransform(targetPoints, sourcePoints)
        return try {
            val sourceToRoi = DoubleArray(9)
            val roiToSource = DoubleArray(9)
            sourceToRoiMat.get(0, 0, sourceToRoi)
            roiToSourceMat.get(0, 0, roiToSource)
            TubeRoiGeometry(revision, quad, sourceToRoi, roiToSource)
        } finally {
            roiToSourceMat.release()
            sourceToRoiMat.release()
            targetPoints.release()
            sourcePoints.release()
        }
    }

    private fun sampleFallback(
        image: AnalysisFrame,
        transform: BufferTransform,
        sourceQuad: FloatArray,
        output: ByteBuffer
    ): Boolean {
        // Non-standard non-direct/pixel-strided Y planes are not used by the target phone.
        // Preserve a correct fallback by sampling the same rectangle parametrically.
        val plane = image.planes[0]
        val source = plane.buffer
        val capacity = source.capacity()
        output.clear()
        for (roiY in 0 until outputHeight) {
            val fy = roiY / (outputHeight - 1f)
            for (roiX in 0 until outputWidth) {
                val fx = roiX / (outputWidth - 1f)
                val topX = sourceQuad[0] * (1f - fx) + sourceQuad[2] * fx
                val topY = sourceQuad[1] * (1f - fx) + sourceQuad[3] * fx
                val bottomX = sourceQuad[6] * (1f - fx) + sourceQuad[4] * fx
                val bottomY = sourceQuad[7] * (1f - fx) + sourceQuad[5] * fx
                val displayX = topX * (1f - fy) + bottomX * fy
                val displayY = topY * (1f - fy) + bottomY * fy
                val bufferX = transform.clampBufXToCrop(
                    transform.bufX(displayX, displayY)
                )
                val bufferY = transform.clampBufYToCrop(
                    transform.bufY(displayX, displayY)
                )
                output.put(
                    bilinear(
                        source,
                        capacity,
                        plane.rowStride,
                        plane.pixelStride,
                        image.width,
                        image.height,
                        bufferX,
                        bufferY
                    ).toInt().coerceIn(0, 255).toByte()
                )
            }
        }
        output.flip()
        return true
    }

    private fun sourceQuad(axis: TubeAxis): FloatArray {
        val halfLong = axis.lengthPx * LONG_MARGIN * 0.5f
        val halfShort = axis.widthPx * SHORT_MARGIN * 0.5f
        val centerX = axis.ox + axis.ux * axis.lengthPx * 0.5f
        val centerY = axis.oy + axis.uy * axis.lengthPx * 0.5f
        val topX = centerX - axis.ux * halfLong
        val topY = centerY - axis.uy * halfLong
        val bottomX = centerX + axis.ux * halfLong
        val bottomY = centerY + axis.uy * halfLong
        return floatArrayOf(
            topX + axis.vx * halfShort,
            topY + axis.vy * halfShort,
            topX - axis.vx * halfShort,
            topY - axis.vy * halfShort,
            bottomX - axis.vx * halfShort,
            bottomY - axis.vy * halfShort,
            bottomX + axis.vx * halfShort,
            bottomY + axis.vy * halfShort
        )
    }

    private fun canonicalTargetPoints() = MatOfPoint2f(
        Point(0.0, 0.0),
        Point((outputWidth - 1).toDouble(), 0.0),
        Point((outputWidth - 1).toDouble(), (outputHeight - 1).toDouble()),
        Point(0.0, (outputHeight - 1).toDouble())
    )

    private fun bilinear(
        source: ByteBuffer,
        capacity: Int,
        rowStride: Int,
        pixelStride: Int,
        width: Int,
        height: Int,
        x: Float,
        y: Float
    ): Float {
        val x0 = floor(x).toInt()
        val y0 = floor(y).toInt()
        val fx = x - x0
        val fy = y - y0
        fun value(xx: Int, yy: Int): Float {
            val clampedX = xx.coerceIn(0, width - 1)
            val clampedY = yy.coerceIn(0, height - 1)
            val index = clampedY * rowStride + clampedX * pixelStride
            if (index !in 0 until capacity) return 0f
            return (source.get(index).toInt() and 0xff).toFloat()
        }
        val top = value(x0, y0) * (1f - fx) + value(x0 + 1, y0) * fx
        val bottom = value(x0, y0 + 1) * (1f - fx) + value(x0 + 1, y0 + 1) * fx
        return top * (1f - fy) + bottom * fy
    }

    companion object {
        const val OUTPUT_WIDTH = 152
        const val OUTPUT_HEIGHT = 1024
        const val LONG_MARGIN = 1.02f
        const val SHORT_MARGIN = 1.10f

        /** Maps the canonical model row onto the complete YOLO tube long axis. */
        internal fun canonicalAxialPx(
            roiY: Float,
            roiHeight: Int,
            axisLengthPx: Float
        ): Float = (
            roiY / (roiHeight - 1f) * LONG_MARGIN -
                (LONG_MARGIN - 1f) * 0.5f
            ) * axisLengthPx

        /**
         * Maps the model column across 110% of the YOLO tube width. Positive values point
         * display-left because [TubeAxis.v] does; canonical x itself grows left-to-right.
         */
        internal fun canonicalLateralPx(
            roiX: Float,
            roiWidth: Int,
            axisWidthPx: Float
        ): Float = (0.5f - roiX / (roiWidth - 1f)) * axisWidthPx * SHORT_MARGIN
    }
}
