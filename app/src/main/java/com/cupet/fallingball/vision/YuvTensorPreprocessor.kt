package com.cupet.fallingball.vision

import java.nio.ByteBuffer
import java.nio.FloatBuffer
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.round

/**
 * Converts one CameraX YUV frame into the FP32 NCHW tensor consumed by YOLO.
 *
 * The interface deliberately contains the complete frame layout and geometry for one call,
 * hiding native availability and fallback selection from the camera pipeline. Direct buffers
 * use the native implementation; unsupported layouts retain a numerically equivalent Kotlin
 * adapter.
 */
class YuvTensorPreprocessor {

    data class Frame(
        val y: ByteBuffer,
        val u: ByteBuffer,
        val v: ByteBuffer,
        val width: Int,
        val height: Int,
        val yRowStride: Int,
        val yPixelStride: Int,
        val uRowStride: Int,
        val uPixelStride: Int,
        val vRowStride: Int,
        val vPixelStride: Int
    )

    data class Geometry(
        val rotationDegrees: Int,
        val cropLeft: Int,
        val cropTop: Int,
        val cropWidth: Int,
        val cropHeight: Int,
        val roiX: Int,
        val roiY: Int,
        val roiWidth: Int,
        val roiHeight: Int,
        val modelSize: Int
    )

    data class Result(val nativeUsed: Boolean)

    fun preprocess(frame: Frame, geometry: Geometry, output: FloatBuffer): Result {
        validate(frame, geometry, output)
        // A smaller profile leaves the reusable max-sized buffer with a smaller limit.
        // Restore its full writable range before either implementation writes by index.
        output.clear()
        val canUseNative = nativeLibraryLoaded &&
            frame.y.isDirect && frame.u.isDirect && frame.v.isDirect && output.isDirect
        if (canUseNative && nativePreprocess(
                frame.y,
                frame.u,
                frame.v,
                output,
                frame.width,
                frame.height,
                frame.yRowStride,
                frame.yPixelStride,
                frame.uRowStride,
                frame.uPixelStride,
                frame.vRowStride,
                frame.vPixelStride,
                geometry.rotationDegrees,
                geometry.cropLeft,
                geometry.cropTop,
                geometry.cropWidth,
                geometry.cropHeight,
                geometry.roiX,
                geometry.roiY,
                geometry.roiWidth,
                geometry.roiHeight,
                geometry.modelSize
            )
        ) {
            prepareForRead(output, geometry.modelSize)
            return Result(nativeUsed = true)
        }

        preprocessKotlin(frame, geometry, output)
        prepareForRead(output, geometry.modelSize)
        return Result(nativeUsed = false)
    }

    internal fun preprocessKotlin(frame: Frame, geometry: Geometry, output: FloatBuffer) {
        val affine = affine(geometry)
        val modelSize = geometry.modelSize
        val scale = modelSize.toFloat() / max(geometry.roiWidth, geometry.roiHeight)
        val newW = round(geometry.roiWidth * scale).toInt().coerceAtLeast(1)
        val newH = round(geometry.roiHeight * scale).toInt().coerceAtLeast(1)
        val padLeft = round((modelSize - newW) / 2f - 0.1f).toInt().coerceAtLeast(0)
        val padTop = round((modelSize - newH) / 2f - 0.1f).toInt().coerceAtLeast(0)
        val channelSize = modelSize * modelSize
        val required = 3 * channelSize
        val padValue = 114f / 255f
        for (index in 0 until required) output.put(index, padValue)

        val maxChromaX = (frame.width + 1) / 2 - 1
        val maxChromaY = (frame.height + 1) / 2 - 1
        fun yAt(x: Int, y: Int): Float {
            val xx = x.coerceIn(0, frame.width - 1)
            val yy = y.coerceIn(0, frame.height - 1)
            val index = yy * frame.yRowStride + xx * frame.yPixelStride
            return if (index in 0 until frame.y.capacity()) {
                (frame.y.get(index).toInt() and 0xFF).toFloat()
            } else {
                0f
            }
        }
        fun uvAt(
            x: Int,
            y: Int,
            buffer: ByteBuffer,
            rowStride: Int,
            pixelStride: Int
        ): Int {
            val xx = (x / 2).coerceIn(0, maxChromaX)
            val yy = (y / 2).coerceIn(0, maxChromaY)
            val index = yy * rowStride + xx * pixelStride
            return if (index in 0 until buffer.capacity()) {
                buffer.get(index).toInt() and 0xFF
            } else {
                128
            }
        }

        val scaleSrcX = geometry.roiWidth.toFloat() / newW
        val scaleSrcY = geometry.roiHeight.toFloat() / newH
        val inv255 = 1f / 255f
        for (dy in 0 until newH) {
            val logicalY = geometry.roiY + (dy + 0.5f) * scaleSrcY - 0.5f
            val rowBufferX = affine.xky * logicalY + affine.xOff
            val rowBufferY = affine.yky * logicalY + affine.yOff
            val destinationRow = (padTop + dy) * modelSize + padLeft
            for (dx in 0 until newW) {
                val logicalX = geometry.roiX + (dx + 0.5f) * scaleSrcX - 0.5f
                val bufferX = affine.xkx * logicalX + rowBufferX
                val bufferY = affine.ykx * logicalX + rowBufferY
                val x0 = floor(bufferX).toInt()
                val y0 = floor(bufferY).toInt()
                val fx = (bufferX - x0).coerceIn(0f, 1f)
                val fy = (bufferY - y0).coerceIn(0f, 1f)
                val y00 = yAt(x0, y0)
                val y10 = yAt(x0 + 1, y0)
                val y01 = yAt(x0, y0 + 1)
                val y11 = yAt(x0 + 1, y0 + 1)
                val yUpper = y00 + (y10 - y00) * fx
                val yLower = y01 + (y11 - y01) * fx
                val yValue = yUpper + (yLower - yUpper) * fy
                val chromaX = (bufferX + 0.5f).toInt()
                val chromaY = (bufferY + 0.5f).toInt()
                val uf = (
                    uvAt(
                        chromaX,
                        chromaY,
                        frame.u,
                        frame.uRowStride,
                        frame.uPixelStride
                    ) - 128
                    ).toFloat()
                val vf = (
                    uvAt(
                        chromaX,
                        chromaY,
                        frame.v,
                        frame.vRowStride,
                        frame.vPixelStride
                    ) - 128
                    ).toFloat()
                val destination = destinationRow + dx
                output.put(
                    destination,
                    ((yValue + 1.402f * vf) * inv255).coerceIn(0f, 1f)
                )
                output.put(
                    channelSize + destination,
                    ((yValue - 0.34414f * uf - 0.71414f * vf) * inv255)
                        .coerceIn(0f, 1f)
                )
                output.put(
                    2 * channelSize + destination,
                    ((yValue + 1.772f * uf) * inv255).coerceIn(0f, 1f)
                )
            }
        }
    }

    private fun validate(frame: Frame, geometry: Geometry, output: FloatBuffer) {
        require(frame.width > 0 && frame.height > 0) { "Frame dimensions must be positive" }
        require(geometry.cropWidth > 0 && geometry.cropHeight > 0) {
            "Crop dimensions must be positive"
        }
        require(geometry.roiWidth > 0 && geometry.roiHeight > 0) {
            "ROI dimensions must be positive"
        }
        require(geometry.modelSize > 0) { "Model size must be positive" }
        require(
            geometry.rotationDegrees == 0 ||
                geometry.rotationDegrees == 90 ||
                geometry.rotationDegrees == 180 ||
                geometry.rotationDegrees == 270
        ) {
            "Unsupported rotation ${geometry.rotationDegrees}"
        }
        val required = 3 * geometry.modelSize * geometry.modelSize
        require(output.capacity() >= required) {
            "Output capacity ${output.capacity()} is smaller than $required"
        }
    }

    private data class Affine(
        val xkx: Float,
        val xky: Float,
        val xOff: Float,
        val ykx: Float,
        val yky: Float,
        val yOff: Float
    )

    private fun affine(geometry: Geometry): Affine = when (geometry.rotationDegrees) {
        90 -> Affine(
            xkx = 0f,
            xky = 1f,
            xOff = geometry.cropLeft.toFloat(),
            ykx = -1f,
            yky = 0f,
            yOff = (geometry.cropTop + geometry.cropHeight - 1).toFloat()
        )
        180 -> Affine(
            xkx = -1f,
            xky = 0f,
            xOff = (geometry.cropLeft + geometry.cropWidth - 1).toFloat(),
            ykx = 0f,
            yky = -1f,
            yOff = (geometry.cropTop + geometry.cropHeight - 1).toFloat()
        )
        270 -> Affine(
            xkx = 0f,
            xky = -1f,
            xOff = (geometry.cropLeft + geometry.cropWidth - 1).toFloat(),
            ykx = 1f,
            yky = 0f,
            yOff = geometry.cropTop.toFloat()
        )
        else -> Affine(
            xkx = 1f,
            xky = 0f,
            xOff = geometry.cropLeft.toFloat(),
            ykx = 0f,
            yky = 1f,
            yOff = geometry.cropTop.toFloat()
        )
    }

    private fun prepareForRead(output: FloatBuffer, modelSize: Int) {
        output.position(0)
        output.limit(3 * modelSize * modelSize)
    }

    @Suppress("LongParameterList")
    private external fun nativePreprocess(
        y: ByteBuffer,
        u: ByteBuffer,
        v: ByteBuffer,
        output: FloatBuffer,
        frameWidth: Int,
        frameHeight: Int,
        yRowStride: Int,
        yPixelStride: Int,
        uRowStride: Int,
        uPixelStride: Int,
        vRowStride: Int,
        vPixelStride: Int,
        rotationDegrees: Int,
        cropLeft: Int,
        cropTop: Int,
        cropWidth: Int,
        cropHeight: Int,
        roiX: Int,
        roiY: Int,
        roiWidth: Int,
        roiHeight: Int,
        modelSize: Int
    ): Boolean

    companion object {
        private val nativeLibraryLoaded: Boolean = try {
            System.loadLibrary("fallingball_native")
            true
        } catch (_: Throwable) {
            false
        }

        val isNativeAvailable: Boolean
            get() = nativeLibraryLoaded
    }
}
