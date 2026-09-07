package com.cupet.fallingball.vision

import android.graphics.Rect
import android.media.Image
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

/** Borrowed native-resolution planes, valid until close(). */
class AnalysisFrame(
    val width: Int,
    val height: Int,
    val cropRect: Rect,
    val imageInfo: Info,
    val planes: List<Plane>,
    val sourceFrameIndex: Int = -1,
    private val release: () -> Unit = {}
) : AutoCloseable {
    constructor(
        width: Int, height: Int, cropRect: Rect,
        rotationDegrees: Int, timestampNs: Long, sourceFrameIndex: Long,
        planes: Array<Plane>
    ) : this(width, height, cropRect, Info(timestampNs, rotationDegrees), planes.toList(),
        Math.toIntExact(sourceFrameIndex))

    data class Info(val timestamp: Long, val rotationDegrees: Int)
    data class Plane(val buffer: ByteBuffer, val rowStride: Int, val pixelStride: Int)
    private val closed = AtomicBoolean(false)

    init {
        require(width > 0 && height > 0 && planes.size == 3)
        require(imageInfo.rotationDegrees in listOf(0, 90, 180, 270))
        require(cropRect.left >= 0 && cropRect.top >= 0 &&
            cropRect.right <= width && cropRect.bottom <= height && !cropRect.isEmpty)
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) release()
    }

    companion object {
        fun fromCamera(image: ImageProxy): AnalysisFrame = AnalysisFrame(
            image.width, image.height, Rect(image.cropRect),
            Info(image.imageInfo.timestamp, image.imageInfo.rotationDegrees),
            image.planes.map { Plane(it.buffer, it.rowStride, it.pixelStride) },
            release = image::close
        )

        fun fromVideo(image: Image, timestampNs: Long, rotation: Int, index: Int): AnalysisFrame =
            AnalysisFrame(
                image.width, image.height, Rect(image.cropRect), Info(timestampNs, rotation),
                image.planes.map { Plane(it.buffer, it.rowStride, it.pixelStride) },
                index, image::close
            )
    }
}

typealias AnalysisPlane = AnalysisFrame.Plane
