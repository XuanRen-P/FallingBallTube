package com.cupet.fallingball.camera

import com.cupet.fallingball.vision.YuvTensorPreprocessor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Reusable owned camera-plane snapshot for asynchronous YOLO preprocessing. */
internal class YuvFrameSnapshotBuffer {
    private var y = emptyDirectBuffer()
    private var u = emptyDirectBuffer()
    private var v = emptyDirectBuffer()

    fun copyFrom(source: YuvTensorPreprocessor.Frame): YuvTensorPreprocessor.Frame {
        y = copyPlane(source.y, y)
        u = copyPlane(source.u, u)
        v = copyPlane(source.v, v)
        return source.copy(
            y = y.duplicate().order(ByteOrder.nativeOrder()),
            u = u.duplicate().order(ByteOrder.nativeOrder()),
            v = v.duplicate().order(ByteOrder.nativeOrder())
        )
    }

    private fun copyPlane(source: ByteBuffer, destination: ByteBuffer): ByteBuffer {
        val owned = if (destination.capacity() == source.capacity()) {
            destination
        } else {
            ByteBuffer.allocateDirect(source.capacity()).order(ByteOrder.nativeOrder())
        }
        val input = source.duplicate().apply { clear() }
        owned.clear()
        owned.put(input)
        owned.flip()
        return owned
    }

    companion object {
        private fun emptyDirectBuffer(): ByteBuffer =
            ByteBuffer.allocateDirect(0).order(ByteOrder.nativeOrder())
    }
}
