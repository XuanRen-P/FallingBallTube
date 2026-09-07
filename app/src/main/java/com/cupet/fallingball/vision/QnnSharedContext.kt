package com.cupet.fallingball.vision

import java.nio.ByteBuffer

/**
 * Owns the single native QNN graph while live, video and benchmark detectors come and go.
 * Native execution mutates global tensor pointers, so it shares the ownership lock with
 * creation and destruction. A paused detector keeps a lease without blocking video inference.
 */
internal class QnnSharedContext(private val native: Native) {
    internal interface Native {
        fun create(htpLibraryPath: String, contextBinary: ByteArray): Boolean
        fun execute(input: ByteBuffer, heatmap: ByteBuffer, offset: ByteBuffer): Boolean
        fun destroy()
    }

    private val lock = Any()
    private var owners = 0

    /** All production callers use the same frozen context asset and HTP library. */
    fun acquire(htpLibraryPath: String, contextBinary: ByteArray): Lease? = synchronized(lock) {
        // Allocate the owner before native resources so a failed allocation cannot leak them.
        val lease = Lease()
        if (owners == 0) {
            val created = try {
                native.create(htpLibraryPath, contextBinary)
            } catch (failure: Throwable) {
                try {
                    native.destroy()
                } catch (cleanup: Throwable) {
                    failure.addSuppressed(cleanup)
                }
                throw failure
            }
            if (!created) {
                native.destroy()
                return@synchronized null
            }
        }
        owners++
        lease
    }

    inner class Lease internal constructor() : AutoCloseable {
        private var closed = false

        fun execute(input: ByteBuffer, heatmap: ByteBuffer, offset: ByteBuffer): Boolean =
            synchronized(lock) {
                check(!closed) { "QNN context lease is closed" }
                native.execute(input, heatmap, offset)
            }

        override fun close() = synchronized(lock) {
            if (closed) return@synchronized
            closed = true
            check(owners > 0) { "QNN context has no remaining owner" }
            owners--
            if (owners == 0) native.destroy()
        }
    }
}
