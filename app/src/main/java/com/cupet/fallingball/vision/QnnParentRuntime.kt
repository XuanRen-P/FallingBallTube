package com.cupet.fallingball.vision

import java.nio.ByteBuffer

/**
 * Thin JNI boundary over the QNN HTP executor in `qnn_parent_executor.cpp`.
 *
 * The native side owns one process-global graph. Each detector holds a lease;
 * all native calls share a lock and the last lease releases the graph, so closing
 * the live camera cannot tear down an offline analysis using the same accelerator.
 */
internal object QnnParentRuntime {
    @Volatile
    private var libraryLoaded = false

    private val sharedContext = QnnSharedContext(object : QnnSharedContext.Native {
        override fun create(htpLibraryPath: String, contextBinary: ByteArray): Boolean =
            nativeCreateFromBuffer(htpLibraryPath, contextBinary)

        override fun execute(input: ByteBuffer, heatmap: ByteBuffer, offset: ByteBuffer): Boolean =
            nativeExecute(input, heatmap, offset)

        override fun destroy() = nativeDestroy()
    })

    /** Loads the JNI bindings once; safe to call repeatedly. */
    @Synchronized
    fun ensureLibraryLoaded() {
        if (libraryLoaded) return
        System.loadLibrary("fallingball_native")
        libraryLoaded = true
    }

    /**
     * Deserializes [contextBinary] (AI Hub `qnn_context_binary` output) onto the
     * HTP device and retrieves the single graph. Returns a lease only when the
     * frozen I/O contract was validated against the binary's metadata.
     */
    fun acquire(htpLibraryPath: String, contextBinary: ByteArray): QnnSharedContext.Lease? =
        try {
            ensureLibraryLoaded()
            sharedContext.acquire(htpLibraryPath, contextBinary)
        } catch (error: Throwable) {
            android.util.Log.w("TinyHeatmapNet", "QNN context construction failed", error)
            null
        }

    private external fun nativeCreateFromBuffer(
        htpLibraryPath: String,
        contextBinary: ByteArray
    ): Boolean

    private external fun nativeExecute(
        input: ByteBuffer,
        heatmap: ByteBuffer,
        offset: ByteBuffer
    ): Boolean

    private external fun nativeDestroy()
}
