package com.cupet.fallingball.camera

import android.content.Context
import android.util.Log
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/** Small bounded background log readable with `adb shell run-as` on debug builds. */
internal class RuntimeDiagnosticsRecorder(context: Context) : AutoCloseable {
    private val directory = context.filesDir.resolve("diagnostics")
    private val current = directory.resolve("runtime.log")
    private val previous = directory.resolve("runtime.previous.log")
    private val executor = Executors.newSingleThreadExecutor()
    private val closed = AtomicBoolean(false)

    fun record(message: String) {
        if (closed.get()) return
        runCatching {
            executor.execute {
                runCatching {
                    directory.mkdirs()
                    rotateIfNeeded()
                    current.appendText("${System.currentTimeMillis()} $message\n", Charsets.UTF_8)
                }.onFailure { Log.w(TAG, "Unable to write runtime diagnostics", it) }
            }
        }.onFailure { Log.w(TAG, "Unable to queue runtime diagnostics", it) }
    }

    private fun rotateIfNeeded() {
        if (!current.exists() || current.length() < MAX_BYTES) return
        if (previous.exists() && !previous.delete()) {
            Log.w(TAG, "Unable to remove previous diagnostics log")
        }
        if (!current.renameTo(previous)) {
            current.writeText("", Charsets.UTF_8)
        }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        executor.shutdown()
        if (!executor.awaitTermination(2, TimeUnit.SECONDS)) {
            executor.shutdownNow()
        }
    }

    companion object {
        private const val TAG = "RuntimeDiagnostics"
        private const val MAX_BYTES = 256L * 1024L
    }
}
