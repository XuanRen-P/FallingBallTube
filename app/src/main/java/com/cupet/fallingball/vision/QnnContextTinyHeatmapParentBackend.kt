package com.cupet.fallingball.vision

import android.content.Context
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * TinyHeatmapNet parent backend executing the AI Hub-compiled HTP context
 * binary (`tinyheatmapnet_sep_qnn_context.bin`) through [QnnParentRuntime].
 *
 * Placement evidence: the context binary was compiled by AI Hub for the HTP
 * (v75, FP16) target, and this backend only reports ready after the native
 * side deserialized it onto a live QNN HTP device.  A context binary built for
 * a different Hexagon arch fails at deserialization, which propagates as a
 * `null` creation and lets the detector fall back to the FP32 CPU backend.
 */
internal class QnnContextTinyHeatmapParentBackend private constructor(
    private val contextLease: QnnSharedContext.Lease,
    private val inputBuffer: ByteBuffer,
    private val heatmapBuffer: ByteBuffer,
    private val offsetBuffer: ByteBuffer,
    private val heatmapOut: FloatArray,
    private val offsetOut: FloatArray
) : TinyHeatmapParentBackend {
    private val latencyAccumulator = TinyHeatmapLatencyAccumulator()
    private var warmedUp = false
    private var closed = false

    override val backendName: String = "qnn_htp_context"
    override val computeUnit: String = "NPU"
    override val placementVerified: Boolean = true
    override val placementSummary: String = "qnn_htp_context_binary_fp16"

    override val latencyStats: TinyHeatmapLatencyStats
        get() = latencyAccumulator.snapshot()

    @Synchronized
    override fun warmUp() {
        check(!closed) { "TinyHeatmapNet parent backend is closed" }
        if (warmedUp) return
        infer(FloatArray(INPUT_FLOATS) { 0.5f })
        warmedUp = true
    }

    @Synchronized
    override fun infer(inputValues: FloatArray): TinyHeatmapParentOutputs {
        check(!closed) { "TinyHeatmapNet parent backend is closed" }
        require(inputValues.size == INPUT_FLOATS) {
            "TinyHeatmapNet parent input size mismatch: ${inputValues.size}"
        }
        inputBuffer.clear()
        inputBuffer.asFloatBuffer().put(inputValues)
        inputBuffer.rewind()
        val started = System.nanoTime()
        val ok = contextLease.execute(inputBuffer, heatmapBuffer, offsetBuffer)
        val elapsedMs = (System.nanoTime() - started) / 1_000_000f
        if (!ok) {
            throw IllegalStateException("QNN HTP parent execution failed")
        }
        latencyAccumulator.record(elapsedMs)
        heatmapBuffer.rewind()
        offsetBuffer.rewind()
        heatmapBuffer.asFloatBuffer().get(heatmapOut)
        offsetBuffer.asFloatBuffer().get(offsetOut)
        return TinyHeatmapParentOutputs(heatmapOut, offsetOut)
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        contextLease.close()
    }

    companion object {
        internal const val CONTEXT_ASSET = "tinyheatmapnet_sep_qnn_context.bin"
        internal const val INPUT_FLOATS = 5 * 1024 * 152
        internal const val HEATMAP_FLOATS = 512 * 76
        internal const val OFFSET_FLOATS = 2 * 512 * 76

        /** Returns `null` when the HTP context cannot be created on this device. */
        fun create(context: Context): QnnContextTinyHeatmapParentBackend? =
            runCatching {
                val binary = context.assets.open(CONTEXT_ASSET).use { it.readBytes() }
                val htpPath = "${context.applicationInfo.nativeLibraryDir}/libQnnHtp.so"
                val backend = create { QnnParentRuntime.acquire(htpPath, binary) }
                if (backend == null) {
                    Log.w(
                        TAG,
                        "QNN HTP context unavailable; TinyHeatmapNet parent stays on CPU"
                    )
                    return@runCatching null
                }
                Log.i(TAG, "QNN HTP parent context ready (fp16 context binary)")
                backend
            }
                .onFailure {
                    Log.w(TAG, "QNN HTP backend construction failed; using FP32 CPU fallback", it)
                }
                .getOrNull()

        /** Native acquisition is isolated so ownership can be tested across real backends. */
        internal fun create(acquireContext: () -> QnnSharedContext.Lease?): QnnContextTinyHeatmapParentBackend? {
            // Allocate before acquiring a lease; resource exhaustion must not retain a graph.
            val inputBuffer = directFloatBuffer(INPUT_FLOATS)
            val heatmapBuffer = directFloatBuffer(HEATMAP_FLOATS)
            val offsetBuffer = directFloatBuffer(OFFSET_FLOATS)
            val heatmapOut = FloatArray(HEATMAP_FLOATS)
            val offsetOut = FloatArray(OFFSET_FLOATS)
            val lease = acquireContext() ?: return null
            return try {
                QnnContextTinyHeatmapParentBackend(
                    lease, inputBuffer, heatmapBuffer, offsetBuffer, heatmapOut, offsetOut
                )
            } catch (failure: Throwable) {
                try {
                    lease.close()
                } catch (cleanup: Throwable) {
                    failure.addSuppressed(cleanup)
                }
                throw failure
            }
        }

        private const val TAG = "TinyHeatmapNet"

        private fun directFloatBuffer(floats: Int): ByteBuffer =
            ByteBuffer.allocateDirect(floats * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
    }
}
