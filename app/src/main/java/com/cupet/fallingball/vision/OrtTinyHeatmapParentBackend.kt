package com.cupet.fallingball.vision

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * The deliberately boring fallback adapter: one fixed ORT CPU session and reusable buffers.
 * Keeping it behind the same interface as NCNN makes fallback observable instead of silently
 * changing the meaning of a benchmark.
 */
internal class OrtTinyHeatmapParentBackend(
    environment: OrtEnvironment,
    modelBytes: ByteArray,
    private val inputShape: LongArray,
    private val heatmapSize: Int,
    private val offsetSize: Int,
    threads: Int,
    override val backendName: String = "ort_cpu",
    override val computeUnit: String = "CPU",
    override val placementVerified: Boolean = backendName == "ort_cpu" && computeUnit == "CPU",
    override val placementSummary: String = if (placementVerified) {
        "ort_cpu_full_graph"
    } else {
        "unverified"
    }
) : TinyHeatmapParentBackend {
    private val environment = environment
    private val inputSize = inputShape.fold(1L) { product, dimension -> product * dimension }
        .also { require(it in 1..Int.MAX_VALUE) { "Input tensor is too large" } }
        .toInt()
    private val inputBuffer = directFloatBuffer(inputSize)
    private val warmupInput = FloatArray(inputSize) { 0.5f }
    private val heatmap = FloatArray(heatmapSize)
    private val offset = FloatArray(offsetSize)
    private val latencyAccumulator = TinyHeatmapLatencyAccumulator()
    private val session: OrtSession = createSession(environment, modelBytes, threads)
    private val inputName = session.inputNames.firstOrNull()
        ?: error("TinyHeatmapNet parent has no input")
    /**
     * The tensor points at [inputBuffer], which is only mutated while this backend is synchronized.
     * Reusing it avoids allocating and registering a Java/native tensor on every camera frame.
     */
    private val inputTensor: OnnxTensor = OnnxTensor.createTensor(
        environment,
        inputBuffer,
        inputShape
    )
    private var warmedUp = false
    private var closed = false

    override val latencyStats: TinyHeatmapLatencyStats
        get() = latencyAccumulator.snapshot()

    @Synchronized
    override fun warmUp() {
        check(!closed) { "TinyHeatmapNet parent backend is closed" }
        if (warmedUp) return
        repeat(WARMUP_RUNS) { runInternal(warmupInput, recordLatency = false) }
        warmedUp = true
    }

    @Synchronized
    override fun infer(input: FloatArray): TinyHeatmapParentOutputs {
        check(!closed) { "TinyHeatmapNet parent backend is closed" }
        require(input.size == inputSize) {
            "Expected $inputSize parent input values, got ${input.size}"
        }
        if (!warmedUp) warmUp()
        return runInternal(input, recordLatency = true)
    }

    private fun runInternal(
        values: FloatArray,
        recordLatency: Boolean
    ): TinyHeatmapParentOutputs {
        inputBuffer.clear()
        inputBuffer.put(values)
        inputBuffer.flip()
        val started = System.nanoTime()
        session.run(mapOf(inputName to inputTensor)).use { result ->
            val heatmapBuffer = (result[0] as? OnnxTensor)?.floatBuffer
                ?: error("TinyHeatmapNet parent did not return a heatmap")
            val offsetBuffer = (result[1] as? OnnxTensor)?.floatBuffer
                ?: error("TinyHeatmapNet parent did not return an offset")
            require(heatmapBuffer.remaining() >= heatmapSize) {
                "Parent heatmap has ${heatmapBuffer.remaining()} values; expected $heatmapSize"
            }
            require(offsetBuffer.remaining() >= offsetSize) {
                "Parent offset has ${offsetBuffer.remaining()} values; expected $offsetSize"
            }
            for (index in 0 until heatmapSize) heatmap[index] = heatmapBuffer.get(index)
            for (index in 0 until offsetSize) offset[index] = offsetBuffer.get(index)
            val elapsedMs = (System.nanoTime() - started) / 1_000_000f
            if (recordLatency) latencyAccumulator.record(elapsedMs)
            return TinyHeatmapParentOutputs(heatmap, offset)
        }
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        runCatching { inputTensor.close() }
        session.close()
    }

    private fun createSession(
        environment: OrtEnvironment,
        modelBytes: ByteArray,
        threads: Int
    ): OrtSession {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads.coerceAtLeast(1))
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setMemoryPatternOptimization(true)
        }
        return try {
            environment.createSession(modelBytes, options)
        } finally {
            options.close()
        }
    }

    private fun directFloatBuffer(size: Int): FloatBuffer =
        ByteBuffer.allocateDirect(size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

    private companion object {
        const val WARMUP_RUNS = 2
    }
}
