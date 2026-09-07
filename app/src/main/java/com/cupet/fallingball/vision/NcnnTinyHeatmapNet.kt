package com.cupet.fallingball.vision

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * NCNN parent runner for a fixed TinyHeatmapNet graph.
 *
 * The asset names are deliberately constructor arguments.  This keeps experimental
 * BatchNorm graphs available to the benchmark actions without accidentally replacing the
 * production GroupNorm graph (whose output contract and quality are already frozen).
 */
internal class NcnnTinyHeatmapNet(
    context: Context,
    useVulkan: Boolean,
    private val paramAsset: String = DEFAULT_PARAM_ASSET,
    private val modelAsset: String = DEFAULT_MODEL_ASSET,
    private val requireVerifiedPlacement: Boolean = true,
) : TinyHeatmapParentBackend {
    private val useVulkan = useVulkan
    private val input: FloatBuffer = directFloatBuffer(INPUT_SIZE)
    private val heatmap: FloatBuffer = directFloatBuffer(HEATMAP_SIZE)
    private val offset: FloatBuffer = directFloatBuffer(OFFSET_SIZE)
    private val heatmapValues = FloatArray(HEATMAP_SIZE)
    private val offsetValues = FloatArray(OFFSET_SIZE)
    private val latencyAccumulator = TinyHeatmapLatencyAccumulator()
    private var handle: Long = nativeCreate(
        context.applicationContext.assets,
        paramAsset,
        modelAsset,
        useVulkan
    )
    override val placementVerified: Boolean = nativePlacementVerified(handle)
    override val placementSummary: String = nativePlacementSummary(handle)
    private var warmedUp = false
    private var closed = false

    init {
        check(handle != 0L) { "Unable to create NCNN TinyHeatmapNet session" }
        if (requireVerifiedPlacement && !placementVerified) {
            nativeDestroy(handle)
            handle = 0L
            error("NCNN runtime placement was not verified: $placementSummary")
        }
    }

    override val backendName: String = if (useVulkan) "ncnn_vulkan" else "ncnn_cpu"
    override val computeUnit: String = if (useVulkan) "GPU" else "CPU"
    override val latencyStats: TinyHeatmapLatencyStats
        get() = latencyAccumulator.snapshot()

    @Synchronized
    override fun infer(inputValues: FloatArray): TinyHeatmapParentOutputs {
        check(!closed && handle != 0L) { "NCNN TinyHeatmapNet session is closed" }
        require(inputValues.size == INPUT_SIZE) {
            "Expected $INPUT_SIZE input values, got ${inputValues.size}"
        }
        if (!warmedUp) warmUp()
        input.clear()
        input.put(inputValues)
        input.flip()
        val started = System.nanoTime()
        check(run()) { "NCNN inference failed" }
        heatmap.rewind()
        offset.rewind()
        heatmap.get(heatmapValues)
        offset.get(offsetValues)
        latencyAccumulator.record((System.nanoTime() - started) / 1_000_000f)
        return TinyHeatmapParentOutputs(heatmapValues, offsetValues)
    }

    @Synchronized
    override fun warmUp() {
        check(!closed && handle != 0L) { "NCNN TinyHeatmapNet session is closed" }
        if (warmedUp) return
        input.clear()
        repeat(INPUT_SIZE) { input.put(0.5f) }
        input.flip()
        try {
            repeat(WARMUP_RUNS) { check(run()) { "NCNN warmup failed" } }
            // Do not mark the backend ready until every warm-up invocation succeeds. If a
            // Vulkan shader compile fails halfway through, the caller must be able to retry or
            // fall back to ORT rather than paying the failed warm-up on every camera frame.
            warmedUp = true
        } catch (error: Throwable) {
            warmedUp = false
            throw error
        }
    }

    /** Retained for the debug benchmark action; warm-up calls are excluded from the stats. */
    @Synchronized
    fun benchmark(iterations: Int = 8): FloatArray {
        return benchmarkResult(iterations).samplesMs
    }

    /** Returns the path that ran and preserves whether its placement evidence was verified. */
    @Synchronized
    internal fun benchmarkResult(iterations: Int = 8): TinyHeatmapBenchmarkResult {
        require(iterations in 1..100) { "Benchmark iterations must be in 1..100" }
        warmUp()
        val samples = FloatArray(iterations) {
            val started = System.nanoTime()
            check(run()) { "NCNN inference failed" }
            (System.nanoTime() - started) / 1_000_000f
        }
        return TinyHeatmapBenchmarkResult(
            backendName = backendName,
            computeUnit = computeUnit,
            providerPlacementVerified = placementVerified,
            samplesMs = samples,
            assetName = modelAsset,
            placementSummary = placementSummary
        )
    }

    private fun run(): Boolean = nativeRun(handle, input, heatmap, offset)

    private external fun nativeCreate(
        assets: android.content.res.AssetManager,
        paramAsset: String,
        modelAsset: String,
        useVulkan: Boolean
    ): Long

    private external fun nativeRun(
        handle: Long,
        input: FloatBuffer,
        heatmap: FloatBuffer,
        offset: FloatBuffer
    ): Boolean

    private external fun nativePlacementVerified(handle: Long): Boolean

    private external fun nativePlacementSummary(handle: Long): String

    private external fun nativeDestroy(handle: Long)

    @Synchronized
    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
        warmedUp = false
        closed = true
    }

    private companion object {
        init {
            System.loadLibrary("fallingball_native")
        }

        /** Exact production GroupNorm graph converted from the frozen FP32 checkpoint. */
        const val DEFAULT_PARAM_ASSET = "tinyheatmapnet_final.ncnn.param"
        const val DEFAULT_MODEL_ASSET = "tinyheatmapnet_final.ncnn.bin"
        const val INPUT_SIZE = 5 * 1024 * 152
        const val HEATMAP_SIZE = 1 * 512 * 76
        const val OFFSET_SIZE = 2 * 512 * 76
        const val WARMUP_RUNS = 2

        fun directFloatBuffer(size: Int): FloatBuffer =
            ByteBuffer.allocateDirect(size * Float.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()

    }
}
