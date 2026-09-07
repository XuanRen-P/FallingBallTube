package com.cupet.fallingball.vision

/**
 * End-to-end timing for one fresh YOLO result.
 *
 * Durations are split at the seams where work changes ownership: camera-to-worker queue,
 * camera tensor preprocessing, ONNX tensor setup, ONNX execution, and output processing.
 * Runtime state is carried with the sample so the phone can show which implementation and
 * thread count produced the numbers.
 */
data class PipelineTimings(
    val queueMs: Float = 0f,
    val preprocessMs: Float = 0f,
    val tensorSetupMs: Float = 0f,
    val inferenceRunMs: Float = 0f,
    val postprocessMs: Float = 0f,
    val totalMs: Float = 0f,
    /** Zero means ONNX Runtime selected its automatic thread pool. */
    val cpuThreads: Int = 0,
    val nativePreprocess: Boolean = false,
    val tuning: Boolean = false
) {
    fun smoothedWith(latest: PipelineTimings, latestWeight: Float = 0.15f): PipelineTimings {
        val a = latestWeight.coerceIn(0f, 1f)
        fun blend(old: Float, new: Float) = old * (1f - a) + new * a
        return PipelineTimings(
            queueMs = blend(queueMs, latest.queueMs),
            preprocessMs = blend(preprocessMs, latest.preprocessMs),
            tensorSetupMs = blend(tensorSetupMs, latest.tensorSetupMs),
            inferenceRunMs = blend(inferenceRunMs, latest.inferenceRunMs),
            postprocessMs = blend(postprocessMs, latest.postprocessMs),
            totalMs = blend(totalMs, latest.totalMs),
            cpuThreads = latest.cpuThreads,
            nativePreprocess = latest.nativePreprocess,
            tuning = latest.tuning
        )
    }
}
