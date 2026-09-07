package com.cupet.fallingball

internal enum class DebugTestAction(val wireValue: String) {
    RELOCK("relock"),
    START_TRACKING("start_tracking"),
    STOP_TRACKING("stop_tracking"),
    START_TEMPORAL_CAPTURE("start_temporal_capture"),
    START_PIPELINE_TIMING_CAPTURE("start_pipeline_timing_capture"),
    STOP_TEMPORAL_CAPTURE("stop_temporal_capture"),
    BENCHMARK_TINY_PARENT("benchmark_tiny_parent"),
    BENCHMARK_TINY_CPU_THREADS("benchmark_tiny_cpu_threads"),
    BENCHMARK_TINY_CANDIDATE("benchmark_tiny_candidate"),
    BENCHMARK_TINY_NCNN("benchmark_tiny_ncnn")
}

/**
 * Narrow debug-build control vocabulary used by the unattended real-device runner.
 *
 * MainActivity checks ApplicationInfo.FLAG_DEBUGGABLE before acting on these values, so
 * release builds do not expose a remote control surface.
 */
internal object DebugTestControl {
    const val EXTRA_ACTION = "com.cupet.fallingball.DEBUG_TEST_ACTION"
    const val LOG_TAG = "FallingBallTestControl"

    fun parse(raw: String?): DebugTestAction? =
        DebugTestAction.entries.firstOrNull { it.wireValue == raw }
}
