package com.cupet.fallingball.camera

import kotlin.math.ceil

/**
 * Preview-only retention for asynchronous primary-detector results.
 *
 * A result is already roughly one inference period old when it reaches the camera thread.
 * A fixed 200 ms limit therefore hid every point on the measured 2.3 FPS phone path. This
 * window follows the real cadence and inference time; measurement timestamps remain unchanged.
 */
internal object TemporalResultDisplayPolicy {
    private const val MIN_AGE_MS = 250L
    private const val MAX_AGE_MS = 1_500L
    private const val PERIODS_TO_KEEP = 1.5f

    /** Below this cadence the hand-off adds whole periods, so the window has to grow too. */
    private const val SLOW_CADENCE_FPS = 8f
    private const val SLOW_PERIODS_TO_KEEP = 3.0f

    fun maxAgeMs(modelFps: Float, inferenceMs: Float?): Long {
        val periodMs = modelFps
            .takeIf { it.isFinite() && it > 0f }
            ?.let { 1_000f / it }
            ?: 0f
        val runtimeMs = inferenceMs?.takeIf { it.isFinite() && it > 0f } ?: 0f
        val base = maxOf(MIN_AGE_MS.toFloat(), periodMs, runtimeMs)
        // A slow device queues the result behind the camera phase and the next hand-off, so a
        // 1.5-period window hid every point on the 4-5 FPS Dimensity path even though the same
        // frames were being recorded (record only requires ordering).  Measured there:
        // ages 300-635 ms against a 375 ms window.  Fast cadence keeps the tight window.
        val periods = if (modelFps.isFinite() && modelFps > 0f && modelFps < SLOW_CADENCE_FPS) {
            SLOW_PERIODS_TO_KEEP
        } else {
            PERIODS_TO_KEEP
        }
        return ceil(base * periods).toLong().coerceIn(MIN_AGE_MS, MAX_AGE_MS)
    }
}

internal data class TemporalResultUsage(
    val preview: Boolean,
    val record: Boolean
)

/** Keeps preview retention independent from exactly-once measurement consumption. */
internal class TemporalResultUsageTracker {
    private var lastRecordedTimestampNs = 0L

    fun resolve(
        frameTimestampNs: Long,
        resultTimestampNs: Long,
        previewMaxAgeMs: Long
    ): TemporalResultUsage {
        val ordered = resultTimestampNs > 0L && frameTimestampNs >= resultTimestampNs
        val ageMs = if (ordered) {
            (frameTimestampNs - resultTimestampNs) / 1_000_000L
        } else {
            Long.MAX_VALUE
        }
        val record = ordered && resultTimestampNs > lastRecordedTimestampNs
        if (record) lastRecordedTimestampNs = resultTimestampNs
        return TemporalResultUsage(
            preview = ordered && ageMs <= previewMaxAgeMs.coerceAtLeast(0L),
            record = record
        )
    }

    fun reset() {
        lastRecordedTimestampNs = 0L
    }
}
