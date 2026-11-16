package com.cupet.fallingball.camera

import java.util.Locale

/**
 * Separates "the sensor is slow" from "our analyzer is slow" on the 1 Hz diagnostics line.
 *
 * CameraX hands the analyzer one ImageProxy at a time and [com.cupet.fallingball.vision.AnalysisFrame]
 * keeps it borrowed until the frame is fully processed. The analyzer's own cadence therefore bounds
 * the arrival rate, so a 49 ms frame period on its own cannot tell a 20 fps sensor from a 30 fps
 * sensor feeding a 49 ms analyzer. Three measurements separate the cases:
 *
 * - `analyzer`: wall time per frame. Equal to the frame period means we are the limit.
 * - `idle`: gap between the previous frame's end and this frame's start. Near zero means a frame
 *   was already waiting when the analyzer became free (the camera keeps up); a non-trivial idle
 *   gap is the signature of a sensor-side limit.
 * - `dt`: raw ImageProxy timestamp deltas. A 30 fps sensor in front of a ~40 ms analyzer alternates
 *   33/67 ms; a genuine 20 fps sensor gives a flat ~50 ms; a 60 fps sensor gives multiples of 16.7.
 *
 * The applied AE range and sensor frame duration come from capture results, because the request
 * we send is not necessarily the one the pipeline runs (stabilization and vendor overrides can
 * rewrite it).
 */
class CameraRateDiagnostics(private val deltaCapacity: Int = 8) {
    private val deltas = LongArray(deltaCapacity)
    private var deltaCount = 0
    private var deltaIndex = 0
    private var analyzerMs = UNSET
    private var idleMs = UNSET
    private var previousEndNs = 0L
    private var previousTimestampNs = 0L
    private var aeMin = 0
    private var aeMax = 0
    private var frameDurationNs = 0L

    fun onFrameStart(timestampNs: Long, nowNs: Long) {
        if (previousEndNs != 0L && nowNs > previousEndNs) {
            idleMs = ema(idleMs, (nowNs - previousEndNs) / NANOS_PER_MILLISECOND)
        }
        if (previousTimestampNs != 0L && timestampNs > previousTimestampNs) {
            deltas[deltaIndex] = timestampNs - previousTimestampNs
            deltaIndex = (deltaIndex + 1) % deltaCapacity
            if (deltaCount < deltaCapacity) deltaCount += 1
        }
        previousTimestampNs = timestampNs
    }

    fun onFrameEnd(durationNs: Long, nowNs: Long) {
        if (durationNs > 0L) analyzerMs = ema(analyzerMs, durationNs / NANOS_PER_MILLISECOND)
        previousEndNs = nowNs
    }

    fun onCaptureResult(aeMin: Int?, aeMax: Int?, frameDurationNs: Long?) {
        if (aeMin != null && aeMax != null && aeMax > 0) {
            this.aeMin = aeMin
            this.aeMax = aeMax
        }
        if (frameDurationNs != null && frameDurationNs > 0L) {
            this.frameDurationNs = frameDurationNs
        }
    }

    fun reset() {
        deltaCount = 0
        deltaIndex = 0
        analyzerMs = UNSET
        idleMs = UNSET
        previousEndNs = 0L
        previousTimestampNs = 0L
    }

    /** One-line summary; every field prints `none` until it has been measured. */
    fun describe(): String {
        val ae = if (aeMax > 0) "$aeMin-$aeMax" else "none"
        val frameDuration = if (frameDurationNs > 0L) {
            formatMillis(frameDurationNs / NANOS_PER_MILLISECOND)
        } else {
            "none"
        }
        return "analyzer=${formatMillis(analyzerMs)} idle=${formatMillis(idleMs)} " +
            "dt=${describeDeltas()} ae=$ae frameDur=$frameDuration"
    }

    private fun describeDeltas(): String {
        if (deltaCount == 0) return "none"
        val builder = StringBuilder()
        for (i in 0 until deltaCount) {
            val slot = (deltaIndex - deltaCount + i + deltaCapacity * 2) % deltaCapacity
            if (builder.isNotEmpty()) builder.append(',')
            builder.append((deltas[slot] / NANOS_PER_MILLISECOND).toInt())
        }
        return builder.toString()
    }

    private fun formatMillis(value: Float): String =
        if (value < 0f) "none" else format(value) + "ms"

    private fun format(value: Float): String = String.format(Locale.ROOT, "%.1f", value)

    private fun ema(previous: Float, sample: Float): Float =
        if (previous < 0f) sample else previous * (1f - SMOOTHING) + sample * SMOOTHING

    private companion object {
        const val NANOS_PER_MILLISECOND = 1_000_000f
        const val SMOOTHING = 0.2f
        const val UNSET = -1f
    }
}
