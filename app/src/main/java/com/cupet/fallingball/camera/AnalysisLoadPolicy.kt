package com.cupet.fallingball.camera

import java.util.concurrent.atomic.AtomicLong
import kotlin.math.ceil

/** Platform-neutral thermal signal, mapped from Android's PowerManager by the adapter. */
internal enum class ThermalSignal {
    NORMAL,
    LIGHT,
    MODERATE,
    SEVERE,
    CRITICAL
}

internal enum class AnalysisLoadLevel {
    NORMAL,
    WARM,
    HOT,
    CRITICAL
}

/** Runtime cadence limits. Native-resolution ruler sampling is intentionally not throttled. */
internal data class AnalysisLoadProfile(
    val level: AnalysisLoadLevel,
    val yoloIntervalNs: Long,
    val yoloCooldownNs: Long,
    val temporalCooldownNs: Long,
    val maxTrackedAxisAgeNs: Long
) {
    val yoloFpsLimit: Float get() = 1_000_000_000f / yoloIntervalNs
    val trackerLossTimeoutMs: Long
        get() = maxTrackedAxisAgeNs / 1_000_000L + TRACKER_RETENTION_MARGIN_MS
    val expectedTemporalPeriodMs: Long
        get() = withSchedulingMargin(
            BASE_TEMPORAL_INFERENCE_MS + temporalCooldownNs / 1_000_000L
        )
    fun expectedTemporalPeriodMs(observedModelFps: Float): Long {
        val observedPeriodMs = observedModelFps
            .takeIf { it.isFinite() && it > 0f }
            ?.let { ceil(1_000f / it).toLong() }
            ?: 0L
        return maxOf(expectedTemporalPeriodMs, withSchedulingMargin(observedPeriodMs))
    }
    val serializeModelInference: Boolean
        get() = level != AnalysisLoadLevel.NORMAL

    companion object {
        private const val TRACKER_RETENTION_MARGIN_MS = 500L
        private const val BASE_TEMPORAL_INFERENCE_MS = 500L
        private const val TEMPORAL_SCHEDULING_MARGIN = 1.5

        private fun withSchedulingMargin(periodMs: Long): Long =
            ceil(periodMs.coerceAtLeast(0L) * TEMPORAL_SCHEDULING_MARGIN).toLong()

        val NORMAL = AnalysisLoadProfile(
            level = AnalysisLoadLevel.NORMAL,
            yoloIntervalNs = 200_000_000L,
            yoloCooldownNs = 0L,
            temporalCooldownNs = 0L,
            maxTrackedAxisAgeNs = 1_500_000_000L
        )
        val WARM = AnalysisLoadProfile(
            level = AnalysisLoadLevel.WARM,
            yoloIntervalNs = 333_000_000L,
            yoloCooldownNs = 200_000_000L,
            temporalCooldownNs = 200_000_000L,
            maxTrackedAxisAgeNs = 2_000_000_000L
        )
        val HOT = AnalysisLoadProfile(
            level = AnalysisLoadLevel.HOT,
            yoloIntervalNs = 500_000_000L,
            yoloCooldownNs = 500_000_000L,
            temporalCooldownNs = 500_000_000L,
            maxTrackedAxisAgeNs = 3_000_000_000L
        )
        val CRITICAL = AnalysisLoadProfile(
            level = AnalysisLoadLevel.CRITICAL,
            yoloIntervalNs = 1_000_000_000L,
            yoloCooldownNs = 1_000_000_000L,
            temporalCooldownNs = 800_000_000L,
            maxTrackedAxisAgeNs = 5_000_000_000L
        )

        fun forLevel(level: AnalysisLoadLevel): AnalysisLoadProfile = when (level) {
            AnalysisLoadLevel.NORMAL -> NORMAL
            AnalysisLoadLevel.WARM -> WARM
            AnalysisLoadLevel.HOT -> HOT
            AnalysisLoadLevel.CRITICAL -> CRITICAL
        }
    }
}

/** Thread-safe cooldown measured from inference completion, not submission. */
internal class InferenceCompletionThrottle {
    private val lastCompletionNs = AtomicLong(0L)

    fun canStart(nowNs: Long, cooldownNs: Long): Boolean {
        return remainingCooldownNs(nowNs, cooldownNs) == 0L
    }

    fun remainingCooldownNs(nowNs: Long, cooldownNs: Long): Long {
        if (cooldownNs <= 0L) return 0L
        val completedNs = lastCompletionNs.get()
        if (completedNs == 0L) return 0L
        return (cooldownNs - (nowNs - completedNs)).coerceAtLeast(0L)
    }

    fun onCompleted(nowNs: Long) {
        if (nowNs > 0L) lastCompletionNs.accumulateAndGet(nowNs, ::maxOf)
    }
}

/**
 * Combines Android's thermal status with battery temperature. Rising heat sheds load
 * immediately; cooling uses a small hysteresis so workers do not oscillate near a boundary.
 */
internal class AnalysisLoadPolicy {
    private var currentLevel = AnalysisLoadLevel.NORMAL

    @Synchronized
    fun update(signal: ThermalSignal, batteryTemperatureC: Float?): AnalysisLoadProfile {
        val signalLevel = when (signal) {
            ThermalSignal.NORMAL,
            ThermalSignal.LIGHT -> AnalysisLoadLevel.NORMAL
            ThermalSignal.MODERATE -> AnalysisLoadLevel.WARM
            ThermalSignal.SEVERE -> AnalysisLoadLevel.HOT
            ThermalSignal.CRITICAL -> AnalysisLoadLevel.CRITICAL
        }
        val batteryLevel = temperatureLevel(batteryTemperatureC)
        currentLevel = maxOf(signalLevel, batteryLevel)
        return AnalysisLoadProfile.forLevel(currentLevel)
    }

    private fun temperatureLevel(temperatureC: Float?): AnalysisLoadLevel {
        if (temperatureC == null || !temperatureC.isFinite()) return AnalysisLoadLevel.NORMAL
        return when {
            temperatureC >= 45f -> AnalysisLoadLevel.CRITICAL
            currentLevel == AnalysisLoadLevel.CRITICAL && temperatureC >= 43f ->
                AnalysisLoadLevel.CRITICAL
            temperatureC >= 42f -> AnalysisLoadLevel.HOT
            currentLevel >= AnalysisLoadLevel.HOT && temperatureC >= 40f ->
                AnalysisLoadLevel.HOT
            temperatureC >= 38.5f -> AnalysisLoadLevel.WARM
            currentLevel >= AnalysisLoadLevel.WARM && temperatureC >= 37f ->
                AnalysisLoadLevel.WARM
            else -> AnalysisLoadLevel.NORMAL
        }
    }
}
