package com.cupet.fallingball.camera

internal data class ObbFrameDecision(
    val submitDetection: Boolean,
    val trackedAxisUsable: Boolean
)

/** Keeps slow tube detection scheduling out of the high-rate ball path. */
internal class ObbRefreshPolicy(
    private val defaultRefreshIntervalNs: Long = DEFAULT_REFRESH_INTERVAL_NS,
    private val defaultMaxTrackedAxisAgeNs: Long = DEFAULT_MAX_TRACKED_AXIS_AGE_NS
) {
    private var lastSubmissionTimestampNs = 0L

    init {
        require(defaultRefreshIntervalNs > 0L)
        require(defaultMaxTrackedAxisAgeNs >= defaultRefreshIntervalNs)
    }

    @Synchronized
    fun decide(
        frameTimestampNs: Long,
        lastDetectionTimestampNs: Long,
        workerBusy: Boolean,
        hasTrackedAxis: Boolean,
        refreshIntervalNs: Long = defaultRefreshIntervalNs,
        maxTrackedAxisAgeNs: Long = defaultMaxTrackedAxisAgeNs
    ): ObbFrameDecision {
        require(refreshIntervalNs > 0L)
        require(maxTrackedAxisAgeNs >= refreshIntervalNs)
        val ageNs = if (lastDetectionTimestampNs <= 0L) {
            Long.MAX_VALUE
        } else {
            (frameTimestampNs - lastDetectionTimestampNs).coerceAtLeast(0L)
        }
        val submissionAgeNs = if (lastSubmissionTimestampNs <= 0L) {
            Long.MAX_VALUE
        } else {
            (frameTimestampNs - lastSubmissionTimestampNs).coerceAtLeast(0L)
        }
        return ObbFrameDecision(
            submitDetection = submissionAgeNs >= refreshIntervalNs && !workerBusy,
            trackedAxisUsable = hasTrackedAxis && ageNs <= maxTrackedAxisAgeNs
        )
    }

    @Synchronized
    fun onDetectionSubmitted(frameTimestampNs: Long) {
        require(frameTimestampNs > 0L)
        lastSubmissionTimestampNs = frameTimestampNs
    }

    @Synchronized
    fun reset() {
        lastSubmissionTimestampNs = 0L
    }

    companion object {
        const val DEFAULT_REFRESH_INTERVAL_NS = 200_000_000L
        const val DEFAULT_MAX_TRACKED_AXIS_AGE_NS = 750_000_000L
    }
}
