package com.cupet.fallingball.camera

/** Serial video scheduling follows presentation time, independently of decode speed and FPS. */
internal class OfflineTubeRefreshPolicy(private val refreshIntervalNs: Long = 250_000_000L) {
    private var lastAttemptTimestampNs: Long? = null

    init {
        require(refreshIntervalNs >= 0L)
    }

    fun shouldRefresh(
        timestampNs: Long,
        hasTrackedAxis: Boolean,
        lastDetectionTimestampNs: Long,
        maxTrackedAxisAgeNs: Long
    ): Boolean {
        require(maxTrackedAxisAgeNs > 0L)
        val previous = lastAttemptTimestampNs
        val expired = lastDetectionTimestampNs <= 0L ||
            timestampNs - lastDetectionTimestampNs > maxTrackedAxisAgeNs
        val due = previous == null || timestampNs <= previous ||
            timestampNs - previous >= refreshIntervalNs
        // An existing axis may remain cached after its TTL expires. Re-detect on this same
        // frame before the shared usable-axis gate can reject it, including large VFR gaps.
        val refresh = !hasTrackedAxis || expired || due
        if (refresh) lastAttemptTimestampNs = timestampNs
        return refresh
    }
}
