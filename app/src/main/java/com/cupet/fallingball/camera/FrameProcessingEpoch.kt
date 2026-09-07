package com.cupet.fallingball.camera

/**
 * Owns frame admission, capture-clock changes and short result-publication transactions.
 * Inference and frame sampling must stay outside this lock so camera rebinding never waits on them.
 */
internal class FrameProcessingEpoch {
    @Volatile private var binding = 0L
    @Volatile private var work = 0L
    @Volatile var latestCaptureTimestampNs = 0L
        private set

    fun isCurrentBinding(token: Long): Boolean = token == binding
    fun isCurrentWork(token: Long): Boolean = token == work

    @Synchronized
    fun replaceBinding(resetPublishedState: () -> Unit = {}): Long {
        binding++
        work++
        latestCaptureTimestampNs = 0L
        resetPublishedState()
        return binding
    }

    @Synchronized
    fun invalidateWork(resetPublishedState: () -> Unit = {}) {
        work++
        resetPublishedState()
    }

    /** A null binding token is reserved for the serial offline source. */
    @Synchronized
    fun enterFrame(bindingToken: Long?, timestampNs: Long): Long? {
        if (bindingToken != null && bindingToken != binding) return null
        latestCaptureTimestampNs = maxOf(latestCaptureTimestampNs, timestampNs)
        return work
    }

    @Synchronized
    fun publishIfCurrent(workToken: Long, publish: () -> Unit): Boolean {
        if (workToken != work) return false
        publish()
        return true
    }
}
