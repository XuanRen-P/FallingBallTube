package com.cupet.fallingball.vision

import java.nio.ByteBuffer

data class TemporalSequenceStatus(
    val frameCount: Int,
    val sequenceLength: Int,
    val inputFps: Float,
    val lastGapMs: Long,
    val discontinuityCount: Int
)

/** Preallocated oldest-to-current model-step window shared by camera and model workers. */
class TemporalRoiSequenceBuffer(
    private val sequenceLength: Int = 8,
    private val pixelsPerFrame: Int = TubeRoiSampler.OUTPUT_WIDTH * TubeRoiSampler.OUTPUT_HEIGHT,
    private val maxFrameGapNs: Long = MAX_FRAME_GAP_NS
) {
    private val frames = Array(sequenceLength) { ByteArray(pixelsPerFrame) }
    private var frameCount = 0
    private var nextSlot = 0
    private var lastTimestampNs = 0L
    private var lastObservedTimestampNs = 0L
    private var inputFps = 0f
    private var lastGapMs = 0L
    private var discontinuityCount = 0

    init {
        require(sequenceLength > 0) { "Sequence length must be positive" }
        require(pixelsPerFrame > 0) { "Frame size must be positive" }
        require(maxFrameGapNs > 0L) { "Maximum frame gap must be positive" }
    }

    val isFull: Boolean get() = frameCount == sequenceLength

    @Synchronized
    fun reset() {
        clearWindow()
        lastObservedTimestampNs = 0L
        inputFps = 0f
        lastGapMs = 0L
        discontinuityCount = 0
    }

    @Synchronized
    fun status(): TemporalSequenceStatus = TemporalSequenceStatus(
        frameCount = frameCount,
        sequenceLength = sequenceLength,
        inputFps = inputFps,
        lastGapMs = lastGapMs,
        discontinuityCount = discontinuityCount
    )

    private fun clearWindow() {
        frameCount = 0
        nextSlot = 0
        lastTimestampNs = 0L
    }

    /** Records camera continuity even when the slower recurrent worker skips this frame. */
    @Synchronized
    fun observe(timestampNs: Long): Boolean {
        require(timestampNs > 0L) { "Frame timestamp must be positive" }
        val gapNs = if (lastObservedTimestampNs == 0L) {
            0L
        } else {
            timestampNs - lastObservedTimestampNs
        }
        val discontinuity = lastObservedTimestampNs != 0L &&
            (gapNs <= 0L || gapNs > maxFrameGapNs)
        if (lastObservedTimestampNs != 0L && gapNs > 0L) {
            lastGapMs = gapNs / 1_000_000L
            if (!discontinuity) {
                val instantaneousFps = 1_000_000_000f / gapNs
                inputFps = if (inputFps <= 0f) {
                    instantaneousFps
                } else {
                    inputFps * 0.8f + instantaneousFps * 0.2f
                }
            }
        }
        if (discontinuity) {
            discontinuityCount += 1
            inputFps = 0f
            clearWindow()
        }
        lastObservedTimestampNs = timestampNs
        return discontinuity
    }

    /** Returns true when accepted model steps are too far apart to share recurrent state. */
    @Synchronized
    fun append(frame: ByteBuffer, timestampNs: Long): Boolean {
        require(frame.remaining() >= pixelsPerFrame) {
            "Expected $pixelsPerFrame ROI bytes, got ${frame.remaining()}"
        }
        val cameraDiscontinuity = if (timestampNs == lastObservedTimestampNs) {
            false
        } else {
            observe(timestampNs)
        }
        if (lastTimestampNs != 0L) {
            val acceptedGapNs = timestampNs - lastTimestampNs
            lastGapMs = if (acceptedGapNs > 0L) acceptedGapNs / 1_000_000L else 0L
        }
        val modelOrderDiscontinuity = lastTimestampNs != 0L && timestampNs <= lastTimestampNs
        if (modelOrderDiscontinuity) {
            discontinuityCount += 1
            clearWindow()
        }
        frame.duplicate().get(frames[nextSlot])
        nextSlot = (nextSlot + 1) % sequenceLength
        frameCount = (frameCount + 1).coerceAtMost(sequenceLength)
        lastTimestampNs = timestampNs
        return cameraDiscontinuity || modelOrderDiscontinuity
    }

    /** Copies a stable inference snapshot only when the worker is ready to consume it. */
    @Synchronized
    fun copyLatestTo(output: ByteBuffer) {
        check(isFull) { "A complete temporal window is not available" }
        require(output.capacity() >= sequenceLength * pixelsPerFrame) {
            "Sequence output buffer is too small"
        }
        output.clear()
        for (time in 0 until sequenceLength) {
            output.put(frames[(nextSlot + time) % sequenceLength])
        }
        output.flip()
    }

    companion object {
        /** Keep mobile scheduling stalls, but never join windows across a real capture pause. */
        const val MAX_FRAME_GAP_NS = 2_000_000_000L
    }
}
