package com.cupet.fallingball.vision

/** A stabilized camera frame retained as causal context for the next model input. */
data class TinyHeatmapHistoryFrame(
    val bytes: ByteArray,
    val geometry: TubeRoiGeometry,
    val timestampNs: Long,
    val sourceFrameIndex: Int
)

/** Keeps the exact preceding camera frames needed by TinyHeatmapNet's lag-1/lag-3 channels. */
class TinyHeatmapInputHistory(
    private val capacity: Int = 3
) {
    private val frames = ArrayDeque<TinyHeatmapHistoryFrame>(capacity)

    init {
        require(capacity > 0) { "History capacity must be positive" }
    }

    fun reset() {
        frames.clear()
    }

    /** Appends the current frame and returns a snapshot of its preceding frames. */
    fun appendAndSnapshot(frame: TinyHeatmapHistoryFrame): List<TinyHeatmapHistoryFrame> {
        val previous = frames.toList()
        frames.addLast(frame)
        while (frames.size > capacity) frames.removeFirst()
        return previous
    }
}
