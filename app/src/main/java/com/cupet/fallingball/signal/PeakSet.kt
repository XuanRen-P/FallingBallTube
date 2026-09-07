package com.cupet.fallingball.signal

/**
 * Reusable container for peak-finding output, laid out as parallel arrays.
 *
 * Detection runs on every frame, so this is designed to be allocated once and refilled:
 * [reset] keeps the backing arrays, [ensureCapacity] only grows them. Reading any index
 * `>= count` is meaningless -- stale values from an earlier frame are deliberately left
 * in place rather than cleared.
 */
class PeakSet(initialCapacity: Int = 256) {

    var indices = IntArray(initialCapacity); private set
    var prominences = FloatArray(initialCapacity); private set
    var leftBases = IntArray(initialCapacity); private set
    var rightBases = IntArray(initialCapacity); private set
    var widths = FloatArray(initialCapacity); private set
    var widthHeights = FloatArray(initialCapacity); private set
    var leftIps = FloatArray(initialCapacity); private set
    var rightIps = FloatArray(initialCapacity); private set

    var count: Int = 0; private set

    fun reset() {
        count = 0
    }

    fun ensureCapacity(n: Int) {
        if (indices.size >= n) return
        var cap = indices.size.coerceAtLeast(16)
        while (cap < n) cap *= 2
        indices = indices.copyOf(cap)
        prominences = prominences.copyOf(cap)
        leftBases = leftBases.copyOf(cap)
        rightBases = rightBases.copyOf(cap)
        widths = widths.copyOf(cap)
        widthHeights = widthHeights.copyOf(cap)
        leftIps = leftIps.copyOf(cap)
        rightIps = rightIps.copyOf(cap)
    }

    /** Appends a peak index; prominence/width fields are filled in later passes. */
    fun add(index: Int) {
        ensureCapacity(count + 1)
        indices[count] = index
        count++
    }

    /** Index of the highest-prominence peak, or -1 when empty. */
    fun argMaxProminence(): Int {
        var best = -1
        var bestVal = Float.NEGATIVE_INFINITY
        for (i in 0 until count) {
            if (prominences[i] > bestVal) {
                bestVal = prominences[i]
                best = i
            }
        }
        return best
    }

    fun maxProminence(): Float {
        val i = argMaxProminence()
        return if (i < 0) 0f else prominences[i]
    }
}
