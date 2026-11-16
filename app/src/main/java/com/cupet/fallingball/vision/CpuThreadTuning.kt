package com.cupet.fallingball.vision

/** Pure selection policy shared by the on-device thread benchmark and JVM tests. */
object CpuThreadTuning {

    data class Benchmark(val threads: Int, val medianMs: Float)

    /**
     * Zero represents ONNX Runtime's automatic pool. The explicit values cover small,
     * medium, and all-core pools without benchmarking every integer on larger devices.
     *
     * [maxThreads] caps the pool on devices whose whole fast cluster is four cores or fewer.
     * When the budget is binding, the automatic pool (0) is dropped: ORT's auto sizing would
     * take every core back and undo the cap.
     */
    fun candidates(availableProcessors: Int, maxThreads: Int = Int.MAX_VALUE): List<Int> {
        val cores = availableProcessors.coerceAtLeast(1)
        val limit = maxThreads.coerceIn(1, cores)
        val explicit = buildList {
            if (cores == 1) add(1) else listOf(2, 3, 4, 6, 8).filterTo(this) { it <= limit }
            if (limit !in this) add(limit)
        }
        return buildList {
            if (limit == cores) add(0)
            addAll(explicit)
        }.distinct()
    }

    /** Selects the lowest valid median; ties keep the candidate measured first. */
    fun select(results: List<Benchmark>): Int? {
        val valid = results.filter { it.medianMs.isFinite() && it.medianMs > 0f }
        return valid.minByOrNull { it.medianMs }?.threads
    }
}
