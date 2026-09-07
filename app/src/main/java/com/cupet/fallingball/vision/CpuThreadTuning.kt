package com.cupet.fallingball.vision

/** Pure selection policy shared by the on-device thread benchmark and JVM tests. */
object CpuThreadTuning {

    data class Benchmark(val threads: Int, val medianMs: Float)

    /**
     * Zero represents ONNX Runtime's automatic pool. The explicit values cover small,
     * medium, and all-core pools without benchmarking every integer on larger devices.
     */
    fun candidates(availableProcessors: Int): List<Int> {
        val cores = availableProcessors.coerceAtLeast(1)
        val explicit = if (cores == 1) listOf(1) else listOf(2, 3, 4, 6, 8)
        return buildList {
            add(0)
            explicit.filterTo(this) { it <= cores }
            if (cores !in this) add(cores)
        }.distinct()
    }

    /** Selects the lowest valid median; ties keep the candidate measured first. */
    fun select(results: List<Benchmark>): Int? {
        val valid = results.filter { it.medianMs.isFinite() && it.medianMs > 0f }
        return valid.minByOrNull { it.medianMs }?.threads
    }
}
