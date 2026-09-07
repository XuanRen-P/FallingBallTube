package com.cupet.fallingball.vision

/** Exact local-maximum top-k selection, with the original ascending-index tie ordering. */
internal fun selectHeatmapLocalMaxima(
    scores: FloatArray,
    width: Int,
    height: Int,
    radius: Int,
    selectedIndices: IntArray,
    selectedScores: FloatArray
) {
    require(width > 0 && height > 0 && scores.size == width * height)
    require(radius >= 0 && selectedIndices.isNotEmpty() && selectedIndices.size == selectedScores.size)
    selectedIndices.fill(-1)
    selectedScores.fill(-1f)
    val lastSlot = selectedIndices.lastIndex
    var selectedNaNs = 0
    for (index in scores.indices) {
        val score = scores[index]
        val lastIndex = selectedIndices[lastSlot]
        val cutoff = selectedScores[lastSlot]
        // The list contains only confirmed local maxima. Once full, a candidate below its
        // weakest member cannot enter top-k, even if its entire neighbourhood is lower.
        // Equal scores prefer the earlier index. Keep the old comparison behaviour for NaN
        // model output, for which the original insertion ordering is not a total order.
        if (lastIndex >= 0 && selectedNaNs == 0 &&
            (score < cutoff || score == cutoff && index > lastIndex)) continue

        val y = index / width
        val x = index % width
        var localMaximum = true
        loop@ for (nearY in maxOf(0, y - radius)..minOf(height - 1, y + radius)) {
            for (nearX in maxOf(0, x - radius)..minOf(width - 1, x + radius)) {
                if (scores[nearY * width + nearX] > score) {
                    localMaximum = false
                    break@loop
                }
            }
        }
        if (!localMaximum) continue

        var insertion = 0
        while (insertion < selectedScores.size &&
            (score < selectedScores[insertion] ||
                score == selectedScores[insertion] && index > selectedIndices[insertion])) {
            insertion++
        }
        if (insertion >= selectedScores.size) continue
        if (selectedScores[lastSlot].isNaN()) selectedNaNs--
        if (score.isNaN()) selectedNaNs++
        for (slot in lastSlot downTo insertion + 1) {
            selectedScores[slot] = selectedScores[slot - 1]
            selectedIndices[slot] = selectedIndices[slot - 1]
        }
        selectedScores[insertion] = score
        selectedIndices[insertion] = index
    }
}
