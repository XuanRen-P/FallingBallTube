package com.cupet.fallingball.ui

enum class AnalysisResolution(val label: String, val width: Int, val height: Int) {
    P720("720p", 720, 1280),
    P1080("1080p", 1080, 1920);

    companion object {
        fun fromButtonId(is1080: Boolean): AnalysisResolution = if (is1080) P1080 else P720

        fun fromStoredName(value: String?): AnalysisResolution =
            entries.firstOrNull { it.name == value } ?: P720
    }
}
