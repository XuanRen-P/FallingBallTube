package com.cupet.fallingball.analysis

/**
 * Low-saturation label a user can attach to a saved measurement.
 *
 * The label is deliberately stored beside the immutable `.fbr` record (see
 * [MeasurementHistoryStore.setMark]) instead of inside it: measurement data stays a
 * write-once physical artifact, while the label is a mutable, purely presentational
 * user preference.
 */
enum class HistoryMarkColor {
    BLUE,
    GREEN,
    YELLOW,
    PINK,
    PURPLE,
    ORANGE;

    companion object {
        fun fromName(value: String?): HistoryMarkColor? =
            entries.firstOrNull { it.name.equals(value?.trim(), ignoreCase = true) }
    }
}
