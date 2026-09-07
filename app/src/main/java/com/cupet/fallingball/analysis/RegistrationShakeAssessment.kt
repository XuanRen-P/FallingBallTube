package com.cupet.fallingball.analysis

internal object RegistrationShakeAssessment {
    const val LARGE_SHIFT_THRESHOLD_PX = 12f
    const val REQUIRED_CONSECUTIVE_SAMPLES = 3

    fun shouldWarn(shiftsPx: Iterable<Float>): Boolean {
        var consecutiveLargeShifts = 0
        for (shiftPx in shiftsPx) {
            consecutiveLargeShifts = if (
                shiftPx.isFinite() &&
                kotlin.math.abs(shiftPx) > LARGE_SHIFT_THRESHOLD_PX
            ) {
                consecutiveLargeShifts + 1
            } else {
                0
            }
            if (consecutiveLargeShifts >= REQUIRED_CONSECUTIVE_SAMPLES) return true
        }
        return false
    }
}
