package com.cupet.fallingball.camera

import kotlin.math.abs
import kotlin.math.max

/**
 * Confirms a manual-focus request from capture feedback, including approximate lenses
 * whose reported actuator position can differ from the requested diopters. Such lenses
 * must actually respond in the commanded direction and then report stationary frames;
 * merely seeing the request tag or AF_OFF is insufficient.
 */
internal class ManualFocusConfirmation(
    private val version: Long,
    private val expectedDistance: Float,
    private val baselineDistance: Float?,
    private val previousRequestDistance: Float?,
    nearestDistance: Float,
    private val calibrated: Boolean,
    private val previousConfirmedDistance: Float? = null
) {
    private val distanceTolerance = max(0.12f, nearestDistance * 0.025f)
    private val stabilityTolerance = max(0.005f, nearestDistance * 0.0025f)
    // Entering MF rounds the observed AF distance to one of the slider's 101 positions.
    private val autofocusHoldTolerance = max(0.00001f, nearestDistance * 0.0051f)
    private var stationaryCount = 0
    private var stationaryAnchor: Float? = null

    fun observe(
        requestVersion: Long,
        manualModeConfirmed: Boolean,
        reportedDistance: Float?,
        stationary: Boolean,
        moving: Boolean
    ): Boolean {
        if (requestVersion != version) return false
        if (!manualModeConfirmed || reportedDistance == null || !reportedDistance.isFinite() ||
            reportedDistance < 0f || moving) {
            resetSettling()
            return false
        }
        if (abs(expectedDistance - reportedDistance) <= distanceTolerance) return true
        if (calibrated || !stationary) {
            resetSettling()
            return false
        }
        val baseline = baselineDistance?.takeIf { it.isFinite() && it >= 0f } ?: return false
        val actualChange = reportedDistance - baseline
        val commandedChange = expectedDistance - (previousRequestDistance ?: baseline)
        val holdingAutofocusPosition = previousRequestDistance == null && abs(commandedChange) <= autofocusHoldTolerance
        val movedInRequestedDirection = abs(actualChange) > stabilityTolerance &&
            (holdingAutofocusPosition || actualChange * commandedChange > 0f)
        // Reapplying a confirmed target (including failure rollback) need not move the
        // actuator again, but its capture feedback must match that confirmed position.
        val confirmedTargetRepeated = previousRequestDistance != null && previousConfirmedDistance != null &&
            abs(expectedDistance - previousRequestDistance) <= 0.00001f &&
            abs(reportedDistance - previousConfirmedDistance) <= stabilityTolerance
        if (!movedInRequestedDirection && !confirmedTargetRepeated) {
            resetSettling()
            return false
        }
        val anchor = stationaryAnchor
        if (anchor == null || abs(anchor - reportedDistance) > stabilityTolerance) {
            stationaryAnchor = reportedDistance
            stationaryCount = 1
        } else stationaryCount++
        return stationaryCount >= 3
    }

    private fun resetSettling() {
        stationaryCount = 0
        stationaryAnchor = null
    }
}
