package com.cupet.fallingball.vision

import android.util.Log

/**
 * Pins the whole process to the device's fast CPU cores.
 *
 * Measured win on the SM8650 reference device: the scheduler parked inference
 * threads on the little cluster under load, inflating parent inference ~10%.
 * Pinning every thread to the >= 85% max-frequency cores restored most of it
 * (parent p50 82-86 ms -> 74.7 ms, effective FPS 7.05 -> 7.66).  Core
 * selection is derived from cpufreq per device, so other SoCs adapt
 * automatically.  Failures are non-fatal and logged; behavior without the pin
 * matches the previous release.
 */
internal object CpuAffinity {
    private const val TAG = "TinyHeatmapNet"

    @Volatile
    private var attempted = false

    /** Returns the number of fast cores pinned to, or 0 when unavailable. */
    @Synchronized
    fun pinProcessToFastCores(): Int {
        if (attempted) return 0
        attempted = true
        return try {
            System.loadLibrary("fallingball_native")
            val fastCores = nativePinProcessToFastCores()
            if (fastCores > 0) {
                Log.i(TAG, "CpuAffinity: process pinned to $fastCores fast cores")
            }
            fastCores
        } catch (error: Throwable) {
            Log.w(TAG, "CpuAffinity unavailable; default scheduler affinity kept", error)
            0
        }
    }

    private external fun nativePinProcessToFastCores(): Int
}
