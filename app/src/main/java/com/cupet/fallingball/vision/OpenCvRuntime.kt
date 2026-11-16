package com.cupet.fallingball.vision

import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core

/** Loads the bundled OpenCV runtime exactly once before any native Mat is created. */
object OpenCvRuntime {
    private const val TAG = "TinyHeatmapNet"

    @Volatile
    private var loaded = false

    fun ensureLoaded() {
        if (loaded) return
        synchronized(this) {
            if (loaded) return
            check(OpenCVLoader.initLocal()) { "OpenCV 4.12.0 initialization failed" }
            capWorkerThreads()
            loaded = true
        }
    }

    /**
     * OpenCV keeps its own worker pool and is not covered by the ORT thread budget.
     *
     * Every blur and warp in the camera phase can fan out to one worker per core, which on a
     * four-fast-core phone is the third pipeline competing for the same cluster (Dimensity
     * 1000+, 2026-11-15).  Half the fast cluster is enough for these small 152x1024 planes;
     * wide-cluster devices keep OpenCV's default.
     */
    private fun capWorkerThreads() {
        val fastCores = CpuAffinity.fastCoreCount
        if (fastCores !in 1..4) return
        val threads = (fastCores / 2).coerceAtLeast(1)
        Core.setNumThreads(threads)
        Log.i(TAG, "OpenCV: worker threads capped to $threads of $fastCores fast cores")
    }
}
