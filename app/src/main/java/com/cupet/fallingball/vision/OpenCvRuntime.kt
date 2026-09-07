package com.cupet.fallingball.vision

import org.opencv.android.OpenCVLoader

/** Loads the bundled OpenCV runtime exactly once before any native Mat is created. */
object OpenCvRuntime {
    @Volatile
    private var loaded = false

    fun ensureLoaded() {
        if (loaded) return
        synchronized(this) {
            if (loaded) return
            check(OpenCVLoader.initLocal()) { "OpenCV 4.12.0 initialization failed" }
            loaded = true
        }
    }
}
