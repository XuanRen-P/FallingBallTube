package com.cupet.fallingball.ui

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.cupet.fallingball.vision.CameraLevelEstimator
import com.cupet.fallingball.vision.CameraLevelReading

/**
 * Android sensor adapter for the camera-level indicator.
 *
 * Gravity is preferred because it excludes most hand acceleration. Devices without that
 * virtual sensor fall back to the accelerometer; [CameraLevelEstimator] smooths either
 * source before the UI sees it.
 */
class DeviceLevelMonitor(
    context: Context,
    private val onReading: (CameraLevelReading?) -> Unit
) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val sensor = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
        ?: sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val estimator = CameraLevelEstimator()
    private var registered = false

    fun start(): Boolean {
        if (registered) return true
        val source = sensor ?: return false
        estimator.reset()
        registered = sensorManager.registerListener(
            this,
            source,
            SensorManager.SENSOR_DELAY_UI
        )
        return registered
    }

    fun stop() {
        if (registered) sensorManager.unregisterListener(this)
        registered = false
        estimator.reset()
        onReading(null)
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.values.size < 3) return
        onReading(
            estimator.update(
                gravityX = event.values[0],
                gravityY = event.values[1],
                gravityZ = event.values[2]
            )
        )
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
}
