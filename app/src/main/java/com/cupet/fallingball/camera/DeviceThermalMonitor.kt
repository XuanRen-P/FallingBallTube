package com.cupet.fallingball.camera

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import androidx.core.content.ContextCompat

internal data class DeviceThermalSnapshot(
    val signal: ThermalSignal,
    val batteryTemperatureC: Float?,
    val profile: AnalysisLoadProfile
)

/** Android adapter for [AnalysisLoadPolicy]. No values are rendered in the camera UI. */
internal class DeviceThermalMonitor(
    context: Context,
    private val onSnapshot: (DeviceThermalSnapshot) -> Unit
) : AutoCloseable {
    private val appContext = context.applicationContext
    private val powerManager = appContext.getSystemService(PowerManager::class.java)
    private val policy = AnalysisLoadPolicy()
    private var thermalSignal = ThermalSignal.NORMAL
    private var batteryTemperatureC: Float? = null
    private var started = false

    private val batteryReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action != Intent.ACTION_BATTERY_CHANGED) return
            val tenthsC = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            batteryTemperatureC = tenthsC.takeIf { it != Int.MIN_VALUE }?.div(10f)
            publish()
        }
    }

    private val thermalListener = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        PowerManager.OnThermalStatusChangedListener { status ->
            thermalSignal = mapThermalStatus(status)
            publish()
        }
    } else {
        null
    }

    @Synchronized
    fun start() {
        if (started) return
        started = true
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            thermalSignal = mapThermalStatus(powerManager.currentThermalStatus)
            thermalListener?.let {
                powerManager.addThermalStatusListener(
                    ContextCompat.getMainExecutor(appContext),
                    it
                )
            }
        }
        val sticky = ContextCompat.registerReceiver(
            appContext,
            batteryReceiver,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED),
            ContextCompat.RECEIVER_NOT_EXPORTED
        )
        val tenthsC = sticky?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        batteryTemperatureC = tenthsC
            ?.takeIf { it != Int.MIN_VALUE }
            ?.div(10f)
        publish()
    }

    @Synchronized
    private fun publish() {
        if (!started) return
        val profile = policy.update(thermalSignal, batteryTemperatureC)
        onSnapshot(DeviceThermalSnapshot(thermalSignal, batteryTemperatureC, profile))
    }

    @Synchronized
    override fun close() {
        if (!started) return
        started = false
        runCatching { appContext.unregisterReceiver(batteryReceiver) }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            thermalListener?.let { listener ->
                runCatching { powerManager.removeThermalStatusListener(listener) }
            }
        }
    }

    private fun mapThermalStatus(status: Int): ThermalSignal = when {
        status >= PowerManager.THERMAL_STATUS_CRITICAL -> ThermalSignal.CRITICAL
        status >= PowerManager.THERMAL_STATUS_SEVERE -> ThermalSignal.SEVERE
        status >= PowerManager.THERMAL_STATUS_MODERATE -> ThermalSignal.MODERATE
        status >= PowerManager.THERMAL_STATUS_LIGHT -> ThermalSignal.LIGHT
        else -> ThermalSignal.NORMAL
    }
}
