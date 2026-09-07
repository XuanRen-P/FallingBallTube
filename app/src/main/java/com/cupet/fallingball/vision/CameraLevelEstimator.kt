package com.cupet.fallingball.vision

import kotlin.math.asin
import kotlin.math.sqrt

data class CameraLevelReading(
    val angleDegrees: Float,
    val isLevel: Boolean,
    val lineLengthFraction: Float
)

/**
 * Estimates how far the rear-camera optical axis points above or below the horizon.
 *
 * Android's gravity vector is expressed in device coordinates. Its Z component is zero
 * when the phone's screen plane is vertical, which makes the camera axis horizontal
 * regardless of portrait/landscape display rotation.
 */
class CameraLevelEstimator(
    private val smoothing: Float = 0.35f
) {
    companion object {
        const val LEVEL_TOLERANCE_DEGREES = 10f
        private const val MIN_GRAVITY = 1f
        private const val RADIANS_TO_DEGREES = 57.29578f
        private const val ANGLE_EPSILON_DEGREES = 0.01f
    }

    private var gravityX = Float.NaN
    private var gravityY = Float.NaN
    private var gravityZ = Float.NaN

    init {
        require(smoothing in 0f..1f)
    }

    fun update(
        gravityX: Float,
        gravityY: Float,
        gravityZ: Float
    ): CameraLevelReading? {
        val sampleMagnitude = sqrt(
            gravityX * gravityX + gravityY * gravityY + gravityZ * gravityZ
        )
        if (!sampleMagnitude.isFinite() || sampleMagnitude < MIN_GRAVITY) return null

        if (this.gravityX.isNaN()) {
            this.gravityX = gravityX
            this.gravityY = gravityY
            this.gravityZ = gravityZ
        } else {
            this.gravityX += (gravityX - this.gravityX) * smoothing
            this.gravityY += (gravityY - this.gravityY) * smoothing
            this.gravityZ += (gravityZ - this.gravityZ) * smoothing
        }
        val magnitude = sqrt(
            this.gravityX * this.gravityX +
                this.gravityY * this.gravityY +
                this.gravityZ * this.gravityZ
        )
        val angle = asin((this.gravityZ / magnitude).coerceIn(-1f, 1f)) *
            RADIANS_TO_DEGREES
        return CameraLevelReading(
            angleDegrees = angle,
            isLevel = kotlin.math.abs(angle) <=
                LEVEL_TOLERANCE_DEGREES + ANGLE_EPSILON_DEGREES,
            lineLengthFraction = (kotlin.math.abs(angle) / 90f).coerceIn(0f, 1f)
        )
    }

    fun reset() {
        gravityX = Float.NaN
        gravityY = Float.NaN
        gravityZ = Float.NaN
    }
}
