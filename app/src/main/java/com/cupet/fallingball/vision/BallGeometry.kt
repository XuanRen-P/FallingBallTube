package com.cupet.fallingball.vision

import kotlin.math.abs

data class TubeBallPoint(
    val x: Float,
    val y: Float,
    val axialPx: Float,
    val lateralPx: Float
)

enum class WallContactState {
    UNKNOWN,
    CLEAR,
    LEFT_WALL,
    RIGHT_WALL
}

data class WallContactResult(
    val state: WallContactState,
    val clearancePx: Float?,
    val normalizedLateral: Float?,
    val lateralMm: Float? = null,
    val clearanceMm: Float? = null
)

object BallGeometry {
    /** Metric tolerance absorbs sub-pixel heatmap and ruler calibration error. */
    const val CONTACT_TOLERANCE_MM = 0.2f

    fun isInsideTubeAxialRange(axialPx: Float, axialMaxPx: Float): Boolean =
        axialPx.isFinite() && axialMaxPx.isFinite() && axialMaxPx > 0f &&
            axialPx in 0f..axialMaxPx

    /** Matches the PC canonical ROI: x grows left-to-right, y grows top-to-bottom. */
    fun canonicalToDisplay(
        roiX: Float,
        roiY: Float,
        roiWidth: Int,
        roiHeight: Int,
        axis: TubeAxis
    ): TubeBallPoint {
        require(roiWidth > 1 && roiHeight > 1) { "Canonical dimensions must exceed one" }
        val axial = TubeRoiSampler.canonicalAxialPx(roiY, roiHeight, axis.lengthPx)
        val lateral = TubeRoiSampler.canonicalLateralPx(roiX, roiWidth, axis.widthPx)
        return TubeBallPoint(
            x = axis.pointX(axial, lateral),
            y = axis.pointY(axial, lateral),
            axialPx = axial,
            lateralPx = lateral
        )
    }

    fun classifyWallContact(
        lateralPx: Float,
        tubeInnerWidthPx: Float?,
        sphereRadiusPx: Float?,
        pixelsPerMm: Float? = null,
        contactTolerancePx: Float = 2f
    ): WallContactResult {
        val validScale = pixelsPerMm?.takeIf { it.isFinite() && it > 0f }
        val lateralMm = validScale?.let { lateralPx / it }
        if (!lateralPx.isFinite() || tubeInnerWidthPx == null ||
            !tubeInnerWidthPx.isFinite() || tubeInnerWidthPx <= 0f
        ) {
            return WallContactResult(
                state = WallContactState.UNKNOWN,
                clearancePx = null,
                normalizedLateral = null,
                lateralMm = lateralMm
            )
        }
        val normalizedLateral = lateralPx / (tubeInnerWidthPx * 0.5f)
        if (sphereRadiusPx == null || !sphereRadiusPx.isFinite() ||
            sphereRadiusPx <= 0f || tubeInnerWidthPx <= sphereRadiusPx * 2f
        ) {
            return WallContactResult(
                state = WallContactState.UNKNOWN,
                clearancePx = null,
                normalizedLateral = normalizedLateral,
                lateralMm = lateralMm
            )
        }
        val clearance = tubeInnerWidthPx * 0.5f - abs(lateralPx) - sphereRadiusPx
        val state = when {
            clearance > contactTolerancePx -> WallContactState.CLEAR
            lateralPx > 0f -> WallContactState.LEFT_WALL
            else -> WallContactState.RIGHT_WALL
        }
        return WallContactResult(
            state = state,
            clearancePx = clearance,
            normalizedLateral = normalizedLateral,
            lateralMm = lateralMm,
            clearanceMm = validScale?.let { clearance / it }
        )
    }

    fun classifyWallContactMm(
        lateralMm: Float,
        tubeInnerDiameterMm: Double?,
        sphereRadiusMm: Double?
    ): WallContactResult = classifyWallContact(
        lateralPx = lateralMm,
        tubeInnerWidthPx = tubeInnerDiameterMm?.toFloat(),
        sphereRadiusPx = sphereRadiusMm?.toFloat(),
        pixelsPerMm = 1f,
        contactTolerancePx = CONTACT_TOLERANCE_MM
    )
}
