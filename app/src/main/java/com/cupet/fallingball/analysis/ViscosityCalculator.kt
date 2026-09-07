package com.cupet.fallingball.analysis

import kotlin.math.pow

data class ViscometryParameters(
    val sphereRadiusMm: Double? = null,
    val tubeInnerDiameterMm: Double? = null,
    val liquidDensityGPerCm3: Double? = null,
    val sphereDensityGPerCm3: Double? = null
) {
    val sphereDiameterMm: Double?
        get() = sphereRadiusMm?.times(2.0)

    val isComplete: Boolean
        get() = sphereRadiusMm != null &&
            tubeInnerDiameterMm != null &&
            liquidDensityGPerCm3 != null &&
            sphereDensityGPerCm3 != null
}

enum class ViscosityWarning {
    REYNOLDS_OUT_OF_RANGE,
    DIAMETER_RATIO_LARGE,
    OFF_AXIS_OR_WALL_CONTACT
}

data class ViscosityResult(
    val wallCorrectedStokesPaS: Double,
    val oseenViscosityPaS: Double,
    val faxenFactor: Double,
    val unboundedVelocityMmPerS: Double,
    val reynoldsNumber: Double,
    val diameterRatio: Double,
    val warnings: Set<ViscosityWarning>,
    val methodVersion: String = ViscosityCalculator.METHOD_VERSION
) {
    /** Compatibility field name; current records store the selected cylindrical-wall factor. */
    val wallCorrectionFactor: Double get() = faxenFactor

    val uncorrectedStokesPaS: Double
        get() = wallCorrectedStokesPaS / wallCorrectionFactor

    val wallCorrectionModel: String
        get() = if (methodVersion == ViscosityCalculator.METHOD_VERSION) {
            when {
                diameterRatio <= ViscosityCalculator.LADENBURG_MAX_DIAMETER_RATIO ->
                    "LADENBURG"
                diameterRatio <= ViscosityCalculator.RECOMMENDED_MAX_DIAMETER_RATIO ->
                    "FAXEN"
                else -> "HABERMAN_SAYRE"
            }
        } else {
            if (diameterRatio > ViscosityCalculator.RECOMMENDED_MAX_DIAMETER_RATIO) {
                "HABERMAN_SAYRE"
            } else {
                "FAXEN"
            }
        }
}

object ViscosityCalculator {
    const val METHOD_VERSION = "STOKES_CYLINDER_WALL_V3"
    const val STANDARD_GRAVITY_M_PER_S2 = 9.80665
    const val MAX_REYNOLDS = 1.0
    const val LADENBURG_MAX_DIAMETER_RATIO = 0.09
    const val RECOMMENDED_MAX_DIAMETER_RATIO = 0.1

    fun calculate(
        parameters: ViscometryParameters,
        terminalVelocityMmPerS: Double?
    ): ViscosityResult? {
        if (!parameters.isComplete || terminalVelocityMmPerS == null) return null

        val radiusMm = requireNotNull(parameters.sphereRadiusMm)
        val tubeDiameterMm = requireNotNull(parameters.tubeInnerDiameterMm)
        val liquidDensity = requireNotNull(parameters.liquidDensityGPerCm3) * 1_000.0
        val sphereDensity = requireNotNull(parameters.sphereDensityGPerCm3) * 1_000.0
        require(
            radiusMm.isFinite() &&
                tubeDiameterMm.isFinite() &&
                liquidDensity.isFinite() &&
                sphereDensity.isFinite() &&
                terminalVelocityMmPerS.isFinite()
        ) { "All physical inputs must be finite" }
        require(radiusMm > 0.0) { "Sphere radius must be positive" }
        require(tubeDiameterMm > radiusMm * 2.0) {
            "Tube inner diameter must exceed sphere diameter"
        }
        require(liquidDensity > 0.0) { "Liquid density must be positive" }
        require(sphereDensity > liquidDensity) {
            "Sphere density must exceed liquid density"
        }
        require(terminalVelocityMmPerS > 0.0) { "Terminal velocity must be positive" }

        val sphereDiameterM = radiusMm * 2.0 / 1_000.0
        val diameterRatio = radiusMm * 2.0 / tubeDiameterMm
        val faxenFactor = 1.0 -
            2.10444 * diameterRatio +
            2.08877 * diameterRatio.pow(3) -
            0.94813 * diameterRatio.pow(5)
        require(faxenFactor > 0.0) { "Faxen correction is not physical" }
        val habermanSayreFactor = (
            1.0 -
                2.105 * diameterRatio +
                2.0865 * diameterRatio.pow(3) -
                1.7068 * diameterRatio.pow(5) +
                0.72603 * diameterRatio.pow(6)
            ) / (1.0 - 0.75857 * diameterRatio.pow(5))
        require(habermanSayreFactor > 0.0) {
            "Haberman-Sayre correction is not physical"
        }
        val ladenburgFactor = 1.0 / (1.0 + 2.4 * diameterRatio)
        val wallFactor = when {
            diameterRatio <= LADENBURG_MAX_DIAMETER_RATIO -> ladenburgFactor
            diameterRatio <= RECOMMENDED_MAX_DIAMETER_RATIO -> faxenFactor
            else -> habermanSayreFactor
        }

        val measuredVelocityMPerS = terminalVelocityMmPerS / 1_000.0
        val unboundedVelocityMPerS = measuredVelocityMPerS / wallFactor
        val uncorrectedStokes = sphereDiameterM.pow(2) *
            (sphereDensity - liquidDensity) *
            STANDARD_GRAVITY_M_PER_S2 /
            (18.0 * measuredVelocityMPerS)
        val wallCorrectedStokes = uncorrectedStokes * wallFactor
        val oseen = wallCorrectedStokes -
            3.0 * liquidDensity * sphereDiameterM * unboundedVelocityMPerS / 16.0
        require(oseen > 0.0) { "Oseen correction produced non-positive viscosity" }

        val reynolds = liquidDensity * sphereDiameterM * unboundedVelocityMPerS / oseen
        val warnings = buildSet {
            if (reynolds >= MAX_REYNOLDS) add(ViscosityWarning.REYNOLDS_OUT_OF_RANGE)
            if (diameterRatio > RECOMMENDED_MAX_DIAMETER_RATIO) {
                add(ViscosityWarning.DIAMETER_RATIO_LARGE)
            }
        }
        return ViscosityResult(
            wallCorrectedStokesPaS = wallCorrectedStokes,
            oseenViscosityPaS = oseen,
            faxenFactor = wallFactor,
            unboundedVelocityMmPerS = unboundedVelocityMPerS * 1_000.0,
            reynoldsNumber = reynolds,
            diameterRatio = diameterRatio,
            warnings = warnings
        )
    }
}
