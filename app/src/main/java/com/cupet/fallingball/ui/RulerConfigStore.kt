package com.cupet.fallingball.ui

import android.content.Context
import com.cupet.fallingball.R
import com.cupet.fallingball.analysis.MeasurementDefaults
import com.cupet.fallingball.analysis.ViscometryParameters
import com.cupet.fallingball.vision.ModelProfile
import com.cupet.fallingball.vision.RulerConfig
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout

/**
 * Persists the cylinder parameters and presents the shared parameter dialog.
 *
 * The values are remembered between launches: they describe the user's glassware and camera
 * setup, not this particular measurement, and re-typing them on every start would be tedious
 * enough that people would leave them wrong. The dialog also returns the YOLO profile; its
 * persistence remains owned by [ModelProfileStore].
 */
object RulerConfigStore {

    private const val PREFS = "ruler_config"
    private const val KEY_SPACING = "tick_spacing_mm"
    private const val KEY_COUNT = "tick_count"
    private const val KEY_CONFIGURED = "configured"
    private const val KEY_ANALYSIS_RESOLUTION = "analysis_resolution"
    private const val KEY_SPHERE_DIAMETER = "sphere_diameter_mm"
    private const val KEY_LEGACY_SPHERE_RADIUS = "sphere_radius_mm"
    private const val KEY_TUBE_DIAMETER = "tube_inner_diameter_mm"
    private const val KEY_LIQUID_DENSITY = "liquid_density_g_cm3"
    private const val KEY_SPHERE_DENSITY = "sphere_density_g_cm3"
    private const val KEY_LIQUID_NAME = "liquid_name"
    private const val KEY_SPHERE_MATERIAL = "sphere_material"
    private const val KEY_TEMPERATURE = "temperature_c"

    fun load(context: Context): RulerConfig {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return RulerConfig(
            tickSpacingMm = p.getFloat(KEY_SPACING, RulerConfig.DEFAULT.tickSpacingMm),
            expectedTickCount = p.getInt(KEY_COUNT, RulerConfig.COUNT_UNKNOWN)
        )
    }

    /** Whether the user has ever completed the dialog, used to decide whether to prompt. */
    fun isConfigured(context: Context): Boolean =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getBoolean(KEY_CONFIGURED, false)

    fun loadAnalysisResolution(context: Context): AnalysisResolution {
        val stored = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(KEY_ANALYSIS_RESOLUTION, null)
        return AnalysisResolution.fromStoredName(stored)
    }

    fun loadMeasurementDefaults(context: Context): MeasurementDefaults {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        fun optionalDouble(key: String) = p.getString(key, null)?.toDoubleOrNull()
        val sphereDiameterMm = optionalDouble(KEY_SPHERE_DIAMETER)
            ?: optionalDouble(KEY_LEGACY_SPHERE_RADIUS)
        return MeasurementDefaults(
            viscometryParameters = ViscometryParameters(
                sphereRadiusMm = sphereDiameterMm?.div(2.0),
                tubeInnerDiameterMm = optionalDouble(KEY_TUBE_DIAMETER),
                liquidDensityGPerCm3 = optionalDouble(KEY_LIQUID_DENSITY),
                sphereDensityGPerCm3 = optionalDouble(KEY_SPHERE_DENSITY)
            ),
            liquidName = p.getString(KEY_LIQUID_NAME, "").orEmpty(),
            sphereMaterial = p.getString(KEY_SPHERE_MATERIAL, "").orEmpty(),
            temperatureC = optionalDouble(KEY_TEMPERATURE)
        )
    }

    private fun save(
        context: Context,
        config: RulerConfig,
        analysisResolution: AnalysisResolution,
        defaults: MeasurementDefaults
    ) {
        val editor = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putFloat(KEY_SPACING, config.tickSpacingMm)
            .putInt(KEY_COUNT, config.expectedTickCount)
            .putBoolean(KEY_CONFIGURED, true)
            .putString(KEY_ANALYSIS_RESOLUTION, analysisResolution.name)
            .putString(KEY_LIQUID_NAME, defaults.liquidName)
            .putString(KEY_SPHERE_MATERIAL, defaults.sphereMaterial)
        fun putOptional(key: String, value: Double?) {
            if (value == null) editor.remove(key) else editor.putString(key, value.toString())
        }
        putOptional(KEY_SPHERE_DIAMETER, defaults.viscometryParameters.sphereDiameterMm)
        editor.remove(KEY_LEGACY_SPHERE_RADIUS)
        putOptional(KEY_TUBE_DIAMETER, defaults.viscometryParameters.tubeInnerDiameterMm)
        putOptional(KEY_LIQUID_DENSITY, defaults.viscometryParameters.liquidDensityGPerCm3)
        putOptional(KEY_SPHERE_DENSITY, defaults.viscometryParameters.sphereDensityGPerCm3)
        putOptional(KEY_TEMPERATURE, defaults.temperatureC)
        editor.apply()
    }

    /**
     * Shows the parameter dialog, calling [onAccepted] with the validated ruler values and
     * selected YOLO profile and camera analysis resolution.
     *
     * Validation happens in place rather than on dismissal so a typo is corrected while the
     * field is still on screen. The graduation count may be left blank, which turns the
     * count-based checks off rather than guessing a number.
     */
    fun prompt(
        context: Context,
        current: RulerConfig,
        currentModelProfile: ModelProfile,
        currentAnalysisResolution: AnalysisResolution,
        currentDefaults: MeasurementDefaults,
        cameraSettingsEnabled: Boolean,
        cancellable: Boolean,
        cameraSettingsVisible: Boolean = true,
        persistChanges: Boolean = true,
        onAccepted: (
            RulerConfig,
            ModelProfile,
            AnalysisResolution,
            MeasurementDefaults
        ) -> Unit
    ) {
        val view = android.view.LayoutInflater.from(context)
            .inflate(R.layout.dialog_ruler_config, null)
        val spacingLayout = view.findViewById<TextInputLayout>(R.id.spacingLayout)
        val countLayout = view.findViewById<TextInputLayout>(R.id.countLayout)
        val spacingInput = view.findViewById<TextInputEditText>(R.id.spacingInput)
        val countInput = view.findViewById<TextInputEditText>(R.id.countInput)
        val modelProfileGroup =
            view.findViewById<MaterialButtonToggleGroup>(R.id.modelProfileGroup)
        val resolutionGroup =
            view.findViewById<MaterialButtonToggleGroup>(R.id.resolutionGroup)
        if (!cameraSettingsVisible) {
            listOf(R.id.modelProfileLabel, R.id.resolutionLabel,
                R.id.modelProfileGroup, R.id.resolutionGroup).forEach { id ->
                view.findViewById<android.view.View>(id).visibility = android.view.View.GONE
            }
        }
        val sphereDiameterLayout = view.findViewById<TextInputLayout>(R.id.sphereDiameterLayout)
        val tubeDiameterLayout = view.findViewById<TextInputLayout>(R.id.tubeDiameterLayout)
        val liquidDensityLayout = view.findViewById<TextInputLayout>(R.id.liquidDensityLayout)
        val sphereDensityLayout = view.findViewById<TextInputLayout>(R.id.sphereDensityLayout)
        val sphereDiameterInput =
            view.findViewById<TextInputEditText>(R.id.sphereDiameterInput)
        val tubeDiameterInput =
            view.findViewById<TextInputEditText>(R.id.tubeDiameterInput)
        val liquidDensityInput =
            view.findViewById<TextInputEditText>(R.id.liquidDensityInput)
        val sphereDensityInput =
            view.findViewById<TextInputEditText>(R.id.sphereDensityInput)
        val liquidNameInput = view.findViewById<TextInputEditText>(R.id.liquidNameInput)
        val sphereMaterialInput =
            view.findViewById<TextInputEditText>(R.id.sphereMaterialInput)
        val temperatureLayout = view.findViewById<TextInputLayout>(R.id.temperatureLayout)
        val temperatureInput = view.findViewById<TextInputEditText>(R.id.temperatureInput)

        spacingInput.setText(trimZeros(current.tickSpacingMm))
        if (current.hasExpectedCount) {
            countInput.setText(
                String.format(java.util.Locale.ROOT, "%d", current.expectedTickCount)
            )
        }
        modelProfileGroup.check(modelProfileButtonId(currentModelProfile))
        resolutionGroup.check(
            if (currentAnalysisResolution == AnalysisResolution.P1080) {
                R.id.btn1080p
            } else {
                R.id.btn720p
            }
        )
        val currentParameters = currentDefaults.viscometryParameters
        sphereDiameterInput.setText(trimOptional(currentParameters.sphereDiameterMm))
        tubeDiameterInput.setText(trimOptional(currentParameters.tubeInnerDiameterMm))
        liquidDensityInput.setText(trimOptional(currentParameters.liquidDensityGPerCm3))
        sphereDensityInput.setText(trimOptional(currentParameters.sphereDensityGPerCm3))
        liquidNameInput.setText(currentDefaults.liquidName)
        sphereMaterialInput.setText(currentDefaults.sphereMaterial)
        temperatureInput.setText(trimOptional(currentDefaults.temperatureC))
        for (index in 0 until modelProfileGroup.childCount) {
            modelProfileGroup.getChildAt(index).isEnabled = cameraSettingsEnabled
        }
        for (index in 0 until resolutionGroup.childCount) {
            resolutionGroup.getChildAt(index).isEnabled = cameraSettingsEnabled
        }

        val dialog = MaterialAlertDialogBuilder(context)
            .setTitle(R.string.config_title)
            .setView(view)
            .setCancelable(cancellable)
            .setPositiveButton(android.R.string.ok, null)
            .apply { if (cancellable) setNegativeButton(android.R.string.cancel, null) }
            .create()

        dialog.setOnShowListener {
            dialog.getButton(android.app.AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val spacing = spacingInput.text?.toString()?.trim()?.toFloatOrNull()
                if (spacing == null || !spacing.isFinite() ||
                    spacing <= 0f || spacing > RulerConfig.MAX_SPACING_MM
                ) {
                    spacingLayout.error = context.getString(R.string.config_invalid_spacing)
                    return@setOnClickListener
                }
                spacingLayout.error = null

                val countText = countInput.text?.toString()?.trim().orEmpty()
                val count = if (countText.isEmpty()) RulerConfig.COUNT_UNKNOWN
                else countText.toIntOrNull() ?: -1
                if (count != RulerConfig.COUNT_UNKNOWN &&
                    count !in RulerConfig.MIN_TICKS..RulerConfig.MAX_TICKS
                ) {
                    countLayout.error = context.getString(
                        R.string.config_invalid_count,
                        RulerConfig.MIN_TICKS,
                        RulerConfig.MAX_TICKS
                    )
                    return@setOnClickListener
                }
                countLayout.error = null

                fun optionalPositive(
                    input: TextInputEditText,
                    layout: TextInputLayout
                ): Double? {
                    val text = input.text?.toString()?.trim().orEmpty()
                    if (text.isEmpty()) {
                        layout.error = null
                        return null
                    }
                    val value = text.toDoubleOrNull()
                    if (value == null || !value.isFinite() || value <= 0.0) {
                        layout.error = context.getString(R.string.config_optional_positive)
                        throw IllegalArgumentException("invalid optional number")
                    }
                    layout.error = null
                    return value
                }
                val parameters = try {
                    ViscometryParameters(
                        sphereRadiusMm = optionalPositive(
                            sphereDiameterInput,
                            sphereDiameterLayout
                        )?.div(2.0),
                        tubeInnerDiameterMm = optionalPositive(
                            tubeDiameterInput,
                            tubeDiameterLayout
                        ),
                        liquidDensityGPerCm3 = optionalPositive(
                            liquidDensityInput,
                            liquidDensityLayout
                        ),
                        sphereDensityGPerCm3 = optionalPositive(
                            sphereDensityInput,
                            sphereDensityLayout
                        )
                    )
                } catch (_: IllegalArgumentException) {
                    return@setOnClickListener
                }
                val temperatureText = temperatureInput.text?.toString()?.trim().orEmpty()
                val temperature = temperatureText.takeIf { it.isNotEmpty() }?.toDoubleOrNull()
                if (temperatureText.isNotEmpty() &&
                    (temperature == null || !temperature.isFinite())
                ) {
                    temperatureLayout.error =
                        context.getString(R.string.config_invalid_temperature)
                    return@setOnClickListener
                }
                temperatureLayout.error = null
                if (parameters.sphereRadiusMm != null &&
                    parameters.tubeInnerDiameterMm != null &&
                    parameters.tubeInnerDiameterMm <= parameters.sphereRadiusMm * 2.0
                ) {
                    tubeDiameterLayout.error =
                        context.getString(R.string.config_tube_too_small)
                    return@setOnClickListener
                }
                if (parameters.liquidDensityGPerCm3 != null &&
                    parameters.sphereDensityGPerCm3 != null &&
                    parameters.sphereDensityGPerCm3 <= parameters.liquidDensityGPerCm3
                ) {
                    sphereDensityLayout.error =
                        context.getString(R.string.config_sphere_not_denser)
                    return@setOnClickListener
                }

                val config = RulerConfig(spacing, count)
                val modelProfile = when (modelProfileGroup.checkedButtonId) {
                    R.id.btnYolo320 -> ModelProfile.FASTEST_320
                    R.id.btnYolo416 -> ModelProfile.FAST_416
                    R.id.btnYolo512 -> ModelProfile.BALANCED_512
                    else -> ModelProfile.ACCURATE_640
                }
                val analysisResolution = AnalysisResolution.fromButtonId(
                    resolutionGroup.checkedButtonId == R.id.btn1080p
                )
                val defaults = MeasurementDefaults(
                    viscometryParameters = parameters,
                    liquidName = liquidNameInput.text?.toString()?.trim().orEmpty(),
                    sphereMaterial = sphereMaterialInput.text?.toString()?.trim().orEmpty(),
                    temperatureC = temperature
                )
                if (persistChanges) save(
                    context,
                    config,
                    analysisResolution,
                    defaults
                )
                onAccepted(
                    config,
                    modelProfile,
                    analysisResolution,
                    defaults
                )
                dialog.dismiss()
            }
        }
        dialog.show()
    }

    private fun trimZeros(v: Float): String =
        if (v == v.toInt().toFloat()) v.toInt().toString()
        else String.format(java.util.Locale.US, "%.2f", v).trimEnd('0').trimEnd('.')

    private fun trimOptional(v: Double?): String =
        v?.let {
            String.format(java.util.Locale.US, "%.6f", it).trimEnd('0').trimEnd('.')
        }.orEmpty()

    private fun modelProfileButtonId(profile: ModelProfile): Int = when (profile) {
        ModelProfile.FASTEST_320 -> R.id.btnYolo320
        ModelProfile.FAST_416 -> R.id.btnYolo416
        ModelProfile.BALANCED_512 -> R.id.btnYolo512
        ModelProfile.ACCURATE_640 -> R.id.btnYolo640
    }
}
