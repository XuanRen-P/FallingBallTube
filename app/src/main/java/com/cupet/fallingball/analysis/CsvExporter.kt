package com.cupet.fallingball.analysis

import java.io.OutputStream
import java.util.Locale

/**
 * Serialises a session to CSV: a commented metadata header followed by one row per sample.
 *
 * Every number is formatted with [Locale.ROOT]. Under a Chinese or European locale the
 * default formatter emits a comma as the decimal separator, which in a comma-separated file
 * silently shifts every column -- the file still opens, it is just wrong.
 */
object CsvExporter {

    private const val COLUMNS =
        "t_ms,position_mm,velocity_mm_s,axial_px,ruler_valid,extrapolated,shift_px," +
            "position_ticks,lateral_mm,lateral_px,normalized_lateral," +
            "wall_clearance_mm,used_in_terminal_fit,excluded_as_outlier,wall_contact_state," +
            "detection_source,detection_confidence"

    fun buildCsv(session: FallSession, timestampIso: String): String {
        val sb = StringBuilder(4096)
        val s = session.summary

        sb.appendLine("# FallingBallTube export")
        sb.appendLine("# timestamp,$timestampIso")
        sb.appendLine("# tick_spacing_mm,${num(session.config.tickSpacingMm)}")
        sb.appendLine(
            "# expected_tick_count," +
                if (session.config.hasExpectedCount) "${session.config.expectedTickCount}"
                else "unset"
        )
        sb.appendLine("# resolution,${session.resolutionLabel}")
        sb.appendLine("# end_reason,${session.endReason ?: "IN_PROGRESS"}")
        sb.appendLine("# samples,${session.samples.size}")
        sb.appendLine("# valid_samples,${session.validSampleCount}")
        sb.appendLine("# mean_fps,${num(session.meanFps())}")
        sb.appendLine("# peak_shift_px,${num(session.peakShiftPx())}")
        appendAnalysisQuality(sb, session.analysisQuality)

        if (s != null) {
            sb.appendLine("# plateau_found,${s.plateauFound}")
            sb.appendLine(
                "# terminal_velocity_mm_s," +
                    (s.terminalVelocityMmPerS?.let { num(it) } ?: "")
            )
            sb.appendLine("# mean_velocity_mm_s,${s.meanVelocityMmPerS?.let { num(it) } ?: ""}")
            sb.appendLine("# plateau_range_mm,${num(s.plateauStartMm)},${num(s.plateauEndMm)}")
            sb.appendLine("# plateau_duration_ms,${s.plateauDurationMs}")
            sb.appendLine("# r2,${num(s.r2, 4)}")
            sb.appendLine("# velocity_std_dev_mm_s,${num(s.velocityStdDev)}")
        }

        sb.appendLine(COLUMNS)

        val t0 = session.samples.firstOrNull()?.tNs ?: 0L
        for (sample in session.samples) {
            sb.append(num((sample.tNs - t0) / 1_000_000.0f, 2)).append(',')
            sb.append(sample.positionMm?.let { num(it) } ?: "").append(',')
            sb.append(sample.velocityMmPerS?.let { num(it) } ?: "").append(',')
            sb.append(num(sample.axialPx, 1)).append(',')
            sb.append(sample.rulerValid).append(',')
            sb.append(sample.extrapolated).append(',')
            sb.append(num(sample.shiftPx, 2)).append(',')
            sb.append(sample.positionTicks?.let { num(it) } ?: "").append(',')
            sb.append(sample.lateralMm?.let { num(it, 4) } ?: "").append(',')
            sb.append(sample.lateralPx?.let { num(it, 2) } ?: "").append(',')
            sb.append(sample.normalizedLateral?.let { num(it, 5) } ?: "").append(',')
            sb.append(sample.wallClearanceMm?.let { num(it, 4) } ?: "").append(',')
            sb.append(sample.usedInTerminalFit).append(',')
            sb.append(sample.excludedAsOutlier).append(',')
            sb.append(sample.wallContactState.name).append(',')
            sb.append(sample.detectionSource.name).append(',')
            sb.append(sample.detectionConfidence?.let { num(it, 5) } ?: "").append('\n')
        }
        return sb.toString()
    }

    fun buildCsv(record: MeasurementRecord, timestampIso: String): String {
        val sb = StringBuilder(4096)
        val summary = record.summary
        val parameters = record.viscometryParameters
        val viscosity = record.viscosityResult.takeIf {
            record.hasReliableTerminalEstimate
        }

        sb.appendLine("# FallingBallTube export")
        // CSV v7 makes sphere diameter and the selected PC wall correction canonical.
        // independent from the binary FBR codec.
        sb.appendLine("# format_version,8")
        sb.appendLine("# timestamp,${csv(timestampIso)}")
        sb.appendLine("# record_id,${csv(record.id)}")
        appendSource(sb, record)
        sb.appendLine("# record_name,${csv(record.metadata.recordName)}")
        sb.appendLine("# liquid_name,${csv(record.metadata.liquidName)}")
        sb.appendLine("# sphere_material,${csv(record.metadata.sphereMaterial)}")
        sb.appendLine("# temperature_c,${parametersOrBlank(record.metadata.temperatureC)}")
        sb.appendLine("# notes,${csv(record.metadata.notes)}")
        sb.appendLine("# tick_spacing_mm,${num(record.config.tickSpacingMm)}")
        sb.appendLine(
            "# expected_tick_count," +
                if (record.config.hasExpectedCount) record.config.expectedTickCount else "unset"
        )
        sb.appendLine("# sphere_diameter_mm,${parametersOrBlank(parameters.sphereDiameterMm)}")
        // Compatibility alias for consumers that still need the internal radius.
        sb.appendLine("# sphere_radius_mm,${parametersOrBlank(parameters.sphereRadiusMm)}")
        sb.appendLine(
            "# tube_inner_diameter_mm," +
                parametersOrBlank(parameters.tubeInnerDiameterMm)
        )
        sb.appendLine(
            "# liquid_density_g_cm3," +
                parametersOrBlank(parameters.liquidDensityGPerCm3)
        )
        sb.appendLine(
            "# sphere_density_g_cm3," +
                parametersOrBlank(parameters.sphereDensityGPerCm3)
        )
        sb.appendLine("# resolution,${csv(record.resolutionLabel)}")
        sb.appendLine("# end_reason,${record.endReason}")
        sb.appendLine("# samples,${record.samples.size}")
        sb.appendLine("# valid_samples,${record.validSampleCount}")
        sb.appendLine("# mean_fps,${num(record.meanFps)}")
        sb.appendLine("# peak_shift_px,${num(record.peakShiftPx)}")
        sb.appendLine("# wall_contact_samples,${record.wallContactSampleCount}")
        sb.appendLine("# sustained_wall_contact,${record.sustainedWallContact}")
        appendAnalysisQuality(sb, record.analysisQuality)
        sb.appendLine("# terminal_estimate_reliable,${record.hasReliableTerminalEstimate}")
        sb.appendLine("# analysis_method_version,${record.analysisMethodVersion}")
        appendSummary(sb, summary)
        sb.appendLine(
            "# viscosity_pa_s," +
                (viscosity?.wallCorrectedStokesPaS?.let { num(it, 8) } ?: "")
        )
        sb.appendLine(
            "# viscosity_mpa_s," +
                (viscosity?.wallCorrectedStokesPaS?.times(1000.0)?.let { num(it, 5) } ?: "")
        )
        sb.appendLine(
            "# stokes_uncorrected_pa_s," +
                (viscosity?.uncorrectedStokesPaS?.let { num(it, 8) } ?: "")
        )
        sb.appendLine(
            "# stokes_wall_corrected_pa_s," +
                (viscosity?.wallCorrectedStokesPaS?.let { num(it, 8) } ?: "")
        )
        sb.appendLine(
            "# wall_correction_model," +
                csv(viscosity?.wallCorrectionModel.orEmpty())
        )
        sb.appendLine(
            "# wall_correction_factor," +
                (viscosity?.wallCorrectionFactor?.let { num(it, 8) } ?: "")
        )
        // Kept as a format-v3 compatibility alias for existing CSV consumers.
        sb.appendLine(
            "# faxen_factor," +
                (viscosity?.wallCorrectionFactor?.let { num(it, 8) } ?: "")
        )
        sb.appendLine(
            "# unbounded_velocity_mm_s," +
                (viscosity?.unboundedVelocityMmPerS?.let { num(it, 5) } ?: "")
        )
        sb.appendLine(
            "# oseen_diagnostic_pa_s," +
                (viscosity?.oseenViscosityPaS?.let { num(it, 8) } ?: "")
        )
        sb.appendLine(
            "# reynolds_number," +
                (viscosity?.reynoldsNumber?.let { num(it, 8) } ?: "")
        )
        sb.appendLine(
            "# calculation_warnings," +
                csv(viscosity?.warnings?.joinToString("|") { it.name }.orEmpty())
        )
        sb.appendLine(
            "# calculation_method_version," +
                csv(viscosity?.methodVersion ?: ViscosityCalculator.METHOD_VERSION)
        )
        sb.appendLine(COLUMNS)
        for (sample in record.samples) {
            sb.append(num(sample.elapsedMs, 2)).append(',')
            sb.append(sample.positionMm?.let { num(it) } ?: "").append(',')
            sb.append(sample.velocityMmPerS?.let { num(it) } ?: "").append(',')
            sb.append(num(sample.axialPx, 1)).append(',')
            sb.append(sample.rulerValid).append(',')
            sb.append(sample.extrapolated).append(',')
            sb.append(num(sample.shiftPx, 2)).append(',')
            sb.append(sample.positionTicks?.let { num(it) } ?: "").append(',')
            sb.append(sample.lateralMm?.let { num(it, 4) } ?: "").append(',')
            sb.append(sample.lateralPx?.let { num(it, 2) } ?: "").append(',')
            sb.append(sample.normalizedLateral?.let { num(it, 5) } ?: "").append(',')
            sb.append(sample.wallClearanceMm?.let { num(it, 4) } ?: "").append(',')
            sb.append(sample.usedInTerminalFit).append(',')
            sb.append(sample.excludedAsOutlier).append(',')
            sb.append(sample.wallContactState.name).append(',')
            sb.append(sample.detectionSource.name).append(',')
            sb.append(sample.detectionConfidence?.let { num(it, 5) } ?: "").append('\n')
        }
        return sb.toString()
    }

    fun writeTo(stream: OutputStream, session: FallSession, timestampIso: String) {
        stream.bufferedWriter(Charsets.UTF_8).use { it.write(buildCsv(session, timestampIso)) }
    }

    fun writeTo(stream: OutputStream, record: MeasurementRecord, timestampIso: String) {
        stream.bufferedWriter(Charsets.UTF_8).use { it.write(buildCsv(record, timestampIso)) }
    }

    private fun appendSource(sb: StringBuilder, record: MeasurementRecord) {
        sb.appendLine("# measurement_source,${record.source.name}")
        val video = record.videoSource ?: return
        sb.appendLine("# video_name,${csv(video.sourceName)}")
        sb.appendLine("# video_uri,${csv(video.sourceUri)}")
        sb.appendLine("# video_width,${video.width}")
        sb.appendLine("# video_height,${video.height}")
        sb.appendLine("# video_rotation_degrees,${video.rotationDegrees}")
        sb.appendLine("# video_nominal_frame_rate,${video.nominalFrameRate?.let { num(it) } ?: ""}")
        sb.appendLine("# video_duration_us,${video.durationUs}")
        sb.appendLine("# video_clip_start_us,${video.clipStartUs}")
        sb.appendLine("# video_clip_end_us,${video.clipEndUs}")
        sb.appendLine("# video_decoded_frames,${video.decodedFrameCount}")
        sb.appendLine("# video_processed_frames,${video.processedFrameCount}")
        sb.appendLine("# video_inference_frames,${video.inferenceFrameCount}")
        sb.appendLine("# video_analysis_task_id,${csv(video.taskId)}")
    }

    private fun appendSummary(sb: StringBuilder, summary: RecordedVelocitySummary?) {
        if (summary == null) return
        sb.appendLine("# plateau_found,${summary.plateauFound}")
        sb.appendLine(
            "# terminal_velocity_mm_s," +
                (summary.terminalVelocityMmPerS?.let { num(it) } ?: "")
        )
        sb.appendLine(
            "# mean_velocity_mm_s," +
                (summary.meanVelocityMmPerS?.let { num(it) } ?: "")
        )
        sb.appendLine(
            "# plateau_range_mm,${num(summary.plateauStartMm)}," +
                num(summary.plateauEndMm)
        )
        sb.appendLine("# plateau_duration_ms,${summary.plateauDurationMs}")
        sb.appendLine("# r2,${num(summary.r2, 4)}")
        sb.appendLine("# velocity_std_dev_mm_s,${num(summary.velocityStdDev)}")
    }

    private fun appendAnalysisQuality(
        sb: StringBuilder,
        quality: VelocityAnalysisQuality?
    ) {
        if (quality == null) return
        sb.appendLine("# effective_sample_rate_fps,${num(quality.effectiveSampleRateFps)}")
        sb.appendLine("# velocity_window_ms,${quality.velocityWindowMs}")
        sb.appendLine("# usable_position_samples,${quality.usablePositionSampleCount}")
        sb.appendLine("# velocity_samples,${quality.velocitySampleCount}")
        sb.appendLine("# outlier_samples,${quality.outlierSampleCount}")
        sb.appendLine("# low_sample_rate,${quality.lowSampleRate}")
        sb.appendLine("# excessive_outliers,${quality.excessiveOutliers}")
        sb.appendLine("# track_discontinuity,${quality.trackDiscontinuity}")
        sb.appendLine(
            "# terminal_fit_relative_disagreement," +
                (quality.terminalFitRelativeDisagreement?.let { num(it, 5) } ?: "")
        )
        sb.appendLine("# terminal_fit_check_attempted,${quality.terminalFitCheckAttempted}")
        sb.appendLine("# unstable_terminal_fit,${quality.unstableTerminalFit}")
        sb.appendLine(
            "# terminal_fit_check_unavailable,${quality.terminalFitCheckUnavailable}"
        )
    }

    private fun parametersOrBlank(value: Double?): String = value?.let { num(it, 8) } ?: ""

    private fun csv(value: String): String =
        if (value.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) {
            "\"${value.replace("\"", "\"\"")}\""
        } else {
            value
        }

    private fun num(v: Float, decimals: Int = 3): String =
        String.format(Locale.ROOT, "%.${decimals}f", v)

    private fun num(v: Double, decimals: Int = 3): String =
        String.format(Locale.ROOT, "%.${decimals}f", v)
}
