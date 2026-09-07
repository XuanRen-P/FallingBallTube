package com.cupet.fallingball.analysis

import com.cupet.fallingball.vision.BallEndReason
import com.cupet.fallingball.vision.BallDetectionSource
import com.cupet.fallingball.vision.BallGeometry
import com.cupet.fallingball.vision.RulerConfig
import com.cupet.fallingball.vision.TinyPipelineStage
import com.cupet.fallingball.vision.TinyPipelineStageTimer
import com.cupet.fallingball.vision.WallContactState
import kotlin.math.abs
import kotlin.math.roundToLong

data class MeasurementMetadata(
    val recordName: String = "",
    val liquidName: String = "",
    val sphereMaterial: String = "",
    val temperatureC: Double? = null,
    val notes: String = ""
)

/** Optional values copied into a record when a new drop begins. */
data class MeasurementDefaults(
    val viscometryParameters: ViscometryParameters = ViscometryParameters(),
    val liquidName: String = "",
    val sphereMaterial: String = "",
    val temperatureC: Double? = null
) {
    fun metadataSnapshot() = MeasurementMetadata(
        liquidName = liquidName,
        sphereMaterial = sphereMaterial,
        temperatureC = temperatureC
    )
}

data class RecordedVelocitySummary(
    val terminalVelocityMmPerS: Float?,
    val plateauFound: Boolean,
    val plateauStartMm: Float,
    val plateauEndMm: Float,
    val plateauDurationMs: Long,
    val r2: Float,
    val velocityStdDev: Float,
    val sampleCount: Int,
    val validSampleCount: Int,
    val meanVelocityMmPerS: Float?
)

data class RecordedBallSample(
    val elapsedMs: Float,
    val positionTicks: Float?,
    val positionMm: Float?,
    val velocityMmPerS: Float?,
    val axialPx: Float,
    val rulerValid: Boolean,
    val extrapolated: Boolean,
    val shiftPx: Float,
    val usedInTerminalFit: Boolean,
    val excludedAsOutlier: Boolean = false,
    val lateralPx: Float? = null,
    val lateralMm: Float? = null,
    val normalizedLateral: Float? = null,
    val wallClearanceMm: Float? = null,
    val wallContactState: WallContactState = WallContactState.UNKNOWN,
    val detectionSource: BallDetectionSource = BallDetectionSource.CONVGRU,
    val detectionConfidence: Float? = null
)

/**
 * Bounded, non-image evidence retained for teaching and quality explanations.
 *
 * Pipeline counters are nullable because records written before FBR v11 cannot reconstruct
 * scheduler or gate activity. Sample-derived fields are always available for new records.
 */
data class MeasurementDiagnosticDigest(
    val observedSampleCount: Int,
    val rulerValidSampleCount: Int,
    val localTrackerSampleCount: Int,
    val heatmapSampleCount: Int,
    val otherDetectorSampleCount: Int,
    val rejectedOutlierCount: Int,
    val wallContactSampleCount: Int,
    val medianDetectionConfidence: Float?,
    val peakRegistrationShiftPx: Float,
    val excessiveRegistrationShake: Boolean,
    val schedulerSkipCount: Int? = null,
    val selectorRejectionCount: Int? = null,
    val gateRejectionCount: Int? = null
) {
    companion object {
        fun fromRecord(
            record: MeasurementRecord,
            pipeline: MeasurementPipelineDiagnostics? = null
        ): MeasurementDiagnosticDigest {
            val confidence = record.samples.mapNotNull { it.detectionConfidence }
                .filter(Float::isFinite)
                .sorted()
            val median = when {
                confidence.isEmpty() -> null
                confidence.size % 2 == 1 -> confidence[confidence.size / 2]
                else -> (confidence[confidence.size / 2 - 1] +
                    confidence[confidence.size / 2]) / 2f
            }
            return MeasurementDiagnosticDigest(
                observedSampleCount = record.samples.size,
                rulerValidSampleCount = record.samples.count { it.rulerValid },
                localTrackerSampleCount = record.samples.count {
                    it.detectionSource == BallDetectionSource.LOCAL_TRACKER
                },
                heatmapSampleCount = record.samples.count {
                    it.detectionSource == BallDetectionSource.CLASSICAL_HEATMAP ||
                        it.detectionSource == BallDetectionSource.TINY_HEATMAP
                },
                otherDetectorSampleCount = record.samples.count {
                    it.detectionSource != BallDetectionSource.LOCAL_TRACKER &&
                        it.detectionSource != BallDetectionSource.CLASSICAL_HEATMAP &&
                        it.detectionSource != BallDetectionSource.TINY_HEATMAP
                },
                rejectedOutlierCount = record.samples.count { it.excludedAsOutlier },
                wallContactSampleCount = record.samples.count { it.wallContactState.isContact() },
                medianDetectionConfidence = median,
                peakRegistrationShiftPx = record.peakShiftPx,
                excessiveRegistrationShake = record.hasExcessiveRegistrationShake,
                schedulerSkipCount = pipeline?.schedulerSkipCount,
                selectorRejectionCount = pipeline?.selectorRejectionCount,
                gateRejectionCount = pipeline?.gateRejectionCount
            )
        }

        private fun WallContactState.isContact(): Boolean =
            this == WallContactState.LEFT_WALL || this == WallContactState.RIGHT_WALL
    }
}

data class WallContactEpisodeSummary(
    val sampleCount: Int,
    val startElapsedMs: Float,
    val endElapsedMs: Float,
    val startPositionMm: Float?,
    val endPositionMm: Float?,
    val minimumClearanceMm: Float?
) {
    val durationMs: Float get() = (endElapsedMs - startElapsedMs).coerceAtLeast(0f)
}

/**
 * Durable, Android-independent snapshot of one completed [FallSession].
 *
 * Capture timestamps are converted to elapsed milliseconds because the camera's monotonic
 * clock has no meaning after the process exits. Everything needed for history display and
 * later CSV reconstruction remains in the snapshot.
 */
data class MeasurementRecord(
    val id: String,
    val recordedAtEpochMs: Long,
    val config: RulerConfig,
    val resolutionLabel: String,
    val endReason: BallEndReason,
    val summary: RecordedVelocitySummary?,
    val samples: List<RecordedBallSample>,
    val metadata: MeasurementMetadata = MeasurementMetadata(),
    val viscometryParameters: ViscometryParameters = ViscometryParameters(),
    val viscosityResult: ViscosityResult? = null,
    val analysisQuality: VelocityAnalysisQuality? = null,
    val analysisMethodVersion: Int = VelocityEstimator.ANALYSIS_METHOD_VERSION,
    val diagnosticDigest: MeasurementDiagnosticDigest? = null,
    val source: MeasurementSource = MeasurementSource.UNKNOWN,
    val videoSource: VideoMeasurementSource? = null
) {
    init {
        validateMeasurementSource(source, videoSource)
    }

    val validSampleCount: Int
        get() = samples.count { it.rulerValid && it.positionMm != null }

    val meanFps: Float
        get() {
            if (samples.size < 2) return 0f
            val spanSeconds = (samples.last().elapsedMs - samples.first().elapsedMs) / 1_000f
            return if (spanSeconds <= 0f) 0f else (samples.size - 1) / spanSeconds
        }

    val peakShiftPx: Float
        get() = samples.maxOfOrNull { kotlin.math.abs(it.shiftPx) } ?: 0f

    val hasExcessiveRegistrationShake: Boolean
        get() = RegistrationShakeAssessment.shouldWarn(samples.map { it.shiftPx })

    val wallContactSampleCount: Int
        get() = samples.count { it.wallContactState.isContact() }

    val twoDimensionalSampleCount: Int
        get() = samples.count { it.lateralPx != null }

    val wallContactKnownSampleCount: Int
        get() = samples.count { it.wallContactState != WallContactState.UNKNOWN }

    val sustainedWallContact: Boolean
        get() = samples.hasSustainedWallContact { it.wallContactState }

    val averageCenterOffsetMm: Float?
        get() = samples.mapNotNull(::centerOffsetMm).takeIf { it.isNotEmpty() }
            ?.average()
            ?.toFloat()

    fun centerOffsetMm(sample: RecordedBallSample): Float? {
        val lateralMm = sample.lateralMm?.takeIf(Float::isFinite) ?: return null
        val tubeRadiusMm = viscometryParameters.tubeInnerDiameterMm
            ?.takeIf { it.isFinite() && it > 0.0 }
            ?.div(2.0)
            ?.toFloat()
        val offsetMm = abs(lateralMm)
        return tubeRadiusMm?.let(offsetMm::coerceAtMost) ?: offsetMm
    }

    val minimumWallClearanceMm: Float?
        get() = samples.mapNotNull { it.wallClearanceMm }.minOrNull()

    val longestWallContactEpisode: WallContactEpisodeSummary?
        get() {
            var bestStart = -1
            var bestEnd = -1
            var currentStart = -1
            for (index in 0..samples.size) {
                val contact = index < samples.size && samples[index].wallContactState.isContact()
                if (contact && currentStart < 0) currentStart = index
                if (!contact && currentStart >= 0) {
                    val currentEnd = index - 1
                    if (bestStart < 0 || currentEnd - currentStart > bestEnd - bestStart) {
                        bestStart = currentStart
                        bestEnd = currentEnd
                    }
                    currentStart = -1
                }
            }
            if (bestStart < 0) return null
            val episode = samples.subList(bestStart, bestEnd + 1)
            return WallContactEpisodeSummary(
                sampleCount = episode.size,
                startElapsedMs = episode.first().elapsedMs,
                endElapsedMs = episode.last().elapsedMs,
                startPositionMm = episode.first().positionMm,
                endPositionMm = episode.last().positionMm,
                minimumClearanceMm = episode.mapNotNull { it.wallClearanceMm }.minOrNull()
            )
        }

    val hasReliableTerminalEstimate: Boolean
        get() = summary?.let {
            VelocityEstimator.isReliableTerminalEstimate(
                plateauFound = it.plateauFound,
                terminalVelocityMmPerS = it.terminalVelocityMmPerS,
                validSampleCount = it.validSampleCount,
                quality = analysisQuality
            )
        } == true

    /**
     * Applies editable measurement parameters to immutable tick/time observations.
     *
     * Every millimetre value, velocity, terminal-fit membership, and viscosity result is
     * regenerated. The raw tick coordinate is never overwritten by an edit.
     */
    fun recalculate(
        config: RulerConfig,
        viscometryParameters: ViscometryParameters,
        metadata: MeasurementMetadata
    ): MeasurementRecord {
        require(config.tickSpacingMm > 0f &&
            config.tickSpacingMm <= RulerConfig.MAX_SPACING_MM
        ) { "Invalid tick spacing" }

        val distanceScale = config.tickSpacingMm / this.config.tickSpacingMm
        val geometryChanged =
            viscometryParameters.sphereRadiusMm != this.viscometryParameters.sphereRadiusMm ||
                viscometryParameters.tubeInnerDiameterMm !=
                this.viscometryParameters.tubeInnerDiameterMm
        val analysisSamples = samples.map { sample ->
            val lateralMm = sample.lateralMm?.times(distanceScale)
            val wall = lateralMm?.let {
                BallGeometry.classifyWallContactMm(
                    lateralMm = it,
                    tubeInnerDiameterMm = viscometryParameters.tubeInnerDiameterMm,
                    sphereRadiusMm = viscometryParameters.sphereRadiusMm
                )
            }
            BallSample(
                tNs = (sample.elapsedMs * 1_000_000.0).roundToLong(),
                axialPx = sample.axialPx,
                positionMm = sample.positionTicks?.times(config.tickSpacingMm),
                rulerValid = sample.rulerValid,
                extrapolated = sample.extrapolated,
                shiftPx = sample.shiftPx,
                positionTicks = sample.positionTicks,
                lateralPx = sample.lateralPx,
                lateralMm = lateralMm,
                normalizedLateral = if (wall != null) {
                    wall.normalizedLateral
                } else {
                    sample.normalizedLateral
                },
                wallClearanceMm = if (wall != null) {
                    wall.clearanceMm
                } else {
                    sample.wallClearanceMm?.times(distanceScale)
                },
                wallContactState = when {
                    wall != null -> wall.state
                    geometryChanged && sample.lateralMm == null -> WallContactState.UNKNOWN
                    else -> sample.wallContactState
                },
                detectionSource = sample.detectionSource,
                detectionConfidence = sample.detectionConfidence
            )
        }
        val analysis = VelocityEstimator.analyze(analysisSamples)
        val recordedSamples = analysisSamples.mapIndexed { index, sample ->
            RecordedBallSample(
                elapsedMs = samples[index].elapsedMs,
                positionTicks = sample.positionTicks,
                positionMm = sample.positionMm,
                velocityMmPerS = sample.velocityMmPerS,
                axialPx = sample.axialPx,
                rulerValid = sample.rulerValid,
                extrapolated = sample.extrapolated,
                shiftPx = sample.shiftPx,
                usedInTerminalFit = analysis.usedInTerminalFit[index],
                excludedAsOutlier = analysis.excludedAsOutlier[index],
                lateralPx = sample.lateralPx,
                lateralMm = sample.lateralMm,
                normalizedLateral = sample.normalizedLateral,
                wallClearanceMm = sample.wallClearanceMm,
                wallContactState = sample.wallContactState,
                detectionSource = sample.detectionSource,
                detectionConfidence = sample.detectionConfidence
            )
        }
        val recordedSummary = analysis.summary.toRecorded()
        val reliableTerminal = VelocityEstimator.isReliableTerminalEstimate(
            plateauFound = recordedSummary.plateauFound,
            terminalVelocityMmPerS = recordedSummary.terminalVelocityMmPerS,
            validSampleCount = recordedSummary.validSampleCount,
            quality = analysis.quality
        )
        val viscosity = if (reliableTerminal) {
            runCatching {
                ViscosityCalculator.calculate(
                    parameters = viscometryParameters,
                    terminalVelocityMmPerS =
                        recordedSummary.terminalVelocityMmPerS?.toDouble()
                )?.withRecordedWallContactWarning(recordedSamples)
            }.getOrNull()
        } else {
            null
        }
        val updated = copy(
            config = config,
            summary = recordedSummary,
            samples = recordedSamples,
            metadata = metadata,
            viscometryParameters = viscometryParameters,
            viscosityResult = viscosity,
            analysisQuality = analysis.quality,
            analysisMethodVersion = VelocityEstimator.ANALYSIS_METHOD_VERSION,
            diagnosticDigest = null
        )
        val refreshedDigest = diagnosticDigest?.let { previous ->
            MeasurementDiagnosticDigest.fromRecord(updated).copy(
                schedulerSkipCount = previous.schedulerSkipCount,
                selectorRejectionCount = previous.selectorRejectionCount,
                gateRejectionCount = previous.gateRejectionCount
            )
        }
        return updated.copy(diagnosticDigest = refreshedDigest)
    }

    companion object {
        fun fromSession(
            session: FallSession,
            recordedAtEpochMs: Long,
            id: String,
            viscometryParameters: ViscometryParameters = ViscometryParameters(),
            metadata: MeasurementMetadata = MeasurementMetadata(),
            stageTimer: TinyPipelineStageTimer = TinyPipelineStageTimer.disabled()
        ): MeasurementRecord {
            val recordBuildStartedNs = stageTimer.start()
            val endReason = requireNotNull(session.endReason) {
                "Only completed sessions can be stored"
            }
            val sourceSamples = session.samples
            val t0 = sourceSamples.firstOrNull()?.tNs ?: session.startNs
            val sampleCopyStartedNs = stageTimer.start()
            val recordedSamples = sourceSamples.map { sample ->
                RecordedBallSample(
                    elapsedMs = (sample.tNs - t0) / 1_000_000f,
                    positionTicks = sample.positionTicks
                        ?: sample.positionMm?.div(session.config.tickSpacingMm),
                    positionMm = sample.positionMm,
                    velocityMmPerS = sample.velocityMmPerS,
                    axialPx = sample.axialPx,
                    rulerValid = sample.rulerValid,
                    extrapolated = sample.extrapolated,
                    shiftPx = sample.shiftPx,
                    usedInTerminalFit = sample.usedInTerminalFit,
                    excludedAsOutlier = sample.excludedAsOutlier,
                    lateralPx = sample.lateralPx,
                    lateralMm = sample.lateralMm,
                    normalizedLateral = sample.normalizedLateral,
                    wallClearanceMm = sample.wallClearanceMm,
                    wallContactState = sample.wallContactState,
                    detectionSource = sample.detectionSource,
                    detectionConfidence = sample.detectionConfidence
                )
            }
            stageTimer.recordSince(TinyPipelineStage.RECORD_SAMPLE_COPY, sampleCopyStartedNs)
            val viscosityStartedNs = stageTimer.start()
            val viscosity = if (
                session.summary?.let {
                    VelocityEstimator.isReliableTerminalEstimate(
                        plateauFound = it.plateauFound,
                        terminalVelocityMmPerS = it.terminalVelocityMmPerS,
                        validSampleCount = it.validSampleCount,
                        quality = session.analysisQuality
                    )
                } == true
            ) {
                runCatching {
                    ViscosityCalculator.calculate(
                        parameters = viscometryParameters,
                        terminalVelocityMmPerS = session.summary?.terminalVelocityMmPerS?.toDouble()
                    )?.withSessionWallContactWarning(sourceSamples)
                }.getOrNull()
            } else {
                null
            }
            stageTimer.recordSince(TinyPipelineStage.VISCOSITY_CALCULATION, viscosityStartedNs)
            val record = MeasurementRecord(
                id = id,
                recordedAtEpochMs = recordedAtEpochMs,
                config = session.config,
                resolutionLabel = session.resolutionLabel,
                endReason = endReason,
                summary = session.summary?.toRecorded(),
                samples = recordedSamples,
                metadata = metadata,
                viscometryParameters = viscometryParameters,
                analysisQuality = session.analysisQuality,
                viscosityResult = viscosity,
                source = MeasurementSource.LIVE_CAMERA
            )
            val completed = record.copy(
                diagnosticDigest = MeasurementDiagnosticDigest.fromRecord(
                    record,
                    session.pipelineDiagnostics
                )
            )
            stageTimer.recordSince(TinyPipelineStage.RECORD_BUILD_TOTAL, recordBuildStartedNs)
            return completed
        }

        internal fun VelocitySummary.toRecorded() = RecordedVelocitySummary(
            terminalVelocityMmPerS = terminalVelocityMmPerS,
            plateauFound = plateauFound,
            plateauStartMm = plateauStartMm,
            plateauEndMm = plateauEndMm,
            plateauDurationMs = plateauDurationMs,
            r2 = r2,
            velocityStdDev = velocityStdDev,
            sampleCount = sampleCount,
            validSampleCount = validSampleCount,
            meanVelocityMmPerS = meanVelocityMmPerS
        )

        private const val MIN_WALL_CONTACT_WARNING_SAMPLES = 3

        private fun WallContactState.isContact(): Boolean =
            this == WallContactState.LEFT_WALL || this == WallContactState.RIGHT_WALL

        private fun ViscosityResult.withSessionWallContactWarning(
            samples: List<BallSample>
        ): ViscosityResult = if (
            samples.hasSustainedWallContact { it.wallContactState }
        ) {
            copy(warnings = warnings + ViscosityWarning.OFF_AXIS_OR_WALL_CONTACT)
        } else {
            this
        }

        private fun ViscosityResult.withRecordedWallContactWarning(
            samples: List<RecordedBallSample>
        ): ViscosityResult = if (
            samples.hasSustainedWallContact { it.wallContactState }
        ) {
            copy(warnings = warnings + ViscosityWarning.OFF_AXIS_OR_WALL_CONTACT)
        } else {
            this
        }

        private inline fun <T> Iterable<T>.hasSustainedWallContact(
            stateOf: (T) -> WallContactState
        ): Boolean {
            var consecutive = 0
            for (sample in this) {
                consecutive = if (stateOf(sample).isContact()) consecutive + 1 else 0
                if (consecutive >= MIN_WALL_CONTACT_WARNING_SAMPLES) return true
            }
            return false
        }
    }
}
