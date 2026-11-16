package com.cupet.fallingball.learning

import com.cupet.fallingball.analysis.MeasurementRecord
import kotlinx.coroutines.flow.Flow
import java.security.MessageDigest
import java.util.UUID

enum class LearningSessionStatus { ACTIVE, PAUSED, ENDED }

enum class LearningMessageRole { USER, ASSISTANT }

enum class LearningMessageState { STREAMING, COMPLETE, STOPPED, ERROR }

enum class LearningActionState { RUNNING, COMPLETE, ERROR }

data class LearningTrajectoryPoint(
    val elapsedMs: Float,
    val positionTicks: Float?,
    val positionMm: Float?,
    val velocityMmPerS: Float?,
    val lateralMm: Float?,
    val rulerValid: Boolean,
    val usedInTerminalFit: Boolean,
    val excludedAsOutlier: Boolean,
    val wallContactState: String,
    val detectionSource: String,
    val detectionConfidence: Float?
)

data class LearningMeasurementFacts(
    val terminalVelocityMmPerS: Float?,
    val meanVelocityMmPerS: Float?,
    val fittedR2: Float?,
    val fitStartMm: Float?,
    val fitEndMm: Float?,
    val fitDurationMs: Long?,
    val correctedViscosityPaS: Double?,
    val uncorrectedViscosityPaS: Double?,
    val wallCorrectionModel: String?,
    val reynoldsNumber: Double?,
    val sampleCount: Int,
    val validSampleCount: Int,
    val effectiveSampleRateFps: Float,
    val outlierSampleCount: Int,
    val wallContactSampleCount: Int,
    val wallContactKnownSampleCount: Int,
    val averageCenterOffsetMm: Float?,
    val sphereDiameterMm: Double?,
    val tubeInnerDiameterMm: Double?,
    val liquidDensityGPerCm3: Double?,
    val sphereDensityGPerCm3: Double?,
    val reliableTerminalEstimate: Boolean,
    val diagnosticsAvailable: Boolean
)

data class LearningMeasurementSnapshot(
    val measurementId: String,
    val fingerprint: String,
    val recordedAtEpochMs: Long,
    val recordName: String,
    val liquidName: String,
    val sphereMaterial: String,
    val temperatureC: Double?,
    val notes: String,
    val resolutionLabel: String,
    val endReason: String,
    val tickSpacingMm: Float,
    val expectedTickCount: Int,
    val facts: LearningMeasurementFacts,
    val diagnostics: Map<String, String>,
    val trajectory: List<LearningTrajectoryPoint>
) {
    companion object {
        fun fromRecord(record: MeasurementRecord): LearningMeasurementSnapshot {
            val summary = record.summary
            val quality = record.analysisQuality
            val viscosity = record.viscosityResult.takeIf { record.hasReliableTerminalEstimate }
            val diagnostic = record.diagnosticDigest
            return LearningMeasurementSnapshot(
                measurementId = record.id,
                fingerprint = MeasurementFingerprint.sha256(record),
                recordedAtEpochMs = record.recordedAtEpochMs,
                recordName = record.metadata.recordName,
                liquidName = record.metadata.liquidName,
                sphereMaterial = record.metadata.sphereMaterial,
                temperatureC = record.metadata.temperatureC,
                notes = record.metadata.notes,
                resolutionLabel = record.resolutionLabel,
                endReason = record.endReason.name,
                tickSpacingMm = record.config.tickSpacingMm,
                expectedTickCount = record.config.expectedTickCount,
                facts = LearningMeasurementFacts(
                    terminalVelocityMmPerS = summary?.terminalVelocityMmPerS,
                    meanVelocityMmPerS = summary?.meanVelocityMmPerS,
                    fittedR2 = summary?.r2,
                    fitStartMm = summary?.plateauStartMm,
                    fitEndMm = summary?.plateauEndMm,
                    fitDurationMs = summary?.plateauDurationMs,
                    correctedViscosityPaS = viscosity?.wallCorrectedStokesPaS,
                    uncorrectedViscosityPaS = viscosity?.oseenViscosityPaS,
                    wallCorrectionModel = viscosity?.wallCorrectionModel,
                    reynoldsNumber = viscosity?.reynoldsNumber,
                    sampleCount = record.samples.size,
                    validSampleCount = record.validSampleCount,
                    effectiveSampleRateFps = quality?.effectiveSampleRateFps ?: record.meanFps,
                    outlierSampleCount = quality?.outlierSampleCount
                        ?: record.samples.count { it.excludedAsOutlier },
                    wallContactSampleCount = record.wallContactSampleCount,
                    wallContactKnownSampleCount = record.wallContactKnownSampleCount,
                    averageCenterOffsetMm = record.averageCenterOffsetMm,
                    sphereDiameterMm = record.viscometryParameters.sphereDiameterMm,
                    tubeInnerDiameterMm = record.viscometryParameters.tubeInnerDiameterMm,
                    liquidDensityGPerCm3 = record.viscometryParameters.liquidDensityGPerCm3,
                    sphereDensityGPerCm3 = record.viscometryParameters.sphereDensityGPerCm3,
                    reliableTerminalEstimate = record.hasReliableTerminalEstimate,
                    diagnosticsAvailable = diagnostic != null
                ),
                diagnostics = if (diagnostic == null) {
                    mapOf("availability" to "unavailable")
                } else {
                    buildMap {
                        put("availability", "available")
                        put("observed_samples", diagnostic.observedSampleCount.toString())
                        put("ruler_valid_samples", diagnostic.rulerValidSampleCount.toString())
                        put("local_tracker_samples", diagnostic.localTrackerSampleCount.toString())
                        put("heatmap_samples", diagnostic.heatmapSampleCount.toString())
                        put("other_detector_samples", diagnostic.otherDetectorSampleCount.toString())
                        put("rejected_outliers", diagnostic.rejectedOutlierCount.toString())
                        put("wall_contact_samples", diagnostic.wallContactSampleCount.toString())
                        diagnostic.medianDetectionConfidence?.let {
                            put("median_detection_confidence", it.toString())
                        }
                        put("peak_registration_shift_px", diagnostic.peakRegistrationShiftPx.toString())
                        put("excessive_registration_shake", diagnostic.excessiveRegistrationShake.toString())
                        diagnostic.schedulerSkipCount?.let { put("scheduler_skips", it.toString()) }
                        diagnostic.selectorRejectionCount?.let { put("selector_rejections", it.toString()) }
                        diagnostic.gateRejectionCount?.let { put("gate_rejections", it.toString()) }
                    }
                },
                trajectory = record.samples.map { sample ->
                    LearningTrajectoryPoint(
                        elapsedMs = sample.elapsedMs,
                        positionTicks = sample.positionTicks,
                        positionMm = sample.positionMm,
                        velocityMmPerS = sample.velocityMmPerS,
                        lateralMm = sample.lateralMm,
                        rulerValid = sample.rulerValid,
                        usedInTerminalFit = sample.usedInTerminalFit,
                        excludedAsOutlier = sample.excludedAsOutlier,
                        wallContactState = sample.wallContactState.name,
                        detectionSource = sample.detectionSource.name,
                        detectionConfidence = sample.detectionConfidence
                    )
                }
            )
        }
    }
}

data class LearningState(
    val currentGoal: String = "理解本次落球法测量中的物理关系与数据质量",
    val unresolvedQuestions: List<String> = emptyList(),
    val masteredConcepts: List<String> = emptyList(),
    val learnerLevel: String = "unknown",
    val updatedAtEpochMs: Long = System.currentTimeMillis()
)

data class LearningMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: LearningMessageRole,
    val text: String,
    /** The model's streamed reasoning, kept separate so it can be collapsed in the UI. */
    val reasoning: String = "",
    /** User-facing reason persisted when this message ends in [LearningMessageState.ERROR]. */
    val failureReason: String = "",
    /** Display-only attachment metadata; binary content is sent once and never stored here. */
    val attachmentNames: List<String> = emptyList(),
    /** Stable ownership key shared by the user message, assistant message, and tool actions. */
    val turnId: String = "",
    val createdAtEpochMs: Long = System.currentTimeMillis(),
    val state: LearningMessageState = LearningMessageState.COMPLETE
)

/** Identity allocated once by the caller so streaming and persisted rows stay the same item. */
data class LearningTurnIdentity(
    val turnId: String = UUID.randomUUID().toString(),
    val userMessageId: String = UUID.randomUUID().toString(),
    val assistantMessageId: String = UUID.randomUUID().toString()
)

data class LearningAction(
    val id: String = UUID.randomUUID().toString(),
    val turnId: String,
    val toolName: String,
    val label: String,
    /** Responses function-call identity and bounded replay payload for later turns. */
    val callId: String = "",
    val arguments: String = "{}",
    val output: String = "",
    /** Complete Responses output item for built-in/MCP calls that have no function output pair. */
    val outputItemJson: String = "",
    val startedAtEpochMs: Long = System.currentTimeMillis(),
    val completedAtEpochMs: Long? = null,
    val state: LearningActionState = LearningActionState.RUNNING,
    val resultSummary: String = ""
)

data class LearningReference(
    val id: String = UUID.randomUUID().toString(),
    val turnId: String,
    val title: String,
    val source: String,
    val url: String? = null
)

data class LearningSession(
    val id: String = UUID.randomUUID().toString(),
    val measurementId: String,
    val measurementFingerprint: String,
    val title: String,
    val titleUserRenamed: Boolean = false,
    val titleAutoGenerated: Boolean = false,
    val status: LearningSessionStatus = LearningSessionStatus.ACTIVE,
    val createdAtEpochMs: Long = System.currentTimeMillis(),
    val updatedAtEpochMs: Long = createdAtEpochMs,
    val snapshot: LearningMeasurementSnapshot,
    val learningState: LearningState = LearningState(),
    val rollingSummary: String = "",
    val messages: List<LearningMessage> = emptyList(),
    val actions: List<LearningAction> = emptyList(),
    val references: List<LearningReference> = emptyList()
) {
    val hasConversation: Boolean get() = messages.isNotEmpty()
    val needsGeneratedTitle: Boolean
        get() = hasConversation && !titleUserRenamed && !titleAutoGenerated
}

data class PreparedQuestions(
    val measurementId: String,
    val fingerprint: String,
    val questions: List<String>,
    val generatedAtEpochMs: Long = System.currentTimeMillis()
)

data class LearningInputAttachment(
    val name: String,
    val mediaType: String,
    val base64: String
)

sealed interface LearningEvent {
    data class ActionStarted(val action: LearningAction) : LearningEvent
    data class ActionCompleted(val action: LearningAction) : LearningEvent
    data object ReasoningStarted : LearningEvent
    data class ReasoningDelta(val text: String) : LearningEvent
    data class TextDelta(val text: String) : LearningEvent
    data class Citation(val reference: LearningReference) : LearningEvent
    data class Completed(val session: LearningSession) : LearningEvent
    data class Stopped(val session: LearningSession) : LearningEvent
    data class Error(val message: String, val recoverable: Boolean) : LearningEvent
}

interface LearningTutor {
    suspend fun prepareQuestions(
        record: MeasurementRecord,
        chartPngBase64: String? = null
    ): PreparedQuestions
    suspend fun openSession(record: MeasurementRecord, sessionId: String? = null): LearningSession
    suspend fun startNewSession(record: MeasurementRecord): LearningSession
    fun send(
        sessionId: String,
        message: String,
        chartPngBase64: String? = null,
        attachments: List<LearningInputAttachment> = emptyList(),
        turnIdentity: LearningTurnIdentity = LearningTurnIdentity()
    ): Flow<LearningEvent>
    fun retry(
        sessionId: String,
        assistantMessageId: String,
        chartPngBase64: String? = null,
        attachments: List<LearningInputAttachment> = emptyList(),
        turnIdentity: LearningTurnIdentity = LearningTurnIdentity()
    ): Flow<LearningEvent>
    fun stopGenerating()
    suspend fun endSession(sessionId: String): LearningSession
}

object LearningQuestionValidator {
    const val MAX_QUESTION_LENGTH = 48

    fun validate(values: List<String>): List<String> {
        val cleaned = values.map { it.trim() }.filter { it.isNotEmpty() }
        require(cleaned.size == 3) { "Exactly three questions are required" }
        require(cleaned.distinct().size == 3) { "Questions must be unique" }
        require(cleaned.all { it.length <= MAX_QUESTION_LENGTH }) {
            "Question is too long"
        }
        return cleaned
    }
}

object MeasurementFingerprint {
    fun sha256(record: MeasurementRecord): String {
        val digest = MessageDigest.getInstance("SHA-256")
        fun add(value: Any?) {
            val text = when (value) {
                null -> "null"
                is Float -> "f:${value.toRawBits()}"
                is Double -> "d:${value.toRawBits()}"
                else -> value.toString()
            }
            digest.update(text.toByteArray(Charsets.UTF_8))
            digest.update(0)
        }
        add(record.id)
        add(record.recordedAtEpochMs)
        add(record.config.tickSpacingMm)
        add(record.config.expectedTickCount)
        add(record.resolutionLabel)
        add(record.endReason.name)
        with(record.metadata) {
            add(recordName); add(liquidName); add(sphereMaterial); add(temperatureC); add(notes)
        }
        with(record.viscometryParameters) {
            add(sphereRadiusMm); add(tubeInnerDiameterMm)
            add(liquidDensityGPerCm3); add(sphereDensityGPerCm3)
        }
        record.summary?.let {
            add(it.terminalVelocityMmPerS); add(it.plateauFound); add(it.plateauStartMm)
            add(it.plateauEndMm); add(it.plateauDurationMs); add(it.r2)
            add(it.velocityStdDev); add(it.sampleCount); add(it.validSampleCount)
            add(it.meanVelocityMmPerS)
        } ?: add(null)
        record.analysisQuality?.let {
            add(it.effectiveSampleRateFps); add(it.velocityWindowMs)
            add(it.usablePositionSampleCount); add(it.velocitySampleCount)
            add(it.outlierSampleCount); add(it.trackDiscontinuity)
            add(it.terminalFitRelativeDisagreement); add(it.terminalFitCheckAttempted)
        } ?: add(null)
        record.viscosityResult?.let {
            add(it.wallCorrectedStokesPaS); add(it.oseenViscosityPaS)
            add(it.faxenFactor); add(it.unboundedVelocityMmPerS)
            add(it.reynoldsNumber); add(it.diameterRatio); add(it.wallCorrectionModel)
        } ?: add(null)
        record.samples.forEach {
            add(it.elapsedMs); add(it.positionTicks); add(it.positionMm); add(it.velocityMmPerS)
            add(it.axialPx); add(it.rulerValid); add(it.extrapolated); add(it.shiftPx)
            add(it.usedInTerminalFit); add(it.excludedAsOutlier); add(it.lateralPx)
            add(it.lateralMm); add(it.normalizedLateral); add(it.wallClearanceMm)
            add(it.wallContactState.name); add(it.detectionSource.name)
            add(it.detectionConfidence)
        }
        record.diagnosticDigest?.let {
            add(it.observedSampleCount); add(it.rulerValidSampleCount)
            add(it.localTrackerSampleCount); add(it.heatmapSampleCount)
            add(it.otherDetectorSampleCount); add(it.rejectedOutlierCount)
            add(it.wallContactSampleCount); add(it.medianDetectionConfidence)
            add(it.peakRegistrationShiftPx); add(it.excessiveRegistrationShake)
            add(it.schedulerSkipCount); add(it.selectorRejectionCount); add(it.gateRejectionCount)
        } ?: add(null)
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
