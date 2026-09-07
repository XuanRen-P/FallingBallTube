package com.cupet.fallingball.camera

import com.cupet.fallingball.analysis.FallFrameObservation
import com.cupet.fallingball.analysis.FallTrackingGate
import com.cupet.fallingball.analysis.FallTrackingPhase
import com.cupet.fallingball.analysis.FallTrackingRejection
import com.cupet.fallingball.analysis.FallTrackingStatus
import com.cupet.fallingball.vision.BallDetectionSource
import com.cupet.fallingball.vision.BallGeometry
import com.cupet.fallingball.vision.TemporalBallPeak
import com.cupet.fallingball.vision.TemporalDetectionThresholds
import kotlin.math.abs

internal data class MappedTemporalPeak(
    val peak: TemporalBallPeak,
    val axialPx: Float,
    val axialMaxPx: Float,
    val positionTicks: Float?,
    val trajectoryLateralFraction: Float?
)

internal enum class TemporalSelectionOutcome {
    SELECTED,
    BELOW_RECOVERY_THRESHOLD,
    BELOW_PRESENCE_THRESHOLD,
    OUTSIDE_TUBE,
    RULER_POSITION_UNAVAILABLE,
    NO_COHERENT_STARTUP_MOTION,
    NO_PLAUSIBLE_CONTINUATION,
    FINISHED,
    OUT_OF_ORDER
}

internal data class TemporalSelectionDiagnostics(
    val candidateCount: Int = 0,
    val validCandidateCount: Int = 0,
    val startupTrackCount: Int = 0,
    val readyStartupTrackCount: Int = 0,
    val outcome: TemporalSelectionOutcome = TemporalSelectionOutcome.BELOW_RECOVERY_THRESHOLD
)

/**
 * Acquires a moving heatmap peak without letting a brighter static reflection monopolize
 * the shared fall gate. Each local maximum keeps a small independent motion hypothesis;
 * only a hypothesis with measurable downward travel is handed to the recording gate.
 */
internal class TemporalDetectionSelector {
    private data class StartupTrack(
        val id: Long,
        val gate: FallTrackingGate,
        var latest: MappedTemporalPeak,
        var lastTimestampNs: Long,
        var hitCount: Int,
        var scoreTotal: Float,
        var updatedTimestampNs: Long,
        var latestDownwardStepTicks: Float,
        var latestDownwardSpeedTicksPerSecond: Float
    ) {
        val averageScore: Float
            get() = scoreTotal / hitCount.coerceAtLeast(1)

        val ready: Boolean
            get() = gate.status().let { status ->
                status.phase == FallTrackingPhase.TRACKING ||
                    status.phase == FallTrackingPhase.CONFIRMING && status.motionSeeded
            }
    }

    private data class CandidateMatch(
        val trackIndex: Int,
        val candidateIndex: Int,
        val cost: Float
    )

    private val startupTracks = ArrayList<StartupTrack>(MAX_STARTUP_TRACKS)
    private var nextTrackId = 1L
    private var preferredTrackId: Long? = null
    private var lastTimestampNs = Long.MIN_VALUE
    private var cachedSelection: MappedTemporalPeak? = null
    internal var lastDiagnostics = TemporalSelectionDiagnostics()
        private set

    @Synchronized
    fun reset() {
        startupTracks.clear()
        nextTrackId = 1L
        preferredTrackId = null
        lastTimestampNs = Long.MIN_VALUE
        cachedSelection = null
        lastDiagnostics = TemporalSelectionDiagnostics()
    }

    @Synchronized
    fun select(
        primary: MappedTemporalPeak,
        alternatives: List<MappedTemporalPeak>,
        gate: FallTrackingGate,
        timestampNs: Long
    ): MappedTemporalPeak? {
        if (timestampNs == lastTimestampNs) return cachedSelection
        if (timestampNs < lastTimestampNs) {
            lastDiagnostics = lastDiagnostics.copy(outcome = TemporalSelectionOutcome.OUT_OF_ORDER)
            return null
        }

        val status = gate.status()
        val candidates = sequenceOf(primary).plus(alternatives.asSequence()).toList()
        val canRecoverTrack = status.phase == FallTrackingPhase.TRACKING ||
            status.phase == FallTrackingPhase.CONFIRMING && status.motionSeeded
        val currentImageIsStationary = status.phase == FallTrackingPhase.TRACKING &&
            status.lastRejection == FallTrackingRejection.NO_PROGRESS
        val selected = when {
            currentImageIsStationary -> {
                clearStartupTracks()
                null
            }
            canRecoverTrack -> {
                clearStartupTracks()
                selectTrackContinuation(
                    primary,
                    alternatives,
                    gate,
                    timestampNs
                )
            }
            status.phase == FallTrackingPhase.FINISHED -> {
                clearStartupTracks()
                primary.takeIf(::isInsideTube)
            }
            else -> selectStartupCandidate(candidates, status, timestampNs)
        }

        lastTimestampNs = timestampNs
        cachedSelection = selected
        val inside = candidates.filter(::isInsideTube)
        val mapped = inside.filter { it.positionTicks?.isFinite() == true }
        val aboveRecovery = mapped.filter {
            it.peak.score >= TemporalDetectionThresholds.TRACK_RECOVERY_THRESHOLD
        }
        lastDiagnostics = TemporalSelectionDiagnostics(
            candidateCount = candidates.size,
            validCandidateCount = aboveRecovery.size,
            startupTrackCount = startupTracks.size,
            readyStartupTrackCount = startupTracks.count { it.ready },
            outcome = when {
                selected != null -> TemporalSelectionOutcome.SELECTED
                candidates.none {
                    it.peak.score >= TemporalDetectionThresholds.TRACK_RECOVERY_THRESHOLD
                } -> TemporalSelectionOutcome.BELOW_RECOVERY_THRESHOLD
                inside.isEmpty() -> TemporalSelectionOutcome.OUTSIDE_TUBE
                mapped.isEmpty() -> TemporalSelectionOutcome.RULER_POSITION_UNAVAILABLE
                status.phase == FallTrackingPhase.FINISHED ->
                    TemporalSelectionOutcome.FINISHED
                !canRecoverTrack && aboveRecovery.none {
                    it.peak.score >= TemporalDetectionThresholds.PRESENCE_THRESHOLD
                } -> TemporalSelectionOutcome.BELOW_PRESENCE_THRESHOLD
                canRecoverTrack -> TemporalSelectionOutcome.NO_PLAUSIBLE_CONTINUATION
                else -> TemporalSelectionOutcome.NO_COHERENT_STARTUP_MOTION
            }
        )
        return selected
    }

    private fun selectTrackContinuation(
        primary: MappedTemporalPeak,
        alternatives: List<MappedTemporalPeak>,
        gate: FallTrackingGate,
        timestampNs: Long
    ): MappedTemporalPeak? {
        val previous = cachedSelection
        val predictedTicks = gate.predictedContinuationPositionTicks(timestampNs)
        val enforcePrediction = gate.status().phase == FallTrackingPhase.TRACKING &&
            gate.hasRecentMotionEstimate()
        val plausible = sequenceOf(primary)
            .plus(alternatives.asSequence())
            .filter(::isInsideTube)
            .filter { it.peak.score >= TemporalDetectionThresholds.TRACK_RECOVERY_THRESHOLD }
            .filter {
                gate.isPlausibleCandidateContinuation(
                    timestampNs = timestampNs,
                    positionTicks = it.positionTicks,
                    trajectoryLateralFraction = it.trajectoryLateralFraction
                )
            }
            .filter { candidate ->
                val comparableTicks = candidate.positionTicks
                !enforcePrediction || predictedTicks == null || comparableTicks == null ||
                    abs(comparableTicks - predictedTicks) <=
                    MAX_MODEL_CONTINUATION_ERROR_TICKS
            }
            .toList()
        return plausible.minWithOrNull(
            compareBy<MappedTemporalPeak> {
                continuationAssociationCost(predictedTicks, previous, it)
            }.thenByDescending { it.peak.score }
        )
    }

    private fun selectStartupCandidate(
        candidates: List<MappedTemporalPeak>,
        status: FallTrackingStatus,
        timestampNs: Long
    ): MappedTemporalPeak? {
        // This path is used only before the shared gate has a motion seed. A single accepted
        // reflection must not reserve the next several model steps and hide a coherent ball
        // hypothesis that appears behind it.
        preferredTrackId = null

        val validCandidates = candidates.filter(::isValidStartupCandidate)
        val allowedGapMs = status.allowedGapMs.coerceAtLeast(1L)
        val allowedGapNs = allowedGapMs * NANOS_PER_MILLISECOND
        startupTracks.removeAll { track ->
            timestampNs - track.lastTimestampNs > allowedGapNs ||
                track.gate.status().phase == FallTrackingPhase.FINISHED
        }
        if (startupTracks.none { it.id == preferredTrackId }) preferredTrackId = null

        val matches = ArrayList<CandidateMatch>()
        startupTracks.forEachIndexed { trackIndex, track ->
            track.gate.setExpectedObservationIntervalMs(allowedGapMs)
            validCandidates.forEachIndexed { candidateIndex, candidate ->
                if (track.gate.isPlausibleCandidateContinuation(
                        timestampNs = timestampNs,
                        positionTicks = candidate.positionTicks,
                        trajectoryLateralFraction = candidate.trajectoryLateralFraction
                    )
                ) {
                    matches.add(
                        CandidateMatch(
                            trackIndex = trackIndex,
                            candidateIndex = candidateIndex,
                            cost = associationCost(track.latest, candidate)
                        )
                    )
                }
            }
        }
        matches.sortBy { it.cost }

        val assignedTracks = BooleanArray(startupTracks.size)
        val assignedCandidates = BooleanArray(validCandidates.size)
        for (match in matches) {
            if (assignedTracks[match.trackIndex] || assignedCandidates[match.candidateIndex]) {
                continue
            }
            val track = startupTracks[match.trackIndex]
            val candidate = validCandidates[match.candidateIndex]
            val stepTicks = requireNotNull(candidate.positionTicks) -
                requireNotNull(track.latest.positionTicks)
            val elapsedSeconds = (timestampNs - track.lastTimestampNs) / NANOS_PER_SECOND
            track.gate.observe(candidate.asObservation(timestampNs), timestampNs)
            track.latest = candidate
            track.lastTimestampNs = timestampNs
            track.updatedTimestampNs = timestampNs
            track.hitCount += 1
            track.scoreTotal += candidate.peak.score
            track.latestDownwardStepTicks = stepTicks.coerceAtLeast(0f)
            track.latestDownwardSpeedTicksPerSecond = if (elapsedSeconds > 0f) {
                (stepTicks / elapsedSeconds).coerceAtLeast(0f)
            } else {
                0f
            }
            assignedTracks[match.trackIndex] = true
            assignedCandidates[match.candidateIndex] = true
        }

        validCandidates.forEachIndexed { index, candidate ->
            if (!assignedCandidates[index]) {
                startupTracks.add(newStartupTrack(candidate, timestampNs, allowedGapMs))
            }
        }
        trimStartupTracks()

        preferredTrackId?.let { id ->
            startupTracks.firstOrNull { it.id == id }?.let { preferred ->
                if (preferred.updatedTimestampNs == timestampNs && preferred.ready) {
                    return preferred.latest
                }
                if (status.phase != FallTrackingPhase.WAITING || preferred.ready) {
                    return null
                }
            }
            preferredTrackId = null
        }

        val selected = startupTracks
            .asSequence()
            .filter { it.updatedTimestampNs == timestampNs && it.ready }
            .filter {
                status.phase != FallTrackingPhase.WAITING ||
                    it.latest.peak.score >= TemporalDetectionThresholds.PRESENCE_THRESHOLD
            }
            .maxWithOrNull(
                compareBy<StartupTrack> { it.latestDownwardSpeedTicksPerSecond }
                    .thenBy { it.latestDownwardStepTicks }
                    .thenBy { it.latest.peak.score }
                    .thenBy { it.hitCount }
                    .thenBy { it.averageScore }
            )
            ?: return null
        preferredTrackId = selected.id
        return selected.latest
    }

    private fun newStartupTrack(
        candidate: MappedTemporalPeak,
        timestampNs: Long,
        allowedGapMs: Long
    ): StartupTrack {
        val hypothesisGate = FallTrackingGate().apply {
            setExpectedObservationIntervalMs(allowedGapMs)
            observe(candidate.asObservation(timestampNs), timestampNs)
        }
        return StartupTrack(
            id = nextTrackId++,
            gate = hypothesisGate,
            latest = candidate,
            lastTimestampNs = timestampNs,
            hitCount = 1,
            scoreTotal = candidate.peak.score,
            updatedTimestampNs = timestampNs,
            latestDownwardStepTicks = 0f,
            latestDownwardSpeedTicksPerSecond = 0f
        )
    }

    private fun trimStartupTracks() {
        if (startupTracks.size <= MAX_STARTUP_TRACKS) return
        startupTracks.sortWith(
            compareByDescending<StartupTrack> {
                if (it.id == preferredTrackId) 1 else 0
            }
                .thenByDescending { it.lastTimestampNs }
                .thenByDescending { if (it.ready) 1 else 0 }
                .thenByDescending { it.latestDownwardSpeedTicksPerSecond }
                .thenByDescending { it.latest.peak.score }
                .thenByDescending { it.hitCount }
                .thenByDescending { it.averageScore }
        )
        while (startupTracks.size > MAX_STARTUP_TRACKS) {
            startupTracks.removeAt(startupTracks.lastIndex)
        }
    }

    private fun clearStartupTracks() {
        startupTracks.clear()
        preferredTrackId = null
    }

    private fun associationCost(
        previous: MappedTemporalPeak,
        current: MappedTemporalPeak
    ): Float {
        val axial = abs(
            requireNotNull(current.positionTicks) - requireNotNull(previous.positionTicks)
        )
        val lateral = if (
            previous.trajectoryLateralFraction != null &&
            current.trajectoryLateralFraction != null
        ) {
            abs(current.trajectoryLateralFraction - previous.trajectoryLateralFraction)
        } else {
            0f
        }
        return axial + lateral * LATERAL_ASSOCIATION_WEIGHT
    }

    private fun continuationAssociationCost(
        predictedTicks: Float?,
        previous: MappedTemporalPeak?,
        current: MappedTemporalPeak
    ): Float {
        val axialReference = predictedTicks ?: previous?.positionTicks
        val axial = if (axialReference != null) {
            abs(requireNotNull(current.positionTicks) - axialReference)
        } else {
            0f
        }
        val lateral = if (
            previous?.trajectoryLateralFraction != null &&
            current.trajectoryLateralFraction != null
        ) {
            abs(current.trajectoryLateralFraction - previous.trajectoryLateralFraction)
        } else {
            0f
        }
        return axial + lateral * LATERAL_ASSOCIATION_WEIGHT
    }

    private fun isValidStartupCandidate(candidate: MappedTemporalPeak): Boolean =
        isInsideTube(candidate) &&
            candidate.peak.score >= TemporalDetectionThresholds.TRACK_RECOVERY_THRESHOLD &&
            candidate.positionTicks?.isFinite() == true &&
            candidate.progress <= MAX_STARTUP_PROGRESS &&
            candidate.trajectoryLateralFraction?.isFinite() != false

    private val MappedTemporalPeak.progress: Float
        get() = if (axialMaxPx > 0f) {
            (axialPx / axialMaxPx).coerceIn(0f, 1f)
        } else {
            1f
        }

    private fun MappedTemporalPeak.asObservation(timestampNs: Long) = FallFrameObservation(
        timestampNs = timestampNs,
        axialPx = axialPx,
        absolutePositionMm = null,
        referenceOrdinal = null,
        rulerValid = true,
        shiftPx = 0f,
        source = BallDetectionSource.TINY_HEATMAP,
        positionTicks = requireNotNull(positionTicks),
        progressFraction = progress,
        trajectoryLateralFraction = trajectoryLateralFraction
    )

    private fun isInsideTube(candidate: MappedTemporalPeak): Boolean =
        BallGeometry.isInsideTubeAxialRange(candidate.axialPx, candidate.axialMaxPx)

    private companion object {
        private const val MAX_STARTUP_TRACKS = 24
        private const val MAX_STARTUP_PROGRESS = 0.90f
        private const val MAX_MODEL_CONTINUATION_ERROR_TICKS = 1.5f
        private const val LATERAL_ASSOCIATION_WEIGHT = 2f
        private const val NANOS_PER_MILLISECOND = 1_000_000L
        private const val NANOS_PER_SECOND = 1_000_000_000f
    }
}

internal fun temporalDetectionThreshold(status: FallTrackingStatus): Float =
    if (status.hasUsableMotionTrajectory) {
        TemporalDetectionThresholds.TRACK_RECOVERY_THRESHOLD
    } else {
        TemporalDetectionThresholds.PRESENCE_THRESHOLD
    }

/** A seeded confirmation already has enough causal motion evidence for guided image tracking. */
internal val FallTrackingStatus.hasUsableMotionTrajectory: Boolean
    get() = phase == FallTrackingPhase.TRACKING ||
        phase == FallTrackingPhase.CONFIRMING && motionSeeded
