package com.cupet.fallingball.vision

import kotlin.math.abs

/** Primary-detector position expressed independently of one particular OBB pixel grid. */
data class DenseBallAnchor(
    val timestampNs: Long,
    val positionTicks: Float,
    val lateralFraction: Float,
    val confidence: Float
)

/** Camera-frame observation derived from current image pixels, never interpolation. */
data class DenseTrackedBall(
    val timestampNs: Long,
    val axialPx: Float,
    val positionTicks: Float,
    val lateralPx: Float,
    val lateralFraction: Float,
    val confidence: Float
)

internal data class DenseTrackerDiagnostics(
    val backgroundReadyBands: Int = 0,
    val candidateCount: Int = 0,
    val predictionAvailable: Boolean = false,
    val plausibleCandidateCount: Int = 0,
    val predictedTicks: Float? = null,
    val closestCandidateErrorTicks: Float? = null,
    val closestCandidateTicks: Float? = null,
    val startupTrackCount: Int = 0,
    val readyStartupTrackCount: Int = 0,
    val unanchoredAcquisitionReady: Boolean = false,
    val recoveryTrackProgress: Float? = null,
    val recoveryAnchorAgeSeconds: Float? = null,
    val recoveryAllowed: Boolean? = null
)

internal data class DenseCandidateDiagnostics(
    val positionTicks: Float,
    val lateralFraction: Float,
    val confidence: Float
)

/**
 * Camera-rate 2D tracker backed by independent ruler-registered profiles across the tube.
 * Current-image residuals can acquire the trajectory independently. Fresh primary-detector
 * observations provide an optional initial anchor and periodic correction.
 */
class DenseRegisteredBallTracker(
    lateralFractions: FloatArray,
    private val captureCandidateDiagnostics: Boolean = false
) {
    private data class Candidate(
        val bandIndex: Int,
        val axialPx: Float,
        val positionTicks: Float,
        val lateralFraction: Float,
        val confidence: Float
    )

    private data class StartupTrack(
        val id: Long,
        val firstPositionTicks: Float,
        val firstTimestampNs: Long,
        var latest: Candidate,
        var lastTimestampNs: Long,
        var hitCount: Int,
        var confidenceTotal: Float,
        var downwardStreakHits: Int,
        var downwardStreakTravelTicks: Float,
        var downwardStreakStartTimestampNs: Long
    ) {
        val downwardTravelTicks: Float
            get() = latest.positionTicks - firstPositionTicks

        val averageConfidence: Float
            get() = confidenceTotal / hitCount.coerceAtLeast(1)
    }

    private data class StartupMatch(
        val trackIndex: Int,
        val candidateIndex: Int,
        val cost: Float
    )

    private data class StartupSelection(
        val candidate: Candidate?,
        val trackCount: Int,
        val readyTrackCount: Int,
        val selectedSpeedTicksPerSecond: Float?
    )

    private val fractions = lateralFractions.copyOf()
    private val detectors = Array(fractions.size) { RegisteredBallDetector() }
    private var lastAnchor: DenseBallAnchor? = null
    private var anchorVelocityTicksPerSecond = 0f
    private var anchorLateralVelocityPerSecond = 0f
    private var lastTrack: DenseTrackedBall? = null
    private var trackStartPositionTicks: Float? = null
    private var committedTrackObservations = 0
    private var trackVelocityTicksPerSecond = 0f
    private var trackLateralVelocityPerSecond = 0f
    private val recentTrackVelocities = ArrayList<Float>(MAX_TRACK_VELOCITY_SAMPLES)
    private val recentTrackLateralVelocities = ArrayList<Float>(MAX_TRACK_VELOCITY_SAMPLES)
    private var pendingProposal: DenseTrackedBall? = null
    private var pendingStartupVelocityTicksPerSecond: Float? = null
    private var reacquisitionBlockedUntilAnchor = false
    private var blockedReacquisitionFloorTicks: Float? = null
    private var rulerEndRecoveryGraceUntilNs = Long.MIN_VALUE
    private val startupTracks = ArrayList<StartupTrack>(MAX_STARTUP_TRACKS)
    private var nextStartupTrackId = 1L
    internal var lastDiagnostics = DenseTrackerDiagnostics()
        private set
    internal var lastCandidateDiagnostics: List<DenseCandidateDiagnostics> = emptyList()
        private set

    init {
        require(fractions.size >= 3)
        require((1 until fractions.size).all { index ->
            fractions[index] > fractions[index - 1]
        })
    }

    @Synchronized
    fun reset() {
        detectors.forEach(RegisteredBallDetector::reset)
        resetTrajectory()
    }

    @Synchronized
    fun resetTrajectory() {
        detectors.forEach(RegisteredBallDetector::resetTrajectory)
        lastAnchor = null
        anchorVelocityTicksPerSecond = 0f
        anchorLateralVelocityPerSecond = 0f
        lastTrack = null
        trackStartPositionTicks = null
        committedTrackObservations = 0
        trackVelocityTicksPerSecond = 0f
        trackLateralVelocityPerSecond = 0f
        recentTrackVelocities.clear()
        recentTrackLateralVelocities.clear()
        pendingProposal = null
        pendingStartupVelocityTicksPerSecond = null
        reacquisitionBlockedUntilAnchor = false
        blockedReacquisitionFloorTicks = null
        rulerEndRecoveryGraceUntilNs = Long.MIN_VALUE
        clearStartupTracks()
        lastDiagnostics = DenseTrackerDiagnostics()
        lastCandidateDiagnostics = emptyList()
    }

    /**
     * Produces a current-frame image observation without changing the accepted trajectory.
     * The caller must [commit] it only after the global fall gate accepts the same proposal.
     */
    @Synchronized
    @Suppress("LongParameterList")
    fun propose(
        signals: Array<FloatArray>,
        signalLength: Int,
        signalOffsetPx: Float,
        ruler: RulerSnapshot,
        tickSpacingMm: Float,
        tubeWidthPx: Float,
        timestampNs: Long,
        learnBackground: Boolean,
        outputEnabled: Boolean,
        anchor: DenseBallAnchor? = null,
        committedTrackRetentionMs: Long = DEFAULT_TRACK_RETENTION_MS
    ): DenseTrackedBall? {
        require(signals.size == detectors.size)
        pendingProposal = null
        pendingStartupVelocityTicksPerSecond = null
        updateAnchor(anchor)

        val prediction = if (outputEnabled) {
            predictionAt(timestampNs, committedTrackRetentionMs)
        } else {
            null
        }
        val predictionElapsedSeconds = prediction?.let {
            elapsedSeconds(it.timestampNs, timestampNs)
        } ?: 0f
        val predictedTicks = prediction?.let {
            it.positionTicks + it.velocityTicksPerSecond * predictionElapsedSeconds
        }
        val predictedLateral = prediction?.let {
            it.lateralFraction + it.lateralVelocityPerSecond * predictionElapsedSeconds
        }
        val axialWindow = prediction?.let {
            maxOf(
                MIN_AXIAL_WINDOW_TICKS,
                abs(it.velocityTicksPerSecond) * predictionElapsedSeconds * 1.75f +
                    AXIAL_WINDOW_ALLOWANCE_TICKS
            )
        } ?: MIN_AXIAL_WINDOW_TICKS
        val lateralWindow = prediction?.let {
            maxOf(
                MIN_LATERAL_WINDOW,
                abs(it.lateralVelocityPerSecond) * predictionElapsedSeconds * 1.75f +
                    LATERAL_WINDOW_ALLOWANCE
            )
        } ?: MIN_LATERAL_WINDOW
        val detectorAxialWindow = if (prediction?.fromTrack == true) {
            minOf(axialWindow, MIN_AXIAL_WINDOW_TICKS)
        } else {
            axialWindow
        }

        val candidates = ArrayList<Candidate>(detectors.size)
        var backgroundReadyBands = 0
        for (index in detectors.indices) {
            val result = detectors[index].observe(
                signal = signals[index],
                signalLength = signalLength,
                signalOffsetPx = signalOffsetPx,
                ruler = ruler,
                rawBallAxialPx = null,
                learnBackground = learnBackground,
                timestampNs = timestampNs,
                preferredPositionTicks = predictedTicks,
                preferredWindowTicks = detectorAxialWindow,
                trackingSensitivity = prediction?.fromTrack == true &&
                    predictedLateral != null &&
                    abs(fractions[index] - predictedLateral) <= TRACKING_LATERAL_SENSITIVITY
            )
            if (result.backgroundReady) backgroundReadyBands++
            val axial = result.ballAxialPx ?: continue
            val positionTicks = ruler.toMm(axial, tickSpacingMm, refOrdinal = 0)
                ?.div(tickSpacingMm)
                ?.takeIf { it.isFinite() }
                ?: continue
            candidates += Candidate(
                bandIndex = index,
                axialPx = axial,
                positionTicks = positionTicks,
                lateralFraction = fractions[index],
                confidence = result.confidence
            )
        }
        lastCandidateDiagnostics = if (captureCandidateDiagnostics) {
            candidates.take(MAX_RECORDED_DIAGNOSTIC_CANDIDATES).map { candidate ->
                DenseCandidateDiagnostics(
                    candidate.positionTicks,
                    candidate.lateralFraction,
                    candidate.confidence
                )
            }
        } else {
            emptyList()
        }
        if (!outputEnabled) {
            clearStartupTracks()
            lastDiagnostics = DenseTrackerDiagnostics(
                backgroundReadyBands = backgroundReadyBands,
                candidateCount = candidates.size
            )
            return null
        }
        if (reacquisitionBlockedUntilAnchor) {
            val blockedFloor = blockedReacquisitionFloorTicks
            val committedTrack = lastTrack
            val registeredLast = ruler.ordinals.lastOrNull()?.toFloat()
            val guardedTerminalProtection = isTerminalRecoveryProtected(
                committedTrack,
                signalOffsetPx,
                signalLength
            )
            val guardedCandidates = if (blockedFloor != null) {
                candidates.filter { candidate ->
                    val clearsAbove = candidate.positionTicks <= blockedFloor -
                        MIN_GUARDED_REACQUISITION_HEADROOM_TICKS
                    val clearsBelow = committedTrack != null &&
                        candidate.positionTicks >= blockedFloor +
                        MIN_GUARDED_REACQUISITION_HEADROOM_TICKS &&
                        candidate.positionTicks <= committedTrack.positionTicks +
                        MAX_STALE_RECOVERY_ADVANCE_TICKS &&
                        canUseRulerEndRecoveryCandidate(
                            candidate,
                            registeredLast,
                            timestampNs
                        ) &&
                        (!guardedTerminalProtection || canUseTerminalRecoveryCandidate(
                            candidate,
                            signalOffsetPx,
                            signalLength,
                            timestampNs
                        )) &&
                        abs(candidate.lateralFraction - committedTrack.lateralFraction) <=
                        MAX_STALE_RECOVERY_LATERAL_DELTA
                    clearsAbove || clearsBelow
                }
            } else {
                emptyList()
            }
            val guardedRecovery = selectStartupCandidate(guardedCandidates, timestampNs)
            lastDiagnostics = DenseTrackerDiagnostics(
                backgroundReadyBands = backgroundReadyBands,
                candidateCount = candidates.size,
                startupTrackCount = guardedRecovery.trackCount,
                readyStartupTrackCount = guardedRecovery.readyTrackCount,
                unanchoredAcquisitionReady = guardedRecovery.candidate != null
            )
            val selected = guardedRecovery.candidate ?: return null
            reacquisitionBlockedUntilAnchor = false
            blockedReacquisitionFloorTicks = null
            return buildProposal(
                best = selected,
                candidates = candidates,
                tubeWidthPx = tubeWidthPx,
                timestampNs = timestampNs,
                startupVelocityTicksPerSecond = guardedRecovery.selectedSpeedTicksPerSecond
            )
        }

        val registeredFirst = ruler.ordinals.firstOrNull() ?: 0
        val registeredSpan = (ruler.ordinals.lastOrNull() ?: registeredFirst) - registeredFirst
        val unanchoredCandidates = if (registeredSpan > 0) {
            candidates.filter { candidate ->
                (candidate.positionTicks - registeredFirst) / registeredSpan <=
                    MAX_UNANCHORED_START_PROGRESS_FRACTION
            }
        } else {
            emptyList()
        }
        val startup = if (prediction == null) {
            selectStartupCandidate(unanchoredCandidates, timestampNs)
        } else {
            null
        }
        if (prediction == null) {
            lastDiagnostics = DenseTrackerDiagnostics(
                backgroundReadyBands = backgroundReadyBands,
                candidateCount = candidates.size,
                startupTrackCount = startup?.trackCount ?: 0,
                readyStartupTrackCount = startup?.readyTrackCount ?: 0,
                unanchoredAcquisitionReady = startup?.candidate != null
            )
            val selected = startup?.candidate ?: return null
            return buildProposal(
                best = selected,
                candidates = candidates,
                tubeWidthPx = tubeWidthPx,
                timestampNs = timestampNs,
                startupVelocityTicksPerSecond = startup.selectedSpeedTicksPerSecond
            )
        }
        check(predictedTicks != null && predictedLateral != null)
        val independentAnchorStartup = if (!prediction.fromTrack) {
            selectStartupCandidate(unanchoredCandidates, timestampNs)
        } else {
            null
        }
        val plausible = candidates.filter { candidate ->
            candidate.positionTicks >= predictedTicks - MAX_UPWARD_TICKS &&
                abs(candidate.positionTicks - predictedTicks) <= axialWindow &&
                abs(candidate.lateralFraction - predictedLateral) <= lateralWindow
        }
        val continuationCandidates = if (prediction.fromTrack) {
            val continuationErrorTicks = (
                MAX_TRACK_CONTINUATION_ERROR_TICKS +
                    abs(prediction.velocityTicksPerSecond) * predictionElapsedSeconds *
                    TRACK_CONTINUATION_CADENCE_ALLOWANCE
                ).coerceAtMost(MAX_TRACK_CONTINUATION_ERROR_CEILING_TICKS)
            plausible.filter { candidate ->
                abs(candidate.positionTicks - predictedTicks) <=
                    continuationErrorTicks
            }
        } else {
            plausible
        }
        val registeredProgress = if (registeredSpan > 0) {
            (predictedTicks - registeredFirst) / registeredSpan
        } else {
            0f
        }
        val committedTrack = lastTrack
        val committedTubeProgress = committedTrack?.let { track ->
            val sampledProgress = (track.axialPx - signalOffsetPx) /
                (signalLength - 1f).coerceAtLeast(1f)
            val registeredRulerProgress = if (registeredSpan > 0) {
                (track.positionTicks - registeredFirst) / registeredSpan
            } else {
                0f
            }
            maxOf(sampledProgress, registeredRulerProgress)
        } ?: 0f
        val directTrackGapSeconds = if (
            committedTubeProgress >= BOTTOM_REACQUISITION_PROGRESS_FRACTION
        ) {
            MAX_BOTTOM_DIRECT_TRACK_GAP_SECONDS
        } else {
            MAX_MID_TUBE_DIRECT_TRACK_GAP_SECONDS
        }
        if (prediction.fromTrack && predictionElapsedSeconds > directTrackGapSeconds) {
            val registeredLast = ruler.ordinals.lastOrNull()?.toFloat()
            val wideStaleRecovery = predictionElapsedSeconds >=
                MIN_WIDE_STALE_RECOVERY_GAP_SECONDS
            val terminalProtection = isTerminalRecoveryProtected(
                committedTrack,
                signalOffsetPx,
                signalLength
            )
            val recoveryCandidates = if (committedTrack != null) {
                // Retaining a committed track must not reopen unrestricted global startup.
                // Even after a long camera gap, only a bounded continuation can be associated
                // with that track; a farther residual waits until the global gate expires it.
                val recoveryCeiling = maxOf(
                    committedTrack.positionTicks + MAX_STALE_RECOVERY_ADVANCE_TICKS,
                    predictedTicks + MAX_PREDICTED_RECOVERY_ERROR_TICKS
                )
                candidates.filter { candidate ->
                    candidate.positionTicks >= committedTrack.positionTicks - MAX_UPWARD_TICKS &&
                        candidate.positionTicks <= recoveryCeiling &&
                        (wideStaleRecovery || canUseRulerEndRecoveryCandidate(
                            candidate,
                            registeredLast,
                            timestampNs
                        )) &&
                        (!terminalProtection || canUseTerminalRecoveryCandidate(
                            candidate,
                            signalOffsetPx,
                            signalLength,
                            timestampNs
                        )) &&
                        abs(candidate.lateralFraction - committedTrack.lateralFraction) <=
                        if (wideStaleRecovery) {
                            MAX_WIDE_STALE_RECOVERY_LATERAL_DELTA
                        } else {
                            MAX_STALE_RECOVERY_LATERAL_DELTA
                        }
                }
            } else {
                emptyList()
            }
            val recovery = selectStartupCandidate(
                candidates = recoveryCandidates,
                timestampNs = timestampNs,
                blockOnStationary = committedTubeProgress >=
                    BOTTOM_REACQUISITION_PROGRESS_FRACTION,
                minimumTravelTicks = if (wideStaleRecovery) {
                    MIN_WIDE_STALE_RECOVERY_TRAVEL_TICKS
                } else {
                    MIN_STARTUP_TRAVEL_TICKS
                }
            )
            lastDiagnostics = DenseTrackerDiagnostics(
                backgroundReadyBands = backgroundReadyBands,
                candidateCount = candidates.size,
                predictionAvailable = true,
                plausibleCandidateCount = recoveryCandidates.size,
                predictedTicks = predictedTicks,
                startupTrackCount = recovery.trackCount,
                readyStartupTrackCount = recovery.readyTrackCount,
                unanchoredAcquisitionReady = recovery.candidate != null,
                recoveryTrackProgress = committedTubeProgress,
                recoveryAnchorAgeSeconds = lastAnchor?.let {
                    elapsedAgeSeconds(it.timestampNs, timestampNs)
                },
                recoveryAllowed = true
            )
            if (reacquisitionBlockedUntilAnchor) return null
            val selected = recovery.candidate ?: return null
            return buildProposal(
                best = selected,
                candidates = recoveryCandidates,
                tubeWidthPx = tubeWidthPx,
                timestampNs = timestampNs,
                startupVelocityTicksPerSecond = recovery.selectedSpeedTicksPerSecond
            )
        }
        val closestCandidate = candidates.minByOrNull { candidate ->
            abs(candidate.positionTicks - predictedTicks)
        }
        val closestCandidateError = closestCandidate?.let {
            abs(it.positionTicks - predictedTicks)
        }
        if (continuationCandidates.isEmpty()) {
            // Only a committed local track is authoritative until its direct-tracking window
            // expires. A primary-detector anchor is optional evidence: if current pixels disagree with
            // it, local startup must keep accumulating from the full candidate set.
            val startup = if (prediction.fromTrack) {
                clearStartupTracks()
                null
            } else {
                independentAnchorStartup
            }
            lastDiagnostics = DenseTrackerDiagnostics(
                backgroundReadyBands = backgroundReadyBands,
                candidateCount = candidates.size,
                predictionAvailable = true,
                plausibleCandidateCount = plausible.size,
                predictedTicks = predictedTicks,
                closestCandidateErrorTicks = closestCandidateError,
                closestCandidateTicks = closestCandidate?.positionTicks,
                startupTrackCount = startup?.trackCount ?: 0,
                readyStartupTrackCount = startup?.readyTrackCount ?: 0,
                unanchoredAcquisitionReady = startup?.candidate != null
            )
            val selected = startup?.candidate ?: return null
            return buildProposal(
                best = selected,
                candidates = candidates,
                tubeWidthPx = tubeWidthPx,
                timestampNs = timestampNs,
                startupVelocityTicksPerSecond = startup.selectedSpeedTicksPerSecond
            )
        }
        val independentCandidate = independentAnchorStartup?.candidate
        if (independentCandidate != null) {
            return buildProposal(
                best = independentCandidate,
                candidates = candidates,
                tubeWidthPx = tubeWidthPx,
                timestampNs = timestampNs,
                startupVelocityTicksPerSecond =
                    independentAnchorStartup.selectedSpeedTicksPerSecond,
                diagnostics = DenseTrackerDiagnostics(
                    backgroundReadyBands = backgroundReadyBands,
                    candidateCount = candidates.size,
                    predictionAvailable = true,
                    plausibleCandidateCount = continuationCandidates.size,
                    predictedTicks = predictedTicks,
                    closestCandidateErrorTicks = closestCandidateError,
                    closestCandidateTicks = closestCandidate?.positionTicks,
                    startupTrackCount = independentAnchorStartup.trackCount,
                    readyStartupTrackCount = independentAnchorStartup.readyTrackCount,
                    unanchoredAcquisitionReady = true
                )
            )
        }
        if (prediction.fromTrack) clearStartupTracks()

        val nearestContinuation = continuationCandidates.minBy { candidate ->
            abs(candidate.positionTicks - predictedTicks) +
                0.35f * abs(candidate.lateralFraction - predictedLateral) -
                0.05f * candidate.confidence.coerceIn(0f, 5f)
        }
        val strongEndCorrection = if (
            registeredProgress >= STRONG_CORRECTION_PROGRESS_FRACTION
        ) {
            plausible
                .asSequence()
                .filter { candidate ->
                    candidate.positionTicks >= nearestContinuation.positionTicks +
                        MIN_STRONG_CORRECTION_ADVANCE_TICKS &&
                        candidate.positionTicks - predictedTicks <=
                        MAX_STRONG_CORRECTION_ERROR_TICKS &&
                        candidate.confidence >= nearestContinuation.confidence *
                        STRONG_CORRECTION_CONFIDENCE_RATIO
                }
                .maxByOrNull { it.confidence }
        } else {
            null
        }
        val best = strongEndCorrection ?: nearestContinuation
        return buildProposal(
            best = best,
            candidates = if (strongEndCorrection != null) plausible else continuationCandidates,
            tubeWidthPx = tubeWidthPx,
            timestampNs = timestampNs,
            diagnostics = DenseTrackerDiagnostics(
                backgroundReadyBands = backgroundReadyBands,
                candidateCount = candidates.size,
                predictionAvailable = true,
                plausibleCandidateCount = continuationCandidates.size,
                predictedTicks = predictedTicks,
                closestCandidateErrorTicks = closestCandidateError,
                closestCandidateTicks = closestCandidate?.positionTicks,
                startupTrackCount = independentAnchorStartup?.trackCount ?: 0,
                readyStartupTrackCount = independentAnchorStartup?.readyTrackCount ?: 0
            )
        )
    }

    private fun buildProposal(
        best: Candidate,
        candidates: List<Candidate>,
        tubeWidthPx: Float,
        timestampNs: Long,
        diagnostics: DenseTrackerDiagnostics = lastDiagnostics,
        startupVelocityTicksPerSecond: Float? = null
    ): DenseTrackedBall? {
        val cluster = candidates.filter { candidate ->
            abs(candidate.bandIndex - best.bandIndex) <= 2 &&
                abs(candidate.positionTicks - best.positionTicks) <= CLUSTER_AXIAL_TICKS
        }
        var weightSum = 0f
        var axialSum = 0f
        var tickSum = 0f
        var lateralSum = 0f
        var confidenceSum = 0f
        for (candidate in cluster) {
            val weight = candidate.confidence.coerceIn(0.25f, 5f)
            weightSum += weight
            axialSum += candidate.axialPx * weight
            tickSum += candidate.positionTicks * weight
            lateralSum += candidate.lateralFraction * weight
            confidenceSum += candidate.confidence * weight
        }
        if (weightSum <= 0f) return null
        val lateralFraction = (lateralSum / weightSum).coerceIn(-1f, 1f)
        val tracked = DenseTrackedBall(
            timestampNs = timestampNs,
            axialPx = axialSum / weightSum,
            positionTicks = tickSum / weightSum,
            lateralPx = lateralFraction * tubeWidthPx * 0.5f,
            lateralFraction = lateralFraction,
            confidence = (confidenceSum / weightSum / 2f).coerceIn(0f, 1f)
        )
        lastDiagnostics = diagnostics
        pendingProposal = tracked
        pendingStartupVelocityTicksPerSecond = startupVelocityTicksPerSecond
        return tracked
    }

    private fun selectStartupCandidate(
        candidates: List<Candidate>,
        timestampNs: Long,
        blockOnStationary: Boolean = false,
        minimumTravelTicks: Float = MIN_STARTUP_TRAVEL_TICKS
    ): StartupSelection {
        startupTracks.removeAll { track ->
            timestampNs <= track.lastTimestampNs ||
                timestampNs - track.lastTimestampNs > MAX_STARTUP_TRACK_GAP_NS
        }

        val matches = ArrayList<StartupMatch>()
        startupTracks.forEachIndexed { trackIndex, track ->
            candidates.forEachIndexed { candidateIndex, candidate ->
                val axialStep = candidate.positionTicks - track.latest.positionTicks
                val lateralStep = abs(candidate.lateralFraction - track.latest.lateralFraction)
                if (isPlausibleStartupContinuation(track, candidate, timestampNs)) {
                    matches += StartupMatch(
                        trackIndex = trackIndex,
                        candidateIndex = candidateIndex,
                        cost = abs(axialStep) + lateralStep * STARTUP_LATERAL_WEIGHT
                    )
                }
            }
        }
        matches.sortBy { it.cost }

        val assignedTracks = BooleanArray(startupTracks.size)
        val assignedCandidates = BooleanArray(candidates.size)
        var sawStationaryMatch = false
        var sawProgressMatch = false
        var stationaryFloorTicks: Float? = null
        for (match in matches) {
            if (assignedTracks[match.trackIndex] || assignedCandidates[match.candidateIndex]) {
                continue
            }
            val track = startupTracks[match.trackIndex]
            val candidate = candidates[match.candidateIndex]
            val axialStep = candidate.positionTicks - track.latest.positionTicks
            if (axialStep >= MIN_STARTUP_PROGRESS_STEP_TICKS) {
                sawProgressMatch = true
                if (track.downwardStreakHits <= 1) {
                    track.downwardStreakStartTimestampNs = track.lastTimestampNs
                    track.downwardStreakTravelTicks = axialStep
                    track.downwardStreakHits = 2
                } else {
                    track.downwardStreakTravelTicks += axialStep
                    track.downwardStreakHits += 1
                }
            } else {
                sawStationaryMatch = true
                stationaryFloorTicks = maxOf(
                    stationaryFloorTicks ?: candidate.positionTicks,
                    candidate.positionTicks
                )
                track.downwardStreakHits = 1
                track.downwardStreakTravelTicks = 0f
                track.downwardStreakStartTimestampNs = timestampNs
            }
            track.latest = candidate
            track.lastTimestampNs = timestampNs
            track.hitCount += 1
            track.confidenceTotal += candidate.confidence
            assignedTracks[match.trackIndex] = true
            assignedCandidates[match.candidateIndex] = true
        }
        if (blockOnStationary && sawStationaryMatch && !sawProgressMatch) {
            reacquisitionBlockedUntilAnchor = true
            blockedReacquisitionFloorTicks = stationaryFloorTicks
        }
        candidates.forEachIndexed { index, candidate ->
            if (!assignedCandidates[index]) {
                startupTracks += StartupTrack(
                    id = nextStartupTrackId++,
                    firstPositionTicks = candidate.positionTicks,
                    firstTimestampNs = timestampNs,
                    latest = candidate,
                    lastTimestampNs = timestampNs,
                    hitCount = 1,
                    confidenceTotal = candidate.confidence,
                    downwardStreakHits = 1,
                    downwardStreakTravelTicks = 0f,
                    downwardStreakStartTimestampNs = timestampNs
                )
            }
        }
        trimStartupTracks()

        val ready = startupTracks.filter { track ->
            track.lastTimestampNs == timestampNs &&
                track.downwardStreakHits >= MIN_STARTUP_HITS &&
                track.downwardStreakTravelTicks >= minimumTravelTicks &&
                startupSpeedTicksPerSecond(track, timestampNs) >= MIN_STARTUP_SPEED_TICKS_PER_SECOND
        }
        val selected = ready.maxWithOrNull(
            compareBy<StartupTrack> { it.downwardStreakHits }
                .thenBy { it.downwardStreakTravelTicks }
                .thenBy { it.averageConfidence }
        )
        return StartupSelection(
            candidate = selected?.latest,
            trackCount = startupTracks.size,
            readyTrackCount = ready.size,
            selectedSpeedTicksPerSecond = selected?.let {
                startupSpeedTicksPerSecond(it, timestampNs)
            }
        )
    }

    private fun canUseRulerEndRecoveryCandidate(
        candidate: Candidate,
        registeredLast: Float?,
        timestampNs: Long
    ): Boolean {
        if (timestampNs <= rulerEndRecoveryGraceUntilNs || registeredLast == null) return true
        val headroomTicks = registeredLast - candidate.positionTicks
        if (headroomTicks >= MIN_RULER_END_RECOVERY_START_HEADROOM_TICKS) return true
        if (headroomTicks < MIN_RULER_END_RECOVERY_HEADROOM_TICKS) return false
        return startupTracks.any { track ->
            isPlausibleStartupContinuation(track, candidate, timestampNs)
        }
    }

    private fun hasFreshBottomRecoveryAnchor(timestampNs: Long): Boolean {
        val anchor = lastAnchor ?: return false
        return elapsedAgeSeconds(anchor.timestampNs, timestampNs) <=
            MAX_BOTTOM_RECOVERY_ANCHOR_AGE_SECONDS
    }

    private fun canUseTerminalRecoveryCandidate(
        candidate: Candidate,
        signalOffsetPx: Float,
        signalLength: Int,
        timestampNs: Long
    ): Boolean {
        val progress = (candidate.axialPx - signalOffsetPx) /
            (signalLength - 1f).coerceAtLeast(1f)
        return progress < TERMINAL_RECOVERY_PROGRESS_FRACTION ||
            hasFreshBottomRecoveryAnchor(timestampNs)
    }

    private fun isTerminalRecoveryProtected(
        committedTrack: DenseTrackedBall?,
        signalOffsetPx: Float,
        signalLength: Int
    ): Boolean {
        committedTrack ?: return false
        val startTicks = trackStartPositionTicks ?: return false
        val travelTicks = committedTrack.positionTicks - startTicks
        val progress = (committedTrack.axialPx - signalOffsetPx) /
            (signalLength - 1f).coerceAtLeast(1f)
        return travelTicks >= MIN_TERMINAL_TRACK_TRAVEL_TICKS &&
            committedTrackObservations >= MIN_TERMINAL_TRACK_OBSERVATIONS &&
            progress >= BOTTOM_REACQUISITION_PROGRESS_FRACTION
    }

    private fun isPlausibleStartupContinuation(
        track: StartupTrack,
        candidate: Candidate,
        timestampNs: Long
    ): Boolean {
        if (timestampNs <= track.lastTimestampNs ||
            timestampNs - track.lastTimestampNs > MAX_STARTUP_TRACK_GAP_NS
        ) {
            return false
        }
        val elapsedSeconds = elapsedSeconds(track.lastTimestampNs, timestampNs)
        val maximumDownwardStep = STARTUP_STEP_ALLOWANCE_TICKS +
            MAX_STARTUP_SPEED_TICKS_PER_SECOND * elapsedSeconds
        val axialStep = candidate.positionTicks - track.latest.positionTicks
        val lateralStep = abs(candidate.lateralFraction - track.latest.lateralFraction)
        return axialStep >= -MAX_STARTUP_UPWARD_STEP_TICKS &&
            axialStep <= maximumDownwardStep &&
            lateralStep <= MAX_STARTUP_LATERAL_STEP
    }

    private fun startupSpeedTicksPerSecond(track: StartupTrack, timestampNs: Long): Float {
        val elapsed = elapsedSeconds(track.downwardStreakStartTimestampNs, timestampNs)
        return if (elapsed > 0f) track.downwardStreakTravelTicks / elapsed else 0f
    }

    private fun trimStartupTracks() {
        if (startupTracks.size <= MAX_STARTUP_TRACKS) return
        startupTracks.sortWith(
            compareByDescending<StartupTrack> { it.lastTimestampNs }
                .thenByDescending { it.hitCount }
                .thenByDescending { it.downwardTravelTicks }
                .thenByDescending { it.averageConfidence }
                .thenBy { it.id }
        )
        while (startupTracks.size > MAX_STARTUP_TRACKS) {
            startupTracks.removeAt(startupTracks.lastIndex)
        }
    }

    private fun clearStartupTracks() {
        startupTracks.clear()
        nextStartupTrackId = 1L
    }

    /** Commits exactly the latest proposal after the global trajectory gate accepts it. */
    @Synchronized
    fun commit(proposal: DenseTrackedBall): Boolean {
        if (pendingProposal !== proposal) return false
        updateTrack(proposal, pendingStartupVelocityTicksPerSecond)
        clearStartupTracks()
        pendingProposal = null
        pendingStartupVelocityTicksPerSecond = null
        return true
    }

    /** Discards exactly the latest proposal without changing position or velocity state. */
    @Synchronized
    fun discard(proposal: DenseTrackedBall) {
        if (pendingProposal === proposal) {
            pendingProposal = null
            pendingStartupVelocityTicksPerSecond = null
        }
    }

    private data class Prediction(
        val timestampNs: Long,
        val positionTicks: Float,
        val lateralFraction: Float,
        val velocityTicksPerSecond: Float,
        val lateralVelocityPerSecond: Float,
        val fromTrack: Boolean
    )

    private fun predictionAt(timestampNs: Long, committedTrackRetentionMs: Long): Prediction? {
        val track = lastTrack
        val retentionSeconds = committedTrackRetentionMs.coerceAtLeast(0L) / 1_000f
        if (track != null && elapsedAgeSeconds(track.timestampNs, timestampNs) <= retentionSeconds) {
            return Prediction(
                timestampNs = track.timestampNs,
                positionTicks = track.positionTicks,
                lateralFraction = track.lateralFraction,
                velocityTicksPerSecond = trackVelocityTicksPerSecond,
                lateralVelocityPerSecond = trackLateralVelocityPerSecond,
                fromTrack = true
            )
        }
        val anchor = lastAnchor ?: return null
        if (elapsedSeconds(anchor.timestampNs, timestampNs) > MAX_ANCHOR_AGE_SECONDS) {
            return null
        }
        return Prediction(
            timestampNs = anchor.timestampNs,
            positionTicks = anchor.positionTicks,
            lateralFraction = anchor.lateralFraction,
            velocityTicksPerSecond = anchorVelocityTicksPerSecond,
            lateralVelocityPerSecond = anchorLateralVelocityPerSecond,
            fromTrack = false
        )
    }

    private fun updateAnchor(anchor: DenseBallAnchor?) {
        anchor ?: return
        if (!anchor.positionTicks.isFinite() || !anchor.lateralFraction.isFinite()) return
        val previous = lastAnchor
        if (previous != null && anchor.timestampNs <= previous.timestampNs) return
        rulerEndRecoveryGraceUntilNs = anchor.timestampNs + MAX_ANCHOR_STARTUP_GAP_NS
        reacquisitionBlockedUntilAnchor = false
        blockedReacquisitionFloorTicks = null
        if (previous != null) {
            val elapsed = elapsedSeconds(previous.timestampNs, anchor.timestampNs)
            if (elapsed > 0f && elapsed <= MAX_ANCHOR_GAP_SECONDS) {
                val measuredAxial = (anchor.positionTicks - previous.positionTicks) / elapsed
                val measuredLateral = (anchor.lateralFraction - previous.lateralFraction) / elapsed
                anchorVelocityTicksPerSecond = smoothVelocity(
                    anchorVelocityTicksPerSecond,
                    measuredAxial
                )
                anchorLateralVelocityPerSecond = smoothVelocity(
                    anchorLateralVelocityPerSecond,
                    measuredLateral
                )
            }
        }
        lastAnchor = anchor
        val localStartup = startupTracks
            .asSequence()
            .filter { track ->
                track.lastTimestampNs <= anchor.timestampNs &&
                    anchor.timestampNs - track.lastTimestampNs <= MAX_ANCHOR_STARTUP_GAP_NS &&
                    track.downwardStreakHits >= MIN_STARTUP_HITS &&
                    track.downwardStreakTravelTicks >= MIN_STARTUP_TRAVEL_TICKS &&
                    abs(track.latest.positionTicks - anchor.positionTicks) <=
                        ANCHOR_RESET_DISTANCE_TICKS
            }
            .maxWithOrNull(
                compareBy<StartupTrack> { it.downwardStreakTravelTicks }
                    .thenBy { it.downwardStreakHits }
            )
        if (localStartup != null) {
            val startupVelocity = startupSpeedTicksPerSecond(localStartup, anchor.timestampNs)
            if (startupVelocity.isFinite() && startupVelocity >= MIN_STARTUP_SPEED_TICKS_PER_SECOND) {
                anchorVelocityTicksPerSecond = startupVelocity
                anchorLateralVelocityPerSecond = 0f
            }
        }
        val track = lastTrack ?: return
        val projectedTicks = anchor.positionTicks + anchorVelocityTicksPerSecond *
            elapsedSeconds(anchor.timestampNs, track.timestampNs)
        if (abs(projectedTicks - track.positionTicks) > ANCHOR_RESET_DISTANCE_TICKS) {
            lastTrack = null
            trackVelocityTicksPerSecond = anchorVelocityTicksPerSecond
            trackLateralVelocityPerSecond = anchorLateralVelocityPerSecond
            recentTrackVelocities.clear()
            recentTrackLateralVelocities.clear()
        }
    }

    private fun updateTrack(
        current: DenseTrackedBall,
        startupVelocityTicksPerSecond: Float?
    ) {
        if (trackStartPositionTicks == null) {
            trackStartPositionTicks = current.positionTicks
        }
        committedTrackObservations++
        val previous = lastTrack
        if (previous != null && current.timestampNs > previous.timestampNs) {
            val elapsed = elapsedSeconds(previous.timestampNs, current.timestampNs)
            if (elapsed > 0f) {
                rememberTrackVelocity(
                    recentTrackVelocities,
                    (current.positionTicks - previous.positionTicks) / elapsed
                )
                rememberTrackVelocity(
                    recentTrackLateralVelocities,
                    (current.lateralFraction - previous.lateralFraction) / elapsed
                )
                trackVelocityTicksPerSecond = median(recentTrackVelocities)
                trackLateralVelocityPerSecond = median(recentTrackLateralVelocities)
            }
        } else if (previous == null) {
            recentTrackVelocities.clear()
            recentTrackLateralVelocities.clear()
            trackVelocityTicksPerSecond = startupVelocityTicksPerSecond
                ?.takeIf { it.isFinite() && it >= 0f }
                ?: anchorVelocityTicksPerSecond
            trackLateralVelocityPerSecond = anchorLateralVelocityPerSecond
            // A lone anchor has no velocity measurement. Do not seed the median with the
            // numeric zero default or the first real motion interval is halved.
            if (trackVelocityTicksPerSecond.isFinite() &&
                trackVelocityTicksPerSecond != 0f
            ) {
                rememberTrackVelocity(recentTrackVelocities, trackVelocityTicksPerSecond)
            }
            if (trackLateralVelocityPerSecond.isFinite()) {
                rememberTrackVelocity(
                    recentTrackLateralVelocities,
                    trackLateralVelocityPerSecond
                )
            }
        }
        lastTrack = current
    }

    private fun rememberTrackVelocity(samples: MutableList<Float>, velocity: Float) {
        if (!velocity.isFinite()) return
        samples += velocity
        if (samples.size > MAX_TRACK_VELOCITY_SAMPLES) samples.removeAt(0)
    }

    private fun median(samples: List<Float>): Float {
        if (samples.isEmpty()) return 0f
        val sorted = samples.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 1) {
            sorted[middle]
        } else {
            (sorted[middle - 1] + sorted[middle]) * 0.5f
        }
    }

    private fun smoothVelocity(previous: Float, measured: Float): Float =
        if (!measured.isFinite()) previous
        else if (previous == 0f) measured
        else previous * 0.65f + measured * 0.35f

    private fun elapsedSeconds(startNs: Long, endNs: Long): Float {
        if (startNs <= 0L || endNs <= startNs) return 0f
        return ((endNs - startNs) / 1_000_000_000f).coerceAtMost(MAX_ELAPSED_SECONDS)
    }

    private fun elapsedAgeSeconds(startNs: Long, endNs: Long): Float {
        if (startNs <= 0L || endNs <= startNs) return 0f
        return (endNs - startNs) / 1_000_000_000f
    }

    private companion object {
        const val MIN_AXIAL_WINDOW_TICKS = 1.5f
        const val MAX_TRACK_CONTINUATION_ERROR_TICKS = 0.25f
        const val TRACK_CONTINUATION_CADENCE_ALLOWANCE = 0.75f
        const val MAX_TRACK_CONTINUATION_ERROR_CEILING_TICKS = 1f
        const val AXIAL_WINDOW_ALLOWANCE_TICKS = 0.8f
        const val MAX_UPWARD_TICKS = 0.35f
        const val MIN_LATERAL_WINDOW = 0.75f
        const val LATERAL_WINDOW_ALLOWANCE = 0.25f
        const val TRACKING_LATERAL_SENSITIVITY = 0.40f
        const val CLUSTER_AXIAL_TICKS = 0.6f
        // The production caller supplies the gate's cadence-dependent loss window. The
        // maximum is the conservative default for tests and other non-production callers.
        const val DEFAULT_TRACK_RETENTION_MS = 6_000L
        const val MAX_TRACK_VELOCITY_SAMPLES = 7
        const val MAX_STALE_RECOVERY_ADVANCE_TICKS = 4f
        const val MAX_PREDICTED_RECOVERY_ERROR_TICKS = 1.5f
        const val MAX_STALE_RECOVERY_LATERAL_DELTA = 0.75f
        const val MAX_WIDE_STALE_RECOVERY_LATERAL_DELTA = 0.90f
        const val MIN_WIDE_STALE_RECOVERY_GAP_SECONDS = 1f
        const val MIN_WIDE_STALE_RECOVERY_TRAVEL_TICKS = 0.20f
        const val MIN_RULER_END_RECOVERY_HEADROOM_TICKS = 0.5f
        const val MIN_RULER_END_RECOVERY_START_HEADROOM_TICKS = 1f
        const val MAX_BOTTOM_DIRECT_TRACK_GAP_SECONDS = 0.10f
        const val MAX_MID_TUBE_DIRECT_TRACK_GAP_SECONDS = 0.5f
        const val BOTTOM_REACQUISITION_PROGRESS_FRACTION = 0.85f
        const val TERMINAL_RECOVERY_PROGRESS_FRACTION = 0.93f
        const val MIN_TERMINAL_TRACK_TRAVEL_TICKS = 5f
        const val MIN_TERMINAL_TRACK_OBSERVATIONS = 32
        const val STRONG_CORRECTION_PROGRESS_FRACTION = 0.85f
        const val MIN_STRONG_CORRECTION_ADVANCE_TICKS = 0.5f
        const val MAX_STRONG_CORRECTION_ERROR_TICKS = 1.5f
        const val STRONG_CORRECTION_CONFIDENCE_RATIO = 1.75f
        const val MAX_ANCHOR_AGE_SECONDS = 0.75f
        const val MAX_BOTTOM_RECOVERY_ANCHOR_AGE_SECONDS = 0.35f
        const val MAX_ANCHOR_GAP_SECONDS = 2.5f
        const val MAX_ELAPSED_SECONDS = 3f
        const val ANCHOR_RESET_DISTANCE_TICKS = 1.5f
        const val MAX_STARTUP_TRACKS = 32
        const val MIN_STARTUP_HITS = 4
        const val MIN_STARTUP_TRAVEL_TICKS = 0.10f
        const val MIN_STARTUP_SPEED_TICKS_PER_SECOND = 0.20f
        const val MAX_STARTUP_SPEED_TICKS_PER_SECOND = 15f
        const val MAX_UNANCHORED_START_PROGRESS_FRACTION = 0.25f
        const val MIN_STARTUP_PROGRESS_STEP_TICKS = 0.02f
        const val MIN_GUARDED_REACQUISITION_HEADROOM_TICKS = 0.5f
        const val MAX_STARTUP_UPWARD_STEP_TICKS = 0.12f
        const val STARTUP_STEP_ALLOWANCE_TICKS = 0.20f
        const val MAX_STARTUP_LATERAL_STEP = 0.30f
        const val STARTUP_LATERAL_WEIGHT = 2f
        // Camera analysis p95 reaches roughly 267 ms on the captured WARM-profile runs.
        // Preserve a motion hypothesis across the documented 500 ms diagnostic gap limit.
        const val MAX_STARTUP_TRACK_GAP_NS = 500_000_000L
        const val MAX_RECORDED_DIAGNOSTIC_CANDIDATES = 12
        const val MAX_ANCHOR_STARTUP_GAP_NS = 250_000_000L
    }
}
