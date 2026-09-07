package com.cupet.fallingball.analysis

import com.cupet.fallingball.vision.BallDetectionSource
import com.cupet.fallingball.vision.BallEndReason
import com.cupet.fallingball.vision.WallContactState

enum class FallTrackingPhase {
    WAITING,
    CONFIRMING,
    TRACKING,
    FINISHED
}

enum class FallTrackingRejection {
    NONE,
    NO_DETECTION,
    RULER_INVALID,
    GAP_TOO_LONG,
    UPWARD_JUMP,
    DOWNWARD_JUMP,
    LATERAL_JUMP,
    NO_PROGRESS,
    OUT_OF_ORDER,
    NEED_MORE_EVIDENCE
}

data class FallTrackingStatus(
    val phase: FallTrackingPhase,
    val pendingObservations: Int,
    val allowedGapMs: Long,
    val lastRejection: FallTrackingRejection,
    val motionSeeded: Boolean,
    /** How long a confirmed trajectory remains recoverable before BALL_LOST. */
    val lossTimeoutMs: Long = 4_000L
)

/**
 * One detector observation expressed in ruler coordinates.
 *
 * Absolute millimetres and the reference ordinal are retained while a possible start is
 * confirmed, allowing the caller to restore the first few real frames without recording
 * unconfirmed detector noise.
 */
data class FallFrameObservation(
    val timestampNs: Long,
    val axialPx: Float,
    val absolutePositionMm: Float?,
    val referenceOrdinal: Int?,
    val rulerValid: Boolean,
    val shiftPx: Float,
    val source: BallDetectionSource,
    val positionTicks: Float?,
    val progressFraction: Float,
    val lateralPx: Float? = null,
    val lateralMm: Float? = null,
    val normalizedLateral: Float? = null,
    val wallClearanceMm: Float? = null,
    val wallContactState: WallContactState = WallContactState.UNKNOWN,
    /** Lateral position divided by the current YOLO half-width; available without geometry. */
    val trajectoryLateralFraction: Float? = null,
    val detectionConfidence: Float? = null
)

data class FallTrackingUpdate(
    val phase: FallTrackingPhase,
    val acceptedObservations: List<FallFrameObservation> = emptyList(),
    val endReason: BallEndReason? = null
)

/**
 * Separates an armed measurement from a confirmed falling-ball track.
 *
 * Before confirmation, detector candidates may be buffered but can neither add measurement
 * samples nor end a run. This keeps a single reflection, tube-end shadow, or jittered raw
 * trough from turning an indefinitely armed session into a short failed measurement.
 */
class FallTrackingGate {

    companion object {
        private const val MIN_CONFIRMATION_GAP_MS = 150L
        private const val EXPECTED_INTERVAL_JITTER_DIVISOR = 4L
        /**
         * The first two detections arrive before cadence can be estimated. Keep this large
         * enough for low-rate primary-detector observations, then tighten around the
         * observed cadence for the remaining confirmation window.
         */
        private const val INITIAL_CONFIRMATION_GAP_MS = 500L
        private const val MAX_CONFIRMATION_GAP_MS = 2_500L
        private const val CONFIRMATION_GAP_PERIODS = 2.5f
        private const val CADENCE_ALPHA = 0.25f
        private const val MIN_STRONG_OBSERVATIONS = 4
        private const val MIN_STRONG_HITS = 3
        private const val MIN_STRONG_TRAVEL_TICKS = 0.50f
        private const val MIN_RAW_OBSERVATIONS = 5
        private const val MIN_RAW_TRAVEL_TICKS = 0.75f
        private const val MIN_PROGRESS_STEP_TICKS = 0.05f
        private const val MIN_TRACK_PROGRESS_TICKS = 0.02f
        private const val MAX_CROSS_SOURCE_CONTINUATION_ERROR_TICKS = 0.75f
        private const val MIN_MOTION_SEED_TRAVEL_TICKS = 0.15f
        private const val MAX_UPWARD_JITTER_TICKS = 0.35f
        private const val MIN_MAX_DOWNWARD_STEP_TICKS = 4f
        private const val MAX_DOWNWARD_SPEED_TICKS_PER_SECOND = 15f
        private const val MAX_ADAPTIVE_DOWNWARD_SPEED_TICKS_PER_SECOND = 60f
        private const val MIN_ADAPTIVE_DOWNWARD_STEP_TICKS = 1.25f
        private const val ADAPTIVE_DOWNWARD_MULTIPLIER = 2.5f
        private const val ADAPTIVE_DOWNWARD_TOLERANCE_TICKS = 0.5f
        private const val MIN_SPEED_INTERVALS = 2
        private const val MAX_RECENT_OBSERVATIONS = 7
        private const val MIN_MAX_LATERAL_STEP = 0.65f
        private const val MAX_LATERAL_STEP = 0.90f
        private const val MAX_LATERAL_SPEED_PER_SECOND = 1.25f
        private const val BOTTOM_FRACTION = 0.95f
        private const val MIN_BALL_LOST_MS = 4_000L
        private const val MAX_BALL_LOST_MS = 6_000L
        private const val BALL_LOST_PERIODS = 8f
        private const val MAX_CONFIRM_OBSERVATIONS = 8
    }

    private val pending = ArrayList<FallFrameObservation>(4)
    private val recentObservations = ArrayList<FallFrameObservation>(MAX_RECENT_OBSERVATIONS)
    private var heldNoProgressObservation: FallFrameObservation? = null
    private var lastAcceptedNs = 0L
    private var lastAcceptedObservation: FallFrameObservation? = null
    private val furthestAcceptedPositionTicks =
        mutableMapOf<BallDetectionSource, Float>()
    private var confirmationCadenceMs = 0f
    private var expectedInitialGapMs = INITIAL_CONFIRMATION_GAP_MS
    private var lastRejection = FallTrackingRejection.NONE

    @Volatile
    var phase: FallTrackingPhase = FallTrackingPhase.WAITING
        private set

    /** A possible ball freezes the pre-drop template until that candidate is resolved. */
    val allowsBackgroundLearning: Boolean
        get() = phase == FallTrackingPhase.WAITING

    @Synchronized
    fun reset() {
        pending.clear()
        recentObservations.clear()
        heldNoProgressObservation = null
        lastAcceptedNs = 0L
        lastAcceptedObservation = null
        furthestAcceptedPositionTicks.clear()
        confirmationCadenceMs = 0f
        lastRejection = FallTrackingRejection.NONE
        phase = FallTrackingPhase.WAITING
    }

    @Synchronized
    fun setExpectedObservationIntervalMs(intervalMs: Long) {
        val schedulingAllowanceMs = intervalMs / EXPECTED_INTERVAL_JITTER_DIVISOR
        expectedInitialGapMs = (intervalMs + schedulingAllowanceMs).coerceIn(
            MIN_CONFIRMATION_GAP_MS,
            MAX_CONFIRMATION_GAP_MS
        )
    }

    @Synchronized
    fun status(): FallTrackingStatus = FallTrackingStatus(
        phase = phase,
        pendingObservations = pending.size,
        allowedGapMs = confirmationGapMs(),
        lastRejection = lastRejection,
        motionSeeded = hasMotionSeed(),
        lossTimeoutMs = ballLostTimeoutMs()
    )

    /** Uses the same motion limits as [observe] to screen alternate heatmap peaks. */
    @Synchronized
    fun isPlausibleCandidateContinuation(
        timestampNs: Long,
        positionTicks: Float?,
        trajectoryLateralFraction: Float?
    ): Boolean {
        val previous = when {
            phase == FallTrackingPhase.TRACKING -> lastAcceptedObservation
            phase == FallTrackingPhase.CONFIRMING -> pending.lastOrNull()
            else -> null
        } ?: return true
        if (timestampNs <= previous.timestampNs) return false
        if (positionTicks == null || !positionTicks.isFinite()) return false
        val gapMs = elapsedMs(previous.timestampNs, timestampNs)
        val continuationLimitMs = if (phase == FallTrackingPhase.TRACKING) {
            ballLostTimeoutMs()
        } else {
            confirmationGapMs()
        }
        if (gapMs > continuationLimitMs) return false
        val candidate = previous.copy(
            timestampNs = timestampNs,
            positionTicks = positionTicks,
            normalizedLateral = null,
            trajectoryLateralFraction = trajectoryLateralFraction
        )
        return movementRejection(previous, candidate, gapMs) == FallTrackingRejection.NONE
    }

    @Synchronized
    fun predictedContinuationPositionTicks(timestampNs: Long): Float? {
        val previous = when {
            phase == FallTrackingPhase.TRACKING -> lastAcceptedObservation
            phase == FallTrackingPhase.CONFIRMING -> pending.lastOrNull()
            else -> null
        } ?: return null
        if (timestampNs <= previous.timestampNs) return null
        val previousTicks = previous.positionTicks?.takeIf { it.isFinite() } ?: return null
        val speed = recentTrajectorySpeedTicksPerSecond() ?: return previousTicks
        val elapsedSeconds = elapsedMs(previous.timestampNs, timestampNs) / 1_000f
        return previousTicks + speed * elapsedSeconds
    }

    @Synchronized
    fun hasRecentMotionEstimate(): Boolean = recentTrajectorySpeedTicksPerSecond() != null

    private fun hasMotionSeed(): Boolean {
        if (phase != FallTrackingPhase.CONFIRMING || pending.size < 2) return false
        val firstTicks = pending.first().positionTicks ?: return false
        val lastTicks = pending.last().positionTicks ?: return false
        return firstTicks.isFinite() && lastTicks.isFinite() &&
            lastTicks - firstTicks >= MIN_MOTION_SEED_TRAVEL_TICKS
    }

    @Synchronized
    fun observe(
        observation: FallFrameObservation?,
        timestampNs: Long
    ): FallTrackingUpdate {
        if (phase == FallTrackingPhase.FINISHED) {
            return FallTrackingUpdate(phase)
        }

        if (phase == FallTrackingPhase.TRACKING) {
            if (observation == null || observation.source == BallDetectionSource.NONE) {
                return rejectTracking(FallTrackingRejection.NO_DETECTION, timestampNs)
            }
            if (!observation.rulerValid || observation.positionTicks == null) {
                return rejectTracking(FallTrackingRejection.RULER_INVALID, timestampNs)
            }
            val previous = lastAcceptedObservation
            if (previous != null) {
                if (observation.timestampNs <= previous.timestampNs) {
                    return rejectTracking(FallTrackingRejection.OUT_OF_ORDER, timestampNs)
                }
                val gapMs = elapsedMs(previous.timestampNs, observation.timestampNs)
                if (previous.source == BallDetectionSource.LOCAL_TRACKER &&
                    (observation.source == BallDetectionSource.CONVGRU ||
                        observation.source == BallDetectionSource.CLASSICAL_HEATMAP ||
                        observation.source == BallDetectionSource.TINY_HEATMAP)
                ) {
                    val recentSpeed = recentTrajectorySpeedTicksPerSecond()
                    val expectedTicks = recentSpeed?.let { speed ->
                        requireNotNull(previous.positionTicks) + speed * gapMs / 1_000f
                    }
                    if (expectedTicks != null &&
                        kotlin.math.abs(observation.positionTicks - expectedTicks) >
                        MAX_CROSS_SOURCE_CONTINUATION_ERROR_TICKS
                    ) {
                        return rejectTracking(FallTrackingRejection.NO_PROGRESS, timestampNs)
                    }
                }
                val step = observation.positionTicks - requireNotNull(previous.positionTicks)
                val rejection = movementRejection(previous, observation, gapMs)
                if (rejection != FallTrackingRejection.NONE) {
                    return rejectTracking(rejection, timestampNs)
                }
                if (gapMs > ballLostTimeoutMs()) {
                    return rejectTracking(FallTrackingRejection.GAP_TOO_LONG, timestampNs)
                }
                val furthest = furthestAcceptedPositionTicks[observation.source]
                if (observation.progressFraction < BOTTOM_FRACTION &&
                    furthest != null &&
                    observation.positionTicks < furthest + MIN_TRACK_PROGRESS_TICKS
                ) {
                    heldNoProgressObservation = observation.takeIf {
                        it.source == BallDetectionSource.LOCAL_TRACKER &&
                            requireNotNull(it.positionTicks) >=
                            requireNotNull(previous.positionTicks) &&
                            gapMs <= confirmationGapMs()
                    }
                    return rejectTracking(FallTrackingRejection.NO_PROGRESS, timestampNs)
                }
                // Once a fall is confirmed, a short image-tracker dropout must remain
                // recoverable. Capping the cadence update prevents one miss from redefining
                // the normal frame period while still adapting gradually to thermal slowdown.
                updateConfirmationCadence(gapMs.coerceAtMost(confirmationGapMs()))
            }
            lastRejection = FallTrackingRejection.NONE
            val recoveredNoProgress = heldNoProgressObservation?.takeIf { held ->
                held.source == observation.source &&
                    held.timestampNs > (previous?.timestampNs ?: Long.MIN_VALUE) &&
                    held.timestampNs < observation.timestampNs &&
                    requireNotNull(held.positionTicks) <= observation.positionTicks &&
                    observation.positionTicks - requireNotNull(held.positionTicks) >=
                    MIN_TRACK_PROGRESS_TICKS
            }
            heldNoProgressObservation = null
            lastAcceptedNs = observation.timestampNs
            lastAcceptedObservation = observation
            furthestAcceptedPositionTicks[observation.source] = maxOf(
                furthestAcceptedPositionTicks[observation.source] ?: observation.positionTicks,
                observation.positionTicks
            )
            recoveredNoProgress?.let(::remember)
            remember(observation)
            return if (observation.progressFraction >= BOTTOM_FRACTION) {
                phase = FallTrackingPhase.FINISHED
                FallTrackingUpdate(
                    phase = phase,
                    acceptedObservations = listOfNotNull(recoveredNoProgress, observation),
                    endReason = BallEndReason.REACHED_BOTTOM
                )
            } else {
                FallTrackingUpdate(
                    phase,
                    listOfNotNull(recoveredNoProgress, observation)
                )
            }
        }

        if (observation == null ||
            observation.source == BallDetectionSource.NONE
        ) {
            lastRejection = FallTrackingRejection.NO_DETECTION
            expireConfirmation(timestampNs)
            return FallTrackingUpdate(phase)
        }
        if (!observation.rulerValid || observation.positionTicks == null) {
            lastRejection = FallTrackingRejection.RULER_INVALID
            expireConfirmation(timestampNs)
            return FallTrackingUpdate(phase)
        }

        if (phase == FallTrackingPhase.WAITING) {
            lastRejection = FallTrackingRejection.NEED_MORE_EVIDENCE
            beginConfirmation(observation)
            return FallTrackingUpdate(phase)
        }

        val previous = pending.last()
        if (observation.timestampNs <= previous.timestampNs) {
            lastRejection = FallTrackingRejection.OUT_OF_ORDER
            return FallTrackingUpdate(phase)
        }
        val gapMs = elapsedMs(previous.timestampNs, observation.timestampNs)
        val gapTooLong = gapMs > confirmationGapMs()
        val movementRejection = movementRejection(previous, observation, gapMs)
        if (gapTooLong || movementRejection != FallTrackingRejection.NONE) {
            lastRejection = when {
                gapTooLong -> FallTrackingRejection.GAP_TOO_LONG
                else -> movementRejection
            }
            pending.clear()
            recentObservations.clear()
            confirmationCadenceMs = 0f
            beginConfirmation(observation)
            return FallTrackingUpdate(phase)
        }

        updateConfirmationCadence(gapMs)
        pending.add(observation)
        remember(observation)
        if (pending.size > MAX_CONFIRM_OBSERVATIONS) pending.removeAt(0)
        val accepted = confirmedWindow()
        if (accepted != null) {
            pending.clear()
            lastRejection = FallTrackingRejection.NONE
            lastAcceptedNs = accepted.last().timestampNs
            lastAcceptedObservation = accepted.last()
            accepted.groupBy { it.source }.forEach { (source, observations) ->
                furthestAcceptedPositionTicks[source] =
                    observations.maxOf { requireNotNull(it.positionTicks) }
            }
            return if (accepted.last().progressFraction >= BOTTOM_FRACTION) {
                phase = FallTrackingPhase.FINISHED
                FallTrackingUpdate(
                    phase = phase,
                    acceptedObservations = accepted,
                    endReason = BallEndReason.REACHED_BOTTOM
                )
            } else {
                phase = FallTrackingPhase.TRACKING
                FallTrackingUpdate(phase, accepted)
            }
        }
        lastRejection = FallTrackingRejection.NEED_MORE_EVIDENCE
        return FallTrackingUpdate(phase)
    }

    private fun confirmedWindow(): List<FallFrameObservation>? {
        for (windowSize in minOf(pending.size, MAX_CONFIRM_OBSERVATIONS)
            downTo MIN_STRONG_OBSERVATIONS
        ) {
            if (pending.size < windowSize) continue
            val from = pending.size - windowSize
            val window = pending.subList(from, pending.size)
            val continuouslyDownward = (1 until window.size).all { index ->
                requireNotNull(window[index].positionTicks) -
                    requireNotNull(window[index - 1].positionTicks) >= MIN_PROGRESS_STEP_TICKS
            }
            if (!continuouslyDownward) continue

            val travel = requireNotNull(window.last().positionTicks) -
                requireNotNull(window.first().positionTicks)
            val strongHits = window.count { it.source.isStrong() }
            val strongConfirmed = strongHits >= MIN_STRONG_HITS &&
                travel >= MIN_STRONG_TRAVEL_TICKS
            val rawConfirmed = windowSize >= MIN_RAW_OBSERVATIONS &&
                strongHits < MIN_STRONG_HITS &&
                travel >= MIN_RAW_TRAVEL_TICKS
            if (strongConfirmed || rawConfirmed) return window.toList()
        }
        return null
    }

    private fun beginConfirmation(observation: FallFrameObservation) {
        recentObservations.clear()
        pending.add(observation)
        remember(observation)
        phase = FallTrackingPhase.CONFIRMING
    }

    private fun movementRejection(
        previous: FallFrameObservation,
        current: FallFrameObservation,
        gapMs: Long
    ): FallTrackingRejection {
        val step = requireNotNull(current.positionTicks) - requireNotNull(previous.positionTicks)
        if (step < -MAX_UPWARD_JITTER_TICKS) return FallTrackingRejection.UPWARD_JUMP
        if (step > maxDownwardStepTicks(gapMs)) return FallTrackingRejection.DOWNWARD_JUMP
        val previousLateral = previous.trajectoryLateralFraction ?: previous.normalizedLateral
        val currentLateral = current.trajectoryLateralFraction ?: current.normalizedLateral
        if (previousLateral != null && currentLateral != null &&
            previousLateral.isFinite() && currentLateral.isFinite() &&
            kotlin.math.abs(currentLateral - previousLateral) > maxLateralStep(gapMs)
        ) {
            return FallTrackingRejection.LATERAL_JUMP
        }
        return FallTrackingRejection.NONE
    }

    private fun maxDownwardStepTicks(gapMs: Long): Float {
        val gapSeconds = gapMs / 1_000f
        val initialLimit = maxOf(
            MIN_MAX_DOWNWARD_STEP_TICKS,
            gapSeconds * MAX_DOWNWARD_SPEED_TICKS_PER_SECOND
        )
        val recentSpeed = recentDownwardSpeedTicksPerSecond() ?: return initialLimit
        val adaptiveLimit = maxOf(
            MIN_ADAPTIVE_DOWNWARD_STEP_TICKS,
            recentSpeed * gapSeconds * ADAPTIVE_DOWNWARD_MULTIPLIER +
                ADAPTIVE_DOWNWARD_TOLERANCE_TICKS
        )
        val physicalLimit = maxOf(
            MIN_MAX_DOWNWARD_STEP_TICKS,
            gapSeconds * MAX_ADAPTIVE_DOWNWARD_SPEED_TICKS_PER_SECOND
        )
        return minOf(adaptiveLimit, physicalLimit)
    }

    private fun recentDownwardSpeedTicksPerSecond(
        source: BallDetectionSource? = null
    ): Float? = recentDownwardSpeeds(source)?.lastOrNull()

    private fun recentTrajectorySpeedTicksPerSecond(
        source: BallDetectionSource? = null
    ): Float? {
        val speeds = recentDownwardSpeeds(source) ?: return null
        return if (speeds.size % 2 == 1) {
            speeds[speeds.size / 2]
        } else {
            (speeds[speeds.size / 2 - 1] + speeds[speeds.size / 2]) * 0.5f
        }
    }

    private fun recentDownwardSpeeds(
        source: BallDetectionSource? = null
    ): List<Float>? {
        if (recentObservations.size < MIN_SPEED_INTERVALS + 1) return null
        val speeds = ArrayList<Float>(recentObservations.size - 1)
        for (index in 1 until recentObservations.size) {
            val older = recentObservations[index - 1]
            val newer = recentObservations[index]
            if (source != null && (older.source != source || newer.source != source)) continue
            val olderTicks = older.positionTicks ?: continue
            val newerTicks = newer.positionTicks ?: continue
            val elapsedMs = elapsedMs(older.timestampNs, newer.timestampNs)
            if (elapsedMs <= 0L) continue
            val speed = (newerTicks - olderTicks) * 1_000f / elapsedMs
            if (speed.isFinite() && speed > 0f) speeds += speed
        }
        if (speeds.size < MIN_SPEED_INTERVALS) return null
        speeds.sort()
        return speeds
    }

    private fun remember(observation: FallFrameObservation) {
        val last = recentObservations.lastOrNull()
        if (last?.timestampNs == observation.timestampNs) {
            recentObservations[recentObservations.lastIndex] = observation
            return
        }
        recentObservations += observation
        if (recentObservations.size > MAX_RECENT_OBSERVATIONS) {
            recentObservations.removeAt(0)
        }
    }

    private fun maxLateralStep(gapMs: Long): Float = minOf(
        MAX_LATERAL_STEP,
        maxOf(
            MIN_MAX_LATERAL_STEP,
            gapMs / 1_000f * MAX_LATERAL_SPEED_PER_SECOND
        )
    )

    private fun rejectTracking(
        rejection: FallTrackingRejection,
        timestampNs: Long
    ): FallTrackingUpdate {
        lastRejection = rejection
        if (lastAcceptedNs != 0L &&
            elapsedMs(lastAcceptedNs, timestampNs) >= ballLostTimeoutMs()
        ) {
            phase = FallTrackingPhase.FINISHED
            return FallTrackingUpdate(phase, endReason = BallEndReason.BALL_LOST)
        }
        return FallTrackingUpdate(phase)
    }

    private fun expireConfirmation(timestampNs: Long) {
        val lastPendingNs = pending.lastOrNull()?.timestampNs ?: return
        if (phase == FallTrackingPhase.CONFIRMING &&
            elapsedMs(lastPendingNs, timestampNs) > confirmationGapMs()
        ) {
            pending.clear()
            recentObservations.clear()
            confirmationCadenceMs = 0f
            phase = FallTrackingPhase.WAITING
        }
    }

    private fun updateConfirmationCadence(gapMs: Long) {
        if (gapMs <= 0L) return
        confirmationCadenceMs = if (confirmationCadenceMs <= 0f) {
            gapMs.toFloat()
        } else {
            confirmationCadenceMs * (1f - CADENCE_ALPHA) + gapMs * CADENCE_ALPHA
        }
    }

    private fun confirmationGapMs(): Long {
        if (confirmationCadenceMs <= 0f) return expectedInitialGapMs
        val cadenceAllowance = (confirmationCadenceMs * CONFIRMATION_GAP_PERIODS).toLong()
        return cadenceAllowance.coerceIn(MIN_CONFIRMATION_GAP_MS, MAX_CONFIRMATION_GAP_MS)
    }

    private fun ballLostTimeoutMs(): Long {
        if (confirmationCadenceMs <= 0f) return MIN_BALL_LOST_MS
        return (confirmationCadenceMs * BALL_LOST_PERIODS).toLong()
            .coerceIn(MIN_BALL_LOST_MS, MAX_BALL_LOST_MS)
    }

    private fun BallDetectionSource.isStrong(): Boolean {
        return this == BallDetectionSource.BACKGROUND ||
            this == BallDetectionSource.FUSED ||
            this == BallDetectionSource.CONVGRU ||
            this == BallDetectionSource.CLASSICAL_HEATMAP ||
            this == BallDetectionSource.TINY_HEATMAP ||
            this == BallDetectionSource.LOCAL_TRACKER
    }

    private fun elapsedMs(startNs: Long, endNs: Long): Long {
        if (endNs <= startNs) return 0L
        return (endNs - startNs) / 1_000_000L
    }
}
