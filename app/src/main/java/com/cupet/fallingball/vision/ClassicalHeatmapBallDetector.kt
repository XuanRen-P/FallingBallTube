package com.cupet.fallingball.vision

import java.nio.ByteBuffer
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.sqrt

enum class ClassicalBallPolarity {
    DARK,
    BRIGHT
}

/** Frozen phone profiles derived from the PC heatmap-C experiment profiles. */
data class ClassicalHeatmapProfile(
    val name: String,
    val polarity: ClassicalBallPolarity = ClassicalBallPolarity.DARK,
    val minimumScore: Float = 0.18f,
    val warmupFrames: Int = 8,
    val startupConfirmFrames: Int = 3,
    val startupMinDropPx: Float = 2.5f,
    val startupMinMedianStepPx: Float = 0f,
    val startupMinYFraction: Float = 0.03f,
    val startupMaxYFraction: Float = 0.55f,
    val startupXMinFraction: Float = 0.25f,
    val startupXMaxFraction: Float = 0.55f,
    val startupTinyPeaks: Boolean = false,
    val startupNormalContinuityWithTinyPeaks: Boolean = true,
    val startupTinyConfirmFrames: Int = 16,
    val startupTinyMinDropPx: Float = 8f,
    val startupTinyMinMedianStepPx: Float = 0f,
    val startupTinyMinYFraction: Float = 0.03f,
    val startupTinyXMinFraction: Float = 0.25f,
    val startupTinyXMaxFraction: Float = 0.55f,
    val tinyDynamicTracking: Boolean = false,
    val tinyDynamicTrackingAfterTinyStartOnly: Boolean = false,
    val searchAheadPx: Float = 36f,
    val searchSidePx: Float = 12f
) {
    companion object {
        fun forBall(sphereRadiusMm: Double?, material: String?): ClassicalHeatmapProfile {
            val diameterMm = sphereRadiusMm?.times(2.0)
            val bright = material.orEmpty().contains("陶瓷", ignoreCase = true) ||
                material.orEmpty().contains("ceramic", ignoreCase = true)
            if (bright) {
                if ((diameterMm ?: 0.0) >= 3.0) {
                    return ClassicalHeatmapProfile(
                        name = "bright_3mm_partial",
                        polarity = ClassicalBallPolarity.BRIGHT
                    )
                }
                return ClassicalHeatmapProfile(
                    name = "bright_2_5mm_partial",
                    polarity = ClassicalBallPolarity.BRIGHT,
                    startupConfirmFrames = 8,
                    startupMinDropPx = 4f,
                    startupMinMedianStepPx = 0.3f,
                    startupMinYFraction = 0.12f,
                    startupTinyPeaks = (diameterMm ?: 0.0) < 3.0,
                    startupTinyConfirmFrames = 8,
                    startupTinyMinDropPx = 4f,
                    startupTinyMinMedianStepPx = 0.3f,
                    startupTinyMinYFraction = 0.12f,
                    tinyDynamicTracking = true
                )
            }
            return when {
                diameterMm == null -> ClassicalHeatmapProfile(name = "dark_default")
                diameterMm >= 5.0 -> ClassicalHeatmapProfile(
                    name = "dark_5mm_cross_camera_validated",
                    startupTinyPeaks = true,
                    startupNormalContinuityWithTinyPeaks = false,
                    startupTinyConfirmFrames = 8,
                    startupTinyMinDropPx = 24f,
                    startupTinyMinMedianStepPx = 2f,
                    startupTinyMinYFraction = 0.20f,
                    startupTinyXMaxFraction = 0.68f,
                    tinyDynamicTracking = true,
                    tinyDynamicTrackingAfterTinyStartOnly = true
                )
                diameterMm >= 4.0 -> ClassicalHeatmapProfile(name = "dark_large_validated")
                diameterMm >= 2.5 -> ClassicalHeatmapProfile(
                    name = "dark_mid_validated",
                    startupConfirmFrames = 8,
                    startupMinDropPx = 10f,
                    startupMinMedianStepPx = 1f,
                    startupMinYFraction = 0.12f,
                    startupTinyPeaks = true,
                    startupTinyConfirmFrames = 8,
                    startupTinyMinDropPx = 10f,
                    startupTinyMinMedianStepPx = 1f,
                    startupTinyMinYFraction = 0.12f
                )
                diameterMm >= 2.0 -> ClassicalHeatmapProfile(
                    name = "dark_2mm_experimental",
                    startupConfirmFrames = 8,
                    startupMinDropPx = 6f,
                    startupMinMedianStepPx = 0.35f,
                    startupMinYFraction = 0.12f,
                    startupTinyPeaks = true,
                    startupTinyConfirmFrames = 24,
                    startupTinyMinDropPx = 18f,
                    startupTinyMinMedianStepPx = 0.5f,
                    startupTinyMinYFraction = 0.03f,
                    startupTinyXMaxFraction = 0.58f,
                    tinyDynamicTracking = true,
                    searchAheadPx = 12f,
                    searchSidePx = 6f
                )
                else -> ClassicalHeatmapProfile(
                    name = "dark_small_experimental",
                    startupConfirmFrames = 8,
                    startupMinDropPx = 6f,
                    startupMinMedianStepPx = 0.35f,
                    startupMinYFraction = 0.12f,
                    startupTinyPeaks = true,
                    startupTinyConfirmFrames = 16,
                    startupTinyMinDropPx = 6f,
                    startupTinyMinMedianStepPx = 0.35f,
                    startupTinyMinYFraction = 0.03f,
                    tinyDynamicTracking = true
                )
            }
        }
    }
}

data class ClassicalHeatmapDetection(
    val roiX: Float,
    val roiY: Float,
    val score: Float
)

enum class ClassicalHeatmapPhase {
    WARMUP,
    STARTUP_NORMAL,
    STARTUP_TINY,
    TRACKING
}

enum class ClassicalHeatmapRejection {
    NONE,
    WARMUP,
    NO_CANDIDATE,
    OUTSIDE_STARTUP_REGION,
    STARTUP_COLLECTING,
    STARTUP_DISCONTINUITY,
    STARTUP_DURATION,
    STARTUP_DROP,
    STARTUP_MEDIAN_STEP,
    TRACKING_WINDOW_MISS,
    TRACKING_STATIONARY
}

data class ClassicalHeatmapDiagnostics(
    val rawPeakScore: Float,
    val strongestCandidateScore: Float?,
    val tinyPeakScore: Float?,
    val normalCandidateCount: Int,
    val tinyCandidateCount: Int,
    val normalStartupSamples: Int,
    val tinyStartupSamples: Int,
    val phase: ClassicalHeatmapPhase,
    val rejection: ClassicalHeatmapRejection,
    val trackingFromTiny: Boolean,
    val recoveryPeakScore: Float? = null
)

data class ClassicalHeatmapResult(
    val detection: ClassicalHeatmapDetection?,
    val backgroundReady: Boolean,
    val framesSeen: Int,
    val resetByGap: Boolean = false,
    val diagnostics: ClassicalHeatmapDiagnostics
)

/**
 * Causal, non-neural port of the PC heatmap-C detector.
 *
 * The caller supplies the already stabilized 152x1024 grayscale tube ROI. The detector keeps
 * only past background/motion state, confirms a falling start, and publishes one 2-D centre.
 */
class ClassicalHeatmapBallDetector(
    private val width: Int = TubeRoiSampler.OUTPUT_WIDTH,
    private val height: Int = TubeRoiSampler.OUTPUT_HEIGHT,
    initialProfile: ClassicalHeatmapProfile = ClassicalHeatmapProfile.forBall(null, null)
) {
    companion object {
        /**
         * The PC heatmap-C path does not clear its causal background for a normal model
         * cadence. On phones the production path commonly delivers inputs about 0.43-0.50 s
         * apart, so 500 ms treats scheduler jitter as a capture pause and restarts warmup.
         * Keep a bounded guard for a genuine multi-second camera pause instead.
         */
        const val MAXIMUM_FRAME_GAP_NS = 2_000_000_000L
        private const val MAX_BACKGROUND_FRAMES = 45
        private const val MAX_MISSED_FRAMES = 6
        private const val MAX_STATIONARY_FRAMES = 10
        private const val MIN_TRACK_PROGRESS_PX = 2.5f
        private const val TRACK_PROGRESS_WINDOW = 12
        private const val MAX_SPEED_PX_PER_SECOND = 360f
        private const val MAX_HORIZONTAL_SPEED_PX_PER_SECOND = 180f
        private const val ALLOW_UPWARD_PX_PER_FRAME = 3f
        private const val TRACK_RECOVERY_MIN_SCORE = 0.12f
        private const val TRACK_RECOVERY_SIDE_PX = 9f
        private const val TRACK_RECOVERY_AHEAD_PX = 20f
        private const val VALID_X_MIN_FRACTION = 0.20f
        private const val VALID_X_MAX_FRACTION = 0.64f
        private const val VALID_Y_MAX_FRACTION = 0.965f
        private const val NOMINAL_FPS = 60f
        private const val NOMINAL_FRAME_PERIOD_SECONDS = 1f / NOMINAL_FPS
        private const val NOMINAL_FRAME_PERIOD_NS = 16_666_667L
        private const val MINIMUM_STARTUP_SAMPLES = 3
        private const val STARTUP_CENTER_PRIOR_WEIGHT = 0.35f
    }

    private data class Candidate(
        val x: Float,
        val y: Float,
        val confidence: Float,
        val area: Int
    )

    private val pixelCount = width * height
    private var profile = initialProfile
    private var pendingProfile: ClassicalHeatmapProfile? = null
    private val current = FloatArray(pixelCount)
    private val background = FloatArray(pixelCount)
    private val previous = FloatArray(pixelCount)
    private val local = FloatArray(pixelCount)
    private val residual = FloatArray(pixelCount)
    private val motion = FloatArray(pixelCount)
    private val score = FloatArray(pixelCount)
    private val tinyScore = FloatArray(pixelCount)
    private val scratchA = FloatArray(pixelCount)
    private val scratchB = FloatArray(pixelCount)
    private val scratchC = FloatArray(pixelCount)
    private val histogram = IntArray(1024)
    private val binaryA = ByteArray(pixelCount)
    private val binaryB = ByteArray(pixelCount)
    private val visited = ByteArray(pixelCount)
    private val componentQueue = IntArray(pixelCount)
    private val rowDeque = IntArray(max(width, height) + 2)

    private var framesSeen = 0
    private var lastTimestampNs = 0L
    private var pendingCandidate: Candidate? = null
    private val pendingY = ArrayDeque<Float>()
    private val pendingTimestampsNs = ArrayDeque<Long>()
    private var tinyPendingCandidate: Candidate? = null
    private val tinyPendingY = ArrayDeque<Float>()
    private val tinyPendingTimestampsNs = ArrayDeque<Long>()
    private var lastCandidate: Candidate? = null
    private var lastCandidateTimestampNs = 0L
    private var velocityPxPerSecond = 0f
    private val trackY = ArrayDeque<Float>()
    private var stationaryFrames = 0
    private var missedFrames = 0
    private var trackStartedFromTiny = false
    private var observationRejection = ClassicalHeatmapRejection.WARMUP
    private var observationRecoveryPeakScore: Float? = null

    init {
        require(width > 16 && height > 16)
    }

    @Synchronized
    fun setProfile(next: ClassicalHeatmapProfile) {
        if (next != profile) pendingProfile = next
    }

    @Synchronized
    fun currentProfile(): ClassicalHeatmapProfile = pendingProfile ?: profile

    @Synchronized
    fun reset() {
        resetState(clearPixels = true)
    }

    @Synchronized
    fun observe(
        input: ByteBuffer,
        timestampNs: Long,
        stabilization: RoiStabilizationTransform = RoiStabilizationTransform(0f, 0f, 1f)
    ): ClassicalHeatmapResult {
        pendingProfile?.let {
            profile = it
            pendingProfile = null
            resetState(clearPixels = true)
        }
        require(input.remaining() >= pixelCount) { "Expected $pixelCount grayscale bytes" }
        val resetByGap = lastTimestampNs > 0L &&
            (timestampNs <= lastTimestampNs || timestampNs - lastTimestampNs > MAXIMUM_FRAME_GAP_NS)
        if (resetByGap) resetState(clearPixels = true)
        val source = input.duplicate()
        for (i in 0 until pixelCount) current[i] = (source.get().toInt() and 0xff).toFloat()
        gaussianBlur(current, scratchA, SIGMA_PREPROCESS)
        scratchA.copyInto(current)

        if (framesSeen == 0) {
            current.copyInto(background)
        } else {
            updateBackground()
        }
        framesSeen = (framesSeen + 1).coerceAtMost(MAX_BACKGROUND_FRAMES)

        compactResponse()
        backgroundResidual()
        causalMotion()
        for (i in 0 until pixelCount) {
            score[i] = 0.45f * local[i] + 0.35f * residual[i] + 0.20f * motion[i]
        }
        val rawPeakScore = normalizeAndMask(score, stabilization)

        val ready = framesSeen >= profile.warmupFrames && lastTimestampNs > 0L
        var selected: Candidate? = null
        var startedFromTiny = false
        var phase = ClassicalHeatmapPhase.WARMUP
        var normalCandidateCount = 0
        var tinyCandidateCount = 0
        var strongestCandidateScore: Float? = null
        var tinyPeakScore: Float? = null
        observationRejection = ClassicalHeatmapRejection.WARMUP
        observationRecoveryPeakScore = null
        if (ready) {
            if (lastCandidate == null) {
                phase = ClassicalHeatmapPhase.STARTUP_NORMAL
                val normalCandidates = extractCandidates(score, preserveTiny = false)
                    .filter { inStartupRegion(it, tiny = false, stabilization = stabilization) }
                normalCandidateCount = normalCandidates.size
                strongestCandidateScore = normalCandidates.maxOfOrNull { it.confidence }
                val normal = selectStartupCandidate(
                    normalCandidates,
                    tiny = false,
                    timestampNs = timestampNs,
                    stabilization = stabilization
                )
                selected = confirmStartup(
                    normal,
                    tiny = false,
                    timestampNs = timestampNs,
                    stabilization = stabilization
                )
                if (selected != null) {
                    clearPending(tiny = true)
                } else if (profile.startupTinyPeaks) {
                    phase = ClassicalHeatmapPhase.STARTUP_TINY
                    tinyPeakScore = tinyDynamicResponse(stabilization)
                    val tinyCandidates = extractCandidates(tinyScore, preserveTiny = true)
                        .filter { inStartupRegion(it, tiny = true, stabilization = stabilization) }
                    tinyCandidateCount = tinyCandidates.size
                    tinyCandidates.maxOfOrNull { it.confidence }?.let { candidateScore ->
                        strongestCandidateScore = maxOf(strongestCandidateScore ?: 0f, candidateScore)
                    }
                    val tiny = selectStartupCandidate(
                        tinyCandidates,
                        tiny = true,
                        timestampNs = timestampNs,
                        stabilization = stabilization
                    )
                    selected = confirmStartup(
                        tiny,
                        tiny = true,
                        timestampNs = timestampNs,
                        stabilization = stabilization
                    )
                    if (selected != null) {
                        startedFromTiny = true
                        clearPending(tiny = false)
                    }
                }
            } else {
                phase = ClassicalHeatmapPhase.TRACKING
                val useTiny = profile.startupTinyPeaks && profile.tinyDynamicTracking &&
                    (!profile.tinyDynamicTrackingAfterTinyStartOnly || trackStartedFromTiny)
                if (useTiny) tinyPeakScore = tinyDynamicResponse(stabilization)
                val activeScore = if (useTiny) tinyScore else score
                val trackingCandidates = extractCandidates(activeScore, preserveTiny = useTiny)
                strongestCandidateScore = trackingCandidates.maxOfOrNull { it.confidence }
                if (useTiny) tinyCandidateCount = trackingCandidates.size
                else normalCandidateCount = trackingCandidates.size
                selected = selectTrackingCandidate(
                    trackingCandidates,
                    timestampNs
                ) ?: recoverTrackCandidate(activeScore, timestampNs)
                if (selected == null) {
                    observationRejection = ClassicalHeatmapRejection.TRACKING_WINDOW_MISS
                }
            }
        }

        var published: ClassicalHeatmapDetection? = null
        if (selected != null && acceptTrackCandidate(selected)) {
            if (lastCandidate == null) trackStartedFromTiny = startedFromTiny
            updateTrack(selected, timestampNs)
            missedFrames = 0
            published = ClassicalHeatmapDetection(
                roiX = selected.x,
                roiY = selected.y,
                score = selected.confidence
            )
            observationRejection = ClassicalHeatmapRejection.NONE
        } else if (lastCandidate != null) {
            missedFrames++
            if (missedFrames > MAX_MISSED_FRAMES) clearTrack()
        }

        current.copyInto(previous)
        lastTimestampNs = timestampNs
        return ClassicalHeatmapResult(
            detection = published,
            backgroundReady = ready,
            framesSeen = framesSeen,
            resetByGap = resetByGap,
            diagnostics = ClassicalHeatmapDiagnostics(
                rawPeakScore = rawPeakScore,
                strongestCandidateScore = strongestCandidateScore ?: published?.score,
                tinyPeakScore = tinyPeakScore,
                normalCandidateCount = normalCandidateCount,
                tinyCandidateCount = tinyCandidateCount,
                normalStartupSamples = pendingY.size,
                tinyStartupSamples = tinyPendingY.size,
                phase = phase,
                rejection = observationRejection,
                trackingFromTiny = trackStartedFromTiny,
                recoveryPeakScore = observationRecoveryPeakScore
            )
        )
    }

    private fun resetState(clearPixels: Boolean) {
        framesSeen = 0
        lastTimestampNs = 0L
        clearPending(false)
        clearPending(true)
        clearTrack()
        if (clearPixels) {
            background.fill(0f)
            previous.fill(0f)
        }
    }

    private fun updateBackground() {
        val dark = profile.polarity == ClassicalBallPolarity.DARK
        for (i in 0 until pixelCount) {
            val bg = background[i]
            val value = current[i]
            val nonTarget = if (dark) value >= bg else value <= bg
            val updated = bg + (if (nonTarget) 0.08f else 0.004f) * (value - bg)
            background[i] = if (dark) max(updated, value - 2f) else minOf(updated, value + 2f)
        }
    }

    private fun compactResponse() {
        gaussianBlur(current, scratchA, SIGMA_LOCAL_BACKGROUND)
        val dark = profile.polarity == ClassicalBallPolarity.DARK
        for (i in 0 until pixelCount) {
            local[i] = max(if (dark) scratchA[i] - current[i] else current[i] - scratchA[i], 0f)
        }
        horizontalOpening(local, scratchA, ((width * 0.20f).roundToInt() or 1).coerceAtLeast(9))
        maxFilter2d(scratchA, scratchB, 2)
        for (i in 0 until pixelCount) {
            scratchA[i] = max(local[i] - scratchA[i] - 0.20f * scratchB[i], 0f)
        }
        gaussianBlur(scratchA, local, SIGMA_COMPACT)
    }

    private fun backgroundResidual() {
        val dark = profile.polarity == ClassicalBallPolarity.DARK
        for (i in 0 until pixelCount) {
            residual[i] = max(if (dark) background[i] - current[i] else current[i] - background[i], 0f)
        }
        val baseline = percentile(residual, 0.50f)
        for (i in 0 until pixelCount) residual[i] = max(residual[i] - baseline, 0f)
        horizontalOpening(residual, scratchA, ((width * 0.25f).roundToInt() or 1).coerceAtLeast(9))
        for (i in 0 until pixelCount) scratchA[i] = max(residual[i] - 0.85f * scratchA[i], 0f)
        gaussianBlur(scratchA, residual, SIGMA_RESIDUAL)
    }

    private fun causalMotion() {
        if (framesSeen <= 1) {
            motion.fill(0f)
            return
        }
        val dark = profile.polarity == ClassicalBallPolarity.DARK
        for (i in 0 until pixelCount) {
            motion[i] = max(if (dark) previous[i] - current[i] else current[i] - previous[i], 0f)
        }
        val baseline = percentile(motion, 0.50f)
        for (i in 0 until pixelCount) motion[i] = max(motion[i] - baseline, 0f)
        gaussianBlur(motion, scratchA, SIGMA_MOTION_BACKGROUND)
        for (i in 0 until pixelCount) scratchA[i] = max(motion[i] - 0.85f * scratchA[i], 0f)
        gaussianBlur(scratchA, motion, SIGMA_RESIDUAL)
    }

    private fun tinyDynamicResponse(stabilization: RoiStabilizationTransform): Float {
        for (i in 0 until pixelCount) tinyScore[i] = 0.65f * residual[i] + 0.35f * motion[i]
        horizontalOpening(
            tinyScore,
            scratchA,
            ((width * 0.07f).roundToInt() or 1).coerceAtLeast(7)
        )
        for (i in 0 until pixelCount) tinyScore[i] = max(tinyScore[i] - 0.95f * scratchA[i], 0f)
        return normalizeAndMask(tinyScore, stabilization)
    }

    private fun normalizeAndMask(
        values: FloatArray,
        stabilization: RoiStabilizationTransform
    ): Float {
        val high = percentile(values, 0.995f, (width * 0.05f).toInt(), (width * 0.95f).toInt())
            .coerceAtLeast(1e-3f)
        val x0 = (width * VALID_X_MIN_FRACTION + stabilization.sourceToStabilizedX).roundToInt()
        val x1 = (width * VALID_X_MAX_FRACTION + stabilization.sourceToStabilizedX).roundToInt()
        val y0 = (height * 0.03f + stabilization.sourceToStabilizedY).roundToInt()
        val y1 = (height * VALID_Y_MAX_FRACTION + stabilization.sourceToStabilizedY).roundToInt()
        var peak = 0f
        for (y in 0 until height) {
            val row = y * width
            for (x in 0 until width) {
                val i = row + x
                values[i] = if (x in x0 until x1 && y in y0 until y1) {
                    (values[i] / high).coerceIn(0f, 1f)
                } else {
                    0f
                }
                peak = max(peak, values[i])
            }
        }
        return peak
    }

    private fun extractCandidates(values: FloatArray, preserveTiny: Boolean): List<Candidate> {
        val positiveP = percentile(values, 0.988f, positiveOnly = true)
        val threshold = max(profile.minimumScore, positiveP)
        binaryA.fill(0)
        for (y in 0 until height) {
            val y0 = (y - 3).coerceAtLeast(0)
            val y1 = (y + 3).coerceAtMost(height - 1)
            for (x in 0 until width) {
                val i = y * width + x
                val value = values[i]
                if (value < threshold) continue
                val x0 = (x - 5).coerceAtLeast(0)
                val x1 = (x + 5).coerceAtMost(width - 1)
                var maximum = value
                loop@ for (yy in y0..y1) {
                    var p = yy * width + x0
                    for (xx in x0..x1) {
                        if (values[p] > maximum + 1e-6f) {
                            maximum = values[p]
                            break@loop
                        }
                        p++
                    }
                }
                if (maximum <= value + 1e-6f) binaryA[i] = 1
            }
        }
        if (!preserveTiny) {
            binaryErode3(binaryA, binaryB)
            binaryDilate3(binaryB, binaryA)
        }
        visited.fill(0)
        val result = ArrayList<Candidate>()
        for (start in 0 until pixelCount) {
            if (binaryA[start].toInt() == 0 || visited[start].toInt() != 0) continue
            var head = 0
            var tail = 0
            componentQueue[tail++] = start
            visited[start] = 1
            var area = 0
            var sumX = 0f
            var sumY = 0f
            var peak = 0f
            var density = 0f
            var minX = width
            var maxX = 0
            var minY = height
            var maxY = 0
            while (head < tail) {
                val index = componentQueue[head++]
                val y = index / width
                val x = index - y * width
                area++
                sumX += x
                sumY += y
                peak = max(peak, values[index])
                density += values[index]
                minX = minOf(minX, x)
                maxX = maxOf(maxX, x)
                minY = minOf(minY, y)
                maxY = maxOf(maxY, y)
                for (yy in (y - 1).coerceAtLeast(0)..(y + 1).coerceAtMost(height - 1)) {
                    for (xx in (x - 1).coerceAtLeast(0)..(x + 1).coerceAtMost(width - 1)) {
                        val next = yy * width + xx
                        if (binaryA[next].toInt() != 0 && visited[next].toInt() == 0) {
                            visited[next] = 1
                            componentQueue[tail++] = next
                        }
                    }
                }
            }
            val minimumArea = if (preserveTiny) 1 else 2
            val componentWidth = maxX - minX + 1
            val componentHeight = maxY - minY + 1
            if (area !in minimumArea..180 ||
                componentWidth > width * 0.24f || componentHeight > height * 0.08f ||
                componentWidth > componentHeight * 5 || componentHeight > componentWidth * 5
            ) {
                continue
            }
            val confidence = (0.75f * peak + 0.25f * minOf(1f, density / area * 2f))
                .coerceIn(0f, 1f)
            result += Candidate(sumX / area, sumY / area, confidence, area)
        }
        return result
    }

    private fun selectStartupCandidate(
        candidates: List<Candidate>,
        tiny: Boolean,
        timestampNs: Long,
        stabilization: RoiStabilizationTransform
    ): Candidate? {
        if (candidates.isEmpty()) return null
        val pending = if (tiny) tinyPendingCandidate else pendingCandidate
        val pendingTimestamps = if (tiny) tinyPendingTimestampsNs else pendingTimestampsNs
        if (pending != null && (tiny || profile.startupNormalContinuityWithTinyPeaks)) {
            val dt = pendingTimestamps.lastOrNull()?.let { previousTimestampNs ->
                elapsedSeconds(previousTimestampNs, timestampNs)
            } ?: NOMINAL_FRAME_PERIOD_SECONDS
            val tolerance = max(
                max(12f, height * 0.018f),
                MAX_SPEED_PX_PER_SECOND * dt + profile.searchAheadPx
            )
            val upward = ALLOW_UPWARD_PX_PER_FRAME * max(1f, dt * NOMINAL_FPS)
            if (pendingStartupStalled(tiny, timestampNs)) {
                val alternative = candidates.filter {
                    abs(it.x - pending.x) > profile.searchSidePx ||
                        it.y < pending.y - upward ||
                        it.y > pending.y + tolerance
                }
                clearPending(tiny)
                return bestStartupCandidate(
                    if (alternative.isNotEmpty()) alternative else candidates,
                    tiny,
                    stabilization
                )
            }
            candidates.filter {
                abs(it.x - pending.x) <= profile.searchSidePx &&
                    it.y >= pending.y - upward &&
                    it.y <= pending.y + tolerance
            }.maxByOrNull { it.confidence }?.let { return it }
        }
        return bestStartupCandidate(candidates, tiny, stabilization)
    }

    private fun bestStartupCandidate(
        candidates: List<Candidate>,
        tiny: Boolean,
        stabilization: RoiStabilizationTransform
    ): Candidate? {
        val minX = if (tiny) profile.startupTinyXMinFraction else profile.startupXMinFraction
        val maxX = if (tiny) profile.startupTinyXMaxFraction else profile.startupXMaxFraction
        val expectedStableX = width * (minX + maxX) * 0.5f +
            stabilization.sourceToStabilizedX
        val halfWidth = max(1f, width * (maxX - minX) * 0.5f)
        return candidates.maxWithOrNull(compareBy<Candidate> {
            it.confidence - STARTUP_CENTER_PRIOR_WEIGHT *
                abs(it.x - expectedStableX) / halfWidth
        }.thenBy { it.area })
    }

    private fun pendingStartupStalled(tiny: Boolean, timestampNs: Long): Boolean {
        val history = if (tiny) tinyPendingY else pendingY
        val timestamps = if (tiny) tinyPendingTimestampsNs else pendingTimestampsNs
        if (history.size < MINIMUM_STARTUP_SAMPLES || timestamps.isEmpty()) return false
        val confirmFrames = if (tiny) profile.startupTinyConfirmFrames else profile.startupConfirmFrames
        val requiredDurationNs = ((max(2, confirmFrames) - 1) * NOMINAL_FRAME_PERIOD_NS)
            .toLong()
        val elapsedNs = (timestampNs - timestamps.first()).coerceAtLeast(0L)
        if (elapsedNs < requiredDurationNs) return false
        val minimumDrop = if (tiny) profile.startupTinyMinDropPx else profile.startupMinDropPx
        val minimumStep = if (tiny) {
            profile.startupTinyMinMedianStepPx
        } else {
            profile.startupMinMedianStepPx
        }
        return history.last() - history.first() < minimumDrop || medianSteps(history) < minimumStep
    }

    private fun confirmStartup(
        candidate: Candidate?,
        tiny: Boolean,
        timestampNs: Long,
        stabilization: RoiStabilizationTransform
    ): Candidate? {
        val valid = candidate?.takeIf { inStartupRegion(it, tiny, stabilization) }
        if (valid == null) {
            observationRejection = if (candidate == null) {
                ClassicalHeatmapRejection.NO_CANDIDATE
            } else {
                ClassicalHeatmapRejection.OUTSIDE_STARTUP_REGION
            }
            clearPending(tiny)
            return null
        }
        val previousPending = if (tiny) tinyPendingCandidate else pendingCandidate
        val history = if (tiny) tinyPendingY else pendingY
        val timestamps = if (tiny) tinyPendingTimestampsNs else pendingTimestampsNs
        if (previousPending == null) {
            if (tiny) tinyPendingCandidate = valid else pendingCandidate = valid
            history.addLast(valid.y)
            timestamps.addLast(timestampNs)
            observationRejection = ClassicalHeatmapRejection.STARTUP_COLLECTING
            return null
        }
        val dt = timestamps.lastOrNull()?.let { previousTimestampNs ->
            elapsedSeconds(previousTimestampNs, timestampNs)
        } ?: NOMINAL_FRAME_PERIOD_SECONDS
        val tolerance = max(
            max(12f, height * 0.018f),
            MAX_SPEED_PX_PER_SECOND * dt + profile.searchAheadPx
        )
        val upward = ALLOW_UPWARD_PX_PER_FRAME * max(1f, dt * NOMINAL_FPS)
        if (valid.y < previousPending.y - upward ||
            abs(valid.y - previousPending.y) > tolerance
        ) {
            clearPending(tiny)
            if (tiny) tinyPendingCandidate = valid else pendingCandidate = valid
            history.addLast(valid.y)
            timestamps.addLast(timestampNs)
            observationRejection = ClassicalHeatmapRejection.STARTUP_DISCONTINUITY
            return null
        }
        if (tiny) tinyPendingCandidate = valid else pendingCandidate = valid
        history.addLast(valid.y)
        timestamps.addLast(timestampNs)
        val confirmFrames = if (tiny) profile.startupTinyConfirmFrames else profile.startupConfirmFrames
        val minimumDrop = if (tiny) profile.startupTinyMinDropPx else profile.startupMinDropPx
        val minimumStep = if (tiny) {
            profile.startupTinyMinMedianStepPx
        } else {
            profile.startupMinMedianStepPx
        }
        val drop = history.last() - history.first()
        val medianStep = medianSteps(history)
        val requiredDurationNs = ((max(2, confirmFrames) - 1) * NOMINAL_FRAME_PERIOD_NS)
            .toLong()
        val elapsedNs = (timestamps.last() - timestamps.first()).coerceAtLeast(0L)
        if (history.size >= MINIMUM_STARTUP_SAMPLES && elapsedNs >= requiredDurationNs &&
            drop >= minimumDrop && medianStep >= minimumStep
        ) {
            clearPending(tiny)
            observationRejection = ClassicalHeatmapRejection.NONE
            return valid
        }
        observationRejection = when {
            history.size < MINIMUM_STARTUP_SAMPLES || elapsedNs < requiredDurationNs ->
                ClassicalHeatmapRejection.STARTUP_DURATION
            drop < minimumDrop -> ClassicalHeatmapRejection.STARTUP_DROP
            else -> ClassicalHeatmapRejection.STARTUP_MEDIAN_STEP
        }
        return null
    }

    private fun inStartupRegion(
        candidate: Candidate,
        tiny: Boolean,
        stabilization: RoiStabilizationTransform
    ): Boolean {
        val sourceX = candidate.x - stabilization.sourceToStabilizedX
        val sourceY = candidate.y - stabilization.sourceToStabilizedY
        val minX = if (tiny) profile.startupTinyXMinFraction else profile.startupXMinFraction
        val maxX = if (tiny) profile.startupTinyXMaxFraction else profile.startupXMaxFraction
        val minY = if (tiny) profile.startupTinyMinYFraction else profile.startupMinYFraction
        return sourceX in width * minX..width * maxX &&
            sourceY in height * minY..height * profile.startupMaxYFraction
    }

    private fun selectTrackingCandidate(candidates: List<Candidate>, timestampNs: Long): Candidate? {
        val last = lastCandidate ?: return candidates.maxByOrNull { it.confidence }
        if (candidates.isEmpty()) return null
        val dt = elapsedSeconds(timestampNs)
        val predicted = last.y + velocityPxPerSecond * dt
        val maxStep = MAX_SPEED_PX_PER_SECOND * dt + profile.searchAheadPx
        val maxSide = MAX_HORIZONTAL_SPEED_PX_PER_SECOND * dt + profile.searchSidePx
        val predictionTolerance = max(8f, abs(velocityPxPerSecond) * dt * 0.5f + 8f)
        val minY = last.y - ALLOW_UPWARD_PX_PER_FRAME * max(1f, dt * 60f)
        val maxY = minOf(height - 1f, predicted + maxStep)
        return candidates.filter {
            it.y in minY..maxY && abs(it.x - last.x) <= maxSide
        }.maxByOrNull {
            it.confidence - 0.75f * abs(it.y - predicted) / predictionTolerance -
                0.25f * abs(it.x - last.x) / max(1f, maxSide)
        }
    }

    private fun recoverTrackCandidate(values: FloatArray, timestampNs: Long): Candidate? {
        val last = lastCandidate ?: return null
        val dt = elapsedSeconds(timestampNs)
        val predictedY = last.y + velocityPxPerSecond * dt
        val side = TRACK_RECOVERY_SIDE_PX + 0.5f * MAX_HORIZONTAL_SPEED_PX_PER_SECOND * dt
        val x0 = floor(last.x - side).toInt().coerceAtLeast(0)
        val x1 = ceil(last.x + side + 1f).toInt().coerceAtMost(width)
        val upward = ALLOW_UPWARD_PX_PER_FRAME * max(1f, dt * 60f)
        val y0 = floor(last.y - upward).toInt().coerceAtLeast(0)
        val y1 = ceil(max(last.y, predictedY) + TRACK_RECOVERY_AHEAD_PX + 1f)
            .toInt().coerceAtMost(height)
        var peak = 0f
        var peakX = -1
        var peakY = -1
        for (y in y0 until y1) {
            for (x in x0 until x1) {
                val value = values[y * width + x]
                if (value > peak) {
                    peak = value
                    peakX = x
                    peakY = y
                }
            }
        }
        observationRecoveryPeakScore = peak
        if (peakX < 0 || peak < TRACK_RECOVERY_MIN_SCORE) return null
        val support = max(TRACK_RECOVERY_MIN_SCORE, peak * 0.55f)
        var area = 0
        var total = 0f
        var weightedX = 0f
        var weightedY = 0f
        for (y in (peakY - 3).coerceAtLeast(0)..(peakY + 3).coerceAtMost(height - 1)) {
            for (x in (peakX - 3).coerceAtLeast(0)..(peakX + 3).coerceAtMost(width - 1)) {
                val value = values[y * width + x]
                if (value >= support) {
                    area++
                    total += value
                    weightedX += x * value
                    weightedY += y * value
                }
            }
        }
        if (area !in 1..180 || total <= 0f) return null
        return Candidate(weightedX / total, weightedY / total, peak.coerceIn(0f, 1f), area)
    }

    private fun acceptTrackCandidate(candidate: Candidate): Boolean {
        val last = lastCandidate
        if (last == null) {
            trackY.clear()
            trackY.addLast(candidate.y)
            stationaryFrames = 0
            return true
        }
        val delta = candidate.y - last.y
        stationaryFrames = if (delta < -ALLOW_UPWARD_PX_PER_FRAME || delta <= 0.5f) {
            stationaryFrames + 1
        } else {
            0
        }
        trackY.addLast(candidate.y)
        while (trackY.size > TRACK_PROGRESS_WINDOW) trackY.removeFirst()
        val progress = trackY.last() - trackY.first()
        if ((stationaryFrames >= MAX_STATIONARY_FRAMES && progress < MIN_TRACK_PROGRESS_PX) ||
            (trackY.size >= TRACK_PROGRESS_WINDOW && progress < -MIN_TRACK_PROGRESS_PX)
        ) {
            clearTrack()
            observationRejection = ClassicalHeatmapRejection.TRACKING_STATIONARY
            return false
        }
        return true
    }

    private fun updateTrack(candidate: Candidate, timestampNs: Long) {
        lastCandidate?.let { previousCandidate ->
            val dt = elapsedSeconds(timestampNs)
            val measured = (candidate.y - previousCandidate.y) / dt
            velocityPxPerSecond = 0.75f * velocityPxPerSecond + 0.25f * measured
        }
        lastCandidate = candidate
        lastCandidateTimestampNs = timestampNs
    }

    private fun clearTrack() {
        lastCandidate = null
        lastCandidateTimestampNs = 0L
        velocityPxPerSecond = 0f
        trackY.clear()
        stationaryFrames = 0
        missedFrames = 0
        trackStartedFromTiny = false
    }

    private fun clearPending(tiny: Boolean) {
        if (tiny) {
            tinyPendingCandidate = null
            tinyPendingY.clear()
            tinyPendingTimestampsNs.clear()
        } else {
            pendingCandidate = null
            pendingY.clear()
            pendingTimestampsNs.clear()
        }
    }

    private fun elapsedSeconds(timestampNs: Long): Float {
        if (lastCandidateTimestampNs <= 0L || timestampNs <= lastCandidateTimestampNs) return 1f / 30f
        return elapsedSeconds(lastCandidateTimestampNs, timestampNs)
    }

    private fun elapsedSeconds(previousTimestampNs: Long, timestampNs: Long): Float =
        if (previousTimestampNs <= 0L || timestampNs <= previousTimestampNs) {
            NOMINAL_FRAME_PERIOD_SECONDS
        } else {
            ((timestampNs - previousTimestampNs) / 1_000_000_000f)
                .coerceIn(1f / 120f, 0.25f)
        }

    private fun medianSteps(values: ArrayDeque<Float>): Float {
        if (values.size < 2) return 0f
        var previousValue: Float? = null
        var count = 0
        for (value in values) {
            previousValue?.let { scratchA[count++] = value - it }
            previousValue = value
        }
        scratchA.sort(0, count)
        val middle = count / 2
        return if (count and 1 == 1) scratchA[middle]
        else (scratchA[middle - 1] + scratchA[middle]) * 0.5f
    }

    private fun gaussianBlur(input: FloatArray, output: FloatArray, kernel: FloatArray) {
        val radius = kernel.size / 2
        for (y in 0 until height) {
            val row = y * width
            for (x in 0 until width) {
                var sum = 0f
                for (k in -radius..radius) {
                    sum += input[row + (x + k).coerceIn(0, width - 1)] * kernel[k + radius]
                }
                scratchB[row + x] = sum
            }
        }
        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0f
                for (k in -radius..radius) {
                    sum += scratchB[(y + k).coerceIn(0, height - 1) * width + x] * kernel[k + radius]
                }
                output[y * width + x] = sum
            }
        }
    }

    private fun horizontalOpening(input: FloatArray, output: FloatArray, kernelWidth: Int) {
        val radius = kernelWidth / 2
        horizontalExtrema(input, scratchB, radius, maximum = false)
        horizontalExtrema(scratchB, output, radius, maximum = true)
    }

    private fun horizontalExtrema(
        input: FloatArray,
        output: FloatArray,
        radius: Int,
        maximum: Boolean
    ) {
        for (y in 0 until height) {
            val row = y * width
            var head = 0
            var tail = 0
            var right = -1
            for (x in 0 until width) {
                val wantedRight = (x + radius).coerceAtMost(width - 1)
                while (right < wantedRight) {
                    right++
                    val value = input[row + right]
                    while (tail > head) {
                        val last = rowDeque[tail - 1]
                        val lastValue = input[row + last]
                        if (if (maximum) lastValue > value else lastValue < value) break
                        tail--
                    }
                    rowDeque[tail++] = right
                }
                val left = (x - radius).coerceAtLeast(0)
                while (tail > head && rowDeque[head] < left) head++
                output[row + x] = input[row + rowDeque[head]]
            }
        }
    }

    private fun maxFilter2d(input: FloatArray, output: FloatArray, radius: Int) {
        horizontalExtrema(input, scratchC, radius, maximum = true)
        for (x in 0 until width) {
            var head = 0
            var tail = 0
            var bottom = -1
            for (y in 0 until height) {
                val wantedBottom = (y + radius).coerceAtMost(height - 1)
                while (bottom < wantedBottom) {
                    bottom++
                    val value = scratchC[bottom * width + x]
                    while (tail > head) {
                        val last = rowDeque[tail - 1]
                        if (scratchC[last * width + x] > value) break
                        tail--
                    }
                    rowDeque[tail++] = bottom
                }
                val top = (y - radius).coerceAtLeast(0)
                while (tail > head && rowDeque[head] < top) head++
                output[y * width + x] = scratchC[rowDeque[head] * width + x]
            }
        }
    }

    private fun binaryErode3(input: ByteArray, output: ByteArray) {
        output.fill(0)
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                var all = true
                for (yy in y - 1..y + 1) {
                    for (xx in x - 1..x + 1) {
                        if (input[yy * width + xx].toInt() == 0) all = false
                    }
                }
                if (all) output[y * width + x] = 1
            }
        }
    }

    private fun binaryDilate3(input: ByteArray, output: ByteArray) {
        output.fill(0)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var any = false
                for (yy in (y - 1).coerceAtLeast(0)..(y + 1).coerceAtMost(height - 1)) {
                    for (xx in (x - 1).coerceAtLeast(0)..(x + 1).coerceAtMost(width - 1)) {
                        if (input[yy * width + xx].toInt() != 0) any = true
                    }
                }
                if (any) output[y * width + x] = 1
            }
        }
    }

    private fun percentile(
        values: FloatArray,
        quantile: Float,
        x0: Int = 0,
        x1: Int = width,
        positiveOnly: Boolean = false
    ): Float {
        var maximum = 0f
        var count = 0
        for (y in 0 until height) {
            val row = y * width
            for (x in x0 until x1) {
                val value = values[row + x]
                if (positiveOnly && value <= 0f) continue
                maximum = max(maximum, value)
                count++
            }
        }
        if (count == 0 || maximum <= 0f) return 0f
        histogram.fill(0)
        val scale = (histogram.size - 1) / maximum
        for (y in 0 until height) {
            val row = y * width
            for (x in x0 until x1) {
                val value = values[row + x]
                if (positiveOnly && value <= 0f) continue
                histogram[(value * scale).roundToInt().coerceIn(0, histogram.lastIndex)]++
            }
        }
        val target = ceil(count * quantile.coerceIn(0f, 1f)).toInt().coerceAtLeast(1)
        var cumulative = 0
        for (i in histogram.indices) {
            cumulative += histogram[i]
            if (cumulative >= target) return i / scale
        }
        return maximum
    }
}

private fun gaussianKernel(sigma: Float): FloatArray {
    val radius = ceil(sigma * 3f).toInt().coerceAtLeast(1)
    val result = FloatArray(radius * 2 + 1)
    var sum = 0f
    for (i in -radius..radius) {
        val value = exp(-(i * i) / (2f * sigma * sigma))
        result[i + radius] = value
        sum += value
    }
    for (i in result.indices) result[i] /= sum
    return result
}

private val SIGMA_PREPROCESS = floatArrayOf(0.25f, 0.5f, 0.25f)
private val SIGMA_LOCAL_BACKGROUND = gaussianKernel(3f)
private val SIGMA_COMPACT = gaussianKernel(1.2f)
private val SIGMA_RESIDUAL = gaussianKernel(1f)
private val SIGMA_MOTION_BACKGROUND = gaussianKernel(5f)
