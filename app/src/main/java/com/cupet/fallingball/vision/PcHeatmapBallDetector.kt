package com.cupet.fallingball.vision

import java.nio.ByteBuffer
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Range
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Direct Android port of PC [CausalHeatmapBallDetector] heatmap-C.
 *
 * Production uses the same OpenCV 4.12 operators, ROI dimensions, frame-count startup rules,
 * real timestamp deltas, accumulated phase-correlation stabilization and geometry reprojection.
 */
class PcHeatmapBallDetector(
    private val width: Int = TubeRoiSampler.OUTPUT_WIDTH,
    private val height: Int = TubeRoiSampler.OUTPUT_HEIGHT,
    initialProfile: ClassicalHeatmapProfile = ClassicalHeatmapProfile.forBall(null, null)
) : AutoCloseable {
    companion object {
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
        private const val MAX_FRAME_SHIFT_PX = 6.0
        private const val MAX_ACCUMULATED_SHIFT_PX = 32.0

        /** Maps the selected current-frame peak back from stabilized to raw ROI coordinates. */
        internal fun publishCurrentFrameDetection(
            stabilizedX: Float,
            stabilizedY: Float,
            score: Float,
            stabilizationX: Float,
            stabilizationY: Float
        ) = ClassicalHeatmapDetection(
            roiX = stabilizedX - stabilizationX,
            roiY = stabilizedY - stabilizationY,
            score = score
        )
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

    private val rawBytes = ByteArray(pixelCount)
    private val stableBytes = ByteArray(pixelCount)
    private val percentileScratch = FloatArray(pixelCount)
    private val scoreValues = FloatArray(pixelCount)

    private val raw8: Mat
    private val rawGray: Mat
    private val stableGray: Mat
    private val stable8: Mat
    private val background: Mat
    private val previousGray: Mat
    private val previousRaw: Mat
    private val local: Mat
    private val residual: Mat
    private val motion: Mat
    private val score: Mat
    private val tinyScore: Mat
    private val workA: Mat
    private val workB: Mat
    private val workC: Mat
    private val binary: Mat
    private val binaryMask: Mat
    private val peakMap: Mat
    private val labels: Mat
    private val stats: Mat
    private val centroids: Mat
    private val horizontal20: Mat
    private val horizontal25: Mat
    private val horizontal07: Mat
    private val kernel3: Mat
    private val kernel5: Mat
    private val peakKernel: Mat
    private val previousSmall: Mat
    private val currentSmall: Mat
    private val hanning: Mat
    private val affine: Mat

    private val cropLeft = (width * 0.12).toInt()
    private val cropRight = (width * 0.88).toInt()
    private val cropTop = (height * 0.05).toInt()
    private val cropBottom = (height * 0.95).toInt()
    private val stabilizationScale = min(
        1.0,
        320.0 / max(cropRight - cropLeft, cropBottom - cropTop)
    )
    private val smallWidth = max(16, pyRound((cropRight - cropLeft) * stabilizationScale))
    private val smallHeight = max(16, pyRound((cropBottom - cropTop) * stabilizationScale))

    private var backgroundFrames = 0
    private var hasPreviousGray = false
    private var hasPreviousRaw = false
    private var lastInputTimestampNs = 0L
    private var stabilizationX = 0.0
    private var stabilizationY = 0.0
    private var lastPhaseResponse = 0.0
    private var currentGeometry: TubeRoiGeometry? = null

    private var pendingCandidate: Candidate? = null
    private var pendingCount = 0
    private var pendingStartY = 0f
    private val pendingY = ArrayDeque<Float>()
    private var tinyPendingCandidate: Candidate? = null
    private var tinyPendingCount = 0
    private var tinyPendingStartY = 0f
    private val tinyPendingY = ArrayDeque<Float>()
    private var lastCandidate: Candidate? = null
    private var lastCandidateTimestampNs = 0L
    private var velocityPxPerSecond = 0f
    private val trackY = ArrayDeque<Float>()
    private var stationaryFrames = 0
    private var missedFrames = 0
    private var trackStartedFromTiny = false
    private var observationRejection = ClassicalHeatmapRejection.WARMUP
    private var observationRecoveryPeakScore: Float? = null

    var lastStabilization = RoiStabilizationTransform(0f, 0f, 1f)
        private set

    init {
        require(width > 16 && height > 16)
        OpenCvRuntime.ensureLoaded()
        raw8 = Mat(height, width, CvType.CV_8UC1)
        rawGray = Mat(height, width, CvType.CV_32FC1)
        stableGray = Mat(height, width, CvType.CV_32FC1)
        stable8 = Mat(height, width, CvType.CV_8UC1)
        background = Mat(height, width, CvType.CV_32FC1)
        previousGray = Mat(height, width, CvType.CV_32FC1)
        previousRaw = Mat(height, width, CvType.CV_32FC1)
        local = Mat(height, width, CvType.CV_32FC1)
        residual = Mat(height, width, CvType.CV_32FC1)
        motion = Mat(height, width, CvType.CV_32FC1)
        score = Mat(height, width, CvType.CV_32FC1)
        tinyScore = Mat(height, width, CvType.CV_32FC1)
        workA = Mat(height, width, CvType.CV_32FC1)
        workB = Mat(height, width, CvType.CV_32FC1)
        workC = Mat(height, width, CvType.CV_32FC1)
        binary = Mat(height, width, CvType.CV_8UC1)
        binaryMask = Mat(height, width, CvType.CV_8UC1)
        peakMap = Mat(height, width, CvType.CV_32FC1)
        labels = Mat()
        stats = Mat()
        centroids = Mat()
        horizontal20 = Mat.ones(1, oddKernel(width * 0.20, 9), CvType.CV_8U)
        horizontal25 = Mat.ones(1, oddKernel(width * 0.25, 9), CvType.CV_8U)
        horizontal07 = Mat.ones(1, oddKernel(width * 0.07, 7), CvType.CV_8U)
        kernel3 = Mat.ones(3, 3, CvType.CV_8U)
        kernel5 = Mat.ones(5, 5, CvType.CV_8U)
        peakKernel = Mat.ones(7, 11, CvType.CV_32F)
        previousSmall = Mat(smallHeight, smallWidth, CvType.CV_32FC1)
        currentSmall = Mat(smallHeight, smallWidth, CvType.CV_32FC1)
        hanning = Mat(smallHeight, smallWidth, CvType.CV_32FC1)
        hanning.put(0, 0, buildHanning(smallWidth, smallHeight))
        affine = Mat(2, 3, CvType.CV_64FC1)
    }

    @Synchronized
    fun setProfile(next: ClassicalHeatmapProfile) {
        if (next != profile) pendingProfile = next
    }

    @Synchronized
    fun currentProfile(): ClassicalHeatmapProfile = pendingProfile ?: profile

    @Synchronized
    fun reset() = resetState(clearGeometry = true)

    @Synchronized
    fun observe(
        input: ByteBuffer,
        timestampNs: Long,
        geometry: TubeRoiGeometry,
        stableOutput: ByteBuffer
    ): ClassicalHeatmapResult {
        pendingProfile?.let {
            profile = it
            pendingProfile = null
            resetState(clearGeometry = true)
        }
        require(input.remaining() >= pixelCount) { "Expected $pixelCount grayscale bytes" }
        require(stableOutput.capacity() >= pixelCount) { "Stable output requires $pixelCount bytes" }

        val resetByGap = lastInputTimestampNs > 0L && timestampNs <= lastInputTimestampNs
        if (resetByGap) resetState(clearGeometry = true)
        currentGeometry?.let { previousGeometry ->
            if (previousGeometry.revision != geometry.revision ||
                !previousGeometry.sourceToRoi.contentEquals(geometry.sourceToRoi)
            ) {
                reprojectGeometry(previousGeometry, geometry)
            }
        }
        currentGeometry = geometry

        input.duplicate().get(rawBytes, 0, pixelCount)
        raw8.put(0, 0, rawBytes)
        raw8.convertTo(rawGray, CvType.CV_32FC1)
        Imgproc.GaussianBlur(rawGray, rawGray, Size(3.0, 3.0), 0.0)
        lastStabilization = stabilize()
        stableGray.convertTo(stable8, CvType.CV_8UC1)
        stable8.get(0, 0, stableBytes)
        stableOutput.clear()
        stableOutput.put(stableBytes)
        stableOutput.flip()

        if (backgroundFrames == 0) {
            stableGray.copyTo(background)
            backgroundFrames = 1
        } else {
            updateBackground()
            backgroundFrames = min(backgroundFrames + 1, MAX_BACKGROUND_FRAMES)
        }

        compactResponse()
        backgroundResidual()
        causalMotion()
        Core.addWeighted(local, 0.45, residual, 0.35, 0.0, score)
        Core.addWeighted(score, 1.0, motion, 0.20, 0.0, score)
        val rawPeakScore = normalizeAndMask(score)

        val ready = backgroundFrames >= profile.warmupFrames && hasPreviousGray
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
            val startupSearch = lastCandidate == null
            if (startupSearch && profile.startupTinyPeaks) {
                phase = ClassicalHeatmapPhase.STARTUP_NORMAL
                val normalCandidates = extractCandidates(score, preserveTiny = false)
                normalCandidateCount = normalCandidates.size
                strongestCandidateScore = normalCandidates.maxOfOrNull { it.confidence }
                selected = confirmStartup(
                    selectCandidate(normalCandidates, timestampNs),
                    tiny = false
                )
                if (selected != null) {
                    clearPending(tiny = true)
                } else {
                    phase = ClassicalHeatmapPhase.STARTUP_TINY
                    tinyPeakScore = tinyDynamicResponse()
                    val tinyCandidates = extractCandidates(tinyScore, preserveTiny = true)
                        .filter { inStartupRegion(it, tiny = true) }
                    tinyCandidateCount = tinyCandidates.size
                    tinyCandidates.maxOfOrNull { it.confidence }?.let {
                        strongestCandidateScore = max(strongestCandidateScore ?: 0f, it)
                    }
                    selected = confirmStartup(
                        selectPendingStartCandidate(tinyCandidates, tinyPendingCandidate),
                        tiny = true
                    )
                    if (selected != null) {
                        startedFromTiny = true
                        clearPending(tiny = false)
                    }
                }
            } else {
                phase = if (startupSearch) {
                    ClassicalHeatmapPhase.STARTUP_NORMAL
                } else {
                    ClassicalHeatmapPhase.TRACKING
                }
                val useTiny = profile.startupTinyPeaks && tinyDynamicTrackEnabled()
                val activeScore = if (useTiny) {
                    tinyPeakScore = tinyDynamicResponse()
                    tinyScore
                } else {
                    score
                }
                val candidates = extractCandidates(activeScore, preserveTiny = useTiny)
                if (useTiny) tinyCandidateCount = candidates.size else normalCandidateCount = candidates.size
                strongestCandidateScore = candidates.maxOfOrNull { it.confidence }
                selected = selectCandidate(candidates, timestampNs)
                if (startupSearch) selected = confirmStartup(selected, tiny = false)
            }

            if (selected == null && lastCandidate != null) {
                val activeScore = if (profile.startupTinyPeaks && tinyDynamicTrackEnabled()) {
                    if (tinyPeakScore == null) tinyPeakScore = tinyDynamicResponse()
                    tinyScore
                } else {
                    score
                }
                selected = recoverTrackCandidate(activeScore, timestampNs)
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
            val currentFrameDetection = publishCurrentFrameDetection(
                stabilizedX = selected.x,
                stabilizedY = selected.y,
                score = selected.confidence,
                stabilizationX = stabilizationX.toFloat(),
                stabilizationY = stabilizationY.toFloat()
            )
            if (detectionInRawBounds(currentFrameDetection)) {
                published = currentFrameDetection
                observationRejection = ClassicalHeatmapRejection.NONE
            } else {
                clearTrack()
            }
        } else if (lastCandidate != null) {
            missedFrames++
            if (missedFrames > MAX_MISSED_FRAMES) clearTrack()
        }

        stableGray.copyTo(previousGray)
        rawGray.copyTo(previousRaw)
        hasPreviousGray = true
        hasPreviousRaw = true
        lastInputTimestampNs = timestampNs
        return ClassicalHeatmapResult(
            detection = published,
            backgroundReady = ready,
            framesSeen = backgroundFrames,
            resetByGap = resetByGap,
            diagnostics = ClassicalHeatmapDiagnostics(
                rawPeakScore = rawPeakScore,
                strongestCandidateScore = strongestCandidateScore ?: published?.score,
                tinyPeakScore = tinyPeakScore,
                normalCandidateCount = normalCandidateCount,
                tinyCandidateCount = tinyCandidateCount,
                normalStartupSamples = pendingCount,
                tinyStartupSamples = tinyPendingCount,
                phase = phase,
                rejection = observationRejection,
                trackingFromTiny = trackStartedFromTiny,
                recoveryPeakScore = observationRecoveryPeakScore
            )
        )
    }

    private fun stabilize(): RoiStabilizationTransform {
        var response = 1.0
        if (hasPreviousRaw) {
            val previousCrop = previousRaw.submat(Range(cropTop, cropBottom), Range(cropLeft, cropRight))
            val currentCrop = rawGray.submat(Range(cropTop, cropBottom), Range(cropLeft, cropRight))
            try {
                Imgproc.resize(
                    previousCrop,
                    previousSmall,
                    Size(smallWidth.toDouble(), smallHeight.toDouble()),
                    0.0,
                    0.0,
                    Imgproc.INTER_AREA
                )
                Imgproc.resize(
                    currentCrop,
                    currentSmall,
                    Size(smallWidth.toDouble(), smallHeight.toDouble()),
                    0.0,
                    0.0,
                    Imgproc.INTER_AREA
                )
            } finally {
                currentCrop.release()
                previousCrop.release()
            }
            Core.normalize(previousSmall, previousSmall, 0.0, 1.0, Core.NORM_MINMAX)
            Core.normalize(currentSmall, currentSmall, 0.0, 1.0, Core.NORM_MINMAX)
            val responseOut = doubleArrayOf(0.0)
            val shift = Imgproc.phaseCorrelate(previousSmall, currentSmall, hanning, responseOut)
            var shiftX = shift.x / stabilizationScale
            var shiftY = shift.y / stabilizationScale
            response = responseOut[0]
            lastPhaseResponse = response
            if (!shiftX.isFinite() || !shiftY.isFinite() ||
                abs(shiftX) > MAX_FRAME_SHIFT_PX || abs(shiftY) > MAX_FRAME_SHIFT_PX ||
                response < 0.01
            ) {
                shiftX = 0.0
                shiftY = 0.0
            }
            stabilizationX = (stabilizationX - shiftX)
                .coerceIn(-MAX_ACCUMULATED_SHIFT_PX, MAX_ACCUMULATED_SHIFT_PX)
            stabilizationY = (stabilizationY - shiftY)
                .coerceIn(-MAX_ACCUMULATED_SHIFT_PX, MAX_ACCUMULATED_SHIFT_PX)
        }

        affine.put(
            0,
            0,
            *doubleArrayOf(1.0, 0.0, stabilizationX, 0.0, 1.0, stabilizationY)
        )
        Imgproc.warpAffine(
            rawGray,
            stableGray,
            affine,
            Size(width.toDouble(), height.toDouble()),
            Imgproc.INTER_LINEAR,
            Core.BORDER_REPLICATE
        )
        return RoiStabilizationTransform(
            stabilizationX.toFloat(),
            stabilizationY.toFloat(),
            response.toFloat()
        )
    }

    private fun updateBackground() {
        val dark = profile.polarity == ClassicalBallPolarity.DARK
        Core.addWeighted(background, 0.996, stableGray, 0.004, 0.0, workA)
        Core.addWeighted(background, 0.92, stableGray, 0.08, 0.0, workB)
        Core.compare(
            stableGray,
            background,
            binary,
            if (dark) Core.CMP_GE else Core.CMP_LE
        )
        workA.copyTo(background)
        workB.copyTo(background, binary)
        if (dark) {
            Core.subtract(stableGray, Scalar(2.0), workC)
            Core.max(background, workC, background)
        } else {
            Core.add(stableGray, Scalar(2.0), workC)
            Core.min(background, workC, background)
        }
    }

    private fun compactResponse() {
        Imgproc.GaussianBlur(stableGray, workA, Size(0.0, 0.0), 3.0)
        if (profile.polarity == ClassicalBallPolarity.DARK) {
            Core.subtract(workA, stableGray, local)
        } else {
            Core.subtract(stableGray, workA, local)
        }
        thresholdToZero(local)
        Imgproc.morphologyEx(local, workA, Imgproc.MORPH_OPEN, horizontal20)
        Imgproc.dilate(workA, workB, kernel5)
        Core.addWeighted(workA, 1.0, workB, 0.20, 0.0, workC)
        Core.subtract(local, workC, workA)
        thresholdToZero(workA)
        Imgproc.GaussianBlur(workA, local, Size(0.0, 0.0), 1.2)
    }

    private fun backgroundResidual() {
        if (profile.polarity == ClassicalBallPolarity.DARK) {
            Core.subtract(background, stableGray, residual)
        } else {
            Core.subtract(stableGray, background, residual)
        }
        thresholdToZero(residual)
        val baseline = percentile(residual, 0.50, positiveOnly = false) ?: 0f
        Core.subtract(residual, Scalar(baseline.toDouble()), residual)
        thresholdToZero(residual)
        Imgproc.morphologyEx(residual, workA, Imgproc.MORPH_OPEN, horizontal25)
        Core.addWeighted(residual, 1.0, workA, -0.85, 0.0, workB)
        thresholdToZero(workB)
        Imgproc.GaussianBlur(workB, residual, Size(0.0, 0.0), 1.0)
    }

    private fun causalMotion() {
        if (!hasPreviousGray) {
            motion.setTo(Scalar(0.0))
            return
        }
        if (profile.polarity == ClassicalBallPolarity.DARK) {
            Core.subtract(previousGray, stableGray, motion)
        } else {
            Core.subtract(stableGray, previousGray, motion)
        }
        thresholdToZero(motion)
        val baseline = percentile(motion, 0.50, positiveOnly = false) ?: 0f
        Core.subtract(motion, Scalar(baseline.toDouble()), motion)
        thresholdToZero(motion)
        Imgproc.GaussianBlur(motion, workA, Size(0.0, 0.0), 5.0)
        Core.addWeighted(motion, 1.0, workA, -0.85, 0.0, workB)
        thresholdToZero(workB)
        Imgproc.GaussianBlur(workB, motion, Size(0.0, 0.0), 1.0)
    }

    private fun tinyDynamicResponse(): Float {
        Core.addWeighted(residual, 0.65, motion, 0.35, 0.0, tinyScore)
        Imgproc.morphologyEx(tinyScore, workA, Imgproc.MORPH_OPEN, horizontal07)
        Core.addWeighted(tinyScore, 1.0, workA, -0.95, 0.0, tinyScore)
        thresholdToZero(tinyScore)
        return normalizeAndMask(tinyScore)
    }

    private fun normalizeAndMask(values: Mat): Float {
        val validX0 = max(0, (width * 0.05).toInt())
        val validX1 = min(width, (width * 0.95).toInt())
        val high = (percentile(values, 0.995, validX0, validX1, positiveOnly = false) ?: 1f)
            .coerceAtLeast(1e-3f)
        Core.multiply(values, Scalar(1.0 / high), values)
        Core.min(values, Scalar(1.0), values)

        val x0 = max(0, pyRound(width * VALID_X_MIN_FRACTION + stabilizationX))
        val x1 = min(width, pyRound(width * VALID_X_MAX_FRACTION + stabilizationX))
        val y0 = max(0, pyRound(height * 0.03 + stabilizationY))
        val y1 = min(height, pyRound(height * VALID_Y_MAX_FRACTION + stabilizationY))
        workA.setTo(Scalar(0.0))
        if (x1 > x0 && y1 > y0) {
            val source = values.submat(y0, y1, x0, x1)
            val target = workA.submat(y0, y1, x0, x1)
            try {
                source.copyTo(target)
            } finally {
                target.release()
                source.release()
            }
        }
        workA.copyTo(values)
        return Core.minMaxLoc(values).maxVal.toFloat()
    }

    private fun extractCandidates(values: Mat, preserveTiny: Boolean): List<Candidate> {
        val positiveP = percentile(values, 0.988, positiveOnly = true) ?: 1f
        val threshold = max(profile.minimumScore, positiveP)
        Core.compare(values, Scalar(threshold.toDouble()), binary, Core.CMP_GE)
        Imgproc.dilate(values, peakMap, peakKernel)
        Core.add(values, Scalar(1e-6), workA)
        Core.compare(workA, peakMap, binaryMask, Core.CMP_LT)
        binary.setTo(Scalar(0.0), binaryMask)
        if (!preserveTiny) {
            Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, kernel3)
        }

        val count = Imgproc.connectedComponentsWithStats(binary, labels, stats, centroids, 8)
        val minimumArea = if (preserveTiny) 1 else 2
        val candidates = ArrayList<Candidate>(max(0, count - 1))
        for (label in 1 until count) {
            val x = stat(label, Imgproc.CC_STAT_LEFT)
            val y = stat(label, Imgproc.CC_STAT_TOP)
            val componentWidth = stat(label, Imgproc.CC_STAT_WIDTH)
            val componentHeight = stat(label, Imgproc.CC_STAT_HEIGHT)
            val area = stat(label, Imgproc.CC_STAT_AREA)
            if (area !in minimumArea..180 ||
                componentWidth > width * 0.24f || componentHeight > height * 0.08f ||
                componentWidth > componentHeight * 5 || componentHeight > componentWidth * 5
            ) {
                continue
            }
            val patch = values.submat(Rect(x, y, componentWidth, componentHeight))
            val peak: Double
            val density: Double
            try {
                peak = Core.minMaxLoc(patch).maxVal
                density = Core.mean(patch).`val`[0]
            } finally {
                patch.release()
            }
            val centroidX = centroids.get(label, 0)[0].toFloat()
            val centroidY = centroids.get(label, 1)[0].toFloat()
            val confidence = (0.75 * peak + 0.25 * min(1.0, density * 2.0))
                .coerceIn(0.0, 1.0).toFloat()
            candidates += Candidate(centroidX, centroidY, confidence, area)
        }
        return candidates
    }

    private fun confirmStartup(candidate: Candidate?, tiny: Boolean): Candidate? {
        val valid = candidate?.takeIf { inStartupRegion(it, tiny) }
        if (valid == null) {
            observationRejection = if (candidate == null) {
                ClassicalHeatmapRejection.NO_CANDIDATE
            } else {
                ClassicalHeatmapRejection.OUTSIDE_STARTUP_REGION
            }
            clearPending(tiny)
            return null
        }
        val previous = if (tiny) tinyPendingCandidate else pendingCandidate
        if (previous == null) {
            setFirstPending(valid, tiny)
            observationRejection = ClassicalHeatmapRejection.STARTUP_COLLECTING
            return null
        }
        val tolerance = max(12f, height * 0.018f)
        if (valid.y < previous.y - 3f || abs(valid.y - previous.y) > tolerance) {
            clearPending(tiny)
            setFirstPending(valid, tiny)
            observationRejection = ClassicalHeatmapRejection.STARTUP_DISCONTINUITY
            return null
        }
        val history = if (tiny) tinyPendingY else pendingY
        history.addLast(valid.y)
        if (tiny) {
            tinyPendingCandidate = valid
            tinyPendingCount++
        } else {
            pendingCandidate = valid
            pendingCount++
        }
        val count = if (tiny) tinyPendingCount else pendingCount
        val startY = if (tiny) tinyPendingStartY else pendingStartY
        val confirmFrames = if (tiny) profile.startupTinyConfirmFrames else profile.startupConfirmFrames
        val minimumDrop = if (tiny) profile.startupTinyMinDropPx else profile.startupMinDropPx
        val minimumStep = if (tiny) {
            profile.startupTinyMinMedianStepPx
        } else {
            profile.startupMinMedianStepPx
        }
        val drop = valid.y - startY
        val medianStep = medianSteps(history)
        if (count >= max(2, confirmFrames) && drop >= minimumDrop && medianStep >= minimumStep) {
            clearPending(tiny)
            observationRejection = ClassicalHeatmapRejection.NONE
            return valid
        }
        observationRejection = when {
            count < max(2, confirmFrames) -> ClassicalHeatmapRejection.STARTUP_DURATION
            drop < minimumDrop -> ClassicalHeatmapRejection.STARTUP_DROP
            else -> ClassicalHeatmapRejection.STARTUP_MEDIAN_STEP
        }
        return null
    }

    private fun setFirstPending(candidate: Candidate, tiny: Boolean) {
        if (tiny) {
            tinyPendingCandidate = candidate
            tinyPendingCount = 1
            tinyPendingStartY = candidate.y
            tinyPendingY.addLast(candidate.y)
        } else {
            pendingCandidate = candidate
            pendingCount = 1
            pendingStartY = candidate.y
            pendingY.addLast(candidate.y)
        }
    }

    private fun selectPendingStartCandidate(
        candidates: List<Candidate>,
        pending: Candidate?
    ): Candidate? {
        if (candidates.isEmpty()) return null
        if (pending != null) {
            val tolerance = max(12f, height * 0.018f)
            candidates.filter {
                abs(it.x - pending.x) <= profile.searchSidePx &&
                    it.y >= pending.y - ALLOW_UPWARD_PX_PER_FRAME &&
                    it.y <= pending.y + tolerance
            }.maxWithOrNull(compareBy<Candidate> { it.confidence }.thenBy { it.area })
                ?.let { return it }
        }
        return candidates.maxWithOrNull(compareBy<Candidate> { it.confidence }.thenBy { it.area })
    }

    private fun selectCandidate(candidates: List<Candidate>, timestampNs: Long): Candidate? {
        if (candidates.isEmpty()) return null
        val last = lastCandidate
        if (last == null || lastCandidateTimestampNs <= 0L) {
            if (profile.startupTinyPeaks && profile.startupNormalContinuityWithTinyPeaks &&
                pendingCandidate != null
            ) {
                return selectPendingStartCandidate(candidates, pendingCandidate)
            }
            return candidates.maxWithOrNull(compareBy<Candidate> { it.confidence }.thenBy { it.area })
        }
        val dt = elapsedSeconds(lastCandidateTimestampNs, timestampNs)
        val predicted = last.y + velocityPxPerSecond * dt
        val maxStep = MAX_SPEED_PX_PER_SECOND * dt + profile.searchAheadPx
        val maxSideStep = MAX_HORIZONTAL_SPEED_PX_PER_SECOND * dt + profile.searchSidePx
        val minY = last.y - ALLOW_UPWARD_PX_PER_FRAME * max(1f, dt * 60f)
        val maxY = min(height - 1f, predicted + maxStep)
        return candidates.filter {
            it.y in minY..maxY && abs(it.x - last.x) <= maxSideStep
        }.maxByOrNull {
            it.confidence - 0.35f * abs(it.y - predicted) / max(height, 1) -
                0.20f * abs(it.x - last.x) / max(1f, profile.searchSidePx)
        }
    }

    private fun recoverTrackCandidate(values: Mat, timestampNs: Long): Candidate? {
        val last = lastCandidate ?: return null
        val dt = elapsedSeconds(lastCandidateTimestampNs, timestampNs)
        val predictedY = last.y + velocityPxPerSecond * dt
        val side = TRACK_RECOVERY_SIDE_PX + 0.5f * MAX_HORIZONTAL_SPEED_PX_PER_SECOND * dt
        val x0 = floor(last.x - side).toInt().coerceAtLeast(0)
        val x1 = ceil(last.x + side + 1f).toInt().coerceAtMost(width)
        val upward = ALLOW_UPWARD_PX_PER_FRAME * max(1f, dt * 60f)
        val y0 = floor(last.y - upward).toInt().coerceAtLeast(0)
        val y1 = ceil(max(last.y, predictedY) + TRACK_RECOVERY_AHEAD_PX + 1f)
            .toInt().coerceAtMost(height)
        if (x1 <= x0 || y1 <= y0) return null
        val window = values.submat(y0, y1, x0, x1)
        val peakResult = try {
            Core.minMaxLoc(window)
        } finally {
            window.release()
        }
        val peak = peakResult.maxVal.toFloat()
        observationRecoveryPeakScore = peak
        if (peak < TRACK_RECOVERY_MIN_SCORE) return null
        val peakX = x0 + peakResult.maxLoc.x.toInt()
        val peakY = y0 + peakResult.maxLoc.y.toInt()
        values.get(0, 0, scoreValues)
        val support = max(TRACK_RECOVERY_MIN_SCORE, peak * 0.55f)
        var area = 0
        var total = 0f
        var weightedX = 0f
        var weightedY = 0f
        for (y in max(0, peakY - 3)..min(height - 1, peakY + 3)) {
            for (x in max(0, peakX - 3)..min(width - 1, peakX + 3)) {
                val value = scoreValues[y * width + x]
                if (value >= support) {
                    area++
                    total += value
                    weightedX += x * value
                    weightedY += y * value
                }
            }
        }
        if (area !in 1..180 || total <= 1e-6f) return null
        return Candidate(weightedX / total, weightedY / total, peak.coerceIn(0f, 1f), area)
    }

    private fun inStartupRegion(candidate: Candidate, tiny: Boolean): Boolean {
        val raw = candidateToRaw(candidate)
        val minX = if (tiny) profile.startupTinyXMinFraction else profile.startupXMinFraction
        val maxX = if (tiny) profile.startupTinyXMaxFraction else profile.startupXMaxFraction
        val minY = if (tiny) profile.startupTinyMinYFraction else profile.startupMinYFraction
        return raw.x in width * minX..width * maxX &&
            raw.y in height * minY..height * profile.startupMaxYFraction
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
        while (trackY.size > max(2, TRACK_PROGRESS_WINDOW)) trackY.removeFirst()
        val progress = trackY.last() - trackY.first()
        if ((stationaryFrames >= MAX_STATIONARY_FRAMES && progress < MIN_TRACK_PROGRESS_PX) ||
            (trackY.size >= max(2, TRACK_PROGRESS_WINDOW) && progress < -MIN_TRACK_PROGRESS_PX)
        ) {
            clearTrack()
            observationRejection = ClassicalHeatmapRejection.TRACKING_STATIONARY
            return false
        }
        return true
    }

    private fun updateTrack(candidate: Candidate, timestampNs: Long) {
        lastCandidate?.let { previous ->
            val dt = elapsedSeconds(lastCandidateTimestampNs, timestampNs)
            val measured = (candidate.y - previous.y) / dt
            velocityPxPerSecond = 0.75f * velocityPxPerSecond + 0.25f * measured
        }
        lastCandidate = candidate
        lastCandidateTimestampNs = timestampNs
    }

    private fun tinyDynamicTrackEnabled(): Boolean =
        profile.tinyDynamicTracking &&
            (!profile.tinyDynamicTrackingAfterTinyStartOnly || trackStartedFromTiny)

    private fun candidateToRaw(candidate: Candidate): Candidate = candidate.copy(
        x = candidate.x - stabilizationX.toFloat(),
        y = candidate.y - stabilizationY.toFloat()
    )

    private fun detectionInRawBounds(detection: ClassicalHeatmapDetection): Boolean =
        detection.roiX in width * VALID_X_MIN_FRACTION..width * VALID_X_MAX_FRACTION &&
            detection.roiY in 0f..height * VALID_Y_MAX_FRACTION

    private fun clearTrack() {
        lastCandidate = null
        lastCandidateTimestampNs = 0L
        velocityPxPerSecond = 0f
        trackY.clear()
        stationaryFrames = 0
        missedFrames = 0
        trackStartedFromTiny = false
        clearPending(tiny = false)
        clearPending(tiny = true)
    }

    private fun clearPending(tiny: Boolean) {
        if (tiny) {
            tinyPendingCandidate = null
            tinyPendingCount = 0
            tinyPendingStartY = 0f
            tinyPendingY.clear()
        } else {
            pendingCandidate = null
            pendingCount = 0
            pendingStartY = 0f
            pendingY.clear()
        }
    }

    private fun reprojectGeometry(old: TubeRoiGeometry, next: TubeRoiGeometry) {
        val oldToNewRaw = multiply3x3(next.sourceToRoi, old.roiToSource)
        val stableToNew = multiply3x3(
            oldToNewRaw,
            doubleArrayOf(
                1.0, 0.0, -stabilizationX,
                0.0, 1.0, -stabilizationY,
                0.0, 0.0, 1.0
            )
        )
        val stableMatrix = matrix3x3(stableToNew)
        val rawMatrix = matrix3x3(oldToNewRaw)
        try {
            if (backgroundFrames > 0) warpState(background, stableMatrix)
            if (hasPreviousGray) warpState(previousGray, stableMatrix)
            if (hasPreviousRaw) warpState(previousRaw, rawMatrix)

            val oldLast = lastCandidate
            val oldPending = pendingCandidate
            val oldTinyPending = tinyPendingCandidate
            lastCandidate = lastCandidate?.let { transformCandidate(it, stableToNew) }
            pendingCandidate = pendingCandidate?.let { transformCandidate(it, stableToNew) }
            tinyPendingCandidate = tinyPendingCandidate?.let { transformCandidate(it, stableToNew) }
            val anchorX = oldLast?.x ?: oldPending?.x ?: oldTinyPending?.x ?: width * 0.5f
            transformDequeY(pendingY, anchorX, stableToNew)
            transformDequeY(tinyPendingY, anchorX, stableToNew)
            transformDequeY(trackY, anchorX, stableToNew)
            if (pendingCount > 0) pendingStartY = transformPoint(anchorX, pendingStartY, stableToNew).second
            if (tinyPendingCount > 0) {
                tinyPendingStartY = transformPoint(anchorX, tinyPendingStartY, stableToNew).second
            }
            if (oldLast != null && lastCandidate != null) {
                val probeY = transformPoint(oldLast.x, oldLast.y + 1f, stableToNew).second
                val scaleY = (probeY - requireNotNull(lastCandidate).y).coerceIn(0.25f, 4f)
                velocityPxPerSecond *= scaleY
            }
        } finally {
            rawMatrix.release()
            stableMatrix.release()
        }
        stabilizationX = 0.0
        stabilizationY = 0.0
        lastPhaseResponse = 0.0
        lastStabilization = RoiStabilizationTransform(0f, 0f, 0f)
    }

    private fun warpState(state: Mat, matrix: Mat) {
        Imgproc.warpPerspective(
            state,
            workA,
            matrix,
            Size(width.toDouble(), height.toDouble()),
            Imgproc.INTER_LINEAR,
            Core.BORDER_REPLICATE
        )
        workA.copyTo(state)
    }

    private fun transformCandidate(candidate: Candidate, matrix: DoubleArray): Candidate {
        val (x, y) = transformPoint(candidate.x, candidate.y, matrix)
        return candidate.copy(x = x, y = y)
    }

    private fun transformDequeY(values: ArrayDeque<Float>, anchorX: Float, matrix: DoubleArray) {
        if (values.isEmpty()) return
        val transformed = values.map { transformPoint(anchorX, it, matrix).second }
        values.clear()
        transformed.forEach(values::addLast)
    }

    private fun transformPoint(x: Float, y: Float, matrix: DoubleArray): Pair<Float, Float> {
        val denominator = matrix[6] * x + matrix[7] * y + matrix[8]
        if (abs(denominator) < 1e-12) return x to y
        return (
            ((matrix[0] * x + matrix[1] * y + matrix[2]) / denominator).toFloat() to
                ((matrix[3] * x + matrix[4] * y + matrix[5]) / denominator).toFloat()
            )
    }

    private fun resetState(clearGeometry: Boolean) {
        backgroundFrames = 0
        hasPreviousGray = false
        hasPreviousRaw = false
        lastInputTimestampNs = 0L
        stabilizationX = 0.0
        stabilizationY = 0.0
        lastPhaseResponse = 0.0
        lastStabilization = RoiStabilizationTransform(0f, 0f, 1f)
        clearTrack()
        if (clearGeometry) currentGeometry = null
        background.setTo(Scalar(0.0))
        previousGray.setTo(Scalar(0.0))
        previousRaw.setTo(Scalar(0.0))
    }

    private fun thresholdToZero(mat: Mat) {
        Imgproc.threshold(mat, mat, 0.0, 0.0, Imgproc.THRESH_TOZERO)
    }

    private fun stat(label: Int, field: Int): Int = stats.get(label, field)[0].toInt()

    private fun percentile(
        mat: Mat,
        quantile: Double,
        x0: Int = 0,
        x1: Int = width,
        positiveOnly: Boolean
    ): Float? {
        val left = x0.coerceIn(0, width)
        val right = x1.coerceIn(left, width)
        if (right <= left) return null
        val view = mat.submat(0, height, left, right)
        val contiguous = view.clone()
        val total = contiguous.rows() * contiguous.cols()
        try {
            contiguous.get(0, 0, percentileScratch)
        } finally {
            contiguous.release()
            view.release()
        }
        var count = total
        if (positiveOnly) {
            count = 0
            for (i in 0 until total) {
                val value = percentileScratch[i]
                if (value > 0f) percentileScratch[count++] = value
            }
        }
        if (count == 0) return null
        val index = (count - 1) * quantile.coerceIn(0.0, 1.0)
        val lowerIndex = floor(index).toInt()
        val upperIndex = ceil(index).toInt()
        val lower = selectKth(percentileScratch, count, lowerIndex)
        if (lowerIndex == upperIndex) return lower
        val upper = selectKth(percentileScratch, count, upperIndex)
        return (lower + (upper - lower) * (index - lowerIndex)).toFloat()
    }

    private fun selectKth(values: FloatArray, size: Int, rank: Int): Float {
        var left = 0
        var right = size - 1
        val target = rank.coerceIn(0, right)
        while (left < right) {
            val pivot = values[(left + right) ushr 1]
            var i = left
            var j = right
            while (i <= j) {
                while (values[i] < pivot) i++
                while (values[j] > pivot) j--
                if (i <= j) {
                    val swap = values[i]
                    values[i] = values[j]
                    values[j] = swap
                    i++
                    j--
                }
            }
            when {
                target <= j -> right = j
                target >= i -> left = i
                else -> return values[target]
            }
        }
        return values[target]
    }

    private fun medianSteps(values: ArrayDeque<Float>): Float {
        if (values.size < 2) return 0f
        var previous: Float? = null
        var count = 0
        for (value in values) {
            previous?.let { percentileScratch[count++] = value - it }
            previous = value
        }
        val middle = count / 2
        val upper = selectKth(percentileScratch, count, middle)
        return if (count and 1 == 1) upper
        else (selectKth(percentileScratch, count, middle - 1) + upper) * 0.5f
    }

    private fun elapsedSeconds(previousTimestampNs: Long, timestampNs: Long): Float =
        if (previousTimestampNs <= 0L || timestampNs <= previousTimestampNs) {
            1f / 30f
        } else {
            max(1e-3f, (timestampNs - previousTimestampNs) / 1_000_000_000f)
        }

    override fun close() {
        listOf(
            raw8, rawGray, stableGray, stable8, background, previousGray, previousRaw,
            local, residual, motion, score, tinyScore, workA, workB, workC, binary,
            binaryMask, peakMap, labels, stats, centroids, horizontal20, horizontal25,
            horizontal07, kernel3, kernel5, peakKernel, previousSmall, currentSmall,
            hanning, affine
        ).forEach(Mat::release)
    }
}

private fun pyRound(value: Double): Int = round(value).toInt()

private fun oddKernel(value: Double, minimum: Int): Int =
    max(minimum, pyRound(value).let { if (it and 1 == 0) it + 1 else it })

private fun buildHanning(width: Int, height: Int): FloatArray {
    val x = DoubleArray(width) { index ->
        if (width <= 1) 1.0 else 0.5 - 0.5 * kotlin.math.cos(2.0 * Math.PI * index / (width - 1))
    }
    val y = DoubleArray(height) { index ->
        if (height <= 1) 1.0 else 0.5 - 0.5 * kotlin.math.cos(2.0 * Math.PI * index / (height - 1))
    }
    return FloatArray(width * height) { index ->
        (y[index / width] * x[index % width]).toFloat()
    }
}

private fun matrix3x3(values: DoubleArray): Mat =
    Mat(3, 3, CvType.CV_64FC1).also { it.put(0, 0, *values) }

private fun multiply3x3(left: DoubleArray, right: DoubleArray): DoubleArray {
    require(left.size == 9 && right.size == 9)
    return DoubleArray(9) { index ->
        val row = index / 3
        val column = index % 3
        left[row * 3] * right[column] +
            left[row * 3 + 1] * right[3 + column] +
            left[row * 3 + 2] * right[6 + column]
    }
}
