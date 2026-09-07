package com.cupet.fallingball.vision

import kotlin.math.abs

/** Which observation produced the selected ball position for this frame. */
enum class BallDetectionSource {
    NONE,
    RAW_PROFILE,
    BACKGROUND,
    FUSED,
    CONVGRU,
    LOCAL_TRACKER,
    /** Causal non-neural 2-D centre heatmap; appended to preserve stored enum names. */
    CLASSICAL_HEATMAP,
    /** Frozen TinyHeatmapNet + CandidateReranker preview/final sequence output. */
    TINY_HEATMAP
}

/** Immutable output of [RegisteredBallDetector.observe]. */
data class RegisteredBallResult(
    val ballAxialPx: Float?,
    val source: BallDetectionSource,
    val confidence: Float,
    val backgroundReady: Boolean
)

/**
 * Detects a moving ball against a pre-drop background expressed in ruler coordinates.
 *
 * The YOLO box only supplies the approximate tube axis used to build [signal]. Its unstable
 * top edge is deliberately absent from the stored background: every profile is resampled
 * against the stable graduation ordinals in [RulerSnapshot]. A whole-frame translation or
 * box-length change therefore moves the input pixels without moving the background grid.
 *
 * Not thread safe; owned by the camera analysis thread.
 */
class RegisteredBallDetector {

    companion object {
        /** Roughly 0.4 seconds at 30 fps; enough for a robust pre-drop median. */
        const val MIN_BACKGROUND_FRAMES = 12

        private const val MAX_BACKGROUND_FRAMES = 21
        private const val SAMPLES_PER_TICK = 12
        private const val MIN_BACKGROUND_COVERAGE = 0.70f
        private const val MIN_RESIDUAL_LUMA = 6f
        private const val RESIDUAL_NOISE_MULTIPLIER = 3f
        private const val TRACKING_MIN_RESIDUAL_LUMA = 3.5f
        private const val TRACKING_RESIDUAL_NOISE_MULTIPLIER = 1.75f
        private const val MIN_BALL_WIDTH_TICKS = 0.20f
        private const val MAX_BALL_WIDTH_TICKS = 6f
        private const val FUSION_DISTANCE_TICKS = 0.75f
        private const val STATIONARY_RAW_DISTANCE_TICKS = 0.75f
        private const val MIN_STATIONARY_RAW_HITS = 3
        private const val MAX_FIRST_RAW_STEP_TICKS = 4f
        private const val MAX_RAW_UPWARD_STEP_TICKS = 1f
        private const val MIN_RAW_GATE_TICKS = 1.5f
    }

    private var firstOrdinal = 0
    private var lastOrdinal = -1
    private var canonicalSize = 0
    private var backgroundFrames = arrayOfNulls<FloatArray>(MAX_BACKGROUND_FRAMES)
    private var nextBackgroundFrame = 0
    private var backgroundFrameCount = 0
    private var background = FloatArray(0)
    private var backgroundReady = false
    private var canonicalScratch = FloatArray(0)
    private var residualScratch = FloatArray(0)
    private var validBackgroundScratch = FloatArray(0)
    private var validCurrentScratch = FloatArray(0)
    private var statsScratch = FloatArray(0)
    private var inlierScratch = BooleanArray(0)
    private val backgroundMedianScratch = FloatArray(MAX_BACKGROUND_FRAMES)
    private var stationaryRawQ = Float.NaN
    private var stationaryRawHits = 0
    private var trackedQ = Float.NaN
    private var trackedVelocityQps = 0f
    private var trackedTimestampNs = 0L
    private var trackedObservations = 0

    fun reset() {
        firstOrdinal = 0
        lastOrdinal = -1
        canonicalSize = 0
        backgroundFrames = arrayOfNulls(MAX_BACKGROUND_FRAMES)
        nextBackgroundFrame = 0
        backgroundFrameCount = 0
        background = FloatArray(0)
        backgroundReady = false
        canonicalScratch = FloatArray(0)
        residualScratch = FloatArray(0)
        validBackgroundScratch = FloatArray(0)
        validCurrentScratch = FloatArray(0)
        statsScratch = FloatArray(0)
        inlierScratch = BooleanArray(0)
        stationaryRawQ = Float.NaN
        stationaryRawHits = 0
        resetTrack()
    }

    /** Starts a new fall without discarding the learned empty-tube background. */
    fun resetTrajectory() {
        resetTrack()
    }

    /**
     * Processes one native-resolution axial luminance profile.
     *
     * [learnBackground] must only be true before or while waiting for a drop. Once a
     * template exists, a moving residual suppresses learning for that frame. The raw
     * detector alone does not: a stationary end shadow may look like a wide dark peak on
     * every frame, and refusing to learn it would prevent the background ever becoming ready.
     */
    fun observe(
        signal: FloatArray,
        signalLength: Int,
        signalOffsetPx: Float,
        ruler: RulerSnapshot,
        rawBallAxialPx: Float?,
        learnBackground: Boolean,
        timestampNs: Long = 0L,
        preferredPositionTicks: Float? = null,
        preferredWindowTicks: Float = Float.POSITIVE_INFINITY,
        trackingSensitivity: Boolean = false
    ): RegisteredBallResult {
        val freshRegistration = hasFreshRegistration(ruler)
        val canonical = if (freshRegistration) {
            canonicalize(
                signal, signalLength, signalOffsetPx, ruler, initialise = learnBackground
            )
        } else {
            null
        }

        // Once a template exists, always detect before learning. This lets an armed session
        // keep refining its pre-drop background without absorbing the first ball frame when
        // the raw dark-trough detector happens to miss it.
        val preferredBackgroundCandidate = if (canonical != null && backgroundReady) {
            detectAgainstBackground(
                canonical,
                ruler,
                preferredPositionTicks,
                preferredWindowTicks,
                trackingSensitivity
            )
        } else {
            null
        }
        val backgroundCandidate = preferredBackgroundCandidate ?: if (
            canonical != null &&
            preferredPositionTicks != null &&
            preferredPositionTicks.isFinite()
        ) {
            // A primary-detector hint is optional. If its local window is empty, retain the global
            // residual so another lateral band can establish an independent moving track.
            detectAgainstBackground(
                canonical,
                ruler,
                preferredPositionTicks = null,
                preferredWindowTicks = Float.POSITIVE_INFINITY,
                trackingSensitivity = false
            )
        } else {
            null
        }

        if (canonical != null && learnBackground && backgroundCandidate == null) {
            addBackgroundFrame(canonical)
        }

        val rawQ = rawBallAxialPx?.let { safeAxialToCanonical(it, ruler) }
        if (learnBackground && freshRegistration && backgroundCandidate == null && rawQ != null) {
            observeStationaryRaw(rawQ)
        }

        var result = select(
            backgroundCandidate,
            rawBallAxialPx,
            rawQ,
            ruler,
            suppressRawOnly = learnBackground &&
                backgroundReady &&
                rawQ != null &&
                stationaryRawHits >= MIN_STATIONARY_RAW_HITS &&
                abs(rawQ - stationaryRawQ) <= STATIONARY_RAW_DISTANCE_TICKS
        )

        // After the first accepted observation, a raw-only fallback must remain associated
        // with that same falling object. This is intentionally a broad alpha-beta-style
        // gate: it rejects jumps to a distant end shadow without imposing a tight physics
        // model on liquids with very different viscosities.
        if (result.source == BallDetectionSource.RAW_PROFILE &&
            !learnBackground &&
            !rawTrackPlausible(rawQ, timestampNs)
        ) {
            result = RegisteredBallResult(null, BallDetectionSource.NONE, 0f, backgroundReady)
        }

        if (result.ballAxialPx == null) {
            if (learnBackground) resetTrack()
        } else {
            if (learnBackground) resetTrack()
            val selectedQ = safeAxialToCanonical(result.ballAxialPx, ruler)
            if (selectedQ != null) updateTrack(selectedQ, timestampNs)
        }
        return result
    }

    private data class Candidate(val axialPx: Float, val canonicalQ: Float, val confidence: Float)

    private fun select(
        backgroundCandidate: Candidate?,
        rawBallAxialPx: Float?,
        rawQ: Float?,
        ruler: RulerSnapshot,
        suppressRawOnly: Boolean
    ): RegisteredBallResult {
        if (backgroundCandidate == null && rawBallAxialPx == null) {
            return RegisteredBallResult(null, BallDetectionSource.NONE, 0f, backgroundReady)
        }
        if (backgroundCandidate == null) {
            if (suppressRawOnly) {
                return RegisteredBallResult(null, BallDetectionSource.NONE, 0f, backgroundReady)
            }
            return RegisteredBallResult(
                rawBallAxialPx, BallDetectionSource.RAW_PROFILE, 1f, backgroundReady
            )
        }
        if (rawBallAxialPx == null) {
            return RegisteredBallResult(
                backgroundCandidate.axialPx,
                BallDetectionSource.BACKGROUND,
                backgroundCandidate.confidence,
                backgroundReady
            )
        }

        if (rawQ != null && abs(rawQ - backgroundCandidate.canonicalQ) <= FUSION_DISTANCE_TICKS) {
            val q = backgroundCandidate.canonicalQ * 0.7f + rawQ * 0.3f
            val axial = canonicalToAxial(q, ruler) ?: backgroundCandidate.axialPx
            return RegisteredBallResult(
                axial,
                BallDetectionSource.FUSED,
                maxOf(1f, backgroundCandidate.confidence),
                backgroundReady
            )
        }

        // A strong moving residual is harder for a static numeral or end shadow to mimic
        // than a single dark trough, so prefer it when the two detectors disagree.
        return RegisteredBallResult(
            backgroundCandidate.axialPx,
            BallDetectionSource.BACKGROUND,
            backgroundCandidate.confidence,
            backgroundReady
        )
    }

    private fun canonicalize(
        signal: FloatArray,
        signalLength: Int,
        signalOffsetPx: Float,
        ruler: RulerSnapshot,
        initialise: Boolean
    ): FloatArray? {
        if (!hasFreshRegistration(ruler) || ruler.count < 2 || signalLength < 2) return null
        if (!strictlyIncreasing(ruler.axialPx) || !strictlyIncreasing(ruler.ordinals)) return null

        if (canonicalSize == 0) {
            if (!initialise) return null
            firstOrdinal = ruler.ordinals.first()
            lastOrdinal = ruler.ordinals.last()
            if (lastOrdinal <= firstOrdinal) return null
            canonicalSize = (lastOrdinal - firstOrdinal) * SAMPLES_PER_TICK + 1
            background = FloatArray(canonicalSize) { Float.NaN }
            canonicalScratch = FloatArray(canonicalSize)
            residualScratch = FloatArray(canonicalSize)
            validBackgroundScratch = FloatArray(canonicalSize)
            validCurrentScratch = FloatArray(canonicalSize)
            statsScratch = FloatArray(canonicalSize)
            inlierScratch = BooleanArray(canonicalSize)
        }

        val out = canonicalScratch
        out.fill(Float.NaN)
        var hi = 1
        for (i in 0 until canonicalSize) {
            val q = firstOrdinal + i.toFloat() / SAMPLES_PER_TICK
            if (q < ruler.ordinals.first() || q > ruler.ordinals.last()) continue
            while (hi < ruler.count && ruler.ordinals[hi] < q) hi++
            if (hi >= ruler.count) continue
            val lo = hi - 1
            val ordinalSpan = ruler.ordinals[hi] - ruler.ordinals[lo]
            if (ordinalSpan <= 0) continue
            val frac = (q - ruler.ordinals[lo]) / ordinalSpan
            val axial = ruler.axialPx[lo] +
                (ruler.axialPx[hi] - ruler.axialPx[lo]) * frac
            val source = axial - signalOffsetPx
            if (source < 0f || source > signalLength - 1f) continue
            val sourceLo = source.toInt().coerceIn(0, signalLength - 1)
            val sourceHi = (sourceLo + 1).coerceAtMost(signalLength - 1)
            val sourceFrac = source - sourceLo
            out[i] = signal[sourceLo] +
                (signal[sourceHi] - signal[sourceLo]) * sourceFrac
        }
        return out
    }

    private fun hasFreshRegistration(ruler: RulerSnapshot): Boolean {
        return ruler.state == RulerState.VALID &&
            ruler.invalidReason == null &&
            ruler.shiftAgreement >= TickRuler.MIN_SHIFT_AGREEMENT
    }

    private fun observeStationaryRaw(q: Float) {
        if (stationaryRawQ.isNaN()) {
            stationaryRawQ = q
            stationaryRawHits = 1
            return
        }
        if (abs(q - stationaryRawQ) <= STATIONARY_RAW_DISTANCE_TICKS) {
            stationaryRawHits++
            val weight = 1f / stationaryRawHits.coerceAtMost(8)
            stationaryRawQ += (q - stationaryRawQ) * weight
        } else if (stationaryRawHits < MIN_STATIONARY_RAW_HITS) {
            // Before a trough proves repeatable, let a different observation replace it.
            // Once stable, keep the learned shadow so the first real ball cannot overwrite it.
            stationaryRawQ = q
            stationaryRawHits = 1
        }
    }

    private fun rawTrackPlausible(q: Float?, timestampNs: Long): Boolean {
        if (q == null || trackedObservations == 0 || trackedQ.isNaN()) return true
        val step = q - trackedQ
        if (step < -MAX_RAW_UPWARD_STEP_TICKS) return false
        if (trackedObservations == 1) return step <= MAX_FIRST_RAW_STEP_TICKS

        val dt = elapsedSeconds(timestampNs)
        val predicted = trackedQ + trackedVelocityQps * dt
        val gate = maxOf(
            MIN_RAW_GATE_TICKS,
            abs(trackedVelocityQps) * dt * 1.5f + 0.75f
        )
        return abs(q - predicted) <= gate
    }

    private fun updateTrack(q: Float, timestampNs: Long) {
        if (trackedObservations > 0 && !trackedQ.isNaN()) {
            val dt = elapsedSeconds(timestampNs)
            val measuredVelocity = (q - trackedQ) / dt
            trackedVelocityQps = if (trackedObservations == 1) {
                measuredVelocity
            } else {
                trackedVelocityQps * 0.65f + measuredVelocity * 0.35f
            }
        }
        trackedQ = q
        trackedTimestampNs = timestampNs
        trackedObservations++
    }

    private fun elapsedSeconds(timestampNs: Long): Float {
        if (timestampNs <= 0L || trackedTimestampNs <= 0L || timestampNs <= trackedTimestampNs) {
            return 1f / 30f
        }
        return ((timestampNs - trackedTimestampNs) / 1_000_000_000f).coerceIn(1f / 120f, 0.25f)
    }

    private fun resetTrack() {
        trackedQ = Float.NaN
        trackedVelocityQps = 0f
        trackedTimestampNs = 0L
        trackedObservations = 0
    }

    private fun addBackgroundFrame(frame: FloatArray) {
        val stored = backgroundFrames[nextBackgroundFrame]
            ?: FloatArray(canonicalSize).also { backgroundFrames[nextBackgroundFrame] = it }
        frame.copyInto(stored, endIndex = canonicalSize)
        nextBackgroundFrame = (nextBackgroundFrame + 1) % MAX_BACKGROUND_FRAMES
        if (backgroundFrameCount < MAX_BACKGROUND_FRAMES) backgroundFrameCount++
        if (backgroundFrameCount < MIN_BACKGROUND_FRAMES) return

        var validBins = 0
        for (i in 0 until canonicalSize) {
            var n = 0
            for (f in 0 until backgroundFrameCount) {
                val value = backgroundFrames[f]?.get(i) ?: continue
                if (!value.isNaN()) backgroundMedianScratch[n++] = value
            }
            if (n >= MIN_BACKGROUND_FRAMES) {
                background[i] = median(backgroundMedianScratch, n)
                validBins++
            } else {
                background[i] = Float.NaN
            }
        }
        backgroundReady = validBins >= canonicalSize * MIN_BACKGROUND_COVERAGE
    }

    private fun detectAgainstBackground(
        current: FloatArray,
        ruler: RulerSnapshot,
        preferredPositionTicks: Float?,
        preferredWindowTicks: Float,
        trackingSensitivity: Boolean
    ): Candidate? {
        val residuals = residualScratch
        val validBackground = validBackgroundScratch
        val validCurrent = validCurrentScratch
        val scratch = statsScratch
        var valid = 0
        for (i in 0 until canonicalSize) {
            val b = background[i]
            val c = current[i]
            if (b.isNaN() || c.isNaN()) {
                residuals[i] = Float.NaN
            } else {
                validBackground[valid] = b
                validCurrent[valid] = c
                valid++
            }
        }
        if (valid < canonicalSize * MIN_BACKGROUND_COVERAGE) return null

        // Fit current ~= gain * background + offset. The second pass excludes local
        // foreground residuals, so auto-exposure gain changes do not turn a broad static
        // shadow into apparent motion.
        val exposure = robustAffine(validBackground, validCurrent, valid, scratch)
        valid = 0
        for (i in 0 until canonicalSize) {
            val b = background[i]
            val c = current[i]
            if (b.isNaN() || c.isNaN()) continue
            val residual = exposure.gain * b + exposure.offset - c
            residuals[i] = residual
            scratch[valid++] = residual
        }
        val residualBaseline = median(scratch, valid)
        for (i in 0 until valid) scratch[i] = abs(scratch[i] - residualBaseline)
        val noise = median(scratch, valid) * 1.4826f
        val threshold = if (trackingSensitivity) {
            maxOf(TRACKING_MIN_RESIDUAL_LUMA, noise * TRACKING_RESIDUAL_NOISE_MULTIPLIER)
        } else {
            maxOf(MIN_RESIDUAL_LUMA, noise * RESIDUAL_NOISE_MULTIPLIER)
        }

        var peakIndex = -1
        var peak = threshold
        for (i in 0 until canonicalSize) {
            if (preferredPositionTicks != null && preferredPositionTicks.isFinite()) {
                val positionTicks = firstOrdinal + i.toFloat() / SAMPLES_PER_TICK
                if (abs(positionTicks - preferredPositionTicks) > preferredWindowTicks) continue
            }
            val value = residuals[i]
            if (value.isNaN()) continue
            val residual = value - residualBaseline
            if (residual > peak) {
                peak = residual
                peakIndex = i
            }
        }
        if (peakIndex < 0) return null

        val edge = maxOf(threshold, peak * 0.35f)
        var lo = peakIndex
        while (lo > 0 && residualAt(residuals, residualBaseline, lo - 1) >= edge) lo--
        var hi = peakIndex
        while (hi + 1 < canonicalSize &&
            residualAt(residuals, residualBaseline, hi + 1) >= edge
        ) {
            hi++
        }

        val widthTicks = (hi - lo + 1).toFloat() / SAMPLES_PER_TICK
        if (widthTicks < MIN_BALL_WIDTH_TICKS || widthTicks > MAX_BALL_WIDTH_TICKS) return null

        var weightedIndex = 0f
        var weight = 0f
        for (i in lo..hi) {
            val w = (residualAt(residuals, residualBaseline, i) - threshold).coerceAtLeast(0f)
            weightedIndex += i * w
            weight += w
        }
        if (weight <= 0f) return null
        val centreIndex = weightedIndex / weight
        val q = firstOrdinal + centreIndex / SAMPLES_PER_TICK
        val axial = canonicalToAxial(q, ruler) ?: return null
        return Candidate(axial, q, peak / threshold)
    }

    private data class Affine(val gain: Float, val offset: Float)

    private fun robustAffine(
        x: FloatArray,
        y: FloatArray,
        count: Int,
        scratch: FloatArray
    ): Affine {
        val first = leastSquares(x, y, count, null)
        for (i in 0 until count) scratch[i] = y[i] - (first.gain * x[i] + first.offset)
        val centre = median(scratch, count)
        for (i in 0 until count) scratch[i] = abs(scratch[i] - centre)
        val sigma = median(scratch, count) * 1.4826f
        val limit = maxOf(MIN_RESIDUAL_LUMA, sigma * 3f)
        var inliers = 0
        for (i in 0 until count) {
            val residual = y[i] - (first.gain * x[i] + first.offset)
            if (abs(residual - centre) <= limit) {
                inlierScratch[i] = true
                inliers++
            } else {
                inlierScratch[i] = false
            }
        }
        return if (inliers >= count / 2) {
            leastSquares(x, y, count, inlierScratch)
        } else {
            first
        }
    }

    private fun leastSquares(
        x: FloatArray,
        y: FloatArray,
        count: Int,
        include: BooleanArray?
    ): Affine {
        var n = 0
        var sx = 0.0
        var sy = 0.0
        var sxx = 0.0
        var sxy = 0.0
        for (i in 0 until count) {
            if (include != null && !include[i]) continue
            val xi = x[i].toDouble()
            val yi = y[i].toDouble()
            n++
            sx += xi
            sy += yi
            sxx += xi * xi
            sxy += xi * yi
        }
        val denom = n * sxx - sx * sx
        if (n < 2 || abs(denom) < 1e-9) {
            val offset = if (n == 0) 0f else ((sy - sx) / n).toFloat()
            return Affine(1f, offset)
        }
        val gain = ((n * sxy - sx * sy) / denom).toFloat().coerceIn(0.5f, 2f)
        val offset = ((sy - gain * sx) / n).toFloat()
        return Affine(gain, offset)
    }

    private fun residualAt(residuals: FloatArray, baseline: Float, i: Int): Float {
        val value = residuals[i]
        return if (value.isNaN()) Float.NEGATIVE_INFINITY else value - baseline
    }

    private fun canonicalToAxial(q: Float, ruler: RulerSnapshot): Float? {
        val ordinals = ruler.ordinals
        val positions = ruler.axialPx
        if (q < ordinals.first() || q > ordinals.last()) return null
        var hi = 1
        while (hi < ordinals.size && ordinals[hi] < q) hi++
        if (hi >= ordinals.size) return positions.last()
        val lo = hi - 1
        val span = ordinals[hi] - ordinals[lo]
        if (span <= 0) return null
        val frac = (q - ordinals[lo]) / span
        return positions[lo] + (positions[hi] - positions[lo]) * frac
    }

    private fun axialToCanonical(axial: Float, ruler: RulerSnapshot): Float? {
        val positions = ruler.axialPx
        val ordinals = ruler.ordinals
        if (axial < positions.first() || axial > positions.last()) return null
        var hi = 1
        while (hi < positions.size && positions[hi] < axial) hi++
        if (hi >= positions.size) return ordinals.last().toFloat()
        val lo = hi - 1
        val span = positions[hi] - positions[lo]
        if (span <= 1e-3f) return null
        val frac = (axial - positions[lo]) / span
        return ordinals[lo] + (ordinals[hi] - ordinals[lo]) * frac
    }

    private fun safeAxialToCanonical(axial: Float, ruler: RulerSnapshot): Float? {
        if (ruler.count < 2 || ruler.ordinals.size != ruler.count) return null
        if (!strictlyIncreasing(ruler.axialPx) || !strictlyIncreasing(ruler.ordinals)) return null
        return axialToCanonical(axial, ruler)
    }

    private fun strictlyIncreasing(values: FloatArray): Boolean {
        for (i in 1 until values.size) if (values[i] <= values[i - 1]) return false
        return true
    }

    private fun strictlyIncreasing(values: IntArray): Boolean {
        for (i in 1 until values.size) if (values[i] <= values[i - 1]) return false
        return true
    }

    private fun median(values: FloatArray, count: Int): Float {
        if (count <= 0) return 0f
        values.sort(0, count)
        val mid = count ushr 1
        return if (count and 1 == 1) values[mid] else (values[mid - 1] + values[mid]) * 0.5f
    }
}
