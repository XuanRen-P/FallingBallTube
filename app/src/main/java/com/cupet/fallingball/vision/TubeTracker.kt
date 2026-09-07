package com.cupet.fallingball.vision

/**
 * Temporal lock + camera-rate SmoothStep interpolation for tube OBB detections.
 *
 * Split across two threads on purpose. [onDetection] runs on the inference worker whenever a
 * detection completes; [advance] is called by the analysis thread on *every* frame, including
 * the many frames that carry no new detection. The high-rate call samples a short, bounded
 * transition between the latest two OBB observations, so the scale path and overlay do not
 * inherit the low-frequency detector's step response.
 *
 * Mutable state is guarded by the instance lock and only ever published as an immutable
 * [TubeTrackResult] through a volatile field, so the reader never sees a half-updated box.
 */
class TubeTracker(
    private val lockFramesNeeded: Int = 2,
    loseMs: Long = 1200,
    private val defaultTransitionMs: Long = 180L
) {
    data class StateSnapshot(
        val result: TubeTrackResult?,
        val axis: TubeAxis?,
        val detectionTimestampNs: Long,
        val geometryRevision: Long
    )

    private var stableCount = 0
    private var locked = false
    private var startObs: TubeObservation? = null
    private var targetObs: TubeObservation? = null
    private var transitionStartNs = 0L
    private var transitionDurationNs = 0L
    /** Latest camera timestamp rendered by the high-rate analysis thread. */
    private var lastAdvancedTimestampNs = 0L
    private var lastHitNs = 0L
    private var lossTimeoutMs = loseMs

    init {
        require(loseMs > 0L)
        require(defaultTransitionMs >= 0L)
    }

    /** Keeps the retained tube lock aligned with the active thermal load profile. */
    fun setLossTimeoutMs(timeoutMs: Long) = synchronized(this) {
        require(timeoutMs > 0L)
        lossTimeoutMs = timeoutMs
    }

    private var lastTsNs = 0L
    private var obbFps = 0f
    private var lastInferenceMs = 0f
    private var lastModelInputSize = ModelProfile.DEFAULT.inputSize
    private var activeProfileGeneration = 0L
    private var geometryRevision = 0L
    private var lastPipelineTimings: PipelineTimings? = null

    @Volatile
    private var published: TubeTrackResult? = null
    private var publishedObs: TubeObservation? = null

    /** Axis of the most recent accepted detection, cached so the reader need not recompute. */
    @Volatile
    var axis: TubeAxis? = null
        private set

    /** Capture timestamp the published result was derived from. */
    @Volatile
    var lastDetectionNs = 0L
        private set

    fun reset() = synchronized(this) {
        resetLocked()
    }

    /**
     * Starts a new YOLO profile generation. A detection already running for an older
     * generation may still finish, but [onDetection] will discard it under this same lock.
     */
    fun resetForProfile(profileGeneration: Long) = synchronized(this) {
        activeProfileGeneration = profileGeneration
        resetLocked()
    }

    private fun resetLocked() {
        stableCount = 0
        locked = false
        startObs = null
        targetObs = null
        transitionStartNs = 0L
        transitionDurationNs = defaultTransitionMs * 1_000_000L
        lastAdvancedTimestampNs = 0L
        lastHitNs = 0L
        lastTsNs = 0L
        obbFps = 0f
        lastInferenceMs = 0f
        lastModelInputSize = ModelProfile.DEFAULT.inputSize
        lastPipelineTimings = null
        published = null
        publishedObs = null
        axis = null
        lastDetectionNs = 0L
        geometryRevision = 0L
    }

    /** Latest result, or null before the first detection. Safe from any thread. */
    fun snapshot(): TubeTrackResult? = published

    /** One-lock view for consumers that must use geometry and its timestamp together. */
    fun stateSnapshot(): StateSnapshot = synchronized(this) {
        stateSnapshotLocked()
    }

    /**
     * Samples the current geometry at the camera timestamp. This must run before that frame
     * consumes the axis, ruler, or ROI. Intermediate samples keep the accepted YOLO revision;
     * a revision identifies a detection epoch, not an animation frame.
     */
    fun advance(timestampNs: Long): StateSnapshot = synchronized(this) {
        val sampleTimestampNs = maxOf(timestampNs, lastAdvancedTimestampNs)
        lastAdvancedTimestampNs = sampleTimestampNs
        val current = sampleAt(sampleTimestampNs)
        // A failed pre-lock detection publishes a searching result without geometry. Do not
        // resurrect that result merely because an older target is retained internally.
        if (current != null && published?.left != null && current !== publishedObs) {
            publish(current, locked = locked)
        }
        stateSnapshotLocked()
    }

    /**
     * Feeds one completed detection. [obs] is null when the model found nothing.
     *
     * Loss is measured in milliseconds rather than frames: the detection rate now varies
     * with device speed and with whether a session is running, so a frame count would mean
     * a different duration in each case.
     */
    fun onDetection(
        obs: TubeObservation?,
        timestampNs: Long,
        frameW: Int,
        frameH: Int,
        inferenceMs: Float = 0f,
        modelInputSize: Int = ModelProfile.DEFAULT.inputSize,
        profileGeneration: Long = 0L,
        timings: PipelineTimings? = null
    ) {
        synchronized(this) {
            if (profileGeneration != activeProfileGeneration) return
            updateFps(timestampNs)
            lastModelInputSize = modelInputSize
            if (timings != null) {
                lastPipelineTimings = lastPipelineTimings
                    ?.smoothedWith(timings)
                    ?: timings
                lastInferenceMs = requireNotNull(lastPipelineTimings).totalMs
            } else {
                lastInferenceMs = inferenceMs
            }

            if (obs == null) {
                val heldFor = if (lastHitNs == 0L) Long.MAX_VALUE
                else (timestampNs - lastHitNs).coerceAtLeast(0L) / 1_000_000L
                if (locked && heldFor < lossTimeoutMs && targetObs != null) {
                    val current = sampleAt(maxOf(timestampNs, lastAdvancedTimestampNs))
                    if (current != null) publish(current, locked = true)
                    return
                }
                if (heldFor >= lossTimeoutMs) {
                    locked = false
                    stableCount = 0
                    startObs = null
                    targetObs = null
                    transitionStartNs = 0L
                    axis = null
                }
                published = TubeTrackResult(
                    mode = TrackMode.SEARCHING,
                    confidence = 0f,
                    frameWidth = frameW,
                    frameHeight = frameH,
                    inferenceFps = obbFps,
                    inferenceMs = lastInferenceMs,
                    modelInputSize = lastModelInputSize,
                    profileGeneration = activeProfileGeneration,
                    pipelineTimings = lastPipelineTimings
                )
                publishedObs = null
                return
            }

            lastHitNs = timestampNs
            val transitionTimestampNs = maxOf(timestampNs, lastAdvancedTimestampNs)
            val current = sampleAt(transitionTimestampNs) ?: obs
            startObs = current
            targetObs = obs
            transitionStartNs = transitionTimestampNs
            transitionDurationNs = transitionDurationForCurrentFps()

            if (!locked) {
                stableCount++
                if (stableCount >= lockFramesNeeded) locked = true
            }

            // Show the box as soon as we have a detection, even before a full lock.
            geometryRevision += 1L
            publish(current, locked = locked)
        }
    }

    private fun stateSnapshotLocked(): StateSnapshot = StateSnapshot(
        result = published,
        axis = axis,
        detectionTimestampNs = lastDetectionNs,
        geometryRevision = geometryRevision
    )

    /**
     * Keep the transition below 70% of the observed detector period when inference is
     * unusually fast. At the normal 2-4 FPS cadence this remains near the configured 180 ms;
     * thermal slowdown never makes a transition drag out longer.
     */
    private fun transitionDurationForCurrentFps(): Long {
        val configuredNs = defaultTransitionMs * 1_000_000L
        if (configuredNs <= 0L) return 0L
        if (obbFps <= 0f || !obbFps.isFinite()) return configuredNs
        val detectorPeriodNs = (1_000_000_000.0 / obbFps)
            .toLong()
            .coerceAtLeast(1_000_000L)
        return minOf(configuredNs, (detectorPeriodNs * 0.7).toLong())
            .coerceAtLeast(1_000_000L)
    }

    private fun sampleAt(timestampNs: Long): TubeObservation? {
        val start = startObs ?: targetObs ?: return null
        val target = targetObs ?: return start
        if (start === target || transitionDurationNs <= 0L) return target

        val elapsedNs = (timestampNs - transitionStartNs).coerceAtLeast(0L)
        if (elapsedNs >= transitionDurationNs) return target
        val progress = (elapsedNs.toDouble() / transitionDurationNs.toDouble())
            .coerceIn(0.0, 1.0)
            .toFloat()
        val weight = progress * progress * (3f - 2f * progress)
        return interpolateObservation(start, target, weight)
    }

    private fun publish(obs: TubeObservation, locked: Boolean) {
        val result = TubeTrackResult(
            mode = if (locked) TrackMode.LOCKED else TrackMode.SEARCHING,
            left = obs.left,
            right = obs.right,
            topY = obs.topY,
            bottomY = obs.bottomY,
            confidence = obs.confidence,
            frameWidth = obs.frameWidth,
            frameHeight = obs.frameHeight,
            inferenceFps = obbFps,
            inferenceMs = lastInferenceMs,
            modelInputSize = lastModelInputSize,
            profileGeneration = activeProfileGeneration,
            geometryCaptureTimestampNs = lastHitNs,
            geometryRevision = geometryRevision,
            corners = obs.corners,
            pipelineTimings = lastPipelineTimings
        )
        axis = TubeAxis.fromCorners(obs.corners)
        lastDetectionNs = lastHitNs
        publishedObs = obs
        published = result
    }

    private fun interpolateObservation(
        prev: TubeObservation,
        cur: TubeObservation,
        a: Float
    ): TubeObservation {
        val left = prev.left.withSmoothed(cur.left, a)
        val right = prev.right.withSmoothed(cur.right, a)
        val meanSlope = (left.slope + right.slope) * 0.5f
        val leftFixed = left.copy(slope = left.slope * 0.4f + meanSlope * 0.6f)
        val rightFixed = right.copy(slope = right.slope * 0.4f + meanSlope * 0.6f)

        fun blend(py: Float?, cy: Float?): Float? {
            if (cy == null) return py
            if (py == null) return cy
            return py * (1 - a) + cy * a
        }

        return TubeObservation(
            left = leftFixed,
            right = rightFixed,
            topY = blend(prev.topY, cur.topY),
            bottomY = blend(prev.bottomY, cur.bottomY),
            confidence = prev.confidence * (1 - a) + cur.confidence * a,
            frameWidth = cur.frameWidth,
            frameHeight = cur.frameHeight,
            corners = interpolateCorners(prev.corners, cur.corners, a)
        )
    }

    private fun interpolateCorners(
        prev: List<Pair<Float, Float>>?,
        cur: List<Pair<Float, Float>>?,
        a: Float
    ): List<Pair<Float, Float>>? {
        if (cur == null || cur.size != 4) return prev
        if (prev == null || prev.size != 4) return cur
        val matchedPrev = topologyPreservingCornerMatch(prev, cur)
        return List(4) { i ->
            val px = matchedPrev[i].first
            val py = matchedPrev[i].second
            val cx = cur[i].first
            val cy = cur[i].second
            (px * (1 - a) + cx * a) to (py * (1 - a) + cy * a)
        }
    }

    /**
     * OBB corner numbering may rotate or reverse, but its cyclic adjacency must not change.
     * A greedy point match can pair the leading edge with the old trailing edge during a
     * translation larger than the tube width, creating a self-crossed, collapsed rectangle.
     */
    private fun topologyPreservingCornerMatch(
        prev: List<Pair<Float, Float>>,
        cur: List<Pair<Float, Float>>
    ): Array<Pair<Float, Float>> {
        var bestCost = Float.POSITIVE_INFINITY
        var bestShift = 0
        var bestReversed = false
        for (reversed in listOf(false, true)) {
            for (shift in 0 until 4) {
                var cost = 0f
                for (i in 0 until 4) {
                    val previousIndex = if (reversed) {
                        (shift - i).mod(4)
                    } else {
                        (shift + i).mod(4)
                    }
                    val dx = cur[i].first - prev[previousIndex].first
                    val dy = cur[i].second - prev[previousIndex].second
                    cost += dx * dx + dy * dy
                }
                if (cost < bestCost) {
                    bestCost = cost
                    bestShift = shift
                    bestReversed = reversed
                }
            }
        }
        return Array(4) { i ->
            val previousIndex = if (bestReversed) {
                (bestShift - i).mod(4)
            } else {
                (bestShift + i).mod(4)
            }
            prev[previousIndex]
        }
    }

    private fun updateFps(timestampNs: Long) {
        if (lastTsNs != 0L) {
            val dt = (timestampNs - lastTsNs) / 1_000_000_000.0
            if (dt > 1e-4) {
                val inst = (1.0 / dt).toFloat()
                obbFps = if (obbFps <= 0f) inst else obbFps * 0.85f + inst * 0.15f
            }
        }
        lastTsNs = timestampNs
    }
}
