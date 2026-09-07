package com.cupet.fallingball.ui

import com.cupet.fallingball.vision.ScaleReading
import com.cupet.fallingball.vision.TubeAxis
import com.cupet.fallingball.vision.TubeBallPoint
import com.cupet.fallingball.vision.TubeTrackResult
import com.cupet.fallingball.vision.WallLine

/** Smooths complete analysis results on the display clock without knowing their source. */
internal class TubePresentationInterpolator(
    transitionMs: Long = DEFAULT_TRANSITION_MS
) {
    private val maximumTransitionNs = transitionMs * 1_000_000L
    private var start: TubeTrackResult? = null
    private var target: TubeTrackResult? = null
    private var transitionStartNs = 0L
    private var transitionDurationNs = maximumTransitionNs
    private var lastAcceptedTimestampNs = 0L

    init {
        require(transitionMs > 0L)
    }

    @Synchronized
    fun reset() {
        start = null
        target = null
        transitionStartNs = 0L
        transitionDurationNs = maximumTransitionNs
        lastAcceptedTimestampNs = 0L
    }

    @Synchronized
    fun accept(result: TubeTrackResult, timestampNs: Long) {
        val acceptedAtNs = maxOf(timestampNs, lastAcceptedTimestampNs)
        val oldTarget = target
        if (oldTarget == null || !hasGeometry(oldTarget) || !hasGeometry(result)) {
            start = result
            target = result
            transitionStartNs = acceptedAtNs
        } else if (sameGeometry(oldTarget, result)) {
            target = result
            if (!isAnimatingLocked(acceptedAtNs)) start = result
        } else {
            start = sampleLocked(acceptedAtNs) ?: oldTarget
            target = result
            transitionStartNs = acceptedAtNs
            val sourceIntervalNs = if (lastAcceptedTimestampNs > 0L) {
                acceptedAtNs - lastAcceptedTimestampNs
            } else {
                maximumTransitionNs
            }
            transitionDurationNs = minOf(
                maximumTransitionNs,
                (sourceIntervalNs * SOURCE_INTERVAL_FRACTION).toLong()
            ).coerceAtLeast(MINIMUM_TRANSITION_NS)
        }
        lastAcceptedTimestampNs = acceptedAtNs
    }

    @Synchronized
    fun sample(timestampNs: Long): TubeTrackResult? = sampleLocked(timestampNs)

    @Synchronized
    fun isAnimatingAt(timestampNs: Long): Boolean = isAnimatingLocked(timestampNs)

    private fun isAnimatingLocked(timestampNs: Long): Boolean {
        val from = start ?: return false
        val to = target ?: return false
        return !sameGeometry(from, to) &&
            timestampNs - transitionStartNs < transitionDurationNs
    }

    private fun sampleLocked(timestampNs: Long): TubeTrackResult? {
        val from = start ?: return target
        val to = target ?: return from
        if (sameGeometry(from, to) || transitionDurationNs <= 0L) return to
        val elapsedNs = (timestampNs - transitionStartNs).coerceAtLeast(0L)
        if (elapsedNs >= transitionDurationNs) {
            start = to
            return to
        }
        val progress = (elapsedNs.toDouble() / transitionDurationNs.toDouble())
            .coerceIn(0.0, 1.0)
            .toFloat()
        if (progress <= 0f) return from
        val weight = progress * progress * (3f - 2f * progress)
        return interpolate(from, to, weight)
    }

    private fun interpolate(
        from: TubeTrackResult,
        to: TubeTrackResult,
        weight: Float
    ): TubeTrackResult {
        val corners = interpolateCorners(from.corners, to.corners, weight)
        val presentedAxis = TubeAxis.fromCorners(corners)
        return to.copy(
            left = interpolateLine(from.left, to.left, weight),
            right = interpolateLine(from.right, to.right, weight),
            topY = interpolateNullable(from.topY, to.topY, weight),
            bottomY = interpolateNullable(from.bottomY, to.bottomY, weight),
            confidence = lerp(from.confidence, to.confidence, weight),
            corners = corners,
            scale = presentedAxis?.let { interpolateScale(from.scale, to.scale, it, weight) }
                ?: to.scale
        )
    }

    private fun interpolateScale(
        from: ScaleReading?,
        to: ScaleReading?,
        axis: TubeAxis,
        weight: Float
    ): ScaleReading? {
        to ?: return null
        val matchingTicks = from != null &&
            from.axialPx.size == to.axialPx.size &&
            from.ordinals.contentEquals(to.ordinals)
        val axialPx = if (matchingTicks) {
            FloatArray(to.axialPx.size) { index ->
                lerp(requireNotNull(from).axialPx[index], to.axialPx[index], weight)
            }
        } else {
            to.axialPx
        }
        val ballAxialPx = to.ballAxialPx?.let { targetAxial ->
            from?.ballAxialPx?.let { sourceAxial ->
                lerp(sourceAxial, targetAxial, weight)
            } ?: targetAxial
        }
        val targetBall = to.ballCenter
        val ballCenter = if (ballAxialPx != null && targetBall != null) {
            val lateralPx = interpolateNullable(
                from?.ballCenter?.lateralPx,
                targetBall.lateralPx,
                weight
            ) ?: targetBall.lateralPx
            TubeBallPoint(
                x = axis.pointX(ballAxialPx, lateralPx),
                y = axis.pointY(ballAxialPx, lateralPx),
                axialPx = ballAxialPx,
                lateralPx = lateralPx
            )
        } else {
            null
        }
        return ScaleReading(
            axis = axis,
            axialPx = axialPx,
            ordinals = to.ordinals,
            ballAxialPx = ballAxialPx,
            ballCenter = ballCenter,
            ballConfidence = lerp(from?.ballConfidence ?: to.ballConfidence, to.ballConfidence, weight),
            ballSource = to.ballSource,
            wallContact = to.wallContact,
            medianGapPx = lerp(from?.medianGapPx ?: to.medianGapPx, to.medianGapPx, weight),
            bandHalfWidthPx = lerp(
                from?.bandHalfWidthPx ?: to.bandHalfWidthPx,
                to.bandHalfWidthPx,
                weight
            ),
            state = to.state,
            invalidReason = to.invalidReason
        )
    }

    private fun interpolateLine(from: WallLine?, to: WallLine?, weight: Float): WallLine? {
        to ?: return from
        from ?: return to
        return WallLine(
            slope = lerp(from.slope, to.slope, weight),
            intercept = lerp(from.intercept, to.intercept, weight),
            score = lerp(from.score, to.score, weight)
        )
    }

    private fun interpolateNullable(from: Float?, to: Float?, weight: Float): Float? {
        to ?: return from
        from ?: return to
        return lerp(from, to, weight)
    }

    private fun interpolateCorners(
        from: List<Pair<Float, Float>>?,
        to: List<Pair<Float, Float>>?,
        weight: Float
    ): List<Pair<Float, Float>>? {
        if (to == null || to.size != 4) return from
        if (from == null || from.size != 4) return to
        val matchedFrom = topologyPreservingMatch(from, to)
        return List(4) { index ->
            lerp(matchedFrom[index].first, to[index].first, weight) to
                lerp(matchedFrom[index].second, to[index].second, weight)
        }
    }

    private fun topologyPreservingMatch(
        from: List<Pair<Float, Float>>,
        to: List<Pair<Float, Float>>
    ): Array<Pair<Float, Float>> {
        var bestCost = Float.POSITIVE_INFINITY
        var bestShift = 0
        var bestReversed = false
        for (reversed in listOf(false, true)) {
            for (shift in 0 until 4) {
                var cost = 0f
                for (index in 0 until 4) {
                    val fromIndex = if (reversed) {
                        (shift - index).mod(4)
                    } else {
                        (shift + index).mod(4)
                    }
                    val dx = to[index].first - from[fromIndex].first
                    val dy = to[index].second - from[fromIndex].second
                    cost += dx * dx + dy * dy
                }
                if (cost < bestCost) {
                    bestCost = cost
                    bestShift = shift
                    bestReversed = reversed
                }
            }
        }
        return Array(4) { index ->
            val fromIndex = if (bestReversed) {
                (bestShift - index).mod(4)
            } else {
                (bestShift + index).mod(4)
            }
            from[fromIndex]
        }
    }

    private fun hasGeometry(result: TubeTrackResult): Boolean =
        result.corners?.size == 4 || (result.left != null && result.right != null)

    private fun sameGeometry(left: TubeTrackResult, right: TubeTrackResult): Boolean =
        left.corners == right.corners &&
            left.left == right.left &&
            left.right == right.right &&
            left.topY == right.topY &&
            left.bottomY == right.bottomY

    private fun lerp(from: Float, to: Float, weight: Float): Float =
        from + (to - from) * weight

    private companion object {
        const val DEFAULT_TRANSITION_MS = 180L
        const val MINIMUM_TRANSITION_NS = 16_000_000L
        const val SOURCE_INTERVAL_FRACTION = 0.9
    }
}
