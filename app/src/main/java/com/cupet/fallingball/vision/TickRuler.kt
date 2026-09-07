package com.cupet.fallingball.vision

import com.cupet.fallingball.signal.RobustStats
import kotlin.math.abs
import kotlin.math.roundToInt

enum class RulerState {
    /** No lock yet; nothing is being reported. */
    ACQUIRING,

    /** Locked and trusted. */
    VALID,

    /** Recent frames were rejected; still showing remembered ticks, but flagged. */
    DEGRADED,

    /** Rejected for long enough that the memory was thrown away. */
    LOST
}

/** Why a frame was rejected. Surfaced to the UI so the user can act on it. */
enum class InvalidReason {
    SHORT_SIGNAL,
    GAP_TOO_SMALL,
    TILT,
    NOT_FULLY_IN_FRAME,
    COUNT,
    GAP_JUMP,
    GEOM_INCONSISTENT,
    IRREGULAR,
    SHIFT
}

/** Immutable per-frame view of the ruler, safe to hand to the UI thread. */
class RulerSnapshot(
    val state: RulerState,
    /** Axial distance of each remembered tick, ascending. */
    val axialPx: FloatArray,
    /** Stable identity of each tick; index i of [axialPx] has ordinal `ordinals[i]`. */
    val ordinals: IntArray,
    val medianGapPx: Float,
    val shiftPx: Float,
    val shiftAgreement: Float,
    val invalidReason: InvalidReason?
) {
    val count get() = axialPx.size
    val isUsable get() = state == RulerState.VALID || state == RulerState.DEGRADED

    /**
     * Converts an axial pixel position into millimetres by interpolating between the two
     * bracketing graduations.
     *
     * Deliberately not `spacingMm / medianGapPx * px`. A single global scale is wrong
     * whenever perspective makes the graduations non-uniform along the tube (10-20% end to
     * end when the camera is close), and it re-introduces sensitivity to hand shake:
     * interpolating between physical features means a whole-frame translation cancels out,
     * because the ball and the graduations move together.
     *
     * @return null when the ruler is unusable or has fewer than two graduations
     */
    fun toMm(axial: Float, spacingMm: Float, refOrdinal: Int): Float? {
        if (count < 2) return null
        // Binary search for the bracketing pair.
        var lo = 0
        var hi = count - 1
        if (axial <= axialPx[0]) {
            lo = 0; hi = 1
        } else if (axial >= axialPx[count - 1]) {
            lo = count - 2; hi = count - 1
        } else {
            while (hi - lo > 1) {
                val mid = (lo + hi) ushr 1
                if (axialPx[mid] <= axial) lo = mid else hi = mid
            }
        }
        val span = axialPx[hi] - axialPx[lo]
        if (abs(span) < 1e-3f) return null
        val frac = (axial - axialPx[lo]) / span
        val ordinal = (ordinals[lo] - refOrdinal) + frac * (ordinals[hi] - ordinals[lo])
        return ordinal * spacingMm
    }

    /** Ordinal of the last graduation at or above [axial]; used as a session's zero. */
    fun ordinalAbove(axial: Float): Int? {
        if (count == 0) return null
        var best: Int? = null
        for (i in 0 until count) if (axialPx[i] <= axial) best = ordinals[i]
        return best ?: ordinals[0]
    }

    /** Visible ruler geometry only; retained off-axis ticks remain private tracking memory. */
    fun clippedToAxis(axisLengthPx: Float): RulerSnapshot {
        if (!axisLengthPx.isFinite() || axisLengthPx <= 0f || count == 0) return EMPTY
        var visibleCount = 0
        for (value in axialPx) {
            if (value in 0f..axisLengthPx) visibleCount++
        }
        if (visibleCount == count) return this
        val visibleAxial = FloatArray(visibleCount)
        val visibleOrdinals = IntArray(visibleCount)
        var destination = 0
        for (index in axialPx.indices) {
            if (axialPx[index] !in 0f..axisLengthPx) continue
            visibleAxial[destination] = axialPx[index]
            visibleOrdinals[destination] = ordinals[index]
            destination++
        }
        return RulerSnapshot(
            state = state,
            axialPx = visibleAxial,
            ordinals = visibleOrdinals,
            medianGapPx = medianGapPx,
            shiftPx = shiftPx,
            shiftAgreement = shiftAgreement,
            invalidReason = invalidReason
        )
    }

    companion object {
        val EMPTY = RulerSnapshot(
            RulerState.ACQUIRING, FloatArray(0), IntArray(0), 0f, 0f, 0f, null
        )
    }
}

/**
 * Turns per-frame tick detections into a stable, identified ruler.
 *
 * Two problems this solves that [TickDetector] cannot, because they are inherently
 * temporal:
 *
 *  1. **Identity.** The reference implementation keeps a sorted list of positions. Entries
 *     come and go at both ends as the detection box breathes, so "the i-th tick" refers to
 *     a different physical graduation over time. Any ball position expressed against that
 *     list drifts by whole divisions. Here every remembered tick carries an integer
 *     ordinal assigned when it was first seen and never renumbered.
 *
 *  2. **Registration.** Frame to frame the whole tick set slides, because a stale or
 *     jittery box moves the axial origin. Matching each detection to its nearest memory is
 *     ambiguous once that slide exceeds half a division, and the wrong match shifts every
 *     position by exactly one graduation while looking perfectly self-consistent.
 *
 * On the limits of (2), stated plainly: **a comb of identical graduations offers no way to
 * tell "did not move" from "moved exactly one division".** Shift the set by one period and
 * every detection lands on a different memory entry with zero residual and total
 * agreement. No amount of care with the tick data alone can separate those cases.
 *
 * What breaks the tie is information from outside the comb: the tube geometry. The caller
 * supplies `coarseShiftPx`, the axial displacement of the coordinate origin derived from
 * the oriented box, which is unaliased but only accurate to a few pixels. Matching is then
 * done around that prior instead of around zero, and the fine estimate is rejected when it
 * disagrees. This catches the realistic cause of a division slip -- a box whose end
 * breathes by roughly a division, moving the axial origin under a stationary tube -- which
 * is the case that matters, since a rigid cylinder cannot actually slide one graduation
 * relative to a box that reports no motion.
 *
 * Not thread safe; confined to the analysis thread.
 */
class TickRuler(private var config: RulerConfig) {

    companion object {
        /** Rejected frames are absorbed silently for this long. */
        const val TOLERATE_MS = 250L

        /** Beyond this, report degradation; beyond [LOST_MS], drop the memory. */
        const val DEGRADED_MS = 250L
        const val LOST_MS = 1000L

        /** Consecutive good frames needed to (re)acquire. */
        const val LOCK_FRAMES = 3

        // --- gates ---
        const val MIN_GAP_PX = 8f
        const val COUNT_TOLERANCE = 0.20f
        const val GAP_JUMP_TOLERANCE = 0.25f
        const val GEOM_TOLERANCE = 0.25f
        const val MAX_GAP_CV = 0.20f
        const val MIN_SHIFT_AGREEMENT = 0.7f

        /** Match window around the prior, as a fraction of the gap. */
        private const val MATCH_FRAC = 0.6f

        /** Residuals within this of the median count as agreeing. */
        private const val AGREE_FRAC = 0.25f

        /**
         * Accepted `|fineShift - prior|`, as a fraction of the gap.
         *
         * The fine estimate is wrapped into +/-half a gap by the matching window, so this
         * comparison fires exactly when the geometry claims the set moved by appreciably
         * more than one division -- the phone was swung, or the box jumped -- which is when
         * registration genuinely cannot be trusted. With current geometry the prior is a
         * real measurement good to a few pixels, so the window is tight; with extrapolated
         * geometry it is a constant-velocity guess and has to be looser.
         */
        private const val SHIFT_LIMIT_FRESH = 0.35f
        private const val SHIFT_LIMIT_STALE = 0.75f

        /** Fuse observation into memory at this weight (reference uses 0.3). */
        private const val MERGE_ALPHA = 0.3f
        private const val MERGE_FRAC = 0.4f

        /** Gap smoothing. */
        private const val GAP_ALPHA = 0.15f

        /** Discard memory outside this multiple of the tube length. */
        private const val PRUNE_MARGIN = 0.2f
    }

    private class Tick(var axialPx: Float, val ordinal: Int, var age: Int, var hits: Int)

    private val ticks = ArrayList<Tick>(128)
    private var minOrdinal = 0
    private var maxOrdinal = -1
    private var smoothedGap = 0f
    private var state = RulerState.ACQUIRING
    private var goodStreak = 0
    private var firstInvalidNs = 0L
    private var lastShift = 0f
    private var lastReason: InvalidReason? = null

    private var scratch = FloatArray(256)

    fun updateConfig(newConfig: RulerConfig) {
        if (newConfig != config) {
            config = newConfig
            reset()
        }
    }

    fun reset() {
        ticks.clear()
        minOrdinal = 0
        maxOrdinal = -1
        smoothedGap = 0f
        state = RulerState.ACQUIRING
        goodStreak = 0
        firstInvalidNs = 0L
        lastShift = 0f
        lastReason = null
    }

    /**
     * @param coarseShiftPx independent prediction of how far the whole tick set should
     *   have moved along the axis since the last accepted frame, obtained from the tube
     *   geometry (`currentAxis.axialOf(previousAxis.origin)`). This is the *only*
     *   unaliased information available -- see the class note on what that buys.
     * @param geometryChanged whether the coordinate frame moved since the previous consumer
     *   frame. Real YOLO targets and interpolated geometry use the same path here.
     */
    fun onFrame(
        ticksIn: FloatArray,
        tickCount: Int,
        interpolated: BooleanArray,
        medianGapPx: Float,
        axisLengthPx: Float,
        tiltRad: Float,
        fullyInFrame: Boolean,
        signalLen: Int,
        coarseShiftPx: Float,
        geometryChanged: Boolean,
        timestampNs: Long,
        geometryScale: Float = 1f
    ): RulerSnapshot {
        if (geometryChanged && ticks.isNotEmpty()) {
            val scale = geometryScale.takeIf { it.isFinite() && it > 0f } ?: 1f
            for (tick in ticks) {
                tick.axialPx = tick.axialPx * scale + coarseShiftPx
            }
            smoothedGap *= scale
            // The coordinate transform happened once above; it is not frame velocity.
            lastShift = 0f
        }

        val reason = validateFrame(
            tickCount, medianGapPx, axisLengthPx, tiltRad, fullyInFrame, signalLen
        )
        if (reason != null) return reject(reason, timestampNs)

        if (state == RulerState.ACQUIRING || ticks.isEmpty()) {
            return acquire(ticksIn, tickCount, medianGapPx, timestampNs)
        }

        // --- registration -------------------------------------------------------------
        val prior = if (geometryChanged) 0f else lastShift
        val limit = (if (geometryChanged) SHIFT_LIMIT_FRESH else SHIFT_LIMIT_STALE) * smoothedGap

        var matched = 0
        ensureScratch(tickCount)
        for (i in 0 until tickCount) {
            if (interpolated[i]) continue // never register against a position we invented
            val p = ticksIn[i]
            var bestD = Float.MAX_VALUE
            var bestTick = -1
            for (t in ticks.indices) {
                val d = abs(ticks[t].axialPx + prior - p)
                if (d < bestD) {
                    bestD = d
                    bestTick = t
                }
            }
            if (bestTick >= 0 && bestD < MATCH_FRAC * smoothedGap) {
                scratch[matched++] = p - ticks[bestTick].axialPx
            }
        }
        if (matched < 2 || matched < ticks.size / 2) return reject(InvalidReason.SHIFT, timestampNs)

        val residuals = scratch.copyOf(matched)
        val shift = RobustStats.median(scratch, matched)
        var agree = 0
        for (i in 0 until matched) {
            if (abs(residuals[i] - shift) < AGREE_FRAC * smoothedGap) agree++
        }
        val agreement = agree.toFloat() / matched
        if (agreement < MIN_SHIFT_AGREEMENT || abs(shift - prior) > limit) {
            return reject(InvalidReason.SHIFT, timestampNs)
        }

        // --- fuse ---------------------------------------------------------------------
        for (t in ticks) {
            t.axialPx += shift
            t.age++
        }
        val used = BooleanArray(tickCount)
        for (t in ticks) {
            var bestD = Float.MAX_VALUE
            var bestI = -1
            for (i in 0 until tickCount) {
                if (used[i]) continue
                val d = abs(ticksIn[i] - t.axialPx)
                if (d < bestD) {
                    bestD = d
                    bestI = i
                }
            }
            if (bestI >= 0 && bestD < MERGE_FRAC * medianGapPx) {
                t.axialPx = t.axialPx * (1f - MERGE_ALPHA) + ticksIn[bestI] * MERGE_ALPHA
                t.hits++
                t.age = 0
                used[bestI] = true
            }
            // Unmatched memory is *kept*: the graduation is probably just hidden this
            // frame (box shrank, ball passing over it, glare).
        }

        // Newly revealed graduations. Ordinal is derived from a neighbour, never by
        // renumbering the sorted list.
        for (i in 0 until tickCount) {
            if (used[i] || interpolated[i]) continue
            val p = ticksIn[i]
            val nearest = ticks.minByOrNull { abs(it.axialPx - p) } ?: continue
            val step = ((p - nearest.axialPx) / medianGapPx).roundToInt()
            if (step == 0) continue
            val ordinal = nearest.ordinal + step
            if (ticks.any { it.ordinal == ordinal }) continue
            ticks.add(Tick(p, ordinal, 0, 1))
            if (ordinal < minOrdinal) minOrdinal = ordinal
            if (ordinal > maxOrdinal) maxOrdinal = ordinal
        }

        prune(axisLengthPx)
        ticks.sortBy { it.axialPx }
        smoothedGap = smoothedGap * (1f - GAP_ALPHA) + medianGapPx * GAP_ALPHA
        lastShift = if (geometryChanged) 0f else shift
        firstInvalidNs = 0L
        lastReason = null
        state = RulerState.VALID
        val reportedShift = if (geometryChanged) coarseShiftPx + shift else shift
        return snapshot(reportedShift, agreement, null)
    }

    // --- gates ------------------------------------------------------------------------

    private fun validateFrame(
        tickCount: Int,
        medianGapPx: Float,
        axisLengthPx: Float,
        tiltRad: Float,
        fullyInFrame: Boolean,
        signalLen: Int
    ): InvalidReason? {
        if (signalLen < TickDetector.MIN_SIGNAL_LEN) return InvalidReason.SHORT_SIGNAL
        if (medianGapPx < MIN_GAP_PX) return InvalidReason.GAP_TOO_SMALL
        if (abs(tiltRad) > TubeAxis.MAX_TILT_RAD) return InvalidReason.TILT
        if (!fullyInFrame) return InvalidReason.NOT_FULLY_IN_FRAME

        // The count gate needs the user's graduation total. Without it there is nothing to
        // compare against, and rejecting every frame against a guessed number would show
        // no ticks at all -- worse than not checking.
        if (config.hasExpectedCount) {
            val expected = config.expectedVisibleCount
            if (tickCount < expected * (1f - COUNT_TOLERANCE) ||
                tickCount > expected * (1f + COUNT_TOLERANCE)
            ) {
                return InvalidReason.COUNT
            }
        }

        // Checked before the geometry cross-check on purpose: a jumped gap is a transient
        // bad frame, whereas GEOM_INCONSISTENT tells the user their configuration is wrong.
        // A sudden gap change violates both, and blaming the configuration for one bad
        // frame sends the user off to re-check a number that was fine.
        if (state == RulerState.VALID && smoothedGap > 0f &&
            abs(medianGapPx - smoothedGap) > GAP_JUMP_TOLERANCE * smoothedGap
        ) {
            return InvalidReason.GAP_JUMP
        }

        // Cross-check the user's tick count against the measured geometry. Catches a
        // mistyped count and a lock onto the wrong object in one gate.
        if (config.hasExpectedCount) {
            val impliedSpan = medianGapPx * (config.expectedTickCount - 1)
            val visibleSpan = axisLengthPx * (1f - 2f * AxialSampler.AXIAL_TRIM)
            if (abs(impliedSpan - visibleSpan) > GEOM_TOLERANCE * visibleSpan) {
                return InvalidReason.GEOM_INCONSISTENT
            }
        }
        return null
    }

    private fun checkRegularity(ticksIn: FloatArray, tickCount: Int): Boolean {
        if (tickCount < 3) return false
        ensureScratch(tickCount)
        for (i in 1 until tickCount) scratch[i - 1] = ticksIn[i] - ticksIn[i - 1]
        return RobustStats.coefficientOfVariation(scratch, tickCount - 1) <= MAX_GAP_CV
    }

    // --- state transitions ------------------------------------------------------------

    private fun acquire(
        ticksIn: FloatArray,
        tickCount: Int,
        medianGapPx: Float,
        timestampNs: Long
    ): RulerSnapshot {
        if (!checkRegularity(ticksIn, tickCount)) {
            return reject(InvalidReason.IRREGULAR, timestampNs)
        }
        goodStreak++
        if (goodStreak < LOCK_FRAMES) {
            return snapshot(0f, 0f, null, RulerState.ACQUIRING)
        }

        ticks.clear()
        for (i in 0 until tickCount) ticks.add(Tick(ticksIn[i], i, 0, 1))
        minOrdinal = 0
        maxOrdinal = tickCount - 1
        smoothedGap = medianGapPx
        lastShift = 0f
        firstInvalidNs = 0L
        lastReason = null
        state = RulerState.VALID
        return snapshot(0f, 1f, null)
    }

    private fun reject(reason: InvalidReason, timestampNs: Long): RulerSnapshot {
        goodStreak = 0
        lastReason = reason
        if (firstInvalidNs == 0L) firstInvalidNs = timestampNs
        val elapsedMs = (timestampNs - firstInvalidNs) / 1_000_000L

        // Thresholds are in time, not frames: the sample rate moves between roughly 10 and
        // 30 fps depending on how often the detector gets a turn, so a frame count would
        // mean different durations on different devices and in different modes.
        state = when {
            // Sticky: once the memory is gone the user needs to keep seeing the failure
            // until a genuine re-acquisition, not flicker back to "searching".
            state == RulerState.LOST -> RulerState.LOST
            ticks.isEmpty() -> RulerState.ACQUIRING
            elapsedMs <= TOLERATE_MS -> RulerState.VALID
            elapsedMs <= LOST_MS -> RulerState.DEGRADED
            else -> {
                ticks.clear()
                minOrdinal = 0
                maxOrdinal = -1
                smoothedGap = 0f
                RulerState.LOST
            }
        }
        return snapshot(lastShift, 0f, reason)
    }

    private fun prune(axisLengthPx: Float) {
        val lo = -PRUNE_MARGIN * axisLengthPx
        val hi = axisLengthPx * (1f + PRUNE_MARGIN)
        ticks.removeAll { it.axialPx < lo || it.axialPx > hi }
    }

    private fun snapshot(
        shift: Float,
        agreement: Float,
        reason: InvalidReason?,
        forced: RulerState? = null
    ): RulerSnapshot {
        val n = ticks.size
        val axial = FloatArray(n)
        val ords = IntArray(n)
        for (i in 0 until n) {
            axial[i] = ticks[i].axialPx
            ords[i] = ticks[i].ordinal
        }
        return RulerSnapshot(
            state = forced ?: state,
            axialPx = axial,
            ordinals = ords,
            medianGapPx = smoothedGap,
            shiftPx = shift,
            shiftAgreement = agreement,
            invalidReason = reason
        )
    }

    private fun ensureScratch(n: Int) {
        if (scratch.size < n) {
            var cap = scratch.size
            while (cap < n) cap *= 2
            scratch = FloatArray(cap)
        }
    }
}
