package com.cupet.fallingball.vision

import com.cupet.fallingball.signal.PeakFinder
import com.cupet.fallingball.signal.PeakSet
import com.cupet.fallingball.signal.RobustStats
import com.cupet.fallingball.signal.SavitzkyGolay
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Single-frame, stateless detection of scale lines and the ball from an axial profile.
 *
 * Port of the core of `kedu_detect.py`, with every threshold made scale-invariant. The
 * reference hard-codes pixel and grey-level constants (`prominence=15`, `width=(8,100)`,
 * a +/-25 px mask) that were tuned for one video's resolution and exposure. Here they are
 * expressed relative to measured frame contrast and tube width, because:
 *
 *  - Some devices deliver Y in video range (16-235), compressing contrast by ~14%.
 *  - The tube's pixel width varies with distance and with 720p/1080p.
 *
 * All state lives in the caller; this object only owns reusable scratch buffers, so it
 * must be confined to the analysis thread.
 */
class TickDetector {

    companion object {
        /**
         * Ball detection uses fixed pixel thresholds, matching `kedu_detect.py` verbatim.
         *
         * The reference calls `find_peaks(inv_signal, prominence=15, width=(8, 100))` on the
         * raw (unsmoothed) inverted signal, takes the strongest qualifying peak, and masks
         * ±25 px around it. That logic is reproduced here with the margins slightly widened
         * so blobs that are borderline on the PC still get picked up on a handheld phone.
         */
        private const val BALL_PROMINENCE = 12f    // reference: 15
        private const val BALL_WIDTH_MIN_PX = 6f   // reference: 8
        private const val BALL_WIDTH_MAX_PX = 200f // reference: 100
        private const val BALL_MASK_HALF_PX = 30f  // reference: 25

        /** Tick prominence floor before the relative-strength filter. */
        private const val TICK_PROM_ABS = 0.8f
        private const val TICK_PROM_REL = 0.02f

        /** Keep peaks at least this fraction of the typical tick strength (reference: 0.35). */
        private const val STRONG_FRAC = 0.35f

        /**
         * How many of the strongest peaks define "typical tick strength".
         *
         * The reference scales its cutoff off the *maximum* prominence, which makes the
         * whole tick set hostage to a single outlier: when the ball slips past the width
         * gate its prominence (2-3x a tick's) triples the cutoff and every genuine tick is
         * discarded, so detection fails with no indication of why. A percentile is no
         * better -- it depends on how many noise peaks happen to sit below the ticks, and
         * flips from working to useless when a few lines go missing. Median of a fixed
         * count from the top is independent of the noise population and still tolerates
         * four outliers.
         */
        private const val STRONG_REF_TOP_K = 9

        /**
         * A reconstructed tick must look at least this fraction as strong as a typical
         * tick to be snapped onto a local maximum; otherwise it goes at the predicted
         * position. Without this, gap filling snaps onto noise and lands the line up to a
         * third of a gap away from where it belongs.
         */
        private const val FILL_SNAP_MIN_FRAC = 0.3f

        /** Printed numerals are fatter than scale lines; drop anything this wide. */
        private const val MAX_TICK_WIDTH_FRAC_OF_GAP = 0.6f

        /** Gap-fill search window, as a fraction of the fitted local gap. */
        private const val FILL_SEARCH_FRAC = 0.22f

        /** A measured peak farther from the winning comb phase is an unrelated feature. */
        private const val GRID_MATCH_FRAC = 0.22f
        private const val PERSPECTIVE_SEED_MATCH_FRAC = 0.49f

        /** Search a few integer multiples so dense weak artifacts cannot halve the period. */
        private const val MAX_PERIOD_MULTIPLIER = 3
        private const val MAX_PHASE_ANCHORS = 64

        /**
         * Pair-derived gap refinement is useful for sub-pixel period estimates, but scoring
         * every pair against every candidate is cubic. Uniformly sample the ordered pair
         * space while still scoring every candidate for each retained hypothesis.
         */
        private const val MAX_REFINED_GAP_HYPOTHESES = 128

        /** A promoted fundamental must retain most of the evidence carried by its harmonic. */
        private const val HARMONIC_MIN_QUALITY_RATIO = 0.45f
        private const val HARMONIC_MIN_STRENGTH_RATIO = 0.52f
        private const val HARMONIC_MIN_PHASE_STRENGTH_SHARE = 0.52f
        private const val HARMONIC_RATIO_TOLERANCE = 0.12f

        /**
         * A configured visible count is a soft model-selection prior, not a hard detector gate.
         * The smallest valid total (5) leaves 4.5 visible graduations after the 5% end trims.
         */
        private const val COUNT_PRIOR_MIN = 4f
        private const val COUNT_PRIOR_TIE_FRAC = 0.05f
        /** A badly mismatched configuration must fall back to signal evidence. */
        private const val COUNT_PRIOR_MAX_ERROR = 0.35f

        private const val MIN_PERSPECTIVE_ANCHORS = 6
        private const val MIN_PERSPECTIVE_SPAN_TEETH = 8
        private const val MAX_PERSPECTIVE_GAP_RATIO = 1.25f
        private const val PERSPECTIVE_MIN_QUALITY_GAIN = 0.5f
        private const val LOCAL_DOUBLET_MAX_FRAC = 0.80f
        private const val LOCAL_DOUBLET_TARGET_TOL = 0.25f
        private const val LOCAL_DOUBLET_PASSES = 2
        private const val MIN_GRID_ANCHORS = 5
        private const val MIN_GRID_SPAN_TEETH = 4

        /** Beyond this many missing ticks in one run, assume a real discontinuity. */
        private const val MAX_FILL_RUN = 4

        private const val MIN_STRONG_PEAKS = 4

        /**
         * Shorter profiles cannot carry a usable number of scale lines. Declared here
         * rather than in [AxialSampler] so this class stays free of Android types and can
         * be unit tested on the JVM.
         */
        const val MIN_SIGNAL_LEN = 120
    }

    /**
     * Result of one frame. Arrays are reused -- copy out anything you keep.
     *
     * All positions are **signal indices**, not axial distances: add
     * [AxialSampler.startOffsetPx] to convert to distance from the tube's top end.
     */
    class FrameTicks {
        var positions = FloatArray(256); internal set
        var interpolated = BooleanArray(256); internal set
        var count = 0; internal set
        var medianGapPx = 0f; internal set
        var contrast = 0f; internal set

        /**
         * Strongest wide dark trough: the ball. -1 when none qualifies.
         *
         * Same rule as the reference implementation -- highest prominence among the peaks
         * that pass the width band. No temporal reasoning here; the caller just draws it.
         */
        var ballAxialPx = -1f; internal set
        var ballProminence = 0f; internal set
        var ballWidthPx = 0f; internal set

        // Diagnostics: why a frame produced the tick set it did. Cheap to keep, and the
        // only practical way to tell "threshold too low" from "signal too weak" on device.
        var rawPeakCount = 0; internal set
        var tickPromRef = 0f; internal set
        var strongCut = 0f; internal set
        var strongCount = 0; internal set

        val hasBall get() = ballAxialPx >= 0f

        internal fun ensure(n: Int) {
            if (positions.size >= n) return
            var cap = positions.size
            while (cap < n) cap *= 2
            positions = positions.copyOf(cap)
            interpolated = interpolated.copyOf(cap)
        }
    }

    private val peaks = PeakSet()
    private var inverted = FloatArray(2048)
    private var smoothed = FloatArray(2048)
    private var scratch = FloatArray(512)
    private var strongIdx = IntArray(512)
    private var candidatePositions = IntArray(512)
    private var candidateStrengths = FloatArray(512)
    private var selectedPositions = IntArray(512)
    private var selectedStrengths = FloatArray(512)
    private var selectedOrdinals = IntArray(512)
    private var fittedGridGap = 0f
    private val modelAnchors = IntArray(MAX_PERIOD_MULTIPLIER)
    private val modelGaps = FloatArray(MAX_PERIOD_MULTIPLIER)
    private val modelQualities = FloatArray(MAX_PERIOD_MULTIPLIER)
    private val modelSupports = IntArray(MAX_PERIOD_MULTIPLIER)
    private val modelResiduals = FloatArray(MAX_PERIOD_MULTIPLIER)
    private val modelStrengths = FloatArray(MAX_PERIOD_MULTIPLIER)
    private val modelSpans = IntArray(MAX_PERIOD_MULTIPLIER)
    private val harmonicPhaseStrengths = FloatArray(MAX_PERIOD_MULTIPLIER)
    private val normalMatrix = DoubleArray(12)

    private var gridCenterOrdinal = 0f
    private var gridCenterPosition = 0f
    private var gridLinear = 0f
    private var gridQuadratic = 0f
    private var associationQuality = 0f
    private var associationMinOrdinal = 0
    private var associationMaxOrdinal = 0

    /** Span blanked out around the ball this frame, or -1 when there was no ball. */
    private var maskLo = -1
    private var maskHi = -1

    val result = FrameTicks()

    /**
     * Runs detection over `signal[0 until n]` (raw luminance, dark features = low values).
     *
     * @param tubeWidthPx kept for signature compatibility; ball thresholds are now fixed pixels
     * @param expectedVisibleCount optional visible graduation count from the user's ruler
     *   configuration. A non-positive value leaves model selection fully data-driven.
     * @return false when the frame yields no usable tick set
     */
    fun detect(
        signal: FloatArray,
        n: Int,
        tubeWidthPx: Float,
        expectedVisibleCount: Float = 0f
    ): Boolean {
        result.count = 0
        result.ballAxialPx = -1f
        result.medianGapPx = 0f
        if (n < SavitzkyGolay.WINDOW || n < MIN_SIGNAL_LEN) return false

        ensureScratch(n)

        // Invert so dark features (ticks, ball) become peaks.
        for (i in 0 until n) inverted[i] = -signal[i]

        // Contrast drives every threshold below, so it must be measured per frame.
        System.arraycopy(inverted, 0, scratch, 0, n)
        val p95 = RobustStats.percentile(scratch, n, 0.95f)
        System.arraycopy(inverted, 0, scratch, 0, n)
        val p05 = RobustStats.percentile(scratch, n, 0.05f)
        val contrast = (p95 - p05).coerceAtLeast(1f)
        result.contrast = contrast

        // --- Ball: strongest wide trough, reference-style fixed thresholds -----------
        PeakFinder.findPeaks(inverted, n, peaks)
        // find_peaks(inv_signal, prominence=BALL_PROMINENCE, width=(BALL_WIDTH_MIN_PX, BALL_WIDTH_MAX_PX))
        // reproduced manually so we can keep the scipy-compatible PeakFinder unchanged.
        var ballIdx = -1
        var ballBest = 0f
        for (p in 0 until peaks.count) {
            val prom = peaks.prominences[p]
            val w = peaks.widths[p]
            if (prom < BALL_PROMINENCE || w < BALL_WIDTH_MIN_PX || w > BALL_WIDTH_MAX_PX) continue
            if (prom > ballBest) { ballBest = prom; ballIdx = p }
        }

        // Mask ±BALL_MASK_HALF_PX around the ball so its flanks don't dominate the tick
        // prominence ranking (reference does exactly this with a fixed ±25 px window).
        maskLo = -1; maskHi = -1
        if (ballIdx >= 0) {
            result.ballAxialPx = peaks.indices[ballIdx].toFloat()
            result.ballProminence = peaks.prominences[ballIdx]
            result.ballWidthPx = peaks.widths[ballIdx]
            maskLo = (result.ballAxialPx - BALL_MASK_HALF_PX).toInt().coerceIn(0, n - 1)
            maskHi = (result.ballAxialPx + BALL_MASK_HALF_PX).toInt().coerceIn(0, n - 1)
            System.arraycopy(inverted, 0, scratch, 0, n)
            val fill = RobustStats.median(scratch, n)
            for (i in maskLo..maskHi) inverted[i] = fill
        }


        // --- Ticks ---------------------------------------------------------------------
        if (!SavitzkyGolay.smooth(inverted, n, smoothed)) return false
        PeakFinder.findPeaks(smoothed, n, peaks)
        if (peaks.count < 3) return false

        val tickPromMin = maxOf(TICK_PROM_ABS, TICK_PROM_REL * contrast)

        // Typical tick strength, measured over candidates outside the masked span. Peaks
        // inside the mask are excluded here as well as below: the flat fill reads as one
        // enormous plateau peak, and letting it into the reference statistic would defeat
        // the point of measuring robustly.
        var promCount = 0
        for (p in 0 until peaks.count) {
            val idx = peaks.indices[p]
            if (isMasked(idx)) continue
            if (isNearEdge(idx, n)) continue
            if (promCount >= scratch.size) break
            scratch[promCount++] = peaks.prominences[p]
        }
        if (promCount < MIN_STRONG_PEAKS) return false
        val tickPromRef = RobustStats.medianOfTopK(scratch, promCount, STRONG_REF_TOP_K)
        val strongCut = tickPromRef * STRONG_FRAC
        result.rawPeakCount = peaks.count
        result.tickPromRef = tickPromRef
        result.strongCut = strongCut

        // Collect peak *slots*, not positions, so width lookups stay O(1) later.
        var strongCount = 0
        for (p in 0 until peaks.count) {
            val idx = peaks.indices[p]
            if (isMasked(idx)) continue
            if (isNearEdge(idx, n)) continue
            if (peaks.prominences[p] < tickPromMin) continue
            if (peaks.prominences[p] < strongCut) continue
            if (strongCount >= strongIdx.size) break
            strongIdx[strongCount++] = p
        }
        if (strongCount < MIN_STRONG_PEAKS) return false
        result.strongCount = strongCount

        // Robust median gap: ignore diffs that span a missing tick.
        var d = 0
        for (i in 1 until strongCount) {
            scratch[d++] = (peaks.indices[strongIdx[i]] - peaks.indices[strongIdx[i - 1]]).toFloat()
        }
        if (d == 0) return false
        // `inverted` is spent by this point (smoothed holds what we still need).
        System.arraycopy(scratch, 0, inverted, 0, d)
        val rawMedian = RobustStats.median(inverted, d)
        var kept = 0
        for (i in 0 until d) if (scratch[i] < rawMedian * 1.5f) inverted[kept++] = scratch[i]
        val medianGap = if (kept > 0) RobustStats.median(inverted, kept) else rawMedian
        if (medianGap < 2f) return false

        // Drop over-wide candidates (printed numerals) now that a gap scale exists.
        val maxTickWidth = MAX_TICK_WIDTH_FRAC_OF_GAP * medianGap
        var filtered = 0
        for (i in 0 until strongCount) {
            val p = strongIdx[i]
            if (peaks.widths[p] > maxTickWidth) continue
            candidatePositions[filtered++] = peaks.indices[p]
            candidateStrengths[filtered - 1] = peaks.prominences[p]
        }
        if (filtered < MIN_STRONG_PEAKS) return false

        filtered = pruneLocalDoublets(candidatePositions, candidateStrengths, filtered, medianGap)
        if (filtered < MIN_STRONG_PEAKS) return false
        filtered = fitGlobalGrid(
            candidatePositions,
            candidateStrengths,
            filtered,
            medianGap,
            expectedVisibleCount
        )
        if (filtered < MIN_GRID_ANCHORS ||
            associationMaxOrdinal - associationMinOrdinal < MIN_GRID_SPAN_TEETH
        ) {
            return false
        }

        fillGaps(selectedPositions, selectedOrdinals, filtered, n, tickPromRef)
        result.medianGapPx = fittedGridGap
        return result.count >= MIN_STRONG_PEAKS
    }

    /**
     * Removes a candidate that sits between two unusually short gaps. This is phase-free:
     * it catches a reflection inserted between two real graduations even when perspective
     * has already moved the global comb far enough to make phase association ambiguous.
     */
    private fun pruneLocalDoublets(
        positions: IntArray,
        strengths: FloatArray,
        initialCount: Int,
        initialGap: Float
    ): Int {
        var count = initialCount
        for (pass in 0 until LOCAL_DOUBLET_PASSES) {
            if (count < 3) break
            var out = 0
            var removed = false
            var i = 0
            while (i < count) {
                if (i in 1 until count - 1) {
                    val leftGap = positions[i] - positions[i - 1]
                    val rightGap = positions[i + 1] - positions[i]
                    val outerGap = positions[i + 1] - positions[i - 1]
                    val oneGapError = abs(outerGap - initialGap) / initialGap
                    if (oneGapError <= LOCAL_DOUBLET_TARGET_TOL &&
                        leftGap < initialGap * LOCAL_DOUBLET_MAX_FRAC &&
                        rightGap < initialGap * LOCAL_DOUBLET_MAX_FRAC
                    ) {
                        removed = true
                        i++
                        continue
                    }
                }
                positions[out] = positions[i]
                strengths[out] = strengths[i]
                out++
                i++
            }
            count = out
            if (!removed) break
        }
        return count
    }

    /**
     * Finds the comb phase supported by the largest number of candidates, then performs a
     * one-to-one association between grid teeth and measured peaks. Strength only breaks
     * ties: an isolated liquid edge must not move the comb merely because it is darker than
     * a printed graduation.
     */
    private fun fitGlobalGrid(
        positions: IntArray,
        strengths: FloatArray,
        count: Int,
        initialGap: Float,
        expectedVisibleCount: Float
    ): Int {
        for (model in 0 until MAX_PERIOD_MULTIPLIER) {
            modelAnchors[model] = 0
            modelGaps[model] = initialGap * (model + 1)
            modelQualities[model] = Float.NEGATIVE_INFINITY
            modelSupports[model] = -1
            modelResiduals[model] = Float.POSITIVE_INFINITY
            modelStrengths[model] = Float.NEGATIVE_INFINITY
            modelSpans[model] = 0
        }

        fun consider(model: Int, anchor: Int, gap: Float) {
            val tolerance = gap * GRID_MATCH_FRAC
            val origin = positions[anchor].toFloat()
            var quality = 0f
            var support = 0
            var residualSum = 0f
            var strengthSum = 0f
            var selectedOrdinal = Int.MIN_VALUE
            var selectedNormalized = Float.POSITIVE_INFINITY
            var selectedQuality = 0f
            var selectedStrength = 0f
            var firstOrdinal = Int.MAX_VALUE
            var lastOrdinal = Int.MIN_VALUE
            for (candidate in 0 until count) {
                val delta = positions[candidate] - origin
                val ordinal = (delta / gap).roundToInt()
                val residual = abs(delta - ordinal * gap)
                if (residual > tolerance) continue
                val normalized = residual / tolerance
                val candidateQuality = 1f - normalized * normalized
                if (ordinal == selectedOrdinal) {
                    if (normalized < selectedNormalized ||
                        (abs(normalized - selectedNormalized) <= 1e-4f &&
                            strengths[candidate] > selectedStrength)
                    ) {
                        selectedNormalized = normalized
                        selectedQuality = candidateQuality
                        selectedStrength = strengths[candidate]
                    }
                    continue
                }
                if (selectedOrdinal != Int.MIN_VALUE) {
                    quality += selectedQuality
                    support++
                    residualSum += selectedNormalized
                    strengthSum += selectedStrength
                    if (firstOrdinal == Int.MAX_VALUE) firstOrdinal = selectedOrdinal
                    lastOrdinal = selectedOrdinal
                }
                selectedOrdinal = ordinal
                selectedNormalized = normalized
                selectedQuality = candidateQuality
                selectedStrength = strengths[candidate]
            }
            if (selectedOrdinal != Int.MIN_VALUE) {
                quality += selectedQuality
                support++
                residualSum += selectedNormalized
                strengthSum += selectedStrength
                if (firstOrdinal == Int.MAX_VALUE) firstOrdinal = selectedOrdinal
                lastOrdinal = selectedOrdinal
            }
            val span = if (support > 0) lastOrdinal - firstOrdinal + 1 else 0
            val sameQuality = abs(quality - modelQualities[model]) <= 1e-4f
            if (quality > modelQualities[model] + 1e-4f ||
                (sameQuality && support > modelSupports[model]) ||
                (sameQuality && support == modelSupports[model] &&
                    residualSum < modelResiduals[model] - 1e-4f) ||
                (sameQuality && support == modelSupports[model] &&
                    abs(residualSum - modelResiduals[model]) <= 1e-4f &&
                    strengthSum > modelStrengths[model])
            ) {
                modelAnchors[model] = anchor
                modelGaps[model] = gap
                modelQualities[model] = quality
                modelSupports[model] = support
                modelResiduals[model] = residualSum
                modelStrengths[model] = strengthSum
                modelSpans[model] = span
            }
        }

        val pairCount = count.toLong() * (count - 1L) / 2L
        val hypothesisCount = minOf(
            pairCount,
            MAX_REFINED_GAP_HYPOTHESES.toLong()
        )
        val anchorCount = minOf(count, MAX_PHASE_ANCHORS)

        // Anchor scoring is cheap and runs for every period. Pair refinement is the costly
        // phase search; when the user supplied a visible count, refine only the period whose
        // initial ordinal span is compatible with that count. Unknown-count mode retains the
        // original all-period search.
        for (model in 0 until MAX_PERIOD_MULTIPLIER) {
            val nominalGap = initialGap * (model + 1)
            for (sample in 0 until anchorCount) {
                val anchor = if (anchorCount <= 1) {
                    0
                } else {
                    sample * (count - 1) / (anchorCount - 1)
                }
                consider(model, anchor, nominalGap)
            }
        }

        val countKnown = expectedVisibleCount.isFinite() && expectedVisibleCount >= COUNT_PRIOR_MIN
        var pairModel = -1
        var bestCountError = Float.POSITIVE_INFINITY
        if (countKnown) {
            for (model in 0 until MAX_PERIOD_MULTIPLIER) {
                val error = modelCountError(model, expectedVisibleCount)
                if (error <= COUNT_PRIOR_MAX_ERROR && error < bestCountError) {
                    bestCountError = error
                    pairModel = model
                }
            }
        }

        val restrictPairSearch = countKnown && pairModel >= 0
        for (model in 0 until MAX_PERIOD_MULTIPLIER) {
            if (restrictPairSearch && model != pairModel) continue
            val nominalGap = initialGap * (model + 1)
            for (sample in 0L until hypothesisCount) {
                val pairOrdinal = if (hypothesisCount <= 1L) {
                    0L
                } else {
                    sample * (pairCount - 1L) / (hypothesisCount - 1L)
                }
                val pairCode = pairCodeForOrdinal(count, pairOrdinal)
                val left = (pairCode ushr 32).toInt()
                val right = pairCode.toInt()
                val distance = (positions[right] - positions[left]).toFloat()
                val steps = (distance / nominalGap).roundToInt()
                if (steps <= 0) continue
                val candidateGap = distance / steps
                if (candidateGap < nominalGap * 0.75f ||
                    candidateGap > nominalGap * 1.25f
                ) {
                    continue
                }
                consider(model, left, candidateGap)
            }
        }

        val countPriorCompatible = countKnown &&
            (0 until MAX_PERIOD_MULTIPLIER).any { model ->
                modelSupports[model] >= MIN_GRID_ANCHORS &&
                    modelCountError(model, expectedVisibleCount) <= COUNT_PRIOR_MAX_ERROR
            }

        var selected = 0
        for (model in 1 until MAX_PERIOD_MULTIPLIER) {
            if (preferModel(model, selected, expectedVisibleCount)) selected = model
        }
        val rawSelected = selected

        fun dominantRawPhaseShare(harmonic: Int): Float {
            for (phase in 0 until harmonic) harmonicPhaseStrengths[phase] = 0f
            val rawGap = modelGaps[rawSelected]
            val rawOrigin = positions[modelAnchors[rawSelected]].toFloat()
            setConstantGrid(rawOrigin, rawGap)
            val rawCount = associateCandidates(positions, strengths, count, GRID_MATCH_FRAC)
            var total = 0f
            for (candidate in 0 until rawCount) {
                val phase = ((selectedOrdinals[candidate] % harmonic) + harmonic) % harmonic
                harmonicPhaseStrengths[phase] += selectedStrengths[candidate]
                total += selectedStrengths[candidate]
            }
            if (total <= 0f) return 0f
            var strongest = 0f
            for (phase in 0 until harmonic) {
                if (harmonicPhaseStrengths[phase] > strongest) {
                    strongest = harmonicPhaseStrengths[phase]
                }
            }
            return strongest / total
        }

        // A known count suppresses harmonic promotion only when at least one model actually
        // fits it. A badly mismatched configuration follows the original defensive path.
        if (!countPriorCompatible) {
            for (model in 0 until MAX_PERIOD_MULTIPLIER) {
                if (modelGaps[model] <= modelGaps[rawSelected]) continue
                val ratio = modelGaps[model] / modelGaps[rawSelected]
                val harmonic = ratio.roundToInt()
                if (harmonic !in 2..MAX_PERIOD_MULTIPLIER) continue
                if (abs(ratio - harmonic) > HARMONIC_RATIO_TOLERANCE) continue
                if (modelQualities[model] < modelQualities[rawSelected] * HARMONIC_MIN_QUALITY_RATIO) continue
                if (modelStrengths[model] <
                    modelStrengths[rawSelected] * HARMONIC_MIN_STRENGTH_RATIO
                ) {
                    continue
                }
                if (dominantRawPhaseShare(harmonic) < HARMONIC_MIN_PHASE_STRENGTH_SHARE) continue
                if (modelGaps[model] > modelGaps[selected]) selected = model
            }
        }

        val bestAnchor = modelAnchors[selected]
        val bestGap = modelGaps[selected]
        val origin = positions[bestAnchor].toFloat()
        setConstantGrid(origin, bestGap)
        val constantCount = associateCandidates(positions, strengths, count)
        val constantQuality = associationQuality
        val constantCenterOrdinal = gridCenterOrdinal
        val constantCenterPosition = gridCenterPosition
        val constantLinear = gridLinear
        val constantSpan = associationMaxOrdinal - associationMinOrdinal
        val seedCount = if (constantCount >= MIN_PERSPECTIVE_ANCHORS &&
            constantSpan >= MIN_PERSPECTIVE_SPAN_TEETH
        ) {
            associateCandidates(positions, strengths, count, PERSPECTIVE_SEED_MATCH_FRAC)
        } else {
            0
        }
        if (seedCount > 0) normalizeSeedOrdinals(seedCount, bestGap)
        if (seedCount >= MIN_PERSPECTIVE_ANCHORS &&
            associationMaxOrdinal - associationMinOrdinal >= MIN_PERSPECTIVE_SPAN_TEETH &&
            fitQuadraticFromSelected(seedCount)
        ) {
            var perspectiveCount = associateCandidates(positions, strengths, count)
            if (perspectiveCount >= MIN_PERSPECTIVE_ANCHORS &&
                fitQuadraticFromSelected(perspectiveCount)
            ) {
                perspectiveCount = associateCandidates(positions, strengths, count)
            }
            val perspectiveAccepted = perspectiveCount >= constantCount &&
                associationMaxOrdinal - associationMinOrdinal >= MIN_PERSPECTIVE_SPAN_TEETH &&
                associationQuality >= constantQuality + PERSPECTIVE_MIN_QUALITY_GAIN
            if (perspectiveAccepted) {
                updateReportedGap()
                return perspectiveCount
            }
        }

        gridCenterOrdinal = constantCenterOrdinal
        gridCenterPosition = constantCenterPosition
        gridLinear = constantLinear
        gridQuadratic = 0f
        val restoredCount = associateCandidates(positions, strengths, count)
        updateReportedGap()
        return restoredCount
    }

    /** Relative error between a model's observed ordinal span and the configured count. */
    private fun modelCountError(model: Int, expectedVisibleCount: Float): Float {
        val span = modelSpans[model]
        if (span <= 0 || !expectedVisibleCount.isFinite() || expectedVisibleCount < COUNT_PRIOR_MIN) {
            return Float.POSITIVE_INFINITY
        }
        // span is inclusive ordinal coverage: [firstOrdinal, lastOrdinal] already counts
        // both endpoints, so do not add another tooth here.
        return abs(span.toFloat() - expectedVisibleCount) / expectedVisibleCount
    }

    /**
     * Uses count as a soft prior only when it has enough support. A badly framed or mistyped
     * configuration must still fall back to the signal score and let TickRuler reject it later.
     */
    private fun preferModel(candidate: Int, current: Int, expectedVisibleCount: Float): Boolean {
        val countKnown = expectedVisibleCount.isFinite() && expectedVisibleCount >= COUNT_PRIOR_MIN
        if (!countKnown) return modelQualities[candidate] > modelQualities[current]
        val candidateError = modelCountError(candidate, expectedVisibleCount)
        val currentError = modelCountError(current, expectedVisibleCount)
        val candidateUsable = modelSupports[candidate] >= MIN_GRID_ANCHORS &&
            candidateError.isFinite() && candidateError <= COUNT_PRIOR_MAX_ERROR
        val currentUsable = modelSupports[current] >= MIN_GRID_ANCHORS &&
            currentError.isFinite() && currentError <= COUNT_PRIOR_MAX_ERROR
        if (candidateUsable && !currentUsable) return true
        if (!candidateUsable && currentUsable) return false
        if (candidateUsable && currentUsable) {
            if (candidateError + COUNT_PRIOR_TIE_FRAC < currentError) return true
            if (currentError + COUNT_PRIOR_TIE_FRAC < candidateError) return false
        }

        val sameQuality = abs(modelQualities[candidate] - modelQualities[current]) <= 1e-4f
        if (modelQualities[candidate] > modelQualities[current] + 1e-4f) return true
        if (!sameQuality || modelSupports[candidate] < modelSupports[current]) return false
        if (modelSupports[candidate] > modelSupports[current]) return true
        if (modelResiduals[candidate] < modelResiduals[current] - 1e-4f) return true
        if (abs(modelResiduals[candidate] - modelResiduals[current]) > 1e-4f) return false
        return modelStrengths[candidate] > modelStrengths[current]
    }

    /** Maps a lexicographic pair ordinal directly to (left, right) without scanning pairs. */
    private fun pairCodeForOrdinal(count: Int, ordinal: Long): Long {
        var low = 0
        var high = count - 2
        while (low <= high) {
            val middle = (low + high) ushr 1
            val before = middle.toLong() * (2L * count - middle - 1L) / 2L
            val after = (middle + 1L) * (2L * count - middle - 2L) / 2L
            if (ordinal < before) {
                high = middle - 1
            } else if (ordinal >= after) {
                low = middle + 1
            } else {
                val right = middle + 1 + (ordinal - before).toInt()
                return (middle.toLong() shl 32) + right.toLong()
            }
        }
        error("pair ordinal out of range: $ordinal")
    }

    private fun setConstantGrid(origin: Float, gap: Float) {
        gridCenterOrdinal = 0f
        gridCenterPosition = origin
        gridLinear = gap
        gridQuadratic = 0f
    }

    /** Corrects phase-drift ordinal jumps using local spacing before fitting perspective. */
    private fun normalizeSeedOrdinals(count: Int, referenceGap: Float) {
        for (i in 1 until count) {
            val observedGap = selectedPositions[i] - selectedPositions[i - 1]
            val steps = (observedGap / referenceGap).roundToInt().coerceAtLeast(1)
            selectedOrdinals[i] = selectedOrdinals[i - 1] + steps
        }
    }

    private fun gridPosition(ordinal: Int): Float {
        val z = ordinal - gridCenterOrdinal
        return gridCenterPosition + gridLinear * z + gridQuadratic * z * z
    }

    private fun localGridGap(ordinal: Int): Float {
        val z = ordinal - gridCenterOrdinal
        return gridLinear + gridQuadratic * (2f * z + 1f)
    }

    private fun nearestGridOrdinal(position: Float): Int {
        val approximate = ((position - gridCenterPosition) / gridLinear + gridCenterOrdinal)
            .roundToInt()
        var best = approximate
        var bestResidual = abs(position - gridPosition(best))
        for (ordinal in approximate - 3..approximate + 3) {
            val residual = abs(position - gridPosition(ordinal))
            if (residual < bestResidual) {
                best = ordinal
                bestResidual = residual
            }
        }
        return best
    }

    private fun associateCandidates(
        positions: IntArray,
        strengths: FloatArray,
        count: Int,
        toleranceFrac: Float = GRID_MATCH_FRAC
    ): Int {
        var out = 0
        var lastOrdinal = Int.MIN_VALUE
        var lastNormalizedResidual = Float.POSITIVE_INFINITY
        var lastQuality = 0f
        associationQuality = 0f

        for (candidate in 0 until count) {
            val position = positions[candidate].toFloat()
            val ordinal = nearestGridOrdinal(position)
            val localGap = localGridGap(ordinal).coerceAtLeast(1f)
            val tolerance = localGap * toleranceFrac
            val residual = abs(position - gridPosition(ordinal))
            if (residual > tolerance) continue
            val normalized = residual / tolerance
            val quality = 1f - normalized * normalized

            if (ordinal == lastOrdinal) {
                val previous = out - 1
                if (normalized < lastNormalizedResidual ||
                    (abs(normalized - lastNormalizedResidual) <= 1e-4f &&
                        strengths[candidate] > selectedStrengths[previous])
                ) {
                    associationQuality += quality - lastQuality
                    selectedPositions[previous] = positions[candidate]
                    selectedStrengths[previous] = strengths[candidate]
                    lastNormalizedResidual = normalized
                    lastQuality = quality
                }
                continue
            }

            selectedPositions[out] = positions[candidate]
            selectedStrengths[out] = strengths[candidate]
            selectedOrdinals[out] = ordinal
            associationQuality += quality
            out++
            lastOrdinal = ordinal
            lastNormalizedResidual = normalized
            lastQuality = quality
        }

        if (out > 0) {
            associationMinOrdinal = selectedOrdinals[0]
            associationMaxOrdinal = selectedOrdinals[out - 1]
        }
        return out
    }

    private fun fitQuadraticFromSelected(count: Int): Boolean {
        if (count < MIN_PERSPECTIVE_ANCHORS) return false
        val minOrdinal = selectedOrdinals[0]
        val maxOrdinal = selectedOrdinals[count - 1]
        if (maxOrdinal - minOrdinal < MIN_PERSPECTIVE_SPAN_TEETH) return false
        val center = (minOrdinal + maxOrdinal) * 0.5

        normalMatrix.fill(0.0)
        for (i in 0 until count) {
            val z = selectedOrdinals[i] - center
            val z2 = z * z
            val position = selectedPositions[i].toDouble()
            normalMatrix[0] += 1.0
            normalMatrix[1] += z
            normalMatrix[2] += z2
            normalMatrix[3] += position
            normalMatrix[4] += z
            normalMatrix[5] += z2
            normalMatrix[6] += z2 * z
            normalMatrix[7] += position * z
            normalMatrix[8] += z2
            normalMatrix[9] += z2 * z
            normalMatrix[10] += z2 * z2
            normalMatrix[11] += position * z2
        }

        for (column in 0..2) {
            var pivot = column
            for (row in column + 1..2) {
                if (kotlin.math.abs(normalMatrix[row * 4 + column]) >
                    kotlin.math.abs(normalMatrix[pivot * 4 + column])
                ) {
                    pivot = row
                }
            }
            if (kotlin.math.abs(normalMatrix[pivot * 4 + column]) < 1e-9) return false
            if (pivot != column) {
                for (entry in column..3) {
                    val a = column * 4 + entry
                    val b = pivot * 4 + entry
                    val swap = normalMatrix[a]
                    normalMatrix[a] = normalMatrix[b]
                    normalMatrix[b] = swap
                }
            }
            val pivotValue = normalMatrix[column * 4 + column]
            for (entry in column..3) normalMatrix[column * 4 + entry] /= pivotValue
            for (row in 0..2) {
                if (row == column) continue
                val factor = normalMatrix[row * 4 + column]
                for (entry in column..3) {
                    normalMatrix[row * 4 + entry] -= factor * normalMatrix[column * 4 + entry]
                }
            }
        }

        val position = normalMatrix[3].toFloat()
        val linear = normalMatrix[7].toFloat()
        val quadratic = normalMatrix[11].toFloat()
        val minZ = minOrdinal - center
        val maxZ = maxOrdinal - center
        val firstGap = linear + quadratic * (2f * minZ + 1f)
        val lastGap = linear + quadratic * (2f * maxZ - 1f)
        val smallerGap = minOf(firstGap, lastGap)
        val largerGap = maxOf(firstGap, lastGap)
        if (!position.isFinite() || !linear.isFinite() || !quadratic.isFinite()) return false
        if (smallerGap < 2f || largerGap / smallerGap > MAX_PERSPECTIVE_GAP_RATIO) return false

        gridCenterOrdinal = center.toFloat()
        gridCenterPosition = position
        gridLinear = linear
        gridQuadratic = quadratic
        return true
    }

    private fun updateReportedGap() {
        val middleOrdinal = Math.floorDiv(associationMinOrdinal + associationMaxOrdinal, 2)
        fittedGridGap = localGridGap(middleOrdinal)
    }

    /**
     * Walks the strong peaks and reconstructs ticks the detector missed, following the
     * reference: a gap of k median-gaps means k-1 lines were dropped, so search for a
     * local maximum near each expected position. Reconstructed entries are flagged so the
     * shift estimator can ignore them -- matching against a position we invented would let
     * a drift reinforce itself.
     */
    private fun fillGaps(
        src: IntArray,
        ordinals: IntArray,
        count: Int,
        n: Int,
        tickPromRef: Float
    ) {
        result.ensure(count * 2 + 8)
        var out = 0

        fun push(pos: Float, interp: Boolean) {
            result.ensure(out + 1)
            result.positions[out] = pos
            result.interpolated[out] = interp
            out++
        }

        push(src[0].toFloat(), false)

        for (i in 1 until count) {
            val p = src[i]
            val ordinalSpan = ordinals[i] - ordinals[i - 1]

            if (ordinalSpan in 2..MAX_FILL_RUN) {
                for (ordinal in ordinals[i - 1] + 1 until ordinals[i]) {
                    val expected = gridPosition(ordinal)
                    val localGap = localGridGap(ordinal).coerceAtLeast(1f)
                    val searchWin = maxOf(2f, localGap * FILL_SEARCH_FRAC).toInt()
                    val lo = (expected - searchWin).toInt().coerceIn(0, n - 1)
                    val hi = (expected + searchWin).toInt().coerceIn(0, n - 1)
                    if (lo < hi) {
                        var best = lo
                        var worst = smoothed[lo]
                        for (q in lo..hi) {
                            if (smoothed[q] > smoothed[best]) best = q
                            if (smoothed[q] < worst) worst = smoothed[q]
                        }
                        // Only snap onto a feature that actually looks like a tick.
                        // Otherwise the "local maximum" is just noise and would place
                        // the line up to searchWin away from where the grid says it is.
                        val standsOut = smoothed[best] - worst >= tickPromRef * FILL_SNAP_MIN_FRAC
                        push(if (standsOut) best.toFloat() else expected, true)
                    } else {
                        push(expected, true)
                    }
                }
            }
            push(p.toFloat(), false)
        }
        result.count = out
    }

    /**
     * Peaks this close to either end of the profile are not trustworthy, for two
     * independent reasons:
     *
     *  - Savitzky-Golay smooths the edge samples with a polynomial fit rather than the
     *    sliding kernel, so a peak there has been shaped by a different estimator.
     *  - The prominence walk runs off the end of the array and takes the boundary as a
     *    base instead of a real trough, which distorts the value either way.
     *
     * They are the ticks nearest the tube's end caps, so they are also the ones most
     * affected by the meniscus and by the detection box breathing. Dropping them here
     * costs one graduation at each end and removes a whole class of flicker.
     */
    private fun isNearEdge(idx: Int, n: Int): Boolean =
        idx < SavitzkyGolay.WINDOW || idx >= n - SavitzkyGolay.WINDOW

    /**
     * Whether [idx] falls inside a masked wide feature.
     *
     * The flat median fill left behind by masking reads to [PeakFinder] as one enormous
     * plateau peak whenever both its sides happen to be lower. The reference implementation
     * lets those through and survives only because the strength filter usually discards
     * them; relying on that is not something to inherit.
     */
    private fun isMasked(idx: Int): Boolean = maskLo >= 0 && idx in maskLo..maskHi

    private fun ensureScratch(n: Int) {
        if (inverted.size < n) {
            var cap = inverted.size
            while (cap < n) cap *= 2
            inverted = FloatArray(cap)
            smoothed = FloatArray(cap)
        }
        if (scratch.size < n) {
            var cap = scratch.size
            while (cap < n) cap *= 2
            scratch = FloatArray(cap)
            strongIdx = IntArray(cap)
            candidatePositions = IntArray(cap)
            candidateStrengths = FloatArray(cap)
            selectedPositions = IntArray(cap)
            selectedStrengths = FloatArray(cap)
            selectedOrdinals = IntArray(cap)
        }
    }
}
