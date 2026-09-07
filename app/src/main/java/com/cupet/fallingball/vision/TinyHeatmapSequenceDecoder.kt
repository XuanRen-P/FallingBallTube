package com.cupet.fallingball.vision

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

data class TinyHeatmapSequenceCandidate(
    val sourceX: Float,
    val sourceY: Float,
    val logit: Float
)

data class TinyHeatmapSequenceFrame(
    val sourceFrameIndex: Int,
    val candidates: List<TinyHeatmapSequenceCandidate>
)

data class TinyHeatmapSequenceResult(
    val selectedCandidate: IntArray,
    val predictedPresent: BooleanArray,
    val expectedVerticalVelocity: Float
)

/** Exact Kotlin port of the frozen PC whole-sequence candidate and presence decoders. */
object TinyHeatmapSequenceDecoder {
    const val CANDIDATE_THRESHOLD = 0.08f
    const val CANDIDATE_TRANSITION_WEIGHT = 0.1f
    const val PRESENCE_TRANSITION_PENALTY = 2f
    const val MAX_FRAME_GAP = 20
    const val MAX_MOTION_RESIDUAL = 4f
    const val HORIZONTAL_VELOCITY_SCALE = 1f
    const val VERTICAL_VELOCITY_SCALE = 4f

    private const val VELOCITY_MINIMUM_PROBABILITY = 0.8f
    private const val VELOCITY_MAX_HORIZONTAL_SPEED = 3f
    private const val VELOCITY_MIN_VERTICAL_SPEED = -3f
    private const val VELOCITY_MAX_VERTICAL_SPEED = 10f
    private const val INVALID_TRANSITION_PENALTY = 1_000_000f

    val fallbackVelocities = mapOf(
        1.0f to 0.45f,
        2.5f to 0.55f,
        3.0f to 1.2f,
        4.0f to 3.2f,
        5.0f to 4.6f
    )

    /** Converts optional measurement metadata without making final detection depend on it. */
    internal fun diameterMmOrNeutral(sphereRadiusMm: Double?): Float =
        sphereRadiusMm?.times(2.0)?.toFloat() ?: 0f

    fun decode(frames: List<TinyHeatmapSequenceFrame>, diameterMm: Float): TinyHeatmapSequenceResult {
        require(frames.isNotEmpty()) { "A sequence must contain at least one frame" }
        val candidateCount = frames.first().candidates.size
        require(candidateCount > 0) { "Every frame must contain candidates" }
        require(frames.all { it.candidates.size == candidateCount }) {
            "Every frame must have the same candidate count"
        }
        // Diameter is measurement metadata, not a detector gate. Keep the frozen PC
        // fallback where it exists; otherwise let the same-frame velocity estimate run
        // with a neutral fallback instead of refusing to decode the sequence.
        val fallback = fallbackVelocities[diameterMm] ?: 0f
        val expectedVelocity = estimateVideoVelocity(frames, fallback)
        val selected = decodeCandidateSequences(frames, expectedVelocity)
        val present = decodePresence(frames, selected, expectedVelocity)
        return TinyHeatmapSequenceResult(selected, present, expectedVelocity)
    }

    internal fun estimateVideoVelocity(
        frames: List<TinyHeatmapSequenceFrame>,
        fallbackVelocity: Float
    ): Float {
        val velocities = ArrayList<Float>()
        for (index in 1 until frames.size) {
            val previous = frames[index - 1]
            val current = frames[index]
            val frameGap = current.sourceFrameIndex - previous.sourceFrameIndex
            if (frameGap <= 0 || frameGap > MAX_FRAME_GAP) continue
            val previousCandidate = previous.candidates[maxLogitIndex(previous.candidates)]
            val currentCandidate = current.candidates[maxLogitIndex(current.candidates)]
            if (sigmoid(previousCandidate.logit) < VELOCITY_MINIMUM_PROBABILITY ||
                sigmoid(currentCandidate.logit) < VELOCITY_MINIMUM_PROBABILITY
            ) {
                continue
            }
            val horizontalSpeed =
                (currentCandidate.sourceX - previousCandidate.sourceX) / frameGap
            val verticalSpeed =
                (currentCandidate.sourceY - previousCandidate.sourceY) / frameGap
            if (!horizontalSpeed.isFinite() || !verticalSpeed.isFinite()) continue
            if (abs(horizontalSpeed) >= VELOCITY_MAX_HORIZONTAL_SPEED) continue
            if (verticalSpeed <= VELOCITY_MIN_VERTICAL_SPEED ||
                verticalSpeed >= VELOCITY_MAX_VERTICAL_SPEED
            ) {
                continue
            }
            velocities.add(verticalSpeed)
        }
        if (velocities.isEmpty()) return fallbackVelocity
        velocities.sort()
        val middle = velocities.size / 2
        return if (velocities.size % 2 == 0) {
            (velocities[middle - 1] + velocities[middle]) * 0.5f
        } else {
            velocities[middle]
        }
    }

    internal fun decodeCandidateSequences(
        frames: List<TinyHeatmapSequenceFrame>,
        expectedVerticalVelocity: Float
    ): IntArray {
        val selected = IntArray(frames.size)
        for ((start, end) in segments(frames)) {
            decodeCandidateSegment(
                frames = frames,
                start = start,
                end = end,
                expectedVerticalVelocity = expectedVerticalVelocity,
                destination = selected
            )
        }
        return selected
    }

    private fun decodeCandidateSegment(
        frames: List<TinyHeatmapSequenceFrame>,
        start: Int,
        end: Int,
        expectedVerticalVelocity: Float,
        destination: IntArray
    ) {
        val candidateCount = frames[start].candidates.size
        var dynamicScore = FloatArray(candidateCount) { candidate ->
            logSigmoid(frames[start].candidates[candidate].logit)
        }
        val backPointers = ArrayList<IntArray>(end - start - 1)
        for (frameIndex in start + 1 until end) {
            val previous = frames[frameIndex - 1]
            val current = frames[frameIndex]
            val frameGap = maxOf(1, current.sourceFrameIndex - previous.sourceFrameIndex)
            val nextScore = FloatArray(candidateCount)
            val backPointer = IntArray(candidateCount)
            for (currentCandidateIndex in 0 until candidateCount) {
                val currentCandidate = current.candidates[currentCandidateIndex]
                var bestPrevious = 0
                var bestTransition = Float.NEGATIVE_INFINITY
                for (previousCandidateIndex in 0 until candidateCount) {
                    val previousCandidate = previous.candidates[previousCandidateIndex]
                    val horizontalVelocity =
                        (currentCandidate.sourceX - previousCandidate.sourceX) / frameGap
                    val verticalVelocity =
                        (currentCandidate.sourceY - previousCandidate.sourceY) / frameGap
                    var penalty = CANDIDATE_TRANSITION_WEIGHT * (
                        square(horizontalVelocity / HORIZONTAL_VELOCITY_SCALE) +
                            square(
                                (verticalVelocity - expectedVerticalVelocity) /
                                    VERTICAL_VELOCITY_SCALE
                            )
                        )
                    if (!penalty.isFinite()) penalty = INVALID_TRANSITION_PENALTY
                    val transition = dynamicScore[previousCandidateIndex] - penalty
                    if (transition > bestTransition) {
                        bestTransition = transition
                        bestPrevious = previousCandidateIndex
                    }
                }
                backPointer[currentCandidateIndex] = bestPrevious
                nextScore[currentCandidateIndex] =
                    logSigmoid(currentCandidate.logit) + bestTransition
            }
            dynamicScore = nextScore
            backPointers.add(backPointer)
        }
        var selected = maxIndex(dynamicScore)
        destination[end - 1] = selected
        for (frameIndex in end - 2 downTo start) {
            selected = backPointers[frameIndex - start][selected]
            destination[frameIndex] = selected
        }
    }

    internal fun decodePresence(
        frames: List<TinyHeatmapSequenceFrame>,
        selectedCandidate: IntArray,
        expectedVerticalVelocity: Float
    ): BooleanArray {
        require(selectedCandidate.size == frames.size)
        val predicted = BooleanArray(frames.size)
        val thresholdLogit = ln(
            CANDIDATE_THRESHOLD.toDouble() / (1.0 - CANDIDATE_THRESHOLD)
        ).toFloat()
        for ((start, end) in segments(frames)) {
            val backPointers = ArrayList<IntArray>(end - start - 1)
            var dynamicScore = presenceEmission(
                frames[start].candidates[selectedCandidate[start]].logit,
                thresholdLogit
            )
            for (frameIndex in start + 1 until end) {
                val nextScore = FloatArray(2)
                val backPointer = IntArray(2)
                val emission = presenceEmission(
                    frames[frameIndex].candidates[selectedCandidate[frameIndex]].logit,
                    thresholdLogit
                )
                for (currentState in 0..1) {
                    var bestPrevious = 0
                    var bestTransition = Float.NEGATIVE_INFINITY
                    for (previousState in 0..1) {
                        val penalty = if (currentState == previousState) {
                            0f
                        } else {
                            PRESENCE_TRANSITION_PENALTY
                        }
                        val transition = dynamicScore[previousState] - penalty
                        if (transition > bestTransition) {
                            bestTransition = transition
                            bestPrevious = previousState
                        }
                    }
                    backPointer[currentState] = bestPrevious
                    nextScore[currentState] = emission[currentState] + bestTransition
                }
                dynamicScore = nextScore
                backPointers.add(backPointer)
            }
            val states = IntArray(end - start)
            states[states.lastIndex] = maxIndex(dynamicScore)
            for (localIndex in states.lastIndex - 1 downTo 0) {
                states[localIndex] = backPointers[localIndex][states[localIndex + 1]]
            }
            applyMotionConsistency(
                frames,
                selectedCandidate,
                start,
                end,
                expectedVerticalVelocity,
                states
            )
            for (localIndex in states.indices) {
                predicted[start + localIndex] = states[localIndex] != 0
            }
        }
        return predicted
    }

    private fun applyMotionConsistency(
        frames: List<TinyHeatmapSequenceFrame>,
        selectedCandidate: IntArray,
        start: Int,
        end: Int,
        expectedVerticalVelocity: Float,
        states: IntArray
    ) {
        val residuals = FloatArray(end - start) { Float.POSITIVE_INFINITY }
        for (frameIndex in start + 1 until end) {
            val previous = frames[frameIndex - 1]
                .candidates[selectedCandidate[frameIndex - 1]]
            val current = frames[frameIndex].candidates[selectedCandidate[frameIndex]]
            val frameGap = frames[frameIndex].sourceFrameIndex -
                frames[frameIndex - 1].sourceFrameIndex
            val horizontalVelocity = (current.sourceX - previous.sourceX) / frameGap
            val verticalVelocity = (current.sourceY - previous.sourceY) / frameGap
            val residual = sqrt(
                square(horizontalVelocity / HORIZONTAL_VELOCITY_SCALE) +
                    square(
                        (verticalVelocity - expectedVerticalVelocity) /
                            VERTICAL_VELOCITY_SCALE
                    )
            )
            if (!residual.isFinite()) continue
            val previousLocal = frameIndex - start - 1
            val currentLocal = frameIndex - start
            residuals[previousLocal] = minOf(residuals[previousLocal], residual)
            residuals[currentLocal] = minOf(residuals[currentLocal], residual)
        }
        for (index in states.indices) {
            if (residuals[index] > MAX_MOTION_RESIDUAL) states[index] = 0
        }
    }

    private fun segments(frames: List<TinyHeatmapSequenceFrame>): List<Pair<Int, Int>> {
        val result = ArrayList<Pair<Int, Int>>()
        var start = 0
        for (index in 1 until frames.size) {
            if (frames[index].sourceFrameIndex - frames[index - 1].sourceFrameIndex >
                MAX_FRAME_GAP
            ) {
                result.add(start to index)
                start = index
            }
        }
        result.add(start to frames.size)
        return result
    }

    private fun presenceEmission(logit: Float, thresholdLogit: Float): FloatArray {
        val centered = logit - thresholdLogit
        return floatArrayOf(logSigmoid(-centered), logSigmoid(centered))
    }

    private fun maxLogitIndex(candidates: List<TinyHeatmapSequenceCandidate>): Int {
        var bestIndex = 0
        var bestLogit = candidates[0].logit
        for (index in 1 until candidates.size) {
            if (candidates[index].logit > bestLogit) {
                bestLogit = candidates[index].logit
                bestIndex = index
            }
        }
        return bestIndex
    }

    private fun maxIndex(values: FloatArray): Int {
        var bestIndex = 0
        var bestValue = values[0]
        for (index in 1 until values.size) {
            if (values[index] > bestValue) {
                bestValue = values[index]
                bestIndex = index
            }
        }
        return bestIndex
    }

    private fun square(value: Float): Float = value * value

    private fun sigmoid(value: Float): Float =
        (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()

    private fun logSigmoid(value: Float): Float = when {
        value >= 0f -> -kotlin.math.ln1p(exp(-value.toDouble())).toFloat()
        else -> value - kotlin.math.ln1p(exp(value.toDouble())).toFloat()
    }
}
