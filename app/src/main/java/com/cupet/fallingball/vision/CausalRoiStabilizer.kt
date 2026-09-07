package com.cupet.fallingball.vision

import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.roundToInt
import kotlin.math.sqrt

data class RoiStabilizationTransform(
    val sourceToStabilizedX: Float,
    val sourceToStabilizedY: Float,
    val score: Float
) {
    fun stabilizedToSourceX(x: Float): Float = x - sourceToStabilizedX
    fun stabilizedToSourceY(y: Float): Float = y - sourceToStabilizedY
}

/**
 * Causal residual translation alignment for canonical tube ROIs.
 *
 * The PC training pipeline aligns each raw ROI to its first-frame coordinates. This small
 * integer NCC search mirrors that behavior without requiring OpenCV on Android. It compares
 * only the previous raw frame, so it never uses future information.
 */
class CausalRoiStabilizer(
    private val width: Int = TubeRoiSampler.OUTPUT_WIDTH,
    private val height: Int = TubeRoiSampler.OUTPUT_HEIGHT,
    private val maxFrameShiftPx: Int = 6,
    // Sparse NCC drifts on the real 004 ROI beyond 16 px; PC's 32 px bound belongs to its
    // phase-correlation estimator and is not numerically interchangeable with this one.
    private val maxAccumulatedShiftPx: Float = 16f
) {
    private val pixelCount = width * height
    private val correlationCropLeft = (width * 0.12f).roundToInt()
    private val correlationCropRight = (width * 0.88f).roundToInt()
    private val correlationCropTop = (height * 0.05f).roundToInt()
    private val correlationCropBottom = (height * 0.95f).roundToInt()
    private val correlationStepX = (width / 40).coerceAtLeast(1)
    private val correlationStepY = (height / 100).coerceAtLeast(1)
    /** The default 152x1024 ROI has a 16 px safety margin around the sparse NCC crop. */
    private val correlationWindowAlwaysInBounds =
        correlationCropLeft >= maxAccumulatedShiftPx.roundToInt() &&
            width - correlationCropRight >= maxAccumulatedShiftPx.roundToInt() &&
            correlationCropTop >= maxAccumulatedShiftPx.roundToInt() &&
            height - correlationCropBottom >= maxAccumulatedShiftPx.roundToInt()
    private var previousRaw: ByteArray? = null
    private val currentRaw = ByteArray(pixelCount)
    private var translationX = 0f
    private var translationY = 0f
    private var lastTimestampNs = 0L
    private var cachedPreviousCount = 0
    private var cachedPreviousSum = 0.0
    private var cachedPreviousSumSquares = 0.0

    init {
        require(width > 2 && height > 2)
        require(maxFrameShiftPx >= 0)
        require(maxAccumulatedShiftPx > 0f)
    }

    fun reset() {
        previousRaw = null
        translationX = 0f
        translationY = 0f
        lastTimestampNs = 0L
    }

    fun observe(
        input: ByteBuffer,
        output: ByteBuffer,
        timestampNs: Long = 0L
    ): RoiStabilizationTransform {
        require(input.remaining() >= pixelCount) { "Expected $pixelCount ROI bytes" }
        require(output.capacity() >= pixelCount) { "Output requires $pixelCount bytes" }
        input.duplicate().get(currentRaw)

        var score = 1f
        previousRaw?.let { previous ->
            val estimate = estimateShift(
                previous,
                currentRaw,
                searchRadius(timestampNs)
            )
            score = estimate.score
            if (estimate.isReliable && estimate.score >= MIN_CORRELATION) {
                translationX = (translationX - estimate.dx)
                    .coerceIn(-maxAccumulatedShiftPx, maxAccumulatedShiftPx)
                translationY = (translationY - estimate.dy)
                    .coerceIn(-maxAccumulatedShiftPx, maxAccumulatedShiftPx)
            }
        }

        output.clear()
        val integerX = translationX.roundToInt()
        val integerY = translationY.roundToInt()
        if (translationX == integerX.toFloat() && translationY == integerY.toFloat()) {
            // The residual search only produces integer translations. In that common case
            // bilinear sampling has zero fractional weights, so a clamped row/column copy is
            // exactly equivalent and avoids 155k floor/coerce/function calls per frame.
            for (y in 0 until height) {
                val sourceY = (y - integerY).coerceIn(0, height - 1)
                val rowOffset = sourceY * width
                for (x in 0 until width) {
                    val sourceX = (x - integerX).coerceIn(0, width - 1)
                    output.put(currentRaw[rowOffset + sourceX])
                }
            }
        } else {
            for (y in 0 until height) {
                val sourceY = y - translationY
                for (x in 0 until width) {
                    val sourceX = x - translationX
                    output.put(sampleBilinear(currentRaw, sourceX, sourceY).toByte())
                }
            }
        }
        output.flip()
        if (previousRaw == null) previousRaw = ByteArray(pixelCount)
        currentRaw.copyInto(requireNotNull(previousRaw))
        if (timestampNs > 0L) lastTimestampNs = timestampNs
        return RoiStabilizationTransform(translationX, translationY, score)
    }

    private data class ShiftEstimate(
        val dx: Float,
        val dy: Float,
        val score: Float,
        val isReliable: Boolean
    )

    private fun estimateShift(
        previous: ByteArray,
        current: ByteArray,
        searchRadiusPx: Int
    ): ShiftEstimate {
        val maximumRadius = maxAccumulatedShiftPx.roundToInt()
        val guardRadius = (searchRadiusPx + OUT_OF_RANGE_GUARD_PX)
            .coerceAtMost(maximumRadius)
        var bestInnerDx = 0
        var bestInnerDy = 0
        var bestInnerScore = Float.NEGATIVE_INFINITY
        var bestOuterScore = Float.NEGATIVE_INFINITY
        if (correlationWindowAlwaysInBounds) {
            cachedPreviousCount = 0
            cachedPreviousSum = 0.0
            cachedPreviousSumSquares = 0.0
            var y = correlationCropTop
            while (y < correlationCropBottom) {
                var x = correlationCropLeft
                while (x < correlationCropRight) {
                    val value = previous[y * width + x].toInt() and 0xff
                    cachedPreviousSum += value
                    cachedPreviousSumSquares += value.toDouble() * value
                    cachedPreviousCount += 1
                    x += correlationStepX
                }
                y += correlationStepY
            }
        }
        for (dy in -guardRadius..guardRadius) {
            for (dx in -guardRadius..guardRadius) {
                val score = correlation(previous, current, dx, dy)
                if (abs(dx) <= searchRadiusPx && abs(dy) <= searchRadiusPx) {
                    if (score > bestInnerScore) {
                        bestInnerScore = score
                        bestInnerDx = dx
                        bestInnerDy = dy
                    }
                } else if (score > bestOuterScore) {
                    bestOuterScore = score
                }
            }
        }
        val hasSentinelRing = guardRadius > searchRadiusPx
        val innerPeakTouchesUnverifiedBoundary = !hasSentinelRing &&
            (abs(bestInnerDx) == searchRadiusPx || abs(bestInnerDy) == searchRadiusPx)
        val strongerPeakOutsideSearch = hasSentinelRing && bestOuterScore > bestInnerScore
        return ShiftEstimate(
            bestInnerDx.toFloat(),
            bestInnerDy.toFloat(),
            bestInnerScore,
            isReliable = !innerPeakTouchesUnverifiedBoundary && !strongerPeakOutsideSearch
        )
    }

    private fun searchRadius(timestampNs: Long): Int {
        if (timestampNs <= 0L || lastTimestampNs <= 0L || timestampNs <= lastTimestampNs) {
            return maxFrameShiftPx
        }
        val periods = ceil((timestampNs - lastTimestampNs).toDouble() / NOMINAL_FRAME_PERIOD_NS)
            .toInt()
            .coerceAtLeast(1)
        return (maxFrameShiftPx * periods)
            .coerceAtMost(maxAccumulatedShiftPx.roundToInt())
    }

    /** Compare previous(x,y) with current(x+dx,y+dy). */
    private fun correlation(previous: ByteArray, current: ByteArray, dx: Int, dy: Int): Float {
        if (correlationWindowAlwaysInBounds) {
            var sumB = 0.0
            var sumBB = 0.0
            var sumAB = 0.0
            var y = correlationCropTop
            while (y < correlationCropBottom) {
                val previousRow = y * width
                val currentRow = (y + dy) * width
                var x = correlationCropLeft
                while (x < correlationCropRight) {
                    val a = previous[previousRow + x].toInt() and 0xff
                    val b = current[currentRow + x + dx].toInt() and 0xff
                    sumB += b
                    sumBB += b.toDouble() * b
                    sumAB += a.toDouble() * b
                    x += correlationStepX
                }
                y += correlationStepY
            }
            return normalizedCorrelation(
                cachedPreviousCount,
                cachedPreviousSum,
                cachedPreviousSumSquares,
                sumB,
                sumBB,
                sumAB
            )
        }

        var count = 0
        var sumA = 0.0
        var sumB = 0.0
        var sumAA = 0.0
        var sumBB = 0.0
        var sumAB = 0.0
        var y = correlationCropTop
        while (y < correlationCropBottom) {
            val currentY = y + dy
            if (currentY in 0 until height) {
                var x = correlationCropLeft
                while (x < correlationCropRight) {
                    val currentX = x + dx
                    if (currentX in 0 until width) {
                        val a = previous[y * width + x].toInt() and 0xff
                        val b = current[currentY * width + currentX].toInt() and 0xff
                        sumA += a
                        sumB += b
                        sumAA += a.toDouble() * a
                        sumBB += b.toDouble() * b
                        sumAB += a.toDouble() * b
                        count += 1
                    }
                    x += correlationStepX
                }
            }
            y += correlationStepY
        }
        return normalizedCorrelation(count, sumA, sumAA, sumB, sumBB, sumAB)
    }

    private fun normalizedCorrelation(
        count: Int,
        sumA: Double,
        sumAA: Double,
        sumB: Double,
        sumBB: Double,
        sumAB: Double
    ): Float {
        if (count < 16) return Float.NEGATIVE_INFINITY
        val covariance = sumAB - sumA * sumB / count
        val varianceA = sumAA - sumA * sumA / count
        val varianceB = sumBB - sumB * sumB / count
        val denominator = sqrt((varianceA * varianceB).coerceAtLeast(0.0))
        return if (denominator <= 1e-6) Float.NEGATIVE_INFINITY
        else (covariance / denominator).toFloat()
    }

    private fun sampleBilinear(source: ByteArray, x: Float, y: Float): Int {
        val x0 = kotlin.math.floor(x).toInt().coerceIn(0, width - 1)
        val y0 = kotlin.math.floor(y).toInt().coerceIn(0, height - 1)
        val x1 = (x0 + 1).coerceAtMost(width - 1)
        val y1 = (y0 + 1).coerceAtMost(height - 1)
        val fx = (x - x0).coerceIn(0f, 1f)
        val fy = (y - y0).coerceIn(0f, 1f)
        fun value(xx: Int, yy: Int) = source[yy * width + xx].toInt() and 0xff
        val top = value(x0, y0) * (1f - fx) + value(x1, y0) * fx
        val bottom = value(x0, y1) * (1f - fx) + value(x1, y1) * fx
        return (top * (1f - fy) + bottom * fy).roundToInt().coerceIn(0, 255)
    }

    companion object {
        private const val MIN_CORRELATION = 0.10f
        private const val NOMINAL_FRAME_PERIOD_NS = 33_333_333.0
        private const val OUT_OF_RANGE_GUARD_PX = 2
    }
}
