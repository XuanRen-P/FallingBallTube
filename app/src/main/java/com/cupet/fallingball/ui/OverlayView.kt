package com.cupet.fallingball.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import com.cupet.fallingball.vision.DetectionRoi
import com.cupet.fallingball.vision.ScaleReading
import com.cupet.fallingball.vision.TrackMode
import com.cupet.fallingball.vision.TubeAxis
import com.cupet.fallingball.vision.TubeTrackResult
import com.cupet.fallingball.vision.WallContactState
import kotlin.math.max

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val guidePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x66FFFFFF.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF2196F3.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 6f
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
    }
    private val boxSearchPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF64B5F6.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 5f
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x332196F3
        style = Paint.Style.FILL
    }
    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 5f
        strokeCap = Paint.Cap.ROUND
    }
    private val tickPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFF3B30.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }
    // Translucent fill, not a hairline stroke: the tick marks are drawn wider than the
    // band and completely hid a 2px 40%-alpha outline on a live camera image.
    private val bandFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x33FFCC00
        style = Paint.Style.FILL
    }
    private val bandEdgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xCCFFCC00.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }
    private val ballFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x6600E5A8
        style = Paint.Style.FILL
    }
    private val ballCrossPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF00E5A8.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 3.5f
        strokeCap = Paint.Cap.ROUND
    }
    private val legacyBallLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFB300.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val wallContactPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFF5252.toInt()
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    // Preallocated so onDraw stays allocation-free: at 30 fps with ~100 ticks, per-frame
    // allocation is enough garbage to cause visible jank.
    private val bandPath = Path()
    private val outlinePath = Path()
    private val mappedCorners = FloatArray(8)
    private var coordinateScale = 1f
    private var coordinateDx = 0f
    private var coordinateDy = 0f

    private val presentationInterpolator = TubePresentationInterpolator()
    private var frameW: Int = 1
    private var frameH: Int = 1
    var mirrorHorizontally: Boolean = false
        set(value) {
            field = value
            invalidate()
        }

    fun updateResult(track: TubeTrackResult?) {
        if (track == null) {
            presentationInterpolator.reset()
        } else {
            presentationInterpolator.accept(track, System.nanoTime())
        }
        if (track != null && track.frameWidth > 0 && track.frameHeight > 0) {
            frameW = track.frameWidth
            frameH = track.frameHeight
        }
        // Ensure redraw even if called off the UI thread.
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        // Aiming guide — only meaningful while framing a live scene. Drawn straight from
        // DetectionRoi so it marks exactly the region the detector is fed, nothing looser.
        val roiL = w * DetectionRoi.LEFT
        val roiT = h * DetectionRoi.TOP
        val roiR = roiL + w * DetectionRoi.WIDTH
        val roiB = roiT + h * DetectionRoi.HEIGHT
        canvas.drawRect(roiL, roiT, roiR, roiB, guidePaint)
        canvas.drawLine(w * 0.5f, roiT, w * 0.5f, roiB, guidePaint)

        val presentationTimestampNs = System.nanoTime()
        val r = presentationInterpolator.sample(presentationTimestampNs) ?: return
        if (presentationInterpolator.isAnimatingAt(presentationTimestampNs)) {
            postInvalidateOnAnimation()
        }
        updateCoordinateMapping(w, h)
        val stroke = if (r.mode == TrackMode.LOCKED) boxPaint else boxSearchPaint

        // Prefer true OBB polygon whenever corners are available.
        val corners = r.corners
        if (corners != null && corners.size == 4) {
            outlinePath.reset()
            for (i in 0 until 4) {
                val offset = i * 2
                mappedCorners[offset] = mapX(corners[i].first)
                mappedCorners[offset + 1] = mapY(corners[i].second)
                if (i == 0) {
                    outlinePath.moveTo(mappedCorners[offset], mappedCorners[offset + 1])
                } else {
                    outlinePath.lineTo(mappedCorners[offset], mappedCorners[offset + 1])
                }
            }
            outlinePath.close()
            canvas.drawPath(outlinePath, fillPaint)
            canvas.drawPath(outlinePath, stroke)
            for (i in 0 until 4) {
                val prevOffset = ((i + 3) % 4) * 2
                val offset = i * 2
                val nextOffset = ((i + 1) % 4) * 2
                drawCornerTick(
                    canvas,
                    mappedCorners[offset], mappedCorners[offset + 1],
                    mappedCorners[prevOffset], mappedCorners[prevOffset + 1],
                    mappedCorners[nextOffset], mappedCorners[nextOffset + 1]
                )
            }
            r.scale?.let { drawScale(canvas, it) }
            return
        }

        // Fallback: left/right walls if OBB corners missing
        val left = r.left ?: return
        val right = r.right ?: return
        val top = r.topY ?: 0f
        val bot = r.bottomY ?: frameH.toFloat()
        outlinePath.reset()
        outlinePath.moveTo(mapX(left.xAt(top)), mapY(top))
        outlinePath.lineTo(mapX(right.xAt(top)), mapY(top))
        outlinePath.lineTo(mapX(right.xAt(bot)), mapY(bot))
        outlinePath.lineTo(mapX(left.xAt(bot)), mapY(bot))
        outlinePath.close()
        canvas.drawPath(outlinePath, fillPaint)
        canvas.drawPath(outlinePath, stroke)
    }

    /**
     * Draws the sampling band, the detected scale lines and the ball.
     *
     * Tick marks are drawn **perpendicular to the tube axis**, not horizontally. On a
     * tilted tube horizontal lines would visibly fail to follow the graduations, which
     * makes it impossible to tell a real detection error from a drawing error.
     */
    private fun drawScale(canvas: Canvas, s: ScaleReading) {
        val axis = s.axis
        val half = s.bandHalfWidthPx

        // Sampling band: the strip the 1D profile was actually averaged over.
        val s0 = s.axialPx.firstOrNull()
        val s1 = s.axialPx.lastOrNull()
        if (s0 != null && s1 != null) {
            bandPath.reset()
            appendBandCorner(bandPath, axis, s0, -half, move = true)
            appendBandCorner(bandPath, axis, s1, -half, move = false)
            appendBandCorner(bandPath, axis, s1, +half, move = false)
            appendBandCorner(bandPath, axis, s0, +half, move = false)
            bandPath.close()
            canvas.drawPath(bandPath, bandFillPaint)
            canvas.drawPath(bandPath, bandEdgePaint)

            // Tick detection alone uses the yellow centre band. The ball-detector ROI is the
            // full blue YOLO tube box (plus its trained 10% short-axis margin).
            val paint = tickPaint.also { it.alpha = if (s.degraded) 90 else 255 }
            val tickHalf = half * 0.85f
            for (i in s.axialPx.indices) {
                val d = s.axialPx[i]
                canvas.drawLine(
                    mapX(axis.pointX(d, -tickHalf)),
                    mapY(axis.pointY(d, -tickHalf)),
                    mapX(axis.pointX(d, +tickHalf)),
                    mapY(axis.pointY(d, +tickHalf)),
                    paint
                )
            }
        }

        // A real 2D ball point gets a translucent disc and a two-axis crosshair. The
        // legacy 1D fallback has no lateral coordinate and remains an amber axial line.
        s.ballAxialPx?.let { d ->
            val center = s.ballCenter
            if (center != null) {
                val crossRadius = 22f
                val centerX = mapX(center.x)
                val centerY = mapY(center.y)
                canvas.drawCircle(centerX, centerY, 12f, ballFillPaint)
                canvas.drawLine(
                    mapX(axis.pointX(center.axialPx - crossRadius, center.lateralPx)),
                    mapY(axis.pointY(center.axialPx - crossRadius, center.lateralPx)),
                    mapX(axis.pointX(center.axialPx + crossRadius, center.lateralPx)),
                    mapY(axis.pointY(center.axialPx + crossRadius, center.lateralPx)),
                    ballCrossPaint
                )
                canvas.drawLine(
                    mapX(axis.pointX(center.axialPx, center.lateralPx - crossRadius)),
                    mapY(axis.pointY(center.axialPx, center.lateralPx - crossRadius)),
                    mapX(axis.pointX(center.axialPx, center.lateralPx + crossRadius)),
                    mapY(axis.pointY(center.axialPx, center.lateralPx + crossRadius)),
                    ballCrossPaint
                )
                val contact = s.wallContact.state != WallContactState.CLEAR &&
                    s.wallContact.state != WallContactState.UNKNOWN
                if (contact) canvas.drawCircle(centerX, centerY, 16f, wallContactPaint)
            } else {
                val wide = half * 3f
                canvas.drawLine(
                    mapX(axis.pointX(d, -wide)),
                    mapY(axis.pointY(d, -wide)),
                    mapX(axis.pointX(d, +wide)),
                    mapY(axis.pointY(d, +wide)),
                    legacyBallLinePaint
                )
            }
        }
    }

    private fun appendBandCorner(
        path: Path,
        axis: TubeAxis,
        s: Float,
        t: Float,
        move: Boolean
    ) {
        val x = mapX(axis.pointX(s, t))
        val y = mapY(axis.pointY(s, t))
        if (move) path.moveTo(x, y) else path.lineTo(x, y)
    }

    /**
     * Map analysis/frame coordinates to overlay view coordinates, mirroring however the
     * view underneath scales its own content:
     *
     *  - camera -> PreviewView scaleType="fillCenter" -> max (crop)
     *
     * The camera path shares a CameraX ViewPort with ImageAnalysis, so the frame aspect
     * already equals the preview aspect and the two scales coincide to within a pixel.
     */
    private fun updateCoordinateMapping(viewW: Float, viewH: Float) {
        val fw = frameW.toFloat().coerceAtLeast(1f)
        val fh = frameH.toFloat().coerceAtLeast(1f)
        coordinateScale = max(viewW / fw, viewH / fh)
        coordinateDx = (viewW - fw * coordinateScale) * 0.5f
        coordinateDy = (viewH - fh * coordinateScale) * 0.5f
    }

    private fun mapX(x: Float): Float {
        val mapped = x * coordinateScale + coordinateDx
        return if (mirrorHorizontally) width - mapped else mapped
    }

    private fun mapY(y: Float): Float = y * coordinateScale + coordinateDy

    private fun drawCornerTick(
        canvas: Canvas,
        x1: Float, y1: Float,
        x2: Float, y2: Float,
        x3: Float, y3: Float
    ) {
        val len = 40f
        val vx1 = x2 - x1
        val vy1 = y2 - y1
        val vx2 = x3 - x1
        val vy2 = y3 - y1
        val n1 = length(vx1, vy1).coerceAtLeast(1f)
        val n2 = length(vx2, vy2).coerceAtLeast(1f)
        canvas.drawLine(x1, y1, x1 + vx1 / n1 * len, y1 + vy1 / n1 * len, cornerPaint)
        canvas.drawLine(x1, y1, x1 + vx2 / n2 * len, y1 + vy2 / n2 * len, cornerPaint)
    }

    private fun length(x: Float, y: Float): Float = kotlin.math.sqrt(x * x + y * y)
}
