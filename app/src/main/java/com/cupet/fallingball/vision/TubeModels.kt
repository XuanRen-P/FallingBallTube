package com.cupet.fallingball.vision

/**
 * Fraction of the camera frame that is actually fed to the detector.
 *
 * The excluded margins are never sampled into the model input, so clutter at the edges
 * of the bench cannot produce a detection at all. [com.cupet.fallingball.ui.OverlayView]
 * draws its guide rectangle from these same numbers, so the on-screen guide always shows
 * exactly where detection happens.
 *
 * Camera mode uses the live preview viewport and the same canonical ROI as the PC pipeline.
 */
object DetectionRoi {
    /** Left edge as a fraction of frame width (excludes the leftmost 15%). */
    const val LEFT = 0.15f

    /** Width as a fraction of frame width (middle 70%). */
    const val WIDTH = 0.70f

    /** Top edge as a fraction of frame height (excludes the topmost 5%). */
    const val TOP = 0.05f

    /** Height as a fraction of frame height (middle 90%). */
    const val HEIGHT = 0.90f
}

/**
 * Describes the region the square model input was letterboxed from, and how to map
 * detections back into full-frame display coordinates.
 *
 * Detection may run on a centre crop (see [DetectionRoi]) while the overlay still draws
 * in full-frame space, so the inverse-letterbox has to be followed by a translation.
 */
data class InputRegion(
    /** Size of the ROI the tensor was letterboxed from. */
    val roiW: Int,
    val roiH: Int,
    /** ROI origin within the full frame, in display coordinates. */
    val offsetX: Int = 0,
    val offsetY: Int = 0,
    /** Full frame size, reported to the overlay for coordinate mapping. */
    val frameW: Int = roiW,
    val frameH: Int = roiH
)

/** Why a fall recording stopped. */
enum class BallEndReason { BALL_LOST, REACHED_BOTTOM, MANUAL }

/**
 * Line model: x = slope * y + intercept  (works well for near-vertical tube walls).
 */
data class WallLine(
    val slope: Float,
    val intercept: Float,
    val score: Float = 1f
) {
    fun xAt(y: Float): Float = slope * y + intercept

    fun withSmoothed(other: WallLine, alpha: Float): WallLine {
        val a = alpha.coerceIn(0f, 1f)
        return WallLine(
            slope = slope * (1 - a) + other.slope * a,
            intercept = intercept * (1 - a) + other.intercept * a,
            score = score * (1 - a) + other.score * a
        )
    }
}

enum class TrackMode {
    SEARCHING,
    LOCKED
}

data class TubeObservation(
    val left: WallLine,
    val right: WallLine,
    val topY: Float? = null,
    val bottomY: Float? = null,
    val confidence: Float = 0f,
    val frameWidth: Int,
    val frameHeight: Int,
    /** True OBB corners in frame coordinates. */
    val corners: List<Pair<Float, Float>>? = null
)

data class TubeTrackResult(
    val mode: TrackMode,
    val left: WallLine? = null,
    val right: WallLine? = null,
    val topY: Float? = null,
    val bottomY: Float? = null,
    val confidence: Float = 0f,
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    /** Camera analysis rate; scale sampling and UI updates run at this cadence. */
    val fps: Float = 0f,
    /** Rate at which fresh YOLO boxes are produced. */
    val inferenceFps: Float = 0f,
    /** End-to-end ONNX detection latency for the most recent fresh box. */
    val inferenceMs: Float = 0f,
    /** Square YOLO tensor side selected by the user. */
    val modelInputSize: Int = ModelProfile.DEFAULT.inputSize,
    /** Monotonic profile version used to reject stale inference/UI results after a switch. */
    val profileGeneration: Long = 0L,
    /** Capture time of the YOLO frame that produced this geometry. */
    val geometryCaptureTimestampNs: Long = 0L,
    /** Increments only when a non-null YOLO observation changes the retained geometry. */
    val geometryRevision: Long = 0L,
    /** True OBB corners in frame coordinates when available. */
    val corners: List<Pair<Float, Float>>? = null,
    /** Smoothed end-to-end stage timings for the latest fresh YOLO results. */
    val pipelineTimings: PipelineTimings? = null,
    /** Scale-line and ball readings for this frame, null until the tube is detected. */
    val scale: ScaleReading? = null
)

/**
 * Per-frame scale-line and ball measurement, in the tube's own axial coordinates.
 *
 * Arrays here are private copies: [TickDetector] reuses its buffers every frame, so
 * anything handed to the UI thread has to be snapshotted first.
 *
 * @param axialPx tick distances from the tube's top end, along the axis
 * @param ballAxialPx ball distance along the axis, or null when not seen
 */
class ScaleReading(
    val axis: TubeAxis,
    val axialPx: FloatArray,
    val ordinals: IntArray,
    val ballAxialPx: Float?,
    val ballCenter: TubeBallPoint? = null,
    val ballConfidence: Float = 0f,
    val ballSource: BallDetectionSource = BallDetectionSource.NONE,
    val wallContact: WallContactResult = WallContactResult(
        WallContactState.UNKNOWN,
        null,
        null
    ),
    val medianGapPx: Float,
    val bandHalfWidthPx: Float,
    val state: RulerState,
    val invalidReason: InvalidReason?
) {
    val count get() = axialPx.size

    /** Ticks are drawn faded while the reading is only being remembered, not confirmed. */
    val degraded get() = state == RulerState.DEGRADED
}
