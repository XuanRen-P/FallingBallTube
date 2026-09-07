package com.cupet.fallingball.vision

/**
 * Maps display (logical) coordinates to YUV buffer coordinates for one camera frame.
 *
 * Two independent paths sample the same frame -- the selectable square model tensor and
 * the 1D axial profile used for scale-line detection -- and they must agree exactly. Keeping the
 * mapping in one place is the only way to guarantee that; if the two ever drift apart the
 * symptom is an overlay whose tick lines sit somewhere other than where the pixels were
 * actually read, which is close to undebuggable.
 *
 * The four rotation cases are the real-device-verified mapping recorded in the
 * integration guide. Do not re-derive them:
 * ```
 *   90  -> bufX = cropLeft + logicalY             bufY = cropTop + (cropH-1) - logicalX
 *   180 -> bufX = cropLeft + (cropW-1) - logicalX bufY = cropTop + (cropH-1) - logicalY
 *   270 -> bufX = cropLeft + (cropW-1) - logicalY bufY = cropTop + logicalX
 *   0   -> bufX = cropLeft + logicalX             bufY = cropTop + logicalY
 * ```
 * They are stored as affine coefficients so the per-pixel `when` can be hoisted out of
 * sampling loops.
 *
 * Note that every coefficient is 0 or +/-1: **display pixels and buffer pixels are 1:1**.
 * That is what lets the axial sampler work at native sensor resolution while the tensor
 * path downsamples, and it means a position measured in the 1D signal can be handed
 * straight to `OverlayView` without conversion.
 */
class BufferTransform private constructor(
    private val xkx: Float,
    private val xky: Float,
    private val xOff: Float,
    private val ykx: Float,
    private val yky: Float,
    private val yOff: Float,
    /** ViewPort crop in buffer coordinates. */
    val cropLeft: Int,
    val cropTop: Int,
    val cropW: Int,
    val cropH: Int,
    /** Crop size after rotation, i.e. the coordinate space detections live in. */
    val displayW: Int,
    val displayH: Int
) {

    fun bufX(logicalX: Float, logicalY: Float): Float = xkx * logicalX + xky * logicalY + xOff

    fun bufY(logicalX: Float, logicalY: Float): Float = ykx * logicalX + yky * logicalY + yOff

    /** Row-invariant part of [bufX], for hoisting out of an inner loop over logicalX. */
    fun rowBaseX(logicalY: Float): Float = xky * logicalY + xOff

    /** Row-invariant part of [bufY]. */
    fun rowBaseY(logicalY: Float): Float = yky * logicalY + yOff

    /** Column term of [bufX] to add to [rowBaseX]. */
    fun colTermX(logicalX: Float): Float = xkx * logicalX

    /** Column term of [bufY] to add to [rowBaseY]. */
    fun colTermY(logicalX: Float): Float = ykx * logicalX

    /** True when the buffer point lies inside the ViewPort crop the user can actually see. */
    fun isInsideCrop(bufX: Float, bufY: Float): Boolean =
        bufX >= cropLeft && bufX <= cropLeft + cropW - 1 &&
            bufY >= cropTop && bufY <= cropTop + cropH - 1

    /** OpenCV BORDER_REPLICATE equivalent for canonical ROIs extending past the crop. */
    fun clampBufXToCrop(bufX: Float): Float =
        bufX.coerceIn(cropLeft.toFloat(), (cropLeft + cropW - 1).toFloat())

    fun clampBufYToCrop(bufY: Float): Float =
        bufY.coerceIn(cropTop.toFloat(), (cropTop + cropH - 1).toFloat())

    companion object {
        fun of(rotationDegrees: Int, cropLeft: Int, cropTop: Int, cropW: Int, cropH: Int): BufferTransform {
            val xkx: Float; val xky: Float; val xOff: Float
            val ykx: Float; val yky: Float; val yOff: Float
            when (rotationDegrees) {
                90 -> {
                    xkx = 0f; xky = 1f; xOff = cropLeft.toFloat()
                    ykx = -1f; yky = 0f; yOff = (cropTop + cropH - 1).toFloat()
                }
                180 -> {
                    xkx = -1f; xky = 0f; xOff = (cropLeft + cropW - 1).toFloat()
                    ykx = 0f; yky = -1f; yOff = (cropTop + cropH - 1).toFloat()
                }
                270 -> {
                    xkx = 0f; xky = -1f; xOff = (cropLeft + cropW - 1).toFloat()
                    ykx = 1f; yky = 0f; yOff = cropTop.toFloat()
                }
                else -> {
                    xkx = 1f; xky = 0f; xOff = cropLeft.toFloat()
                    ykx = 0f; yky = 1f; yOff = cropTop.toFloat()
                }
            }
            val rotated = rotationDegrees == 90 || rotationDegrees == 270
            return BufferTransform(
                xkx, xky, xOff, ykx, yky, yOff,
                cropLeft, cropTop, cropW, cropH,
                displayW = if (rotated) cropH else cropW,
                displayH = if (rotated) cropW else cropH
            )
        }
    }
}
