package com.cupet.fallingball.vision

/** Preview-relative point at which CameraX should meter and lock focus. */
data class FocusTarget(val xFraction: Float, val yFraction: Float)

/**
 * Produces one focus request after a tube lock and then stays quiet until [reset].
 *
 * Camera autofocus is deliberately not driven from every detector result. Re-triggering it
 * continuously makes the lens hunt around an already suitable position ("focus breathing"),
 * which also changes the apparent graduation geometry.
 */
class TubeFocusLockGate {
    private var requestPending = false
    private var focusSettled = false
    private var attemptCount = 0

    fun takeRequest(track: TubeTrackResult): FocusTarget? {
        if (requestPending || focusSettled || track.mode != TrackMode.LOCKED ||
            track.frameWidth <= 0 || track.frameHeight <= 0
        ) {
            return null
        }

        val centre = track.corners
            ?.takeIf { it.isNotEmpty() }
            ?.let { corners ->
                corners.map { it.first }.average().toFloat() to
                    corners.map { it.second }.average().toFloat()
            }
            ?: run {
                val left = track.left ?: return null
                val right = track.right ?: return null
                val y = when {
                    track.topY != null && track.bottomY != null ->
                        (track.topY + track.bottomY) / 2f
                    else -> track.frameHeight / 2f
                }
                ((left.xAt(y) + right.xAt(y)) / 2f) to y
            }

        requestPending = true
        attemptCount += 1
        return FocusTarget(
            xFraction = (centre.first / track.frameWidth).coerceIn(0f, 1f),
            yFraction = (centre.second / track.frameHeight).coerceIn(0f, 1f)
        )
    }

    /**
     * Allows one retry after a failed CameraX request. A hard cap prevents an unsupported or
     * difficult scene from turning into continuous lens hunting.
     */
    fun onRequestCompleted(success: Boolean) {
        requestPending = false
        focusSettled = success || attemptCount >= MAX_ATTEMPTS
    }

    fun reset() {
        requestPending = false
        focusSettled = false
        attemptCount = 0
    }

    private companion object {
        const val MAX_ATTEMPTS = 2
    }
}
