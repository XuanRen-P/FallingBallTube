package com.cupet.fallingball.camera

import com.cupet.fallingball.vision.RulerSnapshot

/** Maps a temporal peak through either a fresh ruler or its short-lived retained memory. */
internal fun RulerSnapshot.temporalPositionTicks(
    axialPx: Float,
    tickSpacingMm: Float
): Float? {
    if (!isUsable || !axialPx.isFinite() ||
        !tickSpacingMm.isFinite() || tickSpacingMm <= 0f
    ) {
        return null
    }
    return toMm(axialPx, tickSpacingMm, refOrdinal = 0)?.div(tickSpacingMm)
}
