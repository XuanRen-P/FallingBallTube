package com.cupet.fallingball.vision

/**
 * User-supplied description of the cylinder being measured.
 *
 * @param tickSpacingMm physical distance between adjacent graduations. Used only to turn
 *   pixel positions into millimetres; the detection algorithm never sees it.
 * @param expectedTickCount total graduations on the cylinder, or [COUNT_UNKNOWN] before the
 *   user has told us. When known it acts as a validation gate: a frame whose detected count
 *   is far from this is rejected. That assumes the whole cylinder is in shot, which is also
 *   what the OBB-derived sampling span assumes.
 */
data class RulerConfig(
    val tickSpacingMm: Float,
    val expectedTickCount: Int
) {
    /**
     * Whether the count-based gates can run at all. With no count supplied there is nothing
     * to compare against, so detection has to be trusted as-is -- better than rejecting
     * every frame against a guessed number and showing the user nothing.
     */
    val hasExpectedCount: Boolean get() = expectedTickCount >= MIN_TICKS

    /** Full span between the outermost graduations, in mm. */
    val totalSpanMm: Float get() = tickSpacingMm * (expectedTickCount - 1)

    /**
     * How many graduations should actually be visible in the profile.
     *
     * The sampler trims each end of the tube, and [TickDetector] additionally discards
     * peaks in the unreliable boundary region, so the visible count is systematically
     * below [expectedTickCount]. Comparing raw counts against the user's number would
     * make the count gate fire on perfectly good frames.
     */
    val expectedVisibleCount: Float
        get() = expectedTickCount * (1f - 2f * AxialSampler.AXIAL_TRIM)

    fun isValid(): Boolean =
        tickSpacingMm > 0f && tickSpacingMm <= MAX_SPACING_MM &&
            expectedTickCount in MIN_TICKS..MAX_TICKS

    companion object {
        const val MIN_TICKS = 5
        const val MAX_TICKS = 500
        const val MAX_SPACING_MM = 100f

        /** No graduation count supplied yet; the count and geometry gates stay off. */
        const val COUNT_UNKNOWN = 0

        val DEFAULT = RulerConfig(tickSpacingMm = 1f, expectedTickCount = COUNT_UNKNOWN)
    }
}
