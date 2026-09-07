package com.cupet.fallingball.ui

/**
 * Light appearance tokens. Names describe the role a color plays, not its hue.
 * Keeping the palette here makes custom Views follow the same hierarchy as XML
 * Material components and avoids reusing dark-theme colors in a light surface.
 */
object UiColors {
    const val PAGE_BACKGROUND: Int = 0xFFF5F7F8.toInt()
    const val SURFACE: Int = 0xFFFFFFFF.toInt()
    const val SURFACE_CONTAINER: Int = 0xFFEDF2F3.toInt()
    const val SURFACE_CONTAINER_HIGH: Int = 0xFFE4EAEC.toInt()
    const val SURFACE_TINT: Int = 0xFFF9FBFB.toInt()

    const val ON_SURFACE: Int = 0xFF18252B.toInt()
    const val ON_SURFACE_VARIANT: Int = 0xFF4B5B63.toInt()
    const val ON_SURFACE_MUTED: Int = 0xFF60727A.toInt()
    const val ON_SURFACE_DISABLED: Int = 0xFF7B8A91.toInt()

    const val PRIMARY: Int = 0xFF006B52.toInt()
    const val ON_PRIMARY: Int = 0xFFFFFFFF.toInt()
    const val PRIMARY_CONTAINER: Int = 0xFFC8F2E3.toInt()
    const val ON_PRIMARY_CONTAINER: Int = 0xFF00382A.toInt()
    const val SECONDARY: Int = 0xFF245B9E.toInt()
    const val SECONDARY_CONTAINER: Int = 0xFFD8E8EB.toInt()
    const val ON_SECONDARY_CONTAINER: Int = 0xFF102F37.toInt()

    const val SUCCESS: Int = 0xFF007457.toInt()
    const val SUCCESS_CONTAINER: Int = 0xFFD1F4E7.toInt()
    const val WARNING: Int = 0xFF8A5200.toInt()
    const val WARNING_CONTAINER: Int = 0xFFFFE8C2.toInt()
    const val ERROR: Int = 0xFFB3261E.toInt()
    const val ERROR_CONTAINER: Int = 0xFFFFDAD6.toInt()
    const val ON_ERROR_CONTAINER: Int = 0xFF5F1110.toInt()

    const val OUTLINE: Int = 0xFF73858D.toInt()
    const val OUTLINE_VARIANT: Int = 0xFFB8C5C9.toInt()
    const val DIVIDER: Int = 0xFFD7E0E3.toInt()
    const val CHART_GRID: Int = 0xFFD4DEE1.toInt()
    const val CHART_LINE: Int = 0xFF006B52.toInt()
    const val CHART_POINT: Int = 0xFF174B5E.toInt()
    const val CHART_WARNING: Int = 0xFF8A5200.toInt()

    /** WCAG relative luminance contrast ratio for opaque sRGB colors. */
    fun contrastRatio(foreground: Int, background: Int): Double {
        fun channel(value: Int): Double {
            val normalized = value / 255.0
            return if (normalized <= 0.03928) normalized / 12.92
            else Math.pow((normalized + 0.055) / 1.055, 2.4)
        }
        fun luminance(color: Int): Double =
            0.2126 * channel(color ushr 16 and 0xFF) +
                0.7152 * channel(color ushr 8 and 0xFF) +
                0.0722 * channel(color and 0xFF)
        val lighter = maxOf(luminance(foreground), luminance(background))
        val darker = minOf(luminance(foreground), luminance(background))
        return (lighter + 0.05) / (darker + 0.05)
    }
}
