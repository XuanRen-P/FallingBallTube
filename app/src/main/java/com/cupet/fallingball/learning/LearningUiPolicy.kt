package com.cupet.fallingball.learning

internal object LearningUiPolicy {
    const val FACT_ALPHA_CLEAR = 1f
    const val FACT_ALPHA_OBSCURED = 0.2f
    val QUESTION_EMOJI_OPTIONS = listOf("🤔", "🔬", "👀", "🤯", "💡", "🧪", "📊", "🧭")
    const val QUESTION_FADE_DURATION_MS = 220L
    const val QUESTION_STAGGER_MS = 70L
    const val QUESTION_START_TRANSLATION_DP = 8
    const val EMPTY_FOREGROUND_RESERVED_DP = 328
    const val MAX_REASONING_DISPLAY_CHARS = 1_600
    const val REASONING_RENDER_BATCH_CHARS = 24
    const val RUBBER_BAND_FACTOR = 0.4f
    const val RUBBER_REBOUND_DURATION_MS = 400L
    const val INPUT_BOTTOM_LIFT_DP = 12
    const val SYSTEM_OVERSCROLL_ENABLED = false

    fun factAlpha(obscured: Boolean): Float =
        if (obscured) FACT_ALPHA_OBSCURED else FACT_ALPHA_CLEAR

    fun randomQuestionEmojis(count: Int): List<String> {
        if (count <= 0) return emptyList()
        val shuffled = QUESTION_EMOJI_OPTIONS.shuffled()
        return List(count) { index -> shuffled[index % shuffled.size] }
    }

    fun decorateSuggestedQuestion(question: String, emoji: String): String =
        "$emoji  ${question.trim()}"

    fun conversationScrollable(hasMessages: Boolean): Boolean = hasMessages

    fun rubberBandTranslation(dragDistancePx: Float): Float =
        dragDistancePx * RUBBER_BAND_FACTOR

    fun canRetry(
        role: LearningMessageRole,
        state: LearningMessageState,
        sessionActive: Boolean
    ): Boolean = sessionActive && role == LearningMessageRole.ASSISTANT &&
        state != LearningMessageState.STREAMING

    fun shouldAttachChart(hasPriorUserMessage: Boolean, message: String): Boolean =
        !hasPriorUserMessage || Regex("图|曲线|轨迹|位移|拟合|异常点").containsMatchIn(message)
}
