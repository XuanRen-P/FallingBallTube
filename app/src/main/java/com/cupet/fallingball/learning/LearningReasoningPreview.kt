package com.cupet.fallingball.learning

internal object LearningReasoningPreview {
    fun phase(state: LearningMessageState, answerStarted: Boolean): LearningReasoningPhase = when {
        answerStarted || state == LearningMessageState.COMPLETE -> LearningReasoningPhase.COMPLETE
        state == LearningMessageState.STOPPED -> LearningReasoningPhase.STOPPED
        state == LearningMessageState.ERROR -> LearningReasoningPhase.ERROR
        else -> LearningReasoningPhase.ACTIVE
    }

    fun latestSentence(reasoning: String): String = reasoning
        .trim()
        .split(Regex("(?<=[。！？!?；;])|(?<=\\.)\\s+|\\n+"))
        .asReversed()
        .firstOrNull { it.isNotBlank() }
        .orEmpty()
        .trim()
        .replace(Regex("^#{1,6}\\s+"), "")
        .removeSurrounding("**")
}

internal enum class LearningReasoningPhase {
    ACTIVE,
    COMPLETE,
    STOPPED,
    ERROR
}
