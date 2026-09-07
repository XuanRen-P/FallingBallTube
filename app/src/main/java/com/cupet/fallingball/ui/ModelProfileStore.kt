package com.cupet.fallingball.ui

import android.content.Context
import com.cupet.fallingball.vision.ModelProfile

/** Remembers the user's YOLO speed/accuracy choice across launches. */
object ModelProfileStore {
    private const val PREFS = "model_profile"
    private const val KEY_INPUT_SIZE = "input_size"

    fun load(context: Context): ModelProfile =
        ModelProfile.fromInputSize(
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getInt(KEY_INPUT_SIZE, ModelProfile.DEFAULT.inputSize)
        )

    fun save(context: Context, profile: ModelProfile) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putInt(KEY_INPUT_SIZE, profile.inputSize)
            .apply()
    }
}
