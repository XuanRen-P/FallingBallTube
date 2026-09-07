package com.cupet.fallingball.vision

/**
 * User-selectable YOLO input sizes, ordered from lowest latency to highest detail.
 *
 * Every size is divisible by the model's largest stride (32). The dynamic ONNX graph
 * produces one output anchor for each cell at strides 8, 16, and 32.
 */
enum class ModelProfile(val inputSize: Int) {
    FASTEST_320(320),
    FAST_416(416),
    BALANCED_512(512),
    ACCURATE_640(640);

    val expectedAnchorCount: Int
        get() =
            square(inputSize / 8) +
                square(inputSize / 16) +
                square(inputSize / 32)

    companion object {
        val DEFAULT = ACCURATE_640

        fun fromInputSize(inputSize: Int): ModelProfile =
            entries.firstOrNull { it.inputSize == inputSize } ?: DEFAULT

        private fun square(value: Int): Int = value * value
    }
}
