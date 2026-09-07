package com.cupet.fallingball.analysis

enum class MeasurementSource {
    UNKNOWN,
    LIVE_CAMERA,
    IMPORTED_VIDEO
}

/** Original media facts and complete-pass counters retained independently of the media URI. */
data class VideoMeasurementSource(
    val sourceName: String,
    val sourceUri: String,
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val nominalFrameRate: Float?,
    val durationUs: Long,
    val clipStartUs: Long,
    val clipEndUs: Long,
    val decodedFrameCount: Long,
    val processedFrameCount: Long,
    val inferenceFrameCount: Long,
    val taskId: String
) {
    init {
        require(sourceName.isNotBlank()) { "Missing source video name" }
        require(sourceUri.isNotBlank()) { "Missing source video URI" }
        require(width > 0 && height > 0) { "Invalid source video dimensions" }
        require(rotationDegrees in setOf(0, 90, 180, 270)) { "Invalid video rotation" }
        require(nominalFrameRate == null || nominalFrameRate.isFinite() && nominalFrameRate > 0f) {
            "Invalid source video frame rate"
        }
        require(durationUs > 0 && clipStartUs >= 0 && clipStartUs < clipEndUs &&
            clipEndUs <= durationUs
        ) { "Invalid source video clip range" }
        require(decodedFrameCount >= 0 && processedFrameCount in 0..decodedFrameCount &&
            inferenceFrameCount in 0..processedFrameCount
        ) { "Invalid source video frame counts" }
        require(taskId.isNotBlank()) { "Missing video analysis task ID" }
    }
}

internal fun validateMeasurementSource(
    source: MeasurementSource,
    videoSource: VideoMeasurementSource?
) {
    require((source == MeasurementSource.IMPORTED_VIDEO) == (videoSource != null)) {
        "Video source metadata does not match measurement source"
    }
}
