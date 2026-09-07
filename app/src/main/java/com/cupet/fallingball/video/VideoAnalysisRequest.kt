package com.cupet.fallingball.video

import androidx.work.Data
import com.cupet.fallingball.analysis.MeasurementDefaults
import com.cupet.fallingball.analysis.ViscometryParameters
import com.cupet.fallingball.vision.RulerConfig
import org.json.JSONObject

/** An immutable snapshot: retries never read mutable camera preferences. */
data class VideoAnalysisRequest(
    val uri: String,
    val videoName: String,
    val clipStartUs: Long,
    val clipEndUs: Long,
    val rulerConfig: RulerConfig,
    val defaults: MeasurementDefaults,
    val requestedAtEpochMs: Long
) {
    init {
        require(uri.isNotBlank()) { "视频地址为空" }
        require(clipStartUs >= 0 && clipEndUs > clipStartUs) { "视频片段时间无效" }
        require(clipEndUs <= (Long.MAX_VALUE - TIMESTAMP_OFFSET_NS) / 1_000L) {
            "视频时长超出支持范围"
        }
        require(rulerConfig.tickSpacingMm.isFinite() && rulerConfig.tickSpacingMm > 0f &&
            rulerConfig.tickSpacingMm <= RulerConfig.MAX_SPACING_MM) { "刻度间距无效" }
        require(rulerConfig.expectedTickCount == RulerConfig.COUNT_UNKNOWN ||
            rulerConfig.expectedTickCount in RulerConfig.MIN_TICKS..RulerConfig.MAX_TICKS) {
            "刻度数量无效"
        }
    }

    fun toData(): Data = Data.Builder().putString(INPUT_SNAPSHOT, toJson()).build()

    fun toJson(): String {
        val parameters = defaults.viscometryParameters
        return JSONObject().apply {
            put("uri", uri)
            put("videoName", videoName)
            put("clipStartUs", clipStartUs)
            put("clipEndUs", clipEndUs)
            put("tickSpacingMm", rulerConfig.tickSpacingMm.toDouble())
            put("expectedTickCount", rulerConfig.expectedTickCount)
            put("requestedAtEpochMs", requestedAtEpochMs)
            put("liquidName", defaults.liquidName)
            put("sphereMaterial", defaults.sphereMaterial)
            putOptional("temperatureC", defaults.temperatureC)
            putOptional("sphereRadiusMm", parameters.sphereRadiusMm)
            putOptional("tubeInnerDiameterMm", parameters.tubeInnerDiameterMm)
            putOptional("liquidDensityGPerCm3", parameters.liquidDensityGPerCm3)
            putOptional("sphereDensityGPerCm3", parameters.sphereDensityGPerCm3)
        }.toString()
    }

    companion object {
        const val INPUT_SNAPSHOT = "video_input_snapshot"
        const val TIMESTAMP_OFFSET_NS = 1_000_000_000L

        fun fromData(data: Data): VideoAnalysisRequest = fromJson(
            requireNotNull(data.getString(INPUT_SNAPSHOT)) { "视频任务参数缺失" }
        )

        fun fromJson(encoded: String): VideoAnalysisRequest {
            val json = JSONObject(encoded)
            return VideoAnalysisRequest(
                uri = json.getString("uri"),
                videoName = json.getString("videoName"),
                clipStartUs = json.getLong("clipStartUs"),
                clipEndUs = json.getLong("clipEndUs"),
                rulerConfig = RulerConfig(
                    json.getDouble("tickSpacingMm").toFloat(),
                    json.getInt("expectedTickCount")
                ),
                defaults = MeasurementDefaults(
                    viscometryParameters = ViscometryParameters(
                        sphereRadiusMm = json.optionalDouble("sphereRadiusMm"),
                        tubeInnerDiameterMm = json.optionalDouble("tubeInnerDiameterMm"),
                        liquidDensityGPerCm3 = json.optionalDouble("liquidDensityGPerCm3"),
                        sphereDensityGPerCm3 = json.optionalDouble("sphereDensityGPerCm3")
                    ),
                    liquidName = json.getString("liquidName"),
                    sphereMaterial = json.getString("sphereMaterial"),
                    temperatureC = json.optionalDouble("temperatureC")
                ),
                requestedAtEpochMs = json.getLong("requestedAtEpochMs")
            )
        }

        private fun JSONObject.optionalDouble(key: String): Double? =
            if (isNull(key)) null else getDouble(key)

        private fun JSONObject.putOptional(key: String, value: Double?) {
            put(key, value ?: JSONObject.NULL)
        }
    }
}

/** Half-open clip bounds avoid counting a boundary frame in two adjacent clips. */
internal class VideoFrameTimeline(private val startUs: Long, private val endUs: Long) {
    private var lastPtsUs: Long? = null
    var decodedFrameCount: Long = 0L
        private set
    var processedFrameCount: Long = 0L
        private set

    init {
        require(startUs >= 0 && endUs > startUs)
    }

    fun accept(presentationTimeUs: Long): Boolean {
        val previous = lastPtsUs
        require(previous == null || presentationTimeUs > previous) {
            "视频帧时间戳重复或倒退，无法可靠测速"
        }
        lastPtsUs = presentationTimeUs
        if (presentationTimeUs < startUs || presentationTimeUs >= endUs) return false
        decodedFrameCount++
        return true
    }

    fun frameProcessed() {
        check(processedFrameCount < decodedFrameCount)
        processedFrameCount++
    }

    fun timestampNs(presentationTimeUs: Long): Long =
        Math.addExact(Math.multiplyExact(presentationTimeUs, 1_000L), VideoAnalysisRequest.TIMESTAMP_OFFSET_NS)

    fun verifyComplete(expectedFrameCount: Long) {
        check(decodedFrameCount == expectedFrameCount && processedFrameCount == expectedFrameCount) {
            "视频帧处理不完整，无法保存为完整分析结果"
        }
    }
}

/** Extractor chunks can be fragments of one compressed frame; only its last chunk counts. */
internal class EncodedVideoFrameCounter(private val startUs: Long, private val endUs: Long) {
    private var pendingTimestampUs: Long? = null
    private var count = 0L

    fun accept(timestampUs: Long, partialFrame: Boolean) {
        require(pendingTimestampUs == null || pendingTimestampUs == timestampUs) {
            "视频分片时间戳不一致，文件可能损坏"
        }
        if (partialFrame) {
            pendingTimestampUs = timestampUs
        } else {
            pendingTimestampUs = null
            if (timestampUs >= startUs && timestampUs < endUs) count++
        }
    }

    fun finish(): Long {
        check(pendingTimestampUs == null) { "视频最后一帧不完整，文件可能损坏" }
        return count
    }
}
