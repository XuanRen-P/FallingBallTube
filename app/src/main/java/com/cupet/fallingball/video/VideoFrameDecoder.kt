package com.cupet.fallingball.video

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Rect
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.SystemClock
import android.provider.OpenableColumns
import com.cupet.fallingball.vision.AnalysisFrame
import com.cupet.fallingball.vision.AnalysisPlane
import java.io.IOException

data class VideoMetadata(
    val uri: String,
    val videoName: String,
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val nominalFrameRate: Float?,
    val durationUs: Long
) {
    val displayWidth: Int get() = if (rotationDegrees % 180 == 0) width else height
    val displayHeight: Int get() = if (rotationDegrees % 180 == 0) height else width
}

data class VideoDecodeSummary(
    val metadata: VideoMetadata,
    val decodedFrameCount: Long,
    val processedFrameCount: Long
)

/** Only one output Image is live; its planes remain valid until the callback returns. */
class VideoFrameDecoder(private val context: Context) {
    fun metadata(uri: Uri): VideoMetadata = withExtractor(uri) { extractor, track ->
        metadata(uri, extractor.getTrackFormat(track))
    }

    fun decode(
        request: VideoAnalysisRequest,
        checkCancellation: () -> Unit,
        onFrame: (AnalysisFrame, Long, Long) -> Unit
    ): VideoDecodeSummary = withExtractor(Uri.parse(request.uri)) { extractor, track ->
        val format = extractor.getTrackFormat(track)
        val metadata = metadata(Uri.parse(request.uri), format)
        require(request.clipEndUs <= metadata.durationUs) { "所选片段超出视频时长" }
        // Count encoded presentation samples independently of decoder callbacks. Matching two
        // counters incremented by the same callback would not detect a decoder dropping frames.
        val encodedFrames = EncodedVideoFrameCounter(request.clipStartUs, request.clipEndUs)
        while (extractor.sampleTime >= 0L) {
            checkCancellation()
            check(extractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_ENCRYPTED == 0) {
                "不支持加密视频"
            }
            encodedFrames.accept(extractor.sampleTime,
                extractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_PARTIAL_FRAME != 0)
            if (!extractor.advance()) break
        }
        val expectedFrames = encodedFrames.finish()
        check(expectedFrames > 0L) { "所选片段没有视频帧" }
        extractor.seekTo(request.clipStartUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
        val colorTransfer = format.optionalInt(MediaFormat.KEY_COLOR_TRANSFER)
        require(colorTransfer != MediaFormat.COLOR_TRANSFER_ST2084 &&
            colorTransfer != MediaFormat.COLOR_TRANSFER_HLG) { "暂不支持 HDR 视频，请使用普通 SDR 原速视频" }
        val mime = requireNotNull(format.getString(MediaFormat.KEY_MIME))
        // Byte-buffer decoding leaves orientation to the shared analysis frame adapter.
        format.setInteger(MediaFormat.KEY_ROTATION, 0)
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
        val codec = try {
            MediaCodec.createDecoderByType(mime)
        } catch (error: Exception) {
            throw IOException("此手机不支持该视频编码：$mime", error)
        }
        var started = false
        try {
            codec.configure(format, null, null, 0)
            codec.start()
            started = true
            extractor.seekTo(request.clipStartUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
            val timeline = VideoFrameTimeline(request.clipStartUs, request.clipEndUs)
            val outputInfo = MediaCodec.BufferInfo()
            var inputEnded = false
            var outputEnded = false
            var lastProgressMs = SystemClock.elapsedRealtime()
            var sourceFrameIndex = 0L
            while (!outputEnded) {
                checkCancellation()
                var progressed = false
                if (!inputEnded) {
                    val inputIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                    if (inputIndex >= 0) {
                        val inputBuffer = requireNotNull(codec.getInputBuffer(inputIndex)) {
                            "视频解码器未提供输入缓冲区"
                        }
                        val size = extractor.readSampleData(inputBuffer, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputEnded = true
                        } else {
                            check(extractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_ENCRYPTED == 0) {
                                "不支持加密视频"
                            }
                            val flags = if (extractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_PARTIAL_FRAME != 0) {
                                MediaCodec.BUFFER_FLAG_PARTIAL_FRAME
                            } else 0
                            codec.queueInputBuffer(inputIndex, 0, size, extractor.sampleTime, flags)
                            extractor.advance()
                        }
                        progressed = true
                    }
                }
                when (val outputIndex = codec.dequeueOutputBuffer(outputInfo, TIMEOUT_US)) {
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val output = codec.outputFormat
                        val outputTransfer = output.optionalInt(MediaFormat.KEY_COLOR_TRANSFER)
                        require(outputTransfer != MediaFormat.COLOR_TRANSFER_ST2084 &&
                            outputTransfer != MediaFormat.COLOR_TRANSFER_HLG) { "暂不支持 HDR 视频，请使用普通 SDR 原速视频" }
                        val outputWidth = output.getInteger(MediaFormat.KEY_WIDTH)
                        val outputHeight = output.getInteger(MediaFormat.KEY_HEIGHT)
                        check(outputWidth >= metadata.width && outputHeight >= metadata.height) {
                            "解码器降低了视频分辨率，分析已停止"
                        }
                        progressed = true
                    }
                    MediaCodec.INFO_TRY_AGAIN_LATER, MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit
                    else -> if (outputIndex >= 0) {
                        try {
                            if (outputInfo.size > 0 && outputInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                                val ptsUs = outputInfo.presentationTimeUs
                                if (timeline.accept(ptsUs)) {
                                    val image = codec.getOutputImage(outputIndex)
                                        ?: throw IOException("解码器无法提供原分辨率 YUV 帧，请使用兼容的普通 SDR 视频")
                                    image.use {
                                        check(image.format == ImageFormat.YUV_420_888 && image.planes.size == 3) {
                                            "该视频的像素格式不受支持"
                                        }
                                        val crop = Rect(image.cropRect)
                                        check(crop.width() == metadata.width && crop.height() == metadata.height) {
                                            "解码器输出尺寸与原视频不符，分析已停止"
                                        }
                                        val frame = AnalysisFrame(
                                            width = image.width,
                                            height = image.height,
                                            cropRect = crop,
                                            rotationDegrees = metadata.rotationDegrees,
                                            timestampNs = timeline.timestampNs(ptsUs),
                                            sourceFrameIndex = sourceFrameIndex,
                                            planes = image.planes.map { plane ->
                                                AnalysisPlane(plane.buffer.duplicate(), plane.rowStride, plane.pixelStride)
                                            }.toTypedArray()
                                        )
                                        onFrame(frame, ptsUs, timeline.decodedFrameCount)
                                        timeline.frameProcessed()
                                    }
                                }
                                sourceFrameIndex++
                                // Presentation output is ordered even when encoded B-frames are not.
                                if (ptsUs >= request.clipEndUs) outputEnded = true
                            }
                            if (outputInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputEnded = true
                        } finally {
                            codec.releaseOutputBuffer(outputIndex, false)
                        }
                        progressed = true
                    }
                }
                if (progressed) lastProgressMs = SystemClock.elapsedRealtime()
                check(SystemClock.elapsedRealtime() - lastProgressMs < STALL_TIMEOUT_MS) {
                    "视频解码超时，文件可能损坏或编码不受支持"
                }
            }
            check(timeline.processedFrameCount > 0) { "所选片段没有可分析的视频帧" }
            timeline.verifyComplete(expectedFrames)
            VideoDecodeSummary(metadata, timeline.decodedFrameCount, timeline.processedFrameCount)
        } finally {
            if (started) runCatching { codec.stop() }
            codec.release()
        }
    }

    private fun metadata(uri: Uri, format: MediaFormat): VideoMetadata {
        val durationUs = format.optionalLong(MediaFormat.KEY_DURATION)
            ?: throw IOException("无法读取视频时长")
        require(durationUs > 0) { "视频时长无效" }
        val rotation = ((format.optionalInt(MediaFormat.KEY_ROTATION) ?: 0) % 360 + 360) % 360
        require(rotation % 90 == 0) { "不支持该视频旋转角度" }
        val nominalFps = runCatching { format.getFloat(MediaFormat.KEY_FRAME_RATE) }.getOrNull()
            ?: format.optionalInt(MediaFormat.KEY_FRAME_RATE)?.toFloat()
        val width = format.getInteger(MediaFormat.KEY_WIDTH)
        val height = format.getInteger(MediaFormat.KEY_HEIGHT)
        val visibleWidth = format.optionalInt("crop-right")?.let {
            it + 1 - (format.optionalInt("crop-left") ?: 0)
        } ?: width
        val visibleHeight = format.optionalInt("crop-bottom")?.let {
            it + 1 - (format.optionalInt("crop-top") ?: 0)
        } ?: height
        require(visibleWidth in 1..width && visibleHeight in 1..height) { "视频尺寸或裁剪区域无效" }
        val name = context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor ->
                if (cursor.moveToFirst()) cursor.getString(0) else null
            } ?: uri.lastPathSegment ?: "视频"
        return VideoMetadata(
            uri.toString(), name, visibleWidth,
            visibleHeight, rotation,
            nominalFps?.takeIf { it.isFinite() && it > 0f }, durationUs
        )
    }

    private inline fun <T> withExtractor(uri: Uri, block: (MediaExtractor, Int) -> T): T {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(context, uri, null)
            val track = (0 until extractor.trackCount).firstOrNull {
                extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
            } ?: throw IOException("文件中没有视频轨道")
            extractor.selectTrack(track)
            return block(extractor, track)
        } finally {
            extractor.release()
        }
    }

    private fun MediaFormat.optionalInt(key: String): Int? =
        if (containsKey(key)) runCatching { getInteger(key) }.getOrNull() else null

    private fun MediaFormat.optionalLong(key: String): Long? =
        if (containsKey(key)) runCatching { getLong(key) }.getOrNull() else null

    private companion object {
        const val TIMEOUT_US = 10_000L
        const val STALL_TIMEOUT_MS = 30_000L
    }
}
