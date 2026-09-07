package com.cupet.fallingball.video

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import androidx.work.Data
import androidx.work.ForegroundInfo
import androidx.work.WorkManager
import androidx.work.Worker
import androidx.work.WorkerParameters
import com.cupet.fallingball.R
import com.cupet.fallingball.analysis.MeasurementHistoryStore
import com.cupet.fallingball.analysis.MeasurementRecord
import com.cupet.fallingball.analysis.MeasurementSource
import com.cupet.fallingball.analysis.VideoMeasurementSource
import com.cupet.fallingball.camera.MeasurementFrameProcessor
import com.cupet.fallingball.vision.ModelProfile
import com.cupet.fallingball.vision.ObbDetector
import com.cupet.fallingball.vision.CpuAffinity
import kotlinx.coroutines.CancellationException

class VideoAnalysisWorker(context: Context, parameters: WorkerParameters) :
    Worker(context, parameters) {
    private val commitLock = Any()
    private var wroteRecord = false
    private val history = MeasurementHistoryStore(context.filesDir.resolve("measurement_history"))

    override fun doWork(): Result {
        var detector: ObbDetector? = null
        var processor: MeasurementFrameProcessor? = null
        var ownsRuntime = false
        val updates = PendingWorkerUpdates()
        try {
            val request = VideoAnalysisRequest.fromData(inputData)
            // A process may die after the atomic save but before WorkManager commits success.
            if (history.load(id.toString()) != null) return successData()
            // Confirm foreground-service creation once before starting expensive work. Later
            // notification/progress writes run asynchronously and are drained before completion.
            updates.track(setForegroundAsync(foreground(
                applicationContext.getString(R.string.video_phase_initializing), 0)))
            updates.awaitAll()
            updates.track(setProgressAsync(progressData(PHASE_INITIALIZING, 0L, 0L, 0L, 0)))
            if (isStopped) throw CancellationException("视频分析已取消")
            VideoAnalysisWork.enterRuntime {
                if (isStopped) throw CancellationException("视频分析已取消")
            }
            ownsRuntime = true
            if (isStopped) throw CancellationException("视频分析已取消")
            // WorkManager can restore a task without ever creating MainActivity.
            CpuAffinity.pinProcessToFastCores()
            detector = ObbDetector(applicationContext, ModelProfile.ACCURATE_640)
            val engine = MeasurementFrameProcessor(applicationContext, detector, offline = true)
            processor = engine
            val decoder = VideoFrameDecoder(applicationContext)
            val metadata = decoder.metadata(android.net.Uri.parse(request.uri))
            engine.configureOffline(
                config = request.rulerConfig,
                defaults = request.defaults,
                resolutionLabel = "${metadata.width}x${metadata.height}",
                runId = id.toString()
            )
            var inferredFrames = 0L
            var rulerWasValid = false
            var lastUpdateMs = 0L
            val processingStartedMs = SystemClock.elapsedRealtime()
            fun checkCancellation() {
                if (isStopped) throw CancellationException("视频分析已取消")
                updates.checkCompleted()
            }
            val decoded = decoder.decode(request, ::checkCancellation) { frame, ptsUs, count ->
                checkCancellation()
                val outcome = engine.processOfflineFrame(frame)
                if (outcome.ballInferenceCompleted) inferredFrames++
                rulerWasValid = rulerWasValid || outcome.rulerValid
                val now = SystemClock.elapsedRealtime()
                if (now - lastUpdateMs >= PROGRESS_INTERVAL_MS || count == 1L) {
                    lastUpdateMs = now
                    val percent = ((ptsUs - request.clipStartUs).toDouble() * 95.0 /
                        (request.clipEndUs - request.clipStartUs)).toInt().coerceIn(0, 95)
                    val phase = if (rulerWasValid) PHASE_ANALYZING else PHASE_CALIBRATING
                    val runtime = engine.ballTrackingDiagnostics()?.runtime
                    val progress = Data.Builder().putAll(progressData(
                        phase, ptsUs - request.clipStartUs, count, inferredFrames, percent))
                        .putDouble(KEY_PROCESSING_FPS,
                            count * 1000.0 / (now - processingStartedMs).coerceAtLeast(1L))
                        .putBoolean(KEY_NPU_ACTIVE, runtime?.computeUnit == "NPU" && runtime.placementVerified)
                        .build()
                    updates.track(setProgressAsync(progress))
                    val label = applicationContext.getString(
                        if (rulerWasValid) R.string.video_phase_analyzing else R.string.video_phase_calibrating
                    )
                    updates.track(setForegroundAsync(foreground("$label · $count 帧", percent)))
                }
            }
            checkCancellation()
            updates.track(setProgressAsync(progressData(PHASE_FINALIZING, request.clipEndUs - request.clipStartUs,
                decoded.processedFrameCount, inferredFrames, 96)))
            updates.track(setForegroundAsync(foreground(applicationContext.getString(R.string.video_phase_finalizing), 96)))
            val session = engine.finishOffline()
            checkCancellation()
            val decodedMetadata = decoded.metadata
            val record = MeasurementRecord.fromSession(
                session = session,
                recordedAtEpochMs = request.requestedAtEpochMs,
                id = id.toString(),
                viscometryParameters = request.defaults.viscometryParameters,
                metadata = request.defaults.metadataSnapshot()
            ).copy(
                resolutionLabel = "${decodedMetadata.width}x${decodedMetadata.height}",
                source = MeasurementSource.IMPORTED_VIDEO,
                videoSource = VideoMeasurementSource(
                    sourceName = request.videoName,
                    sourceUri = request.uri,
                    width = decodedMetadata.width,
                    height = decodedMetadata.height,
                    rotationDegrees = decodedMetadata.rotationDegrees,
                    nominalFrameRate = decodedMetadata.nominalFrameRate,
                    durationUs = decodedMetadata.durationUs,
                    clipStartUs = request.clipStartUs,
                    clipEndUs = request.clipEndUs,
                    decodedFrameCount = decoded.decodedFrameCount,
                    processedFrameCount = decoded.processedFrameCount,
                    inferenceFrameCount = inferredFrames,
                    taskId = id.toString()
                )
            )
            updates.track(setProgressAsync(progressData(PHASE_SAVING, request.clipEndUs - request.clipStartUs,
                decoded.processedFrameCount, inferredFrames, 99)))
            // A foreground/progress failure must be reported before an idempotent record exists.
            updates.awaitAll()
            synchronized(commitLock) {
                checkCancellation()
                history.save(record)
                wroteRecord = true
                if (isStopped) {
                    history.delete(id.toString())
                    wroteRecord = false
                    throw CancellationException("视频分析已取消")
                }
            }
            return successData()
        } catch (cancelled: CancellationException) {
            removeCancelledRecord()
            throw cancelled
        } catch (memory: OutOfMemoryError) {
            return Result.failure(Data.Builder().putString(KEY_ERROR, "内存不足，无法按原视频分辨率完成分析").build())
        } catch (failure: Exception) {
            if (isStopped) {
                removeCancelledRecord()
                throw CancellationException("视频分析已取消", failure)
            }
            val message = when (failure) {
                is SecurityException -> "视频读取权限已失效，请重新选择视频"
                else -> failure.message?.take(1_000) ?: "视频分析失败，请检查文件及片段内容"
            }
            return Result.failure(Data.Builder().putString(KEY_ERROR, message).build())
        } finally {
            // Error and cancellation paths must not leave updates racing WorkManager completion.
            runCatching { updates.awaitAll() }
            runCatching { processor?.closeOffline() }
            runCatching { detector?.close() }
            if (ownsRuntime) VideoAnalysisWork.leaveRuntime()
        }
    }

    override fun onStopped() {
        super.onStopped()
        removeCancelledRecord()
    }

    private fun removeCancelledRecord() = synchronized(commitLock) {
        if (wroteRecord) {
            history.delete(id.toString())
            wroteRecord = false
        }
    }

    private fun successData(): Result = Result.success(
        Data.Builder().putString(KEY_RECORD_ID, id.toString()).putInt(KEY_PERCENT, 100).build()
    )

    private fun foreground(label: String, percent: Int): ForegroundInfo {
        val manager = applicationContext.getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(
            CHANNEL_ID, applicationContext.getString(R.string.video_notification_channel), NotificationManager.IMPORTANCE_LOW
        ))
        val open = PendingIntent.getActivity(applicationContext, 0,
            Intent(applicationContext, VideoAnalysisActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_search)
            .setContentTitle(applicationContext.getString(R.string.video_title))
            .setContentText(label)
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setProgress(100, percent, percent == 0)
            .addAction(android.R.drawable.ic_menu_close_clear_cancel,
                applicationContext.getString(android.R.string.cancel),
                WorkManager.getInstance(applicationContext).createCancelPendingIntent(id))
            .build()
        val type = when {
            Build.VERSION.SDK_INT >= 35 -> ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING
            Build.VERSION.SDK_INT >= 29 -> ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            else -> 0
        }
        return ForegroundInfo(NOTIFICATION_ID, notification, type)
    }

    companion object {
        const val KEY_PHASE = "phase"
        const val KEY_PROCESSED_US = "processed_us"
        const val KEY_FRAME_COUNT = "frame_count"
        const val KEY_INFERRED_COUNT = "inferred_count"
        const val KEY_PERCENT = "percent"
        const val KEY_PROCESSING_FPS = "processing_fps"
        const val KEY_NPU_ACTIVE = "npu_active"
        const val KEY_RECORD_ID = "record_id"
        const val KEY_ERROR = "error"
        const val PHASE_INITIALIZING = "initializing"
        const val PHASE_CALIBRATING = "calibrating"
        const val PHASE_ANALYZING = "analyzing"
        const val PHASE_FINALIZING = "finalizing"
        const val PHASE_SAVING = "saving"
        private const val CHANNEL_ID = "offline_video_analysis"
        private const val NOTIFICATION_ID = 2402
        // WorkManager progress is persisted in its database. Updating twice per second can
        // contend with the decoder on high frame rate clips; one update per second is ample for
        // a user facing progress indicator and keeps the analysis loop focused on inference.
        private const val PROGRESS_INTERVAL_MS = 1_000L

        private fun progressData(phase: String, processedUs: Long, count: Long, inferred: Long, percent: Int): Data =
            Data.Builder().putString(KEY_PHASE, phase).putLong(KEY_PROCESSED_US, processedUs)
                .putLong(KEY_FRAME_COUNT, count).putLong(KEY_INFERRED_COUNT, inferred)
                .putInt(KEY_PERCENT, percent).build()
    }
}
