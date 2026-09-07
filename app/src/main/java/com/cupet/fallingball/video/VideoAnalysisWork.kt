package com.cupet.fallingball.video

import android.content.Context
import androidx.lifecycle.LiveData
import androidx.lifecycle.MediatorLiveData
import androidx.lifecycle.MutableLiveData
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import java.util.UUID
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

object VideoAnalysisWork {
    const val UNIQUE_WORK_NAME = "offline_video_analysis"
    private const val PREFS = "offline_video_analysis"
    private const val LAST_REQUEST = "last_request"
    private const val LAST_ID = "last_id"
    private const val TASK_REQUEST_PREFIX = "request_"
    private val runtimeCount = AtomicInteger(0)
    private val runtimeChanges = MutableLiveData(0)
    private val processorPermit = Semaphore(1)

    fun observe(context: Context): LiveData<List<WorkInfo>> {
        val work = WorkManager.getInstance(context).getWorkInfosForUniqueWorkLiveData(UNIQUE_WORK_NAME)
        return MediatorLiveData<List<WorkInfo>>().apply {
            addSource(work) { value = it }
            // Cancellation marks WorkInfo finished before the decoder has released native models.
            addSource(runtimeChanges) { work.value?.let { infos -> value = infos } }
        }
    }

    fun isRunning(infos: List<WorkInfo>): Boolean = runtimeCount.get() > 0 || infos.any { !it.state.isFinished }

    internal fun enterRuntime(checkCancellation: () -> Unit) {
        while (!processorPermit.tryAcquire(250, TimeUnit.MILLISECONDS)) checkCancellation()
        runtimeChanges.postValue(runtimeCount.incrementAndGet())
    }

    internal fun leaveRuntime() {
        runtimeChanges.postValue(runtimeCount.decrementAndGet())
        processorPermit.release()
    }

    /** Called on an IO dispatcher; KEEP also guards against overlapping UI submissions. */
    @Synchronized
    fun enqueue(context: Context, request: VideoAnalysisRequest): UUID {
        val manager = WorkManager.getInstance(context)
        val running = manager.getWorkInfosForUniqueWork(UNIQUE_WORK_NAME).get()
            .firstOrNull { !it.state.isFinished }
        if (running != null) return running.id
        val work = OneTimeWorkRequestBuilder<VideoAnalysisWorker>()
            .setInputData(request.toData())
            .addTag(UNIQUE_WORK_NAME)
            .build()
        persistTaskRequest(context, work.id, request)
        manager.enqueueUniqueWork(UNIQUE_WORK_NAME, ExistingWorkPolicy.KEEP, work).result.get()
        // This pointer is only a convenience for opening the page. The durable task-keyed
        // snapshot above is authoritative if the process exits before this update reaches disk.
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putString(LAST_REQUEST, request.toJson())
            .putString(LAST_ID, work.id.toString())
            .apply()
        return work.id
    }

    fun cancel(context: Context) {
        WorkManager.getInstance(context).cancelUniqueWork(UNIQUE_WORK_NAME)
    }

    fun lastRequest(context: Context): VideoAnalysisRequest? {
        lastId(context)?.let { id -> requestForTask(context, id)?.let { return it } }
        val encoded = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getString(LAST_REQUEST, null) ?: return null
        return runCatching { VideoAnalysisRequest.fromJson(encoded) }.getOrNull()
    }

    fun requestForTask(context: Context, taskId: UUID): VideoAnalysisRequest? {
        val preferences = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val encoded = preferences.getString("$TASK_REQUEST_PREFIX$taskId", null)
            ?: preferences.getString(LAST_REQUEST, null).takeIf { lastId(context) == taskId }
            ?: return null
        return runCatching { VideoAnalysisRequest.fromJson(encoded) }.getOrNull()
    }

    /** An orphan snapshot is harmless if scheduling fails; the last-task pointer stays intact. */
    internal fun persistTaskRequest(context: Context, taskId: UUID, request: VideoAnalysisRequest) {
        val preferences = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val editor = preferences.edit()
        val previousId = lastId(context)
        val previousRequest = preferences.getString(LAST_REQUEST, null)
        if (previousId != null && previousRequest != null &&
            !preferences.contains("$TASK_REQUEST_PREFIX$previousId")) {
            editor.putString("$TASK_REQUEST_PREFIX$previousId", previousRequest)
        }
        editor.putString("$TASK_REQUEST_PREFIX$taskId", request.toJson())
        check(editor.commit()) { "无法保存视频任务参数，分析尚未启动" }
    }

    fun lastId(context: Context): UUID? = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        .getString(LAST_ID, null)?.let { runCatching { UUID.fromString(it) }.getOrNull() }
}
