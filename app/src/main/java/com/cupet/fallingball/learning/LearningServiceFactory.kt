package com.cupet.fallingball.learning

import android.content.Context
import androidx.work.Constraints
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf
import com.cupet.fallingball.R

object LearningServiceFactory {
    @Volatile
    private var services: LearningServices? = null

    fun get(context: Context): LearningServices = services ?: synchronized(this) {
        services ?: create(context.applicationContext).also { services = it }
    }

    private fun create(context: Context): LearningServices {
        val knowledge = context.resources.openRawResource(R.raw.learning_knowledge_cards).use {
            KnowledgeIndex.fromJson(it)
        }
        val store = LearningStore(context.filesDir.resolve("learning"))
        // One HTTP client keeps the connection pool and dispatcher shared, so the follow-up
        // request of a tool turn reuses the TLS connection opened by the first request.
        val httpClient = AliyunResponsesAdapter.defaultClient()
        val gateway = AliyunResponsesAdapter(
            baseUrl = context.getString(R.string.aliyun_openai_base_url),
            apiKey = context.getString(R.string.aliyun_api_key),
            client = httpClient
        )
        val summaryGateway = AliyunResponsesAdapter(
            baseUrl = context.getString(R.string.aliyun_openai_base_url),
            apiKey = context.getString(R.string.aliyun_api_key),
            client = httpClient
        )
        val questionGateway = AliyunResponsesAdapter(
            baseUrl = context.getString(R.string.aliyun_openai_base_url),
            apiKey = context.getString(R.string.aliyun_api_key),
            client = httpClient
        )
        return LearningServices(
            store = store,
            knowledgeIndex = knowledge,
            tutor = AliyunLearningTutor(
                gateway = gateway,
                store = store,
                knowledgeIndex = knowledge,
                summaryGateway = summaryGateway,
                questionGateway = questionGateway
            )
        )
    }
}

data class LearningServices(
    val store: LearningStore,
    val knowledgeIndex: KnowledgeIndex,
    val tutor: LearningTutor
)

object QuestionWorkScheduler {
    /**
     * Only the most recently requested measurement drafts questions. Browsing a dozen records
     * used to queue a dozen chart-image model calls that competed with the student's own turn,
     * so the queue is intentionally collapsed to a single slot.
     */
    fun enqueue(context: Context, measurementId: String) {
        val request = OneTimeWorkRequestBuilder<QuestionPreparationWorker>()
            .setInputData(workDataOf(QuestionPreparationWorker.KEY_MEASUREMENT_ID to measurementId))
            .setConstraints(
                Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build()
            )
            .addTag(TAG)
            .build()
        // Suggestion drafting is best-effort: a missing scheduler must never break the caller.
        runCatching {
            WorkManager.getInstance(context).enqueueUniqueWork(
                WORK_NAME,
                ExistingWorkPolicy.REPLACE,
                request
            )
        }.onFailure {
            LearningLog.warn("question drafting is unavailable", it)
            return
        }
        preferences(context).edit().putString(KEY_PENDING_MEASUREMENT, measurementId).apply()
    }

    /** Cancels the queued draft only when it belongs to the measurement being deleted. */
    fun cancel(context: Context, measurementId: String) {
        if (preferences(context).getString(KEY_PENDING_MEASUREMENT, null) != measurementId) return
        cancelPending(context)
    }

    /** Drops the queued draft; used once the student starts a real conversation instead. */
    fun cancelPending(context: Context) {
        runCatching { WorkManager.getInstance(context).cancelUniqueWork(WORK_NAME) }
            .onFailure { LearningLog.warn("cannot cancel question drafting", it) }
        preferences(context).edit().remove(KEY_PENDING_MEASUREMENT).apply()
    }

    private fun preferences(context: Context) =
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)

    private const val WORK_NAME = "learning-question-preparation"
    private const val TAG = "learning-question-preparation"
    private const val PREFERENCES = "learning_question_work"
    private const val KEY_PENDING_MEASUREMENT = "pending_measurement_id"
}
