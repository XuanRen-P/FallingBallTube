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
        val gateway = AliyunResponsesAdapter(
            baseUrl = context.getString(R.string.aliyun_openai_base_url),
            apiKey = context.getString(R.string.aliyun_api_key)
        )
        val summaryGateway = AliyunResponsesAdapter(
            baseUrl = context.getString(R.string.aliyun_openai_base_url),
            apiKey = context.getString(R.string.aliyun_api_key)
        )
        return LearningServices(
            store = store,
            knowledgeIndex = knowledge,
            tutor = AliyunLearningTutor(
                gateway = gateway,
                store = store,
                knowledgeIndex = knowledge,
                summaryGateway = summaryGateway
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
    fun enqueue(context: Context, measurementId: String, replace: Boolean = false) {
        val request = OneTimeWorkRequestBuilder<QuestionPreparationWorker>()
            .setInputData(workDataOf(QuestionPreparationWorker.KEY_MEASUREMENT_ID to measurementId))
            .setConstraints(
                Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build()
            )
            .addTag(TAG)
            .build()
        WorkManager.getInstance(context).enqueueUniqueWork(
            workName(measurementId),
            if (replace) ExistingWorkPolicy.REPLACE else ExistingWorkPolicy.KEEP,
            request
        )
    }

    fun cancel(context: Context, measurementId: String) {
        WorkManager.getInstance(context).cancelUniqueWork(workName(measurementId))
    }

    private fun workName(measurementId: String) = "learning-questions-$measurementId"

    private const val TAG = "learning-question-preparation"
}
