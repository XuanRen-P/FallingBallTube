package com.cupet.fallingball.learning

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.cupet.fallingball.analysis.MeasurementHistoryStore

class QuestionPreparationWorker(
    appContext: Context,
    workerParameters: WorkerParameters
) : CoroutineWorker(appContext, workerParameters) {
    override suspend fun doWork(): Result {
        val measurementId = inputData.getString(KEY_MEASUREMENT_ID) ?: return Result.failure()
        val record = MeasurementHistoryStore(
            applicationContext.filesDir.resolve("measurement_history")
        ).load(measurementId) ?: return Result.failure()
        if (record.samples.isEmpty()) return Result.success()
        val chart = runCatching {
            ChartImageEncoder.encodeRecord(applicationContext, record)
        }.getOrNull()
        return runCatching {
            LearningServiceFactory.get(applicationContext).tutor
                .prepareQuestions(record, chart)
        }.fold(
            onSuccess = { Result.success() },
            onFailure = {
                if (shouldRetryQuestionPreparation(runAttemptCount)) {
                    Result.retry()
                } else {
                    Result.failure()
                }
            }
        )
    }

    companion object {
        const val KEY_MEASUREMENT_ID = "measurement_id"
        const val MAX_TOTAL_ATTEMPTS = 3
    }
}

internal fun shouldRetryQuestionPreparation(
    runAttemptCount: Int,
    maxTotalAttempts: Int = QuestionPreparationWorker.MAX_TOTAL_ATTEMPTS
): Boolean = runAttemptCount + 1 < maxTotalAttempts
