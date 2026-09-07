package com.cupet.fallingball.learning

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.io.InputStreamReader
import java.io.InterruptedIOException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

sealed interface AliyunStreamEvent {
    data object ReasoningStarted : AliyunStreamEvent
    data class ReasoningDelta(val delta: String) : AliyunStreamEvent
    data class TextDelta(val delta: String) : AliyunStreamEvent
    data class OutputItemDone(val item: JSONObject) : AliyunStreamEvent
    data class Completed(val response: JSONObject) : AliyunStreamEvent
}

class AliyunHttpException(
    val statusCode: Int,
    val retryAfterSeconds: Long?,
    message: String
) : IOException(message)

class AliyunTimeoutException(cause: IOException) :
    IOException("Alibaba Cloud request timed out", cause)

class AliyunResponseException(message: String) : IOException(message)

interface ResponsesGateway {
    fun stream(requestBody: JSONObject): Flow<AliyunStreamEvent>
    fun cancelActive()
}

/** OpenAI-compatible Alibaba Cloud Responses API with bounded retries and SSE streaming. */
class AliyunResponsesAdapter(
    baseUrl: String,
    private val apiKey: String,
    private val client: OkHttpClient = defaultClient(),
    private val maxAttempts: Int = 2
) : ResponsesGateway {
    private val endpoint = "${baseUrl.trimEnd('/')}/responses"
    private val activeCall = AtomicReference<Call?>()
    private val cancellationGeneration = AtomicLong(0L)

    init {
        require(endpoint.startsWith("https://") || endpoint.startsWith("http://localhost") ||
            endpoint.startsWith("http://127.0.0.1")) { "Responses endpoint must use HTTPS" }
        require(maxAttempts in 1..3) { "Invalid retry count" }
    }

    override fun stream(requestBody: JSONObject): Flow<AliyunStreamEvent> = flow {
        val generation = cancellationGeneration.get()
        var attempt = 0
        var emittedAny = false
        while (true) {
            attempt += 1
            var call: Call? = null
            try {
                val request = Request.Builder()
                    .url(endpoint)
                    .header("Authorization", "Bearer $apiKey")
                    .header("Accept", "text/event-stream")
                    .post(requestBody.toString().toRequestBody(JSON_MEDIA_TYPE))
                    .build()
                call = client.newCall(request)
                activeCall.set(call)
                val response = withContext(Dispatchers.IO) { call.execute() }
                response.use { value ->
                    if (!value.isSuccessful) {
                        val errorBody = value.body.string().take(MAX_ERROR_BODY_CHARS)
                        throw AliyunHttpException(
                            statusCode = value.code,
                            retryAfterSeconds = value.header("Retry-After")?.toLongOrNull(),
                            message = safeErrorMessage(value.code, errorBody)
                        )
                    }
                    val parser = SseEventParser()
                    val reader = InputStreamReader(value.body.byteStream(), Charsets.UTF_8)
                    val buffer = CharArray(READ_CHUNK_CHARS)
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        val count = withContext(Dispatchers.IO) { reader.read(buffer) }
                        if (count < 0) break
                        val chunk = String(buffer, 0, count)
                        parser.feed(chunk).forEach { payload ->
                            parsePayload(payload)?.let { event ->
                                emittedAny = true
                                emit(event)
                            }
                        }
                    }
                    parser.finish().forEach { payload ->
                        parsePayload(payload)?.let { event ->
                            emittedAny = true
                            emit(event)
                        }
                    }
                }
                return@flow
            } catch (cancelled: CancellationException) {
                call?.cancel()
                throw cancelled
            } catch (error: IOException) {
                if (cancellationGeneration.get() != generation) {
                    throw IOException("Alibaba Cloud request was cancelled", error)
                }
                val retryableStatus = when (error) {
                    is AliyunResponseException -> false
                    is AliyunHttpException ->
                        error.statusCode == 429 || error.statusCode in 500..599
                    else -> true
                }
                if (emittedAny || !retryableStatus || attempt >= maxAttempts) {
                    if (error is InterruptedIOException) {
                        throw AliyunTimeoutException(error)
                    }
                    throw error
                }
                val retryAfterMs = (error as? AliyunHttpException)?.retryAfterSeconds
                    ?.coerceIn(0L, MAX_RETRY_AFTER_SECONDS)
                    ?.times(1_000L)
                delay(retryAfterMs ?: (BASE_RETRY_DELAY_MS * attempt))
            } finally {
                if (call != null) activeCall.compareAndSet(call, null)
            }
        }
    }

    override fun cancelActive() {
        cancellationGeneration.incrementAndGet()
        activeCall.getAndSet(null)?.cancel()
    }

    private fun parsePayload(payload: String): AliyunStreamEvent? {
        if (payload == "[DONE]") return null
        val json = runCatching { JSONObject(payload) }.getOrElse {
            throw IOException("Alibaba Cloud returned malformed SSE data")
        }
        return when (json.optString("type")) {
            "response.output_item.added" -> {
                val item = json.optJSONObject("item")
                if (item?.optString("type") == "reasoning") {
                    AliyunStreamEvent.ReasoningStarted
                } else {
                    null
                }
            }
            "response.reasoning_summary_text.delta" ->
                AliyunStreamEvent.ReasoningDelta(json.optString("delta"))
            "response.reasoning_text.delta" ->
                AliyunStreamEvent.ReasoningDelta(json.optString("delta"))
            "response.output_text.delta" -> AliyunStreamEvent.TextDelta(json.optString("delta"))
            "response.output_item.done" -> json.optJSONObject("item")
                ?.let(AliyunStreamEvent::OutputItemDone)
            "response.completed" -> AliyunStreamEvent.Completed(
                json.optJSONObject("response") ?: json
            )
            "response.failed", "response.incomplete" -> {
                val response = json.optJSONObject("response")
                val error = response?.optJSONObject("error") ?: json.optJSONObject("error")
                throw AliyunResponseException(sanitize(
                    error?.optString("message")?.take(MAX_ERROR_BODY_CHARS)
                        ?.ifBlank { null } ?: "Alibaba Cloud response failed"
                ))
            }
            else -> null
        }
    }

    private fun safeErrorMessage(code: Int, body: String): String {
        val parsed = runCatching { JSONObject(body) }.getOrNull()
        val message = parsed?.optJSONObject("error")?.optString("message")
            ?: parsed?.optString("message")
        return sanitize(
            "Alibaba Cloud HTTP $code" +
                message?.takeIf(String::isNotBlank)
                    ?.let { ": ${it.take(MAX_ERROR_BODY_CHARS)}" }.orEmpty()
        )
    }

    private fun sanitize(value: String): String =
        if (apiKey.isBlank()) value else value.replace(apiKey, "[redacted]")

    companion object {
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
        private const val READ_CHUNK_CHARS = 4_096
        private const val MAX_ERROR_BODY_CHARS = 1_000
        private const val BASE_RETRY_DELAY_MS = 250L
        private const val MAX_RETRY_AFTER_SECONDS = 10L

        fun defaultClient(): OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(20, TimeUnit.SECONDS)
            .readTimeout(5, TimeUnit.MINUTES)
            .writeTimeout(30, TimeUnit.SECONDS)
            .callTimeout(5, TimeUnit.MINUTES)
            .retryOnConnectionFailure(true)
            .build()
    }
}

/** Incremental SSE parser that tolerates arbitrary byte-to-string fragmentation and CRLF. */
internal class SseEventParser {
    private val pending = StringBuilder()
    private val dataLines = ArrayList<String>()

    fun feed(chunk: String): List<String> {
        pending.append(chunk)
        val output = ArrayList<String>()
        while (true) {
            val lineEnd = pending.indexOf("\n")
            if (lineEnd < 0) break
            val line = pending.substring(0, lineEnd).removeSuffix("\r")
            pending.delete(0, lineEnd + 1)
            consumeLine(line, output)
        }
        return output
    }

    fun finish(): List<String> {
        val output = ArrayList<String>()
        if (pending.isNotEmpty()) {
            consumeLine(pending.toString().removeSuffix("\r"), output)
            pending.clear()
        }
        flush(output)
        return output
    }

    private fun consumeLine(line: String, output: MutableList<String>) {
        if (line.isEmpty()) {
            flush(output)
            return
        }
        if (line.startsWith(":")) return
        if (line.startsWith("data:")) dataLines += line.substring(5).trimStart()
    }

    private fun flush(output: MutableList<String>) {
        if (dataLines.isEmpty()) return
        output += dataLines.joinToString("\n")
        dataLines.clear()
    }
}
