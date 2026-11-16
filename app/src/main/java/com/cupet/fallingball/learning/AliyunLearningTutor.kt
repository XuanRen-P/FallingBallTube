package com.cupet.fallingball.learning

import com.cupet.fallingball.analysis.MeasurementRecord
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeout
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean

class ToolTurnBudget(private val maximum: Int = MAX_TOOL_CALLS) {
    var used: Int = 0
        private set

    fun tryConsume(): Boolean {
        if (used >= maximum) return false
        used += 1
        return true
    }

    val exhausted: Boolean get() = used >= maximum

    companion object { const val MAX_TOOL_CALLS = 10 }
}

class AliyunLearningTutor(
    private val gateway: ResponsesGateway,
    private val store: LearningStore,
    private val knowledgeIndex: KnowledgeIndex,
    private val clock: () -> Long = System::currentTimeMillis,
    private val turnTimeoutMs: Long = TURN_TIMEOUT_MS,
    private val questionTimeoutMs: Long = QUESTION_TIMEOUT_MS,
    private val summaryGateway: ResponsesGateway? = null,
    private val questionGateway: ResponsesGateway = gateway,
    private val backgroundScope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
) : LearningTutor {
    private val toolExecutor = LearningToolExecutor(knowledgeIndex)
    private val contextBuilder = LearningContextBuilder(knowledgeIndex)
    private val sessionMeasurements = ConcurrentHashMap<String, String>()
    private val stopRequested = AtomicBoolean(false)
    private val generationMutex = Mutex()
    /** Question drafting is background work and must never block a student turn. */
    private val questionMutex = Mutex()
    private val summaryMutex = Mutex()
    private val summaryJobs = ConcurrentHashMap<String, Job>()
    private val titleJobs = ConcurrentHashMap<String, Job>()

    override suspend fun prepareQuestions(
        record: MeasurementRecord,
        chartPngBase64: String?
    ): PreparedQuestions = questionMutex.withLock {
        val snapshot = LearningMeasurementSnapshot.fromRecord(record)
        store.loadQuestions(record.id, snapshot.fingerprint)?.let { return@withLock it }
        val questions = generateQuestions(snapshot, chartPngBase64)
        PreparedQuestions(
            measurementId = record.id,
            fingerprint = snapshot.fingerprint,
            questions = questions,
            generatedAtEpochMs = clock()
        ).also(store::saveQuestions)
    }

    override suspend fun openSession(record: MeasurementRecord, sessionId: String?): LearningSession {
        val opened = store.openSession(record, sessionId)
        val active = if (opened.status == LearningSessionStatus.PAUSED) {
            store.activateSession(opened.measurementId, opened.id)
        } else {
            opened
        }
        sessionMeasurements[active.id] = active.measurementId
        return active
    }

    override suspend fun startNewSession(record: MeasurementRecord): LearningSession =
        store.createSession(LearningMeasurementSnapshot.fromRecord(record)).also {
            sessionMeasurements[it.id] = it.measurementId
        }

    override fun send(
        sessionId: String,
        message: String,
        chartPngBase64: String?,
        attachments: List<LearningInputAttachment>,
        turnIdentity: LearningTurnIdentity
    ): Flow<LearningEvent> = flow {
        val userText = message.trim()
        require(userText.isNotEmpty()) { "Message cannot be empty" }
        gateway.cancelActive()
        generationMutex.withLock {
        val measurementId = sessionMeasurements[sessionId]
        if (measurementId == null) {
            emit(LearningEvent.Error("学习会话已不可用", recoverable = false))
            return@withLock
        }
        val original = store.loadSession(measurementId, sessionId)
        if (original == null) {
            sessionMeasurements.remove(sessionId)
            emit(LearningEvent.Error("学习会话已被删除", recoverable = false))
            return@withLock
        }
        if (original.status != LearningSessionStatus.ACTIVE) {
            emit(LearningEvent.Error("学习会话已结束", recoverable = false))
            return@withLock
        }
        stopRequested.set(false)
        val turnId = turnIdentity.turnId
        val userMessage = LearningMessage(
            id = turnIdentity.userMessageId,
            role = LearningMessageRole.USER,
            text = userText,
            attachmentNames = attachments.map { it.name.take(120) },
            turnId = turnId,
            createdAtEpochMs = clock()
        )
        val assistantId = turnIdentity.assistantMessageId
        val provisionalTitle = if (original.messages.isEmpty() &&
            (original.title == original.snapshot.recordName || original.title == "本次数据学习")
        ) {
            userText.replace(Regex("\\s+"), " ").take(30).ifBlank { original.title }
        } else {
            null
        }
        val titledOriginal = provisionalTitle?.let {
            store.applyProvisionalTitle(
                original.measurementId,
                original.id,
                original.title,
                it
            )
        } ?: original
        var session = titledOriginal.copy(
            updatedAtEpochMs = clock(),
            messages = titledOriginal.messages + userMessage + LearningMessage(
                id = assistantId,
                role = LearningMessageRole.ASSISTANT,
                text = "",
                turnId = turnId,
                createdAtEpochMs = clock(),
                state = LearningMessageState.STREAMING
            )
        )
        store.saveSession(session)
        val input = contextBuilder.build(titledOriginal, userText, chartPngBase64, attachments)
        val firstTutorTurn = titledOriginal.messages.none {
            it.role == LearningMessageRole.ASSISTANT &&
                it.state == LearningMessageState.COMPLETE
        }
        // Evidence already carried by the developer context is not re-fetched through a tool.
        val readEvidenceSections = priorEvidenceSections(titledOriginal) +
            LearningContextBuilder.prefilledEvidenceSections(titledOriginal.snapshot)
        var pendingEvidenceSection = MeasurementEvidencePolicy.requiredSection(
            userText,
            readEvidenceSections
        )
        val turnContract = SocraticPrompt.turnContract(firstTutorTurn, pendingEvidenceSection)
        val budget = ToolTurnBudget()
        val reasoning = StringBuilder()
        val text = StringBuilder()
        val reusableToolResults = reusableToolResults(titledOriginal)
        var lastPersistedLength = 0
        var allowWebTools = false
        var allowKnowledgeSearchTool = !knowledgeIndex.hasSufficientCoverage(userText)
        var allowMeasurementEvidenceTool =
            MeasurementEvidencePolicy.hasUnreadSections(readEvidenceSections)
        var forceToolFreeCompletion = false
        try {
            withTimeout(turnTimeoutMs) {
                val currentUser = input.remove(input.length() - 1)
                input.put(JSONObject().put("role", "developer").put("content", turnContract))
                input.put(currentUser)
                while (true) {
                    throwIfStopRequested()
                    val toolCalls = ArrayList<FunctionCall>()
                    var repeatedReadOnlyCall = false
                    val tools = when {
                        budget.exhausted || forceToolFreeCompletion -> JSONArray()
                        pendingEvidenceSection != null -> measurementEvidenceTools()
                        else -> tools(
                            allowWebTools,
                            allowMeasurementEvidenceTool,
                            allowKnowledgeSearchTool
                        )
                    }
                    val toolChoice: Any = when {
                        budget.exhausted || forceToolFreeCompletion -> "none"
                        // qwen3.7-plus rejects required/object tool_choice in thinking mode; the
                        // turn contract already asks for the missing evidence section first.
                        else -> "auto"
                    }
                    val body = responseBody(
                        input = input,
                        reasoningEffort = "high",
                        tools = tools,
                        toolChoice = toolChoice,
                        maxOutputTokens = ANSWER_MAX_OUTPUT_TOKENS
                    )
                    gateway.stream(body).collect { event ->
                        when (event) {
                            AliyunStreamEvent.ReasoningStarted -> emit(LearningEvent.ReasoningStarted)
                            is AliyunStreamEvent.ReasoningDelta -> if (event.delta.isNotEmpty()) {
                                reasoning.append(event.delta)
                                emit(LearningEvent.ReasoningDelta(event.delta))
                            }
                            is AliyunStreamEvent.TextDelta -> if (event.delta.isNotEmpty()) {
                                text.append(event.delta)
                                emit(LearningEvent.TextDelta(event.delta))
                                if (text.length - lastPersistedLength >= STREAM_PERSIST_CHARS) {
                                    session = session.withAssistantText(
                                        assistantId,
                                        text.toString(),
                                        reasoning.toString(),
                                        LearningMessageState.STREAMING,
                                        clock()
                                    )
                                    store.saveSession(session)
                                    lastPersistedLength = text.length
                                }
                            }
                            is AliyunStreamEvent.OutputItemDone -> {
                                val item = event.item
                                when (item.optString("type")) {
                                    "function_call" -> functionCall(item)?.let(toolCalls::add)
                                    "web_search_call", "web_extractor_call", "mcp_call",
                                    "web_search_image_call", "image_search_call", "file_search_call",
                                    "code_interpreter_call" -> {
                                        if (budget.tryConsume()) {
                                            val action = completedBuiltinAction(item, turnId)
                                            session = session.copy(
                                                actions = session.actions + action,
                                                updatedAtEpochMs = clock()
                                            )
                                            store.saveSession(session)
                                            emit(LearningEvent.ActionCompleted(action))
                                        }
                                        webReferences(item, turnId).forEach { reference ->
                                            session = session.copy(
                                                references = (session.references + reference)
                                                    .distinctBy { it.url ?: "${it.source}:${it.title}" }
                                            )
                                            emit(LearningEvent.Citation(reference))
                                        }
                                    }
                                }
                            }
                            is AliyunStreamEvent.Completed -> Unit
                        }
                    }
                    throwIfStopRequested()
                    if (toolCalls.isEmpty()) break
                    for (call in toolCalls) {
                        throwIfStopRequested()
                        val callKey = reusableToolKey(call.name, call.arguments)
                        if (callKey != null && reusableToolResults.containsKey(callKey)) {
                            // Reuse read-only evidence already obtained for this immutable
                            // snapshot, but still acknowledge the new call id to Responses.
                            input.put(call.toInputItem())
                            input.put(
                                JSONObject().put("type", "function_call_output")
                                    .put("call_id", call.callId)
                                    .put("output", reusableToolResults.getValue(callKey))
                            )
                            repeatedReadOnlyCall = true
                            if (call.name == LearningToolExecutor.GET_MEASUREMENT_EVIDENCE) {
                                pendingEvidenceSection = null
                                allowMeasurementEvidenceTool = false
                            }
                            continue
                        }
                        if (!budget.tryConsume()) break
                        val action = LearningAction(
                            turnId = turnId,
                            toolName = call.name,
                            label = toolLabel(call.name),
                            callId = call.callId,
                            arguments = call.arguments.take(LearningContextBuilder.MAX_TOOL_ARGUMENT_CHARS),
                            startedAtEpochMs = clock()
                        )
                        session = session.copy(actions = session.actions + action, updatedAtEpochMs = clock())
                        store.saveSession(session)
                        emit(LearningEvent.ActionStarted(action))

                        val execution = toolExecutor.execute(session, turnId, call.name, call.arguments)
                        throwIfStopRequested()
                        session = execution.session
                        val completedAction = action.copy(
                            completedAtEpochMs = clock(),
                            state = if (JSONObject(execution.output).has("error")) {
                                LearningActionState.ERROR
                            } else {
                                LearningActionState.COMPLETE
                            },
                            resultSummary = execution.summary,
                            output = execution.output.take(LearningContextBuilder.MAX_TOOL_OUTPUT_CHARS)
                        )
                        session = session.copy(
                            actions = session.actions.map {
                                if (it.id == action.id) completedAction else it
                            },
                            updatedAtEpochMs = clock()
                        )
                        execution.references.forEach { emit(LearningEvent.Citation(it)) }
                        store.saveSession(session)
                        emit(LearningEvent.ActionCompleted(completedAction))
                        input.put(call.toInputItem())
                        input.put(
                            JSONObject().put("type", "function_call_output")
                                .put("call_id", call.callId)
                                .put("output", execution.output)
                        )
                        if (callKey != null) reusableToolResults[callKey] = execution.output
                        if (call.name == LearningToolExecutor.SEARCH_COURSE_KNOWLEDGE) {
                            allowKnowledgeSearchTool = false
                            allowWebTools = runCatching {
                                !JSONObject(execution.output).getBoolean("local_retrieval_sufficient")
                            }.getOrDefault(false)
                        }
                        if (call.name == LearningToolExecutor.GET_MEASUREMENT_EVIDENCE) {
                            pendingEvidenceSection = null
                            allowMeasurementEvidenceTool = false
                            input.put(JSONObject().put("role", "developer").put(
                                "content",
                                "本轮所需的测量证据已在上一步取得，请直接完成本轮回答，不要重复读取同一 section。"
                            ))
                        }
                    }
                    if (repeatedReadOnlyCall) {
                        forceToolFreeCompletion = true
                        input.put(JSONObject().put("role", "developer").put(
                            "content",
                            "该冻结测量证据已经在历史中取得并已回放。不得再次调用相同只读工具；请只根据现有证据完成本轮苏格拉底式引导。"
                        ))
                    }
                    if (budget.exhausted) {
                        input.put(JSONObject().put("role", "developer").put(
                            "content",
                            "本轮已达到 10 次本地工具调用上限。不得继续调用工具；请只根据已取得证据提出本轮最关键的一个引导问题。"
                        ))
                    }
                }
            }
            session = session.withAssistantText(
                assistantId,
                text.toString().ifBlank { "我们先从一个最关键的数据现象开始：你认为图中哪一段最接近匀速？" },
                reasoning.toString(),
                LearningMessageState.COMPLETE,
                clock()
            ).withRollingSummary()
            store.saveSession(session)
            scheduleTopicTitle(session)
            emit(LearningEvent.Completed(session))
            scheduleRollingSummary(session)
        } catch (timeout: TimeoutCancellationException) {
            gateway.cancelActive()
            // A partial stream may contain a leaked or syntactically incomplete conclusion.
            // Discard it rather than presenting truncated model output as finished teaching.
            val fallback = "根据我们已经取得的证据，你认为下一步最应该核验哪个数据特征？"
            session = session.withAssistantText(
                assistantId, fallback, reasoning.toString(), LearningMessageState.COMPLETE, clock()
            ).withRollingSummary()
            store.saveSession(session)
            scheduleTopicTitle(session)
            emit(LearningEvent.Completed(session))
            scheduleRollingSummary(session)
        } catch (cancelled: CancellationException) {
            gateway.cancelActive()
            session = session.withAssistantText(
                assistantId, text.toString(), reasoning.toString(), LearningMessageState.STOPPED, clock()
            )
            store.saveSession(session)
            throw cancelled
        } catch (error: IOException) {
            val stopped = stopRequested.get()
            val reason = if (stopped) "" else error.userMessage()
            if (!stopped) LearningLog.warn("turn failed: ${error.javaClass.simpleName}", error)
            session = session.withAssistantText(
                assistantId,
                text.toString(),
                reasoning.toString(),
                if (stopped) LearningMessageState.STOPPED else LearningMessageState.ERROR,
                clock(),
                reason
            )
            store.saveSession(session)
            if (stopped) emit(LearningEvent.Stopped(session))
            else emit(LearningEvent.Error(reason, recoverable = true))
        } catch (error: Exception) {
            LearningLog.warn("turn failed unexpectedly", error)
            val reason = error.message?.takeIf(String::isNotBlank) ?: "生成失败"
            session = session.withAssistantText(
                assistantId,
                text.toString(),
                reasoning.toString(),
                LearningMessageState.ERROR,
                clock(),
                reason
            )
            store.saveSession(session)
            emit(LearningEvent.Error(reason, recoverable = false))
        }
        }
    }

    override fun retry(
        sessionId: String,
        assistantMessageId: String,
        chartPngBase64: String?,
        attachments: List<LearningInputAttachment>,
        turnIdentity: LearningTurnIdentity
    ): Flow<LearningEvent> = flow {
        val measurementId = sessionMeasurements[sessionId] ?: return@flow
        val original = store.loadSession(measurementId, sessionId) ?: return@flow
        val assistantIndex = original.messages.indexOfFirst { it.id == assistantMessageId }
        val user = original.messages.getOrNull(assistantIndex - 1)
        if (assistantIndex <= 0 || user?.role != LearningMessageRole.USER) return@flow
        val removedTurnIds = original.messages
            .drop(assistantIndex - 1)
            .mapNotNull { it.turnId.takeIf(String::isNotBlank) }
            .toSet()
        val cleaned = original.copy(
            messages = original.messages.take(assistantIndex - 1),
            actions = if (removedTurnIds.isNotEmpty()) {
                original.actions.filterNot { it.turnId in removedTurnIds }
            } else {
                original.actions.filter { it.startedAtEpochMs < user.createdAtEpochMs }
            },
            references = original.references.filter { reference ->
                if (removedTurnIds.isNotEmpty()) {
                    reference.turnId !in removedTurnIds
                } else {
                    original.actions.firstOrNull { it.turnId == reference.turnId }?.startedAtEpochMs
                        ?.let { it < user.createdAtEpochMs } ?: true
                }
            }
        )
        store.saveSession(cleaned)
        send(sessionId, user.text, chartPngBase64, attachments, turnIdentity).collect { emit(it) }
    }

    override fun stopGenerating() {
        stopRequested.set(true)
        gateway.cancelActive()
    }

    override suspend fun endSession(sessionId: String): LearningSession {
        val measurementId = requireNotNull(sessionMeasurements[sessionId]) {
            "Open the learning session before ending it"
        }
        return store.endSession(measurementId, sessionId)
    }

    private suspend fun generateQuestions(
        snapshot: LearningMeasurementSnapshot,
        chartPngBase64: String?
    ): List<String> {
        val questionTool = JSONObject()
            .put("type", "function")
            .put("name", QUESTION_FUNCTION)
            .put(
                "description",
                "提交恰好三个针对当前测量、由浅入深且不泄露答案的中文学习问题。"
            )
            .put("parameters", JSONObject()
                .put("type", "object")
                        .put("properties", JSONObject().put(
                    "questions",
                    JSONObject().put("type", "array")
                        .put("minItems", 3).put("maxItems", 3)
                        .put("items", JSONObject().put("type", "string").put(
                            "maxLength", LearningQuestionValidator.MAX_QUESTION_LENGTH
                        ))
                ))
                .put("required", JSONArray().put("questions"))
                .put("additionalProperties", false))
        val content = JSONArray()
            .put(JSONObject().put("type", "input_text").put(
                "text",
                "根据这份测量快照生成三个可点击问题。不要回答问题。\n" + snapshotForModel(snapshot)
            ))
        chartPngBase64?.takeIf(String::isNotBlank)?.let {
            content.put(JSONObject().put("type", "input_image")
                .put("image_url", "data:image/png;base64,$it"))
        }
        val input = JSONArray()
            .put(JSONObject().put("role", "system").put("content", SocraticPrompt.questionGeneration))
            .put(JSONObject().put("role", "user").put("content", content))
        var result: List<String>? = null
        withTimeout(questionTimeoutMs) {
            questionGateway.stream(responseBody(
                input = input,
                reasoningEffort = "medium",
                tools = JSONArray().put(questionTool),
                toolChoice = "required",
                maxOutputTokens = QUESTION_MAX_OUTPUT_TOKENS
            )).collect { event ->
                if (event is AliyunStreamEvent.OutputItemDone &&
                    event.item.optString("type") == "function_call" &&
                    event.item.optString("name") == QUESTION_FUNCTION
                ) {
                    val args = JSONObject(event.item.getString("arguments"))
                    val array = args.getJSONArray("questions")
                    result = LearningQuestionValidator.validate(
                        (0 until array.length()).map(array::getString)
                    )
                }
            }
        }
        return requireNotNull(result) { "Model did not submit three questions" }
    }

    private fun scheduleRollingSummary(session: LearningSession) {
        val summarizer = summaryGateway ?: return
        val complete = session.messages.filter { it.state == LearningMessageState.COMPLETE }
        if (complete.size <= LearningContextBuilder.MAX_RECENT_MESSAGES) return
        summaryJobs.remove(session.id)?.cancel()
        val job = backgroundScope.launch {
            summaryMutex.withLock {
                runCatching {
                    val olderMessages = complete.dropLast(LearningContextBuilder.MAX_RECENT_MESSAGES)
                    val source = JSONObject()
                        .put("previous_summary", session.rollingSummary)
                        .put("current_goal", session.learningState.currentGoal)
                        .put("unresolved_questions", JSONArray(session.learningState.unresolvedQuestions))
                        .put("mastered_concepts", JSONArray(session.learningState.masteredConcepts))
                        .put("learner_level", session.learningState.learnerLevel)
                        .put("older_messages", JSONArray(olderMessages.map { message ->
                            JSONObject()
                                .put("role", message.role.name.lowercase())
                                .put("text", message.text.take(600))
                        }))
                    val input = JSONArray()
                        .put(JSONObject().put("role", "system").put(
                            "content",
                            "你只负责压缩学习进度，不回答实验问题。必须调用提交摘要函数；不得添加输入中不存在的测量事实。"
                        ))
                        .put(JSONObject().put("role", "user").put(
                            "content",
                            "把以下较早对话压缩为可供后续苏格拉底式教学继续使用的结构化摘要：${source}"
                        ))
                    var result: JSONObject? = null
                    summarizer.stream(responseBody(
                        input = input,
                        reasoningEffort = "medium",
                        tools = JSONArray().put(summaryTool()),
                        toolChoice = "required",
                        maxOutputTokens = SUMMARY_MAX_OUTPUT_TOKENS
                    )).collect { event ->
                        if (event is AliyunStreamEvent.OutputItemDone &&
                            event.item.optString("type") == "function_call" &&
                            event.item.optString("name") == SUMMARY_FUNCTION
                        ) {
                            result = JSONObject(event.item.getString("arguments"))
                        }
                    }
                    val summary = requireNotNull(result)
                    val latest = store.loadSession(session.measurementId, session.id)
                        ?: return@runCatching
                    val level = summary.optString("learner_level", latest.learningState.learnerLevel)
                        .takeIf { it in LEARNER_LEVELS } ?: latest.learningState.learnerLevel
                    store.saveSession(latest.copy(
                        rollingSummary = summary.getString("summary")
                            .trim().take(LearningContextBuilder.MAX_SUMMARY_CHARS),
                        learningState = latest.learningState.copy(
                            unresolvedQuestions = summary.stringList("unresolved_questions", 8)
                                ?: latest.learningState.unresolvedQuestions,
                            masteredConcepts = summary.stringList("mastered_concepts", 12)
                                ?: latest.learningState.masteredConcepts,
                            learnerLevel = level,
                            updatedAtEpochMs = clock()
                        ),
                        updatedAtEpochMs = clock()
                    ))
                }
            }
        }
        summaryJobs[session.id] = job
        job.invokeOnCompletion { summaryJobs.remove(session.id, job) }
    }

    private fun scheduleTopicTitle(session: LearningSession) {
        val titleGateway = summaryGateway ?: return
        val current = store.loadSession(session.measurementId, session.id) ?: return
        if (!current.needsGeneratedTitle) return
        val complete = current.messages.filter { it.state == LearningMessageState.COMPLETE }
        if (complete.none { it.role == LearningMessageRole.USER } ||
            complete.none { it.role == LearningMessageRole.ASSISTANT }
        ) return
        val provisionalTitle = current.title
        titleJobs.remove(current.id)?.cancel()
        val job = backgroundScope.launch {
            summaryMutex.withLock {
                runCatching {
                    val context = complete.joinToString("\n") { message ->
                        val role = if (message.role == LearningMessageRole.USER) "用户" else "数据助手"
                        "$role：${message.text.take(400)}"
                    }.take(TITLE_CONTEXT_CHARS)
                    val input = JSONArray()
                        .put(JSONObject().put("role", "system").put(
                            "content",
                            "你只负责生成中文学习话题。根据会话概括一个不超过10个汉字的标题；不要标点、引号、Markdown或“标题”前缀，只输出标题。"
                        ))
                        .put(JSONObject().put("role", "user").put("content", context))
                    val generated = StringBuilder()
                    withTimeout(TITLE_TIMEOUT_MS) {
                        titleGateway.stream(responseBody(
                            input = input,
                            reasoningEffort = "low",
                            tools = JSONArray(),
                            toolChoice = "none",
                            maxOutputTokens = TITLE_MAX_OUTPUT_TOKENS
                        )).collect { event ->
                            if (event is AliyunStreamEvent.TextDelta) generated.append(event.delta)
                        }
                    }
                    val title = sanitizeTopicTitle(generated.toString())
                    if (title.isNotBlank()) {
                        store.applyGeneratedTitle(
                            current.measurementId,
                            current.id,
                            provisionalTitle,
                            title
                        )
                    }
                }
            }
        }
        titleJobs[current.id] = job
        job.invokeOnCompletion { titleJobs.remove(current.id, job) }
    }

    private fun sanitizeTopicTitle(value: String): String = value
        .lineSequence()
        .map(String::trim)
        .firstOrNull(String::isNotBlank)
        .orEmpty()
        .replace(Regex("^(标题|话题)\\s*[:：]\\s*"), "")
        .replace(Regex("[\\p{P}\\p{S}\\s]+"), "")
        .take(TITLE_MAX_CHARS)

    private fun summaryTool() = JSONObject()
        .put("type", "function")
        .put("name", SUMMARY_FUNCTION)
        .put("description", "提交较早学习对话的结构化滚动摘要。")
        .put("parameters", JSONObject()
            .put("type", "object")
            .put("properties", JSONObject()
                .put("summary", JSONObject().put("type", "string").put(
                    "maxLength", LearningContextBuilder.MAX_SUMMARY_CHARS
                ))
                .put("unresolved_questions", JSONObject().put("type", "array")
                    .put("maxItems", 8)
                    .put("items", JSONObject().put("type", "string").put("maxLength", 200)))
                .put("mastered_concepts", JSONObject().put("type", "array")
                    .put("maxItems", 12)
                    .put("items", JSONObject().put("type", "string").put("maxLength", 200)))
                .put("learner_level", JSONObject().put("type", "string")
                    .put("enum", JSONArray(LEARNER_LEVELS))))
            .put("required", JSONArray().put("summary"))
            .put("additionalProperties", false))

    private fun JSONObject.stringList(name: String, limit: Int): List<String>? {
        if (!has(name)) return null
        val values = optJSONArray(name) ?: return emptyList()
        return (0 until values.length()).mapNotNull { index ->
            values.optString(index).trim().takeIf(String::isNotEmpty)?.take(200)
        }.distinct().take(limit)
    }

    private fun throwIfStopRequested() {
        if (stopRequested.get()) throw IOException("Alibaba Cloud request was cancelled")
    }

    private fun tools(
        allowWeb: Boolean,
        allowMeasurementEvidence: Boolean = true,
        allowKnowledgeSearch: Boolean = true
    ): JSONArray {
        val local = toolExecutor.toolDefinitions()
        val result = JSONArray()
        repeat(local.length()) {
            val definition = local.getJSONObject(it)
            val name = definition.optString("name")
            val allowed = when (name) {
                LearningToolExecutor.GET_MEASUREMENT_EVIDENCE -> allowMeasurementEvidence
                LearningToolExecutor.SEARCH_COURSE_KNOWLEDGE -> allowKnowledgeSearch
                LearningToolExecutor.GET_LEARNING_STATE -> false
                else -> true
            }
            if (allowed) {
                result.put(definition)
            }
        }
        if (allowWeb) {
            result.put(JSONObject().put("type", "web_search"))
            result.put(JSONObject().put("type", "web_extractor"))
        }
        return result
    }

    private fun responseBody(
        input: JSONArray,
        reasoningEffort: String,
        tools: JSONArray,
        toolChoice: Any,
        maxOutputTokens: Int
    ): JSONObject = JSONObject()
        .put("model", MODEL)
        .put("input", input)
        .put("stream", true)
        .put("store", false)
        .put("reasoning", JSONObject().put("effort", reasoningEffort))
        .put("tools", tools)
        .put("tool_choice", toolChoice)
        .put("parallel_tool_calls", false)
        .put("max_output_tokens", maxOutputTokens)

    private fun measurementEvidenceTools(): JSONArray {
        val definitions = toolExecutor.toolDefinitions()
        return JSONArray().apply {
            repeat(definitions.length()) { index ->
                val definition = definitions.getJSONObject(index)
                if (definition.optString("name") == LearningToolExecutor.GET_MEASUREMENT_EVIDENCE) {
                    put(definition)
                }
            }
            check(length() == 1) { "Measurement evidence tool definition is unavailable" }
        }
    }

    private fun snapshotForModel(snapshot: LearningMeasurementSnapshot): String {
        val f = snapshot.facts
        return JSONObject()
            .put("measurement_id", snapshot.measurementId)
            .put("terminal_velocity_mm_per_s", f.terminalVelocityMmPerS ?: JSONObject.NULL)
            .put("corrected_viscosity_pa_s", f.correctedViscosityPaS ?: JSONObject.NULL)
            .put("r_squared", f.fittedR2 ?: JSONObject.NULL)
            .put("sample_count", f.sampleCount)
            .put("valid_sample_count", f.validSampleCount)
            .put("effective_sample_rate_fps", f.effectiveSampleRateFps)
            .put("outlier_sample_count", f.outlierSampleCount)
            .put("wall_contact_sample_count", f.wallContactSampleCount)
            .put("reliable_terminal_estimate", f.reliableTerminalEstimate)
            .put("missing_parameters", JSONArray(buildList {
                if (f.sphereDiameterMm == null) add("sphere_diameter_mm")
                if (f.tubeInnerDiameterMm == null) add("tube_inner_diameter_mm")
                if (f.liquidDensityGPerCm3 == null) add("liquid_density_g_per_cm3")
                if (f.sphereDensityGPerCm3 == null) add("sphere_density_g_per_cm3")
            })).toString()
    }

    private fun priorEvidenceSections(session: LearningSession): Set<String> =
        session.actions.asSequence()
            .filter {
                it.toolName == LearningToolExecutor.GET_MEASUREMENT_EVIDENCE &&
                    it.state == LearningActionState.COMPLETE
            }
            .mapNotNull { action ->
                runCatching { JSONObject(action.arguments).optString("section")
                    .takeIf(String::isNotBlank) }.getOrNull()
            }
            .toSet()

    private fun reusableToolResults(session: LearningSession): MutableMap<String, String> =
        session.actions.asSequence()
            .filter {
                it.state == LearningActionState.COMPLETE && it.output.isNotBlank() &&
                    it.toolName in REUSABLE_TOOL_NAMES
            }
            .mapNotNull { action ->
                reusableToolKey(action.toolName, action.arguments)?.let { it to action.output }
            }
            .toMap(LinkedHashMap())

    private fun reusableToolKey(name: String, arguments: String): String? {
        if (name !in REUSABLE_TOOL_NAMES) return null
        val normalized = runCatching {
            JSONObject(arguments.ifBlank { "{}" }).let(::canonicalArguments)
        }
            .getOrElse { arguments.trim() }
        return "$name:$normalized"
    }

    private fun canonicalArguments(source: JSONObject): String = JSONObject().apply {
        source.keys().asSequence().toList().sorted().forEach { key ->
            put(key, source.get(key))
        }
    }.toString()

    private fun functionCall(item: JSONObject): FunctionCall? {
        val name = item.optString("name")
        val callId = item.optString("call_id")
        if (name.isBlank() || callId.isBlank()) return null
        return FunctionCall(name, item.optString("arguments", "{}"), callId)
    }

    private fun webReferences(item: JSONObject, turnId: String): List<LearningReference> {
        val sources = item.optJSONObject("action")?.optJSONArray("sources")
            ?: item.optJSONArray("sources") ?: return emptyList()
        return (0 until sources.length()).mapNotNull { index ->
            val source = sources.optJSONObject(index) ?: return@mapNotNull null
            val url = source.optString("url").takeIf(String::isNotBlank) ?: return@mapNotNull null
            LearningReference(
                turnId = turnId,
                title = source.optString("title").ifBlank { url },
                source = "阿里云联网检索",
                url = url
            )
        }
    }

    private fun completedBuiltinAction(item: JSONObject, turnId: String): LearningAction {
        val type = item.optString("type")
        val sourceCount = item.optJSONObject("action")?.optJSONArray("sources")?.length() ?: 0
        val name = item.optString("name").ifBlank { type }
        val label = when (type) {
            "web_search_call", "web_search_image_call", "image_search_call" -> "联网检索"
            "web_extractor_call" -> "读取网页"
            "mcp_call" -> "MCP 调用"
            "file_search_call" -> "检索文件知识"
            "code_interpreter_call" -> "运行代码分析"
            else -> "内置工具"
        }
        return LearningAction(
            turnId = turnId,
            toolName = name,
            label = label,
            callId = item.optString("call_id"),
            arguments = item.optString("arguments", "{}").take(LearningContextBuilder.MAX_TOOL_ARGUMENT_CHARS),
            output = item.optString("output").take(LearningContextBuilder.MAX_TOOL_OUTPUT_CHARS),
            outputItemJson = boundedBuiltinOutputItem(item),
            startedAtEpochMs = clock(),
            completedAtEpochMs = clock(),
            state = LearningActionState.COMPLETE,
            resultSummary = when {
                sourceCount > 0 -> "找到 ${sourceCount} 个来源"
                item.optString("output").isNotBlank() -> "已取得工具结果"
                else -> "已完成"
            }
        )
    }

    /** Keep a built-in/MCP history item valid JSON while bounding large search/tool payloads. */
    private fun boundedBuiltinOutputItem(item: JSONObject): String {
        val copy = JSONObject(item.toString())
        copy.optString("output").takeIf(String::isNotBlank)?.let {
            copy.put("output", it.take(LearningContextBuilder.MAX_TOOL_OUTPUT_CHARS / 2))
        }
        copy.optJSONObject("action")?.let { action ->
            action.optJSONArray("sources")?.let { sources ->
                val bounded = org.json.JSONArray()
                for (index in 0 until minOf(sources.length(), 5)) {
                    val source = sources.optJSONObject(index) ?: continue
                    bounded.put(JSONObject()
                        .put("type", source.optString("type", "url"))
                        .put("url", source.optString("url"))
                        .put("title", source.optString("title")))
                }
                action.put("sources", bounded)
            }
        }
        if (copy.toString().length <= LearningContextBuilder.MAX_TOOL_OUTPUT_CHARS) {
            return copy.toString()
        }
        val fallback = JSONObject().put("type", item.optString("type"))
        listOf("id", "status", "name", "call_id").forEach { key ->
            item.optString(key).takeIf(String::isNotBlank)?.let { fallback.put(key, it) }
        }
        item.optString("output").takeIf(String::isNotBlank)?.let {
            fallback.put("output", it.take(LearningContextBuilder.MAX_TOOL_OUTPUT_CHARS / 3))
        }
        return fallback.toString()
    }

    private fun toolLabel(name: String): String = when (name) {
        LearningToolExecutor.GET_MEASUREMENT_EVIDENCE -> "读取实验数据"
        LearningToolExecutor.SEARCH_COURSE_KNOWLEDGE -> "检索课程知识"
        LearningToolExecutor.GET_LEARNING_STATE -> "读取学习状态"
        LearningToolExecutor.UPDATE_LEARNING_STATE -> "更新学习状态"
        else -> "调用工具"
    }

    private fun LearningSession.withAssistantText(
        assistantId: String,
        value: String,
        reasoning: String,
        state: LearningMessageState,
        now: Long,
        failureReason: String = ""
    ) = copy(
        messages = messages.map { message ->
            if (message.id == assistantId) message.copy(
                text = value,
                reasoning = reasoning,
                failureReason = failureReason,
                state = state
            ) else message
        },
        updatedAtEpochMs = now
    )

    private fun LearningSession.withRollingSummary(): LearningSession {
        val complete = messages.filter { it.state == LearningMessageState.COMPLETE }
        if (complete.size <= LearningContextBuilder.MAX_RECENT_MESSAGES) return this
        val older = complete.dropLast(LearningContextBuilder.MAX_RECENT_MESSAGES)
        val structuralSummary = buildString {
            older.forEach { message ->
                append(if (message.role == LearningMessageRole.USER) "学生：" else "导师：")
                append(message.text.replace('\n', ' ').take(240)).append('\n')
            }
            if (learningState.unresolvedQuestions.isNotEmpty()) {
                append("尚未解决：").append(learningState.unresolvedQuestions.joinToString("；"))
            }
        }.takeLast(LearningContextBuilder.MAX_SUMMARY_CHARS)
        return copy(rollingSummary = structuralSummary)
    }

    private fun IOException.userMessage(): String {
        val http = this as? AliyunHttpException
        return when (http?.statusCode) {
            401, 403 -> "模型服务拒绝了当前凭据"
            429 -> "模型服务当前请求较多，请稍后重试"
            in 400..499 -> "模型请求被拒绝（HTTP ${http?.statusCode}）：${http?.message.orEmpty().take(180)}"
            in 500..599 -> "模型服务暂时不可用，请稍后重试"
            else -> when (this) {
                is AliyunTimeoutException -> "模型请求超时，请重试"
                is AliyunResponseException -> message.orEmpty().take(180)
                    .ifBlank { "模型未能完成本次回答" }
                else -> "网络连接中断，请检查网络后重试"
            }
        }
    }

    private data class FunctionCall(
        val name: String,
        val arguments: String,
        val callId: String
    ) {
        fun toInputItem() = JSONObject()
            .put("type", "function_call")
            .put("name", name)
            .put("arguments", arguments)
            .put("call_id", callId)
    }

    companion object {
        const val MODEL = "qwen3.7-plus"
        const val TURN_TIMEOUT_MS = 5 * 60 * 1_000L
        const val QUESTION_TIMEOUT_MS = 2 * 60 * 1_000L
        const val ANSWER_MAX_OUTPUT_TOKENS = 4_096
        const val QUESTION_MAX_OUTPUT_TOKENS = 1_200
        const val STREAM_PERSIST_CHARS = 80
        const val QUESTION_FUNCTION = "submit_learning_questions"
        const val SUMMARY_FUNCTION = "submit_learning_summary"
        const val SUMMARY_MAX_OUTPUT_TOKENS = 1_600
        const val TITLE_MAX_OUTPUT_TOKENS = 256
        const val TITLE_TIMEOUT_MS = 20_000L
        const val TITLE_CONTEXT_CHARS = 1_000
        const val TITLE_MAX_CHARS = 10
        private val LEARNER_LEVELS = listOf("unknown", "beginner", "intermediate", "advanced")
        private val REUSABLE_TOOL_NAMES = setOf(
            LearningToolExecutor.GET_MEASUREMENT_EVIDENCE,
            LearningToolExecutor.SEARCH_COURSE_KNOWLEDGE
        )
    }
}

/**
 * Thin logcat wrapper. Failures raised while streaming a model answer used to disappear into a
 * transient toast, so every aborted turn now leaves one line under a stable tag.
 */
internal object LearningLog {
    private const val TAG = "LearningTutor"

    fun warn(message: String, error: Throwable? = null) {
        runCatching { android.util.Log.w(TAG, message, error) }
    }
}
