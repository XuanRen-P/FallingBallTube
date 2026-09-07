package com.cupet.fallingball.learning

import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import kotlin.math.ceil

enum class MeasurementEvidenceSection { OVERVIEW, FIT, TRAJECTORY, WALL, DIAGNOSTICS }

data class ToolExecution(
    val output: String,
    val label: String,
    val summary: String,
    val session: LearningSession,
    val references: List<LearningReference> = emptyList()
)

class LearningToolExecutor(private val knowledgeIndex: KnowledgeIndex) {
    fun execute(
        session: LearningSession,
        turnId: String,
        name: String,
        arguments: String
    ): ToolExecution {
        val args = runCatching { JSONObject(arguments.ifBlank { "{}" }) }.getOrElse {
            return ToolExecution(
                output = JSONObject().put("error", "invalid_arguments").toString(),
                label = "参数无效",
                summary = "工具参数不是有效 JSON",
                session = session
            )
        }
        return when (name) {
            GET_MEASUREMENT_EVIDENCE -> measurementEvidence(session, args)
            SEARCH_COURSE_KNOWLEDGE -> searchKnowledge(session, turnId, args)
            GET_LEARNING_STATE -> getLearningState(session)
            UPDATE_LEARNING_STATE -> updateLearningState(session, args)
            else -> ToolExecution(
                output = JSONObject().put("error", "unknown_tool").toString(),
                label = "未知工具",
                summary = "没有名为 $name 的本地工具",
                session = session
            )
        }
    }

    fun toolDefinitions(): JSONArray = JSONArray()
        .put(functionTool(
            GET_MEASUREMENT_EVIDENCE,
            "读取当前会话冻结测量快照中的有界证据。凡涉及本次实验的数字、曲线或质量判断，必须先调用此工具。",
            JSONObject()
                .put("type", "object")
                .put("properties", JSONObject().put(
                    "section",
                    JSONObject().put("type", "string").put(
                        "enum",
                        JSONArray(MeasurementEvidenceSection.entries.map { it.name })
                    )
                ))
                .put("required", JSONArray().put("section"))
                .put("additionalProperties", false)
        ))
        .put(functionTool(
            SEARCH_COURSE_KNOWLEDGE,
            "从本地已核验的落球法、数据质量和黏度知识卡中检索最多 5 条相关材料。",
            JSONObject()
                .put("type", "object")
                .put("properties", JSONObject().put(
                    "query",
                    JSONObject().put("type", "string").put("maxLength", 200)
                ))
                .put("required", JSONArray().put("query"))
                .put("additionalProperties", false)
        ))
        .put(functionTool(
            GET_LEARNING_STATE,
            "读取学生当前学习目标、尚未解决的问题、已掌握概念和能力层级。",
            JSONObject().put("type", "object").put("properties", JSONObject())
                .put("additionalProperties", false)
        ))
        .put(functionTool(
            UPDATE_LEARNING_STATE,
            "在学生表现出新的理解或困难后更新学习状态。不得用它结束会话。",
            JSONObject()
                .put("type", "object")
                .put("properties", JSONObject()
                    .put("current_goal", JSONObject().put("type", "string").put("maxLength", 200))
                    .put("unresolved_questions", JSONObject().put("type", "array").put(
                        "items", JSONObject().put("type", "string").put("maxLength", 200)
                    ).put("maxItems", 8))
                    .put("mastered_concepts", JSONObject().put("type", "array").put(
                        "items", JSONObject().put("type", "string").put("maxLength", 100)
                    ).put("maxItems", 12))
                    .put("learner_level", JSONObject().put("type", "string").put(
                        "enum", JSONArray(listOf("unknown", "beginner", "intermediate", "advanced"))
                    )))
                .put("additionalProperties", false)
        ))

    private fun measurementEvidence(session: LearningSession, args: JSONObject): ToolExecution {
        val section = runCatching {
            MeasurementEvidenceSection.valueOf(args.getString("section"))
        }.getOrElse {
            return ToolExecution(
                JSONObject().put("error", "invalid_section").toString(),
                "数据范围无效",
                "未读取数据",
                session
            )
        }
        val snapshot = session.snapshot
        val facts = snapshot.facts
        val output = when (section) {
            MeasurementEvidenceSection.OVERVIEW -> JSONObject()
                .put("measurement_id", snapshot.measurementId)
                .put("recorded_at_epoch_ms", snapshot.recordedAtEpochMs)
                .put("record_name", snapshot.recordName)
                .put("liquid_name", snapshot.liquidName)
                .put("sphere_material", snapshot.sphereMaterial)
                .putNullable("temperature_c", snapshot.temperatureC)
                .put("resolution", snapshot.resolutionLabel)
                .put("end_reason", snapshot.endReason)
                .put("tick_spacing_mm", snapshot.tickSpacingMm)
                .put("expected_tick_count", snapshot.expectedTickCount)
                .put("sample_count", facts.sampleCount)
                .put("valid_sample_count", facts.validSampleCount)
                .put("effective_sample_rate_fps", facts.effectiveSampleRateFps)
                .put("reliable_terminal_estimate", facts.reliableTerminalEstimate)
                .putNullable("terminal_velocity_mm_per_s", facts.terminalVelocityMmPerS)
                .putNullable("corrected_viscosity_pa_s", facts.correctedViscosityPaS)
            MeasurementEvidenceSection.FIT -> JSONObject()
                .put("reliable", facts.reliableTerminalEstimate)
                .putNullable("terminal_velocity_mm_per_s", facts.terminalVelocityMmPerS)
                .putNullable("mean_velocity_mm_per_s", facts.meanVelocityMmPerS)
                .putNullable("r_squared", facts.fittedR2)
                .putNullable("fit_start_mm", facts.fitStartMm)
                .putNullable("fit_end_mm", facts.fitEndMm)
                .putNullable("fit_duration_ms", facts.fitDurationMs)
                .put("outlier_sample_count", facts.outlierSampleCount)
                .put("effective_sample_rate_fps", facts.effectiveSampleRateFps)
            MeasurementEvidenceSection.TRAJECTORY -> trajectoryEvidence(snapshot)
            MeasurementEvidenceSection.WALL -> JSONObject()
                .put("known_sample_count", facts.wallContactKnownSampleCount)
                .put("contact_sample_count", facts.wallContactSampleCount)
                .putNullable("average_center_offset_mm", facts.averageCenterOffsetMm)
                .putNullable("sphere_diameter_mm", facts.sphereDiameterMm)
                .putNullable("tube_inner_diameter_mm", facts.tubeInnerDiameterMm)
                .putNullable("correction_model", facts.wallCorrectionModel)
                .putNullable("corrected_viscosity_pa_s", facts.correctedViscosityPaS)
                .put("scope_note", "当前修正假设中心轴运动；二维贴壁不能推断三维液膜间隙")
            MeasurementEvidenceSection.DIAGNOSTICS -> JSONObject()
                .put("availability", snapshot.diagnostics["availability"] ?: "unavailable")
                .put("values", JSONObject(snapshot.diagnostics))
                .put(
                    "missing_evidence_note",
                    if (facts.diagnosticsAvailable) JSONObject.NULL
                    else "该记录早于 FBR v11 或未写入诊断摘要，相关判断未知"
                )
        }
        return ToolExecution(
            output = output.toString(),
            label = "读取${sectionLabel(section)}",
            summary = "已读取 ${sectionLabel(section)}",
            session = session
        )
    }

    private fun trajectoryEvidence(snapshot: LearningMeasurementSnapshot): JSONObject {
        val source = snapshot.trajectory
        val step = ceil(source.size / MAX_TRAJECTORY_POINTS.toDouble()).toInt().coerceAtLeast(1)
        val selected = source.indices.filter { it % step == 0 || it == source.lastIndex }
            .distinct()
            .take(MAX_TRAJECTORY_POINTS)
            .map(source::get)
        return JSONObject()
            .put("total_point_count", source.size)
            .put("returned_point_count", selected.size)
            .put("sampling", if (selected.size == source.size) "all" else "uniform_bounded")
            .put("points", JSONArray(selected.map { point ->
                JSONObject()
                    .put("t_ms", point.elapsedMs)
                    .putNullable("position_mm", point.positionMm)
                    .putNullable("velocity_mm_per_s", point.velocityMmPerS)
                    .putNullable("lateral_mm", point.lateralMm)
                    .put("ruler_valid", point.rulerValid)
                    .put("in_terminal_fit", point.usedInTerminalFit)
                    .put("outlier", point.excludedAsOutlier)
                    .put("wall", point.wallContactState)
                    .put("source", point.detectionSource)
            }))
    }

    private fun searchKnowledge(
        session: LearningSession,
        turnId: String,
        args: JSONObject
    ): ToolExecution {
        val query = args.optString("query").trim().take(200)
        val hits = knowledgeIndex.search(query)
        val references = hits.map { hit ->
            LearningReference(
                turnId = turnId,
                title = hit.card.title,
                source = hit.card.source,
                url = hit.card.sourceUrl
            )
        }
        val result = JSONObject()
            .put("query", query)
            .put("count", hits.size)
            .put("local_retrieval_sufficient", knowledgeIndex.hasSufficientCoverage(hits))
            .put("cards", JSONArray(hits.mapIndexed { index, hit ->
                JSONObject()
                    .put("reference_id", references[index].id)
                    .put("title", hit.card.title)
                    .put("level", hit.card.level)
                    .put("source", hit.card.source)
                    .putNullable("url", hit.card.sourceUrl)
                    .put("content", hit.boundedContent)
            }))
        return ToolExecution(
            output = result.toString(),
            label = "检索课程知识",
            summary = "找到 ${hits.size} 张本地知识卡",
            session = session.copy(references = (session.references + references).distinctBy { it.id }),
            references = references
        )
    }

    private fun getLearningState(session: LearningSession): ToolExecution {
        val state = session.learningState
        val result = JSONObject()
            .put("current_goal", state.currentGoal)
            .put("unresolved_questions", JSONArray(state.unresolvedQuestions))
            .put("mastered_concepts", JSONArray(state.masteredConcepts))
            .put("learner_level", state.learnerLevel)
        return ToolExecution(result.toString(), "读取学习状态", "已读取学习状态", session)
    }

    private fun updateLearningState(session: LearningSession, args: JSONObject): ToolExecution {
        fun boundedStrings(name: String, limit: Int): List<String>? {
            if (!args.has(name)) return null
            val array = args.optJSONArray(name) ?: return emptyList()
            return (0 until array.length()).mapNotNull { index ->
                array.optString(index).trim().takeIf(String::isNotEmpty)?.take(200)
            }.distinct().take(limit)
        }
        val previous = session.learningState
        val level = args.optString("learner_level", previous.learnerLevel).let {
            if (it in setOf("unknown", "beginner", "intermediate", "advanced")) it
            else previous.learnerLevel
        }
        val updated = previous.copy(
            currentGoal = args.optString("current_goal", previous.currentGoal).trim()
                .take(200).ifBlank { previous.currentGoal },
            unresolvedQuestions = boundedStrings("unresolved_questions", 8)
                ?: previous.unresolvedQuestions,
            masteredConcepts = boundedStrings("mastered_concepts", 12)
                ?: previous.masteredConcepts,
            learnerLevel = level,
            updatedAtEpochMs = System.currentTimeMillis()
        )
        return ToolExecution(
            output = JSONObject().put("updated", true).toString(),
            label = "更新学习状态",
            summary = "已更新学习目标与掌握情况",
            session = session.copy(learningState = updated)
        )
    }

    private fun functionTool(name: String, description: String, parameters: JSONObject) =
        JSONObject().put("type", "function").put("name", name)
            .put("description", description).put("parameters", parameters)

    private fun sectionLabel(section: MeasurementEvidenceSection): String = when (section) {
        MeasurementEvidenceSection.OVERVIEW -> "测量概览"
        MeasurementEvidenceSection.FIT -> "拟合证据"
        MeasurementEvidenceSection.TRAJECTORY -> "轨迹证据"
        MeasurementEvidenceSection.WALL -> "管壁证据"
        MeasurementEvidenceSection.DIAGNOSTICS -> "诊断摘要"
    }

    private fun JSONObject.putNullable(name: String, value: Any?): JSONObject =
        put(name, value ?: JSONObject.NULL)

    companion object {
        const val GET_MEASUREMENT_EVIDENCE = "get_measurement_evidence"
        const val SEARCH_COURSE_KNOWLEDGE = "search_course_knowledge"
        const val GET_LEARNING_STATE = "get_learning_state"
        const val UPDATE_LEARNING_STATE = "update_learning_state"
        const val MAX_TRAJECTORY_POINTS = 64
    }
}

object SocraticPrompt {
    val system: String = """
        你是落球法黏度实验的苏格拉底式学习导师。你使用苏格拉底式方法帮助学生理解实验推理，
        但苏格拉底式方法主要用于解释、判断和结论，不用于阻止学生获得事实知识。

        教学规则：
        1. 先判断学生的问题属于事实、方法、计算还是实验结论。
        2. 当前测量数据，以及知识库中的定义、公式、理论值、参考范围、单位换算、计算步骤和方法说明，
           可以直接回答。不要强行反问，也不要要求学生先自行查找。
        3. 直接回答知识库内容时，说明适用条件、温度、单位、来源和参考值性质；不同来源有差异时明确说明。
        4. 当前测量中的数字、统计量和图表事实可以直接说出；它们不等于实验结论。
           如果相同证据已在上下文或历史中出现，直接复用，不要让学生重复查询。
        5. 对当前实验结果的最终解释、综合可靠性判断、理论与实验差异的最终原因和学习目标的最终回答，
           在学生完全没有尝试时不要直接替学生写出。
        6. 结论型问题默认先引用相关证据，再给一条具体提示，最后提出一个核心问题；
           不要只重复“你观察到什么？”。
        7. 提示按以下顺序递进：指出观察对象 → 定位具体数据段或变量 →
           说明物理关系或公式作用 → 提供局部计算结构。
        8. 学生明确说“不会”“不懂”或“直接告诉我”，或连续一次回答偏离时，至少升级一档提示；
           连续两次卡住时可以提供局部脚手架，但不要直接写出当前实验的最终结论。
        9. 学生回答正确时，明确确认其推理并简洁总结；是否进入进阶内容由学生决定。
        10. 学生部分正确时，保留正确部分并指出缺口；回答错误时，指出错误所在的推理步骤。
        11. 当前客户端会显示模型思考内容流；思考内容应围绕本轮教学任务，但不能替代最终可见回复。
        12. 永远不要主动结束学习会话。返回页面不代表结束，只有用户显式选择“结束本次学习”才结束。
        13. 只有当前上下文和历史证据都缺少所需的实验数据时，才调用 get_measurement_evidence。
            缺失数据必须说未知，不得猜测。
        14. developer 上下文会预取本轮相关的本地核验材料。仅当 local_retrieval_sufficient=false
            且确实缺少概念依据时，才调用一次 search_course_knowledge；本地知识已经包含所需理论值、
            公式或定义时，不要要求学生重复查找。只有该工具结果仍明确不足时，下一次请求才可使用联网搜索。
        15. 当前学习状态已直接包含在 developer 上下文中，不要调用 get_learning_state。
            仅当学生表现出新的理解或困难、状态确实变化时才调用 update_learning_state；
            不得把学习状态当成测量事实。
        16. 每轮最多提出一个需要学生回答的核心问题；纯事实问题可以不强制反问。
            若提出问题，只使用一个问号，不要用“还是”或“或者”连接两个各自完整的问题。
    """.trimIndent()

    fun turnContract(firstTutorTurn: Boolean): String = if (firstTutorTurn) {
        """
            本轮是该会话的首轮教学，学生尚未展示任何独立推导。可以直接引用当前测量中的数据、统计量、
            图表事实或知识库理论值，但不要直接写出当前实验的最终结论。
            如果学生问的是纯事实、理论、公式、单位或计算步骤，可以直接答完，不强制反问。
            如果学生问的是综合解释或实验结论，最终可见回复可以有两到四句：给出必要事实、
            一条具体观察或方法提示，最后提出一个核心问题。若提出问题，只使用一个“？”，
            不要用“还是”或“或者”连接两个各自完整的问题。
        """.trimIndent()
    } else {
        """
            本轮可以先直接回答事实、理论、公式、单位或计算步骤。
            如果问题涉及当前实验的综合解释或最终结论，则先引用证据，给一条适当层级的提示，再提出一个核心问题。
            学生说“不会”“不懂”“直接告诉我”或连续一次回答偏离时，至少升级一档提示；
            连续两次卡住时可以提供局部计算结构。纯事实问题不强制反问；若提出问题，只使用一个“？”。
        """.trimIndent()
    }

    val questionGeneration: String = """
        你的任务是为落球法黏度实验的学生起草三个「他可能想向 AI 提出的问题」。
        这些问题会以可点击的卡片展示，学生点击后就作为他本人的提问发送出去。
        你不回答这些问题。

        【核心视角】
        你不是在出题考学生，而是在替一个盯着自己的实验数据、
        不知道该问什么才好的学生开口。因此：
        - 用学生向老师提问的口吻，可以带「我这次」「为什么」「会不会」「怎么」等说法。
        - 不要用考问语气，禁止出现「说明什么」「你认为」「反映了何种」「可能产生什么影响」
          这类把学生当答题者的句式。
        - 每个问题都必须是学生答不上来、但很想知道答案的；
          凡是学生自己看一眼数据或凭常识就能回答的，一律不要出。

        【三个问题的分工】
        1. 读懂本次数据：针对本次测量中最显眼、最容易让人困惑的现象，问它是怎么回事。
        2. 影响与可靠性：问这个现象或某个条件会让最终黏度结果偏大还是偏小、影响有多大、
           结果还能不能信。
        3. 方法与改进：问该怎么算、怎么修正、下次怎么测更准，或与理论值/其他条件如何比较。

        【选题原则】
        - 只针对本次测量真正突出的特征提问：贴壁下落、明显偏心、温度缺失、
          雷诺数偏高、球径与管径之比偏大、拟合区间异常等，优先挑对最终黏度有实质影响的。
        - 尊重落球法物理常识，不要制造伪问题：
          · 直径 1–3 mm 的小球入液后在极短时间和极短距离内即达匀速，
            因此很小的起始下落距离（如 1–2 mm）不构成端部效应疑点，
            除非诊断明确显示拟合区间紧贴液面或管底。
          · 手机自动追踪在数十上百个采样点中出现少数被自动剔除的异常点属正常现象，
            对结论无实质影响，不要拿它做选题。
          · R² 极高只是说明拟合得好，不要围绕它提问；这类问题每次测量都能问，没有区分度。
        - 系统已判定为既定事实的特征（如贴壁下落、偏心、温度未记录），
          当作前提直接问后果，不要再让学生去确认这个事实是否成立。
        - 不罗列学生看不懂的内部统计量或原始计数（如「76 个壁接触样本」「88 个样本中 3 个异常值」）；
          要用学生能理解的说法描述现象（如「小球这次是贴着管壁下落的」）。
        - 三个问题中至少有一个是能被直接讲清楚的（公式、修正方法、参考值、计算步骤），
          让学生点了就有实质收获。

        【表述要求】
        - 每个问题一句话，尽量不超过 30 个字，句末一个问号，只问一件事。
        - 不在问题里写出答案或暗示唯一结论。
        - 三个问题之间不重复同一个知识点。

        【示例】
        不好：R²=0.9999 说明位移与时间呈什么关系？（学生自己就知道，且是考问语气）
        不好：88 个样本中有 3 个异常值，它们可能出现在曲线的哪个阶段？（噪声，且不像人话）
        不好：76 个壁接触样本对终端速度测量可能产生什么影响？（原始计数 + 含糊语气）
        较好：我这次小球是贴着管壁下落的，算出来的黏度会偏大还是偏小？
        较好：没记温度的话，这次的黏度值还能和标准值比吗？
        较好：壁面修正的公式是怎么算的，这次修正了多少？

        只能通过 submit_learning_questions 函数返回恰好三个问题，
        不输出前言、解释、答案或额外文本。
    """.trimIndent()
}

object MeasurementEvidencePolicy {
    private val measurementTerms = listOf(
        "这次", "本次", "当前", "记录", "结果", "数据", "图", "曲线", "轨迹",
        "终端速度", "黏度", "r²", "拟合", "可靠", "质量", "样本", "采样",
        "异常", "贴壁", "抖动", "参数"
    )
    private val conceptualTerms = listOf(
        "原理", "概念", "定义", "理论", "公式", "一般来说", "是什么", "为什么"
    )

    fun mustReadMeasurementEvidence(
        userText: String,
        firstTutorTurn: Boolean,
        priorEvidenceSections: Set<String> = emptySet()
    ): Boolean {
        if (firstTutorTurn) return true
        val normalized = userText.lowercase()
        val requestedSection = when {
            listOf("轨迹", "曲线", "位移").any(normalized::contains) -> "TRAJECTORY"
            listOf("贴壁", "管壁").any(normalized::contains) -> "WALL"
            listOf("诊断", "抖动", "采样", "异常", "质量", "可靠").any(normalized::contains) -> "DIAGNOSTICS"
            listOf("拟合", "终端速度", "黏度", "速度").any(normalized::contains) -> "FIT"
            measurementTerms.any(normalized::contains) -> "OVERVIEW"
            else -> null
        }
        if (requestedSection != null) return requestedSection !in priorEvidenceSections
        // Short follow-ups such as "继续" or "再看看" do not introduce a new
        // measurement claim. Requiring evidence for every such turn creates a
        // repeated read of the same frozen snapshot even though it is already
        // replayed in the request history.
        return measurementTerms.any(normalized::contains) && conceptualTerms.none(normalized::contains)
    }
}

class LearningContextBuilder(private val knowledgeIndex: KnowledgeIndex) {
    fun build(
        session: LearningSession,
        userText: String,
        chartPngBase64: String? = null,
        attachments: List<LearningInputAttachment> = emptyList()
    ): JSONArray {
        val input = JSONArray()
        input.put(message("system", SocraticPrompt.system))
        val state = session.learningState
        val localKnowledge = knowledgeIndex.search(userText, limit = 5, maxContentChars = 600)
        val localKnowledgeSufficient = knowledgeIndex.hasSufficientCoverage(localKnowledge)
        val context = buildString {
            append("学习目标：").append(state.currentGoal)
            append("\n学习层级：").append(state.learnerLevel)
            append("\n当前测量的有界快照：").append(snapshotOverview(session.snapshot))
            if (state.unresolvedQuestions.isNotEmpty()) {
                append("\n未解决：").append(state.unresolvedQuestions.joinToString("；"))
            }
            if (state.masteredConcepts.isNotEmpty()) {
                append("\n已掌握：").append(state.masteredConcepts.joinToString("；"))
            }
            if (session.rollingSummary.isNotBlank()) {
                append("\n历史摘要：").append(session.rollingSummary.take(MAX_SUMMARY_CHARS))
            }
            append("\n本轮本地知识预取：local_retrieval_sufficient=")
                .append(localKnowledgeSufficient)
            if (localKnowledge.isNotEmpty()) {
                append("；候选材料：")
                localKnowledge.forEach { hit ->
                    append("\n- ").append(hit.card.title)
                        .append("（来源：").append(hit.card.source)
                    hit.card.sourceUrl?.let { url -> append("；URL：").append(url) }
                    append("）：").append(hit.boundedContent)
                }
            }
            val replayableActions = session.actions.filter {
                it.state == LearningActionState.COMPLETE &&
                    (it.output.isNotBlank() || it.outputItemJson.isNotBlank())
            }
            if (replayableActions.isNotEmpty()) {
                append("\n历史已取得工具证据（同一冻结测量不应重复调用，除非需要尚未读取的 section）：")
                replayableActions.asReversed()
                    .distinctBy(::toolEvidenceKey)
                    .take(MAX_RECENT_TOOL_ACTIONS)
                    .asReversed()
                    .forEach { action ->
                        append("\n- ").append(action.toolName).append(" ")
                            .append(action.arguments.take(MAX_TOOL_ARGUMENT_CHARS))
                        append(" => ").append(toolEvidence(action))
                    }
            }
        }.take(MAX_CONTEXT_CHARS)
        input.put(message("developer", context))

        val completeMessages = session.messages
            .filter { it.state == LearningMessageState.COMPLETE }
            .takeLast(MAX_RECENT_MESSAGES)
        val latestImmutableActionIdByKey = session.actions.asSequence()
            .filter {
                it.state == LearningActionState.COMPLETE &&
                    it.toolName in IMMUTABLE_READ_TOOL_NAMES &&
                    (it.output.isNotBlank() || it.outputItemJson.isNotBlank())
            }
            .groupBy(::toolEvidenceKey)
            // Store append order is authoritative when two actions share a clock tick.
            .mapValues { (_, actions) -> actions.last().id }
        completeMessages.forEachIndexed { index, stored ->
            input.put(historyMessage(stored))
            if (stored.role == LearningMessageRole.USER) {
                val nextUserAt = completeMessages.drop(index + 1)
                    .firstOrNull { it.role == LearningMessageRole.USER }
                    ?.createdAtEpochMs ?: Long.MAX_VALUE
                val turnActions = if (stored.turnId.isNotBlank()) {
                    // New sessions carry an explicit ownership key. This avoids losing a
                    // tool result when two user turns share the same millisecond timestamp.
                    session.actions.filter { it.turnId == stored.turnId }
                } else {
                    // Legacy sessions (before turn_id was persisted) retain the timestamp
                    // fallback so their existing history remains usable.
                    session.actions.filter { action ->
                        action.startedAtEpochMs >= stored.createdAtEpochMs &&
                            action.startedAtEpochMs < nextUserAt
                    }
                }
                turnActions
                    .filter { action ->
                        action.state == LearningActionState.COMPLETE &&
                            ((action.callId.isNotBlank() && action.output.isNotBlank()) ||
                                action.outputItemJson.isNotBlank()) &&
                            (action.toolName !in IMMUTABLE_READ_TOOL_NAMES ||
                                latestImmutableActionIdByKey[toolEvidenceKey(action)] == action.id)
                    }
                    .sortedBy(LearningAction::startedAtEpochMs)
                    .distinctBy(LearningAction::id)
                    .take(MAX_RECENT_TOOL_ACTIONS)
                    .forEach { action -> replayActionHistory(input, action) }
            }
        }
        val userContent = JSONArray().put(
            JSONObject().put("type", "input_text").put("text", userText.take(MAX_USER_CHARS))
        )
        chartPngBase64?.takeIf(String::isNotBlank)?.let {
            userContent.put(
                JSONObject().put("type", "input_image")
                    .put("image_url", "data:image/png;base64,$it")
            )
        }
        attachments.take(MAX_ATTACHMENTS).forEach { attachment ->
            val data = "data:${attachment.mediaType};base64,${attachment.base64}"
            val item = if (attachment.mediaType.startsWith("image/")) {
                JSONObject().put("type", "input_image").put("image_url", data)
            } else {
                JSONObject().put("type", "input_file")
                    .put("filename", attachment.name.take(120))
                    .put("file_data", data)
            }
            userContent.put(item)
        }
        input.put(JSONObject().put("role", "user").put("content", userContent))
        return input
    }

    private fun message(role: String, text: String) =
        JSONObject().put("role", role).put("content", text)

    private fun historyMessage(message: LearningMessage): JSONObject {
        val user = message.role == LearningMessageRole.USER
        val content = JSONArray().put(
            JSONObject()
                .put("type", if (user) "input_text" else "output_text")
                .put("text", message.text)
        )
        return JSONObject()
            .put("role", if (user) "user" else "assistant")
            .put("content", content)
    }

    private fun functionCallHistory(action: LearningAction): JSONObject = JSONObject()
        .put("type", "function_call")
        .put("name", action.toolName)
        .put("arguments", action.arguments.take(MAX_TOOL_ARGUMENT_CHARS))
        .put("call_id", action.callId)

    private fun functionOutputHistory(action: LearningAction): JSONObject = JSONObject()
        .put("type", "function_call_output")
        .put("call_id", action.callId)
        .put("output", action.output.take(MAX_TOOL_OUTPUT_CHARS))

    private fun replayActionHistory(input: JSONArray, action: LearningAction) {
        if (action.outputItemJson.isNotBlank()) {
            runCatching { input.put(JSONObject(action.outputItemJson)) }
                .onFailure {
                    // Keep old/corrupt records usable by falling back to the function pair.
                    if (action.callId.isNotBlank() && action.output.isNotBlank()) {
                        input.put(functionCallHistory(action))
                        input.put(functionOutputHistory(action))
                    }
                }
            return
        }
        if (action.callId.isNotBlank() && action.output.isNotBlank()) {
            input.put(functionCallHistory(action))
            input.put(functionOutputHistory(action))
        }
    }

    private fun toolEvidenceKey(action: LearningAction): String =
        "${action.toolName}:${canonicalArguments(action.arguments)}"

    private fun canonicalArguments(value: String): String = runCatching {
        val source = JSONObject(value.ifBlank { "{}" })
        JSONObject().apply {
            source.keys().asSequence().toList().sorted().forEach { key ->
                put(key, source.get(key))
            }
        }.toString()
    }.getOrElse { value.trim() }

    private fun toolEvidence(action: LearningAction): String = when {
        action.output.isNotBlank() -> action.output.take(MAX_EVIDENCE_OUTPUT_CHARS)
        action.outputItemJson.isNotBlank() -> action.outputItemJson.take(MAX_EVIDENCE_OUTPUT_CHARS)
        action.resultSummary.isNotBlank() -> action.resultSummary.take(MAX_EVIDENCE_OUTPUT_CHARS)
        else -> "已完成（无可回放结果）"
    }

    private fun snapshotOverview(snapshot: LearningMeasurementSnapshot): String {
        val facts = snapshot.facts
        return JSONObject()
            .put("measurement_id", snapshot.measurementId)
            .put("fingerprint", snapshot.fingerprint)
            .put("recorded_at_epoch_ms", snapshot.recordedAtEpochMs)
            .put("sample_count", facts.sampleCount)
            .put("valid_sample_count", facts.validSampleCount)
            .put("effective_sample_rate_fps", facts.effectiveSampleRateFps)
            .put("reliable_terminal_estimate", facts.reliableTerminalEstimate)
            .put("terminal_velocity_mm_per_s", facts.terminalVelocityMmPerS ?: JSONObject.NULL)
            .put("corrected_viscosity_pa_s", facts.correctedViscosityPaS ?: JSONObject.NULL)
            .put("r_squared", facts.fittedR2 ?: JSONObject.NULL)
            .put("diagnostics_available", facts.diagnosticsAvailable)
            .toString()
    }

    companion object {
        const val MAX_RECENT_MESSAGES = 24
        const val MAX_RECENT_TOOL_ACTIONS = 10
        const val MAX_TOOL_ARGUMENT_CHARS = 1_200
        const val MAX_TOOL_OUTPUT_CHARS = 12_000
        const val MAX_EVIDENCE_OUTPUT_CHARS = 1_600
        const val MAX_CONTEXT_CHARS = 32_000
        const val MAX_SUMMARY_CHARS = 6_000
        const val MAX_USER_CHARS = 4_000
        const val MAX_ATTACHMENTS = 3
        private val IMMUTABLE_READ_TOOL_NAMES = setOf(
            LearningToolExecutor.GET_MEASUREMENT_EVIDENCE,
            LearningToolExecutor.SEARCH_COURSE_KNOWLEDGE
        )
    }
}
