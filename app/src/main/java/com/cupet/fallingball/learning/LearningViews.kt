package com.cupet.fallingball.learning

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import com.cupet.fallingball.ui.UiColors
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.RenderEffect
import android.graphics.RenderNode
import android.graphics.Shader
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.text.Html
import android.text.Spanned
import android.text.TextUtils
import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.ValueAnimator
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.animation.DecelerateInterpolator
import android.view.animation.PathInterpolator
import android.os.SystemClock
import android.widget.EdgeEffect
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.widget.AppCompatTextView
import androidx.dynamicanimation.animation.DynamicAnimation
import androidx.dynamicanimation.animation.SpringAnimation
import androidx.dynamicanimation.animation.SpringForce
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.RecyclerView
import com.cupet.fallingball.R
import kotlin.math.abs

internal sealed interface ConversationItem {
    val timestamp: Long

    data class Message(
        val value: LearningMessage,
        val actions: List<LearningAction> = emptyList(),
        val references: List<LearningReference> = emptyList(),
        val sessionActive: Boolean = true
    ) : ConversationItem {
        override val timestamp: Long get() = value.createdAtEpochMs
    }
}

private const val ASSISTANT_BUBBLE_WIDTH_FRACTION = 0.84f

internal class LearningConversationAdapter(
    private val context: Context,
    private val onRetry: (String) -> Unit,
    private val onCopy: (LearningMessage) -> Unit,
    private val onContentHeightChanged: () -> Unit
) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {
    private val items = ArrayList<ConversationItem>()
    private val stableIds = mutableMapOf<String, Long>()
    private var nextStableId = 0L
    private val expandedActions = mutableSetOf<String>()
    private val expandedReasoning = mutableMapOf<String, Boolean>()
    private var glassTexture: Bitmap? = null
    private var textureRoot: View? = null
    private var liveBackdrop: LiveGlassBackdrop? = null
    private var streamingReasoning: String = ""

    init {
        setHasStableIds(true)
    }

    fun submit(session: LearningSession, reasoning: String = "") {
        streamingReasoning = reasoning
        val updated = buildConversationItems(session)
        val previous = items.toList()
        val diff = DiffUtil.calculateDiff(object : DiffUtil.Callback() {
            override fun getOldListSize(): Int = previous.size
            override fun getNewListSize(): Int = updated.size
            override fun areItemsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean =
                conversationItemKey(previous[oldItemPosition]) ==
                    conversationItemKey(updated[newItemPosition])
            override fun areContentsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean =
                previous[oldItemPosition] == updated[newItemPosition]

            override fun getChangePayload(oldItemPosition: Int, newItemPosition: Int): Any? =
                PAYLOAD_SAME_MESSAGE.takeIf {
                    conversationItemKey(previous[oldItemPosition]) ==
                        conversationItemKey(updated[newItemPosition])
                }
        })
        items.clear()
        items += updated
        diff.dispatchUpdatesTo(this)
    }

    /** Updates only the currently streaming assistant item while text/reasoning deltas arrive. */
    fun updateStreaming(session: LearningSession, reasoning: String = streamingReasoning) {
        streamingReasoning = reasoning
        val streaming = session.messages.lastOrNull {
            it.role == LearningMessageRole.ASSISTANT &&
                it.state == LearningMessageState.STREAMING
        } ?: return
        val position = items.indexOfFirst {
            it is ConversationItem.Message && it.value.id == streaming.id
        }
        if (position < 0) {
            submit(session, reasoning)
            return
        }
        val previous = items[position] as? ConversationItem.Message
        items[position] = ConversationItem.Message(
            value = streaming,
            actions = previous?.actions.orEmpty(),
            references = previous?.references.orEmpty(),
            sessionActive = session.status == LearningSessionStatus.ACTIVE
        )
        notifyItemChanged(position, PAYLOAD_STREAMING)
    }

    fun updateStructure(session: LearningSession, reasoning: String = streamingReasoning) {
        streamingReasoning = reasoning
        val updated = buildConversationItems(session)
        val previous = items.toList()
        val diff = DiffUtil.calculateDiff(object : DiffUtil.Callback() {
            override fun getOldListSize(): Int = previous.size
            override fun getNewListSize(): Int = updated.size

            override fun areItemsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean =
                conversationItemKey(previous[oldItemPosition]) ==
                    conversationItemKey(updated[newItemPosition])

            override fun areContentsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean =
                previous[oldItemPosition] == updated[newItemPosition]

            override fun getChangePayload(oldItemPosition: Int, newItemPosition: Int): Any? =
                PAYLOAD_SAME_MESSAGE.takeIf {
                    conversationItemKey(previous[oldItemPosition]) ==
                        conversationItemKey(updated[newItemPosition])
                }
        })
        items.clear()
        items += updated
        diff.dispatchUpdatesTo(this)
    }

    fun updateGlassTexture(bitmap: Bitmap?, root: View) {
        glassTexture = bitmap
        textureRoot = root
        notifyItemRangeChanged(0, itemCount)
    }

    fun updateLiveBackdrop(backdrop: LiveGlassBackdrop?) {
        liveBackdrop = backdrop
        notifyItemRangeChanged(0, itemCount)
    }

    override fun getItemViewType(position: Int): Int = TYPE_MESSAGE

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder =
        MessageHolder(LinearLayout(context).apply {
            layoutParams = RecyclerView.LayoutParams(-1, -2)
            setPadding(dp(12), dp(5), dp(12), dp(5))
        })

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        when (val item = items[position]) {
            is ConversationItem.Message -> (holder as MessageHolder).bind(item, streamingReasoning)
        }
    }

    override fun onBindViewHolder(
        holder: RecyclerView.ViewHolder,
        position: Int,
        payloads: MutableList<Any>
    ) {
        if (payloads.isEmpty()) {
            onBindViewHolder(holder, position)
            return
        }
        when (val item = items[position]) {
            is ConversationItem.Message -> {
                if (payloads.contains(PAYLOAD_STREAMING)) {
                    (holder as MessageHolder).updateStreaming(item, streamingReasoning)
                } else {
                    (holder as MessageHolder).bind(item, streamingReasoning)
                }
            }
        }
    }

    override fun getItemCount(): Int = items.size

    override fun getItemId(position: Int): Long =
        stableIds.getOrPut(conversationItemKey(items[position])) { nextStableId++ }

    fun latestTurnStartPosition(): Int = items.indexOfLast {
        it is ConversationItem.Message && it.value.role == LearningMessageRole.USER
    }.coerceAtLeast(0)

    private inner class MessageHolder(private val container: LinearLayout) :
        RecyclerView.ViewHolder(container) {
        private val reasoningPanel = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            setPadding(dp(10), dp(1), dp(10), dp(1))
        }
        private val reasoningHeader = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(10), dp(1), dp(8), dp(1))
            minimumHeight = dp(36)
            isClickable = true
            isFocusable = true
            background = roundedBackground(UiColors.SURFACE_CONTAINER, UiColors.OUTLINE)
        }
        private val reasoningIcon = ImageView(context).apply {
            setImageResource(R.drawable.ic_learning_reasoning)
            setColorFilter(UiColors.ON_SURFACE_VARIANT)
        }
        private val reasoningLabel = AppCompatTextView(context).apply {
            textSize = 12.5f
            setTextColor(UiColors.ON_SURFACE_VARIANT)
            maxLines = 2
            ellipsize = TextUtils.TruncateAt.END
            includeFontPadding = false
            gravity = Gravity.CENTER_VERTICAL
        }
        private val reasoningChevron = ImageView(context).apply {
            setImageResource(R.drawable.ic_learning_chevron_down)
            setColorFilter(UiColors.ON_SURFACE_VARIANT)
        }
        private val reasoningBody = AppCompatTextView(context).apply {
            textSize = 12f
            setTextColor(UiColors.ON_SURFACE_VARIANT)
            setLineSpacing(0f, 1.08f)
            includeFontPadding = false
            setPadding(dp(14), dp(7), dp(14), dp(7))
            background = roundedBackground(UiColors.SURFACE_CONTAINER, UiColors.OUTLINE)
        }
        /** Tool timeline lives inside the assistant turn so inserting a tool step cannot move
         * the whole conversation or look like a detached dialog. */
        private val actionPanel = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            setPadding(dp(10), dp(1), dp(10), dp(1))
        }
        private val actionHeader = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(10), dp(1), dp(8), dp(1))
            minimumHeight = dp(36)
            background = roundedBackground(UiColors.SUCCESS_CONTAINER, UiColors.OUTLINE)
            isClickable = true
        }
        private val actionIcon = ImageView(context).apply {
            setImageResource(R.drawable.ic_learning_evidence)
            setColorFilter(UiColors.SUCCESS)
        }
        private val actionTitle = AppCompatTextView(context).apply {
            textSize = 11.5f
            maxLines = 1
            ellipsize = TextUtils.TruncateAt.END
            includeFontPadding = false
            gravity = Gravity.CENTER_VERTICAL
            setTextColor(UiColors.ON_PRIMARY_CONTAINER)
        }
        private val actionChevron = ImageView(context).apply {
            setImageResource(R.drawable.ic_learning_chevron_down)
            setColorFilter(UiColors.ON_SURFACE_VARIANT)
        }
        private val actionDetails = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            setPadding(dp(11), dp(5), dp(11), dp(7))
        }
        private val bubble = GlassMessageView(context)
        private val bubbleHost = FrameLayout(context)
        private lateinit var richText: LearningRichTextView
        private var reasoningHeightAnimator: ValueAnimator? = null
        private var actionDetailsHeightAnimator: ValueAnimator? = null
        private var boundRichMessageId: String? = null
        private var boundRichRenderToken = 0L
        private var boundReasoningMessageId: String? = null
        private var boundReasoningPreview = ""
        private var boundReasoningPhase = LearningReasoningPhase.ACTIVE
        private val attachmentPanel = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END
            visibility = View.GONE
            setPadding(0, 0, 0, dp(4))
        }
        private val actionBar = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.START
            visibility = View.GONE
        }
        private var boundActionTurnId: String? = null

        init {
            container.orientation = LinearLayout.VERTICAL
            container.gravity = Gravity.START
            richText = LearningRichTextView(context) { height, renderToken ->
                if (boundRichRenderToken != renderToken) return@LearningRichTextView
                val hostParams = bubbleHost.layoutParams as? LinearLayout.LayoutParams
                    ?: return@LearningRichTextView
                val targetHeight = height.coerceAtLeast(dp(44))
                if (hostParams.height != targetHeight) {
                    // The host owns the measured height. Both children fill it, so a shorter
                    // Markdown result cannot leave the old plain-text height behind.
                    hostParams.height = targetHeight
                    bubbleHost.layoutParams = hostParams
                    bubbleHost.requestLayout()
                }
                onContentHeightChanged()
            }
            reasoningHeader.addView(reasoningIcon, LinearLayout.LayoutParams(dp(18), dp(18)))
            reasoningHeader.addView(reasoningLabel, LinearLayout.LayoutParams(0, -2, 1f).apply {
                marginStart = dp(8)
                marginEnd = dp(4)
            })
            reasoningHeader.addView(reasoningChevron, LinearLayout.LayoutParams(dp(18), dp(18)))
            reasoningPanel.addView(reasoningHeader, LinearLayout.LayoutParams(-1, -2))
            reasoningPanel.addView(reasoningBody, LinearLayout.LayoutParams(-1, -2).apply {
                topMargin = dp(3)
            })
            container.addView(reasoningPanel, LinearLayout.LayoutParams(-1, -2))
            actionHeader.addView(actionIcon, LinearLayout.LayoutParams(dp(18), dp(18)))
            actionHeader.addView(actionTitle, LinearLayout.LayoutParams(0, -2, 1f).apply {
                marginStart = dp(8)
                marginEnd = dp(4)
            })
            actionHeader.addView(actionChevron, LinearLayout.LayoutParams(dp(18), dp(18)))
            actionPanel.addView(actionHeader, LinearLayout.LayoutParams(-1, -2))
            actionPanel.addView(actionDetails, LinearLayout.LayoutParams(-1, -2))
            actionHeader.setOnClickListener {
                val id = boundActionTurnId ?: return@setOnClickListener
                val next = !expandedActions.contains(id)
                if (next) expandedActions += id else expandedActions -= id
                setActionsExpanded(next, animate = true)
            }
            container.addView(actionPanel, LinearLayout.LayoutParams(-1, -2))
            // The bubble is added after the reasoning row so the two streams read as one AI turn.
            bubbleHost.addView(bubble, FrameLayout.LayoutParams(-1, -2))
            bubbleHost.addView(richText, FrameLayout.LayoutParams(-1, -1).apply {
                gravity = Gravity.FILL
            })
            richText.visibility = View.GONE
            container.addView(attachmentPanel, LinearLayout.LayoutParams(-1, -2))
            container.addView(bubbleHost, LinearLayout.LayoutParams(-2, -2))
            container.addView(actionBar, LinearLayout.LayoutParams(-1, dp(48)))
        }

        fun bind(item: ConversationItem.Message, reasoningText: String) {
            val message = item.value
            val user = message.role == LearningMessageRole.USER
            reasoningHeightAnimator?.cancel()
            actionDetailsHeightAnimator?.cancel()
            bubble.animate().cancel()
            bubble.alpha = 1f
            bubble.contentAlpha = 255
            attachmentPanel.removeAllViews()
            if (user && message.attachmentNames.isNotEmpty()) {
                message.attachmentNames.take(3).forEach { name ->
                    attachmentPanel.addView(TextView(context).apply {
                        text = "▧  ${name.take(22)}"
                        textSize = 11f
                        setTextColor(UiColors.ON_PRIMARY_CONTAINER)
                        setPadding(dp(9), dp(5), dp(9), dp(5))
                        background = roundedBackground(UiColors.PRIMARY_CONTAINER, UiColors.OUTLINE)
                        maxLines = 1
                        ellipsize = TextUtils.TruncateAt.END
                    }, LinearLayout.LayoutParams(0, dp(30), 1f).apply {
                        marginStart = dp(5)
                    })
                }
                attachmentPanel.visibility = View.VISIBLE
            } else {
                attachmentPanel.visibility = View.GONE
            }
            val completeReasoning = message.reasoning.ifBlank { reasoningText }
            val showReasoning = !user && (message.state == LearningMessageState.STREAMING || completeReasoning.isNotBlank())
            reasoningPanel.visibility = if (showReasoning) View.VISIBLE else View.GONE
            reasoningBody.animate().cancel()
            reasoningBody.alpha = 1f
            reasoningBody.translationY = 0f
            (reasoningPanel.layoutParams as? LinearLayout.LayoutParams)?.let { params ->
                params.width = if (user) 0 else (
                    context.resources.displayMetrics.widthPixels * ASSISTANT_BUBBLE_WIDTH_FRACTION
                ).toInt()
                params.gravity = Gravity.START
                reasoningPanel.layoutParams = params
            }
            val isExpanded = expandedReasoning[message.id] ?: false
            boundReasoningMessageId = message.id
            boundReasoningPreview = completeReasoning
            boundReasoningPhase = LearningReasoningPreview.phase(
                state = message.state,
                answerStarted = message.text.isNotBlank()
            )
            updateReasoningHeader(isExpanded)
            reasoningBody.text = if (completeReasoning.isBlank()) "正在整理思路…" else {
                if (message.state == LearningMessageState.STREAMING) completeReasoning
                else MarkdownRenderer.render(completeReasoning)
            }
            setReasoningExpanded(message.id, isExpanded, animate = false)
            reasoningHeader.setOnClickListener {
                val nextExpanded = !(expandedReasoning[message.id] ?: false)
                val adapterPosition = bindingAdapterPosition
                if (adapterPosition != RecyclerView.NO_POSITION) {
                    setReasoningExpanded(message.id, nextExpanded, animate = true)
                }
            }
            bindActions(item.actions, item.references)
            val useRichText = !user && message.text.isNotBlank()
            val newRichBinding = boundRichMessageId != message.id
            if (newRichBinding) {
                richText.invalidateRenderToken()
                boundRichRenderToken = richText.currentRenderToken()
            }
            boundRichMessageId = if (useRichText) message.id else null
            bubble.isUser = user
            if (useRichText) {
                prepareRichTextHost(newRichBinding)
                bubble.visibility = View.VISIBLE
                bubble.text = ""
                bubble.contentAlpha = 0
                richText.setMarkdown(message.text)
                boundRichRenderToken = richText.currentRenderToken()
            } else {
                bubble.text = MarkdownRenderer.render(message.text.ifBlank {
                    when (message.state) {
                        LearningMessageState.STREAMING -> " "
                        LearningMessageState.ERROR -> message.failureReason
                            .takeIf(String::isNotBlank)
                            ?.let { "本次生成未完成：$it" }
                            ?: "本次生成未完成，可点击重试"
                        LearningMessageState.STOPPED -> "已停止生成"
                        LearningMessageState.COMPLETE -> ""
                    }
                })
                bubble.contentAlpha = 255
                bubble.visibility = if (!user && message.text.isBlank() &&
                    message.state == LearningMessageState.STREAMING
                ) View.GONE else View.VISIBLE
                richText.invalidateRenderToken()
                richText.visibility = View.GONE
                richText.alpha = 1f
                setBubbleFillHost(false)
            }
            bubble.alpha = if (message.state == LearningMessageState.ERROR) 0.64f else 1f
            bubble.setTexture(glassTexture, textureRoot)
            bubble.setLiveBackdrop(liveBackdrop)
            bubble.contentDescription = if (user) "你的消息：${message.text}" else "数据助手：${message.text}"
            val params = bubbleHost.layoutParams as LinearLayout.LayoutParams
            params.width = if (useRichText) {
                (context.resources.displayMetrics.widthPixels * ASSISTANT_BUBBLE_WIDTH_FRACTION).toInt()
            } else {
                -2
            }
            if (!useRichText) params.height = ViewGroup.LayoutParams.WRAP_CONTENT
            params.gravity = if (user) Gravity.END else Gravity.START
            params.topMargin = if (showReasoning) dp(2) else 0
            bubbleHost.layoutParams = params
            actionBar.removeAllViews()
            actionBar.gravity = if (user) Gravity.END else Gravity.START
            val canRetry = LearningUiPolicy.canRetry(
                message.role,
                message.state,
                item.sessionActive
            )
            if (message.state != LearningMessageState.STREAMING &&
                (message.text.isNotBlank() || canRetry)
            ) {
                actionBar.visibility = View.VISIBLE
                if (message.text.isNotBlank()) {
                    actionBar.addView(iconAction(R.drawable.ic_learning_copy, "复制", {
                        onCopy(message)
                    }))
                }
                if (canRetry) {
                    actionBar.addView(iconAction(R.drawable.ic_learning_retry, "重试", {
                        onRetry(message.id)
                    }))
                }
            } else actionBar.visibility = View.GONE
        }

        fun updateStreaming(item: ConversationItem.Message, reasoningText: String) {
            val message = item.value
            if (message.role == LearningMessageRole.USER) return
            val completeReasoning = message.reasoning.ifBlank { reasoningText }
            reasoningPanel.visibility = View.VISIBLE
            // Keep the streaming path plain text. Re-parsing Markdown on every delta forces
            // a full TextView layout and is the main source of visible jitter.
            reasoningBody.text = completeReasoning.ifBlank { "正在整理思路…" }
            val expanded = expandedReasoning[message.id] ?: false
            boundReasoningMessageId = message.id
            boundReasoningPreview = completeReasoning
            boundReasoningPhase = LearningReasoningPreview.phase(
                state = message.state,
                answerStarted = message.text.isNotBlank()
            )
            updateReasoningHeader(expanded)
            setReasoningExpanded(message.id, expanded, animate = false)
            if (message.text.isBlank()) {
                boundRichMessageId = null
                richText.invalidateRenderToken()
                boundRichRenderToken = richText.currentRenderToken()
                bubble.visibility = View.GONE
                richText.visibility = View.GONE
                setBubbleFillHost(false)
                (bubbleHost.layoutParams as? LinearLayout.LayoutParams)?.let { params ->
                    params.height = ViewGroup.LayoutParams.WRAP_CONTENT
                    bubbleHost.layoutParams = params
                }
            } else {
                val newRichBinding = boundRichMessageId != message.id
                if (newRichBinding) {
                    richText.invalidateRenderToken()
                    boundRichRenderToken = richText.currentRenderToken()
                }
                boundRichMessageId = message.id
                bubble.visibility = View.VISIBLE
                bubble.text = ""
                bubble.contentAlpha = 0
                prepareRichTextHost(newRichBinding)
                richText.setMarkdown(message.text)
                boundRichRenderToken = richText.currentRenderToken()
            }
            bubble.alpha = 1f
            bubble.setTexture(glassTexture, textureRoot)
            bubble.setLiveBackdrop(liveBackdrop)
        }

        private fun setReasoningExpanded(id: String, isExpanded: Boolean, animate: Boolean) {
            expandedReasoning[id] = isExpanded
            if (boundReasoningMessageId == id) updateReasoningHeader(isExpanded)
            reasoningHeightAnimator?.cancel()
            reasoningHeightAnimator = animateExpandable(
                view = reasoningBody,
                expanded = isExpanded,
                animate = animate,
                durationMs = 200L
            )
        }

        private fun updateReasoningHeader(expanded: Boolean) {
            val label = when {
                boundReasoningPhase == LearningReasoningPhase.COMPLETE && expanded -> context.getString(
                    R.string.learning_reasoning_complete_expanded
                )
                boundReasoningPhase == LearningReasoningPhase.COMPLETE -> context.getString(
                    R.string.learning_reasoning_complete
                )
                boundReasoningPhase == LearningReasoningPhase.STOPPED && expanded -> context.getString(
                    R.string.learning_reasoning_stopped_expanded
                )
                boundReasoningPhase == LearningReasoningPhase.STOPPED -> context.getString(
                    R.string.learning_reasoning_stopped
                )
                boundReasoningPhase == LearningReasoningPhase.ERROR && expanded -> context.getString(
                    R.string.learning_reasoning_error_expanded
                )
                boundReasoningPhase == LearningReasoningPhase.ERROR -> context.getString(
                    R.string.learning_reasoning_error
                )
                expanded -> context.getString(R.string.learning_reasoning_streaming_expanded)
                else -> LearningReasoningPreview.latestSentence(boundReasoningPreview)
                    .ifBlank { context.getString(R.string.learning_reasoning_streaming) }
            }
            reasoningLabel.text = label
            reasoningHeader.contentDescription = label
            reasoningIcon.setColorFilter(
                when (boundReasoningPhase) {
                    LearningReasoningPhase.ACTIVE -> UiColors.WARNING
                    LearningReasoningPhase.ERROR -> UiColors.ERROR
                    LearningReasoningPhase.STOPPED -> UiColors.ON_SURFACE_MUTED
                    LearningReasoningPhase.COMPLETE -> UiColors.PRIMARY
                }
            )
            reasoningChevron.rotation = if (expanded) 0f else -90f
        }

        private fun bindActions(
            actions: List<LearningAction>,
            references: List<LearningReference>
        ) {
            if (actions.isEmpty() && references.isEmpty()) {
                boundActionTurnId = null
                actionPanel.animate().cancel()
                actionPanel.alpha = 1f
                actionPanel.translationY = 0f
                actionPanel.visibility = View.GONE
                return
            }
            val reveal = actionPanel.visibility != View.VISIBLE
            val turnId = actions.firstOrNull()?.turnId
                ?: references.firstOrNull()?.turnId.orEmpty()
            boundActionTurnId = turnId
            (actionPanel.layoutParams as? LinearLayout.LayoutParams)?.let { params ->
                params.width = (context.resources.displayMetrics.widthPixels * TOOL_PANEL_WIDTH_FRACTION).toInt()
                params.gravity = Gravity.START
                actionPanel.layoutParams = params
            }
            val running = actions.lastOrNull { it.state == LearningActionState.RUNNING }
            val failed = actions.count { it.state == LearningActionState.ERROR }
            actionTitle.text = if (running != null) {
                "正在读取  ·  ${running.label}"
            } else {
                val suffix = if (references.isEmpty()) "" else " / ${references.size} 个来源"
                "已参考 ${actions.size} 项数据$suffix" +
                    if (failed > 0) "  ·  $failed 项失败" else ""
            }
            actionIcon.setColorFilter(
                if (running != null) UiColors.WARNING else UiColors.SUCCESS
            )
            actionDetails.removeAllViews()
            actions.forEach { action ->
                val status = when (action.state) {
                    LearningActionState.RUNNING -> "执行中"
                    LearningActionState.COMPLETE -> "完成"
                    LearningActionState.ERROR -> "失败"
                }
                actionDetails.addView(TextView(context).apply {
                    text = "${action.label}  ·  $status" +
                        action.resultSummary.takeIf { it.isNotBlank() }
                            ?.let { "  ·  $it" }.orEmpty()
                    textSize = 11f
                    setTextColor(
                        if (action.state == LearningActionState.ERROR) {
                            UiColors.ERROR
                        } else UiColors.ON_SURFACE_VARIANT
                    )
                    setPadding(dp(22), dp(3), dp(2), dp(3))
                    maxLines = 2
                    ellipsize = TextUtils.TruncateAt.END
                })
                val inputSummary = compactJson(action.arguments, 360)
                if (inputSummary.isNotBlank() && inputSummary != "{}") {
                    actionDetails.addView(actionDetailText("输入  ·  $inputSummary"))
                }
                val outputSummary = compactJson(action.output, 520)
                if (outputSummary.isNotBlank()) {
                    actionDetails.addView(actionDetailText("结果  ·  $outputSummary"))
                }
            }
            references.forEach { reference ->
                actionDetails.addView(TextView(context).apply {
                    text = "来源  ·  ${reference.title}"
                    textSize = 11f
                    setTextColor(UiColors.SUCCESS)
                    setPadding(dp(22), dp(4), dp(2), dp(4))
                    maxLines = 2
                    ellipsize = TextUtils.TruncateAt.END
                })
            }
            actionPanel.visibility = View.VISIBLE
            if (reveal) {
                actionPanel.alpha = 0f
                actionPanel.translationY = dp(-4).toFloat()
                actionPanel.animate().alpha(1f).translationY(0f)
                    .setDuration(170L).start()
            }
            setActionsExpanded(expandedActions.contains(turnId), animate = false)
        }

        private fun setActionsExpanded(expanded: Boolean, animate: Boolean) {
            actionChevron.animate().cancel()
            actionChevron.animate().rotation(if (expanded) 180f else 0f)
                .setDuration(if (animate) 180L else 0L).start()
            actionDetailsHeightAnimator?.cancel()
            actionDetailsHeightAnimator = animateExpandable(
                view = actionDetails,
                expanded = expanded,
                animate = animate,
                durationMs = 190L
            )
        }

        private fun actionDetailText(value: String) = TextView(context).apply {
            text = value
            textSize = 10.5f
            setTextColor(UiColors.ON_SURFACE_MUTED)
            setPadding(dp(22), dp(1), dp(8), dp(4))
            maxLines = 4
            ellipsize = TextUtils.TruncateAt.END
            setTextIsSelectable(true)
        }

        private fun compactJson(value: String, limit: Int): String {
            if (value.isBlank()) return ""
            return value.replace(Regex("\\s+"), " ").trim().take(limit)
        }

        private fun prepareRichTextHost(resetHeight: Boolean) {
            val hostParams = bubbleHost.layoutParams as? LinearLayout.LayoutParams ?: return
            hostParams.width = (
                context.resources.displayMetrics.widthPixels * ASSISTANT_BUBBLE_WIDTH_FRACTION
            ).toInt()
            hostParams.gravity = Gravity.START
            if (resetHeight) hostParams.height = dp(44)
            bubbleHost.layoutParams = hostParams
            setBubbleFillHost(true)
            richText.visibility = View.VISIBLE
            richText.alpha = 1f
        }

        private fun setBubbleFillHost(fill: Boolean) {
            val params = bubble.layoutParams as? FrameLayout.LayoutParams ?: return
            val targetWidth = if (fill) {
                ViewGroup.LayoutParams.MATCH_PARENT
            } else {
                ViewGroup.LayoutParams.WRAP_CONTENT
            }
            val targetHeight = if (fill) {
                ViewGroup.LayoutParams.MATCH_PARENT
            } else {
                ViewGroup.LayoutParams.WRAP_CONTENT
            }
            if (params.width != targetWidth || params.height != targetHeight) {
                params.width = targetWidth
                params.height = targetHeight
                bubble.layoutParams = params
            }
        }

        /** Android has no CSS grid-row transition, so animate expandable reasoning/tool panels. */
        private fun animateExpandable(
            view: View,
            expanded: Boolean,
            animate: Boolean,
            durationMs: Long
        ): ValueAnimator? {
            view.animate().cancel()
            val params = view.layoutParams ?: return null
            if (!animate || !container.isLaidOut) {
                params.height = ViewGroup.LayoutParams.WRAP_CONTENT
                view.layoutParams = params
                view.visibility = if (expanded) View.VISIBLE else View.GONE
                view.alpha = 1f
                view.translationY = 0f
                return null
            }

            val currentHeight = if (view.visibility == View.VISIBLE) {
                (params.height.takeIf { it >= 0 } ?: view.height).coerceAtLeast(0)
            } else {
                0
            }
            val availableWidth = ((view.parent as? View)?.width
                ?: (context.resources.displayMetrics.widthPixels * ASSISTANT_BUBBLE_WIDTH_FRACTION).toInt()
            ).coerceAtLeast(dp(80))
            params.height = ViewGroup.LayoutParams.WRAP_CONTENT
            view.layoutParams = params
            view.measure(
                View.MeasureSpec.makeMeasureSpec(availableWidth, View.MeasureSpec.AT_MOST),
                View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED)
            )
            val measuredHeight = view.measuredHeight.coerceAtLeast(1)
            val startHeight = when {
                currentHeight > 0 -> currentHeight.coerceAtMost(measuredHeight)
                expanded -> 0
                else -> measuredHeight
            }
            val endHeight = if (expanded) measuredHeight else 0
            params.height = startHeight
            view.layoutParams = params
            view.visibility = View.VISIBLE
            val startProgress = startHeight / measuredHeight.toFloat()
            view.alpha = startProgress
            view.translationY = -dp(4) * (1f - startProgress)

            return ValueAnimator.ofInt(startHeight, endHeight).apply {
                duration = durationMs
                interpolator = DecelerateInterpolator()
                addUpdateListener { animator ->
                    params.height = animator.animatedValue as Int
                    view.layoutParams = params
                    val progress = (params.height / measuredHeight.toFloat()).coerceIn(0f, 1f)
                    view.alpha = progress
                    view.translationY = -dp(4) * (1f - progress)
                }
                addListener(object : AnimatorListenerAdapter() {
                    private var cancelled = false

                    override fun onAnimationCancel(animation: Animator) {
                        cancelled = true
                    }

                    override fun onAnimationEnd(animation: Animator) {
                        if (cancelled) return
                        params.height = ViewGroup.LayoutParams.WRAP_CONTENT
                        view.layoutParams = params
                        view.visibility = if (expanded) View.VISIBLE else View.GONE
                        view.alpha = 1f
                        view.translationY = 0f
                    }
                })
                start()
            }
        }

        private fun iconAction(icon: Int, description: String, click: () -> Unit) =
            ImageButton(context).apply {
                setImageResource(icon)
                contentDescription = description
                setColorFilter(UiColors.ON_SURFACE_VARIANT)
                setBackgroundResource(R.drawable.bg_learning_icon_button)
                setOnClickListener { click() }
                layoutParams = LinearLayout.LayoutParams(dp(48), dp(48)).apply {
                    marginEnd = dp(4)
                }
            }
    }

    private fun dp(value: Int): Int =
        (value * context.resources.displayMetrics.density + 0.5f).toInt()

    private companion object {
        const val TYPE_MESSAGE = 1
        const val TOOL_PANEL_WIDTH_FRACTION = 0.78f
        const val PAYLOAD_STREAMING = "streaming"
        const val PAYLOAD_SAME_MESSAGE = "same_message"
    }
}

internal fun buildConversationItems(session: LearningSession): List<ConversationItem> {
    val actionsByTurn = session.actions
        .filter { it.turnId.isNotBlank() }
        .groupBy(LearningAction::turnId)
    val referencesByTurn = session.references.groupBy(LearningReference::turnId)
    return session.messages.mapIndexed { index, message ->
        val actions = if (message.role != LearningMessageRole.ASSISTANT) {
            emptyList()
        } else if (message.turnId.isNotBlank()) {
            actionsByTurn[message.turnId].orEmpty()
        } else {
            // Legacy sessions predate message.turnId. Keep their actions inside the closest
            // assistant turn instead of creating a detached row.
            val userAt = session.messages.take(index)
                .lastOrNull { it.role == LearningMessageRole.USER }
                ?.createdAtEpochMs ?: Long.MIN_VALUE
            val nextUserAt = session.messages.drop(index + 1)
                .firstOrNull { it.role == LearningMessageRole.USER }
                ?.createdAtEpochMs ?: Long.MAX_VALUE
            session.actions.filter {
                it.startedAtEpochMs >= userAt && it.startedAtEpochMs < nextUserAt
            }
        }
        val turnId = message.turnId
        val refs = if (turnId.isNotBlank()) {
            referencesByTurn[turnId].orEmpty()
        } else {
            actions.firstOrNull()?.turnId?.let(referencesByTurn::get).orEmpty()
        }
        ConversationItem.Message(
            value = message,
            actions = actions,
            references = refs,
            sessionActive = session.status == LearningSessionStatus.ACTIVE
        )
    }
}

private fun roundedBackground(fill: Int, stroke: Int): GradientDrawable =
    GradientDrawable().apply {
        setColor(fill)
        setStroke(1, stroke)
        cornerRadius = 8f
    }

/** Small, dependency-free renderer for the subset used by tutor responses. */
internal object MarkdownRenderer {
    fun render(source: String): Spanned {
        if (source.isBlank()) return Html.fromHtml("", Html.FROM_HTML_MODE_LEGACY)
        var html = TextUtils.htmlEncode(source)
        html = html.replace(Regex("(?s)```(\\w*)\\n?(.*?)```")) {
            "<pre><code>${it.groupValues[2].trim()}</code></pre>"
        }
        html = html.replace(Regex("(?m)^###\\s+(.+)$"), "<b>$1</b><br>")
            .replace(Regex("(?m)^##\\s+(.+)$"), "<b>$1</b><br>")
            .replace(Regex("(?m)^#\\s+(.+)$"), "<b>$1</b><br>")
            .replace(Regex("\\*\\*(.+?)\\*\\*"), "<b>$1</b>")
            .replace(Regex("(?<!\\*)\\*([^*]+)\\*(?!\\*)"), "<i>$1</i>")
            .replace(Regex("`([^`]+)`"), "<tt>$1</tt>")
            .replace(Regex("(?m)^[-*]\\s+(.+)$"), "• $1<br>")
            .replace("\n", "<br>")
        html = html.replace(Regex("(?s)\\$\\$(.+?)\\$\\$|\\\\\\((.+?)\\\\\\)|(?<!\\$)\\$([^$\\n]+)\\$(?!\\$)")) {
            val formula = listOf(it.groupValues[1], it.groupValues[2], it.groupValues[3])
                .firstOrNull(String::isNotBlank).orEmpty()
            "<font face=\"monospace\"><i>${formatLatex(formula)}</i></font>"
        }
        return Html.fromHtml(html, Html.FROM_HTML_MODE_LEGACY)
    }

    private fun formatLatex(value: String): String = value
        .replace(Regex("\\\\frac\\s*\\{([^{}]+)\\}\\s*\\{([^{}]+)\\}"), "($1)/($2)")
        .replace(Regex("\\\\sqrt\\s*\\{([^{}]+)\\}"), "√($1)")
        .replace("\\\\times", "×")
        .replace("\\\\cdot", "·")
        .replace("\\\\rho", "ρ")
        .replace("\\\\mu", "μ")
        .replace(Regex("\\^\\{([^{}]+)\\}"), "^$1")
        .replace(Regex("_\\{([^{}]+)\\}"), "_$1")
        .replace("\\\\", "")
}

private fun conversationItemKey(item: ConversationItem): String = when (item) {
    is ConversationItem.Message -> "message:${item.value.id}"
}

internal class GlassMessageView(context: Context) : AppCompatTextView(context) {
    var contentAlpha: Int = 255
        set(value) {
            field = value.coerceIn(0, 255)
            invalidate()
        }
    var isUser: Boolean = false
        set(value) {
            field = value
            setTextColor(if (value) UiColors.ON_PRIMARY_CONTAINER else UiColors.ON_SURFACE)
            updatePadding()
            invalidate()
        }
    private var texture: Bitmap? = null
    private var textureRoot: View? = null
    private var liveBackdrop: LiveGlassBackdrop? = null
    private val shape = Path()
    private val texturePaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
    private val tintPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(244, 255, 255, 255)
    }
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = UiColors.OUTLINE
        style = Paint.Style.STROKE
        strokeWidth = dp(1).toFloat()
    }
    private var lastTextureRefresh = 0L

    init {
        setTextColor(UiColors.ON_SURFACE)
        textSize = 15f
        setLineSpacing(0f, 1.08f)
        includeFontPadding = false
        updatePadding()
        setWillNotDraw(false)
    }

    fun setTexture(value: Bitmap?, root: View?) {
        texture = value
        textureRoot = root
        invalidate()
    }

    fun setLiveBackdrop(value: LiveGlassBackdrop?) {
        liveBackdrop = value
        invalidate()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        if (MeasureSpec.getMode(widthMeasureSpec) == MeasureSpec.EXACTLY) {
            super.onMeasure(widthMeasureSpec, heightMeasureSpec)
            return
        }
        val maxWidth = (resources.displayMetrics.widthPixels * ASSISTANT_BUBBLE_WIDTH_FRACTION).toInt()
        val limited = MeasureSpec.makeMeasureSpec(
            minOf(MeasureSpec.getSize(widthMeasureSpec), maxWidth),
            MeasureSpec.AT_MOST
        )
        super.onMeasure(limited, heightMeasureSpec)
    }

    override fun onDraw(canvas: Canvas) {
        buildShape()
        canvas.save()
        canvas.clipPath(shape)
        val backdropDrawn = liveBackdrop?.draw(canvas, this, shape) == true
        if (!backdropDrawn) drawFallbackBackdrop(canvas)
        tintPaint.color = if (isUser) {
            Color.argb(236, 200, 242, 227)
        } else {
            Color.argb(242, 255, 255, 255)
        }
        canvas.drawPath(shape, tintPaint)
        canvas.restore()
        canvas.drawPath(shape, strokePaint)
        if (contentAlpha >= 255) {
            super.onDraw(canvas)
        } else if (contentAlpha > 0) {
            val contentLayer = canvas.saveLayerAlpha(
                0f,
                0f,
                width.toFloat(),
                height.toFloat(),
                contentAlpha
            )
            super.onDraw(canvas)
            canvas.restoreToCount(contentLayer)
        }
    }

    private fun drawFallbackBackdrop(canvas: Canvas) {
        val bitmap = texture
        val root = textureRoot
        if (bitmap == null || bitmap.isRecycled || root == null) return
        if (SystemClock.uptimeMillis() - lastTextureRefresh >= 48L) {
            refreshBlurredFactTexture(root, bitmap)
            lastTextureRefresh = SystemClock.uptimeMillis()
        }
        val bubbleLocation = IntArray(2).also(::getLocationOnScreen)
        val rootLocation = IntArray(2).also(root::getLocationOnScreen)
        val left = (bubbleLocation[0] - rootLocation[0]).coerceIn(0, (bitmap.width - 1).coerceAtLeast(0))
        val top = (bubbleLocation[1] - rootLocation[1]).coerceIn(0, (bitmap.height - 1).coerceAtLeast(0))
        val src = Rect(
            left,
            top,
            (left + width).coerceAtMost(bitmap.width),
            (top + height).coerceAtMost(bitmap.height)
        )
        if (src.width() > 0 && src.height() > 0) {
            canvas.drawBitmap(bitmap, src, Rect(0, 0, width, height), texturePaint)
        }
    }

    private fun buildShape() {
        shape.reset()
        val tail = dp(9).toFloat()
        val body = if (isUser) {
            RectF(0f, 0f, width - tail, height.toFloat())
        } else {
            RectF(tail, 0f, width.toFloat(), height.toFloat())
        }
        shape.addRoundRect(body, dp(8).toFloat(), dp(8).toFloat(), Path.Direction.CW)
        val center = height * 0.67f
        if (isUser) {
            shape.moveTo(width - tail, center - dp(6))
            shape.lineTo(width.toFloat(), center)
            shape.lineTo(width - tail, center + dp(6))
        } else {
            shape.moveTo(tail, center - dp(6))
            shape.lineTo(0f, center)
            shape.lineTo(tail, center + dp(6))
        }
        shape.close()
    }

    private fun updatePadding() {
        if (isUser) setPadding(dp(14), dp(11), dp(22), dp(11))
        else setPadding(dp(22), dp(11), dp(14), dp(11))
    }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density + 0.5f).toInt()
}

/** GPU-backed backdrop blur for API 31+, with no bitmap copy in the normal path. */
internal class LiveGlassBackdrop(private val source: View) {
    private val renderNode: RenderNode? = if (Build.VERSION.SDK_INT >= 29) {
        RenderNode("learning-fact-backdrop")
    } else null
    private var recordedAt = 0L

    fun draw(canvas: Canvas, target: View, clip: Path): Boolean {
        if (Build.VERSION.SDK_INT < 31 || renderNode == null || !canvas.isHardwareAccelerated ||
            source.width <= 0 || source.height <= 0
        ) {
            return false
        }
        val now = SystemClock.uptimeMillis()
        if (now - recordedAt >= 32L) {
            renderNode.setPosition(0, 0, source.width, source.height)
            renderNode.setRenderEffect(
                RenderEffect.createBlurEffect(
                    target.resources.displayMetrics.density * 18f,
                    target.resources.displayMetrics.density * 18f,
                    Shader.TileMode.CLAMP
                )
            )
            val recording = renderNode.beginRecording(source.width, source.height)
            source.draw(recording)
            renderNode.endRecording()
            recordedAt = now
        }
        val targetLocation = IntArray(2).also(target::getLocationOnScreen)
        val sourceLocation = IntArray(2).also(source::getLocationOnScreen)
        canvas.save()
        canvas.clipPath(clip)
        canvas.translate(
            (sourceLocation[0] - targetLocation[0]).toFloat(),
            (sourceLocation[1] - targetLocation[1]).toFloat()
        )
        canvas.drawRenderNode(renderNode)
        canvas.restore()
        return true
    }
}

internal fun createBlurredFactTexture(view: View): Bitmap? {
    if (view.width <= 0 || view.height <= 0) return null
    return refreshBlurredFactTexture(
        view,
        Bitmap.createBitmap(view.width, view.height, Bitmap.Config.ARGB_8888)
    )
}

private fun refreshBlurredFactTexture(view: View, target: Bitmap): Bitmap {
    val smallWidth = (view.width / 8).coerceAtLeast(1)
    val smallHeight = (view.height / 8).coerceAtLeast(1)
    val small = Bitmap.createBitmap(smallWidth, smallHeight, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(small)
    canvas.scale(smallWidth / view.width.toFloat(), smallHeight / view.height.toFloat())
    view.draw(canvas)
    boxBlur(small)
    Canvas(target).drawBitmap(small, null, Rect(0, 0, target.width, target.height), Paint(Paint.FILTER_BITMAP_FLAG))
    small.recycle()
    return target
}

private fun boxBlur(bitmap: Bitmap) {
    if (bitmap.width < 3 || bitmap.height < 3) return
    val source = IntArray(bitmap.width * bitmap.height)
    val output = IntArray(source.size)
    bitmap.getPixels(source, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
    source.copyInto(output)
    for (y in 1 until bitmap.height - 1) for (x in 1 until bitmap.width - 1) {
        var a = 0; var r = 0; var g = 0; var b = 0
        for (dy in -1..1) for (dx in -1..1) {
            val color = source[(y + dy) * bitmap.width + x + dx]
            a += color ushr 24; r += color shr 16 and 0xff
            g += color shr 8 and 0xff; b += color and 0xff
        }
        output[y * bitmap.width + x] = (a / 9 shl 24) or (r / 9 shl 16) or
            (g / 9 shl 8) or (b / 9)
    }
    bitmap.setPixels(output, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
}

internal class SpringEdgeEffectFactory : RecyclerView.EdgeEffectFactory() {
    override fun createEdgeEffect(view: RecyclerView, direction: Int): EdgeEffect {
        val sign = if (direction == DIRECTION_TOP) 1f else -1f
        return object : EdgeEffect(view.context) {
            override fun onPull(deltaDistance: Float) {
                pull(deltaDistance)
            }

            override fun onPull(deltaDistance: Float, displacement: Float) {
                pull(deltaDistance)
            }

            override fun onPullDistance(deltaDistance: Float, displacement: Float): Float {
                pull(deltaDistance)
                return deltaDistance
            }

            override fun onRelease() {
                springBack(view)
            }

            override fun onAbsorb(velocity: Int) {
                view.translationY = (view.translationY + sign * velocity.coerceAtMost(6_000) / 120f)
                    .coerceIn(-maxPull(view), maxPull(view))
                springBack(view)
            }

            override fun draw(canvas: Canvas): Boolean = false

            override fun isFinished(): Boolean = abs(view.translationY) < 0.5f

            private fun pull(distance: Float) {
                view.translationY = (view.translationY + sign * view.height * distance * 0.22f)
                    .coerceIn(-maxPull(view), maxPull(view))
                view.invalidate()
            }
        }
    }

    private fun springBack(view: View) {
        SpringAnimation(view, DynamicAnimation.TRANSLATION_Y, 0f).apply {
            spring = SpringForce(0f).apply {
                stiffness = SpringForce.STIFFNESS_MEDIUM
                dampingRatio = SpringForce.DAMPING_RATIO_MEDIUM_BOUNCY
            }
        }.start()
    }

    private fun maxPull(view: View): Float = 72f * view.resources.displayMetrics.density
}

/**
 * Touch-driven rubber band used by the conversation surface. It mirrors the reference web
 * interaction: 0.4x finger travel at a boundary and a non-bouncy eased return.
 */
internal class ElasticRecyclerView(context: Context) : RecyclerView(context) {
    private enum class Gesture { UNDECIDED, HORIZONTAL, VERTICAL, RUBBER_TOP, RUBBER_BOTTOM }

    private var gesture = Gesture.UNDECIDED
    private var downX = 0f
    private var downY = 0f
    private var startedAtTop = false
    private var startedAtBottom = false
    private var rebound: ValueAnimator? = null
    private val touchSlop = android.view.ViewConfiguration.get(context).scaledTouchSlop

    override fun onInterceptTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> prepareDown(event)
            MotionEvent.ACTION_MOVE -> {
                lockDirection(event)
                if (isRubberGesture()) {
                    parent?.requestDisallowInterceptTouchEvent(true)
                    return true
                }
            }
        }
        return isRubberGesture() || super.onInterceptTouchEvent(event)
    }

    /**
     * Normal list movement remains native. Only the unconsumed outward drag at a real edge moves
     * the transparent conversation layer; the fact layer underneath never changes position.
     */
    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                if (gesture == Gesture.UNDECIDED) prepareDown(event)
                return super.onTouchEvent(event)
            }
            MotionEvent.ACTION_MOVE -> {
                lockDirection(event)
                if (isRubberGesture()) {
                    parent?.requestDisallowInterceptTouchEvent(true)
                    val drag = event.y - downY
                    val outwardDrag = when (gesture) {
                        Gesture.RUBBER_TOP -> drag.coerceAtLeast(0f)
                        Gesture.RUBBER_BOTTOM -> drag.coerceAtMost(0f)
                        else -> 0f
                    }
                    translationY = LearningUiPolicy.rubberBandTranslation(outwardDrag)
                    invalidate()
                    return true
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                val wasRubber = isRubberGesture() || abs(translationY) > 0.5f
                val handled = if (wasRubber) true else super.onTouchEvent(event)
                gesture = Gesture.UNDECIDED
                parent?.requestDisallowInterceptTouchEvent(false)
                if (wasRubber) rebound()
                return handled
            }
        }
        return super.onTouchEvent(event)
    }

    private fun prepareDown(event: MotionEvent) {
        rebound?.cancel()
        gesture = Gesture.UNDECIDED
        downX = event.x
        downY = event.y - translationY / LearningUiPolicy.RUBBER_BAND_FACTOR
        startedAtTop = !canScrollVertically(-1)
        startedAtBottom = !canScrollVertically(1)
        parent?.requestDisallowInterceptTouchEvent(false)
    }

    private fun lockDirection(event: MotionEvent) {
        if (gesture != Gesture.UNDECIDED) return
        val dx = event.x - downX
        val dy = event.y - downY
        if (abs(dx) <= touchSlop && abs(dy) <= touchSlop) return
        gesture = when {
            abs(dx) > abs(dy) -> Gesture.HORIZONTAL
            startedAtTop && dy > 0f -> Gesture.RUBBER_TOP
            startedAtBottom && dy < 0f -> Gesture.RUBBER_BOTTOM
            else -> Gesture.VERTICAL
        }
    }

    private fun isRubberGesture(): Boolean =
        gesture == Gesture.RUBBER_TOP || gesture == Gesture.RUBBER_BOTTOM

    private fun rebound() {
        rebound?.cancel()
        val start = translationY
        if (abs(start) <= 0.5f) {
            translationY = 0f
            return
        }
        rebound = ValueAnimator.ofFloat(start, 0f).apply {
            duration = LearningUiPolicy.RUBBER_REBOUND_DURATION_MS
            interpolator = PathInterpolator(0.25f, 1f, 0.5f, 1f)
            addUpdateListener { translationY = it.animatedValue as Float }
            start()
        }
    }

    override fun onDetachedFromWindow() {
        rebound?.cancel()
        rebound = null
        super.onDetachedFromWindow()
    }
}

internal class LiveGlassSurfaceLayout(context: Context) : LinearLayout(context) {
    private val shape = Path()
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(1).toFloat()
        color = UiColors.OUTLINE
    }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(248, 255, 255, 255)
    }
    private var radius = dp(18).toFloat()
    private var backdrop: LiveGlassBackdrop? = null

    init {
        setWillNotDraw(false)
        background = null
    }

    fun setBackdropSource(source: View) {
        backdrop = LiveGlassBackdrop(source)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        shape.reset()
        shape.addRoundRect(RectF(0f, 0f, width.toFloat(), height.toFloat()), radius, radius, Path.Direction.CW)
        canvas.save()
        canvas.clipPath(shape)
        val drawn = backdrop?.draw(canvas, this, shape) == true
        if (!drawn) canvas.drawColor(Color.argb(248, 255, 255, 255))
        canvas.drawPath(shape, fill)
        canvas.restore()
        canvas.drawPath(shape, stroke)
        super.onDraw(canvas)
    }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density + 0.5f).toInt()
}
