package com.cupet.fallingball.learning

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.getSystemService
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import androidx.recyclerview.widget.DefaultItemAnimator
import com.cupet.fallingball.MeasurementHistoryActivity
import com.cupet.fallingball.R
import com.cupet.fallingball.analysis.MeasurementHistoryStore
import com.cupet.fallingball.analysis.MeasurementRecord
import com.cupet.fallingball.ui.DisplacementTimeChartView
import com.cupet.fallingball.ui.UiColors
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.provider.OpenableColumns
import android.util.Base64
import java.util.Locale

class DataAssistantActivity : AppCompatActivity() {
    private lateinit var services: LearningServices
    private lateinit var measurementStore: MeasurementHistoryStore
    private lateinit var root: FrameLayout
    private lateinit var toolbar: LinearLayout
    private lateinit var factLayer: LinearLayout
    private lateinit var conversation: ElasticRecyclerView
    private lateinit var promptPanel: LinearLayout
    private lateinit var questionsPanel: LinearLayout
    private lateinit var inputBar: LiveGlassSurfaceLayout
    private lateinit var inputArea: LinearLayout
    private lateinit var attachmentStrip: LinearLayout
    private lateinit var input: EditText
    private lateinit var sendStop: ImageButton
    private lateinit var oldDataLabel: TextView
    private lateinit var adapter: LearningConversationAdapter
    private var record: MeasurementRecord? = null
    private var session: LearningSession? = null
    private var generationJob: Job? = null
    private var questionPollingJob: Job? = null
    private var glassTexture: Bitmap? = null
    private var liveBackdrop: LiveGlassBackdrop? = null
    private var generating = false
    private var sessionTransitioning = false
    private var navigationBottom = 0
    private var imeBottom = 0
    private var systemTop = 0
    private var detailOpened = false
    private var followTailPosted = false
    private var followRichContentTail = false
    private var renderedSessionId: String? = null
    private val selectedAttachments = mutableListOf<LearningInputAttachment>()
    /** Keeps attachment bytes available for an in-process retry without persisting them to disk. */
    private val attachmentCacheByMessageId = mutableMapOf<String, List<LearningInputAttachment>>()

    private val attachmentLauncher = registerForActivityResult(
        ActivityResultContracts.OpenMultipleDocuments()
    ) { uris ->
        uris.forEach { uri -> readAttachment(uri)?.let { attachment ->
            if (selectedAttachments.size >= MAX_ATTACHMENTS) {
                return@let
            }
            if (selectedAttachments.none { it.name == attachment.name && it.base64 == attachment.base64 }) {
                selectedAttachments += attachment
            }
        } }
        renderAttachmentStrip()
    }

    private val historyLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode != Activity.RESULT_OK) return@registerForActivityResult
        val action = result.data?.getStringExtra(LearningSessionHistoryActivity.EXTRA_ACTION)
        when (action) {
            LearningSessionHistoryActivity.ACTION_NEW -> createNewSession()
            LearningSessionHistoryActivity.ACTION_SWITCH -> {
                result.data?.getStringExtra(LearningSessionHistoryActivity.EXTRA_SESSION_ID)
                    ?.let(::switchSession)
            }
            LearningSessionHistoryActivity.ACTION_ENDED -> {
                result.data?.getStringExtra(LearningSessionHistoryActivity.EXTRA_SESSION_ID)
                    ?.let(::switchSession)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        services = LearningServiceFactory.get(this)
        measurementStore = MeasurementHistoryStore(filesDir.resolve("measurement_history"))
        setContentView(buildScreen())
        configureInsets()
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() = finish()
        })
        val measurementId = intent.getStringExtra(EXTRA_MEASUREMENT_ID)
        if (measurementId == null) {
            showLoadFailure()
        } else {
            load(measurementId, intent.getStringExtra(EXTRA_SESSION_ID))
        }
    }

    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        if (event.actionMasked == MotionEvent.ACTION_DOWN &&
            ::input.isInitialized && input.hasFocus()
        ) {
            val inputBounds = Rect()
            val insideInput = input.getGlobalVisibleRect(inputBounds) &&
                inputBounds.contains(event.rawX.toInt(), event.rawY.toInt())
            if (!insideInput) {
                input.clearFocus()
                getSystemService<InputMethodManager>()
                    ?.hideSoftInputFromWindow(input.windowToken, 0)
            }
        }
        return super.dispatchTouchEvent(event)
    }

    private fun buildScreen(): View {
        root = FrameLayout(this).apply { setBackgroundColor(UiColors.PAGE_BACKGROUND) }
        factLayer = LinearLayout(this).apply {
            id = R.id.learning_fact_layer
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(dp(18), dp(18), dp(18), dp(14))
        }
        root.addView(factLayer, FrameLayout.LayoutParams(-1, -1))

        conversation = ElasticRecyclerView(this).apply {
            id = R.id.learning_conversation
            layoutManager = LinearLayoutManager(this@DataAssistantActivity).apply {
                stackFromEnd = false
            }
            itemAnimator = DefaultItemAnimator().apply {
                addDuration = 180L
                removeDuration = 160L
                changeDuration = 0L
                supportsChangeAnimations = false
            }
            setBackgroundColor(Color.TRANSPARENT)
            clipToPadding = false
            overScrollMode = View.OVER_SCROLL_NEVER
            visibility = View.GONE
        }
        adapter = LearningConversationAdapter(
            this,
            ::retryMessage,
            ::copyMessage,
            onContentHeightChanged = {
                if (followRichContentTail) followStreamingTail(true)
            }
        )
        conversation.adapter = adapter
        conversation.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                super.onScrolled(recyclerView, dx, dy)
                if (dy < 0) followRichContentTail = false
            }
        })
        root.addView(conversation, FrameLayout.LayoutParams(-1, -1))

        promptPanel = LinearLayout(this).apply {
            id = R.id.learning_prompt_panel
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(dp(18), 0, dp(18), dp(8))
        }
        promptPanel.addView(TextView(this).apply {
            setText(R.string.learning_prompt)
            textSize = 22f
            setTextColor(UiColors.ON_SURFACE)
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
        }, LinearLayout.LayoutParams(-1, dp(42)))
        questionsPanel = LinearLayout(this).apply {
            id = R.id.learning_questions
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
        }
        promptPanel.addView(questionsPanel, LinearLayout.LayoutParams(-1, -2))
        root.addView(promptPanel, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))

        attachmentStrip = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(10), dp(4), dp(10), 0)
            visibility = View.GONE
        }
        inputBar = LiveGlassSurfaceLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(12), dp(8), dp(10), dp(8))
            setBackdropSource(factLayer)
        }
        input = EditText(this).apply {
            id = R.id.learning_input
            hint = getString(R.string.learning_input_hint)
            setHintTextColor(UiColors.ON_SURFACE_MUTED)
            setTextColor(UiColors.ON_SURFACE)
            textSize = 15f
            maxLines = 4
            minHeight = dp(48)
            background = null
            setPadding(dp(12), 0, dp(8), 0)
            setOnFocusChangeListener { _, focused ->
                updateFactEmphasis(
                    session?.hasConversation == true || focused || imeBottom > 0
                )
            }
        }
        inputBar.addView(ImageButton(this).apply {
            id = R.id.learning_attachment
            setImageResource(R.drawable.ic_learning_attachment)
            contentDescription = "添加附件"
            setColorFilter(UiColors.ON_SURFACE_VARIANT)
            setBackgroundResource(R.drawable.bg_learning_icon_button)
            minimumWidth = dp(48)
            minimumHeight = dp(48)
            setOnClickListener {
                attachmentLauncher.launch(arrayOf("image/*", "text/*", "application/pdf"))
            }
        }, LinearLayout.LayoutParams(dp(48), dp(48)))
        inputBar.addView(input, LinearLayout.LayoutParams(0, -2, 1f))
        sendStop = ImageButton(this).apply {
            id = R.id.learning_send_stop
            setImageResource(R.drawable.ic_learning_send)
            contentDescription = getString(R.string.learning_send)
            setColorFilter(UiColors.PRIMARY)
            setBackgroundResource(R.drawable.bg_learning_icon_button)
            minimumWidth = dp(48)
            minimumHeight = dp(48)
            setOnClickListener { if (generating) stopGeneration() else sendInput() }
        }
        inputBar.addView(sendStop, LinearLayout.LayoutParams(dp(52), dp(52)))
        inputArea = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(attachmentStrip, LinearLayout.LayoutParams(-1, -2))
            addView(inputBar, LinearLayout.LayoutParams(-1, -2))
        }
        root.addView(inputArea, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))

        toolbar = buildToolbar()
        root.addView(toolbar, FrameLayout.LayoutParams(-1, dp(58), Gravity.TOP))
        return root
    }

    private fun buildToolbar(): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        setPadding(dp(4), 0, dp(6), 0)
        setBackgroundColor(UiColors.SURFACE)
        addView(ImageButton(this@DataAssistantActivity).apply {
            id = R.id.learning_back
            setImageResource(R.drawable.ic_learning_back)
            setColorFilter(UiColors.ON_SURFACE_VARIANT)
            setBackgroundResource(R.drawable.bg_learning_icon_button)
            contentDescription = getString(R.string.history_back)
            setOnClickListener { finish() }
        }, LinearLayout.LayoutParams(dp(50), dp(52)))
        addView(TextView(this@DataAssistantActivity).apply {
            setText(R.string.learning_title)
            textSize = 19f
            setTextColor(UiColors.ON_SURFACE)
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER_VERTICAL
        }, LinearLayout.LayoutParams(0, -1, 1f))
        oldDataLabel = TextView(this@DataAssistantActivity).apply {
            setText(R.string.learning_old_data)
            textSize = 11f
            setTextColor(UiColors.WARNING)
            gravity = Gravity.CENTER
            visibility = View.GONE
            setPadding(dp(5), 0, dp(5), 0)
        }
        addView(oldDataLabel, LinearLayout.LayoutParams(-2, -1))
        addView(MaterialButton(
            this@DataAssistantActivity,
            null,
            com.google.android.material.R.attr.borderlessButtonStyle
        ).apply {
            setText(R.string.learning_history)
            textSize = 11f
            minWidth = 0
            maxLines = 1
            insetTop = 0
            insetBottom = 0
            setOnClickListener { openHistory() }
        }, LinearLayout.LayoutParams(dp(76), dp(48)))
        addView(MaterialButton(
            this@DataAssistantActivity,
            null,
            com.google.android.material.R.attr.borderlessButtonStyle
        ).apply {
            setText(R.string.learning_view_detail)
            textSize = 11f
            minWidth = 0
            maxLines = 1
            insetTop = 0
            insetBottom = 0
            setOnClickListener { openDetail() }
        }, LinearLayout.LayoutParams(dp(104), dp(48)))
    }

    private fun configureInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            val ime = insets.getInsets(WindowInsetsCompat.Type.ime())
            systemTop = bars.top
            navigationBottom = bars.bottom
            imeBottom = ime.bottom
            updateInsetLayout()
            updateFactEmphasis(session?.hasConversation == true || imeBottom > 0 || input.hasFocus())
            insets
        }
    }

    private fun updateInsetLayout() {
        toolbar.layoutParams = (toolbar.layoutParams as FrameLayout.LayoutParams).apply {
            height = dp(58) + systemTop
        }
        toolbar.setPadding(dp(4), systemTop, dp(6), 0)
        factLayer.layoutParams = (factLayer.layoutParams as FrameLayout.LayoutParams).apply {
            topMargin = dp(58) + systemTop
            bottomMargin = navigationBottom
        }
        val movingBottom = maxOf(navigationBottom, imeBottom)
        val comfortableBottom = movingBottom + dp(LearningUiPolicy.INPUT_BOTTOM_LIFT_DP)
        inputArea.layoutParams = (inputArea.layoutParams as FrameLayout.LayoutParams).apply {
            bottomMargin = comfortableBottom
        }
        promptPanel.layoutParams = (promptPanel.layoutParams as FrameLayout.LayoutParams).apply {
            bottomMargin = comfortableBottom + dp(68) +
                if (selectedAttachments.isEmpty()) 0 else dp(42)
        }
        conversation.setPadding(
            0,
            dp(58) + systemTop + dp(6),
            0,
            comfortableBottom + dp(72)
        )
    }

    private fun load(measurementId: String, requestedSessionId: String?) {
        lifecycleScope.launch {
            val loaded = withContext(Dispatchers.IO) { measurementStore.load(measurementId) }
            if (loaded == null || loaded.samples.isEmpty()) {
                showLoadFailure()
                return@launch
            }
            record = loaded
            renderFacts(loaded)
            val opened = withContext(Dispatchers.IO) {
                services.tutor.openSession(loaded, requestedSessionId)
            }
            session = opened
            renderSession(opened)
            val forceQuestionRegeneration = isDebugBuild() &&
                intent.getBooleanExtra(EXTRA_FORCE_QUESTION_REGENERATION, false)
            if (opened.status != LearningSessionStatus.ENDED &&
                (forceQuestionRegeneration || !opened.hasConversation)
            ) {
                QuestionWorkScheduler.enqueue(
                    this@DataAssistantActivity,
                    loaded.id
                )
                if (!opened.hasConversation) pollQuestions(opened)
            }
        }
    }

    private fun isDebugBuild(): Boolean =
        applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0

    private fun renderFacts(value: MeasurementRecord) {
        factLayer.removeAllViews()
        val viscosity = value.viscosityResult.takeIf { value.hasReliableTerminalEstimate }
            ?.wallCorrectedStokesPaS
        val resultRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            addView(
                factValue(
                    getString(R.string.learning_viscosity),
                    viscosity?.let { String.format(Locale.ROOT, "%.4f Pa·s", it) }
                        ?: getString(R.string.learning_value_unavailable),
                    27f
                ),
                LinearLayout.LayoutParams(0, -1, 1f)
            )
            addView(View(context).apply {
                setBackgroundColor(UiColors.DIVIDER)
            }, LinearLayout.LayoutParams(dp(1), dp(72)).apply {
                gravity = Gravity.CENTER_VERTICAL
            })
            addView(
                factValue(
                    getString(R.string.learning_terminal_velocity),
                    value.summary?.terminalVelocityMmPerS
                        ?.takeIf { value.hasReliableTerminalEstimate }
                        ?.let { String.format(Locale.ROOT, "%.2f mm/s", it) }
                        ?: getString(R.string.learning_value_unavailable),
                    27f
                ),
                LinearLayout.LayoutParams(0, -1, 1f)
            )
        }
        factLayer.addView(resultRow, LinearLayout.LayoutParams(-1, dp(88)))
        factLayer.addView(DisplacementTimeChartView(this).apply {
            id = R.id.learning_chart
            setRecord(value)
            isEnabled = false
            isClickable = false
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
            contentDescription = "只读位移时间图"
        }, LinearLayout.LayoutParams(-1, 0, 1f).apply {
            topMargin = dp(8)
            bottomMargin = dp(LearningUiPolicy.EMPTY_FOREGROUND_RESERVED_DP)
        })
        factLayer.post {
            glassTexture?.recycle()
            glassTexture = if (android.os.Build.VERSION.SDK_INT < 31) {
                createBlurredFactTexture(factLayer)
            } else {
                null
            }
            adapter.updateGlassTexture(glassTexture, factLayer)
            liveBackdrop = LiveGlassBackdrop(factLayer)
            adapter.updateLiveBackdrop(liveBackdrop)
        }
    }

    private fun factValue(label: String, value: String, valueSize: Float): View =
        LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            addView(TextView(context).apply {
                text = label
                textSize = 12f
                setTextColor(UiColors.ON_SURFACE_VARIANT)
                gravity = Gravity.CENTER
            }, LinearLayout.LayoutParams(-1, dp(24)))
            addView(TextView(context).apply {
                text = value
                textSize = valueSize
                setTextColor(UiColors.ON_SURFACE)
                typeface = Typeface.DEFAULT_BOLD
                gravity = Gravity.CENTER
            }, LinearLayout.LayoutParams(-1, 0, 1f))
        }

    private fun renderSession(
        value: LearningSession,
        preserveScroll: Boolean = false,
        reasoning: String = ""
    ) {
        val changedSession = renderedSessionId?.let { it != value.id } == true
        if (changedSession) {
            // An empty session hides the RecyclerView before its remove animations can finish.
            // Detach its adapter at the session boundary so old holders cannot reappear on the
            // first turn when the hidden list becomes visible again.
            conversation.visibility = View.GONE
            conversation.itemAnimator?.endAnimations()
            conversation.adapter = null
        }
        session = value
        if (!preserveScroll) followRichContentTail = false
        adapter.submit(value, reasoning)
        if (changedSession) {
            conversation.recycledViewPool.clear()
            conversation.adapter = adapter
        }
        renderedSessionId = value.id
        val currentFingerprint = record?.let(MeasurementFingerprint::sha256)
        oldDataLabel.visibility = if (currentFingerprint != null &&
            currentFingerprint != value.measurementFingerprint
        ) View.VISIBLE else View.GONE
        val hasMessages = value.hasConversation
        conversation.visibility = if (hasMessages) View.VISIBLE else View.GONE
        promptPanel.visibility = if (!hasMessages && value.status != LearningSessionStatus.ENDED) {
            View.VISIBLE
        } else {
            View.GONE
        }
        conversation.isNestedScrollingEnabled = LearningUiPolicy.conversationScrollable(hasMessages)
        conversation.overScrollMode = if (
            hasMessages && LearningUiPolicy.SYSTEM_OVERSCROLL_ENABLED
        ) View.OVER_SCROLL_ALWAYS else View.OVER_SCROLL_NEVER
        input.isEnabled = value.status == LearningSessionStatus.ACTIVE &&
            !generating && !sessionTransitioning
        sendStop.isEnabled = value.status == LearningSessionStatus.ACTIVE && !sessionTransitioning
        if (value.status == LearningSessionStatus.ENDED) {
            input.hint = getString(R.string.learning_session_ended)
        } else {
            input.hint = getString(R.string.learning_input_hint)
        }
        updateFactEmphasis(hasMessages || imeBottom > 0 || input.hasFocus())
        if (!preserveScroll && hasMessages) {
            conversation.post {
                (conversation.layoutManager as? LinearLayoutManager)
                    ?.scrollToPositionWithOffset(adapter.latestTurnStartPosition(), 0)
            }
        }
    }

    private fun pollQuestions(opened: LearningSession) {
        questionPollingJob?.cancel()
        questionPollingJob = lifecycleScope.launch {
            while (isActive && session?.id == opened.id && session?.hasConversation == false) {
                val prepared = withContext(Dispatchers.IO) {
                    services.store.loadQuestions(opened.measurementId, opened.measurementFingerprint)
                }
                if (prepared != null) {
                    showQuestions(prepared.questions)
                    break
                }
                delay(600)
            }
        }
    }

    private fun showQuestions(questions: List<String>) {
        if (session?.hasConversation == true || session?.status != LearningSessionStatus.ACTIVE) return
        questionsPanel.removeAllViews()
        val questionEmojis = LearningUiPolicy.randomQuestionEmojis(questions.size)
        questions.forEachIndexed { index, question ->
            questionsPanel.addView(TextView(this).apply {
                text = LearningUiPolicy.decorateSuggestedQuestion(question, questionEmojis[index])
                textSize = 13f
                setTextColor(UiColors.ON_SURFACE)
                gravity = Gravity.CENTER_VERTICAL
                maxLines = 3
                minHeight = dp(44)
                setLineSpacing(0f, 1.06f)
                setPadding(dp(14), dp(8), dp(14), dp(8))
                background = android.graphics.drawable.GradientDrawable().apply {
                    setColor(UiColors.SURFACE)
                    setStroke(dp(1), UiColors.OUTLINE)
                    cornerRadius = dp(7).toFloat()
                }
                alpha = 0f
                translationY = dp(LearningUiPolicy.QUESTION_START_TRANSLATION_DP).toFloat()
                setOnClickListener { sendMessage(question) }
                postDelayed({
                    animate().alpha(1f).translationY(0f)
                        .setDuration(LearningUiPolicy.QUESTION_FADE_DURATION_MS).start()
                }, index * LearningUiPolicy.QUESTION_STAGGER_MS)
            }, LinearLayout.LayoutParams(-1, -2).apply {
                topMargin = dp(6)
            })
        }
        questionsPanel.visibility = View.VISIBLE
    }

    private fun sendInput() {
        val value = input.text.toString().trim()
        if (value.isEmpty()) return
        input.text.clear()
        sendMessage(value)
    }

    private fun readAttachment(uri: Uri): LearningInputAttachment? = runCatching {
        val mediaType = contentResolver.getType(uri).orEmpty().ifBlank { "application/octet-stream" }
        val name = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
            ?: "附件"
        val bytes = contentResolver.openInputStream(uri)?.use { stream ->
            stream.readBytes().also { require(it.size <= MAX_ATTACHMENT_BYTES) { "单个附件不能超过 8 MB" } }
        } ?: return null
        LearningInputAttachment(name.take(120), mediaType, Base64.encodeToString(bytes, Base64.NO_WRAP))
    }.onFailure {
        Toast.makeText(this, it.message ?: "无法读取附件", Toast.LENGTH_LONG).show()
    }.getOrNull()

    private fun renderAttachmentStrip() {
        if (!::attachmentStrip.isInitialized) return
        attachmentStrip.removeAllViews()
        selectedAttachments.forEachIndexed { index, attachment ->
            attachmentStrip.addView(TextView(this).apply {
                text = "${attachment.name}  ×"
                textSize = 12f
                setTextColor(UiColors.ON_PRIMARY_CONTAINER)
                setPadding(dp(10), dp(5), dp(10), dp(5))
                background = android.graphics.drawable.GradientDrawable().apply {
                    setColor(UiColors.PRIMARY_CONTAINER)
                    setStroke(dp(1), UiColors.OUTLINE)
                    cornerRadius = dp(12).toFloat()
                }
                maxLines = 1
                ellipsize = android.text.TextUtils.TruncateAt.END
                setOnClickListener {
                    selectedAttachments.removeAt(index)
                    renderAttachmentStrip()
                }
            }, LinearLayout.LayoutParams(0, dp(32), 1f).apply {
                marginEnd = dp(6)
            })
        }
        attachmentStrip.visibility = if (selectedAttachments.isEmpty()) View.GONE else View.VISIBLE
        if (::root.isInitialized) updateInsetLayout()
    }

    private fun sendMessage(value: String, retryAssistantId: String? = null) {
        val current = session ?: return
        if (generating || sessionTransitioning || current.status != LearningSessionStatus.ACTIVE) return
        val attachments = selectedAttachments.toList()
        selectedAttachments.clear()
        renderAttachmentStrip()
        // The student is asking a real question now, so the queued suggestion drafting for this
        // measurement is pointless and would only compete for the same model capacity.
        QuestionWorkScheduler.cancelPending(this)
        questionsPanel.removeAllViews()
        questionsPanel.visibility = View.GONE
        promptPanel.visibility = View.GONE
        getSystemService<InputMethodManager>()?.hideSoftInputFromWindow(input.windowToken, 0)
        val retryingMessage = retryAssistantId?.let { id -> current.messages.firstOrNull { it.id == id } }
        val retryIndex = retryAssistantId?.let { id -> current.messages.indexOfFirst { it.id == id } } ?: -1
        val retryUser = retryIndex.takeIf { it > 0 }?.let { current.messages[it - 1] }
        val requestAttachments = if (retryingMessage != null && attachments.isEmpty()) {
            attachmentCacheByMessageId[retryUser?.id].orEmpty()
        } else {
            attachments
        }
        val base = if (retryIndex > 0) current.copy(
            messages = current.messages.take(retryIndex - 1),
            actions = if (retryUser?.turnId?.isNotBlank() == true) {
                current.actions.filter { it.turnId != retryUser.turnId }
            } else {
                current.actions.filter { it.startedAtEpochMs < retryUser?.createdAtEpochMs ?: Long.MIN_VALUE }
            }
        ) else current
        val turnId = retryUser?.turnId?.takeIf(String::isNotBlank)
            ?: java.util.UUID.randomUUID().toString()
        val localUser = retryUser?.copy(
            state = LearningMessageState.COMPLETE,
            turnId = turnId
        ) ?: LearningMessage(
            role = LearningMessageRole.USER,
            text = value,
            attachmentNames = attachments.map { it.name },
            turnId = turnId
        )
        val localAssistant = LearningMessage(
            role = LearningMessageRole.ASSISTANT,
            text = "",
            turnId = turnId,
            state = LearningMessageState.STREAMING
        )
        val turnIdentity = LearningTurnIdentity(
            turnId = turnId,
            userMessageId = localUser.id,
            assistantMessageId = localAssistant.id
        )
        localUser.let { attachmentCacheByMessageId[it.id] = requestAttachments }
        var display = base.copy(messages = base.messages + localUser + localAssistant)
        val streamingReasoning = StringBuilder()
        var renderedReasoningLength = 0
        var lastStreamingRenderAt = 0L
        renderSession(display)
        setGenerating(true)
        generationJob = lifecycleScope.launch {
            var pendingTextRenderJob: Job? = null
            fun flushStreamingText() {
                val followTail = captureStreamingTail()
                adapter.updateStreaming(display, streamingReasoning.toString())
                lastStreamingRenderAt = System.currentTimeMillis()
                followStreamingTail(followTail)
            }
            fun cancelPendingTextRender() {
                pendingTextRenderJob?.cancel()
                pendingTextRenderJob = null
            }
            try {
                val chart = if (shouldAttachChart(current, value)) {
                    withContext(Dispatchers.Default) {
                        record?.let { ChartImageEncoder.encodeRecord(this@DataAssistantActivity, it) }
                    }
                } else {
                    null
                }
                val request = if (retryingMessage != null) {
                    services.tutor.retry(
                        current.id,
                        retryingMessage.id,
                        chart,
                        requestAttachments,
                        turnIdentity
                    )
                } else {
                    services.tutor.send(current.id, value, chart, requestAttachments, turnIdentity)
                }
                request.collect sessionEvent@ { event ->
                    if (session?.id != current.id) return@sessionEvent
                    if (event is LearningEvent.Completed && event.session.id != current.id) {
                        return@sessionEvent
                    }
                    if (event is LearningEvent.Stopped && event.session.id != current.id) {
                        return@sessionEvent
                    }
                    when (event) {
                    LearningEvent.ReasoningStarted -> {
                        cancelPendingTextRender()
                        if (streamingReasoning.isNotEmpty()) streamingReasoning.append('\n')
                        val followTail = captureStreamingTail()
                        adapter.updateStreaming(display, streamingReasoning.toString())
                        renderedReasoningLength = streamingReasoning.length
                        lastStreamingRenderAt = System.currentTimeMillis()
                        followStreamingTail(followTail)
                    }
                    is LearningEvent.ReasoningDelta -> {
                        val remaining = LearningUiPolicy.MAX_REASONING_DISPLAY_CHARS -
                            streamingReasoning.length
                        if (remaining > 0) streamingReasoning.append(event.text.take(remaining))
                        val shouldRender = streamingReasoning.length - renderedReasoningLength >=
                            LearningUiPolicy.REASONING_RENDER_BATCH_CHARS ||
                            event.text.any { it == '\n' || it == '。' || it == '？' || it == '！' }
                        val timeReady = System.currentTimeMillis() - lastStreamingRenderAt >=
                            STREAMING_RENDER_INTERVAL_MS
                        if (shouldRender || timeReady || remaining <= event.text.length) {
                            cancelPendingTextRender()
                            val followTail = captureStreamingTail()
                            adapter.updateStreaming(display, streamingReasoning.toString())
                            renderedReasoningLength = streamingReasoning.length
                            lastStreamingRenderAt = System.currentTimeMillis()
                            followStreamingTail(followTail)
                        }
                    }
                    is LearningEvent.TextDelta -> {
                        display = display.copy(messages = display.messages.map { message ->
                            if (message.id == localAssistant.id) {
                                message.copy(text = message.text + event.text)
                            } else message
                        })
                        val elapsed = System.currentTimeMillis() - lastStreamingRenderAt
                        val timeReady = elapsed >= STREAMING_RENDER_INTERVAL_MS
                        if (timeReady || event.text.any { it == '\n' || it == '。' || it == '？' || it == '！' }) {
                            cancelPendingTextRender()
                            flushStreamingText()
                        } else {
                            cancelPendingTextRender()
                            pendingTextRenderJob = launch {
                                delay((STREAMING_RENDER_INTERVAL_MS - elapsed).coerceAtLeast(1L))
                                pendingTextRenderJob = null
                                flushStreamingText()
                            }
                        }
                    }
                    is LearningEvent.ActionStarted -> {
                        cancelPendingTextRender()
                        val followTail = captureStreamingTail()
                        display = display.copy(
                            messages = display.messages.map { message ->
                                if (message.id == localAssistant.id && event.action.turnId.isNotBlank()) {
                                    message.copy(turnId = event.action.turnId)
                                } else message
                            },
                            actions = display.actions + event.action
                        )
                        adapter.updateStructure(display, streamingReasoning.toString())
                        followStreamingTail(followTail)
                    }
                    is LearningEvent.ActionCompleted -> {
                        cancelPendingTextRender()
                        val followTail = captureStreamingTail()
                        display = display.copy(
                            messages = display.messages.map { message ->
                                if (message.id == localAssistant.id && event.action.turnId.isNotBlank()) {
                                    message.copy(turnId = event.action.turnId)
                                } else message
                            },
                            actions = display.actions.map {
                                if (it.id == event.action.id) event.action else it
                            }
                        )
                        adapter.updateStructure(display, streamingReasoning.toString())
                        followStreamingTail(followTail)
                    }
                    is LearningEvent.Citation -> {
                        cancelPendingTextRender()
                        display = display.copy(
                            references = (display.references + event.reference).distinctBy { it.id }
                        )
                        adapter.updateStructure(display, streamingReasoning.toString())
                    }
                    is LearningEvent.Completed -> {
                        cancelPendingTextRender()
                        if (requestAttachments.isNotEmpty()) {
                            event.session.messages.lastOrNull {
                                it.role == LearningMessageRole.USER
                            }?.let { persistedUser ->
                                attachmentCacheByMessageId[persistedUser.id] = requestAttachments
                            }
                        }
                        // Do not re-anchor the list when the user has browsed older turns while
                        // generation was finishing. The streaming path already owns tail-follow.
                        renderSession(
                            event.session,
                            preserveScroll = true,
                            reasoning = streamingReasoning.toString()
                        )
                        setGenerating(false)
                    }
                    is LearningEvent.Stopped -> {
                        cancelPendingTextRender()
                        if (requestAttachments.isNotEmpty()) {
                            event.session.messages.lastOrNull {
                                it.role == LearningMessageRole.USER
                            }?.let { persistedUser ->
                                attachmentCacheByMessageId[persistedUser.id] = requestAttachments
                            }
                        }
                        renderSession(
                            event.session,
                            preserveScroll = true,
                            reasoning = streamingReasoning.toString()
                        )
                        setGenerating(false)
                    }
                    is LearningEvent.Error -> {
                        cancelPendingTextRender()
                        Toast.makeText(this@DataAssistantActivity, event.message, Toast.LENGTH_LONG).show()
                        session = withContext(Dispatchers.IO) {
                            services.store.loadSession(current.measurementId, current.id)
                        }
                        if (requestAttachments.isNotEmpty()) {
                            session?.messages?.lastOrNull {
                                it.role == LearningMessageRole.USER
                            }?.let { persistedUser ->
                                attachmentCacheByMessageId[persistedUser.id] = requestAttachments
                            }
                        }
                        session?.let { renderSession(it, preserveScroll = true) }
                        setGenerating(false)
                    }
                    }
                }
            } finally {
                cancelPendingTextRender()
                val completedJob = coroutineContext[Job]
                if (generationJob === completedJob) {
                    generationJob = null
                    setGenerating(false)
                }
            }
        }
    }

    private fun shouldAttachChart(current: LearningSession, value: String): Boolean =
        LearningUiPolicy.shouldAttachChart(
            hasPriorUserMessage = current.messages.any { it.role == LearningMessageRole.USER },
            message = value
        )

    private fun stopGeneration() {
        services.tutor.stopGenerating()
        generationJob?.cancel()
        sendStop.isEnabled = false
    }

    private fun setGenerating(value: Boolean) {
        generating = value
        input.isEnabled = !value && !sessionTransitioning &&
            session?.status == LearningSessionStatus.ACTIVE
        sendStop.isEnabled = !sessionTransitioning &&
            session?.status == LearningSessionStatus.ACTIVE
        sendStop.setImageResource(
            if (value) R.drawable.ic_learning_stop else R.drawable.ic_learning_send
        )
        sendStop.contentDescription = getString(
            if (value) R.string.learning_stop else R.string.learning_send
        )
    }

    private fun isFollowingStreamingTail(): Boolean {
        val manager = conversation.layoutManager as? LinearLayoutManager ?: return false
        return !conversation.canScrollVertically(1) ||
            manager.findLastCompletelyVisibleItemPosition() >= adapter.itemCount - 1
    }

    private fun captureStreamingTail(): Boolean {
        val follow = followRichContentTail || isFollowingStreamingTail()
        followRichContentTail = follow
        return follow
    }

    private fun followStreamingTail(follow: Boolean) {
        if (!follow || followTailPosted) return
        followTailPosted = true
        conversation.postOnAnimation {
            followTailPosted = false
            // Preserve the current list origin and consume only the new height. Repositioning
            // the last row with an offset on every delta makes the whole conversation jump.
            // Use the intent captured before the WebView grew; checking layout state here would
            // see the newly added height and incorrectly conclude that the user left the tail.
            if (followRichContentTail && conversation.canScrollVertically(1)) {
                conversation.scrollBy(0, Int.MAX_VALUE)
            }
        }
    }

    private fun updateFactEmphasis(dim: Boolean) {
        factLayer.animate().cancel()
        factLayer.animate()
            .alpha(LearningUiPolicy.factAlpha(dim))
            .setDuration(180)
            .withLayer()
            .setUpdateListener {
                // Keep the GPU backdrop in lockstep with the fact layer animation. Without an
                // invalidation here, RenderNode may keep showing the first captured frame.
                factLayer.invalidate()
                conversation.invalidate()
                inputBar.invalidate()
            }
            .start()
    }

    private fun openHistory() {
        val current = session ?: return
        historyLauncher.launch(
            Intent(this, LearningSessionHistoryActivity::class.java)
                .putExtra(EXTRA_MEASUREMENT_ID, current.measurementId)
                .putExtra(LearningSessionHistoryActivity.EXTRA_CURRENT_SESSION_ID, current.id)
        )
    }

    private fun openDetail() {
        val measurementId = record?.id ?: return
        detailOpened = true
        startActivity(
            Intent(this, MeasurementHistoryActivity::class.java)
                .putExtra(MeasurementHistoryActivity.EXTRA_RECORD_ID, measurementId)
                .putExtra(MeasurementHistoryActivity.EXTRA_DETAIL_ONLY, true)
        )
    }

    override fun onResume() {
        super.onResume()
        if (!detailOpened) return
        detailOpened = false
        val measurementId = record?.id ?: return
        val previousFingerprint = record?.let(MeasurementFingerprint::sha256) ?: return
        lifecycleScope.launch {
            val latest = withContext(Dispatchers.IO) { measurementStore.load(measurementId) }
            if (latest == null) {
                finish()
                return@launch
            }
            val latestFingerprint = MeasurementFingerprint.sha256(latest)
            if (latestFingerprint == previousFingerprint) return@launch
            record = latest
            renderFacts(latest)
            val opened = withContext(Dispatchers.IO) {
                services.tutor.openSession(latest)
            }
            renderSession(opened)
            QuestionWorkScheduler.enqueue(this@DataAssistantActivity, latest.id)
            pollQuestions(opened)
        }
    }

    private fun createNewSession() {
        val currentRecord = record ?: return
        beginSessionTransition()
        lifecycleScope.launch {
            try {
                val created = withContext(Dispatchers.IO) {
                    services.tutor.startNewSession(currentRecord)
                }
                questionsPanel.removeAllViews()
                renderSession(created)
                pollQuestions(created)
            } finally {
                endSessionTransition()
            }
        }
    }

    private fun switchSession(sessionId: String) {
        val currentRecord = record ?: return
        beginSessionTransition()
        lifecycleScope.launch {
            try {
                val selected = withContext(Dispatchers.IO) {
                    services.tutor.openSession(currentRecord, sessionId)
                }
                questionsPanel.removeAllViews()
                renderSession(selected)
                if (!selected.hasConversation && selected.status == LearningSessionStatus.ACTIVE) {
                    pollQuestions(selected)
                }
            } finally {
                endSessionTransition()
            }
        }
    }

    private fun beginSessionTransition() {
        sessionTransitioning = true
        questionPollingJob?.cancel()
        questionPollingJob = null
        if (generationJob != null) services.tutor.stopGenerating()
        generationJob?.cancel()
        input.isEnabled = false
        sendStop.isEnabled = false
    }

    private fun endSessionTransition() {
        sessionTransitioning = false
        setGenerating(false)
    }

    private fun retryMessage(assistantMessageId: String) {
        if (generating) return
        val current = session ?: return
        val assistant = current.messages.firstOrNull { it.id == assistantMessageId } ?: return
        if (assistant.role != LearningMessageRole.ASSISTANT) return
        val user = current.messages.getOrNull(current.messages.indexOf(assistant) - 1) ?: return
        sendMessage(user.text, retryAssistantId = assistantMessageId)
    }

    private fun copyMessage(message: LearningMessage) {
        val clipboard = getSystemService<android.content.ClipboardManager>() ?: return
        clipboard.setPrimaryClip(android.content.ClipData.newPlainText("数据助手", message.text))
        Toast.makeText(this, "已复制", Toast.LENGTH_SHORT).show()
    }

    private fun showLoadFailure() {
        Toast.makeText(this, R.string.learning_load_error, Toast.LENGTH_LONG).show()
        finish()
    }

    override fun onDestroy() {
        questionPollingJob?.cancel()
        glassTexture?.recycle()
        glassTexture = null
        liveBackdrop = null
        super.onDestroy()
    }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density + 0.5f).toInt()

    companion object {
        const val EXTRA_MEASUREMENT_ID = "measurement_id"
        const val EXTRA_SESSION_ID = "session_id"
        const val EXTRA_FORCE_QUESTION_REGENERATION =
            "com.cupet.fallingball.FORCE_QUESTION_REGENERATION"
        private const val MAX_ATTACHMENTS = 3
        private const val MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
        private const val STREAMING_RENDER_INTERVAL_MS = 48L
    }
}
