package com.cupet.fallingball.learning

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import com.cupet.fallingball.ui.UiColors
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.cupet.fallingball.R
import com.cupet.fallingball.analysis.MeasurementHistoryStore
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class LearningSessionHistoryActivity : AppCompatActivity() {
    private lateinit var services: LearningServices
    private lateinit var measurementId: String
    private var currentSessionId: String? = null
    private lateinit var list: ListView
    private lateinit var empty: TextView
    private var titleRefreshJob: Job? = null
    private var renderedSessionSignature = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val id = intent.getStringExtra(DataAssistantActivity.EXTRA_MEASUREMENT_ID)
        if (id == null) {
            finish()
            return
        }
        measurementId = id
        currentSessionId = intent.getStringExtra(EXTRA_CURRENT_SESSION_ID)
        services = LearningServiceFactory.get(this)
        setContentView(buildScreen())
    }

    override fun onStart() {
        super.onStart()
        startSessionRefresh()
    }

    private fun buildScreen(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(UiColors.PAGE_BACKGROUND)
        }
        val toolbar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(6), dp(4), dp(8), dp(4))
            setBackgroundColor(UiColors.SURFACE)
        }
        toolbar.addView(ImageButton(this).apply {
            id = R.id.learning_history_back
            setImageResource(R.drawable.ic_learning_back)
            setColorFilter(UiColors.ON_SURFACE_VARIANT)
            setBackgroundResource(R.drawable.bg_learning_icon_button)
            minimumWidth = dp(48)
            minimumHeight = dp(48)
            contentDescription = getString(R.string.history_back)
            setOnClickListener { finish() }
        }, LinearLayout.LayoutParams(dp(52), dp(52)))
        toolbar.addView(TextView(this).apply {
            setText(R.string.learning_history)
            textSize = 20f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(UiColors.ON_SURFACE)
            gravity = Gravity.CENTER_VERTICAL
        }, LinearLayout.LayoutParams(0, -1, 1f))
        toolbar.addView(MaterialButton(
            this, null, com.google.android.material.R.attr.borderlessButtonStyle
        ).apply {
            setText(R.string.learning_new_session)
            minWidth = 0
            setOnClickListener { returnAction(ACTION_NEW, null) }
        }, LinearLayout.LayoutParams(-2, dp(48)))
        toolbar.addView(MaterialButton(
            this, null, com.google.android.material.R.attr.borderlessButtonStyle
        ).apply {
            setText(R.string.learning_end_session)
            minWidth = 0
            setTextColor(UiColors.ERROR)
            setOnClickListener { confirmEndCurrent() }
        }, LinearLayout.LayoutParams(-2, dp(48)))
        root.addView(toolbar, LinearLayout.LayoutParams(-1, dp(60)))

        list = ListView(this).apply {
            dividerHeight = dp(1)
            setOnItemLongClickListener { _, _, position, _ ->
                renameSession((adapter as SessionAdapter).getItem(position))
                true
            }
        }
        empty = TextView(this).apply {
            setText(R.string.learning_session_empty)
            setTextColor(UiColors.ON_SURFACE_VARIANT)
            textSize = 15f
            gravity = Gravity.CENTER
        }
        val body = android.widget.FrameLayout(this).apply {
            addView(list, android.widget.FrameLayout.LayoutParams(-1, -1))
            addView(empty, android.widget.FrameLayout.LayoutParams(-1, -1))
        }
        root.addView(body, LinearLayout.LayoutParams(-1, 0, 1f))
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        return root
    }

    private fun startSessionRefresh() {
        titleRefreshJob?.cancel()
        titleRefreshJob = lifecycleScope.launch {
            while (isActive) {
                val waitingForGeneratedTitle = refreshSessionsOnce()
                if (!waitingForGeneratedTitle) return@launch
                delay(TITLE_REFRESH_INTERVAL_MS)
            }
        }
    }

    private suspend fun refreshSessionsOnce(): Boolean {
        val (sessions, currentFingerprint) = withContext(Dispatchers.IO) {
            val fingerprint = MeasurementHistoryStore(
                filesDir.resolve("measurement_history")
            ).load(measurementId)?.let(MeasurementFingerprint::sha256)
            services.store.listSessions(measurementId) to fingerprint
        }
        val signature = buildString {
            append(currentFingerprint).append('|')
            sessions.forEach {
                append(it.id).append(':')
                    .append(it.title).append(':')
                    .append(it.titleUserRenamed).append(':')
                    .append(it.titleAutoGenerated).append(':')
                    .append(it.status).append(':')
                    .append(it.measurementFingerprint).append(':')
                    .append(it.messages.size).append('|')
            }
        }
        if (signature != renderedSessionSignature) {
            renderedSessionSignature = signature
            list.adapter = SessionAdapter(sessions, currentFingerprint)
            empty.visibility = if (sessions.isEmpty()) View.VISIBLE else View.GONE
        }
        return sessions.any { it.needsGeneratedTitle }
    }

    override fun onStop() {
        titleRefreshJob?.cancel()
        titleRefreshJob = null
        super.onStop()
    }

    private fun confirmEndCurrent() {
        val sessionId = currentSessionId ?: return
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.learning_end_confirm_title)
            .setMessage(R.string.learning_end_confirm_message)
            .setPositiveButton(R.string.learning_end_session) { _, _ -> endCurrent(sessionId) }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun endCurrent(sessionId: String) {
        lifecycleScope.launch {
            val ended = withContext(Dispatchers.IO) {
                runCatching { services.store.endSession(measurementId, sessionId) }.getOrNull()
            }
            if (ended == null) {
                Toast.makeText(
                    this@LearningSessionHistoryActivity,
                    R.string.learning_load_error,
                    Toast.LENGTH_LONG
                ).show()
            } else {
                returnAction(ACTION_ENDED, ended.id)
            }
        }
    }

    private fun returnAction(action: String, sessionId: String?) {
        setResult(Activity.RESULT_OK, Intent().putExtra(EXTRA_ACTION, action).apply {
            sessionId?.let { putExtra(EXTRA_SESSION_ID, it) }
        })
        finish()
    }

    private inner class SessionAdapter(
        private val sessions: List<LearningSession>,
        private val currentFingerprint: String?
    ) : BaseAdapter() {
        private val date = SimpleDateFormat("MM-dd HH:mm", Locale.getDefault())

        override fun getCount(): Int = sessions.size
        override fun getItem(position: Int): LearningSession = sessions[position]
        override fun getItemId(position: Int): Long = position.toLong()

        override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
            val session = getItem(position)
            val lifecycleStatus = getString(when (session.status) {
                LearningSessionStatus.ACTIVE -> R.string.learning_status_active
                LearningSessionStatus.PAUSED -> R.string.learning_status_paused
                LearningSessionStatus.ENDED -> R.string.learning_status_ended
            })
            val status = if (session.id == currentSessionId) {
                "${getString(R.string.learning_status_current)} · $lifecycleStatus"
            } else {
                lifecycleStatus
            }
            val old = currentFingerprint != null && currentFingerprint != session.measurementFingerprint
            val label = getString(
                R.string.learning_session_item,
                session.title.ifBlank { date.format(Date(session.createdAtEpochMs)) },
                if (old) "$status · ${getString(R.string.learning_old_data)}" else status,
                session.messages.size
            )
            val row = (convertView as? LinearLayout) ?: LinearLayout(
                this@LearningSessionHistoryActivity
            ).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(dp(16), dp(8), dp(8), dp(8))
                background = android.graphics.drawable.GradientDrawable().apply {
                    setColor(UiColors.SURFACE)
                    setStroke(dp(1), UiColors.OUTLINE)
                    cornerRadius = dp(8).toFloat()
                }
                addView(TextView(this@LearningSessionHistoryActivity).apply {
                    textSize = 14f
                    setLineSpacing(0f, 1.12f)
                    maxLines = 3
                    layoutParams = LinearLayout.LayoutParams(0, -2, 1f)
                })
                addView(ImageButton(this@LearningSessionHistoryActivity).apply {
                    id = R.id.learning_session_edit
                    setImageResource(R.drawable.ic_learning_edit)
                    contentDescription = "重命名话题"
                    setColorFilter(UiColors.ON_SURFACE_VARIANT)
                    setBackgroundResource(R.drawable.bg_learning_icon_button)
                    layoutParams = LinearLayout.LayoutParams(dp(48), dp(48)).apply {
                        marginStart = dp(8)
                    }
                })
            }
            row.setPadding(dp(16), dp(8), dp(8), dp(8))
            val title = row.getChildAt(0) as TextView
            title.text = label
            title.setTextColor(if (session.id == currentSessionId) UiColors.PRIMARY else UiColors.ON_SURFACE)
            row.setOnClickListener {
                returnAction(ACTION_SWITCH, session.id)
            }
            (row.getChildAt(1) as ImageButton).setOnClickListener {
                renameSession(session)
            }
            return row
        }
    }

    private fun renameSession(session: LearningSession) {
        val editor = android.widget.EditText(this).apply {
            setText(session.title)
            setSelection(length())
            hint = "给这次对话起个名字"
            setSingleLine(true)
            setPadding(dp(20), dp(4), dp(20), dp(4))
        }
        MaterialAlertDialogBuilder(this)
            .setTitle("重命名话题")
            .setView(editor)
            .setPositiveButton("保存") { _, _ ->
                lifecycleScope.launch {
                    withContext(Dispatchers.IO) {
                        runCatching { services.store.renameSession(measurementId, session.id, editor.text.toString()) }
                    }
                    startSessionRefresh()
                }
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density + 0.5f).toInt()

    companion object {
        const val EXTRA_CURRENT_SESSION_ID = "current_session_id"
        const val EXTRA_ACTION = "action"
        const val EXTRA_SESSION_ID = "session_id"
        const val ACTION_NEW = "new"
        const val ACTION_SWITCH = "switch"
        const val ACTION_ENDED = "ended"
        private const val TITLE_REFRESH_INTERVAL_MS = 500L
    }
}
