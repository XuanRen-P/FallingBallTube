package com.cupet.fallingball.camera

import android.content.Context
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import com.cupet.fallingball.analysis.FallSession
import com.cupet.fallingball.vision.ObbDetector
import com.cupet.fallingball.vision.TinyPipelineStageTimer
import com.cupet.fallingball.vision.TubeTrackResult

/** Live CameraX adapter; video analysis uses the same processing implementation. */
class CameraController(
    context: Context,
    lifecycleOwner: LifecycleOwner,
    previewView: PreviewView,
    detector: ObbDetector,
    onResult: (TubeTrackResult) -> Unit,
    onStabilizationInfo: (Boolean) -> Unit,
    onSessionRecordReady: (FallSession, TinyPipelineStageTimer) -> Unit = { _, _ -> },
    onSessionFinished: (FallSession) -> Unit = {}
) : MeasurementFrameProcessor(
    context, lifecycleOwner, previewView, detector, onResult,
    onStabilizationInfo, onSessionRecordReady, onSessionFinished
)
