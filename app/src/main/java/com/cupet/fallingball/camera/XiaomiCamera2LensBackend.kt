package com.cupet.fallingball.camera

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureFailure
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.MeteringRectangle
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.camera.view.PreviewView
import androidx.concurrent.futures.CallbackToFutureAdapter
import androidx.core.content.ContextCompat
import com.cupet.fallingball.vision.AnalysisFrame
import com.google.common.util.concurrent.ListenableFuture
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.CompletableFuture
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** A verified auxiliary lens; the ID is only used on Xiaomi's aurora device family. */
data class XiaomiCamera2Lens(val id: String, val baseZoomRatio: Float, val characteristics: CameraCharacteristics)

/**
 * Public Camera2 adapter for Xiaomi 14 Ultra telephoto cameras that open normally but are
 * omitted from cameraIdList, hence unavailable to CameraX's camera repository. No vendor
 * request keys, reflection, package impersonation, or privileged permissions are used.
 */
class XiaomiCamera2LensBackend(
    private val context: Context,
    val lens: XiaomiCamera2Lens,
    private val preview: PreviewView,
    requestedSize: Size,
    private val onFrame: (AnalysisFrame) -> Unit,
    private val onReady: () -> Unit,
    private val onFocusDistance: (Float?) -> Unit,
    private val onFailure: (Throwable) -> Unit
) : AutoCloseable {
    private data class Stamp(val focus: Long, val zoom: Long)
    private val mainExecutor = ContextCompat.getMainExecutor(context)
    private val thread by lazy { HandlerThread("Xiaomi-lens-${lens.id}").apply { start() } }
    private val handler by lazy { Handler(thread.looper) }
    private val closed = AtomicBoolean()
    private val started = AtomicBoolean()
    private val ready = AtomicBoolean()
    private var device: CameraDevice? = null
    private var opening = false
    private var cameraDeviceClosed = true
    private var cameraCleanupComplete = false
    val closedFuture = CompletableFuture<Unit>()
    private var session: CameraCaptureSession? = null
    private var texture: TextureView? = null
    private var previewSurface: Surface? = null
    private var reader: ImageReader? = null
    private val frameLock = Any()
    private var borrowedFrames = 0
    private var deferredReaderClose: ImageReader? = null
    private var automatic = true
    private var manualDistance = 0f
    private var tubeAfActive = false
    private var focusRegion: MeteringRectangle? = null
    private var focusVersion = 0L
    private var zoomVersion = 0L
    private var zoomRatio = 1f
    private var pendingFocus: Pair<Long, CallbackToFutureAdapter.Completer<Unit>>? = null
    private var pendingZoom: Pair<Long, CallbackToFutureAdapter.Completer<Unit>>? = null
    private var pendingTubeFocus: CallbackToFutureAdapter.Completer<Boolean>? = null
    private var lastControlFeedback = "no capture result"
    private var lastReportedFocusDistance: Float? = null
    private var lastConfirmedManualDistance: Float? = null
    private var lastConfirmedManualFeedback: Float? = null
    private var manualFocusConfirmation: ManualFocusConfirmation? = null
    private val characteristics = lens.characteristics
    private val sensorArray = Rect(requireNotNull(characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)))
    private val streamSize = chooseStreamSize(characteristics, requestedSize)
    private val sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
    private var rotationDegrees = sensorOrientation
    private val supportsAf = characteristics.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
        ?.contains(CaptureRequest.CONTROL_AF_MODE_AUTO) == true
    private val supportsAfRegion = (characteristics.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AF) ?: 0) > 0
    private val supportsAeRegion = (characteristics.get(CameraCharacteristics.CONTROL_MAX_REGIONS_AE) ?: 0) > 0
    private val usesZoomRatio = Build.VERSION.SDK_INT >= 30 &&
        characteristics.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE) != null

    val minimumZoomRatio: Float get() = 1f
    val maximumZoomRatio: Float = if (Build.VERSION.SDK_INT >= 30) {
        characteristics.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)?.upper
            ?: characteristics.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f
    } else characteristics.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f

    /** Called on the UI thread after CameraX has unbound its use cases. */
    fun start() {
        check(!closed.get())
        if (!started.compareAndSet(false, true)) return
        val parent = preview.parent as ViewGroup
        texture = TextureView(context).also { view ->
            // A sibling is essential: PreviewView may remove all of its own children.
            parent.addView(view, parent.indexOfChild(preview) + 1,
                FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
            view.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                    configureTransform(width, height)
                    surface.setDefaultBufferSize(streamSize.width, streamSize.height)
                    handler.post { open(surface) }
                }
                override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
                    configureTransform(width, height)
                }
                override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                    if (!closed.get()) fail(IllegalStateException("Telephoto preview surface was lost"))
                    return true
                }
                override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit
            }
            if (view.isAvailable) view.surfaceTexture?.let { surface ->
                configureTransform(view.width, view.height)
                surface.setDefaultBufferSize(streamSize.width, streamSize.height)
                handler.post { open(surface) }
            }
        }
        handler.postDelayed({
            if (!closed.get() && !ready.get()) fail(IllegalStateException("Telephoto camera did not produce a capture"))
        }, 10_000)
    }

    private fun configureTransform(width: Int, height: Int) {
        if (width <= 0 || height <= 0) return
        val displayRotation = when (preview.display?.rotation) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        rotationDegrees = (sensorOrientation - displayRotation + 360) % 360
        val rotated = rotationDegrees == 90 || rotationDegrees == 270
        val logicalWidth = if (rotated) streamSize.height else streamSize.width
        val logicalHeight = if (rotated) streamSize.width else streamSize.height
        val scale = max(width.toFloat() / logicalWidth, height.toFloat() / logicalHeight)
        // Camera2's SurfaceTexture producer applies sensor/display orientation. Apply only
        // the same FILL_CENTER aspect crop used by OverlayView to the oriented content.
        texture?.setTransform(Matrix().apply {
            setScale(logicalWidth * scale / width, logicalHeight * scale / height,
                width / 2f, height / 2f)
            postRotate(-displayRotation.toFloat(), width / 2f, height / 2f)
        })
    }

    @android.annotation.SuppressLint("MissingPermission")
    private fun open(surfaceTexture: SurfaceTexture) {
        if (closed.get()) return
        try {
            previewSurface = Surface(surfaceTexture)
            reader = ImageReader.newInstance(streamSize.width, streamSize.height, ImageFormat.YUV_420_888, 4).also {
                it.setOnImageAvailableListener({ source -> deliverFrame(source) }, handler)
            }
            opening = true
            cameraDeviceClosed = false
            context.getSystemService(CameraManager::class.java).openCamera(lens.id,
                object : CameraDevice.StateCallback() {
                    override fun onOpened(camera: CameraDevice) {
                        opening = false
                        if (closed.get()) { camera.close(); return }
                        device = camera
                        configureSession(camera)
                    }
                    override fun onDisconnected(camera: CameraDevice) {
                        opening = false
                        camera.close()
                        fail(IllegalStateException("Telephoto camera disconnected"))
                    }
                    override fun onError(camera: CameraDevice, error: Int) {
                        opening = false
                        camera.close()
                        fail(IllegalStateException("Telephoto camera error $error"))
                    }
                    override fun onClosed(camera: CameraDevice) {
                        cameraDeviceClosed = true
                        if (device === camera) device = null
                        if (closed.get()) completeClose()
                    }
                }, handler)
        } catch (failure: Exception) {
            opening = false
            cameraDeviceClosed = true
            if (closed.get()) completeClose() else fail(failure)
        }
    }

    @Suppress("DEPRECATION")
    private fun configureSession(camera: CameraDevice) {
        try {
            camera.createCaptureSession(listOf(requireNotNull(previewSurface), requireNotNull(reader).surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(configured: CameraCaptureSession) {
                        if (closed.get()) { configured.close(); return }
                        session = configured
                        runCatching { repeat() }.onFailure(::fail)
                    }
                    override fun onConfigureFailed(failed: CameraCaptureSession) {
                        failed.close()
                        fail(IllegalStateException("Telephoto preview/analysis stream combination is unsupported"))
                    }
                }, handler)
        } catch (failure: Exception) { fail(failure) }
    }

    private fun deliverFrame(source: ImageReader) {
        val image = runCatching { source.acquireLatestImage() }.getOrNull() ?: return
        if (closed.get()) { image.close(); return }
        synchronized(frameLock) { borrowedFrames++ }
        var frame: AnalysisFrame? = null
        try {
            frame = AnalysisFrame(image.width, image.height, Rect(image.cropRect),
                AnalysisFrame.Info(image.timestamp, rotationDegrees),
                image.planes.map { AnalysisFrame.Plane(it.buffer, it.rowStride, it.pixelStride) },
                release = {
                    image.close()
                    synchronized(frameLock) {
                        borrowedFrames--
                        if (borrowedFrames == 0) {
                            deferredReaderClose?.close()
                            deferredReaderClose = null
                        }
                    }
                })
            onFrame(frame)
        } catch (failure: Exception) {
            if (frame != null) frame.close() else {
                image.close()
                synchronized(frameLock) { borrowedFrames-- }
            }
            fail(failure)
        }
    }

    fun setFocus(automatic: Boolean, distance: Float): ListenableFuture<Unit> =
        request("focus") { completer ->
            val previousManualDistance = if (this.automatic) null else lastConfirmedManualDistance
            this.automatic = automatic
            manualDistance = distance
            tubeAfActive = false
            pendingTubeFocus?.set(false)
            pendingTubeFocus = null
            focusRegion = null
            focusVersion++
            manualFocusConfirmation = if (automatic) null else ManualFocusConfirmation(
                focusVersion, distance, lastReportedFocusDistance, previousManualDistance,
                characteristics.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f,
                characteristics.get(CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION) !=
                    CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION_APPROXIMATE &&
                    characteristics.get(CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION) !=
                    CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION_UNCALIBRATED,
                lastConfirmedManualFeedback
            )
            pendingFocus?.second?.setException(IllegalStateException("Focus request replaced"))
            pendingFocus = focusVersion to completer
            repeat()
            expireControl(completer, true)
        }

    fun setZoomRatio(ratio: Float): ListenableFuture<Unit> = request("zoom") { completer ->
        require(ratio in minimumZoomRatio..maximumZoomRatio)
        zoomRatio = ratio
        zoomVersion++
        pendingZoom?.second?.setException(IllegalStateException("Zoom request replaced"))
        pendingZoom = zoomVersion to completer
        repeat()
        expireControl(completer, false)
    }

    fun autofocus(xFraction: Float, yFraction: Float): ListenableFuture<Boolean> =
        CallbackToFutureAdapter.getFuture { completer ->
            handler.post {
                if (closed.get() || !automatic || !supportsAf) { completer.set(false); return@post }
                try {
                    pendingTubeFocus?.set(false)
                    pendingTubeFocus = completer
                    tubeAfActive = true
                    focusRegion = meteringRegion(xFraction, yFraction)
                    session?.capture(buildRequest(CaptureRequest.CONTROL_AF_TRIGGER_START), captureCallback, handler)
                    repeat()
                    handler.postDelayed({
                        if (pendingTubeFocus === completer) { pendingTubeFocus = null; completer.set(false) }
                    }, 5_000)
                } catch (failure: Exception) { completer.setException(failure) }
            }
            "Xiaomi-tube-autofocus"
        }

    fun cancelAutofocus() {
        handler.post {
            if (closed.get() || !automatic) return@post
            pendingTubeFocus?.set(false)
            pendingTubeFocus = null
            tubeAfActive = false
            focusRegion = null
            runCatching {
                session?.capture(buildRequest(CaptureRequest.CONTROL_AF_TRIGGER_CANCEL), captureCallback, handler)
                repeat()
            }.onFailure { Log.w(TAG, "Unable to release tube focus", it) }
        }
    }

    private fun request(label: String, action: (CallbackToFutureAdapter.Completer<Unit>) -> Unit): ListenableFuture<Unit> =
        CallbackToFutureAdapter.getFuture { completer ->
            if (closed.get()) completer.setException(IllegalStateException("Telephoto camera is closed"))
            else handler.post {
                if (closed.get()) completer.setException(IllegalStateException("Telephoto camera is closed"))
                else runCatching { action(completer) }.onFailure { completer.setException(it) }
            }
            "Xiaomi-$label"
        }

    private fun expireControl(completer: CallbackToFutureAdapter.Completer<Unit>, focus: Boolean) {
        handler.postDelayed({
            val pending = if (focus) pendingFocus else pendingZoom
            if (pending?.second === completer) {
                if (focus) {
                    pendingFocus = null
                    manualFocusConfirmation = null
                } else pendingZoom = null
                Log.w(TAG, "camera=${lens.id} control_timeout kind=${if (focus) "focus" else "zoom"} " +
                    "pendingVersion=${pending?.first} latest={$lastControlFeedback}")
                val failure = IllegalStateException("Telephoto control was not confirmed by capture metadata")
                completer.setException(failure)
                // A focus actuator may quantize a small step. Let the controls owner
                // restore its last confirmed focus without discarding a working lens.
                if (!focus) fail(failure)
            }
        }, 5_000)
    }

    private fun repeat() {
        check(!closed.get())
        requireNotNull(session).setRepeatingRequest(buildRequest(), captureCallback, handler)
    }

    private fun buildRequest(trigger: Int = CaptureRequest.CONTROL_AF_TRIGGER_IDLE): CaptureRequest {
        val builder = requireNotNull(device).createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        builder.addTarget(requireNotNull(previewSurface))
        builder.addTarget(requireNotNull(reader).surface)
        builder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
        builder.set(CaptureRequest.CONTROL_AF_MODE, when {
            !automatic -> CaptureRequest.CONTROL_AF_MODE_OFF
            tubeAfActive -> CaptureRequest.CONTROL_AF_MODE_AUTO
            else -> CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
        })
        if (!automatic) builder.set(CaptureRequest.LENS_FOCUS_DISTANCE, manualDistance)
        builder.set(CaptureRequest.CONTROL_AF_TRIGGER, trigger)
        if (usesZoomRatio && Build.VERSION.SDK_INT >= 30) {
            builder.set(CaptureRequest.CONTROL_ZOOM_RATIO, zoomRatio)
        } else builder.set(CaptureRequest.SCALER_CROP_REGION, zoomCrop())
        val fps = characteristics.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
            ?.filter { it.contains(30) }?.minByOrNull { abs(it.lower - 30) + abs(it.upper - 30) }
        fps?.let { builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it) }
        // Keep preview and YUV geometry identical; digital stabilization can crop only one stream.
        builder.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
        if (characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION)
                ?.contains(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_ON) == true) {
            builder.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_ON)
        }
        focusRegion?.let { region ->
            if (supportsAfRegion) builder.set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(region))
            if (supportsAeRegion) builder.set(CaptureRequest.CONTROL_AE_REGIONS, arrayOf(region))
        }
        builder.setTag(Stamp(focusVersion, zoomVersion))
        return builder.build()
    }

    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
            if (closed.get()) return
            if (ready.compareAndSet(false, true)) {
                Log.i(TAG, "camera=${lens.id} capture_ready size=$streamSize rotation=$rotationDegrees baseZoom=${lens.baseZoomRatio}")
                mainExecutor.execute { if (!closed.get()) onReady() }
            }
            lastReportedFocusDistance = result.get(CaptureResult.LENS_FOCUS_DISTANCE)
            onFocusDistance(lastReportedFocusDistance)
            (request.tag as? Stamp)?.let { stamp ->
                if (pendingFocus != null || pendingZoom != null) {
                    val requestedZoom = if (Build.VERSION.SDK_INT >= 30) request.get(CaptureRequest.CONTROL_ZOOM_RATIO) else null
                    val reportedZoom = if (Build.VERSION.SDK_INT >= 30) result.get(CaptureResult.CONTROL_ZOOM_RATIO) else null
                    lastControlFeedback = "focusVersion=${stamp.focus} zoomVersion=${stamp.zoom} " +
                        "requestedAF=${request.get(CaptureRequest.CONTROL_AF_MODE)} reportedAF=${result.get(CaptureResult.CONTROL_AF_MODE)} " +
                        "requestedD=${request.get(CaptureRequest.LENS_FOCUS_DISTANCE)} reportedD=${result.get(CaptureResult.LENS_FOCUS_DISTANCE)} " +
                        "afState=${result.get(CaptureResult.CONTROL_AF_STATE)} lensState=${result.get(CaptureResult.LENS_STATE)} " +
                        "calibration=${characteristics.get(CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION)} " +
                        "requestedZoom=$requestedZoom reportedZoom=$reportedZoom " +
                        "requestedCrop=${request.get(CaptureRequest.SCALER_CROP_REGION)} reportedCrop=${result.get(CaptureResult.SCALER_CROP_REGION)}"
                }
                val requestedAfMode = request.get(CaptureRequest.CONTROL_AF_MODE)
                val reportedAfMode = result.get(CaptureResult.CONTROL_AF_MODE)
                pendingFocus?.takeIf { stamp.focus == it.first }?.let { pending ->
                    val modeConfirmed = requestedAfMode == reportedAfMode && requestedAfMode != null
                    val manual = requestedAfMode == CaptureRequest.CONTROL_AF_MODE_OFF
                    val focusConfirmed = if (manual) manualFocusConfirmation?.observe(
                        stamp.focus, modeConfirmed, lastReportedFocusDistance,
                        result.get(CaptureResult.LENS_STATE) == CaptureResult.LENS_STATE_STATIONARY,
                        result.get(CaptureResult.LENS_STATE) == CaptureResult.LENS_STATE_MOVING
                    ) == true else modeConfirmed
                    if (focusConfirmed) {
                        lastConfirmedManualDistance = if (manual) request.get(CaptureRequest.LENS_FOCUS_DISTANCE) else null
                        lastConfirmedManualFeedback = if (manual) lastReportedFocusDistance else null
                        pendingFocus = null
                        manualFocusConfirmation = null
                        pending.second.set(Unit)
                    }
                }
                val zoomConfirmed = if (usesZoomRatio && Build.VERSION.SDK_INT >= 30) {
                    val expected = request.get(CaptureRequest.CONTROL_ZOOM_RATIO)
                    val reported = result.get(CaptureResult.CONTROL_ZOOM_RATIO)
                    expected != null && reported != null && abs(expected - reported) <= 0.01f * max(1f, expected)
                } else {
                    val expected = request.get(CaptureRequest.SCALER_CROP_REGION)
                    val reported = result.get(CaptureResult.SCALER_CROP_REGION)
                    expected != null && reported != null && abs(expected.left - reported.left) <= 2 &&
                        abs(expected.top - reported.top) <= 2 && abs(expected.right - reported.right) <= 2 &&
                        abs(expected.bottom - reported.bottom) <= 2
                }
                if (zoomConfirmed) pendingZoom?.takeIf { stamp.zoom >= it.first }
                    ?.let { pendingZoom = null; it.second.set(Unit) }
            }
            when (result.get(CaptureResult.CONTROL_AF_STATE)) {
                CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED -> { pendingTubeFocus?.set(true); pendingTubeFocus = null }
                CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED -> { pendingTubeFocus?.set(false); pendingTubeFocus = null }
            }
        }
        override fun onCaptureFailed(session: CameraCaptureSession, request: CaptureRequest, failure: CaptureFailure) {
            if (!ready.get()) fail(IllegalStateException("Telephoto capture failed: ${failure.reason}"))
        }
    }

    private fun zoomCrop(): Rect {
        val width = (sensorArray.width() / zoomRatio).toInt().coerceAtLeast(2)
        val height = (sensorArray.height() / zoomRatio).toInt().coerceAtLeast(2)
        val left = sensorArray.centerX() - width / 2
        val top = sensorArray.centerY() - height / 2
        return Rect(left, top, left + width, top + height)
    }

    private fun meteringRegion(x: Float, y: Float): MeteringRectangle {
        val (u, v) = when (rotationDegrees) {
            90 -> y to (1f - x)
            180 -> (1f - x) to (1f - y)
            270 -> (1f - y) to x
            else -> x to y
        }
        // Zoom-ratio API uses the unzoomed active-array coordinate space for metering.
        val crop = if (usesZoomRatio) Rect(sensorArray) else zoomCrop()
        val aspect = streamSize.width.toFloat() / streamSize.height
        if (crop.width().toFloat() / crop.height() > aspect) {
            val wanted = (crop.height() * aspect).toInt()
            crop.inset((crop.width() - wanted) / 2, 0)
        } else {
            val wanted = (crop.width() / aspect).toInt()
            crop.inset(0, (crop.height() - wanted) / 2)
        }
        val radius = (min(crop.width(), crop.height()) * 0.09f).toInt().coerceAtLeast(1)
        val cx = (crop.left + u.coerceIn(0f, 1f) * crop.width()).toInt().coerceIn(crop.left + radius, crop.right - radius)
        val cy = (crop.top + v.coerceIn(0f, 1f) * crop.height()).toInt().coerceIn(crop.top + radius, crop.bottom - radius)
        return MeteringRectangle(Rect(cx - radius, cy - radius, cx + radius, cy + radius), MeteringRectangle.METERING_WEIGHT_MAX)
    }

    private fun fail(failure: Throwable) {
        if (closed.get()) return
        Log.e(TAG, "Camera ${lens.id} failed", failure)
        mainExecutor.execute { if (!closed.get()) onFailure(failure) }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        texture?.let { view -> (view.parent as? ViewGroup)?.removeView(view) }
        texture = null
        handler.post {
            val failure = IllegalStateException("Telephoto camera is closed")
            pendingFocus?.second?.setException(failure)
            pendingZoom?.second?.setException(failure)
            pendingTubeFocus?.set(false)
            pendingFocus = null; pendingZoom = null; pendingTubeFocus = null
            runCatching { session?.stopRepeating() }
            runCatching { session?.close() }
            val closingDevice = device
            runCatching { closingDevice?.close() }
            session = null; device = null
            reader?.setOnImageAvailableListener(null, null)
            synchronized(frameLock) {
                if (borrowedFrames == 0) reader?.close() else deferredReaderClose = reader
            }
            reader = null
            previewSurface?.release()
            previewSurface = null
            cameraCleanupComplete = true
            completeClose()
        }
    }

    private fun completeClose() {
        // onClosed may arrive after the UI marks closed but before it posts cleanup. Keep
        // the handler alive until both device shutdown and our resource cleanup finished.
        if (!closed.get() || !cameraCleanupComplete || !cameraDeviceClosed || opening) return
        closedFuture.complete(Unit)
        thread.quitSafely()
    }

    companion object {
        private const val TAG = "XiaomiCamera2Lens"

        fun discover(context: Context): List<XiaomiCamera2Lens> {
            if (!Build.MANUFACTURER.equals("Xiaomi", true) ||
                !(Build.DEVICE.equals("aurora", true) || Build.MODEL == "24031PN0DC")) return emptyList()
            val manager = context.getSystemService(CameraManager::class.java)
            val main = runCatching { manager.getCameraCharacteristics("0") }.getOrNull() ?: return emptyList()
            val mainFocal = main.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: return emptyList()
            val mainWidth = main.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.width ?: return emptyList()
            return listOf("4" to 3.2f, "5" to 5f).mapNotNull { (id, nominalZoom) ->
                runCatching {
                    val info = manager.getCameraCharacteristics(id)
                    require(info.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK)
                    val focal = requireNotNull(info.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull())
                    val width = requireNotNull(info.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)).width
                    val equivalentZoom = (focal / width) / (mainFocal / mainWidth)
                    require(abs(equivalentZoom / nominalZoom - 1f) < 0.12f)
                    require(info.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                        ?.getOutputSizes(ImageFormat.YUV_420_888)?.isNotEmpty() == true)
                    XiaomiCamera2Lens(id, nominalZoom, info)
                }.getOrNull()
            }
        }

        private fun chooseStreamSize(info: CameraCharacteristics, requested: Size): Size {
            val map = requireNotNull(info.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP))
            val previewSizes = map.getOutputSizes(SurfaceTexture::class.java).toSet()
            val choices = map.getOutputSizes(ImageFormat.YUV_420_888).filter { it in previewSizes }
            require(choices.isNotEmpty()) { "No shared telephoto preview and YUV resolution" }
            val wantedWidth = max(requested.width, requested.height)
            val wantedHeight = min(requested.width, requested.height)
            return choices.minBy { size ->
                abs(size.width.toDouble() / size.height - wantedWidth.toDouble() / wantedHeight) * 10_000_000 +
                    abs(size.width.toLong() * size.height - wantedWidth.toLong() * wantedHeight)
            }
        }
    }
}
