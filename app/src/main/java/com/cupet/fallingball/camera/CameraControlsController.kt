package com.cupet.fallingball.camera

import android.content.Context
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import androidx.annotation.MainThread
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExtendableBuilder
import androidx.camera.core.ZoomState
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.Observer
import com.cupet.fallingball.R
import com.google.common.util.concurrent.ListenableFuture
import java.util.Locale

data class AvailableCameraLens(
    val id: String, val label: String, val isFront: Boolean,
    val baseZoomRatio: Float = 1f, val nativeCamera2: Boolean = false
)

data class CameraOpticsState(
    val cameras: List<AvailableCameraLens> = emptyList(),
    val selectedCameraId: String? = null,
    val automaticFocus: Boolean = true,
    val manualFocusSupported: Boolean = false,
    val focusPosition: Int = 50,
    val zoomRatio: Float = 1f,
    val requestedZoomRatio: Float? = null,
    val minimumZoomRatio: Float = 1f,
    val maximumZoomRatio: Float = 1f,
    val bound: Boolean = false,
    val locked: Boolean = false,
    val applying: Boolean = false
)

/**
 * Main-thread owner of user-selected optics. Keep this instance while camera pages are paused.
 * CameraController still owns binding and tube autofocus; the callbacks synchronize those owners.
 */
@androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
class CameraControlsController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val onCameraSelected: (String) -> Unit,
    private val onBeforeOpticsChange: () -> Boolean,
    private val onAutofocusModeChanged: (Boolean) -> Unit,
    private val onOpticsApplied: () -> Unit = {},
    private val onError: (String) -> Unit = {}
) {
    private data class FocusRequest(val automatic: Boolean, val position: Int, val recovery: Boolean = false)

    var state = CameraOpticsState()
        private set
    var onStateChanged: ((CameraOpticsState) -> Unit)? = null
        set(value) {
            field = value
            value?.invoke(state)
        }

    private val mainExecutor = ContextCompat.getMainExecutor(context)
    private var camera: Camera? = null
    private var nativeCamera: XiaomiCamera2LensBackend? = null
    private val nativeLenses by lazy { XiaomiCamera2LensBackend.discover(context) }
    private val unavailableNativeIds = mutableSetOf<String>()
    private var pendingBindingZoom: Float? = null
    private var rearMinimumZoom = 0.5f
    private var rearMaximumZoom = 8f
    private var zoomObserver: Observer<ZoomState>? = null
    private var bindingGeneration = 0L
    private var minimumFocusDistance = 0f
    private var focusInFlight = false
    private var pendingFocus: FocusRequest? = null
    private var confirmedFocus = FocusRequest(true, 50)
    private val zoomRequests = CameraZoomRequests()
    private val rememberedFocus = mutableMapOf<String, Int>()
    private val rememberedZoom = mutableMapOf<String, Float>()
    @Volatile private var captureCameraId: String? = null
    @Volatile private var observedFocusDistance: Float? = null
    @Volatile private var captureCallbackEpoch = 0L

    val selector: CameraSelector
        get() = state.selectedCameraId?.let(::selectorFor) ?: CameraSelector.DEFAULT_BACK_CAMERA

    val selectedNativeLens: XiaomiCamera2Lens?
        get() = nativeLenses.firstOrNull { it.id == state.selectedCameraId && it.id !in unavailableNativeIds }

    /** Invoked only for the currently bound native camera, never for a stale capture. */
    fun observeNativeFocus(cameraId: String, distance: Float?) {
        if (captureCameraId == cameraId) observedFocusDistance = distance
    }

    @MainThread
    fun nativeBindingFailed(cameraId: String) {
        if (cameraId != state.selectedCameraId) return
        unavailableNativeIds += cameraId
        rememberedZoom["0"] = 1f
        pendingBindingZoom = 1f
        state = state.copy(selectedCameraId = "0", cameras = state.cameras.filterNot { it.id == cameraId })
        onError("长焦镜头暂不可用，已返回主摄")
        publish()
    }

    /** Attach to Preview.Builder before build, so entering MF can retain the measured lens position. */
    fun <T> attachCaptureCallback(builder: ExtendableBuilder<T>) {
        val callbackEpoch = ++captureCallbackEpoch
        Camera2Interop.Extender(builder).setSessionCaptureCallback(
            object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(
                    session: CameraCaptureSession,
                    request: CaptureRequest,
                    result: TotalCaptureResult
                ) {
                    if (callbackEpoch == captureCallbackEpoch && session.device.id == captureCameraId) {
                        observedFocusDistance = result.get(CaptureResult.LENS_FOCUS_DISTANCE)
                    }
                }
            }
        )
    }

    @MainThread
    fun bind(provider: ProcessCameraProvider, boundCamera: Camera) {
        val desiredAutomatic = state.automaticFocus
        unbind()
        camera = boundCamera
        val info = Camera2CameraInfo.from(boundCamera.cameraInfo)
        val cameraId = info.cameraId
        captureCameraId = cameraId
        minimumFocusDistance = info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
        val afModes = info.getCameraCharacteristic(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
        val requestKeys = runCatching {
            context.getSystemService(CameraManager::class.java)
                .getCameraCharacteristics(cameraId).availableCaptureRequestKeys
        }.getOrNull()
        val manualSupported = CameraOpticsPolicy.supportsManualFocus(
            minimumFocusDistance,
            afModes?.contains(CaptureRequest.CONTROL_AF_MODE_OFF) == true,
            requestKeys?.contains(CaptureRequest.LENS_FOCUS_DISTANCE) == true &&
                requestKeys.contains(CaptureRequest.CONTROL_AF_MODE)
        )
        val zoom = boundCamera.cameraInfo.zoomState.value
        val minimumZoom = zoom?.minZoomRatio ?: 1f
        val maximumZoom = zoom?.maxZoomRatio ?: 1f
        if (cameraId == "0") { rearMinimumZoom = minimumZoom; rearMaximumZoom = maximumZoom }
        zoomRequests.reset(zoom?.zoomRatio ?: 1f)
        val desiredZoom = (pendingBindingZoom ?: rememberedZoom[cameraId] ?: 1f).coerceIn(minimumZoom, maximumZoom)
        pendingBindingZoom = null
        val automatic = desiredAutomatic || !manualSupported
        val position = rememberedFocus[cameraId] ?: 50
        confirmedFocus = FocusRequest(true, position)
        state = state.copy(
            cameras = enumerate(provider), selectedCameraId = cameraId, bound = true,
            automaticFocus = automatic, manualFocusSupported = manualSupported,
            focusPosition = position, zoomRatio = zoomRequests.confirmedZoom,
            minimumZoomRatio = minimumZoom, maximumZoomRatio = maximumZoom
        )
        val generation = bindingGeneration
        zoomObserver = Observer<ZoomState> { newZoom ->
            if (generation == bindingGeneration) {
                // CameraX publishes a requested ratio before its future finishes. Only use
                // this observer for capabilities; complete() owns confirmed UI values.
                state = state.copy(minimumZoomRatio = newZoom.minZoomRatio,
                    maximumZoomRatio = newZoom.maxZoomRatio)
                publish()
            }
        }.also { boundCamera.cameraInfo.zoomState.observe(lifecycleOwner, it) }
        onAutofocusModeChanged(false)
        pendingFocus = FocusRequest(automatic, position)
        drainFocus()
        if (zoomRequests.request(desiredZoom)) {
            drainZoom()
        }
        publish()
    }

    @MainThread
    fun bindNative(provider: ProcessCameraProvider, backend: XiaomiCamera2LensBackend) {
        val desiredAutomatic = state.automaticFocus
        unbind()
        nativeCamera = backend
        val lens = backend.lens
        val info = lens.characteristics
        captureCameraId = lens.id
        minimumFocusDistance = info.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
        val keys = info.availableCaptureRequestKeys
        val manualSupported = CameraOpticsPolicy.supportsManualFocus(minimumFocusDistance,
            info.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)?.contains(CaptureRequest.CONTROL_AF_MODE_OFF) == true,
            keys?.contains(CaptureRequest.LENS_FOCUS_DISTANCE) == true && keys.contains(CaptureRequest.CONTROL_AF_MODE))
        val maximum = minOf(rearMaximumZoom, lens.baseZoomRatio * backend.maximumZoomRatio)
            .coerceAtLeast(lens.baseZoomRatio)
        val desiredZoom = (pendingBindingZoom ?: rememberedZoom[lens.id] ?: lens.baseZoomRatio)
            .coerceIn(lens.baseZoomRatio, maximum)
        pendingBindingZoom = null
        zoomRequests.reset(lens.baseZoomRatio)
        val position = rememberedFocus[lens.id] ?: 50
        val automatic = desiredAutomatic || !manualSupported
        confirmedFocus = FocusRequest(true, position)
        state = state.copy(cameras = enumerate(provider), selectedCameraId = lens.id, bound = true,
            automaticFocus = automatic, manualFocusSupported = manualSupported, focusPosition = position,
            minimumZoomRatio = rearMinimumZoom, maximumZoomRatio = maximum)
        onAutofocusModeChanged(false)
        pendingFocus = FocusRequest(automatic, position)
        drainFocus()
        if (zoomRequests.request(desiredZoom)) drainZoom()
        publish()
    }

    /** Call when use cases are detached; remembered values survive and stale completions are ignored. */
    @MainThread
    fun unbind() {
        bindingGeneration++
        zoomObserver?.let { camera?.cameraInfo?.zoomState?.removeObserver(it) }
        zoomObserver = null
        camera = null
        nativeCamera = null
        captureCameraId = null
        observedFocusDistance = null
        pendingFocus = null
        zoomRequests.reset(zoomRequests.confirmedZoom)
        focusInFlight = false
        state = state.copy(bound = false, applying = false)
        publish()
    }

    @MainThread
    fun setLocked(locked: Boolean) {
        state = state.copy(locked = locked)
        publish()
    }

    @MainThread
    fun selectCamera(cameraId: String) {
        if (cameraId == state.selectedCameraId || state.cameras.none { it.id == cameraId }) return
        if (!beginChange()) return
        val previousId = state.selectedCameraId
        pendingBindingZoom = null
        state = state.copy(selectedCameraId = cameraId)
        publish()
        runCatching { onCameraSelected(cameraId) }.onFailure {
            pendingBindingZoom = null
            state = state.copy(selectedCameraId = previousId)
            onError(context.getString(R.string.camera_optics_switch_failed))
            publish()
        }
    }

    @MainThread
    fun setAutomaticFocus(automatic: Boolean) {
        if (automatic == state.automaticFocus || (!automatic && !state.manualFocusSupported)) return
        if (!beginChange()) return
        val position = if (automatic) state.focusPosition else CameraOpticsPolicy.focusPosition(
            observedFocusDistance, minimumFocusDistance, state.selectedCameraId?.let(rememberedFocus::get)
        )
        state = state.copy(automaticFocus = automatic, focusPosition = position)
        // Tube AF must stop before cancelFocusAndMetering or a pending lock could overwrite MF.
        onAutofocusModeChanged(false)
        pendingFocus = FocusRequest(automatic, position)
        drainFocus()
    }

    @MainThread
    fun setFocusPosition(position: Int) {
        if (state.automaticFocus || !state.manualFocusSupported) return
        val clamped = position.coerceIn(0, 100)
        if (clamped == state.focusPosition || !beginChange()) return
        state = state.copy(focusPosition = clamped)
        pendingFocus = FocusRequest(false, clamped)
        drainFocus()
    }

    @MainThread
    fun setZoomRatio(ratio: Float) {
        if (!ratio.isFinite() || state.minimumZoomRatio == state.maximumZoomRatio) return
        val clamped = ratio.coerceIn(state.minimumZoomRatio, state.maximumZoomRatio)
        if (pendingBindingZoom == clamped) return
        val inRearFamily = state.selectedCameraId == "0" || nativeLenses.any { it.id == state.selectedCameraId }
        if (inRearFamily) {
            val target = nativeLenses.filter { it.id !in unavailableNativeIds && clamped >= it.baseZoomRatio }
                .maxByOrNull { it.baseZoomRatio }?.id ?: "0"
            if (target != state.selectedCameraId) {
                if (!beginZoomChange()) return
                val previousId = state.selectedCameraId
                pendingBindingZoom = clamped
                state = state.copy(selectedCameraId = target)
                publish()
                runCatching { onCameraSelected(target) }.onFailure {
                    pendingBindingZoom = null
                    state = state.copy(selectedCameraId = previousId)
                    onError(context.getString(R.string.camera_optics_switch_failed))
                    publish()
                }
                return
            }
        }
        // A lens is opening asynchronously. Keep the latest drag target even though no
        // CameraControl exists yet; bind()/bindNative() will apply it after the first capture.
        if (pendingBindingZoom != null && !state.bound) {
            if (!beginZoomChange()) return
            pendingBindingZoom = clamped
            publish()
            return
        }
        // Keep the UI on the last CameraX-confirmed value while a request is in flight.
        // This avoids presenting a requested (but rejected) 3.2x/5x value as active.
        if (clamped == zoomRequests.targetZoom || !beginChange()) return
        zoomRequests.request(clamped)
        drainZoom()
    }

    private fun beginChange(): Boolean = state.bound && !state.locked && onBeforeOpticsChange()

    private fun beginZoomChange(): Boolean = (state.bound || pendingBindingZoom != null) &&
        !state.locked && onBeforeOpticsChange()

    private fun drainFocus() {
        if (focusInFlight) { publish(); return }
        val request = pendingFocus ?: return
        nativeCamera?.let { native ->
            pendingFocus = null
            focusInFlight = true
            val generation = bindingGeneration
            publish()
            val distance = if (request.automatic) 0f else CameraOpticsPolicy.focusDistance(request.position, minimumFocusDistance)
            val future = runCatching { native.setFocus(request.automatic, distance) }
                .getOrElse { finishFocus(generation, request, false); return }
            await(future) { finishFocus(generation, request, it) }
            return
        }
        val activeCamera = camera ?: return
        pendingFocus = null
        focusInFlight = true
        val generation = bindingGeneration
        publish()
        fun apply() {
            if (generation != bindingGeneration) return
            val future = runCatching {
                val control = Camera2CameraControl.from(activeCamera.cameraControl)
                if (request.automatic) control.clearCaptureRequestOptions()
                else control.setCaptureRequestOptions(
                    CaptureRequestOptions.Builder()
                        .setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                        .setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE,
                            CameraOpticsPolicy.focusDistance(request.position, minimumFocusDistance))
                        .build()
                )
            }.getOrElse { finishFocus(generation, request, false); return }
            await(future) { success -> finishFocus(generation, request, success) }
        }
        // CameraX cancellation waits for its default AF mode in capture results. An existing
        // MF override forces AF_OFF, so cancelling again while in MF would wait indefinitely.
        if (request.automatic || !confirmedFocus.automatic) apply()
        else {
            val cancel = runCatching { activeCamera.cameraControl.cancelFocusAndMetering() }
                .getOrElse { finishFocus(generation, request, false); return }
            await(cancel) { success ->
                if (generation == bindingGeneration) {
                    if (success) apply() else finishFocus(generation, request, false)
                }
            }
        }
    }

    private fun finishFocus(generation: Long, request: FocusRequest, success: Boolean) {
        if (generation != bindingGeneration) return
        focusInFlight = false
        if (success) {
            confirmedFocus = request.copy(recovery = false)
            if (!request.automatic) state.selectedCameraId?.let { rememberedFocus[it] = request.position }
            if (pendingFocus == null) {
                state = state.copy(automaticFocus = request.automatic, focusPosition = request.position)
                onAutofocusModeChanged(request.automatic)
            }
            onOpticsApplied()
        } else {
            onError(context.getString(R.string.camera_optics_focus_failed))
            if (pendingFocus == null) {
                state = state.copy(automaticFocus = confirmedFocus.automatic, focusPosition = confirmedFocus.position)
                if (!request.recovery) pendingFocus = confirmedFocus.copy(recovery = true)
                else {
                    // Do not claim a usable lens state after the vendor rejected both change and rollback.
                    state = state.copy(bound = false)
                    onAutofocusModeChanged(false)
                }
            }
        }
        publish()
        drainFocus()
    }

    private fun drainZoom() {
        val activeCamera = camera
        val native = nativeCamera
        if (activeCamera == null && native == null) return
        val ratio = zoomRequests.startNext() ?: run { publish(); return }
        val generation = bindingGeneration
        publish()
        val future = runCatching {
            native?.setZoomRatio(ratio / native.lens.baseZoomRatio)
                ?: requireNotNull(activeCamera).cameraControl.setZoomRatio(ratio)
        }.getOrElse {
            finishZoom(generation, ratio, false); return
        }
        await(future) { finishZoom(generation, ratio, it) }
    }

    private fun finishZoom(generation: Long, ratio: Float, success: Boolean) {
        if (generation != bindingGeneration) return
        if (!zoomRequests.complete(ratio, success)) return
        if (success) {
            // This means CameraX confirmed the request, not that an optical lens switched.
            state.selectedCameraId?.let { rememberedZoom[it] = zoomRequests.confirmedZoom }
            onOpticsApplied()
        } else {
            onError(context.getString(R.string.camera_optics_zoom_failed))
        }
        publish()
        drainZoom()
    }

    private fun await(future: ListenableFuture<*>, complete: (Boolean) -> Unit) {
        future.addListener({ complete(runCatching { future.get(); true }.getOrDefault(false)) }, mainExecutor)
    }

    private fun publish() {
        state = state.copy(
            zoomRatio = zoomRequests.confirmedZoom,
            requestedZoomRatio = pendingBindingZoom ?: zoomRequests.requestedZoom,
            applying = focusInFlight || pendingFocus != null || pendingBindingZoom != null || zoomRequests.requestedZoom != null
        )
        onStateChanged?.invoke(state)
    }

    private fun enumerate(provider: ProcessCameraProvider): List<AvailableCameraLens> =
        (provider.availableCameraInfos.mapNotNull { cameraInfo ->
            runCatching {
                val info = Camera2CameraInfo.from(cameraInfo)
                val front = info.getCameraCharacteristic(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT
                val facing = context.getString(if (front) R.string.camera_optics_front else R.string.camera_optics_back)
                val focal = info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
                val detail = focal?.let { String.format(Locale.ROOT, "%.1f mm", it) }.orEmpty()
                AvailableCameraLens(info.cameraId, "$facing $detail [${info.cameraId}]", front)
            }.getOrNull()
        } + nativeLenses.filter { it.id !in unavailableNativeIds }.map { lens ->
            AvailableCameraLens(lens.id, "后置 ${lens.baseZoomRatio}x 长焦", false, lens.baseZoomRatio, true)
        }).distinctBy { it.id }.sortedWith(compareBy({ it.isFront }, { it.id }))

    companion object {
        fun selectorFor(cameraId: String): CameraSelector = CameraSelector.Builder()
            .addCameraFilter { infos -> infos.filter { Camera2CameraInfo.from(it).cameraId == cameraId } }
            .build()
    }
}

/** Serial CameraX requests with one replaceable pending target and a separately confirmed value. */
internal class CameraZoomRequests {
    var confirmedZoom = 1f
        private set
    var inFlightZoom: Float? = null
        private set
    private var pendingZoom: Float? = null

    val requestedZoom: Float? get() = pendingZoom ?: inFlightZoom
    val targetZoom: Float get() = requestedZoom ?: confirmedZoom

    fun reset(confirmed: Float) {
        confirmedZoom = confirmed
        inFlightZoom = null
        pendingZoom = null
    }

    fun request(ratio: Float): Boolean {
        require(ratio.isFinite() && ratio > 0f)
        if (ratio == targetZoom) return false
        // Returning to the already submitted value cancels a queued detour.
        pendingZoom = ratio.takeUnless { it == inFlightZoom }
        return true
    }

    fun startNext(): Float? {
        if (inFlightZoom != null) return null
        val ratio = pendingZoom ?: return null
        pendingZoom = null
        inFlightZoom = ratio
        return ratio
    }

    fun complete(ratio: Float, success: Boolean): Boolean {
        if (inFlightZoom != ratio) return false
        inFlightZoom = null
        if (success) confirmedZoom = ratio
        return true
    }
}
