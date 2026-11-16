package com.cupet.fallingball.vision

import android.content.Context
import android.os.Build
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import java.nio.FloatBuffer
import java.security.MessageDigest
import java.util.EnumMap
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors

/**
 * YOLOv8n-OBB ONNX detector for glass tube.
 *
 * Expects input preprocessed to match ultralytics training:
 *   RGB, square letterbox, normalized [0,1], NCHW layout.
 *
 * Model input:  float32 (1, 3, height, width) - "images"
 * Model output: float32 (1, 6, anchors)       - [cx, cy, w, h, confidence, angle_rad]
 *   ch4 = confidence (direct probability), ch5 = angle (radians)
 */
class ObbDetector(
    context: Context,
    initialProfile: ModelProfile = ModelProfile.DEFAULT
) {

    companion object {
        private const val TAG = "ObbDetector"
        private const val MODEL_FILE = "best.onnx"
        private const val OUTPUT_CHANNELS = 6

        /**
         * Only detections strictly above this are accepted. Raised from the ultralytics
         * default of 0.25 to suppress low-confidence false positives on cluttered
         * benches; lower it again if a genuine tube stops being picked up.
         */
        private const val CONF_THRESHOLD = 0.7f
        private const val IOU_THRESHOLD = 0.45f
        private const val MAX_DET = 50
        private const val TUNING_PREFS = "cpu_thread_tuning_v1"
        private const val TUNING_VERSION = 2
        private const val BENCHMARK_WARMUPS = 1
        private const val BENCHMARK_RUNS = 5
        // Production camera startup must not launch dozens of extra ONNX runs and heat the
        // phone. Existing cached choices remain honored; benchmarking is an offline task.
        private const val ENABLE_RUNTIME_TUNING = false

        // Set true only while comparing against PC diagnostics
        private const val DEBUG_DUMP = false

        // Per-frame inference logging costs a String.format on the analysis thread.
        private const val LOG_TIMINGS = false
    }

    private val appContext = context.applicationContext
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val modelBytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
    private val modelHash = sha256(modelBytes)
    private val tuningPreferences =
        appContext.getSharedPreferences(TUNING_PREFS, Context.MODE_PRIVATE)
    private val tuningExecutor = Executors.newSingleThreadExecutor()
    private val tuningProfiles = ConcurrentHashMap.newKeySet<ModelProfile>()
    private val directInputTensors = EnumMap<ModelProfile, OnnxTensor>(ModelProfile::class.java)
    private var directInputBuffer: FloatBuffer? = null
    private var session: OrtSession
    private val inputName: String
    @Volatile
    private var activeThreads: Int
    @Volatile
    private var closed = false

    /**
     * Direct buffer reused for every inference. ORT can read a direct buffer in place,
     * so this avoids allocating and copying a fresh ~4.9 MB tensor each frame.
     */
    init {
        val fallbackThreads = (Runtime.getRuntime().availableProcessors() / 2).coerceIn(2, 4)
        // A stored choice from an earlier run must not exceed the current budget: the phone
        // may have been tuned before the budget existed.
        activeThreads = clampThreads(cachedThreads(initialProfile) ?: fallbackThreads)
        session = createSession(activeThreads)
        inputName = session.inputNames.first()
        Log.d(
            TAG,
            "ONNX model loaded - inputs: ${session.inputNames}, " +
                "outputs: ${session.outputNames}, intraOpThreads=$activeThreads"
        )
    }

    data class LetterboxParams(
        val scale: Float,
        val newW: Int,
        val newH: Int,
        val padL: Int,
        val padT: Int
    )

    data class Detection(
        val observation: TubeObservation?,
        val inputSize: Int,
        val inferenceMs: Float,
        val timings: PipelineTimings = PipelineTimings(totalMs = inferenceMs)
    )

    /**
     * Ultralytics-compatible square letterbox geometry:
     * scale = inputSize / max(w,h), round sizes, asymmetric pad via round(d/2 +/- 0.1).
     */
    fun letterboxParams(
        frameW: Int,
        frameH: Int,
        inputSize: Int = ModelProfile.DEFAULT.inputSize
    ): LetterboxParams {
        val scale = inputSize.toFloat() / max(frameW, frameH)
        val newW = kotlin.math.round(frameW * scale).toInt().coerceAtLeast(1)
        val newH = kotlin.math.round(frameH * scale).toInt().coerceAtLeast(1)
        val dw = (inputSize - newW).toFloat()
        val dh = (inputSize - newH).toFloat()
        val padL = kotlin.math.round(dw / 2f - 0.1f).toInt().coerceAtLeast(0)
        val padT = kotlin.math.round(dh / 2f - 0.1f).toInt().coerceAtLeast(0)
        return LetterboxParams(scale, newW, newH, padL, padT)
    }

    /**
     * Detect glass tube from a preprocessed square RGB NCHW tensor.
     *
     * [region] must describe the exact rectangle [rgb] was letterboxed from — the inverse
     * letterbox here recomputes the same parameters from `region.roiW/roiH`, so a mismatch
     * with the preprocessing side shows up as a scaled or offset box.
     *
     * Synchronised because the direct input tensor is reused by the camera inference worker.
     */
    /** Runs a camera tensor already stored in native-order direct memory. The buffer is read
     * in place by ORT and its tensor wrapper is cached per profile, eliminating the former
     * FloatArray-to-direct-buffer copy and per-frame input tensor allocation.
     */
    @Synchronized
    fun detect(
        rgb: FloatBuffer,
        region: InputRegion,
        profile: ModelProfile = ModelProfile.DEFAULT
    ): Detection {
        require(rgb.isDirect) { "Camera inference requires a direct FloatBuffer" }
        val required = 3 * profile.inputSize * profile.inputSize
        require(rgb.capacity() >= required) {
            "Expected capacity >= $required for YOLO ${profile.inputSize}, got ${rgb.capacity()}"
        }
        applyCachedThreads(profile)
        val t0 = System.nanoTime()
        val inputTensor = directInputTensor(rgb, profile)
        val detection = runDetection(inputTensor, region, profile, t0, System.nanoTime())
        val tuningScheduled = scheduleTuningIfNeeded(rgb, profile)
        return if (tuningScheduled) {
            detection.copy(timings = detection.timings.copy(tuning = true))
        } else {
            detection
        }
    }

    private fun runDetection(
        inputTensor: OnnxTensor,
        region: InputRegion,
        profile: ModelProfile,
        startedNs: Long,
        tensorReadyNs: Long
    ): Detection {
        val outputs = session.run(mapOf(inputName to inputTensor))
        val inferenceFinishedNs = System.nanoTime()
        val rawTensor = outputs.get(0) as? OnnxTensor
        val bestBox = try {
            if (rawTensor == null) {
                null
            } else {
                val raw = rawTensor.floatBuffer
                val floatCount = raw.remaining()
                require(floatCount % OUTPUT_CHANNELS == 0) {
                    "Unexpected OBB output length $floatCount"
                }
                val anchorCount = floatCount / OUTPUT_CHANNELS
                require(anchorCount == profile.expectedAnchorCount) {
                    "YOLO ${profile.inputSize} produced $anchorCount anchors; " +
                        "expected ${profile.expectedAnchorCount}"
                }
                val box = findBestObb(raw, anchorCount)
                if (DEBUG_DUMP) dumpInferenceDiag(raw, anchorCount)
                box
            }
        } finally {
            outputs.close()
        }
        val observation = bestBox?.let { toObservation(it, region, profile.inputSize) }
        val finishedNs = System.nanoTime()
        val totalMs = nanosToMs(finishedNs - startedNs)
        val timings = PipelineTimings(
            tensorSetupMs = nanosToMs(tensorReadyNs - startedNs),
            inferenceRunMs = nanosToMs(inferenceFinishedNs - tensorReadyNs),
            postprocessMs = nanosToMs(finishedNs - inferenceFinishedNs),
            totalMs = totalMs,
            cpuThreads = activeThreads,
            nativePreprocess = false,
            tuning = tuningProfiles.isNotEmpty()
        )
        if (LOG_TIMINGS) {
            Log.d(
                TAG,
                if (bestBox == null) {
                    "Inference: %.1f ms input=%d no detection".format(
                        totalMs,
                        profile.inputSize
                    )
                } else {
                    "Inference: %.1f ms input=%d conf=%.3f".format(
                        totalMs,
                        profile.inputSize,
                        bestBox.confidence
                    )
                }
            )
        }
        return Detection(
            observation = observation,
            inputSize = profile.inputSize,
            inferenceMs = totalMs,
            timings = timings
        )
    }

    private fun directInputTensor(buffer: FloatBuffer, profile: ModelProfile): OnnxTensor {
        if (directInputBuffer !== buffer) {
            closeDirectInputTensors()
            directInputBuffer = buffer
        }
        return directInputTensors.getOrPut(profile) {
            val required = 3 * profile.inputSize * profile.inputSize
            buffer.position(0)
            buffer.limit(required)
            OnnxTensor.createTensor(
                env,
                buffer,
                longArrayOf(
                    1,
                    3,
                    profile.inputSize.toLong(),
                    profile.inputSize.toLong()
                )
            )
        }
    }

    private fun scheduleTuningIfNeeded(
        source: FloatBuffer,
        profile: ModelProfile
    ): Boolean {
        if (!ENABLE_RUNTIME_TUNING) return false
        if (closed || cachedThreads(profile) != null || !tuningProfiles.add(profile)) return false
        val required = 3 * profile.inputSize * profile.inputSize
        val copy = try {
            ByteBuffer.allocateDirect(required * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .also { destination ->
                    val input = source.duplicate()
                    input.position(0)
                    input.limit(required)
                    destination.put(input)
                    destination.flip()
                }
        } catch (t: Throwable) {
            tuningProfiles.remove(profile)
            Log.w(TAG, "Unable to copy tensor for CPU tuning", t)
            return false
        }

        return try {
            tuningExecutor.execute {
                try {
                    tuneThreads(profile, copy)
                } catch (t: Throwable) {
                    Log.w(TAG, "CPU thread tuning failed for ${profile.inputSize}", t)
                } finally {
                    tuningProfiles.remove(profile)
                }
            }
            true
        } catch (t: Throwable) {
            tuningProfiles.remove(profile)
            Log.w(TAG, "Unable to schedule CPU tuning", t)
            false
        }
    }

    private fun tuneThreads(profile: ModelProfile, input: FloatBuffer) {
        if (cachedThreads(profile) != null || tuningCancelled()) return
        val candidates = CpuThreadTuning.candidates(
            Runtime.getRuntime().availableProcessors(),
            threadBudget()
        )
        val results = ArrayList<CpuThreadTuning.Benchmark>(candidates.size)
        for (threads in candidates) {
            if (tuningCancelled()) return
            val median = benchmarkThreads(threads, profile, input)
            Log.i(
                TAG,
                "CPU tuning input=${profile.inputSize} threads=$threads median=${"%.1f".format(median)}ms"
            )
            results += CpuThreadTuning.Benchmark(threads, median)
        }
        val selected = CpuThreadTuning.select(results)
        if (selected == null) {
            Log.w(TAG, "CPU tuning produced no valid result for input=${profile.inputSize}")
            return
        }
        if (closed || cachedThreads(profile) != null) return
        // Cache only. The next normal-priority camera inference creates and swaps
        // the production session in applyCachedThreads(), outside the tuning executor.
        tuningPreferences.edit().putInt(tuningKey(profile), selected).apply()
        Log.i(
            TAG,
            "CPU tuning selected threads=$selected for input=${profile.inputSize}; results=$results"
        )
    }

    private fun benchmarkThreads(
        threads: Int,
        profile: ModelProfile,
        input: FloatBuffer
    ): Float {
        var candidateSession: OrtSession? = null
        var inputTensor: OnnxTensor? = null
        return try {
            if (tuningCancelled()) return Float.NaN
            candidateSession = createSession(threads)
            input.position(0)
            input.limit(3 * profile.inputSize * profile.inputSize)
            inputTensor = OnnxTensor.createTensor(
                env,
                input,
                longArrayOf(1, 3, profile.inputSize.toLong(), profile.inputSize.toLong())
            )
            repeat(BENCHMARK_WARMUPS) {
                if (tuningCancelled()) return Float.NaN
                candidateSession.run(mapOf(inputName to inputTensor)).close()
            }
            val samples = FloatArray(BENCHMARK_RUNS)
            repeat(BENCHMARK_RUNS) { index ->
                if (tuningCancelled()) return Float.NaN
                val started = System.nanoTime()
                candidateSession.run(mapOf(inputName to inputTensor)).close()
                samples[index] = nanosToMs(System.nanoTime() - started)
            }
            samples.sort()
            samples[samples.size / 2]
        } catch (t: Throwable) {
            Log.w(TAG, "CPU benchmark failed for threads=$threads", t)
            Float.NaN
        } finally {
            try {
                inputTensor?.close()
            } catch (_: Throwable) {
            }
            try {
                candidateSession?.close()
            } catch (_: Throwable) {
            }
        }
    }

    private fun tuningCancelled(): Boolean =
        closed || Thread.currentThread().isInterrupted

    private fun applyCachedThreads(profile: ModelProfile) {
        val cached = clampThreads(cachedThreads(profile) ?: return)
        if (cached != activeThreads) replaceSession(cached)
    }

    /**
     * Thread budget for this graph.
     *
     * The tube detector shares the fast cluster with the camera analyzer and the parent
     * session.  On a phone whose whole fast cluster is four cores, each pipeline taking half
     * of every core made all of them three to six times slower (measured on a Dimensity 1000+
     * on 2026-11-15: parent 44 ms isolated vs 287 ms inside the pipeline).  The budget leaves
     * two cores for the others; wide-cluster devices keep the historical behaviour.
     */
    private fun threadBudget(): Int {
        val fastCores = CpuAffinity.fastCoreCount
        return if (fastCores in 1..4) (fastCores - 2).coerceAtLeast(1) else Int.MAX_VALUE
    }

    /** Clamps a stored/auto thread choice to [threadBudget]; 0 (ORT auto) becomes the budget. */
    private fun clampThreads(threads: Int): Int {
        val budget = threadBudget()
        if (budget == Int.MAX_VALUE) return threads
        return if (threads <= 0) budget else threads.coerceAtMost(budget)
    }

    private fun replaceSession(threads: Int) {
        val replacement = createSession(threads)
        val previous = session
        session = replacement
        activeThreads = threads
        try {
            previous.close()
        } catch (_: Throwable) {
        }
    }

    private fun createSession(threads: Int): OrtSession {
        val options = OrtSession.SessionOptions()
        return try {
            if (threads > 0) options.setIntraOpNumThreads(threads)
            options.setInterOpNumThreads(1)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            options.setMemoryPatternOptimization(true)
            env.createSession(modelBytes, options)
        } finally {
            options.close()
        }
    }

    private fun cachedThreads(profile: ModelProfile): Int? {
        val key = tuningKey(profile)
        return if (tuningPreferences.contains(key)) tuningPreferences.getInt(key, 0) else null
    }

    private fun tuningKey(profile: ModelProfile): String =
        "$TUNING_VERSION|${Build.FINGERPRINT}|$modelHash|${profile.inputSize}"

    private fun closeDirectInputTensors() {
        directInputTensors.values.forEach { tensor ->
            try {
                tensor.close()
            } catch (_: Throwable) {
            }
        }
        directInputTensors.clear()
    }

    private fun toObservation(
        box: ObbBox,
        region: InputRegion,
        inputSize: Int
    ): TubeObservation? {
        // Undo the letterbox back into ROI coordinates, then translate into full-frame
        // coordinates so the overlay can draw without knowing about the crop.
        val lb = letterboxParams(region.roiW, region.roiH, inputSize)
        val cxOrig = (box.cx - lb.padL) / lb.scale + region.offsetX
        val cyOrig = (box.cy - lb.padT) / lb.scale + region.offsetY
        val wOrig = box.w / lb.scale
        val hOrig = box.h / lb.scale

        val corners = computeRotatedCorners(cxOrig, cyOrig, wOrig, hOrig, box.angle)
        val byX = corners.sortedBy { it.first }
        val leftPts = listOf(byX[0], byX[1]).sortedBy { it.second }
        val rightPts = listOf(byX[2], byX[3]).sortedBy { it.second }

        val leftLine = wallLineFromPoints(leftPts[0], leftPts[1], box.confidence)
        val rightLine = wallLineFromPoints(rightPts[0], rightPts[1], box.confidence)
        if (leftLine == null || rightLine == null) return null

        val topY = corners.minOf { it.second }
        val bottomY = corners.maxOf { it.second }

        return TubeObservation(
            left = leftLine.copy(score = box.confidence),
            right = rightLine.copy(score = box.confidence),
            topY = topY,
            bottomY = bottomY,
            confidence = box.confidence,
            frameWidth = region.frameW,
            frameHeight = region.frameH,
            corners = corners.toList()
        )
    }

    // -----------------------------------------------------------------
    // Post-processing: best-conf + class-agnostic OBB NMS (no soft average)
    // -----------------------------------------------------------------

    private fun findBestObb(raw: FloatBuffer, anchorCount: Int): ObbBox? {
        // ch4 = confidence (direct probability, NOT a logit)
        // ch5 = angle (radians)
        val candidates = ArrayList<ObbBox>(64)
        for (i in 0 until anchorCount) {
            val conf = raw.get(4 * anchorCount + i)
            if (conf <= CONF_THRESHOLD) continue
            var w = raw.get(2 * anchorCount + i)
            var h = raw.get(3 * anchorCount + i)
            var angle = raw.get(5 * anchorCount + i)
            // regularize_rboxes: make w the longer side, angle in [0, pi)
            if (h > w) {
                val tmp = w
                w = h
                h = tmp
                angle += (PI / 2.0).toFloat()
            }
            angle = normalizeAngle(angle)
            candidates.add(
                ObbBox(
                    cx = raw.get(i),
                    cy = raw.get(anchorCount + i),
                    w = w,
                    h = h,
                    angle = angle,
                    confidence = conf
                )
            )
        }

        if (candidates.isEmpty()) return null
        candidates.sortByDescending { it.confidence }

        // Simple greedy OBB NMS using AABB IoU of rotated corners.
        val kept = ArrayList<ObbBox>(8)
        for (cand in candidates) {
            if (kept.size >= MAX_DET) break
            var suppressed = false
            for (k in kept) {
                if (obbIou(cand, k) > IOU_THRESHOLD) {
                    suppressed = true
                    break
                }
            }
            if (!suppressed) kept.add(cand)
        }
        return kept.maxByOrNull { it.confidence }
    }

    /** Normalize angle to [0, pi). */
    private fun normalizeAngle(a: Float): Float {
        var x = a
        val pi = PI.toFloat()
        while (x < 0f) x += pi
        while (x >= pi) x -= pi
        return x
    }

    /**
     * IoU of two OBBs via their axis-aligned bounding boxes of corners.
     * Fast and sufficient for NMS on elongated near-vertical tubes.
     */
    private fun obbIou(a: ObbBox, b: ObbBox): Float {
        val ca = computeRotatedCorners(a.cx, a.cy, a.w, a.h, a.angle)
        val cb = computeRotatedCorners(b.cx, b.cy, b.w, b.h, b.angle)
        var aMinX = Float.POSITIVE_INFINITY
        var aMaxX = Float.NEGATIVE_INFINITY
        var aMinY = Float.POSITIVE_INFINITY
        var aMaxY = Float.NEGATIVE_INFINITY
        for (p in ca) {
            if (p.first < aMinX) aMinX = p.first
            if (p.first > aMaxX) aMaxX = p.first
            if (p.second < aMinY) aMinY = p.second
            if (p.second > aMaxY) aMaxY = p.second
        }
        var bMinX = Float.POSITIVE_INFINITY
        var bMaxX = Float.NEGATIVE_INFINITY
        var bMinY = Float.POSITIVE_INFINITY
        var bMaxY = Float.NEGATIVE_INFINITY
        for (p in cb) {
            if (p.first < bMinX) bMinX = p.first
            if (p.first > bMaxX) bMaxX = p.first
            if (p.second < bMinY) bMinY = p.second
            if (p.second > bMaxY) bMaxY = p.second
        }
        val interW = max(0f, min(aMaxX, bMaxX) - max(aMinX, bMinX))
        val interH = max(0f, min(aMaxY, bMaxY) - max(aMinY, bMinY))
        val inter = interW * interH
        if (inter <= 0f) return 0f
        val areaA = max(0f, aMaxX - aMinX) * max(0f, aMaxY - aMinY)
        val areaB = max(0f, bMaxX - bMinX) * max(0f, bMaxY - bMinY)
        val union = areaA + areaB - inter
        return if (union <= 0f) 0f else inter / union
    }

    // -----------------------------------------------------------------
    // Geometry helpers
    // -----------------------------------------------------------------

    private data class ObbBox(
        val cx: Float, val cy: Float,
        val w: Float, val h: Float,
        val angle: Float, val confidence: Float
    )

    /** Compute 4 corners of a rotated bounding box. Local order: TL, TR, BR, BL before rotation. */
    private fun computeRotatedCorners(
        cx: Float, cy: Float,
        w: Float, h: Float,
        angleRad: Float
    ): Array<Pair<Float, Float>> {
        val cosA = cos(angleRad.toDouble()).toFloat()
        val sinA = sin(angleRad.toDouble()).toFloat()
        val hw = w * 0.5f
        val hh = h * 0.5f
        val dx = floatArrayOf(-hw, +hw, +hw, -hw)
        val dy = floatArrayOf(-hh, -hh, +hh, +hh)
        return Array(4) { i ->
            val rx = dx[i] * cosA - dy[i] * sinA
            val ry = dx[i] * sinA + dy[i] * cosA
            (cx + rx) to (cy + ry)
        }
    }

    private fun wallLineFromPoints(
        p0: Pair<Float, Float>,
        p1: Pair<Float, Float>,
        confidence: Float
    ): WallLine? {
        val (x0, y0) = p0
        val (x1, y1) = p1
        val dy = y1 - y0
        return if (abs(dy) < 0.01f) null
        else {
            val slope = (x1 - x0) / dy
            val intercept = x0 - slope * y0
            WallLine(slope, intercept, confidence.coerceIn(0f, 1f))
        }
    }

    private fun dumpInferenceDiag(raw: FloatBuffer, anchorCount: Int) {
        Log.d(TAG, "===== DIAG: Inference Output =====")
        val chNames = arrayOf("cx", "cy", "w", "h", "conf", "angle_rad")
        for (c in 0..5) {
            var minV = Float.MAX_VALUE
            var maxV = -Float.MAX_VALUE
            var sum = 0f
            for (i in 0 until anchorCount) {
                val v = raw.get(c * anchorCount + i)
                if (v < minV) minV = v
                if (v > maxV) maxV = v
                sum += v
            }
            Log.d(
                TAG,
                "Ch[$c] %s: min=%.4f max=%.4f mean=%.4f"
                    .format(chNames[c], minV, maxV, sum / anchorCount)
            )
        }
        data class Cand(val idx: Int, val conf: Float)
        val cands = ArrayList<Cand>()
        for (i in 0 until anchorCount) {
            val conf = raw.get(4 * anchorCount + i)
            if (conf > CONF_THRESHOLD) cands.add(Cand(i, conf))
        }
        cands.sortByDescending { it.conf }
        Log.d(TAG, "Candidates above $CONF_THRESHOLD: ${cands.size}")
        for (k in 0 until min(10, cands.size)) {
            val c = cands[k]
            val cx = raw.get(c.idx)
            val cy = raw.get(anchorCount + c.idx)
            val w = raw.get(2 * anchorCount + c.idx)
            val h = raw.get(3 * anchorCount + c.idx)
            val a = Math.toDegrees(raw.get(5 * anchorCount + c.idx).toDouble())
            Log.d(
                TAG,
                "#%d: conf=%.4f  cx=%.2f cy=%.2f w=%.2f h=%.2f angle=%.3f deg"
                    .format(k + 1, c.conf, cx, cy, w, h, a)
            )
        }
    }

    private fun nanosToMs(nanos: Long): Float = nanos / 1_000_000f

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString(separator = "") { byte -> "%02x".format(byte) }

    /**
     * Closes this detector's session only. [OrtEnvironment] is a process-wide singleton —
     * closing it here would tear down any other live session in the app.
     */
    fun close() {
        closed = true
        tuningExecutor.shutdownNow()
        synchronized(this) {
            closeDirectInputTensors()
            try {
                session.close()
            } catch (_: Throwable) {
            }
        }
    }
}
