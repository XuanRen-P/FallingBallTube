package com.cupet.fallingball.analysis

import com.cupet.fallingball.vision.BallEndReason
import com.cupet.fallingball.vision.BallDetectionSource
import com.cupet.fallingball.vision.RulerConfig
import com.cupet.fallingball.vision.WallContactState
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption

data class MeasurementRecordSummary(
    val id: String,
    val recordedAtEpochMs: Long,
    val metadata: MeasurementMetadata,
    val terminalVelocityMmPerS: Float?,
    val plateauFound: Boolean,
    val parametersComplete: Boolean,
    val viscosityResult: ViscosityResult?,
    val sampleCount: Int,
    val source: MeasurementSource = MeasurementSource.UNKNOWN
)

data class MeasurementHistoryScan(
    val summaries: List<MeasurementRecordSummary>,
    val corruptedFileNames: List<String>
)

/**
 * Versioned file history with lightweight list reads and atomic record replacement.
 *
 * The public interface keeps file layout, migration, and corruption isolation behind one
 * seam. Callers must execute it away from the Android main thread.
 */
class MeasurementHistoryStore(private val directory: File) {

    fun save(record: MeasurementRecord) {
        validateId(record.id)
        ensureDirectory()
        check(!recordFile(record.id).exists()) { "Measurement already exists" }
        writeAtomically(record)
    }

    fun update(record: MeasurementRecord) {
        validateId(record.id)
        ensureDirectory()
        check(recordFile(record.id).exists()) { "Measurement does not exist" }
        writeAtomically(record)
    }

    fun scanSummaries(): MeasurementHistoryScan {
        val summaries = ArrayList<MeasurementRecordSummary>()
        val corrupted = ArrayList<String>()
        recordFiles().forEach { file ->
            runCatching {
                DataInputStream(BufferedInputStream(FileInputStream(file))).use {
                    MeasurementRecordCodec.readSummaryOnly(it)
                }
            }.fold(
                onSuccess = summaries::add,
                onFailure = { corrupted += file.name }
            )
        }
        return MeasurementHistoryScan(
            summaries = summaries.sortedByDescending { it.recordedAtEpochMs },
            corruptedFileNames = corrupted.sorted()
        )
    }

    fun listSummaries(): List<MeasurementRecordSummary> = scanSummaries().summaries

    fun load(id: String): MeasurementRecord? {
        if (!VALID_ID.matches(id)) return null
        val file = recordFile(id)
        if (!file.isFile) return null
        return runCatching {
            DataInputStream(BufferedInputStream(FileInputStream(file))).use {
                MeasurementRecordCodec.read(it)
            }
        }.getOrNull()
    }

    /** Compatibility helper for existing callers; new list screens should use summaries. */
    fun list(): List<MeasurementRecord> =
        recordFiles().mapNotNull { file ->
            runCatching {
                DataInputStream(BufferedInputStream(FileInputStream(file))).use {
                    MeasurementRecordCodec.read(it)
                }
            }.getOrNull()
        }.sortedByDescending { it.recordedAtEpochMs }

    fun delete(id: String): Boolean {
        if (!VALID_ID.matches(id)) return false
        val file = recordFile(id)
        return file.exists() && file.delete()
    }

    private fun writeAtomically(record: MeasurementRecord) {
        val target = recordFile(record.id)
        val temporary = File(directory, ".${record.id}.tmp")
        try {
            DataOutputStream(BufferedOutputStream(FileOutputStream(temporary))).use { output ->
                MeasurementRecordCodec.write(output, record)
            }
            try {
                Files.move(
                    temporary.toPath(),
                    target.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING
                )
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(
                    temporary.toPath(),
                    target.toPath(),
                    StandardCopyOption.REPLACE_EXISTING
                )
            }
        } finally {
            if (temporary.exists()) temporary.delete()
        }
    }

    private fun recordFiles(): List<File> =
        directory.listFiles { file ->
            file.isFile &&
                file.name.startsWith(FILE_PREFIX) &&
                file.extension == FILE_EXTENSION
        }?.toList().orEmpty()

    private fun ensureDirectory() {
        check(directory.exists() || directory.mkdirs()) {
            "Cannot create measurement history directory"
        }
    }

    private fun validateId(id: String) {
        require(VALID_ID.matches(id)) { "Invalid measurement id" }
    }

    private fun recordFile(id: String) = File(directory, "$FILE_PREFIX$id.$FILE_EXTENSION")

    private companion object {
        const val FILE_PREFIX = "measurement_"
        const val FILE_EXTENSION = "fbr"
        val VALID_ID = Regex("[A-Za-z0-9_-]{1,100}")
    }
}

private object MeasurementRecordCodec {
    private const val MAGIC = 0x46425452
    private const val VERSION_1 = 1
    private const val VERSION_2 = 2
    private const val VERSION_3 = 3
    private const val VERSION_4 = 4
    private const val VERSION_5 = 5
    private const val VERSION_6 = 6
    private const val VERSION_7 = 7
    private const val VERSION_8 = 8
    private const val VERSION_9 = 9
    private const val VERSION_10 = 10
    private const val VERSION_11 = 11
    private const val VERSION_12 = 12
    private const val MAX_SAMPLES = 100_000

    private data class Header(
        val version: Int,
        val id: String,
        val recordedAtEpochMs: Long,
        val config: RulerConfig,
        val resolutionLabel: String,
        val endReason: BallEndReason,
        val summary: RecordedVelocitySummary?,
        val analysisQuality: VelocityAnalysisQuality?,
        val analysisMethodVersion: Int,
        val metadata: MeasurementMetadata,
        val viscometryParameters: ViscometryParameters,
        val viscosityResult: ViscosityResult?,
        val sampleCount: Int
    )

    fun write(output: DataOutputStream, record: MeasurementRecord) {
        require(record.samples.size <= MAX_SAMPLES) {
            "Measurement has too many samples"
        }
        output.writeInt(MAGIC)
        output.writeInt(VERSION_12)
        output.writeUTF(record.id)
        output.writeLong(record.recordedAtEpochMs)
        output.writeFloat(record.config.tickSpacingMm)
        output.writeInt(record.config.expectedTickCount)
        output.writeUTF(record.resolutionLabel)
        output.writeUTF(record.endReason.name)
        writeVelocitySummary(output, record.summary)
        writeAnalysisQuality(output, record.analysisQuality)
        output.writeInt(record.analysisMethodVersion)
        writeMetadata(output, record.metadata)
        writeViscometryParameters(output, record.viscometryParameters)
        writeViscosityResult(output, record.viscosityResult)
        output.writeInt(record.samples.size)
        for (sample in record.samples) {
            output.writeFloat(sample.elapsedMs)
            output.writeNullableFloat(sample.positionTicks)
            output.writeNullableFloat(sample.positionMm)
            output.writeNullableFloat(sample.velocityMmPerS)
            output.writeFloat(sample.axialPx)
            output.writeBoolean(sample.rulerValid)
            output.writeBoolean(sample.extrapolated)
            output.writeFloat(sample.shiftPx)
            output.writeBoolean(sample.usedInTerminalFit)
            output.writeBoolean(sample.excludedAsOutlier)
            output.writeUTF(sample.wallContactState.name)
            output.writeNullableFloat(sample.lateralPx)
            output.writeNullableFloat(sample.lateralMm)
            output.writeNullableFloat(sample.normalizedLateral)
            output.writeNullableFloat(sample.wallClearanceMm)
        }
        // V10 keeps the V9 sample body byte-for-byte stable and appends diagnostic evidence.
        // This isolates source metadata from the physics schema and keeps legacy fixtures valid.
        output.writeInt(record.samples.size)
        for (sample in record.samples) {
            output.writeUTF(sample.detectionSource.name)
            output.writeNullableFloat(sample.detectionConfidence)
        }
        writeDiagnosticDigest(output, record.diagnosticDigest)
        writeSource(output, record.source, record.videoSource)
    }

    fun readSummaryOnly(input: DataInputStream): MeasurementRecordSummary {
        val header = readHeader(input)
        // Old analysis versions need their raw trajectories reclassified before display.
        if (
            header.version == VERSION_7 ||
            header.version == VERSION_8 ||
            header.analysisMethodVersion != VelocityEstimator.ANALYSIS_METHOD_VERSION
        ) {
            return readRecordBody(input, header).toSummary()
        }
        // Current records can validate the complete irreplaceable file without allocating a
        // RecordedBallSample for every frame. A truncated or invalid enum still fails loudly.
        repeat(header.sampleCount) { consumeSample(input, header.version) }
        if (header.version >= VERSION_10) consumeDetectionEvidence(input, header.sampleCount)
        if (header.version >= VERSION_11) readDiagnosticDigest(input)
        val source = if (header.version >= VERSION_12) readSource(input).first
            else MeasurementSource.UNKNOWN
        return header.toSummary().copy(source = source)
    }

    fun read(input: DataInputStream): MeasurementRecord {
        val header = readHeader(input)
        return readRecordBody(input, header)
    }

    private fun readRecordBody(
        input: DataInputStream,
        header: Header
    ): MeasurementRecord {
        val samples = ArrayList<RecordedBallSample>(header.sampleCount)
        repeat(header.sampleCount) {
            samples += when (header.version) {
                VERSION_1 -> readVersion1Sample(input, header.config)
                VERSION_2 -> readVersion2Sample(input)
                VERSION_3, VERSION_4 -> readVersion3Sample(input)
                VERSION_5 -> readVersion5Sample(input)
                else -> readVersion6Sample(input)
            }
        }
        if (header.version >= VERSION_10) {
            val evidenceCount = input.readInt()
            require(evidenceCount == header.sampleCount) { "Invalid detection evidence count" }
            repeat(evidenceCount) { index ->
                val source = BallDetectionSource.valueOf(input.readUTF())
                require(source != BallDetectionSource.NONE) { "Recorded sample has no source" }
                val confidence = input.readNullableFloat()
                require(confidence == null || confidence.isFinite()) {
                    "Invalid detection confidence"
                }
                samples[index] = samples[index].copy(
                    detectionSource = source,
                    detectionConfidence = confidence
                )
            }
        }
        val diagnosticDigest = if (header.version >= VERSION_11) {
            readDiagnosticDigest(input)
        } else {
            null
        }
        val source = if (header.version >= VERSION_12) readSource(input)
            else MeasurementSource.UNKNOWN to null
        val record = MeasurementRecord(
            id = header.id,
            recordedAtEpochMs = header.recordedAtEpochMs,
            config = header.config,
            resolutionLabel = header.resolutionLabel,
            endReason = header.endReason,
            summary = header.summary,
            samples = samples,
            metadata = header.metadata,
            viscometryParameters = header.viscometryParameters,
            viscosityResult = header.viscosityResult,
            analysisQuality = header.analysisQuality,
            analysisMethodVersion = header.analysisMethodVersion,
            diagnosticDigest = diagnosticDigest,
            source = source.first,
            videoSource = source.second
        )
        // Older versions did not persist the current analysis classification. Re-run the
        // pure analysis in memory; the file remains unchanged until the user edits it.
        return if (
            header.version == VERSION_7 ||
            header.version == VERSION_8 ||
            header.analysisMethodVersion != VelocityEstimator.ANALYSIS_METHOD_VERSION
        ) {
            record.recalculate(
                config = record.config,
                viscometryParameters = record.viscometryParameters,
                metadata = record.metadata
            )
        } else {
            record
        }
    }

    private fun Header.toSummary() = MeasurementRecordSummary(
        id = id,
        recordedAtEpochMs = recordedAtEpochMs,
        metadata = metadata,
        terminalVelocityMmPerS = summary?.terminalVelocityMmPerS,
        plateauFound = summary?.let {
            VelocityEstimator.isReliableTerminalEstimate(
                plateauFound = it.plateauFound,
                terminalVelocityMmPerS = it.terminalVelocityMmPerS,
                validSampleCount = it.validSampleCount,
                quality = analysisQuality
            )
        } == true,
        parametersComplete = viscometryParameters.isComplete,
        viscosityResult = viscosityResult.takeIf {
            summary?.let { recorded ->
                VelocityEstimator.isReliableTerminalEstimate(
                    plateauFound = recorded.plateauFound,
                    terminalVelocityMmPerS = recorded.terminalVelocityMmPerS,
                    validSampleCount = recorded.validSampleCount,
                    quality = analysisQuality
                )
            } == true
        },
        sampleCount = sampleCount
    )

    private fun MeasurementRecord.toSummary() = MeasurementRecordSummary(
        id = id,
        recordedAtEpochMs = recordedAtEpochMs,
        metadata = metadata,
        terminalVelocityMmPerS = summary?.terminalVelocityMmPerS,
        plateauFound = hasReliableTerminalEstimate,
        parametersComplete = viscometryParameters.isComplete,
        viscosityResult = viscosityResult.takeIf { hasReliableTerminalEstimate },
        sampleCount = samples.size,
        source = source
    )

    private fun readHeader(input: DataInputStream): Header {
        require(input.readInt() == MAGIC) { "Unknown measurement record" }
        val version = input.readInt()
        require(
            version == VERSION_1 ||
                version == VERSION_2 ||
                version == VERSION_3 ||
                version == VERSION_4 ||
                version == VERSION_5 ||
                version == VERSION_6 ||
                version == VERSION_7 ||
                version == VERSION_8 ||
                version == VERSION_9 ||
                version == VERSION_10 ||
                version == VERSION_11 ||
                version == VERSION_12
        ) {
            "Unsupported measurement record version"
        }
        val id = input.readUTF()
        val recordedAtEpochMs = input.readLong()
        val config = RulerConfig(input.readFloat(), input.readInt())
        val resolutionLabel = input.readUTF()
        val endReason = BallEndReason.valueOf(input.readUTF())
        val summary = readVelocitySummary(input)
        val analysisQuality =
            if (version >= VERSION_3) readAnalysisQuality(input, version)
            else null
        val analysisMethodVersion =
            if (version >= VERSION_4) input.readInt()
            else 0
        val metadata = if (version >= VERSION_2) readMetadata(input) else MeasurementMetadata()
        val parameters =
            if (version >= VERSION_2) readViscometryParameters(input, version)
            else ViscometryParameters()
        val viscosity =
            if (version >= VERSION_2) readViscosityResult(input)
            else null
        val sampleCount = input.readInt()
        require(sampleCount in 0..MAX_SAMPLES) { "Invalid sample count" }
        return Header(
            version = version,
            id = id,
            recordedAtEpochMs = recordedAtEpochMs,
            config = config,
            resolutionLabel = resolutionLabel,
            endReason = endReason,
            summary = summary,
            analysisQuality = analysisQuality,
            analysisMethodVersion = analysisMethodVersion,
            metadata = metadata,
            viscometryParameters = parameters,
            viscosityResult = viscosity,
            sampleCount = sampleCount
        )
    }

    private fun readVersion1Sample(
        input: DataInputStream,
        config: RulerConfig
    ): RecordedBallSample {
        val elapsedMs = input.readFloat()
        val positionMm = input.readNullableFloat()
        return RecordedBallSample(
            elapsedMs = elapsedMs,
            positionTicks = positionMm?.div(config.tickSpacingMm),
            positionMm = positionMm,
            velocityMmPerS = input.readNullableFloat(),
            axialPx = input.readFloat(),
            rulerValid = input.readBoolean(),
            extrapolated = input.readBoolean(),
            shiftPx = input.readFloat(),
            usedInTerminalFit = false,
            excludedAsOutlier = false
        )
    }

    private fun readVersion2Sample(input: DataInputStream) = RecordedBallSample(
        elapsedMs = input.readFloat(),
        positionTicks = input.readNullableFloat(),
        positionMm = input.readNullableFloat(),
        velocityMmPerS = input.readNullableFloat(),
        axialPx = input.readFloat(),
        rulerValid = input.readBoolean(),
        extrapolated = input.readBoolean(),
        shiftPx = input.readFloat(),
        usedInTerminalFit = input.readBoolean(),
        excludedAsOutlier = false
    )

    private fun readVersion3Sample(input: DataInputStream) = RecordedBallSample(
        elapsedMs = input.readFloat(),
        positionTicks = input.readNullableFloat(),
        positionMm = input.readNullableFloat(),
        velocityMmPerS = input.readNullableFloat(),
        axialPx = input.readFloat(),
        rulerValid = input.readBoolean(),
        extrapolated = input.readBoolean(),
        shiftPx = input.readFloat(),
        usedInTerminalFit = input.readBoolean(),
        excludedAsOutlier = input.readBoolean()
    )

    private fun readVersion5Sample(input: DataInputStream) = RecordedBallSample(
        elapsedMs = input.readFloat(),
        positionTicks = input.readNullableFloat(),
        positionMm = input.readNullableFloat(),
        velocityMmPerS = input.readNullableFloat(),
        axialPx = input.readFloat(),
        rulerValid = input.readBoolean(),
        extrapolated = input.readBoolean(),
        shiftPx = input.readFloat(),
        usedInTerminalFit = input.readBoolean(),
        excludedAsOutlier = input.readBoolean(),
        wallContactState = WallContactState.valueOf(input.readUTF())
    )

    private fun readVersion6Sample(input: DataInputStream) = RecordedBallSample(
        elapsedMs = input.readFloat(),
        positionTicks = input.readNullableFloat(),
        positionMm = input.readNullableFloat(),
        velocityMmPerS = input.readNullableFloat(),
        axialPx = input.readFloat(),
        rulerValid = input.readBoolean(),
        extrapolated = input.readBoolean(),
        shiftPx = input.readFloat(),
        usedInTerminalFit = input.readBoolean(),
        excludedAsOutlier = input.readBoolean(),
        wallContactState = WallContactState.valueOf(input.readUTF()),
        lateralPx = input.readNullableFloat(),
        lateralMm = input.readNullableFloat(),
        normalizedLateral = input.readNullableFloat(),
        wallClearanceMm = input.readNullableFloat()
    )

    private fun consumeSample(input: DataInputStream, version: Int) {
        input.readFloat() // elapsed ms
        if (version == VERSION_1) {
            input.readNullableFloat() // position mm
            input.readNullableFloat() // velocity
            input.readFloat() // axial px
            input.readBoolean() // ruler valid
            input.readBoolean() // extrapolated
            input.readFloat() // shift px
            return
        }
        input.readNullableFloat() // position ticks
        input.readNullableFloat() // position mm
        input.readNullableFloat() // velocity
        input.readFloat() // axial px
        input.readBoolean() // ruler valid
        input.readBoolean() // extrapolated
        input.readFloat() // shift px
        input.readBoolean() // terminal fit membership
        if (version >= VERSION_3) input.readBoolean() // excluded as outlier
        if (version >= VERSION_5) {
            WallContactState.valueOf(input.readUTF())
        }
        if (version >= VERSION_6) {
            repeat(4) { input.readNullableFloat() }
        }
    }

    private fun consumeDetectionEvidence(input: DataInputStream, expectedCount: Int) {
        val evidenceCount = input.readInt()
        require(evidenceCount == expectedCount) { "Invalid detection evidence count" }
        repeat(evidenceCount) {
            val source = BallDetectionSource.valueOf(input.readUTF())
            require(source != BallDetectionSource.NONE) { "Recorded sample has no source" }
            val confidence = input.readNullableFloat()
            require(confidence == null || confidence.isFinite()) {
                "Invalid detection confidence"
            }
        }
    }

    private fun writeSource(
        output: DataOutputStream,
        source: MeasurementSource,
        video: VideoMeasurementSource?
    ) {
        validateMeasurementSource(source, video)
        output.writeUTF(source.name)
        output.writeBoolean(video != null)
        if (video == null) return
        output.writeUTF(video.sourceName)
        output.writeUTF(video.sourceUri)
        output.writeInt(video.width)
        output.writeInt(video.height)
        output.writeInt(video.rotationDegrees)
        output.writeNullableFloat(video.nominalFrameRate)
        output.writeLong(video.durationUs)
        output.writeLong(video.clipStartUs)
        output.writeLong(video.clipEndUs)
        output.writeLong(video.decodedFrameCount)
        output.writeLong(video.processedFrameCount)
        output.writeLong(video.inferenceFrameCount)
        output.writeUTF(video.taskId)
    }

    private fun readSource(
        input: DataInputStream
    ): Pair<MeasurementSource, VideoMeasurementSource?> {
        val source = MeasurementSource.valueOf(input.readUTF())
        val video = if (input.readBoolean()) VideoMeasurementSource(
            sourceName = input.readUTF(),
            sourceUri = input.readUTF(),
            width = input.readInt(),
            height = input.readInt(),
            rotationDegrees = input.readInt(),
            nominalFrameRate = input.readNullableFloat(),
            durationUs = input.readLong(),
            clipStartUs = input.readLong(),
            clipEndUs = input.readLong(),
            decodedFrameCount = input.readLong(),
            processedFrameCount = input.readLong(),
            inferenceFrameCount = input.readLong(),
            taskId = input.readUTF()
        ) else null
        validateMeasurementSource(source, video)
        return source to video
    }

    private fun writeMetadata(output: DataOutputStream, metadata: MeasurementMetadata) {
        output.writeUTF(metadata.recordName)
        output.writeUTF(metadata.liquidName)
        output.writeUTF(metadata.sphereMaterial)
        output.writeNullableDouble(metadata.temperatureC)
        output.writeUTF(metadata.notes)
    }

    private fun readMetadata(input: DataInputStream) = MeasurementMetadata(
        recordName = input.readUTF(),
        liquidName = input.readUTF(),
        sphereMaterial = input.readUTF(),
        temperatureC = input.readNullableDouble(),
        notes = input.readUTF()
    )

    private fun writeViscometryParameters(
        output: DataOutputStream,
        parameters: ViscometryParameters
    ) {
        output.writeNullableDouble(parameters.sphereRadiusMm)
        output.writeNullableDouble(parameters.tubeInnerDiameterMm)
        output.writeNullableDouble(parameters.liquidDensityGPerCm3)
        output.writeNullableDouble(parameters.sphereDensityGPerCm3)
    }

    private fun readViscometryParameters(
        input: DataInputStream,
        version: Int
    ): ViscometryParameters {
        val storedSphereSizeMm = input.readNullableDouble()
        return ViscometryParameters(
            // Version 8 was written by a UI that asked for "radius", while the experiment
            // and filenames consistently supplied nominal ball diameter. V9 makes the UI
            // semantics explicit and keeps radius only as the internal/storage unit.
            sphereRadiusMm = if (version == VERSION_8) {
                storedSphereSizeMm?.div(2.0)
            } else {
                storedSphereSizeMm
            },
            tubeInnerDiameterMm = input.readNullableDouble(),
            liquidDensityGPerCm3 = input.readNullableDouble(),
            sphereDensityGPerCm3 = input.readNullableDouble()
        )
    }

    private fun writeViscosityResult(
        output: DataOutputStream,
        result: ViscosityResult?
    ) {
        output.writeBoolean(result != null)
        if (result == null) return
        output.writeDouble(result.wallCorrectedStokesPaS)
        output.writeDouble(result.oseenViscosityPaS)
        output.writeDouble(result.faxenFactor)
        output.writeDouble(result.unboundedVelocityMmPerS)
        output.writeDouble(result.reynoldsNumber)
        output.writeDouble(result.diameterRatio)
        output.writeInt(result.warnings.fold(0) { mask, warning ->
            mask or (1 shl warning.ordinal)
        })
        output.writeUTF(result.methodVersion)
    }

    private fun readViscosityResult(input: DataInputStream): ViscosityResult? {
        if (!input.readBoolean()) return null
        val stokes = input.readDouble()
        val oseen = input.readDouble()
        val faxen = input.readDouble()
        val unboundedVelocity = input.readDouble()
        val reynolds = input.readDouble()
        val diameterRatio = input.readDouble()
        val warningMask = input.readInt()
        val warnings = ViscosityWarning.entries.filterTo(linkedSetOf()) { warning ->
            warningMask and (1 shl warning.ordinal) != 0
        }
        return ViscosityResult(
            wallCorrectedStokesPaS = stokes,
            oseenViscosityPaS = oseen,
            faxenFactor = faxen,
            unboundedVelocityMmPerS = unboundedVelocity,
            reynoldsNumber = reynolds,
            diameterRatio = diameterRatio,
            warnings = warnings,
            methodVersion = input.readUTF()
        )
    }

    private fun writeVelocitySummary(
        output: DataOutputStream,
        summary: RecordedVelocitySummary?
    ) {
        output.writeBoolean(summary != null)
        if (summary == null) return
        output.writeNullableFloat(summary.terminalVelocityMmPerS)
        output.writeBoolean(summary.plateauFound)
        output.writeFloat(summary.plateauStartMm)
        output.writeFloat(summary.plateauEndMm)
        output.writeLong(summary.plateauDurationMs)
        output.writeFloat(summary.r2)
        output.writeFloat(summary.velocityStdDev)
        output.writeInt(summary.sampleCount)
        output.writeInt(summary.validSampleCount)
        output.writeNullableFloat(summary.meanVelocityMmPerS)
    }

    private fun readVelocitySummary(input: DataInputStream): RecordedVelocitySummary? {
        if (!input.readBoolean()) return null
        return RecordedVelocitySummary(
            terminalVelocityMmPerS = input.readNullableFloat(),
            plateauFound = input.readBoolean(),
            plateauStartMm = input.readFloat(),
            plateauEndMm = input.readFloat(),
            plateauDurationMs = input.readLong(),
            r2 = input.readFloat(),
            velocityStdDev = input.readFloat(),
            sampleCount = input.readInt(),
            validSampleCount = input.readInt(),
            meanVelocityMmPerS = input.readNullableFloat()
        )
    }

    private fun writeAnalysisQuality(
        output: DataOutputStream,
        quality: VelocityAnalysisQuality?
    ) {
        output.writeBoolean(quality != null)
        if (quality == null) return
        output.writeFloat(quality.effectiveSampleRateFps)
        output.writeLong(quality.velocityWindowMs)
        output.writeInt(quality.usablePositionSampleCount)
        output.writeInt(quality.velocitySampleCount)
        output.writeInt(quality.outlierSampleCount)
        output.writeBoolean(quality.trackDiscontinuity)
        output.writeNullableFloat(quality.terminalFitRelativeDisagreement)
        output.writeBoolean(quality.terminalFitCheckAttempted)
    }

    private fun readAnalysisQuality(
        input: DataInputStream,
        recordVersion: Int
    ): VelocityAnalysisQuality? {
        if (!input.readBoolean()) return null
        return VelocityAnalysisQuality(
            effectiveSampleRateFps = input.readFloat(),
            velocityWindowMs = input.readLong(),
            usablePositionSampleCount = input.readInt(),
            velocitySampleCount = input.readInt(),
            outlierSampleCount = input.readInt(),
            trackDiscontinuity = input.readBoolean(),
            terminalFitRelativeDisagreement = if (recordVersion >= VERSION_7) {
                input.readNullableFloat()
            } else {
                null
            },
            terminalFitCheckAttempted = recordVersion >= VERSION_8 && input.readBoolean()
        )
    }

    private fun writeDiagnosticDigest(
        output: DataOutputStream,
        digest: MeasurementDiagnosticDigest?
    ) {
        output.writeBoolean(digest != null)
        if (digest == null) return
        output.writeInt(digest.observedSampleCount)
        output.writeInt(digest.rulerValidSampleCount)
        output.writeInt(digest.localTrackerSampleCount)
        output.writeInt(digest.heatmapSampleCount)
        output.writeInt(digest.otherDetectorSampleCount)
        output.writeInt(digest.rejectedOutlierCount)
        output.writeInt(digest.wallContactSampleCount)
        output.writeNullableFloat(digest.medianDetectionConfidence)
        output.writeFloat(digest.peakRegistrationShiftPx)
        output.writeBoolean(digest.excessiveRegistrationShake)
        output.writeNullableInt(digest.schedulerSkipCount)
        output.writeNullableInt(digest.selectorRejectionCount)
        output.writeNullableInt(digest.gateRejectionCount)
    }

    private fun readDiagnosticDigest(input: DataInputStream): MeasurementDiagnosticDigest? {
        if (!input.readBoolean()) return null
        return MeasurementDiagnosticDigest(
            observedSampleCount = input.readNonNegativeInt("observed sample count"),
            rulerValidSampleCount = input.readNonNegativeInt("ruler-valid sample count"),
            localTrackerSampleCount = input.readNonNegativeInt("local sample count"),
            heatmapSampleCount = input.readNonNegativeInt("heatmap sample count"),
            otherDetectorSampleCount = input.readNonNegativeInt("other sample count"),
            rejectedOutlierCount = input.readNonNegativeInt("outlier count"),
            wallContactSampleCount = input.readNonNegativeInt("wall-contact count"),
            medianDetectionConfidence = input.readNullableFloat()?.also {
                require(it.isFinite() && it in 0f..1f) { "Invalid detection confidence" }
            },
            peakRegistrationShiftPx = input.readFloat().also {
                require(it.isFinite() && it >= 0f) { "Invalid peak registration shift" }
            },
            excessiveRegistrationShake = input.readBoolean(),
            schedulerSkipCount = input.readNullableNonNegativeInt("scheduler skip count"),
            selectorRejectionCount = input.readNullableNonNegativeInt("selector rejection count"),
            gateRejectionCount = input.readNullableNonNegativeInt("gate rejection count")
        )
    }

    private fun DataInputStream.readNonNegativeInt(label: String): Int = readInt().also {
        require(it >= 0) { "Invalid $label" }
    }

    private fun DataOutputStream.writeNullableInt(value: Int?) {
        writeBoolean(value != null)
        if (value != null) writeInt(value)
    }

    private fun DataInputStream.readNullableNonNegativeInt(label: String): Int? =
        if (readBoolean()) readNonNegativeInt(label) else null

    private fun DataOutputStream.writeNullableFloat(value: Float?) {
        writeBoolean(value != null)
        if (value != null) writeFloat(value)
    }

    private fun DataInputStream.readNullableFloat(): Float? =
        if (readBoolean()) readFloat() else null

    private fun DataOutputStream.writeNullableDouble(value: Double?) {
        writeBoolean(value != null)
        if (value != null) writeDouble(value)
    }

    private fun DataInputStream.readNullableDouble(): Double? =
        if (readBoolean()) readDouble() else null
}
