package com.cupet.fallingball.analysis

import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.util.Locale

/** Sort orders offered by the measurement history list. */
enum class HistorySortOrder {
    TIME_DESC,
    SPEED_DESC,
    VISCOSITY_ASC
}

/** Date presets offered by the history filter dialog. */
enum class HistoryDatePreset {
    ALL,
    TODAY,
    LAST_3_DAYS,
    LAST_7_DAYS,
    LAST_30_DAYS,
    CUSTOM
}

/** Date filter state; [startEpochMs, endEpochMs) is used when [preset] is CUSTOM. */
data class HistoryDateFilter(
    val preset: HistoryDatePreset = HistoryDatePreset.ALL,
    val startEpochMs: Long? = null,
    val endEpochMs: Long? = null
)

/** Search / mark-filter / date-filter / sort state of the history list. */
data class HistoryQuery(
    val text: String = "",
    val marks: Set<HistoryMarkColor> = emptySet(),
    val dateFilter: HistoryDateFilter = HistoryDateFilter(),
    val sort: HistorySortOrder = HistorySortOrder.TIME_DESC
)

/** Pure filtering and sorting for the history list; unit-testable without Android. */
object MeasurementHistoryQuery {

    fun apply(
        items: List<MeasurementRecordSummary>,
        query: HistoryQuery,
        nowEpochMs: Long = System.currentTimeMillis()
    ): List<MeasurementRecordSummary> {
        val marked = if (query.marks.isEmpty()) {
            items
        } else {
            items.filter { it.mark != null && it.mark in query.marks }
        }
        val searched = if (query.text.isBlank()) {
            marked
        } else {
            marked.filter { matches(it, query.text) }
        }
        val dated = if (query.dateFilter.preset == HistoryDatePreset.ALL) {
            searched
        } else {
            searched.filter { matchesDate(it, query.dateFilter, nowEpochMs) }
        }
        return when (query.sort) {
            HistorySortOrder.TIME_DESC -> dated.sortedByDescending { it.recordedAtEpochMs }
            HistorySortOrder.SPEED_DESC -> dated.sortedWith(
                compareByDescending<MeasurementRecordSummary> {
                    it.terminalVelocityMmPerS ?: Float.NEGATIVE_INFINITY
                }.thenByDescending { it.recordedAtEpochMs }
            )
            HistorySortOrder.VISCOSITY_ASC -> dated.sortedWith(
                compareBy<MeasurementRecordSummary> {
                    it.viscosityResult?.wallCorrectedStokesPaS ?: Double.MAX_VALUE
                }.thenByDescending { it.recordedAtEpochMs }
            )
        }
    }

    /** Matches record name, liquid, sphere material and temperature text. */
    fun matches(item: MeasurementRecordSummary, query: String): Boolean {
        val needle = query.trim().lowercase(Locale.ROOT)
        if (needle.isEmpty()) return true
        val temperature = item.metadata.temperatureC?.let {
            String.format(Locale.ROOT, "%.1f", it)
        }.orEmpty()
        return listOf(
            item.metadata.recordName,
            item.metadata.liquidName,
            item.metadata.sphereMaterial,
            temperature
        ).any { it.lowercase(Locale.ROOT).contains(needle) }
    }

    /** Filters by local calendar days: TODAY = today, LAST_3_DAYS = today + previous 2 days, etc. */
    fun matchesDate(
        item: MeasurementRecordSummary,
        filter: HistoryDateFilter,
        nowEpochMs: Long,
        zone: ZoneId = ZoneId.systemDefault()
    ): Boolean {
        if (filter.preset == HistoryDatePreset.ALL) return true
        if (filter.preset == HistoryDatePreset.CUSTOM) {
            val start = filter.startEpochMs ?: return true
            val end = filter.endEpochMs ?: return true
            return item.recordedAtEpochMs >= start && item.recordedAtEpochMs < end
        }
        val today = Instant.ofEpochMilli(nowEpochMs).atZone(zone).toLocalDate()
        val startDate: LocalDate = when (filter.preset) {
            HistoryDatePreset.TODAY -> today
            HistoryDatePreset.LAST_3_DAYS -> today.minusDays(2)
            HistoryDatePreset.LAST_7_DAYS -> today.minusDays(6)
            HistoryDatePreset.LAST_30_DAYS -> today.minusDays(29)
            else -> return true
        }
        val start = startDate.atStartOfDay(zone).toInstant().toEpochMilli()
        return item.recordedAtEpochMs >= start && item.recordedAtEpochMs <= nowEpochMs
    }
}
