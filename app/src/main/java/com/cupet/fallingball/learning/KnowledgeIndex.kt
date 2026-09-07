package com.cupet.fallingball.learning

import org.json.JSONArray
import java.io.InputStream
import kotlin.math.ln

data class KnowledgeCard(
    val id: String,
    val title: String,
    val keywords: List<String>,
    val level: String,
    val source: String,
    val sourceUrl: String?,
    val content: String
)

data class KnowledgeHit(
    val card: KnowledgeCard,
    val score: Double,
    val boundedContent: String
)

/** Small local BM25 index tuned for mixed Chinese physics terms and ASCII symbols. */
class KnowledgeIndex(
    cards: List<KnowledgeCard>,
    private val synonyms: Map<String, Set<String>> = DEFAULT_SYNONYMS
) {
    private val documents = cards.distinctBy { it.id }
    private val documentTokens = documents.map { card ->
        tokenize(listOf(card.title, card.keywords.joinToString(" "), card.content).joinToString(" "))
    }
    private val averageDocumentLength = documentTokens.map { it.size }.average().takeIf {
        it.isFinite() && it > 0.0
    } ?: 1.0
    private val documentFrequency: Map<String, Int> = buildMap {
        documentTokens.forEach { tokens ->
            tokens.toSet().forEach { token -> put(token, (get(token) ?: 0) + 1) }
        }
    }

    init {
        require(documents.map { it.id }.toSet().size == documents.size) {
            "Knowledge card ids must be unique"
        }
    }

    fun search(query: String, limit: Int = 5, maxContentChars: Int = 800): List<KnowledgeHit> {
        if (limit <= 0 || query.isBlank()) return emptyList()
        val queryTokens = expandSynonyms(tokenize(query)).distinct()
        if (queryTokens.isEmpty()) return emptyList()
        return documents.indices.mapNotNull { index ->
            val tokens = documentTokens[index]
            val termFrequency = tokens.groupingBy { it }.eachCount()
            var score = 0.0
            queryTokens.forEach { token ->
                val frequency = termFrequency[token] ?: 0
                if (frequency == 0) return@forEach
                val df = documentFrequency[token] ?: 0
                val idf = ln(1.0 + (documents.size - df + 0.5) / (df + 0.5))
                val denominator = frequency + K1 *
                    (1.0 - B + B * tokens.size / averageDocumentLength)
                score += idf * frequency * (K1 + 1.0) / denominator
            }
            if (score <= 0.0) null else KnowledgeHit(
                card = documents[index],
                score = score,
                boundedContent = documents[index].content.take(maxContentChars.coerceAtLeast(80))
            )
        }.sortedWith(compareByDescending<KnowledgeHit> { it.score }.thenBy { it.card.id })
            .take(limit.coerceAtMost(5))
    }

    fun hasSufficientCoverage(query: String): Boolean {
        val hits = search(query, limit = 2, maxContentChars = 80)
        return hasSufficientCoverage(hits)
    }

    internal fun hasSufficientCoverage(hits: List<KnowledgeHit>): Boolean {
        return hits.size >= 2 && hits.first().score >= SUFFICIENT_SCORE
    }

    val size: Int get() = documents.size

    private fun expandSynonyms(tokens: List<String>): List<String> = buildList {
        tokens.forEach { token ->
            add(token)
            synonyms[token]?.let(::addAll)
        }
    }

    companion object {
        fun fromJson(input: InputStream): KnowledgeIndex = input.bufferedReader().use { reader ->
            val array = JSONArray(reader.readText())
            KnowledgeIndex(
                (0 until array.length()).map { index ->
                    val value = array.getJSONObject(index)
                    KnowledgeCard(
                        id = value.getString("id"),
                        title = value.getString("title"),
                        keywords = value.getJSONArray("keywords").let { keywords ->
                            (0 until keywords.length()).map(keywords::getString)
                        },
                        level = value.getString("level"),
                        source = value.getString("source"),
                        sourceUrl = if (value.isNull("source_url")) null
                        else value.getString("source_url"),
                        content = value.getString("content")
                    )
                }
            )
        }

        internal fun tokenize(value: String): List<String> {
            val normalized = value.lowercase()
            val ascii = Regex("[a-z0-9]+(?:[._/-][a-z0-9]+)*")
                .findAll(normalized)
                .map { it.value }
                .toList()
            val cjkRuns = Regex("[\\u3400-\\u9fff]+").findAll(normalized).map { it.value }
            val cjk = buildList {
                cjkRuns.forEach { run ->
                    run.forEach { add(it.toString()) }
                    if (run.length >= 2) {
                        for (index in 0 until run.lastIndex) add(run.substring(index, index + 2))
                    }
                    if (run.length >= 3) {
                        for (index in 0 until run.length - 2) add(run.substring(index, index + 3))
                    }
                }
            }
            return ascii + cjk
        }

        private const val K1 = 1.35
        private const val B = 0.72
        private const val SUFFICIENT_SCORE = 1.0

        private val DEFAULT_SYNONYMS = mapOf(
            "粘度" to setOf("黏度", "viscosity"),
            "黏度" to setOf("粘度", "viscosity"),
            "终端" to setOf("稳定", "平台"),
            "平台" to setOf("终端", "稳定"),
            "贴壁" to setOf("管壁", "偏心"),
            "管壁" to setOf("贴壁", "修正"),
            "异常" to setOf("离群", "outlier"),
            "离群" to setOf("异常", "outlier"),
            "雷诺" to setOf("reynolds", "re"),
            "速度" to setOf("velocity", "速率"),
            "位移" to setOf("position", "轨迹")
        )
    }
}
