package com.gomoku.app.ai

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.gomoku.app.game.GameEngine

object BenchmarkRunner {
    private const val TAG = "BenchmarkRunner"

    data class InferenceStats(
        val totalCalls: Int,
        val avgMs: Double,
        val medianMs: Double,
        val minMs: Long,
        val maxMs: Long,
        val p95Ms: Double,
        val allTimesMs: List<Long>
    ) {
        override fun toString(): String = buildString {
            appendLine("========== Inference Benchmark ==========")
            appendLine("Total calls  : $totalCalls")
            appendLine("Avg          : ${"%.2f".format(avgMs)} ms")
            appendLine("Median       : ${"%.2f".format(medianMs)} ms")
            appendLine("Min          : $minMs ms")
            appendLine("Max          : $maxMs ms")
            appendLine("P95          : ${"%.2f".format(p95Ms)} ms")
            appendLine("==========================================")
        }
    }

    fun run(
        context: Context,
        playerFactory: () -> AIPlayer,
        numGames: Int = 10
    ): InferenceStats? {
        val player = playerFactory()
        if (!player.isLoaded) {
            Log.e(TAG, "Model failed to load: ${player.lastError}")
            player.close()
            return null
        }

        when (player) {
            is SNPEAIPlayer -> player.inferenceTimesMs.clear()
            is TFLiteAIPlayer -> player.inferenceTimesMs.clear()
        }

        val engine = GameEngine()

        try {
            for (game in 1..numGames) {
                engine.reset()
                while (!engine.gameOver) {
                    val move = player.selectMove(engine, engine.curPlayer)
                    if (move < 0 || !engine.makeAIMove(move / GameEngine.BOARD_SIZE, move % GameEngine.BOARD_SIZE)) {
                        Log.w(TAG, "Game $game: invalid move=$move, aborting")
                        break
                    }
                }
                Log.d(TAG, "Game $game: winner=${engine.winner}, moves=${engine.moveCount}")
            }
        } finally {
            player.close()
        }

        val times = when (player) {
            is SNPEAIPlayer -> player.inferenceTimesMs.toList()
            is TFLiteAIPlayer -> player.inferenceTimesMs.toList()
            else -> emptyList()
        }

        if (times.isEmpty()) {
            Log.w(TAG, "No inference times recorded")
            return null
        }

        val sorted = times.sorted()
        val n = sorted.size
        return InferenceStats(
            totalCalls = n,
            avgMs = sorted.average(),
            medianMs = if (n % 2 == 1) sorted[n / 2].toDouble()
                        else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
            minMs = sorted.first(),
            maxMs = sorted.last(),
            p95Ms = sorted[(n * 0.95).toInt().coerceIn(0, n - 1)].toDouble(),
            allTimesMs = sorted
        )
    }
}
