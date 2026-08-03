package com.gomoku.app

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.gomoku.app.ai.AIPlayer
import com.gomoku.app.ai.AIPlayerFactory
import com.gomoku.app.databinding.ActivityMainBinding
import com.gomoku.app.game.GameEngine
import com.gomoku.app.ui.BoardView
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var boardView: BoardView
    private lateinit var tvStatus: TextView
    private lateinit var btnNewGame: Button

    private val engine = GameEngine()
    private lateinit var aiPlayer: AIPlayer

    private val humanPlayer = GameEngine.BLACK
    private val aiPlayerStone = GameEngine.WHITE

    private val aiExecutor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private var aiThinking = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        boardView = binding.boardView
        tvStatus = binding.tvStatus
        btnNewGame = binding.btnNewGame

        boardView.gameEngine = engine
        boardView.onCellTapped = { row, col -> onHumanMove(row, col) }

        btnNewGame.setOnClickListener { startNewGame() }

        loadModelAndStart()
    }

    private fun loadModelAndStart() {
        tvStatus.text = "Loading AI model…"
        aiExecutor.execute {
            aiPlayer = AIPlayerFactory.create(this)
            mainHandler.post {
                if (aiPlayer.isLoaded) {
                    val engineType = if (aiPlayer is com.gomoku.app.ai.SNPEAIPlayer) "DLC (SNPE)" else "TFLite"
                    Log.d("MainActivity", "AI engine loaded: $engineType, useCPU=${aiPlayer.useCPU}")
                    startNewGame()
                } else {
                    showModelNotFoundDialog()
                }
            }
        }
    }

    private fun showModelNotFoundDialog() {
        tvStatus.text = getString(R.string.model_not_found)
        val errorMsg = aiPlayer.lastError ?: "Unknown error"
        AlertDialog.Builder(this)
            .setTitle("Model Load Failed")
            .setMessage(errorMsg)
            .setPositiveButton("OK") { _, _ -> finish() }
            .setCancelable(false)
            .show()
    }

    private fun startNewGame() {
        engine.reset()
        boardView.inputEnabled = true
        aiThinking = false
        tvStatus.text = getString(R.string.your_turn)
        boardView.invalidate()
    }

    private fun onHumanMove(row: Int, col: Int) {
        if (aiThinking || engine.gameOver) return
        if (engine.curPlayer != humanPlayer) return

        val success = engine.makeMove(row, col)
        if (!success) return

        boardView.invalidate()

        if (engine.gameOver) {
            handleGameEnd()
            return
        }

        boardView.inputEnabled = false
        aiThinking = true
        tvStatus.text = getString(R.string.ai_thinking)

        aiExecutor.execute {
            try {
                val move = aiPlayer.selectMove(engine, aiPlayerStone)
                Log.d("MainActivity", "AI selected move: $move")
                mainHandler.post { onAIMove(move) }
            } catch (e: Exception) {
                Log.e("MainActivity", "selectMove crashed", e)
                mainHandler.post {
                    aiThinking = false
                    boardView.inputEnabled = true
                    tvStatus.text = "AI error: ${e.message}"
                }
            }
        }
    }

    private fun onAIMove(action: Int) {
        if (action < 0) {
            aiThinking = false
            boardView.inputEnabled = true
            tvStatus.text = "AI cannot move"
            return
        }

        val row = action / GameEngine.BOARD_SIZE
        val col = action % GameEngine.BOARD_SIZE

        engine.makeAIMove(row, col)
        boardView.invalidate()
        aiThinking = false

        if (engine.gameOver) {
            handleGameEnd()
        } else {
            boardView.inputEnabled = true
            tvStatus.text = getString(R.string.your_turn)
        }
    }

    private fun handleGameEnd() {
        boardView.inputEnabled = false
        when (engine.winner) {
            humanPlayer -> {
                tvStatus.text = getString(R.string.you_win)
                Toast.makeText(this, "Congratulations, you beat the AI!", Toast.LENGTH_LONG).show()
            }
            aiPlayerStone -> {
                tvStatus.text = getString(R.string.ai_wins)
                Toast.makeText(this, "AI wins, try another round!", Toast.LENGTH_LONG).show()
            }
            2 -> {
                tvStatus.text = getString(R.string.draw)
                Toast.makeText(this, "Board is full, it's a draw!", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        aiPlayer.close()
        aiExecutor.shutdown()
    }
}
