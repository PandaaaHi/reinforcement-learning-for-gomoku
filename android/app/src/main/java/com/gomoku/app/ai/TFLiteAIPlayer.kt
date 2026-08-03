package com.gomoku.app.ai

import android.content.Context
import android.util.Log
import com.gomoku.app.game.GameEngine
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TFLiteAIPlayer(
    private val context: Context,
    private val modelFileName: String = "gomoku_model.tflite"
) : AIPlayer {
    companion object {
        private const val TAG = "TFLiteAIPlayer"
        private const val BOARD_SQ = GameEngine.BOARD_SIZE * GameEngine.BOARD_SIZE
        private const val INPUT_SIZE = GameEngine.NUM_CHANNELS * BOARD_SQ
        private const val INPUT_BYTES = INPUT_SIZE * 4
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    override var useCPU: Boolean = false
        private set

    override var isLoaded: Boolean = false
        private set

    override var lastError: String? = null
        private set

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(INPUT_BYTES).apply { order(ByteOrder.nativeOrder()) }

    override fun loadModel(): Boolean {
        if (tryLoadWithGPU()) return true
        Log.w(TAG, "GPU delegate unavailable, using CPU")
        return tryLoadWithCPU()
    }

    private fun tryLoadWithGPU(): Boolean {
        return try {
            val modelBuffer = FileUtil.loadMappedFile(context, modelFileName)
            gpuDelegate = GpuDelegate()
            interpreter = Interpreter(modelBuffer, Interpreter.Options().apply {
                addDelegate(gpuDelegate)
                setNumThreads(4)
            })

            val inputShape = interpreter!!.getInputTensor(0).shape()
            Log.d(TAG, "TFLite model input shape: ${inputShape.contentToString()}")

            testInference()
            Log.d(TAG, "GPU delegate verified OK")
            isLoaded = true
            true
        } catch (e: Exception) {
            Log.w(TAG, "GPU delegate load/inference failed: ${e.message}")
            interpreter?.close()
            interpreter = null
            gpuDelegate?.close()
            gpuDelegate = null
            false
        }
    }

    private fun tryLoadWithCPU(): Boolean {
        return try {
            val modelBuffer = FileUtil.loadMappedFile(context, modelFileName)
            interpreter = Interpreter(modelBuffer, Interpreter.Options().apply {
                setNumThreads(4)
            })
            gpuDelegate = null
            useCPU = true
            isLoaded = true

            val inputShape = interpreter!!.getInputTensor(0).shape()
            Log.d(TAG, "TFLite model input shape: ${inputShape.contentToString()}")
            Log.d(TAG, "CPU model loaded OK")
            true
        } catch (e: Exception) {
            lastError = "TFLite model load failed: ${e.message}"
            Log.e(TAG, lastError!!, e)
            isLoaded = false
            false
        }
    }

    private fun testInference() {
        val zeroInput = ByteBuffer.allocateDirect(INPUT_BYTES)
            .apply { order(ByteOrder.nativeOrder()) }
        val policyOut = Array(1) { FloatArray(BOARD_SQ) }
        val valueOut = Array(1) { FloatArray(1) }
        val outputs = mutableMapOf<Int, Any>(0 to policyOut, 1 to valueOut)
        interpreter!!.runForMultipleInputsOutputs(arrayOf(zeroInput), outputs)
    }

    override fun selectMove(engine: GameEngine, forPlayer: Int): Int {
        if (!isLoaded || interpreter == null) {
            Log.w(TAG, "selectMove: model not loaded")
            return -1
        }

        val validMoves = engine.getValidMoves()
        if (validMoves.isEmpty()) return -1

        if (engine.moveCount == 0) {
            return (GameEngine.BOARD_SIZE / 2) * GameEngine.BOARD_SIZE + (GameEngine.BOARD_SIZE / 2)
        }

        val stateArray = engine.buildState(forPlayer)
        fillInputBuffer(stateArray)

        val policyOutput = Array(1) { FloatArray(BOARD_SQ) }
        val valueOutput = Array(1) { FloatArray(1) }
        val outputs = mutableMapOf<Int, Any>(0 to policyOutput, 1 to valueOutput)

        if (!runInference(outputs)) {
            Log.w(TAG, "Inference failed, returning first valid move")
            return validMoves.first()
        }

        val logits = policyOutput[0]
        Log.d(TAG, "Policy logits range: [${logits.minOrNull()}, ${logits.maxOrNull()}]")
        Log.d(TAG, "Value: ${valueOutput[0][0]}")

        val maskedLogits = FloatArray(BOARD_SQ) { Float.NEGATIVE_INFINITY }
        for (move in validMoves) {
            maskedLogits[move] = logits[move]
        }

        val maxLogit = maskedLogits.filter { it.isFinite() }.maxOrNull() ?: return validMoves.first()
        var sumExp = 0f
        val probs = FloatArray(maskedLogits.size)
        for (i in maskedLogits.indices) {
            if (maskedLogits[i].isFinite()) {
                probs[i] = Math.exp((maskedLogits[i] - maxLogit).toDouble()).toFloat()
                sumExp += probs[i]
            }
        }
        if (sumExp > 0f) {
            for (i in probs.indices) {
                probs[i] /= sumExp
            }
        }

        var bestMove = validMoves.first()
        var bestProb = probs[bestMove]
        for (move in validMoves) {
            if (probs[move] > bestProb) {
                bestProb = probs[move]
                bestMove = move
            }
        }

        Log.d(TAG, "Best move idx=$bestMove, prob=$bestProb, " +
                "row=${bestMove / GameEngine.BOARD_SIZE}, col=${bestMove % GameEngine.BOARD_SIZE}")
        return bestMove
    }

    override fun evaluatePosition(engine: GameEngine, forPlayer: Int): Float {
        if (!isLoaded || interpreter == null) return 0f

        fillInputBuffer(engine.buildState(forPlayer))
        val policyOutput = Array(1) { FloatArray(BOARD_SQ) }
        val valueOutput = Array(1) { FloatArray(1) }
        val outputs = mutableMapOf<Int, Any>(0 to policyOutput, 1 to valueOutput)

        if (!runInference(outputs)) return 0f
        return valueOutput[0][0]
    }

    private fun fillInputBuffer(data: FloatArray) {
        inputBuffer.rewind()
        inputBuffer.asFloatBuffer().put(data)
    }

    private fun runInference(outputs: MutableMap<Int, Any>): Boolean {
        return try {
            val inputs = arrayOf(inputBuffer)
            interpreter!!.runForMultipleInputsOutputs(inputs, outputs)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed: ${e.message}", e)
            false
        }
    }

    override fun close() {
        try { interpreter?.close() } catch (_: Exception) {}
        interpreter = null
        try { gpuDelegate?.close() } catch (_: Exception) {}
        gpuDelegate = null
        isLoaded = false
        useCPU = false
    }
}
