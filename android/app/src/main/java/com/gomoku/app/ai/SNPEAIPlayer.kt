package com.gomoku.app.ai

import android.app.Application
import android.content.Context
import android.util.Log
import com.gomoku.app.game.GameEngine
import com.qualcomm.qti.snpe.FloatTensor
import com.qualcomm.qti.snpe.NeuralNetwork
import com.qualcomm.qti.snpe.SNPE
import java.io.File
import java.io.FileOutputStream

class SNPEAIPlayer(
    private val context: Context,
    private val modelFileName: String = "gomoku_model.dlc"
) : AIPlayer {
    companion object {
        private const val TAG = "SNPEAIPlayer"
        private const val BOARD_SQ = GameEngine.BOARD_SIZE * GameEngine.BOARD_SIZE
        private const val NUM_CHANNELS = GameEngine.NUM_CHANNELS
    }

    private var network: NeuralNetwork? = null
    private var inputTensor: FloatTensor? = null
    private var inputTensorName: String = ""
    private var outputNames: Set<String> = emptySet()

    override var isLoaded: Boolean = false
        private set

    override var useCPU: Boolean = false
        private set

    override var lastError: String? = null
        private set

    override fun loadModel(): Boolean {
        try {
            // WARNING: caching with exists() need to be avoided here.
            // If the APK ships a newer model, the stale cached copy in filesDir
            // would still be used, causing a model-version mismatch.

            // val dlcFile = File(context.filesDir, modelFileName)
            // if (!dlcFile.exists()) {
            //     try {
            //         context.assets.open(modelFileName).use { input ->
            //             FileOutputStream(dlcFile).use { output ->
            //                 input.copyTo(output)
            //             }
            //         }
            //     } catch (e: Exception) {
            //         lastError = "DLC file not found: $modelFileName\nPlease place it in app/src/main/assets/"
            //         Log.e(TAG, lastError!!, e)
            //         return false
            //     }
            // }

            val dlcFile = File(context.filesDir, modelFileName)
            try {
                context.assets.open(modelFileName).use { input ->
                    FileOutputStream(dlcFile).use { output ->
                        input.copyTo(output)
                    }
                }
            } catch (e: Exception) {
                lastError = "DLC file not found: $modelFileName\nPlease place it in app/src/main/assets/"
                Log.e(TAG, lastError!!, e)
                return false
            }
            Log.d(TAG, "DLC copied to: ${dlcFile.absolutePath} (${dlcFile.length()} bytes)")

            if (tryBuild(dlcFile, NeuralNetwork.Runtime.DSP, NeuralNetwork.Runtime.GPU_FLOAT16, NeuralNetwork.Runtime.CPU)) {
                Log.d(TAG, "SNPE model loaded on DSP")
                return finishLoad()
            }

            Log.w(TAG, "DSP unavailable, trying GPU_FLOAT16")
            if (tryBuild(dlcFile, NeuralNetwork.Runtime.GPU_FLOAT16, NeuralNetwork.Runtime.CPU)) {
                Log.d(TAG, "SNPE model loaded on GPU")
                return finishLoad()
            }

            Log.w(TAG, "GPU unavailable, using CPU")
            if (tryBuild(dlcFile, NeuralNetwork.Runtime.CPU)) {
                useCPU = true
                Log.d(TAG, "SNPE model loaded on CPU")
                return finishLoad()
            }

            lastError = buildErrors.joinToString("\n")
            if (lastError.isNullOrEmpty()) {
                lastError = "All SNPE runtimes failed to load, no detailed error available"
            }
            Log.e(TAG, lastError!!)
            return false
        } catch (e: Exception) {
            lastError = "SNPE model load exception: ${e.message}"
            Log.e(TAG, lastError!!, e)
            releaseResources()
            return false
        }
    }

    private val buildErrors = mutableListOf<String>()

    private fun tryBuild(dlcFile: File, vararg runtimes: NeuralNetwork.Runtime): Boolean {
        return try {
            val builder = SNPE.NeuralNetworkBuilder(context.applicationContext as Application)
                .setModel(dlcFile)
                .setRuntimeOrder(*runtimes)
                .setUnconsumedTensorsOutput(true) // required; otherwise the policy output may be lost

            network = builder.build()
            true
        } catch (e: Exception) {
            val msg = "[${runtimes.first()}] ${e.javaClass.simpleName}: ${e.message}"
            Log.w(TAG, "Build failed: $msg")
            buildErrors.add(msg)
            network?.release()
            network = null
            false
        }
    }

    private fun finishLoad(): Boolean {
        try {
            val net = network ?: return false

            inputTensorName = net.inputTensorsNames.first()
            outputNames = net.outputTensorsNames

            Log.d(TAG, "Input tensor: '$inputTensorName', Outputs: $outputNames")

            inputTensor = net.createFloatTensor(1, NUM_CHANNELS, GameEngine.BOARD_SIZE, GameEngine.BOARD_SIZE)

            val shape = inputTensor!!.shape
            Log.d(TAG, "Input shape: ${shape.joinToString()}")

            testInference()

            isLoaded = true
            Log.d(TAG, "SNPE model ready")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "finishLoad failed: ${e.message}", e)
            releaseResources()
            return false
        }
    }

    private fun testInference() {
        val zeroData = FloatArray(NUM_CHANNELS * BOARD_SQ)
        inputTensor!!.write(zeroData, 0, zeroData.size)
        val inputMap = mapOf(inputTensorName to inputTensor!!)
        val outputs = network!!.execute(inputMap)
        for ((name, tensor) in outputs) {
            Log.d(TAG, "Output '$name': shape=${tensor.shape.joinToString()}")
            tensor.release()
        }
        Log.d(TAG, "Test inference OK")
    }

    override fun selectMove(engine: GameEngine, forPlayer: Int): Int {
        if (!isLoaded || network == null || inputTensor == null) {
            Log.w(TAG, "selectMove: model not loaded")
            return -1
        }

        val validMoves = engine.getValidMoves()
        if (validMoves.isEmpty()) return -1

        if (engine.moveCount == 0) {
            val center = GameEngine.BOARD_SIZE / 2
            return center * GameEngine.BOARD_SIZE + center
        }

        val stateArray = engine.buildState(forPlayer)
        fillInputTensor(stateArray)

        val outputs = runInference() ?: run {
            Log.w(TAG, "Inference failed, returning first valid move")
            return validMoves.first()
        }

        val allOutputs = mutableListOf<Triple<String, FloatTensor, Long>>()
        for ((name, tensor) in outputs) {
            val numel = tensor.shape.fold(1L) { acc, d -> acc * d }
            allOutputs.add(Triple(name, tensor, numel))
            Log.d(TAG, "  output name='$name' shape=[${tensor.shape.joinToString()}] numel=$numel")
        }

        if (allOutputs.isEmpty()) {
            Log.w(TAG, "No output tensors returned from execute()")
            return validMoves.first()
        }

        val (policyName, policyTensor, policyNumel) =
            allOutputs.firstOrNull { it.first.contains("policy", ignoreCase = true) }
                ?: allOutputs.firstOrNull { it.third == BOARD_SQ.toLong() }
                ?: allOutputs.maxByOrNull { it.third }!!

        val logits = FloatArray(BOARD_SQ)
        try {
            policyTensor.read(logits, 0, BOARD_SQ)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to read policy output '${policyName}' (numel=$policyNumel): ${e.message}")
        }

        for ((_, tensor, _) in allOutputs) {
            tensor.release()
        }

        Log.d(TAG, "Policy logits range: [${logits.minOrNull()}, ${logits.maxOrNull()}]")

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
        if (!isLoaded || network == null || inputTensor == null) return 0f

        fillInputTensor(engine.buildState(forPlayer))
        val outputs = runInference() ?: return 0f

        val allOutputs = outputs.map { (name, tensor) ->
            val numel = tensor.shape.fold(1L) { acc, d -> acc * d }
            Triple(name, tensor, numel)
        }

        if (allOutputs.isEmpty()) {
            Log.w(TAG, "No output tensors for evaluatePosition")
            return 0f
        }

        val (_, valueTensor, _) =
            allOutputs.firstOrNull { it.first.contains("value", ignoreCase = true) }
                ?: allOutputs.firstOrNull { it.third == 1L }
                ?: allOutputs.minByOrNull { it.third }!!

        val value = FloatArray(1)
        try {
            valueTensor.read(value, 0, 1)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to read value output: ${e.message}")
        }

        for ((_, tensor, _) in allOutputs) {
            tensor.release()
        }

        return value[0]
    }

    private fun fillInputTensor(data: FloatArray) {
        inputTensor?.write(data, 0, data.size)
    }

    private fun runInference(): Map<String, FloatTensor>? {
        return try {
            val inputMap = mapOf(inputTensorName to inputTensor!!)
            network!!.execute(inputMap)
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed: ${e.message}", e)
            null
        }
    }

    override fun close() {
        releaseResources()
        isLoaded = false
        useCPU = false
    }

    private fun releaseResources() {
        try { inputTensor?.release() } catch (_: Exception) {}
        inputTensor = null
        try { network?.release() } catch (_: Exception) {}
        network = null
    }
}
