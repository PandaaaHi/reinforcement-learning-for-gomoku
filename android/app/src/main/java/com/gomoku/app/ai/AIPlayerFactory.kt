package com.gomoku.app.ai

import android.content.Context
import android.util.Log

object AIPlayerFactory {
    private const val TAG = "AIPlayerFactory"

    fun create(context: Context): AIPlayer {
        val ctx = context.applicationContext

        Log.d(TAG, "Attempting to create SNPE AIPlayer...")
        val snpePlayer = SNPEAIPlayer(ctx)
        if (snpePlayer.loadModel()) {
            Log.d(TAG, "SNPE AIPlayer loaded successfully")
            return snpePlayer
        }
        Log.w(TAG, "SNPE failed: ${snpePlayer.lastError}. Falling back to TFLite.")
        snpePlayer.close()

        Log.d(TAG, "Attempting to create TFLite AIPlayer...")
        val tflitePlayer = TFLiteAIPlayer(ctx)
        if (tflitePlayer.loadModel()) {
            Log.d(TAG, "TFLite AIPlayer loaded successfully")
            return tflitePlayer
        }

        Log.e(TAG, "Both engines failed. TFLite error: ${tflitePlayer.lastError}")
        return tflitePlayer
    }
}
