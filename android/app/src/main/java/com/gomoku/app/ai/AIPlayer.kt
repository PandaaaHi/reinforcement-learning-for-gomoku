package com.gomoku.app.ai

import com.gomoku.app.game.GameEngine

interface AIPlayer {

    val isLoaded: Boolean
    val useCPU: Boolean
    val lastError: String?

    fun loadModel(): Boolean
    fun selectMove(engine: GameEngine, forPlayer: Int): Int
    fun evaluatePosition(engine: GameEngine, forPlayer: Int): Float
    fun close()
}
