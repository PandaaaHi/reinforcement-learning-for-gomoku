package com.gomoku.app.game

class GameEngine {

    companion object {
        const val BOARD_SIZE = 11
        const val WIN_LENGTH = 5
        const val NUM_CHANNELS = 12

        val OFFSETS = PatternDetector.OFFSETS

        const val EMPTY = 0
        const val BLACK = 1
        const val WHITE = -1
    }

    val board = Array(BOARD_SIZE) { IntArray(BOARD_SIZE) }

    var curPlayer: Int = BLACK
        private set

    var winner: Int = 0
        private set

    var moveCount: Int = 0
        private set

    var lastMoveRow: Int = -1
        private set
    var lastMoveCol: Int = -1
        private set

    var aiLastMoveRow: Int = -1
        private set
    var aiLastMoveCol: Int = -1
        private set

    private val detector = PatternDetector()
    var gameOver: Boolean = false
        private set

    fun reset() {
        for (r in 0 until BOARD_SIZE) {
            for (c in 0 until BOARD_SIZE) {
                board[r][c] = EMPTY
            }
        }
        curPlayer = BLACK
        winner = 0
        moveCount = 0
        lastMoveRow = -1
        lastMoveCol = -1
        aiLastMoveRow = -1
        aiLastMoveCol = -1
        gameOver = false
    }

    fun makeMove(row: Int, col: Int): Boolean {
        if (gameOver) return false
        if (row !in 0 until BOARD_SIZE || col !in 0 until BOARD_SIZE) return false
        if (board[row][col] != EMPTY) return false

        board[row][col] = curPlayer
        moveCount++
        lastMoveRow = row
        lastMoveCol = col

        if (checkWin(row, col)) {
            winner = curPlayer
            gameOver = true
            return true
        }

        if (moveCount >= BOARD_SIZE * BOARD_SIZE) {
            winner = 2
            gameOver = true
            return true
        }

        curPlayer = if (curPlayer == BLACK) WHITE else BLACK
        return true
    }

    fun makeAIMove(row: Int, col: Int): Boolean {
        if (board[row][col] != EMPTY) return false
        board[row][col] = curPlayer
        moveCount++
        aiLastMoveRow = row
        aiLastMoveCol = col

        if (checkWin(row, col)) {
            winner = curPlayer
            gameOver = true
            return true
        }
        if (moveCount >= BOARD_SIZE * BOARD_SIZE) {
            winner = 2
            gameOver = true
            return true
        }

        curPlayer = if (curPlayer == BLACK) WHITE else BLACK
        lastMoveRow = -1
        lastMoveCol = -1
        return true
    }

    fun getValidMoves(): List<Int> {
        val moves = mutableListOf<Int>()
        for (r in 0 until BOARD_SIZE) {
            for (c in 0 until BOARD_SIZE) {
                if (board[r][c] == EMPTY) {
                    moves.add(r * BOARD_SIZE + c)
                }
            }
        }
        return moves
    }

    fun checkWin(row: Int, col: Int): Boolean {
        val player = board[row][col]
        if (player == EMPTY) return false

        for ((dr, dc) in OFFSETS) {
            var count = 1

            var r = row + dr
            var c = col + dc
            while (r in 0 until BOARD_SIZE && c in 0 until BOARD_SIZE && board[r][c] == player) {
                count++
                r += dr
                c += dc
            }

            r = row - dr
            c = col - dc
            while (r in 0 until BOARD_SIZE && c in 0 until BOARD_SIZE && board[r][c] == player) {
                count++
                r -= dr
                c -= dc
            }

            if (count >= WIN_LENGTH) return true
        }
        return false
    }

    fun buildState(forPlayer: Int): FloatArray {
        val state = FloatArray(NUM_CHANNELS * BOARD_SIZE * BOARD_SIZE)
        val opponent = if (forPlayer == BLACK) WHITE else BLACK

        val curResult = detector.calPatternCount(board, forPlayer)
        val oppResult = detector.calPatternCount(board, opponent)

        for (r in 0 until BOARD_SIZE) {
            for (c in 0 until BOARD_SIZE) {
                state[0 * 121 + r * 11 + c] = if (board[r][c] == forPlayer) 1f else 0f
                state[1 * 121 + r * 11 + c] = if (board[r][c] == opponent) 1f else 0f
                state[2 * 121 + r * 11 + c] = forPlayer.toFloat()
                state[3 * 121 + r * 11 + c] = curResult.state[PatternDetector.IDX_OPEN_ONE][r][c]
                state[4 * 121 + r * 11 + c] = curResult.state[PatternDetector.IDX_OPEN_TWO][r][c]
                state[5 * 121 + r * 11 + c] = curResult.state[PatternDetector.IDX_OPEN_THREE][r][c]
                state[6 * 121 + r * 11 + c] = curResult.state[PatternDetector.IDX_OPEN_FOUR][r][c]
                state[7 * 121 + r * 11 + c] = curResult.state[PatternDetector.IDX_CLOSED_FOUR][r][c]
                state[8 * 121 + r * 11 + c] = oppResult.state[PatternDetector.IDX_OPEN_ONE][r][c]
                state[9 * 121 + r * 11 + c] = oppResult.state[PatternDetector.IDX_OPEN_TWO][r][c]
                state[10 * 121 + r * 11 + c] = oppResult.state[PatternDetector.IDX_OPEN_THREE][r][c]
                state[11 * 121 + r * 11 + c] = oppResult.state[PatternDetector.IDX_CLOSED_FOUR][r][c]
            }
        }

        return state
    }
}
