package com.gomoku.app.game

class PatternDetector {

    companion object {
        const val BOARD_SIZE = 11
        const val WIN_LENGTH = 5

        val OFFSETS = arrayOf(
            intArrayOf(0, 1),
            intArrayOf(1, 0),
            intArrayOf(1, 1),
            intArrayOf(-1, 1)
        )

        const val IDX_OPEN_ONE = 0
        const val IDX_OPEN_TWO = 1
        const val IDX_OPEN_THREE = 2
        const val IDX_OPEN_FOUR = 3
        const val IDX_CLOSED_FOUR = 4
    }

    data class PatternResult(
        val count: Map<String, Int>,
        val state: Array<Array<FloatArray>>
    )

    fun calPatternCount(board: Array<IntArray>, player: Int): PatternResult {
        val patternCount = mutableMapOf(
            "five" to 0,
            "open_four" to 0, "closed_four" to 0,
            "open_three" to 0, "closed_three" to 0,
            "open_two" to 0, "closed_two" to 0,
            "open_one" to 0, "closed_one" to 0
        )

        val patternState = Array(5) { Array(BOARD_SIZE) { FloatArray(BOARD_SIZE) } }
        val countedLines = mutableSetOf<String>()

        for (r in 0 until BOARD_SIZE) {
            for (c in 0 until BOARD_SIZE) {
                if (board[r][c] != player) continue

                for ((dr, dc) in OFFSETS) {
                    var rr = r
                    var cc = c
                    while (true) {
                        val pr = rr - dr
                        val pc = cc - dc
                        if (pr !in 0 until BOARD_SIZE || pc !in 0 until BOARD_SIZE) break
                        rr = pr
                        cc = pc
                    }

                    val key = "$rr,$cc,$dr,$dc"
                    if (key in countedLines) continue

                    val line = getFullLine(board, rr, cc, dr, dc, player)
                    if (line.length < WIN_LENGTH) continue

                    countedLines.add(key)

                    var tmp = line
                    for (p in 0..4) {
                        var idx = 0
                        while (idx <= tmp.length - WIN_LENGTH) {
                            val window = tmp.substring(idx, idx + WIN_LENGTH)
                            val leftExt = if (idx >= 1) line[idx - 1] else null
                            val rightExt = if (idx + WIN_LENGTH < line.length) line[idx + WIN_LENGTH] else null

                            val pattern = when (p) {
                                0 -> classifyFive(window)
                                1 -> classifyFour(window, leftExt, rightExt)
                                2 -> classifyThree(window, leftExt, rightExt)
                                3 -> classifyTwo(window, leftExt, rightExt)
                                else -> classifyOne(window, leftExt, rightExt)
                            }

                            if (pattern != null) {
                                patternCount[pattern] = patternCount[pattern]!! + 1

                                val baseR = rr + dr * idx
                                val baseC = cc + dc * idx
                                val stateIdx = when (pattern) {
                                    "open_one" -> IDX_OPEN_ONE
                                    "open_two" -> IDX_OPEN_TWO
                                    "open_three" -> IDX_OPEN_THREE
                                    "open_four" -> IDX_OPEN_FOUR
                                    "closed_four" -> IDX_CLOSED_FOUR
                                    else -> -1
                                }
                                if (stateIdx >= 0) {
                                    drawPattern(patternState[stateIdx], baseR, baseC, dr, dc, window)
                                }

                                val sb = StringBuilder(tmp)
                                for (wi in idx until idx + WIN_LENGTH) {
                                    sb.setCharAt(wi, '.')
                                }
                                tmp = sb.toString()
                            }

                            idx++
                        }
                    }
                }
            }
        }

        return PatternResult(patternCount, patternState)
    }

    private fun getFullLine(
        board: Array<IntArray>,
        r: Int, c: Int, dr: Int, dc: Int,
        player: Int
    ): String {
        val sb = StringBuilder()
        var cr = r
        var cc = c
        while (cr in 0 until BOARD_SIZE && cc in 0 until BOARD_SIZE) {
            val v = board[cr][cc]
            sb.append(
                when {
                    v == player -> 'x'
                    v == -player -> 'o'
                    else -> '.'
                }
            )
            cr += dr
            cc += dc
        }
        return sb.toString()
    }

    private fun drawPattern(
        state: Array<FloatArray>,
        r: Int, c: Int, dr: Int, dc: Int,
        window: String
    ) {
        var cr = r
        var cc = c
        for (ch in window) {
            if (cr !in 0 until BOARD_SIZE || cc !in 0 until BOARD_SIZE) break
            if (ch == 'x') {
                state[cr][cc] = 1f
            }
            cr += dr
            cc += dc
        }
    }

    private fun classifyFive(window: String): String? {
        return if (window == "xxxxx") "five" else null
    }

    private fun classifyFour(window: String, leftExt: Char?, rightExt: Char?): String? {
        val countX = window.count { it == 'x' }
        val countO = window.count { it == 'o' }
        if (countX != 4 || countO > 0) return null

        if (window in setOf(".xxxx", "xxxx.")) {
            return if (isSideOpen(window, "left", leftExt) && isSideOpen(window, "right", rightExt))
                "open_four" else "closed_four"
        }
        return "closed_four"
    }

    private fun classifyThree(window: String, leftExt: Char?, rightExt: Char?): String? {
        val countX = window.count { it == 'x' }
        val countO = window.count { it == 'o' }
        if (countX != 3 || countO > 0) return null

        if (window in setOf("xxx..", ".xxx.", "..xxx")) {
            return if (isSideOpen(window, "left", leftExt) && isSideOpen(window, "right", rightExt))
                "open_three" else "closed_three"
        }
        if (window in setOf("xx.x.", "x.xx.", ".xx.x", ".x.xx")) {
            return if (isSideOpen(window, "left", leftExt) && isSideOpen(window, "right", rightExt))
                "open_three" else "closed_three"
        }
        return "closed_three"
    }

    private fun classifyTwo(window: String, leftExt: Char?, rightExt: Char?): String? {
        val countX = window.count { it == 'x' }
        val countO = window.count { it == 'o' }
        if (countX != 2 || countO > 0) return null

        if (window in setOf("xx...", ".xx..", "..xx.", "...xx")) {
            return if (isSideOpen(window, "left", leftExt) && isSideOpen(window, "right", rightExt))
                "open_two" else "closed_two"
        }
        val indices = window.mapIndexedNotNull { i, ch -> if (ch == 'x') i else null }
        if (indices.size == 2) {
            val gap = indices[1] - indices[0]
            if (gap == 2) {
                return if (isSideOpen(window, "left", leftExt) && isSideOpen(window, "right", rightExt))
                    "open_two" else "closed_two"
            }
        }
        return null
    }

    private fun classifyOne(window: String, leftExt: Char?, rightExt: Char?): String? {
        val countX = window.count { it == 'x' }
        val countO = window.count { it == 'o' }
        if (countX != 1 || countO > 0) return null

        if (window in setOf("x....", ".x...", "..x..", "...x.", "....x")) {
            return if (isSideOpen(window, "left", leftExt) && isSideOpen(window, "right", rightExt))
                "open_one" else "closed_one"
        }
        return null
    }

    private fun isSideOpen(window: String, side: String, extension: Char?): Boolean {
        return if (side == "left") {
            if (window[0] == '.') true
            else extension != null && extension == '.'
        } else {
            if (window.last() == '.') true
            else extension != null && extension == '.'
        }
    }
}
