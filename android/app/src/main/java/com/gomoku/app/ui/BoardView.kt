package com.gomoku.app.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import com.gomoku.app.game.GameEngine

class BoardView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    companion object {
        private const val BOARD_SIZE = GameEngine.BOARD_SIZE
        private const val MARGIN_RATIO = 0.08f
        private const val STONE_FRACTION = 0.42f
        private const val STAR_FRACTION = 0.12f
    }

    private val boardPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#EEB866")
        style = Paint.Style.FILL
    }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#33000000")
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }
    private val blackStonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val blackStoneBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF333333")
        style = Paint.Style.STROKE
        strokeWidth = 1.5f
    }
    private val whiteStonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFF0F0F0")
        style = Paint.Style.FILL
    }
    private val whiteStoneBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFCCCCCC")
        style = Paint.Style.STROKE
        strokeWidth = 1.5f
    }
    private val starPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF000000")
        style = Paint.Style.FILL
    }
    private val highlightPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFFF4444")
        style = Paint.Style.STROKE
        strokeWidth = 2.5f
    }
    private val aiHighlightPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF4488FF")
        style = Paint.Style.STROKE
        strokeWidth = 2.5f
    }

    private var boardLeft = 0f
    private var boardTop = 0f
    private var cellSize = 0f
    private var stoneRadius = 0f
    private var margin = 0f

    var gameEngine: GameEngine? = null
    var onCellTapped: ((row: Int, col: Int) -> Unit)? = null
    var inputEnabled: Boolean = true

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF666666")
        textSize = 24f
        textAlign = Paint.Align.CENTER
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        computeLayout()

        canvas.drawRoundRect(
            boardLeft - margin * 0.5f,
            boardTop - margin * 0.5f,
            boardLeft + cellSize * (BOARD_SIZE - 1) + margin * 0.5f,
            boardTop + cellSize * (BOARD_SIZE - 1) + margin * 0.5f,
            8f, 8f, boardPaint
        )

        for (i in 0 until BOARD_SIZE) {
            val pos = margin + i * cellSize
            canvas.drawLine(margin, pos, margin + cellSize * (BOARD_SIZE - 1), pos, linePaint)
            canvas.drawLine(pos, margin, pos, margin + cellSize * (BOARD_SIZE - 1), linePaint)
        }

        val center = BOARD_SIZE / 2
        val centerX = margin + center * cellSize
        val centerY = margin + center * cellSize
        canvas.drawCircle(centerX, centerY, STAR_FRACTION * cellSize, starPaint)

        for (c in 0 until BOARD_SIZE) {
            val x = margin + c * cellSize
            val y = margin - 18f
            canvas.drawText(('A' + c).toString(), x, y, labelPaint)
        }

        for (r in 0 until BOARD_SIZE) {
            val x = margin - 22f
            val y = margin + r * cellSize + 8f
            canvas.drawText((r + 1).toString(), x, y, labelPaint)
        }

        val engine = gameEngine ?: return
        for (r in 0 until BOARD_SIZE) {
            for (c in 0 until BOARD_SIZE) {
                val stone = engine.board[r][c]
                if (stone == GameEngine.EMPTY) continue

                val cx = margin + c * cellSize
                val cy = margin + r * cellSize

                if (stone == GameEngine.BLACK) {
                    drawBlackStone(canvas, cx, cy)
                } else {
                    drawWhiteStone(canvas, cx, cy)
                }
            }
        }

        if (engine.lastMoveRow >= 0 && engine.lastMoveCol >= 0) {
            val hx = margin + engine.lastMoveCol * cellSize
            val hy = margin + engine.lastMoveRow * cellSize
            canvas.drawCircle(hx, hy, stoneRadius * 0.5f, highlightPaint)
        }

        if (engine.aiLastMoveRow >= 0 && engine.aiLastMoveCol >= 0) {
            val ax = margin + engine.aiLastMoveCol * cellSize
            val ay = margin + engine.aiLastMoveRow * cellSize
            canvas.drawCircle(ax, ay, stoneRadius * 0.5f, aiHighlightPaint)
        }
    }

    private fun drawBlackStone(canvas: Canvas, cx: Float, cy: Float) {
        val gradient = RadialGradient(
            cx - stoneRadius * 0.3f, cy - stoneRadius * 0.3f,
            stoneRadius,
            intArrayOf(Color.parseColor("#FF4A4A4A"), Color.parseColor("#FF0A0A0A")),
            floatArrayOf(0.3f, 1f),
            Shader.TileMode.CLAMP
        )
        blackStonePaint.shader = gradient
        canvas.drawCircle(cx, cy, stoneRadius, blackStonePaint)
        blackStonePaint.shader = null
        canvas.drawCircle(cx, cy, stoneRadius, blackStoneBorderPaint)
    }

    private fun drawWhiteStone(canvas: Canvas, cx: Float, cy: Float) {
        val gradient = RadialGradient(
            cx - stoneRadius * 0.3f, cy - stoneRadius * 0.3f,
            stoneRadius,
            intArrayOf(Color.WHITE, Color.parseColor("#FFD8D8D8")),
            floatArrayOf(0.3f, 1f),
            Shader.TileMode.CLAMP
        )
        whiteStonePaint.shader = gradient
        canvas.drawCircle(cx, cy, stoneRadius, whiteStonePaint)
        whiteStonePaint.shader = null
        canvas.drawCircle(cx, cy, stoneRadius, whiteStoneBorderPaint)
    }

    private fun computeLayout() {
        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        val size = minOf(viewWidth, viewHeight)
        margin = size * MARGIN_RATIO
        cellSize = (size - 2 * margin) / (BOARD_SIZE - 1)
        stoneRadius = cellSize * STONE_FRACTION
        boardLeft = margin
        boardTop = margin
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (!inputEnabled || gameEngine == null) return false
        if (event.action != MotionEvent.ACTION_DOWN) return false

        computeLayout()

        val x = event.x
        val y = event.y

        val col = ((x - margin + cellSize / 2) / cellSize).toInt()
        val row = ((y - margin + cellSize / 2) / cellSize).toInt()

        if (row !in 0 until BOARD_SIZE || col !in 0 until BOARD_SIZE) return false

        val cx = margin + col * cellSize
        val cy = margin + row * cellSize
        val dist = Math.hypot((x - cx).toDouble(), (y - cy).toDouble())

        if (dist > stoneRadius * 1.2) return false

        onCellTapped?.invoke(row, col)
        return true
    }
}
