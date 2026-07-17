import pygame
import sys
import torch
import numpy as np

from ai import RuleAI, RLAI
from config import *

class GomokuBoard:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))

    def get_board_pos(self, pos):
        x, y = pos
        col = round((x - MARGIN) / CELL_SIZE)
        row = round((y - MARGIN) / CELL_SIZE)
        if row >= 0 and row < BOARD_SIZE and col >= 0 and col < BOARD_SIZE:
            if self.board[row, col] == 0:
                return POS(row, col)
        return None

    def update_board(self, pos, stone):
        self.board[pos.row][pos.col] = stone
        return self.is_win(pos, stone)

    def get_board(self):
        return self.board
    
    def is_win(self, pos, stone):
        for offset in OFFSET:
            count = 1

            for step in range(1, 5):
                row = pos.row + step * offset[0]
                col = pos.col + step * offset[1]
                if row >= 0 and row < BOARD_SIZE and col >= 0 and col < BOARD_SIZE:
                    if self.board[row, col] == stone:
                        count += 1
                    else:
                        break
                else:
                    break

            for step in range(1, 5):
                row = pos.row - step * offset[0]
                col = pos.col - step * offset[1]
                if row >= 0 and row < BOARD_SIZE and col >= 0 and col < BOARD_SIZE:
                    if self.board[row, col] == stone:
                        count += 1
                    else:
                        break
                else:
                    break

            if count >= 5:
                return True
        return False
    
    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))

    def draw_board(self, screen):
        screen.fill(BOARD_COLOR)

        for i in range(BOARD_SIZE):
            start = (MARGIN, MARGIN + i * CELL_SIZE)
            end = (MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, MARGIN + i * CELL_SIZE)
            pygame.draw.line(screen, LINE_COLOR, start, end, LINE_WIDTH)

            start = (MARGIN + i * CELL_SIZE, MARGIN)
            end = (MARGIN + i * CELL_SIZE, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE)
            pygame.draw.line(screen, LINE_COLOR, start, end, LINE_WIDTH)

        for pos in STAR_POS:
            row, col = pos
            x = MARGIN + CELL_SIZE * col
            y = MARGIN + CELL_SIZE * row
            if (BOARD_SIZE == 19 and pos != (9, 9)) or (BOARD_SIZE == 15 and pos != (7, 7)):
                pygame.draw.circle(screen, STAR_COLOR, (x, y), STAR_RADIUS)
            else:
                pygame.draw.circle(screen, STAR_COLOR, (x, y), TENGEN_RADIUS)

    def draw_stones(self, screen):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row, col] == 0:
                    continue
                color = BLACK_STONE_COLOR if self.board[row, col] == 1 else WHITE_STONE_COLOR
                x = MARGIN + CELL_SIZE * col
                y = MARGIN + CELL_SIZE * row
                pygame.draw.circle(screen, color, (x, y), STONE_RADIUS)

    def show_message(self, screen, player):
        font = pygame.font.Font(None, 32)
        if player is not None:
            text = font.render('{} wins'.format(player.name), True, (255, 0, 0))
        else:
            text = font.render('draw', True, (255, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (WINDOW_WIDTH / 2, WINDOW_HEIGHT - MARGIN / 2)
        screen.blit(text, text_rect)

def player_vs_ai(screen, board, ai):
    win = False
    winner = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and not win:
                    mouse_pos = event.pos
                    board_pos = board.get_board_pos(mouse_pos)
                    if board_pos is not None:
                        win = board.update_board(board_pos, BLACK_PLAYER.stone)
                        board.draw_stones(screen)
                        pygame.display.flip()
                        if win:
                            winner = BLACK_PLAYER
                            break
                    else:
                        break

                    if not np.any(board.get_board() == 0):
                        win = True
                        winner = None
                        continue
                    
                    pygame.time.wait(np.random.randint(100, 200))

                    board_pos = ai.make_move(board.get_board())
                    if board_pos is not None:
                        win = board.update_board(board_pos, WHITE_PLAYER.stone)
                        board.draw_stones(screen)
                        if win:
                            winner = WHITE_PLAYER
                            break

                    if not np.any(board.get_board() == 0):
                        win = True
                        winner = None
                        board.show_message(screen, None)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and win:
                    win = False
                    winner = None
                    board.reset()
                    board.draw_board(screen)
            elif event.type == pygame.QUIT:
                sys.exit()

        if win:
            board.show_message(screen, winner)
        pygame.display.flip()

def ai_vs_ai(screen, board, black, white):
    win = False
    winner = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and win:
                    win = False
                    winner = None
                    board.reset()
                    board.draw_board(screen)
            elif event.type == pygame.QUIT:
                sys.exit()

        if win:
            board.show_message(screen, winner)
        pygame.display.flip()

        if win:
            pygame.time.wait(50)
            continue

        board_pos = black.make_move(board.get_board())
        if board_pos is not None:
            win = board.update_board(board_pos, BLACK_PLAYER.stone)
            board.draw_stones(screen)
            pygame.display.flip()
            if win:
                winner = BLACK_PLAYER
                continue

        if not np.any(board.get_board() == 0):
            win = True
            winner = None
            board.show_message(screen, None)
            continue

        pygame.time.wait(1500)

        board_pos = white.make_move(board.get_board())
        if board_pos is not None:
            win = board.update_board(board_pos, WHITE_PLAYER.stone)
            board.draw_stones(screen)
            pygame.display.flip()
            if win:
                winner = WHITE_PLAYER
                continue

        if not np.any(board.get_board() == 0):
            win = True
            winner = None
            board.show_message(screen, None)
            continue

        pygame.time.wait(1500)

def player_vs_player(screen, board):
    win = False
    winner = None
    player = BLACK_PLAYER

    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and not win:
                    mouse_pos = event.pos
                    board_pos = board.get_board_pos(mouse_pos)
                    if board_pos is not None:
                        win = board.update_board(board_pos, player.stone)
                        board.draw_stones(screen)
                        pygame.display.flip()
                        if win:
                            winner = player
                            break
                        player = WHITE_PLAYER if player == BLACK_PLAYER else BLACK_PLAYER
                    else:
                        break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and win:
                    win = False
                    winner = None
                    player = BLACK_PLAYER
                    board.reset()
                    board.draw_board(screen)
            elif event.type == pygame.QUIT:
                sys.exit()

        if win:
            board.show_message(screen, winner)
        pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('Gomoku')
    
    board = GomokuBoard()
    board.draw_board(screen)

    checkpoints = torch.load('../best_model/best_model_steps_6492_games_640540_rate_0.94.pt')
    weights = checkpoints['model_state_dict']
    ai = RLAI(WHITE_PLAYER, weights)
    # ai = RuleAI(WHITE_PLAYER)
    player_vs_ai(screen, board, ai)

    black, white = RLAI(BLACK_PLAYER, weights), RuleAI(WHITE_PLAYER)
    # black, white = RuleAI(BLACK_PLAYER), RLAI(WHITE_PLAYER, weights)
    # black, white = RLAI(BLACK_PLAYER, weights), RLAI(WHITE_PLAYER, weights)
    # black, white = RuleAI(BLACK_PLAYER), RuleAI(WHITE_PLAYER)
    ai_vs_ai(screen, board, black, white)

    # player_vs_player(screen, board)

if __name__ == '__main__':
    main()