import torch
import numpy as np
from enum import Enum

from model import GomokuNet
from env import GomokuEnv
from config import *

class Space(Enum):
    NONE = 0
    END_EMPTY = 1
    CONSECUTIVE_EMPTY = 2

class RuleAI:
    def __init__(self, player):
        self.myself = player
        self.opponent = BLACK_PLAYER if player.name == 'white' else WHITE_PLAYER

    def make_move(self, board):
        if np.all(board == 0):
            board_pos = POS(BOARD_SIZE // 2, BOARD_SIZE // 2)
            return board_pos
        
        max_score = 0
        board_pos = None

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row, col] != 0:
                    continue
                score = 0
                for offset in OFFSET:
                    score += self.cal_score(board, POS(row, col), offset[0], offset[1])
                if score > max_score:
                    max_score = score
                    board_pos = POS(row, col)
                elif score == max_score:
                    if board_pos is None:
                        board_pos = POS(row, col)
                    elif np.random.uniform() < 0.5:
                        board_pos = POS(row, col)
        return board_pos
    
    def cal_score(self, board, pos, offset_row, offset_col):
        args = {
            'consecutive_count': 0,
            'consecutive_count_': 0,
            'space': Space.NONE,
            'space_': Space.NONE,
            'block_count': 0,
            'block_count_': 0,
        }

        row = pos.row + offset_row
        col = pos.col + offset_col
        same_color = self.is_same_color_with_neighbour(board, row, col)
        if same_color is not None:
            self.count(board, pos, offset_row, offset_col, same_color, args)

        if args['space'] is Space.END_EMPTY:
            args['space'] = Space.NONE
        if args['space_'] is Space.END_EMPTY:
            args['space_'] = Space.NONE

        row = pos.row - offset_row
        col = pos.col - offset_col
        same_color = self.is_same_color_with_neighbour(board, row, col)
        if same_color is not None:
            self.count(board, pos, -offset_row, -offset_col, same_color, args)

        score = 0
        if args['consecutive_count'] >= 4:
            score = 10000
        elif args['consecutive_count_'] >= 4:
            score = 9000
        elif args['consecutive_count'] == 3:
            if args['block_count'] == 0:
                score = 1000
            elif args['block_count'] == 1:
                score = 100
        elif args['consecutive_count_'] == 3:
            if args['block_count_'] == 0:
                score = 900
            elif args['block_count_'] == 1:
                score = 90
        elif args['consecutive_count'] == 2:
            if args['block_count'] == 0:
                score = 100
            elif args['block_count'] == 1:
                score = 10
        elif args['consecutive_count_'] == 2:
            if args['block_count_'] == 0:
                score = 90
            elif args['block_count_'] == 1:
                score = 9
        elif args['consecutive_count'] == 1:
            score = 10
        elif args['consecutive_count_'] == 1:
            score = 9

        if args['space'] or args['space_']:
            score /= 2

        return score

    def is_same_color_with_neighbour(self, board, row, col):
        if row >= 0 and row < BOARD_SIZE and col >= 0 and col < BOARD_SIZE:
            if board[row, col] == self.myself.stone:
                return True
            elif board[row, col] == self.opponent.stone:
                return False
        return None

    def count(self, board, pos, offset_row, offset_col, same_color, args):
        for step in range(1, 6):
            row = pos.row + step * offset_row
            col = pos.col + step * offset_col

            if row >= 0 and row < BOARD_SIZE and col >= 0 and col < BOARD_SIZE:
                if same_color:
                    if board[row, col] == self.myself.stone:
                        args['consecutive_count'] += 1
                        if args['space'] == Space.END_EMPTY:
                            args['space'] = Space.CONSECUTIVE_EMPTY
                    elif board[row, col] == self.opponent.stone:
                        args['block_count'] += 1
                        break
                    else:
                        if args['space'] == Space.NONE:
                            args['space'] = Space.END_EMPTY
                        else:
                            break
                else:
                    if board[row, col] == self.opponent.stone:
                        args['consecutive_count_'] += 1
                        if args['space_'] == Space.END_EMPTY:
                            args['space_'] = Space.CONSECUTIVE_EMPTY
                    elif board[row, col] == self.myself.stone:
                        args['block_count_'] += 1
                        break
                    else:
                        if args['space_'] == Space.NONE:
                            args['space_'] = Space.END_EMPTY
                        else:
                            break
            else:
                if same_color:
                    args['block_count'] += 1
                else:
                    args['block_count_'] += 1

class RLAI:
    def __init__(self, player, weights):
        self.myself = player
        self.opponent = BLACK_PLAYER if player.name == 'white' else WHITE_PLAYER
        self.device = torch.device('cpu')
        self.model = GomokuNet().to(self.device)
        self.model.load_state_dict(weights)
        self.model.eval()
        self.env = GomokuEnv()
        self.env.cur_player = self.myself.stone
    
    def select_action(self, state, valid_moves):
        probs, _ = self.model.get_action_probs(state, valid_moves)
        return valid_moves[probs[valid_moves].argmax().item()]

    def make_move(self, board):
        self.env.board = board
        valid_moves = self.env.get_valid_moves()

        _, cur_pattern_state = self.env.potential.cal_pattern_count(self.env.board, self.myself.stone)
        _, opp_pattern_state = self.env.potential.cal_pattern_count(self.env.board, self.opponent.stone)
        
        # self.env.state[3:6] = cur_pattern_state
        # self.env.state[6:] = opp_pattern_state

        # self.env.state[3:7] = cur_pattern_state
        # self.env.state[7] = opp_pattern_state[1]
        # self.env.state[8] = opp_pattern_state[3]

        self.env.state[3:8] = cur_pattern_state
        self.env.state[8] = opp_pattern_state[0]
        self.env.state[9] = opp_pattern_state[1]
        self.env.state[10] = opp_pattern_state[2]
        self.env.state[11] = opp_pattern_state[4]

        state = self.env.get_state()
        action = self.select_action(state, valid_moves)

        row, col = divmod(action, BOARD_SIZE)
        if board[row, col] != 0:
            return None
        
        return POS(row, col)