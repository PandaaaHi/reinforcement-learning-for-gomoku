import numpy as np
from config import *

class Potential:
    def __init__(self):
        pass
    
    def draw_pattern(self, state, r, c, dr, dc, window):
        for ch in window:
            if not (r >= 0 and r < BOARD_SIZE and c >= 0 and c < BOARD_SIZE):
                break

            if ch == 'x':
                state[r, c] = 1
            
            r += dr
            c += dc
    
    def cal_pattern_count(self, board, player):
        pattern_count = {
            'five': 0,
            'open_four': 0,
            'closed_four': 0,
            'open_three': 0,
            'closed_three': 0,
            'open_two': 0,
            'closed_two': 0,
            'open_one': 0,
            'closed_one': 0,
        }

        pattern_state = np.zeros((5, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        counted_lines = set()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] != player:
                    continue

                for dr, dc in OFFSET:
                    rr, cc = r, c
                    while True:
                        pr, pc = rr - dr, cc - dc
                        if not (pr >= 0 and pr < BOARD_SIZE and pc >= 0 and pc < BOARD_SIZE):
                            break
                        rr, cc = pr, pc
                    key = (rr, cc, dr, dc)
                    if key in counted_lines:
                        continue
                
                    line = self.get_full_line(board, rr, cc, dr, dc, player)
                    if len(line) < WIN_LENGTH:
                        continue

                    counted_lines.add(key)

                    # for i in range(len(line) - WIN_LENGTH + 1):
                    #     window = line[i:i+WIN_LENGTH]
                    #     # left_extension = line[i-2:i] if i >= 2 else None
                    #     # right_extension = line[i+5:i+7] if i+5 < len(line) else None
                    #     left_extension = line[i-1] if i >= 1 else None
                    #     right_extension = line[i+5] if i+5 < len(line) else None
                    #     pattern = self.classify_window(window, left_extension, right_extension)
                    #     if pattern is not None:
                    #         pattern_count[pattern] += 1
                    #         base_r = rr + dr * i
                    #         base_c = cc + dc * i
                    #         if pattern == 'open_four' or pattern == 'closed_four':
                    #             self.draw_pattern(pattern_state[0], base_r, base_c, dr, dc, window)
                    #         elif pattern == 'open_three' or pattern == 'closed_three':
                    #             self.draw_pattern(pattern_state[1], base_r, base_c, dr, dc, window)
                    #         elif pattern == 'open_two' or pattern == 'closed_two':
                    #             self.draw_pattern(pattern_state[2], base_r, base_c, dr, dc, window)

                    tmp = line
                    for p in range(5):
                        for i in range(len(line) - WIN_LENGTH + 1):
                            window = tmp[i:i+WIN_LENGTH]
                            left_extension = line[i-1] if i >= 1 else None
                            right_extension = line[i+5] if i+5 < len(line) else None

                            pattern = None
                            if p == 0:
                                pattern = self.classify_five(window, left_extension, right_extension)
                            elif p == 1:
                                pattern = self.classify_four(window, left_extension, right_extension)
                            elif p == 2:
                                pattern = self.classify_three(window, left_extension, right_extension)
                            elif p == 3:
                                pattern = self.classify_two(window, left_extension, right_extension)
                            else:
                                pattern = self.classify_one(window, left_extension, right_extension)

                            if pattern is None:
                                continue

                            pattern_count[pattern] += 1

                            base_r = rr + dr * i
                            base_c = cc + dc * i
                            if pattern == 'open_one':
                                self.draw_pattern(pattern_state[0], base_r, base_c, dr, dc, window)
                            elif pattern == 'open_two':
                                self.draw_pattern(pattern_state[1], base_r, base_c, dr, dc, window)
                            elif pattern == 'open_three':
                                self.draw_pattern(pattern_state[2], base_r, base_c, dr, dc, window)
                            elif pattern == 'open_four':
                                self.draw_pattern(pattern_state[3], base_r, base_c, dr, dc, window)
                            elif pattern == 'closed_four':
                                self.draw_pattern(pattern_state[4], base_r, base_c, dr, dc, window)

                            tmp = list(tmp)
                            tmp[i:i+WIN_LENGTH] = ['.'] * WIN_LENGTH
                            tmp = ''.join(tmp)
                            
        return pattern_count, pattern_state
    
    def get_full_line(self, board, r, c, dr, dc, player):
        seq = []
        while r >= 0 and r < BOARD_SIZE and c >= 0 and c < BOARD_SIZE:
            val = board[r, c]
            if val == player:
                seq.append('x')
            elif val == -player:
                seq.append('o')
            else:
                seq.append('.')
            r += dr
            c += dc
        return ''.join(seq)
    
    def classify_five(self, window, left_extension, right_extension):        
        if window == 'xxxxx':
            return 'five'
        
        return None

    def classify_four(self, window, left_extension, right_extension):
        count_x = window.count('x')
        count_o = window.count('o')
        if count_x != 4 or count_o > 0:
            return None
        
        if window in ('.xxxx', 'xxxx.'):
            left_ok = self.is_side_open(window, 'left', left_extension)
            right_ok = self.is_side_open(window, 'right', right_extension)
            if left_ok and right_ok:
                return 'open_four'
            else:
                return 'closed_four'
        
        return 'closed_four'
    
    def classify_three(self, window, left_extension, right_extension):
        count_x = window.count('x')
        count_o = window.count('o')
        if count_x != 3 or count_o > 0:
            return None

        if window in ('xxx..', '.xxx.', '..xxx'):
            left_ok = self.is_side_open(window, 'left', left_extension)
            right_ok = self.is_side_open(window, 'right', right_extension)
            if left_ok and right_ok:
                return 'open_three'
            else:
                return 'closed_three'
            
        if window in ('xx.x.', 'x.xx.', '.xx.x', '.x.xx'):
            left_ok = self.is_side_open(window, 'left', left_extension)
            right_ok = self.is_side_open(window, 'right', right_extension)
            if left_ok and right_ok:
                return 'open_three'
            else:
                return 'closed_three'
            
        return 'closed_three'
    
    def classify_two(self, window, left_extension, right_extension):
        count_x = window.count('x')
        count_o = window.count('o')
        if count_x != 2 or count_o > 0:
            return None

        if window in ('xx...', '.xx..', '..xx.', '...xx'):
            left_ok = self.is_side_open(window, 'left', left_extension)
            right_ok = self.is_side_open(window, 'right', right_extension)
            if left_ok and right_ok:
                return 'open_two'
            else:
                return 'closed_two'
            
        indices = [i for i, ch in enumerate(window) if ch == 'x']
        if len(indices) == 2:
            gap = indices[1] - indices[0]
            left_ok = self.is_side_open(window, 'left', left_extension)
            right_ok = self.is_side_open(window, 'right', right_extension)
            if gap == 2:
                if left_ok and right_ok:
                    return 'open_two'
                else:
                    return 'closed_two'
                
        return None
    
    def classify_one(self, window, left_extension, right_extension):
        count_x = window.count('x')
        count_o = window.count('o')
        if count_x != 1 or count_o > 0:
            return None
        
        if window in ('x....', '.x...', '..x..', '...x.', '....x'):
            left_ok = self.is_side_open(window, 'left', left_extension)
            right_ok = self.is_side_open(window, 'right', right_extension)
            if left_ok and right_ok:
                return 'open_one'
            else:
                return 'closed_one'
            
        return None

    def classify_window(self, window, left_extension, right_extension):
        count_x = window.count('x')
        count_o = window.count('o')
        if count_o > 0:
            return None
        
        if window == 'xxxxx':
            return 'five'
        
        if count_x == 4:
            if window in ('.xxxx', 'xxxx.'):
                left_ok = self.is_side_open(window, 'left', left_extension)
                right_ok = self.is_side_open(window, 'right', right_extension)
                if left_ok and right_ok:
                    return 'open_four'
                else:
                    return 'closed_four'
            
            return 'closed_four'
            
        if count_x == 3:
            if window in ('xxx..', '.xxx.', '..xxx'):
                left_ok = self.is_side_open(window, 'left', left_extension)
                right_ok = self.is_side_open(window, 'right', right_extension)
                if left_ok and right_ok:
                    return 'open_three'
                else:
                    return 'closed_three'
                
            if window in ('xx.x.', 'x.xx.', '.xx.x', '.x.xx'):
                left_ok = self.is_side_open(window, 'left', left_extension)
                right_ok = self.is_side_open(window, 'right', right_extension)
                if left_ok and right_ok:
                    return 'open_three'
                else:
                    return 'closed_three'
                
            return 'closed_three'
        
        if count_x == 2:
            if window in ('xx...', '.xx..', '..xx.', '...xx'):
                left_ok = self.is_side_open(window, 'left', left_extension)
                right_ok = self.is_side_open(window, 'right', right_extension)
                if left_ok and right_ok:
                    return 'open_two'
                else:
                    return 'closed_two'
                
            indices = [i for i, ch in enumerate(window) if ch == 'x']
            if len(indices) == 2:
                gap = indices[1] - indices[0]
                left_ok = self.is_side_open(window, 'left', left_extension)
                right_ok = self.is_side_open(window, 'right', right_extension)
                if gap == 2:
                    if left_ok and right_ok:
                        return 'open_two'
                    else:
                        return 'closed_two'
        
        return None
    
    def is_side_open(self, window, side, extension):
        if side == 'left':
            if window[0] == '.':
                return True
            if extension is not None:
                return extension[-1] == '.'
            return False
        else:
            if window[-1] == '.':
                return True
            if extension is not None:
                return extension[0] == '.'
            return False

class GomokuEnv:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.state = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.cur_player = 1 # 1: black, -1: white
        self.win_player = 0
        self.move_count = 0
        self.potential = Potential()
        self.cached_potential = None
        self.cached_cur_pattern_count = None
        self.cached_opp_pattern_count = None

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.state = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.cur_player = 1
        self.win_player = 0
        self.move_count = 0
        self.cached_potential = None
        self.cached_cur_pattern_count = None
        self.cached_opp_pattern_count = None
        return self.get_state()
    
    def get_state(self):
        state = self.state.copy()
        state[0] = (self.board == self.cur_player).astype(np.float32) 
        state[1] = (self.board == -self.cur_player).astype(np.float32)
        state[2] = np.full((BOARD_SIZE, BOARD_SIZE), self.cur_player, dtype=np.float32)
        return state
    
    def get_valid_moves(self):
        return np.where(self.board.flatten() == 0)[0]
    
    def cal_score(self, count_pattern):
        score = 0
        for pattern, count in count_pattern.items():
            score += PATTERN_SCORES[pattern] * count
        return score
    
    def step(self, action):
        row, col = divmod(action, BOARD_SIZE)
        if self.board[row, col] != 0:
            return None, REWARD_INVALID_MOVE, True, {'error': 'invalid move'}

        old_cur_pattern_count = self.cached_cur_pattern_count if self.cached_cur_pattern_count is not None else self.potential.cal_pattern_count(self.board, self.cur_player)[0]
        old_opp_pattern_count = self.cached_opp_pattern_count if self.cached_opp_pattern_count is not None else self.potential.cal_pattern_count(self.board, -self.cur_player)[0]

        if self.move_count == 0:
            row = col = BOARD_SIZE // 2
        self.board[row, col] = self.cur_player
        self.move_count += 1

        if self.move_count == BOARD_SIZE * BOARD_SIZE and not self.check_win(POS(row, col)):
            self.win_player = 0            
            return None, REWARD_DRAW, True, {'draw': True}
        
        new_cur_pattern_count, cur_pattern_state = self.potential.cal_pattern_count(self.board, self.cur_player)
        new_opp_pattern_count, opp_pattern_state = self.potential.cal_pattern_count(self.board, -self.cur_player)

        attack_potential = self.cal_score(new_cur_pattern_count) - self.cal_score(old_cur_pattern_count)
        attack_potential *= ATTACK_POTENTIAL_FACTOR

        defense_potential = 0
        for pattern, score in DEFENSE_SCORES.items():
            reduction = old_opp_pattern_count[pattern] - new_opp_pattern_count[pattern]
            if reduction > 0:
                defense_potential += reduction * score * DEFENSE_FACTOR
        defense_potential *= DEFENSE_POTENTIAL_FACTOR

        if self.check_win(POS(row, col)):
            self.win_player = self.cur_player
            return self.get_state(), {'attack': attack_potential, 'defense': defense_potential}, True, {'winner': self.cur_player}
        
        self.cached_cur_pattern_count = new_opp_pattern_count
        self.cached_opp_pattern_count = new_cur_pattern_count
        self.cur_player *= -1

        # self.state[3:6] = opp_pattern_state
        # self.state[6:] = cur_pattern_state
        
        # self.state[3:7] = opp_pattern_state
        # self.state[7] = cur_pattern_state[1]
        # self.state[8] = cur_pattern_state[3]
        
        self.state[3:8] = opp_pattern_state
        self.state[8] = cur_pattern_state[0]
        self.state[9] = cur_pattern_state[1]
        self.state[10] = cur_pattern_state[2]
        self.state[11] = cur_pattern_state[4]

        return self.get_state(), {'attack': attack_potential, 'defense': defense_potential}, False, {}

    def check_win(self, pos):
        max_count = 0
        for offset in OFFSET:
            count = 1

            for step in range(1, 5):
                row = pos.row + step * offset[0]
                col = pos.col + step * offset[1]
                if row >= 0 and row < BOARD_SIZE and col >= 0 and col < BOARD_SIZE:
                    if self.board[row, col] == self.cur_player:
                        count += 1
                    else:
                        break
                else:
                    break

            for step in range(1, 5):
                row = pos.row - step * offset[0]
                col = pos.col - step * offset[1]
                if row >= 0 and row < BOARD_SIZE and col >= 0 and col < BOARD_SIZE:
                    if self.board[row, col] == self.cur_player:
                        count += 1
                    else:
                        break
                else:
                    break

            if count > max_count:
                max_count = count

        return max_count >= WIN_LENGTH