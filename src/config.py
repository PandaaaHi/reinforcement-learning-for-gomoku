from collections import namedtuple

# env
POS = namedtuple('Pos', 'row col')
BOARD_SIZE = 11
WIN_LENGTH = 5
OFFSET = [(0, 1), (1, 0), (1, 1), (-1, 1)]
ATTACK_POTENTIAL_FACTOR = 5e-2
DEFENSE_POTENTIAL_FACTOR = 5e-2
DEFENSE_FACTOR = 0.5

REWARD_INVALID_MOVE = 0
REWARD_WIN = 10
REWARD_LOSE = -10
REWARD_DRAW = 0

# PATTERN_SCORES = {
#     'five': 1000,
#     'open_four': 200,
#     'closed_four': 80,
#     'open_three': 30,
#     'closed_three': 10,
#     'open_two': 4,
#     'closed_two': 1,
# }

PATTERN_SCORES = {
    'five': 500,
    'open_four': 150,
    'closed_four': 60,
    'open_three': 50,
    'closed_three': 15,
    'open_two': 10,
    'closed_two': 5,
    'open_one': 3,
    'closed_one': 1,
}

DEFENSE_SCORES = {
    'closed_four': PATTERN_SCORES['five'],
    'open_three': PATTERN_SCORES['open_four'],
    # 'closed_three': PATTERN_SCORES['closed_four'],
    'open_two': PATTERN_SCORES['open_three'],
    # 'closed_two': PATTERN_SCORES['closed_three'],
    'open_one': PATTERN_SCORES['open_two'],
}

# ppo
LR = 3e-4
WEIGHT_DECAY = 1e-4
GAMMA = 0.9
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
C1 = 1
C2 = 0.01
NUM_SHUFFLE = 10

# model
NUM_CHANNELS = 12
NUM_FILTERS = 64
KERNEL_SIZE = 3
NUM_RESIDUAL_BLOCKS = 3

# ray
NUM_GPUS = 1
NUM_WORKERS = 16
NUM_GAMES_PER_WORKER = 1

# train
MIN_BATCH_SIZE = 256
MAX_BATCH_SIZE = 4096
BATCH_SIZE = 1024

MAX_BUFFER_SIZE = 500000
INIT_BUFFER_SIZE = BATCH_SIZE * 20

TOTAL_GAMES = 1000000
NUM_TRANSFORMS = 1 # min:1 max: 8

# evaluate
EVALUATE_STEPS = 10
NUM_EVALUATIONS = 100
SAVE_STEPS = 200
SAVE_PATH = '../checkpoints'

# play
CELL_SIZE = 30
MARGIN = 50
WINDOW_HEIGHT = MARGIN * 2 + (BOARD_SIZE - 1) * CELL_SIZE 
WINDOW_WIDTH = WINDOW_HEIGHT
LINE_WIDTH = 1
STONE_RADIUS = CELL_SIZE // 2 - 2
STAR_RADIUS = STONE_RADIUS // 3
TENGEN_RADIUS = STONE_RADIUS // 2

BOARD_COLOR = (238, 184, 102)
LINE_COLOR = (0, 0, 0)
BLACK_STONE_COLOR = (20, 20, 20)
WHITE_STONE_COLOR = (240, 240, 240)
STAR_COLOR = (0, 0, 0)

PLAYER = namedtuple('Player', 'name stone')
BLACK_PLAYER = PLAYER('black', 1)
WHITE_PLAYER = PLAYER('white', -1)

STAR_POS = [
    # (3, 3), (3, 11),
    # (7, 7),
    # (11, 3), (11, 11)
    (BOARD_SIZE // 2, BOARD_SIZE // 2)
]