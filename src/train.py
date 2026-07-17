import numpy as np
import ray
import time
import threading
import os
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import deque

from model import GomokuNet
from ai import RuleAI
from env import GomokuEnv
from config import *

np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'
    
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.next_values = []
        self.players = []

    def store(self, state, action, prob, reward, value, done, value_, player):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.next_values.append(value_)
        self.players.append(player)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.next_values.clear()
        self.players.clear()

    def get_data(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.dones),
            np.array(self.next_values),
            np.array(self.players)
        )
    
    def merge(self, other):
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.probs.extend(other.probs)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.dones.extend(other.dones)
        self.next_values.extend(other.next_values)
        self.players.extend(other.players)

class PPO:
    def __init__(self, model):
        self.model = model
        self.optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    def update(self, memory, num_shuffle):
        self.model.train()
        device = next(self.model.parameters()).device

        states, actions, old_probs, advantages, returns = memory

        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        batch_size = len(states)
        for _ in range(num_shuffle):
            perm = torch.randperm(batch_size, device=device)
            states, actions, old_probs, advantages, returns = states[perm], actions[perm], old_probs[perm], advantages[perm], returns[perm]

            policy_logits, values_pred = self.model(states)
            values_pred = values_pred.squeeze(1)

            occupied = (states[:, 0] != 0) | (states[:, 1] != 0)
            invalid_mask = occupied.view(occupied.size(0), -1)
            masked_logits = policy_logits.clone()
            masked_logits[invalid_mask] = float('-inf')

            log_probs = F.log_softmax(masked_logits, dim=-1)
            probs = log_probs.exp()
            current_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()

            ratio = current_probs / (old_probs + 1e-8)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_pred, returns)

            entropy = -(probs * log_probs.masked_fill(invalid_mask, 0.0)).sum(dim=-1).mean()

            total_loss = policy_loss + C1 * value_loss - C2 * entropy

            self.optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optim.step()

        return policy_loss.item(), C1 * value_loss.item(), C2 * entropy.item()

@ray.remote
class ExperienceQueue:
    def __init__(self):
        self.chunks = deque()
        self.buffer_size = 0
        self.num_data_type = 0
        self.flag = False
        self.stat = {
            'total_games': 0,
            'winner_attack': 0,
            'winner_defense': 0,
            'loser_attack': 0,
            'loser_defense': 0,
            'moves': 0
        }

    def push(self, exp_data, stat):
        self.chunks.append(exp_data)
        self.buffer_size += len(exp_data[0])
        self.num_data_type = len(exp_data)

        self.stat['total_games'] += stat['num_games']
        self.stat['winner_attack'] = stat['winner_attack'] / stat['moves']
        self.stat['winner_defense'] = stat['winner_defense'] / stat['moves']
        self.stat['loser_attack'] = stat['loser_attack'] / stat['moves']
        self.stat['loser_defense'] = stat['loser_defense'] / stat['moves']
        self.stat['moves'] = stat['moves'] / stat['num_games']
    
    def pop(self, n):
        collected = []
        need = n

        while need > 0:
            chunk = self.chunks[0]
            chunk_size = len(chunk[0])

            if chunk_size <= need:
                collected.append(self.chunks.popleft())
                need -= chunk_size
                self.buffer_size -= chunk_size
            else:
                front = tuple(arr[:need] for arr in chunk)
                back = tuple(arr[need:] for arr in chunk)
                collected.append(front)
                self.chunks[0] = back
                self.buffer_size -= need
                need = 0

        result = tuple(
            np.concatenate([c[i] for c in collected], axis=0)
            for i in range(self.num_data_type)
        )

        return result, self.stat
    
    def size(self):
        return self.buffer_size
    
    def get_total_games(self):
        return self.stat['total_games']

    def stop(self):
        self.flag = True

    def is_stopped(self):
        return self.flag
    
@ray.remote
class ParameterServer:
    def __init__(self):
        self.weights = None

    def update(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights
    
@ray.remote
class AsyncMultiEnvRemoteWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.device = torch.device('cpu')
        self.envs = [GomokuEnv() for _ in range(NUM_GAMES_PER_WORKER)]
        self.model = GomokuNet().to(self.device)
        self.executor = ThreadPoolExecutor(max_workers=NUM_GAMES_PER_WORKER)

    def play_forever(self, param_server_handle, queue_handle):
        while True:
            if ray.get(queue_handle.is_stopped.remote()):
                print('worker %d stopped' % self.worker_id)
                break

            if ray.get(queue_handle.size.remote()) > MAX_BUFFER_SIZE:
                print('worker %d: experience queue exceeds the limit size %d' % (self.worker_id, MAX_BUFFER_SIZE))
                while ray.get(queue_handle.size.remote()) > MAX_BUFFER_SIZE * 0.6:
                    time.sleep(0.1)

            weights = ray.get(param_server_handle.get_weights.remote())
            if weights is not None:
                self.model.load_state_dict(weights)

            progress = ray.get(queue_handle.get_total_games.remote()) / TOTAL_GAMES
            temperature = max(0.1, 1.0 - progress)
            exp_data, stat = self.self_play_games(temperature)
            queue_handle.push.remote(exp_data, stat)

    def get_transforms(self):
        transforms = []

        def action_to_coord(action):
            return divmod(action, BOARD_SIZE)
        
        def coord_to_action(row, col):
            return row * BOARD_SIZE + col

        def t1(state, action):
            return state.copy(), action
        transforms.append(t1)

        def t2(state, action):
            row, col = action_to_coord(action)
            new_state = np.rot90(state, k=-1, axes=(1,2))
            new_action = coord_to_action(col, BOARD_SIZE - 1 - row)
            return new_state, new_action
        transforms.append(t2)
        
        def t3(state, action):
            row, col = action_to_coord(action)
            new_state = np.rot90(state, k=-2, axes=(1,2))
            new_action = coord_to_action(BOARD_SIZE - 1 - row, BOARD_SIZE - 1 - col)
            return new_state, new_action
        transforms.append(t3)

        def t4(state, action):
            row, col = action_to_coord(action)
            new_state = np.rot90(state, k=-3, axes=(1,2))
            new_action = coord_to_action(BOARD_SIZE - 1 - col, row)
            return new_state, new_action
        transforms.append(t4)

        def t5(state, action):
            row, col = action_to_coord(action)
            new_state = np.flip(state, axis=2)
            new_action = coord_to_action(row, BOARD_SIZE - 1 - col)
            return new_state, new_action
        transforms.append(t5)
        
        def t6(state, action):
            row, col = action_to_coord(action)
            new_state = np.flip(state, axis=1)
            new_action = coord_to_action(BOARD_SIZE - 1 - row, col)
            return new_state, new_action
        transforms.append(t6)

        def t7(state, action):
            row, col = action_to_coord(action)
            new_state = np.transpose(state, axes=(0,2,1))
            new_action = coord_to_action(col, row)
            return new_state, new_action
        transforms.append(t7)

        def t8(state, action):
            row, col = action_to_coord(action)
            new_state = np.rot90(np.transpose(state, axes=(0,2,1)), k=-2, axes=(1,2))
            new_action = coord_to_action(BOARD_SIZE - 1 - col, BOARD_SIZE - 1 - row)
            return new_state, new_action
        transforms.append(t8)

        return transforms
    
    def self_play_games(self, temperature: float=1.0):
        total_memory = PPOMemory()
        stat = {
            'num_games': 0,
            'num_draws': 0,
            'num_invalid': 0,
            'winner_attack': 0,
            'winner_defense': 0,
            'loser_attack': 0,
            'loser_defense': 0,
            'moves': 0
        }

        futures = [
            self.executor.submit(
                self.play_one_game,
                self.envs[i],
                self.model,
                temperature,
                stat.copy()
            )
            for i in range(NUM_GAMES_PER_WORKER)
        ]

        transforms = self.get_transforms()
        transforms = transforms[0:NUM_TRANSFORMS]

        for future in as_completed(futures):
            exp_data, game_stat = future.result()
            states, actions = exp_data[0:2]
            res_data = [list(arr) if isinstance(arr, np.ndarray) else arr for arr in exp_data[2:]]

            for transform in transforms:
                new_states, new_actions = [], []
                for state, action in zip(states, actions):
                    new_state, new_action = transform(state, action)
                    new_states.append(new_state)
                    new_actions.append(new_action)

                memory = PPOMemory()
                memory.states = new_states
                memory.actions = new_actions
                memory.probs, memory.rewards, memory.values, memory.dones, memory.next_values, memory.players = res_data
                total_memory.merge(memory)

            for k, v in game_stat.items():
                stat[k] += v

        def compute_gae(rewards, values, dones, next_values, players):
            # advantages = [0] * len(rewards)
            # last_advantage = 0
            # for t in reversed(range(len(rewards))):
            #     delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
            #     last_advantage = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_advantage
            #     advantages[t] = last_advantage
            # returns = [adv + val for adv, val in zip(advantages, values)]

            # returns = [0] * len(rewards)
            # discounted_reward = 0
            # next_player = None
            # for t in reversed(range(len(rewards))):
            #     player = players[t]
            #     if dones[t]:
            #         discounted_reward = 0
            #         next_player = None
            #     elif next_player is not None and player != next_player:
            #         discounted_reward = -GAMMA * discounted_reward
            #     else:
            #         discounted_reward = GAMMA * discounted_reward
            #     discounted_reward = rewards[t] + discounted_reward
            #     returns[t] = discounted_reward
            #     next_player = player
            # advantages = [r - v for r, v in zip(returns, values)]

            # returns = [0] * len(rewards)
            # done = False
            # winner = players[-1]
            # winner_discounted_reward = 0
            # loser_discounted_reward = 0
            # for t in reversed(range(len(rewards))):
            #     player = players[t]
            #     if dones[t]:
            #         winner = players[t]
            #         winner_discounted_reward = 0
            #         done = True
            #     elif done:
            #         loser_discounted_reward = 0
            #         done = False
            #     else:
            #         if player == winner:
            #             winner_discounted_reward = GAMMA * winner_discounted_reward
            #         else:
            #             loser_discounted_reward = GAMMA * loser_discounted_reward
                
            #     if player == winner:
            #         winner_discounted_reward = rewards[t] + winner_discounted_reward
            #         returns[t] = winner_discounted_reward
            #     else:
            #         loser_discounted_reward = rewards[t] + loser_discounted_reward
            #         returns[t] = loser_discounted_reward
            # advantages = [r - v for r, v in zip(returns, values)]

            returns = rewards
            advantages = [r - v for r, v in zip(returns, values)]

            return np.array(advantages), np.array(returns)
        
        states, actions, old_probs, rewards, values, dones, next_values, players = total_memory.get_data()
        advantages, returns = compute_gae(rewards, values, dones, next_values, players)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return (states, actions, old_probs, advantages, returns), stat

    def play_one_game(self, env, model, temperature, game_stat):
        state = env.reset()
        done = False

        local_memory = PPOMemory()
        player_potential = {1: {'attack': 0, 'defense': 0}, -1: {'attack': 0, 'defense': 0}}

        while not done:
            valid_moves = env.get_valid_moves()
            cur_player = env.cur_player

            action, prob, value = AsyncMultiEnvRemoteWorker.select_action(state, valid_moves, temperature, model)
            state_, reward, done, info = env.step(action)
            if isinstance(reward, dict):
                total_reward = sum(reward.values())
            else:
                total_reward = reward

            # if len(local_memory.next_values) >= 1:
            #     local_memory.next_values[-1] = -value

            if not done:
                if isinstance(reward, dict):
                    player_potential[cur_player]['attack'] += reward['attack']
                    player_potential[cur_player]['defense'] += reward['defense']
            else:
                if 'winner' in info:
                    game_stat['winner_attack'] = player_potential[cur_player]['attack']
                    game_stat['winner_defense'] = player_potential[cur_player]['defense']
                    game_stat['loser_attack'] = player_potential[-cur_player]['attack']
                    game_stat['loser_defense'] = player_potential[-cur_player]['defense']
                if 'draw' in info:
                    game_stat['winner_attack'] = player_potential[cur_player]['attack']
                    game_stat['winner_defense'] = player_potential[cur_player]['defense']
                    game_stat['loser_attack'] = player_potential[-cur_player]['attack']
                    game_stat['loser_defense'] = player_potential[-cur_player]['defense']
                    game_stat['num_draws'] = 1
                if 'error' in info:
                    game_stat['num_invalid'] = 1
                game_stat['num_games'] = 1
                game_stat['moves'] = env.move_count

            local_memory.store(state, action, prob, total_reward, value, done, 0, cur_player)
            state = state_

        return local_memory.get_data(), game_stat
    
    @staticmethod
    def select_action(state, valid_moves, temperature, model):
        probs, value = model.get_action_probs(state, valid_moves)        
        probs = probs.cpu().numpy()
        scaled_probs = probs[valid_moves] ** (1.0 / max(0.1, temperature))
        scaled_probs = scaled_probs / scaled_probs.sum()
        idx = np.random.choice(len(valid_moves), p=scaled_probs)
        action = valid_moves[idx]
        action_prob = probs[action].item()

        return action, action_prob, value

@ray.remote(num_gpus=NUM_GPUS)
class RemoteTrainer:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GomokuNet().to(self.device)
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'loaded pretrained weights from {checkpoint_path}')
        self.ppo = PPO(self.model)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
    
    def train_step(self, all_experiences, num_shuffle):
        policy_loss, value_loss, entropy = self.ppo.update(all_experiences, num_shuffle)
        return policy_loss, value_loss, entropy

def evaluate_on_cpu(weights):
    device = torch.device('cpu')
    model = GomokuNet().to(device)
    model.load_state_dict(weights)
    model.eval()
    env = GomokuEnv()

    def select_action(state, valid_moves):
        probs, _ = model.get_action_probs(state, valid_moves)
        return valid_moves[probs[valid_moves].argmax().item()]

    wins, losses, draws = 0, 0, 0
    for _ in range(NUM_EVALUATIONS):
        state = env.reset()
        done = False
        rl_ai_black = np.random.random() < 0.5

        if rl_ai_black:
            rule_ai = RuleAI(WHITE_PLAYER)
        else:
            rule_ai = RuleAI(BLACK_PLAYER)

        while not done:
            cur_player = env.cur_player
            if (cur_player == 1 and rl_ai_black) or (cur_player == -1 and not rl_ai_black):
                valid_moves = env.get_valid_moves()
                action = select_action(state, valid_moves)
            else:
                board_pos = rule_ai.make_move(env.board)
                action = board_pos.row * BOARD_SIZE + board_pos.col

            state, reward, done, info = env.step(action)

            if done:
                if 'winner' in info:
                    if (info['winner'] == 1 and rl_ai_black) or (info['winner'] == -1 and not rl_ai_black):
                        wins += 1
                    else:
                        losses += 1
                elif 'draw' in info:
                    draws += 1
                else:
                    print('error occurs')
    
    win_rate = wins / NUM_EVALUATIONS
    
    return win_rate, wins, losses, draws

def main():
    now = datetime.now().replace(microsecond=0).strftime("%Y-%m-%d-%H-%M-%S")
    saved_path = os.path.join(SAVE_PATH, now)
    os.makedirs(saved_path, exist_ok=True)

    ray.init(ignore_reinit_error=True)

    checkpoint_path = None
    trainer = RemoteTrainer.options(num_gpus=NUM_GPUS).remote(checkpoint_path)
    param_server = ParameterServer.remote()
    queue = ExperienceQueue.remote()

    init_weights = ray.get(trainer.get_weights.remote())
    ray.get(param_server.update.remote(init_weights))

    worker_futures = []
    for i in range(NUM_WORKERS):
        worker = AsyncMultiEnvRemoteWorker.remote(i)
        future = worker.play_forever.remote(param_server, queue)
        worker_futures.append(future)

    print('put some data into buffer...')
    buffer_size = ray.get(queue.size.remote())
    while buffer_size < INIT_BUFFER_SIZE:
        print('\rbuffer size: %d' % buffer_size, end='')
        time.sleep(1)
        buffer_size = buffer_size = ray.get(queue.size.remote())
    print()

    eval_lock = threading.Lock()
    eval_result = None
    def async_evaluate():
        nonlocal eval_result
        try:
            weights = ray.get(param_server.get_weights.remote())
            win_rate, wins, losses, draws = evaluate_on_cpu(weights)
            with eval_lock:
                eval_result = (win_rate, wins, losses, draws)
        except Exception as e:
            print('evalute error: ', e)

    steps = 0
    max_win_rate = 0
    stop_training = False
    batch_size = BATCH_SIZE

    count_add = 0
    count_sub = 0
    count_change = 10
    buffer_size = 0
    num_shuffle = NUM_SHUFFLE

    while True:
        cur_buffer_size = ray.get(queue.size.remote())
        if cur_buffer_size > buffer_size:
            count_add += 1
        else:
            count_add = 0

        if cur_buffer_size < buffer_size:
            count_sub += 1
        else:
            count_sub = 0

        buffer_size = cur_buffer_size

        if count_add >= count_change or buffer_size > MAX_BUFFER_SIZE:
            if num_shuffle > 1:
                tmp = num_shuffle
                num_shuffle = max(1, num_shuffle - 1)
                count_add = 0
                print(f'decrease num_shuffle from {tmp} to {num_shuffle}')
            elif batch_size < MAX_BATCH_SIZE:
                tmp = batch_size
                batch_size = min(MAX_BATCH_SIZE, batch_size * 2)
                count_add = 0
                print(f'increase batch_size from {tmp} to {batch_size}')
        elif count_sub >= count_change or buffer_size < batch_size:
            if num_shuffle < NUM_SHUFFLE:
                tmp = num_shuffle
                num_shuffle = min(NUM_SHUFFLE, num_shuffle + 1)
                count_sub = 0
                print(f'increase num_shuffle from {tmp} to {num_shuffle}')
            elif batch_size > MIN_BATCH_SIZE:
                tmp = batch_size
                batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
                count_sub = 0
                print(f'decrease batch_size from {tmp} to {batch_size}')

        while buffer_size < batch_size and not stop_training:
            print('buffer size %d less than %d' % (buffer_size, batch_size))
            time.sleep(1)
            buffer_size = ray.get(queue.size.remote())

        if stop_training and buffer_size < batch_size:
            print('stop training...')
            break

        exp_list, stats = ray.get(queue.pop.remote(batch_size))

        if stats['total_games'] > TOTAL_GAMES and not stop_training:
            stop_training = True
            ray.get(queue.stop.remote())

        policy_loss, value_loss, entropy = ray.get(trainer.train_step.remote(exp_list, num_shuffle))
        steps += 1

        latest_weights = ray.get(trainer.get_weights.remote())
        param_server.update.remote(latest_weights)

        stats_str = ', '.join(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' for k, v in stats.items())
        print(f'steps: {steps}, shuffle: {num_shuffle}, batch: {batch_size}, '
              f'buffer: {buffer_size}, '
              f'policy: {policy_loss:.4f}, value: {value_loss:.4f}, entropy: {entropy:.4f} '
              f'{stats_str}'
        )
        
        if steps % EVALUATE_STEPS == 0:
            eval_thread = threading.Thread(target=async_evaluate, daemon=True)
            eval_thread.start()

        with eval_lock:
            if eval_result is not None:
                win_rate, wins, losses, draws = eval_result
                print('rl vs rule, win_rate: %f, wins: %d, losses: %d, draws: %d' % (win_rate, wins, losses, draws))
                eval_result = None

                if win_rate > max_win_rate:
                    max_win_rate = win_rate
                    filename = os.path.join(saved_path, f'best_model_steps_{steps}_games_{stats['total_games']}_rate_{round(win_rate, 4)}.pt')
                    saved_weights = ray.get(param_server.get_weights.remote())
                    checkpoint = {
                        'model_state_dict': saved_weights,
                        'steps': steps,
                        'total_games': stats['total_games']
                    }
                    torch.save(checkpoint, filename)
                    print('checkpoint saved to {}'.format(filename))

        if steps % SAVE_STEPS == 0:
            filename = os.path.join(saved_path, f'model_steps_{steps}_games_{stats['total_games']}.pt')
            saved_weights = ray.get(param_server.get_weights.remote())
            checkpoint = {
                'model_state_dict': saved_weights,
                'steps': steps,
                'total_games': stats['total_games']
            }
            torch.save(checkpoint, filename)
            print('checkpoint saved to {}'.format(filename))

    print('now stop all workers...')
    ray.get(queue.stop.remote())

    try:
        ray.get(worker_futures, timeout=30)
        print('all workers stopped')
    except Exception:
        print('some workers not stopped in time')

    ray.shutdown()

if __name__ == '__main__':
    main()