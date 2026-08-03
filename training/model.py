import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(x))
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        return x + residual
    
class GomokuNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_input = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, NUM_FILTERS, KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU()
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(NUM_FILTERS) for _ in range(NUM_RESIDUAL_BLOCKS)]
        )

        self.policy_conv = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        self.value_conv = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.value_fc1 = nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv_input(x)
        x = self.residual_blocks(x)

        policy = self.policy_conv(x)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        value = self.value_conv(x)
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = self.value_fc2(value)

        return policy, value
    
    def get_action_probs(self, state, valid_moves=None):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            policy_logits, value = self.forward(state_tensor)

            if valid_moves is not None and len(valid_moves) > 0:
                mask = torch.ones_like(policy_logits) * float('-inf')
                mask[0, valid_moves] = policy_logits[0, valid_moves]
                probs = F.softmax(mask, dim=-1)
                return probs.squeeze(0), value.item()
            else:
                return None, value.item()