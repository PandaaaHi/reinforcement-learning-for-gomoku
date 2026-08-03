import torch
import sys
import os
import qai_hub as hub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training'))

from training.model import GomokuNet
from training.config import *

checkpoints = torch.load('../training/best_model/best_model_steps_6492_games_640540_rate_0.94.pt')
weights = checkpoints['model_state_dict']

torch_model = GomokuNet().to(torch.device('cpu'))
torch_model.load_state_dict(weights)
torch_model.eval()

input_shape = (1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
example_input = torch.rand(input_shape)
with torch.no_grad():
    exported_torch_model = torch.export.export(torch_model, (example_input,))

device = hub.Device('Samsung Galaxy S25 (Family)')
compile_job = hub.submit_compile_job(
    model=exported_torch_model,
    device=device,
    input_specs=dict(state=input_shape),
    options='--target_runtime tflite'
)
compile_job.download_target_model(os.path.join('outputs_aihub', 'gomoku_model.tflite'))