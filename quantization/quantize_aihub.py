'''
Apply ai-hub for quantization after we have determined the precision.
'''

import torch
import sys
import os
import qai_hub as hub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training'))

from training.model import GomokuNet
from training.config import *

'''
According to https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html, these precisions are supported for different runtimes:
- tflite: w8a8
- qnn: w8a8, w8a16
- onnx: w8a8, w8a16
'''

'''
Method 1: quantization using quantsim results (an exported onnx model, an encodings JSON file containing quantization parameters)
'''
format = 'w8a16'
model_name = format + '.aimet'
model_path = os.path.join('outputs_quantsim', model_name)

device = hub.Device('Samsung Galaxy S25 (Family)')

client = hub.Client()
# compile_job = client.submit_compile_job(
#     model=model_path,
#     device=device,
#     options='--target_runtime tflite',
# )
# compile_job.download_target_model(os.path.join('outputs_aihub', format))

compile_job = client.submit_compile_job(
    model=model_path,
    device=device,
    options='--target_runtime qnn_dlc',
)
compile_job.download_target_model(os.path.join('outputs_aihub', format))


'''
Method 2: quantization from scratch (need to prepare calibration data); torch -> exported torch -> onnx -> quantized_onnx -> tflite/dlc
'''
# checkpoints = torch.load('../training/best_model/best_model_steps_6492_games_640540_rate_0.94.pt')
# weights = checkpoints['model_state_dict']

# torch_model = GomokuNet().to(torch.device('cpu'))
# torch_model.load_state_dict(weights)
# torch_model.eval()

# # export torch model
# input_shape = (1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
# example_input = torch.rand(input_shape)
# with torch.no_grad():
#     exported_torch_model = torch.export.export(torch_model, (example_input,))

# # compile torch model to onnx
# device = hub.Device('Samsung Galaxy S25 (Family)')

# compile_job = hub.submit_compile_job(
#     model=exported_torch_model,
#     device=device,
#     input_specs=dict(state=input_shape),
#     options='--target_runtime onnx'
# )
# unquantized_onnx_model = compile_job.get_target_model()

# # quantize
# import numpy as np

# states = np.load('./data/states.npy') # (1024*8, 12, 11, 11)
# np.random.seed(1)
# np.random.shuffle(states)

# num_calibration = 1024
# sample_inputs = []
# for state in states[:num_calibration]:
#     state = np.expand_dims(state, axis=0)
#     sample_inputs.append(state)
# calibration_data = dict(state=sample_inputs)

# quantize_job = hub.submit_quantize_job(
#     model=unquantized_onnx_model,
#     calibration_data=calibration_data,
#     weights_dtype=hub.QuantizeDtype.INT8,
#     activations_dtype=hub.QuantizeDtype.INT8
# )
# quantized_onnx_model = quantize_job.get_target_model()

# format = 'w8a16'
# # compile_job = hub.submit_compile_job(
# #     model=quantized_onnx_model,
# #     device=device,
# #     input_specs=dict(state=input_shape),
# #     options='--target_runtime tflite'
# # )
# # compile_job.download_target_model(os.path.join('outputs_aihub', format))

# compile_job = hub.submit_compile_job(
#     model=quantized_onnx_model,
#     device=device,
#     options='--target_runtime qnn_dlc'
# )
# compile_job.download_target_model(os.path.join('outputs_aihub', format))