'''
Apply quantization simulation to determine precision
'''

import torch
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training'))

from training.model import GomokuNet
from training.config import *

USE_BATCH_ONE = True # for the convenience of quantization and deployment

states = np.load('./data/states.npy') # (1024*8, 12, 11, 11)
np.random.seed(1)
np.random.shuffle(states)

num_calibration = 1024
calibration_data = states[:num_calibration]
test_data = states[num_calibration:]

# load torch model
checkpoints = torch.load('../training/best_model/best_model_steps_6492_games_640540_rate_0.94.pt')
weights = checkpoints['model_state_dict']

device = torch.device('cpu')
torch_model = GomokuNet().to(device)
torch_model.load_state_dict(weights)
torch_model.eval()

torch_test_data = torch.tensor(test_data, dtype=torch.float32, device=device)
policy_logits, _ = torch_model.forward(torch_test_data)
torch_preds = torch.argmax(policy_logits, dim=1).numpy() # (1024,)

# export torch model to onnx
import onnx
input_shape = (1, 12, 11, 11)
dummy_input = torch.randn(input_shape)
file_name = './outputs_quantsim/gomoku_model.onnx'
torch.onnx.export(
    torch_model.eval(),
    dummy_input,
    file_name,
    training=torch.onnx.TrainingMode.EVAL,
    export_params=True,
    do_constant_folding=False,
    input_names=['input'],
    output_names=['policy', 'value'],
    dynamic_axes= {} if USE_BATCH_ONE else {'input': {0: 'batch_size'}, 'policy': {0: 'batch_size'}, 'value': {0: 'batch_size'},},
    dynamo=False,
)
onnx_model = onnx.load_model(file_name)

from onnxsim import simplify
try:
    onnx_model, _ = simplify(onnx_model)
except:
    print('ONNX Simplifier failed. Proceeding with unsimplified model')

import onnxruntime as ort
providers = ['CPUExecutionProvider']
sess = ort.InferenceSession(onnx_model.SerializePartialToString(), providers=providers)

input_name = sess.get_inputs()[0].name
output_names = [output.name for output in sess.get_outputs()]

if USE_BATCH_ONE:
    onnx_preds = []
    for data in test_data:
        data = np.expand_dims(data, axis=0)
        policy_logits_onnx, _ = sess.run(output_names, {input_name: data})
        onnx_preds.append(np.argmax(policy_logits_onnx, axis=1).squeeze())
    onnx_preds = np.array(onnx_preds)
else:
    result = sess.run(output_names, {input_name: test_data})
    policy_logits_onnx, _ = result
    onnx_preds = np.argmax(policy_logits_onnx, axis=1)

matches = (onnx_preds == torch_preds)
accuracy = np.mean(matches)

print(f'onnx matched: {np.sum(matches)}/{len(torch_preds)}')
print(f'onnx accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n')

# quantization simulation
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
_ = fold_all_batch_norms_to_weight(onnx_model)

from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx import int8, int16, float16

def pass_calibration_data(session):
    input_name = session.get_inputs()[0].name
    if USE_BATCH_ONE:
        for data in calibration_data:
            data = np.expand_dims(data, axis=0)
            session.run(None, {input_name: data})
    else:
        session.run(None, {input_name: calibration_data})

precisions = [(int8, int8), (int8, int16), (int16, int16), (float16, float16)]
names = [('w8', 'a8'), ('w8', 'a16'), ('w16', 'a16'), ('fp16')]

for precision, name in zip(precisions, names):
    param, activation = precision

    sim = QuantizationSimModel(
        model=onnx_model,
        quant_scheme=QuantScheme.post_training_tf,
        param_type=param,
        activation_type=activation,
        providers=providers,
        config_file='quant_config.json' # necessary
    )

    sim.compute_encodings(forward_pass_callback=pass_calibration_data)

    quant_input_name = sim.session.get_inputs()[0].name
    quant_output_names = [output.name for output in sim.session.get_outputs()]

    if USE_BATCH_ONE:
        quant_preds = []
        for data in test_data:
            data = np.expand_dims(data, axis=0)
            quant_policy_logits, _ = sim.session.run(quant_output_names, {quant_input_name: data})
            quant_preds.append(np.argmax(quant_policy_logits, axis=1).squeeze())
        quant_preds = np.array(quant_preds)
    else:
        quant_policy_logits, _ = sim.session.run(quant_output_names, {quant_input_name: test_data})
        quant_preds = np.argmax(quant_policy_logits, axis=1)

    quant_matches = (quant_preds == torch_preds)
    quant_accuracy = np.mean(quant_matches)

    file_name = ''.join(name)
    folder_name = file_name + '.aimet'
    save_path = os.path.join('outputs_quantsim', folder_name)
    os.makedirs(save_path, exist_ok=True)

    print(f'quantiaztion format: {file_name}')
    print(f'quantization matched: {np.sum(quant_matches)}/{len(torch_preds)}')
    print(f'quantization accuracy: {quant_accuracy:.4f} ({quant_accuracy * 100:.2f}%)\n')

    sim.export(
        path=save_path,
        filename_prefix=file_name,
        export_model=True,
        export_int32_bias=True
    )

    del sim