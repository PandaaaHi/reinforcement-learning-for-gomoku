## Reinforcement-Learning-for-Gomoku
This is an implementation of reinforcement learning algorithm (PPO) for playing Gomoku. To accelerate the training process, we adopt [ray](https://github.com/ray-project/ray) to scale our application and introduce asynchronous training to decouple data production and data consumption. We also implement a rule AI as baseline to choose the best model.

### Requirements
To train the AI model from scratch, need:
- numpy=2.4.6
- pygame=2.6.1
- ray=2.55.1
- torch=2.12.0+cu126

### Getting Started
To train the AI model, run the following script from the ``src`` directory:
```
python train.py
```
You may modify config.py to try different training parameters.

To play with provided models, directly run:
```
python play_with_ai.py
```