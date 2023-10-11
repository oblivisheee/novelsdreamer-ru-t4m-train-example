import os
import json

def transformer_config():
    path_to_config_transformer = './transformer_config.json'
    config = {
        "num_layers": 4,
        "d_model": 128,
        "dff": 512,
        "num_heads": 8,
        "dropout_rate": 0.1,
        "input_vocab_size": 8500,
        "target_vocab_size": 8000,
        "maximum_position_encoding": 10000,
        "batch_size": 32,
        "tar_seq_len": 16
    }

    os.makedirs(os.path.dirname(path_to_config_transformer), exist_ok=True)

    if not os.path.isfile(path_to_config_transformer):
        with open(path_to_config_transformer, 'w') as f:
            json.dump(config, f)

    try:
        with open(path_to_config_transformer, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        with open(path_to_config_transformer, 'w') as f:
            json.dump(config, f)
    return config

def trainer_config():
    path_to_config_of_train = './train_config.json'
    config = {
        "learning_rate":  0.001,
        "loss_reduction": 'auto',
        "epoch_train": 30, 
    }

    os.makedirs(os.path.dirname(path_to_config_of_train), exist_ok=True)

    if not os.path.isfile(path_to_config_of_train):
        with open(path_to_config_of_train, 'w') as f:
            json.dump(config, f)

    try:
        with open(path_to_config_of_train, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        with open(path_to_config_of_train, 'w') as f:
            json.dump(config, f)
    return config

trainer_config()
