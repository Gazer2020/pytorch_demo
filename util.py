import json
import logging
from pathlib import Path
import shutil

import torch


class Params(dict):
    """
    Succeed from dict.

    Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params['learning_rate'])
    params['learning_rate'] = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        """
        Args:
            json_path (Path): the path of the json file
        """
        with open(json_path) as f:
            params = json.load(f)
            self.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self, f, indent=4)


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)


def set_logger(log_path: Path, log_name='train'):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (Path) where to log
    """
    logger = logging.getLogger(name=log_name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Logging to a file
        if not log_path.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch()
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)-8s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)-8s - %(message)s', datefmt='%m-%d %H:%M:%S')
        stream_handler.setFormatter(console_formatter)
        logger.addHandler(stream_handler)


def save_dict_to_json(d: dict, json_path: Path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (Path) path of the json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint_dir: Path):
    """Saves model and training parameters at checkpoint_dir + 'last.pth.tar'. If is_best==True, also saves
    checkpoint_dir + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint_dir: (Path) folder where parameters are to be saved
    """
    filepath = checkpoint_dir / 'last.pth.tar'
    if not checkpoint_dir.is_dir():
        print("Checkpoint Directory does not exist! Making directory {}".format(
            str(checkpoint_dir)))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, checkpoint_dir / 'best.pth.tar')


def load_checkpoint(checkpoint: Path, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (Path) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "File doesn't exist {}".format(str(checkpoint)))

    checkpoint = torch.load(checkpoint)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])


def show_dict(d: dict):
    print('#'*31 + ' Hyperparameters ' + '#'*32)
    for k, v in d.items():
        print(f'{k:>38} | {str(v):<39}')
    print('#'*80)
