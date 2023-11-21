import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import util
import model.net as net
import model.data_loader as data_loader
from config import parser
from eval import evaluate


logger = logging.getLogger(name='train')


def train_one_epoch(model, loss_fn, optimizer, dataloader, metrics, params):
    """Train the model one epoch

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        optimizer: (torch.optim) optimizer for parameters of model
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    loss_avg = util.RunningAverage()

    summ = []

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (inputs, targets) in enumerate(dataloader):
            if params['cuda']:
                inputs, targets = inputs.to('cuda'), targets.to('cuda')

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params['save_summary_steps'] == 0:
                outputs = outputs.data.cpu().numpy()
                targets = targets.data.cpu().numpy()

                summary_batch = {metric: func(outputs, targets)
                                 for metric, func in metrics.items()}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss=f'{loss_avg():05.3f}')
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]for x in summ])
                    for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logger.info("- Train metrics: " + metrics_string)


def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """
    # reload weights from restore_file if specified
    if 'restore_file' in params:
        restore_path = params['res_dir'] / \
            params['restore_file'].with_suffix('.pth.tar')
        logger.info("Restoring parameters from {}".format(str(restore_path)))
        util.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params['num_epochs']):
        logger.info("Epoch {}/{}".format(epoch + 1, params['num_epochs']))

        train_one_epoch(model, loss_fn, optimizer,
                        train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        util.save_checkpoint({'epoch': epoch + 1,
                              'state_dict': model.state_dict(),
                              'optim_dict': optimizer.state_dict()},
                             is_best=is_best,
                             checkpoint_dir=params['res_dir'])

        # If best_eval, best_save_path
        if is_best:
            logger.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = params['res_dir'] / "metrics_val_best.json"
            util.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = params['res_dir'] / "metrics_val_last.json"
        util.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':
    args = parser.parse_args()
    json_path = Path(args.model_dir, 'params.json')

    params = util.Params(json_path)

    params['data_dir'] = Path(args.data_dir)
    params['model_dir'] = Path(args.model_dir)
    params['res_dir'] = Path(args.res_dir)

    # use GPU if available
    params['cuda'] = torch.cuda.is_available()

    # if need restore file
    if args.restore_file:
        params['restore_file'] = Path(args.restore_file)

    # Set the random seed
    torch.manual_seed(2023)
    if params['cuda']:
        torch.cuda.manual_seed(2023)

    # Set the logger
    util.set_logger(params['res_dir'] / 'train.log')

    # Create the input data pipeline
    logger.info("Loading the datasets...")

    train_dataloader = data_loader.get_dataloader(params, train=True)
    test_dataloader = data_loader.get_dataloader(params, train=False)

    logger.info("- done.")

    # Define the model and optimizer
    model = net.CNN(params).cuda() if params['cuda'] else net.CNN(params)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # show the hyperparameters
    util.show_dict(params)

    # Train the model
    logger.info("Starting training for {} epoch(s)".format(
        params['num_epochs']))

    train(model, train_dataloader, test_dataloader, optimizer,
          loss_fn, metrics, params)

    logger.info("- train over")
