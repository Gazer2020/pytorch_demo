import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import util
import model.net as net
import model.data_loader as data_loader
from config import parser

logger = logging.getLogger(name='test')


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    with tqdm(iterable=dataloader, desc='Eval', leave=False, position=1) as t:
        for inputs, targets in t:
            if params['cuda']:
                inputs, targets = inputs.to('cuda'), targets.to('cuda')

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            outputs = outputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()

            summary_batch = {metric: metrics[metric](
                outputs, targets) for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                    for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logger.info("Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = Path(args.model_dir, 'params.json')
    assert json_path.is_file(), "No json configuration file found at {}".format(str(json_path))
    params = util.Params(json_path)

    params['data_dir'] = Path(args.data_dir)
    params['model_dir'] = Path(args.model_dir)
    params['res_dir'] = Path(args.res_dir)

    if args.restore_file:
        params['restore_file'] = Path(args.restore_file)
    else:
        params['restore_file'] = Path('best')

    # use GPU if available
    params['cuda'] = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params['cuda']:
        torch.cuda.manual_seed(230)

    # Get the logger
    util.set_logger(log_path=params['res_dir'] / 'evaluate.log',
                    log_name='test')
    logger.info("Start testing.")

    # Create the input data pipeline
    logger.info("Creating the dataset...")

    test_dataloader = data_loader.get_dataloader(params, train=False)

    logger.info("done")

    # Define the model
    model = net.CNN(params).cuda() if params['cuda'] else net.CNN(params)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logger.info("Starting evaluation")

    # Reload weights from the saved file
    util.load_checkpoint(
        params['res_dir'] / params['restore_file'].with_suffix('.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dataloader, metrics, params)

    logger.info("done")

    logger.info("Saving result...")

    save_path = params['res_dir'] / \
        "metrics_test_{}.json".format(str(params['restore_file']))
    util.save_dict_to_json(test_metrics, save_path)

    logger.info("done")

    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"The {k} is {(v*100):05.2f}.")
        else:
            print(f"The {k} is {v}.")
