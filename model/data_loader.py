from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

train_transformer = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor


def get_dataloader(params, train):
    if not params['data_dir'].exists():
        params['data_dir'].mkdir(parents=True, exist_ok=True)
    if params['cuda']:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
    if train:
        train_kwargs = {
            'batch_size': params['batch_size'],
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        dataset = datasets.MNIST(str(params['data_dir']),
                                 train=True,
                                 transform=train_transformer,
                                 download=True)
        data_loader = DataLoader(dataset, **train_kwargs)
    else:
        test_kwargs = {
            'batch_size': params['batch_size'],
            'shuffle': False
        }
        test_kwargs.update(cuda_kwargs)
        dataset = datasets.MNIST(str(params['data_dir']),
                                 train=False,
                                 transform=eval_transformer,
                                 download=True)
        data_loader = DataLoader(dataset, **test_kwargs)

    return data_loader
