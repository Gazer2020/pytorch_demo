# This is my pytorch demo.

This project implement a CNN to do image classification on MNIST.

### Requirements

`pytorch`, `numpy`, `tqdm`

I use python 3.10, pytorch 2.1

### How to run

Just run

```
python train.py
```

and

```
python eval.py
```

The directory config is defined in `config.py`. You can also run `python train.py --data_dir dataset/mnist` to set the config.

The hyperparameter is defined in `model/mnist/params.json`.

The result is under folder `res/`

## Acknowledgement

This is my first demo of pytorch, the project is build from a repo of [stanford-cs230](https://github.com/cs230-stanford/cs230-code-examples). Also I referred from pytorch official doc and [the example repo](https://github.com/pytorch/examples)
