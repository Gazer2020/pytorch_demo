"""
    this file set the directories:
    
        data_dir
        
        model_dir
        
        res_dir
        
        restore_file
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='dataset/mnist',
                    help='Directory containing the dataset')
parser.add_argument('--model_dir', default='model/mnist',
                    help='Directory containing the model')
parser.add_argument('--res_dir', default='res/mnist',
                    help='Directory containing the result')
parser.add_argument('--restore_file', default='',
                    help="File to restore the checkpoint if you want")

if __name__ == '__main__':
    pass
