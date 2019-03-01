import argparse
import numpy as np
import numpy.random
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout, MaxPooling1D, Conv1D, LSTM, Embedding
from tensorflow.python.keras import utils
from tensorflow.python.keras import backend
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import sys
from collections import Counter
import pickle as pkl



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform some initial testing for question intention model training")
    parser.add_argument("--data", default="./training_data.csv", help="Path of training data in csv (headerless)\
     format")
    parser.add_argument("--model", default="./models", help="Directory to output of models")
    parser.add_argument("--vocab-size", default=20000, type=int, help="Tokenizer vocabulary size")
    parser.add_argument("--tokenizer-mode", default="count", help="Tokenizer vectorization method")
    parser.add_argument("--batch", default=500, type=int, help="minibatch batch size")
    parser.add_argument("--epoch", default=1, type=int, help="Number of epochs")
    parser.add_argument("--debug", default=False, action="store_true", help="Output debugging info")
    parser.add_argument("--mode", default="lr", help="""Mode of action:
    lr -- tensorflow linear classifiers, logistic regression
    lr2 -- tensorflow linear classifiers, logistic regression with weight decay + ftrl
    nn -- keras nn, 1 hidden layer, 1 dropout
    nn2 -- keras nn, 2 hidden layers, 1 dropout
    lstmcnn -- keras lstm cnn with trained embedding layer
    """)
    args = parser.parse_args()


