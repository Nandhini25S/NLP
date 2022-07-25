import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import os
os['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LSTM:
    def __init__(self, input_shape, output_shape, epochs, batch_size, verbose, early_stopping_path, patience):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.early_stopping_path = early_stopping_path
        self.patience = patience
    
    def build_model(self):
        tf.compat.v1.disable_eager_execution()
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=self.input_shape))
        pass