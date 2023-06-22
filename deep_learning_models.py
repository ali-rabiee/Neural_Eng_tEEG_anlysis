import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, Flatten, Dense, Bidirectional
from tensorflow.keras.callbacks import Callback


class LSTM_binary:
    def __init__(self, win_size, n_channels):
        
        self.win_size = win_size
        self.n_channels = n_channels

    def build(self):
        # Define the LSTM model
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, input_shape=(self.win_size, self.n_channels)))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(LSTM(64))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return self.model
    

class LSTM_multiclass:
    def __init__(self, win_size, n_channels):
        
        self.win_size = win_size
        self.n_channels = n_channels

    def build(self):
        # Define the LSTM model
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, input_shape=(self.win_size, self.n_channels)))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(LSTM(64))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return self.model
    

class CNN_binary:
    def __init__(self, win_size, n_channels):
        
        self.win_size = win_size
        self.n_channels = n_channels

    def build(self):
        # Define the CNN model
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(self.win_size, self.n_channels)))
        self.model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.5))
        self.model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        self.model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        
        return self.model
    

class CNN_multiclass:
    def __init__(self, win_size, n_channels):
        
        self.win_size = win_size
        self.n_channels = n_channels

    def build(self):
        # Define the CNN model
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(self.win_size, self.n_channels)))
        self.model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.5))
        self.model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        self.model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        
        return self.model



class RCNN_binary:
    def __init__(self, win_size, n_channels):
        
        self.win_size = win_size
        self.n_channels = n_channels

    def build(self):
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(self.win_size, self.n_channels)))
        self.model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.5))
        self.model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        self.model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.5))
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(Dropout(0.5))
        self.model.add(Bidirectional(LSTM(64)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return self.model


class RCNN_multiclass:
    def __init__(self, win_size, n_channels):
        
        self.win_size = win_size
        self.n_channels = n_channels

    def build(self):
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(self.win_size, self.n_channels)))
        self.model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.5))
        self.model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        self.model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.5))
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(Dropout(0.5))
        self.model.add(Bidirectional(LSTM(64)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return self.model


class PrintAccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}, Train Accuracy: {logs['accuracy']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}")