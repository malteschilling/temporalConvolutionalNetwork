'''
Simple example for a 1D-Convolutional model that can be applied to time series.
Architecture is adapted from
https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
(they don't use max-pooling in every block)

It is trained on sensor data from smartphone data set
https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

Model consists on stacked upon each other 1D convolutional layers - there is afterwards
only a single maxpooling layer which connects to a dense layer.
Therefore, the model still keeps up quite a large number of parameters.


Example params:
for 128 time steps in 9 input channels with 64 filters in the layers = 400.000 params
Model performs well - training acc above 95 % and test acc around 91 %
For half the number of filters still on same level; and even for 8 filters only in each
layer around 89 %.

@author: mschilling
'''

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

class Convolutional1DModel:

    def __init__ (self, input_shape, output_number, module_layers=3, 
            filter_number=32, kernel_size=5, 
            dropout=0.25, 
            regression=False):
        self._epochs = 10
        self._batch_size = 32
        
        self.model = Sequential()
        self.model.add(Conv1D(filters=filter_number, kernel_size=kernel_size, 
                 activation='relu', input_shape=input_shape))
        self.model.add(Dropout(dropout))
#        self.model.add(Conv1D(filters=filter_number, kernel_size=kernel_size, 
 #               activation='relu'))
  #      self.model.add(Dropout(dropout))
        self.model.add(MaxPooling1D(pool_size=2))
        for mod_layers in range(1, module_layers):
            self.model.add(Conv1D(filters=filter_number, kernel_size=kernel_size, 
                activation='relu'))
            self.model.add(Dropout(dropout))
#            self.model.add(Conv1D(filters=filter_number, kernel_size=kernel_size, 
 #               activation='relu'))
  #          self.model.add(Dropout(dropout))
            self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        #self.model.add(Dense(16, activation='relu'))
        
        self.regression = regression
        if self.regression:
            self.model.add(Dense(output_number, activation='linear'))
            self.model.compile(loss='mean_squared_error', optimizer='adam')
        else:
            self.model.add(Dense(output_number, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train_model(self, train_X, train_y, val_X=None, val_y=None, validation_split=0.2, epochs=None):
        epochs = self._epochs if epochs is None else epochs
        if (val_X is None):
            # Train the network on the training data.
            self.model.fit(train_X, train_y, validation_split=validation_split, 
                epochs=epochs, batch_size=self._batch_size, verbose=1)
        else:
            # Train the network on the training data.
            self.model.fit(train_X, train_y, validation_data=(val_X, val_y), 
                epochs=epochs, batch_size=self._batch_size, verbose=1)
                
    def get_prediction_without_padding(self, set_X):
        complete_time_series = self.model.predict(set_X)
        return complete_time_series
        
    def get_targets_without_padding(self, set_y):
        return set_y