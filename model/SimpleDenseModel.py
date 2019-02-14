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


@author: mschilling
'''

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

class SimpleDenseModel:

    def __init__ (self, input_shape, output_number, module_layers=3, 
            hidden_size = 64,
            regression=False): 
        self._epochs = 10
        self._batch_size = 32
        
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=input_shape[0]))
        #self.model.add(Dropout(0.25))
        for mod_layers in range(1, module_layers):
            self.model.add(Dense(hidden_size, activation='relu'))
            #self.model.add(Dropout(0.25))
        self.model.add(Dense(output_number, activation='softmax'))
        
        self.regression = regression
        if self.regression:
            self.model.add(Dense(output_number, activation='linear'))
            self.model.compile(loss='mean_squared_error', optimizer='adam')
        else:
            self.model.add(Dense(output_number, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.model.summary()

    def train_model(self, trainX, trainy, epochs=None):
        epochs = self._epochs if epochs is None else epochs
        # Train the network on the training data.
        self.model.fit(trainX, trainy, epochs=epochs, batch_size=self._batch_size, verbose=1)
        
    def get_prediction_without_padding(self, set_X):
        complete_time_series = self.model.predict(set_X)
        return complete_time_series
        
    def get_targets_without_padding(self, set_y):
        return set_y