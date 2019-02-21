'''
Simple LSTM Model.

@author: mschilling
'''

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from keras.callbacks import TensorBoard

class LSTMModel:

    def __init__ (self, input_shape, output_number,  
            hidden_size = 64,
            regression=False): 
        self._epochs = 10
        self._batch_size = 32
	
        self.model = Sequential()
        self.model.add(LSTM(hidden_size, input_shape=input_shape))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(hidden_size/2, activation='relu'))
        
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
        
        import time
        now = time.strftime("%c")
        # Train the network on the training data.
        if (val_X is None):
            # Train the network on the training data.
            self.model.fit(train_X, train_y, validation_split=validation_split, 
                epochs=epochs, batch_size=self._batch_size, verbose=1,
                callbacks=[TensorBoard(log_dir='./logdir/lstm/'+now)] )
        else:
            # Train the network on the training data.
            self.model.fit(train_X, train_y, validation_data=(val_X, val_y), 
                epochs=epochs, batch_size=self._batch_size, verbose=1,
                callbacks=[TensorBoard(log_dir='./logdir/lstm/'+now)] )