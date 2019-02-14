'''
A temporal convolutional network.



@author: malteschilling@googlemail.com
'''

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, add, concatenate, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import optimizers

from keras import backend as K
from keras.layers import Layer

from keras.callbacks import TensorBoard

def TemporalConvolutionalBlock( inputTensor, filter_number, kernel_size=3, 
        padding='causal', dilation_rate=1):
    shortcut = inputTensor

    convLayer1 = Conv1D(filters=filter_number, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate, 
            activation='relu' )
    conv_output = convLayer1(inputTensor)
    conv_output = SpatialDropout1D(0.25)(conv_output)
    
    convLayer2 = Conv1D(filters=filter_number, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate, 
            activation='relu' )
    conv_output = convLayer2(conv_output)
    conv_output = SpatialDropout1D(0.25)(conv_output)

    # Residual connection:
    #multi_resolution_concat = concatenate([conv_output, shortcut])
    #output = Conv1D(filter_number, 1, padding='same')(multi_resolution_concat)
    if (inputTensor.shape[2] != conv_output.shape[2]):
        res_connection = Conv1D(filter_number, 1, padding='same')(inputTensor)
        output = add([conv_output, res_connection])
    else:
        output = add([conv_output, inputTensor])

    return output

class TemporalConvolutionalNetwork:

    def __init__ (self, input_shape, output_number, 
            module_layers=3, 
            filter_number=64, kernel_size=3, padding='causal', 
            regression=False,
            epochs=10):
        self._epochs = epochs
        self._batch_size = 32
        
        # Setup Model 
        print(input_shape)
        print(output_number)
        inputTensor = Input(shape=(None, input_shape[1]))
        
        tcnOutput = TemporalConvolutionalBlock( inputTensor, filter_number=filter_number, kernel_size=kernel_size,
                padding=padding, dilation_rate=1 )
        tcnOutput = TemporalConvolutionalBlock( tcnOutput, filter_number=filter_number, kernel_size=kernel_size,
                padding=padding, dilation_rate=2 )
        tcnOutput = TemporalConvolutionalBlock( tcnOutput, filter_number=filter_number, kernel_size=kernel_size,
                padding=padding, dilation_rate=4 )
        tcnOutput = TemporalConvolutionalBlock( tcnOutput, filter_number=filter_number, kernel_size=kernel_size,
                padding=padding, dilation_rate=2 )
        tcnOutput = TemporalConvolutionalBlock( tcnOutput, filter_number=filter_number, kernel_size=kernel_size,
                padding=padding, dilation_rate=1 )
        
        self.regression = regression
        if self.regression:
            outputLayer = Dense(output_number, activation='linear')
            finalOutput = outputLayer(tcnOutput)
            self.model = Model(inputTensor,finalOutput)
            adam = optimizers.Adam(lr=0.002, clipnorm=1.)
            self.model.compile(adam, loss='mean_squared_error')
        else:
            outputLayer = Dense(output_number, activation='softmax')
            finalOutput = outputLayer(tcnOutput)
            self.model = Model(inputTensor,finalOutput)
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.model.summary()
        print("Size of receptive field: ", self.calculate_receptive_field_size() )

    def calculate_receptive_field_size(self):
        rec_field_size = 1.
        for layer in self.model.layers:
            if isinstance(layer, Conv1D):
                # Calculate size of receptive field:
                # rec_#(n) = rec_#(n-1) + 2 * [kernel_size(n)-1] * dilation(n).
                rec_field_size += (layer.kernel_size[0]-1) * (layer.dilation_rate[0])
        return rec_field_size
        
    def train_model(self, train_X, train_y, val_X=None, val_y=None, validation_split=0.2, epochs=None):    
        epochs = self._epochs if epochs is None else epochs
        
        import time
        now = time.strftime("%c")
        # Train the network on the training data.
        if (val_X is None):
            # Train the network on the training data.
            self.model.fit(train_X, train_y, validation_split=validation_split, 
                epochs=epochs, batch_size=self._batch_size, verbose=1,
                callbacks=[TensorBoard(log_dir='./logdir/block/'+now)] )
        else:
            # Train the network on the training data.
            self.model.fit(train_X, train_y, validation_data=(val_X, val_y), 
                epochs=epochs, batch_size=self._batch_size, verbose=1,
                callbacks=[TensorBoard(log_dir='./logdir/block/'+now)] )
        #print(self.model.predict(train_X[100:110]))
        #print(train_y[100:110])
        #print(self.model.predict(val_X[200:210]))
        #print(val_y[200:210])
        
    def get_prediction_without_padding(self, set_X):
        complete_time_series = self.model.predict(set_X)
        return complete_time_series[:,-1,:]
        
    def get_targets_without_padding(self, set_y):
        return set_y[:,-1,:]