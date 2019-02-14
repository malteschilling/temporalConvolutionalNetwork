import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--dataset', type=str, default="electric_fish", help='Dataset you are using: electric_fish or smartphone')
parser.add_argument('--model', type=str, default="TCN", help='The model you are using: TCN, Conv1D, Dense ')
args = parser.parse_args()

#######################################
#######################################
#### 1 - LOAD DATA SET
#######################################
#######################################
"""
    There are two different datasets:
    Both have in common that they represent time series and have multi modal
    input data.
    - smartphone data: classification = 9 input channels over 128 time steps    
        target output: one of six classes (loaded in one hot encoding)
            six different activities
    - electric fish: regression example = 7 input channels over 150 time steps
        target: x, y position (in [1,81]) and orientation (in degrees [0,360])
            of an electric fish swimming in a tank
"""
if args.dataset=="smartphone":
    #######################################
    # 1-A Classification: Smart Phone Data
    #######################################
    # Loading training and test data for smartphone data set
    from data.dataset_loader import SmartPhoneDataSet
    dataset_loader = SmartPhoneDataSet()
    train_X, train_y, test_X, test_y, regression = dataset_loader.load_dataset('data/UCI_HAR_Dataset/')
else:
    #######################################
    # 1-B Regression: Electric Fish Data
    #######################################
    from data.dataset_loader import ElectricFishDataSet
    dataset_loader = ElectricFishDataSet()
    train_X, train_y, test_X, test_y, regression = dataset_loader.load_dataset('data/')
print("** Training data (", args.dataset, "): ", train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# Visualization of examples of the training data set
#idx = np.arange(train_X.shape[0])
#np.random.shuffle(idx)
#dataset_loader.visualize_sample(train_X[idx[0:10]], train_y[idx[0:10]])

#######################################
#######################################
#### 2 - Load DNN Model
#######################################
#######################################
""" 
    Different deep neural network models are applied (allows for comparison):
    Main goal is to test Temporal Convolutional Networks - this is presented first.
"""
print("** Train on model type: ", args.model)
if args.model=="TCN":
    #######################################
    # 2-A Temporal Convolutional Network
    #######################################
    from model.TemporalConvolutionalNetwork import TemporalConvolutionalNetwork
    # Target data is reshaped to 3-dimensional data set
    # as TCN provides many to many output
    train_y = train_y.reshape((train_y.shape[0], 1, train_y.shape[1]))
    test_y = test_y.reshape((test_y.shape[0], 1, test_y.shape[1]))
    # Setup model
    nn_model = TemporalConvolutionalNetwork(input_shape=(train_X.shape[1],train_X.shape[2]), 
        output_number=train_y.shape[2],
        kernel_size=7, filter_number=21, padding='causal',
        regression=regression)
    nn_model.train_model(train_X, train_y, test_X, test_y, epochs=args.num_epochs)

elif args.model=="Conv1D":
    #######################################
    # 2-B Convolutional 1D Network
    #######################################
    from model.Convolutional1DModel import Convolutional1DModel
    # Setup model
    nn_model = Convolutional1DModel(input_shape=(train_X.shape[1],train_X.shape[2]), 
        output_number=train_y.shape[1], module_layers=4,
        kernel_size=5, filter_number=32, 
        regression=regression)
    nn_model.train_model(train_X, train_y, test_X, test_y, epochs=args.num_epochs)

elif args.model=="Dense":
    #######################################
    # 2-C Fully-connected Dense Network
    #######################################
    from model.SimpleDenseModel import SimpleDenseModel
    
    timesteps_as_input = 128
    #train_y = np.repeat(train_y[:,np.newaxis,:], train_X.shape[1]/timesteps_as_input, axis=1)
    #train_y = train_y.reshape(( (train_y.shape[0]*train_y.shape[1]), train_y.shape[2]))
    #train_X = train_X.reshape( ( (train_X.shape[0]*train_X.shape[1])/timesteps_as_input, 
     #   train_X.shape[2] * timesteps_as_input))
    train_X = train_X.reshape( ( train_X.shape[0], 
        train_X.shape[2] * train_X.shape[1]))
    #test_y = np.repeat(test_y[:,np.newaxis,:], test_X.shape[1]/timesteps_as_input, axis=1)
    #test_y = test_y.reshape(( (test_y.shape[0]*test_y.shape[1]), test_y.shape[2]))
    #test_X = test_X.reshape( ( (test_X.shape[0]*test_X.shape[1])/timesteps_as_input, 
     #   test_X.shape[2] * timesteps_as_input))
    test_X = test_X.reshape( ( test_X.shape[0], 
        test_X.shape[2] * test_X.shape[1]))

    # Setup model
    nn_model = SimpleDenseModel(input_shape=([train_X.shape[1]]), 
        output_number=train_y.shape[1], regression=regression)
    nn_model.train_model(train_X, train_y, epochs=args.num_epochs)

#######################################
#######################################
#### 3 ANALYSE PREDICTIONS
#######################################
#######################################
if nn_model.regression:
    loss = nn_model.model.evaluate(test_X, test_y, verbose=0)
    print("Generalization loss: ", loss)
else:
    loss, accuracy = nn_model.model.evaluate(test_X, test_y, verbose=0)
    print("Generalization accuracy: ", accuracy, " - loss: ", loss)
    print("    (for TCN this includes data with zero padding at beginning)")
dataset_loader.evaluate_last_predictions(nn_model.get_prediction_without_padding(test_X), 
    nn_model.get_targets_without_padding(test_y))