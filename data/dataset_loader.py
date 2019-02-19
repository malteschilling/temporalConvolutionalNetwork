'''
Load data sets as numpy arrays.

Two different data sets are used: one for regression and one for classification.
1) Electric Fish - provides synthetic measurement data for the position and orientation 
   of electric fish 
   Input: measurements come from simulated electrodes around a fish tank
   Targets: position in x and y, plus rotation
2) Classification of behaviours from smartphone sensory data was used. 
   https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
   Input: acceleration sensors ...
   Output: 6 different behaviour classes 
   Data should be put in subfolder UCI_HAR_Dataset

@author: malteschilling@googlemail.com
'''
from numpy import dstack
from pandas import read_csv
from keras.utils import to_categorical

import scipy.io

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class ElectricFishDataSet:
    # Load complete dataset
    # provided as a matlab .mat file
    def load_dataset(self, folder_name='', train_ratio=0.8):
    # Load training and testing data    
        mat = scipy.io.loadmat('data/synth_data_electric_fish.mat')
        data_X = np.stack(np.hstack(mat['synth_data']))
        load_data_y = np.stack(np.hstack(mat['target']))
        #data_y = np.array(load_data_y[:,:,0], dtype=np.float)
        self.norm_factors = np.max(load_data_y, axis=0)
        data_y = np.array(load_data_y, dtype=np.float)/self.norm_factors
        data_y = data_y.reshape( (load_data_y.shape[0], data_y.shape[-1]) )
        
        # Calculate cosine and sine to describe angle in periodic way 
        # (otherwise 0 and 359 are far away from each other)
        cos_y = np.cos(np.pi*load_data_y[:,:,2]/180)
        sin_y = np.sin(np.pi*load_data_y[:,:,2]/180)
        data_y = np.concatenate( (data_y[:,0:2],cos_y, sin_y), axis=1 )
        # Get angle back: np.arctan2(sin_y[0], cos_y[0])*180/np.pi
        #   np.arctan2( data_y[0,2], data_y[0,3])*180/np.pi
        
        # Split data set into training and test set_title
        idx = np.arange(data_X.shape[0])
        np.random.shuffle(idx)
        
        # Split data set into training and test set_title        
        train_X = data_X[idx[0:int(idx.shape[0]*train_ratio)]]
        test_X = data_X[idx[int(idx.shape[0]*train_ratio):]]
        train_y = data_y[idx[0:int(idx.shape[0]*train_ratio)]]
        test_y = data_y[idx[int(idx.shape[0]*train_ratio):]]
        
        regression=True
        return train_X, train_y, test_X, test_y, regression
        
    def visualize_sample(self, samples, targets=None):        
        self.fig_3D = plt.figure(figsize=(10, 8))    
        array_time_steps = np.arange(0, samples[0].shape[0])
        print(samples.shape)
        # Remove the plot frame lines. They are unnecessary chartjunk.  
        ax_3D = plt.subplot(111, projection='3d') 
        for num, sample in enumerate(samples):
            if targets.any():
                ax_3D.set_title('Class ' + str(targets[num]))
            for i in range(0, sample.shape[1]):
                ax_3D.plot(array_time_steps, ys=np.ones(sample.shape[0])*i, zs=sample[:,i], zdir='z')

                ax_3D.set_xlabel('Time Step', fontsize=12)
                ax_3D.set_ylabel('Feature Dimension', fontsize=12, labelpad=12)
                ax_3D.set_zlabel('Activation', fontsize=12)
            plt.pause(1.)
            plt.cla()
        #plt.savefig("Results/Fig_Hidden_ActivationHiddenLayer_3D.pdf")
        #plt.show()
    
    def evaluate_predictions(self, test_pred, test_y):
        if (len(test_pred.shape) > 2):
            last_pred = test_pred[:,-1,:]
            last_y = test_y[:,-1,:]
            #targets_y_ext = np.tile(targets_y, (set_X.shape[1],1))
            # Plot a graph that shows how the mean abs error develops over time.
            # Initially, a net will mostly get zero padding and will perform 
            # therefore worse. For a fair comparison we therefore look at the last 
            # prediction as well as the mean absolute error over time (plus the minimal error
            # as the last time step is quite bad for the sinus curve data set)
            diff_over_time = np.abs(test_pred - test_y)
            abs_error_over_time = np.mean(self.norm_factors[0,0:2]*diff_over_time[:,:,0:2], axis=(0,2))
            print("Minimal abs error over time [1..81]: ", np.min(abs_error_over_time), 
                " - at time: ", np.argmin(abs_error_over_time))             
            ax_error = plt.subplot(111) 
            ax_error.plot(abs_error_over_time)
            plt.xlabel('Time steps of presented time window')
            plt.ylabel('Abs error')
            plt.title('Mean abs error over time window')
            plt.grid(True) 
            plt.show()
        else:
            last_pred = test_pred
            last_y = test_y
        diff_train = np.abs(last_pred - last_y)
        print("SHAPES: ", last_pred.shape, last_y.shape)
        print("Mean (absolute) position difference for last timestep [1..81]: ", np.mean( self.norm_factors[0,0:2]*diff_train[:,0:2], axis=0 ) )
        print("Std dev position difference for last timestep: ", np.std( self.norm_factors[0,0:2]*diff_train[:,0:2], axis=0 ) )
        diff_cos = np.sum(last_y[:,2:4] * last_pred[:,2:4],axis=1)
        diff_cos = diff_cos / np.linalg.norm(last_pred[:,2:4], axis=1)
        np.clip(diff_cos, -1., 1., out=diff_cos)
        diff_angle =np.arccos(diff_cos)
        print("Orientation (degrees) mean/std for last timestep: ", np.mean(diff_angle)*180/np.pi, np.std(diff_angle)*180/np.pi)

class SmartPhoneDataSet(ElectricFishDataSet):
    # Data in the dataset is organised in single files:
    # for each input and output there is a single file
    # - this is organized as sample x timeseries which is loaded into a tensor
    def load_single_file(self, path_data):
        single_tensor = read_csv(path_data, header=None, delim_whitespace=True)
        #print(path_data, single_tensor.shape)
        return single_tensor.values

    # Load complete dataset - for training or testing.
    def load_dataset_group(self, group, prefix=''):
        file_path = prefix + group + '/Inertial Signals/'
        # Setup all sensory input file names
        input_files = ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt', \
            'body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt', \
            'body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
        # Stack all the different sensor inputs into a three dimensional tensor:
        # Sample x Timeseries x Sensor
        loaded = list()
        for sensor_file_name in input_files:
            data = self.load_single_file(file_path + sensor_file_name)
            loaded.append(data)
        X = dstack(loaded)
        #print(X.shape)
        # Load output class - and change to one-hot-encoding (shift beforehand)
        #print(self.load_single_file(prefix + group + '/y_'+group+'.txt'))
        y = to_categorical(self.load_single_file(prefix + group + '/y_'+group+'.txt') - 1)
        return X, y

    # Load complete dataset Human Activity Recognition Using Smartphones Data Set 
    # as provided at https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
    def load_dataset(self, folder_name=''):
        # Load training data set
        train_X, train_y = self.load_dataset_group('train', folder_name )
        print(train_X.shape, train_y.shape)
        # Load test data set 
        test_X, test_y = self.load_dataset_group('test', folder_name )
        print(test_X.shape, test_y.shape)
        # Normalization:
        #train_y = train_y - 1
        #test_y = test_y - 1
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
        regression = False
        return train_X, train_y, test_X, test_y, regression
        
    def evaluate_predictions(self, test_pred, test_y):
        if (len(test_pred.shape) > 2):
            last_pred = test_pred[:,-1,:]
            last_y = test_y[:,-1,:]
            #targets_y_ext = np.tile(targets_y, (set_X.shape[1],1))
            # Plot a graph that shows how the accuracy develops over time.
            # Initially, a net will mostly get zero padding and will perform 
            # therefore worse. For a fair comparison we therefore look at the last 
            # prediction as well as the accuracy/loss over time.
            miss_classification_over_time = 1.* np.sum( 
                np.argmax(test_pred, axis=2) <> np.argmax(test_y,axis=2) , axis=0)
            accuracy_over_time = 1. - miss_classification_over_time/test_y.shape[0]
            #self.fig = plt.figure(figsize=(8, 6))    
            ax_accuracy = plt.subplot(111) 
            ax_accuracy.plot(accuracy_over_time)
            plt.xlabel('Time steps of presented time window')
            plt.ylabel('Accuracy')
            plt.title('Accuracy over time window')
            plt.grid(True) 
            plt.show()
        else:
            last_pred = test_pred
            last_y = test_y
        print("Number of miss classifications when excluding any padding: ", \
            np.sum( np.argmax(last_pred, axis=1) <> np.argmax(last_y,axis=1) ), \
            ", size test set: ", last_y.shape[0], " test accuracy: ",  
            1. - 1.*np.sum( np.argmax(last_pred, axis=1) <> np.argmax(last_y,axis=1) )/last_y.shape[0])
