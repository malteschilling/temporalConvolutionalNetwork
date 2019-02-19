# Temporal Convolutional Network 


--
## Summary

A temporal convolutional network is constructed in keras and applied to a classification and regression task. As baseline comparison we use a dense network as well as a simple 1D-convolutional neural network.

For details on the temporal convolutional network see the original publication: [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/pdf/1803.01271.pdf)

As one difference: weight normalization is not used here.

--

## Requirements

The neural networks are realized in the keras framework (https://keras.io) which either requires [tensorflow](https://www.tensorflow.org) or [theano](http://deeplearning.net/software/theano/).

The scripts are in python and we used python 2.7. Pickle is used to save data from training runs and matplotlib for visualizing the data.

--

## Usage

Training (and evaluating) different models:

	python train_model.py --model=Conv1D --num_epochs=100 --dataset=electric_fish
	
Possible parameters:

* model = Dense, Conv1D, TCN
* num_epochs = number of epochs (default=10)
* dataset = electric_fish (default), smartphone

--

## Structure

There are two main folders:

* data - contains the datasets (smartphone data has to be downloaded)
* model - contains the different models
* after running: a logdir will be created

--

## Results

#### Classification task

For the classification task a [data set](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) for classification of behaviours from smartphone sensory data was used. 

* Length of time series: 128 timesteps.
* Input: 9 dimensional time series (normalized)
* Output: 6 classes, one-hot encoded
* Number of epochs for training:100.
* Training data: 7352 samples
* Test data: 2947 samples.

Different configurations were tested:

* Dense network (5 layers, Total params: 82,544); Generalization accuracy: 0.8591788259246692

* Conv1D (Total params: 30,434); Generalization accuracy: 0.8931116389548693

* TCN original architecture (as described in original paper, with Blocks of 1, 2, 4, 8 dilation size; kernel\_size=5, filter\_number=28; Total params: 29,378);
Receptive field size = 121;
Generalization accuracy: 0.8985117280285035; Generalization accuracy: 0.894087207309262;
Number of miss classifications when excluding any padding: 282 of 2947,  accuracy: 0.9043094672548354

* TCN architecture (with Blocks of 1, 2, 4, 2, 1 dilation size; kernel\_size=7, filter\_number=21; Total params: 29,658);
Receptive field = 121;
Generalization accuracy: 0.9070002332880895;
Generalization accuracy: 0.9109952918424176;
Number of miss classifications when excluding any padding: 220 of 2947, accuracy: 0.9253478113335596

#### Regression task

For the regression task synthetic measurements (simulated) were created from 7 sensors placed around a fish tank that measures current as input from an electric fish. This provides information on position and orientation of the electric fish.

Network is evaluated on mean absolute position error in x and y direction (in range of 1 to 81) and mean absolute orientation error calculated in degrees.

* Input: 7 dimensions, normalized
* each time series 150 steps
* Output: x, y position and 2-dimensional orientation (given as a vector on the unit circle)
* Number of epochs for training: 100.
* Training data: 5248 time series of length 150 
* Test data: 1313 samples.

Different configurations were tested:

* Dense network (fully connected, Total params: 75,864)
	* Mean position difference [1..81]: [ 9.07769757, 19.13281343]
	* Std dev position difference: [ 8.83493939, 11.58760673]))
	* Orientation (degrees) mean/std: 22.868539151225267, 33.89565144003737

* Conv1D simple network (Total params: 33,112)
	* Mean position difference [1..81]: [7.34562092, 7.84315034]
	* Std dev position difference: [7.31536976, 8.13988954]
	* Orientation (degrees) mean/std: 21.036551953053777, 31.359925059993436 

* TCN original architecture (as described in original paper, with Blocks of 1, 2, 4, 8 dilation size; kernel\_size=5, filter\_number=28; Total params: 28,984);
Receptive field size = 121;
	* Mean (absolute) position difference for last timestep [1..81]: [14.66728339, 15.01606241]
	* Std dev position difference for last timestep: [10.95818841, 11.51426357]
	* Orientation (degrees) mean/std for last timestep: 30.990351403099513, 38.488101430834526
	* Minimal abs error over time [1..81]: 13.070564812215247 - at time step: 52

![Electric fish data, error over time](Data/loss_over_time.pdf)
When looking at the electric fish data, one can see that the error is initially very high. The reason is that there is zero padding and the prediction is only based on very little real data. It rises also at the end. This is most probably due to the sinusoidal data.

* TCN architecture (with Blocks of 1, 2, 4, 2, 1 dilation size; kernel\_size=7, filter\_number=21; Total params: 29,278); Receptive field = 121; 
	* Mean position difference [1..81]: [16.87574486, 18.66155783]
	* Minimal abs error over time [1..81]: 14.285876066196264, - at time step 43
	* Std dev position difference: [11.09996728, 11.38420293]
	* Orientation (degrees) mean/std: 26.959646456508253, 37.42577067844571 
