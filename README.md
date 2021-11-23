# XOR Deep Neural Network


This simple Deep Neural Network ("DNN") is able to accurately mimic an XOR logic operation (also Exclusive Disjunction). Which outputs TRUE if its binary
inputs differ, and outputs FALSE if its binary inputs are the same. 

We can represent this in a Truth Table: 

<img width="500" alt="Screen Shot 2021-11-22 at 8 50 10 PM" src="https://user-images.githubusercontent.com/69181932/142967396-76c37280-0b3f-4b7b-afb8-6e887dd6ec62.png">

We can also express the XOR gate as a mathematical expression, where 1 = TRUE and 0 = FALSE:

<img width="500" alt="Screen Shot 2021-11-22 at 8 47 55 PM" src="https://user-images.githubusercontent.com/69181932/142967531-fe1644df-2be0-4697-ab89-9d376788cdf2.png">


The DNN has two input neurons, four neurons for the first hidden layer, four neurons for the second hidden layer, and 2 output neurons. 
The two hidden layers use the ReLU activation function and the output layer uses the softmax activation function.

<img width="950" alt="Screen Shot 2021-11-22 at 9 49 27 PM" src="https://user-images.githubusercontent.com/69181932/142968989-86557c56-00b1-4d73-b241-adb937ae5ee1.png">


When training a DNN, one tries to minimize the Loss function. There are a lot of algorithms that can accomplish the minimization of that function. 
For this network I chose Stochastic Gradient Descent ("SGD"), Where the algorithm iterates through a training set, and performs an update for each 
training example. Several passes can be made over the training set until the algorithm converges. When combining SGD with the backpropagation algorithm, 
it is the standard algorithm for training artificial neural networks.

Here you can see the Loss function after each iteration:

<img width="1200" alt="Screen Shot 2021-11-23 at 1 21 45 PM" src="https://user-images.githubusercontent.com/69181932/143090034-abde87b2-a0f6-4ec6-99b2-3c7d6f936afd.png">


