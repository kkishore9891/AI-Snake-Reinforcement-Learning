# AI-Snake-Reinforcement-Learning.

This is an implementation of using AI to play the popular snake game using Reinforcement learning with Tensorflow 2.0.

Reinforcement learning is a technique which is used to train neural networks without large amount of labelled data. Instead, we let the AI make predictions for a certain input state and give a reward or punishment to the AI based on its performance. Therefore, the AI tries to maximise the reward by updating the weight parameters of the neural network accordingly.

There are two different implementations that I have tried.

1)FULLY CONNECTED NEURAL NETWORKS:

In this method, the neural network has a 1 dimensional vector as input. The vector consists of 11 data points(The distance between snake's head and food in x direction and y direction, the distance between the snake's head and the obstacles in N,E,W,S,NE,NW,SE,SE directions and the length of the snake). This 11 dimensional input is used by the neural network to formulate a Q-Table which consists of the rewards for each possible action that could be taken for each possible Input.

2)CONVOLUTIONAL NEURAL NETWORKS:

In this method, the neural network has an input image of dimensions 16x16 with three colour channels(B,G,R). The Neural network consists of convolutional layers which uses trained kernels to produce feature maps. A series of convolutional and maxpooling layers produces a feature vector, which is used by fully connect layers to formulate the Q-Table.

### Contents:

1) snake.py is the game environment. This can return inputs suitable for both fully connected and convolutional neural network architectures. The code is commented line by line. You can check it for any clarifications.

2) snake_rl_conv.py is used to train the AI using convolutional neural networks. The code can be used to retrain a previously trained neural network to improve its accuracy.

3) snake_rl_fcn.py is used to train the AI using fully connected neural networks. The code can be used to retrain a previously trained neural network to improve its accuracy.

4) snake_test_conv.py is used to test the convolutional neural network trained using RL. Make sure to change the name of the model file inside the code while testing a new neural network.

5) snake_test_fcn.py is used to test the fully coonected neural network trained using RL. Make sure to change the name of the model file inside the code while testing a new neural network.

### Dependencies:

1) Numpy library. Install using the cmd command: pip-install numpy.
2) OpenCV library. Install using the cmd command: pip-install opencv-python.
3) Tensorflow 2.0-gpu with CUDA support. Visit the follwing link install CUDA and tensorflow 2.0-gpu: https://medium.com/ai-club-gectcr/installing-tensorflow-2-0-on-windows-10-x64-e9380c5e20d8

### Note:

1) To train a new model using a new architecture, you have to alter the code. While doing so, you have to comment out the model.load() command in both snake_rl_conv.py and snake_rl_fcn.py in order to prevent an older model from being loaded. You have to do this only for the first time. To retrain the new architecture, uncomment the model.load() statement and change the name of the model file.
For example:
model = model.load("Models/New_Model.hdf5")

2) The code crashes after 400 iterations(Episodes). If you want to train the CNN model for 800 iterations, you have to do so by running the CNN code 2 times in a sequence using the cmd command. Since the code saves the new model at the end of the code, the code will load it into the memory and retrain it for 400 iterations using the model.load() command.
For example:
python snake_rl_conv.py && snake_rl_conv.py

3) Please don't forget to give credits to the author while repurposing it for your own use.

### Links:

1) My LinkedIN post(Implementation): https://www.linkedin.com/posts/kishore-kumar-5a935a163_reinforcementlearning-tensorflow-neuralnetwork-ugcPost-6657648173022056449-yw1V
2) Tutorials for Reinforcement learning: https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/ 
