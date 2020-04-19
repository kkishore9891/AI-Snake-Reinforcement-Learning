#####################################IMPORTING ESSENTIAL LIBRARIES#######################################
import snake												#Snake environment.
from tensorflow import keras										#Keras for building neural networks.
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation,Conv2D,MaxPool2D		#Important layers to be used.
import numpy as np											#For operating with matrices and vectors.
import cv2                                                                                              #Image Processing Library.
#####################################CREATING THE NEURAL NETWORK######################################
model=keras.Sequential()                                                                        	#Create a new model.
model.add(keras.layers.Lambda(lambda x:x/255,input_shape=(16,16,3)))					#Normalise the input image. The input Image has the dimensions 16x16x3 (Color Image).
model.add(Conv2D(64,(3,3),padding='same'))                                                              #Convolutional Layer 1 with 3x3 kernel and 64 feature maps.
model.add(Activation('relu')) 										#ReLU activation function.
model.add(Dropout(0.3)) 										#Dropout probability is 30%.
model.add(MaxPool2D(pool_size=(2,2))) 									#Maxpooling layer to reduce the size to its half.
model.add(Conv2D(128,(3,3),padding='same')) 								#Convolutional Layer 3 with 3x3 kernel and 256 feature maps.
model.add(Activation('relu')) 										#ReLU activation function.
model.add(Dropout(0.3)) 										#Dropout probability is 30%.
model.add(Conv2D(256,(3,3),padding='same'))                                                             #Convolutional Layer 1 with 3x3 kernel and 64 feature maps.
model.add(Activation('relu')) 										#ReLU activation function.
model.add(Dropout(0.3)) 										#Dropout probability is 30%.
model.add(MaxPool2D(pool_size=(2,2))) 									#Maxpooling layer to reduce the size to its half.
model.add(Conv2D(512,(3,3),padding='same')) 								#Convolutional Layer 3 with 3x3 kernel and 256 feature maps.
model.add(Activation('relu')) 										#ReLU activation function.
model.add(Dropout(0.3)) 										#Dropout probability is 30%.
model.add(MaxPool2D(pool_size=(2,2))) 									#Maxpooling layer to reduce the size to its half.
model.add(Flatten()) 											#Flatten the feature map to feed into a fully connected layer.
model.add(Dense(512))											#Fully connected layer with 256 neurons.
model.add(Activation("relu"))										#ReLU activation function.
model.add(Dropout(0.3))											#Dropout probability is 30%.
model.add(Dense(256))											#Fully connected layer with 256 neurons.
model.add(Activation("relu"))										#ReLU activation function.
model.add(Dropout(0.3))											#Dropout probability is 30%.
model.add(Dense(4))											#Output layer with 4 neurons.
model.add(Activation("linear"))										#Activation function is linear to predict Q Values.
model = keras.models.load_model("Models/model_conv.hdf5")                                               #Load a saved model. (Comment if the model doesn't exist while using the first time).
model.compile(loss="mse",optimizer=keras.optimizers.Adam(learning_rate=0.0001),metrics=['accuracy'])	#The loss function is mean square error and the optimiser is adam optimiser. The accuracy of the model is displayed as it trains.
#########################################INITIALISING THE GAME############################################
player = snake.Snake() 											#Create a snake game instance.
val = player.val 											#Initial movement value.
player.render()                                                                                         #Display the changes and create the new input image.		
state = player.state() 											#Obtaining the initial state of the snake.
done = state[1] 											#If the snake eats the food.
hit = state[2] 												#If the snake has hit the wall.
learning_rate = 0.20 											#A constant used in Q value formula.
discount = 0.9 												#Another constant used in Q value formula.
Iterations = 400											#The number of times the game is to be played.
ctr = 0 												#Counter variable.
epsilon = [1] 												#This is used to make the model robust by exploring random paths for different lengths of the snake.
epsilon_decay = 0.9 											#Decay Factor.
min_epsilon = 0.2 											#Minimum value of epsilon.
I = [] 												        #Training inputs.
T = [] 													#Training targets.
#################################TRAINING THE MODEL BY COLLECTING THE DATA##############################
for iteration in range(Iterations): 									#The AI plays the game "Iteration" no. of times.
	history = [] 											#Track where the snake goes.
	print("Iteration:",iteration) 									#The current episode.
	dist = ((player.food[0][0] - player.snake[0][0])**2 + (player.food[0][1] - player.snake[0][1])**2)**0.5#Distance between the food and the snake's head.
	#print(epsilon) 										#Print the epsilon value.
	while not(done or hit or player.snake in history): 						#The AI plays till it wins, loses or gets stuck in a location.
		x = player.game          								#The input vector.
		if len(I) < 2000: 									#The input data(Replay memory) should have a maximum length of 2000.
			I.append(x.reshape(16,16,3)) 							#Save the current input to the training inputs replay memory.
		else: 											#If the length exceeds 2000.
			I[ctr % 2000] = x.reshape(16,16,3) 						#Overwrite on the past data.
		history.append(player.snake)                                                            #Append the position of the snake to histroy vector.
		Q_Values = model.predict(x.reshape(1,16,16,3)) 						#The predicted Q values.
		#print("Prediction:",Q_Values[0]) 							#Print the predicted Q-Value.
		if len(player.snake) <= len(epsilon): 						        #Checking if the length of the snake is lesser than epsilon vector.
			action = np.argmax(Q_Values)							#The move to be taken (0 - Up, 1 - Down, 2 - Left, 3-Right).
			if (np.random.random() < epsilon[len(player.snake)-1]) and epsilon[len(player.snake)-1] > 0.2 and state[1]:#If a random value is lesser than the epsilon value for a specific length of the snake, the epsilon is above a minimum value and the snake ate a food in the previous iteration.
				action = np.random.randint(0,4) 					#Select a random action instead of predicted action.
				epsilon[len(player.snake)-1] *= epsilon_decay 				#Decay the epsilon for the specific length.
				epsilon[len(player.snake)-1] = max(epsilon[len(player.snake)-1],min_epsilon)#If the value goes below minimum, reset it.
				print("Random") 							#A random value was selected.
		else: 											#If the lengths of the snake exceeds the epsilon vector.
			epsilon.append(1) 								#Make a new entry for the new length of the snake.
			action = np.argmax(Q_Values)							#The move to be taken (0 - Up, 1 - Down, 2 - Left, 3-Right).
			if (np.random.random() < epsilon[len(player.snake)-1]) and epsilon[len(player.snake)-1] > 0.2 and state[1]:#If a random value is lesser than the epsilon value for a specific length of the snake, the epsilon is above a minimum value and the snake ate a food in the previous iteration.
				action = np.random.randint(0,4) 					#Select a random action instead of predicted action.
				epsilon[len(player.snake)-1] *= epsilon_decay 				#Decay the epsilon for the specific length.
				epsilon[len(player.snake)-1] = max(epsilon[len(player.snake)-1],min_epsilon)#If the value goes below minimum, reset it.
				print("Random") 							#A random value was selected.
		new_state = player.move(action) 							#Move the snake based on the prediction.
		#print("Epsilon:",epsilon) 								#Print the epsilon vector.
		player.render()                                                                         #Display the changes and create the new input image.		
		new_x = player.game 									#The new state after moving in predicted direction.
		done = new_state[1] 									#Whether the snake has eaten the food after the new move.
		hit = new_state[2] 									#Whether the snake has hit the wall after the new move.
		if done: 										#If the Snake eats the food.
			reward = 20 									#The reward it with the highest score.
		elif hit: 										#If the Snake has hit the wall.
			reward = -40 									#Punish it with a huge negative score.
		else: 											#If it simply moved a step.
			new_dist =  ((player.food[0][0] - player.snake[0][0])**2 + (player.food[0][1] - player.snake[0][1])**2)**0.5    	#Calculate the distance after making the movement.
			if new_dist > dist: 								#If the distance has increased.
				if len(player.snake) <= 4: 						#If the snake's length is lesser than or equal to 4 pixels.
					reward = -5 							#Punish it with a negative reward.
				else: 									#If the length exceeds 5.
					reward = -.5 							#Punish it with a very small negative reward as the game gets complicated.
			else: 										#If the distance has decreased.
				reward = 0.5 								#Give a small positive reward.
			dist = new_dist 								#Replace the old distance with the new distance.
		New_Q_Values = model.predict(new_x.reshape(1,16,16,3)) 					#Get the new set of Q values.
		new_action = np.argmax(New_Q_Values) 							#The new predicted action.
		if reward == -20 or reward == 20: 							#If the snake has hit the wall or eaten the food.
			New_Q_Values[0,new_action] = 0 							#The future value will not be taken into account while calculating Q value.
		Target_Q = (1 - learning_rate)*(Q_Values[0, action]) + learning_rate*(reward + discount*New_Q_Values[0, new_action])#The Q Value formula.
		Q_Values[0, action] = Target_Q 								#The old Q Value is updated using the value calculated.
		#print("Target:",Q_Values[0]) 								#Display the target value.
		y = Q_Values[0] 									#The new target value.
		if len(T) < 2000: 									#The target data should have a maximum length of 2000
			T.append(y) 									#Save the current target value to the training targets list.
		else: 											#If the length exceeds 2000.
			T[ctr % 2000] = y 								#Overwrite the existing data.
		state = new_state 									#Replace the current state with the next state.
		cv2.waitKey(1) 										#Wait for 1 millisecond for the user to press a key.
		ctr+=1 											#Increment the counter.
	X = np.array(I)                                                   				#Converting the inputs list to a numpy array.
	Y = np.array(T) 										#Converting the targets list to a numpy array.
	model.fit(X, Y, batch_size=32, epochs = 1, verbose=1,shuffle=False) 				#Training the model.
	#print(Y) 											#Display the targets.
	#print(model.predict(X)) 									#Display the predictions.
	done = False 											#Resetting done.
	hit = False 											#Resetting hit.
model.save("Models/model_conv.hdf5")                                         				#Saving the trained model.
cv2.destroyAllWindows() 										#Close the SNAKE window.
#########################################################################################################
	

