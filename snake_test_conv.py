#####################################IMPORTING ESSENTIAL LIBRARIES#######################################
import snake                                                                                                                                                                                                                    #Snake environment.
from tensorflow import keras                                                                                                                                                                                    	#Keras for building neural networks.
import numpy as np 												#For operating with matrices and vectors.
import cv2                                                                                                                                                      				#Image Processing Library.
#############################################INITIALISATION################################################
snake.np.random.seed(0)                                                                                                                                         			#For testing the accuracy of the model, the snake and food should start from the same positions everytime.
model = keras.models.load_model("Models/model3.hdf5")                                                                                                    		#Load a saved model.
player = snake.Snake() 											#Create a snake game instance.
val = player.val 												#Initial movement value.
player.render()                                                                                         							#Display the changes and create the new input image.		
state = player.state() 											#Obtaining the initial state of the snake.
done = state[1] 												#If the snake eats the food.
hit = state[2] 												#If the snake has hit the wall.
WINDOW = cv2.namedWindow("SNAKE",cv2.WINDOW_NORMAL) 							#Create a resizable window.
score = 0 													#The current score.
history = [] 												#To prevent the snake from getting stuck in a loop.
##########################################OBTAINING INFERENCES##########################################
while True: 												#Infinite loop.
	X = player.game 											#Obtain input image.
	history.append(player.snake) 										#Record the curent position of the snake.
	action = model.predict_classes(X.reshape(1,16,16,3)) 							#Predict the action to be taken.
	state = player.move(action) 										#Move in the predicted direction.
	player.render() 											#Update the changes.
	done = state[1] 											#If the snake has eaten the game.
	hit = state[2] 											#If the snake has hit the wall.
	if player.snake in history[:-1]:                                                                 						#If the snake has already been in the current position.
		print("The score is:",score) 									#Print the score.
		player.reset_snake() 									#Reset the snake as it is caught in a loop.
		history = [] 										#Clear the history.
		score = 0 											#Reset the score.
	if hit: 												#If the snake
		print("The score is:",score) 									#Print the score.
		score = 0 											#Reset the score.
		history = [] 										#Clear the history.
	if done: 												#If the snake has eaten the food.
		score += 1 										#Increment the score.
		history = [] 										#Clear the history.
	val = player.getKey(val) 										#Get the user key press. (w-0,s-1,a-2,d-3,Esc-4)
	if val == 4: 											#If the escape key is pressed.
		break 											#Close the game.
cv2.destroyAllWindows() 											#Close the opened window.
#########################################################################################################		
