#########################################################################################################
# 													#
#		             This is the snake library used for training Reinforcement learning 	#
# 													#
#		        This library is written by Kishore Kumar from Madras Institute of Technology.   #
# 													#
#			           Please give due credit while using it in your project. 		# 													
# 													#
########################IMPORTING ESSENTIAL LIBRARIES####################################################
import cv2                                                                                              #Image Processing Library.
import numpy as np                                                                                      #Matrices and math library.
#######################CREATING THE GAME ENVIRONMENT#####################################################
class Snake:                                                                                            #Creating a class for the game.
	def __init__(self):                                                                             #Constructor.
		self.back_ground = np.zeros((16,16,3),np.uint8)                                     	#Game Background.
		self.val = None                                                                         #
		self.snake = [[np.random.randint(0,16),np.random.randint(0,16)]]          		#Initial Snake.
		self.food = [[np.random.randint(0,16),np.random.randint(0,16)]]             		#Initial Food.
		self.first=True                                                                         #True if this is the first first time the render() function is called.
		while self.snake == self.food:                                                          #To prevent the snake and the food spawning on the same place.
			self.food = [[np.random.randint(0,16),np.random.randint(0,16)]]          	#Generate food again if it spawns on the snake.
		self.game = self.back_ground.copy()                                                     #Copy the background.
		self.game[self.snake[0][0],self.snake[0][1]] = [127,127,127]                      	#Drawing the snake on the background.
		self.game[self.food[0][0],self.food[0][1]] = [255,0,100]                               	#Drawing the food on the background.
		self.done = False 									#It is true when the snake eats the food.
		self.hit = False                                                                        #If the snake hits a wall.
		return 											#
													#
	def game_over(self): 										#
		if -1 in self.snake[0] or 16 in self.snake[0] or self.snake[0] in self.snake[1:]:	#If the snake touches the top wall.
			self.game = self.back_ground.copy()						#Black screen.
			return True									#
		else: 											#
			return False 									#
	def state(self):										#This function will return 13 values: Food distance in x and y directions and obstacle distance(wall or snake body) in all the 8 directions, the length of the snake, if the snake ate the food and if the snake hit the wall. 
		self.food_x = self.food[0][1] - self.snake[0][1]					#Distance between food and snake in x direction.
		self.food_y = self.food[0][0] - self.snake[0][0]					#Distance between food and snake in y direction.
		self.TOP = None 									#Shortest obstacle present in top direction.
		self.BOTTOM = None 									#Shortest obstacle present in bottom direction.
		self.LEFT = None 									#Shortest obstacle present in left direction.
		self.RIGHT = None 									#Shortest obstacle present in right direction.
		self.T_LEFT = None 									#Shortest obstacle present in top left direction.
		self.T_RIGHT = None 									#Shortest obstacle present in top right direction.
		self.B_LEFT = None 									#Shortest obstacle present in bottom left direction.
		self.B_RIGHT = None 									#Shortest obstacle present in bottom right direction.
		ctr = 1 										#Counter Variable to calculate distance between obstacle and snake.
		while True:										#Infinite loop
			if self.snake[0][0] - ctr == -1 or [self.snake[0][0] - ctr,self.snake[0][1]] in self.snake: #If snake or wall is at a distance of ctr No. of pixels.
				self.TOP = ctr 								#The distance in the upward direction is ctr.
				ctr = 1 								#Reset ctr.
				break 									#Terminate the loop.
			else: 										#If not, 
				ctr += 1 								#Increment  ctr.
		while True:										#Infinite loop.
			if self.snake[0][0] + ctr == 16 or [self.snake[0][0] + ctr,self.snake[0][1]] in self.snake: #If snake or wall is at a distance of ctr No. of pixels.
				self.BOTTOM = ctr 							#The distance in the downward direction is ctr.
				ctr = 1 								#Reset ctr.
				break 									#Terminate the loop.
			else: 										#If not,
				ctr += 1 								#Increment ctr.
		while True:										#Infinite loop.
			if self.snake[0][1] - ctr == -1 or [self.snake[0][0],self.snake[0][1] - ctr] in self.snake: #If snake or wall is at a distance of ctr No. of pixels.
				self.LEFT = ctr 							#The distance in the left direction is ctr.
				ctr = 1 								#Reset ctr.
				break 									#Terminate the loop.
			else: 										#If not,
				ctr += 1 								#Increment ctr.
		while True:										#Infinite loop.
			if self.snake[0][1] + ctr == 16 or [self.snake[0][0],self.snake[0][1] + ctr] in self.snake: #If snake or wall is at a distance of ctr No. of pixels.
				self.RIGHT = ctr 							#The distance in the right direction is ctr.
				ctr = 1 								#Reset ctr.
				break 									#Terminate the loop.
			else: 										#If not,
				ctr += 1 								#Increment ctr.
		while True:										#Infinite loop.
			if [self.snake[0][0] - ctr,self.snake[0][1] - ctr] in self.snake or -1 in [self.snake[0][0] - ctr,self.snake[0][1] - ctr]:#If snake or wall is at a distance of ctr No. of pixels.
				self.T_LEFT = ctr 							#The distance in the top left direction is ctr.
				ctr = 1 								#Reset ctr.
				break 									#Terminate the loop.
			else: 										#If not,
				ctr += 1 								#Increment ctr.
		while True:										#Infinite loop.
			if [self.snake[0][0] - ctr,self.snake[0][1] + ctr] in self.snake or -1 in [self.snake[0][0] - ctr,self.snake[0][1] + ctr] or 16 in [self.snake[0][0] - ctr,self.snake[0][1] + ctr]:#If snake  or wall is at a distance of ctr No. of pixels.
				self.T_RIGHT = ctr 							#The distance in the top right direction is ctr.
				ctr = 1 								#Reset ctr.
				break 									#Terminate the loop.
			else: 										#If not,
				ctr += 1 								#Increment ctr.
		while True:										#Infinite loop.
			if [self.snake[0][0] + ctr,self.snake[0][1] - ctr] in self.snake or -1 in [self.snake[0][0] + ctr,self.snake[0][1] - ctr] or 16 in [self.snake[0][0] + ctr,self.snake[0][1] - ctr]:#If snake or wall is at a distance of ctr No. of pixels.
				self.B_LEFT = ctr 							#The distance in the bottom left direction is ctr.
				ctr = 1 								#Reset ctr.
				break 									#Terminate the loop.
			else: 										#If not,
				ctr += 1 								#Increment ctr.
		while True:										#Infinite loop.
			if [self.snake[0][0] + ctr,self.snake[0][1] + ctr] in self.snake or 16 in [self.snake[0][0] + ctr,self.snake[0][1] + ctr]:#If snake or wall is at a distance of ctr No. of pixels.
				self.B_RIGHT = ctr 							#The distance in the bottom right direction is ctr.
				ctr = 1 								#Reset ctr.
				break 									#Terminate the loop.
			else: 										#If not,
				ctr += 1 								#Increment ctr.
		return 	[np.array((self.food_x/15,self.food_y/15,self.TOP/15,self.BOTTOM/15,self.LEFT/15,self.RIGHT/15,self.T_LEFT/15,self.T_RIGHT/15,self.B_LEFT/15,self.B_RIGHT/15,len(self.snake)/256)),self.done,self.hit]
	def reset_snake(self):										#Does the same thing as __init__ but returns the state of the game.
		self.snake = [[np.random.randint(0,16),np.random.randint(0,16)]]          		#Initial Snake.
		self.food = [[np.random.randint(0,16),np.random.randint(0,16)]]             		#Initial Food.
		while self.snake == self.food:                                                          #To prevent the snake and the food spawning on the same place.
			self.food = [[np.random.randint(0,16),np.random.randint(0,16)]]          	#Generate food again if it spawns on the snake.
		self.hit = True 									#The snake has hit its body or the wal.		
		return  										#Leave the function.
	def reset_food(self):										#Function to respawn the food.
		while self.food[0] in self.snake:                                                       #To prevent the snake and the food spawning on the same place.
			self.food = [[np.random.randint(0,16),np.random.randint(0,16)]]			#Randomly spawn the food.
		self.done = True 									#The snake has eaten the food.
		return 											#Leave the function.
	def getKey(self,val):										#Based on the key pressed, the function returns a number.
		key = cv2.waitKey(1)									#Wait for 65 milli seconds till a key is pressed. If a key is pressed, the ASCII value is saved.										
		if key == ord('w'): 									#If w is pressed the snake should go up
			val = 0 									#The value for up is 0.
		elif key == ord('s'): 									#If s is pressed the snake should go down.
			val = 1 									#The value for down is 1.
		elif key == ord('a'): 									#If a is pressed the snake should go left.
			val = 2 									#The value for left is 2.
		elif key == ord('d'): 									#If d is pressed the snake should go right.
			val = 3 									#The value for right is 3.
		elif key == 27: 									#If escape key is pressed.
			val = 4 									#The value is 4.
		else: 											#if nothing is pressed.
			pass 										#Do nothing.
		return val										#Return the value.
													#
	def move(self,val):										#To take movement decision.
		self.done = self.hit = False 								#Reset done and hit.
		if val == 0:									        #If the val is 0(top).
			if len(self.snake)==2 : 							#If the snake has a length of 2(Special case where the snake can pass through its body).
				if (self.snake[0][0]-1 == self.snake[1][0]): 				#If the snake will hit itself.
					self.reset_snake()						#Reset the snake.
				else:                                                                   #Otherwise,
					self.top() 							#Move upwards.
			else: 										#If the length is not 2,
				self.top() 								#Move upwards.
		elif val == 1:										#If the val is 1(bottom).
			if len(self.snake)==2: 								#If the snake has a length of 2(Special case where the snake can pass through its body).
				if (self.snake[0][0]+1 == self.snake[1][0]): 				#If the snake will hit itself.
					self.reset_snake()						#Reset the snake.
				else: 									#Otherwise,
					self.bottom() 							#move downwards.
			else:  										#If the length is not 2,
				self.bottom()								#move downwards.
		elif val == 2:										#If the val is 2(left).
			if len(self.snake)==2: 								#If the snake has a length of 2(Special case where the snake can pass through its body).
				if (self.snake[0][1]-1 == self.snake[1][1]): 				#If the snake will hit itself.
					self.reset_snake()						#Reset the snake.
				else: 									#Otherwise,
					self.left() 							#Move left.
			else:  										#If the length is not 2,
				self.left()								#Move left.
		elif val == 3:										#If the val is 3(right).
			if len(self.snake)==2: 								#If the snake has a length of 2(Special case where the snake can pass through its body).
				if (self.snake[0][1]+1 == self.snake[1][1]): 				#If the snake will hit itself.
					self.reset_snake()						#Reset the snake.
				else: 									#Otherwise,
					self.right() 							#Move right.
			else:  										#If the length is not 2,
				self.right()								#Move right.
		if self.game_over(): 									#If the game is over,
			self.reset_snake() 								#Reset the snake.
		return self.state()		                                                        #Return the state of the game.
													#
	def top(self):                                                                                  #To make the snake move upwards.
		if [self.snake[0][0]-1,self.snake[0][1]] == self.food[0]:                               #If the snake touches the food.
			self.snake = self.food+self.snake                                               #The snake expands.
			self.reset_food() 								#Reset the food.
		else:                                                                                   #If not,
			self.snake = [[self.snake[0][0]-1,self.snake[0][1]]]+self.snake[:-1]            #The snake is shifted 1 position up.
		return											#Leave the function.
													#
	def bottom(self):                                                                               #To make the snake move downwards.
		if [self.snake[0][0]+1,self.snake[0][1]] == self.food[0]:                               #If the snake touches the food.
			self.snake = self.food+self.snake                                               #The snake expands.
			self.reset_food()                                                               #Reset the food.
		else:                                                                                   #If not,
			self.snake = [[self.snake[0][0]+1,self.snake[0][1]]]+self.snake[:-1]            #The snake is shifted 1 position down.
		return											#Leave the function.
													#
	def left(self):                                                                                 #To make the snake move left.
		if [self.snake[0][0],self.snake[0][1]-1] == self.food[0]:                               #If the snake touches the food.
			self.snake = self.food+self.snake                                               #The snake expands.
			self.reset_food() 								#Reset the food.
		else:                                                                                   #If not,
			self.snake = [[self.snake[0][0],self.snake[0][1]-1]]+self.snake[:-1]            #The snake is shifted 1 position left.
		return											#Leave the function.
													#
	def right(self):                                                                                #To make the snake move right.
		if [self.snake[0][0],self.snake[0][1]+1] == self.food[0]:                               #If the snake touches the food.
			self.snake = self.food+self.snake                                               #The snake expands.
			self.reset_food() 								#Reset the food.
		else:                                                                                   #If not,
			self.snake = [[self.snake[0][0],self.snake[0][1]+1]]+self.snake[:-1]            #The snake is shifted 1 position right.
		return											#Leave the function.
													#
	def render(self):										#Render the image of the game.
		if not self.game_over(): 								#Render only if the game is not over.
			if self.first: 									#If render is called for the first time.
				cv2.namedWindow("SNAKE",cv2.WINDOW_NORMAL)				#Create a resizable window.
			self.game = self.back_ground.copy()						#Copy the background.
			for i in range(len(self.snake)):						#Iterate across each pixek of the snake.
				if i != 0:								#Everything except the head is white.
					self.game[self.snake[i][0],self.snake[i][1]] = [255,255,255]	#
				else:									#The head is grey.
					self.game[self.snake[i][0],self.snake[i][1]] = [127,127,127]	#
			self.game[self.food[0][0],self.food[0][1]] = [255,0,100] 			#The food is purple.
			cv2.imshow("SNAKE",self.game) 							#Display the image.
			cv2.waitKey(1)                                                  		#Wait for 1 millisecond till a key is pressed.
		else:											#If the game is over, don't render.
			self.reset_snake() 								#If not,
		return 											#Reset the game.
													#
#########################################################################################################
#Import the library using import snake 									#
#Create an Instance as follows: player = snake.Snake() 							#
#An initial value is required. Obtain it as follows: val = player.val 					#													
#To detect user key press use: val = player.getKey(val) 						#
#If nothing is pressed, val is the same as previous value and the snake resumes going in the same 	#
#direction. 												#
#If w(Top) is pressed, val = 0. If s(Bottom) is pressed, val = 1. If a(Left) is pressed, val = 2. If 	#
#d(Right) is pressed, val =3. 										#
#To make the player move in the required direction, the corresponding key value is given using  	#
#player.move(x) where x=0,1,2,3. 									# 
#The move function also returns 13 values: Food distance in x and y directions and obstacle 		#
#distance(wall or snake body) in all the 								#
#8 directions, the length of the snake, if the snake ate the food and if the snake hit the wall. 	#
#The first 11 values are converted into an array							#
#These values can also be obtained using the function player.state(). Convert the 1st 11 values to a 	#
#numpy array before feeding it to the 									#
#model.                                                       	                                        #
#The function player.render() creates an open-cv window and diplays the snake, the food and the 	#
#background. 												#
############################################SAMPLE######################################################
#player = Snake() 											#Create a player.
#val = player.val 											#Initialise
#while val != 4: 											#Game loop.
#	val = player.getKey(val) 									#
#	state = player.move(val) 									#
#	player.render() 										#
#########################################################################################################
