from __future__ import print_function
import numpy as np
import pygame
import random
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, Adadelta, Nadam, RMSprop
import keras.preprocessing.text
import numpy as np

boardSide = 6
empty = 0
black = 1
white = -1
testSpot = 3
legalSpot = 9

gameOver = False

pygame.init()

(width, height) = (600, 600)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Reversi')

block_size = int(width/(boardSide+2))
radius = int(block_size/3)
offset = int((height-(block_size*boardSide))/2)

font = pygame.font.Font(None, int(offset/1.5))

model = Sequential()
model.add(Dense(boardSide**2, activation='relu', input_dim=(boardSide**2)))
model.add(Dropout(0.05))
model.add(Dense(2*boardSide**2, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(boardSide**2, activation='softmax'))
Adadelta= Adadelta(lr=0.1, rho=0.95, epsilon=None, decay=0.0)
filename = "Network Weights/Othello-Epoch-100000-Drop-0.05-Batch-8.hdf5"
model.load_weights(filename)
model.compile(loss='mse',
			  optimizer="Adam",
			  metrics=['accuracy'])


def setUp():
  #for the board 0 is empty, 1 is black, and 2 is white
  boardArray = np.zeros((boardSide,boardSide))
  #Finds the middle of the board
  mid = int(boardSide/2 - 1)
  #These set the black and white pieces in the middle of the board
  boardArray[mid][mid] = white
  boardArray[mid+1][mid] = black
  boardArray[mid][mid+1] = black
  boardArray[mid+1][mid+1] = white
  #print("Starting Board")
  return boardArray

def displayBoard(board):
  #prints each row in the board
  for b in board:
	print (b)
  print()
  
def getWinner(board):
  #Finds the sum of the board
  boardScore = np.sum(board)
  #Returns the winner for the game
  if boardScore < 0:
	return white
  elif boardScore > 0:
	return black
  #If neither is larger then no one wins
  else:
	return empty

#This function is used to find the switches for a placement
#You give it the turn, the other value (more useful when it wasn't 1 and -1)
# and then also an array which is the row, column, or one of the diagonals 
# that the piece was placed into
def findSwitches(turn,other,array):
  
  #This finds the location of the spot that is being tested
  # to see if it creates any switches
  newPieceLoc = np.where(array == testSpot)[0][0]

  #Sets up varibles
  legalMove = False
  changeLocs = []
 

  #The array of peices in front of the point, and the array
  # behind it
  backArray = array[0:newPieceLoc][::-1]
  forwardArray =  array[newPieceLoc:]

  # Set array and varibles to null
  tempChanges = []
  foundTurn = False
  
  #For each item in the back array first you check if it is 
  #the other turn type
  for x in range(len(backArray)):
	if backArray[x] == other:
	  #If it is the other player's piece then you append that number to 
	  # the temporary change locations
	  
	  #This x+1 * -1 is only to make sure it's the right poitn becuase creating
	  #the backwards array messed up the indexing and points. But this works
	  tempChanges.append((x+1)*-1)
	elif backArray[x] == turn:
	  #If you find your on piece then you have another end point and this could
	  # be a legal move
	  foundTurn = True
	  break
	elif backArray[x] == empty  or backArray[x] == legalSpot:
	  #If you don't find that and just find an empty spot then you get nothing
	  break

  #If you found your own piece on the other end add the temp changes to the 
  # list of locations to actually change
  if foundTurn:
	changeLocs += tempChanges

  # Set array and varibles to null
  tempChanges = []
  foundTurn = False  

  #Do the same thing you did to the back array to the forward array
  #find all the pieces between the placed location and another peice 
  # of the same color
  for x in range(len(forwardArray)):
	if forwardArray[x] == other:
	  tempChanges.append(x)
	elif forwardArray[x] == turn:
	  foundTurn = True
	  break
	elif forwardArray[x] == empty or forwardArray[x] == legalSpot:
	  break

  # add those to the changeLocs if you found one of your own
  if foundTurn:
	changeLocs+=tempChanges

  #If there is anything in change locations then this is a legal move
  if len(changeLocs) > 0:
	legalMove = True
	
  #Return the legal move and the location of changes
  return [legalMove,changeLocs]

#This functions find all leagal moves for a board and a turn
def findLegal(board,turn):

  #This is the empty list of legal moves
  legalList = []

  #Goes through each spot on the board. If there is not a piece there already
  # then test it
  for x in range(boardSide):
	for y in range(boardSide):
	  #Make sure it is nto black or white
	  if board[x][y] != black and board[x][y] != white:
		#Make that spot the test spot
		board[x][y] = testSpot
  
		#determine value for other
		other = -1*turn

		#Get the column, row, and both diagonals that the test piece is in
		col = board[:,y]
		row = board[x]
		diagLeftRight = board.diagonal(y-x)
		#This is a bit harder as you need to flip the board and then get a changed 
		# diagonal here. This works though, by shifting it by y and x
		diagRightLeft = np.fliplr(board).diagonal(((boardSide-1)-y)-x)
  
		#Run find switches on the column, row and both diagonals. If any of them
		# has a legal move then append that point to the legal list
		# Either way reset the board position to empty 
		if findSwitches(turn,other,col)[0] or findSwitches(turn,other,row)[0] or findSwitches(turn,other,diagLeftRight)[0] or findSwitches(turn,other,diagRightLeft)[0]:
		  legalList.append([x,y])
		  #board[x][y] = legalSpot
		  board[x][y] = 0
		else:
		  board[x][y] = 0

  #return the list of legal moves
  return legalList

#This function takes a board, move, turn, and list of legal moves and makes a move
def makeMove(board,move,turn,legalList):
  #print(move)
  #print(legalList)
  
  #If the move is not on the legal return the same board and turn
  if move not in legalList:
	#print("Illegal Move")
	return [turn,board]

  #Set the point on the board where the move is to the test spot
  board[move[0]][move[1]] = testSpot

  #Find turn of the other
  other = -1*turn

  #Get the column, row, and both diagonals that the test piece is in
  col = board[:,move[1]]
  row = board[move[0]]
  diagLeftRight = board.diagonal(move[1]-move[0])
  #This is a bit harder as you need to flip the board and then get a changed 
  # diagonal here. This works though, by shifting it by y and x
  diagRightLeft = np.fliplr(board).diagonal(((boardSide-1)-move[1])-move[0])

  #This gets the list of all the moves for the column, row, and diagonals
  #However they are just the indexes in the array of the column, row, and 
  # diagonals. So they need to be put on the board properly
  colChanges = findSwitches(turn,other,col)[1]
  rowChanges = findSwitches(turn,other,row)[1]
  diagLeftRightChanges = findSwitches(turn,other,diagLeftRight)[1]
  diagRightLeftChanges= findSwitches(turn,other,diagRightLeft)[1]

  #For each column changes, change the point there to turn
  #You keep the same Y for move and just adjust the X value
  # with the move point in the middle
  for c in colChanges:
	board[move[0]+c][move[1]] = turn

  #For each row changes, change the point there to turn
  #You keep the same X for move and just adjust the Y value
  # with the move point in the middle
  for r in rowChanges:
	board[move[0]][move[1]+r] = turn

  #For each Diagonal Right to Left change, change the point there to turn
  #You add the value to X and Y as you move up
  for dlr in diagLeftRightChanges:
	board[move[0]+dlr][move[1]+dlr] = turn

  #For each Diagonal Left to Right change, change the point there to turn
  #You add the value to X and subtract from Y as you go down
  for drl in diagRightLeftChanges:
	board[move[0]+drl][move[1]-drl] = turn

  #Set the point of hte move to turn
  board[move[0]][move[1]] = turn
  
  
  # The turn was successful so return the next turn and the new board
  #print(str(turn)+"'s turn successful")
  return [other,board]

def pygameDisplay(board,gameOver,turn,legalList):
	screen.fill((255,255,255))

	numBlack = 0
	numWhite = 0

	for x in range(boardSide):
		for y in range(boardSide):
			if [x,y] in legalList:
				color = (2, 154, 255)
				size = 3
			else:
				color = (0,0,0)
				size = 1

			# text = font.render(str(x)+" , "+str(y), 1, (0,0,0))
			# screen.blit(text,[(offset/2)+y*block_size,(offset/2)+x*block_size])
			rect = pygame.Rect(offset+y*block_size,offset+x*block_size, block_size, block_size)
			pygame.draw.rect(screen, color, rect, size)

	for x in range(boardSide):
		for y in range(boardSide):
			piece = board[x][y]
			if piece == black:
				numBlack += 1
				pygame.draw.circle(screen, (0,0,0), (int(offset+(block_size/2)+y*block_size),int(offset+(block_size/2)+x*block_size)), radius)
			if piece == white:
				numWhite += 1
				pygame.draw.circle(screen, (0,0,0), (int(offset+(block_size/2)+y*block_size),int(offset+(block_size/2)+x*block_size)), radius,2)

	text = font.render("Black: "+str(numBlack)+" White: "+str(numWhite), 1, (0,0,0))
	screen.blit(text,[width/2-offset*2,offset/3])

	if numBlack+numWhite == boardSide**2:
		gameOver = True

	if not gameOver:
		if turn == black:
			text = font.render("Black's Turn", 1, (0,0,0))
		else:
			text = font.render("White's Turn", 1, (0,0,0))
		screen.blit(text,[width/2-offset,height-offset/1.5])

	else:
		if numBlack > numWhite:
			text = font.render("Black Wins!", 1, (0,0,0))
		elif numWhite > numBlack:
			text = font.render("White Wins!", 1, (0,0,0))
		else:
			text = font.render("It's a Tie!", 1, (0,0,0))
		screen.blit(text,[width/2-offset,height-offset/1.5])

def getLegal(guess,legalList):
  indexArray = np.asarray([[[y,x] for x in range (boardSide)] for y in range(boardSide)]).reshape(boardSide**2,2)
  print("Guessed Moves")
  print(np.round(guess*100))

  sumAll = 0
  for i in range(boardSide):
	for j in range(boardSide):
	  if [i,j] not in legalList:
		guess[i][j] = 0.0
	  else:
		sumAll += guess[i][j]
			  
  print("Legal Moves")
  print(np.round(guess*100))
  #print(np.round(guess.reshape((boardSide,boardSide))*100))			  
  guess = np.ndarray.flatten(guess/sumAll)
  #Now we need to choose a move
  choice = np.random.choice(boardSide**2, p=guess)
  print("")
  return(list(indexArray[choice]))

running = True
timer_event = pygame.USEREVENT + 1
pygame.time.set_timer(timer_event, 100)

turn = black

board = setUp()
legalList = findLegal(board,turn)
pygameDisplay(board,gameOver,turn,legalList)
pygame.display.flip()

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

		legalList = findLegal(board,turn)
		if len(legalList) == 0:
			#print("No Moves for",turn)
			turn = -1*turn
			legalList = findLegal(board,turn)
			if len(legalList) == 0:
				gameOver = True

		pygameDisplay(board,gameOver,turn,legalList)
		pygame.display.flip()

		if not gameOver:
			if turn == black and event.type == pygame.MOUSEBUTTONUP:
				pos = pygame.mouse.get_pos()
				xLoc = int((pos[1]-offset)/block_size)
				yLoc = int((pos[0]-offset)/block_size)
				(turn,board) = makeMove(board,[xLoc,yLoc],turn,legalList)
				pygameDisplay(board,gameOver,turn,legalList)
				pygame.display.flip()


			elif turn == white:
				time.sleep(0.5)
				inputlayer = board.reshape(1,boardSide**2)
				legalList = findLegal(board,turn)
				print("The Board")
				displayBoard(board)
				guess = model.predict(inputlayer*turn, batch_size=1, verbose=0)
				guess = guess[0].reshape(boardSide,boardSide)
				guessMove = getLegal(guess,legalList)
				(turn,board) = makeMove(board,guessMove,turn,legalList)
				pygameDisplay(board,gameOver,turn,legalList)
				pygame.display.flip()

	

