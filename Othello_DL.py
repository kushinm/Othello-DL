import numpy as np
import random
import math

boardSide = 6
empty = 0
black = 1
white = -1
testSpot = 3
legalSpot = 9

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
  print("Starting Board")
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



'''
turn = black
gameOver = False
board = setUp()
legalList = findLegal(board,turn)
displayBoard(board)

while not gameOver:
  [turn,board] = makeMove(board,random.choice(legalList),turn,legalList)
  displayBoard(board)
  legalList = findLegal(board,turn)
   #How you tell if game is over
  if len(legalList) == 0:
    print("No Moves for",turn)
    turn = -1*turn
    legalList = findLegal(board,turn)
  if len(legalList) == 0:
    print("No Moves for",turn)
    gameOver = True
    print("GameOver")
    print("The Winner is",getWinner(board))'''

'''def softmax(x):
    #Compute softmax values for each sets of scores in x.
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def relu(x):
  return np.maximum(x, 0)

def sigmoid(val):
  return 1/(1+math.e**(-1*val))-0.5

class ThreeLayerNetwork:
   def __init__(self,inputSize,hiddenSize,outputSize):
      # Initiate first layer weights / middle to output weights
      #Input
      self.inputLayer = np.zeros((inputSize,1))
      self.inputLayerWeights = np.random.random((inputSize,hiddenSize))

      #Hidden
      self.hiddenLayer = np.zeros((hiddenSize,1))
      self.hiddenLayerWeights = np.random.random((hiddenSize,outputSize))

      #Output
      self.outputLayer = np.zeros((outputSize,1))
      
      self.indexArray = np.asarray([[[x,y] for x in range (boardSide)] for y in range(boardSide)]).reshape(boardSide**2,2)
   
   def forwardPass(self,board,turn):
      self.inputLayer = board.reshape(self.inputLayer.shape)
      self.hiddenLayer = sigmoid(np.sum(self.inputLayerWeights*self.inputLayer,axis=0)).reshape(self.hiddenLayer.shape)
      self.outputLayer = softmax(np.sum(self.hiddenLayerWeights*self.hiddenLayer,axis=0)).reshape(self.outputLayer.shape)
      
      boardSide = len(board)
      legalList = findLegal(board,turn)
      self.outputLayer = self.outputLayer.reshape(boardSide,boardSide)
      
      sumAll = 0
      for i in range(boardSide):
        for j in range(boardSide):
          if [i,j] not in legalList:
             self.outputLayer[i][j] = 0.0
          else:
             sumAll += self.outputLayer[i][j]
             
            
      self.outputLayer = np.ndarray.flatten(self.outputLayer/sumAll)
     
      
      #Now we need to choose a move      
      choice = np.random.choice(boardSide**2, p=self.outputLayer)
      
      return(self.indexArray[choice])
    


      
  
#network = ThreeLayerNetwork(boardSide**2,10,boardSide**2)
#print(legalList)
#for x in range(20):
  #print(network.forwardPass(board,turn))

#How to Train
#If a move is correct the output has that one be 1, and all other board spots are 0
#If a move is considered incorrect the output is 0, and all other board spots are 
#   probalistically increased with the wrong amount removed
#Do all backprop at once, so just one update witht he whole change


 def backprop()'''



import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
import keras.preprocessing.text
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
#random.seed(7)

model = Sequential()
model.add(Dense(boardSide**2, activation='relu', input_dim=(boardSide**2)))
model.add(Dropout(0.5))
model.add(Dense(2*boardSide**2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(boardSide**2, activation='softmax'))
Adam= Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_absolute_error',
              optimizer=Adam,
              metrics=['accuracy'])

def getLegal(guess,legalList):
  indexArray = np.asarray([[[y,x] for x in range (boardSide)] for y in range(boardSide)]).reshape(boardSide**2,2)
  
  sumAll = 0
  for i in range(boardSide):
    for j in range(boardSide):
      if [i,j] not in legalList:
        guess[i][j] = 0.0
      else:
        sumAll += guess[i][j]
              
  guess = np.ndarray.flatten(guess/sumAll)
      
  #Now we need to choose a move
  choice = np.random.choice(boardSide**2, p=guess)
  
  #print(guess.reshape(boardSide,boardSide))
  #print(guess)
  #print(indexArray)
  #print(list(indexArray[choice]))
      
  return(list(indexArray[choice]))


#turn = black
#gameOver = False
#board = setUp()
#legalList = findLegal(board,turn)
#displayBoard(board)
epochs = 100

#player1Boards = np.zeros(boardSide**2)
#player2Boards = np.zeros(boardSide**2)
#player1Guesses = np.zeros(boardSide**2)
#player2Guesses = np.zeros(boardSide**2)
#player1Moves = np.zeros(2)
#player2Moves = np.zeros(2)






for i in range(epochs):
  print(i)
  #print(model.get_weights())
  turn = black
  gameOver = False
  board = setUp()
  legalList = findLegal(board,turn)
  player1Boards = np.zeros(boardSide**2)
  player2Boards = np.zeros(boardSide**2)
  player1Guesses = np.zeros(boardSide**2)
  player2Guesses = np.zeros(boardSide**2)
  player1Moves = np.zeros(2)
  player2Moves = np.zeros(2)
  nturns = 0
  gameOver = False
  while not gameOver:  
    nturns = nturns+1
    #displayBoard(board)
    legalList = findLegal(board,turn)
    
    if len(legalList) == 0:
      #print("No Moves for",turn)
      turn = -1*turn
      legalList = findLegal(board,turn)
    if len(legalList) == 0:
      #print("No Moves for",turn)
      gameOver = True
      #print("GameOver")
      winner = getWinner(board)
      #print("The Winner is",winner)
    
    if not gameOver:
      inputlayer = board.reshape(1,boardSide**2)
      legalList = findLegal(board,turn)
      guess = model.predict(inputlayer*turn, batch_size=1, verbose=0, steps=None)
      guess = guess[0].reshape(boardSide,boardSide)
      guessMove = getLegal(guess,legalList)
      #print(guess)
      #print(legalList)

      if turn == 1:
        player1Guesses = np.vstack((player1Guesses,guess.reshape(boardSide**2)))
        player1Boards = np.vstack((player1Boards,inputlayer*turn))
        player1Moves = np.vstack((player1Moves,guessMove))
      else:
        player2Guesses = np.vstack((player2Guesses,guess.reshape(boardSide**2)))
        player2Boards = np.vstack((player2Boards,inputlayer*turn))
        player2Moves = np.vstack((player2Moves,guessMove))

      [turn,board] = makeMove(board,guessMove,turn,legalList)
  
  #The first for all of them are 0 so ignore the first array in each

  
  player1Moves = player1Moves[1:]
  player1Guesses = player1Guesses[1:]
  player1Boards = player1Boards[1:]
  player2Moves = player2Moves[1:]
  player2Guesses = player2Guesses[1:]
  player2Boards = player2Boards[1:]
 
  
  if getWinner(board)== black:
    winTarget = np.zeros((len(player1Moves),boardSide,boardSide))
    loseTarget = np.zeros((len(player2Moves),boardSide,boardSide))
    
    for i in range(len(player1Moves)):
  
      winTarget[i][int(player1Moves[i][0])][int(player1Moves[i][1])]= 1           #haven't dealt with the first entries being 0s in playerBoards, and playerMoves yet
    for i in range(len(player2Moves)):

      loseTarget[i]= player2Guesses[i].reshape(boardSide,boardSide)
      loseTarget[i,int(player2Moves[i][0]),int(player2Moves[i][1])] = 0
      
    loseTarget = np.divide(loseTarget, sum(loseTarget))
    xTrain = np.vstack((player1Boards,player2Boards))
    yTrain = np.vstack((winTarget.reshape((len(player1Moves),boardSide**2)),loseTarget.reshape((len(player2Moves),boardSide**2))))
    #model.fit(player1Boards,winTarget.reshape((len(player1Moves),boardSide**2)),batch_size=len(player1Moves))
    #model.fit(player2Boards,loseTarget.reshape((len(player2Moves),boardSide**2)),epochs=1, batch_size=len(player2Moves))
                        
    model.fit(xTrain, yTrain, epochs=1, batch_size=len(player1Moves)+len(player2Moves))
  elif getWinner(board)== white:
    winTarget = np.zeros((len(player2Moves),boardSide, boardSide))
    loseTarget = np.zeros((len(player1Moves),boardSide, boardSide))
    
    for i in range(len(player2Moves)):
      winTarget[i][int (player2Moves[i][0])][int (player2Moves[i][1])]= 1    
    for i in range(len(player1Moves)):
      loseTarget[i]= player1Guesses[i].reshape(boardSide,boardSide) 
      loseTarget[i,int(player1Moves[i][0]),int(player1Moves[i][1])] = 0
      
    loseTarget = np.divide( loseTarget, sum(loseTarget))
    xTrain = np.vstack((player2Boards,player1Boards))
    yTrain = np.vstack((winTarget.reshape((len(player2Moves),boardSide**2)),loseTarget.reshape((len(player1Moves),boardSide**2))))
                        
    #model.fit(player2Boards, winTarget.reshape((len(player2Moves),boardSide**2)),epochs=1,batch_size=len(player2Moves))
    #model.fit(player1Boards,loseTarget.reshape((len(player1Moves),boardSide**2)),epochs=1, batch_size=len(player1Moves))       
    model.fit( xTrain, yTrain, epochs=1, batch_size=len(player1Moves)+len(player2Moves))
    
  #If there's a tie don't train on it
  else:
    pass
  
testBoard = setUp().reshape(1,boardSide**2)
model.predict(testBoard)

 