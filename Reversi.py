import numpy as np

boardSide = 6
black = 1
white = 2

def setUp():
	#for the board 0 is empty, 1 is black, and 2 is white
	boardArray = np.zeros((boardSide,boardSide))
	mid = boardSide/2 - 1
	boardArray[mid][mid] = white
	boardArray[mid+1][mid] = black
	boardArray[mid][mid+1] = black
	boardArray[mid+1][mid+1] = white
	print "Starting Board"
	return boardArray

def displayBoard(board):
	for b in board:
		print b
	print 

def findSwitches(turn,other,array):

	newPieceLoc = np.where(array == 3)[0][0]

	legalMove = False
	changeLocs = []

	backArray = array[0:newPieceLoc][::-1]
	forwardArray =  array[newPieceLoc:]

	tempChanges = []
	foundTurn = False

	for x in range(len(backArray)):
		if backArray[x] == other:
			tempChanges.append(newPieceLoc-x-1)
		if backArray[x] == turn:
			foundTurn = True
			break

	if foundTurn:
		changeLocs += tempChanges

	tempChanges = []
	foundTurn = False	 

	for x in range(len(forwardArray)):
		if forwardArray[x] == other:
			tempChanges.append(newPieceLoc+x)
		if forwardArray[x] == turn:
			foundTurn = True
			break

	if foundTurn:
		changeLocs+=tempChanges

	if len(changeLocs) > 0:
		legalMove = True

	return [legalMove,changeLocs]



def makeMove(board,move,turn):
	if board[move[0]][move[1]] == black or board[move[0]][move[1]] == white:
		print "Illegal Move: You cannot place a piece on another piece"
		return board

	board[move[0]][move[1]] = 3

	if turn == black:
		other = white
	else:
		other = black

	legalMove = False

	col = board[:,move[1]]
	row = board[move[0]]
	diagLeftRight = board.diagonal(move[1]-move[0])
	diagRightLeft = np.fliplr(board).diagonal((5-move[1])-move[0])

	# print "Column",col
	# print "Row",row
	# print "Diag \\",diagLeftRight
	# print "Diag /",diagRightLeft

	colResults = findSwitches(turn,other,col)
	rowResults = findSwitches(turn,other,row)
	diagLeftRightResults = findSwitches(turn,other,diagLeftRight)
	diagRightLeftResults = findSwitches(turn,other,diagRightLeft)

	legalMove = colResults[0] or rowResults[0] or diagLeftRightResults[0] or diagRightLeftResults[0]

	if not legalMove:
		print "Illegal Move: Nothing flipped"
		return board

	for c in colResults[1]:
		board[c][move[1]] = turn

	for r in rowResults[1]:
		board[move[0]][r] = turn

	
	board[move[0]][move[1]] = turn

	print str(turn)+"'s turn successful"
	return board


board = setUp()
displayBoard(board)
board = makeMove(board,(1,2),black)
displayBoard(board)
board = makeMove(board,(3,1),white)
displayBoard(board)
board = makeMove(board,(4,2),black)
displayBoard(board)

