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
	return boardArray

def displayBoard(board):
	for b in board:
		print b
	print 

def makeMove(board,move,turn):
	board[move[0]][move[1]] = 3
	if turn == 1:
		other = 2
	else:
		other = 1

	col = board[:,move[1]]
	row = board[move[0]]
	diagLeftRight = board.diagonal(move[0])
	diagRightLeft = np.fliplr(board).diagonal(move[1])

	print "Column",col
	print "Row",row
	print "Diag \\",diagLeftRight
	print "Diag /",diagRightLeft
	print

	print
	return board


board = setUp()
displayBoard(board)
board = makeMove(board,(1,2),black)
displayBoard(board)

