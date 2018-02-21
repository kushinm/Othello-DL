import numpy as np
import pygame

boardSide = 6
black = 1
white = 2


(width, height) = (600, 600)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Reversi')

block_size = width/(boardSide+2)
radius = block_size/3
offset = (height-(block_size*boardSide))/2

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
			tempChanges.append((x+1)*-1)
		elif backArray[x] == turn:
			foundTurn = True
			break
		elif backArray[x] == 0:
			break


	if foundTurn:
		changeLocs += tempChanges

	tempChanges = []
	foundTurn = False	 

	for x in range(len(forwardArray)):
		if forwardArray[x] == other:
			tempChanges.append(x)
		elif forwardArray[x] == turn:
			foundTurn = True
			break
		elif forwardArray[x] == 0:
			break

	if foundTurn:
		changeLocs+=tempChanges

	if len(changeLocs) > 0:
		legalMove = True

	return [legalMove,changeLocs]


def makeMove(board,move,turn):
	if board[move[0]][move[1]] == black or board[move[0]][move[1]] == white:
		print "Illegal Move: You cannot place a piece on another piece"
		return turn

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
		board[move[0]][move[1]] = 0
		return turn

	for c in colResults[1]:
		board[move[0]+c][move[1]] = turn

	for r in rowResults[1]:
		board[move[0]][move[1]+r] = turn

	for dlr in diagLeftRightResults[1]:
		board[move[0]+dlr][move[1]+dlr] = turn

	for drl in diagRightLeftResults[1]:
		board[move[0]+drl][move[1]-drl] = turn

	
	board[move[0]][move[1]] = turn

	print str(turn)+"'s turn successful"
	return other

def pygameDisplay(board):
	screen.fill((255,255,255))
	for x in range(boardSide):
		for y in range(boardSide):
			rect = pygame.Rect(offset+x*block_size,offset+y*block_size, block_size, block_size)
			pygame.draw.rect(screen, (0,0,0), rect, 1)

	for x in range(boardSide):
		for y in range(boardSide):
			piece = board[x][y]
			if piece == 1:
				pygame.draw.circle(screen, (0,0,0), (offset+(block_size/2)+x*block_size,offset+(block_size/2)+y*block_size), radius)
			if piece == 2:
				pygame.draw.circle(screen, (0,0,0), (offset+(block_size/2)+x*block_size,offset+(block_size/2)+y*block_size), radius,2)




board = setUp()
displayBoard(board)

pygameDisplay(board)
pygame.display.flip()

running = True
turn = black
while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		if event.type == pygame.MOUSEBUTTONUP:
			pos = pygame.mouse.get_pos()
			xLoc = (pos[0]-offset)/block_size
			yLoc = (pos[1]-offset)/block_size
			turn = makeMove(board,(xLoc,yLoc),turn)
			pygameDisplay(board)
			pygame.display.flip() 

	

