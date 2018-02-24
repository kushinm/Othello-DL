import numpy as np
import pygame
import random

boardSide = 8
empty = 0
black = 1
white = 2
testSpot = 3
legalSpot = 9

gameOver = False

pygame.init()

(width, height) = (600, 600)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Reversi')

block_size = width/(boardSide+2)
radius = block_size/3
offset = (height-(block_size*boardSide))/2

font = pygame.font.Font(None, int(offset/1.5))

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

	newPieceLoc = np.where(array == testSpot)[0][0]

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
		elif backArray[x] == empty  or backArray[x] == legalSpot:
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
		elif forwardArray[x] == empty or forwardArray[x] == legalSpot:
			break

	if foundTurn:
		changeLocs+=tempChanges

	if len(changeLocs) > 0:
		legalMove = True

	return [legalMove,changeLocs]

def findLegal(board,turn):

	legalList = []

	for x in range(boardSide):
		for y in range(boardSide):
			if board[x][y] != black and board[x][y] != white:
				board[x][y] = testSpot

				if turn == black:
					other = white
				else:
					other = black

				col = board[:,y]
				row = board[x]
				diagLeftRight = board.diagonal(y-x)
				diagRightLeft = np.fliplr(board).diagonal(((boardSide-1)-y)-x)

				if findSwitches(turn,other,col)[0] or findSwitches(turn,other,row)[0] or findSwitches(turn,other,diagLeftRight)[0] or findSwitches(turn,other,diagRightLeft)[0]:
					legalList.append([x,y])
					board[x][y] = legalSpot
				else:
					board[x][y] = 0

	return legalList


def makeMove(board,move,turn,legalList):
	if move not in legalList:
		print "Illegal Move"
		return [turn,board]

	board[move[0]][move[1]] = testSpot

	if turn == black:
		other = white
	else:
		other = black

	col = board[:,move[1]]
	row = board[move[0]]
	diagLeftRight = board.diagonal(move[1]-move[0])
	diagRightLeft = np.fliplr(board).diagonal(((boardSide-1)-move[1])-move[0])

	colChanges = findSwitches(turn,other,col)[1]
	rowChanges = findSwitches(turn,other,row)[1]
	diagLeftRightChanges = findSwitches(turn,other,diagLeftRight)[1]
	diagRightLeftChanges= findSwitches(turn,other,diagRightLeft)[1]

	for c in colChanges:
		board[move[0]+c][move[1]] = turn

	for r in rowChanges:
		board[move[0]][move[1]+r] = turn

	for dlr in diagLeftRightChanges:
		board[move[0]+dlr][move[1]+dlr] = turn

	for drl in diagRightLeftChanges:
		board[move[0]+drl][move[1]-drl] = turn

	board[move[0]][move[1]] = turn

	print str(turn)+"'s turn successful"
	return [other,board]

def pygameDisplay(board,gameOver,turn):
	screen.fill((255,255,255))

	numBlack = 0
	numWhite = 0

	for x in range(boardSide):
		for y in range(boardSide):
			if board[x][y] == legalSpot:
				color = (0,255,0)
				size = 3
			else:
				color = (0,0,0)
				size = 1

			rect = pygame.Rect(offset+x*block_size,offset+y*block_size, block_size, block_size)
			pygame.draw.rect(screen, color, rect, size)

	for x in range(boardSide):
		for y in range(boardSide):
			piece = board[x][y]
			if piece == black:
				numBlack += 1
				pygame.draw.circle(screen, (0,0,0), (offset+(block_size/2)+x*block_size,offset+(block_size/2)+y*block_size), radius)
			if piece == white:
				numWhite += 1
				pygame.draw.circle(screen, (0,0,0), (offset+(block_size/2)+x*block_size,offset+(block_size/2)+y*block_size), radius,2)

	text = font.render("Black: "+str(numBlack)+" White: "+str(numWhite), 1, (0,0,0))
	screen.blit(text,[width/2-offset*2,offset/3])

	if not gameOver:
		if turn == black:
			text = font.render("Black's Turn", 1, (0,0,0))
		else:
			text = font.render("White's Turn", 1, (0,0,0))
		screen.blit(text,[width/2-offset,height-offset/1.5])


	if numBlack+numWhite == boardSide**2 or gameOver:
		if numBlack > numWhite:
			text = font.render("Black Wins!", 1, (0,0,0))
		elif numWhite > numBlack:
			text = font.render("White Wins!", 1, (0,0,0))
		else:
			text = font.render("It's a Tie!", 1, (0,0,0))
		screen.blit(text,[width/2-offset,height-offset/1.5])


running = True
timer_event = pygame.USEREVENT + 1
pygame.time.set_timer(timer_event, 100)

turn = black

board = setUp()
legalList = findLegal(board,turn)
pygameDisplay(board,gameOver,turn)
pygame.display.flip()

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		if event.type == pygame.MOUSEBUTTONUP:
			pos = pygame.mouse.get_pos()
			xLoc = (pos[0]-offset)/block_size
			yLoc = (pos[1]-offset)/block_size
			[turn,board] = makeMove(board,[xLoc,yLoc],turn,legalList)
			legalList = findLegal(board,turn)
			if len(legalList) == 0:
				print "No Moves for",turn
				if turn == black:
					turn = white
				else:
					turn = black
				legalList = findLegal(board,turn)
				if len(legalList) == 0:
					gameOver = True

			pygameDisplay(board,gameOver,turn)
			pygame.display.flip()
		elif event.type == timer_event:
			if len(legalList) > 0:
				move = random.choice(legalList)
				[turn,board] = makeMove(board,move,turn,legalList)
				legalList = findLegal(board,turn)
			if len(legalList) == 0:
				print "No Moves for",turn
				if turn == black:
					turn = white
				else:
					turn = black
				legalList = findLegal(board,turn)
				if len(legalList) == 0:
					gameOver = True

			pygameDisplay(board,gameOver,turn)
			pygame.display.flip()

	

