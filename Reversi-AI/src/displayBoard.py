# displayBoard.py
from graphics import *
from revfuncs import *
import time
def isDangerousMove(x, y):
    return ((x==1 and y==0) or (x==1 and y==1) or (x==0 and y==1)
            or (x==6 and y==7) or (x==6 and y==6) or (x==7 and y==6)
            or (x==0 and y==6) or (x==1 and y==6) or (x==1 and y==7)
            or (x==6 and y==0) or (x==6 and y==1) or (x==7 and y==1))

def displayBoard():
    size = 540
    win = GraphWin("Reversi", size, size)
    rect = Rectangle(Point(size/9,size/9),Point(size,size))
    rect.draw(win)
    for i in range(1,9):
        line = Line(Point(size/9,size/9*i),Point(size,size/9*i))
        line.setWidth(5)
        line.draw(win)
        linev = Line(Point(size/9*i,size/9),Point(size/9*i,size))
        linev.setWidth(5)
        linev.draw(win)
        label = Text(Point(size/9*i+30,30),'%d' % i)
        label.setSize(16)
        label.draw(win)
        labelv = Text(Point(30,size/9*i+30),'%d' % i)
        labelv.setSize(16)
        labelv.draw(win)
    return win
           

def drawBoard(board,win):
    for i in range(8):
        for j in range(8):
            if board[i][j] == 'X':
                cir = Circle(Point(i*60+90,j*60+90),20)
                cir.setFill("black")
                cir.draw(win)
            elif board[i][j] == 'O':
                cir = Circle(Point(i*60+90,j*60+90),20)
                cir.setFill("white")
                cir.draw(win)
            elif board[i][j] == '.':
                text = Text(Point(i*60+90,j*60+90),'X')
                text.setSize(30)
                text.draw(win)
#            elif board[i][j] == 'f':
    return win

def getPlayerMoveMouse(board,playerTile,win):
    DIGITS1TO8 = '0 1 2 3 4 5 6 7'.split()
    while True:
        print('Enter your move, or type quit to end the game, or hints to turn off/on hints.')
        
        point = win.getMouse()
        x = int((point.getX()-60)/60)
        y = int((point.getY()-60)/60)
        print([x,y])
        
        if isValidMove(board, playerTile, x, y) == False:
            print('That is not a valid move. Click on the board again.')
            continue
        else:
            break
        
            
    
    return [x,y]

