# function test
from revfuncs import *
from myAI import *

# compare two algorithms by having them play against each other for many times
# input algorithm 1 and algorithm 2 and the board to start with
# out put the number of time that algorithm 1 wins 
def agent_compare(algo_1,algo_2,total):
    result = []
    run = 0
    win = 0
    tie = 0
    
    
    while run < total:
#        window = displayBoard()
        mainBoard = getNewBoard()
        resetBoard(mainBoard)
        run = run + 1
        if run < total/2:
            playerTile = 'X'
            computerTile = 'O'
        else:
            playerTile = 'O'
            computerTile = 'X'
        turn = whoGoesFirst()
        
        while True:
            if turn == 'player':
                # Player's turn.
#                drawBoard(mainBoard, window)
#                showPoints(mainBoard, playerTile, computerTile)
                x,y = algo_1(mainBoard, playerTile)
                makeMove(mainBoard, playerTile, x, y)
                if getValidMoves(mainBoard, computerTile) == []:
                    break
                else:
                    turn = 'computer'

            else:
                # Computer's turn.
 #               drawBoard(mainBoard, window)
 #               showPoints(mainBoard, playerTile, computerTile)
 #               time.sleep(1)
                x, y = algo_2(mainBoard, computerTile)
                makeMove(mainBoard, computerTile, x, y)
                if getValidMoves(mainBoard, playerTile) == []:
                    break
                else:
                    turn = 'player'
        scores = getScoreOfBoard(mainBoard)
        if scores[playerTile] > scores[computerTile]:
            win = win + 1
        elif scores[playerTile] == scores[computerTile]:
            tie = tie + 1
#        window.close()
    lose = total - win - tie
#    print(lose)
    result = [win,tie,lose]
    print(result)





print(agent_compare(maximum_agent,frontier_agent,10))

