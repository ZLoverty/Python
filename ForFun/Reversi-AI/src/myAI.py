from revfuncs import *
from displayBoard import *
from graphics import *
import time


def maximum_agent(board, AITile):
    possibleMoves = getValidMoves(board, AITile)
#    print('possible: ',possibleMoves)
    safeMoves = []
    bestMove = possibleMoves[0]
    
    random.shuffle(possibleMoves)
    for x, y  in possibleMoves:
        if isOnCorner(x, y):
            return [x,y]

        # avoid dangerous moves
        if not isDangerousMove(x, y):
            safeMoves.append([x, y])
    if not safeMoves == []:
        possibleMoves = safeMoves
    bestScore = -1
    for x, y in possibleMoves:
        dupeBoard = getBoardCopy(board)
        makeMove(dupeBoard, AITile, x, y)
        score = getScoreOfBoard(dupeBoard)[AITile]
        if score > bestScore:
            bestMove = [x, y]
            bestScore = score
    return bestMove
    
# Minimax algorithm, compute 2 steps ahead to give minimal loss
def minimax_agent(board, AITile):
    if AITile == 'X':
        AI2Tile = 'O'
    else:
        AI2Tile = 'X'
    safeMoves = []
    possibleMoves = getValidMoves(board, AITile)
    random.shuffle(possibleMoves)
    for x, y  in possibleMoves:
        if isOnCorner(x, y):
            return [x,y]
        # avoid dangerous moves
        if not isDangerousMove(x, y):
            safeMoves.append([x, y])
    if not safeMoves == []:
        possibleMoves = safeMoves
    dupeBoard = getBoardCopy(board)
    moveScore = -1
    random.shuffle(possibleMoves)
    for x, y in possibleMoves:
#        dupeBoard[x][y] = '.'
        makeMove(dupeBoard, AITile, x, y)
        possibleMoves1 = getValidMoves(dupeBoard, AI2Tile)
        random.shuffle(possibleMoves1)
        lowest = 64
        move = [x, y]
        for z, w in possibleMoves1:
            dupeBoard1 = getBoardCopy(dupeBoard)
            makeMove(dupeBoard1, AI2Tile, z, w)
            possibleMoves2 = getValidMoves(dupeBoard1, AITile)
            random.shuffle(possibleMoves2)
            bestScore = -1;
            for u, v in possibleMoves2:
                makeMove(dupeBoard1, AITile, u, v)
                score = getScoreOfBoard(dupeBoard1)[AITile]
                # find the move with best score for AI
                if score > bestScore:
                    bestScore = score
                    bestMove = [u, v]
            # find the worst situation that can happen to AI
            if bestScore < lowest:
                lowest = bestScore
                lowestMove = [z, w]
        # find the best move for AI
        if lowest > moveScore:
            moveScore = lowest
            move = [x, y]
    return move
    
def evaporation_agent(board, AITile):
    possibleMoves = getValidMoves(board, AITile)
#    print('possible: ',possibleMoves)
    safeMoves = []
    bestMove = possibleMoves[0]
    random.shuffle(possibleMoves)
    for x, y in possibleMoves:
        if isOnCorner(x, y):
            return [x,y]

        # avoid dangerous moves
        if not isDangerousMove(x, y):
            safeMoves.append([x, y])
    if not safeMoves == []:
        possibleMoves = safeMoves
    bestScore = 64
    for x, y in possibleMoves:
        dupeBoard = getBoardCopy(board)
        makeMove(dupeBoard, AITile, x, y)
        score = getScoreOfBoard(dupeBoard)[AITile]
        if score < bestScore:
            bestMove = [x, y]
            bestScore = score
    return bestMove

def frontier_agent(board, AITile):
    possibleMoves = getValidMoves(board, AITile)
    random.shuffle(possibleMoves)
    if AITile == 'X':
        AI2Tile = 'O'
    else:
        AI2Tile = 'X'
    # use (opFrontier - myFrontier) as the evaluation of move
    bestScore = -64
    bestMove = possibleMoves[0]
    safeMoves = []
    for x, y in possibleMoves:
        if isOnCorner(x, y):
            return [x,y]

        # avoid dangerous moves
        if not isDangerousMove(x, y):
            safeMoves.append([x, y])
    if not safeMoves == []:
        possibleMoves = safeMoves
    dupeboard = getNewBoard()
    for x, y in possibleMoves:
        for i in range(8):
            for j in range(8):
                dupeboard[i][j] = board[i][j]

        makeMove(dupeboard, AITile, x, y)
        myFrontier = 0
        opFrontier = 0
        for i in range(8):
            for j in range(8):
                if dupeboard[i][j] == AITile:
                    for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
                        if isOnBoard(i+xdirection,j+ydirection):
                            
                            if board[i+xdirection][j+ydirection] == '':
                                myFrontier = myFrontier + 1
                                break
                            else:
                                continue
                if dupeboard[i][j] == AI2Tile:
                    for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
                        if isOnBoard(i+xdirection,j+ydirection):
                            if board[i+xdirection][j+ydirection] == '':
                                opFrontier = opFrontier + 1
                                break
                            else:
                                continue
        score = opFrontier - myFrontier
        if score > bestScore:
            bestScore = score
            bestMove = [x,y]
    return bestMove



def gameGow():
    while True:
        # Reset the board and game.
        mainBoard = getNewBoard()
        resetBoard(mainBoard)
        AI1Tile, AI2Tile = enterAI1Tile()
        showHints = False
        turn = whoGoFirst()
        print('The ' + turn + ' will go first.')
        
        while True:
            if turn =='player':
                drawBoard(mainBoard)
                showPoints(mainBoard, AI1Tile, AI2Tile)
                input('Press Enter to see the AI\'s move.')
                move = getAImove1(mainBoard, AI1Tile)
                makeMove(mainBoard, AI1Tile, move[0], move[1])

                if getValidMoves(mainBoard, AI2Tile) == []:
                    break
                else:
                    turn = 'computer'
            else:
                drawBoard(mainBoard)
                showPoints(mainBoard, AI1Tile, AI2Tile)
                input('Press Enter to see the computer\'s move.')
                x, y = getAImove2(mainBoard, AI2Tile, AI1Tile)
                makeMove(mainBoard, AI2Tile, x, y)
                
                if getValidMoves(mainBoard, AI2Tile) == []:
                    break
                else:
                    turn = 'player'
    drawBoard(mainBoard)
    print('Best Move: ',getAImove1(mainBoard,AI1Tile))

def gameGowo():
    while True:
        # Reset the board and game.
        mainBoard = getNewBoard()
        resetBoard(mainBoard)
        AI1Tile, AI2Tile = enterAI1Tile()
        showHints = False
        turn = whoGoFirst()
        print('The ' + turn + ' will go first.')

        AI1 = 'max score AI'
        AI2 = 'revised max score AI'
        while True:
            if turn =='AI1':
                move = getAImove1(mainBoard, AI1Tile)
                makeMove(mainBoard, AI1Tile, move[0], move[1])

                if getValidMoves(mainBoard, AI2Tile) == []:
                    break
                else:
                    turn = 'AI2'
            else:
                x, y = getAImove2(mainBoard, AI2Tile)
                makeMove(mainBoard, AI2Tile, x, y)
                
                if getValidMoves(mainBoard, AI1Tile) == []:
                    break
                else:
                    turn = 'AI1'
        drawBoard(mainBoard)
        scores = getScoreOfBoard(mainBoard)
        print('X scored %s points. O scored %s points.' % (scores['X'], scores['O']))

def singlePlayerGame():
    print('Welcome to Reversi!')
    win = displayBoard()
    while True:
        # Reset the board and game.
        mainBoard = getNewBoard()
        resetBoard(mainBoard)
        drawBoard(mainBoard,win)
        playerTile, computerTile = enterPlayerTile()
        showHints = False
        turn = whoGoesFirst()
        print('The ' + turn + ' will go first.')

    
        while True:
            if turn == 'player':
                # Player's turn.
                if showHints:
                    validMovesBoard = getBoardWithValidMoves(mainBoard, playerTile)
                    drawBoard(validMovesBoard)
                else:
                    drawBoard(mainBoard, win)
                    showPoints(mainBoard, playerTile, computerTile)
                    move = getPlayerMoveMouse(mainBoard, playerTile, win)
#                    move = getPlayerMove(mainBoard, playerTile)
                if move == 'quit':
                    print('Thanks for playing!')
                    sys.exit() # terminate the program
                elif move == 'hints':
                    showHints = not showHints
                    continue
                else:
                    makeMove(mainBoard, playerTile, move[0], move[1])
                    drawBoard(mainBoard, win)
                if getValidMoves(mainBoard, computerTile) == []:
                    break
                else:
                    turn = 'computer'

            else:
                # Computer's turn.
                drawBoard(mainBoard, win)
                showPoints(mainBoard, playerTile, computerTile)
 #               input('Press Enter to see the computer\'s move.')
                time.sleep(1)
                x, y = frontier_agent(mainBoard, computerTile)
                makeMove(mainBoard, computerTile, x, y)
                drawBoard(mainBoard,win)

                if getValidMoves(mainBoard, playerTile) == []:
                    break
                else:
                    turn = 'player'

        # Display the final score.
        drawBoard(mainBoard, win)
        scores = getScoreOfBoard(mainBoard)
        print('X scored %s points. O scored %s points.' % (scores['X'], scores['O']))
        if scores[playerTile] > scores[computerTile]:
            print('You beat the computer by %s points! Congratulations!' % (scores[playerTile] - scores[computerTile]))
        elif scores[playerTile] < scores[computerTile]:
            print('You lost. The computer beat you by %s points.' % (scores[computerTile] - scores[playerTile]))
        else:
            print('The game was a tie!')

        if not playAgain():
            break

if __name__ == '__main__':
    singlePlayerGame()

