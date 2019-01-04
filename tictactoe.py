#----------------------------------------------------
# Author: Nantha Kumar Sunder
# Description: Program to tic tac toe game
#               using Q-Learning
#----------------------------------------------------

import random
import csv
import os

#---------------------------------------------------
# class for tictactoe
class tictactoe:
    #  tictactoe board
    def __init__(self, player='X', board=' '*9):
          # 1 for AI plays first 
          self.player = player
          self.winner = None
          self.board = board

    def printBoard(self):
        # This function prints out the board that it was passed.
        # "board" is a list of 10 strings representing the board (ignore index 0)
        print(' ' + self.board[0] + ' | ' + self.board[1] + ' | ' + self.board[2])
        print('-----------')
        print(' ' + self.board[3] + ' | ' + self.board[4] + ' | ' + self.board[5])
        print('-----------')
        print(' ' + self.board[6] + ' | ' + self.board[7] + ' | ' + self.board[8])

    def resetBoard(self):
        self.board = ' '*9

    def getMove(self):
        # get input from the user
        posMoves = [i+1 for i in range(9) if self.board[i] == ' ']
        userMove = None
        while not userMove:
            idx = int(input('Choose move for {}, from {} : '.format(self.player, posMoves)))
            if any([i==idx for i in posMoves]):
                userMove = self.board[:idx-1] + self.player + self.board[idx:]
        self.makeMove(userMove)

    def makeMove(self, move):
        self.board = move
        isGameOver = self.isGameOver(condition=False)
        if isGameOver:
            self.winner = self.player
        elif self.player == 'X':
            self.player = 'O'
        else:
            self.player = 'X'

    def isNotOccupied(self, move):
        return self.board[move] == ' '

    def isBoardFull(self):
        for pos in self.board:
            if pos == ' ':
                return 0
        return 1

    def isGameOver(self, condition=True):
        if condition:
            if self.player == 'X':
                letter = 'O'  # type: str
            else:
                letter = 'X'
        else:
            letter = self.player
        won = ((self.board[6] == letter and self.board[7] == letter and self.board[8] == letter) or # across the top
        (self.board[3] == letter and self.board[4] == letter and self.board[5] == letter) or # across the middle
        (self.board[0] == letter and self.board[1] == letter and self.board[2] == letter) or # across the bottom
        (self.board[6] == letter and self.board[3] == letter and self.board[0] == letter) or # down the left side
        (self.board[7] == letter and self.board[4] == letter and self.board[1] == letter) or # down the middle
        (self.board[8] == letter and self.board[5] == letter and self.board[2] == letter) or # down the right side
        (self.board[6] == letter and self.board[4] == letter and self.board[2] == letter) or # diagonal
        (self.board[8] == letter and self.board[4] == letter and self.board[0] == letter)) # diagonal
        if won == 1:
            self.winner = letter
        else:
            self.winner = None
        return won

    def playFirst(self):
        return random.randint(0,1)

# ---------------------------------------------------
# class for AI
class ai:
    def __init__(self, gameType, alpha, epsilon, aiLetter, gamma=0.95, maxEpsilson=1.0, minEpsilon=0.01):
        self.qTable = dict()
        self.game = gameType
        self.alpha = alpha
        self.epsilon = epsilon
        self.aiLetter = aiLetter
        self.gamma = gamma
        self.maxEpsilon = maxEpsilson
        self.minEpsilon = minEpsilon

    def write2csv(self):
        with open('qTable.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['State', 'Value'])
            allStates = list(self.qTable.keys())
            allStates.sort()
            for state in allStates:
                writer.writerow([state, self.qTable[state]])

    def loadFromCsv(self):
        reader = csv.reader(open('qTable.csv', 'r'))
        for k, v in reader:
            if v == 'Value':
                self.qTable[k] = v
            else:
                self.qTable[k] = float(v)

    def trainFromEpisode(self, episodeNum=50000 ):
        decayRate = (self.maxEpsilon - self.minEpsilon) / episodeNum
        for episode in range(episodeNum):
            self.epsilon = self.epsilon * decayRate
            self.learnFromEpisode()
        print("Training Done")

    def learnFromEpisode(self):
        NewGame = self.game()
        _, move = self.getMove(NewGame)
        while move:
          NewGame.makeMove(move)
          reward = self.giveReward(NewGame)
          nextReward = 0.0
          selectedMove = None
          if ( not NewGame.isBoardFull() and not NewGame.isGameOver() ):
              bestNextMove, selectedMove = self.getMove(NewGame)
              nextReward = self.qTableValue(bestNextMove)
          currentQValue = self.qTableValue(move)
          cummulativeReward = reward + self.gamma * nextReward
          self.qTable[move] = currentQValue + self.alpha * (cummulativeReward - currentQValue)
          move = selectedMove

    def getMoveVsHuman(self):
        posMoves = self.getQTableValues(self.possibleMoves(self.game))
        ## exploitation
        if self.game.player == 'X':
            move = self.maxExploit(posMoves)
        else:
            move = self.minExploit(posMoves)
        
        self.game.makeMove( move )

    def getMove(self, game):
        posMoves = self.getQTableValues(self.possibleMoves(game))
        ## exploitation
        if game.player == 'X':
            move = self.maxExploit(posMoves)
        else:
            move = self.minExploit(posMoves)
        randomMove = move
        ## exploration
        if random.random() < self.epsilon:
            randomMove = random.choice(list(posMoves.keys()))
        return (move, randomMove)

    def possibleMoves(self, game):
        posStates = list()
        for i in range(0,9):
            if game.board[i] == ' ':
                tempBoard = game.board[:i] + game.player + game.board[i+1:]
                posStates.append(tempBoard)
        return posStates

    def qTableValue(self, state):
        return self.qTable.get(state, 0.0)
    
    def getQTableValues(self, posStatesDict):
        return dict((state, self.qTableValue(state)) for state in posStatesDict)

    def maxExploit(self, posStatesDict):
        maxValue = max(posStatesDict.values())
        chosenState = random.choice([state for state,\
         val in posStatesDict.items() if val == maxValue])
        return chosenState

    def minExploit(self, posStatesDict):
        minValue = min(posStatesDict.values())
        chosenState = random.choice([state for state,\
         val in posStatesDict.items() if val == minValue])
        return chosenState        

    def giveReward(self, game):
        if game.winner == 'X':
            return 1.0
        elif game.winner == 'O':
            return -1.0
        else:
            return 0.0
##---------------------------------------------------
## to play against AI
def play():
    ticGame = tictactoe('X', ' '*9)
    firstTurn = ticGame.playFirst()
    print(firstTurn)
    if firstTurn == 1:
        aiCharacter = 'X'
        userLetter = 'O'
    else:
        aiCharacter = 'O'
        userLetter = 'X'
    ticGame.resetBoard()
    aiAgent = ai(ticGame, alpha = 1, epsilon = 0.1, aiLetter = aiCharacter)
    aiAgent.loadFromCsv()
    turn = firstTurn
    gameOn = 1
    while gameOn == 1:
        ## 1 means AI turn
        if turn == 1:
            aiAgent.getMoveVsHuman()
            if ticGame.isGameOver(condition=False):
                ticGame.printBoard()
                print("AI has won the game!!!!")
                gameOn =0
            else:
                if ticGame.isBoardFull():
                    ticGame.printBoard()
                    print("The game is a tie!!")
                    gameOn = 0
                else:
                    turn = 0
        # user turn
        else:
            ticGame.printBoard()
            ticGame.getMove()
            if ticGame.isGameOver(condition=False):
                ticGame.printBoard()
                print("You have won the game!!!!")
                gameOn =0
            else:
                if ticGame.isBoardFull():
                    ticGame.printBoard()
                    print("The game is a tie!!")
                    gameOn = 0
                else:
                    turn = 1

# ---------------------------------------------------
# to Train the AI
def trainAI():
    aiAgent = ai(tictactoe, alpha=1, epsilon=1, aiLetter='X')
    aiAgent.trainFromEpisode()
    aiAgent.write2csv()
 
# ---------------------------------------------------
# main
os.system('clear')
print("Welcome to Tic tac toe using Q learning")
while True:
    # Displaying instructions
    print("Enter: 1 to play")
    print("Enter: 2 to train the AI")
    print("Enter any other character top exit")
    # taking user input
    userInput = input()
    # response for user selection
    if userInput == '1':
        play() # if the user input is 1
    elif userInput == '2':
        trainAI() # if the user input is 2
    else:
        print("Please enter either 1 or 2")  # Do Nothing
    # getting the user input
    restart = input("Do you want to continue? [y/n]")
    if restart != 'y':
        break
