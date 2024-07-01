# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        ghost_position = successorGameState.getGhostPosition(1)
        foods = newFood.asList()
        capsules = currentGameState.getCapsules()
        shortest_food_distance = 500
        shortest_ghost_distance = 300

        # successor pacman closer to food so better
        for food in foods:
            distance = manhattanDistance(food, newPos)
            shortest_food_distance = min(distance, shortest_food_distance)
        score += 100 / shortest_food_distance

        # successor ate a food so better
        if len(foods) < len(currentGameState.getFood().asList()):
            score += 100

        # successor pacman farther from ghost is better
        ghost_idx = 1
        for ghost_state in newGhostStates:
            ghost_position = successorGameState.getGhostPosition(ghost_idx)
            shortest_ghost_distance = min(manhattanDistance(ghost_position, newPos), shortest_ghost_distance)
            ghost_idx += 1
        score += shortest_ghost_distance
        
        # successor pacman ate capsule so better, pacman stopped so worse
        if newPos in capsules:
            score += 50
        if action == Directions.STOP:
            score -= 20

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    # Given the state of the game, return the max action to tak
    def max_value(self, gameState: GameState, agentIndex, depth):
        legal_actions = []
        for action in gameState.getLegalActions(agentIndex):
            legal_actions.append((self.minimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0], action))
        return max(legal_actions)
    
    def min_value(self, gameState: GameState, agentIndex, depth):
        legal_actions = []
        for action in gameState.getLegalActions(agentIndex):
            legal_actions.append((self.minimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0], action))
        return min(legal_actions)

    def minimax(self, gameState: GameState, agentIndex, depth):
        # terminal test
        if (gameState.isWin() or gameState.isLose() or depth == 0):
            return ( self.evaluationFunction(gameState), Directions.STOP)
        else:
            numAgents = gameState.getNumAgents()
            agentIndex %= numAgents
            if agentIndex == numAgents - 1:
                depth -= 1
            
            if agentIndex == 0:
                return self.max_value(gameState, agentIndex, depth)
            else:
                return self.min_value(gameState, agentIndex, depth)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        return self.minimax(gameState, 0, self.depth)[1]
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def max_value(self, gameState, agentIndex, depth):
        legal_actions = []
        for action in gameState.getLegalActions(agentIndex):
            legal_actions.append((self.expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0], action))   
        return max(legal_actions)
    
    def min_value(self, gameState, agentIndex, depth):
        legal_actions = []
        running_total = 0
        for action in gameState.getLegalActions(agentIndex):
            value = self.expectimax(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, depth)[0]
            running_total += value
            legal_actions.append(action)
        
        return (running_total / len(legal_actions), action)
    
    def expectimax(self, gameState, agentIndex, depth):
        # terminal test
        if (gameState.isWin() or gameState.isLose() or depth == 0):
            return ( self.evaluationFunction(gameState), Directions.STOP)
        
        numAgents = gameState.getNumAgents()
        agentIndex %= numAgents
        if agentIndex == numAgents - 1:
            depth -= 1

        if agentIndex == 0:
            return self.max_value(gameState, agentIndex, depth)
        else:
            return self.min_value(gameState, agentIndex, depth)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, 0, self.depth)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    I created a modified score that uses subscores made from evaluations of the food, capsules, 
    and ghost on the board. Primarily I took into account the minimum distance between the pacman's position
    and each of the food and capsules on the board since the pacman should be eating the closes food or capsule.
    I gave a bigger base score for the foods since the pacman should be eating all the food to win the game.
    The scores for food and capsules are calculated the same which is the base score subtracted by the distance
    to the closest cell subtracted by the number of foods * 10 which was found through testing.
    For the ghost score, I use the scaredTimer value available in the ghost state to determine whether the score should 
    be positive or negative since it is 'good' for a ghost to be scared since it is edible in this state.
    These scores have been intentionally scaled to work with the gameState score.
    """
    "*** YOUR CODE HERE ***"
    pacman = currentGameState.getPacmanPosition()

    #Food Evaluations
    foods = currentGameState.getFood().asList()
    food_distance = 100
    for food in foods:
        food_distance = min(manhattanDistance(pacman, food), food_distance)
    food_score = 500 - len(foods) * 10 - food_distance
    
    #Capsules Evaluations
    capsules = currentGameState.getCapsules()
    capsule_distance = 100
    for capsule in capsules:
        capsule_distance = min(manhattanDistance(pacman, capsule), capsule_distance)
    capsule_score =  300 - len(capsules) * 10 - capsule_distance
    
    #Ghost Evaluations
    ghosts = currentGameState.getGhostStates()
    ghost_distance = 100
    ghost_idx = 1
    ghost_score = 0
    for ghost in ghosts:
        ghost_position = currentGameState.getGhostPosition(ghost_idx)
        ghost_timer = ghost.scaredTimer
        ghost_distance = min(manhattanDistance(ghost_position, pacman), ghost_distance)
    
    if ghost_timer > 0:
        ghost_score += max(100 - ghost_distance, ghost_score)
    else:
        ghost_score -= max(100 - ghost_distance, ghost_score)
        
    # sum of all scores
    score = currentGameState.getScore() + food_score + capsule_score + ghost_score

    return score

# Abbreviation
better = betterEvaluationFunction
