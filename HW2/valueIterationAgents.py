# valueIterationAgents.py
# -----------------------
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


import mdp, util, json

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()

        for copy in [self.values.copy() for _ in range(self.iterations)]:
            for s in [s for s in mdp.getStates() if not mdp.isTerminal(s)]:
                copy[s] = self.__computeActionsAndQValues(s)[1] #the value of the best possible action
            self.values = copy

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        """
            Explanation: to calculate QValue we need to calculate the weighted
            sum that results from multiplying each probability times the sum of
            the discount times the value of the ending state (s') plus the reward received
            by transitioning to that state (s')
        """
        return sum(list(map(lambda stateAndProb: 
            stateAndProb[1] * (self.discount * self.getValue(stateAndProb[0]) 
            + self.mdp.getReward(state, action, stateAndProb[0]))
            , self.mdp.getTransitionStatesAndProbs(state, action))))

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        """
          Explanation:
          to compute the action we need to get Qvalue of every possible action and return
          the highest one
        """
        if self.mdp.isTerminal(state):
            return None

        return self.__computeActionsAndQValues(state)[0]

    def __computeActionsAndQValues(self, state):
        """
            Returns a tuple of the best possible action and its qValue
        """
        return sorted(
            list(
                map(
                    lambda action: (action, self.getQValue(state, action)), self.mdp.getPossibleActions(state))
                    ), 
                key=lambda tuple: tuple[1]
                )[-1]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)