# textDisplay.py
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


import time
import numpy as np
try: 
    import pacman
except:
    pass

SLEEP_TIME = 0 # This can be overwritten by __init__
QUIET = False # Supresses output

class NullGraphics:
    def initialize(self, state, isBlue = False):
        pass

    def update(self, state):
        pass

    def checkNullDisplay(self):
        return True

    def pause(self):
        time.sleep(SLEEP_TIME)

    def draw(self, state):
        print state

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass

class PacmanGraphics:
    def __init__(self, speed=None, display_rate = -1, draw_end = False):
        if speed != None:
            global SLEEP_TIME
            SLEEP_TIME = speed
        self.display_rate = display_rate
        self.draw_end = draw_end

    def initialize(self, state, isBlue = False):
        self.pause()
        self.turn = 0
        self.agentCounter = 0

    def update(self, state):
        numAgents = len(state.agentStates)
        self.agentCounter = (self.agentCounter + 1) % numAgents
        if self.agentCounter == 0:
            self.turn += 1
            if self.display_rate > 0 and self.turn % self.display_rate == 0:
                self.draw(state)
                self.pause()
        if self.draw_end and (state._win or state._lose):
            self.draw(state)

    def pause(self):
        time.sleep(SLEEP_TIME)

    def draw(self, state):
        print state
                           
    def getArray(self, state):
        return np.array([np.array([ord(c) for c in i]) for i in str(state).splitlines()[:-1]])
        
    def finish(self):
        pass
        
class PacPosGraphics:

    def initialize(self, state):
        pass
        
    def update(self, state):
        pass

    def pause(self):
        time.sleep(SLEEP_TIME)

    def draw(self, state):
        print state.data.agentStates[0].configuration.pos
                           
    def getArray(self, state):
        return state.data.agentStates[0].configuration.pos
        
    def finish(self):
        pass
