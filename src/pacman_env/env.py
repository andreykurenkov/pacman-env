import gym
from gym import error, spaces
import random
import numpy as np
import time
from game import Directions, Game
from ghostAgents import RandomGhost, DirectionalGhost
from pacmanAgents import RandomAgent, LeftTurnAgent, GreedyAgent
from graphicsDisplay import PacmanGraphics as VizGraphics
from textDisplay import PacmanGraphics as TextGraphics
from matrixDisplay import PacmanGraphics as MatrixGraphics
from pacman import GameState, ClassicGameRules

RGB = 'rgb'
TERMINAL = 'terminal'

class EnvGame(Game):
    """
    Variant of Game where Env supplies action for pacman
    """

    def __init__(self, ghostAgents, display, rules, layout, percentRandomize=0.5):
        Game.__init__(self, ghostAgents, display, rules)
        initState = GameState()
        initState.initialize(layout, len(ghostAgents) , percentRandomize)
        self.state = initState
        self.initialState = initState.deepCopy()
        
    def getState(self):
        return self.state
        
    def getScore(self):
        return self.state.getScore()
        
    def init(self):
        self.display.initialize(self.state.data)
        self.numMoves = 0

        ###self.display.initialize(self.state.makeObservation(1).data)
        # inform learning agents of the game start
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if not agent:
                self.mute(i)
                # this is a null agent, meaning it failed to load
                # the other team wins
                print >>sys.stderr, "Agent %d failed to load" % i
                self.unmute()
                self._agentCrash(i, quiet=True)
                return
                
        self.numAgents = len( self.agents )

    def do_one_move(self, action):
        """
        Main control loop for game play.
        """
        agentIndex = 0
        # Execute the action
        self.moveHistory.append( (agentIndex, action) )
        self.state = self.state.generateSuccessor( agentIndex, action )
        self.display.update( self.state.data )
        self.rules.process(self.state, self)            
        if self.gameOver:
            self.display.finish()
            return self.state.deepCopy()

        for ghost in self.agents:
            # Fetch the next agent
            observation = self.state.deepCopy()
            action = ghost.getAction(observation)
            agentIndex+=1
            self.state = self.state.generateSuccessor( agentIndex, action )
            # Change the display
            self.display.update( self.state.data )
       
            # Allow for game specific conditions (winning, losing, etc.)
            self.rules.process(self.state, self)
            
            if self.gameOver:
                self.display.finish()
                break
        
        self.numMoves += 1
            
        return self.state.deepCopy()

class PacmanEnv(gym.Env):

    metadata = {'render.modes': ['rgb_array','ansi','matrix']}

    def __init__(self, layout, ghosts, display, timeout=30, percentRandomize=0.5, teacherAgents = []):
        import __main__
        __main__.__dict__['_display'] = display
        self._rules = ClassicGameRules(timeout)
        self._layout = layout
        self._ghosts = ghosts
        self._display = display
        self._percentRandomize = percentRandomize
        if isinstance(display, VizGraphics):
            self._obs_type = 'rgb_array' 
        elif isinstance(display, TextGraphics):
            self._obs_type = 'ansi'
        elif isinstance(display, MatrixGraphics):
            self._obs_type = 'matrix'
        else:
            raise ValueError('Invalid display arg!')

        self.action_set = [Directions.NORTH, 
                           Directions.SOUTH, 
                           Directions.EAST, 
                           Directions.WEST, 
                           Directions.STOP] + teacherAgents
        self.action_space = spaces.Discrete(len(self.action_set))

        self.width = layout.width
        self.height = layout.height
        if self._obs_type == 'ansi':
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.height), dtype=np.uint8)
        elif self._obs_type == 'matrix':
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.width, self.height, display.NUM_CHANNELS), dtype=np.uint8)
        elif self._obs_type == 'rgb_array':
            (screen_width,screen_height) = display.get_size(self.width, self.height)
            self.observation_space = spaces.Box(low=0, high=255, shape=(int(screen_height), int(screen_width), 3), dtype=np.uint8)

    @property 
    def game_over(self):
        return self.game.gameOver

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.real_action = None
        if isinstance(action, int):
            if action > 4:
                self.real_action = action
            action = self.action_set[action]
        if not isinstance(action, str):
            action = action.getAction(self.game.getState())
        legal = self.game.getState().getLegalActions(0)
        if action not in legal:
            action = Directions.STOP
        self.game.do_one_move(action)
        done = self.game.gameOver
        state = self.game.getState()
        observation = self._display.getArray(state)
        new_score = self.game.getScore()
        reward = new_score - self.current_score
        if done:
            return None, reward, True, None
        self.current_score = new_score

        if self.real_action!=None:
            self.real_reward = reward
            if reward > 0:
                reward = reward * 0.9
            else:
                reward = reward * 1.1
        return observation, reward, done, None

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        self.game = EnvGame(self._ghosts, self._display, self._rules, self._layout, self._percentRandomize)
        self.game.init()
        self.current_score = self.game.getScore()
        return self._display.getArray(self.game.getState())

    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """

        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        
        random.seed(seed2)
        return [seed1, seed2]
        
if __name__ == '__main__':
    import layout
    medium_layout = layout.getLayout('originalClassic')
    ghosts = []
    for i in range(2):
        ghosts.append(DirectionalGhost(i+1))
    #display = VizGraphics(includeInfoPane=False, zoom=1)
    display = TextGraphics(draw_end = True)
    #display = MatrixGraphics(medium_layout)
    env = PacmanEnv(medium_layout, ghosts, display)
    env.reset()
    
    state = env.game.state
    pacman = RandomAgent(onlyLegal = False)
    #pacman = LeftTurnAgent()
    #pacman = GreedyAgent()
    totals = []
    total = 0
    games = 0
    while games < 100:
        obs, reward, done, info = env.step(pacman.getAction(state))
        if reward>0:
            reward*=0.9
        else:
            reward*=1.1
        total+=reward
        if done:
            totals.append(total)
            total = 0
            games+=1
            env.reset()
        state = env.game.state
    print(totals)
