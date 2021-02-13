from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

"""
Define a Black Jack environment using OpenAI gym 
"""
# Define our custom environment for the agent to operate in
class BlackJackEnv(Env):

    def __init__(self):
        # Agent can either accept a new card or hold
        self.action_space = Discrete(2)
        # Observation space is a the sum of the card values, with 0 representing a score>21
        self.observation_space = Box(low=np.array([0]), high=np.array([21]))
        # Set start point as any number between 2-21
        self.state = random.randint(2,21)

    def step(self, action):
        # Apply action
        # 0: Draw a card
        # 1: Hold position
        if action == 0:
            self.state += random.randint(1,10)

        # Calculate if agent drew over 21 and end game
            if self.state >=22:
                self.state = 0
                reward = 0
                done = True
            else:
                reward = self.state
                done = False

        # Calculate agent's final score and end game
        elif action ==1:
            done = True
            reward = self.state

        # Set placeholder for info
        info = {}

        # Return Step info
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        # Reset agent's hand
        self.state = random.randint(2, 21)
        return self.state

# Create training environment
trainEnv = BlackJackEnv()



# Test environment with random inputs from action_space
episodes = 20
for episode in range(1, episodes+1):
    state = trainEnv.reset()
    done = False
    score = 0

    while not done:
        action = trainEnv.action_space.sample()
        n_state, reward, done, info, = trainEnv.step(action)
    print('Episode: {0} Score: {1}'.format(episode, reward))

"""
Create a Deep Learning Model with Keras
"""
states = trainEnv.observation_space.shape
actions = trainEnv.action_space.n

print(states)
print(actions)

def build_model(states, actions):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='sigmoid'))
    return model

model = build_model(states, actions)

print(model.summary())

"""
Build Agent with Keras-RL
"""

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions,
                   nb_steps_warmup=50, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
dqn.fit(trainEnv, nb_steps=50000, visualize=False, verbose=1)









