import torch
import random
import numpy as np
from collections import deque
from agent.model import Trainer, Linear_Net
from game.snake_game import SnakeGame
from game.snake_logic import Direction, Point, SnakeEnv
from helper.plot import plot

MAX_MEM = 100000
BATCH_SIZE= 1000
LR = 0.01


class Agent:
    
    def __init__(self):
        self.epsilon = 0
        self.gamma = 0.9
        self.lr = LR
        self.memory = deque(maxlen=MAX_MEM)
        self.model = Linear_Net(11, 256, 3)
        self.trainer = Trainer(self.model, self.lr, self.gamma)
        self.n_games = 0
    
    def get_state(self, game: SnakeEnv):
        return np.array(game.get_state(), dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
            )
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        self.epsilon = 70 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 140) < self.epsilon:
            move = random.randint(0, len(final_move) - 1)
            final_move[move] = 1;
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
    


def train(render=False, plot_visible=True):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame(render=render)

    while game.running:
        state_old = agent.get_state(game.logic)
        final_move = agent.get_action(state_old)
        reward, done, score, running = game.step(final_move)
        
        if not running:
            print("Game stopped by user. Skipping training for this step.")
            break
        
        state_new = agent.get_state(game.logic)

        
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            if plot_visible:
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

            game.reset()

            