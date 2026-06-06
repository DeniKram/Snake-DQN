import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class SnakeEnv(gym.Env):
    def __init__(self, w=10, h=10):
        super().__init__()

        self.w = w
        self.h = h

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, h, w),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        cx, cy = self.w // 2, self.h // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = 3  
        self.spawn_food()
        self.steps = 0
        self.max_steps = self.w * self.h * 2

        return self._get_state(), {}

    def spawn_food(self):
        empty = [
            (x, y)
            for x in range(self.w)
            for y in range(self.h)
            if (x, y) not in self.snake
        ]
        self.food = random.choice(empty) if empty else (0, 0)

    def step(self, action):
        
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        if action != opposite[self.direction]:
            self.direction = action

        dx, dy = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[self.direction]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy

        self.steps += 1

        if nx < 0 or nx >= self.w or ny < 0 or ny >= self.h:
            return self._get_state(), -1.0, True, False, {}

        if (nx, ny) in self.snake[:-1]:
            return self._get_state(), -1.0, True, False, {}

        self.snake.insert(0, (nx, ny))

        if (nx, ny) == self.food:
            reward = 1.0
            self.spawn_food()
        else:
            self.snake.pop()
            
            old_dist = abs(hx - self.food[0]) + abs(hy - self.food[1])
            new_dist = abs(nx - self.food[0]) + abs(ny - self.food[1])
            reward = 0.01 if new_dist < old_dist else -0.01

        done = self.steps >= self.max_steps

        return self._get_state(), reward, done, False, {}

    def _get_state(self):
        state = np.zeros((3, self.h, self.w), dtype=np.float32)

        for x, y in self.snake:
            state[0, y, x] = 1.0

        hx, hy = self.snake[0]
        state[1, hy, hx] = 1.0

        fx, fy = self.food
        state[2, fy, fx] = 1.0

        return state

    def render(self):
        grid = [['.' for _ in range(self.w)] for _ in range(self.h)]
        for x, y in self.snake[1:]:
            grid[y][x] = 'o'
        hx, hy = self.snake[0]
        grid[hy][hx] = 'H'
        fx, fy = self.food
        grid[fy][fx] = 'F'
        for row in grid:
            print(' '.join(row))
        print()