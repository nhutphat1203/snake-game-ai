import random
import numpy as np
from enum import Enum
from collections import namedtuple
from game.const import *

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple("Point", "x, y")

class SnakeEnv:
    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height
        self.state_count = 0
        self.reset()

    def reset(self):
        self.snake = [Point(self.width // 2, self.height // 2)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.food = None
        self._place_food()
        self.game_over = False
        self.state_count = 0
        return self.get_state()

    def _place_food(self):
        x = random.randrange(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randrange(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

    def _map_action(self, action):
        return 0 if action[0] == 1 else (1 if action[1] == 1 else 2)
    
    def max_allowed_steps(self, head):
        """
        Tính số bước tối đa rắn có thể đi trước khi game over,
        dựa vào khoảng cách Manhattan từ đầu rắn tới thức ăn.
        """
        distance_to_food = abs(head.x - self.food.x) // BLOCK_SIZE + abs(head.y - self.food.y) // BLOCK_SIZE
        buffer = len(self.snake) * 3 + min(self.width, self.height) // BLOCK_SIZE

        return 2 * distance_to_food + buffer
        
    def _game_stop_condition(self, head):
        reward_b, stop_b = self._boundary_stop(head)
        reward_e, stop_e = self._eat_itself(head)
        max_distance = self.max_allowed_steps(head)
        stop_distance = self.state_count > max_distance
        reward_distance = -10 if stop_distance else 0
        return reward_b + reward_e + reward_distance, stop_b or stop_e or stop_distance
        
    def _boundary_stop(self, head):
        stop = False
        reward = 0
        if (head.x < 0 or head.x >= self.width or
        head.y < 0 or head.y >= self.height):
            stop = True
            reward = -15
        return reward, stop
    
    def _eat_itself(self, head):
        stop = False
        reward = 0
        if head in self.snake:
            stop = True
            reward = -20
        return reward, stop
        

    def _neer_food_reward(self, head, new_head):
        old_distance = abs(head.x - self.food.x) + abs(head.y - self.food.y)
        new_distance = abs(new_head.x - self.food.x) + abs(new_head.y - self.food.y)
        reward = (old_distance - new_distance) / BLOCK_SIZE
        return 2 * reward

    def step(self, action_player):
        """
        action: 0=straight, 1=left, 2=right
        Trả về: reward, done, score
        """
        if self.game_over:
            return self.get_state(), 0, True, self.score
        
        action = self._map_action(action_player)
        
        self._apply_action(action)

        head = self.snake[0]
        if self.direction == Direction.RIGHT:
            new_head = Point(head.x + BLOCK_SIZE, head.y)
        elif self.direction == Direction.LEFT:
            new_head = Point(head.x - BLOCK_SIZE, head.y)
        elif self.direction == Direction.UP:
            new_head = Point(head.x, head.y - BLOCK_SIZE)
        else:  # DOWN
            new_head = Point(head.x, head.y + BLOCK_SIZE)

        reward = 0

        reward += self._neer_food_reward(head, new_head)
        
        # Kiểm tra va chạm
        reward_stop, stop = self._game_stop_condition(new_head)
        reward += reward_stop
        
        if stop:
            self.game_over = True
            return reward, True, self.score

        self.snake.insert(0, new_head)

        # Ăn thức ăn
        if new_head == self.food:
            self.score += 1
            reward += 10
            self.state_count = 0
            self._place_food()
        else:
            self.snake.pop()

        return reward, False, self.score

    def _apply_action(self, action):
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = directions.index(self.direction)

        if action == 0:  # đi thẳng
            new_dir = directions[idx]
        elif action == 1:  # rẽ trái
            new_dir = directions[(idx - 1) % 4]
        else:  # rẽ phải
            new_dir = directions[(idx + 1) % 4]

        self.direction = new_dir
        self.state_count += 1


    def get_state(self):
        """
        Trả về state đơn giản dạng vector:
        [danger_straight, danger_right, danger_left,
         dir_left, dir_right, dir_up, dir_down,
         food_left, food_right, food_up, food_down]
        """
        head = self.snake[0]

        danger_straight, danger_right, danger_left = self._get_danger()

        # Hướng hiện tại
        dir_left  = self.direction == Direction.LEFT
        dir_right = self.direction == Direction.RIGHT
        dir_up    = self.direction == Direction.UP
        dir_down  = self.direction == Direction.DOWN

        # Vị trí thức ăn so với đầu
        food_left  = self.food.x < head.x
        food_right = self.food.x > head.x
        food_up    = self.food.y < head.y
        food_down  = self.food.y > head.y

        state = [
            danger_straight, danger_right, danger_left,
            dir_left, dir_right, dir_up, dir_down,
            food_left, food_right, food_up, food_down
        ]
        return state

    def _get_danger(self):
        """
        Trả về 3 giá trị boolean:
        - danger_straight
        - danger_right
        - danger_left
        """
        head = self.snake[0]

        # Map hướng sang vector (dx,dy)
        directions = {
            Direction.UP:    (0, -BLOCK_SIZE),
            Direction.DOWN:  (0, BLOCK_SIZE),
            Direction.LEFT:  (-BLOCK_SIZE, 0),
            Direction.RIGHT: (BLOCK_SIZE, 0),
        }
        dx, dy = directions[self.direction]

        straight = Point(head.x + dx, head.y + dy)
        left     = Point(head.x + dy, head.y - dx)
        right    = Point(head.x - dy, head.y + dx) 
        
        danger_straight = self._is_collision(straight)
        danger_left     = self._is_collision(left)
        danger_right    = self._is_collision(right)
        
        if len(self.snake) > 4:
            straight2, right2, left2 = self._n_step_vision()
            danger_straight = danger_straight or straight2
            danger_left = danger_left or left2
            danger_right = danger_right or right2
            
        return danger_straight, danger_right, danger_left

    def _is_collision(self, point):

        if (point.x < 0 or point.x >= self.width or
            point.y < 0 or point.y >= self.height or
            point in self.snake):
            return True
        return False
    
    def _n_step_vision(self, n=2):
        head = self.snake[0]
        # Map hướng sang vector (dx,dy)
        directions = {
            Direction.UP:    (0, -BLOCK_SIZE * n),
            Direction.DOWN:  (0, BLOCK_SIZE * n),
            Direction.LEFT:  (-BLOCK_SIZE * n, 0),
            Direction.RIGHT: (BLOCK_SIZE * n, 0),
        }
        dx, dy = directions[self.direction]

        left     = Point(head.x + dy, head.y - dx)
        right    = Point(head.x - dy, head.y + dx) 
        
        danger_straight = False
        danger_left     = False
        danger_right    = False
        
        snake_len = len(self.snake)
        
        if left in self.snake and self.snake[snake_len - n - 1] == left:
            danger_left = True
        if right in self.snake and self.snake[snake_len - n - 1] == right:
            danger_right = True
        return danger_straight, danger_right, danger_left
