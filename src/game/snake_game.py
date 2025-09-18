import pygame
from game.const import *
from game.snake_logic import SnakeEnv
pygame.init()

# 🎨 Bảng màu
WHITE   = (245, 245, 245)
BLACK   = (30, 30, 30)
GRID_1  = (0, 5, 40)
GRID_2  = (6, 20, 70)
SNAKE_HEAD = (65, 105, 225)
SNAKE_BODY = (50, 205, 50)
FOOD_COLOR = (220, 20, 60)
YELLOW  = (255, 215, 0)

font = pygame.font.SysFont("arial", 25)

# ---------------------------
# 🖼️ GIAO DIỆN
# ---------------------------
class SnakeUI:
    def __init__(self, env: SnakeEnv):
        self.env = env
        self.screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.game_count = 0

        # Vẽ background caro 1 lần
        self.background = pygame.Surface((env.width, env.height))
        for y in range(0, env.height, BLOCK_SIZE):
            for x in range(0, env.width, BLOCK_SIZE):
                color = GRID_1 if (x // BLOCK_SIZE + y // BLOCK_SIZE) % 2 == 0 else GRID_2
                pygame.draw.rect(self.background, color, (x, y, BLOCK_SIZE, BLOCK_SIZE))

    def draw(self):
        # Vẽ background
        self.screen.blit(self.background, (0, 0))

        # Vẽ thức ăn
        pygame.draw.rect(self.screen, FOOD_COLOR,
                         (self.env.food.x, self.env.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Vẽ rắn
        for i, p in enumerate(self.env.snake):
            color = SNAKE_HEAD if i == 0 else SNAKE_BODY
            pygame.draw.rect(self.screen, color, (p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))

        # Điểm số
        score_text = font.render(f"Score: {self.env.score}", True, YELLOW)
        self.screen.blit(score_text, [10, 10])

        pygame.display.flip()

        # Nếu game over -> reset ngay
        if self.env.game_over:
            self.game_count += 1
            self.env.reset()

class SnakeGame:
    def __init__(self, render=True):
        self.logic = SnakeEnv()
        self.render = render
        self.running = True
        if self.render:
            self.ui = SnakeUI(self.logic)

    def step(self, action):
        if not self.running:
            return 0, True, self.logic.score, False  # reward, done, score, running

        reward, done, score = self.logic.step(action)

        # Xử lý sự kiện Pygame
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    return reward, True, score, False

            self.ui.draw()
            self.ui.clock.tick(CLOCK_TICK)

        return reward, done, score, self.running

    def reset(self):
        state = self.logic.reset()
        if self.render:
            self.ui.draw()
        return state
