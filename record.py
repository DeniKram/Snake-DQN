import torch
import numpy as np
import imageio
import pygame
import sys
from DQN import DQN
import Snake


CELL   = 40          
FPS    = 10          
W, H   = 10, 10

device = "cuda" if torch.cuda.is_available() else "cpu"

q_net = DQN(H, W, 4).to(device)
q_net.load_state_dict(torch.load("best_snake.pt", map_location=device))
q_net.eval()

env  = Snake.SnakeEnv(w=W, h=H)
pygame.init()
screen = pygame.display.set_mode((W * CELL, H * CELL))
clock  = pygame.time.Clock()

BG      = (15,  15,  15)
BODY    = (50,  200, 50)
HEAD    = (0,   255, 0)
FOOD    = (220, 50,  50)
GRID    = (30,  30,  30)

def draw(env_obj):
    screen.fill(BG)
    # сетка
    for x in range(W):
        for y in range(H):
            pygame.draw.rect(screen, GRID,
                (x*CELL, y*CELL, CELL, CELL), 1)
            
    for x, y in env_obj.snake[1:]:
        pygame.draw.rect(screen, BODY,
            (x*CELL+2, y*CELL+2, CELL-4, CELL-4),
            border_radius=6)
    
    hx, hy = env_obj.snake[0]
    pygame.draw.rect(screen, HEAD,
        (hx*CELL+2, hy*CELL+2, CELL-4, CELL-4),
        border_radius=8)
    
    fx, fy = env_obj.food
    cx, cy = fx*CELL + CELL//2, fy*CELL + CELL//2
    pygame.draw.circle(screen, FOOD, (cx, cy), CELL//2 - 4)

    pygame.display.flip()
    return pygame.surfarray.array3d(screen).transpose(1, 0, 2)

frames = []
state, _ = env.reset()
done  = False
score = 0

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()

    s_t = torch.tensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        action = q_net(s_t).argmax().item()

    state, reward, done, _, _ = env.step(action)
    if reward == 1.0:
        score += 1

    frame = draw(env)
    frames.append(frame)
    clock.tick(FPS)

pygame.quit()

imageio.mimsave("snake.gif", frames, fps=FPS, loop=0)
print(f"Score: {score} | FPS: {len(frames)}")