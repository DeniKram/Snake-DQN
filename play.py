import time

import torch
from DQN import DQN
import Snake

device = "cuda" if torch.cuda.is_available() else "cpu"

q_net = DQN(10, 10, 4).to(device)
q_net.load_state_dict(torch.load("best_snake.pt", map_location=device))
q_net.eval()

env = Snake.SnakeEnv(w=10, h=10)
state, _ = env.reset()
done = False
score = 0

env.render()

while not done:
    s_t = torch.tensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        action = q_net(s_t).argmax().item()

    state, reward, done, _, _ = env.step(action)
    score += 1 if reward == 1.0 else 0
    env.render()
    time.sleep(0.2)

print(f"Episode finished | score: {score} | done: {done}")