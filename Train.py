import torch
import torch.nn as nn
import numpy as np
from DQN import DQN
from Memory import ReplayBuffer
import Snake

env = Snake.SnakeEnv(w=10, h=10)

h, w    = 10, 10
n_actions = env.action_space.n  # 4

device = "cuda" if torch.cuda.is_available() else "cpu"

q_net      = DQN(h, w, n_actions).to(device)
target_net = DQN(h, w, n_actions).to(device)
target_net.load_state_dict(q_net.state_dict())

optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)
buffer    = ReplayBuffer(size=100000)

gamma         = 0.99
batch_size    = 64
epsilon       = 1.0
epsilon_min   = 0.05
epsilon_decay = 0.99995

step_count = 0
best_score = 0

for episode in range(20000):
    state, _ = env.reset()
    done         = False
    total_reward = 0.0
    score        = 0  

    while not done:

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            s_t = torch.tensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = q_net(s_t).argmax().item()

        next_state, reward, done, _, _ = env.step(action)

        if reward == 1.0:
            score += 1

        buffer.push(state, action, reward, next_state, done)
        state         = next_state
        total_reward += reward

        if len(buffer) > batch_size:
            s, a, r, ns, d = buffer.sample(batch_size)

            s  = torch.tensor(s).to(device)
            ns = torch.tensor(ns).to(device)
            a  = torch.tensor(a).to(device)
            r  = torch.tensor(r).to(device)
            d  = torch.tensor(d).to(device)

            q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_actions = q_net(ns).argmax(dim=1)
                next_q       = target_net(ns).gather(1, next_actions.unsqueeze(1)).squeeze()
                target       = r + gamma * next_q * (1 - d)

            loss = nn.SmoothL1Loss()(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()

        step_count += 1
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if step_count % 1000 == 0:
            target_net.load_state_dict(q_net.state_dict())

    if score > best_score:
        best_score = score
        torch.save(q_net.state_dict(), "best_snake.pt")

    if episode % 50 == 0:
        print(f"Ep {episode:5d} | reward: {total_reward:7.2f} | score: {score:3d} | best: {best_score:3d} | eps: {epsilon:.3f}")