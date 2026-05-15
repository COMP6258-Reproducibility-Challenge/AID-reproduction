import argparse, json, os, random, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

# AID module from paper
class AID(nn.Module):
    def __init__(self, p=0.9):
        super().__init__()
        self.p = float(p)

        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")

    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, self.p)).bool()
            return torch.where(mask, torch.relu(x), -torch.relu(-x))
        else:
            return torch.where(x >= 0, self.p * x, (1.0 - self.p) * x)

# DQN network
def create_activation(mode, p):
    if mode == "relu":
        return nn.ReLU()
    elif mode == "aid":
        return AID(p=p)
    elif mode == "dropout":
        return nn.Sequential(nn.ReLU(), nn.Dropout(p))

class QNet(nn.Module):
    def __init__(self, n_actions, activation="relu", p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            create_activation(activation, p),
            nn.Conv2d(32, 64, 4, stride=2),
            create_activation(activation, p),
            nn.Conv2d(64, 64, 3, stride=1),
            create_activation(activation, p),
            nn.Flatten(),
            nn.Linear(3136, 512),
            create_activation(activation, p),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.net(x)

# p values from table 9 for AID and dropout
aid_p = {
    "Seaquest":0.999,
    "DemonAttack":0.99, # changed from 0.99
    "SpaceInvaders":0.99,
    "Qbert":0.999,
    "DoubleDunk":0.99,
    "MsPacman":0.999,
    "Enduro":0.99,
    "BeamRider":0.99,
    "WizardOfWor":0.999,
    "Jamesbond":0.99,
    "RoadRunner":0.999,
    "Asterix":0.99,
    "Pong":0.999,
    "Zaxxon":0.999,
    "YarsRevenge":0.99,
    "Breakout":0.99,
    "IceHockey":0.99}

dropout_p = {
    "Seaquest": 0.01, 
    "DemonAttack": 0.01, 
    "SpaceInvaders": 0.01, 
    "Qbert": 0.01, 
    "DoubleDunk": 0.01, 
    "MsPacman": 0.01, 
    "Enduro": 0.001, 
    "BeamRider": 0.01, 
    "WizardOfWor": 0.01, 
    "Jamesbond": 0.01, 
    "RoadRunner": 0.01, 
    "Asterix": 0.01, 
    "Pong": 0.01, 
    "Zaxxon": 0.01, 
    "YarsRevenge": 0.01, 
    "Breakout": 0.01, 
    "IceHockey": 0.01
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, choices=list(aid_p))
    ap.add_argument("--activation", required=True, choices=["relu","dropout","aid"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total_frames", type=int, default=10_000_000)
    ap.add_argument("--out_dir", default="./results")
    args = ap.parse_args()

    # assign p values according to the table in the paper
    p = 0.0
    if args.activation == "aid":
        p = aid_p[args.game]

    elif args.activation == "dropout":
        p = dropout_p[args.game]

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # env setup for atari
    env = gym.make(f"ALE/{args.game}-v5", frameskip=1, repeat_action_probability=0.25)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=False, noop_max=30)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.TransformReward(env, np.sign)
    n_actions = env.action_space.n

    obs, _ = env.reset(seed=args.seed)
    obs_arr = np.array(obs)

    # networks and optimiser
    q = QNet(n_actions, args.activation, p).to(device)
    tgt = QNet(n_actions, args.activation, p).to(device)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=6.25e-4, eps=1.5e-4)

    # replay buffer
    cap = 1_000_000
    obs_buf = np.zeros((cap, 4, 84, 84), dtype=np.uint8)
    next_buf = np.zeros((cap, 4, 84, 84), dtype=np.uint8)
    act_buf = np.zeros(cap, dtype=np.int64)
    rew_buf = np.zeros(cap, dtype=np.float32)
    done_buf = np.zeros(cap, dtype=np.float32)
    ptr, size = 0, 0

    # training loop
    obs, _ = env.reset(seed=args.seed)
    obs = np.array(obs, dtype=np.uint8)
    lives = env.unwrapped.ale.lives()
    returns = deque(maxlen=100)
    log = []
    total_steps = args.total_frames // 4
    start = time.time()
    grad_steps = 0
    train_frequency = 4 # RR (1/train_frequency)

    for step in range(total_steps):
        eps = max(0.01, 1.0 - step / 250_000 * 0.99)
        if step < 20_000 or random.random() < eps:
            action = env.action_space.sample()
        else:
            q.eval() # deterministic Q for AID and Dropout
            with torch.no_grad():
                action = int(q(torch.from_numpy(obs).unsqueeze(0).to(device)).argmax(1))

        next_obs, reward, term, trunc, info = env.step(action)
        next_obs = np.array(next_obs, dtype=np.uint8)

        # store transition
        obs_buf[ptr], act_buf[ptr], rew_buf[ptr] = obs, action, reward
        next_buf[ptr], done_buf[ptr] = next_obs, float(term)
        ptr = (ptr + 1) % cap; size = min(size + 1, cap)
        
        new_lives = env.unwrapped.ale.lives()
        real_done = (new_lives == 0) or trunc

        # track only final life score (as terminal_on_life_loss is true)
        if real_done:
            if "episode" in info:
                returns.append(float(info["episode"]["r"]))
                log.append((step, float(info["episode"]["r"])))
            obs, _ = env.reset()
            obs = np.array(obs, dtype=np.uint8)
            lives = env.unwrapped.ale.lives()
        else:
            obs = next_obs
            lives = new_lives

        # gradient update
        if step >= 20_000 and step % train_frequency == 0: 
            q.train()  # AID or Dropout active during update
            idx = np.random.randint(0, size, 32)
            b_obs = torch.from_numpy(obs_buf[idx]).to(device)
            b_act = torch.from_numpy(act_buf[idx]).to(device)
            b_rew = torch.from_numpy(rew_buf[idx]).to(device)
            b_nxt = torch.from_numpy(next_buf[idx]).to(device)
            b_dn = torch.from_numpy(done_buf[idx]).to(device)

            with torch.no_grad():
                tgt.eval()
                target = b_rew + 0.99 * tgt(b_nxt).max(1).values * (1 - b_dn)
            pred = q(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(pred, target)

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            grad_steps += 1

            if grad_steps % 2000 == 0:
                tgt.load_state_dict(q.state_dict())

        if step % 10_000 == 0 and returns:
            sps = (step + 1) / (time.time() - start)
            loss_val = loss.item() if step >= 20_000 else 0.0
            print("step=", step, "frames=", step*4, "eps=", eps, "ret(100)=", round(np.mean(returns), 1), "loss=", round(loss_val, 4), "sps=", int(sps))   


    # save log
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.game}_{args.activation}_seed{args.seed}.json"
    with open(os.path.join(args.out_dir, tag), "w") as f:
        json.dump({"game": args.game, "activation": args.activation, "p": p, "seed": args.seed, "raw_returns": log}, f)
    env.close()


if __name__ == "__main__":
    main()
