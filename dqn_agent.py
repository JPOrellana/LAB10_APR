import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """
    Convierte la observación RGB (H, W, 3) a un frame en escala de grises
    redimensionado a 84x84 y normalizado en [0,1].
    """
    import cv2

    # obs es (H, W, 3) tipo uint8
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Devuelve (1, 84, 84) float32
    return (resized.astype(np.float32) / 255.0)[None, :, :]


class DuelingDQN(nn.Module):
    """
    Red Dueling: separa Value y Advantage para mejorar estabilidad
    y desempeño en Atari.
    """

    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Asumiendo salida de convs: (64, 7, 7)
        self.value_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        value = self.value_stream(x)              # (B, 1)
        advantage = self.advantage_stream(x)      # (B, A)
        # Q = V + (A - mean(A))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        return q_values


@dataclass
class DQNConfig:
    # Hiperparámetros más “agresivos” para Atari
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 32
    buffer_size: int = 500_000
    min_buffer_size: int = 50_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_frames: int = 800_000   # exploración más larga
    target_update_freq: int = 30_000
    train_freq: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Agente DUELING DOUBLE DQN preparado para ser usado con play.py.
    Implementa un método predict(obs) que devuelve una acción entera.
    """

    def __init__(self, n_actions: int, cfg: DQNConfig):
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # 4 frames apilados como en DQN clásico
        self.policy_net = DuelingDQN(in_channels=4, n_actions=n_actions).to(self.device)
        self.target_net = DuelingDQN(in_channels=4, n_actions=n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.replay_buffer = ReplayBuffer(cfg.buffer_size)

        # frame stack interno (4 x 84 x 84)
        self.frame_stack = None
        self.total_frames = 0

    # --------- Utilidades de frame stacking ---------
    def reset_frame_stack(self, first_frame: np.ndarray):
        f = preprocess_obs(first_frame)  # (1,84,84)
        self.frame_stack = np.repeat(f, 4, axis=0)  # (4,84,84)

    def append_frame(self, frame: np.ndarray):
        f = preprocess_obs(frame)
        self.frame_stack = np.concatenate([self.frame_stack[1:], f], axis=0)

    # --------- Política epsilon-greedy para entrenamiento ---------
    def select_action(self, obs: np.ndarray, train: bool = True) -> int:
        """
        obs es una observación cruda (RGB). Internamente se maneja frame_stack.
        """
        if self.frame_stack is None:
            self.reset_frame_stack(obs)
        else:
            self.append_frame(obs)

        self.total_frames += 1

        if train:
            epsilon = self._compute_epsilon()
            if random.random() < epsilon:
                return random.randrange(self.n_actions)
        # Acción greedy
        state_t = torch.from_numpy(self.frame_stack).unsqueeze(0).to(self.device)  # (1,4,84,84)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def _compute_epsilon(self) -> float:
        cfg = self.cfg
        frac = min(1.0, self.total_frames / cfg.epsilon_decay_frames)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    # --------- Método requerido por play.py ---------
    def predict(self, obs: np.ndarray) -> int:
        """
        Versión sin exploración: siempre toma la mejor acción greedy.
        Se usará en evaluación (play.py).
        """
        if self.frame_stack is None:
            self.reset_frame_stack(obs)
        else:
            self.append_frame(obs)

        state_t = torch.from_numpy(self.frame_stack).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # --------- Entrenamiento ---------
    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def optimize(self):
        cfg = self.cfg
        if len(self.replay_buffer) < cfg.min_buffer_size:
            return

        if self.total_frames % cfg.train_freq != 0:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            cfg.batch_size
        )
        # Convertir a tensores
        states_t = torch.from_numpy(states).to(self.device)        # (B,4,84,84)
        actions_t = torch.from_numpy(actions).long().to(self.device)  # (B,)
        rewards_t = torch.from_numpy(rewards).to(self.device)      # (B,)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        # Q(s,a)
        q_values = self.policy_net(states_t)
        state_action_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # --------- DOUBLE DQN ---------
        with torch.no_grad():
            # 1) Política elige la mejor acción en next_state
            next_q_policy = self.policy_net(next_states_t)
            next_actions = next_q_policy.argmax(dim=1)  # (B,)

            # 2) Target evalúa esa acción
            next_q_target = self.target_net(next_states_t)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target_values = rewards_t + cfg.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(state_action_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Actualizar red target
        if self.total_frames % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # --------- Serialización ---------
    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.policy_net.state_dict(),
                "n_actions": self.n_actions,
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "DQNAgent":
        # IMPORTANTE: weights_only=False por PyTorch 2.6+
        data = torch.load(path, map_location="cpu", weights_only=False)
        cfg = DQNConfig(**data["cfg"])
        agent = DQNAgent(n_actions=data["n_actions"], cfg=cfg)
        agent.policy_net.load_state_dict(data["state_dict"])
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.policy_net.eval()
        return agent
