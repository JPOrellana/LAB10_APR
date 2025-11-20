import argparse
import os
from collections import deque

import gymnasium as gym
import numpy as np

from dqn_agent import DQNAgent, DQNConfig


def make_env(seed: int | None = None):
    env = gym.make(
        "ALE/Galaxian-v5",
        render_mode=None,
        frameskip=4,
        repeat_action_probability=0.0,
    )
    if seed is not None:
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
    return env


def train_dqn(
    total_episodes: int = 5000,
    max_steps_per_episode: int = 20000,
    max_frames: int | None = 1_000_000,
    seed: int | None = 0,
    model_out: str = "models/dqn_galaxian.pth",
    checkpoints_dir: str = "models/checkpoints",
    resume: bool = True,
):
    # Carpetas
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    env = make_env(seed)
    n_actions = env.action_space.n
    cfg = DQNConfig()

    # Cargar modelo previo si existe (para continuar entrenamiento)
    if resume and os.path.exists(model_out):
        print(f"Cargando modelo existente desde {model_out} para continuar entrenamiento...")
        agent = DQNAgent.load(model_out)
    else:
        agent = DQNAgent(n_actions=n_actions, cfg=cfg)

    rewards_window = deque(maxlen=100)
    best_mean = -1e9
    global_frame_count = agent.total_frames  # por si venimos de entrenos previos

    for ep in range(1, total_episodes + 1):
        # Si ya alcanzamos el tope de frames, paramos
        if (max_frames is not None) and (global_frame_count >= max_frames):
            print(f"Se alcanzó el máximo de frames: {global_frame_count} >= {max_frames}")
            break

        obs, info = env.reset(seed=seed)
        agent.frame_stack = None  # reiniciar stack para el nuevo episodio
        done = False
        ep_reward = 0.0
        steps = 0

        # inicializar frame stack con la primera observación
        agent.reset_frame_stack(obs)
        state = agent.frame_stack.copy()

        while not done and steps < max_steps_per_episode:
            # seleccionar acción (epsilon-greedy)
            action = agent.select_action(obs, train=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            steps += 1
            global_frame_count += 1

            # CLIPPING de recompensa para estabilidad
            reward_clipped = np.sign(reward)

            # actualizar frame stack con el siguiente frame y armar next_state
            agent.append_frame(next_obs)
            next_state = agent.frame_stack.copy()

            agent.push_transition(state, action, reward_clipped, next_state, float(done))
            agent.optimize()

            obs = next_obs
            state = next_state

            # Corte duro por frames dentro del episodio
            if (max_frames is not None) and (global_frame_count >= max_frames):
                print(f"Se alcanzó el máximo de frames durante el episodio: {global_frame_count} >= {max_frames}")
                break

        rewards_window.append(ep_reward)
        mean_100 = np.mean(rewards_window)

        print(
            f"Ep {ep:5d} | Recompensa: {ep_reward:8.1f} | "
            f"Media(100): {mean_100:8.1f} | Frames: {global_frame_count}"
        )

        # Guardar el mejor modelo según media móvil de 100 episodios
        if mean_100 > best_mean and ep >= 50:
            best_mean = mean_100
            # Guardar "best overall"
            agent.save(model_out)
            print(f"  >> Nuevo mejor promedio {best_mean:.1f}, modelo guardado en {model_out}")

            # Guardar checkpoint histórico
            ckpt_name = f"ep{ep:05d}_frames{global_frame_count:07d}_mean{int(mean_100):05d}.pth"
            ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
            agent.save(ckpt_path)
            print(f"  >> Checkpoint guardado en {ckpt_path}")

    env.close()
    print("Entrenamiento terminado.")
    print(f"Mejor media(100) = {best_mean:.1f}")
    # Guardar último modelo por si sirve con otras seeds
    agent.save(model_out)
    print(f"Modelo final guardado en {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrena un agente DUELING DOUBLE DQN para ALE/Galaxian-v5 con checkpoints."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Máximo de episodios de esta corrida (el entrenamiento puede parar antes si se alcanza max_frames).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Semilla")
    parser.add_argument(
        "--out",
        type=str,
        default="models/dqn_galaxian.pth",
        help="Ruta donde guardar el mejor modelo global.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1_000_000,
        help="Máximo de frames (pasos de entorno) para entrenar. Usa ~1e6 para tu objetivo.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="models/checkpoints",
        help="Carpeta donde guardar checkpoints de mejores promedios.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Si se indica, no intenta cargar un modelo previo y empieza desde cero.",
    )
    args = parser.parse_args()

    train_dqn(
        total_episodes=args.episodes,
        max_steps_per_episode=20000,
        max_frames=args.max_frames,
        seed=args.seed,
        model_out=args.out,
        checkpoints_dir=args.checkpoints_dir,
        resume=not args.no_resume,
    )
