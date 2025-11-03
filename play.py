"""
===========================================================
UNIVERSIDAD DEL VALLE DE GUATEMALA
Facultad de Ingeniería
Departamento de Ciencias de la Computación
===========================================================
Proyecto: Interfaz RL para ALE/Galaxian-v5 
Autor   : José Pablo Orellana
Curso   : Aprendizaje por Refuerzo
===========================================================
Requisitos:
    pip install gymnasium ale-py numpy moviepy imageio imageio-ffmpeg
    AutoROM --accept-license --install-dir ~/.ale_roms
    export ALE_ROM_DIR=~/.ale_roms
"""

# -----------------------------------------------------------
# Librerías y dependencias
# -----------------------------------------------------------
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import gymnasium as gym
import ale_py  


# -----------------------------------------------------------
# Configuración general
# -----------------------------------------------------------
STUDENT_EMAIL_DEFAULT = "ore21970@uvg.edu.gt"
DEFAULT_OUT_DIR = "videos"


# -----------------------------------------------------------
# Comprobación de dependencias de video
# -----------------------------------------------------------
def _ensure_video_deps() -> None:
    try:
        import moviepy  # noqa: F401
    except Exception as e:
        raise RuntimeError("Falta 'moviepy'. Instalar con: pip install moviepy") from e

    try:
        import imageio_ffmpeg  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Falta 'imageio-ffmpeg'. Instalar con: pip install imageio-ffmpeg"
        ) from e


# -----------------------------------------------------------
# Creación del entorno
# -----------------------------------------------------------
def _make_env_for_recording() -> gym.Env:
    return gym.make("ALE/Galaxian-v5", render_mode="rgb_array")


# -----------------------------------------------------------
# Política (interfaz flexible)
# -----------------------------------------------------------
def _coerce_policy(policy: Any, action_space: gym.spaces.Space, seed: int | None):
    if policy is None:
        rng = np.random.default_rng(seed)  # dejo rng por si quiero usarlo luego
        return lambda obs: int(action_space.sample())
    if callable(policy):
        return policy
    if hasattr(policy, "predict") and callable(policy.predict):
        return lambda obs: int(policy.predict(obs))
    raise TypeError("Política inválida: usar callable, .predict(obs) o None (aleatoria).")


# -----------------------------------------------------------
# Ejecución y grabación del episodio
# -----------------------------------------------------------
def _record_episode(
    policy: Any,
    student_email: str,
    seed: int | None,
    out_dir: str,
) -> tuple[str, int, int, str]:
    from gymnasium.wrappers import RecordVideo

    # Preparar carpetas
    out_path = Path(out_dir)
    tmp_dir = out_path / "tmp"
    out_path.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Crear entorno y fijar semillas
    base_env = _make_env_for_recording()
    if seed is not None:
        base_env.reset(seed=seed)
        try:
            base_env.action_space.seed(seed)
            base_env.observation_space.seed(seed)
        except Exception:
            pass

    # Envolver con grabador
    env = RecordVideo(
        base_env,
        video_folder=str(tmp_dir),
        episode_trigger=lambda ep_id: True,  # grabo todos los episodios
        name_prefix="galaxian",
        disable_logger=True,
    )

    # Política efectiva
    pol = _coerce_policy(policy, env.action_space, seed)

    # Ejecutar episodio completo
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action = pol(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated

    # Cerrar para asegurar el flush del archivo de video
    env.close()

    # Localizar y renombrar el MP4
    produced = sorted(tmp_dir.glob("*.mp4"))
    if not produced:
        raise RuntimeError(
            "No se generó MP4. Verificar 'moviepy' e 'imageio-ffmpeg'."
        )
    latest_mp4 = max(produced, key=os.path.getmtime)

    ts = datetime.now()
    ts_str = ts.strftime("%Y%m%d%H%M")
    final_name = f"{student_email}_{ts_str}_{int(total_reward)}.mp4"
    final_path = out_path / final_name
    latest_mp4.replace(final_path)

    return str(final_path), int(total_reward), steps, ts_str


# -----------------------------------------------------------
# Salida a consola 
# -----------------------------------------------------------
def _print_summary_box(video_path: str, score: int, steps: int, ts: str, email: str) -> None:
    lines = [
        "RESUMEN DEL EPISODIO (Galaxian)",
        f"Correo      : {email}",
        f"Timestamp   : {ts}",
        f"Puntuación  : {score}",
        f"Pasos       : {steps}",
        f"Video MP4   : {video_path}",
    ]
    width = max(len(s) for s in lines) + 4
    top = "┌" + "─" * (width - 2) + "┐"
    bottom = "└" + "─" * (width - 2) + "┘"
    print(top)
    for s in lines:
        print("│ " + s.ljust(width - 3) + "│")
    print(bottom)


# -----------------------------------------------------------
# CLI principal
# -----------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta y GRABA por defecto un episodio de ALE/Galaxian-v5."
    )
    parser.add_argument("--seed", type=int, default=None, help="Semilla para reproducibilidad")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_DIR, help="Carpeta de salida del MP4")
    parser.add_argument("--email", type=str, default=STUDENT_EMAIL_DEFAULT, help="Correo para nombrar el MP4")
    args = parser.parse_args()

    # 1) Dependencias de video
    _ensure_video_deps()

    # 2) Ejecutar y grabar (política aleatoria por defecto)
    video_path, score, steps, ts = _record_episode(
        policy=None,            
        student_email=args.email,
        seed=args.seed,
        out_dir=args.out,
    )

    # 3) Resumen ordenado en consola
    _print_summary_box(video_path, score, steps, ts, args.email)


if __name__ == "__main__":
    main()
