# Interfaz RL – ALE/Galaxian-v5 

**Autor:** José Pablo Orellana 

**Curso:** Aprendizaje por Refuerzo 

**Laboratorio:** Laboratorio #10 – Interfaz del Proyecto Final

---

## 1) Propósito del laboratorio
Preparar el entorno y la interfaz necesarios para la presentación del proyecto final. Este laboratorio deja listo el script capaz de reproducir un episodio en ALE/Galaxian-v5, grabar el video de la ejecución y nombrarlo con el patrón solicitado.

---

## 2) Entorno y código base

### 2.1 Objetivos y mecánicas del juego
**Galaxian** es un “shooter” vertical de Atari: la nave se mueve horizontalmente en la parte inferior, dispara hacia arriba y debe eliminar enemigos que descienden y atacan. El puntaje aumenta al destruir enemigos; el éxito se mide por la **puntuación total** al terminar el episodio.

### 2.2 Definición del estado (observación)
El entorno devuelve **frames RGB** por paso. Para esta interfaz, la política consume la **observación cruda** que entrega Gymnasium al ejecutar el episodio.

### 2.3 Acciones disponibles
El espacio de acción es **discreto** (p. ej., `NOOP`, `LEFT`, `RIGHT`, `FIRE` y combinaciones como `LEFTFIRE`, `RIGHTFIRE`). La política debe retornar un **entero** válido del espacio de acción.

---

## 3) Implementación requerida

Archivo principal: **`play.py`** con una función que ejecuta un episodio, lo **graba** y nombra el MP4 con el formato:
```
<correo>_<YYYYMMDDHHMM>_<puntuación>.mp4
```
Ejemplo: `ore21970_20251102_910.mp4`.

En este proyecto, `play.py` **graba por defecto** al ejecutar `python play.py`, genera el MP4 con el nombre requerido y muestra un **resumen en consola** (correo, timestamp, pasos y puntuación).

---

## 4) Estructura del proyecto
```
.
├── play.py            # Ejecuta y graba un episodio por defecto
└── videos/            # MP4 de salida (<correo>_<YYYYMMDDHHMM>_<score>.mp4)
```

---

## 5) Requisitos e instalación

### 5.1 Entorno virtual 
```bash
conda create -n Aprendizaje python=3.10 -y
conda activate Aprendizaje
```
### 5.2 Dependencias
```bash
pip install gymnasium ale-py numpy moviepy imageio imageio-ffmpeg
```
### 5.3 ROMs de Atari (AutoROM)
```bash
pip install autorom
AutoROM --accept-license --install-dir ~/.ale_roms
# Si es necesario en la sesión actual:
export ALE_ROM_DIR=$HOME/.ale_roms
```
**Smoke test (opcional):**
```bash
python -c "import ale_py, gymnasium as gym; env=gym.make('ALE/Galaxian-v5', render_mode='rgb_array'); print(env.action_space)"
# Debe imprimir algo como: Discrete(6)
```

---
## 6) Uso

### 6.1 Ejecutar y grabar (por defecto)
```bash
python play.py
```

---
