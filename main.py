"""Evaluate a policy served over a websocket port against the RoboMME benchmark.

Mirrors robomme_policy_learning/examples/robomme/eval.py, but:
  * uses RoboMME's BenchmarkEnvBuilder directly (no external env_runner module);
  * inlines the MME-VLA websocket protocol (reset / add_buffer / infer) so it
    works with the upstream openpi-client in this repo's pyproject;
  * drops the VLM subgoal predictor path (set prompt = task_goal only).

Example:
    uv run python scripts/main.py --host 0.0.0.0 --port 8011 \
        --policy-name my_vla --only-tasks PickXtimes,BinFill
"""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import imageio
import numpy as np
import tyro
import websockets.sync.client
from openpi_client import msgpack_numpy

from robomme.env_record_wrapper import BenchmarkEnvBuilder


TASK_WITH_VIDEO_DEMO = {
    "VideoUnmask", "VideoUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder",
    "VideoRepick", "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
}


# --------------------------------------------------------------------------- #
# Websocket client matching the MME-VLA server protocol:
#   reset        : send {"reset": True} -> wait until response has reset_finished
#   add_buffer   : send {"images", "state", "add_buffer": True, "exec_start_idx"}
#   infer        : send obs dict, receive {"actions": [...]}
# --------------------------------------------------------------------------- #
class MMEVLAClient:
    def __init__(self, host: str, port: int, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        while True:
            try:
                self._ws = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None,
                    additional_headers=headers,
                    ping_timeout=600, open_timeout=60, close_timeout=60,
                )
                break
            except ConnectionRefusedError:
                print(f"[client] waiting for server at {self._uri}...")
                time.sleep(5)
        self.server_metadata = msgpack_numpy.unpackb(self._ws.recv())

    def _call(self, payload: dict) -> dict:
        self._ws.send(self._packer.pack(payload))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"Server error:\n{resp}")
        return msgpack_numpy.unpackb(resp)

    def reset(self) -> dict:
        return self._call({"reset": True})

    def add_buffer(self, buffer: dict) -> dict:
        return self._call(buffer)

    def infer(self, obs: dict) -> dict:
        return self._call(obs)


# --------------------------------------------------------------------------- #
# Helpers (verbatim shape of robomme_policy_learning/examples/robomme/utils.py)
# --------------------------------------------------------------------------- #
def pack_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)


def pack_buffer(image_buffer, wrist_image_buffer, state_buffer, exec_start_idx: int = 0) -> dict:
    return {
        "images": np.stack(image_buffer, axis=0).astype(np.uint8)[:, None],
        "wrist_images": np.stack(wrist_image_buffer, axis=0).astype(np.uint8)[:, None],
        "state": np.stack(state_buffer, axis=0).astype(np.float32),
        "add_buffer": True,
        "exec_start_idx": exec_start_idx,
    }


class EpisodeState:
    def __init__(self) -> None:
        self.image_buffer: list[np.ndarray] = []
        self.wrist_image_buffer: list[np.ndarray] = []
        self.state_buffer: list[np.ndarray] = []
        self.action_plan: collections.deque = collections.deque()
        self.count = 0
        self.exec_start_idx = 0

    def add_observation(self, img: np.ndarray, wrist: np.ndarray, state: np.ndarray) -> None:
        self.image_buffer.append(img.copy())
        self.wrist_image_buffer.append(wrist.copy())
        self.state_buffer.append(state.copy())

    def clear_buffers(self) -> None:
        self.image_buffer.clear()
        self.wrist_image_buffer.clear()
        self.state_buffer.clear()
        self.exec_start_idx = 0

    def current(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.image_buffer[-1], self.wrist_image_buffer[-1], self.state_buffer[-1]


class RolloutRecorder:
    BORDER_COLOR = (255, 0, 0)
    BORDER_THICKNESS = 10

    def __init__(self, save_dir: Path, fps: int = 30) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        self.fps = fps
        self.frames: list[np.ndarray] = []

    def record(self, image: np.ndarray, wrist: np.ndarray, is_demo: bool = False) -> None:
        frame = np.concatenate([image, wrist], axis=1).astype(np.uint8)
        if is_demo:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), self.BORDER_COLOR, self.BORDER_THICKNESS)
        self.frames.append(frame)

    def save(self, filename: str) -> None:
        imageio.mimsave(self.save_dir / filename, self.frames, fps=self.fps)
        self.frames.clear()


# --------------------------------------------------------------------------- #
# Env wrapper (parallels examples/robomme/env_runner.py)
# --------------------------------------------------------------------------- #
class EnvRunner:
    def __init__(self, env_id: str, max_steps: int) -> None:
        self.env_id = env_id
        self.env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="test",
            action_space="joint_angle",
            gui_render=False,
            max_steps=max_steps,
        )
        self.env: Any = None
        self.episode_id: Optional[int] = None
        self.info: dict = {}
        self.task_goal: str = ""

    @property
    def num_episodes(self) -> int:
        return self.env_builder.get_episode_num()

    def make_env(self, episode_id: int) -> None:
        self.env = self.env_builder.make_env_for_episode(episode_id)
        self.episode_id = episode_id

    def get_init_obs(self) -> dict:
        obs, self.info = self.env.reset()
        goal = self.info["task_goal"]
        self.task_goal = goal[0] if isinstance(goal, list) else goal
        states = [
            pack_state(j, g)
            for j, g in zip(obs["joint_state_list"], obs["gripper_state_list"])
        ]
        return {
            "images": obs["front_rgb_list"],
            "wrist_images": obs["wrist_rgb_list"],
            "states": states,
            "task_goal": self.task_goal,
        }

    def step(self, action: np.ndarray) -> Tuple[Tuple[Any, Any, Any], bool, str]:
        try:
            obs, _, terminated, truncated, self.info = self.env.step(action)
        except Exception as e:
            print(f"[env] step error: {e}")
            return (None, None, None), True, "error"
        img = obs["front_rgb_list"][-1]
        wrist = obs["wrist_rgb_list"][-1]
        state = pack_state(obs["joint_state_list"][-1], obs["gripper_state_list"][-1])
        return (img, wrist, state), terminated or truncated, self.info.get("status", "unknown")

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
            self.episode_id = None


# --------------------------------------------------------------------------- #
# Args & main loop
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 9000
    api_key: Optional[str] = None

    obs_horizon: int = 16
    max_steps: int = 1300

    save_dir: str = "runs/evaluation"
    policy_name: str = "default"
    model_seed: int = 42
    model_ckpt_id: int = 0
    use_history: bool = True
    overwrite: bool = False

    only_tasks: str = ""
    exclude_tasks: str = ""
    re_eval_tasks: str = ""


def _eval_episode(client: MMEVLAClient, runner: EnvRunner, args: Args, video_dir: Path) -> str:
    resp = client.reset()
    while not resp.get("reset_finished", False):
        time.sleep(0.1)

    state = EpisodeState()
    pre = runner.get_init_obs()
    task_goal = pre["task_goal"]
    print(f"[eval] task={runner.env_id} ep={runner.episode_id} goal={task_goal}")

    recorder = RolloutRecorder(video_dir)
    state.image_buffer.extend(pre["images"])
    state.wrist_image_buffer.extend(pre["wrist_images"])
    state.state_buffer.extend(pre["states"])
    for i, (im, wr) in enumerate(zip(pre["images"], pre["wrist_images"])):
        is_demo = runner.env_id in TASK_WITH_VIDEO_DEMO and i < len(pre["images"]) - 1
        recorder.record(im, wr, is_demo=is_demo)

    state.exec_start_idx = len(state.image_buffer) - 1
    img, wrist, robot_state = state.current()
    success_flag = "unknown"

    while True:
        if not state.action_plan:
            if args.use_history:
                resp = client.add_buffer(pack_buffer(
                    state.image_buffer,
                    state.wrist_image_buffer,
                    state.state_buffer,
                    state.exec_start_idx,
                ))
                while not resp.get("add_buffer_finished", False):
                    time.sleep(0.1)

            element = {
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": robot_state,
                "prompt": task_goal,
            }
            action_chunk = client.infer(element)["actions"][: args.obs_horizon]
            state.action_plan.extend(action_chunk)
            state.clear_buffers()

        action = state.action_plan.popleft()
        obs, stop, success_flag = runner.step(action)
        state.count += 1

        if state.count > args.max_steps:
            success_flag = "timeout"
            break
        if success_flag == "error":
            break

        img, wrist, robot_state = obs
        state.add_observation(img, wrist, robot_state)
        recorder.record(img, wrist)

        if stop:
            break

    if success_flag not in ("unknown", "error"):
        recorder.save(f"{runner.env_id}_ep{runner.episode_id}_{success_flag}_{task_goal}.mp4")
    return success_flag


def _setup_save_dir(args: Args) -> Path:
    out = Path(args.save_dir) / args.policy_name / f"ckpt{args.model_ckpt_id}" / f"seed{args.model_seed}"
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
        print(f"[main] overwriting {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_progress(save_dir: Path, args: Args) -> dict:
    prog = save_dir / "progress.json"
    final = save_dir / "log.json"
    if prog.exists():
        log_dict = json.loads(prog.read_text())
    elif final.exists():
        log_dict = json.loads(final.read_text())
        log_dict.pop("success_rate", None)
        log_dict.pop("total_success_rate", None)
    else:
        log_dict = {}

    for task in list(log_dict.keys()):
        for ep in [k for k, v in log_dict[task].items() if v == "error"]:
            log_dict[task].pop(ep)

    if args.re_eval_tasks:
        for t in args.re_eval_tasks.split(","):
            log_dict.pop(t, None)
            os.system(f"rm -f {save_dir / 'videos' / f'{t}_ep*.mp4'}")

    prog.write_text(json.dumps(log_dict, indent=2))
    return log_dict


def main(args: Args) -> None:
    save_dir = _setup_save_dir(args)
    video_dir = save_dir / "videos"
    log_dict = _load_progress(save_dir, args)

    tasks = list(BenchmarkEnvBuilder.get_task_list())
    if args.only_tasks:
        only = set(args.only_tasks.split(","))
        tasks = [t for t in tasks if t in only]
    if args.exclude_tasks:
        excl = set(args.exclude_tasks.split(","))
        tasks = [t for t in tasks if t not in excl]
        for t in excl:
            log_dict[t] = {str(i): False for i in range(50)}

    client = MMEVLAClient(args.host, args.port, args.api_key)
    print(f"[main] connected to {args.host}:{args.port}; server metadata: {client.server_metadata}")

    while not (save_dir / "log.json").exists():
        for task in tasks:
            log_dict.setdefault(task, {})
            runner = EnvRunner(task, max_steps=args.max_steps)
            print(f"[main] task {task}: {runner.num_episodes} episodes")

            for ep in range(runner.num_episodes):
                if str(ep) in log_dict[task]:
                    print(f"[main] {task}/ep{ep} already done, skip")
                    continue

                runner.make_env(ep)
                try:
                    flag = _eval_episode(client, runner, args, video_dir)
                    log_dict[task][ep] = "error" if flag in ("unknown", "error") else (flag == "success")
                except Exception as e:
                    print(f"[main] {task}/ep{ep} failed: {e}")
                    log_dict[task][ep] = "error"

                runner.close()
                (save_dir / "progress.json").write_text(json.dumps(log_dict, indent=2))
            del runner
            time.sleep(1)

        try:
            per_task = {
                t: sum(bool(v) for v in d.values()) / len(d)
                for t, d in log_dict.items() if d
            }
            final = {
                "success_rate": per_task,
                "total_success_rate": sum(per_task.values()) / max(len(per_task), 1),
            }
            (save_dir / "log.json").write_text(json.dumps(final, indent=2))
            print(f"[main] done. total success rate: {final['total_success_rate']:.3f}")
        except Exception as e:
            print(f"[main] error saving final results: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
