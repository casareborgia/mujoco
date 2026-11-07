import argparse
from collections import deque
from pathlib import Path
from typing import Any
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register, registry
from gymnasium.utils import EzPickle
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

XML_PATH = Path(__file__).resolve().parent / "world.xml"
MODEL_SAVE_PATH = Path(__file__).resolve().parent / "ppo_two_stage_maze"
ENV_ID = "TwoStageBallMaze-v0"

ARENA_SURFACE_Z = 0.30
BALL_RADIUS = 0.07
START_XY = np.array([0.0, -1.65], dtype=np.float64)
HOLE_XY = np.array([1.30, 1.30], dtype=np.float64)       # 2단계 진정한 골인 지점 (+1000점)
TRAP_XY = np.array([-1.30, 1.30], dtype=np.float64)      # 2단계 잘못된 선택 함정 홀 (-500점)
HOLE_RADIUS = 0.14
TRAP_RADIUS = 0.16

# 1단계 & 2단계 미로 내 시행착오(Trial & Error, 각 +10점) 탐색 체크포인트 및 분기 구역
TRIAL_NODES = np.array([
    [-1.35, -1.35],  # Node 1: 1단계 좌측 하단 탐색 분기 (+10점)
    [1.35, -1.35],   # Node 2: 1단계 우측 하단 탐색 분기 (+10점)
    [-0.85, -0.55],  # Node 3: 1단계 좌측 중단 막다른 길 시행착오 (+10점)
    [0.00, -0.45],   # Node 4: 1단계 중앙 포켓 시행착오 (+10점)
    [1.15, -0.45],   # Node 5: 1단계 우측 통로 분기 (+10점)
    [-0.35, 0.35],   # Node 6: 2단계 좌/중앙 경계 탐색 (+10점)
    [0.45, 0.45],    # Node 7: 2단계 중앙 배플 우회 분기 (+10점)
    [1.45, 0.45],    # Node 8: 2단계 우측 골인 진입 관문 (+10점)
], dtype=np.float64)

DEFAULT_CAMERA_CONFIG = {
    "distance": 5.2,
    "azimuth": -90.0,
    "elevation": -84.0,
    "lookat": np.array([0.0, 0.0, 0.30]),
}


class GeodesicField:
  """2단계 복합 미로(장애물 + 함정 구역 회피)를 스캔하여 목표 지점까지의 최단 우회 거리장(BFS)을 계산합니다."""

  def __init__(self, model: mujoco.MjModel, target_xy: np.ndarray = HOLE_XY, grid_res: int = 89):
    self.grid_res = grid_res
    self.coords = np.linspace(-1.92, 1.92, grid_res)
    self.step_size = float(self.coords[1] - self.coords[0])
    self.blocked = np.zeros((grid_res, grid_res), dtype=bool)
    self.target_xy = target_xy.copy()

    obs_geoms = []
    for gid in range(model.ngeom):
      name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
      if name.startswith("maze_") or name.startswith("bumper_"):
        pos = model.geom_pos[gid][:2].copy()
        size = model.geom_size[gid].copy()
        quat = model.geom_quat[gid].copy()
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)
        rot2d = mat.reshape(3, 3)[:2, :2]
        gtype = model.geom_type[gid]
        obs_geoms.append((gtype, pos, size, rot2d))

    margin = BALL_RADIUS + 0.035
    for ix, x in enumerate(self.coords):
      for iy, y in enumerate(self.coords):
        pt = np.array([x, y], dtype=np.float64)
        if np.linalg.norm(pt - TRAP_XY) <= TRAP_RADIUS + margin:
          self.blocked[ix, iy] = True
          continue
        for gtype, pos, size, rot2d in obs_geoms:
          rel = rot2d.T @ (pt - pos)
          if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            if np.hypot(rel[0], rel[1]) <= size[0] + margin:
              self.blocked[ix, iy] = True
              break
          elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            if (
                abs(rel[0]) <= size[0] + margin
                and abs(rel[1]) <= size[1] + margin
            ):
              self.blocked[ix, iy] = True
              break

    self.dist_map = np.full((grid_res, grid_res), 12.0, dtype=np.float64)
    hx = int(np.argmin(np.abs(self.coords - target_xy[0])))
    hy = int(np.argmin(np.abs(self.coords - target_xy[1])))
    self.dist_map[hx, hy] = 0.0
    queue = deque([(hx, hy)])
    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.414),
        (-1, 1, 1.414),
        (1, -1, 1.414),
        (1, 1, 1.414),
    ]
    while queue:
      cx, cy = queue.popleft()
      cd = self.dist_map[cx, cy]
      for dx, dy, w in neighbors:
        nx, ny = cx + dx, cy + dy
        if 0 <= nx < grid_res and 0 <= ny < grid_res and not self.blocked[nx, ny]:
          nd = cd + w * self.step_size
          if nd < self.dist_map[nx, ny]:
            self.dist_map[nx, ny] = nd
            queue.append((nx, ny))

  def query(self, xy: np.ndarray) -> tuple[float, np.ndarray]:
    ix = int(np.clip(np.round((xy[0] + 1.92) / self.step_size), 1, self.grid_res - 2))
    iy = int(np.clip(np.round((xy[1] + 1.92) / self.step_size), 1, self.grid_res - 2))
    d = float(self.dist_map[ix, iy])

    best_dir = self.target_xy - xy
    best_d = d
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        if dx == 0 and dy == 0:
          continue
        cand_d = float(self.dist_map[ix + dx, iy + dy])
        if cand_d < best_d:
          best_d = cand_d
          best_dir = np.array([dx, dy], dtype=np.float64)
    norm = float(np.linalg.norm(best_dir))
    return d, best_dir / max(norm, 1e-5)


class TwoStageBallMazeEnv(MujocoEnv, EzPickle):
  """
  Farama Gymnasium 공식 MuJoCo 환경 규격(https://gymnasium.farama.org/environments/mujoco/)을
  준수하는 2단계 선택형 복잡 미로(Two-Stage Branching Maze) 강화학습 환경.

  - 환경 ID: `TwoStageBallMaze-v0`
  - 상속: `gymnasium.envs.mujoco.MujocoEnv`, `gymnasium.utils.EzPickle`
  - 보상 체계:
    1) 골인지점(1.30, 1.30) 도달 시: +1,000점
    2) 시행착오(Trial & Error 탐색 체크포인트/구역 개척) 시: 회당 +10점
    3) 아무것도 하지 않을 시(무행동 정지): -1,000점
    4) 길을 못 찾을 시(2단계 오답 함정 진입 또는 제한시간 초과): -500점
  """

  metadata = {
      "render_modes": [
          "human",
          "rgb_array",
          "depth_array",
          "rgbd_tuple",
      ],
      "render_fps": 50,
  }

  def __init__(
      self,
      xml_file: str | Path = XML_PATH,
      frame_skip: int = 10,
      default_camera_config: dict[str, Any] = DEFAULT_CAMERA_CONFIG,
      curriculum: bool = True,
      render_mode: str | None = None,
      **kwargs,
  ):
    EzPickle.__init__(
        self,
        xml_file,
        frame_skip,
        default_camera_config,
        curriculum,
        render_mode,
        **kwargs,
    )
    self.curriculum = curriculum
    self._ray_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    self._step_count = 0
    self._max_episode_steps = 450

    self.episode_score = 0.0
    self.trial_count = 0
    self.visited_nodes = np.zeros(len(TRIAL_NODES), dtype=bool)
    self.visited_cells: set[tuple[int, int]] = set()
    self._pos_history: deque[np.ndarray] = deque(maxlen=45)
    self._prev_potential = 0.0

    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
    )

    MujocoEnv.__init__(
        self,
        model_path=str(xml_file),
        frame_skip=frame_skip,
        observation_space=observation_space,
        default_camera_config=default_camera_config,
        render_mode=render_mode,
        **kwargs,
    )

    # 행동 공간: 정규화된 2차원 구름 힘 제어 [-1.0, 1.0] (실제 ±3.5 N으로 매핑)
    self.action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(2,), dtype=np.float32
    )
    self.geodesic = GeodesicField(self.model, target_xy=HOLE_XY)
    self.ball_body_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_BODY, "rolling_ball"
    )

  def _cast_lidar_rays(self, ball_pos: np.ndarray) -> np.ndarray:
    max_range = 1.5
    pnt = np.array([ball_pos[0], ball_pos[1], ARENA_SURFACE_Z + BALL_RADIUS], dtype=np.float64)
    geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    geomid = np.zeros(1, dtype=np.int32)
    dists = np.empty(8, dtype=np.float32)
    for i, ang in enumerate(self._ray_angles):
      vec = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float64)
      d = mujoco.mj_ray(
          self.model, self.data, pnt, vec, geomgroup, 1, self.ball_body_id, geomid
      )
      if d < 0 or d > max_range:
        dists[i] = 1.0
      else:
        dists[i] = float(d / max_range)
    return dists

  def _get_obs(self) -> np.ndarray:
    ball_pos = self.data.qpos[0:3]
    ball_vel = self.data.qvel[0:3]
    xy = ball_pos[:2]
    hole_delta = HOLE_XY - xy
    trap_delta = TRAP_XY - xy
    dist_hole = float(np.linalg.norm(hole_delta))
    if dist_hole < 0.42:
      guide_dir = hole_delta / max(dist_hole, 1e-5)
    else:
      _, guide_dir = self.geodesic.query(xy)

    stage_flag = np.array(
        [1.0 if xy[1] >= 0.0 else -1.0, float(self.trial_count) / 10.0],
        dtype=np.float32,
    )
    lidar = self._cast_lidar_rays(ball_pos)
    return np.concatenate([
        (xy / 2.0).astype(np.float32),
        (ball_vel[:2] / 2.5).astype(np.float32),
        (hole_delta / 3.5).astype(np.float32),
        (trap_delta / 3.5).astype(np.float32),
        guide_dir.astype(np.float32),
        stage_flag,
        lidar,
    ])

  def is_in_goal(self) -> bool:
    ball_pos = self.data.qpos[0:3]
    dist_xy = float(np.linalg.norm(ball_pos[:2] - HOLE_XY))
    return dist_xy < HOLE_RADIUS and float(ball_pos[2]) < 0.27

  def is_in_trap(self) -> bool:
    ball_pos = self.data.qpos[0:3]
    dist_xy = float(np.linalg.norm(ball_pos[:2] - TRAP_XY))
    return dist_xy < TRAP_RADIUS

  def reset_model(self) -> np.ndarray:
    """Gymnasium MujocoEnv 표준 reset_model 구현."""
    qpos = self.init_qpos.copy()
    qvel = self.init_qvel.copy()

    if self.curriculum and self.np_random.random() < 0.40:
      candidates = [
          [0.0, -1.65],
          [1.45, -0.80],
          [0.0, -0.30],
          [1.35, 0.20],
          [0.45, 0.40],
      ]
      base_xy = np.array(candidates[int(self.np_random.integers(len(candidates)))])
      start_xy = base_xy + self.np_random.uniform(-0.12, 0.12, size=2)
    else:
      start_xy = START_XY + self.np_random.uniform(low=[-0.22, -0.12], high=[0.22, 0.12])

    qpos[0] = start_xy[0]
    qpos[1] = start_xy[1]
    qpos[2] = ARENA_SURFACE_Z + BALL_RADIUS + 0.002
    if self.model.nq >= 14:
      qpos[7] = start_xy[0]
      qpos[8] = start_xy[1]
      qpos[9] = ARENA_SURFACE_Z + BALL_RADIUS + 0.002

    self.set_state(qpos, qvel)

    self._step_count = 0
    self.episode_score = 0.0
    self.trial_count = 0
    self.visited_nodes[:] = False
    cell_x = int(np.floor((start_xy[0] + 2.0) / 0.65))
    cell_y = int(np.floor((start_xy[1] + 2.0) / 0.65))
    self.visited_cells = {(cell_x, cell_y)}
    self._pos_history.clear()
    self._pos_history.append(start_xy.copy())

    geo_d, _ = self.geodesic.query(start_xy)
    self._prev_potential = -40.0 * geo_d
    return self._get_obs()

  def _get_reset_info(self) -> dict[str, Any]:
    dist_to_hole = float(np.linalg.norm(self.data.qpos[0:2] - HOLE_XY))
    return {
        "distance_to_hole": dist_to_hole,
        "episode_score": self.episode_score,
        "trial_count": self.trial_count,
    }

  def step(self, action: np.ndarray):
    action_clipped = np.clip(action, -1.0, 1.0)
    force_xy = action_clipped * 3.5

    for _ in range(self.frame_skip):
      ball_pos_inner = self.data.qpos[0:3]
      ball_vel_inner = self.data.qvel[0:3]
      dist_inner = float(np.linalg.norm(ball_pos_inner[:2] - HOLE_XY))
      if dist_inner < 0.08 and float(ball_pos_inner[2]) >= 0.27:
        self.data.ctrl[0:2] = 0.0
        self.data.qvel[0:2] *= 0.85
      else:
        self.data.ctrl[0] = force_xy[0] - 0.65 * ball_vel_inner[0]
        self.data.ctrl[1] = force_xy[1] - 0.65 * ball_vel_inner[1]
      mujoco.mj_step(self.model, self.data)
    mujoco.mj_rnePostConstraint(self.model, self.data)

    self._step_count += 1
    ball_pos = self.data.qpos[0:3]
    ball_vel = self.data.qvel[0:3]
    xy = ball_pos[:2].copy()
    self._pos_history.append(xy)

    step_score_delta = 0.0
    outcome = "running"

    # 1) 시행착오(Trial & Error) 점수 판정 (+10점)
    for idx, node_xy in enumerate(TRIAL_NODES):
      if not self.visited_nodes[idx] and float(np.linalg.norm(xy - node_xy)) < 0.32:
        self.visited_nodes[idx] = True
        self.trial_count += 1
        step_score_delta += 10.0

    cell_id = (int(np.floor((xy[0] + 2.0) / 0.65)), int(np.floor((xy[1] + 2.0) / 0.65)))
    if cell_id not in self.visited_cells and self.trial_count < 12:
      self.visited_cells.add(cell_id)
      self.trial_count += 1
      step_score_delta += 10.0

    # 2) 종료 조건 및 대형 보상/페널티 판정
    in_goal = self.is_in_goal()
    in_trap = self.is_in_trap()

    inaction = False
    if len(self._pos_history) >= 45 and self._step_count >= 60:
      pts = np.array(self._pos_history)
      max_disp = float(np.max(np.linalg.norm(pts - pts[0], axis=1)))
      speed = float(np.linalg.norm(ball_vel[:2]))
      if max_disp < 0.04 and speed < 0.05:
        inaction = True

    terminated = False
    truncated = False

    if in_goal:
      step_score_delta += 1000.0
      terminated = True
      outcome = "goal"
    elif inaction:
      step_score_delta -= 1000.0
      terminated = True
      outcome = "inaction"
    elif in_trap:
      step_score_delta -= 500.0
      terminated = True
      outcome = "trap_lost"
    elif self._step_count >= self._max_episode_steps:
      step_score_delta -= 500.0
      truncated = True
      outcome = "timeout_lost"

    self.episode_score += step_score_delta

    # 3) 2단계 미로 최단 우회 경로 진행도(Potential Shaping) 결합
    dist_to_hole = float(np.linalg.norm(xy - HOLE_XY))
    if dist_to_hole < 0.42:
      geo_d = dist_to_hole
      guide_dir = (HOLE_XY - xy) / max(dist_to_hole, 1e-5)
    else:
      geo_d, guide_dir = self.geodesic.query(xy)

    vel_align = float(np.dot(ball_vel[:2], guide_dir))
    cur_potential = -40.0 * geo_d
    shaping = (
        (cur_potential - self._prev_potential)
        + 0.35 * vel_align
        - 0.02 * float(np.sum(np.square(action_clipped)))
    )
    self._prev_potential = cur_potential
    reward = step_score_delta + shaping

    if self.render_mode == "human":
      self.render()

    info = {
        "distance_to_hole": dist_to_hole,
        "in_hole": in_goal,
        "in_trap": in_trap,
        "inaction": inaction,
        "outcome": outcome,
        "episode_score": self.episode_score,
        "trial_count": self.trial_count,
    }
    return self._get_obs(), float(reward), terminated, truncated, info


# 하위 호환 별칭 유지
PureBallRollArenaEnv = TwoStageBallMazeEnv

# Farama Gymnasium 공식 환경 레지스트리에 등록 (https://gymnasium.farama.org/environments/mujoco/)
def register_gymnasium_envs() -> None:
  if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="run:TwoStageBallMazeEnv",
        max_episode_steps=450,
        reward_threshold=1000.0,
    )


register_gymnasium_envs()


def train_rl(total_timesteps: int = 120_000) -> Path:
  """Gymnasium에 등록된 `TwoStageBallMaze-v0` 환경을 사용하여 PPO 학습을 수행합니다."""
  print("=" * 80)
  print(
      f"🚀 [Gymnasium `{ENV_ID}` PPO 강화학습 시작] 목표: 점수의 극대화 (총 {total_timesteps:,} 스텝)"
  )
  print(" - 골인: +1,000점 | 시행착오 탐색: +10점 | 무행동: -1,000점 | 길 못찾음/함정: -500점")
  print("=" * 80)

  vec_env = make_vec_env(
      ENV_ID, n_envs=8, seed=42, env_kwargs={"curriculum": True}
  )
  model = PPO(
      "MlpPolicy",
      vec_env,
      learning_rate=1e-3,
      n_steps=256,
      batch_size=256,
      n_epochs=10,
      gamma=0.99,
      gae_lambda=0.95,
      ent_coef=0.005,
      policy_kwargs={"net_arch": [128, 128]},
      verbose=1,
      seed=42,
  )
  model.learn(total_timesteps=total_timesteps)
  save_file = MODEL_SAVE_PATH.with_suffix(".zip")
  model.save(str(MODEL_SAVE_PATH))
  print(f"\n✅ [PPO 학습 완료] 모델 저장 위치: {save_file}")

  eval_env = gym.make(ENV_ID, curriculum=False)
  successes = 0
  scores = []
  eval_episodes = 5
  print(f"\n📊 [학습된 `{ENV_ID}` PPO 에이전트 평가 테스트 (하단 출발 5회 시도)]")
  for ep in range(1, eval_episodes + 1):
    obs, _ = eval_env.reset(seed=200 + ep)
    done = False
    steps = 0
    info = {}
    while not done:
      action, _ = model.predict(obs, deterministic=True)
      obs, _, terminated, truncated, info = eval_env.step(action)
      done = terminated or truncated
      steps += 1
    sc = info.get("episode_score", 0.0)
    tc = info.get("trial_count", 0)
    scores.append(sc)
    if info.get("in_hole", False):
      successes += 1
      print(
          f"  - 에피소드 #{ep}: 🏆 진정한 골인 성공! | 최종 점수: {sc:+.0f}점"
          f" (골인 +1000점, 시행착오 +{tc*10}점 [{tc}회], 소요 시간={steps*0.02:.2f}s)"
      )
    else:
      print(
          f"  - 에피소드 #{ep}: ❌ 실패({info.get('outcome')}) | 최종 점수: {sc:+.0f}점"
          f" (시행착오 +{tc*10}점, 최종 홀 거리: {info.get('distance_to_hole', 0.0):.2f}m)"
      )
  eval_env.close()
  print(
      f"👉 최종 골인 성공률: {successes}/{eval_episodes} ({successes/eval_episodes*100:.0f}%)"
      f" | 평균 획득 점수: {np.mean(scores):+.1f}점"
  )
  print("=" * 80)
  return save_file


class PurePPOViewerController:
  """2단계 선택형 미로에서 무작위 굴리기(파란 공)와 점수 극대화 PPO 에이전트(주황 공)를 동시에 주행시켜 비교합니다."""

  def __init__(
      self,
      env: TwoStageBallMazeEnv,
      ppo_model: PPO | None,
      episode_sec: float = 8.5,
      seed: int = 42,
  ):
    self.env = env
    self.ppo_model = ppo_model
    self.episode_sec = episode_sec
    self.rng = np.random.default_rng(seed)
    self.attempt = 0
    self.ppo_success_count = 0
    self.rand_success_count = 0
    self.in_hole_time: float | None = None
    self.rand_done_logged: bool = False

    self.ppo_score = 0
    self.ppo_trials = 0
    self.ppo_visited_nodes = np.zeros(len(TRIAL_NODES), dtype=bool)
    self.ppo_visited_cells: set[tuple[int, int]] = set()

    self.rand_score = 0
    self.rand_trials = 0
    self.rand_visited_nodes = np.zeros(len(TRIAL_NODES), dtype=bool)
    self.rand_visited_cells: set[tuple[int, int]] = set()

    self.current_ppo_force = np.zeros(2, dtype=np.float64)
    self.current_rand_force = np.zeros(2, dtype=np.float64)
    self.substep_counter = 0
    self.ep_start_time = 0.0
    self.last_time = -1.0
    self._need_init = True

  def start_new_episode(
      self, model: mujoco.MjModel, data: mujoco.MjData
  ) -> None:
    data.qpos[:] = model.qpos0[:]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0

    self.attempt += 1
    self.in_hole_time = None
    self.rand_done_logged = False
    self.current_ppo_force[:] = 0.0
    self.current_rand_force[:] = 0.0
    self.substep_counter = 0
    self.ep_start_time = float(data.time)
    self.last_time = float(data.time)

    start_x = float(START_XY[0] + self.rng.uniform(-0.20, 0.20))
    start_y = float(START_XY[1] + self.rng.uniform(-0.10, 0.10))

    data.qpos[0] = start_x
    data.qpos[1] = start_y
    data.qpos[2] = ARENA_SURFACE_Z + BALL_RADIUS + 0.002
    if model.nq >= 14:
      data.qpos[7] = start_x
      data.qpos[8] = start_y
      data.qpos[9] = ARENA_SURFACE_Z + BALL_RADIUS + 0.002

    if model.nmocap >= 1:
      data.mocap_pos[0] = [start_x, start_y, ARENA_SURFACE_Z + 0.22]
    if model.nmocap >= 2:
      data.mocap_pos[1] = [start_x, start_y, ARENA_SURFACE_Z + 0.22]

    init_cell = (int(np.floor((start_x + 2.0) / 0.65)), int(np.floor((start_y + 2.0) / 0.65)))
    self.ppo_score = 0
    self.ppo_trials = 0
    self.ppo_visited_nodes[:] = False
    self.ppo_visited_cells = {init_cell}

    self.rand_score = 0
    self.rand_trials = 0
    self.rand_visited_nodes[:] = False
    self.rand_visited_cells = {init_cell}

    self.env.trial_count = 0
    self.env.visited_nodes[:] = False
    self.env.visited_cells = {init_cell}

    print(
        f"\n🏁 [2단계 선택형 미로 비교 주행 #{self.attempt}] 출발=({start_x:.2f}, {start_y:.2f})\n"
        f"   🟠 주황 공(PPO 점수 극대화 에이전트) vs 🔵 파란 공(무작위 시행착오)",
        flush=True,
    )

  def _update_ball_trials(
      self,
      xy: np.ndarray,
      visited_nodes: np.ndarray,
      visited_cells: set[tuple[int, int]],
      trials: int,
      score: int,
  ) -> tuple[int, int]:
    for idx, node_xy in enumerate(TRIAL_NODES):
      if not visited_nodes[idx] and float(np.linalg.norm(xy - node_xy)) < 0.32:
        visited_nodes[idx] = True
        trials += 1
        score += 10
    cell_id = (int(np.floor((xy[0] + 2.0) / 0.65)), int(np.floor((xy[1] + 2.0) / 0.65)))
    if cell_id not in visited_cells and trials < 12:
      visited_cells.add(cell_id)
      trials += 1
      score += 10
    return trials, score

  def step_callback(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if self._need_init or data.time < self.last_time:
      self._need_init = False
      self.start_new_episode(model, data)
      return
    self.last_time = float(data.time)
    ep_elapsed = float(data.time) - self.ep_start_time

    self.env.data = data
    ball_pos = data.qpos[0:3]
    ball_vel = data.qvel[0:3]
    rand_pos = data.qpos[7:10] if model.nq >= 14 else ball_pos
    rand_vel = data.qvel[6:9] if model.nv >= 12 else ball_vel

    if model.nmocap >= 1:
      data.mocap_pos[0] = [ball_pos[0], ball_pos[1], float(ball_pos[2]) + 0.15]
    if model.nmocap >= 2:
      data.mocap_pos[1] = [rand_pos[0], rand_pos[1], float(rand_pos[2]) + 0.15]

    dist_xy = float(np.linalg.norm(ball_pos[:2] - HOLE_XY))
    is_in_hole = dist_xy < HOLE_RADIUS and float(ball_pos[2]) < 0.27

    rand_dist_xy = float(np.linalg.norm(rand_pos[:2] - HOLE_XY))
    rand_in_hole = rand_dist_xy < HOLE_RADIUS and float(rand_pos[2]) < 0.27
    rand_in_trap = float(np.linalg.norm(rand_pos[:2] - TRAP_XY)) < TRAP_RADIUS

    if self.substep_counter % 10 == 0:
      if not is_in_hole:
        self.ppo_trials, self.ppo_score = self._update_ball_trials(
            ball_pos[:2],
            self.ppo_visited_nodes,
            self.ppo_visited_cells,
            self.ppo_trials,
            self.ppo_score,
        )
        self.env.trial_count = self.ppo_trials
        if self.ppo_model is not None:
          obs = self.env._get_obs()
          action, _ = self.ppo_model.predict(obs, deterministic=True)
          self.current_ppo_force = np.clip(action, -1.0, 1.0) * 3.5
        else:
          self.current_ppo_force = self.rng.uniform(-3.5, 3.5, size=2)

      if not rand_in_hole and not self.rand_done_logged:
        self.rand_trials, self.rand_score = self._update_ball_trials(
            rand_pos[:2],
            self.rand_visited_nodes,
            self.rand_visited_cells,
            self.rand_trials,
            self.rand_score,
        )
      if self.substep_counter % 20 == 0 and not rand_in_hole:
        self.current_rand_force = self.rng.uniform(-3.5, 3.5, size=2)

    self.substep_counter += 1

    # 1) 학습된 PPO 공 제어 (ctrl[0], ctrl[1])
    if is_in_hole:
      data.ctrl[0:2] = 0.0
      data.qvel[0:6] *= 0.85
      if self.in_hole_time is None:
        self.in_hole_time = float(data.time)
        self.ppo_success_count += 1
        self.ppo_score += 1000
        rand_projected = self.rand_score - 500
        print(
            f"🏆 [🟠 PPO 에이전트 2단계 골인! (+1000점)] 총점: {self.ppo_score:+d}점"
            f" (골인 +1000, 시행착오 +{self.ppo_trials*10}점 [{self.ppo_trials}회], {ep_elapsed:.2f}s)\n"
            f"   vs 🔵 무작위 공: 1단계 미로에서 방황 중 (시행착오 +{self.rand_trials*10}점,"
            f" 길 못찾음 페널티 -500점 반영 시 {rand_projected:+d}점)",
            flush=True,
        )
    elif dist_xy < 0.08 and float(ball_pos[2]) >= 0.27:
      data.ctrl[0:2] = 0.0
      data.qvel[0:2] *= 0.85
    else:
      data.ctrl[0] = self.current_ppo_force[0] - 0.65 * ball_vel[0]
      data.ctrl[1] = self.current_ppo_force[1] - 0.65 * ball_vel[1]

    # 2) 무작위 비교 공 제어 (qfrc_applied[6:8] 사용 -> Gymnasium 액션 공간(2D) 오염 방지)
    if model.nv >= 12:
      if rand_in_trap and not self.rand_done_logged:
        self.rand_done_logged = True
        self.rand_score -= 500
        data.qfrc_applied[6:8] = 0.0
        data.qvel[6:12] *= 0.8
        print(
            f"💀 [🔵 무작위 공 2단계 오답 함정 진입! (-500점)] 최종 점수: {self.rand_score:+d}점",
            flush=True,
        )
      elif rand_in_hole:
        data.qfrc_applied[6:8] = 0.0
        data.qvel[6:12] *= 0.85
      else:
        data.qfrc_applied[6] = self.current_rand_force[0] - 0.50 * rand_vel[0]
        data.qfrc_applied[7] = self.current_rand_force[1] - 0.50 * rand_vel[1]

    if (
        self.in_hole_time is not None and data.time - self.in_hole_time >= 2.2
    ) or ep_elapsed >= self.episode_sec:
      if not self.rand_done_logged and not rand_in_hole:
        self.rand_score -= 500
      self.start_new_episode(model, data)


def run_viewer(
    xml_path: Path = XML_PATH,
    mode: str = "viewer",
    episode_sec: float = 8.5,
) -> None:
  env = gym.make(ENV_ID, xml_file=xml_path, curriculum=False).unwrapped
  ppo_model = None
  if mode != "random":
    zip_path = MODEL_SAVE_PATH.with_suffix(".zip")
    if not zip_path.exists():
      print("[알림] 학습된 2단계 미로 PPO 모델이 없어 먼저 학습을 진행합니다...")
      train_rl(120_000)
    ppo_model = PPO.load(str(MODEL_SAVE_PATH))

  model = env.model
  data = env.data
  controller = PurePPOViewerController(
      env, ppo_model, episode_sec=episode_sec
  )
  controller._need_init = False
  controller.start_new_episode(model, data)
  mujoco.set_mjcb_control(controller.step_callback)

  print("=" * 80)
  print(f"[MuJoCo Viewer] Gymnasium `{ENV_ID}` 2단계 선택형 복잡 미로 시뮬레이터")
  print(" - 미로 구조: 1단계(다중 지그재그 & 막다른 함정) -> 2단계(좌측 -500점 함정 vs 우측 +1000점 골인)")
  print(" - 보상 체계: 골인 +1000점 | 시행착오 +10점 | 아무것도 안함 -1000점 | 길 못찾음 -500점")
  print("=" * 80)

  mujoco.viewer.launch(model, data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Gymnasium MuJoCo 2단계 선택형 복잡 미로 점수 극대화 강화학습 시뮬레이터"
  )
  parser.add_argument(
      "--mode",
      choices=["viewer", "train", "random"],
      default="viewer",
      help="viewer: 학습된 PPO 뷰어 실행 | train: PPO 학습 실행 | random: 무작위 굴리기 뷰어",
  )
  parser.add_argument(
      "--timesteps",
      type=int,
      default=120_000,
      help="PPO 강화학습 총 스텝 수",
  )
  parser.add_argument(
      "--episode-sec",
      type=float,
      default=8.5,
      help="에피소드당 최대 제한 시간(초)",
  )
  args = parser.parse_args()

  if args.mode == "train":
    train_rl(args.timesteps)
  else:
    run_viewer(XML_PATH, mode=args.mode, episode_sec=args.episode_sec)
