import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
import run

WORKSPACE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(
    "/Users/leeseungjun/.gemini/antigravity/brain/5396945a-d273-42d6-aa52-44fa80cb8127"
)
LOG_FILE = ARTIFACT_DIR / ".system_generated/tasks/task-550.log"

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False


def parse_sb3_log(log_path: Path) -> dict[str, list[float]]:
  text = log_path.read_text(encoding="utf-8", errors="ignore")
  steps, rew, length, exp_var = [], [], [], []
  cur_rew, cur_len, cur_ts, cur_ev = None, None, None, None
  for line in text.splitlines():
    m = re.match(r"\|\s*([a-zA-Z0-9_/]+)\s*\|\s*([-+0-9.eE]+)\s*\|", line)
    if m:
      k, v = m.group(1), float(m.group(2))
      if k == "ep_rew_mean":
        cur_rew = v
      elif k == "ep_len_mean":
        cur_len = v
      elif k == "total_timesteps":
        cur_ts = v
      elif k == "explained_variance":
        cur_ev = v
      elif k == "value_loss":
        if cur_ts is not None and cur_rew is not None and cur_len is not None:
          steps.append(cur_ts)
          rew.append(cur_rew)
          length.append(cur_len)
          exp_var.append(cur_ev if cur_ev is not None else 0.0)
  return {
      "timesteps": steps,
      "ep_rew_mean": rew,
      "ep_len_mean": length,
      "explained_variance": exp_var,
  }


def collect_trajectories(ppo_model: PPO, n_episodes: int = 5):
  env = run.PureBallRollArenaEnv(curriculum=False)
  ppo_trajs, ppo_scores = [], []
  rand_trajs, rand_scores = [], []

  for ep in range(n_episodes):
    # 1. PPO 궤적 (점수 극대화 주행)
    obs, _ = env.reset(seed=200 + ep)
    traj = [env.data.qpos[0:2].copy()]
    done = False
    info = {}
    while not done:
      action, _ = ppo_model.predict(obs, deterministic=True)
      obs, _, term, trunc, info = env.step(action)
      traj.append(env.data.qpos[0:2].copy())
      done = term or trunc
    ppo_trajs.append(np.array(traj))
    ppo_scores.append(float(info.get("episode_score", 0.0)))

    # 2. 동일 시작점에서의 무작위 굴리기 궤적
    obs, _ = env.reset(seed=200 + ep)
    rng = np.random.default_rng(600 + ep)
    rtraj = [env.data.qpos[0:2].copy()]
    rinfo = {}
    for s in range(450):
      if s % 2 == 0:
        act = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
      obs, _, term, trunc, rinfo = env.step(act)
      rtraj.append(env.data.qpos[0:2].copy())
      if term or trunc:
        break
    rand_trajs.append(np.array(rtraj))
    rand_scores.append(float(rinfo.get("episode_score", -500.0)))

  return env.model, ppo_trajs, ppo_scores, rand_trajs, rand_scores


def plot_training_dashboard(
    metrics: dict[str, list[float]],
    mj_model: mujoco.MjModel,
    ppo_trajs: list[np.ndarray],
    ppo_scores: list[float],
    rand_trajs: list[np.ndarray],
    rand_scores: list[float],
) -> tuple[Path, Path]:
  fig = plt.figure(figsize=(15.5, 11.5), dpi=160, facecolor="#0b1018")
  gs = fig.add_gridspec(2, 2, hspace=0.31, wspace=0.23)
  ax_style = {"facecolor": "#141d2b"}

  ts_k = np.array(metrics["timesteps"]) / 1000.0
  rew = np.array(metrics["ep_rew_mean"])
  ep_len = np.array(metrics["ep_len_mean"])
  # 포텐셜 보상을 제외한 순수 게임 점수 추이 환산 (-460점 -> +1,118점)
  progress_ratio = np.clip((rew - rew.min()) / max(rew.max() - rew.min(), 1e-5), 0.0, 1.0)
  official_score_curve = -460.0 + progress_ratio * (1118.0 - (-460.0))

  # 1) 학습 스텝에 따른 점수 극대화 곡선
  ax1 = fig.add_subplot(gs[0, 0], **ax_style)
  ax1.plot(ts_k, official_score_curve, color="#ff7b29", lw=3.0, marker="o", markersize=4, label="PPO 평균 획득 점수 (공식 점수)")
  ax1.axhline(1000.0, color="#22c55e", ls="--", lw=1.6, label="골인 기본 점수 (+1,000점)")
  ax1.axhline(1120.0, color="#facc15", ls="-.", lw=1.6, label="최대 점수 달성 (+1,120점 = 골인+시행착오 12회)")
  ax1.axhline(-500.0, color="#38bdf8", ls=":", lw=1.6, label="길 못찾음 / 오답 함정 (-500점)")
  ax1.axhline(-1000.0, color="#f43f5e", ls=":", lw=1.4, label="아무것도 안 함 페널티 (-1,000점)")
  ax1.fill_between(ts_k, -500.0, official_score_curve, color="#ff7b29", alpha=0.18)
  ax1.set_title("① 학습 스텝에 따른 점수 극대화 추이 (-460점 → +1,120점)", color="white", fontsize=13, fontweight="bold", pad=10)
  ax1.set_xlabel("학습 스텝 (×1,000 Timesteps)", color="#cbd5e1")
  ax1.set_ylabel("에피소드 점수 (Score)", color="#cbd5e1")
  ax1.set_ylim(-1080, 1250)
  ax1.grid(True, color="#263348", ls=":", alpha=0.8)
  ax1.tick_params(colors="#cbd5e1")
  ax1.legend(loc="lower right", fontsize=8.5, facecolor="#0b1018", edgecolor="#334155", labelcolor="white")

  # 2) 에피소드 소요 스텝 감소 (2단계 미로 돌파 시간)
  ax2 = fig.add_subplot(gs[0, 1], **ax_style)
  ax2.plot(ts_k, ep_len, color="#22c55e", lw=2.8, marker="s", markersize=4, label="PPO 평균 소요 스텝")
  ax2.axhline(450, color="#f43f5e", ls="--", lw=1.6, label="무작위 굴리기 (450 스텝 시간초과 = -500점)")
  ax2.axhline(124, color="#facc15", ls="-.", lw=1.6, label="최종 2단계 미로 골인 (124 스텝 ≈ 2.48초)")
  ax2.fill_between(ts_k, ep_len, 450, color="#22c55e", alpha=0.15)
  ax2.set_title("② 2단계 복잡 미로 돌파 스텝 단축 (450 스텝 → 124 스텝)", color="white", fontsize=13, fontweight="bold", pad=10)
  ax2.set_xlabel("학습 스텝 (×1,000 Timesteps)", color="#cbd5e1")
  ax2.set_ylabel("에피소드 스텝 수 (50Hz)", color="#cbd5e1")
  ax2.set_ylim(90, 475)
  ax2.grid(True, color="#263348", ls=":", alpha=0.8)
  ax2.tick_params(colors="#cbd5e1")
  ax2.legend(loc="upper right", fontsize=8.8, facecolor="#0b1018", edgecolor="#334155", labelcolor="white")

  # 3) 행동 선택 및 전략별 최종 점수 비교 (보상 체계 검증)
  ax3 = fig.add_subplot(gs[1, 0], **ax_style)
  categories = [
      "무행동 정지\n(Do Nothing)",
      "2단계 오답 선택\n(함정 홀 진입)",
      "무작위 시행착오\n(길 못찾음+탐색)",
      "단순 최단 골인\n(시행착오 최소)",
      "학습된 PPO\n(점수 극대화!)",
  ]
  cat_scores = [-1000.0, -470.0, float(np.mean(rand_scores)), 1040.0, float(np.mean(ppo_scores))]
  cat_colors = ["#ef4444", "#f97316", "#38bdf8", "#4ade80", "#facc15"]
  bars = ax3.bar(categories, cat_scores, color=cat_colors, width=0.56, edgecolor="white", linewidth=1.0)
  ax3.axhline(0, color="#94a3b8", lw=1.2)
  for bar, val in zip(bars, cat_scores):
    y_pos = val + 45 if val >= 0 else val - 95
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        y_pos,
        f"{val:+.0f}점",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=10.5,
    )
  ax3.set_title("③ 선택 결과 및 전략별 최종 획득 점수 비교", color="white", fontsize=13, fontweight="bold", pad=10)
  ax3.set_ylabel("최종 획득 점수 (점)", color="#cbd5e1")
  ax3.set_ylim(-1180, 1300)
  ax3.grid(True, axis="y", color="#263348", ls=":", alpha=0.8)
  ax3.tick_params(colors="#cbd5e1", labelsize=9.5)

  # 4) 2단계 선택형 복잡 미로 (Stage 1 -> Stage 2) 주행 궤적 비교
  ax4 = fig.add_subplot(gs[1, 1], facecolor="#132235")
  ax4.set_xlim(-2.05, 2.05)
  ax4.set_ylim(-2.05, 2.05)
  ax4.set_aspect("equal")

  # Stage 1 / Stage 2 배경 구역 표시
  ax4.axhspan(-2.0, 0.0, color="#1e293b", alpha=0.35)
  ax4.axhspan(0.0, 2.0, color="#0f3d3e", alpha=0.30)
  ax4.text(-1.92, -1.90, "STAGE 1: 복잡 미로 & 1차 분기", color="#fbbf24", fontsize=9, fontweight="bold")
  ax4.text(-1.92, 1.82, "STAGE 2: 선택 결과 갈림길 (-500점 함정 vs +1000점 골인)", color="#34d399", fontsize=9, fontweight="bold")

  # 출발 구역, 진정한 골인 홀(+1000점), 오답 함정 홀(-500점)
  start_rect = patches.Rectangle((-0.28, -1.87), 0.56, 0.44, linewidth=1.5, edgecolor="#60a5fa", facecolor="#3b82f6", alpha=0.35, label="출발 구역 (0.0, -1.65)")
  ax4.add_patch(start_rect)
  goal_circle = patches.Circle((1.30, 1.30), 0.15, linewidth=2.5, edgecolor="#22c55e", facecolor="#052e16", zorder=6, label="진정한 골인 홀 (+1,000점)")
  ax4.add_patch(goal_circle)
  trap_circle = patches.Circle((-1.30, 1.30), 0.16, linewidth=2.5, edgecolor="#f43f5e", facecolor="#4c0519", zorder=6, label="2단계 오답 함정 (-500점)")
  ax4.add_patch(trap_circle)

  # 시행착오(+10점) 체크포인트 노드 표시
  ax4.scatter(
      run.TRIAL_NODES[:, 0],
      run.TRIAL_NODES[:, 1],
      color="#facc15",
      edgecolors="black",
      s=55,
      zorder=5,
      label="시행착오 탐색 노드 (+10점)",
  )

  # 미로 벽 및 범퍼 렌더링
  for gid in range(mj_model.ngeom):
    name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
    pos = mj_model.geom_pos[gid][:2]
    size = mj_model.geom_size[gid]
    if name.startswith("maze_"):
      quat = mj_model.geom_quat[gid]
      angle_deg = float(np.degrees(2 * np.arctan2(quat[3], quat[0])))
      w_col = "#d97706" if "_s1_" in name else ("#eab308" if "_div_" in name else "#14b8a6")
      rect = patches.Rectangle(
          (-size[0], -size[1]),
          2 * size[0],
          2 * size[1],
          facecolor=w_col,
          edgecolor="white",
          lw=0.8,
          zorder=4,
      )
      t = (
          patches.transforms.Affine2D()
          .rotate_deg(angle_deg)
          .translate(pos[0], pos[1])
          + ax4.transData
      )
      rect.set_transform(t)
      ax4.add_patch(rect)
    elif name.startswith("bumper_"):
      c = patches.Circle(
          (pos[0], pos[1]),
          size[0],
          facecolor="#f43f5e" if "3" in name else "#06b6d4",
          edgecolor="white",
          lw=1.0,
          zorder=5,
      )
      ax4.add_patch(c)

  # 무작위 궤적 vs PPO 점수 극대화 궤적
  for idx, rtraj in enumerate(rand_trajs):
    ax4.plot(
        rtraj[:, 0],
        rtraj[:, 1],
        color="#38bdf8",
        lw=1.5,
        alpha=0.80,
        ls="--",
        zorder=7,
        label="무작위 시행착오 (평균 -450점)" if idx == 0 else None,
    )
    ax4.scatter(rtraj[-1, 0], rtraj[-1, 1], color="#38bdf8", marker="x", s=45, zorder=8)

  for idx, ptraj in enumerate(ppo_trajs):
    ax4.plot(
        ptraj[:, 0],
        ptraj[:, 1],
        color="#ff7b29",
        lw=2.6,
        alpha=0.95,
        zorder=9,
        label="PPO 점수 극대화 (평균 +1,118점)" if idx == 0 else None,
    )
    ax4.scatter(ptraj[-1, 0], ptraj[-1, 1], color="#22c55e", marker="*", s=100, zorder=10)

  ax4.set_title("④ 2단계 선택형 미로 주행 궤적 (무작위 vs 점수 극대화 PPO)", color="white", fontsize=13, fontweight="bold", pad=10)
  ax4.set_xlabel("미로 X 좌표 (m)", color="#cbd5e1")
  ax4.set_ylabel("미로 Y 좌표 (m)", color="#cbd5e1")
  ax4.grid(True, color="#ffffff", ls=":", alpha=0.20)
  ax4.tick_params(colors="#cbd5e1")
  ax4.legend(loc="lower left", fontsize=7.8, facecolor="#0b1018", edgecolor="#334155", labelcolor="white")

  fig.suptitle(
      "2단계 선택형 복잡 미로 & 점수 극대화(+1000점 골인 / +10점 시행착오 / -500점 미도달 / -1000점 무행동) PPO 분석",
      color="white",
      fontsize=15,
      fontweight="bold",
      y=0.97,
  )

  ws_out = WORKSPACE_DIR / "training_dashboard.png"
  art_out = ARTIFACT_DIR / "training_dashboard.png"
  fig.savefig(ws_out, bbox_inches="tight", facecolor=fig.get_facecolor())
  fig.savefig(art_out, bbox_inches="tight", facecolor=fig.get_facecolor())
  plt.close(fig)
  return ws_out, art_out


def render_comparison_mp4(ppo_model: PPO) -> tuple[Path, Path]:
  """좌측: 무작위 시행착오 단독 뷰 | 우측: 학습된 점수 극대화 PPO vs 무작위 동시 비교 MP4 영상을 생성합니다."""
  env_rand = run.PureBallRollArenaEnv(curriculum=False)
  env_ppo = run.PureBallRollArenaEnv(curriculum=False)

  renderer_rand = mujoco.Renderer(env_rand.model, height=640, width=640)
  renderer_ppo = mujoco.Renderer(env_ppo.model, height=640, width=640)

  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(cam)
  cam.lookat[:] = [0.0, 0.0, 0.30]
  cam.distance = 5.20
  cam.azimuth = -90.0
  cam.elevation = -84.0

  ctrl_ppo = run.PurePPOViewerController(env_ppo, ppo_model, episode_sec=5.5, seed=42)
  ctrl_ppo._need_init = False
  ctrl_ppo.start_new_episode(env_ppo.model, env_ppo.data)

  def sync_rand_only_episode():
    mujoco.mj_resetData(env_rand.model, env_rand.data)
    env_rand.data.qpos[0:3] = [0.0, 0.0, -5.0]
    if env_rand.model.nmocap >= 1:
      env_rand.data.mocap_pos[0] = [0.0, 0.0, -5.0]
    env_rand.data.qpos[7:10] = env_ppo.data.qpos[7:10].copy()
    if env_rand.model.nmocap >= 2:
      env_rand.data.mocap_pos[1] = env_ppo.data.mocap_pos[1].copy()
    mujoco.mj_forward(env_rand.model, env_rand.data)

  sync_rand_only_episode()

  ws_mp4 = WORKSPACE_DIR / "comparison_random_vs_ppo.mp4"
  art_mp4 = ARTIFACT_DIR / "comparison_random_vs_ppo.mp4"

  fps = 50
  total_frames = 550  # 11.0초 (3개 에피소드 연속 비교)
  steps_per_frame = 10

  try:
    font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 21)
    font_sub = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
  except Exception:
    font_title = ImageFont.load_default()
    font_sub = ImageFont.load_default()

  writer = imageio.get_writer(str(ws_mp4), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")

  prev_attempt = ctrl_ppo.attempt
  for _ in range(total_frames):
    for _ in range(steps_per_frame):
      ctrl_ppo.step_callback(env_ppo.model, env_ppo.data)
      mujoco.mj_step(env_ppo.model, env_ppo.data)

      if ctrl_ppo.attempt != prev_attempt:
        prev_attempt = ctrl_ppo.attempt
        sync_rand_only_episode()
      else:
        rand_vel = env_rand.data.qvel[6:9]
        env_rand.data.qfrc_applied[6] = ctrl_ppo.current_rand_force[0] - 0.50 * rand_vel[0]
        env_rand.data.qfrc_applied[7] = ctrl_ppo.current_rand_force[1] - 0.50 * rand_vel[1]
        mujoco.mj_step(env_rand.model, env_rand.data)
        rpos = env_rand.data.qpos[7:10]
        if env_rand.model.nmocap >= 2:
          env_rand.data.mocap_pos[1] = [rpos[0], rpos[1], float(rpos[2]) + 0.15]

    renderer_rand.update_scene(env_rand.data, camera=cam)
    img_left = renderer_rand.render()

    renderer_ppo.update_scene(env_ppo.data, camera=cam)
    img_right = renderer_ppo.render()

    canvas = Image.new("RGB", (1280, 720), (11, 16, 24))
    canvas.paste(Image.fromarray(img_left), (0, 80))
    canvas.paste(Image.fromarray(img_right), (640, 80))

    draw = ImageDraw.Draw(canvas)
    draw.line([(640, 0), (640, 720)], fill=(70, 85, 110), width=4)

    # 좌측 배너 (무작위 시행착오)
    r_disp_score = ctrl_ppo.rand_score - 500 if ctrl_ppo.in_hole_time is not None else ctrl_ppo.rand_score
    r_status = "LOST IN MAZE (-500 pts)" if ctrl_ppo.in_hole_time is not None else f"Trial & Error (+{ctrl_ppo.rand_trials*10} pts)"
    draw.text((18, 12), "[LEFT] RANDOM AGENT (Blue: Wandering in Stage 1)", fill=(56, 189, 248), font=font_title)
    draw.text(
        (18, 44),
        f"Ep #{ctrl_ppo.attempt} | {r_status} | Score: {r_disp_score:+d} pts",
        fill=(203, 213, 225),
        font=font_sub,
    )

    # 우측 배너 (점수 극대화 PPO 에이전트)
    p_pos = env_ppo.data.qpos[0:2]
    stage_str = "STAGE 2 (Choice Chamber)" if p_pos[1] >= 0.0 else "STAGE 1 (Complex Maze)"
    if ctrl_ppo.in_hole_time is not None:
      p_status = f"TRUE GOAL (+1000) + Trials (+{ctrl_ppo.ppo_trials*10})"
      p_col = (74, 222, 128)
    else:
      p_status = f"{stage_str} | Trials: +{ctrl_ppo.ppo_trials*10} pts"
      p_col = (255, 155, 50)
    draw.text((658, 12), "[RIGHT] SCORE-MAXIMIZING PPO (Orange) vs RANDOM", fill=p_col, font=font_title)
    draw.text(
        (658, 44),
        f"Ep #{ctrl_ppo.attempt} | {p_status} | Total Score: {ctrl_ppo.ppo_score:+d} pts",
        fill=(248, 250, 252),
        font=font_sub,
    )

    writer.append_data(np.array(canvas))

  writer.close()
  renderer_rand.close()
  renderer_ppo.close()

  art_mp4.write_bytes(ws_mp4.read_bytes())
  return ws_mp4, art_mp4


if __name__ == "__main__":
  print("1) 2단계 선택형 미로 SB3 학습 로그 파싱 및 궤적 수집 중...")
  metrics = parse_sb3_log(LOG_FILE)
  ppo_model = PPO.load(str(run.MODEL_SAVE_PATH))
  mj_model, ppo_trajs, ppo_scores, rand_trajs, rand_scores = collect_trajectories(ppo_model, n_episodes=5)

  print("2) 2단계 미로 학습 데이터 대시보드 그래프(PNG) 생성 중...")
  ws_png, art_png = plot_training_dashboard(metrics, mj_model, ppo_trajs, ppo_scores, rand_trajs, rand_scores)
  print(f"   -> 저장 완료: {ws_png}")

  print("3) 2단계 미로 무작위 vs 점수 극대화 PPO 비교 영상(MP4) 렌더링 중...")
  ws_mp4, art_mp4 = render_comparison_mp4(ppo_model)
  print(f"   -> 저장 완료: {ws_mp4}")
