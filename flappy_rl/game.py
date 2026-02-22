"""Headless Flappy Bird physics engine — exact port of index.html."""

import random
import numpy as np
from typing import Tuple, Optional


class FlappyBirdEnv:
    W = 400
    H = 600
    GROUND_HEIGHT = 60
    BIRD_X = 60        # W * 0.15
    BIRD_RADIUS = 10
    PIPE_WIDTH = 56
    GRAVITY = 0.45
    FLAP_STRENGTH = -7.5

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.reset()

    # ── difficulty curve (exact JS port) ──────────────────────────
    def difficulty(self) -> float:
        t = min(1.0, self.score / 30.0)
        if t < 0.5:
            return 4 * t * t * t
        return 1 - (-2 * t + 2) ** 3 / 2

    def pipe_gap(self) -> float:
        return 210 - 85 * self.difficulty()

    def pipe_speed(self) -> float:
        return 2.0 + 2.0 * self.difficulty()

    def pipe_horizontal_gap(self) -> float:
        return 520 - 320 * self.difficulty()

    # ── environment control ───────────────────────────────────────
    def reset(self) -> np.ndarray:
        self.bird_y = self.H / 2 - 30          # 270
        self.bird_vy = 0.0
        self.score = 0
        self.pipes = []
        self._spawn_pipe()
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        # 1. flap
        if action == 1:
            self.bird_vy = self.FLAP_STRENGTH

        # 2. gravity + position
        self.bird_vy += self.GRAVITY
        self.bird_y += self.bird_vy

        # 3. maybe spawn pipe
        if not self.pipes or self.pipes[-1]["x"] <= self.W - self.pipe_horizontal_gap():
            self._spawn_pipe()

        # 4. move pipes & score
        speed = self.pipe_speed()
        scored = False
        for p in self.pipes:
            p["x"] -= speed
            if not p["scored"] and p["x"] + self.PIPE_WIDTH < self.BIRD_X:
                p["scored"] = True
                self.score += 1
                scored = True

        # 5. remove off-screen pipes
        self.pipes = [p for p in self.pipes if p["x"] > -self.PIPE_WIDTH - 20]

        # 6. collision
        done = self._check_collision()

        # 7. reward
        reward = self._compute_reward(scored, done)

        return self._get_state(), reward, done

    # ── internals ─────────────────────────────────────────────────
    def _spawn_pipe(self):
        gap = self.pipe_gap()
        min_y = 70.0
        max_y = self.H - self.GROUND_HEIGHT - gap - 70.0
        top_h = min_y + self.rng.random() * (max_y - min_y)
        bottom_y = top_h + gap
        self.pipes.append({
            "x": float(self.W + 10),
            "topH": top_h,
            "bottomY": bottom_y,
            "scored": False,
        })

    def _check_collision(self) -> bool:
        r = self.BIRD_RADIUS
        bx = self.BIRD_X
        by = self.bird_y

        # ceiling / ground
        if by - r < 0 or by + r > self.H - self.GROUND_HEIGHT:
            return True

        # pipes
        for p in self.pipes:
            if bx + r > p["x"] and bx - r < p["x"] + self.PIPE_WIDTH:
                if by - r < p["topH"] or by + r > p["bottomY"]:
                    return True
        return False

    def _get_state(self) -> np.ndarray:
        # find next pipe (first not fully past bird)
        next_pipe = None
        for p in self.pipes:
            if p["x"] + self.PIPE_WIDTH >= self.BIRD_X:
                next_pipe = p
                break

        if next_pipe is None:
            dist = 1.0
            gap_center_y = 0.5
            gap_size = self.pipe_gap() / self.H
        else:
            dist = (next_pipe["x"] - self.BIRD_X) / self.W
            gap_center = (next_pipe["topH"] + next_pipe["bottomY"]) / 2.0
            gap_center_y = gap_center / self.H
            gap_size = (next_pipe["bottomY"] - next_pipe["topH"]) / self.H

        return np.array([
            self.bird_y / self.H,
            self.bird_vy / 15.0,
            dist,
            gap_center_y,
            gap_size,
            self.pipe_speed() / 4.0,
        ], dtype=np.float32)

    def _compute_reward(self, scored: bool, done: bool) -> float:
        if done:
            return -5.0
        if scored:
            return 10.0

        # proximity reward: encourage being near the gap center
        next_pipe = None
        for p in self.pipes:
            if p["x"] + self.PIPE_WIDTH >= self.BIRD_X:
                next_pipe = p
                break
        if next_pipe is not None:
            gap_center = (next_pipe["topH"] + next_pipe["bottomY"]) / 2.0
            dist = abs(self.bird_y - gap_center) / (self.H / 2)
            return 0.1 + 0.4 * max(0, 1 - dist)  # 0.1–0.5 based on alignment
        return 0.1
