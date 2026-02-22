"""Flappy Bird RL — Dashboard-only training UI + standalone play mode."""

import time
import math
import numpy as np
import pygame

from .game import FlappyBirdEnv

# how many game steps to run per rendered frame in visual training
STEPS_PER_FRAME = 16

# ── theme ─────────────────────────────────────────────────────────
BG = (12, 13, 20)
CARD = (22, 24, 38)
CARD_HOVER = (30, 33, 50)
BORDER = (42, 46, 68)
BORDER_GLOW = (60, 70, 110)

WHITE = (240, 242, 255)
DIM = (110, 118, 150)
FAINT = (60, 65, 90)

CYAN = (80, 210, 255)
ORANGE = (255, 140, 60)
PURPLE = (170, 100, 255)
GREEN = (80, 230, 140)
GOLD = (255, 210, 60)
RED = (255, 80, 90)
PINK = (255, 100, 180)

GRID = (30, 34, 52)
AXIS = (55, 60, 85)

WIN_W = 900
WIN_H = 640

# ── game rendering colours (for play mode) ───────────────────────
GAME_W = 400
GAME_H = 600
SKY_TOP = (25, 25, 80)
SKY_BOT = (70, 130, 180)


def _lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _fmt_time(s):
    if s < 60:
        return f"{int(s)}s"
    if s < 3600:
        return f"{int(s//60)}m {int(s%60)}s"
    return f"{int(s//3600)}h {int((s%3600)//60)}m"


# ══════════════════════════════════════════════════════════════════
#  Chart renderer
# ══════════════════════════════════════════════════════════════════
def _draw_chart(surf, rect, data, color, title, font_title, font_tick,
                avg_data=None, avg_color=None, log_scale=False,
                fill=True, dot_last=True):
    x, y, w, h = rect
    pad_l, pad_r, pad_t, pad_b = 54, 12, 30, 24

    # card
    pygame.draw.rect(surf, CARD, rect, border_radius=8)
    # subtle top highlight
    pygame.draw.line(surf, BORDER_GLOW, (x + 8, y), (x + w - 8, y), 1)

    # title
    ts = font_title.render(title, True, DIM)
    surf.blit(ts, (x + pad_l, y + 8))

    cx, cy = x + pad_l, y + pad_t
    cw, ch = w - pad_l - pad_r, h - pad_t - pad_b

    if not data or len(data) < 2:
        msg = font_tick.render("Collecting data...", True, FAINT)
        surf.blit(msg, (cx + cw // 2 - msg.get_width() // 2, cy + ch // 2 - 5))
        return

    arr = np.array(data, dtype=np.float64)

    if log_scale:
        pos = arr[arr > 0]
        if len(pos) < 2:
            return
        vmin = np.log10(max(pos.min() * 0.5, 1e-8))
        vmax = np.log10(pos.max() * 2)
    else:
        vmin = min(0, float(arr.min()))
        vmax = max(float(arr.max()) * 1.15, 0.1)

    # grid
    for i in range(5):
        gy = cy + int(ch * i / 4)
        pygame.draw.line(surf, GRID, (cx, gy), (cx + cw, gy), 1)
        raw = vmax - (vmax - vmin) * i / 4
        if log_scale:
            lbl = f"{10**raw:.2g}"
        else:
            lbl = f"{raw:.0f}" if abs(raw) >= 10 else f"{raw:.1f}"
        ts = font_tick.render(lbl, True, FAINT)
        surf.blit(ts, (cx - ts.get_width() - 4, gy - 5))

    # axes
    pygame.draw.line(surf, AXIS, (cx, cy), (cx, cy + ch), 1)
    pygame.draw.line(surf, AXIS, (cx, cy + ch), (cx + cw, cy + ch), 1)

    # x label
    xl = font_tick.render(f"ep {len(data)}", True, FAINT)
    surf.blit(xl, (cx + cw - xl.get_width(), cy + ch + 6))

    def _pts(values, use_log=False):
        n = len(values)
        out = []
        for i, v in enumerate(values):
            px = cx + int(i / max(n - 1, 1) * cw)
            if use_log:
                nv = (math.log10(max(v, 1e-8)) - vmin) / max(vmax - vmin, 1e-8)
            else:
                nv = (v - vmin) / max(vmax - vmin, 1e-8)
            py = cy + ch - int(max(0, min(1, nv)) * ch)
            out.append((px, py))
        return out

    # filled area under curve
    points = _pts(arr, use_log=log_scale)
    if fill and len(points) > 1:
        fill_col = (*color[:3], 25) if len(color) >= 3 else (*color, 25)
        fill_surf = pygame.Surface((cw, ch), pygame.SRCALPHA)
        local_pts = [(px - cx, py - cy) for px, py in points]
        poly = local_pts + [(local_pts[-1][0], ch), (local_pts[0][0], ch)]
        try:
            pygame.draw.polygon(fill_surf, fill_col, poly)
            surf.blit(fill_surf, (cx, cy))
        except (ValueError, TypeError):
            pass

    # main line
    if len(points) > 1:
        pygame.draw.lines(surf, color, False, points, 2)

    # rolling average
    if avg_data and len(avg_data) >= 2:
        avg_pts = _pts(avg_data, use_log=log_scale)
        if len(avg_pts) > 1:
            pygame.draw.lines(surf, avg_color or ORANGE, False, avg_pts, 2)

    # dot on last point
    if dot_last and points:
        lp = points[-1]
        pygame.draw.circle(surf, WHITE, lp, 4)
        pygame.draw.circle(surf, color, lp, 3)

    # current value
    if data:
        cur = data[-1]
        if log_scale:
            vs = f"{cur:.4f}"
        else:
            vs = f"{cur:.1f}" if abs(cur) >= 1 else f"{cur:.3f}"
        cv = font_title.render(vs, True, color)
        surf.blit(cv, (x + w - pad_r - cv.get_width(), y + 8))


# ══════════════════════════════════════════════════════════════════
#  Stat card (top bar)
# ══════════════════════════════════════════════════════════════════
def _draw_stat(surf, x, y, w, h, label, value, color, font_lbl, font_val):
    pygame.draw.rect(surf, CARD, (x, y, w, h), border_radius=8)
    pygame.draw.line(surf, color, (x + 4, y + h - 3), (x + w - 4, y + h - 3), 2)
    ls = font_lbl.render(label, True, DIM)
    vs = font_val.render(str(value), True, color)
    surf.blit(ls, (x + w // 2 - ls.get_width() // 2, y + 6))
    surf.blit(vs, (x + w // 2 - vs.get_width() // 2, y + 22))


# ══════════════════════════════════════════════════════════════════
#  Training UI (dashboard only, no game panel)
# ══════════════════════════════════════════════════════════════════
class TrainingUI:
    FPS = 60

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Flappy Bird RL - Training Dashboard")
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("segoeui", 13, bold=True)
        self.font_val = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_lbl = pygame.font.SysFont("segoeui", 10)
        self.font_tick = pygame.font.SysFont("consolas", 9)
        self.font_hdr = pygame.font.SysFont("segoeui", 22, bold=True)
        self.font_sub = pygame.font.SysFont("segoeui", 11)
        self._frame = 0
        self._start_time = time.time()
        self._cached = None

    def _draw(self, scores, losses, epsilons, agent):
        surf = pygame.Surface((WIN_W, WIN_H))
        surf.fill(BG)

        margin = 14
        # ── header ────────────────────────────────────────────────
        hdr = self.font_hdr.render("FLAPPY BIRD RL", True, WHITE)
        surf.blit(hdr, (margin, 10))
        sub = self.font_sub.render("Double DQN Training Dashboard", True, DIM)
        surf.blit(sub, (margin, 36))

        elapsed = time.time() - self._start_time
        ts = self.font_sub.render(_fmt_time(elapsed), True, FAINT)
        surf.blit(ts, (WIN_W - margin - ts.get_width(), 14))
        spd = self.font_sub.render(f"{STEPS_PER_FRAME}x speed  |  ESC to save & quit", True, FAINT)
        surf.blit(spd, (WIN_W - margin - spd.get_width(), 32))

        # ── stat cards ────────────────────────────────────────────
        card_y = 56
        card_h = 48
        n_cards = 6
        card_gap = 8
        card_w = (WIN_W - margin * 2 - card_gap * (n_cards - 1)) // n_cards

        ep_count = len(scores)
        best = max(scores) if scores else 0
        avg20 = np.mean(scores[-20:]) if len(scores) >= 20 else (np.mean(scores) if scores else 0)
        avg100 = np.mean(scores[-100:]) if len(scores) >= 100 else (np.mean(scores) if scores else 0)
        eps_val = f"{agent.epsilon:.3f}" if agent else "---"
        last_loss = f"{losses[-1]:.4f}" if losses and losses[-1] > 0 else "---"

        cards = [
            ("EPISODES", str(ep_count), CYAN),
            ("BEST SCORE", str(best), GOLD),
            ("AVG (20)", f"{avg20:.1f}", ORANGE),
            ("AVG (100)", f"{avg100:.1f}", PINK),
            ("EPSILON", eps_val, GREEN),
            ("LOSS", last_loss, PURPLE),
        ]
        for i, (lbl, val, col) in enumerate(cards):
            cx = margin + i * (card_w + card_gap)
            _draw_stat(surf, cx, card_y, card_w, card_h, lbl, val, col, self.font_lbl, self.font_val)

        # ── charts ────────────────────────────────────────────────
        chart_top = card_y + card_h + 14
        chart_gap = 10
        # two rows: score (tall) on top, loss + epsilon side by side on bottom
        big_h = 220
        small_h = 195
        half_w = (WIN_W - margin * 2 - chart_gap) // 2

        # score chart (full width)
        avg_data = None
        if scores and len(scores) >= 20:
            k = np.ones(20) / 20
            c = list(np.convolve(scores, k, mode="valid"))
            avg_data = [scores[0]] * 19 + c  # pad start

        _draw_chart(surf, (margin, chart_top, WIN_W - margin * 2, big_h),
                    scores, CYAN, "SCORE PER EPISODE", self.font_title, self.font_tick,
                    avg_data=avg_data, avg_color=ORANGE)

        # loss chart (left half)
        row2_y = chart_top + big_h + chart_gap
        _draw_chart(surf, (margin, row2_y, half_w, small_h),
                    losses, PURPLE, "TRAINING LOSS", self.font_title, self.font_tick,
                    log_scale=True)

        # epsilon chart (right half)
        _draw_chart(surf, (margin + half_w + chart_gap, row2_y, half_w, small_h),
                    epsilons, GREEN, "EXPLORATION (EPSILON)", self.font_title, self.font_tick)

        # ── footer info ───────────────────────────────────────────
        fy = row2_y + small_h + 8
        info = "lr=5e-4  |  batch=128  |  gamma=0.99  |  tau=0.005  |  reward: +10 score, -5 death, 0.1-0.5 proximity"
        ft = self.font_sub.render(info, True, FAINT)
        surf.blit(ft, (margin, fy))

        return surf

    def update(self, env, agent, scores, losses, epsilons) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True

        self._frame += 1
        if self._frame % 10 == 1 or self._cached is None:
            self._cached = self._draw(scores, losses, epsilons, agent)
        self.screen.blit(self._cached, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.FPS)
        return False

    def close(self):
        pygame.quit()


# ══════════════════════════════════════════════════════════════════
#  Standalone play mode (game rendering)
# ══════════════════════════════════════════════════════════════════
def _draw_game(env, big_font):
    surf = pygame.Surface((GAME_W, GAME_H))
    for y in range(0, GAME_H, 4):
        t = y / GAME_H
        pygame.draw.rect(surf, _lerp(SKY_TOP, SKY_BOT, t), (0, y, GAME_W, 4))

    gh = env.GROUND_HEIGHT
    gy = GAME_H - gh
    pygame.draw.rect(surf, (80, 50, 30), (0, gy, GAME_W, gh))
    pygame.draw.line(surf, (50, 35, 20), (0, gy), (GAME_W, gy), 2)

    for p in env.pipes:
        px, pw = int(p["x"]), env.PIPE_WIDTH
        top_h, bot_y = int(p["topH"]), int(p["bottomY"])
        pygame.draw.rect(surf, (46, 160, 46), (px, 0, pw, top_h))
        pygame.draw.rect(surf, (30, 130, 30), (px - 3, top_h - 18, pw + 6, 18))
        pygame.draw.rect(surf, (46, 160, 46), (px, bot_y, pw, gy - bot_y))
        pygame.draw.rect(surf, (30, 130, 30), (px - 3, bot_y, pw + 6, 18))

    bx, by, br = int(env.BIRD_X), int(env.bird_y), env.BIRD_RADIUS + 4
    pygame.draw.circle(surf, (255, 210, 50), (bx, by), br)
    pygame.draw.circle(surf, (255, 255, 255), (bx + 5, by - 3), 4)
    pygame.draw.circle(surf, (20, 20, 20), (bx + 6, by - 3), 2)
    pygame.draw.polygon(surf, (255, 130, 50), [(bx + br, by), (bx + br + 6, by + 2), (bx + br, by + 4)])

    txt = big_font.render(str(env.score), True, (255, 255, 255))
    surf.blit(txt, (GAME_W // 2 - txt.get_width() // 2, 20))
    return surf


def play_game(args):
    pygame.init()
    screen = pygame.display.set_mode((GAME_W, GAME_H))
    pygame.display.set_caption("Flappy Bird RL - Play")
    clock = pygame.time.Clock()
    big_font = pygame.font.SysFont("consolas", 36, bold=True)
    small_font = pygame.font.SysFont("consolas", 18)

    env = FlappyBirdEnv()
    human = getattr(args, "human", False)
    fps = getattr(args, "fps", 60)

    agent = None
    if not human:
        from .agent import DQNAgent
        agent = DQNAgent(device="cpu")
        model_path = getattr(args, "model", "checkpoints/agent_final.pt")
        agent.load(model_path)
        print(f"Loaded model from {model_path}")

    state = env.reset()
    running = True
    game_over = False

    while running:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    if game_over:
                        state = env.reset()
                        game_over = False
                        continue
                    if human:
                        action = 1

        if not running:
            break

        if game_over:
            overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))
            go = big_font.render("GAME OVER", True, (255, 80, 80))
            screen.blit(go, (GAME_W // 2 - go.get_width() // 2, GAME_H // 2 - 40))
            sc = small_font.render(f"Score: {env.score}  |  SPACE to restart", True, (255, 255, 255))
            screen.blit(sc, (GAME_W // 2 - sc.get_width() // 2, GAME_H // 2 + 10))
            pygame.display.flip()
            clock.tick(fps)
            continue

        if not human and agent:
            action = agent.select_action(state, training=False)

        state, _, done = env.step(action)
        if done:
            game_over = True

        screen.blit(_draw_game(env, big_font), (0, 0))
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
