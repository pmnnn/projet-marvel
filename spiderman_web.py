#!/usr/bin/env python3
"""
spiderman_web.py — Real-time Spider-Man web-shooter gesture recognition.

Controls:
  Q / ESC  — quit
  S        — toggle hand skeleton overlay

Gesture: index + middle extended, ring + pinky curled, thumb out.
"""

import sys
import time
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

# ─── MediaPipe setup ─────────────────────────────────────────────────────────

mp_hands        = mp.solutions.hands
mp_drawing      = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─── Landmark index constants ────────────────────────────────────────────────

WRIST                                             = 0
THUMB_CMC, THUMB_MCP, THUMB_IP,   THUMB_TIP      = 1,  2,  3,  4
INDEX_MCP, INDEX_PIP, INDEX_DIP,  INDEX_TIP      = 5,  6,  7,  8
MIDDLE_MCP,MIDDLE_PIP,MIDDLE_DIP, MIDDLE_TIP     = 9, 10, 11, 12
RING_MCP,  RING_PIP,  RING_DIP,   RING_TIP       = 13,14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP,  PINKY_TIP      = 17,18, 19, 20

TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
PIPS = [THUMB_IP,  INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

# ─── Timing / animation constants ────────────────────────────────────────────

GESTURE_HOLD   = 0.30   # seconds gesture must be held before web fires
WEB_FRAMES     = 28     # total animation lifetime (frames)
WEB_GROW_END   = 20     # frame index where growth stops / fade begins
RESPAWN_SECS   = 0.55   # minimum gap between consecutive web spawns

# ─── Math helpers ─────────────────────────────────────────────────────────────

def to_px(lm, w: int, h: int) -> np.ndarray:
    """Normalised landmark → float pixel coords."""
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)

def bz(p0, p1, p2, t: float) -> np.ndarray:
    """Quadratic Bézier point at t ∈ [0, 1]."""
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

def bz_tang(p0, p1, p2, t: float) -> np.ndarray:
    """Normalised tangent of quadratic Bézier at t."""
    v = 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])

# ─── Gesture detection ────────────────────────────────────────────────────────

def finger_up(lms, fidx: int, w: int, h: int) -> bool:
    """True if finger tip is farther from wrist than its PIP joint."""
    wrist = to_px(lms[WRIST], w, h)
    tip   = to_px(lms[TIPS[fidx]], w, h)
    pip   = to_px(lms[PIPS[fidx]], w, h)
    return np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist) * 1.05

def thumb_out(lms, w: int, h: int) -> bool:
    """True if thumb tip is farther from index MCP than the thumb IP joint."""
    tip     = to_px(lms[THUMB_TIP], w, h)
    ip_pt   = to_px(lms[THUMB_IP],  w, h)
    idx_mcp = to_px(lms[INDEX_MCP], w, h)
    return np.linalg.norm(tip - idx_mcp) > np.linalg.norm(ip_pt - idx_mcp)

def is_spiderman(lms, w: int, h: int) -> bool:
    """Detect the Spider-Man web-shooter hand sign.
    Index + pinky up, middle + ring curled, thumb out.
    """
    return (
        finger_up(lms, 1, w, h) and          # index extended
        not finger_up(lms, 2, w, h) and      # middle curled
        not finger_up(lms, 3, w, h) and      # ring curled
        finger_up(lms, 4, w, h) and          # pinky extended
        thumb_out(lms, w, h)                 # thumb extended outward
    )

def hand_props(lms, w: int, h: int):
    """Return (origin_px, direction_unit_vec, scale) for web emission."""
    wrist   = to_px(lms[WRIST], w, h)
    mid_tip = to_px(lms[MIDDLE_TIP], w, h)
    mid_mcp = to_px(lms[MIDDLE_MCP], w, h)

    # Palm centre = average of wrist + 4 knuckles
    palm = np.mean([
        to_px(lms[i], w, h)
        for i in (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
    ], axis=0)

    raw = mid_tip - palm
    nrm = np.linalg.norm(raw)
    direction = raw / nrm if nrm > 1e-9 else np.array([0.0, -1.0])

    # Scale web length to hand size
    scale = np.linalg.norm(mid_mcp - wrist) * 4.0
    return wrist.astype(int), direction, float(scale)

# ─── Web animation ────────────────────────────────────────────────────────────

class WebAnimation:
    """Animated spider-web shooting from a hand."""

    N_STRANDS = 5   # main thread lines

    def __init__(self, origin, direction, scale: float):
        self.origin    = np.array(origin,    dtype=np.float64)
        self.direction = np.array(direction, dtype=np.float64)
        self.scale     = scale
        self.frame     = 0
        self._strands  = self._build_strands()

    # ── strand geometry ──────────────────────────────────────────────────────

    def _build_strands(self):
        o, d, L = self.origin, self.direction, self.scale
        dx, dy  = d
        strands  = []

        for i in range(self.N_STRANDS):
            # Fan strands ±44° around the pointing direction
            angle = np.deg2rad((i - self.N_STRANDS // 2) * 22)
            ca, sa = np.cos(angle), np.sin(angle)
            sd = np.array([dx * ca - dy * sa, dx * sa + dy * ca])

            perp = np.array([-sd[1], sd[0]])
            # Alternate left/right curl for organic look
            curl = 0.12 * ((-1) ** i)
            ctrl = o + sd * L * 0.45 + perp * L * curl
            end  = o + sd * L

            branches = []
            for t in (0.25, 0.52, 0.78):
                pt   = bz(o, ctrl, end, t)
                tang = bz_tang(o, ctrl, end, t)
                perp_t = np.array([-tang[1], tang[0]])
                bl = L * 0.22
                for side in (-1.0, 1.0):
                    be = pt + perp_t * side * bl * 0.65 + tang * bl * 0.30
                    branches.append({'t': t, 'start': pt.copy(), 'end': be})

            strands.append({'s': o.copy(), 'c': ctrl, 'e': end, 'br': branches})
        return strands

    # ── per-frame draw ───────────────────────────────────────────────────────

    def draw(self, img: np.ndarray) -> bool:
        """Draw current frame onto img in-place. Returns True while alive."""
        if self.frame >= WEB_FRAMES:
            return False

        draw_p = min(1.0, self.frame / WEB_GROW_END)   # growth 0→1

        if self.frame < WEB_GROW_END:
            alpha = 1.0
        else:
            alpha = 1.0 - (self.frame - WEB_GROW_END) / (WEB_FRAMES - WEB_GROW_END)
        alpha = max(0.0, alpha)

        overlay = img.copy()
        WEB_COLOR  = (245, 248, 255)   # near-white with slight blue tint
        GLOW_COLOR = (180, 200, 255)   # soft blue glow

        for st in self._strands:
            s, c, e = st['s'], st['c'], st['e']

            # Draw a thick glow pass, then a sharp thin pass
            n = max(2, int(45 * draw_p))
            pts = [bz(s, c, e, k / n * draw_p).astype(int) for k in range(n + 1)]

            for k in range(len(pts) - 1):
                cv2.line(overlay, tuple(pts[k]), tuple(pts[k + 1]),
                         GLOW_COLOR, 3, cv2.LINE_AA)   # soft glow
            for k in range(len(pts) - 1):
                cv2.line(overlay, tuple(pts[k]), tuple(pts[k + 1]),
                         WEB_COLOR, 1, cv2.LINE_AA)    # crisp thread

            # Cross-branches
            for br in st['br']:
                if draw_p > br['t']:
                    bp = min(1.0, (draw_p - br['t']) / 0.20)
                    bs = br['start'].astype(int)
                    be = (br['start'] + (br['end'] - br['start']) * bp).astype(int)
                    cv2.line(overlay, tuple(bs), tuple(be), WEB_COLOR, 1, cv2.LINE_AA)

        # Origin burst (first 6 frames)
        if self.frame < 6:
            burst_r  = int(self.frame * 7 + 8)
            burst_a  = 1.0 - self.frame / 6.0
            for ang in range(0, 360, 45):
                rad = np.deg2rad(ang)
                be  = (self.origin + np.array([np.cos(rad), np.sin(rad)]) * burst_r).astype(int)
                cv2.line(overlay, tuple(self.origin.astype(int)), tuple(be),
                         (160, 200, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, alpha * 0.88, img, 1.0 - alpha * 0.88, 0, img)
        self.frame += 1
        return True

# ─── Per-hand gesture state ───────────────────────────────────────────────────

class HandState:
    def __init__(self):
        self._start:      Optional[float] = None
        self._last_spawn: Optional[float] = None

    def charge(self, now: float) -> float:
        """Charge progress 0→1 toward the GESTURE_HOLD threshold."""
        if self._start is None:
            return 0.0
        return min(1.0, (now - self._start) / GESTURE_HOLD)

    def update(self, detected: bool, now: float) -> bool:
        """Call every frame. Returns True when a web should be spawned."""
        if not detected:
            self._start = None
            return False
        if self._start is None:
            self._start = now
        held = now - self._start
        if held < GESTURE_HOLD:
            return False
        # Threshold crossed — spawn if first time or enough gap since last
        if self._last_spawn is None or (now - self._last_spawn) >= RESPAWN_SECS:
            self._last_spawn = now
            return True
        return False

# ─── HUD ─────────────────────────────────────────────────────────────────────

def draw_hud(img: np.ndarray, fps: float, gesture_on: bool,
             hand_count: int, skeleton_on: bool) -> None:
    H, W    = img.shape[:2]
    PAD     = 10
    LINE_H  = 22
    font    = cv2.FONT_HERSHEY_SIMPLEX
    fs      = 0.52

    lines = [
        (f"FPS: {fps:4.0f}",                             (170, 220, 170)),
        (f"Hands:  {hand_count}",                        (170, 220, 170)),
        (f"Gesture: {'YES !!!' if gesture_on else 'no'}",
                                            (0, 240, 80) if gesture_on else (150, 150, 150)),
        (f"[S] skeleton: {'on' if skeleton_on else 'off'}", (130, 130, 130)),
        ( "[Q] quit",                                    (130, 130, 130)),
    ]

    box_w = 192
    box_h = len(lines) * LINE_H + PAD * 2
    x2    = min(PAD + box_w, W - 1)
    y2    = min(PAD + box_h, H - 1)

    # Semi-transparent dark panel
    roi = img[PAD:y2, PAD:x2]
    if roi.size:
        cv2.addWeighted(np.zeros_like(roi), 0.60, roi, 0.40, 0, roi)

    for i, (text, color) in enumerate(lines):
        y = PAD * 2 + i * LINE_H + 12
        cv2.putText(img, text, (PAD + 8, y), font, fs, color, 1, cv2.LINE_AA)

    # Big centre label when firing
    if gesture_on:
        label = "WEB-SHOOT!"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
        cx = W // 2 - tw // 2
        cv2.putText(img, label, (cx, 58),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 200, 255), 2, cv2.LINE_AA)

# ─── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("ERROR: Cannot open webcam (device 0). Check your camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS,            30)

    model = mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 2,
        min_detection_confidence = 0.70,
        min_tracking_confidence  = 0.60,
    )

    show_skeleton = True
    animations: list[WebAnimation]   = []
    hand_states: dict[str, HandState] = {}

    fps_window: list[float] = []
    t_prev = time.perf_counter()

    print("Spider-Man Web Shooter started.")
    print("  Hold the Spider-Man sign (index+middle up, ring+pinky curled, thumb out)")
    print("  for 0.3 s to fire a web.  S = toggle skeleton  |  Q / ESC = quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)          # mirror so it feels natural
        H, W  = frame.shape[:2]
        now   = time.perf_counter()

        # ── MediaPipe inference ───────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = model.process(rgb)

        gesture_active = False
        hand_count     = 0

        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)

            for hld, hnd in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):
                label = hnd.classification[0].label   # "Left" or "Right"
                lms   = hld.landmark

                # Optional skeleton overlay
                if show_skeleton:
                    mp_drawing.draw_landmarks(
                        frame, hld,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                detected = is_spiderman(lms, W, H)
                if detected:
                    gesture_active = True

                if label not in hand_states:
                    hand_states[label] = HandState()
                hs = hand_states[label]

                # Charge-ring indicator on wrist while charging
                charge = hs.charge(now)
                if detected and 0.0 < charge < 1.0:
                    wrist_px = to_px(lms[WRIST], W, H).astype(int)
                    arc_deg  = int(360 * charge)
                    cv2.ellipse(frame, tuple(wrist_px), (24, 24),
                                -90, 0, arc_deg, (0, 210, 255), 2, cv2.LINE_AA)

                # Spawn web?
                if hs.update(detected, now):
                    origin, direction, scale = hand_props(lms, W, H)
                    animations.append(WebAnimation(origin, direction, scale))

        # ── Draw web animations (before HUD so HUD stays on top) ─────────
        animations = [a for a in animations if a.draw(frame)]

        # ── FPS counter ───────────────────────────────────────────────────
        dt = max(now - t_prev, 1e-9)
        fps_window.append(1.0 / dt)
        if len(fps_window) > 30:
            fps_window.pop(0)
        fps    = sum(fps_window) / len(fps_window)
        t_prev = now

        # ── HUD ───────────────────────────────────────────────────────────
        draw_hud(frame, fps, gesture_active, hand_count, show_skeleton)

        cv2.imshow("Spider-Man Web Shooter  |  S=skeleton  Q=quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # Q or ESC
            break
        if key in (ord('s'), ord('S')):
            show_skeleton = not show_skeleton

    model.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()
