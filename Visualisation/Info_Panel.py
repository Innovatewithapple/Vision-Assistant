import time
import cv2

WHITE, GRAY = (255, 255, 255), (190, 190, 190)
GREEN, BLUE = (90, 220, 90), (255, 160, 60)
YELLOW, ORANGE, RED = (0, 220, 255), (0, 165, 255), (60, 60, 255)

def rounded_rect(img, x, y, w, h, r, color, thickness):
    if thickness < 0:   # filled
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in [(x + r, y + r), (x + w - r, y + r),
                       (x + r, y + h - r), (x + w - r, y + h - r)]:
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:               # outline
        cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + r, y + r), (r, r), 0, 180, 270, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + w - r, y + r), (r, r), 0, 270, 360, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + r, y + h - r), (r, r), 0, 90, 180, color, thickness, cv2.LINE_AA)

SHOW_PANEL = True
latest_stats = []          # every stat, including hidden ones (for logs / LLM)
_last_t, _fps = None, 0.0

def toggle_panel():
    global SHOW_PANEL
    SHOW_PANEL = not SHOW_PANEL

def tick_fps():
    global _last_t, _fps
    now = time.perf_counter()
    if _last_t is not None:
        inst = 1.0 / max(now - _last_t, 1e-6)
        _fps = inst if _fps == 0 else 0.9 * _fps + 0.1 * inst   # smoothed
    _last_t = now
    return _fps

def stat(label, value, kind="text", color=WHITE, section=1, on_panel=True, only_if=True):
    return {"label": label, "value": str(value), "kind": kind, "color": color,
            "section": section, "on_panel": on_panel, "only_if": only_if}

def draw_stats_panel(frame, stats, width=210, margin=10):
    latest_stats[:] = stats
    if not SHOW_PANEL:
        return
    rows = [s for s in stats if s["on_panel"] and s["only_if"]]
    if not rows:
        return

    FONT, pad = cv2.FONT_HERSHEY_SIMPLEX, 12
    h_main, h_small, gap = 28, 21, 8

    # total height
    h, prev = 2 * pad, None
    for s in rows:
        if prev is not None and s["section"] != prev:
            h += gap
        h += h_main if s["section"] == 1 else h_small
        prev = s["section"]

    x, y = frame.shape[1] - width - margin, margin

    # translucent background + border
    overlay = frame.copy()
    rounded_rect(overlay, x, y, width, h, 12, (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    rounded_rect(frame, x, y, width, h, 12, (110, 110, 110), 1)

    cy, prev = y + pad, None
    for s in rows:
        if prev is not None and s["section"] != prev:
            cy += gap
        main = s["section"] == 1
        rh = h_main if main else h_small
        scale = 0.5 if main else 0.4
        base = cy + rh - (8 if main else 6)

        cv2.putText(frame, s["label"] + ":", (x + pad, base), FONT, scale,
                    WHITE if main else GRAY, 1, cv2.LINE_AA)

        if s["kind"] == "badge":
            (tw, th), _ = cv2.getTextSize(s["value"], FONT, 0.42, 1)
            bw, bh = tw + 16, th + 10
            bx, by = x + width - pad - bw, base - th - 5
            rounded_rect(frame, bx, by, bw, bh, bh // 2,
                         tuple(int(c * 0.3) for c in s["color"]), -1)
            rounded_rect(frame, bx, by, bw, bh, bh // 2, s["color"], 1)
            cv2.putText(frame, s["value"], (bx + 8, base), FONT, 0.42,
                        s["color"], 1, cv2.LINE_AA)
        else:
            (tw, _), _ = cv2.getTextSize(s["value"], FONT, scale, 1)
            cv2.putText(frame, s["value"], (x + width - pad - tw, base), FONT,
                        scale, WHITE, 1, cv2.LINE_AA)

        cy += rh
        prev = s["section"]