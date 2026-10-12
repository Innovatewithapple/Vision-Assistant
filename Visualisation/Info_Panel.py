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

    frame_h, frame_w = frame.shape[:2]
    # Proportional HUD sizing; tune PANEL_WIDTH_RATIO and FONT_MULTIPLIER below.
    PANEL_WIDTH_RATIO = 0.32
    FONT_MULTIPLIER = 1.0
    scale = max(0.85, min(frame_w / 960.0, 2.0)) * FONT_MULTIPLIER
    margin = max(6, int(margin * max(0.85, min(frame_w / 960.0, 2.0))))
    pad = max(9, int(12 * max(0.85, min(frame_w / 960.0, 2.0))))
    width = max(
        int(width * max(0.85, min(frame_w / 960.0, 2.0))),
        int(frame_w * PANEL_WIDTH_RATIO),
    )
    width = min(width, max(1, frame_w - 2 * margin))
    x = max(margin, frame_w - width - margin)
    y = margin

    FONT = cv2.FONT_HERSHEY_DUPLEX
    border_thickness = 1
    text_thickness = max(1, int(round(1.0 * max(0.85, min(frame_w / 960.0, 2.0)))))

    # Metrics use a fixed order every frame. Never derive card positions from
    # the incoming stats order because conditional rows can appear/disappear.
    metric_order = [
        "Approaching",
        "People Detected",
        "Min Distance",
        "Avg Distance",
        "Next Contact",
    ]
    stats_by_label = {s["label"]: s for s in stats}
    metric_rows = []
    for label in metric_order:
        item = stats_by_label.get(label)
        if item is None:
            if label == "Next Contact":
                item = {
                    "label": label, "value": "No predicted contact", "kind": "text",
                    "color": (235, 239, 244), "section": 2,
                    "on_panel": True, "only_if": True,
                }
            else:
                continue
        else:
            item = dict(item)
            if label == "Next Contact" and not item["value"]:
                item["value"] = "No predicted contact"
        metric_rows.append(item)

    other_rows = [s for s in rows if s["label"] not in metric_order]
    status_rows = [s for s in other_rows if s["section"] == 1]
    detail_rows = [s for s in other_rows if s["section"] != 1]

    base_scale = max(0.85, min(frame_w / 960.0, 2.0))
    title_h = max(25, int(29 * base_scale))
    status_row_h = max(24, int(28 * base_scale))
    detail_row_h = max(22, int(25 * base_scale))
    section_gap = max(7, int(9 * base_scale))
    card_gap = max(6, int(7 * base_scale))
    card_pad = max(8, int(10 * base_scale))
    label_scale = 0.43 * base_scale
    value_scale = 0.62 * base_scale
    badge_scale = 0.46 * base_scale
    radius = max(7, int(10 * base_scale))

    # Calculate only the content that is actually drawn. Previously an extra
    # Next Contact height was added here even though it was already grid card #5,
    # creating the large empty area below the cards.
    status_height = title_h + sum(status_row_h for s in status_rows if s["label"] != "FPS")
    details_height = sum(detail_row_h for _ in detail_rows)
    metric_count = len(metric_rows)
    metric_grid_rows = (metric_count + 1) // 2
    card_h = max(48, int(55 * base_scale))
    metric_header_height = max(15, int(20 * base_scale)) + max(7, int(8 * base_scale))
    metrics_height = (
        section_gap + metric_header_height + metric_grid_rows * card_h
        + max(0, metric_grid_rows - 1) * card_gap
    ) if metric_rows else 0
    panel_h = int(2 * pad + status_height + details_height + metrics_height)

    # Keep the panel within the frame. The chosen compact layout is intended for
    # portrait video; for unusually short frames, shrink vertical spacing gently.
    if panel_h > frame_h - 2 * margin:
        vertical_factor = max(0.65, (frame_h - 2 * margin) / panel_h)
        title_h = int(title_h * vertical_factor)
        status_row_h = int(status_row_h * vertical_factor)
        detail_row_h = int(detail_row_h * vertical_factor)
        section_gap = int(section_gap * vertical_factor)
        card_h = int(card_h * vertical_factor)
        card_gap = int(card_gap * vertical_factor)
        pad = int(pad * vertical_factor)
        panel_h = min(frame_h - 2 * margin, int(panel_h * vertical_factor))
    panel_h = max(1, panel_h)

    overlay = frame.copy()
    rounded_rect(overlay, x, y, width, panel_h, min(radius, panel_h // 2, width // 2),
                 (20, 22, 26), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    rounded_rect(frame, x, y, width, panel_h, min(radius, panel_h // 2, width // 2),
                 (105, 112, 122), border_thickness)

    cy = y + pad

    # Header.
    cv2.putText(frame, "Observing", (x + pad, cy + int(15 * base_scale)),
                FONT, 0.46 * base_scale, (245, 247, 250), text_thickness, cv2.LINE_AA)
    fps_row = next((s for s in status_rows if s["label"] == "FPS"), None)
    if fps_row is not None:
        fps_text = f'{fps_row["value"]} FPS'
        (tw, th), _ = cv2.getTextSize(fps_text, FONT, label_scale, text_thickness)
        cv2.putText(frame, fps_text, (x + width - pad - tw, cy + int(15 * base_scale)),
                    FONT, label_scale, (220, 225, 232), text_thickness, cv2.LINE_AA)
        status_rows = [s for s in status_rows if s["label"] != "FPS"]
    cy += title_h

    # Status rows with right-aligned colored badges or values.
    for s in status_rows:
        base = cy + int(status_row_h * 0.72)
        cv2.putText(frame, s["label"] + ":", (x + pad, base), FONT,
                    0.48 * base_scale, (232, 235, 240), text_thickness, cv2.LINE_AA)
        if s["kind"] == "badge":
            bs = badge_scale
            (tw, th), _ = cv2.getTextSize(s["value"], FONT, bs, text_thickness)
            bx_pad, by_pad = max(6, int(7 * base_scale)), max(3, int(4 * base_scale))
            bw, bh = tw + 2 * bx_pad, th + 2 * by_pad
            bx = x + width - pad - bw
            by = base - th - by_pad
            rounded_rect(frame, bx, by, bw, bh, bh // 2,
                         tuple(int(c * 0.28) for c in s["color"]), -1)
            rounded_rect(frame, bx, by, bw, bh, bh // 2, s["color"], 1)
            cv2.putText(frame, s["value"], (bx + bx_pad, base), FONT, bs,
                        s["color"], text_thickness, cv2.LINE_AA)
        else:
            val_scale = 0.50 * base_scale
            (tw, _), _ = cv2.getTextSize(s["value"], FONT, val_scale, text_thickness)
            cv2.putText(frame, s["value"], (x + width - pad - tw, base), FONT,
                        val_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
        cy += status_row_h

    # Metrics section: five cards in a two-column grid (last card spans full width).
    if metric_rows:
        cy += section_gap
        cv2.line(frame, (x + pad, cy), (x + width - pad, cy), (72, 78, 87), 1, cv2.LINE_AA)
        cy += max(15, int(20 * base_scale))
        cv2.putText(frame, "PEDESTRIAN METRICS", (x + pad, cy - int(4 * base_scale)),
                    FONT, 0.38 * base_scale, (175, 185, 198), 1, cv2.LINE_AA)
        cy += max(7, int(8 * base_scale))

        inner_w = width - 2 * pad
        col_gap = card_gap
        col_w = max(1, (inner_w - col_gap) // 2)
        for i, s in enumerate(metric_rows):
            row_idx, col_idx = divmod(i, 2)
            # A fifth card spans the full row for a balanced, intentional layout.
            if i == 4:
                col_idx = 0
                this_w = inner_w
            else:
                this_w = col_w
            cx = x + pad + col_idx * (col_w + col_gap)
            card_y = cy + row_idx * (card_h + card_gap)
            # Slightly transparent card background
            card_overlay = frame.copy()

            rounded_rect(
                card_overlay, cx, card_y, this_w, card_h,
                min(8, card_h // 2), (42, 46, 54), -1
            )

            alpha = 0.65  # Lower = more transparent; higher = more opaque
            cv2.addWeighted(card_overlay, alpha, frame, 1 - alpha, 0, frame)

            # Keep the border crisp
            rounded_rect(
                frame, cx, card_y, this_w, card_h,
                min(8, card_h // 2), (63, 69, 79), 1
            )

            label = s["label"]
            val = s["value"]
            small_label_scale = 0.34 * base_scale
            (lw, lh), _ = cv2.getTextSize(label, FONT, small_label_scale, 1)
            cv2.putText(frame, label, (cx + card_pad, card_y + card_pad + lh),
                        FONT, small_label_scale, (183, 192, 204), 1, cv2.LINE_AA)

            if s["kind"] == "badge":
                val_color = s["color"]
            else:
                val_color = (248, 250, 252)
            # Values are larger and bolder than the labels for quick scanning.
            val_thickness = max(1, text_thickness)
            val_font = 0.54 * base_scale
            (vw, vh), _ = cv2.getTextSize(val, FONT, val_font, val_thickness)
            max_value_w = max(10, this_w - 2 * card_pad)
            if vw > max_value_w:
                val_font *= max(0.65, max_value_w / vw)
                (vw, vh), _ = cv2.getTextSize(val, FONT, val_font, val_thickness)
            value_x = cx + card_pad if this_w == inner_w else max(
                cx + card_pad, cx + this_w - card_pad - vw)
            value_y = card_y + card_h - max(6, int(8 * base_scale))
            cv2.putText(frame, val, (value_x, value_y), FONT, val_font,
                        val_color, val_thickness, cv2.LINE_AA)

    # Any remaining optional detail rows are kept out of the metric grid.
    # Next Contact is already rendered as the fifth metric card above.

