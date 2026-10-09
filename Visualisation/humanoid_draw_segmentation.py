from YOLO.segmentation import colors,color_names
import cv2
import numpy as np
import math
from Visualisation.Info_Panel import stat, draw_stats_panel, tick_fps, rounded_rect,WHITE, GREEN, BLUE, YELLOW, ORANGE, RED
import time

allowed_objects = ['person']
font = cv2.FONT_HERSHEY_SIMPLEX   # Standard clean OpenCV font style
font_scale = 0.5                  # Font size (0.5 is compact and readable)
text_thickness = 1 
DANGER_RADIUS = 1.5
CAUTION_RADIUS = 2.7
AWARE_RADIUS = 4.0
zone_state = {}
history={}
ZONE_RANK = {"DANGER": 0, "CAUTION": 1, "AWARE": 2, "CLEAR": 3}
OUTER_RADIUS = {"DANGER": DANGER_RADIUS, "CAUTION": CAUTION_RADIUS, "AWARE": AWARE_RADIUS}
MARGIN = 0.4

def update_zone(track_id, distance):
    candidate = get_zone(distance)          # plain zone, no memory
    old = zone_state.get(track_id)

    if old is None:                         # first time we see this person
        zone_state[track_id] = candidate
    elif ZONE_RANK[candidate] < ZONE_RANK[old]:           # got closer
        zone_state[track_id] = candidate
    elif ZONE_RANK[candidate] > ZONE_RANK[old]:           # moved farther
        if distance > OUTER_RADIUS[old] + MARGIN:         # only if clearly beyond the sticky band
            zone_state[track_id] = candidate
    return zone_state[track_id]

def Draw_Segment_Around(boxes,labels,class_names,frame,track_ids,scores,mask,distance_estimator,center_x,focal_length_x,now):
    frame_overlay = frame.copy()

    if mask is not None and track_ids is not None:
        polygons = mask.xy

        for polygon,track_id,score,label,class_name,box in zip(polygons,track_ids,scores,labels,class_names,boxes):
            class_name = class_names[int(label)]
            if class_name not in allowed_objects or score < 0.5:
                continue
            polygon = polygon.astype(np.int32)
            track_id = int(track_id)
            color = colors[track_id % len(colors)]
            cv2.fillPoly(frame_overlay,[polygon],color)

    alpha = 0.25
    frame = cv2.addWeighted(frame,1-alpha,frame_overlay,alpha,0)     

    people = []
    
    if mask is not None and track_ids is not None:
            polygons = mask.xy
            for polygon,track_id,score,label,class_name,box in zip(polygons,track_ids,scores,labels,class_names,boxes):
                x1,y1,x2,y2 = box.int().tolist()
                class_name = class_names[int(label)]
                if class_name not in allowed_objects or score < 0.5:
                    continue
    
                polygon = polygon.astype(np.int32)
                track_id = int(track_id)
                color = colors[track_id % len(colors)]
                color_name = color_names[track_id % len(colors)]

                #------Distance Calculation-------@
                ys = polygon[:, 1]
                pixel_height = ys.max() - ys.min()
                feet_y = ys.max() - 5            # feet row from the mask itself

                forward_depth = distance_estimator.get_distance_to_base(feet_y)
                x1, y1, x2, y2 = box.int().tolist()
                horizontal_m = ((x1 + x2) / 2.0 - center_x) * forward_depth / focal_length_x
                distance = math.hypot(horizontal_m, forward_depth)
                real_height = 1.45 * pixel_height / (feet_y - 587)
                height_cm = real_height * 100
                height_text = f"{height_cm:.1f}cm"

                HALF_BODY = 0.25
                EDGE_PX = 8
                if x1 < EDGE_PX:
                    lateral_m = (x2 - center_x) * forward_depth / focal_length_x - HALF_BODY
                elif x2 > frame.shape[1] - EDGE_PX:
                    lateral_m = (x1 - center_x) * forward_depth / focal_length_x + HALF_BODY
                else:
                    lateral_m = horizontal_m

                # ---- zone logic ----
                old_zone = zone_state.get(track_id)
                zone = update_zone(track_id, distance)
                person = {"t":now,
                          "track_id":track_id,
                          "distance":distance,
                          "zone":zone,
                          "horizontal_m":lateral_m,
                          "forward_depth":forward_depth,
                          "feet_cut": bool(ys.max() >= frame.shape[0] - 6),
                         "side": side_of(h=lateral_m)}
                people.append(person)
                history.setdefault(track_id,[]).append(person)
                history[track_id] = [e for e in history[track_id] if now - e["t"] <= 2.0]
                closing, motion, ttc = get_motion(track_id, now)
                person["closing"] = closing
                person["motion"] = motion
                person["ttc"] = ttc
                # if track_id in (7, 8):
                #     c = f"{closing:.1f}" if closing is not None else "None"
                #     print(f"id={track_id} dist={distance:.1f} closing={c}")
                t_cpa, miss = get_cpa(track_id, now)
                person["t_cpa"] = t_cpa
                person["miss"] = miss
                update_threat(track_id, person, now)
                # if person["threat"]:
                #     print(f'THREAT id={track_id} dist={distance:.1f} t_cpa={person["t_cpa_shown"]:.1f} miss={person["miss"]}')
                # if zone != old_zone:
                #     print(f"{track_id} ({color_name}): {old_zone} -> {zone} at {distance:.1f} m")

                label_x = (x1 + x2) / 2.0
                label_y = ys.min() + 0.3 * pixel_height
                dist_text = f"ID{track_id}: {distance:.1f}m" #if distance <= 15 else ">15 m"
                cv2.polylines(frame,[polygon],isClosed=True,color=color,thickness=text_thickness,lineType=cv2.LINE_AA)
                draw_distance_label(frame, dist_text, label_x, label_y, color,pixel_height)
                
                # cv2.putText(frame, distance, (text_x, text_y), font, font_scale, (0, 255, 255), text_thickness, cv2.LINE_AA)

    #----Check and delete history if person disappear and in history for more then 2 seconds---@
    before_ids = list(history.keys())          # snapshot before removal
    removed = remove_ids_from_history(now=now)

    # if removed:
    #     print(f"Removed: {removed}")
    #     print(f"Before: {before_ids}")
    #     print(f"After:  {list(history.keys())}")

    # ---- panel: OUTSIDE the if and the for, once per frame ----
    dists = [p["distance"] for p in people]
    danger_ids = [str(p["track_id"]) for p in people if p["zone"] == "DANGER"]
    caution_n = sum(1 for p in people if p["zone"] == "CAUTION")
    aware_n = sum(1 for p in people if p["zone"] == "AWARE")
    
    path_people = [p for p in people if abs(p["horizontal_m"]) < CORRIDOR_HALF and p["forward_depth"] < 3.0]
    path_blocked = len(path_people) > 0
    path_text = ", ".join(f'{p["track_id"]} {p["side"]}' for p in path_people)

    approaching = [p for p in people if p["motion"] == "APPROACHING"]
    threats = [p for p in people if p["threat"]]
    threat = min(threats, key=lambda p: p["t_cpa_shown"], default=None)
    level = ttc_level(threat["t_cpa_shown"]) if threat else None

    if danger_ids or level == "DANGER":    state, state_col = "DANGER", RED
    elif caution_n or level == "CAUTION":  state, state_col = "CAUTION", ORANGE
    elif aware_n or level == "AWARE":      state, state_col = "AWARE", YELLOW
    else:                                  state, state_col = "NORMAL", GREEN

    LEVEL_COLORS = {"DANGER": RED, "CAUTION": ORANGE, "AWARE": YELLOW}
    if threat is not None:
        ttc_text = f'{threat["track_id"]} {threat["side"]} {threat["t_cpa_shown"]:.1f}s'
        ttc_color = LEVEL_COLORS[level]
    else:
        ttc_text, ttc_color = "", ORANGE

    run_early_monitor(people, now)

    stats = [
        stat("State", state, "badge", state_col),
        stat("FPS", f"{tick_fps():.1f}"),
        stat("Zone", "PEDESTRIAN", "badge", BLUE),
        stat("Current Path", "BLOCKED" if path_blocked else "CLEAR", "badge", RED if path_blocked else GREEN),
        stat("In Path",path_text, "badge", RED, only_if=path_blocked),
        stat("Impact", ", ".join(danger_ids), "badge", RED, only_if=bool(danger_ids)),
        stat("Approaching", len(approaching), section=2),
        stat("TTC", ttc_text, "badge", ttc_color, only_if=threat is not None),

        stat("People Detected", len(people), section=2),
        stat("Min Distance", f"{min(dists):.1f} m" if dists else "--", section=2),
        stat("Avg Distance", f"{sum(dists)/len(dists):.1f} m" if dists else "--", section=2),

        stat("Aware Count", aware_n, on_panel=False),
        stat("Caution Count", caution_n, on_panel=False),
    ]
    draw_stats_panel(frame, stats)

    return frame


def draw_distance_label(frame, text, cx, cy, color,person_h):
    color = tuple(int(c) for c in color)
    fill = tuple(int(c * 0.25) for c in color)
    text_color = tuple(int(c + (255 - c) * 0.6) for c in color)

    ui_font = cv2.FONT_HERSHEY_DUPLEX
    scale = float(np.clip(person_h / 600, 0.25, 0.4))   # font size
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, ui_font, scale, thick)
    padx, pady = int(8 * scale / 0.5), int(5 * scale / 0.5)
    w, h = tw + 2 * padx, th + 2 * pady
    r = h // 2                                                  # fully rounded ends

    # keep the pill inside the frame
    x = int(max(0, min(cx - w / 2, frame.shape[1] - w - 1)))
    y = int(max(0, min(cy - h / 2, frame.shape[0] - h - 1)))

    rounded_rect(frame, x, y, w, h, r, fill, -1)
    rounded_rect(frame, x, y, w, h, r, color, 1) # corner thickness
    cv2.putText(frame, text, (x + padx, y + pady + th),
                ui_font, scale, text_color, 1, cv2.LINE_AA)


#-----------Self Aware Zones------------@
def get_zone(distance):
    if distance <= DANGER_RADIUS:
        return "DANGER"
    elif distance <= CAUTION_RADIUS:
        return "CAUTION"
    elif distance <= AWARE_RADIUS:
        return "AWARE"
    else:
        return "CLEAR"

#----Check Which side object appear---@
def side_of(h, center=0.4):
    if h < -center:
        return "LEFT"
    if h > center:
        return "RIGHT"
    return "FRONT"

#----Remove trackid from history dictionary who never appear again or timestamp is older then 2 seconds----@
def remove_ids_from_history(now):
    removed = []
    for tid in list(history.keys()):
        if now - history[tid][-1]['t'] > 2.0:
            del history[tid]
            zone_state.pop(tid, None)
            threat_state.pop(tid, None)
            monitor_state.pop(tid, None)
            closing_hist.pop(tid, None)
            removed.append(tid)
    return removed

#----
APPROACH_SPEED = 0.5   # m/s needed to count as approaching/receding

def get_motion(track_id, now, window=0.8):
    pts = [(e["t"], e["distance"]) for e in history[track_id] if now - e["t"] <= window and not e.get("feet_cut")]
    if len(pts) < 5:                                   # not enough data yet
        return None, "UNKNOWN", None
    t = np.array([p[0] for p in pts])
    d = np.array([p[1] for p in pts])
    closing = -np.polyfit(t, d, 1)[0]                  # m/s, positive = getting closer

    if closing > APPROACH_SPEED:
        motion = "APPROACHING"
    elif closing < -APPROACH_SPEED:
        motion = "RECEDING"
    else:
        motion = "STEADY"

    ttc = d[-1] / closing if closing > APPROACH_SPEED else None
    return closing, motion, ttc

CORRIDOR_HALF = 0.6

def is_threat(p):
    return (p["t_cpa"] is not None
            and p["t_cpa"] < 5.0
            and p["miss"] < MISS_LIMIT
            and p["forward_depth"] < 8.0)

def ttc_level(ttc):
    if ttc < 1.5:  return "DANGER"
    if ttc < 2.5:  return "CAUTION"
    return "AWARE"

#---- Getting time to approach or possible collision with Miss----@
MISS_LIMIT = 0.9     # m: closest approach under this = collision course

def get_cpa(track_id, now, window=1.2):
    pts = [e for e in history[track_id] if now - e["t"] <= window and not e.get("feet_cut")]
    if len(pts) < 5:
        return None, None

    t = np.array([e["t"] for e in pts])
    t = t - t[-1]                                    # newest sample = time 0
    x = np.array([e["horizontal_m"] for e in pts])   # sideways
    z = np.array([e["forward_depth"] for e in pts])  # ahead

    vx, px = np.polyfit(t, x, 1)     # slope = speed, intercept = position now
    vz, pz = np.polyfit(t, z, 1)

    vv = vx * vx + vz * vz
    if vv < 0.25:                    # relative speed under 0.5 m/s: not closing
        return None, None

    t_cpa = -(px * vx + pz * vz) / vv    # time of closest approach
    if t_cpa <= 0 or t_cpa > 6.0:                        # already past / moving away
        return None, None

    miss = math.hypot(px + vx * t_cpa, pz + vz * t_cpa)
    return t_cpa, miss

threat_state = {}   # track_id -> {"streak", "until", "t_cpa", "t_ref"}

def shrinking(p, now, span=1.0, min_drop=0.4):
    pts = [e["distance"] for e in history[p["track_id"]] if now - e["t"] <= span]
    if len(pts) < 8:
        return False
    return np.median(pts[:3]) - np.median(pts[-3:]) >= min_drop

def update_threat(track_id, p, now):
    s = threat_state.setdefault(track_id, {"since": None, "until": 0.0, "t_cpa": None, "t_ref": now})

    if is_threat(p) and shrinking(p, now):
        if s["since"] is None:
            s["since"] = now
        if now - s["since"] >= 0.3:              # 0.3 s of video time, about 9 frames
            s["until"] = now + 0.5
            s["t_cpa"] = p["t_cpa"]
            s["t_ref"] = now
    else:
        s["since"] = None

    p["threat"] = now < s["until"]
    p["t_cpa_shown"] = max(0.0, s["t_cpa"] - (now - s["t_ref"])) if p["threat"] else None

#-----Early Monitoring----@
EARLY_MONITOR_ENABLED = True
monitor_state = {}   # track_id -> (last logged label, time)
early_events = []    # for logs / the LLM
closing_hist = {}    # track_id -> [(t, closing), ...] last 4 s

def run_early_monitor(people, now):
    if not EARLY_MONITOR_ENABLED:
        return
    for p in people:
        tid, c = p["track_id"], p.get("closing")
        if (c is None or p["feet_cut"] or not (3 < p["forward_depth"] < 12)
                or abs(p["horizontal_m"]) > 1.2):
            continue

        buf = closing_hist.setdefault(tid, [])
        buf.append((now, c))
        buf[:] = [x for x in buf if now - x[0] <= 4.0]

        before = [v for t, v in buf if 1.0 <= now - t <= 3.5]
        recent = [v for t, v in buf if now - t <= 0.5]
        if len(before) < 8 or len(recent) < 4:
            continue
        if np.std(before) > 0.5:             # wasn't steady before, can't call it a change
            continue

        jump = float(np.median(recent) - np.median(before))
        if jump >= 1.8:      label = "WALKING_TOWARD"
        elif jump >= 0.7:    label = "STOPPED_OR_SLOWED"
        else:                continue

        last = monitor_state.get(tid)
        if last and last[0] == label and now - last[1] < 3.0:
            continue                          # same event, don't re-log for 3 s
        monitor_state[tid] = (label, now)
        early_events.append({"t": round(now, 1), "id": tid, "label": label,
                             "dist": round(p["distance"], 1), "side": p["side"]})
        print(f"EARLY id={tid} {label} dist={p['distance']:.1f} jump={jump:.1f}")


#   miss        	What it means	        Real-life picture
# 0 to 0.3 m	  Head-on or almost	    You walk into each other
# 0.3 to 0.5 m	  Bodies overlap	    Shoulder bump or collision
# 0.5 to 0.9 m	  Almost touching	    Shoulders or elbows brush, or one of you has to turn a bit
# 0.9 to 1.2 m	  Close pass	        No touch, but you'd feel him go by (personal space)
# 1.2 to 2.0 m	  Safe pass	            Comfortable gap, like passing someone on a sidewalk
# above 2.0 m	  Far pass	            Not even close, no reaction needed