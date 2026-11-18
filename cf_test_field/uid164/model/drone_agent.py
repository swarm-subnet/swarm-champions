"""Swarm Test Field, architecture A21 (A20 + the three root causes of gate 10: (1) a drone carried off flyable ground (the margin beside a crown) flies straight back onto it before any path, as A13 did, instead of hovering while the governor zeroes every command until the time runs out; (2) the steady rule never reports a box that a weaker box of a clearly different size overlaps (the detector torn between part of a crouching man and all of him: two 0.71 boxes of his back were reported and judged no_object); (3) a far close-up (glance) proves a sighting only with one of its own boxes nearer to that sighting than to anyone known before, never with a neighbour's body 3 m away, which had made one-time sightings beside real men worth two empty visits each).  A20 = (A19's flight code unchanged, with the detectors' sixth colour and fifth thermal rounds, trained also on a second rim bank of 2,000 fresh seeds with more wide views: on held-out rim close-ups a sure box 1.5-2.5 m from the edge 89-92 % (A19: 88-91 %), 2.5-3.5 m 93-95 % (92-93 %); on held-out flight close-ups 97.7 % (97.4 %) with no wrong box at 0.8; thermal 84-91 % on rim frames (82-88 %)).  A19 = (A18 + a thermal look completes a report (TZ) only on a close-up box a report may be made on (REPORT_BOX_SCORE, or the steady rule): until now the thermal side reported on whatever close-up was shown, and in gate 6 it reported a 0.70 box of 0.76 m, part of a crouching man at the rim that look_zoom had just refused (no_object, and the man was lost).  A18 = (A17 + a ghost of a reported man is left out of the waiting list also when the time for visits runs short and on a leg to a search stop; only the last pass, when nothing else is left, visits it.  In gate 5 a ghost 2.6 m from a reported man, born at 111 s, still took two visits in the last 45 s, and a man at the north rim seen first at 140 s was not reached).  A17 = (A16 + ghosts of reported men go last: a far close-up check (glance) of a sighting no longer counts a man already reported, 3 m away, as someone at the spot; and a sighting made five times or fewer within 3.5 m of a reported man, never warm and never boxed by a close-up of its own (ghost()), is visited only when nobody else is waiting, also on the way home.  In A15's 2,320 flights 314 visits went to such spots, 2.4-3.2 m from a reported man, and all found nobody; in gate 4 two of them cost about 30 s and two people were still unvisited when the time ran out).  A16 = (A15 + three narrow changes for a man the close-ups see but never surely: a 0.70-0.80 close-up box may be reported when an earlier close-up of the same visit put a box of the same size on the same spot with 0.70 too (STEADY_SCORE; on held-out flight and rim close-ups no such first report was wrong); two agreeing close-ups that wait only for the thermal looks stop further close-ups and the last thermal look reports on the close-up still shown (zz_in_hand: the judge compares a report with the close-up shown); and a revisit goes back to a side whose close-ups boxed him instead of avoiding it like a side that showed nothing (views_saw).  Gate 3's one miss, a thermal-blind man crouching at the rim, was boxed 0.70, 0.71, 0.74 from the west and never from the north, where both later visits went).  A15 = (A12's flight code unchanged, with the detectors' next rounds trained on rim people: every person missed or wrongly boxed in A12's real-map gates stood within 5 m of the field edge, lying or crouching, where the colour detector gave a sure box on only 80-82 % of close-ups; colour v9r adds 2,000 seeds of close-ups taken from where the drone really flies, and lifts that to 88-91 % on held-out rim seeds and 95.3 -> 97.4 % on held-out flight close-ups with no wrong box at 0.8; thermal v7r adds the same seeds' thermal frames).  A12 = (A11b + the published field's ground is KNOWN: when the surveyed edge in the site map is the published one, the ground height everywhere comes from the field's own height map (field_heights.npz, 0.5 m grid) from the first frame, instead of a plane fitted to laser readings during the flight; before, sightings made while the plane was still settling were put 2-3 m off along the line of sight on this 11 % slope and opened second tracks beside real people, which cost visits to nobody; the laser still checks the map and the plane fit takes over if they disagree; and REPORT_BOX_SCORE is 0.80, the cut above which no box on 1,423 people in real flight close-ups failed the judge).  A11b = (A11 + a report is made only on a close-up box that scores REPORT_BOX_SCORE: on close-ups of real-map seeds never trained on, every one of 661 boxes of 0.7 and more fitted the man, while 4 of 38 between 0.5 and 0.7 did not, too big or half of him, and the judge takes such a report for a report on nothing; A11 = A9e logic with both detectors fine-tuned on the published map, swarm-worlds v1.1.1, at first on images rendered there (colour v6r, thermal v5r); A12 carries the third round, trained also on the close-ups and thermal frames real flights took: colour v7r, thermal v6r).  A9e = (A9d + a report on colour close-ups alone needs two agreeing close-ups that BOTH score 0.55: the one false report of the third gate, two dark patches of ground, had 0.61 and 0.45, while the weakest of nineteen valid ones had 0.66 and 0.61) — A9d was (A9c + the time kept for the way home is measured along the way that will really be flown, round the people found, and the careful way home over checked ground is taken only when it is not much longer: one flight had turned back half-way home and landed too late) — A9c was (A9b + a weak sighting made three times or more on one spot gets a visit when nothing else is left, also after an empty close-up check; and when the time for visits runs short, everyone still waiting is visited, whatever a close-up from afar showed) — A9b was (A9 + empty close-up checks never end a sighting the wide camera has made three times or more: it only goes last) — A9 was (A8h + check before you fly: a sighting the colour camera alone has made gets one aimed close-up from where the drone is, and is flown to only if that close-up shows someone; a sighting the close-up shows empty goes last, two empty ones from two places end it; a box with a score just under the bar is kept as a weak sighting that only such a close-up can raise; the smallest whole body measured in flight is 0.76 m, so a box under 0.72 m is a part) — A8h was (A8g + the fixes from the first gate: while the ground model is not yet trusted the drone spins again and never starts scanning, a scan gives way to a new plan every few seconds, a person found while no visit was possible still interrupts later, a box too small to be a whole body (a head, a limb) is no report but a re-aim, and a person with no viewpoint on the usual rings is looked at from farther off) — A8g was (A8f + a person who can only be looked at from far off and flat, as on the high rim, is placed by the laser aimed at his close-up box, not by the ground model, whose height errors grow three to four times along so flat a line) — A8f was (A8e + the time kept for the way home is the measured one: 12 s to land plus a quarter second per metre of the way over flyable ground, with 13 s on top; a look begun while the drone is still braking gets the braking time added to its stay) — A8e was (A8d + a box too small to trust alone is kept, and three of them on one spot make a sighting: a crouching person 25 m away is 15-25 px and was thrown away frame after frame) — A8d was (A8c + nobody found waits long for his visit: a close-up sighting counts as surely real, anyone else is visited ten seconds after he was first seen at the latest, and everyone at once when the time for visits runs short) — A8c was (A8b + a visit that finds nobody where a box was seen from one place only leaves that box's offset as a hint for the shadow, so the next box at the same offset settles it) — A8b was (A8 + the code review's fixes: ground counts as checked only where a whole standing body would have been in the frame and in sight over the crowns, every report stands on a colour close-up, the way home keeps off unchecked ground first, laser readings off trees stay out of the ground fit, a failed visit is tried again, the second look at someone near a reported person comes from the report's side, people on high ground are looked at from nearer, waiting does not replan ten times a second) — A8 was (A7b + the drone's own shadow is told from a person by the fact that it follows the drone, a stop is judged by what its spins really show on the ground as it lies (higher or lower than the dock) and the thermal spin is tilted for the shade it is meant for, ground a stop promised and did not show is left to close-ups, and someone newly found is visited at once, the surely real first) — A7b was (A7 + small leftover patches are checked by aimed close-ups from afar before any flight to a new stop for them) — A7 was (A6b + a person first seen on the way home is still visited while the time allows, and the flight keeps its own account of where the time went) — A6b was (A6 + leftover ground checked by aimed close-ups from afar, stops chosen by what they can really see, sightings just inside a crown circle moved to where a person can stand, thermal kept on once it has seen a warm body) — A6 was (A5 + the thermal camera where colour is weak: ground seen in tree shade is cleared only by a thermal look, a thermal spin follows the colour spin where shade is in reach, and every visit takes thermal looks beside its close-ups; ground hidden behind a crown is not counted as seen) — A5 was (A3b + the code review's fixes: the same answer when asked twice, reported spots pinned, people 3 m apart kept apart, every person with a viewpoint visited before going home, sight lines that end short of the man, a try counted when he was really looked at, maps that change only when something changed) — A3b was (A3 + close-ups through the 3x lens unless he is far, a height check on standing boxes, an easier second look, a wider home reserve) — A3 was (A2 + the world's own rules as filters: nobody within 8 m of the dock or under a crown; two agreeing close-ups per report; a safety governor on every stick command; position hold): look before you fly, confirm before you report.

People on the field never move and never react, so the patrol is a search problem with a safety rule:

  maps      a 1 m grid over the field (dock frame): where the drone may fly (inside the edge by EDGE_CLEAR_M, off every
            crown by CROWN_CLEAR_M), which ground has been CLEARED (seen closely enough that a person there would have
            been found), which ground is COVERED (the scoring rule: 8.3 cm per pixel), and the people found;
  safety    the drone only goes where every bit of ground within SAFE_R_M has been cleared and every person found is
            PERSON_CLEAR_M away: an unseen person can never be under it.  The dock's own disc starts cleared (nobody
            stands within 8 m of the dock);
  search    stop, spin a full turn with the camera SPIN_TILT_DEG down (8 frames), then pick the next stop that clears the
            most ground per second over allowed ground; repeat until every cell of the field is cleared or none is
            reachable;
  confirm   for each person found: fly to a point VIEW_M away, face him, aim the camera centre at him, take the laser
            point and a close-up (3x or 7x by distance), report him once on the close-up box with the laser point as
            his position (the ground plane fit as a fallback);
  fill      with time left, spin where uncovered cells remain;
  home      route to the dock over allowed ground, press return where the straight line is free, keep RESERVE_S.

Eyes are a detector (Sentinel day_zoom_so on both wide and zoom images) or, in local tests only, the truth the runner
passes as observation["_truth"] (never present in a real patrol), filtered by the same size floors a good detector
needs.
"""
import heapq
import math
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, convolve

HERE = Path(__file__).resolve().parent
ACTION_DIM = 26
# ---- flight
CRUISE_M = 9.3
MAX_MPS = 4.0
BRAKE_MPS2 = 1.5
YAW_MAX_DPS = 90.0
YAW_KP, YAW_KD = 1.6, 0.45            # wanted turn rate (deg/s) per degree of error, braked by the measured rate
SPIN_TILT_DEG = -45.0
SPIN_FRAMES = 8                        # 45 degrees a frame at full turn
GIMBAL_RANGE = (-90.0, 90.0)
# ---- maps
GRID_M = 1.0
EDGE_CLEAR_M, CROWN_CLEAR_M = 3.5, 3.0
SAFE_R_M = 5.5                         # the 5 m rule with 0.5 m to spare, in 3D to a person
PERSON_TOP_M = 2.0                     # a 1.8 m person, with 0.2 m to spare
BODY_M = 1.5                           # a lying body reaches this far from its spot
UNKNOWN_RISE_M = 3.0                   # before the slope is known: the ground may be this much higher than the dock's
PERSON_CLEAR_M = 6.5
DOCK_CLEARED_M = 6.5                   # people stand at least 8 m from the dock, a lying body reaches about 1 m
CLEAR_PIX_M = float(os.environ.get("TF_CLEAR_PIX_M", "0.040"))   # ground per pixel at which a person is surely found
COVER_PIX_M = 0.5 / 6.0                # the scoring rule
# ---- cameras
COLOUR_FOV = (82.0, 640, 480)
WIDE_F = 0.5 * math.hypot(640, 480) / math.tan(math.radians(82.0 / 2))
ZOOM_F = {3: 0.5 * math.hypot(640, 480) / math.tan(math.radians(35.0 / 2)),
          7: 0.5 * math.hypot(640, 480) / math.tan(math.radians(15.0 / 2))}
THERMAL_F = 0.5 * math.hypot(640, 512) / math.tan(math.radians(45.0 / 2))      # the thermal main feed: 640 x 512, 45 degrees
CAMERA_FORWARD_M, CAMERA_BELOW_M = 0.16, 0.04
# ---- eyes
DET_THR = 0.35
WIDE_MIN_PX = float(os.environ.get("TF_WIDE_MIN_PX", "28"))     # a wide box this big (longer side) is a sighting
WIDE_MIN_SCORE = 0.45
WIDE_SMALL_PX = 12.0                   # a box from this size up to WIDE_MIN_PX is kept as a small sighting
SMALL_NEED, SMALL_SAME_M, SMALL_KEEP_S = 3, 1.5, 60.0   # this many small sightings on one spot within a minute make a track
ZOOM_MIN_PX = 40.0
ZOOM_SCORE = 0.55                      # one of the two agreeing close-ups must be this sure
ZOOM_SCORE2 = 0.45                     # the other may be this sure: standing on the same spot is evidence too
LENS7_FROM_M = 36.0                    # the 7x lens only from this far: nearer he fills the frame and the box comes out partial
STANDING_MIN_M = 1.25                  # a tall narrow box shorter than this on the ground is half a man
ORACLE_MIN_PIXELS = {"feed": 120, "zoom": 400, "thermal": 300}
THERMAL_SCORE, THERMAL_MIN_PX = 0.5, 24    # a warm body in a thermal frame
TT_SCORE = 0.6                         # for a report on the thermal frame itself
THERMAL_LOOKS = 2                      # thermal frames taken of each person visited, aimed at him
SENSES_AGREE_M = 1.5                   # a warm body and a close-up box this close on the ground are one person
ZZ_SCORE = 0.55                        # both close-ups of a colour-only report must score this
REPORT_BOX_SCORE = 0.80                # the close-up box a report is made on must score this: a less sure box may not fit him
OFF_GROUND_MPS = 2.0                   # speed back onto flyable ground from the margin beside a crown or the edge
RIVAL_SCORE, RIVAL_IOU, RIVAL_SIZE = 0.35, 0.25, 1.3   # a weaker box overlapping the chosen one this much, with a longer
                                       # side this different, means the detector is torn between part of him and all of him
STEADY_SCORE, STEADY_SIZE = 0.70, 1.2  # ...or this, when an earlier close-up of the same visit put a box of the same size
                                       # (width and height within 20 %) on the same spot with this score too
FIELD_HEIGHTS = "field_heights.npz"      # the published field: ground heights (cm above its lowest point) on a grid in the map frame, and its surveyed edge
DOCK_PAD_M = 0.51                      # the dock frame's origin stands this far above the ground under the dock (measured: 0.509, spread 0.04)
FIELD_MATCH_M = 0.3                    # the site map's edge is the published one when every corner fits it this closely after one shift
FIELD_DOUBT_M = 0.8                    # laser ground readings this far off the height map on the whole: not that field after all
ALLOW_ZZ = True                        # report on two agreeing close-ups when the thermal looks showed nobody
ALLOW_TT = False                       # no report on a thermal frame alone: warm look-alikes exist, and one on the real map would repeat in every patrol
SHADE_REL = 0.6                        # ground darker than this share of the brightest soil seen is in shade
THERMAL_CLEAR_PIX_M = 0.045            # ground per thermal pixel at which a warm body is surely found (about 26 m away)
TSPIN_DPS, TSPIN_TILT_DEG, TSPIN_S = 60.0, -35.0, 7.0   # thermal spin: 30 degrees a frame through a 36 degree lens
TSPIN_NEED = 8                         # shaded cells in reach that make a thermal spin worth its seven seconds
SCAN_PIX_M = 0.03                      # ground per close-up pixel at which a scan clears it (3x: to 24 m, 7x: to 43 m)
SCAN_KEEP_ZOOMS = 26                   # close-ups kept back for visits (six people, four each, and spare)
SCAN_AIM_S, SCAN_GIVE_UP_S = 0.2, 4.0
SNAP_M = 2.5                           # a sighting this close to ground a person can stand on is moved onto it
UNDER_CROWN_M = 2.2                    # a sight line this low passes under a crown
SAME_M = 2.0                           # sightings this close are one person (people stand 3 m apart)
SAME_REPORTED_M = 1.8                  # a close-up spot this close to a reported (pinned) spot is that person again
GHOST_NEAR_M, GHOST_N = 3.5, 5          # a sighting made this few times, this near a reported man, with no close-up box of its
                                       # own and not warm, is most likely him seen again with a larger error: visited last
# ---- confirm
VIEW_M = (13.0, 16.0, 10.0, 20.0)       # 13 m first: from 9.3 m up he is seen 35 degrees down, more from the side
VIEW_BEST_M = 13.0
VIEW_MIN_M, VIEW_MAX_M = 9.0, 17.0      # looked at from where the drone is only this near: farther, the spot gets worse
AIM_TOL_DEG = 3.0
CONFIRM_GIVE_UP_S = 25.0
STAY_S = 9.0
LASER_AGREE_M = 2.0
DOCK_KEEP_M = 7.0                      # people stand at least 8 m from the dock: nearer than this it is the dock, or its shadow
CROWN_INSIDE_M = 1.0                   # people stand outside every crown circle: this deep inside one it is the tree
LOOKS_AGREE_M = 1.0                    # two close-ups whose boxes stand this close on the ground are one person, seen twice
ZOOMS_PER_VISIT = 4
HARD_EDGE_M, HARD_CROWN_M = 2.0, 1.6   # the governor's lines: no stick command may carry the drone past them
# ---- clock
HORIZON_S = 180.0
BODY_TOP_M = 1.9                       # ground counts as checked where a body this tall would have been in the frame
SEEN_TOP_M = 1.5                       # ... and where this much of him shows under a crown on the sight line
PIN_NEAR_M = 3.5                       # a new sighting this close to a reported person gets its look from the report's side
SHADOW_MOVE_M = 3.0                    # the drone must have moved this far between two looks to tell what stays from what follows it
SHADOW_SAME_M = 1.0                    # two offsets from the drone this close are the same offset
STATIC_SAME_M = 1.5                    # two sightings this close, from places SHADOW_MOVE_M apart, are a thing that stays put
SHADOW_NEED = 2                        # earlier places with a box at the same offset before it is taken for the drone's own shadow
UNPROVEN_WAIT_S = 10.0                 # ... or once this long has passed since he was first seen
VISIT_ALL_S = 45.0                     # with less than this beyond the way home left, everyone found is visited at once
UNPROVEN_WAIT_M = 8.0                  # someone seen from one place only is visited once the drone has moved this far (or nothing else is left)
TSPIN_TILTS = (-25.0, -35.0, -45.0, -55.0, -65.0)   # the thermal spin picks the tilt that shows the most of the shade in reach
STOP_MIN_CELLS = 30                    # a flight to a new stop and its spin must check this much ground; less is left to close-ups
LATE_LOOK_S = 9.0                      # a look at someone first seen on the way home: the stay at his viewpoint
BODY_MIN_M = 0.72                      # a close-up box whose longer side is under this on the ground is a head or a limb, not a body
WIDE_WEAK_SCORE = 0.40                 # a wide box from this score up to WIDE_MIN_SCORE is a weak sighting: a close-up decides
GLANCE_MAX_M, GLANCE_MIN_M, GLANCE_GIVE_UP_S, GLANCE_KEEP_ZOOMS = 40.0, 6.0, 5.0, 14
GLANCE_SKIP_SCORE = 0.6                # seen three times, to stay put from two places, once with this score: flown to without the check
SCAN_REPLAN_S = 4.0                    # a scan gives way to a new plan this often: visits and real stops come first
RESPIN_MAX = 4                         # spins repeated on the spot while the ground model is not yet trusted
FAR_VIEW_M = (25.0, 30.0)              # view rings of last resort: the laser places him from there
FLAT_LOOK_DEG = float(os.environ.get("TF_FLAT_LOOK_DEG", "28"))    # a look flatter than this takes his spot from the laser
FLAT_LAST_DEG = 18.0                   # ... and one flatter than this is not reported on the ground model at all, unless it is his last try
PROBE_TOL_DEG, PROBE_AIM_S, PROBE_AGREE_M, PROBE_GIVE_UP_S, PROBE_NEAR_M = 0.8, 1.1, 0.45, 8.0, 3.5
HOME_BASE_S, HOME_S_PER_M = 25.0, 0.25    # measured on 640 flights: turn-to-landing = 12.2 s + 0.24 s per metre of way, worst 6 s over that
RESERVE_S = 28.0                       # hover at the dock, press return, descend and land: measured about 19 s
MAX_REPORTS = 20

# state indices (Sentinel's layout)
POS, HEAD, TILT, TIME_LEFT = slice(0, 3), 3, 8, 9
LASER_RANGE, LASER_POINT, LASER_STATUS = 14, slice(15, 18), 18
GROUND_D, GROUND_OK, EDGE_D, PHASE, FRAME_AGE, ZOOM_LENS, ZOOM_AGE, ZOOMS_LEFT = 19, 20, 21, 23, 25, 28, 29, 30
DOCKED, TAKING_OFF, FLYING = 0, 1, 2
CAMERA_FEED, I_THERMAL = 24, 5
I_TURN, I_GIMBAL, I_ZOOM, I_LENS, I_ZCX, I_ZCY, I_ZW, I_ZH = 3, 4, 8, 9, 10, 11, 12, 13
I_REPORT, I_CLASS, I_IMAGE, I_CX, I_CY, I_W, I_H = 14, 15, 16, 17, 18, 19, 20
I_TAKE_OFF, I_RETURN, I_EAST, I_NORTH = 21, 22, 24, 25
SITE_FENCE_COUNT, SITE_FENCE, SITE_CROWN_COUNT, SITE_CROWNS = 0, 1, 258, 259


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def box_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def inside(poly, pts):
    a, b = poly, np.roll(poly, -1, axis=0)
    x, y = pts[:, :1], pts[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1]))
    return cross.sum(axis=1) % 2 == 1


def edge_distance(poly, pts):
    ab = np.roll(poly, -1, axis=0) - poly
    rel = pts[:, None, :] - poly[None, :, :]
    t = np.clip((rel * ab).sum(axis=2) / (ab * ab).sum(axis=1), 0.0, 1.0)
    return np.min(np.linalg.norm(rel - t[..., None] * ab, axis=2), axis=1)


def disk(r_cells):
    k = int(math.ceil(r_cells))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return (xx * xx + yy * yy) <= r_cells * r_cells


class Detector:
    """The Sentinel close-up detector (CenterNet style, split into a fast body and peak heads)."""

    def __init__(self, path):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = 2
        o.inter_op_num_threads = 1
        o.add_session_config_entry("session.intra_op.allow_spinning", "0")
        path = Path(path)
        self.fast = ort.InferenceSession(str(path.with_name(path.stem + "_fast.onnx")), o, providers=["CPUExecutionProvider"])
        with np.load(path.with_name(path.stem + "_heads.npz")) as z:
            self.heads = {h: (z[h + "_w3"].reshape(z[h + "_w3"].shape[0], -1), z[h + "_b3"], z[h + "_w1"], z[h + "_b1"])
                          for h in ("size", "offset")}

    def __call__(self, image, thr, k=40, stride=4):
        x = np.ascontiguousarray(np.asarray(image, np.float32).transpose(2, 0, 1)[None]) - 0.5
        heat, feat = self.fast.run(None, {"image": x})
        heat, feat = heat[0], feat[0]
        prob = 1.0 / (1.0 + np.exp(-np.clip(heat, -60, 60)))
        padded = np.pad(prob, ((0, 0), (1, 1), (1, 1)))
        c, h, w = prob.shape
        pooled = np.max(np.stack([padded[:, dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]), axis=0)
        peaks = np.where(pooled == prob, prob, 0.0).reshape(-1)
        order = np.argpartition(-peaks, min(k, peaks.size - 1))[:k]
        order = order[peaks[order] >= thr]
        order = order[np.argsort(-peaks[order])]
        cls, pix = order // (h * w), order % (h * w)
        ys, xs = pix // w, pix % w
        fp = np.pad(feat, ((0, 0), (1, 1), (1, 1)))
        patches = np.stack([fp[:, y:y + 3, x:x + 3].reshape(-1) for y, x in zip(ys, xs)]) if len(pix) else np.zeros((0, feat.shape[0] * 9), np.float32)
        out = {}
        for name, (w3, b3, w1, b1) in self.heads.items():
            out[name] = np.maximum(patches @ w3.T + b3, 0.0) @ w1.T + b1
        size, off = out["size"], out["offset"]
        cx, cy = (xs + off[:, 0]) * stride, (ys + off[:, 1]) * stride
        bw, bh = np.exp(size[:, 0]), np.exp(size[:, 1])
        return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1), peaks[order], cls


class FullDetector:
    """A centre-point detector run whole (heat, size and offset maps): the thermal feed, 512 x 640 x 1."""

    def __init__(self, path):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = 2
        o.inter_op_num_threads = 1
        o.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])

    def __call__(self, image, thr, k=40, stride=4):
        x = np.asarray(image, np.float32)
        if x.ndim == 2:
            x = x[..., None]
        heat, size, off = self.s.run(None, {"image": np.ascontiguousarray(x.transpose(2, 0, 1)[None]) - 0.5})
        prob = 1.0 / (1.0 + np.exp(-np.clip(heat[0][0], -60, 60)))          # the person channel
        h, w = prob.shape
        pad = np.pad(prob, 1)
        pooled = np.max(np.stack([pad[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]), axis=0)
        peaks = np.where(pooled == prob, prob, 0.0).reshape(-1)
        order = np.argpartition(-peaks, min(k, peaks.size - 1))[:k]
        order = order[peaks[order] >= thr]
        ys, xs = order // w, order % w
        cx = (xs + off[0][0].reshape(-1)[order]) * stride
        cy = (ys + off[0][1].reshape(-1)[order]) * stride
        bw, bh = np.exp(size[0][0].reshape(-1)[order]), np.exp(size[0][1].reshape(-1)[order])
        return [(float(a - c / 2), float(b - d / 2), float(a + c / 2), float(b + d / 2), float(sc))
                for a, b, c, d, sc in zip(cx, cy, bw, bh, peaks[order])]


class Maps:
    """The field on a GRID_M grid in the dock frame: inside, flyable, cleared, covered, people."""

    def __init__(self, edge, crowns):
        self.edge, self.crowns = edge, crowns
        self.low = edge.min(0) - 3.0
        n = np.ceil((edge.max(0) + 3.0 - self.low) / GRID_M).astype(int) + 1
        self.shape = (int(n[0]), int(n[1]))
        xs, ys = np.meshgrid(self.low[0] + GRID_M * np.arange(n[0]), self.low[1] + GRID_M * np.arange(n[1]), indexing="ij")
        self.xy = np.stack([xs, ys], -1)                       # [i, j] -> (east, north)
        pts = self.xy.reshape(-1, 2)
        ins = inside(edge, pts)
        ed = np.where(ins, edge_distance(edge, pts), -1.0)
        crown_gap = np.full(len(pts), np.inf)
        for e, n_, r in crowns:
            crown_gap = np.minimum(crown_gap, np.hypot(pts[:, 0] - e, pts[:, 1] - n_) - r)
        self.inside = ins.reshape(self.shape)
        self.edge_abs = edge_distance(edge, pts).reshape(self.shape)      # distance to the edge line, inside or outside
        self.edge_d = ed.reshape(self.shape)
        self.crown_gap = crown_gap.reshape(self.shape)
        self.flyable = (self.edge_d >= EDGE_CLEAR_M) & (self.crown_gap >= CROWN_CLEAR_M)
        self.hard = (self.edge_d >= HARD_EDGE_M) & (self.crown_gap >= HARD_CROWN_M)
        at_dock = np.hypot(self.xy[..., 0], self.xy[..., 1]) <= 1.5      # the dock stands 5 m inside the edge, 3.4 m off crowns
        self.flyable |= at_dock
        self.hard |= at_dock
        # where a person can stand: 1.5 m inside the edge, 0.5 m off crowns (with slack for our errors)
        self.can_hold = (self.edge_d >= 0.8) & (self.crown_gap >= 0.0)
        # what the scoring counts: 2 m cells inside the edge outside crowns (approximated on this grid)
        self.scored = self.inside & (self.crown_gap >= 0.0)
        self.cleared = ~self.can_hold                          # ground nobody can stand on needs no look
        self.covered = ~self.scored
        d = np.hypot(self.xy[..., 0], self.xy[..., 1])
        self.cleared |= d <= DOCK_CLEARED_M
        self.people = []                                       # [x, y]
        self.shaded = np.zeros(self.shape, bool)               # seen in colour, in shade: waits for a thermal look
        self.failed = np.zeros(self.shape, bool)               # a stop was to show these and did not: left to close-ups
        self.lum = np.full(self.shape, np.inf)                 # brightness a cell was checked at by colour alone
        self.ground = np.full(self.shape, UNKNOWN_RISE_M)
        self.version = 0
        self._allowed = None
        self.safe_k = disk(SAFE_R_M / GRID_M)
        self.person_k = disk(PERSON_CLEAR_M / GRID_M)
        self.home_d = self.spread(self.flyable, self.cell(np.zeros(2)))   # the way home in metres from every flyable cell
        self.home_dp = None                                    # the same round the people found; worked out when they change

    def plausible(self, xy):
        """Whether a person can stand at a spot by the field's own rules: inside the edge, not within reach of the dock,
        not deep inside a crown circle (position errors of about a metre allowed for)."""
        c = self.cell(xy)
        return (bool(self.inside[c]) or self.edge_abs[c] <= 1.0) and self.crown_gap[c] >= -CROWN_INSIDE_M \
            and float(np.hypot(xy[0], xy[1])) >= DOCK_KEEP_M

    def stand_spot(self, xy):
        """The nearest spot where a person can stand, for a sighting that fell where nobody can (a position error near a
        crown or the edge); None when it is farther than SNAP_M."""
        if self.plausible(xy):
            return np.asarray(xy, float)
        ok = self.inside & (self.edge_d >= 1.0) & (self.crown_gap >= 0.3)
        ok &= np.hypot(self.xy[..., 0], self.xy[..., 1]) >= DOCK_KEEP_M + 0.5
        d = np.where(ok, np.hypot(self.xy[..., 0] - xy[0], self.xy[..., 1] - xy[1]), np.inf)
        k = np.unravel_index(int(np.argmin(d)), d.shape)
        return self.xy[k].copy() if d[k] <= SNAP_M else None

    def cell(self, p):
        i, j = np.round((np.asarray(p, float) - self.low) / GRID_M).astype(int)
        return int(np.clip(i, 0, self.shape[0] - 1)), int(np.clip(j, 0, self.shape[1] - 1))

    def point(self, c):
        return self.low + GRID_M * np.asarray(c, float)

    def keep_off(self):
        """Per cell: how far across the ground the drone must stay from a person standing there, from the 5 m rule in 3D
        with the drone at cruise height over the dock and the ground where it is (0 where the drone clears his head by
        5.5 m)."""
        gap = (CRUISE_M - 0.3) - self.ground - PERSON_TOP_M
        r = np.sqrt(np.maximum(SAFE_R_M ** 2 - np.maximum(gap, 0.0) ** 2, 0.0))
        return np.where(r > 0, r + BODY_M, 0.0)

    def allowed(self):
        if self._allowed is None:
            r = self.keep_off()
            danger = np.zeros(self.shape, bool)
            unseen = ~self.cleared
            ppl = np.zeros(self.shape, bool)
            for x, y in self.people:
                ppl[self.cell((x, y))] = True
            for lo in np.arange(0.5, 8.0, 1.0):               # cells grouped by their keep-off radius, one dilation each
                band = (r >= lo) & (r < lo + 1.0)
                if not band.any():
                    continue
                k = disk((lo + 1.0) / GRID_M)
                for src in (unseen & band, ppl & band):
                    if src.any():
                        danger |= binary_dilation(src, structure=k)
            self._allowed = self.flyable & ~danger
        return self._allowed

    def set_ground(self, plane, ok):
        g = plane[0] + plane[1] * self.xy[..., 0] + plane[2] * self.xy[..., 1]
        g = g if ok else np.maximum(g, UNKNOWN_RISE_M)
        if float(np.max(np.abs(g - self.ground))) < 0.15:
            return                                         # nothing a route depends on has moved
        self.ground = g
        self.home_dp = None
        self.changed()

    def changed(self):
        self._allowed = None
        self.version += 1

    def holds(self, p):
        return bool(self.allowed()[self.cell(p)])

    def line_free(self, a, b, mask=None, skip=0.0):
        mask = self.allowed() if mask is None else mask
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.hypot(*(b - a)) / 0.5) + 2)
        for p in np.linspace(a, b, n):
            if skip and np.hypot(*(p - a)) < skip:
                continue
            if not mask[self.cell(p)]:
                return False
        return True

    def path(self, a, b, mask=None):
        """Points from a to b over the mask (allowed by default), pulled straight; None when cut off."""
        mask = self.allowed() if mask is None else mask
        a, b = np.asarray(a, float), np.asarray(b, float)
        if self.line_free(a, b, mask, skip=1.5):
            return [b]
        start, goal = self.cell(a), self.cell(b)
        if not mask[goal]:
            return None
        if not mask[start]:
            cells = np.argwhere(mask)
            if not len(cells):
                return None
            start = tuple(int(v) for v in cells[np.argmin(np.hypot(*(cells - np.array(start)).T))])
        best, came, todo = {start: 0.0}, {}, [(0.0, start)]
        while todo:
            _, c = heapq.heappop(todo)
            if c == goal:
                break
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if not di and not dj:
                        continue
                    n = (c[0] + di, c[1] + dj)
                    if not (0 <= n[0] < self.shape[0] and 0 <= n[1] < self.shape[1]) or not mask[n]:
                        continue
                    g = best[c] + (1.4142 if di and dj else 1.0)
                    if g < best.get(n, math.inf):
                        best[n], came[n] = g, c
                        heapq.heappush(todo, (g + math.hypot(goal[0] - n[0], goal[1] - n[1]), n))
        if goal not in best:
            return None
        cells = [goal]
        while cells[-1] != start:
            cells.append(came[cells[-1]])
        pts = [self.point(c) for c in cells[::-1]] + [b]
        out, here, k = [], a, 0
        while k < len(pts) - 1:
            far = k + 1
            for m in range(len(pts) - 1, k, -1):
                if self.line_free(here, pts[m], mask, skip=1.5 if not out else 0.0):
                    far = m
                    break
            out.append(pts[far]); here = pts[far]; k = far
        return out

    def reach(self, a):
        """Distances in metres over allowed ground from a to every cell (Dijkstra), inf where cut off; worked out once
        per start cell and map version (it is the costliest thing the patrol computes)."""
        key = (self.cell(a), self.version)
        if getattr(self, "_reach_key", None) == key:
            return self._reach
        self._reach_key, self._reach = key, self._reach_calc(a)
        return self._reach

    def _reach_calc(self, a):
        return self.spread(self.allowed(), self.cell(a))

    def spread(self, mask, start):
        """Metres over the mask from a start cell to every cell (Dijkstra on the grid), inf where cut off."""
        dist = np.full(self.shape, np.inf)
        if not mask[start]:
            cells = np.argwhere(mask)
            if not len(cells):
                return dist
            start = tuple(int(v) for v in cells[np.argmin(np.hypot(*(cells - np.array(start)).T))])
        dist[start] = 0.0
        todo = [(0.0, start)]
        while todo:
            d, c = heapq.heappop(todo)
            if d > dist[c]:
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if not di and not dj:
                        continue
                    n = (c[0] + di, c[1] + dj)
                    if 0 <= n[0] < self.shape[0] and 0 <= n[1] < self.shape[1] and mask[n]:
                        nd = d + (1.4142 if di and dj else 1.0) * GRID_M
                        if nd < dist[n]:
                            dist[n] = nd
                            heapq.heappush(todo, (nd, n))
        return dist


class DroneFlightController:
    def __init__(self):
        self.det = Detector(HERE / "day_zoom_so.onnx")
        self.det(np.zeros((480, 640, 3), np.float32), 0.9)          # first run is slow: done before the patrol
        self.tdet = FullDetector(HERE / "thermal_day.onnx")
        self.field = None                                  # (heights m, low corner, step, surveyed edge) of the published field
        try:
            z = np.load(HERE / FIELD_HEIGHTS)
            self.field = (z["h_cm"].astype(np.float32) / 100.0, z["low"].astype(float), float(z["step"]), z["outline"].astype(float))
        except Exception:
            self.field = None
        self.tdet(np.zeros((512, 640, 1), np.float32), 0.9)
        self.reset()

    def reset(self):
        self.maps = None
        self.held = np.zeros(ACTION_DIM, np.float32)
        self.hist, self.ground, self.plane, self.plane_ok = [], [], np.zeros(3), False
        self.hmap = None                                   # (the dock's place in the map frame, the dock origin's height there) once the field is recognised
        self.tracks = []
        self.mode, self.goal, self.leg, self.spin_left, self.target = "takeoff", None, None, 0, None
        self.last_age, self.last_zoom_age, self.zoom_wait, self.last_zoom_lens = None, None, None, 0
        self.pending, self.reports = None, 0
        self.truth = None
        self.log = []
        self.last_lp = None
        self._ground_set = False
        self._asked, self._answer = None, None
        self.wide_due = None
        self.kinds, self.bright, self.tspin, self.stops = [], 0.0, None, []
        self.scan, self.scan_done, self.zooms_left, self.scan_from = None, None, 80, []
        self.spin_at = np.zeros(2)
        self.respins, self.scan_since = 0, None
        self.glance = None
        self.sizes, self.last_size = [], None
        self.late_key, self.spent, self.last_t = None, {}, None
        self.dets, self.shadow, self.next_id, self.hints = [], None, 0, []
        self.small = []
        self.promise, self.promise_next, self.promise_live = None, None, False
        self.leg_ids, self.tspin_tilt = set(), TSPIN_TILT_DEG
        self._idle, self.time_left, self.bar_at = None, HORIZON_S, 0.0

    # ------------------------------------------------------------------ geometry
    def camera_at(self, t):
        hist = self.hist
        later = next((k for k in range(len(hist)) if hist[k][0] >= t), len(hist) - 1)
        a, b = hist[max(later - 1, 0)], hist[later]
        w = min(max((t - a[0]) / max(b[0] - a[0], 1e-6), 0.0), 1.0)
        pos = a[1] + w * (b[1] - a[1])
        head = a[2] + w * wrap(b[2] - a[2])
        tilt = a[3] + w * (b[3] - a[3])
        hd = math.radians(head)
        hv = np.array([math.sin(hd), math.cos(hd), 0.0])
        eye = pos + CAMERA_FORWARD_M * hv
        eye[2] -= CAMERA_BELOW_M
        tr = math.radians(tilt)
        f = math.cos(tr) * hv + math.sin(tr) * np.array([0, 0, 1.0])
        up = -math.sin(tr) * hv + math.cos(tr) * np.array([0, 0, 1.0])
        return eye, f, up, np.cross(f, up)

    @staticmethod
    def pixel_ray(cam, u, v, focal=WIDE_F, cu=320.0, cv=240.0):
        eye, f, up, right = cam
        r = f + ((u - cu) / focal) * right - ((v - cv) / focal) * up
        return r / np.linalg.norm(r)

    @staticmethod
    def project(cam, p, focal=WIDE_F):
        eye, f, up, right = cam
        d = np.asarray(p, float) - eye
        z = float(d @ f)
        if z < 0.5:
            return None
        return 320.0 + focal * float(d @ right) / z, 240.0 - focal * float(d @ up) / z

    def known_field(self, edge):
        """The dock's place on the published field, if the site map's edge is that field's surveyed edge: the same
        corners in the same order, moved by where the dock stands in this seed.  None for any other edge."""
        if self.field is None:
            return None
        H, low, step, outline = self.field
        k = len(edge)
        if k != len(outline):
            return None
        best = None
        for sh in range(k):
            d = outline - np.roll(edge, -sh, axis=0)
            off = np.median(d, axis=0)
            spread = float(np.abs(d - off).max())
            if best is None or spread < best[0]:
                best = (spread, off)
        if best[0] > FIELD_MATCH_M:
            return None
        dock = best[1]
        q = (dock - low) / step
        if not (0 <= q[0] < H.shape[0] - 1 and 0 <= q[1] < H.shape[1] - 1):
            return None
        self.hmap = (dock, 0.0)
        return dock, float(self.ground_at(np.zeros(2))) + DOCK_PAD_M

    def ground_at(self, xy):
        """Height of the ground at a place, in the dock's frame: from the published field's height map when the field
        was recognised, else from the plane fitted to the laser readings."""
        xy = np.asarray(xy, float)
        if self.hmap is not None:
            H, low, step, _ = self.field
            dock, z0 = self.hmap
            q0 = (xy[..., 0] + dock[0] - low[0]) / step
            q1 = (xy[..., 1] + dock[1] - low[1]) / step
            i0 = np.clip(np.floor(q0).astype(int), 0, H.shape[0] - 2)
            i1 = np.clip(np.floor(q1).astype(int), 0, H.shape[1] - 2)
            f0, f1 = np.clip(q0 - i0, 0.0, 1.0), np.clip(q1 - i1, 0.0, 1.0)
            return (H[i0, i1] * (1 - f0) * (1 - f1) + H[i0 + 1, i1] * f0 * (1 - f1)
                    + H[i0, i1 + 1] * (1 - f0) * f1 + H[i0 + 1, i1 + 1] * f0 * f1) - z0
        return self.plane[0] + self.plane[1] * xy[..., 0] + self.plane[2] * xy[..., 1]

    def fit_ground(self):
        g = np.asarray(self.ground[-600:], float)
        if len(g) < 5:
            return
        if self.hmap is not None:
            # the ground is known; the laser only checks it: readings that disagree on the whole mean another field
            if len(g) < 60 or abs(float(np.median(g[:, 2] - self.ground_at(g[:, :2])))) <= FIELD_DOUBT_M:
                return
            self.hmap, self.plane_ok, self._ground_set = None, False, False
        A = np.column_stack([np.ones(len(g)), g[:, 0], g[:, 1]])
        spread = np.ptp(g[:, :2], axis=0)
        lam = np.array([0.0, 1.0 if spread[0] < 8 else 1e-3, 1.0 if spread[1] < 8 else 1e-3]) * len(g) * 30.0
        coef = np.linalg.solve(A.T @ A + np.diag(lam), A.T @ g[:, 2])
        for _ in range(2):
            keep = np.abs(A @ coef - g[:, 2]) < 0.8
            if keep.sum() < 5:
                break
            coef = np.linalg.solve(A[keep].T @ A[keep] + np.diag(lam), A[keep].T @ g[keep, 2])
        self.plane = coef
        xy = g[:, :2] - g[:, :2].mean(0)
        self.plane_ok = len(g) >= 30 and float(np.linalg.svd(xy, compute_uv=False)[-1] / math.sqrt(len(g))) >= 2.0
        if self.maps is not None and (self.plane_ok or not self._ground_set):
            self._ground_set = True
            self.maps.set_ground(self.plane, self.plane_ok)

    def on_ground(self, eye, ray, lift=0.0):
        if self.hmap is not None:
            # along the line of sight until it passes under the ground (raised by lift), then between the two samples
            eye, ray = np.asarray(eye, float), np.asarray(ray, float)
            t = np.arange(0.0, 150.0, 0.5)
            p = eye[None, :] + t[:, None] * ray[None, :]
            d = p[:, 2] - (self.ground_at(p[:, :2]) + lift)
            if d[0] <= 0:
                return None
            under = np.nonzero(d <= 0)[0]
            if not len(under):
                return None
            k = int(under[0])
            tt = t[k - 1] + 0.5 * d[k - 1] / max(d[k - 1] - d[k], 1e-9)
            return eye + tt * ray
        a, b, c = self.plane
        a = a + lift
        denom = ray[2] - b * ray[0] - c * ray[1]
        if denom > -1e-3:
            return None
        t = (a + b * eye[0] + c * eye[1] - eye[2]) / denom
        return eye + t * ray if 0 < t < 150 else None

    # ------------------------------------------------------------------ what a frame saw
    def mark_seen(self, cam, image=None, thermal=False, zoom_focal=None):
        """Cells this main-camera frame saw closely enough to cover (the scoring rule) and to clear (a person there would
        have been found).  Ground behind a crown is not seen.  A colour frame clears sunlit ground only: ground it shows
        in shade is marked and waits for a thermal frame, which clears whatever it sees."""
        m = self.maps
        eye, f, up, right = cam
        focal, half_w, half_h = (THERMAL_F, 320.0, 256.0) if thermal else (zoom_focal or WIDE_F, 320.0, 240.0)
        g = self.ground_at(m.xy)
        pts = np.concatenate([m.xy, g[..., None]], -1) - eye
        depth = pts @ f
        ok = depth > 0.3
        sd = np.where(ok, depth, 1.0)
        u, v = (pts @ right) / sd * focal, (pts @ up) / sd * focal
        inf = ok & (np.abs(u) <= half_w) & (np.abs(v) <= half_h)
        dist = np.linalg.norm(pts, axis=-1)
        nrm = np.array([-self.plane[1], -self.plane[2], 1.0]); nrm /= np.linalg.norm(nrm)
        facing = -(pts @ nrm) / np.maximum(dist, 1e-6)
        size = np.sqrt((sd / focal) ** 2 * (sd / np.maximum(dist, 1e-6)) / np.maximum(facing, 1e-6))
        seen = inf & (facing > 0)
        if zoom_focal is None:
            m.covered |= seen & (size <= COVER_PIX_M * 0.95)   # close-ups do not count for coverage
        # checked means a person standing there would have been found: his head must be in the frame too
        top = pts + np.array([0.0, 0.0, BODY_TOP_M])
        dt = top @ f
        st = np.where(dt > 0.3, dt, 1.0)
        ut, vt = (top @ right) / st * focal, (top @ up) / st * focal
        whole = (dt > 0.3) & (np.abs(ut) <= half_w - 4.0) & (np.abs(u) <= half_w - 4.0) & (vt <= half_h - 4.0) & (v >= -half_h + 2.0)
        near = seen & whole & (size <= (THERMAL_CLEAR_PIX_M if thermal else (SCAN_PIX_M if zoom_focal else CLEAR_PIX_M))) & ~m.cleared
        if not near.any():
            return
        clear = self.sight_from(eye, m.xy[near])
        if not thermal and image is not None:
            lum = np.asarray(image, np.float32)[:480, :640].reshape(60, 8, 80, 8, -1).mean((1, 3, 4))    # 8 px blocks
            col = np.clip(((u[near] + 320.0) / 8.0).astype(int), 0, 79)
            row = np.clip(((240.0 - v[near]) / 8.0).astype(int), 0, 59)
            level = lum[row, col]
            if clear.sum() >= 20:
                self.bright = max(self.bright, float(np.percentile(level[clear], 90)))
            if self.bright > 0.08:
                dark = clear & (level < SHADE_REL * self.bright)
                cells = np.argwhere(near)
                m.shaded[cells[dark, 0], cells[dark, 1]] = True
                clear &= ~dark
            if self.bright > 1.15 * self.bar_at:
                # the first frames may have shown shade only and set the bar too low: ground colour alone checked at a
                # level that is shade by the bar of now goes back to waiting for a thermal look
                self.bar_at = self.bright
                back = m.cleared & m.can_hold & (m.lum < SHADE_REL * self.bright)
                if back.any():
                    m.cleared[back], m.shaded[back], m.lum[back] = False, True, np.inf
                    m.changed()
            cells = np.argwhere(near)[clear]
            m.lum[cells[:, 0], cells[:, 1]] = level[clear]
        cells = np.argwhere(near)[clear]
        if len(cells):
            m.cleared[cells[:, 0], cells[:, 1]] = True
            m.shaded[cells[:, 0], cells[:, 1]] = False
            if thermal or zoom_focal is not None or image is None:
                m.lum[cells[:, 0], cells[:, 1]] = np.inf       # not by colour alone
            m.changed()

    def sight_from(self, eye, pts):
        """Which ground points a camera at eye sees past the crowns: a crown circle on the line hides the point, unless the
        line is already lower than the crown's underside there."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        a = np.asarray(eye, float)[:2]
        ok = np.ones(len(pts), bool)
        ab = pts - a[None, :]
        L2 = np.maximum((ab * ab).sum(1), 1e-9)
        above = max(float(np.asarray(eye, float)[2]) - float(np.mean(self.ground_at(pts))) if len(pts) else CRUISE_M, 3.0)
        # the line to the top of a body there (SEEN_TOP_M of him must show), not to the ground under him
        end = np.clip(1.0 - (UNDER_CROWN_M - SEEN_TOP_M) / max(above - SEEN_TOP_M, 1.0), 0.0, 1.0)
        for e, n, r in self.maps.crowns:
            c = np.array([e, n])
            t = np.clip(((c[None, :] - a[None, :]) * ab).sum(1) / L2, 0.0, end)
            ok &= np.hypot(*(a[None, :] + t[:, None] * ab - c[None, :]).T) >= 0.9 * r
        return ok

    # ------------------------------------------------------------------ eyes
    def boxes(self, kind, image):
        """Person boxes (x0, y0, x1, y1, score) in a wide ('feed') or zoom image."""
        if self.truth is not None:
            rows = self.truth.get("feed" if kind == "thermal" else kind, [])
            return [(r[0], r[1], r[2], r[3], 0.99) for r in rows if r[4] >= ORACLE_MIN_PIXELS[kind]]
        if kind == "thermal":
            return self.tdet(image, DET_THR)
        b, s, c = self.det(image, DET_THR)
        return [(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(sc)) for x, sc, cl in zip(b, s, c) if int(cl) == 0]

    @staticmethod
    def foot(x0, y0, x1, y1):
        """The pixel over his spot (the centre of his body seen from above): between the feet of someone standing, the
        middle of someone lying flat, low in the box of someone crouching."""
        h, w = y1 - y0, x1 - x0
        if h > 1.25 * w:
            return (x0 + x1) / 2, y1 - 0.06 * h
        if w > 1.3 * h:
            return (x0 + x1) / 2, (y0 + y1) / 2 + 0.1 * h
        return (x0 + x1) / 2, y1 - 0.25 * h

    def track_of(self, p, make=True):
        near = min(self.tracks, key=lambda tr: np.hypot(*(tr["xy"] - p)), default=None)
        if near is not None and np.hypot(*(near["xy"] - p)) < SAME_M:
            return near
        if not make:
            return None
        tr = {"xy": np.asarray(p, float).copy(), "pts": [], "n": 0, "reported": False, "tried": 0, "laser": None,
              "empty": 0, "dead": False, "views": [], "id": self.next_id, "obs": [], "warm": False,
              "born": np.asarray(self.hist[-1][1][:2], float).copy() if self.hist else np.zeros(2),
              "born_t": float(self.hist[-1][0]) if self.hist else 0.0}
        self.next_id += 1
        self.tracks.append(tr)
        return tr

    def ghost(self, tr):
        """A sighting that is most likely a reported man seen again: made GHOST_N times or fewer, GHOST_NEAR_M from his
        pinned spot, never warm and never boxed by a close-up of its own.  In A15's flights 314 visits went to such spots,
        2.4-3.2 m from a reported man, and every one found nobody; it is visited only when nobody else is waiting."""
        if tr["reported"] or tr.get("warm") or tr.get("zoomed") or tr["n"] > GHOST_N:
            return False
        return any(o is not tr and o["reported"] and np.hypot(*(o["xy"] - tr["xy"])) < GHOST_NEAR_M for o in self.tracks)

    def proven(self, tr):
        """Surely not the drone's own shadow: a warm body in a thermal frame, or seen at one spot from two places."""
        if tr.get("warm") or tr["reported"] or tr.get("zoomed"):
            return True                                    # a close-up showed him: the shadow is not what a close-up is aimed at
        obs = tr.get("obs", [])
        for i in range(len(obs)):
            for j in range(i):
                if (np.hypot(*(obs[i][0] - obs[j][0])) >= SHADOW_MOVE_M
                        and np.hypot(*(obs[i][2] - obs[j][2])) <= STATIC_SAME_M):
                    return True
        return False

    def own_shadow(self, c, z, q, t):
        """Whether a box in a colour frame is the drone's own shadow, which the detector takes for a person.  People
        never move; the shadow keeps its place from the drone: so many metres down-sun for every metre of height above
        the ground it falls on.  A box at the same offset as boxes seen from SHADOW_NEED other places, none of them a
        thing that stayed where it was, fixes that offset; from then on a box at the offset is the shadow."""
        h = max(float(z) - float(self.ground_at(q)), 1.0)
        k = (q - c) / h
        if self.shadow is not None:
            off = h * self.shadow
            if float(np.hypot(*(q - c - off))) <= max(1.2, 0.1 * float(np.hypot(*off))):
                self.shadow = 0.9 * self.shadow + 0.1 * k          # the offset follows the ground as it is measured better
                return True
        hit = False
        if self.shadow is None and self.hints:
            # a visit found nobody where a box had been seen from one place only: that box was most likely the shadow.
            # A box at the same offset from another place, where nothing was seen to stay, settles it.
            D = np.array(self.dets) if self.dets else np.zeros((0, 8))
            stays = bool(len(D) and ((np.hypot(D[:, 1] - c[0], D[:, 2] - c[1]) >= SHADOW_MOVE_M)
                                     & (np.hypot(D[:, 4] - q[0], D[:, 5] - q[1]) <= STATIC_SAME_M)).any())
            for hc, hk in self.hints:
                if not stays and np.hypot(*(hc - c)) >= SHADOW_MOVE_M and float(np.hypot(*(hk - k))) * h <= SHADOW_SAME_M:
                    self.shadow = 0.5 * (hk + k)
                    self.log.append(("SHADOW", round(float(t), 1), round(float(self.shadow[0]), 2), round(float(self.shadow[1]), 2), "hint"))
                    self.purge_shadow()
                    return True
        if self.dets:
            D = np.array(self.dets)                               # t, cx, cy, h, qx, qy, kx, ky
            moved = np.hypot(D[:, 1] - c[0], D[:, 2] - c[1]) >= SHADOW_MOVE_M
            stays = bool((moved & (np.hypot(D[:, 4] - q[0], D[:, 5] - q[1]) <= STATIC_SAME_M)).any())
            same = moved & (np.hypot(D[:, 6] - k[0], D[:, 7] - k[1]) * h <= SHADOW_SAME_M)
            if not stays and self.shadow is None and same.any():
                places, ks = [], []
                for e in D[same]:
                    # an earlier box that was itself seen to stay put from somewhere else is a thing, not the shadow
                    far = np.hypot(D[:, 1] - e[1], D[:, 2] - e[2]) >= SHADOW_MOVE_M
                    if (far & (np.hypot(D[:, 4] - e[4], D[:, 5] - e[5]) <= STATIC_SAME_M)).any():
                        continue
                    if all(np.hypot(e[1] - pl[0], e[2] - pl[1]) >= SHADOW_MOVE_M for pl in places):
                        places.append(e[1:3])
                    ks.append(e[6:8])
                if len(places) >= SHADOW_NEED:
                    self.shadow = np.mean(np.array(ks + [k]), axis=0)
                    self.log.append(("SHADOW", round(float(t), 1), round(float(self.shadow[0]), 2), round(float(self.shadow[1]), 2)))
                    self.purge_shadow()
                    hit = True
        self.dets = (self.dets + [(float(t), float(c[0]), float(c[1]), h, float(q[0]), float(q[1]), float(k[0]), float(k[1]))])[-400:]
        return hit

    def purge_shadow(self):
        """Tracks that were only ever seen where the drone's shadow fell at that moment were the shadow."""
        gone = []
        for tr in self.tracks:
            obs = tr.get("obs", [])
            if tr["reported"] or tr.get("warm") or not obs or tr.get("zoomed"):
                continue
            if all(np.hypot(*(o[2] - o[0] - o[1] * self.shadow)) <= max(1.5, 0.12 * o[1] * float(np.hypot(*self.shadow))) for o in obs):
                gone.append(tr)
        for tr in gone:
            tr["dead"] = True
        keep = self.target.get("track") if self.target is not None else None
        self.tracks = [tr for tr in self.tracks if not any(tr is g for g in gone) or tr is keep]
        if gone:
            self.sync_people()

    def look_wide(self, image, cam, t=0.0):
        m = self.maps
        c = np.asarray(cam[0][:2], float).copy()
        for x0, y0, x1, y1, s in self.boxes("feed", image):
            mid = self.on_ground(cam[0], self.pixel_ray(cam, (x0 + x1) / 2, (y0 + y1) / 2))    # a shadow lies flat: its middle
            if mid is not None and self.truth is None and self.own_shadow(c, float(cam[0][2]), np.asarray(mid[:2], float), t):
                continue                                   # the drone's own shadow
            size = max(x1 - x0, y1 - y0)
            weak = WIDE_WEAK_SCORE <= s < WIDE_MIN_SCORE and size >= WIDE_MIN_PX
            if (s < WIDE_MIN_SCORE and not weak) or size < WIDE_SMALL_PX:
                continue
            ray = self.pixel_ray(cam, *self.foot(x0, y0, x1, y1))
            p = self.on_ground(cam[0], ray)
            if p is None:
                continue
            if np.hypot(p[0], p[1]) < DOCK_KEEP_M:
                continue                                   # the dock, or its shadow
            spot = m.stand_spot(p[:2])
            if spot is None:
                continue                                   # deep in a tree, or off the field
            seen = [(cam[0].copy(), ray, size ** 2,
                     None if mid is None else (c, max(float(cam[0][2]) - float(self.ground_at(mid[:2])), 1.0), np.asarray(mid[:2], float)))]
            if size < WIDE_MIN_PX and self.track_of(spot, make=False) is None:
                # too small to trust alone (someone crouching far off is 15-25 px): kept, and SMALL_NEED of them on one
                # spot make the sighting
                self.small = [e for e in self.small if t - e[0] <= SMALL_KEEP_S][-80:] + [(t, np.asarray(spot, float).copy(), seen[0])]
                same = [e for e in self.small if np.hypot(*(e[1] - spot)) <= SMALL_SAME_M]
                if len(same) < SMALL_NEED:
                    continue
                spot = np.mean(np.array([e[1] for e in same]), axis=0)
                seen = [e[2] for e in same]
                self.small = [e for e in self.small if not any(e is x for x in same)]
            fresh = self.track_of(spot, make=False) is None
            tr = self.track_of(spot)
            if fresh and weak:
                tr["weak"] = True                          # only a close-up can make him someone to visit
            elif not weak:
                tr["weak"] = False
            tr["best"] = max(tr.get("best", 0.0), float(s))
            for eye, r_, w, ob in seen:
                tr["rays"] = (tr.get("rays", []) + [(eye, r_, w)])[-10:]
                tr["n"] += 1
                if ob is not None:
                    obs = tr["obs"] + [ob]
                    tr["obs"] = obs if len(obs) <= 14 else obs[:4] + obs[-10:]
            self.place(tr)
        self.merge()
        self.sync_people()

    def place(self, tr):
        """A track's spot: the laser on him when there is one, else the weighted mean of his rays on the current plane."""
        if tr.get("pin") is not None:
            tr["xy"] = np.asarray(tr["pin"], float)            # reported: he stays where the report put him
            return
        pts, ws = [], []
        for eye, ray, w in tr.get("rays", []):
            q = self.on_ground(eye, ray)
            if q is not None:
                pts.append(q[:2]); ws.append(w)
        if pts:
            P, W = np.array(pts), np.array(ws)
            tr["xy"] = (P * W[:, None]).sum(0) / W.sum()

    def relocate(self):
        if self.tracks:
            for tr in self.tracks:
                self.place(tr)
            self.merge()
            self.sync_people()

    def merge(self):
        """Two tracks whose spots meet are one person (people stand 3 m apart); a track that has moved to where nobody
        can stand is dropped unless it was reported."""
        for tr in self.tracks:
            if not tr["reported"] and not self.maps.plausible(tr["xy"]):
                spot = self.maps.stand_spot(tr["xy"])
                if spot is not None and np.hypot(tr["xy"][0], tr["xy"][1]) >= DOCK_KEEP_M:
                    tr["xy"] = spot
        self.tracks = [tr for tr in self.tracks if tr["reported"] or self.maps.plausible(tr["xy"])
                       or (self.target is not None and self.target.get("track") is tr)]
        k = 0
        while k < len(self.tracks):
            a = self.tracks[k]
            other = next((b for b in self.tracks[k + 1:]
                          if np.hypot(*(a["xy"] - b["xy"])) < (SAME_REPORTED_M if a["reported"] or b["reported"] else SAME_M)
                          and not (a["reported"] and b["reported"])), None)
            if other is None:
                k += 1
                continue
            if a["reported"] and other["reported"]:
                k += 1                                     # two reports are two people
                continue
            keep, drop = (a, other) if (a["reported"] or a["n"] >= other["n"]) and not other["reported"] else (other, a)
            keep["rays"] = (keep.get("rays", []) + drop.get("rays", []))[-12:]
            keep["n"] += drop["n"]
            keep["reported"] = keep["reported"] or drop["reported"]
            keep["empty"], keep["views"] = max(keep.get("empty", 0), drop.get("empty", 0)), keep.get("views", []) + drop.get("views", [])
            keep["views_saw"] = keep.get("views_saw", []) + drop.get("views_saw", [])
            keep["dead"] = keep.get("dead", False) and drop.get("dead", False)
            obs = keep.get("obs", []) + drop.get("obs", [])
            keep["obs"] = obs if len(obs) <= 14 else obs[:4] + obs[-10:]
            keep["warm"] = keep.get("warm", False) or drop.get("warm", False)
            keep["best"] = max(keep.get("best", 0.0), drop.get("best", 0.0))
            keep["weak"] = bool(keep.get("weak")) and bool(drop.get("weak"))
            keep["glance_empty"] = min(keep.get("glance_empty", 0), drop.get("glance_empty", 0))
            keep["zoomed"] = keep.get("zoomed", False) or drop.get("zoomed", False)
            if self.target is not None and self.target.get("track") is drop:
                self.target["track"] = keep
            if self.zoom_wait is not None and self.zoom_wait["track"] is drop:
                self.zoom_wait["track"] = keep             # the close-up on its way shows the same man
            keep["tried"] = max(keep["tried"], drop["tried"])
            self.tracks = [x for x in self.tracks if x is not drop]
            self.place(keep)

    def sync_people(self):
        people = [tr["xy"].tolist() for tr in self.tracks]
        cells = sorted(self.maps.cell(p) for p in people)
        if cells == getattr(self.maps, "people_cells", None):
            self.maps.people = people
            return                                         # nobody moved to another cell: routes stay as they are
        self.maps.people, self.maps.people_cells = people, cells
        self.maps.home_dp = None
        self.maps.changed()

    # ------------------------------------------------------------------ act
    def act(self, observation):
        state = np.asarray(observation["state"], np.float64)
        key = state.tobytes()
        if key == self._asked:
            return self._answer.copy()                     # asked again with the same observation: the same answer
        a = self._act(observation, state)
        self._asked, self._answer = key, a.copy()
        return a

    def _act(self, observation, state):
        self.truth = observation.get("_truth")                     # local tests only
        t = HORIZON_S - float(state[TIME_LEFT])
        self.zooms_left = int(round(float(state[ZOOMS_LEFT])))
        pos = state[POS].copy()
        self.hist = (self.hist + [(t, pos, float(state[HEAD]), float(state[TILT]))])[-40:]
        if self.maps is None:
            self.read_map(observation.get("site_map"))
        if state[GROUND_OK] > 0.5 and 0.5 < state[GROUND_D] < 16.0 and np.hypot(*pos[:2]) > 2.5:
            self.ground.append((pos[0], pos[1], pos[2] - state[GROUND_D]))
            if len(self.ground) % 10 == 0:
                self.fit_ground()
        lp = state[LASER_POINT].copy()
        if (int(round(state[LASER_STATUS])) == 0 and state[LASER_RANGE] > 0 and (self.last_lp is None or np.linalg.norm(lp - self.last_lp) > 1e-6)
                and self.laser_on_ground(lp)):
            # the laser lands wherever the camera looks: only a reading on open field ground is a ground sample
            for _ in range(5):
                self.ground.append((lp[0], lp[1], lp[2]))
            self.fit_ground()
            self.relocate()
        self.last_lp = lp
        a = np.zeros(ACTION_DIM, np.float32)
        self.tilt(a, SPIN_TILT_DEG)
        phase = int(round(state[PHASE]))
        age, zage = float(state[FRAME_AGE]), float(state[ZOOM_AGE])
        fresh = self.last_age is not None and age < self.last_age - 1e-6
        zlens = int(round(state[ZOOM_LENS]))
        zfresh = self.last_zoom_age is not None and zlens > 0 and (zage < self.last_zoom_age - 1e-6 or self.last_zoom_lens == 0)
        self.last_age, self.last_zoom_age, self.last_zoom_lens = age, zage, zlens
        if self.zoom_wait is not None and t - self.zoom_wait["t"] > 2.0:
            self.zoom_wait = None
        if phase == DOCKED:
            self.press(a, I_TAKE_OFF)
        elif phase == FLYING and self.maps is not None:
            zoom_now = zfresh and self.zoom_wait is not None
            if fresh and self.pending is not None and self.pending[0] == "feed":
                self.pending = None                        # a box on the main frame is only good while that frame is shown
            if fresh and len(self.hist) > 2:               # one detector pass per decision: the close-up goes first
                self.wide_due = (self.camera_at(t - age), int(round(state[CAMERA_FEED])), t - age)
            if zoom_now:
                self.look_zoom(np.asarray(observation["zoom"], np.float32), state)
            elif self.wide_due is not None:
                (cam, feed, t_frame), self.wide_due = self.wide_due, None     # the frame is shown for five decisions
                if feed == 1:
                    image = np.asarray(observation["thermal"], np.float32)
                    if self.mode == "confirm":
                        self.look_thermal(image, cam, t_frame)
                    else:
                        self.mark_seen(cam, thermal=True)
                        self.look_thermal_wide(image, cam)
                else:
                    image = np.asarray(observation["rgb"], np.float32)
                    if image.max() > 0 or self.truth is not None:
                        self.mark_seen(cam, image)
                        self.look_wide(image, cam, t_frame)
            self.decide(state, a, t)
            self.govern(state, a)
            self.report(a)
        self.held = a
        return a

    def laser_on_ground(self, lp):
        """A laser point that lies on open ground of the field: inside the edge, clear of every crown circle, not on
        the dock and not on someone found."""
        m = self.maps
        if m is None or np.hypot(lp[0], lp[1]) < 2.5:
            return False
        if not (m.low[0] <= lp[0] <= m.low[0] + GRID_M * (m.shape[0] - 1) and m.low[1] <= lp[1] <= m.low[1] + GRID_M * (m.shape[1] - 1)):
            return False
        c = m.cell(lp[:2])
        if not m.inside[c] or m.crown_gap[c] <= 0.5:
            return False
        return all(np.hypot(*(tr["xy"] - lp[:2])) > 1.5 for tr in self.tracks)

    def read_map(self, site_map):
        if site_map is None:
            return
        s = np.asarray(site_map, np.float64)
        k = int(round(s[SITE_FENCE_COUNT]))
        if k < 3:
            return
        edge = s[SITE_FENCE:SITE_FENCE + 2 * k].reshape(k, 2)
        rows = s[SITE_CROWNS:SITE_CROWNS + 5 * int(round(s[SITE_CROWN_COUNT]))].reshape(-1, 5)
        crowns = np.column_stack([rows[:, 0], rows[:, 1], rows[:, 2] / 2.0])
        self.maps = Maps(edge, crowns)
        self.scan_done = np.zeros(self.maps.shape, bool)
        self.hmap = None
        known = self.known_field(edge)
        self.hmap = known
        if known is not None:
            # the published field: its ground is known from this first frame, so nothing waits for a fitted plane
            g = self.ground_at(self.maps.xy)
            ins = self.maps.inside
            A = np.column_stack([np.ones(int(ins.sum())), self.maps.xy[ins][:, 0], self.maps.xy[ins][:, 1]])
            self.plane = np.linalg.lstsq(A, g[ins], rcond=None)[0]         # its mean slope, for the code that wants a normal
            self.plane_ok, self._ground_set = True, True
            self.maps.ground = np.asarray(g, float)
            self.maps.home_dp = None
            self.maps.changed()

    def press(self, a, i):
        a[i] = 0.0 if self.held[i] > 0.5 else 1.0

    @staticmethod
    def tilt(a, deg):
        a[I_GIMBAL] = np.clip((deg - GIMBAL_RANGE[0]) / (GIMBAL_RANGE[1] - GIMBAL_RANGE[0]) * 2 - 1, -1, 1)

    # ------------------------------------------------------------------ decide
    def home_needed(self, xy):
        """Seconds to keep for flying home and landing from here: by the length of the way over flyable ground."""
        m = self.maps
        if m is None:
            return HOME_BASE_S + HOME_S_PER_M * 1.4 * float(np.hypot(*xy))
        if m.home_dp is None:
            m.home_dp = m.spread(self.home_mask(), m.cell(np.zeros(2)))     # the way that will be flown: round the people found
        c = m.cell(xy)
        way = float(m.home_dp[c])
        if not math.isfinite(way):
            way = float(m.home_d[c])                       # cut off by someone: the way over the bare field
        if not math.isfinite(way):
            way = 1.4 * float(np.hypot(*xy))               # off the flyable ground: the straight line with a margin
        return HOME_BASE_S + HOME_S_PER_M * way

    def decide(self, state, a, t):
        xy = state[POS][:2]
        what = self.mode if self.mode != "confirm" or self.target is None else ("confirm_fly" if self.target["arrived"] is None else "confirm_look")
        self.spent[what] = self.spent.get(what, 0.0) + (t - self.last_t if self.last_t is not None else 0.0)
        self.last_t = t
        time_left = self.time_left = float(state[TIME_LEFT])
        if self.mode != "home" and time_left < self.home_needed(xy):
            self.mode, self.leg = "home", None
        if self.mode == "home" and self.plane_ok and self.held[I_RETURN] <= 0.5:
            self.late_visit(xy, time_left)                 # someone first seen on the way home: back to him if time allows
        if self.mode == "home" and (not self.log or self.log[-1][0] not in ("HOME", "HOME2", "TIME")):
            first = not any(x[0] == "HOME" for x in self.log)
            way = float(self.maps.home_d[self.maps.cell(xy)])
            self.log.append(("HOME" if first else "HOME2", round(t, 1), len(self.tracks), sum(tr["reported"] for tr in self.tracks),
                             int((~self.maps.cleared).sum()), round(float(np.hypot(*xy)), 1), round(way, 1) if math.isfinite(way) else -1.0))
            self.log.append(("TIME", {k: round(v, 1) for k, v in self.spent.items()}))
        if self.mode == "takeoff":
            self.mode = "spin"; self.spin_left = SPIN_FRAMES + 1; self.spin_at = xy.copy()
        if self.mode == "spin":
            a[I_TURN] = 1.0
            hag = float(state[2] - self.ground_at(xy)) if self.plane_ok else CRUISE_M
            self.tilt(a, self.colour_tilt(hag))
            if self.last_age is not None and state[FRAME_AGE] < 0.11:
                pass
            self.spin_left -= 1.0 / 5.0                    # a decision is a fifth of a frame
            if self.spin_left <= 0:
                self.mode = "next"
                shade, tilt = self.shade_in_reach(state)
                if shade >= TSPIN_NEED and time_left > self.home_needed(xy) + TSPIN_S + 6.0:
                    self.mode, self.tspin, self.tspin_tilt = "tspin", t, tilt     # shade in reach: the same turn again in thermal
                else:
                    self.keep_promise()
            self.hold(state, a, self.spin_at)
            return
        if self.mode == "tspin":
            a[I_THERMAL] = 1.0
            a[I_TURN] = TSPIN_DPS / YAW_MAX_DPS
            self.tilt(a, self.tspin_tilt)
            if t - self.tspin >= TSPIN_S:
                self.mode = "next"
                self.keep_promise()
            self.hold(state, a, self.spin_at)
            return
        if self.mode == "next":
            seen = (self.maps.version, self.plane_ok, len(self.tracks))
            if self._idle is not None and self._idle[0] == seen and t < self._idle[1]:
                self.hover(state, a)                       # nothing has changed since the last plan found nothing to do
                return
            self.promise, self.promise_live = None, False
            self.mode, self.goal, self.leg = self.choose(xy, time_left)
            if self.mode == "respin":                      # the ground model is not trusted yet: another turn on the spot
                self.mode, self.spin_left, self.spin_at = "spin", SPIN_FRAMES + 1, xy.copy()
            if self.mode == "glance":
                tr_g = self.glance_pick(xy, self.tracks)
                if tr_g is None:
                    self.mode = "next"
                else:
                    self.start_glance(tr_g, xy, t, ("next", None, None))
            # those who could be visited now are known to this leg; anyone else, or everyone if no visit was possible
            # (no trusted ground yet), interrupts it later
            self.leg_ids = {tr["id"] for tr in self.waiting(xy)} if self.plane_ok else set()
            self.scan_since = t if self.mode == "scan" else None
            self._idle = (seen, t + 1.0) if self.mode == "next" else None
        if self.mode in ("go", "fill_go", "go_scan", "scan") and self.plane_ok and self.zoom_wait is None and self.pending is None:
            fresh = [tr for tr in self.tracks if tr["id"] not in self.leg_ids and not tr["reported"] and not tr.get("dead")]
            # a leg in flight is cut only for someone surely there (warm in thermal, or shown by a close-up) or when the
            # time for visits runs short; a sighting by the colour camera alone waits for the leg's end and its close-up
            short = self.time_left <= self.home_needed(xy) + VISIT_ALL_S
            new = [tr for tr in self.waiting(xy) if tr["id"] not in self.leg_ids and (tr.get("warm") or tr.get("zoomed") or short)]
            tr_g = self.glance_pick(xy, fresh) if fresh and self.mode == "scan" and float(np.hypot(*state[4:6])) < 0.5 else None
            if tr_g is not None:
                # scanning from a hover: the new sighting gets its close-up at once, the scan goes on after
                self.scan = None
                self.leg_ids.add(tr_g["id"])
                self.start_glance(tr_g, xy, t, ("scan", None, None))
                self.mode = "glance"
            elif new:                                      # someone newly found: he comes before the next stop
                self.leg_ids |= {tr["id"] for tr in new}
                vp = self.visit_now(xy, new, "viewpoint, on the way to a stop")
                if vp is not None:
                    self.promise, self.promise_live, self.scan = None, False, None
                    self.mode, self.goal, self.leg = "confirm", vp, None
        if self.mode in ("go", "fill_go"):
            if self.fly_to(state, a, self.goal):
                self.mode = "spin"; self.spin_left = SPIN_FRAMES + 1; self.spin_at = xy.copy()
                self.stops.append(xy.copy())
                self.scan_done[:] = False                  # from a new place the leftover ground may be in sight
                self.promise_live = self.promise is not None
            return
        if self.mode == "confirm":
            self.do_confirm(state, a, t)
            return
        if self.mode == "go_scan":
            if self.fly_to(state, a, self.goal):
                self.mode, self.scan, self.spin_at = "scan", None, xy.copy()
                self.scan_from.append(xy.copy())
                self.scan_done[:] = False
            return
        if self.mode == "glance":
            if self.glance is None:
                self.mode = "next"
            else:
                self.do_glance(state, a, t)
            return
        if self.mode == "scan":
            if (self.scan_since is not None and t - self.scan_since >= SCAN_REPLAN_S and self.zoom_wait is None
                    and self.pending is None):
                self.scan, self.mode = None, "next"        # a scan is a filler: plan again, visits and real stops first
                self.hold(state, a, self.spin_at)
                return
            self.do_scan(state, a, t)
            return
        if self.mode == "home":
            # the dock's own flight home ignores crowns and drifts about 6 m if pressed at speed: press it hovering at the dock
            at_dock = self.fly_to(state, a, np.zeros(2), mask="flyable", near=1.5)
            if at_dock and float(np.hypot(*state[4:6])) < 0.6:
                self.press(a, I_RETURN)
            return
        self.hover(state, a)

    def scan_target(self, xy):
        """The unchecked cell to aim the next close-up at: in clear sight from here, in a close-up's reach, nearest first;
        None when nothing is left to scan or the close-ups are needed for visits."""
        m = self.maps
        if self.zooms_left <= SCAN_KEEP_ZOOMS:
            return None
        todo = ~m.cleared & m.can_hold & ~self.scan_done
        if not todo.any():
            return None
        d = np.hypot(m.xy[..., 0] - xy[0], m.xy[..., 1] - xy[1])
        todo &= (d >= 6.0) & (d <= 42.0)
        if not todo.any():
            return None
        cells = np.argwhere(todo)
        vis = self.sight_from(np.array([xy[0], xy[1], CRUISE_M]), m.xy[todo])
        cells = cells[vis]
        if not len(cells):
            return None
        dd = d[cells[:, 0], cells[:, 1]]
        return tuple(int(v) for v in cells[int(np.argmin(dd))])

    def scan_place(self, dist, xy):
        """A reachable place from which the most leftover ground is in clear sight and in a close-up's reach, per second
        of flight; None when there is no leftover ground or no such place."""
        m = self.maps
        todo = ~m.cleared & m.can_hold
        if not todo.any() or self.zooms_left <= SCAN_KEEP_ZOOMS:
            return None
        tp = m.xy[todo]
        ok = m.allowed() & np.isfinite(dist)
        ok[::1, ::1] &= ((np.arange(m.shape[0])[:, None] % 4 == 0) & (np.arange(m.shape[1])[None, :] % 4 == 0))   # a 4 m lattice
        for sx, sy in self.scan_from + [xy]:
            ok &= np.hypot(m.xy[..., 0] - sx, m.xy[..., 1] - sy) > 4.0
        best, best_rate = None, 0.0
        for c in np.argwhere(ok):
            p = m.xy[c[0], c[1]]
            d = np.hypot(tp[:, 0] - p[0], tp[:, 1] - p[1])
            reach = (d >= 6.0) & (d <= 40.0)
            if not reach.any():
                continue
            n = int(self.sight_from(np.array([p[0], p[1], CRUISE_M]), tp[reach]).sum())
            rate = n / (dist[c[0], c[1]] / MAX_MPS + 4.0)
            if n >= 2 and rate > best_rate:
                best, best_rate = p.copy(), rate
        return best

    def do_scan(self, state, a, t):
        """Aim at a patch of unchecked ground, take a close-up of it, go on to the next."""
        m = self.maps
        xy = state[POS][:2]
        sc = self.scan
        if sc is None or m.cleared[sc["cell"]] or t - sc["since"] > SCAN_GIVE_UP_S:
            if sc is not None and not m.cleared[sc["cell"]]:
                self.scan_done[sc["cell"]] = True          # could not be scanned from here: not asked again from this stop
            cell = self.scan_target(xy) if self.zoom_wait is None else None
            if cell is None:
                if self.zoom_wait is None:
                    self.scan, self.mode = None, "next"
                self.hold(state, a, self.spin_at)
                return
            self.scan = sc = {"cell": cell, "since": t, "aimed": None, "at": xy.copy()}
        tgt = m.xy[sc["cell"]]
        g = float(self.ground_at(tgt)) + 0.9               # the middle of a body standing there
        d = tgt - xy
        err = self.steer(a, state, math.degrees(math.atan2(d[0], d[1])))
        cam_h = float(state[2]) - CAMERA_BELOW_M
        tilt = -math.degrees(math.atan2(cam_h - g, max(float(np.hypot(*d)), 0.5)))
        self.tilt(a, tilt)
        self.hold(state, a, sc["at"])
        aimed = abs(err) < AIM_TOL_DEG and abs(float(state[TILT]) - tilt) < AIM_TOL_DEG and abs(self.yaw_rate()) < 4.0
        sc["aimed"] = (sc["aimed"] if sc["aimed"] is not None else t) if aimed else None
        # the close-up is aimed along the centre of the frame SHOWN, so that frame must have been taken after the aim settled
        if (aimed and t - sc["aimed"] >= SCAN_AIM_S and t - float(state[FRAME_AGE]) >= sc["aimed"] + 0.05
                and float(np.hypot(*state[4:6])) < 0.3 and self.zoom_wait is None
                and self.pending is None and state[ZOOMS_LEFT] > SCAN_KEEP_ZOOMS and self.held[I_ZOOM] <= 0.5):
            slant = float(np.linalg.norm(np.array([d[0], d[1], cam_h - g])))
            lens = 7 if slant > 22.0 else 3
            a[I_ZOOM], a[I_LENS] = 1.0, (1.0 if lens == 7 else 0.0)
            a[I_ZCX], a[I_ZCY], a[I_ZW], a[I_ZH] = 0.5, 0.5, 0.05, 0.1
            self.zoom_wait = {"t": t, "track": None, "lens": lens, "cam": self.camera_at(t - float(state[FRAME_AGE])), "scan": True}

    def look_scan(self, image, zw):
        """A scanning close-up: people in it become sightings, and the ground it shows closely enough is checked."""
        m = self.maps
        cam, focal = zw["cam"], ZOOM_F[zw["lens"]]
        before = [(x, np.asarray(x["xy"], float).copy()) for x in self.tracks]    # who was known before this close-up
        spots = []
        for x0, y0, x1, y1, s in self.boxes("zoom", image):
            if s < ZOOM_SCORE2 or max(x1 - x0, y1 - y0) < ZOOM_MIN_PX:
                continue
            ray = self.pixel_ray(cam, *self.foot(x0, y0, x1, y1), focal=focal)
            q = self.on_ground(cam[0], ray)
            if q is None or np.hypot(q[0], q[1]) < DOCK_KEEP_M:
                continue
            spot = m.stand_spot(q[:2])
            if spot is None:
                continue
            spots.append(np.asarray(spot, float).copy())
            tr = self.track_of(spot)
            tr["rays"] = (tr.get("rays", []) + [(cam[0].copy(), ray, (max(x1 - x0, y1 - y0) * WIDE_F / focal) ** 2 * 4.0)])[-10:]
            tr["n"] += 2
            tr["zoomed"] = True
            self.place(tr)
        self.merge()
        self.sync_people()
        self.mark_seen(cam, zoom_focal=focal)
        gt = zw.get("glance")
        if gt is not None:
            # the close-up asked for one sighting: someone at his spot makes him real, nobody there counts against him
            here = np.asarray(cam[0][:2], float).copy()
            live = next((x for x in self.tracks if x is gt), None) or self.track_of(gt["xy"], make=False)
            # a box of THIS close-up within 3 m of his spot, and nearer to it than to anyone known before (reported or not),
            # shows him; a neighbour's body beside the spot is the neighbour's and proves nothing about it
            others = [p for x, p in before if x is not gt]
            shown = any(np.hypot(*(q - gt["xy"])) <= 3.0 and all(np.hypot(*(q - gt["xy"])) <= np.hypot(*(q - p)) for p in others)
                        for q in spots)
            if live is not None:
                live["glances"] = live.get("glances", 0) + 1
                if shown:
                    live["zoomed"], live["weak"] = True, False
                    self.leg_ids.discard(live["id"])           # he may be visited now, also in the middle of a leg
                else:
                    apart = all(np.hypot(*(here - g_)) >= 10.0 for g_ in live.get("glance_from", []))
                    live["glance_empty"] = live.get("glance_empty", 0) + (1 if apart else 0)
                    if live["n"] < 3 and (live.get("weak") or live["glance_empty"] >= 2):
                        live["dead"] = True                   # seen once or twice only: a weak box and an empty close-up, or two empty ones from two places
                        # (someone the wide camera has seen three times or more is never ended by close-ups from afar: he goes last)
                        if not (self.target is not None and self.target.get("track") is live):
                            self.tracks = [x for x in self.tracks if x is not live]
                            self.sync_people()
                live["glance_from"] = live.get("glance_from", []) + [here]
        if self.scan is not None:
            self.scan_done[self.scan["cell"]] = True       # this patch has had its close-up
            self.scan["since"] = -1e9                      # go on to the next one

    @staticmethod
    def colour_tilt(hag):
        """The colour spin's tilt for a height above the ground: steeper when the ground is far below."""
        return float(np.clip(SPIN_TILT_DEG * (max(hag, 1.0) / CRUISE_M) ** 0.35, -60.0, -35.0))

    def spin_sees(self, eye, pts, g, thermal, tilt):
        """Which ground points (heights g) a full turn at eye shows closely enough for a person there to be found, the
        camera tilted `tilt` degrees: inside the frame from top to bottom (two degrees kept as margin) and small enough
        per pixel.  The ground is taken as it lies: seen from 9 m above the dock, ground 4 m lower is 13 m below."""
        d = np.hypot(pts[:, 0] - eye[0], pts[:, 1] - eye[1])
        dz = eye[2] - g
        dep = np.degrees(np.arctan2(dz, np.maximum(d, 1e-3)))
        focal, half_v, pix = (THERMAL_F, 256.0, THERMAL_CLEAR_PIX_M) if thermal else (WIDE_F, 240.0, CLEAR_PIX_M)
        half = math.degrees(math.atan(half_v / focal)) - 1.0
        head = np.degrees(np.arctan2(dz - BODY_TOP_M, np.maximum(d, 1e-3)))      # the top of a body standing there
        size = np.hypot(d, dz) / focal / np.sqrt(np.maximum(np.sin(np.radians(np.clip(dep, 0.5, 90.0))), 1e-3))
        return (dz > BODY_TOP_M + 0.5) & (dep <= -tilt + half) & (head >= -tilt - half) & (size <= 0.97 * pix)

    def thermal_view(self, eye, pts, g, vis):
        """The tilt at which a thermal spin at eye shows the most of the given ground points in clear sight, and which."""
        best_tilt, best = TSPIN_TILT_DEG, np.zeros(len(pts), bool)
        for tilt in TSPIN_TILTS:
            got = vis & self.spin_sees(eye, pts, g, True, tilt)
            if got.sum() > best.sum():
                best_tilt, best = tilt, got
        return best_tilt, best

    def shade_in_reach(self, state):
        """How many cells the colour frames showed in shade, still unchecked, a thermal spin from here would show, and
        the tilt that shows the most of them."""
        m = self.maps
        todo = m.shaded & ~m.cleared & m.can_hold
        if not todo.any():
            return 0, TSPIN_TILT_DEG
        xy = state[POS][:2]
        pts = m.xy[todo]
        eye = np.array([xy[0], xy[1], float(state[2]) - CAMERA_BELOW_M])
        near = np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1]) <= 45.0
        vis = np.zeros(len(pts), bool)
        if near.any():
            vis[near] = self.sight_from(eye, pts[near])
        tilt, got = self.thermal_view(eye, pts, self.ground_at(pts), vis)
        return int(got.sum()), tilt

    def keep_promise(self):
        """After a stop's spins: ground the stop was chosen for and did not show is not promised to another stop."""
        if self.promise is not None and self.promise_live:
            lost = self.promise & ~self.maps.cleared
            if lost.any():
                self.maps.failed |= lost
        self.promise, self.promise_live = None, False

    def look_thermal_wide(self, image, cam):
        """Warm bodies in a thermal frame of the search: sightings, as from a colour frame."""
        m = self.maps
        for x0, y0, x1, y1, s in self.boxes("thermal", image):
            if s < THERMAL_SCORE or max(x1 - x0, y1 - y0) < THERMAL_MIN_PX:
                continue
            ray = self.pixel_ray(cam, *self.foot(x0, y0, x1, y1), focal=THERMAL_F, cu=320.0, cv=256.0)
            p = self.on_ground(cam[0], ray)
            if p is None or np.hypot(p[0], p[1]) < DOCK_KEEP_M:
                continue
            spot = m.stand_spot(p[:2])
            if spot is None:
                continue
            tr = self.track_of(spot)
            tr["rays"] = (tr.get("rays", []) + [(cam[0].copy(), ray, (max(x1 - x0, y1 - y0) * WIDE_F / THERMAL_F) ** 2)])[-10:]
            tr["n"] += 1
            tr["warm"] = True                              # a shadow is not warm
            self.place(tr)
        self.merge()
        self.sync_people()

    def choose(self, xy, time_left):
        """Next job: confirm the nearest unreported person, else the next search stop, else fill, else home."""
        m = self.maps
        if not self.plane_ok and self.respins < RESPIN_MAX:
            self.respins += 1                              # nothing can be planned on untrusted ground: more laser readings
            return "respin", None, None
        if self.glance_pick(xy, self.tracks) is not None:
            return "glance", None, None                    # check before you fly: one close-up of a sighting from here
        vp = self.visit_now(xy, self.waiting(xy), "viewpoint")
        if vp is not None:
            return "confirm", vp, None
        dist = m.reach(xy)
        stop = self.best_stop(dist, xy, clear=True, least=STOP_MIN_CELLS)
        if stop is not None:
            self.promise = self.promise_next
            return "go", stop, None
        if self.scan_target(xy) is not None and time_left > self.home_needed(xy) + 6.0:
            return "scan", None, None                      # only patches are left: aimed close-ups from here
        if time_left > self.home_needed(xy) + 14.0:
            spot = self.scan_place(dist, xy)
            if spot is not None:
                return "go_scan", spot, None               # leftover ground is in sight from somewhere else: scan from there
        stop = self.best_stop(dist, xy, clear=True)        # what no close-up reached (none left, or no sight): a stop after all
        if stop is not None:
            self.promise = self.promise_next
            return "go", stop, None
        vp = self.visit_now(xy, self.waiting(xy, sure=False, ghosts=True), "viewpoint")    # seen from one place only, and nothing else is left
        if vp is not None:
            return "confirm", vp, None
        if time_left > self.home_needed(xy) + 15.0:
            stop = self.best_stop(dist, xy, clear=False)
            if stop is not None:
                return "fill_go", stop, None
        waiting = [tr for tr in self.tracks if not tr["reported"] and not tr.get("dead") and tr["tried"] < 3
                   and not (tr.get("weak") and not tr.get("zoomed") and not tr.get("warm"))]
        if waiting and time_left > self.home_needed(xy) + 8.0:
            return "next", None, None                      # someone found is still unreported: wait for a view of him
        return "home", None, None

    def glance_pick(self, xy, pool):
        """A sighting worth one close-up from here before anyone flies to it: made by the colour camera alone (not warm
        in thermal, not yet shown by a close-up), never looked at from near here, in clear sight within the long lens's
        reach, and not so placed that the visit itself could be made from this spot.  Nearest first."""
        if not self.plane_ok or self.zooms_left <= GLANCE_KEEP_ZOOMS:
            return None
        xy = np.asarray(xy, float)
        best = None
        for tr in pool:
            if tr["reported"] or tr.get("dead") or tr["tried"] or tr.get("warm") or tr.get("zoomed"):
                continue
            if any(np.hypot(*(xy - g_)) < 8.0 for g_ in tr.get("glance_from", [])):
                continue
            if tr.get("best", 0.0) >= GLANCE_SKIP_SCORE and tr["n"] >= 3 and not tr.get("weak") and self.proven(tr):
                continue                                   # a strong sighting: the visit needs no check first
            d = float(np.hypot(*(tr["xy"] - xy)))
            if not GLANCE_MIN_M <= d <= GLANCE_MAX_M or not self.sight(xy, tr["xy"]):
                continue
            if VIEW_MIN_M <= d <= VIEW_MAX_M and not tr.get("weak") and self.maps.holds(xy) and not tr.get("views"):
                continue                                   # the visit can start right here: no flight to save
            if best is None or d < best[0]:
                best = (d, tr)
        return None if best is None else best[1]

    def do_glance(self, state, a, t):
        """Hover, face the sighting, one close-up of its spot; what the close-up shows is read in look_scan."""
        gl = self.glance
        tr = gl["track"]
        xy = state[POS][:2]
        gone = tr["reported"] or tr.get("dead") or not any(x is tr for x in self.tracks)
        if gone or t - gl["since"] > GLANCE_GIVE_UP_S or (gl["asked"] is not None and self.zoom_wait is None):
            if gl["asked"] is None and not gone:
                tr["glance_from"] = tr.get("glance_from", []) + [np.asarray(xy, float).copy()]   # no close-up came of it: not again from here
            self.mode, self.goal, self.leg = gl["back"]
            self.glance = None
            self.hold(state, a, gl["at"])
            return
        g = float(self.ground_at(tr["xy"])) + 0.9
        d = tr["xy"] - xy
        err = self.steer(a, state, math.degrees(math.atan2(d[0], d[1])))
        cam_h = float(state[2]) - CAMERA_BELOW_M
        tilt = -math.degrees(math.atan2(cam_h - g, max(float(np.hypot(*d)), 0.5)))
        self.tilt(a, tilt)
        self.hold(state, a, gl["at"])
        aimed = abs(err) < AIM_TOL_DEG and abs(float(state[TILT]) - tilt) < AIM_TOL_DEG and abs(self.yaw_rate()) < 4.0
        gl["aimed"] = (gl["aimed"] if gl["aimed"] is not None else t) if aimed else None
        if (gl["asked"] is None and aimed and t - gl["aimed"] >= SCAN_AIM_S and t - float(state[FRAME_AGE]) >= gl["aimed"] + 0.05
                and float(np.hypot(*state[4:6])) < 0.3 and self.zoom_wait is None and self.pending is None
                and state[ZOOMS_LEFT] >= 1 and self.held[I_ZOOM] <= 0.5):
            slant = float(np.linalg.norm(np.array([d[0], d[1], cam_h - g])))
            lens = 7 if slant > 22.0 else 3
            a[I_ZOOM], a[I_LENS] = 1.0, (1.0 if lens == 7 else 0.0)
            a[I_ZCX], a[I_ZCY], a[I_ZW], a[I_ZH] = 0.5, 0.5, 0.05, 0.1
            gl["asked"] = t
            self.zoom_wait = {"t": t, "track": None, "lens": lens, "cam": self.camera_at(t - float(state[FRAME_AGE])), "scan": True, "glance": tr}

    def start_glance(self, tr, xy, t, back):
        self.glance = {"track": tr, "since": t, "aimed": None, "asked": None, "at": np.asarray(xy, float).copy(), "back": back}
        self.log.append(("glance", round(float(tr["xy"][0]), 1), round(float(tr["xy"][1]), 1), round(float(t), 1)))

    def waiting(self, xy, sure=True, ghosts=False):
        """People found and still to be visited.  sure: those surely real (warm in thermal, or seen at one spot from two
        places) and those the drone has since moved away from far enough for a shadow to have shown itself.  ghosts: also
        the ghosts of reported men, which only the last pass, when nothing else is left, asks for."""
        if sure and self.time_left <= self.home_needed(xy) + VISIT_ALL_S:
            sure = False                                   # the time for visits runs short: everyone still waiting counts
        out = [tr for tr in self.tracks if not tr["reported"] and not tr.get("dead") and tr["tried"] < 3
               and not (tr.get("weak") and not tr.get("zoomed") and not tr.get("warm")           # a weak box alone is never flown to,
                        and (sure or tr["n"] < 3))]                                              # unless it came three times and nothing else is left
        if not ghosts:
            # (also when the time runs short: the search may still find someone real in it, a ghost never is)
            out = [tr for tr in out if not self.ghost(tr)]
        if sure:
            # a sighting a close-up showed empty goes last (when nothing else is left)
            out = [tr for tr in out if tr.get("zoomed") or tr.get("warm") or not tr.get("glance_empty")]
        if sure and self.time_left > self.home_needed(xy) + VISIT_ALL_S:
            now = float(self.hist[-1][0]) if self.hist else 0.0
            out = [tr for tr in out if self.proven(tr) or np.hypot(*(np.asarray(xy, float) - tr["born"])) >= UNPROVEN_WAIT_M
                   or now - tr.get("born_t", 0.0) >= UNPROVEN_WAIT_S]
        return out

    def visit_now(self, xy, pool, why):
        """Start a visit to the nearest of the pool that has a viewpoint now (the surely real before the others); None
        when none has.  A track asked once waits until the maps change."""
        if not self.plane_ok:
            return None
        m = self.maps
        todo = [tr for tr in pool if tr.get("waited_v") != m.version]
        for tr in sorted(todo, key=lambda tr: (self.ghost(tr), not self.proven(tr), np.hypot(*(tr["xy"] - xy))))[:4]:
            tr["waited_v"] = m.version
            vp = self.viewpoint(tr, xy)
            if vp is not None:
                self.visit(tr, vp, why)
                return vp
            self.log.append(("pick", round(float(tr["xy"][0]), 1), round(float(tr["xy"][1]), 1), "NO VIEWPOINT"))
        return None

    def visit(self, tr, vp, why):
        self.log.append(("pick", round(float(tr["xy"][0]), 1), round(float(tr["xy"][1]), 1), why))
        self.target = {"track": tr, "at": vp, "since": None, "arrived": None, "zooms": 0, "lasers": [], "looks": [], "seen": 0,
                       "tlooks": [], "tlooks_n": 0, "last_zoom": None}
        d0 = vp - tr["xy"]
        tr["views"] = tr.get("views", []) + [math.atan2(d0[1], d0[0])]
        self.target["side"] = tr["views"][-1]

    def late_visit(self, xy, time_left):
        """Flying home with someone found and not yet reported (he was first seen on the way, or his visit had to
        wait): turn back to him while the flight there, the look and the way home from there are still in hand.  Asked
        once per state of the maps and of the list of people waiting, never every decision."""
        m = self.maps
        todo = [k for k, tr in enumerate(self.tracks) if not tr["reported"] and not tr.get("dead") and tr["tried"] < 3
                and not (tr.get("weak") and not tr.get("zoomed") and not tr.get("warm"))]
        key = (m.version, tuple(todo))
        if not todo or key == self.late_key:
            return False
        self.late_key = key
        for k in sorted(todo, key=lambda k: (self.ghost(self.tracks[k]), np.hypot(*(self.tracks[k]["xy"] - xy))))[:2]:
            tr = self.tracks[k]
            vp = self.viewpoint(tr, xy)
            if vp is None:
                continue
            if time_left > self.home_needed(vp) + float(np.hypot(*(vp - xy))) / 2.5 + LATE_LOOK_S:
                self.visit(tr, vp, "viewpoint, turned back from the way home")
                self.mode, self.goal, self.leg = "confirm", vp, None
                return True
        return False

    def best_stop(self, dist, xy, clear, least=3):
        """The allowed cell whose spin gains the most (cleared ground, or covered cells) per second of travel and spin."""
        m = self.maps
        want = (~m.cleared & ~m.failed) if clear else (~m.covered)
        if not want.any():
            return None
        H = CRUISE_M
        # ring a spin sees: 1 m to the reach where the pixel rule holds, from about the cruise height
        pix = CLEAR_PIX_M if clear else COVER_PIX_M
        reach = (pix * WIDE_F * math.sqrt(H)) ** (2.0 / 3.0)       # slant range where the pixel rule holds (flat ground)
        r_ground = math.sqrt(max(reach * reach - H * H, 1.0))
        k = disk(r_ground / GRID_M)
        gain = convolve(want.astype(np.float32), k.astype(np.float32), mode="constant")
        ok = m.allowed() & np.isfinite(dist)
        for sx, sy in self.stops:                          # a stop already made saw what it could from there
            ok &= np.hypot(m.xy[..., 0] - sx, m.xy[..., 1] - sy) > 2.5
        if not ok.any():
            return None
        score = np.where(ok, gain / (dist / MAX_MPS + SPIN_FRAMES * 0.5 + 2.0), -1.0)
        if not clear:
            best = np.unravel_index(int(np.argmax(score)), score.shape)
            return m.point(best) if gain[best] >= 3.0 else None
        # the twelve best by reach alone, then judged by what each can really see: a clear line to the ground, in sun for
        # the colour spin or 7-24 m away for the thermal one
        cand = []
        for k in np.argsort(score.ravel())[::-1][:600]:
            c = np.unravel_index(int(k), score.shape)
            if score[c] <= 0:
                break
            if all(math.hypot(c[0] - o[0], c[1] - o[1]) * GRID_M >= 4.0 for o in cand):
                cand.append(c)
                if len(cand) >= 14:
                    break
        todo = np.argwhere(want)
        tp = m.xy[want]
        tg = self.ground_at(tp)
        sh = m.shaded[want]
        best, best_rate, best_got = None, 0.0, None
        for c in cand:
            p = m.xy[c]
            near = np.hypot(tp[:, 0] - p[0], tp[:, 1] - p[1]) <= 45.0
            if not near.any():
                continue
            eye = np.array([p[0], p[1], CRUISE_M - CAMERA_BELOW_M])
            vis = np.zeros(len(tp), bool)
            vis[near] = self.sight_from(eye, tp[near])
            got = vis & ~sh & self.spin_sees(eye, tp, tg, False, self.colour_tilt(float(eye[2] - self.ground_at(p))))
            _, warm = self.thermal_view(eye, tp, tg, vis & sh)
            thermal = int(warm.sum()) >= TSPIN_NEED                # fewer and the thermal spin is not flown
            if thermal:
                got = got | warm
            n = int(got.sum())
            rate = n / (dist[c] / MAX_MPS + SPIN_FRAMES * 0.5 + 2.0 + (TSPIN_S if thermal else 0.0))
            if n >= least and rate > best_rate:
                best, best_rate, best_got = c, rate, got
        if best is None:
            return None
        self.promise_next = np.zeros(m.shape, bool)
        self.promise_next[todo[best_got, 0], todo[best_got, 1]] = True
        return m.point(best)

    def sight(self, a, b):
        """No crown circle between a viewpoint and a person (crowns hide what is under and behind them)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        ab = b - a
        end = max(0.0, 1.0 - 1.5 / max(float(np.hypot(*ab)), 1e-6))    # a crown beside him blocks only views that cross it
        for e, n, r in self.maps.crowns:
            c = np.array([e, n])
            t = float(np.clip(np.dot(c - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, end))
            if np.hypot(*(a + t * ab - c)) < r + 0.3:
                return False
        return True

    def viewpoint(self, tr, xy):
        """Where to look at him from: here if he is 7.5-24 m away in plain sight, else the allowed point on a ring round
        him (9-24 m) in plain sight that is nearest to fly to; None while none is reachable yet."""
        m = self.maps
        xy = np.asarray(xy, float)
        d_here = float(np.hypot(*(xy - tr["xy"])))
        # over high ground the drone is low above him: from the usual rings the look would be flat and every height
        # error would grow on the ground, so nearer rings are offered too (never farther ones)
        k = float(np.clip(((CRUISE_M - 0.3) - float(self.ground_at(tr["xy"]))) / 9.0, 0.5, 1.0))
        rings = tuple(VIEW_M) if k > 0.85 else tuple(max(v * k, 8.0) for v in VIEW_M) + tuple(VIEW_M)
        best_m = VIEW_BEST_M if k > 0.85 else max(VIEW_BEST_M * k, 8.0)
        # someone already reported close by: the second look comes from the side the report was made from, so that both
        # spots carry the same errors and the same man is known again
        side = next((o["pin_side"] for o in self.tracks if o is not tr and o["reported"] and o.get("pin_side") is not None
                     and np.hypot(*(o["xy"] - tr["xy"])) < PIN_NEAR_M), None)
        here_ok = side is None or abs(wrap(math.degrees(math.atan2(xy[1] - tr["xy"][1], xy[0] - tr["xy"][0]) - side))) <= 45.0
        if (VIEW_MIN_M <= d_here <= VIEW_MAX_M and k > 0.85 and here_ok and m.holds(xy) and self.sight(xy, tr["xy"])
                and not tr.get("views")):
            return xy.copy()
        dist = m.reach(xy)
        ang = np.radians(np.arange(0, 360, 10.0))
        rr = np.repeat(np.array(rings), len(ang))
        aa = np.tile(ang, len(rings))
        pts = tr["xy"][None, :] + rr[:, None] * np.stack([np.cos(aa), np.sin(aa)], 1)
        ij = np.clip(np.round((pts - m.low) / GRID_M).astype(int), 0, np.array(m.shape) - 1)
        d = dist[ij[:, 0], ij[:, 1]]
        ok = np.isfinite(d) & self.sight_many(pts, tr["xy"])
        if not ok.any():
            # no viewpoint on the usual rings (unchecked ground, the edge or crowns in the way): from farther off, where
            # the long lens shows him and the laser places him
            rr = np.repeat(np.array(FAR_VIEW_M), len(ang))
            aa = np.tile(ang, len(FAR_VIEW_M))
            pts = tr["xy"][None, :] + rr[:, None] * np.stack([np.cos(aa), np.sin(aa)], 1)
            ij = np.clip(np.round((pts - m.low) / GRID_M).astype(int), 0, np.array(m.shape) - 1)
            d = dist[ij[:, 0], ij[:, 1]]
            ok = np.isfinite(d) & self.sight_many(pts, tr["xy"])
            if not ok.any():
                return None
            best_m = FAR_VIEW_M[0]
        cost = d + 2.0 * np.abs(rr - best_m)                   # near 13 m (nearer over high ground): a good box and a good spot
        if side is not None:
            cost = cost + 40.0 * (np.abs((np.degrees(aa - side) + 180.0) % 360.0 - 180.0) > 45.0)
        saw = tr.get("views_saw", [])
        for v in tr.get("views", []):                          # a side he was already looked at from: another one first,
            if any(abs(wrap(math.degrees(v - s))) < 1.0 for s in saw):
                continue                                       # unless the close-ups from there showed him
            cost = cost + 40.0 * (np.abs((np.degrees(aa - v) + 180.0) % 360.0 - 180.0) < 60.0)
        for s in saw:                                          # a side whose close-ups boxed him: back there first
            cost = cost - 20.0 * (np.abs((np.degrees(aa - s) + 180.0) % 360.0 - 180.0) < 30.0)
        cost = np.where(ok, cost, np.inf)
        return pts[int(np.argmin(cost))].copy()

    def sight_many(self, pts, b):
        """sight() for many viewpoints at once."""
        ok = np.ones(len(pts), bool)
        b = np.asarray(b, float)
        ab = b[None, :] - pts
        L2 = np.maximum((ab * ab).sum(1), 1e-9)
        end = np.maximum(0.0, np.minimum(1.0 - 1.5 / np.sqrt(L2), 1.0 - UNDER_CROWN_M / CRUISE_M))
        for e, n, r in self.maps.crowns:
            c = np.array([e, n])
            t = np.clip(((c[None, :] - pts) * ab).sum(1) / L2, 0.0, end)
            ok &= np.hypot(*(pts + t[:, None] * ab - c[None, :]).T) >= r + 0.3
        return ok

    # ------------------------------------------------------------------ confirm
    def do_confirm(self, state, a, t):
        tg = self.target
        tr = tg["track"]
        xy = state[POS][:2]
        if tg["since"] is None:
            tg["since"] = t
        probing = tg.get("probe") is not None
        spent = tg["zooms"] >= ZOOMS_PER_VISIT and self.zoom_wait is None and self.pending is None and not probing
        empty = (tg["zooms"] >= 2 and tg["seen"] == 0 and self.zoom_wait is None      # two close-ups of bare ground
                 and tg["tlooks_n"] >= THERMAL_LOOKS and not probing)                # and no warm body in thermal
        if (tr["reported"] or tr.get("dead") or tg.get("no_flat") or t - tg["since"] > CONFIRM_GIVE_UP_S
                or (tg["arrived"] is not None and not probing and t - tg["arrived"] > STAY_S + tg.get("grace", 0.0)) or spent or empty):
            if tg["arrived"] is not None and tg["zooms"] >= 1:
                tr["tried"] += 1                           # a try is a look that was really taken
            else:
                tr["lost"] = tr.get("lost", 0) + 1
                if tr["lost"] >= 4:
                    tr["tried"], tr["lost"] = tr["tried"] + 1, 0   # no look came of it four times: that is a try too
            if not tr["reported"] and tg["arrived"] is not None and tg["looks"] and tg.get("side") is not None:
                tr["views_saw"] = tr.get("views_saw", []) + [tg["side"]]     # close-ups boxed him from here: worth another try
            if not tr["reported"] and tg["arrived"] is not None and tg["seen"] == 0 and tg["zooms"] >= 2:
                if self.shadow is None and not self.proven(tr):
                    self.hints = (self.hints + [(o[0].copy(), (o[2] - o[0]) / o[1]) for o in tr.get("obs", [])[:3]])[-9:]
                tr["empty"] = tr.get("empty", 0) + 1
                tr["dead"] = tr["empty"] >= 2                 # nothing there from two sides: not a person
            self.log.append(("end", round(float(tr["xy"][0]), 1), round(float(tr["xy"][1]), 1),
                             "REPORTED" if tr["reported"] else ("never arrived" if tg["arrived"] is None else
                             f"no report: zooms {tg['zooms']} aimed {tg.get('aimed_since') is not None} zoom-results {tg.get('zres', [])}")))
            if tg["arrived"] is not None:
                tr.pop("waited_v", None)
            self.mode, self.target, self.zoom_wait = "next", None, None
            if tr.get("dead"):
                self.tracks = [x for x in self.tracks if x is not tr or tr["reported"]]
                self.sync_people()
            self.hover(state, a)
            return
        if tg["arrived"] is None:
            if self.fly_to(state, a, tg["at"]):
                tg["arrived"] = t
                # stopping and coming back is not part of the stay (a look begun in full flight brakes for up to 6 s)
                tg["grace"] = min(1.5 * max(float(np.hypot(*state[4:6])) - 2.2, 0.0), 6.0)
            return
        if tg.get("probe") is not None:
            self.do_probe(state, a, t, tg, tr)
            return
        # at the viewpoint: face him, aim the camera centre at his middle
        g = float(self.ground_at(tr["xy"]))
        aim = np.array([tr["xy"][0], tr["xy"][1], g + 0.5])
        d = aim[:2] - xy
        want = math.degrees(math.atan2(d[0], d[1]))
        err = self.steer(a, state, want)
        cam_h = float(state[2]) - CAMERA_BELOW_M
        tilt = -math.degrees(math.atan2(cam_h - aim[2], max(float(np.hypot(*d)), 0.5)))
        self.tilt(a, tilt)
        self.hold(state, a, tg["at"])
        if tg["tlooks_n"] < THERMAL_LOOKS or (tg["tlooks"] and not tg["looks"]):
            a[I_THERMAL] = 1.0                             # thermal until he has been looked at in it, and on while only it sees him
        aimed = abs(err) < AIM_TOL_DEG and abs(float(state[TILT]) - tilt) < AIM_TOL_DEG and abs(self.yaw_rate()) < 4.0
        if not aimed:
            tg["aimed_since"] = None
        elif tg.get("aimed_since") is None:
            tg["aimed_since"] = t
        # laser: it reads once a second and a miss still reads normal (the ground behind him), so only a NEW reading that
        # came after the aim had held 1.1 s, close to where we think he stands, counts
        lp = state[LASER_POINT].copy()
        new_reading = tg.get("last_laser") is None or float(np.linalg.norm(lp - tg["last_laser"])) > 1e-6
        tg["last_laser"] = lp
        if (aimed and tg["aimed_since"] is not None and t - tg["aimed_since"] >= 1.1 and new_reading
                and int(round(state[LASER_STATUS])) == 0 and state[LASER_RANGE] > 0):
            if np.hypot(*(lp[:2] - tr["xy"])) < LASER_AGREE_M:
                tg["lasers"].append(lp)
        # close-up: once aimed, the lens that makes him 120-300 px tall
        still = float(np.hypot(*state[4:6])) < 0.3
        if (aimed and still and tg["aimed_since"] is not None and t - tg["aimed_since"] >= 0.3
                and (t - float(state[FRAME_AGE]) >= tg["aimed_since"] + 0.05 or t - tg["aimed_since"] >= 0.8) and self.zoom_wait is None
                and self.pending is None and tg["zooms"] < ZOOMS_PER_VISIT and state[ZOOMS_LEFT] >= 1 and self.held[I_ZOOM] <= 0.5
                and not (tg.get("zz_ready") and tg.get("last_zoom") is not None and tg["tlooks_n"] < THERMAL_LOOKS)):
            slant = float(np.linalg.norm(np.array([d[0], d[1], cam_h - aim[2]])))
            far = float(np.hypot(*(np.asarray(tg["at"], float) - tr["xy"]))) > 22.0       # a viewpoint of last resort
            lens = 7 if slant > (22.0 if far else LENS7_FROM_M) else 3
            a[I_ZOOM], a[I_LENS] = 1.0, (1.0 if lens == 7 else 0.0)
            a[I_ZCX], a[I_ZCY], a[I_ZW], a[I_ZH] = 0.5, 0.5, 0.05, 0.1
            tg["zooms"] += 1
            tg["last_zoom"] = None
            self.zoom_wait = {"t": t, "track": tr, "lens": lens, "cam": self.camera_at(t - float(state[FRAME_AGE]))}

    def look_zoom(self, image, state):
        zw = self.zoom_wait
        self.zoom_wait = None
        if zw.get("scan"):
            self.look_scan(image, zw)
            return
        tr = zw["track"]
        if tr["reported"] or self.reports >= MAX_REPORTS:
            return
        cam, focal = zw["cam"], ZOOM_F[zw["lens"]]
        best, part = None, None
        boxes = self.boxes("zoom", image)
        rivals = []                                        # every box on him the detector draws, weak ones too
        for x0, y0, x1, y1, s in boxes:
            if s < RIVAL_SCORE:
                continue
            q = self.on_ground(cam[0], self.pixel_ray(cam, *self.foot(x0, y0, x1, y1), focal=focal))
            if q is not None and float(np.hypot(*(q[:2] - tr["xy"]))) < 2.5:
                rivals.append((x0, y0, x1, y1))
        for x0, y0, x1, y1, s in boxes:
            if s < ZOOM_SCORE2 or max(x1 - x0, y1 - y0) < ZOOM_MIN_PX:
                continue
            q = self.on_ground(cam[0], self.pixel_ray(cam, *self.foot(x0, y0, x1, y1), focal=focal))
            if q is None:
                continue
            if (y1 - y0) > 1.6 * (x1 - x0) and (y1 - y0) * float(np.linalg.norm(q - cam[0])) / focal < STANDING_MIN_M:
                continue                                   # a standing man's box that is too short: legs or trunk only
            d = float(np.hypot(*(q[:2] - tr["xy"])))
            if max(x1 - x0, y1 - y0) * float(np.linalg.norm(q - cam[0])) / focal < BODY_MIN_M:
                if d < 2.5 and self.maps.plausible(q[:2]) and (part is None or d < part[0]):
                    part = (d, (x0, y0, x1, y1))           # a head or a limb: he is there, but this is not his box
                continue
            if d < 2.5 and self.maps.plausible(q[:2]) and (best is None or d < best[0]):
                best = (d, (x0, y0, x1, y1), q[:2], s, max(x1 - x0, y1 - y0) * float(np.linalg.norm(q - cam[0])) / focal)
        tg = self.target if self.target is not None and self.target["track"] is tr else None
        if tg is not None:
            tg.setdefault("zres", []).append(("part" if part is not None else "none-near") if best is None else ("edge-cut" if best[1][0] < 3 or best[1][1] < 3 or best[1][2] > 637 or best[1][3] > 477 else "ok"))
            tg["seen"] += best is not None or part is not None
        if best is None and part is not None and tg is not None and not tr.get("laser_fixed"):
            px0, py0, px1, py1 = part[1]                   # aim the next close-up at what was seen of him
            tr["rays"] = tr.get("rays", []) + [(cam[0].copy(), self.pixel_ray(cam, *self.foot(px0, py0, px1, py1), focal=focal), 1e5)]
            self.place(tr)
        if best is None or tg is None:
            return
        _, (x0, y0, x1, y1), zxy, zs, size_m = best
        self.last_size = round(float(size_m), 2)           # kept beside the report, for the flight's own record
        if x0 < 3 or y0 < 3 or x1 > 637 or y1 > 477:
            if not tr.get("laser_fixed"):
                tr["rays"] = tr.get("rays", []) + [(cam[0].copy(), self.pixel_ray(cam, *self.foot(x0, y0, x1, y1), focal=focal), 1e6)]
                self.place(tr)                            # cut by the frame edge: re-aim on him and look again
            return
        # a report stands on two close-ups that put him on the same spot: the first one only moves the aim onto him
        share = ((x0 + x1) / 2 / 640, (y0 + y1) / 2 / 480, (x1 - x0) / 640, (y1 - y0) / 480)
        agree = [q for q, _s in tg["looks"] if np.hypot(*(q - zxy)) < LOOKS_AGREE_M]
        sure = zs >= ZOOM_SCORE or any(_s >= ZOOM_SCORE for q, _s in tg["looks"] if np.hypot(*(q - zxy)) < LOOKS_AGREE_M)
        warm = [q for q in tg["tlooks"] if np.hypot(*(q - zxy)) < SENSES_AGREE_M]
        # a box another box of a clearly different size overlaps (the detector is torn between part of him and all of
        # him) is never reported on the steady rule: only a sure box can settle which one the judge's box is
        torn = any(box_iou((x0, y0, x1, y1), r) >= RIVAL_IOU and not (1 / RIVAL_SIZE <= max(x1 - x0, y1 - y0) / max(r[2] - r[0], r[3] - r[1], 1e-6) <= RIVAL_SIZE)
                   for r in rivals)
        box_ok = self.steady(tg, zxy, zs, x1 - x0, y1 - y0, torn)
        tg["looks"].append((np.asarray(zxy, float), zs))
        tg.setdefault("look_box", []).append((np.asarray(zxy, float), zs, x1 - x0, y1 - y0))
        tg["last_zoom"] = (share, np.asarray(zxy, float), cam, focal, (x0, y0, x1, y1), zs, box_ok)
        # two looks on one spot make a report: a warm body in thermal and this close-up (TZ), or, once the thermal looks
        # are taken, two close-ups (ZZ)
        both = sum(_s >= ZZ_SCORE for q, _s in tg["looks"] if np.hypot(*(q - zxy)) < LOOKS_AGREE_M) >= 2    # this close-up is in the list
        kind = "TZ" if warm else ("ZZ" if agree and sure and both and ALLOW_ZZ and tg["tlooks_n"] >= THERMAL_LOOKS else None)
        # two agreeing close-ups wait only for the thermal looks, and this box may be reported: no more close-ups are taken
        # meanwhile, so that it stays the one shown and the last thermal look can report on it (look_thermal)
        tg["zz_ready"] = bool(kind is None and agree and sure and both and ALLOW_ZZ and box_ok)
        if kind is not None and not box_ok:
            # he is there, but this box is not sure enough to be the one the judge compares: another close-up is taken
            tg["zres"][-1] = "low-box"
            kind = None
        elif kind is None:
            if agree:
                return                                     # two close-ups agree: the thermal looks are still to come
        if kind is None:
            tr["rays"] = tr.get("rays", []) + [(cam[0].copy(), self.pixel_ray(cam, *self.foot(x0, y0, x1, y1), focal=focal), 1e6)]
            self.place(tr)
            return
        lasers = tg["lasers"]
        xy = np.mean(np.array(agree + [zxy]), axis=0)
        if lasers:
            lxy = np.median(np.array(lasers)[:, :2], axis=0)
            if np.hypot(*(lxy - xy)) < 0.8:
                xy = 0.5 * (lxy + xy)                         # the laser hit him: it agrees with the close-ups
        if self.flat_look(cam, xy) < FLAT_LOOK_DEG and not tg.get("probed"):
            tg["probe"] = {"cam": cam, "focal": focal, "box": (x0, y0, x1, y1), "share": share, "kind": kind,
                           "est": np.asarray(xy, float), "hits": [], "stage": 0, "since": None, "aimed": None}
            return                                         # too flat for the ground model: the laser places him
        self.finish_report(tr, xy, share, kind, cam)

    def steady(self, tg, zxy, zs, w, h, torn=False):
        """Whether a close-up box may be the one a report is made on: it scores REPORT_BOX_SCORE, or STEADY_SCORE while an
        earlier close-up of this visit put a box of the same size on the same spot with STEADY_SCORE too (a box that
        comes back the same from look to look is his whole body, not a part of him or of the ground), and no box of a
        clearly different size overlaps it (torn)."""
        if zs >= REPORT_BOX_SCORE:
            return True
        if zs < STEADY_SCORE or torn:
            return False
        return any(np.hypot(*(q - zxy)) < LOOKS_AGREE_M and s >= STEADY_SCORE and 1 / STEADY_SIZE <= w / max(bw, 1e-6) <= STEADY_SIZE
                   and 1 / STEADY_SIZE <= h / max(bh, 1e-6) <= STEADY_SIZE for q, s, bw, bh in tg.get("look_box", []))

    def flat_look(self, cam, xy):
        """Degrees the camera looks down at a spot on the ground."""
        eye = cam[0]
        return math.degrees(math.atan2(float(eye[2]) - float(self.ground_at(xy)), max(float(np.hypot(*(np.asarray(xy, float) - eye[:2]))), 0.1)))

    def finish_report(self, tr, xy, share, kind, cam, image_kind="zoom"):
        """The report on a person whose box and spot are settled: unless he is someone already reported."""
        if any(o is not tr and o["reported"] and np.hypot(*(o["xy"] - xy)) < SAME_REPORTED_M for o in self.tracks):
            tr["reported"], tr["dup"] = True, True      # the same person as one already reported: his other track goes
            self.tracks = [x for x in self.tracks if x is not tr]
            self.sync_people()
            return
        tr["xy"] = tr["pin"] = np.asarray(xy, float)
        tr["pin_side"] = math.atan2(float(cam[0][1] - xy[1]), float(cam[0][0] - xy[0]))
        tr["rays"] = []
        tr["reported"] = True
        self.sync_people()
        self.pending = (image_kind, share, float(xy[0]), float(xy[1]), kind)

    def do_probe(self, state, a, t, tg, tr):
        """A flat look (he can only be seen from far off, as on the high rim): the ground model cannot place him there,
        a height error of 0.3 m is a metre along so flat a line.  The laser can.  The camera centre, where the laser
        points, is aimed at the close-up's box.  Someone upright is an upright target: two readings at two heights on
        him lie over one spot, while a reading that passed him lands well behind and apart; two that agree are his spot.
        Someone lying is hit low on the box, where a line cannot pass over him."""
        pr = tg["probe"]
        xy = state[POS][:2]
        if pr["since"] is None:
            pr["since"] = t
        cam, focal = pr["cam"], pr["focal"]
        x0, y0, x1, y1 = pr["box"]
        wide = (x1 - x0) > 1.3 * (y1 - y0)
        # the marks lie inside the box above its lower edge, so a line through one either meets him or passes him and
        # lands behind him, never in front: of the readings, the nearest ones are on him
        marks = ([(x0 + f_ * (x1 - x0), y0 + 0.6 * (y1 - y0)) for f_ in (0.5, 0.25, 0.75)] if wide
                 else [((x0 + x1) / 2, y0 + 0.30 * (y1 - y0)), ((x0 + x1) / 2, y0 + 0.65 * (y1 - y0))])
        look = pr["est"] - cam[0][:2]
        far = float(np.hypot(*look))
        look = look / max(far, 1e-6)
        if len(pr["hits"]) >= len(marks) or t - pr["since"] > PROBE_GIVE_UP_S:
            hits = pr["hits"]
            spot = None
            if hits:
                along = np.array([float((h_[:2] - cam[0][:2]) @ look) for h_ in hits])
                near = [h_ for h_, d_ in zip(hits, along) if d_ <= along.min() + PROBE_AGREE_M]
                # two readings over one spot are him; a single nearest one is taken when it does not lie beyond where
                # the ground model put him (a line that passed him lands behind)
                if len(near) >= 2 or along.min() <= far + 0.5:
                    spot = np.mean(np.array(near)[:, :2], axis=0) + (0.2 if wide else 0.15) * look
            down = self.flat_look(cam, pr["est"])
            tg["probe"], tg["probed"] = None, True
            tg.setdefault("zres", []).append("laser" if spot is not None else "laser-none")
            if spot is None and (down >= FLAT_LAST_DEG or tr["tried"] >= 2):
                spot = pr["est"]                           # no two readings on him: the ground model's spot, if the look allows it
            if spot is None:
                tg["no_flat"] = True                       # not from here: this visit ends, the next comes from another side
            if spot is not None and self.pending is None:
                self.finish_report(tr, np.asarray(spot, float), pr["share"], pr["kind"], cam)
            self.hold(state, a, tg["at"])
            return
        u, v = marks[len(pr["hits"])]
        ray = self.pixel_ray(cam, u, v, focal=focal)
        reach = float(np.hypot(*(pr["est"] - cam[0][:2]))) / max(float(np.hypot(ray[0], ray[1])), 1e-6)
        aim = cam[0] + ray * reach                         # the point on that line of sight at his distance
        d = aim[:2] - xy
        err = self.steer(a, state, math.degrees(math.atan2(d[0], d[1])))
        cam_h = float(state[2]) - CAMERA_BELOW_M
        tilt = -math.degrees(math.atan2(cam_h - aim[2], max(float(np.hypot(*d)), 0.5)))
        self.tilt(a, tilt)
        self.hold(state, a, tg["at"])
        aimed = abs(err) < PROBE_TOL_DEG and abs(float(state[TILT]) - tilt) < PROBE_TOL_DEG and abs(self.yaw_rate()) < 2.0 \
            and float(np.hypot(*state[4:6])) < 0.3
        pr["aimed"] = (pr["aimed"] if pr["aimed"] is not None else t) if aimed else None
        lp = state[LASER_POINT].copy()
        fresh = tg.get("last_laser") is None or float(np.linalg.norm(lp - tg["last_laser"])) > 1e-6
        tg["last_laser"] = lp
        if (aimed and t - pr["aimed"] >= PROBE_AIM_S and fresh and int(round(state[LASER_STATUS])) == 0 and state[LASER_RANGE] > 0
                and np.hypot(*(lp[:2] - pr["est"])) <= PROBE_NEAR_M):
            pr["hits"].append(np.asarray(lp, float))
            pr["aimed"] = None                             # the next mark is a new aim

    def look_thermal(self, image, cam, t_frame):
        """A thermal main frame taken while a person is being looked at: the warm body nearest his spot, if any.  With a
        close-up of him still shown it completes the report (TZ); with two agreeing thermal looks and no close-up box
        after the close-ups had their chance, the thermal box itself is reported on this frame (TT)."""
        tg = self.target
        if tg is None or tg["arrived"] is None or tg.get("aimed_since") is None or t_frame < tg["aimed_since"]:
            return                                         # not aimed at him when the frame was taken
        tr = tg["track"]
        if tr["reported"] or self.reports >= MAX_REPORTS:
            return
        if tg.get("probe") is not None or tg.get("no_flat"):
            return                                         # the laser is placing him (or could not from here): no report past it
        best = None
        for x0, y0, x1, y1, s in self.boxes("thermal", image):
            if s < THERMAL_SCORE or max(x1 - x0, y1 - y0) < THERMAL_MIN_PX:
                continue
            q = self.on_ground(cam[0], self.pixel_ray(cam, *self.foot(x0, y0, x1, y1), focal=THERMAL_F, cu=320.0, cv=256.0))
            if q is None or not self.maps.plausible(q[:2]):
                continue
            d = float(np.hypot(*(q[:2] - tr["xy"])))
            if d < 2.5 and (best is None or d < best[0]):
                best = (d, (x0, y0, x1, y1), np.asarray(q[:2], float), s)
        tg["tlooks_n"] += 1
        tg.setdefault("zres", []).append("T+" if best is not None else "T-")
        if best is None:
            self.zz_in_hand(tg, tr)
            return
        _, (x0, y0, x1, y1), txy, ts = best
        earlier = [q for q in tg["tlooks"] if np.hypot(*(q - txy)) < LOOKS_AGREE_M]
        tg["tlooks"].append(txy)
        tg["seen"] += 1
        if self.pending is not None:
            return
        lz = tg.get("last_zoom")
        # the close-up shown must be one a report may be made on (steady()): a box look_zoom refused (low-box, a part of
        # him) is not reported because a warm body came in after it; the next close-up decides
        if lz is not None and len(lz) >= 7 and lz[6] and self.zoom_wait is None and np.hypot(*(lz[1] - txy)) < SENSES_AGREE_M:
            kind, image_kind, share, xy = "TZ", "zoom", lz[0], lz[1]           # the close-up of him is still the one shown
            if self.flat_look(lz[2], xy) < FLAT_LOOK_DEG and not tg.get("probed") and tg.get("probe") is None:
                tg["probe"] = {"cam": lz[2], "focal": lz[3], "box": lz[4], "share": share, "kind": kind,
                               "est": np.asarray(xy, float), "hits": [], "stage": 0, "since": None, "aimed": None}
                return                                     # too flat for the ground model: the laser places him
        elif (ALLOW_TT and earlier and ts >= TT_SCORE and tg["zooms"] >= 2 and not tg["looks"] and self.zoom_wait is None
              and x0 >= 3 and y0 >= 3 and x1 <= 637 and y1 <= 509):
            kind, image_kind = "TT", "feed"
            share, xy = ((x0 + x1) / 2 / 640, (y0 + y1) / 2 / 512, (x1 - x0) / 640, (y1 - y0) / 512), np.mean(np.array(earlier + [txy]), axis=0)
        else:
            self.zz_in_hand(tg, tr)
            return
        if any(o is not tr and o["reported"] and np.hypot(*(o["xy"] - xy)) < SAME_REPORTED_M for o in self.tracks):
            tr["reported"], tr["dup"] = True, True
            self.tracks = [x for x in self.tracks if x is not tr]
            self.sync_people()
            return
        lasers = tg["lasers"]
        if lasers:
            lxy = np.median(np.array(lasers)[:, :2], axis=0)
            if np.hypot(*(lxy - xy)) < 0.8:
                xy = 0.5 * (lxy + xy)
        tr["xy"] = tr["pin"] = np.asarray(xy, float)
        tr["pin_side"] = math.atan2(float(cam[0][1] - xy[1]), float(cam[0][0] - xy[0]))
        tr["rays"] = []
        tr["reported"] = True
        self.sync_people()
        self.pending = (image_kind, share, float(xy[0]), float(xy[1]), kind)

    def zz_in_hand(self, tg, tr):
        """The last thermal look is in and showed no warm body on him: two agreeing close-ups wanted only it for a
        colour-only report, and the second of them is still the close-up shown (none was taken since), so the report is
        made on it now (ZZ) instead of on a close-up still to come."""
        lz = tg.get("last_zoom")
        if (not tg.get("zz_ready") or lz is None or len(lz) < 7 or not lz[6] or tg["tlooks_n"] < THERMAL_LOOKS
                or self.zoom_wait is not None or self.pending is not None or tr["reported"] or tg.get("probe") is not None):
            return
        tg["zz_ready"] = False
        share, zxy, zcam, zfocal, zbox = lz[0], lz[1], lz[2], lz[3], lz[4]
        xy = np.mean(np.array([q for q, _s in tg["looks"] if np.hypot(*(q - zxy)) < LOOKS_AGREE_M]), axis=0)   # the shown one too
        tg["zres"].append("ZZ-held")
        lasers = tg["lasers"]
        if lasers:
            lxy = np.median(np.array(lasers)[:, :2], axis=0)
            if np.hypot(*(lxy - xy)) < 0.8:
                xy = 0.5 * (lxy + xy)                         # the laser hit him: it agrees with the close-ups
        if self.flat_look(zcam, xy) < FLAT_LOOK_DEG and not tg.get("probed"):
            tg["probe"] = {"cam": zcam, "focal": zfocal, "box": zbox, "share": share, "kind": "ZZ",
                           "est": np.asarray(xy, float), "hits": [], "stage": 0, "since": None, "aimed": None}
            return                                         # too flat for the ground model: the laser places him
        self.finish_report(tr, xy, share, "ZZ", zcam)

    def report(self, a):
        if self.pending is None or self.held[I_REPORT] > 0.5:
            return
        image, (cx, cy, w, h), e, n, kind = self.pending
        self.kinds.append(kind)
        self.sizes.append(self.last_size)
        a[I_REPORT] = 1.0
        a[I_CLASS], a[I_IMAGE] = 0.0, (1.0 if image == "zoom" else 0.0)
        a[I_CX], a[I_CY], a[I_W], a[I_H] = cx, cy, w, h
        a[I_EAST], a[I_NORTH] = np.clip(e / 100.0, -1, 1), np.clip(n / 100.0, -1, 1)
        a[I_ZOOM] = 0.0
        self.reports += 1
        self.pending = None

    # ------------------------------------------------------------------ steering
    def yaw_rate(self):
        h = self.hist
        if len(h) < 2 or h[-1][0] - h[-2][0] <= 1e-6:
            return 0.0
        return wrap(h[-1][2] - h[-2][2]) / (h[-1][0] - h[-2][0])

    def steer(self, a, state, want):
        """Turn toward a heading with the rate braked: a spin leaves the drone turning at 90 deg/s."""
        err = wrap(want - float(state[HEAD]))
        rate = YAW_KP * err - YAW_KD * self.yaw_rate()
        a[I_TURN] = np.clip(rate / YAW_MAX_DPS, -1.0, 1.0)
        return err

    def home_mask(self):
        """Ground the drone may cross going home: flyable, and off every person found (cleared or not, time is up)."""
        m = self.maps
        if not m.people:
            return m.flyable
        ppl = np.zeros(m.shape, bool)
        r = m.keep_off()
        for x, y in m.people:
            c = m.cell((x, y))
            if r[c] > 0:                                   # where the drone passes 5.5 m over his head he is not in the way
                ppl[c] = True
        return m.flyable & ~binary_dilation(ppl, structure=m.person_k) if ppl.any() else m.flyable

    def hold(self, state, a, point):
        """Stay over a point (wind pushes a hovering drone): a small velocity back to it, at cruise height."""
        err = np.asarray(point, float) - state[POS][:2]
        v = 0.8 * err
        sp = float(np.hypot(*v))
        if sp > 1.5:
            v *= 1.5 / sp
        h = math.radians(float(state[HEAD]))
        a[0] = np.clip((v[0] * math.sin(h) + v[1] * math.cos(h)) / 5.0, -1.0, 1.0)
        a[1] = np.clip((v[0] * math.cos(h) - v[1] * math.sin(h)) / 5.0, -1.0, 1.0)
        rise = CRUISE_M - state[7]
        a[2] = np.clip(rise / (3.0 if rise >= 0 else 2.0), -1.0, 1.0)

    def govern(self, state, a):
        """The last line of defence, on every decision in flight: whatever the planner asked for, the sticks never carry
        the drone to within HARD_EDGE_M of the field's edge or HARD_CROWN_M of a crown circle.  The next two seconds are
        flown ahead with the aircraft's own easing (2 m/s2 per axis); a command that would cross a line is halved, then
        zeroed, and if even a stop would cross it the drone is steered back to safely flyable ground."""
        m = self.maps
        xy = np.asarray(state[POS][:2], float)
        v = np.asarray(state[4:6], float)
        h = math.radians(float(state[HEAD]))
        sn, cs = math.sin(h), math.cos(h)
        cmd = 5.0 * np.array([a[0] * sn + a[1] * cs, a[0] * cs - a[1] * sn])

        def safe(c):
            p, vel = xy.copy(), v.copy()
            for _ in range(8):
                vel = vel + np.clip(c - vel, -0.5, 0.5)
                p = p + 0.25 * vel
                if not m.hard[m.cell(p)]:
                    return False
            return True

        if safe(cmd):
            return
        for scale in (0.5, 0.0):
            if safe(cmd * scale):
                a[0], a[1] = a[0] * scale, a[1] * scale
                return
        cells = np.argwhere(m.flyable)
        if not len(cells):
            a[0] = a[1] = 0.0
            return
        d = m.point(cells[int(np.argmin(np.hypot(*(cells - np.array(m.cell(xy))).T)))]) - xy
        n = float(np.hypot(*d))
        w = d / n * 3.0 if n > 1e-6 else np.zeros(2)
        a[0] = np.clip((w[0] * sn + w[1] * cs) / 5.0, -1.0, 1.0)
        a[1] = np.clip((w[0] * cs - w[1] * sn) / 5.0, -1.0, 1.0)

    def hover(self, state, a):
        a[0] = a[1] = 0.0
        rise = CRUISE_M - state[7]
        a[2] = np.clip(rise / (3.0 if rise >= 0 else 2.0), -1.0, 1.0)

    def fly_to(self, state, a, goal, mask="allowed", near=1.2):
        """Steer along a path to goal (over allowed ground); True once there."""
        m = self.maps
        xy = state[POS][:2]
        goal = np.asarray(goal, float)
        if np.hypot(*(goal - xy)) < near:
            self.hover(state, a)
            return True
        if not m.flyable[m.cell(xy)] and float(np.hypot(*xy)) > 2.0:
            # off flyable ground (carried into the margin beside a crown or the edge): straight back onto it first, the
            # way that gains room fastest; a path from here would begin along the margin and the governor would refuse it
            cells = np.argwhere(m.flyable)
            if len(cells):
                back = m.point(cells[int(np.argmin(np.hypot(*(cells - np.array(m.cell(xy))).T)))]) - xy
                n = float(np.hypot(*back))
                east, north = back / max(n, 1e-6) * min(OFF_GROUND_MPS, n + 0.5)
                h = math.radians(state[HEAD])
                a[0] = np.clip((east * math.sin(h) + north * math.cos(h)) / 5.0, -1.0, 1.0)
                a[1] = np.clip((east * math.cos(h) - north * math.sin(h)) / 5.0, -1.0, 1.0)
                rise = CRUISE_M - state[7]
                a[2] = np.clip(rise / (3.0 if rise >= 0 else 2.0), -1.0, 1.0)
                self.leg = None                            # the path is planned afresh from flyable ground
                return False
        key = (round(float(goal[0]), 2), round(float(goal[1]), 2), m.version, mask)
        if self.leg is None or self.leg[0] != key:
            if mask == "allowed":
                path = m.path(xy, goal, None)
            else:
                # home: over ground that is checked and off everyone found if that way is not too long for the time
                # left, else off the people found, else over the bare field (the edge and the crowns always kept)
                def length(q):
                    return float(sum(np.hypot(*(b - a_)) for a_, b in zip([np.asarray(xy, float)] + q[:-1], q)))
                path = m.path(xy, goal, self.home_mask())  # round the people found: the way the time was kept for
                careful = m.path(xy, goal, m.allowed())    # over checked ground too, if that is hardly longer
                if careful and (not path or length(careful) <= 1.15 * length(path) + 3.0):
                    path = careful
                if not path:
                    path = m.path(xy, goal, m.flyable)
            self.leg = (key, path)
        path = self.leg[1]
        if not path:
            if self.mode != "home":
                self.mode = "next"
            self.hover(state, a)                         # no way: wait (the maps change); the governor keeps every line
            return False
        while len(path) > 1 and np.hypot(*(path[0] - xy)) < 1.0:
            path.pop(0)
        tgt = path[0]
        to_go = tgt - xy
        dist = float(np.hypot(*to_go))
        want = math.degrees(math.atan2(to_go[0], to_go[1])) if dist > 1.0 else float(state[HEAD])
        err = self.steer(a, state, want)
        facing = max(0.25, math.cos(math.radians(min(abs(err), 90.0))) ** 2)
        speed = min(MAX_MPS * facing, math.sqrt(2.0 * BRAKE_MPS2 * max(dist, 0.0)) + 0.3)
        east, north = to_go / max(dist, 1e-6) * speed
        h = math.radians(state[HEAD])
        a[0] = np.clip((east * math.sin(h) + north * math.cos(h)) / 5.0, -1.0, 1.0)
        a[1] = np.clip((east * math.cos(h) - north * math.sin(h)) / 5.0, -1.0, 1.0)
        rise = CRUISE_M - state[7]
        a[2] = np.clip(rise / (3.0 if rise >= 0 else 2.0), -1.0, 1.0)
        return False
