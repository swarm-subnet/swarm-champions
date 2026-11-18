"""Swarm Test Field (cf_test_field) patrol agent.

Takes off from the dock, finds the people on the field, reports each one with a box and the spot he stands on, and
lands back in the dock, never near the field's edge, a tree crown, or the 5 m round anyone it has found.

Flight
- The site map gives the field's edge, the tree crowns and the houses, from this seed's dock, on the first
  observation. The free ground is read from it on a 0.5 m grid: EDGE_CLEAR_M inside the edge and CROWN_CLEAR_M off
  every crown, and, as people are found, off each by the across-the-ground distance the 3D 5 m rule asks at the
  drone's height over the ground he stands on.
- The drone holds just under the 10 m ceiling. It sweeps the field in lanes LANE_M apart along its long axis, joined
  over the free ground, the nose along the way it flies and the camera tilted SEARCH_TILT_DEG, so the ground ahead is
  searched before the drone gets there: over the field's high end the drone is only a few metres above a head.
- Out of time, or done, it flies towards the dock over the free ground and presses return home once the dock's own
  straight flight home keeps clear of the edge, the crowns and everyone found.

Seeing
- The ground is a plane with the field's known slopes, bent to what the downward sensor reads as the drone flies
  (and to laser readings that agree with it).
- Every new colour frame goes through the feed detector. Each person box is placed on the ground through the camera's
  pose when the frame was taken; nearby boxes are one candidate.
- A candidate in the frame gets a close-up. The close-up's detector and box refiner give how sure it is and his box;
  sure enough, and with his spot known well enough (a steep enough look, ground read near him), he is reported on the
  close-up with that spot. Not sure, or his spot is not good enough, the drone visits a viewpoint near him.

The flight-path following (_progress, _ahead) and the free-ground search are adapted from the PR's reference pilot
(MIT, Swarm). Everything is in this one file: the packager ships only drone_agent.py and model artifacts.
"""

from __future__ import annotations

import heapq
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {"feed": "feed_colour.onnx", "zoom": "zoom.onnx", "refiner": "zoom_box.onnx"}
ORT_THREADS = int(os.environ.get("TF_AGENT_ORT_THREADS", "2"))    # 2 when validated; 1 for mass patrols on one box
FIELD_FILE = os.path.join(HERE, "field_data.npz")      # the field's surveyed outline and ground (tools/tf_site.py)
MATCH_TOL_M = 0.05               # the site map's edge matches the survey's when every corner agrees this well

# --------------------------------------------------------------------------------------------------- contract
ACTION_DIM = 26
A_FWD, A_RIGHT, A_UP, A_TURN, A_GIMBAL = 0, 1, 2, 3, 4
A_THERMAL, A_NIGHT_MODE, A_NIGHT_VISION = 5, 6, 7
A_ZOOM, A_ZOOM_LENS, A_ZOOM_CX, A_ZOOM_CY, A_ZOOM_W, A_ZOOM_H = 8, 9, 10, 11, 12, 13
A_REPORT, A_REPORT_CLASS, A_REPORT_IMAGE = 14, 15, 16
A_REPORT_CX, A_REPORT_CY, A_REPORT_W, A_REPORT_H = 17, 18, 19, 20
A_TAKE_OFF, A_RETURN, A_CANCEL = 21, 22, 23
A_EAST, A_NORTH = 24, 25

S_HEADING, S_HEIGHT, S_TILT, S_TIME_LEFT, S_WIND = 3, 7, 8, 9, 11
S_DOCK_WIND = 13
S_LASER_RANGE, S_LASER_STATUS, S_GROUND, S_DOWN_OK = 14, 18, 19, 20
S_EDGE, S_INSIDE, S_PHASE, S_FEED, S_FRAME_AGE = 21, 22, 23, 24, 25
S_ZOOM_LENS, S_ZOOM_AGE, S_ZOOMS_LEFT = 28, 29, 30
DOCKED, TAKING_OFF, FLYING, RETURNING, LANDING, LANDED = range(6)

HORIZON_S = 180.0
REPORT_REACH_M = 100.0
MAX_HORIZONTAL_MPS = 5.0
MAX_CLIMB_MPS = 3.0
MAX_DESCENT_MPS = 2.0
MAX_YAW_DEG_S = 90.0
MAX_REPORTS = 20                 # the judge's own cap per patrol

FEED_SHAPE = (480, 640)
FEED_DIAG_DEG = 82.0
ZOOM_SHAPE = (480, 640)
ZOOM_DIAG_DEG = {3: 35.0, 7: 15.0}
CAMERA_OFFSET = np.array([0.158, 0.0, -0.04])          # wide camera from the body centre: forward, left, up
SENSOR_DROP_M = 0.062                                  # the downward sensor under the body centre

# --------------------------------------------------------------------------------------------------- flight
CRUISE_Z = 9.8                   # above the dock: the ceiling is 10 m and a full climb stops within 0.15 m of it
EDGE_CLEAR_M = 3.4 # the free ground starts this far inside the edge (the rule trips at 1 m)
EDGE_GUARD_M = 2.6 # nearer the edge than this, the drone is pushed back in
EDGE_STOP_M = 2.3 # and towards it never faster than it can brake 1.8 m short of it (the rule: 1 m)
FRAME_ZOOM_SPLIT = True # True: one heavy model read per 0.5 s window (the budget is 500 ms for its 5 decisions,
                                 # and a frame read and a close-up read are ~250 ms each on the reference machine): a
                                 # waiting close-up is read and that window's frame only updates the coverage; a
                                 # close-up landing after the frame was read waits for the next window
WIND_GUARD_M_PER_MPS = 0.25 # > 0: the edge and crown margins grow this much per m/s of the wind estimate (real field,
                                 # seed 1044: strong wind carried the drone over the 1 m edge line past the 1.8 m stop)
CROWN_CLEAR_M = 2.0              # the free ground keeps this far off every crown circle (a crown is a crash)
CROWN_GUARD_M = 1.2
CROWN_STOP_M = 0.8               # towards a crown never faster than the drone can brake this far short of it
GUARD_BRAKE_MPS2 = 1.5
GUARD_LAG_S = 0.4                # the drone's speed follows a command about this late: the guard leaves room for the
                                 # ground covered meanwhile (stand-in seed 3024 came within 0.04 m of a crown)
REVERSE_MPS = 1.5                # faster than this, a course turned more than about 80 degrees from the way the drone
REVERSE_COS = 0.2                # moves is flown only after braking straight, nose held (a yawing turn swung the
                                 # drone round into a crown: stand-in seed 2125)
RETURN_EDGE_M = 2.2              # the dock's straight flight home keeps this far inside the edge
RETURN_CROWN_M = 1.2             # and this far off every crown
RETURN_FROM_STOP = True          # return home is pressed only once the line home from where the drone will stop
                                 # (its speed now, GUARD_LAG_S, GUARD_BRAKE_MPS2) is clear as well
TAKEOFF_LULL = 0.0               # OFF, tested no use: 800 new windy seeds, at once 3 crashes, lull 1.0 6, fresh-push 5
                                 # (timing only re-rolls which seeds crash; the climb-window wind is the same at any time)
                                 # > 0: in a strong wind (the dock gauge's first reading, which is the seed's steady
                                 # wind at the cups, at least TAKEOFF_LULL_FROM) take-off waits for a fresh gauge
                                 # reading at most this share of the first one, at most TAKEOFF_WAIT_MAX_S: the lids
                                 # open for 4 s and the climb out between them is where strong gusts crash the drone
TAKEOFF_LULL_FROM = 3.0          # m/s at the cups (strong seeds read 2.2-5.7, light 0.6-2.8)
TAKEOFF_DELAY_S = 0.0            # > 0: take-off is pressed no earlier than this (take-off timing studies)
TAKEOFF_WAIT_MAX_S = 8.0
TAKEOFF_FRESH_S = 2.05           # the gauge is pushed every 2 s: after this long a reading is a new one
TAKEOFF_FRESH_PUSH = False       # True: only a reading that has just changed counts (pushed within
                                 # TAKEOFF_PUSH_AGE_S); one up to 2 s old can miss a rising wind
                                 # (seed 9340: pressed on a stale steady reading, climbed into 9 m/s)
TAKEOFF_PUSH_AGE_S = 0.25
SAFE_M = 5.0                     # the rule: never within 5 m in 3D of a 1.8 m tall person
PERSON_H_M = 1.8
CLEAR_MARGIN_M = 1.3             # over the rule, for where we think he stands and how the drone swings
POCKET_ESCAPE = True             # True: boxed in by keep-away circles (a route fails and the free ground the drone can
                                 # reach is under POCKET_CELLS cells), the circles keep only ESCAPE_MARGIN_M over the rule,
                                 # for the route and the guard alike, until it is out (13304: boxed in against the edge by
                                 # three reported men from 52 to 127 s; a 1.7 m gap opens between two of them at 0.8 m)
                                 # 41 seeds with 3+ failed routes in a row: acted in 3, mistakes 1 -> 0, nearest 6.03 m
POCKET_CELLS = 600               # 150 m2 of 0.5 m cells
ESCAPE_MARGIN_M = 0.8
ESCAPE_MPS = 1.5                 # slow while the margin is small
ESCAPE_MAX_S = 20.0              # the full margin comes back after this at the latest
ESCAPE_MAX = 2                   # escapes a patrol may make
PERSON_SLOPE_M = 0.2             # he stands on the highest ground under his body, a little over his spot's
LANE_M = 13.0
LANE_INSET_M = 6.0               # the outer lanes this far in from the field's extent across its long axis
LANE_WIDE_M = 16.0               # > 0: lanes this far apart when the dock stands low (LANE_WIDE_DOCK_Z): there much of
                                 # the field is high ground under the 10 m ceiling, the drone flies it slowly and the
                                 # patrol runs out of time (87 seeds that did: 16 m lanes, mistakes 8 -> 1, home on
                                 # time 84 -> 13, coverage 0.993 -> 0.998)
                                 # 88 low-dock v5 seeds: 4 -> 1 (1 knock-on: 11126, no close lane pass, no viewpoint)
LANE_WIDE_DOCK_Z = 3.0           # the dock's height (its survey ground + DOCK_ABOVE_GROUND_M) under which lanes widen:
                                 # 22% of seeds, 139 of the 162 patrols that ran out of time
DOCK_ABOVE_GROUND_M = 0.57       # the dock's z over the survey ground under it (0.55-0.61 over 520 seeds)
CRUISE_MPS = 4.8                 # under the 5 m/s limit; frames 2.4 m apart, each sees 15 m ahead
LOW_MPS = 2.5                    # over ground this close under the drone, slower: a head may be near
LOW_AGL_M = 6.5
BRAKE_MPS2 = 1.2
CORNER_DEG = 30.0
CORNER_MPS = 1.0
CARROT_M = 3.5
NEAR_CARROTS_M = (2.0, 1.0)
ARRIVE_M = 1.2
HEIGHT_GAIN = 1.2
HEADING_GAIN = 2.0
TURNING_MPS = 1.0
SEARCH_TILT_DEG = -55.0          # the camera ahead and down: from 10 m it sees from about 1.5 m to 20 m ahead
DANGER_TILT_DEG = -45.0          # over high ground the camera looks farther ahead,
DANGER_MPS = 2.2                 # and the drone slows: it brakes inside 3 m once a man is seen 9 m or more ahead
DANGER_LOOK_M = 10.0
DANGER_MARGIN_M = 0.5
DANGER_GRADED = True             # True: over high ground the speed fits what a man standing there needs instead of the
                                 # flat LOW_MPS / DANGER_MPS caps: his horizontal clearance plus the stopping distance
                                 # (DANGER_REACT_S, DANGER_BRAKE_MPS2) within DANGER_SEE_M. Real field: patrols that run
                                 # out of time (all with the dock under 4 m) spend ~76 s at 2 m/s 5-8 m over the ground
DANGER_SEE_M = 11.0              # 95% of standing people on high ground were first seen farther than 11.9 m (crouching
                                 # 13.2, lying 14.4; 2,336 real-field patrols)
DANGER_REACT_S = 1.2             # a frame's age plus the control lag
DANGER_BRAKE_MPS2 = 1.3
DANGER_GRADED_LOOK_M = 13.0      # the ground looked at ahead: where someone not yet seen may stand
GUARD_EDGE_LAST = True           # True: the edge and crown holds act after the person zones, so a way out of a zone
                                 # can never cross them
ZONE_PUSH_MAX_MPS = 1.5          # > 0: the push out of a person zone the drone finds itself in is at most this fast
ZONE_INSIDE_FIRST = True         # True: the push out of a zone the drone is in comes before the holds of the zones it
                                 # is outside, so it can never carry the drone into another man's zone. Both 5 m
                                 # breaches on record (3141, 10062) were a shadow's zone pushing the drone into the real
                                 # man's. 242 seeds where the drone was inside a zone: mistakes 5 -> 2, breaches 1 -> 0,
                                 # closest pass 4.25 -> 5.73 m (cost: 4713 boxed in on its way home, not landed); 520
                                 # fresh paired seeds: every outcome the same, 12 flights changed, closest 5.62 -> 5.75 m
UNSURE_BOX_MIN = 0.85             # > 0: a look under UNSURE_BOX_SCORE reports only with a box at least this share of the
UNSURE_BOX_SCORE = 0.85          # size the wide frame predicts (real field: 5 bad boxes in 184 reports under 0.85, 0 in
                                 # 16,444 at 0.85 and over; the bad ones boxed 0.65-0.73 of the man)
UNSURE_BOX_MAX = 1.3             # > 0: and no more than this share (seed 7472: 0.83, box 1.38x, overlap 0.49, judged
                                 # no_object and he was missed; under 0.85 and over 1.3x: 3 bad, 5 valid)
LYING_MIN_M = 1.1                # > 0: a box this much wider than tall (LYING_ASPECT), a lying man's, reports only when
                                 # at least this long at its range (40,728 reports: wide and under 1.1 m, 7 bad, 6 valid)
                                 # 9 seeds where such a report was made: mistakes 4 -> 0 (12473, 13069), nothing new
LYING_ASPECT = 1.5
BOX_HOLD_VISIT = 0.8             # > 0: a look this sure held only for its box's size gets a visit (14278: 0.805 held as
                                 # small at 58 s from 37 m; the sweep never came back and time ran out). 153 seeds with
                                 # such a hold: acted in 135, mistakes 4 -> 1, false 1 -> 0 (14278, 13292), nothing new
UNKNOWN_BOX_SCORE = 0.88         # > 0: a look with no wide-frame box to size it against reports only this sure or
                                 # surer (seed 11408: 0.862, box twice his width, judged no_object and he was missed)
                                 # Every bad unchecked report scored under 0.88 (8384, 9433, 11377, 11408); 0 bad in
                                 # 1,195 at 0.88+; re-flown 4 mistakes -> 0
HOME_MPS = 2.2                   # the dock's own flight home averaged this from 19 m out (return to landing 17 s)
RETURN_RESERVE_S = 14.0          # the landing from the cruise height and the pad, with slack
HOME_SLACK_S = 6.0

# --------------------------------------------------------------------------------------------------- seeing
OBJECT_HEIGHT_M = 0.5            # where a box's centre sits over the ground: lying, crouching and standing people
FEED_MIN_SCORE = 0.35            # a person box this sure on a colour frame becomes or feeds a candidate
ZOOM_PERSON_SCORE = 0.30         # a person box on a close-up this sure is weighed at all
ZOOM_SECOND = True               # True: a second sure man in a close-up, ZOOM_SECOND_M or more from the one the report
                                 # goes to, is a candidate of his own, never the one asked about (4327)
ZOOM_SECOND_SCORE = 0.8
ZOOM_SECOND_M = 2.0
REPORT_SURE = 0.80               # one close-up this sure reports him
LOOK_AGAIN = 0.30
PAIR_MIN = 0.62                  # or two close-ups each this sure (stand-in: people's second look 0.62-0.90,
                                 # no false candidate's second look over 0.57)
LOOKS_MAX = 5                    # close-ups a candidate may get
EMPTY_DROP = 2                   # close-ups that find nobody where he should be before he is dropped
WEAK_LOOKS = 3                   # a candidate this many close-ups found, none at WEAK_MAX or more, is dropped
WEAK_MAX = 0.5
VISIT_LOOK_MIN = 0.45            # a visit is flown only for a candidate a close-up found at least this sure
ASSOC_M = 2.0                    # a box this near a candidate is him (people stand 3 m apart or more)
MERGE_M = 1.5                    # two live candidates this close are one man
MERGE_SURVIVOR = True            # True: a box near a candidate merged into another goes to the one he was merged into
                                 # when within MERGE_M of him, and he keeps the sure sight lines (9379: the merged one kept
                                 # the man's spot and took every box of him, 0.97 from overhead, as a duplicate; the
                                 # survivor never saw him again); only for a man with nobody else known within
                                 # MERGE_ALONE_M (12457: two men 3.2 m apart, a box of one pulled the other's candidate)
MERGE_ALONE_M = 5.0              # 335 seeds where boxes went to a merged duplicate: acted in 177, mistakes 3 -> 1
                                 # (9379, 9578 fixed; 13304 is the boxed-in drone), time-outs 27 -> 22
VISIT_FOLLOW_MERGE = True       # True: a visit whose man is merged into his duplicate goes on for the survivor (17659)
VISIT_FOLLOW_ZOOMS = True       # True: ... and keeps its own close-ups however many the merge added to him
SEEN_APART_SCORE = 0.8           # > 0: two candidates boxed in one wide frame, both this sure and SEEN_APART_M apart,
                                 # are two men: never merged, never a repeat of each other (records: 26 such pairs were
                                 # two men, 0 one man; seed 11567: a lying man 3.4 m from a reported one merged as him)
SEEN_APART_M = 2.5               # at 1.5 m a man's second, shadow-like box 2.19 m off reported a repeat (10790)
SEEN_APART_REVIVE = True         # True: a duplicate later boxed apart from the man he was taken for is a man again
                                 # (11567: a lying man 3.4 m from a reported one, merged at his first 0.61 box)
                                 # 11 seeds where it acts: mistakes 2 -> 0 (7266, 11567), nothing new
SIGHTLINE_VIEW = True            # True: with no viewpoint found, fly in along a line the wide camera saw him surely
                                 # from (seed 11126: home 'done' with 93 s left, his steeper look never taken)
SIGHTLINE_SCORE = 0.8
LASER_FIX_M = 0.0                # > 0: a laser hit on a man's body (LASER_BODY_M over the ground) this near a man
                                 # waiting for a steeper look fixes his place, so his next sure close-up reports (11389:
                                 # the laser hit him at 1.21 m at 94 s; two visits later he was still unreported)
LASER_BODY_M = (0.3, 2.0)
VIEW_STEEP_MARGIN = 0.85         # of that distance: the drone hovers up to 0.5 m off its spot (11389: refused at 7.2 m)
VIEW_STEEP = False               # True: a man waiting for a steeper look gets a viewpoint only as far as that look
                                 # allows (or the nearest ring outside his zone when even that is too far)
LEFTOVER_VISITS = True           # True: with the sweep and fill done and LEFTOVER_MIN_S to spare besides the way home,
                                 # a man found but unresolved gets up to LEFTOVER_EXTRA more visits instead of the drone
                                 # going home 'done' (seed 3299: found during the last fill, home 70 s early, missed)
LEFTOVER_EXTRA = 2
FILL_VISITS = False              # True: once the sweep is done, a man a close-up found sure enough whose report waits for a
                                 # look from a second place gets his visit before more gap-fill (13304: found at 53 s in
                                 # the fill, never zoomed again; his visit came only when the fill ran out at 138 s)
LEFTOVER_MIN_S = 12.0            # at 30 s it acted in 1 of 156 seeds; at 12 s it fixed 3299, nothing new
WEAK_RUN_LOOKS = 2               # > 0: after this many close-ups that all scored under WEAK_RUN_MAX, of a man the wide
                                 # frame never boxed WEAK_RUN_MAX or more, one sure look waits for a second (15761: 0.33,
                                 # 0.58, 0.58 then 0.909 on nothing; 56,209 reports: the only one so made, and wrong)
                                 # 11 seeds where it can act: acted in 2, mistakes 1 -> 0, nothing new
WEAK_RUN_MAX = 0.6
WEAK_RUN_EMPTY_FEED = 0.5        # > 0: empty close-ups count as weak ones too when the wide frame never boxed him this
                                 # surely (16449: wide 0.41, close-ups empty and 0.44, then 0.92 on nothing, 26 m off)
                                 # v12's 500 batch seeds: changed 1 flight, mistakes 9 -> 8 (16449), nothing new
ZOOM_EDGE_DETS = True            # True: a man the wide frame boxes in this very frame may get his close-up there even
                                 # when his stored spot projects just outside the frame window (16470: on high ground
                                 # at the frame's top corner, boxed 0.82-0.93 from 96 s, never zoomed; time ran out)
                                 # v12's 500 batch seeds: changed 278 flights, mistakes 9 -> 7 (16470, 16449), nothing new
LEFTOVER_FEED_MIN = 0.7          # > 0: in leftover time a live man the wide frame boxed this surely gets a visit too,
                                 # however weak his close-ups (14972: a lying man boxed 0.74 wide, 0.31 close, never
                                 # visited; home 'done' with 70 s left). 185 seeds where it can act: changed 9
                                 # flights, mistakes 1 -> 0 (14972), nothing new
                                 # v7 (sightline + seen-apart 2.5 m + oversized-box hold) on the 57 seeds where they
                                 # act, paired with v6: mistakes 12 -> 5, false 1 -> 0, nothing new
SAME_PERSON_M = 2.5              # a sure man this near a reported one is him (people stand 3 m apart; two
                                 # reports of one man came 2.05-2.2 m apart, of two men never under 2.7 m)
DOCK_CLEAR_M = 6.0               # nobody stands within 8 m of the dock: a box nearer is the dock or a mistake
EDGE_IN_M = 0.3                  # nobody stands within 1.5 m of the edge: a box outside it by this much is not one
CROWN_IN_M = 0.6                 # nobody stands under a crown: a box this deep inside a crown circle is the tree
ZOOM_TIMEOUT = 6                 # decisions to wait for a close-up before giving up on it
ZOOM_FAR_M = 15.0                # farther than this the 7x lens, nearer the 3x
VIEW_MIN_M = 3.5                 # a viewpoint at least this far from him across the ground
BOX_RATIO_MIN = 0.50             # a close-up box outside these shares of the size the wide frame's box predicts is
BOX_RATIO_MAX = 1.8              # part of him or two men together: on 416 generated close-ups IoU >= 0.5 held for 20%
                                 # under 0.65, 97-98% from 0.8 to 1.8, 82-88% over 1.8
POS_TAN_MIN = 0.65               # his spot is good enough once a look came down at least this steeply (tan of the
POS_TAN_BLIND = 1.0              # angle below the horizon; close-up spots: max error 0.83 m from 0.7 up), or this
                                 # steeply where the ground under him was never read
VISIT_HOLD_S = 4.0               # at a viewpoint this long without a report, the visit ends
VISITS_MAX = 2
VISIT_ZOOMS = 4                  # close-ups a visit adds to what a candidate may get
SHADOW_TOL = 0.10                # boxes whose place from the drone, per metre of its height, agrees this well
SHADOW_MOVED_M = 2.0             # while the place itself moved this far, are the drone's own shadow
SHADOW_VOTES = 3                 # boxes that agree so before the shadow's place is taken as learnt
SHADOW_STICKY = True             # True: a learnt shadow place moves elsewhere only with more boxes than now agree with it
                                 # (the 300-box memory lets a stale first guess fade). 66 v5 seeds whose estimate jumped:
                                 # mistakes 2 -> 1 (11389 fixed), nothing new
SHADOW_REPLAN = True             # True: a visit whose man the shadow keep-out blocks at the viewpoint picks a new one, once
SHADOW_VETO_M = 1.6              # a box this near where the shadow falls is the shadow (more when high)
SHADOW_MAX = 3.0                 # the shadow's place per metre of height is no farther (sun at least 18 deg up)
SHADOW_KEEP_M = 2.5              # no close-up on a man while the drone's shadow falls this near him, and no viewpoint
                                 # that puts it there (the veto would throw his look away, or the box may be the shadow)
SIGHT_CROWN_M = 0.3              # a close-up counts for a report only if the line to him passes this far off crowns
FEED_SIGHT_MIN = 0.8             # > 0: a man the wide frame boxes this surely is in sight in fact, whatever the crown
                                 # circles say: he may get a close-up from it, and it counts (seed 11040: boxed at 0.97
                                 # for 4 s, every close-up refused as 'crown in line', never reported)
                                 # 52 seeds where it acts: mistakes 9 -> 5, false reports 1 -> 0; 592 fresh pairs with
                                 # the unchecked-box rule: 7 -> 5, false 1 -> 0, 90% of flights unchanged
STATIC_MOVED_M = 2.5             # a man is reported only once two looks from places this far apart (or heights
STATIC_HEIGHT_M = 2.0            # this different) put him on the same spot: he stands still, the shadow does not
STATIC_SAME_M = 1.5
STILL_VISIT_S = 0.0              # > 0: a man sure but waiting this long for a look from a second place gets a visit
                                 # (real field, seed 3299: three lying men seen sure while the drone hovered for
                                 # another were never looked at again and went unreported)
KNOWN_FIGURE_M = 1.0             # a report whose spot lies this near a known figure of the field (survey frame; 0 = off)
KNOWN_FIGURE_CORE_M = 0.30       # no report at all with its spot this near a known figure (its false reports sat
                                 # 0.02-0.21 m off it, looks up to 0.944; people near it were reported 0.37 m or more off)
KNOWN_FIGURE_SURE = 0.90         # needs one close-up this sure; the two-look rule does not count there. Real field:
KNOWN_FIGURES = ((8.3, -28.2),)  # every false report of the release candidate (10 in 2,340 fresh seeds) sat within
                                 # 0.21 m of (8.3, -28.2). On; 9 of the 10 gone, no person lost (guard 7000-7519 paired,
                                 # 91 re-flown seeds); its looks reach 0.916 (2 of 457 >= 0.90), people within 1 m of it
                                 # were reported at 0.902-0.977, so the bar stays 0.90
CROWN_LOOK_VISIT = 0.0           # > 0: a close-up this sure seen only past a crown makes him worth a visit once the
                                 # sweep is done (blocked looks are not looks, so he stayed unworthy)
CROWN_VISIT_MIN = 0.0            # > 0: a close-up this sure that saw him only past a crown gets a visit, whose
                                 # viewpoint sees him clear of every crown (seeds 3078 and 1018 went unreported)
CROWD_CHECK = True              # no report on a close-up box that also holds where someone else known stands
CROWD_SAME_M = 2.5               # another candidate this near is the same man, not a second one
REPEAT_VIEW_FAR_M = 15.0         # ... only for close-ups of him from farther than this
REPEAT_VIEW_M = 1.0              # > 0: a reported man counts as inside a close-up box when his place, give or take
                                 # this many metres (projected at his range), is (repeat reports 11270, 11776)
                                 # 156 seeds: false 2 -> 0 (both repeats), nothing new, 151 flights unchanged
CROWD_H_M = 0.4                  # a second man's middle over his ground must not fall inside the box reported: the
CROWD_VISIT_LOOKS = 2            # box may hold both (two people lined up in an angled close-up); after this many such
CROWD_SEP_DEG = 15.0             # looks a viewpoint where the two stand at least this far apart across the view
CROWD_NEAR_M = 10.0
INNER_SCORE = 0.5                # a box holding a sure person box (this score or more) mostly inside it, the inner box
INNER_COVER = 0.6                # this share of its area inside and no more than INNER_AREA of the outer box's area, is
INNER_AREA = 0.6                 # two people boxed as one: the inner man is reported on his own box (seed 2956)
ZOOM_RESERVE = 20                # close-ups kept for after the first half of the patrol
FILL_RINGS_M = (5.0, 7.0, 9.0)   # a field cell left unseen is looked at from this far across the ground
FILL_HOLD_S = 2.5                # at its viewpoint this long without seeing it, the cell is left
FILL_MPS = 3.0
COVER_PIXEL_M = 0.5 / 6.0        # the coverage rule: a person 6 pixels across
COVER_HEIGHT_M = 20.0
EVENT_CAP = 1500


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    a, b = poly, np.roll(poly, -1, axis=0)
    x, y = pts[:, :1], pts[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1]))
    return cross.sum(axis=1) % 2 == 1


def _edge_distance(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    a = poly
    ab = np.roll(poly, -1, axis=0) - a
    out = np.empty(len(pts))
    for k in range(0, len(pts), 4096):
        rel = pts[k:k + 4096, None, :] - a[None, :, :]
        t = np.clip((rel * ab).sum(axis=2) / (ab * ab).sum(axis=1), 0.0, 1.0)
        out[k:k + 4096] = np.min(np.linalg.norm(rel - t[..., None] * ab, axis=2), axis=1)
    return out


def _vfov_half_tan(diag_deg: float, rows: int, cols: int) -> float:
    return math.tan(math.radians(diag_deg / 2.0)) * rows / math.hypot(rows, cols)


def _segment_gap(point, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(point - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.hypot(*(a + t * ab - point)))


def _progress(route: np.ndarray, index: int, point: np.ndarray) -> int:
    """The segment of a route the drone is on: the nearer of the next two, moved past every corner within ARRIVE_M."""
    near = route[index:index + 3]
    index += int(np.argmin([_segment_gap(point, p, q) for p, q in zip(near[:-1], near[1:])] or [0]))
    while index < len(route) - 2 and float(np.hypot(*(route[index + 1] - point))) < ARRIVE_M:
        index += 1
    return index


def _ahead(route: np.ndarray, index: int, point: np.ndarray, reach: float):
    """The point reach metres further along the route but never past the next sharp bend, how far the route still
    runs, and how far that bend lies."""
    a, b = route[index], route[min(index + 1, len(route) - 1)]
    ab = b - a
    t = float(np.clip(np.dot(point - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    here = a + t * ab
    # Repeated points would hide a bend between two zero-length steps (stand-in seed 2480: no corner slow-down, the
    # drone swung wide into a crown): keep only points that move on.
    corners = [here]
    for p in route[index + 1:]:
        if float(np.hypot(*(p - corners[-1]))) > 0.05:
            corners.append(p)
    if len(corners) == 1:
        corners.append(route[-1])
    gone, carrot, bend = 0.0, corners[-1], math.inf
    for k in range(len(corners) - 1):
        step = float(np.hypot(*(corners[k + 1] - corners[k])))
        if carrot is corners[-1] and gone + step >= reach:
            carrot = corners[k] + (corners[k + 1] - corners[k]) * ((reach - gone) / max(step, 1e-9))
        gone += step
        if bend == math.inf and k + 2 < len(corners):
            d1, d2 = corners[k + 1] - corners[k], corners[k + 2] - corners[k + 1]
            norms = float(np.hypot(*d1) * np.hypot(*d2))
            if norms > 0.0 and float(np.dot(d1, d2)) < norms * math.cos(math.radians(CORNER_DEG)):
                bend = gone
                if bend < reach:
                    carrot = corners[k + 1]
    return np.asarray(carrot, dtype=float), gone, bend


def _path_length(points) -> float:
    pts = np.asarray(points, float)
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def _stop_speed(room: float, react_s: float, brake_mps2: float) -> float:
    """The speed from which the drone stops within `room` metres: react_s at that speed, then braking."""
    if room <= 0.0:
        return 0.0
    return brake_mps2 * (-react_s + math.sqrt(react_s * react_s + 2.0 * room / brake_mps2))


# --------------------------------------------------------------------------------------------------- models

class Detector:
    """One ONNX CenterNet-style model: heatmaps for person and dog, box size and offset, on a grid 4x coarser."""

    STRIDE = 4

    def __init__(self, path: str, scale: float = 1.0):
        self.scale = scale
        self.session = None
        if os.path.exists(path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = ORT_THREADS
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
            except Exception:
                self.session = None

    @property
    def ready(self) -> bool:
        return self.session is not None

    def __call__(self, img: np.ndarray, min_score: float = 0.2):
        """Boxes on the image, in its own pixels: (class, score, cx, cy, w, h); class 0 person, 1 dog."""
        if self.session is None:
            return []
        x = np.asarray(img, dtype=np.float32)
        if x.ndim == 2:
            x = x[:, :, None]
        if self.scale != 1.0:
            step = int(round(1.0 / self.scale))
            h, w = x.shape[0] // step * step, x.shape[1] // step * step
            x = x[:h, :w].reshape(h // step, step, w // step, step, x.shape[2]).mean(axis=(1, 3))
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        outs = self.session.run(None, {"x": x})
        heat, size, off = outs[0][0], outs[1][0], outs[2][0]
        peak = heat.copy()
        padded = np.pad(heat, ((0, 0), (1, 1), (1, 1)), constant_values=0.0)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                peak = np.maximum(peak, padded[:, dy:dy + heat.shape[1], dx:dx + heat.shape[2]])
        keep = (heat >= peak) & (heat >= min_score)
        out = []
        for c, iy, ix in zip(*np.nonzero(keep)):
            s = float(heat[c, iy, ix])
            cx = (ix + float(off[0, iy, ix])) * self.STRIDE / self.scale
            cy = (iy + float(off[1, iy, ix])) * self.STRIDE / self.scale
            w = math.exp(float(np.clip(size[0, iy, ix], -2.0, 7.0))) / self.scale
            h = math.exp(float(np.clip(size[1, iy, ix], -2.0, 7.0))) / self.scale
            out.append((int(c), s, cx, cy, w, h))
        out.sort(key=lambda d: -d[1])
        return out


def _area_weights(start: float, side: float, size: int, n: int):
    step = side / size
    edges = start + np.arange(size + 1) * step
    lo, hi = max(0, int(math.floor(edges[0]))), min(n, int(math.ceil(edges[-1])))
    if hi <= lo:
        return np.zeros((size, 0), np.float32), lo, lo
    j = np.arange(lo, hi, dtype=np.float64)
    cover = np.minimum(edges[1:, None], j[None, :] + 1.0) - np.maximum(edges[:-1, None], j[None, :])
    return (np.clip(cover, 0.0, None) / step).astype(np.float32), lo, hi


def area_crop(img: np.ndarray, x0: float, y0: float, side: float, size: int) -> np.ndarray:
    """A square crop of side pixels from (x0, y0), averaged by area to size x size; black outside the image."""
    return _area_resample(img, x0, y0, side, size)


def _area_resample(img: np.ndarray, x0: float, y0: float, side: float, size: int) -> np.ndarray:
    wy, ylo, yhi = _area_weights(y0, side, size, img.shape[0])
    wx, xlo, xhi = _area_weights(x0, side, size, img.shape[1])
    sub = np.asarray(img[ylo:yhi, xlo:xhi], dtype=np.float32)
    out = np.empty((size, size, img.shape[2]), np.float32)
    for c in range(img.shape[2]):
        out[:, :, c] = wy @ sub[:, :, c] @ wx.T
    return out


class BoxRefiner:
    """The close-up box refiner: a square crop around a person box, read again for where his tight box's edges lie."""

    SIZE = 96
    CROP_SCALE = 1.6

    def __init__(self, path: str):
        self.session = None
        self.size = self.SIZE
        if os.path.exists(path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = ORT_THREADS
                opts.inter_op_num_threads = 1
                self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
                shape = self.session.get_inputs()[0].shape
                if isinstance(shape[-1], int):
                    self.size = int(shape[-1])
            except Exception:
                self.session = None

    @property
    def ready(self) -> bool:
        return self.session is not None

    def __call__(self, img: np.ndarray, cx: float, cy: float, w: float, h: float, rounds: int = 2):
        if self.session is None:
            return cx, cy, w, h
        img = np.asarray(img, dtype=np.float32)
        for _ in range(rounds):
            side = float(np.clip(self.CROP_SCALE * max(w, h), 48.0, 480.0))
            x0, y0 = cx - side / 2.0, cy - side / 2.0
            if self.size != self.SIZE:
                patch = area_crop(img, x0, y0, side, self.size)
            else:
                idx = np.floor(x0 + (np.arange(self.SIZE) + 0.5) * side / self.SIZE).astype(int)
                idy = np.floor(y0 + (np.arange(self.SIZE) + 0.5) * side / self.SIZE).astype(int)
                okx, oky = (idx >= 0) & (idx < img.shape[1]), (idy >= 0) & (idy < img.shape[0])
                patch = img[np.clip(idy, 0, img.shape[0] - 1)][:, np.clip(idx, 0, img.shape[1] - 1)].copy()
                patch[~oky] = 0.0
                patch[:, ~okx] = 0.0
            out = self.session.run(None, {"x": np.ascontiguousarray(patch.transpose(2, 0, 1)[None])})[0][0]
            l, t, r, b = (x0 + out[0] * side, y0 + out[1] * side, x0 + out[2] * side, y0 + out[3] * side)
            if r - l < 2.0 or b - t < 2.0:
                break
            cx, cy, w, h = (l + r) / 2.0, (t + b) / 2.0, r - l, b - t
        return cx, cy, w, h


# --------------------------------------------------------------------------------------------------- the ground

class Ground:
    """The terrain under the field in metres above the dock: a plane, starting from the field's known slopes (about
    9 % down to the east and 6 % down to the south) and fitted to the readings, bent by a smoothed map of what the
    readings say beyond it."""

    CELL = 1.0
    SIGMA_CELLS = 3.0
    SLOPE_PRIOR = (-0.09, 0.06)
    RIDGE = 500.0
    SHRINK = 0.3

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, float)
        self.shape = tuple(int(v) for v in np.ceil((np.asarray(hi, float) - self.lo) / self.CELL) + 1)
        self.sum = np.zeros(self.shape)
        self.cnt = np.zeros(self.shape)
        self.coef = np.array([-0.6, self.SLOPE_PRIOR[0], self.SLOPE_PRIOR[1]])
        self.res = np.zeros(self.shape)
        self.conf = np.zeros(self.shape)
        self.dirty = False
        self.samples = 0
        k = int(3 * self.SIGMA_CELLS)
        xs = np.arange(-k, k + 1)
        self.kernel = np.exp(-0.5 * (xs / self.SIGMA_CELLS) ** 2)

    def _index(self, x, y):
        fx = (np.asarray(x, float) - self.lo[0]) / self.CELL
        fy = (np.asarray(y, float) - self.lo[1]) / self.CELL
        return fx, fy

    def add(self, x: float, y: float, z: float, w: float = 1.0) -> None:
        fx, fy = self._index(x, y)
        i, j = int(round(float(fx))), int(round(float(fy)))
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            return
        self.sum[i, j] += w * z
        self.cnt[i, j] += w
        self.samples += 1
        self.dirty = True

    def _smooth(self, a: np.ndarray) -> np.ndarray:
        k = self.kernel
        a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, a)
        return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, a)

    def refresh(self) -> None:
        if not self.dirty:
            return
        self.dirty = False
        m = self.cnt > 0
        if not m.any():
            return
        ii, jj = np.nonzero(m)
        x = self.lo[0] + ii * self.CELL
        y = self.lo[1] + jj * self.CELL
        z = self.sum[m] / self.cnt[m]
        w = np.minimum(self.cnt[m], 4.0)
        a = np.column_stack([np.ones_like(x), x, y])
        lhs = (a * w[:, None]).T @ a + np.diag([1e-6, self.RIDGE, self.RIDGE])
        rhs = (a * w[:, None]).T @ z + np.array([0.0, self.RIDGE * self.SLOPE_PRIOR[0], self.RIDGE * self.SLOPE_PRIOR[1]])
        try:
            self.coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            pass
        r = np.zeros(self.shape)
        wg = np.zeros(self.shape)
        r[m] = (z - a @ self.coef) * w
        wg[m] = w
        s, c = self._smooth(r), self._smooth(wg)
        self.res = s / (c + self.SHRINK)
        self.conf = c

    def _bilinear(self, grid, x, y):
        fx, fy = self._index(x, y)
        fx = np.clip(fx, 0.0, self.shape[0] - 1.001)
        fy = np.clip(fy, 0.0, self.shape[1] - 1.001)
        i, j = np.floor(fx).astype(int), np.floor(fy).astype(int)
        u, v = fx - i, fy - j
        return ((1 - u) * (1 - v) * grid[i, j] + u * (1 - v) * grid[i + 1, j]
                + (1 - u) * v * grid[i, j + 1] + u * v * grid[i + 1, j + 1])

    def z(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return self.coef[0] + self.coef[1] * x + self.coef[2] * y + self._bilinear(self.res, x, y)

    def support(self, x, y) -> float:
        return float(self._bilinear(self.conf, x, y))


class SurveyGround:
    """The ground from the survey shipped with the agent: heights on a grid of the field's own frame, placed by the
    dock's offset found from the site map's edge, and levelled to the dock by the downward sensor's readings."""

    def __init__(self, lo, step: float, heights: np.ndarray, offset):
        self.lo = np.asarray(lo, float)
        self.step = float(step)
        self.h = np.asarray(heights, float)
        self.off = np.asarray(offset, float)          # survey frame = the dock's frame + off
        self.shape = self.h.shape
        self.level: list = []                         # readings of (measured ground - survey height)
        self.zoff = None
        self.samples = 0
        self.coef = np.zeros(3)

    def _survey(self, x, y):
        fx = np.clip((np.asarray(x, float) + self.off[0] - self.lo[0]) / self.step, 0.0, self.shape[0] - 1.001)
        fy = np.clip((np.asarray(y, float) + self.off[1] - self.lo[1]) / self.step, 0.0, self.shape[1] - 1.001)
        i, j = np.floor(fx).astype(int), np.floor(fy).astype(int)
        u, v = fx - i, fy - j
        h = self.h
        return (1 - u) * (1 - v) * h[i, j] + u * (1 - v) * h[i + 1, j] + (1 - u) * v * h[i, j + 1] + u * v * h[i + 1, j + 1]

    def add(self, x: float, y: float, z: float, w: float = 1.0) -> None:
        self.level.append(float(z) - float(self._survey(x, y)))
        if len(self.level) > 200:
            del self.level[:-200]
        self.zoff = float(np.median(self.level))
        self.samples += 1

    def refresh(self) -> None:
        pass

    def z(self, x, y):
        zoff = self.zoff if self.zoff is not None else -float(self._survey(0.0, 0.0)) - 0.6
        return self._survey(x, y) + zoff

    def support(self, x, y) -> float:
        return 10.0 if self.zoff is not None else 0.5


# --------------------------------------------------------------------------------------------------- the field

class Field:
    """The field from the site map, metres from the dock: its edge, the crowns, the houses, and the free ground on a
    0.5 m grid with the paths across it."""

    STEP = 0.5

    def __init__(self, fence: np.ndarray, crowns: np.ndarray, buildings: np.ndarray):
        self.fence = np.asarray(fence, float)
        self.crowns = np.asarray(crowns, float).reshape(-1, 3)
        self.buildings = np.asarray(buildings, float).reshape(-1, 5)
        self.lo = self.fence.min(axis=0) - 3.0
        hi = self.fence.max(axis=0) + 3.0
        self.size = tuple(int(v) for v in np.ceil((hi - self.lo) / self.STEP) + 1)
        gx, gy = np.meshgrid(self.lo[0] + self.STEP * np.arange(self.size[0]),
                             self.lo[1] + self.STEP * np.arange(self.size[1]), indexing="ij")
        self.gx, self.gy = gx, gy
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        inside = _inside(self.fence, pts)
        edge = _edge_distance(self.fence, pts)
        self.edge = np.where(inside, edge, -edge).reshape(self.size)
        if len(self.crowns):
            gap = np.full(len(pts), np.inf)
            for e, n, r in self.crowns:
                gap = np.minimum(gap, np.hypot(pts[:, 0] - e, pts[:, 1] - n) - r)
            self.crown_gap = gap.reshape(self.size)
        else:
            self.crown_gap = np.full(self.size, np.inf)
        self.base = (self.edge >= EDGE_CLEAR_M) & (self.crown_gap >= CROWN_CLEAR_M)
        self.free = self.base.copy()
        self.zones: list = []
        # Coverage cells: 2 m inside the edge and outside every crown, as the judge lays them (its grid turns a little).
        cx, cy = np.meshgrid(np.arange(self.lo[0] + 1.0, hi[0], 2.0), np.arange(self.lo[1] + 1.0, hi[1], 2.0),
                             indexing="ij")
        cells = np.column_stack([cx.ravel(), cy.ravel()])
        keep = _inside(self.fence, cells)
        for e, n, r in self.crowns:
            keep &= np.hypot(cells[:, 0] - e, cells[:, 1] - n) >= r
        self.cells = cells[keep]
        self.covered = np.zeros(len(self.cells), bool)

    # ---- the grid
    def cell(self, xy) -> tuple:
        i = int(round((float(xy[0]) - self.lo[0]) / self.STEP))
        j = int(round((float(xy[1]) - self.lo[1]) / self.STEP))
        return min(max(i, 0), self.size[0] - 1), min(max(j, 0), self.size[1] - 1)

    def point(self, c) -> np.ndarray:
        return self.lo + self.STEP * np.asarray(c, float)

    def _sample(self, grid, xy) -> float:
        fx = float(np.clip((float(xy[0]) - self.lo[0]) / self.STEP, 0.0, self.size[0] - 1.001))
        fy = float(np.clip((float(xy[1]) - self.lo[1]) / self.STEP, 0.0, self.size[1] - 1.001))
        i, j = int(fx), int(fy)
        u, v = fx - i, fy - j
        return float((1 - u) * (1 - v) * grid[i, j] + u * (1 - v) * grid[i + 1, j]
                     + (1 - u) * v * grid[i, j + 1] + u * v * grid[i + 1, j + 1])

    def edge_many(self, pts) -> np.ndarray:
        """edge_at for many points at once."""
        pts = np.asarray(pts, float)
        fx = np.clip((pts[:, 0] - self.lo[0]) / self.STEP, 0.0, self.size[0] - 1.001)
        fy = np.clip((pts[:, 1] - self.lo[1]) / self.STEP, 0.0, self.size[1] - 1.001)
        i, j = fx.astype(int), fy.astype(int)
        u, v = fx - i, fy - j
        g = self.edge
        return ((1 - u) * (1 - v) * g[i, j] + u * (1 - v) * g[i + 1, j] + (1 - u) * v * g[i, j + 1]
                + u * v * g[i + 1, j + 1])

    def edge_at(self, xy) -> float:
        """Metres inside the edge, negative outside it."""
        return self._sample(self.edge, xy)

    def crown_at(self, xy) -> float:
        """Metres off the nearest crown circle, negative inside one."""
        if not len(self.crowns):
            return math.inf
        return float(np.min(np.hypot(self.crowns[:, 0] - xy[0], self.crowns[:, 1] - xy[1]) - self.crowns[:, 2]))

    def inward(self, xy) -> np.ndarray:
        """The way into the field from a point, along the edge distance's slope."""
        d = 0.5
        g = np.array([self.edge_at((xy[0] + d, xy[1])) - self.edge_at((xy[0] - d, xy[1])),
                      self.edge_at((xy[0], xy[1] + d)) - self.edge_at((xy[0], xy[1] - d))])
        n = float(np.hypot(*g))
        return g / n if n > 1e-9 else -np.asarray(xy, float) / max(1e-6, float(np.hypot(*xy)))

    def set_zones(self, zones) -> None:
        """Keep the free ground off every person: zones are (east, north, radius)."""
        free = self.base.copy()
        for e, n, r in zones:
            if r > 0.0:
                free &= np.hypot(self.gx - e, self.gy - n) >= r
        self.free = free
        self.zones = [tuple(z) for z in zones]

    def in_zone(self, xy, pad: float = 0.0) -> bool:
        return any(float(np.hypot(xy[0] - e, xy[1] - n)) < r + pad for e, n, r in self.zones if r > 0.0)

    def holds(self, xy) -> bool:
        return bool(self.free[self.cell(xy)])

    def clear(self, a, b) -> bool:
        a, b = np.asarray(a, float), np.asarray(b, float)
        count = max(2, int(np.hypot(*(b - a)) / 0.4) + 2)
        pts = np.linspace(a, b, count)
        i = np.clip(np.rint((pts[:, 0] - self.lo[0]) / self.STEP), 0, self.size[0] - 1).astype(int)
        j = np.clip(np.rint((pts[:, 1] - self.lo[1]) / self.STEP), 0, self.size[1] - 1).astype(int)
        return bool(self.free[i, j].all())

    def path_clear(self, points) -> bool:
        """Whether every leg of a path stays on the free ground."""
        return all(self.clear(a, b) for a, b in zip(points[:-1], points[1:]))

    def nearest_free(self, xy):
        cells = np.argwhere(self.free)
        if not len(cells):
            return None
        here = (np.asarray(xy, float) - self.lo) / self.STEP
        return tuple(int(v) for v in cells[int(np.argmin(np.hypot(*(cells - here).T)))])

    def path(self, a, b):
        """The shortest way from a to b over the free ground, pulled straight; from the free cell nearest a when a is
        not free, to the free cell nearest b when b is not. None when b cannot be reached."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        if self.holds(a) and self.holds(b) and self.clear(a, b):
            return np.array([a, b])
        start = self.cell(a) if self.holds(a) else self.nearest_free(a)
        goal = self.cell(b) if self.holds(b) else self.nearest_free(b)
        if start is None or goal is None:
            return None
        end = b if self.holds(b) else self.point(goal)
        came = {start: None}
        cost = {start: 0.0}
        heap = [(0.0, start)]
        steps = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        free = self.free
        nx, ny = self.size
        expanded = 0
        while heap:
            _, here = heapq.heappop(heap)
            if here == goal:
                break
            expanded += 1
            if expanded > 12000:
                return None
            base = cost[here]
            for di, dj, step in steps:
                i, j = here[0] + di, here[1] + dj
                if not (0 <= i < nx and 0 <= j < ny) or not free[i, j]:
                    continue
                new = base + step
                nxt = (i, j)
                if new < cost.get(nxt, math.inf):
                    cost[nxt], came[nxt] = new, here
                    heapq.heappush(heap, (new + math.hypot(goal[0] - i, goal[1] - j), nxt))
        if goal not in came:
            return None
        cells = [goal]
        while came[cells[-1]] is not None:
            cells.append(came[cells[-1]])
        points = [a] + [self.point(c) for c in reversed(cells)] + [end]
        return self._pulled(points)

    def _pulled(self, points):
        kept = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.clear(points[i], points[j]):
                j -= 1
            kept.append(points[j])
            i = j
        return np.array(kept, dtype=float)

    def lanes(self, dock_xy=(0.0, 0.0), spacing: float = 0.0):
        """The sweep's runs: lines LANE_M apart along the field's long axis, the outer ones LANE_INSET_M in, cut where
        the ground is not free; each a start and an end, flown back and forth from the lane nearest the dock."""
        centred = self.fence - self.fence.mean(axis=0)
        _, _, axes = np.linalg.svd(centred, full_matrices=False)
        u = axes[0]
        v = np.array([-u[1], u[0]])
        along, across = self.fence @ u, self.fence @ v
        width = float(across.max() - across.min())
        count = max(1, int(math.ceil((width - 2.0 * LANE_INSET_M) / (spacing or LANE_M))) + 1)
        if count == 1:
            offsets = np.array([(across.max() + across.min()) / 2.0])
        else:
            offsets = np.linspace(across.min() + LANE_INSET_M, across.max() - LANE_INSET_M, count)
        dock_across = float(np.asarray(dock_xy) @ v)
        if abs(offsets[-1] - dock_across) < abs(offsets[0] - dock_across):
            offsets = offsets[::-1]
        runs = []
        for k, offset in enumerate(offsets):
            steps = np.arange(along.min(), along.max(), 1.0)
            points = steps[:, None] * u + offset * v
            ok = [self.holds(p) for p in points] + [False]
            lane_runs, start = [], None
            for n, good in enumerate(ok):
                if good and start is None:
                    start = n
                elif not good and start is not None:
                    if n - 1 > start + 1:
                        lane_runs.append(np.array([points[start], points[n - 1]]))
                    start = None
            runs.append(lane_runs)
        # The first lane starts at its end nearer the dock; each next one starts where the last one ended.
        ordered = []
        here = np.asarray(dock_xy, float)
        for lane_runs in runs:
            if not lane_runs:
                continue
            first, last = lane_runs[0][0], lane_runs[-1][1]
            if np.hypot(*(last - here)) < np.hypot(*(first - here)):
                lane_runs = [r[::-1] for r in lane_runs[::-1]]
            ordered += lane_runs
            here = lane_runs[-1][1]
        return ordered

    def cover(self, eye, fwd, up, right, half_v, aspect, ground) -> int:
        """Mark the cells a frame saw closely enough (the coverage rule, flat ground); the number newly marked."""
        if not len(self.cells):
            return 0
        hag = float(eye[2] - ground.z(eye[0], eye[1]))
        if not 0.8 < hag <= COVER_HEIGHT_M:
            return 0
        z = ground.z(self.cells[:, 0], self.cells[:, 1])
        off = np.column_stack([self.cells, z]) - eye[None, :]
        depth = off @ fwd
        ahead = depth > 0.1
        safe = np.where(ahead, depth, 1.0)
        half_h = half_v * aspect
        in_frame = ahead & (np.abs(off @ right) <= half_h * safe) & (np.abs(off @ up) <= half_v * safe)
        dist = np.linalg.norm(off, axis=1)
        facing = -off[:, 2] / np.maximum(dist, 1e-9)
        pitch = 2.0 * half_v / FEED_SHAPE[0]
        area = (safe * pitch) ** 2 * (safe / np.maximum(dist, 1e-9)) / np.maximum(facing, 1e-9)
        seen = in_frame & (facing > 0.0) & (np.sqrt(area) <= COVER_PIXEL_M)
        new = int((seen & ~self.covered).sum())
        self.covered |= seen
        return new


# --------------------------------------------------------------------------------------------------- people

class Cand:
    """A person candidate: where the boxes put him, how sure the feed and the close-ups were."""

    def __init__(self, cid: int, t: float):
        self.id = cid
        self.sum_xy = np.zeros(2)
        self.sum_w = 0.0
        self.zoom_xy = np.zeros(2)
        self.zoom_w = 0.0
        self.hits = 0
        self.feed_best = 0.0
        self.t_first = t
        self.t_last = t
        self.looks: list = []
        self.empties = 0
        self.zooms = 0
        self.status = "new"           # new, reported, dropped
        self.best_tan = 0.0
        self.visits = 0
        self.need_visit = False
        self.report_xy = None
        self.views: list = []         # (spot, drone's ground point, its height over the ground) per look
        self.blocked = 0              # close-ups that saw him only past a crown
        self.blocked_best = 0.0       # the surest of those
        self.crowded = 0              # close-ups whose box also held where someone else stands
        self.wait = ""                # sure, but waiting for a steeper look ("pos") or a second place ("still")
        self.wait_t = None            # when he first waited for a second place
        self.zoom_from = None        # where the drone was for its last close-up of him
        self.apart: set = set()       # candidates boxed apart from him in one frame (two men)
        self.dup_of = None            # the reported man he was taken for, when a duplicate
        self.merged_into = None       # the live candidate he was merged into, when a duplicate
        self.seen_from: list = []     # where the drone was when the wide frame boxed him surely

    def add(self, xy, score: float, tan_dep: float, t: float, zoom: bool = False, drone=None, agl=None) -> None:
        if drone is not None:
            self.views.append((np.asarray(xy, float).copy(), np.asarray(drone, float).copy(), float(agl)))
            if len(self.views) > 40:
                del self.views[:-40]
        w = max(score, 0.05) * min(tan_dep, 3.0) ** 2 * (2.0 if zoom else 1.0)
        if zoom:
            self.zoom_xy += w * np.asarray(xy, float)
            self.zoom_w += w
        else:
            self.hits += 1
            self.feed_best = max(self.feed_best, score)
        self.sum_xy += w * np.asarray(xy, float)
        self.sum_w += w
        self.best_tan = max(self.best_tan, tan_dep)
        self.t_last = t

    @property
    def xy(self) -> np.ndarray:
        return self.sum_xy / max(self.sum_w, 1e-9)

    @property
    def live(self) -> bool:
        return self.status == "new"

    def still(self) -> bool:
        """Whether two looks from places apart put him on the same spot: a man standing, not something that moves
        with the drone."""
        v = self.views
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                apart = float(np.hypot(*(v[i][1] - v[j][1]))) >= STATIC_MOVED_M or abs(v[i][2] - v[j][2]) >= STATIC_HEIGHT_M
                if apart and float(np.hypot(*(v[i][0] - v[j][0]))) <= STATIC_SAME_M:
                    return True
        return False


# --------------------------------------------------------------------------------------------------- the agent

class DroneFlightController:
    """The Test Field patrol: take off, sweep, confirm and report each person with his spot, home and land."""

    def __init__(self):
        self.models = {"feed": Detector(os.path.join(HERE, MODEL_FILES["feed"])),
                       "zoom": Detector(os.path.join(HERE, MODEL_FILES["zoom"]), scale=0.5)}
        self.refiner = BoxRefiner(os.path.join(HERE, MODEL_FILES["refiner"]))
        self.reset()

    def reset(self):
        self.field = None
        self.ground = None
        self.held = np.zeros(ACTION_DIM, dtype=np.float32)
        self.decision = 0
        self.gauge0 = None              # the dock gauge's first reading and when it came
        self.gauge0_t = 0.0
        self.gauge_last = None          # the gauge's reading at the last decision, and when it last changed
        self.gauge_push_t = None
        self.history: list = []
        self.frame = None
        self.last_frame_age = None
        self.last_zoom_age = None
        self.window_reads = 0               # heavy model reads in this 0.5 s window (FRAME_ZOOM_SPLIT)
        self.pending_zoom = None
        self.cands: list = []
        self.next_id = 0
        self.reports_made = 0
        self.report_queue: list = []
        self.mode = "search"
        self.runs: list = []
        self.run_i = 0
        self.path = None
        self.path_i = 0
        self.path_goal = None
        self.visit = None
        self.home_pressed = -10
        self.events: list = []
        self.last_laser = None
        self.last_refresh = -1.0
        self.zone_key = None
        self.zones_full: list = []     # the keep-away circles with the full margin (the dock's flight home keeps off these)
        self.escaping = False
        self.escape_t = 0.0
        self.escapes = 0
        self.offsets: list = []          # (offset per metre of height east, north, spot east, north) of feed boxes
        self.shadow = None               # where the drone's shadow falls, per metre of its height over the ground
        self.shadow_votes = 0
        self.home_clear = None
        self.danger_ahead = False
        self.fill = None
        self.fill_skip: set = set()
        self.plan_retry_at = -1.0
        self.plan_fails = 0
        self.now = 0.0
        self.pos = np.zeros(3)
        self.heading = 0.0

    # ------------------------------------------------------------------------------------------------ the loop

    def act(self, observation):
        try:
            action = self._act(observation)
        except Exception as exc:                 # never let the process die: hover, camera ahead
            self._log("error", repr(exc)[:300])
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            action[A_GIMBAL] = SEARCH_TILT_DEG / 90.0
        action = np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        action[:5] = np.clip(action[:5], -1.0, 1.0)
        action[5:24] = np.clip(action[5:24], 0.0, 1.0)
        action[24:26] = np.clip(action[24:26], -1.0, 1.0)
        self.held = action.copy()
        self.decision += 1
        return action

    def _log(self, *entry) -> None:
        if len(self.events) < EVENT_CAP:
            self.events.append(entry)

    def _press(self, action, index) -> bool:
        """Press a button if it was released on the last decision; True when this decision presses it."""
        if self.held[index] > 0.5:
            action[index] = 0.0
            return False
        action[index] = 1.0
        return True

    def _takeoff_ok(self, state, now) -> bool:
        """Whether to press take-off now: at once, unless a strong wind blows and the gauge's newest reading is over
        TAKEOFF_LULL of its first (the steady wind), for at most TAKEOFF_WAIT_MAX_S."""
        g = float(state[S_DOCK_WIND]) if len(state) > S_DOCK_WIND else 0.0
        if self.gauge0 is None:
            self.gauge0, self.gauge0_t = g, now
        if self.gauge_last is not None and g != self.gauge_last:
            self.gauge_push_t = now
        self.gauge_last = g
        if TAKEOFF_LULL <= 0.0 or self.gauge0 < TAKEOFF_LULL_FROM or now - self.gauge0_t >= TAKEOFF_WAIT_MAX_S:
            return True
        if TAKEOFF_FRESH_PUSH:
            fresh = self.gauge_push_t is not None and now - self.gauge_push_t <= TAKEOFF_PUSH_AGE_S
        else:
            fresh = now - self.gauge0_t >= TAKEOFF_FRESH_S
        if fresh and g <= TAKEOFF_LULL * self.gauge0:
            return True
        if self.decision % 10 == 0:
            self._log("takeoff_wait", round(now, 2), round(g, 1), round(self.gauge0, 1))
        return False

    def _act(self, obs):
        state = np.asarray(obs["state"], dtype=np.float64)
        if self.field is None:
            self._read_site(np.asarray(obs.get("site_map"), dtype=np.float64))
        now = HORIZON_S - float(state[S_TIME_LEFT])
        self.now = now
        self.pos = state[0:3].copy()
        self.heading = float(state[S_HEADING])
        self.history.append((now, self.pos[0], self.pos[1], self.pos[2], self.heading, float(state[S_TILT])))
        if len(self.history) > 40:
            del self.history[:-40]
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[A_GIMBAL] = SEARCH_TILT_DEG / 90.0
        phase = int(round(state[S_PHASE]))
        self._sense(state, phase)
        if phase == DOCKED:
            if (TAKEOFF_DELAY_S <= 0.0 or now >= TAKEOFF_DELAY_S) and self._takeoff_ok(state, now):
                if self._press(action, A_TAKE_OFF):
                    self._log("takeoff_press", round(now, 2), float(state[S_DOCK_WIND]) if len(state) > S_DOCK_WIND else 0.0)
            return action
        if phase in (TAKING_OFF, FLYING):
            self._perceive(obs, state, now)
        if phase == FLYING:
            self._decide(state, now)
            self._fly(state, action, now)
        if phase in (TAKING_OFF, FLYING):
            self._buttons(state, action, now, phase)
        return action

    # ------------------------------------------------------------------------------------------------ the site

    def _read_site(self, site_map: np.ndarray) -> None:
        n = int(round(site_map[0])) if site_map.size else 0
        fence = site_map[1:1 + 2 * n].reshape(-1, 2) if n >= 3 else None
        if fence is None:
            # No map: a square round the dock, so the drone still flies and lands.
            fence = np.array([[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]])
        k = int(round(site_map[258])) if site_map.size > 258 else 0
        rows = site_map[259:259 + 5 * k].reshape(-1, 5) if k else np.zeros((0, 5))
        crowns = np.column_stack([rows[:, 0], rows[:, 1], rows[:, 2] / 2.0]) if len(rows) else np.zeros((0, 3))
        b = int(round(site_map[579])) if site_map.size > 579 else 0
        buildings = site_map[580:580 + 5 * b].reshape(-1, 5) if b else np.zeros((0, 5))
        self.field = Field(fence, crowns, buildings)
        lo, hi = fence.min(axis=0) - 25.0, fence.max(axis=0) + 25.0
        self.ground = self._survey_ground(fence) or Ground(lo, hi)
        spacing = LANE_M
        if LANE_WIDE_M > 0.0 and isinstance(self.ground, SurveyGround):
            dock_z = float(self.ground._survey(0.0, 0.0)) + DOCK_ABOVE_GROUND_M
            if dock_z < LANE_WIDE_DOCK_Z:
                spacing = LANE_WIDE_M
                self._log("lanes_wide", round(dock_z, 2), spacing)
        self.runs = self.field.lanes((0.0, 0.0), spacing)
        self.run_i = 0
        self._log("site", n, k, b, len(self.runs), len(self.field.cells))

    def _survey_ground(self, fence):
        """The shipped survey's ground, placed by matching this seed's edge to the surveyed one; None without it."""
        if not os.path.exists(FIELD_FILE):
            return None
        try:
            d = np.load(FIELD_FILE)
            outline = np.asarray(d["outline"], float)
            if len(outline) != len(fence):
                self._log("survey_mismatch", len(outline), len(fence))
                return None
            for shift in range(len(outline)):
                diff = np.roll(outline, -shift, axis=0) - fence
                off = diff.mean(axis=0)
                if float(np.abs(diff - off).max()) < MATCH_TOL_M:
                    self._log("survey", shift, round(float(off[0]), 3), round(float(off[1]), 3))
                    return SurveyGround(d["lo"], float(d["step"]), np.asarray(d["heights_cm"], float) / 100.0, off)
            self._log("survey_no_match")
        except Exception as exc:  # noqa: BLE001
            self._log("survey_error", repr(exc)[:120])
        return None

    # ------------------------------------------------------------------------------------------------ sensing

    def _sense(self, state, phase) -> None:
        """Ground readings: the downward sensor under the drone, and laser points that agree with the ground model."""
        g = self.ground
        x, y, z = self.pos
        if phase in (FLYING, RETURNING) and state[S_DOWN_OK] > 0.5:
            gd = float(state[S_GROUND])
            if 0.5 < gd < 16.0 and math.hypot(x, y) > 3.0:
                zg = z - SENSOR_DROP_M - gd
                if g.support(x, y) > 2.0 and abs(zg - float(g.z(x, y))) > 1.0:
                    self._log("ground_odd", round(self.now, 1), round(x, 1), round(y, 1), round(zg, 2))
                else:
                    g.add(x, y, zg)
        if int(round(state[S_LASER_STATUS])) == 0:
            p = np.asarray(state[15:18], float)
            if self.last_laser is None or float(np.abs(p - self.last_laser).max()) > 1e-6:
                self.last_laser = p.copy()
                if self.field is not None and self.field.crown_at(p[:2]) > 0.5 and math.hypot(p[0], p[1]) > 3.0:
                    near = [c for c in self.cands if c.status != "dropped" and float(np.hypot(*(c.xy - p[:2]))) < 1.8]
                    model = float(g.z(p[0], p[1]))
                    if near:
                        self._log("laser_cand", round(self.now, 1), near[0].id, round(p[0], 2), round(p[1], 2),
                                  round(p[2] - model, 2))
                        h = float(p[2] - model)
                        if LASER_FIX_M > 0.0 and LASER_BODY_M[0] <= h <= LASER_BODY_M[1]:
                            # The laser on a man's body pins him (real field: 9,634 of 9,875 such hits within
                            # 1.5 m of a person, p50 0.21 m): a man waiting only for a steeper look has it.
                            c = min(near, key=lambda c: float(np.hypot(*(c.xy - p[:2]))))
                            if c.live and c.wait == "pos" and float(np.hypot(*(c.xy - p[:2]))) <= LASER_FIX_M:
                                w = 18.0
                                c.sum_xy += w * p[:2]
                                c.sum_w += w
                                c.best_tan = max(c.best_tan, 3.0)
                                c.wait = ""
                                self._log("laser_fix", round(self.now, 1), c.id, round(h, 2))
                    elif g.support(p[0], p[1]) < 1.0 or abs(p[2] - model) < 0.8:
                        g.add(p[0], p[1], p[2], 1.0)
        if self.now - self.last_refresh >= 0.5:
            g.refresh()
            self.last_refresh = self.now

    # ------------------------------------------------------------------------------------------------ geometry

    def _pose_at(self, t: float):
        hist = self.history
        if not hist:
            return self.pos.copy(), self.heading, SEARCH_TILT_DEG
        if t <= hist[0][0]:
            h = hist[0]
            return np.array(h[1:4]), h[4], h[5]
        for a, b in zip(reversed(hist[:-1]), reversed(hist[1:])):
            if a[0] <= t <= b[0]:
                f = (t - a[0]) / max(1e-6, b[0] - a[0])
                pos = np.array(a[1:4]) * (1 - f) + np.array(b[1:4]) * f
                return pos, a[4] + _wrap_deg(b[4] - a[4]) * f, a[5] * (1 - f) + b[5] * f
        h = hist[-1]
        return np.array(h[1:4]), h[4], h[5]

    @staticmethod
    def _camera(pos: np.ndarray, heading_deg: float, tilt_deg: float):
        """Eye, forward, up and right of the main camera at a pose, as the gimbal holds it level."""
        hr = math.radians(heading_deg)
        nose = np.array([math.sin(hr), math.cos(hr), 0.0])
        t = math.radians(tilt_deg)
        forward = math.cos(t) * nose + math.sin(t) * np.array([0.0, 0.0, 1.0])
        up = -math.sin(t) * nose + math.cos(t) * np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        eye = pos + nose * CAMERA_OFFSET[0] + np.array([0.0, 0.0, CAMERA_OFFSET[2]])
        return eye, forward, up, right

    @staticmethod
    def _ray(forward, up, right, half_v, aspect, cx, cy):
        half_h = half_v * aspect
        r = forward + (2.0 * cx - 1.0) * half_h * right + (1.0 - 2.0 * cy) * half_v * up
        return r / np.linalg.norm(r)

    @staticmethod
    def _project(fr, point):
        """The pixel (column, row) of a point in a frame, or None when it falls behind or outside it."""
        d = np.asarray(point, float) - fr["eye"]
        z = float(d @ fr["fwd"])
        if z <= 0.1:
            return None
        half_v = fr["half_v"]
        half_h = half_v * fr["cols"] / fr["rows"]
        a = float(d @ fr["right"]) / (z * half_h)
        b = float(d @ fr["up"]) / (z * half_v)
        cx, cy = (a + 1.0) / 2.0, (1.0 - b) / 2.0
        if not (0.03 <= cx <= 0.97 and 0.03 <= cy <= 0.97):
            return None
        return cx * fr["cols"], cy * fr["rows"]

    def _hit(self, eye, ray, above: float = OBJECT_HEIGHT_M):
        """Where a ray from the eye meets the ground raised by `above`; None when it never comes down to it."""
        d = ray / np.linalg.norm(ray)
        if d[2] > -0.05:
            return None
        g = self.ground
        s = (eye[2] - (float(g.z(eye[0], eye[1])) + above)) / -d[2]
        for _ in range(6):
            p = eye + max(s, 0.0) * d
            s = (eye[2] - (float(g.z(p[0], p[1])) + above)) / -d[2]
        if s <= 0.0 or s > 80.0:
            return None
        return eye + s * d

    def _clear_r(self, xy, margin=None) -> float:
        """How far across the ground the drone must keep from a person standing at xy, at the cruise height."""
        zg = float(self.ground.z(xy[0], xy[1])) + PERSON_SLOPE_M
        over = CRUISE_Z - zg - PERSON_H_M
        need = SAFE_M + (CLEAR_MARGIN_M if margin is None else margin)
        if over >= need:
            return 0.0
        return math.sqrt(max(0.0, need * need - max(over, 0.0) ** 2))

    def _update_zones(self) -> None:
        zones, full = [], []
        for c in self.cands:
            if c.status == "dropped":
                continue
            e, n = round(float(c.xy[0]), 1), round(float(c.xy[1]), 1)
            full.append((e, n, round(self._clear_r(c.xy), 1)))
            zones.append((e, n, round(self._clear_r(c.xy, ESCAPE_MARGIN_M), 1)) if self.escaping else full[-1])
        self.zones_full = full
        key = tuple(zones)
        if key != self.zone_key:
            self.zone_key = key
            self.field.set_zones(zones)
            # Re-plan only when what is left of the path now crosses someone's zone.
            if self.path is not None:
                rest = np.vstack([self.pos[:2][None, :], self.path[min(self.path_i + 1, len(self.path) - 1):]])
                if not self.field.path_clear(rest):
                    self.path = None

    # ------------------------------------------------------------------------------------------------ seeing

    def _perceive(self, obs, state, now) -> None:
        age = float(state[S_FRAME_AGE])
        new_frame = self.last_frame_age is None or age < self.last_frame_age - 1e-6
        self.last_frame_age = age
        if new_frame:
            self.window_reads = 0               # a new 0.5 s window: its compute budget starts again
        zoom_age = float(state[S_ZOOM_AGE])
        lens = int(round(state[S_ZOOM_LENS]))
        new_zoom = lens > 0 and (self.last_zoom_age is None or zoom_age < self.last_zoom_age - 1e-6)
        zoom_ready = new_zoom and self.pending_zoom is not None
        if new_frame and int(round(state[S_FEED])) == 0:
            if FRAME_ZOOM_SPLIT and zoom_ready:
                self._read_frame(obs, now, age, detect=False)      # the close-up gets this window
                self._log("frame_skip", round(now, 2))
            else:
                self._read_frame(obs, now, age)
                self.window_reads += 1
        if FRAME_ZOOM_SPLIT and zoom_ready and self.window_reads > 0:
            return                              # read it when the next window starts (the image stays in view)
        self.last_zoom_age = zoom_age if lens > 0 else None
        if zoom_ready:
            self._read_zoom(obs, now, lens, zoom_age)
            self.window_reads += 1
        elif self.pending_zoom is not None and self.decision - self.pending_zoom["decision"] > ZOOM_TIMEOUT:
            self._log("zoom_lost", round(now, 2), self.pending_zoom["cand"])
            self.pending_zoom = None

    def _shadow_vote(self, spot, drone, agl, now) -> None:
        """Learn where the drone's own shadow falls: boxes that keep one place relative to the drone, per metre of its
        height over the ground, while their spot moves with it."""
        if agl < 2.0:
            return
        o = (np.asarray(spot, float) - np.asarray(drone, float)) / agl
        if float(np.hypot(*o)) > SHADOW_MAX:
            return
        agree = [k for k in self.offsets if math.hypot(k[0] - o[0], k[1] - o[1]) < SHADOW_TOL
                 and math.hypot(k[2] - spot[0], k[3] - spot[1]) >= SHADOW_MOVED_M
                 and math.hypot(k[4] - drone[0], k[5] - drone[1]) >= SHADOW_MOVED_M]
        self.offsets.append((float(o[0]), float(o[1]), float(spot[0]), float(spot[1]), float(drone[0]),
                             float(drone[1])))
        if len(self.offsets) > 300:
            del self.offsets[:-300]
        votes = len(agree) + 1
        if votes >= SHADOW_VOTES and votes >= self.shadow_votes:
            pts = np.array([[k[0], k[1]] for k in agree] + [list(o)])
            new = np.median(pts, axis=0)
            if SHADOW_STICKY and self.shadow is not None and float(np.hypot(*(new - self.shadow))) > 2.0 * SHADOW_TOL:
                # Another place: it takes over only with more boxes than now agree with the learnt one (seed 11389:
                # clusters of 3 chance boxes flipped it west, onto the man, and every close-up of him was held).
                held = sum(1 for k in self.offsets if math.hypot(k[0] - self.shadow[0], k[1] - self.shadow[1]) < SHADOW_TOL)
                if votes <= held:
                    return
            if self.shadow is None or float(np.hypot(*(new - self.shadow))) > 0.03:
                self._log("shadow", round(now, 2), round(float(new[0]), 3), round(float(new[1]), 3), votes)
            self.shadow = new
            self.shadow_votes = votes
            # Candidates sitting where the shadow fell are dropped.
            for c in self.cands:
                if c.live and c.views and all(self._on_shadow(v[0], v[1], v[2]) for v in c.views):
                    c.status = "dropped"
                    self._log("drop_shadow", round(now, 2), c.id)

    def _shadow_near(self, drone_xy, xy, keep: float = SHADOW_KEEP_M) -> bool:
        """Whether the drone's shadow, with the drone at drone_xy at the cruise height, falls within keep of xy."""
        if self.shadow is None:
            return False
        agl = CRUISE_Z - float(self.ground.z(drone_xy[0], drone_xy[1]))
        where = np.asarray(drone_xy, float) + max(agl, 0.0) * self.shadow
        return float(np.hypot(*(where - np.asarray(xy, float)))) < keep

    def _on_shadow(self, spot, drone, agl) -> bool:
        if self.shadow is None or agl <= 0.0:
            return False
        where = np.asarray(drone, float) + agl * self.shadow
        return float(np.hypot(*(np.asarray(spot, float) - where))) < max(SHADOW_VETO_M, 0.15 * agl)

    def _plausible(self, xy) -> bool:
        f = self.field
        return (f.edge_at(xy) >= EDGE_IN_M and f.crown_at(xy) >= -CROWN_IN_M
                and float(np.hypot(*xy)) >= DOCK_CLEAR_M)

    def _read_frame(self, obs, now, age, detect=True) -> None:
        rows, cols = FEED_SHAPE
        img = np.asarray(obs["rgb"])
        t_cap = now - age
        pos, head, tilt = self._pose_at(t_cap)
        eye, fwd, up, right = self._camera(pos, head, tilt)
        half_v = _vfov_half_tan(FEED_DIAG_DEG, rows, cols)
        fr = {"t": t_cap, "eye": eye, "fwd": fwd, "up": up, "right": right, "half_v": half_v, "rows": rows,
              "cols": cols, "dets": [], "decision": self.decision, "zoomed": False}
        self.frame = fr
        self.field.cover(eye, fwd, up, right, half_v, cols / rows, self.ground)
        if int(now) // 10 != int(now - 0.5) // 10 and len(self.field.cells):
            self._log("cov", round(now, 1), round(float(self.field.covered.mean()), 3), self.mode)
        model = self.models["feed"]
        if not detect or not model.ready or not img.any():
            return
        drone = pos[:2].copy()
        agl = float(eye[2] - self.ground.z(drone[0], drone[1]))
        fr["drone"], fr["agl"] = drone, agl
        dets = []
        for c, s, cx, cy, w, h in model(img, min_score=FEED_MIN_SCORE)[:12]:
            if c != 0:
                continue
            ray = self._ray(fwd, up, right, half_v, cols / rows, cx / cols, cy / rows)
            p = self._hit(eye, ray)
            if p is None:
                continue
            if not self._plausible(p[:2]):
                self._log("det_out", round(now, 2), round(s, 3), round(p[0], 1), round(p[1], 1))
                continue
            tan_dep = -ray[2] / max(1e-6, math.hypot(ray[0], ray[1]))
            self._shadow_vote(p[:2], drone, agl, now)
            if self._on_shadow(p[:2], drone, agl):
                self._log("det_shadow", round(now, 2), round(s, 3), round(p[0], 1), round(p[1], 1))
                continue
            if any(float(np.hypot(*(p[:2] - d[5][:2]))) < 1.0 for d in dets):
                continue
            dets.append((s, cx, cy, w, h, p, tan_dep))
        placed = []
        for s, cx, cy, w, h, p, tan_dep in dets:
            cand = self._associate(p[:2], s, tan_dep, t_cap, drone=drone, agl=agl)
            fr["dets"].append((cand, s, cx, cy, w, h))
            placed.append((cand, s, p[:2]))
            if SIGHTLINE_VIEW and s >= SIGHTLINE_SCORE:
                cand.seen_from.append(np.asarray(drone, float)[:2].copy())
                if len(cand.seen_from) > 20:
                    del cand.seen_from[:-20]
            self._log("det", round(now, 2), round(s, 3), round(p[0], 2), round(p[1], 2), round(tan_dep, 2), cand.id,
                      cand.status)
        if SEEN_APART_SCORE > 0.0:
            for i, (a, sa, pa) in enumerate(placed):
                for b, sb, pb in placed[i + 1:]:
                    if a is not b and min(sa, sb) >= SEEN_APART_SCORE and float(np.hypot(*(pa - pb))) >= SEEN_APART_M:
                        a.apart.add(b.id)
                        b.apart.add(a.id)
                        for x, y in ((a, b), (b, a)):
                            if SEEN_APART_REVIVE and x.status == "dup" and x.dup_of == y.id:
                                # Taken for him at a first, faint sighting (11567): two men after all.
                                x.status = "new"
                                x.dup_of = None
                                self._log("revive", round(now, 2), x.id, y.id)
        if dets:
            self._merge_cands(now)
            self._update_zones()

    def _merge_cands(self, now) -> None:
        """One candidate per man: one near a reported man is him; two live ones close together are one."""
        reported = [c for c in self.cands if c.status == "reported" and c.report_xy is not None]
        live = [c for c in self.cands if c.live]
        for c in live:
            twin = next((r for r in reported if float(np.hypot(*(r.report_xy - c.xy))) < SAME_PERSON_M
                         and r.id not in c.apart), None)
            if twin is not None:
                c.status = "dup"
                c.dup_of = twin.id
                self._log("dup", round(now, 2), c.id, twin.id)
        live = [c for c in self.cands if c.live]
        for i, a in enumerate(live):
            if not a.live:
                continue
            for b in live[i + 1:]:
                if b.live and float(np.hypot(*(a.xy - b.xy))) < MERGE_M and b.id not in a.apart:
                    a.sum_xy += b.sum_xy
                    a.sum_w += b.sum_w
                    a.zoom_xy += b.zoom_xy
                    a.zoom_w += b.zoom_w
                    a.hits += b.hits
                    a.feed_best = max(a.feed_best, b.feed_best)
                    a.looks += b.looks
                    a.views += b.views
                    a.zooms += b.zooms
                    a.empties += b.empties
                    a.crowded += b.crowded
                    a.best_tan = max(a.best_tan, b.best_tan)
                    a.t_last = max(a.t_last, b.t_last)
                    b.status = "dup"
                    if MERGE_SURVIVOR:
                        b.merged_into = a.id
                        a.seen_from += b.seen_from
                        if len(a.seen_from) > 20:
                            del a.seen_from[:-20]
                    self._log("merge", round(now, 2), a.id, b.id)

    def _associate(self, xy, s, tan_dep, t, zoom: bool = False, drone=None, agl=None, exclude=None) -> Cand:
        best, gap = None, ASSOC_M
        for c in self.cands:
            if (c.status == "dropped" and not zoom) or c is exclude:
                continue
            d = float(np.hypot(*(c.xy - xy)))
            if d < gap:
                best, gap = c, d
        if MERGE_SURVIVOR and best is not None:
            # One merged into another is the same man: his boxes go to the one that carries him on.
            for _ in range(4):
                if best.status != "dup" or best.merged_into is None:
                    break
                nxt = next((c for c in self.cands if c.id == best.merged_into), None)
                if nxt is None or nxt is exclude or nxt.status == "dropped" or float(np.hypot(*(nxt.xy - xy))) >= MERGE_M \
                        or self._others(nxt, MERGE_ALONE_M):
                    break               # only a box that could be him, and only where he is alone
                best = nxt
        if best is None:
            best = Cand(self.next_id, t)
            self.next_id += 1
            self.cands.append(best)
        best.add(xy, s, tan_dep, t, zoom=zoom, drone=drone, agl=agl)
        return best

    def _zoom_view(self, pz, now, zoom_age, lens):
        t_cap = now - zoom_age
        pos, head, tilt = self._pose_at(t_cap)
        self.zoom_drone = pos.copy()
        eye, fwd0, up0, _ = self._camera(pos, head, tilt)
        target = pz["target"]
        f = target - eye
        f /= np.linalg.norm(f)
        up = pz["up"] - float(pz["up"] @ f) * f
        if np.linalg.norm(up) < 1e-6:
            up = up0
        up = up / np.linalg.norm(up)
        right = np.cross(f, up)
        right /= np.linalg.norm(right)
        half_v = _vfov_half_tan(ZOOM_DIAG_DEG.get(lens, 15.0), *ZOOM_SHAPE)
        return eye, f, up, right, half_v

    def _read_zoom(self, obs, now, lens, zoom_age) -> None:
        pz = self.pending_zoom
        self.pending_zoom = None
        img = np.asarray(obs["zoom"])
        eye, f, up, right, half_v = self._zoom_view(pz, now, zoom_age, lens)
        drone = self.zoom_drone[:2]
        agl = float(eye[2] - self.ground.z(drone[0], drone[1]))
        rows, cols = ZOOM_SHAPE
        boxes = self.models["zoom"](img, min_score=0.2) if self.models["zoom"].ready else []
        cand = next((c for c in self.cands if c.id == pz["cand"]), None)
        centre = None
        centre_refined = False
        others = []
        seconds = []
        persons = [b for b in boxes if b[0] == 0 and b[1] >= ZOOM_PERSON_SCORE][:4]
        # Only the box nearest where the close-up was aimed is refined (the refiner is the costly part), and it is the
        # one a report carries when it is on the man asked about.
        refined = -1
        if persons and cand is not None:
            refined = min(range(len(persons)), key=lambda i: math.hypot(persons[i][2] - cols / 2.0, persons[i][3] - rows / 2.0))
            if self.refiner.ready:
                c0, s0, cx0, cy0, w0, h0 = persons[refined]
                persons[refined] = (c0, s0, *self.refiner(img, cx0, cy0, w0, h0))
        seen = []
        for k, (c, s, cx, cy, w, h) in enumerate(persons):
            ray = self._ray(f, up, right, half_v, cols / rows, cx / cols, cy / rows)
            p = self._hit(eye, ray)
            if p is None:
                continue
            tan_dep = -ray[2] / max(1e-6, math.hypot(ray[0], ray[1]))
            self._log("zdet", round(now, 2), lens, round(s, 3), round(p[0], 2), round(p[1], 2), round(tan_dep, 2),
                      None if cand is None else cand.id, [round(float(v), 1) for v in (cx, cy, w, h)])
            if self._on_shadow(p[:2], drone, agl):
                self._log("zdet_shadow", round(now, 2), round(s, 3))
                continue
            near_asked = cand is not None and min(float(np.hypot(*(cand.xy - p[:2]))),
                                                  float(np.hypot(*(pz["target"][:2] - p[:2])))) < 2.5
            item = (s, cx, cy, w, h, p, tan_dep)
            seen.append(item)
            if near_asked and (centre is None or k == refined or (s > centre[0] and not centre_refined)):
                centre = item
                centre_refined = k == refined
            elif not near_asked:
                others.append(item)
        if ZOOM_SECOND and centre is not None and cand is not None:
            # A second sure man in the close-up, away from the one the report goes to (4327: a lying man 2.5 m from
            # the standing one that took the report was dropped, never reported, and the standing one reported twice).
            for item in seen:
                if item is centre or any(item is o for o in others) or item[0] < ZOOM_SECOND_SCORE:
                    continue
                if float(np.hypot(*(np.asarray(item[5][:2]) - np.asarray(centre[5][:2])))) >= ZOOM_SECOND_M:
                    seconds.append(item)
        if cand is None:
            return
        cand.zooms += 1
        if centre is None:
            cand.looks.append(0.0)
            cand.empties += 1
            if cand.empties >= EMPTY_DROP and max(cand.looks) < LOOK_AGAIN and cand.status == "new":
                cand.status = "dropped"
                self._log("drop", round(now, 2), cand.id, cand.empties)
                self._update_zones()
            return
        inner = self._inner_box(centre, seen)
        if inner is not None:
            # The chosen box holds a second sure person box: two people boxed as one, which the judge counts as nobody.
            # The man inside is reported on his own box and his own spot (the candidate's spot lies between them).
            self._log("merged_box", round(now, 2), cand.id, round(inner[0], 3))
            centre = inner
        s, cx, cy, w, h, p, tan_dep = centre
        if not self._clear_sight(eye[:2], p[:2]) and not pz.get("feed_seen"):
            # Seen past a crown: it may be the tree. It keeps the candidate alive but cannot report him.
            cand.blocked += 1
            cand.blocked_best = max(cand.blocked_best, s)
            self._log("zdet_crown", round(now, 2), cand.id, round(s, 3))
            cand.add(p[:2], s, tan_dep, now, zoom=True, drone=drone, agl=agl)
            if (cand.blocked >= 3 and not any(v > 0.0 for v in cand.looks)) \
                    or (CROWN_VISIT_MIN > 0.0 and s >= CROWN_VISIT_MIN and cand.status == "new"):
                cand.need_visit = cand.visits < VISITS_MAX
            return
        # Someone else known stands where this box is: it may hold both men (the judge counts that as nobody), and its
        # middle lies between them, so the look neither reports him nor moves his spot.
        crowd = self._crowded(cand, eye, f, up, right, half_v, (cx, cy, w, h)) \
            if CROWD_CHECK and cand.status == "new" else None
        before = [v for v in cand.looks if v > 0.0]
        cand.looks.append(s)
        if crowd is None:
            cand.add(p[:2], s, tan_dep, now, zoom=True, drone=drone, agl=agl)
        if len(cand.looks) >= WEAK_LOOKS and max(cand.looks) < WEAK_MAX and cand.status == "new":
            # Every real man with three close-ups had one at 0.69 or more: this is the ground, not a man.
            cand.status = "dropped"
            self._log("drop_weak", round(now, 2), cand.id, round(max(cand.looks), 3))
            self._update_zones()
            return
        # The box should be about as large as the wide frame's box of him, seen through this lens: much smaller,
        # it is likely part of him (a torso without legs), and the report waits for a close-up that boxes all of him.
        whole = True
        self.look_ratio = None
        if pz.get("feed_diag"):
            expect = pz["feed_diag"] * (2.0 * _vfov_half_tan(FEED_DIAG_DEG, *FEED_SHAPE) / FEED_SHAPE[0]) / (2.0 * half_v / rows)
            ratio = math.hypot(w, h) / max(expect, 1e-6)
            self.look_ratio = ratio
            if not BOX_RATIO_MIN <= ratio <= BOX_RATIO_MAX:
                whole = False
                self._log("box_odd", round(now, 2), cand.id, round(ratio, 2))
            elif UNSURE_BOX_MIN > 0.0 and s < UNSURE_BOX_SCORE and (ratio < UNSURE_BOX_MIN
                                                                  or 0.0 < UNSURE_BOX_MAX < ratio):
                # A weaker look boxing less of him than the wide frame predicts: on the real field every report judged
                # a bad box (part of the man) came from a look under 0.85; the next look may box all of him.
                whole = False
                self._log("box_small", round(now, 2), cand.id, round(ratio, 2), round(s, 3))
        elif UNKNOWN_BOX_SCORE > 0.0 and s < UNKNOWN_BOX_SCORE:
            # No wide-frame box of him to size this one against: a weaker look waits for a surer one (real
            # field: every bad box reported unchecked scored under 0.88, 0 bad in 1,195 at 0.88 and over).
            whole = False
            self._log("box_unknown", round(now, 2), cand.id, round(s, 3))
        box_held = not whole
        if whole and LYING_MIN_M > 0.0 and h > 0.0 and w / h >= LYING_ASPECT:
            # A lying man's box: at its range it should be about his length; much shorter, it holds one end of
            # him and the judge counts it as nobody (12473, 13069: 1.08 m, a third of him, false report + miss).
            length = max(w, h) * 2.0 * half_v * float(np.linalg.norm(p - eye)) / rows
            if length < LYING_MIN_M:
                whole = False
                self._log("box_short", round(now, 2), cand.id, round(length, 2), round(s, 3))
                box_held = True
        if box_held and crowd is None and BOX_HOLD_VISIT > 0.0 and s >= BOX_HOLD_VISIT and cand.status == "new" \
                and not cand.need_visit and cand.visits < VISITS_MAX:
            # A sure look held only for its box: a nearer look from a viewpoint boxes all of him.
            cand.need_visit = True
            self._log("box_visit", round(now, 2), cand.id, round(s, 3))
        if crowd is not None:
            whole = False
            cand.crowded += 1
            self._log("crowded", round(now, 2), cand.id, crowd.id, cand.crowded)
            if cand.crowded >= CROWD_VISIT_LOOKS:
                cand.need_visit = cand.visits < VISITS_MAX
        if whole:
            self._queue_report(cand, s, before, (cx, cy, w, h), tan_dep, now,
                               spot=p[:2] if inner is not None else None)
        # A sure person elsewhere in the close-up is a candidate of his own (or more on one we know).
        for s2, cx2, cy2, w2, h2, p2, tan2 in others:
            if s2 >= 0.5 and self._plausible(p2[:2]) and self._clear_sight(eye[:2], p2[:2]):
                other = self._associate(p2[:2], s2, tan2, now, drone=drone, agl=agl)
                self._log("zother", round(now, 2), other.id, round(s2, 3))
        for s2, cx2, cy2, w2, h2, p2, tan2 in seconds:
            if self._plausible(p2[:2]) and self._clear_sight(eye[:2], p2[:2]):
                other = self._associate(p2[:2], s2, tan2, now, drone=drone, agl=agl, exclude=cand)
                self._log("zsecond", round(now, 2), cand.id, other.id, round(s2, 3))
        self._update_zones()

    @staticmethod
    def _inner_box(centre, seen):
        """A sure person box lying mostly inside the chosen box and much smaller than it, or None."""
        _s, cx, cy, w, h = centre[:5]
        l, t, r, b = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        best = None
        for item in seen:
            if item is centre or item[0] < INNER_SCORE:
                continue
            _s2, cx2, cy2, w2, h2 = item[:5]
            if w2 * h2 > INNER_AREA * w * h:
                continue
            iw = max(0.0, min(r, cx2 + w2 / 2) - max(l, cx2 - w2 / 2))
            ih = max(0.0, min(b, cy2 + h2 / 2) - max(t, cy2 - h2 / 2))
            if iw * ih >= INNER_COVER * w2 * h2 and (best is None or item[0] > best[0]):
                best = item
        return best

    def _others(self, cand, within: float):
        """Spots of the other people known near a candidate: reported ones, and live ones a close-up found fairly
        sure (a second box of the same man from the wide frame alone is not one)."""
        out = []
        for o in self.cands:
            if o is cand or o.status in ("dropped", "dup"):
                continue
            if o.status != "reported" and not any(v >= 0.5 for v in o.looks):
                continue
            xy = o.report_xy if o.report_xy is not None else o.xy
            d = float(np.hypot(*(xy - cand.xy)))
            if CROWD_SAME_M <= d <= within:
                out.append((o, np.asarray(xy, float)))
        return out

    def _crowded(self, cand, eye, f, up, right, half_v, box):
        """Another known person whose middle falls inside a close-up box of this one, or None."""
        rows, cols = ZOOM_SHAPE
        cx, cy, w, h = box
        fr = {"eye": eye, "fwd": f, "up": up, "right": right, "half_v": half_v, "rows": rows, "cols": cols}
        for o, xy in self._others(cand, 1e9):
            p3 = np.array([xy[0], xy[1], float(self.ground.z(xy[0], xy[1])) + CROWD_H_M])
            px = self._project(fr, p3)
            if px is None:
                continue
            m = 0.0
            rng = float(np.linalg.norm(p3 - eye))
            if REPEAT_VIEW_M > 0.0 and o.status == "reported" and rng > REPEAT_VIEW_FAR_M:
                # A man already reported is placed only so well, and a far close-up magnifies that: his
                # place, give or take REPEAT_VIEW_M at this range, inside the box means the box may be him
                # again (11270 at 21.7 m, 11776 at 44.5 m: repeats judged false reports). Near, the margin
                # would hide a second man standing just behind a reported one.
                m = REPEAT_VIEW_M / max(rng, 1.0) * (rows / 2.0) / half_v
            if abs(px[0] - cx) <= w / 2.0 + m and abs(px[1] - cy) <= h / 2.0 + m:
                return o
        return None

    @staticmethod
    def _lined_up(spot, xy, others) -> bool:
        """Whether, seen from spot, someone else stands within CROWD_SEP_DEG of the bearing to xy."""
        a = math.degrees(math.atan2(xy[0] - spot[0], xy[1] - spot[1]))
        for _o, oxy in others:
            b = math.degrees(math.atan2(oxy[0] - spot[0], oxy[1] - spot[1]))
            if abs(_wrap_deg(a - b)) < CROWD_SEP_DEG:
                return True
        return False

    def _clear_sight(self, eye_xy, xy) -> bool:
        """Whether the line across the ground from the camera to a spot passes clear of every crown circle: a tree
        part is always behind or inside one, and nobody stands under one."""
        f = self.field
        if not len(f.crowns):
            return True
        a, b = np.asarray(eye_xy, float), np.asarray(xy, float)
        ab = b - a
        rel = f.crowns[:, :2] - a[None, :]
        t = np.clip((rel @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
        gap = np.hypot(*(a[None, :] + t[:, None] * ab[None, :] - f.crowns[:, :2]).T) - f.crowns[:, 2]
        return bool(np.all(gap >= SIGHT_CROWN_M))

    def _sure(self, s: float, before) -> bool:
        if s >= REPORT_SURE:
            return True
        return bool(before) and s >= PAIR_MIN and max(before) >= PAIR_MIN

    def _figure_dist(self, xy) -> float:
        """How far a spot (the dock's frame) lies from the nearest known figure; infinite unless the shipped survey
        placed the field (its frame = the dock's + offset)."""
        g = self.ground
        if not isinstance(g, SurveyGround) or not KNOWN_FIGURES:
            return math.inf
        w = np.asarray(xy, float) + g.off[:2]
        return min(float(np.hypot(w[0] - fx, w[1] - fy)) for fx, fy in KNOWN_FIGURES)

    def _near_figure(self, xy) -> bool:
        """Whether a spot (the dock's frame) lies within KNOWN_FIGURE_M of a known figure."""
        return self._figure_dist(xy) < KNOWN_FIGURE_M

    def _queue_report(self, cand, s, before, box, tan_dep, now, spot=None) -> None:
        if cand.status != "new":
            return
        xy = cand.xy if spot is None else np.asarray(spot, float)
        if KNOWN_FIGURE_CORE_M > 0.0 and self._figure_dist(xy) < KNOWN_FIGURE_CORE_M:
            # On the figure itself: however sure the look, it is the figure (seen from overhead it scored up to 0.944).
            self._log("known_figure_core", round(now, 2), cand.id, round(s, 3))
            if sum(1 for v in cand.looks if v > 0.0) >= 3:
                cand.status = "dropped"
            return
        if KNOWN_FIGURE_M > 0.0 and s < KNOWN_FIGURE_SURE and self._near_figure(xy):
            # The field's own figure looks like a man to the close-up model: only a very sure look reports there.
            self._log("known_figure", round(now, 2), cand.id, round(s, 3))
            if sum(1 for v in cand.looks if v > 0.0) >= 3:
                cand.status = "dropped"
            return
        if not self._sure(s, before):
            if len(cand.looks) >= LOOKS_MAX:
                cand.need_visit = cand.visits < VISITS_MAX
            return
        looked = len(cand.looks) - 1                    # close-ups before this one, empty ones included
        if WEAK_RUN_LOOKS > 0 and cand.feed_best < WEAK_RUN_MAX and max(before, default=0.0) < WEAK_RUN_MAX and (
                len(before) >= WEAK_RUN_LOOKS
                or (WEAK_RUN_EMPTY_FEED > 0.0 and cand.feed_best < WEAK_RUN_EMPTY_FEED and looked >= WEAK_RUN_LOOKS)):
            # Only weak looks till now, and a weak wide box: one sure look among them may be a fluke; one more.
            self._log("weak_run_wait", round(now, 2), cand.id, round(s, 3), len(before))
            return
        if not self._plausible(xy) or self.field.edge_at(xy) < 0.5:
            self._log("report_skip_place", round(now, 2), cand.id, round(xy[0], 2), round(xy[1], 2))
            return
        if not cand.still():
            if cand.wait != "still" or cand.wait_t is None:
                cand.wait_t = now
            cand.wait = "still"
            self._log("report_wait_still", round(now, 2), cand.id, len(cand.views))
            return
        twin = next((c for c in self.cands if c is not cand and c.status == "reported"
                     and float(np.hypot(*(c.report_xy - xy))) < SAME_PERSON_M and c.id not in cand.apart), None)
        if twin is not None:
            cand.status = "dropped"
            self._log("repeat", round(now, 2), cand.id, twin.id)
            return
        sigma = round(cand.best_tan, 2)
        known = self.ground.support(xy[0], xy[1]) >= 1.0
        if cand.best_tan < (POS_TAN_MIN if known else POS_TAN_BLIND):
            cand.need_visit = cand.visits < VISITS_MAX
            self._log("report_wait_pos", round(now, 2), cand.id, sigma, known)
            if cand.need_visit:
                cand.wait = "pos"
                return
        if self.reports_made >= MAX_REPORTS:
            return
        cx, cy, w, h = box
        rows, cols = ZOOM_SHAPE
        self.report_queue.append({"cand": cand.id, "box": (cx / cols, cy / rows, w / cols, h / rows),
                                  "xy": xy.copy(), "decision": self.decision, "score": s})
        cand.status = "reported"
        cand.report_xy = xy.copy()
        ratio = getattr(self, "look_ratio", None)
        self._log("report", round(now, 2), cand.id, round(s, 3), round(xy[0], 2), round(xy[1], 2), round(sigma, 2),
                  None if ratio is None else round(ratio, 2))

    # ------------------------------------------------------------------------------------------------ buttons

    def _buttons(self, state, action, now, phase) -> None:
        if self.report_queue:
            item = self.report_queue[0]
            if self.decision - item["decision"] > 1:
                # The close-up it was judged on is no longer the one shown: drop it and look again.
                self.report_queue.pop(0)
                cand = next((c for c in self.cands if c.id == item["cand"]), None)
                if cand is not None:
                    cand.status = "new"
                    cand.report_xy = None
                self._log("report_stale", round(now, 2), item["cand"])
            elif self._press(action, A_REPORT):
                self.report_queue.pop(0)
                cx, cy, w, h = item["box"]
                action[A_REPORT_CLASS] = 0.0
                action[A_REPORT_IMAGE] = 1.0
                action[A_REPORT_CX], action[A_REPORT_CY], action[A_REPORT_W], action[A_REPORT_H] = cx, cy, w, h
                action[A_EAST] = float(np.clip(item["xy"][0] / REPORT_REACH_M, -1.0, 1.0))
                action[A_NORTH] = float(np.clip(item["xy"][1] / REPORT_REACH_M, -1.0, 1.0))
                self.reports_made += 1
                self._log("press_report", round(now, 2), item["cand"])
                return
            return
        visiting = self.visit["cand"] if self.mode == "visit" and self.visit is not None else None

        def why(reason):
            # Debug: once per fresh frame, why the man the drone hovers for gets no close-up from it.
            fr_ = self.frame
            if visiting is None or self.visit.get("arrived") is None or fr_ is None or fr_.get("why"):
                return
            fr_["why"] = reason
            self._log("visit_nozoom", round(now, 2), visiting, reason)

        if self.pending_zoom is not None or self.frame is None or state[S_ZOOMS_LEFT] < 1:
            if self.pending_zoom is None:
                why("no frame or no zooms left")
            return
        if now < HORIZON_S / 2.0 and state[S_ZOOMS_LEFT] <= ZOOM_RESERVE:
            return
        fr = self.frame
        if fr["zoomed"] or self.decision - fr["decision"] > 2:
            return
        pick = None
        for c in self.cands:
            mine = c.id == visiting
            sure_unseen = c.live and c.feed_best >= 0.9 and not any(v > 0.0 for v in c.looks)

            def skip(reason, c=c, sure_unseen=sure_unseen):
                # Debug: a man the wide frame boxes surely but no close-up has found yet, passed over: why.
                if sure_unseen and not mine:
                    key = ("skip", c.id)
                    if not fr.get(key):
                        fr[key] = True
                        self._log("sure_nozoom", round(now, 2), c.id, reason)
            if not c.live or c.zooms >= LOOKS_MAX + VISIT_ZOOMS * (c.visits + (1 if mine else 0)):
                if mine:
                    why("not live" if not c.live else f"allowance used ({c.zooms})")
                skip(f"allowance used ({c.zooms})")
                continue
            p3 = np.array([c.xy[0], c.xy[1], float(self.ground.z(c.xy[0], c.xy[1])) + OBJECT_HEIGHT_M])
            px = self._project(fr, p3)
            seen_now = FEED_SIGHT_MIN > 0.0 and any(d[0] is c and d[1] >= FEED_SIGHT_MIN for d in fr["dets"])
            if px is None and ZOOM_EDGE_DETS:
                # Boxed in this frame: he is in it, whatever his stored spot projects to (the close-up aims at the box).
                box = next((d for d in fr["dets"] if d[0] is c and d[1] >= FEED_SIGHT_MIN), None)
                if box is not None:
                    px = (box[2], box[3])
                    if not fr.get(("edge", c.id)):
                        fr[("edge", c.id)] = True
                        self._log("zoom_edge", round(now, 2), c.id, round(box[1], 3))
            if px is None or not (seen_now or self._clear_sight(fr["eye"][:2], c.xy)):
                if mine:
                    why("out of the frame" if px is None else "crown in line")
                skip("out of the frame" if px is None else "crown in line")
                continue
            if self._shadow_near(self.pos[:2], c.xy):
                if mine:
                    why("own shadow near him")
                    if SHADOW_REPLAN and self.visit.get("arrived") is not None and not self.visit.get("replanned"):
                        # The shadow now falls on him from this viewpoint (its estimate moved since the visit was
                        # planned, seed 11389): go to one the shadow keeps clear of instead of hovering till time-out.
                        spot = self._viewpoint(c)
                        self.visit["replanned"] = True
                        if spot is not None and float(np.hypot(*(spot - self.visit["spot"]))) > 2.0:
                            self.visit["spot"], self.visit["arrived"] = spot, None
                            self.path = None
                            self._log("visit_replan_shadow", round(now, 2), c.id)
                skip("own shadow near him")
                continue
            if not self._worth_a_look(c, fr):
                if mine:
                    why(f"not worth a look here (waits {c.wait})")
                skip(f"not worth a look here (waits {c.wait})")
                continue
            rank = (c.id != visiting, len(c.looks), -c.feed_best)
            if pick is None or rank < pick[0]:
                pick = (rank, c, px, p3)
        if pick is None:
            return
        _, cand, (cx, cy), p3 = pick
        # Aim at the box the frame drew him with when it has one.
        feed_diag = None
        for c, s, bx, by, bw, bh in fr["dets"]:
            if c is cand:
                cx, cy = bx, by
                feed_diag = math.hypot(bw, bh)
                break
        if not self._press(action, A_ZOOM):
            return
        ray = self._ray(fr["fwd"], fr["up"], fr["right"], fr["half_v"], fr["cols"] / fr["rows"], cx / fr["cols"],
                        cy / fr["rows"])
        target = self._hit(fr["eye"], ray)
        if target is None:
            target = p3
        slant = float(np.linalg.norm(target - self.pos))
        lens = 7 if slant > ZOOM_FAR_M else 3
        action[A_ZOOM_LENS] = 1.0 if lens == 7 else 0.0
        action[A_ZOOM_CX], action[A_ZOOM_CY] = cx / fr["cols"], cy / fr["rows"]
        action[A_ZOOM_W], action[A_ZOOM_H] = 0.05, 0.05
        fr["zoomed"] = True
        cand.zoom_from = np.asarray(fr["eye"], float).copy()
        feed_seen = FEED_SIGHT_MIN > 0.0 and any(d[0] is cand and d[1] >= FEED_SIGHT_MIN for d in fr["dets"])
        self.pending_zoom = {"cand": cand.id, "target": target, "up": np.asarray(fr["up"], float).copy(),
                             "decision": self.decision, "lens": lens, "feed_diag": feed_diag,
                             "feed_seen": feed_seen}
        self._log("zoom", round(now, 2), cand.id, lens, round(slant, 1))

    def _worth_a_look(self, c, fr) -> bool:
        """Whether a close-up of c from here can do what his report waits for: a man already sure but waiting for a
        steeper look gets one only where the look is steep enough, one waiting for a second place only 2.5 m or more
        from where the drone last looked, and one whose close-ups held someone else too only where the two do not
        line up. Close-ups spent short of that used up his allowance before his visit's viewpoint (seed 2524)."""
        eye = fr["eye"]
        if c.wait == "pos" and c.visits < VISITS_MAX:
            known = self.ground.support(c.xy[0], c.xy[1]) >= 1.0
            drop = float(eye[2]) - (float(self.ground.z(c.xy[0], c.xy[1])) + OBJECT_HEIGHT_M)
            if drop / max(0.3, float(np.hypot(*(c.xy - eye[:2])))) < (POS_TAN_MIN if known else POS_TAN_BLIND):
                return False
        if c.wait == "still" and c.zoom_from is not None and not c.still():
            # His wide-frame boxes may show him still from two places by now; else a new place for the next look.
            if float(np.hypot(*(eye[:2] - c.zoom_from[:2]))) < STATIC_MOVED_M \
                    and abs(float(eye[2]) - float(c.zoom_from[2])) < STATIC_HEIGHT_M:
                return False
        if c.crowded and self._lined_up(eye[:2], c.xy, self._others(c, CROWD_NEAR_M)):
            return False
        return True

    # ------------------------------------------------------------------------------------------------ deciding

    def _home_eta(self) -> float:
        """Seconds from now to the landing if the drone headed home now: over the free ground to where the dock's
        straight flight home is clear, then the dock's own flight and landing."""
        dist = float(np.hypot(*self.pos[:2]))
        if self.decision % 10 == 0 or self.home_clear is None:
            self.home_clear = self._return_clear(self.pos[:2])
        if self.home_clear:
            return dist / HOME_MPS + RETURN_RESERVE_S
        return 1.4 * dist / HOME_MPS + RETURN_RESERVE_S

    def _decide(self, state, now) -> None:
        if self.escaping and (now - self.escape_t > ESCAPE_MAX_S or (self.decision % 5 == 0 and not self._boxed_in(full=True))):
            self.escaping = False
            self._update_zones()
            self._log("escape_end", round(now, 2), round(now - self.escape_t, 1), [round(float(v), 1) for v in self.pos[:2]])
        if self.mode != "home" and now + self._home_eta() + HOME_SLACK_S >= HORIZON_S:
            self._switch("home", now, "time")
        if self.mode == "visit":
            v = self.visit
            cand = next((c for c in self.cands if c.id == v["cand"]), None)
            if VISIT_FOLLOW_MERGE and cand is not None and cand.status == "dup" and cand.merged_into is not None:
                # The man being visited was merged into his duplicate: the visit goes on for the survivor (17659: the
                # merge ended the visit 1.5 s in and nobody came back for him).
                nxt = next((c for c in self.cands if c.id == cand.merged_into), None)
                if nxt is not None and nxt.live and nxt.status == "new":
                    self._log("visit_follow", round(now, 2), cand.id, nxt.id)
                    v["cand"] = nxt.id
                    if VISIT_FOLLOW_ZOOMS:
                        # The merge added the duplicate's close-ups to his: the visit still gets its own (17659: all
                        # nine used, so none at the viewpoint while the laser hit him 0.1 m from where he stood).
                        nxt.zooms = min(nxt.zooms, LOOKS_MAX + VISIT_ZOOMS * nxt.visits)
                    cand = nxt
            if cand is None or not cand.live:
                self._switch("search", now, "visit_done")
            elif v["arrived"] is not None and now - v["arrived"] > VISIT_HOLD_S:
                cand.visits += 1
                cand.need_visit = False
                self._log("visit_give_up", round(now, 2), cand.id)
                self._switch("search", now, "visit_timeout")
        if STILL_VISIT_S > 0.0:
            for c in self.cands:
                if c.live and c.wait == "still" and not c.need_visit and c.visits < VISITS_MAX \
                        and c.wait_t is not None and now - c.wait_t >= STILL_VISIT_S and not c.still():
                    c.need_visit = True
                    self._log("still_visit", round(now, 2), c.id)
        cap = self._visit_cap(now)
        if FILL_VISITS and self.mode == "fill":
            # The sweep's end gave every unsure man a visit before the fill; one found sure in the fill gets his too.
            for c in self.cands:
                if c.live and c.wait == "still" and not c.need_visit and c.visits < cap and self._worth(c):
                    c.need_visit = True
                    self._log("fill_visit", round(now, 2), c.id)
        if self.mode in ("search", "fill"):
            waiting = [c for c in self.cands if c.live and c.need_visit and c.visits < cap and self._worth_now(c, cap)]
            if waiting:
                cand = min(waiting, key=lambda c: float(np.hypot(*(c.xy - self.pos[:2]))))
                spot = self._viewpoint(cand)
                if spot is not None:
                    self.visit = {"cand": cand.id, "spot": spot, "arrived": None}
                    self._switch("visit", now, f"cand {cand.id}")
                    return
                self._log("visit_noview", round(now, 2), cand.id, dict(getattr(self, "view_why", {})))
                cand.visits += 1
                cand.need_visit = False
        if self.mode == "search" and self.run_i >= len(self.runs):
                # The sweep is done: a candidate still unsure gets a visit, else home.
                unsure = [c for c in self.cands if c.live and c.visits < cap and self._worth(c)]
                if not getattr(self, "sweep_logged", False):
                    self.sweep_logged = True
                    blocked = [(c.id, c.status, c.visits, round(c.blocked_best, 2)) for c in self.cands if c.blocked]
                    self._log("sweep_done", round(now, 2), [c.id for c in unsure], blocked)
                if unsure:
                    for c in unsure:
                        c.need_visit = True
                else:
                    self._next_fill(now)
        if self.mode == "fill":
            f = self.fill
            if f is None:
                self._next_fill(now)
            elif self.field.covered[f["cell"]]:
                self._next_fill(now)
            elif f["arrived"] is not None and now - f["arrived"] > FILL_HOLD_S:
                self.fill_skip.add(f["cell"])
                self._log("fill_give_up", round(now, 2), f["cell"])
                self._next_fill(now)

    def _next_fill(self, now: float) -> None:
        """Fly to see the nearest field cell no frame has seen closely enough, if there is time; else home."""
        f = self.field
        left = [k for k in np.nonzero(~f.covered)[0] if k not in self.fill_skip]
        here = self.pos[:2]
        if left:
            cells = f.cells[left]
            order = np.argsort(np.hypot(cells[:, 0] - here[0], cells[:, 1] - here[1]))
            for k in order[:12]:
                cell = cells[k]
                spot = self._view_spot(cell)
                if spot is None:
                    self.fill_skip.add(left[k])
                    continue
                go = float(np.hypot(*(spot - here))) / FILL_MPS + 2.0
                back = float(np.hypot(*spot)) / HOME_MPS + RETURN_RESERVE_S
                if now + go + back + HOME_SLACK_S >= HORIZON_S:
                    break
                self.fill = {"cell": int(left[k]), "xy": cell, "spot": spot, "arrived": None}
                if self.mode != "fill":
                    self._switch("fill", now, f"{len(left)} cells")
                else:
                    self.path = None
                self._log("fill", round(now, 2), int(left[k]), round(float(cell[0]), 1), round(float(cell[1]), 1))
                return
        self.fill = None
        if LEFTOVER_VISITS:
            cap = self._visit_cap(now)
            unsure = [c for c in self.cands if c.live and c.visits < cap and self._worth_now(c, cap)]
            if cap > VISITS_MAX and unsure:
                for c in unsure:
                    c.need_visit = True
                self._log("leftover", round(now, 2), [c.id for c in unsure])
                if self.mode != "search":
                    self._switch("search", now, "leftover")
                return
        self._switch("home", now, "done")

    def _visit_cap(self, now: float) -> int:
        """Visits a candidate may get: more once the sweep is done and time is left over besides the way home."""
        if LEFTOVER_VISITS and self.run_i >= len(self.runs) and now + self._home_eta() + LEFTOVER_MIN_S < HORIZON_S:
            return VISITS_MAX + LEFTOVER_EXTRA
        return VISITS_MAX

    def _view_spot(self, cell):
        """A free point FILL_RINGS_M from a cell, as near the drone as can be."""
        here = self.pos[:2]
        for radius in FILL_RINGS_M:
            best, best_d = None, math.inf
            for k in range(16):
                ang = 2.0 * math.pi * k / 16.0
                spot = np.asarray(cell, float) + radius * np.array([math.cos(ang), math.sin(ang)])
                if not self.field.holds(spot):
                    continue
                d = float(np.hypot(*(spot - here)))
                if d < best_d:
                    best, best_d = spot, d
            if best is not None:
                return best
        return None

    def _worth_now(self, c, cap) -> bool:
        """_worth, or in leftover time (cap over VISITS_MAX) a sure wide-frame box of him."""
        return self._worth(c) or (LEFTOVER_FEED_MIN > 0.0 and cap > VISITS_MAX and c.feed_best >= LEFTOVER_FEED_MIN)

    @staticmethod
    def _worth(c) -> bool:
        """Whether a candidate has evidence enough for a visit: a close-up that found him fairly sure, or, before any
        close-up, two boxes or a sure one in the wide frame."""
        found = [v for v in c.looks if v > 0.0]
        if found:
            return max(found) >= VISIT_LOOK_MIN
        # A close-up seen only past a crown is not a look, but a sure one is reason enough to go and look clear of it
        # (real field: 4 standing men, one blocked look each at 0.77-0.99, went unreported while the drone went home
        # with 50-90 s left).
        if CROWN_LOOK_VISIT > 0.0 and c.blocked_best >= CROWN_LOOK_VISIT:
            return True
        return c.hits >= 2 or c.feed_best >= 0.5

    def _switch(self, mode: str, now: float, why: str) -> None:
        self._log("mode", round(now, 2), mode, why)
        # Leaving a sweep run part-flown: it resumes from the point of it nearest where the drone is now.
        if self.mode == "search" and self.path_goal is not None and self.path_goal[0] == "run" \
                and self.run_i < len(self.runs):
            a, b = self.runs[self.run_i]
            ab = b - a
            t = float(np.clip(np.dot(self.pos[:2] - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0))
            if t > 0.0 and _segment_gap(self.pos[:2], a, b) < 3.0:
                self.runs[self.run_i] = np.array([a + t * ab, b])
        self.mode = mode
        self.path = None

    def _viewpoint(self, cand):
        """A free point near a candidate that sees him past every crown, as near the drone as can be, and where
        nobody else known stands in line with him (else the nearest point where someone does)."""
        r0 = max(VIEW_MIN_M, self._clear_r(cand.xy) + 0.8)
        self.view_why = {"r0": round(r0, 1)}
        near = self._others(cand, CROWD_NEAR_M)
        # A man waiting for a look from a second place is looked at from a spot well away from the last one.
        last = cand.zoom_from[:2] if STILL_VISIT_S > 0.0 and cand.wait == "still" and cand.zoom_from is not None \
            else None
        best, best_d, lined = None, math.inf, None
        max_r = math.inf
        if VIEW_STEEP and cand.wait == "pos":
            # He waits for a steeper look: a viewpoint farther than that look allows is a wasted visit (11389:
            # every look from the chosen spot was 'not steep enough', twice, until time ran out).
            known = self.ground.support(cand.xy[0], cand.xy[1]) >= 1.0
            drop = CRUISE_Z - (float(self.ground.z(cand.xy[0], cand.xy[1])) + OBJECT_HEIGHT_M)
            max_r = max(r0, VIEW_STEEP_MARGIN * drop / (POS_TAN_MIN if known else POS_TAN_BLIND))
            self.view_why["max_r"] = round(max_r, 1)
        radii = (r0, r0 + 2.0, r0 + 4.0)
        if CROWN_LOOK_VISIT > 0.0 and cand.blocked_best >= CROWN_LOOK_VISIT and not any(v > 0.0 for v in cand.looks):
            # Seen only past a crown: near him the ground is mostly crowns and their margins, so look from farther
            # out as well (the 7x close-up makes people out from 15-40 m).
            radii = radii + (r0 + 7.0, r0 + 10.0, r0 + 13.0)
        for radius in radii:
            if radius > max_r + 1e-6:
                continue
            for k in range(24):
                ang = 2.0 * math.pi * k / 24.0
                spot = cand.xy + radius * np.array([math.cos(ang), math.sin(ang)])
                if not self.field.holds(spot):
                    self.view_why["held"] = self.view_why.get("held", 0) + 1
                    continue
                if self._shadow_near(spot, cand.xy, SHADOW_KEEP_M + 0.5):
                    self.view_why["shadow"] = self.view_why.get("shadow", 0) + 1
                    continue
                if last is not None and float(np.hypot(*(spot - last))) < STATIC_MOVED_M + 0.5:
                    continue
                line = np.linspace(spot, cand.xy, 12)
                if any(self.field.crown_at(p) < 0.3 for p in line):
                    self.view_why["crown"] = self.view_why.get("crown", 0) + 1
                    continue
                d = float(np.hypot(*(spot - self.pos[:2])))
                if near and self._lined_up(spot, cand.xy, near):
                    if lined is None or d < lined[1]:
                        lined = (spot, d)
                    continue
                if d < best_d:
                    best, best_d = spot, d
            if best is not None:
                return best
        if lined is None and SIGHTLINE_VIEW and cand.seen_from:
            # No viewpoint passes the crown circles: the wide camera did see him from these places, so fly in
            # along one of those lines to the nearest free spot outside his zone (the steepest look on it).
            for d in reversed(cand.seen_from):
                vec = d - cand.xy
                dist = float(np.hypot(*vec))
                if dist < r0:
                    continue
                u = vec / dist
                for radius in np.arange(r0, min(dist, max_r) + 0.01, 1.0):
                    spot = cand.xy + radius * u
                    if self.field.holds(spot) and not self._shadow_near(spot, cand.xy, SHADOW_KEEP_M + 0.5):
                        self.view_why["sightline"] = round(float(radius), 1)
                        return spot
        return None if lined is None else lined[0]

    # ------------------------------------------------------------------------------------------------ flying

    def _goal(self):
        """Where the flight goes now: the next sweep run's start or end, the visit's viewpoint, or home."""
        if self.mode == "visit":
            return self.visit["spot"], "visit"
        if self.mode == "home":
            return np.zeros(2), "home"
        if self.mode == "fill":
            return (self.fill["spot"], "fill") if self.fill is not None else (None, "none")
        while self.run_i < len(self.runs):
            run = self.runs[self.run_i]
            return run[0], "run"
        return None, "none"

    def _plan(self):
        goal, kind = self._goal()
        if goal is None:
            return None
        here = self.pos[:2]
        if kind == "run":
            run = self.runs[self.run_i]
            lead = self.field.path(here, run[0])
            if lead is None:
                self.run_i += 1
                return None
            tail = self.field.path(run[0], run[1])
            if tail is None:
                pts = lead
            else:
                pts = np.vstack([lead, tail[1:]])
            self.path_goal = ("run", self.run_i)
        else:
            pts = self.field.path(here, goal)
            self.path_goal = (kind, None)
        if pts is None or len(pts) < 2:
            return None
        keep = [pts[0]]
        for p in pts[1:]:
            if float(np.hypot(*(p - keep[-1]))) > 0.05:
                keep.append(p)
        if len(keep) < 2:
            return None
        return np.asarray(keep, dtype=float)

    def _fly(self, state, action, now) -> None:
        here = self.pos[:2]
        vel = np.zeros(2)
        face = None
        aim = None
        if self.mode == "home" and self._return_clear(here):
            # The dock's flight home starts where the drone stops, not where the button is pressed: moving, it drifts
            # on a few metres first (seed 9642: pressed at 2.7 m/s, it drifted 3.5 m and the line home from there cut
            # a crown). Brake until the line from the stopping point is clear too.
            stop_ok = True
            if RETURN_FROM_STOP:
                v = np.asarray(state[4:6], float)
                sp = float(np.hypot(*v))
                if sp > 0.3:
                    stop_ok = self._return_clear(here + v * (GUARD_LAG_S + sp / (2.0 * GUARD_BRAKE_MPS2)))
                    if not stop_ok and self.decision % 10 == 0:
                        self._log("return_wait", round(now, 2), round(sp, 2))
            if stop_ok and self.decision > self.home_pressed + 1:
                action[A_RETURN] = 1.0
                self.home_pressed = self.decision
                self._log("return", round(now, 2), round(here[0], 2), round(here[1], 2))
        else:
            if self.path is None and now >= self.plan_retry_at:
                self.path = self._plan()
                self.path_i = 0
                if self.path is not None:
                    self.plan_fails = 0
                    self._log("path", round(now, 2), self.mode, len(self.path), round(_path_length(self.path), 1))
                elif POCKET_ESCAPE and not self.escaping and self.escapes < ESCAPE_MAX and self._boxed_in():
                    # Boxed in by the circles around people found since the drone came here: out through the gaps
                    # the rule itself leaves, slowly, then the full margin again.
                    self.escaping, self.escape_t = True, now
                    self.escapes += 1
                    self._update_zones()
                    self.plan_retry_at = now
                    self._log("escape", round(now, 2), self.mode, [round(float(v), 1) for v in self.pos[:2]])
                else:
                    # No way there now: wait a second before searching again, and give up a target after two tries.
                    self.plan_fails += 1
                    self.plan_retry_at = now + 1.0
                    self._log("no_path", round(now, 2), self.mode, self.plan_fails)
                    if self.plan_fails >= 2:
                        self._unreachable(now)
            if self.path is not None:
                vel, left = self._follow(here)
                if left < ARRIVE_M and float(np.hypot(*state[4:6])) < 1.0 + (2.0 if self.mode == "search" else 0.0):
                    if self.mode == "search" and self.path_goal[0] == "run":
                        self.run_i += 1
                        self.path = None
                    elif self.mode == "visit" and self.visit["arrived"] is None:
                        self.visit["arrived"] = now
                        self._log("visit_at", round(now, 2), self.visit["cand"])
                    elif self.mode == "fill" and self.fill is not None and self.fill["arrived"] is None:
                        self.fill["arrived"] = now
            if self.mode == "visit":
                cand = next((c for c in self.cands if c.id == self.visit["cand"]), None)
                if cand is not None:
                    face = cand.xy
                    aim = cand.xy
            elif self.mode == "fill" and self.fill is not None:
                face = self.fill["xy"]
                aim = self.fill["xy"]
        actual = np.asarray(state[4:6], float)
        moving = float(np.hypot(*actual))
        ask = float(np.hypot(*vel))
        if moving > REVERSE_MPS and ask > 0.0 and float(vel @ actual) < REVERSE_COS * moving * ask:
            if self.decision % 10 == 0:
                self._log("brake_turn", round(now, 2), round(moving, 2))
            vel = np.zeros(2)
            face = None
        vel = self._guard(here, vel, state)
        if self.escaping:
            sp = float(np.hypot(*vel))
            if sp > ESCAPE_MPS:
                vel = vel * (ESCAPE_MPS / sp)
        speed = float(np.hypot(*vel))
        h = math.radians(self.heading)
        action[A_FWD] = (vel[0] * math.sin(h) + vel[1] * math.cos(h)) / MAX_HORIZONTAL_MPS
        action[A_RIGHT] = (vel[0] * math.cos(h) - vel[1] * math.sin(h)) / MAX_HORIZONTAL_MPS
        vz = float(np.clip(HEIGHT_GAIN * (CRUISE_Z - self.pos[2]), -MAX_DESCENT_MPS, MAX_CLIMB_MPS))
        action[A_UP] = vz / (MAX_CLIMB_MPS if vz >= 0.0 else MAX_DESCENT_MPS)
        want = None
        if face is not None and float(np.hypot(*(face - here))) > 0.5:
            want = math.degrees(math.atan2(face[0] - here[0], face[1] - here[1]))
        elif speed > TURNING_MPS:
            want = math.degrees(math.atan2(vel[0], vel[1]))
        if want is not None:
            err = _wrap_deg(want - self.heading)
            action[A_TURN] = float(np.clip(HEADING_GAIN * err / MAX_YAW_DEG_S, -1.0, 1.0))
        tilt = DANGER_TILT_DEG if self.danger_ahead else SEARCH_TILT_DEG
        if aim is not None:
            eye_z = self.pos[2] + CAMERA_OFFSET[2]
            drop = eye_z - (float(self.ground.z(aim[0], aim[1])) + OBJECT_HEIGHT_M)
            tilt = -math.degrees(math.atan2(drop, max(0.3, float(np.hypot(*(aim - here))))))
        action[A_GIMBAL] = float(np.clip(tilt / 90.0, -1.0, 0.0))

    def _boxed_in(self, full: bool = False) -> bool:
        """Whether the free ground the drone can reach from where it is is a small pocket (under POCKET_CELLS cells);
        with full, on the ground the full margin round everyone leaves (the drone inside a circle is still in)."""
        f = self.field
        if f is None:
            return False
        free = f.free
        if full:
            free = f.base.copy()
            for e, n, r in self.zones_full:
                if r > 0.0:
                    free &= np.hypot(f.gx - e, f.gy - n) >= r
        here = self.pos[:2]
        start = f.cell(here)
        if not free[start]:
            if full:
                return True
            start = f.nearest_free(here)
            if start is None:
                return True
        seen = {start}
        todo = [start]
        nx, ny = f.size
        while todo:
            i, j = todo.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (i + di, j + dj)
                    if nb not in seen and 0 <= nb[0] < nx and 0 <= nb[1] < ny and free[nb]:
                        seen.add(nb)
                        if len(seen) >= POCKET_CELLS:
                            return False
                        todo.append(nb)
        return True

    def _unreachable(self, now) -> None:
        """The current target cannot be reached over the free ground: drop it."""
        self.plan_fails = 0
        self.plan_retry_at = -1.0
        if self.mode == "visit" and self.visit is not None:
            cand = next((c for c in self.cands if c.id == self.visit["cand"]), None)
            if cand is not None:
                cand.visits += 1
                cand.need_visit = False
            self._switch("search", now, "visit_unreachable")
        elif self.mode == "fill" and self.fill is not None:
            self.fill_skip.add(self.fill["cell"])
            self._next_fill(now)
        elif self.mode == "search" and self.run_i < len(self.runs):
            self.run_i += 1
        # Home: keep trying; the guards hold the drone meanwhile and the return check runs every decision.

    def _follow(self, here):
        route = self.path
        self.path_i = _progress(route, self.path_i, here)
        carrot, left, bend = _ahead(route, self.path_i, here, CARROT_M)
        for reach in NEAR_CARROTS_M:
            if self.field.clear(here, carrot):
                break
            carrot = _ahead(route, self.path_i, here, reach)[0]
        top = CRUISE_MPS
        if DANGER_GRADED:
            # As fast as still lets the drone stop clear of a man first seen DANGER_SEE_M ahead on the highest ground
            # it is about to cross, never slower than the flat cap.
            need = self._clearance_ahead(route, here)
            self.danger_ahead = need > 0.0
            if need > 0.0:
                top = min(top, max(DANGER_MPS, _stop_speed(DANGER_SEE_M - need, DANGER_REACT_S, DANGER_BRAKE_MPS2)))
        else:
            agl = self.pos[2] - float(self.ground.z(here[0], here[1]))
            if agl < LOW_AGL_M:
                top = LOW_MPS
            # Ground ahead high enough that someone lying there would be within the 3D rule's reach of the drone:
            # slow, so a man first seen at the edge of what the camera makes out can still be kept clear of.
            self.danger_ahead = self._danger_ahead(route, here)
            if self.danger_ahead:
                top = min(top, DANGER_MPS)
        speed = min(top, math.sqrt(2.0 * BRAKE_MPS2 * left), math.sqrt(CORNER_MPS ** 2 + 2.0 * BRAKE_MPS2 * bend))
        to = carrot - here
        dist = float(np.hypot(*to))
        speed = min(speed, dist * 1.5)
        vel = to / dist * speed if dist > 1e-6 else np.zeros(2)
        return vel, left

    def _danger_ahead(self, route, here) -> bool:
        """Whether the ground within DANGER_LOOK_M along the route (and under the drone) is high enough that a person
        standing there would be within the rule's 5 m (plus DANGER_MARGIN_M) of the drone straight above him."""
        pts = [np.asarray(here, float)]
        gone = 0.0
        last = np.asarray(here, float)
        for p in route[self.path_i + 1:]:
            step = float(np.hypot(*(p - last)))
            if gone + step >= DANGER_LOOK_M:
                pts.append(last + (p - last) * ((DANGER_LOOK_M - gone) / max(step, 1e-9)))
                break
            pts.append(np.asarray(p, float))
            gone += step
            last = np.asarray(p, float)
        line = np.vstack([np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / 2.0) + 2)) for a, b in zip(pts[:-1], pts[1:])]) \
            if len(pts) > 1 else np.array(pts)
        top = float(np.max(self.ground.z(line[:, 0], line[:, 1])))
        return CRUISE_Z - (top + PERSON_SLOPE_M + PERSON_H_M) < SAFE_M + DANGER_MARGIN_M

    def _clearance_ahead(self, route, here) -> float:
        """The horizontal distance the rule's 5 m (plus DANGER_MARGIN_M) needs from a man standing on the highest
        ground within DANGER_GRADED_LOOK_M along the route (and under the drone); 0 when he could stand anywhere."""
        pts = [np.asarray(here, float)]
        gone = 0.0
        last = np.asarray(here, float)
        for p in route[self.path_i + 1:]:
            step = float(np.hypot(*(p - last)))
            if gone + step >= DANGER_GRADED_LOOK_M:
                pts.append(last + (p - last) * ((DANGER_GRADED_LOOK_M - gone) / max(step, 1e-9)))
                break
            pts.append(np.asarray(p, float))
            gone += step
            last = np.asarray(p, float)
        line = np.vstack([np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / 2.0) + 2)) for a, b in zip(pts[:-1], pts[1:])]) \
            if len(pts) > 1 else np.array(pts)
        top = float(np.max(self.ground.z(line[:, 0], line[:, 1])))
        gap = CRUISE_Z - (top + PERSON_SLOPE_M + PERSON_H_M)
        reach = SAFE_M + DANGER_MARGIN_M
        if gap >= reach:
            return 0.0
        return reach if gap <= 0.0 else math.sqrt(reach * reach - gap * gap)

    def _guard(self, here, vel, state):
        """Keep off the edge, the crowns and everyone found, whatever the route says: towards each the drone never
        flies faster than it can brake before a margin (the controller eases at about 2 m/s2, and a turn near a crown
        swings wide), and inside the margin it is pushed back out."""
        f = self.field
        actual = np.asarray(state[4:6], float)

        def hold(vel, away, gap, stop_at, push_at, what, push_max=0.0):
            towards = -float(vel @ away)
            moving = -float(actual @ away)
            room = gap - stop_at - max(0.0, moving) * GUARD_LAG_S
            allowed = math.sqrt(2.0 * GUARD_BRAKE_MPS2 * max(0.0, room))
            # Already closing faster than that: ask for the way out, harder the bigger the excess (the drone's own
            # closing speed counts, not only what was asked of it).
            allowed -= 1.5 * max(0.0, moving - allowed)
            if towards > allowed:
                vel = vel + (towards - allowed) * away
            if gap < push_at:
                push = (push_at - gap) * 2.0
                vel = vel + away * (min(push, push_max) if push_max > 0.0 else push)
                if self.decision % 10 == 0:
                    self._log(what, round(self.now, 2), round(gap, 2))
            return vel

        extra = WIND_GUARD_M_PER_MPS * max(0.0, float(state[S_WIND])) if len(state) > S_WIND else 0.0

        def edge_hold(vel):
            edge = min(f.edge_at(here), float(state[S_EDGE]) if state[S_INSIDE] > 0.5 else -1.0)
            if edge < 8.0 + extra:
                vel = hold(vel, f.inward(here), edge, EDGE_STOP_M + extra, EDGE_GUARD_M + extra, "edge_guard")
            return vel

        def crown_holds(vel):
            if len(f.crowns):
                gaps = np.hypot(f.crowns[:, 0] - here[0], f.crowns[:, 1] - here[1]) - f.crowns[:, 2]
                for k in np.nonzero(gaps < 8.0 + extra)[0]:
                    away = here - f.crowns[k, :2]
                    vel = hold(vel, away / max(1e-6, float(np.hypot(*away))), float(gaps[k]), CROWN_STOP_M + extra,
                               CROWN_GUARD_M + extra, "crown_guard")
            return vel

        def zone_holds(vel):
            near = []
            for e, n_, r in f.zones:
                if r <= 0.0:
                    continue
                away = here - np.array([e, n_])
                d = float(np.hypot(*away))
                if d - r < 8.0:
                    near.append((away / max(d, 1e-6), d - r))
            if ZONE_INSIDE_FIRST:
                # The way out of a zone the drone is in comes first and the holds of the zones it is outside come
                # last, so that way out never carries it into someone else's zone (seeds 3141 and 10062: a shadow
                # taken for a man beside the drone pushed it into the real man's zone, 5 m rule broken).
                near.sort(key=lambda z: z[1] >= 0.0)
            for away, gap in near:
                vel = hold(vel, away, gap, 0.0, 0.3, "person_guard", ZONE_PUSH_MAX_MPS)
            return vel

        if GUARD_EDGE_LAST:
            # A way out of a zone that just appeared around the drone never takes it over the edge or into a crown:
            # those holds come last (seed 3031: pushed out of a zone at ~6 m/s, the drone left the field).
            vel = edge_hold(crown_holds(zone_holds(vel)))
        else:
            vel = zone_holds(crown_holds(edge_hold(vel)))
        sp = float(np.hypot(*vel))
        if sp > CRUISE_MPS:
            vel = vel * (CRUISE_MPS / sp)
        return vel

    def _return_clear(self, here) -> bool:
        """Whether the dock's straight flight home from here keeps off the edge, the crowns and everyone found."""
        f = self.field
        dist = float(np.hypot(*here))
        pts = np.linspace(np.asarray(here, float), np.zeros(2), max(2, int(dist / 0.5) + 2))
        if np.min(f.edge_many(pts)) < RETURN_EDGE_M:
            return False
        if len(f.crowns):
            gaps = np.hypot(pts[:, None, 0] - f.crowns[None, :, 0], pts[:, None, 1] - f.crowns[None, :, 1]) - f.crowns[None, :, 2]
            if float(gaps.min()) < RETURN_CROWN_M:
                return False
        for e, n, r in (self.zones_full if self.escaping else f.zones):
            if r > 0.0 and float(np.min(np.hypot(pts[:, 0] - e, pts[:, 1] - n))) < r + 0.3:
                return False
        return True
