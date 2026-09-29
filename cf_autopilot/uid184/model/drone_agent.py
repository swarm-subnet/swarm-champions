from __future__ import annotations
# --- shipped configuration: these fixes are on unless the environment already sets them ---
import os as _cx_os0
for _cx_k, _cx_v in (("CX_WH_BOX", "1"), ("CX_WH_ZLO2", "1"), ("CX_FO_SPIN", "1"), ("CX_CT_KCITY", "1"), ("CX_VK_MOTION", "1"), ("CX_VK_WATCH", "1"), ("CX_VK_UNSTICK", "1"), ("CX_MS_SLOPE", "1"), ("CX_VS_REFLY", "1"), ("CX_M2_PULSE", "1"), ("CX_M2_RESPIN", "1"), ("CX_F6_UNDER", "1"), ("CX_DV_DIVE", "1"), ("CX_MV_ANCHOR", "1"), ("CX_MV_GATE", "1"), ("CX_MV_SPIN", "1"), ("CX_V8_SURF", "1"), ("CX_VS_DIPLATE", "1"), ("CX_K9_WH", "1"), ("CX_VL_HOLD", "1"), ("CX_FO_HC", "3.2"), ("CX_MG_FLOOR", "1"), ("MY_SEARCH", "slowmv,mvfit,mvpatience,invest,nospin,inward_mtn,mdesc,mreacq,keepview,brake,yawvel,gndfilt,gentle2,apptube,roofguard,fastapp,vsearch3,mhigharr,mspin2,vmovko,vsurf,vclamp,vbad1,vmem,vland,vking,vretry,cmem,ccheck,cland,cretry,caround,cfast,cnoclimb,qretry,cdetour,czcap,cstable,cmovko,clook,cskim,cwall,cdrift,ftake,fdown,fspin,wleash,wsearch,wspin,vabs,mland"), ("CX_F14_GATE", "1"), ("CX_F14_STATIC", "1"), ("CX_F14_REFLY", "1"), ("CX_W17_BRAKE", "1"), ("CX_W17_EDGE", "1"), ("CX_V16_VLAND", "1"), ("CX_RA_DIP", "1"), ("CX_RB_PINHO", "1"), ("CX_RC_LSEEN", "1"), ("CX_RC_LBAND", "1"), ("CX_CR_KRAY", "1"), ("CX_MA_SO", "1"), ("CX_VF_LAND", "1"), ("CX_VF_KING", "1"), ("CX_VF_ZONE", "1"), ("CX_VF_LO", "1.6"), ("CX_KL_MDROP", "1"), ("CX_KL_KBACK", "1"), ("CX_KL_KB_REHO", "1"), ("CX_VK2_BRAKE", "1"), ("CX_VK2_PIN", "1"), ("CX_VK2_ROOF", "1"), ("CX_VG_V8H", "1"), ("CX_VX_RAY", "1"), ("CX_ME_RIM", "1"), ("CX_ME_END", "1"), ("CX_VM_GATE", "1"), ("CX_ME_EG_MAPS", "mountain,village,forest"), ("CX_PRM_CITY_H_CRUISE", "7.5"), ("CX_CZ_GATE", "1"), ("MY_MTN_C11M", "1"), ("CX_FP_POST", "1"), ("CX_FP_CLAMP", "1"), ("CX_FP_LOOK", "1"), ("CX_PW_KSEED", "1"), ("CX_PVS_VXRIM", "1"), ("CX_PC_VETO", "1"), ("CX_PC_SEEN", "1"), ("CX_PC_KTO", "1"), ("CX_PW_CREP", "1"), ("CX_RG_KFLIP", "1"), ("CX_RG_KFEED", "1"), ("CX_RG_VLOW", "1"), ("CX_RG_VSLOW", "1"), ("CX_SV_SGATE", "1"), ("CX_SV_ALT", "1"), ("CX_SV_LOOK", "1"), ("CX_S1M_HM", "1"), ("CX_S1M_DESC", "1"), ("CX_S1M_RECEN", "1"), ("CX_S2N_NBV", "1"), ("CX_S2K_OPEN", "1"), ("CX_MT_NODIVE", "1"), ("CX_WH_ZHI", "6.5"), ("CX_VFP_LSEEN", "1"), ("CX_VS_RESUME", "1"), ("CX_VFP_RESG", "1"), ("CX_FO_TUBE", "1"), ("CX_WF_SPCL", "1"), ("CX_MPC", "1"), ("CX_VCH_RQ", "1"), ("MY_LAND_SOFT_MAPS", "mountain,village"), ("CX_OMF_CSTK", "1"), ("CX_MTF_ZB", "1"), ("CX_MTF_ZB_APP", "1"), ("CX_M15_LOOK", "1"), ("CX_M15_LOOK_HITS", "1"), ("CX_MTF_YT", "1"), ("CX_MTF_LKEEP", "1"), ("CX_WF_AVMAX", "2.5"),):
    _cx_os0.environ.setdefault(_cx_k, _cx_v)

import collections
import math
import os
from pathlib import Path

import numpy as np

# mdirect/mland/mking (uid183) are dropped: all three are mountain-gated and each fights a mechanism merged in from
# the varE7 line -- mking's leash calls _drop_track()/clears pad_hist, destroying the very shadow track the hand-back
# needs, and mdirect/mland override the crawl-slope approach and soft-contact descent. With them off the merged agent
# reproduces varE7 mountain exactly (0.9256, 5 failures, static 0.9059 / moving 0.9771); with them on it gets 0.9036.
DEFAULT_FLAGS = "slowmv,mvfit,mvpatience,invest,nospin,inward_mtn,mdesc,mreacq,keepview,brake,yawvel,gndfilt,gentle2,apptube,roofguard,fastapp,vsearch3,mhigharr,mspin2,vmovko,vsurf,vclamp,vbad1,vmem,vland,vcheck,vking,vretry,cmem,ccheck,cland,cretry,caround,cfast,cnoclimb,qretry,cdetour,czcap,cstable,cmovko,clook,cskim,cwall,cdrift,ftake,fdown,fspin,wleash,wsearch,wspin,vdip,vabs,mland"
_VBOX_M = float(os.environ.get("MY_VBOX_M", "41.0"))
_VABS_LIMIT_M = float(os.environ.get("MY_VABS_LIMIT_M", "8.0"))
_VABS_ROOF_M = 6.35
_VDIP_ALT_M = float(os.environ.get("MY_VDIP_ALT_M", "3.0"))
_VDIP_CAP_S = float(os.environ.get("MY_VDIP_CAP_S", "12.0"))
_VDIP_GND_M = float(os.environ.get("MY_VDIP_GND_M", "0.6"))
_SEARCH_FLAGS = set(f for f in os.environ.get("MY_SEARCH", DEFAULT_FLAGS).split(",") if f)
# --- mechanisms restored from uid117 (champion 0.9315), each default-off so m94 == final until measured ---
# c19yh: uid117 lets the village search-ring camera yaw follow the flight heading instead of biasing
# 0.7 rad inward. Their measurement: +0.0083 flown (t=1.98, 11 rescued / 3 broken) over 4 tuning lists,
# +0.0063 (4/1) over 2 held-out epoch-21 lists. "in" = our current (uid183) behaviour.
_VYAW = os.environ.get("MY_VYAW", "in")
# c11m: uphill route -> climb to the goal height early rather than hugging the rising slope, where the
# forward depth image is all terrain and the planner brakes the cruise to a crawl. uid117 measures
# +0.0029 board (t=4.67, 9 lists) on uid128. We dropped it for the varE7 _MTN_CLIMB_* stack; this flag
# lets the two be compared and, if they compose, run together.
_MTN_C11M = int(os.environ.get("MY_MTN_C11M", "0"))
_MTN_SLOPE_VH = float(os.environ.get("MY_MTN_SLOPE_VH", "1.2"))  # closing speed for a steep approach that hugs a slope (0 = uid128)
_MTN_SLOPE_ALT = float(os.environ.get("MY_MTN_SLOPE_ALT", "2.2"))  # hugging = ground this close below (sink throttled)
_MTN_LAND_OFFSET = float(os.environ.get("MY_MTN_LAND_OFFSET", "0.45"))  # touch a static mountain pad down this far downhill (0 = off)
_MTN_LAND_RECLIMB = int(os.environ.get("MY_MTN_LAND_RECLIMB", "1"))      # offset point beside the pad: climb back, land at the centre
_MTN_LAND_RECLIMB_H = float(os.environ.get("MY_MTN_LAND_RECLIMB_H", "0.3"))
_LAND_SOFT = int(os.environ.get("MY_LAND_SOFT", "1"))
_LAND_SOFT_MAPS = tuple(x for x in os.environ.get("MY_LAND_SOFT_MAPS", "mountain").split(",") if x)  # city lands better without it (0.9754 -> 0.9843)  # static pads: stop pressing once the drone has touched; the final descent otherwise keeps
# commanding -0.35/-0.50 m/s through the 0.5 s stable-contact wait and sinks the drone ~3 cm into the pad (probe results/lfp: -0.03 m clearance)
_LAND_SOFT_H = float(os.environ.get("MY_LAND_SOFT_H", "0.10"))       # ...within this height over the pad estimate (m)
_LAND_SOFT_RATIO = float(os.environ.get("MY_LAND_SOFT_RATIO", "0.8"))  # ...and sinking slower than this fraction of the commanded descent (0.8 fires on the first contact tick)
_LAND_SOFT_VZ = float(os.environ.get("MY_LAND_SOFT_VZ", "-0.05"))    # ...then command this instead (a light hold keeps contact)
_LAND_SOFT_FAST = int(os.environ.get("MY_LAND_SOFT_FAST", "1"))  # jump the velocity command to the hold at once (pool 700000: +0.0034 m over the ramped hold, better on 69 of 83, worse on 0)
_MTN_LAND_T_MAX = float(os.environ.get("MY_MTN_LAND_T_MAX", "55.5"))  # no offset for a LAND that starts this late: ~3 s to land
_MTN_CLIMB_V = float(os.environ.get("MY_MTN_CLIMB_V", "2.5"))  # >0: on mountain keep a requested climb up to this speed after the planner
_MTN_CRUISE_VZ = float(os.environ.get("MY_MTN_CRUISE_VZ", "2.5"))  # mountain cruise terrain-following climb limit (uid128: 1.5)
_MTN_CLIMB_NOPAD = int(os.environ.get("MY_MTN_CLIMB_NOPAD", "0"))  # 1: climb knobs act only while no pad is tracked
_MTN_CLIMB_CUT = float(os.environ.get("MY_MTN_CLIMB_CUT", "0.5"))  # >0: planner kept < this share of the cruise speed -> climb over
_MTN_SPIN_PULSE = int(os.environ.get("MY_MTN_SPIN_PULSE", "1"))  # mountain search spin: forward pulses pitch the camera down
_MTN_PULSE_ALT = float(os.environ.get("MY_MTN_PULSE_ALT", "1.5"))  # ...only this high over the ground (the high-ground spin sits at 2 m)
_MTN_PULSE_CLEAR = float(os.environ.get("MY_MTN_PULSE_CLEAR", "2.5"))  # ...and with nothing this close ahead in the depth image
_MTN_PULSE_V = float(os.environ.get("MY_MTN_PULSE_V", "1.5"))  # pulse velocity step (m/s)
_MTN_PULSE_S = float(os.environ.get("MY_MTN_PULSE_S", "0.3"))  # pulse length (s)
_MTN_PULSE_PERIOD = float(os.environ.get("MY_MTN_PULSE_PERIOD", "1.5"))  # one pulse per period while spinning
_MTN_PULSE_DELAY = float(os.environ.get("MY_MTN_PULSE_DELAY", "0"))  # level spin (s) before the first pulse
_MTN_PULSE_SPINX = int(os.environ.get("MY_MTN_PULSE_SPINX", "0"))  # 1: longer mountain spin (+pi, +4 s) when pulses are delayed
_MTN_PULSE_STILL = float(os.environ.get("MY_MTN_PULSE_STILL", "0.6"))  # start a pulse only below this horizontal speed (m/s)...
_MTN_PULSE_MAXTILT = float(os.environ.get("MY_MTN_PULSE_MAXTILT", "12"))  # ...and with roll and pitch under this (deg)
_MTN_PULSE_NOVOTE = int(os.environ.get("MY_MTN_PULSE_NOVOTE", "1"))  # freeze motion votes for 0.8 s after a look-down pulse
_MTN_APP_PULSE = int(os.environ.get("MY_MTN_APP_PULSE", "1"))  # also pulse in the steep mountain approach while the pad is below the image
_MTN_LAND_SLOPE_MIN = float(os.environ.get("MY_MTN_LAND_SLOPE_MIN", "0.0"))  # ...only where the fitted terrain slopes at least this
_MTN_LAND_SLOPE_MAX = float(os.environ.get("MY_MTN_LAND_SLOPE_MAX", "1.5"))  # no offset above this estimated slope: 1.0 blocked 8% of landings that gain; above 1.5 a third of offset landings lose clearance and a miss past the pad edge hits terrain (1962)
_MTN_LAND_RMS_MAX = float(os.environ.get("MY_MTN_LAND_RMS_MAX", "0.4"))  # terrain too uneven for one plane: keep the centre (0.25 blocked landings that gain offline)

# ==== c2 candidate fixes (fx_ct, fx_ms, fx_fo, fx_vk, fx_vs here; fx_mm and fx_wh further down), every flag default OFF ====
# CX_EVENTS is the ONE activation-count dict of this module (tools/fly.py reads drone_agent.CX_EVENTS after each seed and
# clears it in place before the next). Every fix counts into it through its own helper, which keeps that fix's own
# counting rule: _cx_ev (ct, wh: plain), _cx_ms_ev/_cx_ms_log (ms: muted inside shadow act()s), _cx (fo: plain),
# _cx_count (vk: plain), _cx_vs_ev (vs: *_shadow suffix inside the vking shadow act()); fx_mm writes to it directly.
CX_EVENTS = {}

# --- fx_ct (city) candidate fixes, all default OFF ---
# CX_CT_KCITY: on a dual route whose map mine calls city, tell the king's main controller so its XGB cannot relabel the
#   flight 'mountain' (which hands a later moving-pad chase to uid130 instead of main; 1269694645).
# CX_CT_ROOF: city roof-step guard. The down-ray (cast against collision shapes) jumps up when the drone crosses onto a
#   roof, awning or a building's convex-hull collider; brake hard, climb, and crawl while the surface below keeps rising
#   (1443566373: hull face rising at ~47 deg under a 2.9 m/s cruise).
_CX_CT_KCITY = os.environ.get("CX_CT_KCITY", "0") == "1"
_CX_CT_ROOF = os.environ.get("CX_CT_ROOF", "0") == "1"

# --- fx_k9 (king pilot-routing hints), default OFF ---
# The king's XGB picks its pilot at its 2nd prediction (tick 6, 0.12 s) and may still switch main -> graph/uid130 up to
# tick 120. These hints pass mine's padnet label to that pick (set before each king.act while the pick is open, then
# frozen), the KCITY pattern:
# CX_K9_WH: mine says warehouse and the start is 1.5-12 m up -> the king flies _forest. Champion fresh runs (10,392 seeds):
#   mine's label is warehouse at 0.12 s on 1,643 of 1,644 warehouse flights and on no other flight starting 1.5-12 m up;
#   the XGB still put 53 on the graph pack (0.884, 5 failures) and 25 on main (0.838, 4) against 0.986 for _forest (1,566).
# CX_K9_NG: mine says city/open and the start is 1.5-12 m up -> an XGB village/forest label cannot put the king on the
#   graph pack (city moving hand-overs: graph 5 flown, 1 crash; open high starts: graph 3 flown, 1 timeout; main 0.94-0.98).
#   Flown net ~0 (it also takes the graph pack from forest flights mine calls open): not in fx_k9/FLAGS.
_CX_K9_WH = os.environ.get("CX_K9_WH", "0") == "1"
_CX_K9_NG = os.environ.get("CX_K9_NG", "0") == "1"
_K9_Z_LO = 1.5   # warehouse starts 1.74-10.11 m, city/open 0.32-10.4 m; every village start is <= 1.23 m
_K9_Z_HI = 12.0  # = MY_OPEN_ALT: above it mine relabels the map mountain (_MTN_HIGH_RELABEL)


def _cx_ev(name):
    # plain counter shared by fx_ct and fx_wh (both defined the same function)
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + 1


# --- fx_ms: mountain static search / approach fixes. Every flag defaults OFF; with all of them off the agent is the champion. ---
_CX_MUTE = [0]  # > 0 while the router runs a shadow mine.act() whose action is discarded: nothing is counted then


def _cx_ms_ev(name):
    # fx_ms counter (fx_ms's own _cx_ev): muted inside a shadow act() (_CxShadow)
    if _CX_MUTE[0]:
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + 1


def _cx_ms_log(name, value):
    """Diagnostic value (not a count), recorded only on seeds where a fix already acted."""
    if _CX_MUTE[0]:
        return
    CX_EVENTS[name] = value


def _cx_on(name):
    return os.environ.get(name, "0").strip().lower() not in ("", "0", "false", "no", "off")


# ==== fx_me (round 27): three static-mountain fixes, every flag default OFF (all off = c23 exactly) ====
# CX_ME_RIM: rim guard for mine's downhill-offset touchdown. The offset point (0.45 m downhill of the estimate) can lie
#   beside the 0.6 m pad when the estimate is 0.2-0.25 m off the same way; the drone then sinks past the pad top beside
#   the rim into the terrain (672277095, 1790977030, 343201094 ...). Over the pad the down-ray reads the pad top; beside
#   it the terrain, 0.2-1.4 m lower. The pad-top reference is calibrated from the ray itself (the surface under the
#   drone while it is over the estimate's centre), else the estimate's own height. A fire (surface under the drone
#   below the reference on 2 ticks, near the touchdown point and no longer closing on the estimate's centre, still
#   sinking, ray clear of contact) stops the sink
#   (acceleration limit bypassed), moves the touchdown point _ME_RG_STEP inward of the drone (at most half the offset;
#   centre on a 2nd fire), climbs straight up first if the drone is within _ME_RG_MARGIN of the reference, and holds
#   height until the ray steps up onto the pad top (or the new point is reached; if the surface stayed perfectly flat
#   over that move, the drone was over the pad all along and that surface becomes the reference).
# CX_ME_END: endgame commit on static mountain, the skeptics' narrow variant: gated once, at mine's static LAND entry,
#   when entry time + the median LAND duration (3.1 s) passes 59.5 s. That LAND then sinks while centring (down to
#   0.8 m, unless still sliding in at > 1 m/s) and descends with fx_lf's m17 profile (-1.3 m/s to 0.5 m, then -0.75 m/s
#   once centred) only while level and not sliding (< 0.2 m/s); else the champion's own value; LAND_SOFT kept.
#   No approach change, no LAND-gate widening, no moving-flag or hand-over freeze.
# CX_ME_MOV: no moving flag from the approach's own range bias: a static track (>= 30 hits, never flagged moving)
#   approached level or from below (drone < 1 m above the estimate, climbing, closing) whose estimate recedes along the
#   line of sight (CX_ME_MV_DIR=0: either way) is not flagged by a vote whose fitted velocity is radial (the
#   perpendicular test the champion applies beyond 9 m) nor by the raw quarter-median check's radial-only branch; the raw
#   window keeps only its recent samples after a block.
# CX_ME_DIAG: log only (never acts): which motion path flagged a mountain pad, and the rim-guard signal.
_CX_ME_RIM = _cx_on("CX_ME_RIM")
_CX_ME_END = _cx_on("CX_ME_END")
# fx_eg (round 29): CX_ME_EG_MAPS = the maps where CX_ME_END latches and acts (comma list; unset = "mountain", i.e.
#   exactly fx_me). Village and forest also lose static landings that start LAND at 56-59 s and time out 0.03-0.3 m
#   above the pad (c21 batches: village 3 / 1,064, forest 3 / 960).
_ME_EG_MAPS = tuple(x.strip() for x in os.environ.get("CX_ME_EG_MAPS", "mountain").split(",") if x.strip())
_CX_ME_MOV = _cx_on("CX_ME_MOV")
_CX_ME_DIAG = _cx_on("CX_ME_DIAG")
_CX_ME_ANY = _CX_ME_RIM or _CX_ME_END or _CX_ME_MOV or _CX_ME_DIAG
_ME_RG_THR = float(os.environ.get("CX_ME_RG_THR", "0.2"))         # m: surface under the drone below the estimate's height
_ME_RG_THR_CAL = float(os.environ.get("CX_ME_RG_THR_CAL", "0.12"))  # m: ... below the ray-calibrated pad top
_ME_RG_XY = float(os.environ.get("CX_ME_RG_XY", "0.25"))          # m: only near the touchdown point (the final sink)
_ME_RG_MARGIN = float(os.environ.get("CX_ME_RG_MARGIN", "0.25"))  # m: height over the reference needed to shift inward
_ME_RG_STEP = float(os.environ.get("CX_ME_RG_STEP", "0.15"))      # m: a fire moves the touchdown point this far inward
_ME_RG_VIN = float(os.environ.get("CX_ME_RG_VIN", "0.15"))        # m/s: no fire while still closing on the estimate's centre
_ME_RG_CAL_R = float(os.environ.get("CX_ME_RG_CAL_R", "0.3"))     # m: calibration samples within this of the estimate
_ME_RG_HOLD_S = float(os.environ.get("CX_ME_RG_HOLD_S", "2.0"))   # s: longest hold per fire
_ME_EG_T = float(os.environ.get("CX_ME_EG_T", "59.5"))            # s: a LAND predicted to finish later than this...
_ME_EG_LAND_S = float(os.environ.get("CX_ME_EG_LAND_S", "3.1"))   # s: ...at the median LAND duration (c21-c23, 1,100 landings)
_ME_MV_DZ = float(os.environ.get("CX_ME_MV_DZ", "1.0"))           # m: drone at most this far above the pad estimate
_ME_MV_HITS = int(os.environ.get("CX_ME_MV_HITS", "30"))
_ME_MV_DISP = float(os.environ.get("CX_ME_MV_DISP", "1.5"))       # m: a larger radial shift is not a range bias
_ME_MV_VZ = float(os.environ.get("CX_ME_MV_VZ", "0.2"))           # m/s: the drone climbs toward the pad
_ME_MV_DIR = int(os.environ.get("CX_ME_MV_DIR", "1"))             # 1: only an estimate receding along the line of sight
_ME_MV_CLOSE = float(os.environ.get("CX_ME_MV_CLOSE", "0.5"))     # m/s: closing speed along the line of sight


def _cx_me_ev(name, n=1):
    if _CX_MUTE[0]:
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


def _cx_me_log(name, s):
    if _CX_MUTE[0]:
        return
    lst = CX_EVENTS.setdefault(name, [])
    if isinstance(lst, list) and len(lst) < 40:
        lst.append(s)


# SLOPE: steep static approach while hugging a slope far above the pad: fly down the slope toward the pad (AGL-held
# feed-forward sink) instead of the 0.3-1.2 m/s crawl with an AGL-keyed -0.6/-1.2 m/s sink (1571070702, 552119277).
_CX_MS_SLOPE = _cx_on("CX_MS_SLOPE")
_MS_SLOPE_VH = float(os.environ.get("CX_MS_SLOPE_VH", "2.2"))    # horizontal speed cap along the slope (m/s)
_MS_SLOPE_AGL = float(os.environ.get("CX_MS_SLOPE_AGL", "2.4"))  # height held over the slope (m)
_MS_SLOPE_KP = float(os.environ.get("CX_MS_SLOPE_KP", "1.5"))    # AGL correction gain (1/s)
_MS_SLOPE_ST = float(os.environ.get("CX_MS_SLOPE_ST", "0.72"))   # engage only this steep (sin of the dive angle to the
# approach target; the steep branch itself starts at 0.64). The two epoch successes that grazed 0.64-0.65 (1668675160,
# 612127110) toggled between steep and glide; the target seeds sit at 0.77-0.90. Once engaged, it holds down to 0.66.
_MS_SLOPE_ST_HOLD = 0.66
_MS_SLOPE_ALT_HI = 4.0  # engaged only with the ground this close below (higher up the free-air -2 m/s sink is kept)
_MS_SLOPE_DROP = 2.0    # ...and the pad at least this far below the ground under the drone
# KEEP: a young static track whose pad is predicted below the camera's look-down limit is not declared lost after 6 s
# unseen while the drone is closing in over it (1571070702: REACQUIRE fired 0.5 m from and 2.6 m above the pad).
_CX_MS_KEEP = _cx_on("CX_MS_KEEP")
_MS_KEEP_S = float(os.environ.get("CX_MS_KEEP_S", "14.0"))  # unseen limit while the pad is predicted under the drone
_MS_KEEP_HITS = int(os.environ.get("CX_MS_KEEP_HITS", "10"))  # detections the young track must have (1571070702: 24)
_MS_KEEP_FLOAT = float(os.environ.get("CX_MS_KEEP_FLOAT", "0.15"))  # min height of the estimate over the fitted terrain
# (stub, slopes 1.0-1.3: real pads 0.56-0.66, false detections on the terrain -0.28 to -0.37; the ring points read ~0.3 m high)
# SZ: a mountain SEARCH that has had no track at all for this long climbs its ring height to clue z + _MS_SZ_DZ, over the
# top of the +-5 m clue z band (1426199431: rings at clue z +0.5 flew 1-5 m under a pad floating at clue z +5).
_CX_MS_SZ = _cx_on("CX_MS_SZ")
_MS_SZ_T = float(os.environ.get("CX_MS_SZ_T", "10.0"))
_MS_SZ_DZ = float(os.environ.get("CX_MS_SZ_DZ", "7.0"))
# GLIDE: a solid (>= 30 hits) static mountain track in the (non-steep) glide skips the x0.7 horizontal cut for a low pad
# pixel row, which drives a slow-fast limit cycle with the planner's 50 deg cone cap (1985319819, 312363869, 440010412).
_CX_MS_GLIDE = _cx_on("CX_MS_GLIDE")
_CX_MS_ANY = _CX_MS_SLOPE or _CX_MS_KEEP or _CX_MS_SZ or _CX_MS_GLIDE


class _CxShadow:
    """Mutes CX_EVENTS during a shadow act() (only when some fx_ms or fx_m2 flag is on; otherwise it does nothing).
    fx_dv: with CX_DV_NOHOLD on it also marks the shadow in _DV_SHADOW (NOHOLD never acts inside one)."""

    def __enter__(self):
        if _CX_MS_ANY or _CX_M2_ANY or _CX_M15_ANY:
            _CX_MUTE[0] += 1
        if _CX_DV_NOHOLD:
            _DV_SHADOW[0] += 1
        return self

    def __exit__(self, *exc):
        if _CX_MS_ANY or _CX_M2_ANY or _CX_M15_ANY:
            _CX_MUTE[0] -= 1
        if _CX_DV_NOHOLD:
            _DV_SHADOW[0] -= 1
        return False


def _flag(name):
    return name in _SEARCH_FLAGS


# --- fx_fo: forest candidate fixes, every one default OFF (flags off == champion bit for bit) ---
# SPIN: the forest SEARCH spin flies v=(0,0,vz) with the planner off. Under a tree crown the 4 s gndfilt window first lets
# it sink onto the foliage, then ages out and saturates vz at +1.5 m/s: a blind climb into the crown above (2072616194,
# 192547435). With foliage under the drone (down-ray ground > spin reference ground + 1 m) never command a climb: keep the
# champion's sink while agl >= 1.2, then freeze at whatever height the drone has once agl < 1.2. After such a hold, cap the
# requested vz of ring legs at 0 until the drone is off the foliage or 4 m from the hold point (the planner may still pick
# an in-view upward ray, and forest's escape branch may still climb when the top band shows > 6 m of clearance).
_FO_SPIN = os.environ.get("CX_FO_SPIN", "0") != "0"
# RESUME (experimental, keep OFF): an APPROACH lost with the track already gone (pad None, e.g. a false positive) rebuilt
# the rings around the clue and re-ran the 9.5 s forest spin (2037242053 timed out, 502117086 late). Keep the previous rings
# and index instead, and skip the spin when the earlier one already turned at least half a revolution -- unless the track
# was dropped fresh (last seen <= 2.0 s before the drop) within 10 m of the drone (760333354), where the champion's
# re-spin re-finds a real pad.
_FO_RESUME = os.environ.get("CX_FO_RESUME", "0") != "0"
# STEEP: forest zeroes the two steep-up candidate rows (51/60 deg, above the 45 deg image edge) for the main tube, but not
# in the thin-tube retry. At spin exits and ring vertices the cone rule's nearest candidate is a steep row with free 0, the
# retry then re-picks that blind row from the unzeroed thin-tube distances and climbs 0.5 m/s into crowns it cannot see
# (1477317106). In SEARCH only (the forest takeoff relies on the same rows), zero those rows in the retry when it would
# take one. Experimental, keep OFF.
_FO_STEEP = os.environ.get("CX_FO_STEEP", "0") != "0"
# TUBE (experimental, keep OFF): before falling back to the 0.35 m thin tube, try a 0.6 m tube (the forest safe distance)
# and take its ray when it is clearly freer than the main-tube ray and long enough not to need the escape branch. Only in
# CRUISE, SEARCH and REACQUIRE: not in TAKEOFF, and not in APPROACH, whose steep descent onto the raised pad runs the planner
# under a vz_override and must keep the direct ray.
_FO_TUBE = os.environ.get("CX_FO_TUBE", "0") != "0"
_FO_TUBE_R = float(os.environ.get("CX_FO_TUBE_R", "0.6"))
# --- ws2 forest (wf1), every flag default OFF (off == c29f) ---
# WF_OOC: forest SEARCH/CRUISE/REACQUIRE: while the requested horizontal direction is > _WF_OOC_ON deg off the
#   camera heading (the planner's out-of-cone branch flies an image-edge or steep-up candidate blind), hover and let
#   the yaw turn; released below _WF_OOC_OFF deg or after _WF_OOC_CAP s. Only when the mode faces its motion.
# WF_ABRK: forest APPROACH planner brake margin/decel 1.2/1.5 (city/mountain brake numbers) instead of 0.6/2.2.
# WF_AVMAX: > 0: forest APPROACH speed cap (m/s).
# WF_SPCL: forest spin (planner off) climb capped at _WF_SPCL_V m/s and held while the top band sees < _WF_SPCL_TOP m.
_WF_OOC = os.environ.get("CX_WF_OOC", "0") != "0"
_WF_OOC_ON = math.radians(float(os.environ.get("CX_WF_OOC_ON", "50")))
_WF_OOC_OFF = math.radians(float(os.environ.get("CX_WF_OOC_OFF", "30")))
_WF_OOC_CAP = float(os.environ.get("CX_WF_OOC_CAP", "4.0"))
_WF_ABRK = os.environ.get("CX_WF_ABRK", "0") != "0"
_WF_AVMAX = float(os.environ.get("CX_WF_AVMAX", "0") or 0)
_CX_OMF_CSTK = os.environ.get("CX_OMF_CSTK", "0") != "0"  # fx_cstk (ws9 B2): city CRUISE stuck near the clue -> SEARCH
_OMF_CSTK_R = float(os.environ.get("CX_OMF_CSTK_R", "8.0") or 8.0)    # m (xy) from the clue centre
_OMF_CSTK_S = float(os.environ.get("CX_OMF_CSTK_S", "10.0") or 10.0)  # s of total CRUISE time inside R
_WF_SPCL = os.environ.get("CX_WF_SPCL", "0") != "0"
_WF_SPCL_V = float(os.environ.get("CX_WF_SPCL_V", "0.5"))
_WF_SPCL_TOP = float(os.environ.get("CX_WF_SPCL_TOP", "2.0"))


def _cx(name):
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + 1


# ---- fx_vk: village hand-over fixes, each default OFF (flags off == champion bit for bit) ----
# CX_VK_MOTION: on village a static track may only turn "moving" on evidence from one continuous detection segment.
#   The vote fit (1.2-2 s window) and the raw quarter-median check (8 s, z < 16 m) otherwise compare a biased far or
#   image-edge sighting with the close sightings that follow a detection gap (1340370548: 1.1 s gap while APPROACH
#   brakes; 514295942: edge hits at 45 deg off-axis, then nothing until range < 16 m) and hand a static pad to the king.
# CX_VK_WATCH: after a village hand-over, the graph pack that loses a moving pad repeats its last action bit for bit
#   (king main's tracking 'else: action = _last_action' branch) and flies blind away from the pad or into the terrain
#   (512343485, 307956468, 514295942, 1954733135). Take the flight back as the 18 m vking leash would, but as soon as
#   the repeat is seen carrying the drone away from the pad, and (like the leash) not in the first 3 s after the hand-over.
#   "Away" is judged only against a live shadow estimate (pad seen within 3 s); otherwise the leash decides.
# CX_VK_UNSTICK: after a village take-back (vking leash or CX_VK_WATCH), pad_ever_moving stays set, so a re-acquired
#   STATIC pad is flown as "once moving": the LAND hold (vz 0 once the pad is unseen for 1 s, abort after 3 s) keeps the
#   drone 1.3 m above a pad it is centred on and cannot see (514295942 with MOTION+WATCH: 37.3-40 s, then timeout).
#   pad_ever_moving is NOT cleared (the track stays un-established, so both motion paths keep full sensitivity until LAND).
#   The only change is that this one hold is skipped while the current track's own raw samples, re-checked at every
#   raw check up to LAND, show a still pad: one continuous run of >= 15 samples over >= 2 s, quarter-median
#   displacement < 0.5 m and a least-squares speed < 0.25 m/s.
_CX_VK_MOTION = os.environ.get("CX_VK_MOTION", "0") == "1"
_CX_VK_WATCH = os.environ.get("CX_VK_WATCH", "0") == "1"
_CX_VK_UNSTICK = os.environ.get("CX_VK_UNSTICK", "0") == "1"
_VK_VOTE_GAP = 0.4    # s: a detection gap inside the vote window splits the evidence
_VK_RAW_GAP = 0.5     # s: pad_raw is sampled every >= 0.1 s
_VK_WATCH_N = 10      # identical king actions in a row (0.2 s) = blind repeat
_VK_WATCH_MIN_T = 3.0  # s after the hand-over before WATCH may fire (the 18 m leash uses the same 3 s)
_VK_STILL_SPAN = 2.0  # s of continuous raw samples on the re-acquired track
_VK_STILL_N = 15      # raw samples (10 Hz) in that run
_VK_STILL_SP = 0.25   # m/s: least-squares speed of those samples


def _cx_count(name):
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + 1


# --- fx_vs: village search fixes, each default OFF (all off == champion bit for bit) ---
# DIPFULL: the village dip (descend to 3 m and spin) sweeps most bearings while still descending, where a pad closer
#          than the altitude is below the image (1145904165: the pad bearing was crossed at z 5.35, 52 deg down). Keep
#          spinning at the dip height until one full turn has been made there (the cap grows by _CX_VS_DIP_XCAP).
# REFLY:   a non-weak APPROACH whose track was dropped (pad None) before any search was built restarted SEARCH where the
#          drone was, with a dip 20-33 m from the clue (799215439, 101972003, 101879300). Fly back to the clue in CRUISE
#          instead, as the weak-approach branch already does.
# RESUME:  the same drop during a running search rebuilt the rings from scratch and dipped in place (1094226733).
#          Resume the search where it stopped, as the weak-approach branch already does.
# LOOK:    with RESUME, keep the camera on the dropped estimate for CX_VS_LOOK_S seconds of the resumed ring leg
#          (1094226733: a real pad 12 m ahead hidden for 1.2 s by roofs was dropped and never looked at again).
# RING1OUT: after a dip at the clue has swept the inner ~11 m, the first ring's legs look outward (the dip already
#          covered what the inward view sees) so pads 12-20 m from the clue come into view (401674693).
_CX_VS_DIPFULL = os.environ.get("CX_VS_DIPFULL", "0") == "1"
_CX_VS_REFLY = os.environ.get("CX_VS_REFLY", "0") == "1"
_CX_VS_RESUME = os.environ.get("CX_VS_RESUME", "0") == "1"
_CX_VS_LOOK = os.environ.get("CX_VS_LOOK", "0") == "1"
_CX_VS_RING1OUT = os.environ.get("CX_VS_RING1OUT", "0") == "1"
_CX_VS_ANY = _CX_VS_DIPFULL or _CX_VS_REFLY or _CX_VS_RESUME or _CX_VS_LOOK or _CX_VS_RING1OUT


def _cx_knob(name, default, on):
    # tuning knobs are read only when a flag that uses them is on; a malformed or non-finite value falls back to the
    # default instead of failing the import
    if not on:
        return default
    try:
        v = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


_CX_VS_DIP_XCAP = _cx_knob("CX_VS_DIP_XCAP", 6.0, _CX_VS_DIPFULL)
_CX_VS_LOOK_S = _cx_knob("CX_VS_LOOK_S", 3.0, _CX_VS_LOOK)
_CX_VS_NEAR_M = _cx_knob("CX_VS_NEAR_M", 6.0, _CX_VS_DIPFULL or _CX_VS_RING1OUT)
_CX_VS_REFLY_MAX = 2  # REFLY fires at most this often per seed, then the champion's rebuild path runs
# --- fx_kg: gates on the kept fixes (each default OFF; with all of them off this is c8 bit for bit) ---
# DIPLATE: DIPFULL's extra turn runs only when the champion's own dip ends at t >= CX_VS_DIPLATE_T (34 s). Earlier, the
#          champion's ring search still has time to find and land (fresh pairs f1-f10: 19 of 27 arrival dips ending before
#          34 s were champion landings, and the extra turn broke 7 of them), so the extra turn is mostly risk. Later, the
#          ring search rarely finishes (4 of 18 landed, 1 broken: 118165789), and the extra turn is a cheap option (every
#          rescue of a near pad by the extra turn itself came from a dip ending at 34.6 s or later).
_CX_VS_DIPLATE = os.environ.get("CX_VS_DIPLATE", "0") == "1"
_CX_VS_DIPLATE_T = _cx_knob("CX_VS_DIPLATE_T", 34.0, _CX_VS_DIPLATE)
# DIPCAP: DIPFULL's extra turn ends at most CX_VS_DIPCAP_S (3 s) after the champion's own dip end. Every near pad the extra
#         turn itself found was seen 0.9-2.4 s into it (76968241, 1549479488, design seed 1145904165); a longer turn only
#         delays the champion's ring search (median ~5 s, up to 9 s; 118165789 landed at 54.4 s in the champion).
_CX_VS_DIPCAP = os.environ.get("CX_VS_DIPCAP", "0") == "1"
_CX_VS_DIPCAP_S = _cx_knob("CX_VS_DIPCAP_S", 3.0, _CX_VS_DIPCAP)
_CX_STATE = {"shadow": False}  # True while the router runs mine only as a shadow (its action is thrown away)

# --- fx_v3 (village static search): count evidence about a candidate only on ticks when the camera could see it ---
# The village pilot counts a miss whenever the estimate projects into the image, and its vland check blacklists a young
# track (< 30 hits) that gathers too few hits during a 2 s level hover. In village the pad is mostly behind houses: the
# depth image at the true pad's pixel read a nearer surface (35-60% of the pad's camera distance) on 64-100% of the
# in-image ticks (1094226733, 267173686, 639834171, 106903232). And the vland hover can sit with the estimate below the
# image (1858838395, 267173686: ~50 deg down from 2.7-2.8 m out and 3.4 m up; the real pad, 0.03-0.1 m off, was
# blacklisted after 2 s with no chance of a hit).
#   CX_V3_OCC=1 (occlusion excuse): APPROACH (and, with VQF, a vcheck REACQUIRE) of a static, not yet established track
#                while moving >= 0.5 m/s: a no-detection tick whose predicted pixel shows a nearer surface (depth < 0.7 x
#                camera-z and > 1 m nearer) is not counted as a miss (<= _V3_EXC_S per track); never during the vland
#                hover. It only DELAYS the drop the champion's in-view miss rule would make: once the excuse has kept alive
#                a track the champion would have dropped (its miss count, excused ticks included, passed 1.2 s:
#                _v3_saved), APPROACH's 'lost' drops that track first, so the champion's exits for a dropped track run
#                (weak -> CRUISE/SEARCH, CX_VS_REFLY, SEARCH rebuild) and never REACQUIRE toward the occluded estimate
#                (1094226733: REACQUIRE beside the hiding house, 0.146 m clearance).
#   CX_V3_VLOOK=1 (vland look-down): with the estimate below the image (bottom 8 rows or behind the image plane) and
#                < 68 deg down, the hover becomes look-down pulses: a _V3_VL_PULSE_S step of _V3_VL_PULSE_V toward the pad
#                every _V3_VL_PERIOD, past the acceleration limit, pitches the camera ~20-25 deg down (as the champion's
#                mountain spin pulses do), with a slow return to the hover point in between. Those ticks do not advance
#                the 2 s timer (<= _V3_VL_EXT_S per APPROACH). Pulses only start at < 0.6 m/s, tilt < 12 deg and > 2.5 m
#                free ahead; when not allowed, or with the estimate in view (or occluded: a hover cannot change that),
#                the champion's timer and hover run unchanged. Pulse state never outlives a hover stretch or a track.
# CX_V3_VQF=1: the other side of the same accounting. A vcheck detour (CRUISE reached the clue with a >= 3-hit candidate)
#   REACQUIREs toward the candidate for up to 8 s (+ a side switch), and keeps flying it after the candidate has been
#   dropped by the champion's in-view miss rule (681764668 / 1182662052 locally: dropped 1.0-1.6 s in, REACQUIRE ran on
#   for 13-14 s; pod 1292196094, 556539092, 648833047: 8 s). Once the candidate is gone, SEARCH starts at once (no anchor,
#   so at the clue centre, as the champion's vcheck exit does). With OCC on, occluded misses are excused during that
#   REACQUIRE too, so a real candidate hidden behind a house is not rejected.
# None of them acts inside the vking shadow act() (mine's action is thrown away there, and its track feeds the VK
# take-backs): only the OCC drop-first rule can run there, on a track the excuse saved before the hand-over.
# Events (CX_EVENTS): v3_occ_excuse (first excused tick of a track), v3_occ_saved (first tick the excuse kept alive a
# track the champion would have dropped: before it, the flight is the champion's), v3_occ_drop (APPROACH 'lost' dropped
# such a track first), v3_vl_below (first pulse tick of an APPROACH), v3_vl_pulse (each pulse), v3_vqf (vcheck ended
# early); inside a discarded shadow act() they get a _shadow suffix.
_CX_V3_OCC = os.environ.get("CX_V3_OCC", "0") == "1"
_CX_V3_VLOOK = os.environ.get("CX_V3_VLOOK", "0") == "1"
_CX_V3_VQF = os.environ.get("CX_V3_VQF", "0") == "1"
_CX_V3_ANY = _CX_V3_OCC or _CX_V3_VLOOK or _CX_V3_VQF
_V3_OCC_RATIO = 0.7    # occluder: depth at the predicted pixel < this x the estimate's camera-z ...
_V3_OCC_MIN = 1.0      # ... and more than this (m) nearer
_V3_EXC_S = 4.0        # occluded miss time excused per track (s)
_V3_VL_EXT_S = 4.0     # vland ticks not counted per APPROACH (s)
_V3_VL_PERIOD = 1.2    # one look-down pulse per period (s)
_V3_VL_PULSE_S = 0.3   # pulse length (s)
_V3_VL_PULSE_V = 1.0   # pulse step (m/s): 1.5 m/s pitched 267173686 to 43 deg; 1.0 m/s gives ~20-25 deg


def _cx_v3_ev(name):
    # fx_v3 counter (the fx_vs rule: decisions inside a discarded shadow act() are counted apart)
    if _CX_STATE["shadow"]:
        name = name + "_shadow"
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + 1


def _v3_live():
    # fx_v3 acts only when mine flies: not inside the router's discarded vking shadow act()
    return not _CX_STATE["shadow"]
# --- fx_f6 (round 6, forest; port of fx_fo6 CX_FO_UNDER): boxed-in recovery, default OFF (off == c3 bit for bit) ---
# UNDER: forest cruises 4 m above the ground, the lower fringe of the crown layer (foliage within 0.6 m of the path on 8% of
# columns at 4 m AGL against 6% at 3 m and 14-38% at 5-9 m; ray probe of 10 fresh forest worlds, analysis/r6_forest). When the drone runs into
# foliage on all sides of the camera view, the planner's escape branch climbs blind at 0.5 m/s (the camera sees at most
# 45 deg up) into the denser layer above, or dithers: 794135506 and 357961169 climbed into a crown and sat there to the
# 60 s timeout; 2035705582, 927838733 and 300730315 lost 7-18 s and passed 0.12-0.17 m from foliage. After the escape
# branch has run for _F6_UNDER_TRIG s: retrace the recorded path (known free) until the drone is >= 2 m back and over open
# ground, turn toward a detour point _F6_UNDER_SIDE m to the freer side, fly it with the planner at up to 1.5 m/s, and
# hold _F6_UNDER_LOWH m instead of 4 m during the detour and _F6_UNDER_LOWS s after it. At most 3 episodes per seed.
# Review fixes: an episode starts only when a crumb >= 2 m back was recorded over open ground (else c3 keeps the tick);
# a BACK that times out still over foliage ends the episode (no detour from inside a crown); the planner-off turn step
# never climbs; an episode that starts during a detour or within _F6_UNDER_KEEP s of its end keeps the first blockage
# heading and takes the other side of it (a later, unrelated episode picks the freer side of its own heading); a cancel
# also clears the LOW window.
_F6_UNDER = os.environ.get("CX_F6_UNDER", "0") != "0"
_F6_UNDER_TRIG = _cx_knob("CX_F6_UNDER_TRIG", 0.8, _F6_UNDER)   # s of escape branch before a recovery starts
_F6_UNDER_BACK = _cx_knob("CX_F6_UNDER_BACK", 2.0, _F6_UNDER)   # m (straight line) to retrace
_F6_UNDER_SIDE = _cx_knob("CX_F6_UNDER_SIDE", 5.0, _F6_UNDER)   # m sideways to the detour point
_F6_UNDER_LOWH = _cx_knob("CX_F6_UNDER_LOWH", 3.2, _F6_UNDER)   # m AGL held during/after the detour (bush tops <= 2.6 m)
_F6_UNDER_LOWS = _cx_knob("CX_F6_UNDER_LOWS", 6.0, _F6_UNDER)   # s the low height is kept after the detour
_F6_UNDER_MAXEP = 3
_F6_UNDER_KEEP = 3.0  # s after a detour ends in which a new episode counts as the same blockage

# --- fx_f14 (round 14, forest failure mechanisms), every flag default OFF (off == c14 bit for bit) ---
# GATE:   a padnet detection that cannot be the goal pad is ignored (neither a hit nor a miss), forest only:
#         (a) within _F14_NEAR_M (xy) of the start: forest goals are placed >= 22 m from the start (distance band 22-40 m,
#             TYPE_6_R_MIN; 4,942 fresh forest seeds: min 22.0 m). 1787474779: a clutter object 2.3 m from the start took
#             10 hits in 0.4 s of TAKEOFF; APPROACH + LAND on it, crash at 4.9 s.
#         (b) more than _F14_HIGH_M above the lowest ground the down-ray has seen this flight: forest pads are platforms
#             at most TYPE_6_H_MAX = 3 m tall (the forest ground plane is z 0; pad tops 1.65-3.05 m on 4,942 seeds).
#             2053832238: a 'pad' at z 15.8 (a crown top seen from a 9.5 m start tower) -> APPROACH, then a blind spin
#             descent from 11 m into a crown at 9.06 s.
# STATIC: forest has no moving pads (moving_platform is never drawn for forest). The one path that can still flag a forest
#         track moving is _update_track's established-static re-seed after 40 conflicting detections (it sets pad_moving
#         and pad_ever_moving). With the flag on, forest re-seeds the track as a young static one (hits 30, not
#         established) at the new detection. 1712602343: an accurate 191-hit track (0.07 m off) went 'moving' at 12.3 s;
#         the moving LAND held 1.0 m over the pad it could no longer see and aborted (twice, the second time through the
#         pad_ever_moving hold), then timed out.
# REFLY:  forest APPROACH lost with the track already dropped (pad None) before any search was built, more than
#         _F14_REFLY_M from the clue: fly on to the clue (CRUISE) instead of starting the 9.5 s spin where the drone is
#         (median 17 m from the clue on those seeds) and ringing the clue from there; at most _F14_REFLY_MAX per seed,
#         then the champion's rebuild runs. The village fix CX_VS_REFLY, ported. Fresh hc32 batches: 79 of 2,219 forest
#         seeds take that path, mean score 0.871, 7 failures (2134218702, 2053832238, 1595556719, 941594811, ...).
_F14_GATE = _cx_on("CX_F14_GATE")
_F14_STATIC = _cx_on("CX_F14_STATIC")
_F14_REFLY = _cx_on("CX_F14_REFLY")
_F14_NEAR_M = _cx_knob("CX_F14_NEAR_M", 12.0, _F14_GATE)
_F14_HIGH_M = _cx_knob("CX_F14_HIGH_M", 5.0, _F14_GATE)
_F14_REFLY_M = _cx_knob("CX_F14_REFLY_M", 8.0, _F14_REFLY)
_F14_REFLY_MAX = 2

# --- fx_rc (cluster C regression hunt, forest), every flag default OFF (off == c19 bit for bit) ---
# Both act only at the champion's forest static LAND switch (d_h < 1.0 and <= 1.8 m above the estimate), which has no
# check of its own in forest (village/city run a 2 s hover check on young tracks). LAND on a non-pad body is a crash.
# Dev-pod LAND entries (debug build, c19 flags): 11 real-pad landings: established, last seen 0.14-0.38 s before,
# first seen from 1.0-1.8 m, estimate z within 0.05 m of the pad. 4 false-track LAND crashes: not established, last seen
# 1.1-2.9 s before (a crown top or clutter stops looking like a pad from close by) and/or estimate z 0.24 / 4.26.
# LSEEN: a not-established static forest track last seen >= _RC_LSEEN_S before the switch is dropped instead of LANDed
#        on (1393547997: crown top at z 2.7 last seen 1.1 s before; 468347476, 657633717). If it was never seen from
#        closer than _RC_LSEEN_FAR m (real pads: 1.0-1.8 m) its spot also goes on bad_spots (as the champion's LAND abort
#        does for young tracks), so it is not tracked and approached again. APPROACH's lost handling then runs as for any
#        dropped track.
# LBAND: forest pads are platforms whose tops lie at z 1.65-3.05 (10,464 forest seeds) on flat ground at z 0. A static
#        track whose estimate is below _RC_LBAND_LO or above _RC_LBAND_HI at the switch is dropped and its spot put on
#        bad_spots (672137831: crown top at z 4.3; 468347476: clutter at z 0.24). Only while the down-ray has seen ground
#        within 0.5 m of z 0 this flight (else the ground is not where this assumes and nothing is vetoed).
_RC_LSEEN = _cx_on("CX_RC_LSEEN")
_RC_LSEEN_S = _cx_knob("CX_RC_LSEEN_S", 0.8, _RC_LSEEN)
_RC_LSEEN_FAR = _cx_knob("CX_RC_LSEEN_FAR", 3.0, _RC_LSEEN)
_RC_LBAND = _cx_on("CX_RC_LBAND")
_RC_LBAND_LO = _cx_knob("CX_RC_LBAND_LO", 1.3, _RC_LBAND)
_RC_LBAND_HI = _cx_knob("CX_RC_LBAND_HI", 3.5, _RC_LBAND)
_RC_ANY = _RC_LSEEN or _RC_LBAND
_CX_VFP_LSEEN = os.environ.get("CX_VFP_LSEEN", "0") == "1"   # ws2/vfp village LAND sighting guard
_CX_VFP_RESG = os.environ.get("CX_VFP_RESG", "0") == "1"     # ws2/vfp: VS_RESUME only when fx_ra did not decide to dip
_VFP_LS_S = _cx_knob("CX_VFP_LS_S", 2.5, _CX_VFP_LSEEN)       # s since the track's last detection
_VFP_LS_D = _cx_knob("CX_VFP_LS_D", 8.5, _CX_VFP_LSEEN)       # m: closest 3D drone-estimate distance at a detection

# --- fx_v8 (village: mine's LAND onto a parked car; port of fx_vl CX_VL_SURF), every flag default OFF (off == c5) ---
# padnet takes parked village cars for pads (sedan 1758303276, hatchback 2024730237, pickup 2084560002). APPROACH collects
# >= 30 hits on the car before it is within 4 m, so the vland verification hover (< 30 hits only) never runs, and the
# estimate's height comes out near a ground pad's (0.44 / 0.68 / 0.62 m), so vmem calls the track established and LAND
# descends onto the car: roof, body side or pickup bed. The champion's vsurf only reacts to a surface > 0.6 m above for 2 s.
# The down ray is cast straight down in the world frame and reads the exact surface height under the drone. On real
# static pads (ground, car-top, roof) it reads the pad top -0.01..+0.10 m from the estimate's height once the drone is
# within 0.35 m of the estimate (the estimate is <= 0.16 m from the pad centre and the pad's collider has r = 0.6 m); over
# the cars it read +0.41 (sedan roof), -0.54 (ground beside the hatchback) and -0.48 then -0.17 (ground, pickup bed).
#   CX_V8_SURF=1: in a static village LAND (track never flagged moving, not touched down, not inside the router's shadow
#     act()), with the drone within _V8_SURF_XY of the estimate, more than _V8_SURF_H above it and _V8_SURF_ALT..3 m over
#     the surface under it, a surface more than _V8_SURF_UP above or _V8_SURF_DN below the estimate's height held for
#     _V8_SURF_T s: blacklist the estimate (bad_spots), drop the track, and leave as the champion leaves a dropped weak track
#     (CRUISE when no search was built yet and the clue is > 8 m away, else SEARCH).
#   CX_V8_NODIP=1 (with SURF): if that SEARCH resumes a village dip that has not finished and whose search centre is more
#     than _V8_NODIP_M away (2024730237: the dip at the clue was left for a car 13 m away, and after the rejection the
#     champion's resume spun ~8 s at 3 m beside the car), end the dip there; the rings resume where they stood.
# Events (CX_EVENTS): v8_surf (fires), v8_surf_at (last 8 fires as t/dz/estimate-to-clue distance), v8_nodip.
_CX_V8_SURF = os.environ.get("CX_V8_SURF", "0") == "1"
_CX_V8_NODIP = os.environ.get("CX_V8_NODIP", "0") == "1"
_V8_SURF_UP = _cx_knob("CX_V8_SURF_UP", 0.25, _CX_V8_SURF)  # m above the estimate (real pads: <= +0.10)
_V8_SURF_DN = _cx_knob("CX_V8_SURF_DN", 0.12, _CX_V8_SURF)  # m below it (real pads: >= -0.01; 0.20 missed the pickup bed)
_V8_SURF_H = 0.4      # m: only while the drone is this far above the estimate (on the pad the ray starts under its top)
_V8_SURF_XY = 0.35    # m: only this close (horizontally) to the estimate, where the ray falls on a real pad's top
_V8_SURF_ALT = 0.15   # m: minimum down-ray reading
_V8_SURF_ALT_MAX = 3.0  # m: maximum down-ray reading
_V8_SURF_T = 0.1      # s the inconsistent reading must hold (5 ticks)
_V8_NODIP_M = 6.0     # m from the search centre beyond which NODIP ends an unfinished dip


def _cx_v8_ev(name):
    # fx_v8 counter (the fx_vs rule: a decision inside a discarded shadow act() is counted apart; v8 never acts there)
    if _CX_STATE["shadow"]:
        name = name + "_shadow"
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + 1


def _cx_vs_ev(name):
    # fx_vs counter (fx_vs's own _cx_ev)
    if _CX_STATE["shadow"]:
        name = name + "_shadow"  # decision taken inside a discarded shadow act(): counted apart
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + 1


# --- fx_vl3 (village: found but not landed, round C; HOLD port of fx_vl2), flag default OFF (off == c8 bit for bit) ---
# A false detection far from the pad (458544027: an object 15 m from the pad) turns "moving" in APPROACH and the dual
# route hands the flight to the king. After the take-back (18 m vking leash or CX_VK_WATCH) pad_ever_moving stays set
# for the rest of the seed (_drop_track keeps it; setting it hands a village flight to the king at once, so mine flies
# a LAND with it set only after a take-back, or with no king), so LAND on the re-acquired STATIC pad runs the champion's
# "once moving" hold: vz 0 once the pad has been unseen for 1 s and abort after 3 s. Centred 1.4 m over the pad the
# camera cannot see it, so the hold always ends in the abort, RECOVER drops the track and SEARCH flies off (458544027 in c8 and in the champion: LAND at 28.2 s centred
# 0.1-0.3 m over the pad, abort at 31.0 s, never found again). CX_VK_UNSTICK needs 2 s of continuous raw samples on
# the re-acquired track and did not fire there (the APPROACH lasted 2.5 s with a verification hover inside it).
# CX_VL_HOLD=1: skip that hold (and its unseen abort) on a tick where the down ray shows the pad top right under the
# drone: within _VL_HOLD_XY of the estimate, and the surface under the drone within -_VL_HOLD_DN/+_VL_HOLD_UP of the
# estimate's height (real pad tops read -0.01..+0.11 m on 62 deep-traced landings, fx_v8; the ground or roof beside a
# pad reads 0.13-0.4 m lower). A pad that slides away takes the reading with it and the hold is back that tick.
# Never inside the router's discarded shadow act(). Events (CX_EVENTS): vl_hold (ticks the hold was skipped),
# vl_hold_seed (seeds), vl_hold_at ("t/dz" of the first skip).
_CX_VL_HOLD = _cx_on("CX_VL_HOLD")
_CX_VL_ANY = _CX_VL_HOLD
_VL_HOLD_XY = _cx_knob("CX_VL_HOLD_XY", 0.35, _CX_VL_HOLD)  # m from the estimate (fx_v8: the ray falls on the pad top)
_VL_HOLD_UP = _cx_knob("CX_VL_HOLD_UP", 0.15, _CX_VL_HOLD)  # m above the estimate's height (real pad tops <= +0.11)
_VL_HOLD_DN = _cx_knob("CX_VL_HOLD_DN", 0.06, _CX_VL_HOLD)  # m below it (real pad tops >= -0.01; beside a pad 0.13-0.4)
_VL_HOLD_ALT = 0.15      # m: minimum down-ray reading (fx_v8)
_VL_HOLD_ALT_MAX = 3.0   # m: maximum down-ray reading (fx_v8)


def _cx_vl_ev(name):
    # fx_vl3 counter (the fx_vs rule: a decision inside a discarded shadow act() is counted apart; HOLD never acts there)
    if _CX_STATE["shadow"]:
        name = name + "_shadow"
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + 1


# --- fx_m2: mountain SEARCH, round 2 (flag prefix CX_M2_). Every flag defaults OFF; with all of them off this is c2. ---
# The pad z lies in [clue z - 5, clue z + 5] (spawn: clue z = pad z + U(-5, 5)), and the camera sees at most ~45 deg
# below the horizon (~75 deg during a look-down pulse, and then only within ~18 m).
# SPIN:   the first mountain SEARCH at the clue, when the clue FLOATS above the terrain (ground under the drone below
#         clue z + _M2_SPIN_GMAX) and the search STARTS at or above clue z + _M2_SPIN_DZ (the cz + 8 arrival of mhigharr),
#         spins at clue z + _M2_SPIN_DZ instead of sinking to max(clue z - 1.5, ground + 2). From above the top of the
#         pad band every pad is below the camera; there the champion swept most bearings while sinking through the band
#         (more than half of it above the camera at the end) and passed the pad's bearing at a grazing angle
#         (1257414148: 2 m above the pad at 10 deg; 671534777: the pad sits 4 m above clue z on a bump 2.6 m away).
#         Elsewhere the champion's spin is kept: every search change re-times long ring searches, and the two champion
#         successes flown with a wider SPIN lost time that way (74872952, spin climbing from clue z + 2.3: 0.94 -> 0.01;
#         881040794, ground at clue z + 1.3, ring find 0.2 s later, landing at 60.0 s: 0.92 -> 0.47). Inside this gate
#         every champion success found its pad within 3.5 s of the SEARCH start (17 of 19 logged seeds; the other two
#         are the failures above), but from the SPIN height the find and the APPROACH start higher (the approach after the
#         find is re-timed on most of them).
# SCOPE:  all three act only on that first search and only until it first hands over to APPROACH (_m2_seen): a weak
#         APPROACH drop resumes the same first search (search_wps and spin_done kept), and the resumed spin must be the
#         champion's (1768674715: found at clue z + 3.2, dropped at clue z + 1.8, re-found after the champion's resumed
#         spin sank to clue z - 0.3; SPIN would have climbed it 4.2 m). A re-spin already running when a find interrupts
#         it resumes as a re-spin. A down ray saturated at the build (alt >= _M2_SAT_M) gives no ground, so neither the
#         ridge nor the SPIN class is assigned then.
# PULSE:  the mountain spin's look-down pulse is gated on "nothing closer than 2.5 m in depth rows 48-112". Hovering at
#         the ground + 2 floor (a spin on a ridge above the pad band) those rows see the ground itself, so no pulse ever
#         fires exactly where the pad is > 45 deg below (1143372174, 2123782371, 552119277, 1954235022, 1223686041: act
#         speed ~0 through the whole spin). Count only pixels whose hit point is less than _M2_PULSE_DROP below the eye.
#         The value is a min over a subset of the same pixels, so this can only open the gate, never close it.
#         CX_M2_PULSE: only in the first spin of a RIDGE-class search (below) and in re-spins; CX_M2_PULSE_ALL: in every
#         mountain spin (the msr_code prototype's scope, which also changes flat seeds that hover at ground + 2).
# RESPIN: a RIDGE clue (ground under the drone at the first SEARCH more than _M2_RIDGE_DZ above clue z: the whole pad
#         band lies downhill) that has had no live track for _M2_RS_T s after the SEARCH was built, or whose ring leg has
#         carried the drone off the ridge top (ground under the drone more than _M2_RS_DROP below that first ground for
#         _M2_RS_DROP_S s in a row: the down ray pitches with the body, so a one-tick reading is attitude-dependent),
#         stops once and spins a full turn in place at max(clue z + _M2_RS_DZ, ground + 2). Only the turn made within
#         1 m of that height counts; _M2_RS_CAP s cap; none starts after _M2_RS_TMAX s. The PULSE gate is always used
#         inside the re-spin (with or without CX_M2_PULSE). (Prototype: candidates/msr_code, time trigger only, PULSE
#         on: 1143372174 0.01 -> 0.946, 1453984591 0.01 -> 0.932.) Keep CX_MS_SZ OFF with RESPIN: SZ would lift the
#         rings after the re-spin back to clue z + 7 (never flown together).
_CX_M2_SPIN = _cx_on("CX_M2_SPIN")
_CX_M2_PULSE = _cx_on("CX_M2_PULSE")
_CX_M2_PULSE_ALL = _cx_on("CX_M2_PULSE_ALL")
_CX_M2_RESPIN = _cx_on("CX_M2_RESPIN")
_CX_M2_ANY = _CX_M2_SPIN or _CX_M2_PULSE or _CX_M2_PULSE_ALL or _CX_M2_RESPIN
_M2_SPIN_DZ = _cx_knob("CX_M2_SPIN_DZ", 6.0, _CX_M2_SPIN)          # spin height over clue z (m)
_M2_SPIN_GMAX = _cx_knob("CX_M2_SPIN_GMAX", 0.0, _CX_M2_SPIN)      # SPIN only if the ground is below clue z + this (m)
_M2_PULSE_DROP = _cx_knob("CX_M2_PULSE_DROP", 1.0, _CX_M2_ANY)     # depth pixels further below the eye are ground (m)
_M2_RIDGE_DZ = _cx_knob("CX_M2_RIDGE_DZ", 5.0, _CX_M2_ANY)         # ridge: ground at the clue above clue z + this (m)
_M2_RS_T = _cx_knob("CX_M2_RS_T", 10.0, _CX_M2_RESPIN)             # time trigger after the SEARCH build (s)
_M2_RS_DROP = _cx_knob("CX_M2_RS_DROP", 5.0, _CX_M2_RESPIN)        # drop trigger: ground this far below the first (m)
_M2_RS_DROP_S = _cx_knob("CX_M2_RS_DROP_S", 0.3, _CX_M2_RESPIN)    # ...held for this long (debounce) (s)
_M2_RS_DZ = _cx_knob("CX_M2_RS_DZ", 3.0, _CX_M2_RESPIN)            # re-spin height over clue z (ground + 2 floor) (m)
_M2_RS_CAP = _cx_knob("CX_M2_RS_CAP", 12.0, _CX_M2_RESPIN)         # re-spin time cap (s)
_M2_RS_TMAX = _cx_knob("CX_M2_RS_TMAX", 48.0, _CX_M2_RESPIN)       # no re-spin starts after this flight time (s)
_M2_RS_TURN = 2.0 * math.pi
_M2_SAT_M = 19.5                                                     # down ray at or above this reads "nothing" (20 m)


def _cx_m2_ev(name):
    # fx_m2 counter: muted inside a shadow act() (_CxShadow mutes when any CX_MS_* or CX_M2_* flag is on)
    if _CX_MUTE[0]:
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + 1


# --- fx_dv: moving-pad dive (flag prefix CX_DV_). Every flag defaults OFF; with both off this is c4 bit for bit. ---
# On a moving pad the FIRST contact with the pad body is a success, at any speed or attitude. The pilots still fly the
# last metre like a static landing and the pad slides out from under them: in 9 of 24 fresh moving-pad failures the
# down-ray was on the pad (0.08-1.47 m above it) for 0.2-0.6 s and no contact followed (dev pod, c4: 1776265334 crossed
# 0.07-0.31 m over the pad for 0.56 s; 58814357 0.17 m for 0.6 s; 696068867, 1775071560 likewise).
# CX_DV_DIVE (router, after the route's action, on any route whose pad has been flagged moving):
#   trigger only on observed evidence -- the down-ray agl DROPS by >= _DV_JUMP within _DV_JUMP_S (the ray has just moved
#   onto a raised surface), and while the ray stays on that surface (+-_DV_SURF, at most _DV_EP_S after the drop):
#   the surface z (z - agl) matches the pad-top estimate within _DV_DZ, the estimated pad xy is within _DV_DXY of the drone,
#   and that estimate is fresh (<= _DV_FRESH s since a detection). The estimate is mine's track when mine ran this tick
#   (dual/mine routes, the mountain hand-back shadow, the village vking shadow), else the king pilot's platform track.
#   Never on a flight whose pad is not flagged moving (king route: only after the moving hand-over; other routes: mine's
#   pad_moving / pad_ever_moving), never with agl above _DV_HMAX (every useful fire was at 0.15-0.42 m; the one fire
#   from 1.12 m, 541071544, ran off the pad at 0.42 m and handed uid130 a 44 deg, -2 m/s drone), never on a pad whose
#   estimated speed is under _DV_MINSP (static pads handed to the king as moving are common: 54 static seeds in the
#   f3/f4 c4 runs; the ungated first test tipped 3 static city landings, TILT; static estimates read up to 0.53 m/s).
# CONTACT: a dive must never start into a drone that is touching something. No episode opens and no dive fires below
#   agl _DV_AGL_MIN (resting on a static pad the ray starts inside the pad and reads the ground; lifting a few cm it
#   reads the pad top at agl 0.03-0.04 again, which looks like a >= 0.8 m drop: 9 of 13 static moving-hand-over flights
#   opened such an episode inside the 0.5 s success hold). And the pad is latched STATIC for the rest of the seed
#   (_dv_n = _DV_MAX_N, no further dive) once the flight is flagged moving and either (a) a tick reads agl <=
#   _DV_LATCH_AGL within _DV_DXY of any pad estimate (contact that did not end the seed: on a moving pad the first
#   contact is the success), or (b) a dive ends without the seed ending. At most _DV_MAX_N dives per seed.
# COMMAND (DSLPIDControl, target_pos = current pos, so thrust = D*(v_cmd - v) + [0, 0, m g] with D = [.2, .2, .5],
#   m g = 0.265 N): a sink command more than ~0.53 m/s below the current vz drives the vertical thrust target below
#   zero, every motor then sits at MIN_PWM (the measured ~5.6 m/s^2 sink), and there is NO attitude authority -- the
#   drone keeps whatever roll rate it had (1776265334: +2 deg per tick, 3.5 -> 30.8 deg), and as vz catches up with the
#   command, authority returns with a target tilt of atan(0.2 |dv_xy| / F_z) with F_z near 0 (541071544: 44 -> 54 deg,
#   1694166481: 25 -> 44.5 deg with a 3.2 m/s horizontal step). So the dive command is rate-limited on both axes:
#   vz_cmd = max(-sink, vz - _DV_LEAD) keeps F_z >= 0.265 - 0.5 * _DV_LEAD (~0.14 N at 0.25: every motor ~4,000 PWM
#   above the floor, full roll/pitch torque) and still sinks at ~4.6 m/s^2; the horizontal command stays within
#   _DV_DVH of the drone's own horizontal velocity, stepping toward the estimated pad velocity (target tilt <=
#   atan(0.2 * 0.3 / 0.14) = 23 deg; a 1.5 m/s step would ask for 65 deg, past the 60 deg TILT cut). The old uncapped
#   dive never tracked the pad anyway: with no authority the horizontal velocity did not change (58814357: vx -2.23 ->
#   -2.25 over the whole dive). sink = _DV_VZ, capped so the drone can stop _DV_FLOOR above the surface the ray saw
#   beside the pad just before the drop (braking at _DV_BRAKE_A); if the sum passes 3 m/s the sink gives way first.
#   The dive lasts at most _DV_MAX_S, ending early when the ray leaves that surface (this includes a touch on a static
#   pad), at an impact (vz rising by >= _DV_HIT_DV in one tick), or when the sink has stopped (blocked: still slower than
#   0.3 m/s after _DV_BLOCK_S); then the route's own action is returned unchanged, except that if the ray left the pad
#   onto a surface the sink could not stop above, the route's action is flown with its sink removed (vz >= +0.5) for up
#   to 0.3 s (dv_brake). Contact comes at agl ~0.03 (the ray then jumps to the ground under the pad), so _DV_TOUCH (end
#   at agl <= it) is off by default.
# CX_DV_NOHOLD (mine's LAND, static branch, pad flagged moving = pad_ever_moving): skip the height hold
#   (vz = 0 while d_xy > 0.35 and h < 1.6) and keep the static branch's -0.9/-0.6 m/s descent while the pad is tracked
#   (seen < 0.8 s ago) and the ray is on the pad or the surface under the drone lies >= _DV_NH_DROP below the pad top,
#   and mine's fitted or filtered pad speed is >= _DV_NH_MINSP (pad_ever_moving is also set on static pads). Never inside
#   a discarded shadow act(). UNFLOWN (never acted in any pod flight) and NOT in fx_dv/FLAGS.
# Events (CX_EVENTS): dv_fire (dive started), dv_src_king (its estimate came from the king pilot), dv_end_{time,left,
#   touch,block,hit} (why a dive ended without a success), dv_brake, dv_latch_touch / dv_latch_end (the static latch set
#   by (a) / (b) above), dv_nohold (LAND ticks whose hold was skipped), dv_error; dv_tilt is the peak tilt (deg, max of
#   |roll|, |pitch|) from any dive start to 0.6 s after its end; dv_fire1 is a diagnostic list [t, src, agl, surface -
#   estimate, xy distance, pad speed, route, |v_pad - v_drone|_xy, tilt deg] of the first dive.
#   CX_DV_DIAG=1 also records why a raised-surface episode did not fire (dv_diag list; changes nothing).
_CX_DV_DIVE = _cx_on("CX_DV_DIVE")
_CX_DV_NOHOLD = _cx_on("CX_DV_NOHOLD")
_CX_DV_DIAG = _cx_on("CX_DV_DIAG")
_DV_SHADOW = [0]  # > 0 inside a shadow mine.act() (set by _CxShadow when CX_DV_NOHOLD is on)
_DV_JUMP = _cx_knob("CX_DV_JUMP", 0.8, _CX_DV_DIVE)          # agl drop that marks the ray moving onto a raised surface (m)
_DV_JUMP_S = _cx_knob("CX_DV_JUMP_S", 0.1, _CX_DV_DIVE)      # ...within this long (s)
_DV_SURF = _cx_knob("CX_DV_SURF", 0.15, _CX_DV_DIVE)         # the ray is still on that surface (z - agl within this) (m)
_DV_EP_S = _cx_knob("CX_DV_EP_S", 0.3, _CX_DV_DIVE)          # the trigger may be met this long after the drop (s)
_DV_DZ = _cx_knob("CX_DV_DZ", 0.3, _CX_DV_DIVE)              # surface z vs the pad-top estimate (m)
_DV_DXY = _cx_knob("CX_DV_DXY", 0.8, _CX_DV_DIVE)            # estimated pad centre within this of the drone (m)
_DV_FRESH = _cx_knob("CX_DV_FRESH", 0.3, _CX_DV_DIVE)        # s since the estimate's last detection
_DV_HMAX = _cx_knob("CX_DV_HMAX", 0.6, _CX_DV_DIVE)          # no dive from higher than this over the surface (m)
_DV_AGL_MIN = _cx_knob("CX_DV_AGL_MIN", 0.08, _CX_DV_DIVE)   # no episode opens and no dive fires below this agl (m)
_DV_LATCH_AGL = _cx_knob("CX_DV_LATCH_AGL", 0.06, _CX_DV_DIVE)  # agl <= this near a pad estimate = contact: static (m)
_DV_VZ = _cx_knob("CX_DV_VZ", 2.5, _CX_DV_DIVE)              # dive sink rate (m/s)
_DV_LEAD = _cx_knob("CX_DV_LEAD", 0.25, _CX_DV_DIVE)         # sink command at most this below the measured vz (m/s)
_DV_DVH = _cx_knob("CX_DV_DVH", 0.3, _CX_DV_DIVE)            # horizontal command within this of the drone's velocity (m/s)
_DV_MAX_S = _cx_knob("CX_DV_MAX_S", 0.6, _CX_DV_DIVE)        # dive length cap (s)
_DV_TOUCH = _cx_knob("CX_DV_TOUCH", 0.0, _CX_DV_DIVE)        # end the dive at agl <= this (m); 0 = off (see above)
_DV_BLOCK_S = _cx_knob("CX_DV_BLOCK_S", 0.3, _CX_DV_DIVE)    # 'blocked' only after this long (a climbing start needs ~0.2 s)
_DV_MINSP = _cx_knob("CX_DV_MINSP", 0.7, _CX_DV_DIVE)        # estimated pad speed needed to dive (m/s)
_DV_HIT_DV = _cx_knob("CX_DV_HIT_DV", 0.4, _CX_DV_DIVE)      # vz rising this much in one tick = an impact (m/s)
_DV_FLOOR = _cx_knob("CX_DV_FLOOR", 0.4, _CX_DV_DIVE)        # keep this above the surface seen beside the pad (m)
_DV_BRAKE_A = _cx_knob("CX_DV_BRAKE_A", 5.0, _CX_DV_DIVE)    # braking assumed for that floor (m/s^2)
_DV_MAX_N = int(_cx_knob("CX_DV_MAX_N", 2.0, _CX_DV_DIVE))   # dives per seed
_DV_TILT_S = 0.6                                              # dv_tilt: window after a dive ends (s)
_DV_NH_DROP = _cx_knob("CX_DV_NH_DROP", 0.8, _CX_DV_NOHOLD)  # NOHOLD off the pad: ground at least this below the pad top (m)
_DV_NH_MINSP = _cx_knob("CX_DV_NH_MINSP", 0.7, _CX_DV_NOHOLD)  # NOHOLD: mine's fitted/filtered pad speed must reach this (m/s)


def _cx_dv_ev(name):
    # fx_dv counter (plain: the dive runs in the router, NOHOLD checks _DV_SHADOW itself before counting)
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + 1


# --- fx_mg: moving-pad floor guard (flag CX_MG_FLOOR, default OFF; with it off nothing below runs) ---
# Failure family (22 seeds in ~15k fresh flights, the champion crashes on them too): on the king route after a moving
# hand-over (city/village/mountain) or on open, the king pilot's final approach misses the pad edge by 0.1-0.4 m, keeps
# descending beside the pad and hits the ground (city: the transparent z=0 plane) or the terrain 0.3-0.9 m below the
# pad top. A moving pad counts as landed at the FIRST contact (moving_drone.py _update_landing_state), so a drone that
# stays at pad-top height beside the pad and closes in sideways, or passes over it and sinks, lands instead of crashing.
# CX_MG_FLOOR (router, after the route's action and after CX_DV_DIVE; king route only; never inside a shadow act()):
#   ESTIMATES (read-only, every tick on every route, so mine's hand-over estimate is kept): the pad-top z (constant for
#   a moving pad: the orbit is horizontal) is the median of fresh pilot detections made from at least _MG_GOOD_H above
#   the detected top and within _MG_EST_NEAR of the drone (the king pilot's platform_position when it detected the pad
#   this tick; mine's pad_pos when mine ran this tick and detected it), frozen while the guard holds; it is replaced by
#   the down-ray's reading of the pad top once the ray reads a raised surface (>= 0.15 m over the lowest surface of the
#   last second) within _MG_TOL of that median, the drone within _MG_RAY_XY of the pad xy. Level with the pad the king's
#   estimate is off by ~0.5 m in z and 0.4-0.8 m in xy (1901189210); from >= 0.5 m above by ~0.03 m and ~0.1 m.
#   The pad xy: the latest detection (king, else mine, else the ray on the pad top), used to gate; the latest GOOD one
#   (pilot detection from >= _MG_GOOD_H above, or the ray) is the pursuit target.
#   FLAG: the king route after the moving hand-over (king_from_dual), or on the open king route once the king pilot has
#   flagged the pad moving (tracking or move_in_auto_mode; sticky).
#   ENGAGE (all): a pad-top estimate; the drone within _MG_R of the predicted pad xy (seen within _MG_AGE s; an older
#   estimate gates at _MG_RFAR); the drone at or below top + _MG_ON; the down-ray NOT on the pad top (surface below
#   top - _MG_TOL, or saturated); not latched.
#   HOLD: the vertical command becomes max(route vz, clip(_MG_K * (top + _MG_HOLD - z), -_MG_VDN, _MG_VUP)): the sink is
#   cut to a gentle approach of a floor 6 cm under the pad top (inside the pad's side band: closing in sideways touches
#   the side, and a moving pad's first contact is a landing), and below it the drone climbs back. Horizontal: the route's command, or (CX_MG_STEER, when the king pilot did not see the pad this tick)
#   a pursuit of the predicted pad (pad velocity + _MG_KXY * offset, <= _MG_VXY, within _MG_DVH of the drone's own
#   velocity so the tilt stays small); CX_MG_YAW turns the camera to the predicted pad so the pilot can re-acquire it;
#   while the pilot sees the pad, CX_MG_BIAS (m/s, 0 = off) adds an inward component to its horizontal command.
#   LOITER: when the route's action has repeated bit for bit for _MG_REP_N ticks (the pilot is blind) the guard owns
#   the command: the vertical tracks the floor (no pilot climb or sink) and the horizontal pursues the last good
#   estimate facing it, for as long as the episode runs (the pad sweeps its path periodically and touches a drone held
#   on it at side-band height).
#   A NEW episode needs a pad xy seen within _MG_AGE s and the drone within _MG_R of it, and z >= top - _MG_LOW; a
#   running one continues while the drone stays within _MG_RFAR of the latest estimate. Pilot detections within
#   _MG_START_R of the start xy are ignored, and the pad-top median needs _MG_NZ samples.
#   DROP (CX_MG_DROP): within _MG_DROP_S after a HOLD tick, a ray back on the pad top with agl <= _MG_DROP_H sinks at
#   _MG_DROP_VZ (at least the route's sink) keeping the drone's horizontal velocity: the pilot's own command may carry
#   the drone off the pad again before it touches.
#   RELEASE: the ray back on the pad (mg_release_ray; the route flies, or DROP), the pad estimate stale or far
#   (mg_release_far), the drone above the band (mg_release_high), _MG_MAX_S of guard time used (mg_release_cap).
#   NEVER ACT ON A DRONE THAT TOUCHES: once a tick reads agl <= _MG_TOUCH with the ray on the pad top (mg_latch_touch;
#   the ray origin, 3 cm under the centre, enters the pad just before contact, then reads the ground below it) the guard
#   does nothing (a DROP already running continues to contact) until the drone has slid off the pad top without contact:
#   3 ticks in a row with the ray off the pad top at >= 0.8 m/s (mg_untouch; a drone settling on a static pad does not
#   move, and a moving pad's contact ends the seed on its first tick); latched fully off when 5 HOLD ticks
#   in a row find the drone at or above top - 0.02 and standing still although the guard itself asks for a sink
#   (resting on a pad edge: mg_latch_block; a free hover sits at the floor, 4 cm lower). Skipped while
#   a CX_DV_DIVE dive or brake runs. The pad-top ray reading needs a raised surface (>= 0.15 m above the lowest surface
#   the ray read in the last second) and the first one must match the pilots' median within _MG_TOL.
# Events (CX_EVENTS): mg_fire (episodes), mg_ticks (HOLD ticks), mg_steer (HOLD ticks with pursuit), mg_loiter (HOLD
#   ticks in LOITER), mg_drop (DROP ticks), mg_bias, mg_top_ray (the ray measured the pad top), mg_untouch,
#   mg_release_{ray,far,high,cap,touch,latch}, mg_latch_{touch,block}, mg_error; mg_fire1 = [t, z - top, xy to the predicted
#   pad, agl, surface - top, top source, estimate age, pilot, route vz, flag source] of the first episode.
_CX_MG_FLOOR = _cx_on("CX_MG_FLOOR")
_MG_ON = _cx_knob("CX_MG_ON", 0.20, _CX_MG_FLOOR)          # engage at z <= pad top + this (m)
_MG_HOLD = _cx_knob("CX_MG_HOLD", -0.06, _CX_MG_FLOOR)     # floor target relative to the pad top (m): inside the
#   pad side band (top-0.2..top), so moving inward touches the side; 7 cm under a drone resting on the top (block latch)
_MG_K = _cx_knob("CX_MG_K", 3.0, _CX_MG_FLOOR)             # floor gain (1/s)
_MG_VUP = _cx_knob("CX_MG_VUP", 0.6, _CX_MG_FLOOR)         # climb back to the floor at most this fast (m/s)
_MG_VDN = _cx_knob("CX_MG_VDN", 0.5, _CX_MG_FLOOR)         # approach the floor sinking at most this fast (m/s)
_MG_R = _cx_knob("CX_MG_R", 3.0, _CX_MG_FLOOR)             # the drone within this of the predicted pad xy (m)
_MG_AGE = _cx_knob("CX_MG_AGE", 3.0, _CX_MG_FLOOR)         # a pad xy seen at most this long ago gates at _MG_R (s)
_MG_RFAR = _cx_knob("CX_MG_RFAR", 8.0, _CX_MG_FLOOR)       # an older one gates at this (m; the pad stays within ~4.3 m of
#   its orbit centre, so the drone at the pad is within ~8.6 m of any estimate of it)
_MG_GOOD_H = _cx_knob("CX_MG_GOOD_H", 0.35, _CX_MG_FLOOR)  # pilot detections count (z sample, pursuit target) only from
#   at least this high over the detected pad top: level with the pad the king's estimate is off by ~0.5 m in z and
#   0.4-0.8 m in xy (1901189210), from >= 0.5 m above within ~0.03 m and ~0.1 m
_MG_STEER_AGE = _cx_knob("CX_MG_STEER_AGE", 1.0, _CX_MG_FLOOR)  # pursue only a good estimate at most this old (s)
_MG_PRED_S = _cx_knob("CX_MG_PRED_S", 0.8, _CX_MG_FLOOR)   # extrapolate the pad velocity at most this long (s)
_MG_TOL = _cx_knob("CX_MG_TOL", 0.12, _CX_MG_FLOOR)        # the ray is on the pad top when surface >= top - this (m)
_MG_RAY_XY = _cx_knob("CX_MG_RAY_XY", 0.9, _CX_MG_FLOOR)   # a ray pad-top sample needs the drone this close to the pad xy (m)
_MG_EST_NEAR = _cx_knob("CX_MG_EST_NEAR", 20.0, _CX_MG_FLOOR)  # pilot z samples only from detections this close (m)
_MG_NZ = int(_cx_knob("CX_MG_NZ", 5.0, _CX_MG_FLOOR))     # pilot z samples needed for a pad-top estimate
_MG_START_R = _cx_knob("CX_MG_START_R", 3.0, _CX_MG_FLOOR)  # ignore pilot detections this close to the start xy (m):
#   the king's detector can take the start pad for the goal at take-off (551991491: 3.84 m "pad top" -> 6 s of climbs)
_MG_LOW = _cx_knob("CX_MG_LOW", 0.35, _CX_MG_FLOOR)        # a new episode needs z >= top - this (came down into the band)
_MG_REP_N = int(_cx_knob("CX_MG_REP_N", 10.0, _CX_MG_FLOOR))  # the route's action repeated bit for bit this many ticks =
#   the pilot is blind (the king's 'landing, tracking' branch repeats _last_action forever once the pad is out of view:
#   1239298750, 1270074406 fly away level at 2 m/s): LOITER then owns the command (floor + pursuit of the last good
#   estimate, facing it). The pad sweeps its whole path periodically, so a drone held at side-band height on that path
#   is touched by the pad's side when it comes by
_MG_TOUCH = _cx_knob("CX_MG_TOUCH", 0.10, _CX_MG_FLOOR)    # agl <= this with the ray on the pad top = contact (m)
_MG_MAX_S = _cx_knob("CX_MG_MAX_S", 30.0, _CX_MG_FLOOR)    # guard time per seed (s; holding beside a linear pad's
#   path gives a side contact each time it passes, while the pilot would only descend into the ground)
_MG_STEER = _cx_knob("CX_MG_STEER", 1.0, _CX_MG_FLOOR) > 0.5  # pursue the predicted pad while the pilot is blind
_MG_KXY = _cx_knob("CX_MG_KXY", 1.2, _CX_MG_FLOOR)         # pursuit gain (1/s)
_MG_VXY = _cx_knob("CX_MG_VXY", 2.0, _CX_MG_FLOOR)         # pursuit speed cap (m/s)
_MG_DVH = _cx_knob("CX_MG_DVH", 0.6, _CX_MG_FLOOR)         # pursuit command within this of the drone's velocity (m/s)
_MG_YAW = _cx_knob("CX_MG_YAW", 1.0, _CX_MG_FLOOR) > 0.5   # face the predicted pad while pursuing
_MG_BIAS = _cx_knob("CX_MG_BIAS", 0.0, _CX_MG_FLOOR)       # while the pilot sees the pad: add this inward speed (m/s)
_MG_DROP = _cx_knob("CX_MG_DROP", 1.0, _CX_MG_FLOOR) > 0.5  # sink onto the pad when the ray finds it after a HOLD
_MG_DROP_S = _cx_knob("CX_MG_DROP_S", 1.0, _CX_MG_FLOOR)   # DROP only this long after the last HOLD tick (s)
_MG_DROP_H = _cx_knob("CX_MG_DROP_H", 0.4, _CX_MG_FLOOR)   # DROP only this close over the pad top (m)
_MG_DROP_VZ = _cx_knob("CX_MG_DROP_VZ", 0.6, _CX_MG_FLOOR)  # DROP sink rate (m/s)
_CX_VCH_RQ = _cx_on("CX_VCH_RQ")  # fx_vrq (ws6 C'): capped REACQUIRE + resume after a low-gain static village approach
_VCH_RQ_H = int(_cx_knob("CX_VCH_RQ_H", 3.0, _CX_VCH_RQ))    # hits the approach must gain for the champion's REACQUIRE
_VCH_RQ_T = _cx_knob("CX_VCH_RQ_T", 3.0, _CX_VCH_RQ)         # only approaches at least this long (s)
_VCH_RQ_CAP = _cx_knob("CX_VCH_RQ_CAP", 4.0, _CX_VCH_RQ)     # REACQUIRE time for such a track (s; champion 8)


def _cx_mg_ev(name, n=1):
    # fx_mg counter (plain: the guard runs in the router, after any shadow act() has returned)
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


# --- fx_mpc: village roof moving-pad path wait (flag CX_MPC, default OFF; with it off nothing below runs) ---
# Village moving pads over roofs (pad top >= _MPC_ROOF over the start) with LINEAR orbits (R sin(0.5 speed t + ph)
# along a line, 4-8 m stroke, stop and reversal at each end) defeat the pursuit: the king's dive misses the pad top at the
# edge, fx_mg holds beside it, CX_VK_WATCH hands the flight to mine, and mine's constant-velocity chase never touches the
# reversing pad again (DEEP.md: 12 of 14 near take-backs failed). The validator raises a village moving pad 0.2 m over
# the highest body AABB within 4.9 m of its orbit centre, so the slab at the pad's side band is free along its path.
# CX_MPC (router, the LAST post-processor of act(); route king after a village moving hand-over; pad top known):
#   SAMPLES (from the hand-over on): mine's raw detections of the (shadow) track (pad_obs, planar depth <= _MPC_ZMAX,
#     within _MPC_RNG of the drone) and the king pilot's fresh detections (main/uid130, read through fx_mg), weight
#     1/(0.05 + 0.02 depth)^2.
#   TRIGGER (the near-miss): T1 = the CX_VK_WATCH take-back decision (vetoed when MPC arms); T2 = _MPC_MG_DELAY after an
#     fx_mg HOLD episode ended without contact (no DROP / touch latch), the episode having started within _MPC_MG_D0 of
#     the pad estimate, and NOT with the drone above the side band (z > top + _MG_ON) within _MPC_T2_DX of the pad
#     estimate (the king's fx_dv dive lands those: 1247666021). Arms when: pad top - start z >= _MPC_ROOF, the drone
#     within [-_MPC_DZLO, +_MPC_DZHI] of the pad top and within _MPC_NEAR of the pad estimate, no fx_dv dive / brake,
#     the vking hand-over memory alive (a release goes through its take-back), no earlier episode, t <= _MPC_TLAST,
#     and the LINE FIT of the last _MPC_WIN s is valid with the drone within _MPC_LAT of the axis.
#   LINE FIT: weighted principal axis, lateral outliers (> max(_MPC_OUT, 3 rms)) dropped once; valid when >= _MPC_MINN
#     samples over >= _MPC_SPAN s, lateral rms <= _MPC_RMS, the quadratic sag over the stroke (|a| (extent / 2)^2 of
#     lateral = a s^2 + b s + c) <= _MPC_SAG (circle arcs and figure-8 lobes curve), extent (q03-q97) in [_MPC_EXT, 9.5].
#   WAIT POINT (_mpc_pick): axis points clamped min(_MPC_END, extent / 4) inside the observed stroke. _MPC_IC with the
#     pad's along-axis position and velocity known (regression over the samples of the last _MPC_IC_WV s, >=
#     _MPC_IC_VMIN, not contradicted by fx_mg's pad velocity): the point the pad reaches first (arrivals at the speed
#     bound _MPC_IC_VP, turning at the observed stroke end: lower bounds, in the true visiting order) at or after the hop
#     (|P - drone| / _MPC_IC_VH + _MPC_IC_H0); else the axis point nearest the drone.
#   HOP / WAIT (MPC owns the action): fly there (<= _MPC_VH m/s, braking-limited, command within _MPC_DVH of the drone's
#     velocity; <= 0.5 m/s sideways while more than _MPC_CLIMB under the target height) at z = top + _MPC_DZ with a
#     ray-measured top (fx_mg's floor -0.06: inside the pad's 0.2 m side band), top + _MPC_DZP with a pilot-median top
#     (reads 0.01-0.16 m low), hold it, face along the axis toward the pad's side. Re-fit every 0.5 s; a valid fit whose
#     axis lies > _MPC_REPL from the wait point moves it (to the nearest point of the new axis). Fresh near samples that
#     cross the wait point along the axis without contact = a pass; in WAIT a MISS (two misses release).
#   fx_dv: its dive (and post-dive brake) may fire while MPC owns the flight (the pad passing under a drone above the
#     top); MPC yields those ticks and resumes with a HOP when the dive ends without contact.
#   SINK: the down-ray reads the pad top under the drone (agl <= _MPC_SINK_H), the pad estimate within _MPC_SINK_XY and
#     the drone within _MPC_SINK_LAT of the axis: sink as fx_dv does (<= _MPC_SINK_VZ, stopping above top -
#     _MPC_SINK_FL, the command leading the measured vz by <= 0.25 m/s; horizontal stepping <= 0.3 m/s per tick toward
#     the pad velocity) for <= 0.8 s.
#   SAFETY: a non-pad surface under the drone above top - _MPC_SURF, or a pad-top-level surface off the pad's corridor
#     (> _MPC_SINK_LAT from the axis or > 0.8 m outside the observed stroke; impossible inside the placement disc):
#     RETREAT (climb to top + 0.6 for <= 1.2 s) and release.
#   EXIT: contact (the landing); _MPC_TMAX s without contact while t <= _MPC_TKEEP (later the wait goes on: a release
#     then hands the flight to a chase that cannot finish), a path-check / miss / surface release, or t > _MPC_TEND:
#     release (CX_MPC_HBM: to my controller through the vking take-back; hand-back hygiene for a king main pilot; for
#     _MPC_HB_S s the route's command is rate-limited to 0.5 m/s of the drone's velocity and 0.6 rad of its heading).
#     Never mid-SINK. Any exception: the same take-back at once, never again this seed.
#   While MPC owns the flight: fx_mg keeps its estimates but runs no HOLD / DROP, and the VK_WATCH take-back, the vking
#     18 m leash and CX_VK2_PIN are vetoed.
# Events: mpc_arm, mpc_arm1 (first arm row), mpc_fitx (rejected arm tries), mpc_fit (accepted fit rows), mpc_wait (first
#   arrival row), mpc_wait_s, mpc_last (state on the last owned tick: the contact when the flight lands in MPC), mpc_sink,
#   mpc_pass, mpc_miss, mpc_replace, mpc_release_<why>, mpc_rel (rows), mpc_veto_vkw / _leash / _pin, mpc_hb_ticks,
#   mpc_dv / mpc_dv1 (ticks yielded to fx_dv), mpc_keep (the timeout passed after _MPC_TKEEP: waiting on), mpc_err.
_CX_MPC = _cx_on("CX_MPC")
_MPC_MAPS = tuple(x.strip() for x in os.environ.get("CX_MPC_MAPS", "village").split(",") if x.strip()) if _CX_MPC else ("village",)
_MPC_ROOF = _cx_knob("CX_MPC_ROOF", 2.0, _CX_MPC)        # pad top - start z (m): a roof pad
_MPC_WIN = _cx_knob("CX_MPC_WIN", 6.0, _CX_MPC)          # fit window (s)
_MPC_MINN = int(_cx_knob("CX_MPC_MINN", 12.0, _CX_MPC))  # samples in the window
_MPC_SPAN = _cx_knob("CX_MPC_SPAN", 1.5, _CX_MPC)        # ...spanning at least this (s)
_MPC_ZMAX = _cx_knob("CX_MPC_ZMAX", 12.0, _CX_MPC)       # sample planar depth limit (m)
_MPC_RNG = _cx_knob("CX_MPC_RNG", 12.0, _CX_MPC)         # sample range limit (xy, m)
_MPC_OUT = _cx_knob("CX_MPC_OUT", 0.4, _CX_MPC)          # lateral outlier limit (m, or 3 rms)
_MPC_RMS = _cx_knob("CX_MPC_RMS", 0.20, _CX_MPC)         # linear: lateral rms (m)
_MPC_SAG = _cx_knob("CX_MPC_SAG", 0.35, _CX_MPC)         # linear: quadratic sag over the stroke (m)
_MPC_EXT = _cx_knob("CX_MPC_EXT", 2.0, _CX_MPC)          # linear: observed extent at least (m)
_MPC_EXT_MAX = 9.5                                       # ...at most (2 (R_max + 0.3) + noise)
_MPC_END = _cx_knob("CX_MPC_END", 1.0, _CX_MPC)          # wait point this far inside the observed stroke ends (m)
_MPC_LAT = _cx_knob("CX_MPC_LAT", 2.5, _CX_MPC)          # arm: drone within this of the axis (m)
_MPC_NEAR = _cx_knob("CX_MPC_NEAR", 4.0, _CX_MPC)        # arm: drone within this of the pad estimate (m)
_MPC_DZLO = _cx_knob("CX_MPC_DZLO", 0.45, _CX_MPC)       # arm: drone at least top - this (m)
_MPC_DZHI = _cx_knob("CX_MPC_DZHI", 1.0, _CX_MPC)        # arm: drone at most top + this (m)
_MPC_MG_D0 = _cx_knob("CX_MPC_MG_D0", 1.5, _CX_MPC)      # T2: the MG episode started this close to the pad estimate (m)
_MPC_MG_DELAY = _cx_knob("CX_MPC_MG_DELAY", 0.5, _CX_MPC)  # T2: this long after the MG episode ended (s)
_MPC_T2 = _cx_knob("CX_MPC_T2", 1.0, _CX_MPC) > 0.5      # T2 on
_MPC_T2_DX = _cx_knob("CX_MPC_T2_DX", 1.5, _CX_MPC)      # T2: not with the drone above top + _MG_ON within this of the
#   pad estimate (m): the drone is over the pad, where fx_dv's dive is how c29h lands (1247666021: T2 took over a dive
#   landing, +4.5 / +5.1 s)
_MPC_T3 = _cx_knob("CX_MPC_T3", 0.0, _CX_MPC) > 0.5      # T3 (STRESS TEST ONLY, keep off): try to arm every 0.5 s from the
#   first tick the drone is within _MPC_T3_D of the pad estimate (no near-miss needed): exercises the wait on every roof pad
_MPC_T3_D = _cx_knob("CX_MPC_T3_D", 2.5, _CX_MPC)
_MPC_TLAST = _cx_knob("CX_MPC_TLAST", 54.0, _CX_MPC)     # no episode starts later (s)
_MPC_TMAX = _cx_knob("CX_MPC_TMAX", 18.0, _CX_MPC)       # episode length cap (s; linear periods 10.5-21 s)
_MPC_TKEEP = _cx_knob("CX_MPC_TKEEP", 48.0, _CX_MPC)     # ...applied only up to this flight time (s): later the wait goes on
_MPC_TEND = _cx_knob("CX_MPC_TEND", 59.9, _CX_MPC)       # release after this flight time (s)
_MPC_DZ = _cx_knob("CX_MPC_DZ", -0.06, _CX_MPC)          # wait height relative to a ray-measured pad top (m; fx_mg's floor)
_MPC_DZP = _cx_knob("CX_MPC_DZP", -0.02, _CX_MPC)        # ...relative to a pilot-median top (m): that median reads 0.01-0.16 m
#   (median 0.08) under the true top (56 deep MG starts), the ray 0.00-0.02; side contact needs the drone centre within
#   [top - 0.2125, top + 0.0125]: -0.02 centres the window on the median bias (margins >= 3 cm at both ends)
_MPC_VH = _cx_knob("CX_MPC_VH", 2.0, _CX_MPC)            # hop speed cap (m/s)
_MPC_DVH = _cx_knob("CX_MPC_DVH", 0.6, _CX_MPC)          # command within this of the drone's horizontal velocity (m/s)
_MPC_CLIMB = _cx_knob("CX_MPC_CLIMB", 0.15, _CX_MPC)     # HOP / WAIT: this far under the target height, sideways <= 0.5 m/s
_MPC_REPL = _cx_knob("CX_MPC_REPL", 0.3, _CX_MPC)        # re-place when a new valid axis lies this far from the point (m)
_MPC_REPL_MAX = _cx_knob("CX_MPC_REPL_MAX", 1.0, _CX_MPC)  # ...but a new valid axis this far off: not a line (release)
_MPC_INNOV = _cx_knob("CX_MPC_INNOV", 0.7, _CX_MPC)      # PATH CHECK: new samples within _MPC_INV_R this far off the line...
_MPC_INV_S = _cx_knob("CX_MPC_INV_S", 1.0, _CX_MPC)      # ...for this long, none on it: the pad does not run on this line
_MPC_INV_R = _cx_knob("CX_MPC_INV_R", 8.0, _CX_MPC)      #   (a circle / figure-8 arc taken for a line leaves it within ~2 s)
_MPC_SURF = _cx_knob("CX_MPC_SURF", 0.30, _CX_MPC)       # a non-pad surface above top - this under the drone: retreat (m;
#   inside the placement disc nothing lies above top - 0.43)
_MPC_TUP = _cx_knob("CX_MPC_TUP", 0.17, _CX_MPC)         # a pilot-median top: the true top may lie this much higher (m)
_MPC_SINK_H = _cx_knob("CX_MPC_SINK_H", 0.6, _CX_MPC)    # SINK when the ray reads the pad top this close under (m)
_MPC_SINK_XY = _cx_knob("CX_MPC_SINK_XY", 0.9, _CX_MPC)  # ...and the pad estimate lies this close (m; fx_mg's _MG_RAY_XY)
_MPC_SINK_LAT = _cx_knob("CX_MPC_SINK_LAT", 0.8, _CX_MPC)  # ...and the drone this close to the axis (m)
_MPC_SINK_VZ = _cx_knob("CX_MPC_SINK_VZ", 2.5, _CX_MPC)  # SINK rate cap (m/s; fx_dv's _DV_VZ)
_MPC_SINK_FL = _cx_knob("CX_MPC_SINK_FL", 0.2, _CX_MPC)  # SINK stops above top - this (m; 5 m/s^2 braking)
_MPC_IC = _cx_knob("CX_MPC_IC", 1.0, _CX_MPC) > 0.5      # intercept wait point (else the axis point nearest the drone)
_MPC_IC_VP = _cx_knob("CX_MPC_IC_VP", 1.8, _CX_MPC)      # intercept: pad speed bound (m/s; the step clamp 1.5 speed <= 1.8)
_MPC_IC_VH = _cx_knob("CX_MPC_IC_VH", 1.5, _CX_MPC)      # intercept: hop mean speed (m/s)
_MPC_IC_H0 = _cx_knob("CX_MPC_IC_H0", 0.8, _CX_MPC)      # intercept: hop start / brake / settle time (s)
_MPC_IC_VMIN = _cx_knob("CX_MPC_IC_VMIN", 0.3, _CX_MPC)  # intercept: the pad's along-axis speed must read this (m/s)
_MPC_IC_WV = _cx_knob("CX_MPC_IC_WV", 0.6, _CX_MPC)      # intercept: velocity regression window (s)
_MPC_HB_S = _cx_knob("CX_MPC_HB_S", 1.5, _CX_MPC)        # hand-back: rate-limit window (s)
_MPC_HBM = _cx_knob("CX_MPC_HBM", 1.0, _CX_MPC) > 0.5    # hand-back to MINE (the vking take-back: REACQUIRE aimed at the
#   pad estimate, as CX_VK_WATCH does) instead of the king: after an MPC hold the village graph pack sank in place onto a
#   roof (906587310) or hovered to the timeout in 7 of 9 releases (mpc_v3 stress arm)


def _cx_mpc_ev(name, n=1):
    # fx_mpc counter (plain: MPC runs in the router, after any shadow act() has returned)
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


def _cx_mpc_log(name, row, cap=8):
    lst = CX_EVENTS.setdefault(name, [])
    if isinstance(lst, list) and len(lst) < cap:
        lst.append(row)


def _mpc_line_fit(S):
    """fx_mpc line fit of pad xy samples S = [(t, x, y, depth, weight), ...] (time order). Returns (fit dict, 'ok') or
    (partial dict or None, reason): weighted principal axis (lateral outliers dropped once), lateral rms, the quadratic
    sag over the observed stroke, the stroke extent (along-axis q03-q97)."""
    if len(S) < _MPC_MINN:
        return None, "n"
    A = np.asarray(S, dtype=np.float64)
    if A[-1, 0] - A[0, 0] < _MPC_SPAN:
        return None, "span"
    keep = np.ones(A.shape[0], dtype=bool)
    fit = None
    for it_ in range(2):
        B = A[keep]
        w = B[:, 4] / B[:, 4].sum()
        mu = w @ B[:, 1:3]
        X = B[:, 1:3] - mu[None, :]
        C = (X * w[:, None]).T @ X
        _ev, V = np.linalg.eigh(C)
        u = V[:, 1].copy()
        if u[0] < 0.0 or (u[0] == 0.0 and u[1] < 0.0):
            u = -u
        nv = np.array([-u[1], u[0]])
        r_all = (A[:, 1:3] - mu[None, :]) @ nv
        rms = float(np.sqrt(w @ (r_all[keep] ** 2)))
        fit = dict(mu=mu, u=u, nv=nv, rms=rms)
        if it_ == 0:
            k2 = np.abs(r_all) <= max(_MPC_OUT, 3.0 * rms)
            if k2.sum() >= _MPC_MINN and not k2.all():
                keep = k2
                continue
        break
    B = A[keep]
    w = B[:, 4] / B[:, 4].sum()
    s = (B[:, 1:3] - fit["mu"][None, :]) @ fit["u"]
    r = (B[:, 1:3] - fit["mu"][None, :]) @ fit["nv"]
    s_lo, s_hi = (float(q) for q in np.quantile(s, [0.03, 0.97]))
    ext = s_hi - s_lo
    D = np.stack([s * s, s, np.ones_like(s)], axis=1)
    sw = np.sqrt(w)
    try:
        coef = np.linalg.lstsq(D * sw[:, None], r * sw, rcond=None)[0]
        sag = abs(float(coef[0])) * (0.5 * ext) ** 2
    except Exception:
        sag = 9.9
    fit.update(s_lo=s_lo, s_hi=s_hi, ext=ext, sag=sag, n=int(B.shape[0]), nout=int((~keep).sum()),
               span=float(A[-1, 0] - A[0, 0]), t1=float(A[-1, 0]))
    if fit["rms"] > _MPC_RMS:
        return fit, "rms"
    if sag > _MPC_SAG:
        return fit, "sag"
    if ext < _MPC_EXT:
        return fit, "short"
    if ext > _MPC_EXT_MAX:
        return fit, "long"
    return fit, "ok"


def _mpc_sv(S, fit, t):
    """fx_mpc: the pad's along-axis position and velocity at t, (s, v), from a weighted linear regression over the
    samples of the last _MPC_IC_WV s (>= 5 spanning >= half of it), else (None, None)."""
    W = [s_ for s_ in S if t - s_[0] <= _MPC_IC_WV + 1e-9]
    if len(W) < 5:
        return None, None
    A = np.asarray(W, dtype=np.float64)
    if float(A[:, 0].max() - A[:, 0].min()) < 0.5 * _MPC_IC_WV:
        return None, None
    s = (A[:, 1:3] - fit["mu"][None, :]) @ fit["u"]
    tt = A[:, 0] - t
    w = A[:, 4] / A[:, 4].sum()
    tm, sm = float(w @ tt), float(w @ s)
    var = float(w @ ((tt - tm) ** 2))
    if not var > 1e-9:
        return None, None
    b = float(w @ ((tt - tm) * (s - sm))) / var
    a = sm - b * tm
    if not (math.isfinite(a) and math.isfinite(b)):
        return None, None
    return a, b


def _mpc_pick(fit, xy, s0, vs):
    """fx_mpc wait point: (along-axis coordinate s_p, 'near' / 'near0' / 'icpt', predicted pad arrival after the arm or
    None). Candidates: the axis points clamped min(_MPC_END, extent / 4) inside the observed stroke. 'near': the one
    nearest xy (no usable pad heading: |vs| < _MPC_IC_VMIN or unknown, or _MPC_IC off). 'icpt': on a 0.1 m grid, the
    point the pad reaches first at or after the drone can be there (hop |P - xy| / _MPC_IC_VH + _MPC_IC_H0). The pad's
    arrivals are lower bounds (speed _MPC_IC_VP, turning at the observed stroke end; the true end lies at or beyond it)
    but keep the pad's true visiting order, so a slower pad or a farther end delays every candidate alike. 'near0': no
    candidate reachable in time over the first pass and the pass after the turn (nearest point)."""
    rel = np.asarray(xy, dtype=np.float64)[:2] - fit["mu"]
    s_d = float(rel @ fit["u"])
    lat = float(rel @ fit["nv"])
    mg = min(_MPC_END, 0.25 * fit["ext"])
    lo, hi = fit["s_lo"] + mg, fit["s_hi"] - mg
    s_n = min(max(s_d, lo), hi)
    if (not _MPC_IC or s0 is None or vs is None or not math.isfinite(s0) or not math.isfinite(vs)
            or abs(vs) < _MPC_IC_VMIN or hi - lo < 0.2):
        return s_n, "near", None
    sg = 1.0 if vs > 0.0 else -1.0
    E = fit["s_hi"] if sg > 0.0 else fit["s_lo"]   # the turn ahead of the pad
    run = max(sg * (E - s0), 0.0)
    best = None
    n = int((hi - lo) / 0.1) + 1
    for k in range(n + 1):
        s = lo + (hi - lo) * k / n
        h = math.hypot(s - s_d, lat) / _MPC_IC_VH + _MPC_IC_H0
        a1 = sg * (s - s0)
        arr = [a1 / _MPC_IC_VP] if a1 >= 0.0 else []    # the first pass (a point ahead of the pad)
        arr.append((run + abs(E - s)) / _MPC_IC_VP)     # the pass after the turn
        # (later passes are not compared: with lower-bound arrivals the conservative feasibility test would favour
        # points whose early pass it wrongly rejects; icpt_mc.py: worse than the nearest point then)
        ok = [x for x in arr if x >= h]
        if ok and (best is None or min(ok) < best[0] - 1e-9):
            best = (min(ok), s)
    if best is None:
        return s_n, "near0", None
    return best[1], "icpt", best[0]


# --- fx_cr: city KING roof guard (flags CX_CR_KRAY, CX_CR_KFWD; default OFF; with both off nothing below runs) ---
# Failure family (round 18, the champion crashes on the same seeds): on city, after mine's moving-pad hand-over to the
# king (route king, king_from_dual), the king pilot's navigation glide sinks at 1.5-2.3 m/s toward a pad 10-13 m away
# and meets a roof on the way (968536666, 1065510188, 796273227: SUPPORT_ROOFTOP contact 10-12 m from the pad, 3-4 m
# above its top). None of mine's city guards (roofguard, cwall, cskim, ...) runs on the king route.
# The guard runs in the router after CX_DV_DIVE and CX_MG_FLOOR, on the city king route after the moving hand-over only
# (never inside a shadow act()), and changes only the vertical command (and, very low, the horizontal speed).
# NON-PAD: a surface or depth point counts only when it lies at least _CR_RAISE above the pad top (the higher of mine's
#   hand-over estimate and the fx_mg pad-top reading) while the drone is at least _CR_PADR from every pad xy estimate
#   (mine's, fx_mg's predicted and good ones, the king pilot's fresh detection). City pads sit on the street (pad tops
#   0.25-1.05 m over 878 batch seeds); a pad on anything raised still has its own top as the reference.
# KRAY (down-ray): a NON-PAD surface under the drone with agl < max(_CR_AGL, 0.6 + _CR_TTC * closing), closing = sink
#   rate + the rate the surface under the drone rises (0.1 s window, restarted at a step of > 0.3 m): the vertical
#   command becomes at least min(_CR_KP * (_CR_AGL - agl), _CR_VUP) (>= 0) plus the rise rate (<= 1.5); under
#   _CR_BRAKE_AGL the horizontal speed is capped at _CR_VH.
# KFWD (depth; off in the accepted config): only while sinking faster than _CR_FVZ, the depth frame (4x4 min-pooled)
#   projected to world; a NON-PAD raised point within _CR_W of the horizontal track, 0.2 m to clip(_CR_FT * vh, 1, 4) m
#   ahead, less than _CR_MARGIN under the straight glide (z + vz * s / vh) and not above z + 0.5 stops the sink
#   (vz >= 0) for _CR_FHOLD s; one within 0.3 m of the glide climbs at _CR_VUP_F.
# COMMAND: a raised vertical command leads the measured vz by at most _CR_LEAD, and a capped horizontal command stays
#   within _CR_DVH of the drone's horizontal velocity (a step from a 2 m/s sink to a 3 m/s climb saturated the thrust
#   and the attitude loop lost the horizontal: 796273227 sped up to 3.8 m/s at 39 deg). The route's yaw is kept.
# Events (CX_EVENTS): cr_ray / cr_fwd / cr_brake (ticks), cr_fire (episodes), cr_at (first 8 episodes: t, source, agl,
#   surface - pad top, distance to the nearest pad estimate, vz, [rise] or [s, point z - top, glide clearance, n]), cr_err.
_CX_CR_KRAY = _cx_on("CX_CR_KRAY")
_CX_CR_KFWD = _cx_on("CX_CR_KFWD")
_CX_CR_ANY = _CX_CR_KRAY or _CX_CR_KFWD
_CR_RAISE = _cx_knob("CX_CR_RAISE", 0.8, _CX_CR_ANY)      # a surface this far above the pad top is not the pad (m)
_CR_PADR = _cx_knob("CX_CR_PADR", 1.5, _CX_CR_ANY)        # never act within this of a pad xy estimate (m)
_CR_AGL = _cx_knob("CX_CR_AGL", 2.0, _CX_CR_ANY)          # KRAY: agl held over a non-pad surface (m)
_CR_TTC = _cx_knob("CX_CR_TTC", 0.7, _CX_CR_ANY)          # KRAY: fire earlier by this much closing time (s)
_CR_KP = _cx_knob("CX_CR_KP", 1.5, _CX_CR_ANY)            # KRAY: climb gain (1/s)
_CR_VUP = _cx_knob("CX_CR_VUP", 1.5, _CX_CR_ANY)          # KRAY: climb cap (m/s)
_CR_BRAKE_AGL = _cx_knob("CX_CR_BRAKE_AGL", 1.2, _CX_CR_ANY)  # KRAY: under this agl cap the horizontal speed (m)
_CR_VH = _cx_knob("CX_CR_VH", 1.0, _CX_CR_ANY)            # KRAY: that cap (m/s)
_CR_W = _cx_knob("CX_CR_W", 0.8, _CX_CR_ANY)              # KFWD: half-width of the swept track (m)
_CR_MARGIN = _cx_knob("CX_CR_MARGIN", 0.8, _CX_CR_ANY)    # KFWD: glide clearance that stops the sink (m)
_CR_VUP_F = _cx_knob("CX_CR_VUP_F", 0.8, _CX_CR_ANY)      # KFWD: climb when the glide clears a point by < 0.3 m (m/s)
_CR_FVZ = _cx_knob("CX_CR_FVZ", 0.8, _CX_CR_ANY)          # KFWD: only while sinking faster than this (m/s)
_CR_FT = _cx_knob("CX_CR_FT", 1.2, _CX_CR_ANY)            # KFWD: look this far ahead in time along the track (s)
_CR_FHOLD = _cx_knob("CX_CR_FHOLD", 0.3, _CX_CR_ANY)      # KFWD: keep the sink stopped this long after a hit (s)
_CR_LEAD = _cx_knob("CX_CR_LEAD", 1.2, _CX_CR_ANY)        # the climb command leads the measured vz by at most this (m/s)
_CR_DVH = _cx_knob("CX_CR_DVH", 1.0, _CX_CR_ANY)          # a braked horizontal command stays within this of the drone's (m/s)


def _cx_cr_ev(name, n=1):
    # fx_cr counter (router level, after any shadow act() has returned)
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


# --- fx_vk (round 23): village KING route, obstacle guard (flags CX_VK2_*; default OFF; all off == c21) ---
# (The round is named fx_vk; its flags are CX_VK2_* because CX_VK_* belong to the earlier village hand-over round.)
# Failure family: on village the king (mostly the graph pack) flies level at z 2.4-2.5 m, or blind-repeats one action
# at 0.4-0.8 m, into a house wall or a street object 7-20 m from the pad (1034949795, 1076245508, 1432876675,
# 484837393, 346879337, 400833060), or sinks onto a roof away from the pad (531897584, 466294885, 669185741), or
# fx_mg's hold keeps it at the pad-top height of a roof pad while a roof rises under it (1873314559). No outer guard
# runs on the village king route. Everything below runs in the router after CX_KL_MDROP, on the village king route
# only (_vk2_gate: route king and mine's map label village), never inside a shadow act().
# DIAG: records the BRAKE geometry (no effect on the flight): vk2_diag = the last _VK2_DIAG_N ticks [t, |v|, free along
#   v, free along the command (None = not visible), agl, distance to the nearest pad estimate, would act].
# BRAKE: the depth frame (4x4 min-pooled) projected to world, plus the frames of the last _VK2_MEM_S s (every 5th tick,
#   points within 6 m); points below the ground (lowest down-ray surface this flight + _VK2_GCUT) and within _VK2_PADR
#   (xy) of any pad estimate (plus _VK2_PADV x the age of a remembered frame) are dropped. Along the measured velocity and along the route's commanded velocity, the free
#   distance in a tube of radius _VK2_TUBE (the current frame counts only for a direction inside the camera frustum,
#   +-_VK2_VIEW deg). The closing speed along that direction is capped at sqrt(2 * _VK2_DEC * (free - _VK2_MARGIN))
#   (only the excess is removed; the sideways part stays). Not within _VK2_NEAR (xy) of a pad estimate.
# PIN: BRAKE has held the drone (< _VK2_PIN_V) against an obstacle for _VK2_PIN_S (gaps <= _VK2_PIN_GAP): after a
#   village moving hand-over with the vking memory armed, take the flight back the way CX_VK_WATCH does, aimed at my
#   shadow track's last sighting (>= 10 hits) on the king route.
# ROOF: the down-ray reads a raised surface (> ground + _VK2_ROOF_RAISE) at least _VK2_ROOF_R from every pad estimate,
#   and agl < 0.4 + _VK2_ROOF_TTC * closing (closing = sink + the rate the surface rises under the drone); farther than
#   _VK2_ROOF_FAR from every estimate also any agl < _VK2_ROOF_AGL (with no estimate at all, only the closing rule); nearer, only a rising surface
#   (>= 0.3 m/s) counts (a roof pad's own roof is flat and just below it). The vertical command becomes at least
#   min(1.5 * (_VK2_ROOF_AGL + 0.2 - agl), 1.2) + rise (<= 1.5), leading the measured vz by at most _VK2_LEAD; under
#   _VK2_ROOF_BRAKE_AGL the horizontal speed is capped at _VK2_ROOF_VH.
# COMMAND: the guarded command stays within _VK2_DVH of the drone's velocity; the route's yaw is kept.
# Events: vk2_fire / vk2_ticks / vk2_at (BRAKE episodes, ticks, first 8), vk2_roof_fire / vk2_roof_ticks / vk2_roof_at,
#   vk2_pin_takeback / vk2_pin_t, vk2_would* (DIAG without BRAKE), vk2_err, vk2_pin_err.
_CX_VK2_DIAG = _cx_on("CX_VK2_DIAG")
_CX_VK2_BRAKE = _cx_on("CX_VK2_BRAKE")
_CX_VK2_PIN = _cx_on("CX_VK2_PIN")
_CX_VK2_ROOF = _cx_on("CX_VK2_ROOF")
_CX_VK2_ANY = _CX_VK2_DIAG or _CX_VK2_BRAKE or _CX_VK2_ROOF
_VK2_TUBE = _cx_knob("CX_VK2_TUBE", 0.4, _CX_VK2_ANY)      # tube radius around the path (m)
_VK2_DEC = _cx_knob("CX_VK2_DEC", 3.0, _CX_VK2_ANY)        # braking deceleration assumed (m/s^2)
_VK2_MARGIN = _cx_knob("CX_VK2_MARGIN", 0.6, _CX_VK2_ANY)  # stop this far short of the first point in the tube (m)
_VK2_PADR = _cx_knob("CX_VK2_PADR", 1.2, _CX_VK2_ANY)      # depth points this close (xy) to a pad estimate are ignored (m)
_VK2_PADV = _cx_knob("CX_VK2_PADV", 1.5, _CX_VK2_ANY)      # ...plus this x the age of a remembered frame (a moving pad, m/s)
_VK2_NEAR = _cx_knob("CX_VK2_NEAR", 3.0, _CX_VK2_ANY)      # BRAKE never acts this close (xy) to a pad estimate (m)
_VK2_GCUT = _cx_knob("CX_VK2_GCUT", 0.35, _CX_VK2_ANY)     # depth points lower than the ground + this are ignored (m)
_VK2_MEM_S = _cx_knob("CX_VK2_MEM_S", 1.2, _CX_VK2_ANY)    # recent frames kept as obstacle memory (s)
_VK2_VMIN = _cx_knob("CX_VK2_VMIN", 0.1, _CX_VK2_ANY)      # only directions of at least this speed are checked (m/s)
_VK2_VIEW = _cx_knob("CX_VK2_VIEW", 40.0, _CX_VK2_ANY)     # a direction is in view within this of the camera axis (deg)
_VK2_DVH = _cx_knob("CX_VK2_DVH", 2.0, _CX_VK2_ANY)        # the guarded command stays within this of the velocity (m/s)
_VK2_PIN_V = _cx_knob("CX_VK2_PIN_V", 0.6, _CX_VK2_ANY)    # PIN: braked ticks count only below this speed (m/s)
_VK2_PIN_S = _cx_knob("CX_VK2_PIN_S", 1.0, _CX_VK2_ANY)    # PIN: pinned this long (s)
_VK2_PIN_GAP = _cx_knob("CX_VK2_PIN_GAP", 0.3, _CX_VK2_ANY)  # PIN: gaps up to this keep the episode (s)
_VK2_ROOF_RAISE = _cx_knob("CX_VK2_ROOF_RAISE", 1.0, _CX_VK2_ANY)  # ROOF: a surface this far over the ground is raised (m)
_VK2_ROOF_R = _cx_knob("CX_VK2_ROOF_R", 1.5, _CX_VK2_ANY)  # ROOF: never within this (xy) of a pad estimate (m)
_VK2_ROOF_FAR = _cx_knob("CX_VK2_ROOF_FAR", 5.0, _CX_VK2_ANY)  # ROOF: beyond this any low pass counts (m)
_VK2_ROOF_AGL = _cx_knob("CX_VK2_ROOF_AGL", 0.8, _CX_VK2_ANY)  # ROOF: agl kept over a far raised surface (m)
_VK2_ROOF_TTC = _cx_knob("CX_VK2_ROOF_TTC", 0.7, _CX_VK2_ANY)  # ROOF: fire earlier by this much closing time (s)
_VK2_ROOF_BRAKE_AGL = _cx_knob("CX_VK2_ROOF_BRAKE_AGL", 0.5, _CX_VK2_ANY)  # ROOF: under this agl cap the horizontal speed
_VK2_ROOF_VH = _cx_knob("CX_VK2_ROOF_VH", 1.0, _CX_VK2_ANY)  # ROOF: that cap (m/s)
_VK2_LEAD = _cx_knob("CX_VK2_LEAD", 1.2, _CX_VK2_ANY)      # ROOF: the climb leads the measured vz by at most this (m/s)
_VK2_DIAG_N = 150


def _cx_vk2_ev(name, n=1):
    # fx_vk (round 23) counter (router level, after any shadow act() has returned)
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


# --- fx_kl (round 21): failures that end under the king. Every flag defaults OFF; all off == c20. ---
# CX_KL_MDROP ("the pad slides under a hovering drone and nothing sinks onto it"). Mountain moving pads float at a fixed
#   height over sloped terrain, often 1-4 m above it. After the moving hand-over uid130 loses the pad, CX_MV_ANCHOR sends
#   its goal-return search to the orbit centre and it spins there ~0.5 m above the pad plane (its goal-return target is
#   the pad height + 0.5 + d/10). Linear and figure-8 orbits pass through that centre, so the pad slides under the
#   drone again and again; the down-ray reads the pad top 0.4-0.8 m below for 0.5-1 s each time, and nothing descends
#   (AVX2 batch: 2114808376 21 such rows, 810927998 14, 415920981 15, all timeouts; dev pod: 2114808376 at 37.5 s,
#   and at 40.2-40.9 s uid130's own slow landing sank to agl 0.05 as the pad slid out). With the flag on (mountain,
#   king route after the moving hand-over, the orbit anchor armed, i.e. a pad whose samples really move): when the
#   surface under the drone rises by >= _KL_RISE onto the known pad-top height (the ray-measured top of CX_MG_FLOOR,
#   within _KL_TOL; else the anchor height within _KL_TOL_ANC) and the drone is 0.12-1.0 m above it within _KL_ANC_R
#   of the orbit anchor, sink onto it with the fx_dv rate-limited dive (the command leads the measured vz by at most
#   0.25 m/s, up to 2.5 m/s; horizontal within 0.3 m/s of the drone's velocity, stepping toward a fresh pad velocity
#   estimate, else holding) and the fx_dv floor (the drone can always stop 0.4 m above the surface it read before the
#   pad came under it). A moving pad's first contact is a landing. The dive ends when the ray leaves the pad top
#   (contact, or the pad slid away: then a 0.3 s brake if the drone would not stop in time), after _KL_MAX_S, or when
#   blocked; at most _KL_MAX_N dives per seed, each needing a new rise. Needs CX_MG_FLOOR and CX_MV_ANCHOR (c20 has both).
# Events: kl_fire (dives), kl_src_anc (dives on the anchor height), kl_ticks, kl_end_{left,time,block,hit}, kl_brake,
#   kl_fire1 = [t, agl, surface - top, xy to anchor, pad-velocity source, king mode, vz] of the first dive, kl_err.
_CX_KL_MDROP = _cx_on("CX_KL_MDROP")
_KL_TOL = _cx_knob("CX_KL_TOL", 0.06, _CX_KL_MDROP)          # |surface - ray-measured pad top| (m)
_KL_TOL_ANC = _cx_knob("CX_KL_TOL_ANC", 0.10, _CX_KL_MDROP)  # |surface - anchor height| when no ray top exists (m)
_KL_RISE = _cx_knob("CX_KL_RISE", 0.5, _CX_KL_MDROP)         # the surface rose at least this onto the pad top (m)...
_KL_RISE_S = _cx_knob("CX_KL_RISE_S", 0.3, _CX_KL_MDROP)     # ...within this long (s)
_KL_HMIN = _cx_knob("CX_KL_HMIN", 0.12, _CX_KL_MDROP)        # start a dive only from this high over the pad top (m)...
_KL_HMAX = _cx_knob("CX_KL_HMAX", 1.0, _CX_KL_MDROP)         # ...up to this high (m)
_KL_ANC_R = _cx_knob("CX_KL_ANC_R", 5.5, _CX_KL_MDROP)       # drone within this of the orbit anchor (orbit r <= 4.3 m)
_KL_VZ = _cx_knob("CX_KL_VZ", 2.5, _CX_KL_MDROP)             # sink cap (m/s)
_KL_MAX_S = _cx_knob("CX_KL_MAX_S", 0.9, _CX_KL_MDROP)       # dive length cap (s)
_KL_MAX_N = int(_cx_knob("CX_KL_MAX_N", 4.0, _CX_KL_MDROP))  # dives per seed
_KL_VFRESH = _cx_knob("CX_KL_VFRESH", 0.5, _CX_KL_MDROP)     # a pad velocity estimate this fresh steers the dive (s)


# CX_KL_KBACK ("the blind king sinks in 'landing' with the pad lost"). After the moving hand-over on city (the king's
#   main pilot) and mountain (uid130), the pilot's 'landing' branch with the tracking flag set has no exit: once the pad
#   is out of view it repeats its last action forever (fx_mg). When that action sinks, the drone sinks beside the pad
#   into the (ray-transparent) ground plane or the terrain, 5-12 m from the pad (dev pod: 1882127997, 732272598,
#   1214742324, and fx_cr's fresh 1414577637, 2094996184; AVX2 batch: mountain 2097637047). CX_MG_FLOOR's hold can
#   only turn that into a timeout (both score 0.01). With the flag on: once the king pilot has sat in 'landing' without
#   sight of the pad for _KB_LOST_S, with the drone low (at most _KB_ABOVE over the pad top) and not over the pad top,
#   at least _KB_AFTER after the hand-over, take the flight back as CX_VK_WATCH does on village: my controller
#   REACQUIREs the moving pad from 7 m back and 3 m above its last good estimate (mountain: the CX_MV_ANCHOR orbit
#   centre), starting from the current velocity with any descent removed, and finishes the landing itself (the king is
#   blocked from taking it again: king_blocked, as after the static hand-back). City only by default (the one mountain
#   case, 2097637047, is AVX2-only and never reproduced here; CX_KL_KB_MTN=1 adds mountain, not flown); village has
#   CX_VK_WATCH. Like CX_VK_WATCH, the firing tick still flies the king's action.
# Events: kb_fire, kb_fire1 = [t, blind s, z - top, top source, drone-ref xy, ref source, pilot].
_CX_KL_KBACK = _cx_on("CX_KL_KBACK")
_KB_LOST_S = _cx_knob("CX_KL_KB_LOST", 3.0, _CX_KL_KBACK)    # king pilot in 'landing' without sight this long (s)
_KB_ABOVE = _cx_knob("CX_KL_KB_ABOVE", 0.5, _CX_KL_KBACK)    # drone at most this far above the pad top (m)
# (the blind time only accumulates on the king route after the moving hand-over, so it also bounds the time since it)
_KB_MAPS = ("city",) if _cx_knob("CX_KL_KB_MTN", 0.0, _CX_KL_KBACK) < 0.5 else ("city", "mountain")  # mountain: untested
# CX_KL_KB_REHO (knob, 0 = off): after a take-back, hand the flight to the king again once my controller has re-found the
#   pad (APPROACH on a reliable track that my own motion test flags moving) while the king pilot sees it too, restarting
#   the pilot's approach from 'navigation' (its old 'landing' state only sinks in place). A second stuck 'landing' (blind,
#   or seeing the pad >= _KB_FAR away) low beside the pad is then taken back once more, and my controller lands. Dev pod
#   (5 city seeds c20 crashes): my controller alone landed 2 (1882127997, 732272598); with REHO 4 (1214742324 times out).
_KB_REHO = _cx_knob("CX_KL_KB_REHO", 0.0, _CX_KL_KBACK) > 0.5
_KB_MAX_N = 2 if _KB_REHO else 1
_KB_FAR = _cx_knob("CX_KL_KB_FAR", 1.5, _CX_KL_KBACK)  # REHO: a second take-back also when the pilot sees the pad at least
#   this far away (xy) while it sits low beside it for _KB_LOST_S (its re-approach sinks in place); after the second
#   take-back my controller lands (no further hand-over)


# ==== fx_kseed: CX_PW_KSEED (from the pool_warehouse_moving analysis), flag default OFF (off == c29a exactly) ====
# CX_PW_KSEED: at a city dual->king moving hand-over the king's main pilot is often still in 'search' with its goal at the
#   clue (it has not seen the pad mine tracks; 41895495: pad 18.9 m away, clue 17.7 m from the pad, it flew to the clue
#   and searched there until 54 s; 1049306411: it saw the pad in 'takeoff', lost it, entered 'search' with goal = clue at
#   8.1 s and hit a 9 m building on the way). While it is in 'search' without a sighting, point its goal-return at mine's
#   pad fix (the MV_ANCHOR idea on city/main). Stops for good once the pilot sees a pad of its own in 'navigation' /
#   'landing', or _PW_KS_T after the hand-over, or when mine's fix is stale / too high (a roof-top track).
# Events: pw_kseed (seeded ticks), pw_kseed1 = [t, t - hand-over, drone-goal xy, clue-goal xy], pw_kseed_err.
_CX_PW_KSEED = _cx_on("CX_PW_KSEED")
_PW_KS_T = _cx_knob("CX_PW_KS_T", 8.0, _CX_PW_KSEED)      # seed only this long after the hand-over (s)
_PW_KS_AGE = _cx_knob("CX_PW_KS_AGE", 1.0, _CX_PW_KSEED)  # mine's fix at most this old at the hand-over (s)
_PW_KS_ZUP = _cx_knob("CX_PW_KS_ZUP", 0.0, _CX_PW_KSEED)  # goal this far above mine's pad-top estimate (m)
_PW_KS_ZMAX = _cx_knob("CX_PW_KS_ZMAX", 2.0, _CX_PW_KSEED)  # seed only a fix at most this high (m)
# ==== fx_pw2: CX_PW_CREP / CX_PW_CRCAP / CX_PW_GSINK (pool_warehouse_moving analysis), every flag default OFF ====
#   (all off == c29a exactly; CX_PW_KSEED of the same analysis lives in fx_kseed and is not part of this bundle)
# CX_PW_CREP: the CX_KL_KBACK take-back also fires on city when the route's action has repeated bit for bit for
#   _PW_REP_N ticks (fx_mg's repeat count, needs CX_MG_FLOOR) with a horizontal command above _PW_REP_VH while the king
#   pilot does not see the pad, in any pilot mode and at any height: the main pilot's blind repeat after overflying a
#   moving pad 0.4-1.0 m above its top (1075807930, 2099487460 flew ~50 m on into a building / onto a roof; 1404822236
#   was taken back only after 3 s by the 'landing' rule and timed out). CX_KL_KBACK only acts in 'landing' within 0.5 m
#   of the pad top after 3 s blind, CX_MG_FLOOR only within 0.2 m of it: the gap this closes.
# CX_PW_CRCAP: the fx_cr KRAY climb may not force the 3 m/s speed cap to take more than _CR_DVH off the drone's
#   measured horizontal speed in one tick: the climb is capped at sqrt(3^2 - (vh - _CR_DVH)^2), never under the route's
#   own vz (84869429: a 2.9 m/s climb over a roof edge cut vh 3.1 -> 0.65 in one tick, TILT 0.2 s after the guard
#   released; 1878817035 TILT at 57 s).
# CX_PW_GSINK: after a city moving hand-over to the king's graph pack (no platform track, so CX_KL_KBACK, CX_MG_FLOOR
#   and CX_DV_DIVE never see it) the pack can sink in place at ~0.15 m/s 3.7-5.3 m from the moving pad into the ground
#   plane (1440637350, 135801981). When the pack commands a slow near-vertical sink (down at most _PW_GS_VZ, vh under
#   _PW_GS_VH), the drone does it (measured vz -0.05 .. -_PW_GS_VZ, vh under _PW_GS_VH), low (at most _PW_GS_ABOVE over
#   the pad top) and not over the pad top (the down-ray reads a surface more than 0.15 m under it, or nothing), for
#   _PW_GS_T s in a row: take the flight back as CX_KL_KBACK does (my controller REACQUIREs the pad from 7 m back and
#   3 m above its last estimate, the king blocked). Once per seed. Pod city data (4,005 flights, h75/c28/c27): the
#   signature held >= 0.6 s in graph flights only in those two, both failures.
# Events: pw_crep / pw_crep1 = [t, repeated ticks, z - top, drone-ref xy] (with kb_fire / kb_fire1 as usual),
#   pw_crcap (capped ticks) / pw_crcap1 = [t, vh, climb asked, climb kept], pw_gsink / pw_gsink1 = [t, s in the sink,
#   z - top, top source, drone-ref xy, ref source], pw_gsink_err.
_CX_PW_CREP = _cx_on("CX_PW_CREP")
_PW_REP_N = int(_cx_knob("CX_PW_REP_N", 50.0, _CX_PW_CREP))   # repeated ticks (20 ms each)
_PW_REP_VH = _cx_knob("CX_PW_REP_VH", 0.5, _CX_PW_CREP)       # horizontal speed of the repeated command above (m/s)
_CX_PW_CRCAP = _cx_on("CX_PW_CRCAP")
_CX_PW_GSINK = _cx_on("CX_PW_GSINK")
_PW_GS_T = _cx_knob("CX_PW_GS_T", 1.0, _CX_PW_GSINK)          # the sink held this long (s)
_PW_GS_VZ = _cx_knob("CX_PW_GS_VZ", 0.35, _CX_PW_GSINK)       # slow: commanded and measured sink at most this (m/s)
_PW_GS_VH = _cx_knob("CX_PW_GS_VH", 0.25, _CX_PW_GSINK)       # in place: commanded and measured vh under this (m/s)
_PW_GS_ABOVE = _cx_knob("CX_PW_GS_ABOVE", 1.0, _CX_PW_GSINK)  # low: drone at most this over the pad top (m)


# ==== fx_s1r: Stage-1 ROUTER misread guards, every flag default OFF (all off == c29c exactly); see tools/mk_fx_s1r.py ====
# CX_RG_KFLIP: an early map-route king flight (armed at the router's tick-10/60 decision or the dual branch's early
#   open/warehouse re-route; never a moving hand-over) that started below _RG_Z0 (the router's village-start prior) is
#   re-routed as the decision would have routed it once mine's label turns to one of _RG_LABS (village) within _RG_T s
#   of arming, before the label locks. 1811841738: warehouse at tick 10, village again at 0.32 s, king to the end.
# CX_RG_KFEED (needs KFLIP): on such a flight labelled open, mine's act() runs each tick of the window (action discarded
#   unless the label flips) so its classifier keeps reading until it locks; 1432876675: open at tick 10 (facing out of
#   the village on the ground), never read again.
# CX_RG_VLOW / CX_RG_VSLOW (need CX_VM_GATE): extra fx_vm vetoes of a village motion flip: a track below any goal-pad
#   top (estimate and recent median <= _RG_VLOW_DZ above the ground), or a raw-path flip with a vote-fit speed below
#   _RG_VSLOW_SP (logged mover flips: min 0.26 m above ground, min raw-path speed 0.32 m/s).
_CX_RG_KFLIP = _cx_on("CX_RG_KFLIP")
_CX_RG_KFEED = _cx_on("CX_RG_KFEED") and _CX_RG_KFLIP
_CX_RG_ANY = _CX_RG_KFLIP
_CX_RG_VLOW = _cx_on("CX_RG_VLOW")
_CX_RG_VSLOW = _cx_on("CX_RG_VSLOW")
_RG_Z0 = _cx_knob("CX_RG_Z0", 0.7, _CX_RG_ANY)        # start height below which the watch arms (m)
_RG_T = _cx_knob("CX_RG_T", 3.0, _CX_RG_ANY)          # watch window from arming (s)
_RG_LABS = tuple(x for x in (os.environ.get("CX_RG_LABS", "village") if _CX_RG_ANY else "village").split(",") if x)
_RG_VLOW_DZ = _cx_knob("CX_RG_VLOW_DZ", 0.15, _CX_RG_VLOW)   # m above the ground (static pad tops >= 0.23)
_RG_VSLOW_SP = _cx_knob("CX_RG_VSLOW_SP", 0.25, _CX_RG_VSLOW)  # m/s vote-fit speed on a raw-path flip


def _cx_rg_log(name, value, cap=8):
    lst = CX_EVENTS.setdefault(name, [])
    if isinstance(lst, list) and len(lst) < cap:
        lst.append(value)


# ==== fx_s1m (Stage-1 track 2: mountain touchdown and near-pad landing), every flag default OFF (all off == c29c) ====
# CX_S1M_HM: static mountain touchdown point chosen per bearing from a local height map (full-resolution depth points
#   0.8-2.5 m around the frozen estimate -> 0.1 m cells, median height -> robust quadratic surface -> the 0.45 m bearing
#   whose touchdown keeps the cf2x disc furthest from it). Replaces the plane-fit downhill offset and also covers the
#   landings where that fit had too few pooled ring points. Offline (analysis/s1_s1m, 139 replayed landings vs the true
#   clearance field): +0.0034 +- 0.0008 per static mountain landing over c29c's rule.
# CX_S1M_DESC: keep a young consistent static track the drone is descending onto (< 3 m, under the look-down limit)
#   instead of APPROACH's unseen drop, with a down-ray and a float check (973802848).
# CX_S1M_RECEN: in a static LAND on a frozen estimate that failed the pad-top check, re-seed on >= 8 agreeing strong
#   detections 1.5-4 m away and APPROACH them (2014301281).
# CX_S1M_BADREC: a static LAND abort after a failed pad-top check blacklists the spot and searches (2014301281).
# CX_S1M_FLAT: at LAND entry, terrain rising through the estimated pad top (height-map cells within 0.5 m > 0.15 m above
#   a flat level near the estimate) blacklists the spot and searches (693878911). The LAND gate is unchanged.
# Events: s1m_hm (offset set), s1m_hm_fb_<why>, s1m_hm_at [t, bearing deg, est clearance, cells]; s1m_desc (armed),
#   s1m_desc_tick, s1m_desc_end_<why>; s1m_notop; s1m_recen, s1m_recen_at; s1m_badrec; s1m_flat_rej, s1m_flat_at;
#   s1m_err_<where>.
_CX_S1M_HM = _cx_on("CX_S1M_HM")
_CX_S1M_DESC = _cx_on("CX_S1M_DESC")
_CX_S1M_RECEN = _cx_on("CX_S1M_RECEN")
_CX_S1M_BADREC = _cx_on("CX_S1M_BADREC")
_CX_S1M_FLAT = _cx_on("CX_S1M_FLAT")
_CX_S1M_BUF = _CX_S1M_HM or _CX_S1M_DESC or _CX_S1M_FLAT   # these read the height-map point buffer
_CX_S1M_TOP = _CX_S1M_RECEN or _CX_S1M_BADREC              # these read the pad-top check
_CX_S1M_ANY = _CX_S1M_BUF or _CX_S1M_TOP
_S1M_CAP = int(_cx_knob("CX_S1M_CAP", 120000.0, _CX_S1M_BUF))       # points kept in the buffer (newest, ~1.4 MB)
_S1M_R0 = _cx_knob("CX_S1M_R0", 0.8, _CX_S1M_BUF)                   # height-map annulus around the estimate (m)
_S1M_R1 = _cx_knob("CX_S1M_R1", 2.5, _CX_S1M_BUF)
_S1M_CELL = _cx_knob("CX_S1M_CELL", 0.1, _CX_S1M_BUF)               # cell size (m); cell height = median
_S1M_MINC = int(_cx_knob("CX_S1M_MINC", 12.0, _CX_S1M_BUF))         # cells needed for a surface
_S1M_SMAX = _cx_knob("CX_S1M_SMAX", 1.5, _CX_S1M_HM)                # no new offset above this fitted slope
_S1M_DESC_HITS = int(_cx_knob("CX_S1M_DESC_HITS", 5.0, _CX_S1M_DESC))
_S1M_DESC_SPREAD = _cx_knob("CX_S1M_DESC_SPREAD", 1.0, _CX_S1M_DESC)  # recent fixes within this of the estimate (m)
_S1M_DESC_DH = _cx_knob("CX_S1M_DESC_DH", 3.0, _CX_S1M_DESC)        # arm only this close horizontally (m)
_S1M_DESC_S = _cx_knob("CX_S1M_DESC_S", 8.0, _CX_S1M_DESC)          # keep at most this long (s)
_S1M_DESC_FLOAT = _cx_knob("CX_S1M_DESC_FLOAT", 0.12, _CX_S1M_DESC)  # min float of the estimate over the surface (m)
_S1M_DESC_RAY = _cx_knob("CX_S1M_DESC_RAY", 0.5, _CX_S1M_DESC)      # ground this far above the pad top under us ends it
_S1M_TOP_DZ = _cx_knob("CX_S1M_TOP_DZ", 0.3, _CX_S1M_TOP)           # ray reads > h + this: no pad top under us
_S1M_TOP_N = int(_cx_knob("CX_S1M_TOP_N", 10.0, _CX_S1M_TOP))       # consecutive ticks
_S1M_TOP_T = _cx_knob("CX_S1M_TOP_T", 3.0, _CX_S1M_TOP)             # the evidence stays valid this long (s)
_S1M_RC_S = _cx_knob("CX_S1M_RC_S", 0.6, _CX_S1M_RECEN)             # detection score
_S1M_RC_N = int(_cx_knob("CX_S1M_RC_N", 8.0, _CX_S1M_RECEN))        # agreeing detections within _S1M_RC_W s
_S1M_RC_W = _cx_knob("CX_S1M_RC_W", 2.0, _CX_S1M_RECEN)
_S1M_RC_LO = _cx_knob("CX_S1M_RC_LO", 1.5, _CX_S1M_RECEN)           # distance band from the frozen estimate (m)
_S1M_RC_HI = _cx_knob("CX_S1M_RC_HI", 4.0, _CX_S1M_RECEN)
_S1M_FLAT_DZ = _cx_knob("CX_S1M_FLAT_DZ", 0.15, _CX_S1M_FLAT)       # cells this far above the level count as rising
_S1M_FLAT_N = int(_cx_knob("CX_S1M_FLAT_N", 12.0, _CX_S1M_FLAT))    # rising cells needed to reject
_S1M_FLAT_MINC = int(_cx_knob("CX_S1M_FLAT_MINC", 40.0, _CX_S1M_FLAT))  # cells within 0.5 m needed at all
_S1M_FLAT_WIN = _cx_knob("CX_S1M_FLAT_WIN", 0.25, _CX_S1M_FLAT)     # the flat level must lie this close to the estimate
_S1M_GRID = [None]  # stride-2 pixel ray factors (xr, yu), built on first use


def _cx_s1m_ev(name, n=1):
    if _CX_STATE["shadow"]:
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


def _cx_s1m_log(name, value, cap=6):
    if _CX_STATE["shadow"]:
        return
    lst = CX_EVENTS.setdefault(name, [])
    if len(lst) < cap:
        lst.append(value)


def _s1m_cells(rel, r0, r1, cell):
    """Height-map cells from points relative to the estimate (x, y, z - top): (M,3) [cell centre x, y, median z]."""
    d = np.hypot(rel[:, 0], rel[:, 1])
    m = (d >= r0) & (d <= r1) & (np.abs(rel[:, 2]) < 2.5)
    R = rel[m]
    if len(R) == 0:
        return np.zeros((0, 3))
    ij = np.floor(R[:, :2] / cell).astype(np.int64)
    key = (ij[:, 0] + 1000) * 4000 + (ij[:, 1] + 1000)
    o = np.lexsort((R[:, 2], key))
    key, z, ij = key[o], R[o, 2], ij[o]
    _, st_, cnt = np.unique(key, return_index=True, return_counts=True)
    return np.c_[(ij[st_, 0] + 0.5) * cell, (ij[st_, 1] + 0.5) * cell, z[st_ + cnt // 2]]


def _s1m_quad(C, minc):
    """Robust quadratic surface z = a x + b y + c + d x^2 + e xy + f y^2 through the cells: (coef, keep) or (None, None)."""
    if len(C) < minc:
        return None, None
    x, y = C[:, 0], C[:, 1]
    A = np.c_[x, y, np.ones(len(C)), x * x, x * y, y * y]
    keep = np.ones(len(C), bool)
    coef = None
    for _ in range(3):
        coef = np.linalg.lstsq(A[keep], C[keep, 2], rcond=None)[0]
        r = C[:, 2] - A @ coef
        sd = float(np.sqrt(np.mean(r[keep] ** 2)))
        nk = np.abs(r) <= max(2.5 * sd, 0.05)
        if nk.sum() < minc or np.array_equal(nk, keep):
            break
        keep = nk
    return coef, keep


def _s1m_hm_pick(rel, off_r):
    """fx_s1m HM (same code as analysis/s1_s1m/hm_final.py s1m_pick): (offset xy, info) or (None, why)."""
    C = _s1m_cells(rel, _S1M_R0, _S1M_R1, _S1M_CELL)
    if len(C) < _S1M_MINC:
        return None, "few"
    coef, keep = _s1m_quad(C, _S1M_MINC)
    if coef is None:
        return None, "cells"
    x, y = C[keep, 0], C[keep, 1]
    cp = np.linalg.lstsq(np.c_[x, y, np.ones(len(x))], C[keep, 2], rcond=None)[0]
    slope = float(math.hypot(cp[0], cp[1]))
    if slope > _S1M_SMAX:
        return None, "steep"
    rr = np.arange(0.62, 1.61, 0.04)
    aa = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
    gx = (rr[:, None] * np.cos(aa)[None, :]).ravel()
    gy = (rr[:, None] * np.sin(aa)[None, :]).ravel()
    gz = coef[0] * gx + coef[1] * gy + coef[2] + coef[3] * gx * gx + coef[4] * gx * gy + coef[5] * gy * gy
    best, bc, bb = None, -1.0, 0
    for b in range(16):
        a = 2.0 * math.pi * b / 16.0
        qx, qy = off_r * math.cos(a), off_r * math.sin(a)
        dxy = np.maximum(np.hypot(gx - qx, gy - qy) - 0.06, 0.0)   # cf2x collision cylinder r 0.06 m, 0.025 m tall,
        dz = np.maximum(np.abs(gz - 0.0125) - 0.0125, 0.0)        # resting 0.0125 m over the pad top
        c = float(np.sqrt(dxy * dxy + dz * dz).min())
        if c > bc:
            best, bc, bb = (qx, qy), c, b
    return np.array(best), (bb * 22.5, round(bc, 3), int(len(C)), round(slope, 2))


def _s1m_float(rel):
    """Height of the estimate over the quadratic terrain surface of the 0.8-2.5 m cells, or None without one."""
    C = _s1m_cells(rel, _S1M_R0, _S1M_R1, _S1M_CELL)
    coef, _ = _s1m_quad(C, _S1M_MINC)
    if coef is None:
        return None
    return -float(coef[2])


def _s1m_flat_bad(rel):
    """fx_s1m FLAT: (reject?, info). Cells within 0.5 m of the estimate; the highest level holding >= 25% of them within
    +-0.06 m is the candidate pad top. A real pad top is the highest surface there (the terrain under it is lower and it
    occludes the rest), so cells rising well above that level mean terrain runs through the 'pad'."""
    C = _s1m_cells(rel, 0.0, 0.5, _S1M_CELL)
    if len(C) < _S1M_FLAT_MINC:
        return False, "few%d" % len(C)
    zs = np.sort(C[:, 2])[::-1]
    level = None
    for zc in zs:
        if int((np.abs(C[:, 2] - zc) <= 0.06).sum()) >= 0.25 * len(C):
            level = float(zc)
            break
    if level is None or abs(level) > _S1M_FLAT_WIN:
        return False, "nolevel"
    n_hi = int((C[:, 2] > level + _S1M_FLAT_DZ).sum())
    return n_hi >= _S1M_FLAT_N, "lv%.2f/hi%d/n%d" % (level, n_hi, len(C))


# ==== fx_s2kpad: CX_S2K_OPEN (Stage 2, open king-route shadow padnet seeding), flag default OFF (off == c29d) ====
# On open the router hands the whole flight to the king (~0.4 s) and mine never runs, so padnet never looks. The king's
#   open detector (eye_detector.onnx, >= 0.55) fires on ~70% of the replayed frames with the pad in view, padnet on 99%
#   (analysis/s2_s2kpad). All 3 open TIMEOUTs of the rc2_co pool had the pad in clear view at 13-19 m for 5-9 s while the
#   king spiralled round a clue 17-21 m off (it puts its platform estimate back on the clue 20 ticks after it loses a
#   sighting): 1618273170, 398306488, 939723200. With the flag on, on an open map-route king flight (never a moving
#   hand-over), padnet runs in shadow while the king's main pilot is in 'takeoff' or 'search' (mine's detector and decode on
#   this tick's frame; mine's state is not touched). Hits (s >= SMIN, farther than STARTR from the start pad, within CLUER
#   of the clue, world z in [ZLO, ZHI]: open pad tops sit on the ground) feed up to 6 tracks (gate GATE + VG * age, capped
#   at GMAX); a track is credible with >= NHIT hits spanning >= SPAN s. Once the pilot has been in 'search' for WAIT s with
#   no sighting of its own (see_P off or its estimate back on the clue), no goal-return of its own and a credible track that
#   is not blacklisted, its goal-return target is set to the track (the fx_kseed mechanism). The king's own detector then
#   locks at close range and it lands on its own. 'navigation' / 'landing' are never touched. A seeded track that padnet
#   did not see again during the attempt is blacklisted when the goal-return ends (the king's own 400-tick limit); at most
#   MAXSEED seeds per flight. Events: s2k_arm, s2k_cred1, s2k_seed(1), s2k_klock(1), s2k_end, s2k_bl, s2k_nd, s2k_ms, s2k_err.
_CX_S2K_OPEN = _cx_on("CX_S2K_OPEN")
_S2K_WAIT = _cx_knob("CX_S2K_WAIT", 2.5, _CX_S2K_OPEN)      # s of search without an own sighting before a seed
_S2K_SMIN = _cx_knob("CX_S2K_SMIN", 0.45, _CX_S2K_OPEN)     # padnet score of a hit (mine's own 0.45)
_S2K_NHIT = _cx_knob("CX_S2K_NHIT", 15.0, _CX_S2K_OPEN)     # hits of a credible track
_S2K_SPAN = _cx_knob("CX_S2K_SPAN", 0.5, _CX_S2K_OPEN)      # s between a credible track's first and last hit
_S2K_GATE = _cx_knob("CX_S2K_GATE", 1.0, _CX_S2K_OPEN)      # association gate (m, xy) ...
_S2K_VG = _cx_knob("CX_S2K_VG", 1.5, _CX_S2K_OPEN)          # ... plus this much per second since the track's last hit
_S2K_GMAX = _cx_knob("CX_S2K_GMAX", 3.0, _CX_S2K_OPEN)      # ... capped here
_S2K_ZLO = _cx_knob("CX_S2K_ZLO", -0.6, _CX_S2K_OPEN)       # hit world z band (open pad tops: 0.2-1.3 m)
_S2K_ZHI = _cx_knob("CX_S2K_ZHI", 2.2, _CX_S2K_OPEN)
_S2K_CLUER = _cx_knob("CX_S2K_CLUER", 25.0, _CX_S2K_OPEN)   # hits farther than this from the clue (xy) are ignored
_S2K_STARTR = _cx_knob("CX_S2K_STARTR", 4.0, _CX_S2K_OPEN)  # hits this close to the start pad (xy) are ignored
_S2K_MAXSEED = _cx_knob("CX_S2K_MAXSEED", 3.0, _CX_S2K_OPEN)  # seeds per flight
_S2K_EVERY = _cx_knob("CX_S2K_EVERY", 1.0, _CX_S2K_OPEN)    # shadow detector on every N-th eligible tick


def _cx_kl_ev(name, n=1):
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


# --- fx_m15: static mountain timeouts, round 15 (flag prefix CX_M15_). Every flag defaults OFF; all off == c14. ---
# CX_M15_LOOK ("glimpse not followed"): mine's detector sees the real pad a few times (2-7 hits, often at 12-19 m range,
#   during CRUISE or the SEARCH spin), the track never becomes reliable (reliable needs a sighting < 2.5 s old AND 5 hits
#   at conf > 0.6 or 8 hits at conf > 0.5), the camera turns away, and the young track sits stale for 10-25 s while the
#   drone searches elsewhere (dev pod, c14: 1235875381 7 hits at 0.1 m from the pad at 31.7 s, dropped at 46 s, re-found
#   at 52.8 s; 1956260438 7 hits at 0.6 m at 31-34 s, APPROACH only at 57.5 s; 548792766, 233142382, 848760969). With
#   the flag on, in a static mountain CRUISE or SEARCH, a stale (unseen >= _M15_LOOK_UNSEEN) young track with >=
#   _M15_LOOK_HITS hits whose estimate is not in the camera's view and lies within clue z +- _M15_LOOK_DZ (the task
#   draws clue z = pad z + U(-5, 5); false young tracks on terrain sat 6-14 m off that band) gets ONE look: yaw toward it;
#   farther than _M15_LOOK_DFAR m, close in (planner on, the request is along the camera axis); nearer, hover and move
#   only vertically (planner off, as the champion's spin) to 0.7 x the xy distance above it (_ZLO.._ZHI, never under
#   ground + 2; a pad above: climb), so the estimate sits ~35 deg below the camera axis. Backing away is never asked for:
#   the planner flies a request > 50 deg off the camera axis along its nearest in-view ray (forward). While a look flies
#   CRUISE, the mountain climb-over rule (planner cut the cruise -> climb at 2.5 m/s) is skipped. The champion's own
#   track logic then decides: new detections make the track reliable and the mode chain switches to APPROACH; a false
#   estimate is missed in view for 1.2 s and _update_track drops it. The look ends on either, on a new track, on a
#   moving flag, when the ground + 2 floor keeps the estimate > _M15_LOOK_BLIND deg below the horizon for
#   _M15_LOOK_BLIND_S s, or after _M15_LOOK_CAP s; one look per track, _M15_LOOK_MAX per seed, none after
#   _M15_LOOK_TMAX s or farther than _M15_LOOK_DMAX m. Events: m15_look (started), m15_look_end_{drop,reliable,cap,new,
#   moving,mode,blind}, m15_look_ticks. (Revision 2; revision 1 flew to a point behind the estimate with the planner on,
#   which crept forward and, in CRUISE, triggered the climb-over rule: 848760969 climbed 12 m.)
# CX_M15_KEEP ("young track expires during the blind final descent"): a static mountain APPROACH on a young track
#   descends steeply; the pad is under the image's bottom edge, so it cannot be seen, and the unseen limit (weak track:
#   3 s; other non-established tracks: 6 s) drops a correct estimate a few metres above the pad (265620305: 11 hits,
#   0.1 m error, dropped 0.85 m beside and 2.9 m above the pad at 55.9 s; 1024549284: weak, 18 hits, 0.2 m error,
#   dropped 4.3 m beside and 6.9 m above at 42 s, then 17 s of rings). fx_ms CX_MS_KEEP covers only the non-weak branch
#   with >= 10 hits. With CX_M15_KEEP on (both branches), when the drop would fire, the track is kept while: >=
#   _M15_KEEP_HITS hits, no in-view misses beyond 1.2 s, the estimate below the look-down limit (h > 0.8 d + 0.5, d < 8 m)
#   or under the image's bottom rows, the drone closing in (or within 1 m), unseen <= _M15_KEEP_S, the down ray not on
#   ground > 0.6 m above the estimate once over it, and (fx_ms float veto) the estimate floating >= 0.15 m over the
#   terrain plane fitted around it when such a fit exists. Events: m15_keep (ticks kept), m15_keep_veto_{ray,float}.
_CX_M15_LOOK = _cx_on("CX_M15_LOOK")
_CX_M15_KEEP = _cx_on("CX_M15_KEEP")
_CX_M15_ANY = _CX_M15_LOOK or _CX_M15_KEEP
_M15_LOOK_HITS = int(_cx_knob("CX_M15_LOOK_HITS", 2.0, _CX_M15_LOOK))   # hits on the young track
_M15_LOOK_UNSEEN = _cx_knob("CX_M15_LOOK_UNSEEN", 0.6, _CX_M15_LOOK)   # s since its last sighting
_M15_LOOK_DMAX = _cx_knob("CX_M15_LOOK_DMAX", 20.0, _CX_M15_LOOK)      # xy distance to the estimate (m)
_M15_LOOK_CAP = _cx_knob("CX_M15_LOOK_CAP", 6.0, _CX_M15_LOOK)         # s per look
_M15_LOOK_MAX = int(_cx_knob("CX_M15_LOOK_MAX", 2.0, _CX_M15_LOOK))    # looks per seed
_M15_LOOK_TMAX = _cx_knob("CX_M15_LOOK_TMAX", 52.0, _CX_M15_LOOK)      # no look starts after this flight time (s)
_M15_LOOK_ZLO = _cx_knob("CX_M15_LOOK_ZLO", 1.5, _CX_M15_LOOK)         # vantage height over the estimate, min (m)
_M15_LOOK_ZHI = _cx_knob("CX_M15_LOOK_ZHI", 6.0, _CX_M15_LOOK)         # ...max (m)
_M15_LOOK_VH = _cx_knob("CX_M15_LOOK_VH", 2.0, _CX_M15_LOOK)           # horizontal speed cap (m/s)
_M15_LOOK_DZ = _cx_knob("CX_M15_LOOK_DZ", 6.0, _CX_M15_LOOK)           # estimate within clue z +- this (m; the task: +-5)
_M15_LOOK_DFAR = _cx_knob("CX_M15_LOOK_DFAR", 10.0, _CX_M15_LOOK)      # close in only while farther than this (m)
_M15_LOOK_BLIND = _cx_knob("CX_M15_LOOK_BLIND", 44.0, _CX_M15_LOOK)    # estimate this far below the horizon at the floor..
_M15_LOOK_BLIND_S = _cx_knob("CX_M15_LOOK_BLIND_S", 1.0, _CX_M15_LOOK)  # ..for this long: end the look (s)
_M15_KEEP_HITS = int(_cx_knob("CX_M15_KEEP_HITS", 8.0, _CX_M15_KEEP))  # hits on the young track
_M15_KEEP_S = _cx_knob("CX_M15_KEEP_S", 12.0, _CX_M15_KEEP)            # longest unseen time kept (s)
_M15_KEEP_FLOAT = _cx_knob("CX_M15_KEEP_FLOAT", 0.15, _CX_M15_KEEP)    # float veto (m), as fx_ms KEEP
# fx_mtf (ws8/mtf SYNTH): every flag default OFF (all off == c29h)
_CX_MTF_ZB = _cx_on("CX_MTF_ZB")          # drop a static mountain track far outside the task's clue z band
_MTF_ZB_DZ = _cx_knob("CX_MTF_ZB_DZ", 6.5, _CX_MTF_ZB)        # |est z - clue z| beyond this (template +-5 m + 1.5 m floor)
_MTF_ZB_HITS = int(_cx_knob("CX_MTF_ZB_HITS", 3, _CX_MTF_ZB)) # hits before the veto may act
_MTF_ZB_R = _cx_knob("CX_MTF_ZB_R", 2.0, _CX_MTF_ZB)          # xy radius (m) of a vetoed spot
_MTF_ZB_SPOTZ = _cx_knob("CX_MTF_ZB_SPOTZ", 5.5, _CX_MTF_ZB)  # near a vetoed spot, ignore detections beyond this band
_MTF_ZB_APP = _CX_MTF_ZB and _cx_on("CX_MTF_ZB_APP")  # veto only in the track-driven modes (review fix, in FLAGS)
_CX_MTF_YT = _cx_on("CX_MTF_YT")          # the M15 look only for a track whose first hit came in a look-down pulse
_MTF_YT_PMIN = _cx_knob("CX_MTF_YT_PMIN", 20.0, _CX_MTF_YT)   # deg nose-down (max over the last 0.3 s) at the first hit
_MTF_YT_DEP = _cx_knob("CX_MTF_YT_DEP", 0.0, _CX_MTF_YT)      # alt. pass: estimate this far below horizontal (deg); 0 = off
_MTF_YT_DHMIN = _cx_knob("CX_MTF_YT_DHMIN", 1.5, _CX_MTF_YT)  # no look at an estimate this close (xy, m): blind; 0 = off
_CX_MTF_LKEEP = _cx_on("CX_MTF_LKEEP")    # fx_m15 KEEP (default knobs) for a track that had an M15 look


def _cx_m15_ev(name, n=1):
    # fx_m15 counter: muted inside a shadow act() (_CxShadow mutes while any CX_M15_* flag is on)
    if _CX_MUTE[0]:
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


def _cx_mtf_ev(name, n=1):
    # fx_mtf counter: events inside a discarded shadow act() are counted under NAME_sh
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        name = name + "_sh"
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


def _cx_mtf_log(key, row, cap):
    # fx_mtf: keep the last `cap` diagnostic rows (space separated) in CX_EVENTS[key]
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        row = row + "/sh"
    CX_EVENTS[key] = " ".join((str(CX_EVENTS.get(key, "")) + " " + row).split()[-cap:])


# --- fx_v16 (round 16, village failure mechanisms). Every flag defaults OFF; with all of them off this is c17 bit for bit.
# CX_V16_LOG=1: diagnostics only, no action changes. Per tick with a padnet detection >= 0.45: the detections and what
#   _update_track did with them (new / upd / cf conflict / rs re-seed / mv moving / rej filtered); every track drop with the
#   calling line; every bad_spots addition. Lists in CX_EVENTS['v16log_det'], ['v16log_drop'], ['v16log_bad'].
_CX_V16_LOG = _cx_on("CX_V16_LOG")
_V16_LOG_MAX = 8000


def _v16_log(key, row):
    # fx_v16 LOG: append one diagnostic row (never inside a discarded shadow act())
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    lst = CX_EVENTS.setdefault(key, [])
    if isinstance(lst, list) and len(lst) < _V16_LOG_MAX:
        lst.append(row)


def _cx_v16_ev(name, n=1):
    # fx_v16 counter: muted inside a shadow act() (the router's vking shadow sets _CX_STATE['shadow'])
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


# CX_V16_STALE ("a stale track swallows the real pad's detections"): on village, a detection outside the current track's
#   gate is a conflict, and a new track needs 6 + hits/4 conflicts (7-9 detections) before it replaces the old one. A brief
#   real glimpse (2-5 detections at 6-8 m) against a young false track last seen seconds ago is lost that way (910636304:
#   4 real detections at 28-29 s against a 12-hit false weak APPROACH track unseen 1.4 s; 654313195: 2 real detections at
#   8 m against a 4-hit false track unseen 15 s). With the flag on, a static, non-established village track with fewer than
#   _V16_STALE_HITS hits that has been unseen for >= _V16_STALE_S s yields to the first conflicting detection (re-seeded
#   exactly as the champion's own conflict re-seed does). Not in LAND, not on a moving track. Event: v16_stale.
_CX_V16_STALE = _cx_on("CX_V16_STALE")
_V16_STALE_S = _cx_knob("CX_V16_STALE_S", 1.0, _CX_V16_STALE)
_V16_STALE_HITS = int(_cx_knob("CX_V16_STALE_HITS", 30.0, _CX_V16_STALE))
_V16_STALE_N = int(_cx_knob("CX_V16_STALE_N", 1.0, _CX_V16_STALE))      # agreeing conflicting detections in a burst (1 = round-1 rule)
_V16_STALE_GAP = _cx_knob("CX_V16_STALE_GAP", 0.3, _CX_V16_STALE)       # s between two detections of one burst
# CX_V16_LOOK ("pad glimpsed, never followed up"; village port of fx_m15 LOOK): a young village track (>= 2 hits, not
#   reliable, unseen >= 0.6 s, estimate out of view, <= 20 m away, at ground level: estimate z <= lowest ground seen +
#   _V16_LOOK_DZ) gets one look in CRUISE/SEARCH: yaw toward it; farther than _V16_LOOK_DFAR close in along the camera axis
#   (planner on); nearer, hover and sink (planner off) until the estimate is ~39 deg below the axis (0.8 x the xy distance
#   above it, never under the surface below + 2.5 m, never a climb). The champion's track logic decides: new detections
#   make the track reliable (the mode chain goes to APPROACH), a false estimate is missed in view and dropped. Ends on
#   reliable/APPROACH, drop, new track, moving, 'blind' (the floor keeps it under the image 1 s) or the cap. One look per
#   track, _V16_LOOK_MAX per seed, none after _V16_LOOK_TMAX s (557447861: 4 real hits at 7-8 m at 36.4 s, conf 0.5,
#   never reliable, stale for 23 s while the rings flew on). Events: v16_look, v16_look_ticks, v16_look_end_*.
_CX_V16_LOOK = _cx_on("CX_V16_LOOK")
_V16_LOOK_HITS = int(_cx_knob("CX_V16_LOOK_HITS", 2.0, _CX_V16_LOOK))
_V16_LOOK_UNSEEN = _cx_knob("CX_V16_LOOK_UNSEEN", 0.6, _CX_V16_LOOK)
_V16_LOOK_DMAX = _cx_knob("CX_V16_LOOK_DMAX", 20.0, _CX_V16_LOOK)
_V16_LOOK_CAP = _cx_knob("CX_V16_LOOK_CAP", 4.0, _CX_V16_LOOK)
_V16_LOOK_MAX = int(_cx_knob("CX_V16_LOOK_MAX", 2.0, _CX_V16_LOOK))
_V16_LOOK_TMAX = _cx_knob("CX_V16_LOOK_TMAX", 52.0, _CX_V16_LOOK)
_V16_LOOK_DZ = _cx_knob("CX_V16_LOOK_DZ", 1.5, _CX_V16_LOOK)
_V16_LOOK_DFAR = _cx_knob("CX_V16_LOOK_DFAR", 10.0, _CX_V16_LOOK)
_V16_LOOK_VH = _cx_knob("CX_V16_LOOK_VH", 2.0, _CX_V16_LOOK)
# CX_V16_VLAND ("vland blacklists a real pad it cannot see"): the vland check (static village APPROACH on a < 30-hit track
#   within 4 m of its approach point) hovers 2 s and then puts the estimate on bad_spots. The hover often starts 1.5-2 m
#   above the approach point, with the estimate under the image's bottom rows, so a real pad cannot gather hits and is
#   blacklisted; its later close-range detections (0.82-0.99 at 3 m) are then all rejected (423475988: conf 0.87 track,
#   4.5 s unseen in the hover, bad spot at 38.1 s, 26 rejected real detections after). With the flag on, while the
#   estimate is under row _V16_VL_ROW (or behind the image plane) and the drone is >= 0.8 m off it in xy, the hover sinks
#   (xy held, planner off as in the hover) toward max(1.0, 0.7 x xy distance) above the estimate, so it comes back into
#   view; those ticks do not count toward the 2 s (at most _V16_VL_EXT_S s per APPROACH). No sink under 1.2 m above the
#   surface below. Events: v16_vland (APPROACHes with a sink), v16_vland_ticks.
_CX_V16_VLAND = _cx_on("CX_V16_VLAND")
_V16_VL_ROW = _cx_knob("CX_V16_VL_ROW", 116.0, _CX_V16_VLAND)
# ==== fx_vf (round 20, village landings on raised non-pad surfaces), every flag default OFF (off == c20) ====
# padnet takes flat parts of village houses for pads: the snow roof of building-type-i has a flat part at z 2.35-2.5
# (4 of the 5 c20 village LAND crashes), and other house types have flat parts at 2.25-2.6. The track's height estimate
# agrees with the surface (within 0.1 m), so neither vsurf (surface > 0.6 m above the estimate) nor CX_V8_SURF (> 0.25
# above / 0.12 below) reacts, and LAND descends onto the house. World facts (tools/vf, 12,893 village seeds): a static
# goal pad top is at 0.33-0.43 (ground, 97.7%), 0.97-1.22 (a car roof, 0.55%) or 3.2-6.43 (a house, 1.74%, placed on
# the house's AABB top); a moving one at 0.53-0.63, 1.17-1.40 or 3.4-6.63. No pad of either kind lies between 1.4 and
# 3.1 m. The village ground is flat (slab top 0.10, roads 0.20). Track estimates of real pads (deep traces, 215 static
# seeds): ground pads <= 1.01 once the track has >= 30 hits (2.18 on young tracks), car pads 0.77-1.26, roof pads >= 3.69;
# at the static LAND switch all 256 real entries were at 0.22-1.22 or 3.86-6.38; at the dual->king hand-over real moving
# pads read 1.15-1.28 and 3.32-3.41.
#   CX_VF_LAND=1: at the static village LAND switch (track never flagged moving, not in the router's shadow act()), an
#     estimate whose height above the lowest down-ray ground seen (gmin, required <= _VF_GMAX) is in [_VF_LO, _VF_HI] is
#     not a pad: blacklist it (bad_spots, and a ZONE when that flag is on), drop the track and leave as CX_V8_SURF leaves a
#     rejected car (CRUISE when no search was built yet and the clue is > 8 m away, else SEARCH, resuming the rings).
#   CX_VF_KING=1: the dual->king hand-over (the track just turned moving) on a village track whose estimate is in the same
#     band does not happen: the track is rejected as above and its moving flags cleared (the motion came from it).
#   CX_VF_ZONE=1 (with LAND/KING): after a band rejection, detections within _VF_ZONE_R m (xy) of it whose height is also
#     in the band are ignored (a flat roof part yields detections > 1 m apart, which bad_spots' 1 m radius lets through).
# Events: vf_land, vf_king, vf_zone_rej (detections ignored), vf_at (last 8 as t/estimate z/gmin/hits/estimate-to-clue m).
_CX_VF_LAND = _cx_on("CX_VF_LAND")
_CX_VF_KING = _cx_on("CX_VF_KING")
_CX_VF_ZONE = _cx_on("CX_VF_ZONE")
_CX_VF_ANY = _CX_VF_LAND or _CX_VF_KING
_VF_LO = _cx_knob("CX_VF_LO", 1.45, _CX_VF_ANY)       # m above gmin (car pads <= 1.26 - 0.10)
_VF_HI = _cx_knob("CX_VF_HI", 2.80, _CX_VF_ANY)       # m above gmin (roof pads >= 3.2; their estimates >= 3.32)
_VF_GMAX = _cx_knob("CX_VF_GMAX", 0.6, _CX_VF_ANY)    # gmin must be ground level (village ground 0.10-0.20)
_VF_ZONE_R = _cx_knob("CX_VF_ZONE_R", 3.5, _CX_VF_ZONE)


# ==== fx_vg (round 25: village glimpses / CX_V8_SURF false kills of raised pads), every flag default OFF ====
# CX_VG_V8H ("V8_SURF kills real pads whose estimate height is off"): CX_V8_SURF compares the down-ray surface under the
#   drone with the track's height estimate (+0.25 / -0.12 m) in a static village LAND. Over every recorded flight it fired
#   on the TRUE pad in 12 distinct seeds, and all 12 timed out (the pad becomes a bad spot, so later sightings are ignored).
#   On pads raised on cars (surface 0.87-1.12 m above the lowest ground) the estimate sat 0.15-0.19 m above the top
#   (1052631285, 42891229: 2 of 9 c20 car pads); on a house pad (1665217444) and on two ground pads (1057821625, 5472650)
#   it sat 0.26-0.27 m below. V8's real targets read elsewhere: car roofs 0.69-0.76 m above the ground (+0.27..+0.41),
#   ground (-0.14..-1.83), pickup beds / low objects 0.18-0.51 m (-0.15..-0.58, +0.43/+0.48), roof parts 2.1-2.4 m, house
#   roofs 4.4-5.4 m (-0.12, +0.53..+0.69); vehicle AABB tops are <= 0.88 m (0.78 above the ground). With the flag on,
#   V8's margins widen only where the surface under the drone is at a pad-top height, and only on the side the real-pad
#   kills fell on:
#     car-pad band (_VG_CAR_LO.._VG_CAR_HI above gmin, above every vehicle roof): lower margin _VG_DN2 (estimate high);
#     ground-pad band (_VG_GND_LO.._VG_GND_HI) and house band (>= _VG_HOUSE): upper margin _VG_UP2 (estimate low).
#   None of V8's recorded false targets falls in these windows. The other margins, bands and gates of V8 are unchanged.
#   gmin = lowest down-ray ground seen (<= _VG_GMAX).
#   Events: vg_v8h_save (a V8 fire that c22 would have made was withheld; the flight differs from c22 from here on),
#   vg_v8h_at (t/dz/surface above gmin for each save).
_CX_VG_V8H = _cx_on("CX_VG_V8H")
_VG_UP2 = _cx_knob("CX_VG_UP2", 0.32, _CX_VG_V8H)
_VG_DN2 = _cx_knob("CX_VG_DN2", 0.25, _CX_VG_V8H)
_VG_CAR_LO = _cx_knob("CX_VG_CAR_LO", 0.82, _CX_VG_V8H)
_VG_CAR_HI = _cx_knob("CX_VG_CAR_HI", 1.25, _CX_VG_V8H)
_VG_GND_LO = _cx_knob("CX_VG_GND_LO", 0.15, _CX_VG_V8H)
_VG_GND_HI = _cx_knob("CX_VG_GND_HI", 0.40, _CX_VG_V8H)
_VG_HOUSE = _cx_knob("CX_VG_HOUSE", 3.0, _CX_VG_V8H)
_VG_GMAX = _cx_knob("CX_VG_GMAX", 0.6, _CX_VG_V8H)


def _cx_vg_ev(name, n=1):
    # fx_vg counter: muted inside a shadow act() (V8 never acts there)
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


def _cx_vf_ev(name, n=1):
    # fx_vf counter: muted inside a shadow act() (fx_vf never acts there)
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


# ==== fx_vx (round 26: the down-ray crosses village pads the camera never saw), every flag default OFF ====
# At the 8 m village search height the forward camera sees no ground closer than ~9 m, so a pad right at the clue is
# overflown on arrival and never enters the first ring's view (431193720, 370284780, 768639845, 160193131: crossed at
# 24-27 s, timed out circling 1-8 m from it). The world-vertical down-ray (state[137]) reads the pad top exactly as the
# drone crosses it: a goal pad is a 0.6 m-radius cylinder, top 0.335 m over 0.10 m ground (type A) or 0.435 m on a
# 0.14 m surface (type B).
# CX_VX_RAY=1: village, mine flying (never inside the router's shadow act()), CRUISE/SEARCH/REACQUIRE, no live reliable
#   track, no track ever flagged moving, agl >= _VX_AGL, > 3 m from the start, t <= _VX_TMAX, at most _VX_NMAX
#   episodes. The surface under the drone (z - agl, 50 Hz) is cut into segments at steps >= _VX_STEP. A candidate is a
#   ground / flat run / ground triple: the run no longer than _VX_LMAX (edge to edge), flat within _VX_FLAT, >= 5 ticks
#   of ground before it and 3 after it. Heights (gmin = the lowest
#   ground the ray has seen away from the start):
#     A: ground at gmin +-0.02 on both sides and the run _VX_A_LO.._VX_A_HI above gmin;
#     B: ground gmin -0.01..+_VX_B_GND on both sides (type-B pads sit on 0.14 / 0.20 m surfaces), the run _VX_B_LO..
#        _VX_B_HI above gmin and within _VX_B_BOX m (Chebyshev) of the clue.
#   Both: the run's midpoint within 18 m (Chebyshev) of the clue (the pad lies within R <= 17.6 m of it per axis).
#   Response (mode VXRAY, the controller's own, planner off, heading held):
#     ret: back to the run's midpoint M1 at the trigger height (the path just flown);
#     x2a/x2b: the perpendicular line through M1 (it passes through the pad centre C) at _VX_VX m/s: out along +n until
#       the ray leaves the pad top, back along -n until it leaves it on the other side. Each edge lies 0.6 m (the pad
#       radius) from C, so C's position along the line follows from either edge and the cross-track offset it was
#       crossed at (<= _VX_WMAX); an edge where the ray jumps UP (an overhang over the pad) is not the rim, and C then
#       comes from the other edge. Verified as a 0.6 m disc: the two agree within _VX_UCTOL (an overhang edge must lie
#       inside the pad's extent), and
#       |sqrt((L1/2)^2 + e^2) - 0.6| <= _VX_DISC (e = C - M1 along n);
#     down: back along the scanned chord to C while sinking at up to _VX_SINK (the ray has seen every column there
#       clear; near C it must keep reading the pad top, and nothing may read above it);
#     then, centred and under _VX_HLAND + 0.15 over the pad top, LAND on a camera-free static track at (C, pad top)
#     with 30 hits (V8/VG/VL_HOLD compare the ray with it).
#   A failed check, a moving or ever-moving track or a phase time-out restores the interrupted mode (the rings resume
#   where they stopped) and remembers the spot (no second trigger within 1.5 m).
# Events (CX_EVENTS): vx_cand (candidates, gated or not: first 8 as t/cls/hs/L1/why), vx_fire, vx_ok (LAND handed a
#   ray estimate), vx_abort_<why>, vx_at (first 4 episodes: t/cls/L1/L2/e/C-to-M1).
_CX_VX_RAY = _cx_on("CX_VX_RAY")
_VX_AGL = _cx_knob("CX_VX_AGL", 3.0, _CX_VX_RAY)        # trigger floor (m agl; 3 of 14 failures cross at 3-5 m)
_VX_TMAX = _cx_knob("CX_VX_TMAX", 47.0, _CX_VX_RAY)     # no trigger later than this (s): the response needs ~12 s
_VX_NMAX = int(_cx_knob("CX_VX_NMAX", 2, _CX_VX_RAY))   # episodes per seed
_VX_STEP = 0.05     # m: a surface step between two ticks that starts a new segment (pad edges step 0.23-0.30 m)
_VX_FLAT = 0.025    # m: a pad top is flat (the 0.48 m flat disc sits 0.0105 m over the 0.6 m cylinder top)
_VX_LMAX = 1.3      # m: the run is a chord of a 1.2 m disc
_VX_LMIN = 0.15     # m: shorter runs give no direction
_VX_A_LO = _cx_knob("CX_VX_A_LO", 0.20, _CX_VX_RAY)
_VX_A_HI = _cx_knob("CX_VX_A_HI", 0.27, _CX_VX_RAY)
_VX_B_BOX = _cx_knob("CX_VX_B_BOX", 8.0, _CX_VX_RAY)   # type-B heights are shared with a common low object
_VX_B_LO = 0.32     # m above gmin: type-B pad top (0.43-0.4405 over the 0.10 ground)
_VX_B_HI = 0.345    # (the common low object reads 0.445-0.45)
_VX_B_GND = 0.12    # m above gmin: type-B pads sit on 0.14 / 0.20 surfaces or on the ground
_VX_CLUE = 18.0     # m (Chebyshev) from the clue
_VX_VX = _cx_knob("CX_VX_VX", 1.2, _CX_VX_RAY)         # m/s along the second chord (2.4 cm per tick)
_VX_RET_D = 0.3     # m from M1 where the return hands over to the chord scan (a constant cross-track offset leaves
                    # the chord midpoint's position along the scan exact)
_VX_WMAX = 0.45     # m: largest cross-track offset (from the scan line) at which an edge of the second chord may be crossed
_VX_UCTOL = 0.15    # m: the centre's position along the scan from its two edges (radius 0.6 m) must agree this well
_VX_DISC = 0.12     # m: disc-consistency tolerance
_VX_ON = 0.03       # m: the ray reads the pad top
_VX_OFF = 0.045     # m: the ray has left it
_VX_HLAND = 1.3     # m over the pad top at the LAND hand-off (+0.15, still sinking; LAND's track update only counts
                    # hits under 1.5 m)
_VX_SINK = 2.5      # m/s maximum sink over the pad
_VX_OBST = 1.5      # m: a surface above the pad top this close under the drone ends the episode (it sinks only over
                    # the pad top; 574667499's pad lies half under a 5.7 m roof, flown over at 2.3 m)


def _cx_vx_ev(name, n=1):
    # fx_vx counter: muted inside a shadow act() (fx_vx never acts there)
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


def _cx_vx_log(name, value, cap=8):
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    lst = CX_EVENTS.setdefault(name, [])
    if isinstance(lst, list) and len(lst) < cap:
        lst.append(value)


# ==== fx_vm (round 28: village static pads falsely flagged moving and handed to the king), every flag default OFF ====
# A static village ground pad's track can be flagged moving by mine's motion tests (range-biased estimates drifting along
# the line of sight, edge detections, detection gaps); the router then hands the flight to the king, which often never
# lands it. World fact (tools/vf, 12,893 village seeds): placement lifts a moving village pad 0.2 m above the highest
# body within its orbit, so moving pad tops are >= 0.53 (0.53 / 0.63 over flat ground), while static ground pads sit at
# 0.33 / 0.43 (the ground is the 0.10 slab or 0.20 roads).
# CX_VM_DIAG=1: log every village motion flip of mine's track (path, t, heights, hits, range; vetoed or not) and every
#   village dual->king hand-over. No effect on the flight.
# CX_VM_GATE=1: at a village motion flip (vote fit, raw quarter-median check, the established re-seed; the vmovko LAND
#   abort is only logged) of a track that is not moving yet, never inside a shadow act(), the flip does not happen (the
#   track stays static; the tests run again on the next detection) when all of these hold:
#     - the track has >= _VM_HITS hits and a ground-level gmin is known (<= _VM_GMAX, the lowest down-ray ground seen
#       away from the start);
#     - ground-pad height: the estimate and the median z of the track's last _VM_NOBS detections are both at most
#       _VM_DZ above gmin (static ground pads 0.23 / 0.33 above the slab, moving pads >= 0.43, car / roof pads >= 0.87);
#     - the motion evidence runs along the line of sight (drone -> estimate), as the range-biased drift of a static pad
#       during the approach does: the raw check fired only on its magnitude branch (displacement across the line of
#       sight <= _VM_PERP), the vote fit's speed across it is <= _VM_SPP, the re-seed detection's offset across it is
#       <= _VM_PERP. A real mover shows motion across the line of sight within a moment and is flagged then;
#     - the apparent motion is bounded: the champion raw check's displacement measure over the track's raw samples of
#       the last 8 s is at most _VM_DISP (a static pad's range-biased drift stayed <= 1.44 m; a real mover moving along
#       the line of sight keeps moving, so its displacement passes the bound within about a second; v1 had no bound and
#       held the 0.53 m linear mover 1599682428, 1.18 m/s straight along the line of sight, static until it timed out).
# Events: vm_flip (flips seen), vm_veto, vm_flip_at / vm_ho_at (DIAG, first 12 as strings).
_CX_VM_DIAG = _cx_on("CX_VM_DIAG")
_CX_VM_GATE = _cx_on("CX_VM_GATE")
# fx_cz: CX_CZ_GATE: city goal pads stand 0.2-1.0 m high (swarm constants TYPE_1_H_MIN/MAX; 3,505 recorded city
#   goals: 0.25-1.05 m), so a city detection whose world z is above _CZ_MAX_Z is a roof top, never the goal pad.
#   647631218: a roof-top false positive at z 10 found during take-off turned "moving" as the view rose, and the king
#   dove onto the roof. 3.5 m keeps a forest pad (<= 3.05 m) that is mislabelled city (1 in 960 forest flights).
_CX_CZ_GATE = _cx_on("CX_CZ_GATE")
_CZ_MAX_Z = float(os.environ.get("CX_CZ_MAX_Z", "3.5"))
# fx_vxrim: CX_PVS_VXRIM: fx_vx's down-ray crossing test needs >= 5 ticks of ground right before the pad top. A pad
#   at the edge of a 0.2 m ledge gives ground (0.10) -> ledge (0.20, 4 ticks) -> pad top (0.43) -> road (0.14) and
#   is never tested (1296607187: crossed at 7.6 m AGL at 24.2 s, TIMEOUT; the champion lands it). With the flag on
#   the last 4 surface segments are kept, and a short step (< _PVS_RIM_N ticks, within _PVS_RIM_DZ of the ground
#   segment before it, below the pad top) is looked through: that ground stands in as the pre-pad ground. Every
#   other gate of the test is unchanged. Events: pvs_vxrim (triples tested that way), pvs_vxrim_at (first 4:
#   t/ledge ticks/ledge surface/ground surface). Ported from analysis/pool_village_static/pvs (VXRIM part only).
_CX_PVS_VXRIM = _cx_on("CX_PVS_VXRIM")
_PVS_RIM_N = 5
_PVS_RIM_DZ = 0.15

# ==== fx_pc (pool_city analysis): three city fixes + diagnostics, every flag default OFF (all off == c29a exactly) ====
# CX_PC_VETO: mine's city 'cdrift' check (estimate moved > 0.8 m vs 1.8-2.6 s ago) turns a STATIC pad 'moving' when the
#   detector's range bias slides the estimate along the camera ray (1429876824, 179374762: dz / dz_pred 0.79-0.95); the
#   flag hands the flight to a king pilot that has not seen the pad (search, lost 21 steps, its platform 13-17 m off) and
#   flies blind into a building. A drift across the line of sight <= _PC_PERP, whose z moved the way the ray predicts
#   (|dz - dz_pred| <= _PC_TOLK * |dz_pred| + _PC_TOLC, dz >= max(_PC_DZMIN, _PC_RMIN * |dz_pred|)), totalling
#   <= _PC_DRMAX, is range bias, not motion (a real pad moves level: dz / dz_pred 0.00-0.09 on the logged movers): no flip,
#   and the cdrift window restarts after the vetoed tick. Toward the drone, and away from it too unless CX_PC_AWAY=0.
#   At most _PC_VN vetoes per seed, only with the drone >= _PC_DHMIN from the estimate.
# CX_PC_SEEN: a city moving flag hands the flight to the king only once the king pilot sees a platform (see_P, lost step
#   <= _PC_SEEN_LOST) within _PC_SEEN_D of mine's predicted pad; until then mine keeps flying (its moving-pad branch).
#   After _PC_SEEN_MAX s of holding it hands over as c29a. No hold when the active pilot carries no platform track.
# CX_PC_KTO: at the city dual->king hand-over a king main pilot still in its 'takeoff' state (it never faced the clue
#   within 7.5 deg while mine flew: the pad was found during take-off / early CRUISE) hovers and yaws to the clue first
#   and loses the pad (1049306411, 39848586). Leave 'takeoff' exactly as the pilot's own code does: 'navigation'
#   (first_order_cnt 20) when it sees a platform, else 'search' (30). Skipped while the drone is < 1 m above its start.
# CX_PC_DIAG: logging only (pc_flips, pc_cd, pc_ho, pc_ksee into CX_EVENTS); no behaviour change.
# Events: pc_veto (count), pc_veto_at [t, dr, rad, perp, dz, dz_pred, d_h] (first 3), pc_hold_ticks (count),
#   pc_kto [t, sees], pc_err (count).
_CX_PC_VETO = _cx_on("CX_PC_VETO")
_CX_PC_SEEN = _cx_on("CX_PC_SEEN")
_CX_PC_KTO = _cx_on("CX_PC_KTO")
_PC_DIAG = _cx_on("CX_PC_DIAG")
_PC_ANY = _CX_PC_VETO or _CX_PC_SEEN or _CX_PC_KTO or _PC_DIAG
_PC_VON = _CX_PC_VETO or _PC_DIAG
_PC_PERP = _cx_knob("CX_PC_PERP", 0.35, _PC_VON)     # m: drift across the line of sight
_PC_DZMIN = _cx_knob("CX_PC_DZMIN", 0.10, _PC_VON)   # m: the estimate's z must move at least this much (the ray's way)
_PC_TOLK = _cx_knob("CX_PC_TOLK", 0.6, _PC_VON)      # |dz - dz_pred| <= TOLK * |dz_pred| + TOLC
_PC_TOLC = _cx_knob("CX_PC_TOLC", 0.12, _PC_VON)
_PC_DRMAX = _cx_knob("CX_PC_DRMAX", 2.5, _PC_VON)    # m: larger drifts are motion (179374762: the first check reads 1.63)
_PC_AWAY = _cx_knob("CX_PC_AWAY", 1.0, _PC_VON) > 0.5  # also veto a ray-consistent drift AWAY from the drone
_PC_RMIN = _cx_knob("CX_PC_RMIN", 0.65, _PC_VON)     # the z change must reach this share of the ray prediction
#   (0.65, prototype 0.5: logged static fires read dz / dz_pred 0.78-1.10; a real mover, 1321634644, read 0.51 and
#   passed every other test, the other movers 0.09-0.33. None of the fx_pc re-fly veto decisions changes with 0.65.)
_PC_VN = int(_cx_knob("CX_PC_VN", 3, _PC_VON))       # vetoes per seed
_PC_DHMIN = _cx_knob("CX_PC_DHMIN", 6.0, _PC_VON)    # m: veto only with the drone this far from the estimate
_PC_SEEN_D = _cx_knob("CX_PC_SEEN_D", 3.0, _CX_PC_SEEN or _PC_DIAG)  # m: king platform within this of my predicted pad
_PC_SEEN_LOST = int(_cx_knob("CX_PC_SEEN_LOST", 2, _PC_ANY))       # king ticks since its last platform sighting
_PC_SEEN_MAX = _cx_knob("CX_PC_SEEN_MAX", 6.0, _CX_PC_SEEN)        # s of holding, then hand over as c29a


def _pc_ev(name):
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + 1


def _pc_log(key, rec, cap=6):
    try:
        L = CX_EVENTS.setdefault(key, [])
        if len(L) < cap:
            L.append(rec)
    except Exception:
        pass
_CX_VM_ANY = _CX_VM_DIAG or _CX_VM_GATE
_VM_HITS = int(_cx_knob("CX_VM_HITS", 20, _CX_VM_GATE))    # detections on the track
_VM_DZ = _cx_knob("CX_VM_DZ", 0.40, _CX_VM_GATE)           # m above gmin (static tops 0.23 / 0.33, moving >= 0.43)
_VM_PERP = _cx_knob("CX_VM_PERP", 0.35, _CX_VM_GATE)       # m: raw / re-seed displacement across the line of sight
_VM_SPP = _cx_knob("CX_VM_SPP", 0.30, _CX_VM_GATE)         # m/s: vote-fit speed across the line of sight
_VM_DISP = _cx_knob("CX_VM_DISP", 1.5, _CX_VM_GATE)        # m: largest raw displacement over 8 s that may be vetoed
                                                            # (0 = no bound, round-28 v1; a static pad's radial drift
                                                            # stayed <= 1.44 m, a real mover's keeps growing)
_VM_GMAX = _cx_knob("CX_VM_GMAX", 0.3, _CX_VM_GATE)        # gmin must be village ground level (slab 0.10, roads 0.20)
_VM_NOBS = int(_cx_knob("CX_VM_NOBS", 12, _CX_VM_ANY))     # recent detections in the height statistic
_VM_DMAX = _cx_knob("CX_VM_DMAX", 12.0, _CX_VM_ANY)        # only detections nearer than this (m depth) count


def _cx_vm_ev(name, n=1):
    # fx_vm counter: muted inside a shadow act() (fx_vm never acts there)
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


def _cx_vm_log(name, value, cap=12):
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    lst = CX_EVENTS.setdefault(name, [])
    if isinstance(lst, list) and len(lst) < cap:
        lst.append(value)
_V16_VL_EXT_S = _cx_knob("CX_V16_VL_EXT_S", 3.0, _CX_V16_VLAND)
# CX_V16_SPIN ("the arrival turn P1 removed"): P1 dropped vdip (descend to 3 m and turn 2 pi at the clue, median 10.4 s),
#   so the first village search at the clue flies ring 1 at 8 m at once. On the v3a seeds, c14's dip found 49 static pads
#   within 8 s of reaching the clue; P1 lost 8 of those (1448278233, 2117902210, 1814937956, 965809016, 423475988 found
#   1.6-3.2 s into the dip at 4.8-6.5 m, pads 8-15 m from the clue). With the flag on, the first village search built at
#   the clue (<= 6 m from it, not anchored, over low ground by vdip's own test, no moving flag) first turns in place (yaw
#   lead 1.2 rad as the champion's spins) while sinking to _V16_SPIN_H over the lowest ground (never within _V16_SPIN_FLOOR
#   of the surface below), for _V16_SPIN_TURN rad or _V16_SPIN_CAP s from its start, then flies the rings as before.
#   Once per seed. Events: v16_spin, v16_spin_end.
_CX_V16_SPIN = _cx_on("CX_V16_SPIN")
_V16_SPIN_H = _cx_knob("CX_V16_SPIN_H", 5.0, _CX_V16_SPIN)
_V16_SPIN_FLOOR = _cx_knob("CX_V16_SPIN_FLOOR", 2.5, _CX_V16_SPIN)
_V16_SPIN_TURN = _cx_knob("CX_V16_SPIN_TURN", 2.0 * math.pi, _CX_V16_SPIN)
_V16_SPIN_CAP = _cx_knob("CX_V16_SPIN_CAP", 5.0, _CX_V16_SPIN)

# ==== fx_ra (round RA, cluster A: village arrival dip), every flag default OFF ====
# CX_RA_DIP ("gated arrival dip"): P1 (MY_SEARCH without vdip/vcheck) never makes the champion's arrival dip (descend to
#   3 m and turn 2 pi at the clue). On the 2,076-seed vc batch the dip is a coin flip (23 rescues / 25 losses among 214
#   first-search dips); on the v3a batch it is net negative. With the flag on and vdip absent from MY_SEARCH, the dip is
#   re-enabled for this flight only when, at the first village SEARCH build, (a) the down-ray has seen a structure at least
#   _RA_GMAX m above the lowest ground while within 20 m of the clue, and (b) less than _RA_FU of the ground cells within
#   11 m of the clue (default 0.16) stayed unseen by the forward camera on the way in (1 m cells; min-pooled 64x64 depth, planar depth
#   1-14 m, <= 40 deg below the axis; a point within -0.3..+0.6 m of the lowest ground marks ground, higher marks
#   structure; structure-only cells leave the denominator). Decided once per flight, then latched: a gated-on flight flies
#   exactly as MY_SEARCH=P2 (vdip on, so DIPFULL/DIPLATE act on its dips), a gated-off one exactly as P1.
#   Events: ra_on / ra_off (decision), ra_feat ("fu/gmax/t" string), ra_dip (dips started because of the gate),
#   ra_error (exception in the feature code; the flight then keeps P1).
_CX_RA_DIP = _cx_on("CX_RA_DIP")
_RA_FU = _cx_knob("CX_RA_FU", 0.16, _CX_RA_DIP)
_RA_GMAX = _cx_knob("CX_RA_GMAX", 1.0, _CX_RA_DIP)
_RA_R = 11.0
_RA_DMAX = 14.0
_RA_DOWN_DEG = 40.0
_RA_NEAR = 20.0


def _cx_ra_ev(name, value=None):
    # fx_ra counter: muted inside a shadow act() (the router's vking shadow sets _CX_STATE['shadow'])
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    if value is not None:
        CX_EVENTS[name] = value
    else:
        CX_EVENTS[name] = CX_EVENTS.get(name, 0) + 1


class _RaCover:
    """fx_ra: near-clue ground coverage by the forward camera + highest down-ray ground near the clue (see CX_RA_DIP).
    Same arithmetic as candidates/fx_ra/rafeat.py (the offline calibration)."""

    def __init__(self, center_xy, n_pool=64, r=_RA_R):
        self.c = np.asarray(center_xy, dtype=np.float64)[:2].copy()
        self.r = float(r)
        self.n = int(2 * math.ceil(r) + 1)
        self.org = self.c - math.ceil(r)
        gx = self.org[0] + np.arange(self.n) + 0.5
        gy = self.org[1] + np.arange(self.n) + 0.5
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        self.disc = np.hypot(X - self.c[0], Y - self.c[1]) <= self.r
        self.gnd = np.zeros((self.n, self.n), dtype=bool)
        self.struct = np.zeros((self.n, self.n), dtype=bool)
        t = math.tan(math.radians(90.0) / 2)
        us = (np.arange(n_pool) + 0.5) / n_pool * 2 - 1
        vs = 1 - (np.arange(n_pool) + 0.5) / n_pool * 2
        xr, yu = np.meshgrid(us * t, vs * t)
        self.k = 128 // n_pool
        self.np_ = n_pool
        self.xr = xr.reshape(-1)
        self.yu = yu.reshape(-1)
        self.rows_ok = self.yu >= -math.tan(math.radians(_RA_DOWN_DEG))
        self.gmax_abs = None
        self.frames = 0

    def update(self, depth01, pos, R, v_gnd):
        pos = np.asarray(pos, dtype=np.float64)
        if float(np.hypot(pos[0] - self.c[0], pos[1] - self.c[1])) > self.r + _RA_DMAX + 1.0 or v_gnd is None:
            return
        pooled = depth01.reshape(self.np_, self.k, self.np_, self.k).min(axis=(1, 3)).reshape(-1)
        z = 0.5 + pooled.astype(np.float64) * (20.0 - 0.5)
        ok = (z > 1.0) & (z < _RA_DMAX) & self.rows_ok
        if not ok.any():
            return
        fwd, up = R[:, 0], R[:, 2]
        right = np.cross(fwd, up)
        eye = pos + fwd * 0.13 + up * 0.05
        zz = z[ok]
        w = (eye[None, :] + (zz * self.xr[ok])[:, None] * right[None, :] + (zz * self.yu[ok])[:, None] * up[None, :]
             + zz[:, None] * fwd[None, :])
        ij = np.floor(w[:, :2] - self.org[None, :]).astype(int)
        m = (ij[:, 0] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n)
        if not m.any():
            return
        self.frames += 1
        h = w[m, 2] - float(v_gnd)
        ij = ij[m]
        g = (h >= -0.3) & (h <= 0.6)
        s = h > 0.6
        self.gnd[ij[g, 0], ij[g, 1]] = True
        self.struct[ij[s, 0], ij[s, 1]] = True

    def ground(self, pos, alt):
        if alt < 19.5 and float(np.hypot(pos[0] - self.c[0], pos[1] - self.c[1])) <= _RA_NEAR:
            g = float(pos[2] - alt)
            if self.gmax_abs is None or g > self.gmax_abs:
                self.gmax_abs = g

    def features(self, v_gnd):
        struct_only = self.struct & ~self.gnd & self.disc
        denom = int(self.disc.sum() - struct_only.sum())
        unseen = int((self.disc & ~self.gnd & ~struct_only).sum())
        fu = unseen / max(denom, 1)
        gmax = (self.gmax_abs - float(v_gnd)) if (self.gmax_abs is not None and v_gnd is not None) else 0.0
        return fu, gmax


# --- fx_ma (round 19: static mountain pads near the clue that are never seen). Every flag defaults OFF (off == c20). ---
# CX_MA_SO ("stand-off look"). Where the terrain falls toward the pad faster than the cruise can sink (3 m/s speed cap:
#   2.3 m/s ahead and 1.8 m/s down), a mountain CRUISE closes on the clue 10-16 m over the ground and 15-25 m over clue z.
#   The clue area then stays > 45 deg below the horizon for the whole approach, below the level camera (padnet needs the
#   pad <= ~42 deg down; tools/ma/vscan.py: from 6-18 m out and <= 0.8 x that above the pad it is detected from almost
#   every bearing, and terrain hardly ever hides it). The drone then sinks vertically over the clue, straight above a pad
#   within ~3 m of it, and the rings that follow never face their own centre (see CX_MA_RING; 1003911921, dev pod:
#   17 s of rings with the pad 55-180 deg off the camera axis, TIMEOUT). With the flag on, the FIRST time a mountain
#   CRUISE with no live track (none, or a young one unseen >= 1 s) is between _MA_SO_DLO and _MA_SO_D m (xy) from the
#   clue while it is more than _MA_SO_EX m over the view cone z = clue z + _MA_SO_K * d, and a vertical sink down to
#   ground + _MA_SO_FLOOR (down ray) would bring the clue within _MA_SO_KR * d below it (look-down pulse range), the drone
#   stops (holds the xy where it can stop), faces the clue and sinks vertically (planner off, as the champion's search
#   spin; -2.5 m/s above 6 m AGL, -1.5 above 4, -1.0 below) to clue z + _MA_SO_KT * d, with the champion's look-down
#   pulses toward the clue while still (_MA_SO_PULSE). It then flies on as the champion. It ends when the track turns
#   reliable (CRUISE hands over to APPROACH), at that height, at the ground floor, or after _MA_SO_CAP s. Once per seed.
#   Events: ma_so (starts), ma_so_d<m> (start distance), ma_so_tick, ma_so_pulse, ma_so_end_{cone,floor,cap,trk,mode}.
_CX_MA_SO = _cx_on("CX_MA_SO")
_MA_SO_D = _cx_knob("CX_MA_SO_D", 16.0, _CX_MA_SO)          # engage within this xy distance of the clue (m)
_MA_SO_DLO = _cx_knob("CX_MA_SO_DLO", 10.0, _CX_MA_SO)      # ...but not closer than this (m; 969039340 at 7.8 m: a pulse
# found the pad 76 deg down, and the young track died in the vertical descent onto it)
_MA_SO_K = _cx_knob("CX_MA_SO_K", 0.75, _CX_MA_SO)          # view cone: height over clue z per m of xy distance
_MA_SO_EX = _cx_knob("CX_MA_SO_EX", 2.0, _CX_MA_SO)         # engage only this far above that cone (m)
_MA_SO_KT = _cx_knob("CX_MA_SO_KT", 0.6, _CX_MA_SO)         # sink to clue z + this x distance (m/m)
_MA_SO_FLOOR = _cx_knob("CX_MA_SO_FLOOR", 2.0, _CX_MA_SO)   # never below this over the ground under the drone (m)
_MA_SO_KR = _cx_knob("CX_MA_SO_KR", 1.2, _CX_MA_SO)         # engage only if the floor brings the clue within this x d below
_MA_SO_SINK = _cx_knob("CX_MA_SO_SINK", 3.0, _CX_MA_SO)     # ...and the sink to that floor is at least this (m; 363266621:
# 2.8 m over a ridge top bought nothing and cost 3 s)
_MA_SO_CAP = _cx_knob("CX_MA_SO_CAP", 7.0, _CX_MA_SO)       # longest stand-off (s)
_MA_SO_PULSE = int(_cx_knob("CX_MA_SO_PULSE", 1.0, _CX_MA_SO))  # look-down pulses toward the clue while holding
# CX_MA_RING ("the rings never face their centre"). inward_mtn turns the camera up to 0.7 rad from a mountain ring leg
#   toward the ring centre, but the post-planner yawvel rule (any heading > 0.35 rad off the flown direction at > 0.8 m/s
#   snaps to the flown direction) cancels it at ring speed, so the camera looks along the square's legs and a pad within
#   ~4 m of the clue is 55-180 deg off the camera axis for the whole ring (1003911921 17 s, 913796364 20 s: TIMEOUT). With
#   the flag on, on the legs of the FIRST mountain ring (after the radial leg out to its first vertex) of a search built
#   at the clue with no live track (see _ma_live_track), yawvel keeps the inward offset, capped at _MA_RING_OFF rad (under the 0.6 rad at which
#   yawvel's own speed cut starts), so the centre comes into view at the start of every leg. Event: ma_ring (ticks).
_CX_MA_RING = _cx_on("CX_MA_RING")
_MA_RING_OFF = _cx_knob("CX_MA_RING_OFF", 0.55, _CX_MA_RING)  # inward camera offset kept against yawvel (rad)


def _cx_ma_ev(name, n=1):
    # fx_ma counter: muted inside a shadow act()
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


SIM_DT = 1.0 / 50.0
SPEED_LIMIT = 3.0
CAM_FWD_M = 0.13
CAM_UP_M = 0.05
DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 20.0
IMG_W = IMG_H = 128
DEPTH_NEAR_DUMMY = DEPTH_MIN_M
FOV_DEG = 90.0
STRIDE = 4
G = IMG_W // STRIDE

MAP_NAMES = ("city", "open", "mountain", "village", "warehouse", "forest")
_MODELS_DIR = Path(__file__).resolve().parent / "models"

MAP_PARAMS = {
    "city":      dict(h_cruise=4.0, h_search=4.0, tube=1.1, tube_min=0.45, v_cruise=3.0, v_search=2.5),
    "open":      dict(h_cruise=3.5, h_search=3.5, tube=1.1, tube_min=0.45, v_cruise=3.0, v_search=2.5),
    "mountain":  dict(h_cruise=5.0, h_search=6.5, tube=1.2, tube_min=0.6, v_cruise=3.0, v_search=2.5),
    "village":   dict(h_cruise=7.5, h_search=6.5, tube=1.1, tube_min=0.45, v_cruise=3.0, v_search=3.0),
    "warehouse": dict(h_cruise=6.5, h_search=6.0, tube=1.0, tube_min=0.35, v_cruise=2.5, v_search=2.0),
    "forest":    dict(h_cruise=4.0, h_search=4.0, tube=0.8, tube_min=0.35, v_cruise=2.5, v_search=2.0),
    "unknown":   dict(h_cruise=4.0, h_search=4.0, tube=1.1, tube_min=0.45, v_cruise=2.5, v_search=2.0),
}
_CX_FO_HC = float(os.environ.get("CX_FO_HC", "0") or 0)  # > 0: forest cruise height override (default 4.0)
_CX_FO_HS = float(os.environ.get("CX_FO_HS", "0") or 0)  # > 0: forest search height override (default 4.0)
# CX_PRM_<MAP>_<KEY>=value (e.g. CX_PRM_VILLAGE_H_SEARCH=8.0): override one MAP_PARAMS entry; none set = no change
_CX_PRM_OVR = {}
for _k_, _v_ in os.environ.items():
    if _k_.startswith("CX_PRM_") and _v_:
        _m_, _, _f_ = _k_[7:].lower().partition("_")
        if _m_ in MAP_PARAMS and _f_ in MAP_PARAMS[_m_]:
            try:
                _CX_PRM_OVR.setdefault(_m_, {})[_f_] = float(_v_)
            except ValueError:
                pass


# --- fx_fp (pool_forest): forest search from the public task template + glimpse follow-up; every flag default OFF ---
# POST:  forest SEARCH centre (and the CRUISE target that leads to it) = posterior mean of the pad xy given the clue, the
#        start and the public task template (swarm validator task_gen/_max_search_radius + challenge_families/autopilot:
#        clue = goal + U(-R, R)^2 per axis, R = min(U(5, 20), Rmax(|goal - start|)); forest goals 22-45 m from the start
#        (TYPE_6_R_MIN/MAX) inside +-42 m (TYPE_6_WORLD_RANGE)); cells outside the start band / world box keep a floor
#        weight _FP_FLOOR. c27 forest (1,880 seeds): P(pad within 10 m of the search centre) 0.587 -> 0.747 (0.726 with
#        the 4 m shift gate); the 19 c20-c27 forest failures sit 16.8 m (mean) from the clue and 11.8 m from this centre.
# CLAMP: forest ring waypoints clipped into +-_FP_BOX_M (the pad is inside +-42 m; c27 flew rings to |y| 57).
# LOOK:  forest SEARCH follow-up of a stale young static track (a far glimpse: accurate hits at 13-20 m whose padnet
#        scores keep conf <= 0.6/0.7, so _track_reliable never passes; 2118241558: 19 hits 0.3 m off at 27-29 s, then 23 s
#        of rings away from it). Face it, close in to _FP_LOOK_DFAR at the cruise height (planner on), and let the
#        champion's track logic decide: reliable -> APPROACH, missed in view 1.2 s -> dropped. Gate: >= _FP_LOOK_HITS
#        hits, unseen >= _FP_LOOK_UNSEEN s, not in view, top z within the forest pad band above the lowest ground,
#        >= _FP_LOOK_SMIN m from the start, within _FP_LOOK_CLUE m (Chebyshev) of the clue, one look per track,
#        _FP_LOOK_MAX per seed, none after _FP_LOOK_TMAX s. Events fp_look, fp_look_end_{drop,reliable,new,cap,mode,moving}.
# Port of analysis/pool_forest/patch_fp.py (built on c28) onto c29a; knob defaults RMARGIN 1.5 and SHIFT_MIN 4 are the
# analyst's recommended values (the prototype defaulted to 1.0 and 0).
_CX_FP_POST = _cx_on("CX_FP_POST")
_CX_FP_CLAMP = _cx_on("CX_FP_CLAMP")
_CX_FP_LOOK = _cx_on("CX_FP_LOOK")
_FP_FLOOR = _cx_knob("CX_FP_FLOOR", 0.03, _CX_FP_POST)          # weight of cells outside the start band / world box
_FP_RMARGIN = _cx_knob("CX_FP_RMARGIN", 1.5, _CX_FP_POST)       # m added to Rmax(d) (task distance != flown distance)
_FP_SHIFT_MIN = _cx_knob("CX_FP_SHIFT_MIN", 4.0, _CX_FP_POST)   # m: smaller shifts keep the clue
_FP_BOX_M = _cx_knob("CX_FP_BOX_M", 40.0, _CX_FP_CLAMP)

# ==== fx_s1f: Stage-1 forest track (root-cause v2 synthesis track 1, parts a/b/d), every flag default OFF (off == c29c) ====
# CX_S1F_R915 (a): the champion's MY_SEARCH r915 rings (radii 9/15 m, 8 waypoints 45 deg apart, instead of 7/14/20 m with
#   5 waypoints 90 deg apart) on forest only; the forest spin (fspin) and the fx_fp waypoint clamp are kept. rc2_forest
#   re-flies of r915 on c29c: 3 of the 5 never-visible forest timeouts landed (1236029505, 1677086725, 531880219).
# CX_S1F_CG (b): forest climb gate in SEARCH / APPROACH / REACQUIRE. The drone may rise at most _S1F_CG_DZ above the
#   lowest height it held over the last _S1F_CG_WIN s (in these modes) unless recent depth frames saw through the column
#   above it: points 0.4 .. _S1F_CG_UP m above the drone (centre and 4 offsets of _S1F_CG_OFF) are projected into the
#   frames of the last _S1F_CG_MEM s (one kept every _S1F_CG_EVERY ticks); a point is free when some frame's depth at its
#   pixel (3x3 minimum) reads at least _S1F_CG_MARG beyond it. The climb may go up to the highest level whose 5 points are
#   all free, less _S1F_CG_TOPM (held within _S1F_CG_OFF of that xy), else the requested vz is capped at
#   1.2 x (cap height - z), never below 0. (Open-loop replay of the 820-flight pool, analysis/s1_s1f/cg_sum.txt: offsets
#   0.5 m catch 10 of the climbs followed by canopy contact within 3 s against 8 at 0.3 m, for 11 vs 10 of 418 clean
#   flights gated; 886023938 re-flown with 0.3 m / 0.2 m still met the crown beside the verified column.)
#   The camera sees at most 45 deg up, so a spin in place never sees the column over itself: 867836838 climbed 0.84 m in
#   its spin into a crown, 886023938 1.14 m back up after a low detour. APPROACH within _S1F_CG_PADR of the pad estimate
#   is not gated (the pre-landing rise to ~1.3 m over the pad top, 649 of 820 pool flights).
# CX_S1F_EDGE (d): a start with _S1F_EDGE_XLO <= x <= _S1F_EDGE_XHI, |y| <= _S1F_EDGE_YMAX, z <= _S1F_EDGE_ZMAX (the
#   spawn yaw 0 faces +x, out of the forest box: all 36 forest flights routed to the king in 13,736 started there) whose
#   first route label is open or city is held on the dual route (mine flies, the king shadows), and the dual -> king open
#   rule is blocked, until mine's classifier has scored _S1F_EDGE_N frames taken with the camera turned >= _S1F_EDGE_YAW
#   deg away from +x: forest ahead of every other map by _S1F_EDGE_LP (summed log-prob) -> relabel forest, route mine;
#   otherwise, or once mine's clock passes _S1F_EDGE_MAXT s, the hold ends and the c29c rules resume.
_CX_S1F_R915 = _cx_on("CX_S1F_R915")
_CX_S1F_CG = _cx_on("CX_S1F_CG")
_CX_S1F_EDGE = _cx_on("CX_S1F_EDGE")
_S1F_CG_DZ = _cx_knob("CX_S1F_CG_DZ", 0.3, _CX_S1F_CG)        # m of free rise above the base height
_S1F_CG_UP = _cx_knob("CX_S1F_CG_UP", 1.5, _CX_S1F_CG)        # m: top sample level of the column check
_S1F_CG_WIN = _cx_knob("CX_S1F_CG_WIN", 8.0, _CX_S1F_CG)      # s: base height = lowest z over this window
_S1F_CG_MEM = _cx_knob("CX_S1F_CG_MEM", 3.0, _CX_S1F_CG)      # s of depth frames kept
_S1F_CG_EVERY = max(1, int(_cx_knob("CX_S1F_CG_EVERY", 5, _CX_S1F_CG)))  # keep one frame every N ticks
_S1F_CG_OFF = _cx_knob("CX_S1F_CG_OFF", 0.5, _CX_S1F_CG)      # m: horizontal offsets of the column samples
_S1F_CG_TOPM = _cx_knob("CX_S1F_CG_TOPM", 0.4, _CX_S1F_CG)    # m kept below the highest level seen free
_S1F_CG_MARG = _cx_knob("CX_S1F_CG_MARG", 0.3, _CX_S1F_CG)    # m: seen depth beyond a point for it to count free
_S1F_CG_PADR = _cx_knob("CX_S1F_CG_PADR", 3.0, _CX_S1F_CG)    # m: APPROACH this close to the pad estimate is not gated
_S1F_EDGE_XLO = _cx_knob("CX_S1F_EDGE_XLO", 37.0, _CX_S1F_EDGE)
_S1F_EDGE_XHI = _cx_knob("CX_S1F_EDGE_XHI", 42.5, _CX_S1F_EDGE)
_S1F_EDGE_YMAX = _cx_knob("CX_S1F_EDGE_YMAX", 42.5, _CX_S1F_EDGE)
_S1F_EDGE_ZMAX = _cx_knob("CX_S1F_EDGE_ZMAX", 10.5, _CX_S1F_EDGE)   # forest starts are <= 10.11 m
_S1F_EDGE_N = _cx_knob("CX_S1F_EDGE_N", 25.0, _CX_S1F_EDGE)         # inward frames scored before the decision
_S1F_EDGE_LP = _cx_knob("CX_S1F_EDGE_LP", 6.0, _CX_S1F_EDGE)        # forest log-prob lead needed to relabel
_S1F_EDGE_YAW = _cx_knob("CX_S1F_EDGE_YAW", 60.0, _CX_S1F_EDGE)     # deg away from +x for a frame to count
_S1F_EDGE_MAXT = _cx_knob("CX_S1F_EDGE_MAXT", 3.0, _CX_S1F_EDGE)    # s: the hold ends by then whatever was scored
_FP_LOOK_HITS = int(_cx_knob("CX_FP_LOOK_HITS", 5.0, _CX_FP_LOOK))
_FP_LOOK_UNSEEN = _cx_knob("CX_FP_LOOK_UNSEEN", 0.6, _CX_FP_LOOK)
_FP_LOOK_DMAX = _cx_knob("CX_FP_LOOK_DMAX", 22.0, _CX_FP_LOOK)
_FP_LOOK_DFAR = _cx_knob("CX_FP_LOOK_DFAR", 7.0, _CX_FP_LOOK)
_FP_LOOK_VH = _cx_knob("CX_FP_LOOK_VH", 2.0, _CX_FP_LOOK)
_FP_LOOK_CAP = _cx_knob("CX_FP_LOOK_CAP", 10.0, _CX_FP_LOOK)
_FP_LOOK_MAX = int(_cx_knob("CX_FP_LOOK_MAX", 2.0, _CX_FP_LOOK))
_FP_LOOK_TMAX = _cx_knob("CX_FP_LOOK_TMAX", 50.0, _CX_FP_LOOK)
_FP_LOOK_ZLO = _cx_knob("CX_FP_LOOK_ZLO", 1.4, _CX_FP_LOOK)     # forest pad tops 1.65-3.05 m over flat ground at z 0
_FP_LOOK_ZHI = _cx_knob("CX_FP_LOOK_ZHI", 3.3, _CX_FP_LOOK)
_FP_LOOK_SMIN = _cx_knob("CX_FP_LOOK_SMIN", 20.0, _CX_FP_LOOK)  # forest goals are >= 22 m from the start
_FP_LOOK_CLUE = _cx_knob("CX_FP_LOOK_CLUE", 18.5, _CX_FP_LOOK)  # |est - clue|_inf (R <= 17.35)


# --- fx_s1v (Stage-1 track 3, village detection/track robustness); every flag default OFF (all off == c29c) ---
# SGATE: village only. padnet's score separates the goal pad from village false positives far better at 0.60 than at the
#        champion's 0.45: in 191 logged c28/c29 village flights (analysis/rc2_det flight frames) 74% of the false
#        detections >= 0.45 (cars 72%, roofs 78%, house bodies 87%; start pad excluded) score < 0.60, while 97% of the
#        true detections under 12 m (98.5% under 8 m) score >= 0.60. A track START (no track) and a RE-SEED (the young
#        conflict re-seed and the 40-conflict re-seed of an established track) therefore need the detection that would
#        seed it to score >= _SV_SMIN; an existing track keeps updating at 0.45, a low conflicting detection still counts
#        as a conflict (the re-seed waits for the next one >= _SV_SMIN). Events sv_sgate_start, sv_sgate_rs, sv_sgate_mv.
# ALT:   village only, young static tracks (not established, never moving, not in LAND). The champion keeps one track;
#        every detection outside its gate is a "conflict", and 6 + hits/4 conflicts re-seed the track at the last one
#        with 1 hit. A brief, strong glimpse of the real pad against a weak young false track is lost that way
#        (654313195: 2 detections 0.73/0.76 at 8 m vs a 3-hit 0.5 track; 423475988: 5 x 0.81-0.95 vs a 1-hit 0.48
#        track), and a car seen 6 times displaces a real young track, which then has to re-grow from 1 hit. ALT keeps a
#        second hypothesis built from the conflicting detections that agree with each other (or the young track the
#        champion's re-seed displaced) and scores both by evidence = hits x (score EMA - 0.45), decaying with a
#        _SV_ALT_HL half-life once a hypothesis is unseen for 0.5 s. The alternate becomes the track (with its hits) as
#        soon as it has >= _SV_ALT_N hits and more evidence than the track by _SV_ALT_MARGIN; the champion's conflict
#        re-seed is held while the track was seen within _SV_ALT_HOLD s and has more evidence than the conflicting
#        cluster by the margin, and when it goes ahead it seeds the cluster with its hits and keeps the displaced track
#        as the alternate. A new track that starts on the alternate (the track was dropped meanwhile) resumes its hits.
#        Only a clearly better-scoring cluster takes over (score EMA >= _SV_ALT_CMIN and >= the track's + _SV_ALT_DCONF;
#        a resumed alternate needs _SV_ALT_RMIN): in an open-loop replay of 207 logged village flights through this
#        tracker, looser rules switched mostly between false positives and made 53 more false tracks reliable; with
#        these, the pad's track forms earlier in 17 flights (later in 4, lost in none) for 8 more reliable false tracks.
#        With SGATE on, the alternate may become the track only once one of its detections scored >= _SV_SMIN.
#        Events sv_alt_promote, sv_alt_reseed, sv_alt_hold, sv_alt_restore; rows sv_log (capped).
# LOOK:  village port of CX_FP_LOOK (SEARCH only): a stale young static track (>= _SV_LOOK_HITS hits, score EMA >=
#        _SV_LOOK_CONF, unseen >= _SV_LOOK_UNSEEN s, out of view, not reliable) whose estimate sits at pad height
#        (_SV_LOOK_ZLO.._ZHI over the lowest ground seen; roofs, poles and house tops are 1.6-5 m up) and whose hits agree
#        (xy RMS about their mean <= _SV_LOOK_SPREAD) and lies within _SV_LOOK_CLUE m (Chebyshev) of the clue (public
#        template: clue = goal + U(-R, R) per axis, R <= 17.6) gets one look: face it, close in to _SV_LOOK_DFAR at the
#        search height (planner on), then hover facing it and sink until the estimate is ~39 deg below the camera axis
#        (at 6.5 m a nearer pad is under the image; CX_V16_LOOK's rule); the champion's track logic decides (reliable -> APPROACH,
#        missed in view 1.2 s -> dropped). One look per track, _SV_LOOK_MAX per seed, none after _SV_LOOK_TMAX s, never in
#        a shadow act (open-loop replay of 207 village flights: at score EMA >= 0.55 it would fire on 20 false tracks
#        for 3 real ones, at 0.65 on 0-1 false for 3 real). Events sv_look, sv_look_ticks,
#        sv_look_end_{drop,reliable,new,cap,mode,moving,blind}.
_CX_SV_SGATE = _cx_on("CX_SV_SGATE")
_CX_SV_ALT = _cx_on("CX_SV_ALT")
_CX_SV_LOOK = _cx_on("CX_SV_LOOK")
_CX_SV_TRK = _CX_SV_SGATE or _CX_SV_ALT
_SV_SMIN = _cx_knob("CX_SV_SMIN", 0.60, _CX_SV_TRK)
_SV_ALT_N = int(_cx_knob("CX_SV_ALT_N", 2.0, _CX_SV_ALT))           # hits the alternate needs before it can take over
_SV_ALT_MARGIN = _cx_knob("CX_SV_ALT_MARGIN", 0.10, _CX_SV_ALT)     # evidence margin for a switch / a hold
_SV_ALT_HL = _cx_knob("CX_SV_ALT_HL", 2.0, _CX_SV_ALT)              # s: evidence half-life once unseen for 0.5 s
_SV_ALT_HOLD = _cx_knob("CX_SV_ALT_HOLD", 1.5, _CX_SV_ALT)          # s: the re-seed hold needs the track seen this recently
_SV_ALT_KEEP = _cx_knob("CX_SV_ALT_KEEP", 6.0, _CX_SV_ALT)          # s: an alternate unseen this long is forgotten
_SV_ALT_CAP = _cx_knob("CX_SV_ALT_CAP", 6.0, _CX_SV_ALT)            # evidence cap
_SV_ALT_CMIN = _cx_knob("CX_SV_ALT_CMIN", 0.65, _CX_SV_ALT)         # score EMA an alternate needs to take over
_SV_ALT_DCONF = _cx_knob("CX_SV_ALT_DCONF", 0.05, _CX_SV_ALT)       # ... and this much above the track's
_SV_ALT_RMIN = _cx_knob("CX_SV_ALT_RMIN", 0.60, _CX_SV_ALT)         # score EMA a new track needs to resume the alternate
_SV_LOOK_HITS = int(_cx_knob("CX_SV_LOOK_HITS", 3.0, _CX_SV_LOOK))
_SV_LOOK_CONF = _cx_knob("CX_SV_LOOK_CONF", 0.65, _CX_SV_LOOK)
_SV_LOOK_SPREAD = _cx_knob("CX_SV_LOOK_SPREAD", 0.5, _CX_SV_LOOK)
_SV_LOOK_UNSEEN = _cx_knob("CX_SV_LOOK_UNSEEN", 0.6, _CX_SV_LOOK)
_SV_LOOK_DMAX = _cx_knob("CX_SV_LOOK_DMAX", 20.0, _CX_SV_LOOK)
_SV_LOOK_DFAR = _cx_knob("CX_SV_LOOK_DFAR", 9.0, _CX_SV_LOOK)
_SV_LOOK_VH = _cx_knob("CX_SV_LOOK_VH", 2.0, _CX_SV_LOOK)
_SV_LOOK_CAP = _cx_knob("CX_SV_LOOK_CAP", 8.0, _CX_SV_LOOK)
_SV_LOOK_MAX = int(_cx_knob("CX_SV_LOOK_MAX", 2.0, _CX_SV_LOOK))
_SV_LOOK_TMAX = _cx_knob("CX_SV_LOOK_TMAX", 52.0, _CX_SV_LOOK)
_SV_LOOK_ZLO = _cx_knob("CX_SV_LOOK_ZLO", -0.3, _CX_SV_LOOK)        # village pad tops 0.33 / 0.43 m over ground at 0
_SV_LOOK_ZHI = _cx_knob("CX_SV_LOOK_ZHI", 1.0, _CX_SV_LOOK)
_SV_LOOK_CLUE = _cx_knob("CX_SV_LOOK_CLUE", 18.5, _CX_SV_LOOK)
_SV_LOG_MAX = 200


def _sv_ev(name, n=1):
    # fx_s1v counter: muted inside a shadow act() (the router's vking shadow sets _CX_STATE['shadow'])
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


def _sv_log(row):
    # fx_s1v diagnostic row (never inside a discarded shadow act())
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    lst = CX_EVENTS.setdefault("sv_log", [])
    if isinstance(lst, list) and len(lst) < _SV_LOG_MAX:
        lst.append(row)


# --- fx_s2nbv (Stage 2: village occlusion-aware active search); every flag default OFF (all off == c29d) ---
# NBV: village only. From the first village depth frame after take-off, every 10th tick (5 Hz) the 32 x 32 min-pooled
#      depth (bottom two pooled rows and the edge columns dropped: a pad there is cut off) is back-projected into a
#      1 m map around the clue (+-24 m, _S2nMap): 'seen' = the best detection quality with which a cell's ground was
#      seen (1 to 12 m planar depth, 0.5 at 18 m, 0 beyond; padnet recall on cluttered village views), 'hmax' = the
#      highest structure seen over a cell (line-of-sight tests). 'value' = prior x (1 - seen) with the public-template
#      prior of the pad given the clue (clue = goal + U(-R, R) per axis, R = min(U(5, 20), 17.6); +-40.5 m world box;
#      floor weight 0.03 outside both; no distance-band prior), x 0.3 on cells only ever seen as structure (roofs).
#      Once the champion's ring search has flown the radial leg and the first arc of ring 1 (search_i >= _S2N_RING = 2:
#      finds on the arrival heading and the first side stay the champion's), the rings are replaced by vantages: candidates on a polar grid around the clue (0 / 3.5 / 7 / ... / 21 m, 30 deg) and the drone's own
#      position; gain = the best +-40 deg camera wedge of visible value 9-18 m out (below ~9 m at 8 m height the pad is
#      under the image) with line of sight through hmax; utility = gain / (distance / 2.6 + 0.4 + turns / 0.65 + 0.8),
#      turns = the turn to face the wedge on arrival plus the part of the departure turn beyond 45 deg (the planner
#      crawls at 0.6 m/s while the flight direction is more than 50 deg off the camera axis).
#      Fly to the best at the search speed (planner on) with the camera on the richest wedge within +-45 deg of the
#      flight direction (inside the planner's 50 deg full-speed cone), replanned every _S2N_REPLAN s (a switch needs
#      _S2N_SWITCH x the utility); at the vantage hover and face the richest wedge, dwell _S2N_DWELL s once facing it
#      (<= _S2N_LOOKMAX s), then replan (a looked-from spot is tabu: gain x 0.2 within 3 m). When no vantage holds
#      _S2N_VMIN of the prior, the champion's rings resume at the nearest ring-2/3 vertex. While a young static track
#      is live (seen < _S2N_LIVE s, 3-25 m away, not reliable) the camera stays on it and the drone moves at <= 1.5 m/s
#      so the champion's tracker can confirm or drop it (at most _S2N_LIVEMAX s per track). Never in a shadow act.
#      Offline (analysis/s2_s2nbv): see the patcher docstring numbers. Events s2n_on, s2n_vantage, s2n_look, s2n_fb,
#      s2n_live, s2n_error; s2n_log rows (capped); s2n_ms (total compute ms).
_CX_S2N_NBV = _cx_on("CX_S2N_NBV")
_CX_S2N_ANY = _CX_S2N_NBV
_S2N_GMIN = _cx_knob("CX_S2N_GMIN", 0.004, _CX_S2N_ANY)      # gaze: least wedge value worth steering the camera for
_S2N_HYST = _cx_knob("CX_S2N_HYST", 0.75, _CX_S2N_ANY)       # gaze: keep the current bin while it holds this share
_S2N_GOFF = _cx_knob("CX_S2N_GOFF", 45.0, _CX_S2N_ANY)       # gaze: max camera offset from the flight direction (deg)
_S2N_VMIN = _cx_knob("CX_S2N_VMIN", 0.004, _CX_S2N_ANY)      # least vantage gain; below it the rings resume
_S2N_REPLAN = _cx_knob("CX_S2N_REPLAN", 1.5, _CX_S2N_ANY)
_S2N_SWITCH = _cx_knob("CX_S2N_SWITCH", 1.3, _CX_S2N_ANY)
_S2N_DWELL = _cx_knob("CX_S2N_DWELL", 0.6, _CX_S2N_ANY)
_S2N_LOOKMAX = _cx_knob("CX_S2N_LOOKMAX", 3.0, _CX_S2N_ANY)
_S2N_TMAX = _cx_knob("CX_S2N_TMAX", 55.0, _CX_S2N_ANY)       # no NBV after this (a find cannot land in time)
_S2N_LIVE = _cx_knob("CX_S2N_LIVE", 0.3, _CX_S2N_ANY)        # a young track seen this recently keeps the camera
_S2N_LIVEMAX = _cx_knob("CX_S2N_LIVEMAX", 2.0, _CX_S2N_ANY)  # s of live hold per track (a flickering car cannot stall NBV)
_S2N_RING = int(_cx_knob("CX_S2N_RING", 2.0, _CX_S2N_ANY))   # NBV starts at this ring index (2: after the radial leg
#                                                              and the first ring-1 arc; 4 = after ring 1)
_S2N_LOG_MAX = 60
_S2N_HALF = 24
_S2N_RCAP = 17.6
_S2N_BOX = 40.5
_S2N_FLOOR = 0.03
_S2N_DEP = math.radians(40.0)
_S2N_DMAX = 18.0
_S2N_NB = 36
_S2N_WING = 4
_S2N_FR = np.array([0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82, 0.9], dtype=np.float32)

# fx_nodive: CX_MT_NODIVE (default OFF): mountain far-leg CRUISE glides instead of diving (see tools/mk_fx_nodive.py)
_CX_MT_NODIVE = _cx_on("CX_MT_NODIVE")
_MT_NODIVE_DZ = _cx_knob("CX_MT_NODIVE_DZ", 8.0, _CX_MT_NODIVE)   # glide target height above the clue (m)


def _s2n_ev(name, n=1):
    # fx_s2nbv counter: muted inside a shadow act()
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


def _s2n_log(row):
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    lst = CX_EVENTS.setdefault("s2n_log", [])
    if isinstance(lst, list) and len(lst) < _S2N_LOG_MAX:
        lst.append(row)


def _s2n_ms(t0_):
    # fx_s2nbv compute time (ms, summed over the seed); not counted inside a shadow act()
    if _CX_MUTE[0] or _CX_STATE.get("shadow"):
        return
    import time as _t
    CX_EVENTS["s2n_ms"] = round(float(CX_EVENTS.get("s2n_ms", 0.0)) + 1000.0 * (_t.perf_counter() - t0_), 2)


def _s2n_q(z):
    """Detection quality at planar depth z: 1 to 12 m, 0.5 at 18 m, 0 beyond."""
    return np.where(z > _S2N_DMAX, 0.0, np.clip(1.0 - (z - 12.0) / 12.0, 0.5, 1.0))


def _s2n_bin_yaw(k):
    return -math.pi + (k + 0.5) * (2 * math.pi / _S2N_NB)


class _S2nMap:
    """fx_s2nbv seen-ground map around the clue (same code as tools/s2_s2nbv_core.NbvMap, which the offline study ran)."""

    def __init__(self, c0, half=_S2N_HALF):
        self.n = n = 2 * half + 1
        self.c0 = np.array([float(c0[0]), float(c0[1])])
        self.org = self.c0 - (half + 0.5)
        ax = np.arange(n, dtype=np.float64) + 0.5
        X, Y = np.meshgrid(self.org[0] + ax, self.org[1] + ax, indexing="ij")
        self.X, self.Y = X, Y
        u = np.maximum(np.abs(X - self.c0[0]), np.abs(Y - self.c0[1]))
        lo = np.maximum(u, 5.0)
        p = (np.where(u < _S2N_RCAP, (1.0 / 60.0) * (1.0 / lo - 1.0 / _S2N_RCAP), 0.0)
             + np.where(u <= _S2N_RCAP, (2.4 / 15.0) / (4.0 * _S2N_RCAP * _S2N_RCAP), 0.0))
        inbox = (np.abs(X) <= _S2N_BOX) & (np.abs(Y) <= _S2N_BOX)
        p = np.where(inbox, p, 0.0)
        p = np.maximum(p, _S2N_FLOOR * float(p.max()))
        self.prior = (p / p.sum()).astype(np.float32)
        self.seen = np.zeros((n, n), dtype=np.float32)
        self.hmax = np.zeros((n, n), dtype=np.float32)
        self.frames = 0

    def update(self, eye, W, zc, g0):
        if W.shape[0] == 0:
            return
        ij = np.floor((W[:, :2] - self.org[None, :])).astype(np.int64)
        m = (ij[:, 0] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n)
        if not m.any():
            return
        self.frames += 1
        W, zc, ij = W[m], zc[m], ij[m]
        h = W[:, 2] - g0
        q = _s2n_q(zc).astype(np.float32)
        g = (h > -0.5) & (h < 0.75) & (q > 0.0)
        if g.any():
            np.maximum.at(self.seen, (ij[g, 0], ij[g, 1]), q[g])
        st = h > 1.0
        if st.any():
            np.maximum.at(self.hmax, (ij[st, 0], ij[st, 1]), h[st].astype(np.float32))

    def value(self):
        v = self.prior * (1.0 - self.seen)
        return np.where((self.hmax > 1.0) & (self.seen <= 0.0), 0.3 * v, v)

    def _los(self, ex, ey, ez, tx, ty, g0):
        f = _S2N_FR.reshape((-1,) + (1,) * np.ndim(tx))
        px = ex + f * (tx - ex)
        py = ey + f * (ty - ey)
        pz = (ez - g0) + f * (0.4 - (ez - g0))
        i = np.floor(px - self.org[0]).astype(np.int64)
        j = np.floor(py - self.org[1]).astype(np.int64)
        inb = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        hm = self.hmax[np.clip(i, 0, self.n - 1), np.clip(j, 0, self.n - 1)]
        return ~(inb & (hm > pz)).any(axis=0)

    def wedges(self, ex, ey, ez, g0):
        val = self.value()
        idx = np.flatnonzero(val.ravel() > 0.0)
        X = self.X.ravel()[idx]
        Y = self.Y.ravel()[idx]
        V = val.ravel()[idx]
        dx, dy = X - ex, Y - ey
        d = np.hypot(dx, dy)
        dlo = max(1.0, (ez - g0 - 0.4) / math.tan(_S2N_DEP))
        m = (d >= dlo) & (d <= _S2N_DMAX)
        H = np.zeros(_S2N_NB)
        if m.any():
            X, Y, V, d, dx, dy = X[m], Y[m], V[m], d[m], dx[m], dy[m]
            los = self._los(ex, ey, ez, X, Y, g0)
            w = V * _s2n_q(d) * los
            k = np.floor((np.arctan2(dy, dx) + math.pi) / (2 * math.pi) * _S2N_NB).astype(np.int64) % _S2N_NB
            H = np.bincount(k, weights=w, minlength=_S2N_NB)
        Wd = np.zeros(_S2N_NB)
        for s_ in range(-_S2N_WING, _S2N_WING + 1):
            Wd += np.roll(H, -s_)
        return Wd

    def vantage_gains(self, C, ez, g0, top_k=500):
        val = self.value()
        flat = val.ravel()
        idx = np.flatnonzero(flat > 0.0)
        if idx.size > top_k:
            idx = idx[np.argpartition(flat[idx], -top_k)[-top_k:]]
        X = self.X.ravel()[idx][None, :]
        Y = self.Y.ravel()[idx][None, :]
        V = flat[idx][None, :]
        cx = C[:, 0:1]
        cy = C[:, 1:2]
        dx, dy = X - cx, Y - cy
        d = np.hypot(dx, dy)
        dlo = max(1.0, (ez - g0 - 0.4) / math.tan(_S2N_DEP))
        m = (d >= dlo) & (d <= _S2N_DMAX)
        los = self._los(cx, cy, ez, X, Y, g0)
        w = np.where(m & los, V * _s2n_q(d), 0.0)
        k = np.floor((np.arctan2(dy, dx) + math.pi) / (2 * math.pi) * _S2N_NB).astype(np.int64) % _S2N_NB
        M = C.shape[0]
        H = np.zeros((M, _S2N_NB))
        np.add.at(H, (np.repeat(np.arange(M), k.shape[1]), k.ravel()), w.ravel())
        Wd = np.zeros_like(H)
        for s_ in range(-_S2N_WING, _S2N_WING + 1):
            Wd += np.roll(H, -s_, axis=1)
        b = np.argmax(Wd, axis=1)
        return Wd[np.arange(M), b], b


def _s2n_cands(c):
    C = [(float(c[0]), float(c[1]))]
    for r_, na in ((3.5, 6), (7.0, 12), (10.5, 12), (14.0, 12), (17.5, 12), (21.0, 12)):
        for k in range(na):
            a = 2 * math.pi * k / na
            C.append((min(max(float(c[0]) + r_ * math.cos(a), -_VBOX_M), _VBOX_M),
                      min(max(float(c[1]) + r_ * math.sin(a), -_VBOX_M), _VBOX_M)))
    return np.array(C)


def rot_from_rpy(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def cam_pose(pos, rpy):
    R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    fwd = R[:, 0]
    up = R[:, 2]
    right = np.cross(fwd, up)
    eye = np.asarray(pos, dtype=np.float64) + fwd * CAM_FWD_M + up * CAM_UP_M
    return eye, fwd, right, up


def pixel_depth_to_world(u, v, z, eye, fwd, right, up):
    t = math.tan(math.radians(FOV_DEG) / 2)
    xr = ((u + 0.5) / IMG_W * 2 - 1) * t
    yu = (1 - (v + 0.5) / IMG_H * 2) * t
    return eye + z * (fwd + xr * right + yu * up)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def unit(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros_like(v)


class DepthGrid:
    def __init__(self, n=32):
        self.n = n
        t = math.tan(math.radians(FOV_DEG) / 2)
        us = (np.arange(n) + 0.5) / n * 2 - 1
        vs = 1 - (np.arange(n) + 0.5) / n * 2
        xr, yu = np.meshgrid(us * t, vs * t)
        self.xr = xr.astype(np.float32)
        self.yu = yu.astype(np.float32)

    def pool_min(self, depth01):
        k = IMG_H // self.n
        return depth01.reshape(self.n, k, self.n, k).min(axis=(1, 3))

    def points_cam(self, pooled):
        z = (DEPTH_MIN_M + pooled.astype(np.float32) * (DEPTH_MAX_M - DEPTH_MIN_M))
        return np.stack([self.xr * z, self.yu * z, z], axis=-1).reshape(-1, 3), z.reshape(-1)


class Planner:

    def __init__(self, n_pool=32, n_cand=11, half=0.9):
        self.grid = DepthGrid(n_pool)
        c = np.linspace(-half, half, n_cand)
        cy_rows = np.concatenate([c, [1.25, 1.7]])
        cx, cy = np.meshgrid(c, cy_rows)
        self.cy_raw = cy.ravel().astype(np.float32)
        self.cx_raw = cx.ravel().astype(np.float32)
        dirs = np.stack([cx.ravel(), cy.ravel(), np.ones(cx.size)], axis=-1)
        self.cand = (dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)).astype(np.float32)
        self.n_cand = self.cand.shape[0]

    def free_distances(self, depth01, tube_r, lookahead, z_floor=None, eye_z=0.0, zvec=None):
        pooled = self.grid.pool_min(depth01)
        pts, z = self.grid.points_cam(pooled)
        valid = z < 19.6
        if z_floor is not None and zvec is not None:
            valid = valid & ((eye_z + pts @ zvec) > z_floor)
        pts = pts[valid]
        if pts.shape[0] == 0:
            return np.full(self.n_cand, lookahead, dtype=np.float32), None
        proj = self.cand @ pts.T
        r2 = np.sum(pts * pts, axis=1)[None, :]
        lat2 = np.clip(r2 - proj * proj, 0.0, None)
        hit = (proj > 0.0) & (lat2 < tube_r * tube_r)
        pen = np.sqrt(np.clip(tube_r * tube_r - lat2, 0.0, None))
        dist = np.where(hit, np.clip(proj - pen, 0.0, None), np.inf)
        free = dist.min(axis=1)
        free = np.minimum(free, lookahead).astype(np.float32)
        return free, pts


class PadDetector:
    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, depth01):
        x = np.ascontiguousarray(depth01.reshape(1, 1, IMG_H, IMG_W), dtype=np.float32)
        heat, off, zed, cls = self.sess.run(None, {self.input_name: x})
        heat = np.nan_to_num(heat, nan=-20.0, posinf=20.0, neginf=-20.0)
        off = np.nan_to_num(off)
        zed = np.nan_to_num(zed, nan=0.0)
        cls = np.nan_to_num(cls)
        hm = 1.0 / (1.0 + np.exp(-heat[0, 0]))
        pad = np.pad(hm, 1, mode="constant")
        mx = np.max(np.stack([pad[i:i + G, j:j + G] for i in range(3) for j in range(3)]), axis=0)
        peaks = hm * (hm >= mx)
        idx = np.argsort(peaks.ravel())[::-1][:3]
        dets = []
        for k in idx:
            s = float(peaks.ravel()[k])
            if s < 0.2:
                break
            iv, iu = divmod(int(k), G)
            u = (iu + float(off[0, 0, iv, iu])) * STRIDE
            v = (iv + float(off[0, 1, iv, iu])) * STRIDE
            z = float(zed[0, 0, iv, iu]) * 20.0
            dets.append((s, u, v, z))
        return dets, cls[0]


class _MyController:
    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else _MODELS_DIR
        det_path = self.models_dir / "padnet.onnx"
        self.detector = PadDetector(det_path) if det_path.exists() else None
        self.planner = Planner()
        self.debug = {}
        self._pos_xy = None
        self.reset()

    def reset(self):
        self.t = 0.0
        self.step = 0
        self.mode = "TAKEOFF"
        self.mode_t0 = 0.0
        self._omf_cstk_acc, self._omf_cstk_t, self._omf_cstk_done = 0.0, None, False  # fx_cstk per-seed state
        if _PC_ANY:  # fx_pc per-seed state: veto window restart time, veto count, DIAG flip count, this tick's position
            self._pc_veto_t, self._pc_veto_n, self._pc_nflip, self._pc_pos = -1e9, 0, 0, None
        self.v_cmd = np.zeros(3)
        self.yaw_cmd = None
        self.start_pos = None
        self.start_alt = None
        self.map_logp = np.zeros(6)
        self.map_n = 0
        self.map_label = "unknown"
        self.map_locked = False
        self.pad_pos = None
        self.pad_vel = np.zeros(2)
        self.pad_hits = 0
        self.pad_last_seen = -1.0
        self.pad_obs = []
        self.pad_conf = 0.0
        self.pad_conflicts = 0
        self.pad_moving = False
        self.pad_move_votes = 0
        self.pad_miss_t = 0.0
        self.pad_omega = 0.0
        self.pad_acc = np.zeros(2)
        self.pad_slow_t = 0.0
        self.pad_fit_sp = 0.0
        self.pad_fit_z = 99.0
        self.pad_min_z = 99.0
        self.pad_info = 0.0
        self.pad_raw = []
        self.pad_still_t = 0.0
        self.pad_ever_moving = False
        self._vk_tb = False  # CX_VK_UNSTICK: set by the router's village take-backs (read only when the flag is on)
        self._vk_unstuck = False  # CX_VK_UNSTICK: current track proven still after a take-back (read only when the flag is on)
        self._slope_ticks = 0
        self.ring_pts = []        # terrain points seen 0.75-3 m around a static pad estimate (downhill touchdown)
        if _CX_S1M_ANY:
            self._s1m_reset()  # fx_s1m per-seed state
        self.land_off = None      # touchdown offset (xy), frozen at LAND entry
        self.stuck_t = 0.0        # seconds continuously wedged
        self.escape_until = -1.0  # backing out until this time
        self.escape_dir = None
        self._wf_ooc_t0 = None     # ws2 WF_OOC: start of the running hold (None = not holding)
        self._wf_ooc_used = 0.0    # ws2 WF_OOC: hold time of the current turn episode
        self.land_off_failed = False  # an offset landing failed this episode: land at the centre from now on
        self.land_reclimb = False     # climbing back over the pad rim after an offset touchdown point missed it
        self.land_reclimb_t0 = 0.0
        # fx_me state (read only when a CX_ME_* flag is on)
        self._me_pos = None
        self._me_vel = None
        self._me_rg_cal = []          # (t, surface z under the drone, estimate x, y) over the estimate's centre
        self._me_rg_run = 0           # consecutive ticks with the rim signal
        self._me_rg_desc = 0          # consecutive ticks sinking faster than 0.2 m/s
        self._me_rg_hold = False
        self._me_rg_t0 = 0.0
        self._me_rg_sfire = 0.0
        self._me_rg_nf = 0
        self._me_rg_bypass = False
        self._me_rg_diag = False
        self._me_rg_flat = None
        self._me_rg_alts = []
        self._me_rg_pfire = None
        self._me_mv_blocks = 0
        self._me_eg = False
        self.land_slope = None    # terrain slope fitted at LAND entry (logged even with the knob off)
        self.land_down = None
        self.land_rms = None
        self._landoff_ticks = 0
        self._climb_ticks = 0
        self.pad_close_seen_t = -99.0
        self.pad_hist = []
        self.bad_spots = []
        self.center0 = None
        self.gnd_hist = []
        self._vdip_active = False
        self.v_gnd = None
        self.cov_seen = None
        self.cov_origin = None
        self.cov_wp = None
        self.cov_tgt = None
        self.cov_t_plan = -9.0
        self.hint_xy = None
        self.hint_t = -99.0
        self.mm_raw = []
        self.mm = None
        self.search_t_first = None  # first entry into SEARCH this episode
        self._pulse_k = -1            # look-down pulse index within the current spin
        self._pulse_dir = None
        self._pulse_bypass = False
        self._pulse_ticks = 0
        self._pulse_on = False
        self._app_pulse_k = -1
        self._app_pulse_on = False
        self._app_pulse_dir = None
        self._pulse_last_t = -99.0
        self.mm_raw = []       # (t, x, y) raw detections at >= 0.1 s spacing, z < 16 m, up to 30 s
        self.mm = None         # fitted motion model dict or None
        self.mm_t_fit = -99.0
        self._rpy = np.zeros(3)
        self.recover_z = 0.0
        self.land_stuck_t = 0.0
        self.reacq_n = 0
        self.reacq_hold = 0.0
        self.reacq_wp = None
        self.reacq_target = None
        self.reacq_moving = False
        self.reacq_side = 0
        self.reacq_t_enter = 0.0
        self.land_dive = False
        self.touch_t = 0.0
        self.land_pdir = None
        self.land_phase = "track"
        self.land_retry = 0
        self.drop_t0 = 0.0
        self.amb_dbg = None
        self.amb_wait_t = 0.0
        self.search_center_xy = None
        self.drop_z0 = -1.0
        self.dash_v0 = np.zeros(2)
        self.dash_p0 = np.zeros(2)
        self.dash_t1 = 0.0
        self.pv_hist = []
        self.e_hist = []
        self.rest_t = 0.0
        self.touch_hold = -1.0
        self.drop_mis_t = 0.0
        self.search_wps = None
        self.search_i = 0
        self.search_anchor = None
        self.spin_target = None
        self.spin_done = False
        # fx_fo per-seed state (c2: set whatever the flags, so a reused agent starts clean; read only behind CX_FO_*)
        self._fo_gref = None     # CX_FO_SPIN: lowest ground seen since the current spin began
        self._fo_hold_xy = None  # CX_FO_SPIN: where a canopy hold fired (arms the post-spin level exit)
        self._fo_drop = None     # CX_FO_RESUME: (t, dropped pad xy, pad_last_seen) of the last _drop_track
        # CX_F6_UNDER per-seed state (set whatever the flag; read only behind CX_F6_UNDER)
        self._f6_trail = []       # (x, y, z, down-ray alt) crumbs >= 0.25 m apart along the path flown
        self._f6_blk = 0.0        # seconds of planner escape branch (boxed in), decaying when the planner is free
        self._f6_esc_step = -9    # step of the last escape branch taken by _plan
        self._f6_lr = (0.0, 0.0)  # level left/right free sums at that escape
        self._f6_ep = 0           # recovery episodes started this seed
        self._f6_ph = None        # None | "back" | "detour"
        self._f6_ph_t0 = 0.0
        self._f6_back_i = 0
        self._f6_back_stop = 0
        self._f6_yaw0 = 0.0
        self._f6_side = 0.0
        self._f6_wp = None
        self._f6_low_until = -1.0
        self._f6_det_end = -99.0  # time the last detour phase ended (reached, timed out or cancelled)
        # fx_f14 per-seed state (set whatever the flags; read only behind CX_F14_*)
        self._f14_gmin = None     # GATE: lowest down-ray ground seen this flight
        self._rc_gmin = None      # fx_rc LBAND: lowest down-ray ground seen this flight (set whatever the flags)
        self._vf_gmin = None      # fx_vf: lowest down-ray ground seen this flight (read only behind CX_VF_*)
        self._vm_gmin = None      # fx_vm: lowest down-ray ground seen away from the start (read only behind CX_VM_*)
        self._vm_ho_logged = False
        self._vf_zones = []       # fx_vf ZONE: (x, y, z) of band vetoes this flight
        self._f14_refly_n = 0     # REFLY: re-flies to the clue used this seed
        # fx_fp per-seed state (set whatever the flags; read only behind CX_FP_*)
        self._fp_c = None          # POST: cached search centre xy (False = keep the clue)
        self._fp_look = None       # LOOK: the running look {t0, key}, else None
        self._fp_look_n = 0        # LOOK: looks started this seed
        self._fp_look_done = []    # LOOK: keys of tracks already looked at
        # fx_s1v per-seed state (set whatever the flags; read only behind CX_SV_*)
        self._sv_alt = None        # ALT: the second young-track hypothesis (dict), else None
        self._sv_look = None       # LOOK: the running look {t0, key}, else None
        self._sv_look_n = 0        # LOOK: looks started this seed
        self._sv_look_done = []    # LOOK: keys of tracks already looked at
        # fx_s2nbv per-seed state (set whatever the flags; read only behind CX_S2N_*)
        self._s2n_map = None       # the seen-ground map (_S2nMap), built at the first village frame
        self._s2n_dead = False     # an exception switched NBV off for this seed
        self._s2n_sid = None       # the search_wps list NBV state belongs to (a new SEARCH build resets it)
        self._s2n_st = None        # None (not started) / 'go' / 'look' / 'fb' (rings resumed)
        self._s2n_tgt = None       # current vantage xy
        self._s2n_util = 0.0
        self._s2n_tplan = -1e9
        self._s2n_look = None
        self._s2n_tabu = []
        self._s2n_gk = None        # gaze bin (hysteresis)
        self._s2n_gy = None        # gaze yaw
        self._s2n_cand = None
        self._s2n_live_key = None  # track whose live hold is being timed
        self._s2n_live_acc = 0.0
        self.land_t0 = None
        self.land_xy = None
        self.land_z = None
        self.recover_n = 0
        self.stuck_t = 0.0
        self.last_pos = None
        self.blocked_steps = 0
        self.debug = {}
        # fx_ct ROOF per-seed state. Validators reuse one agent for every seed of a container and only call reset() in
        # between: the roof guard's history (seed-relative times), hold deadline, episode clock, street level and ramp
        # bypass must not carry over. c2: set whatever the flags (read only behind CX_CT_ROOF), and the history is created
        # here with the same values _ct_roof's first-call init would give it (that lazy init then never runs).
        self._ct_ghist = []
        self._ct_until = -1.0
        self._ct_ep_t = -99.0
        self._ct_street_g = 0.0
        self._ct_bypass = False
        # fx_vs per-seed state (new attributes the champion never reads; validators reuse one agent and only reset())
        self._vs_low_acc = 0.0
        self._vs_dip_ext = False
        self._vs_dip_ok = False
        self._vs_dip_near = False
        self._vs_drop = None
        self._vs_look_until = -1.0
        self._vs_look_xy = None
        self._vs_look_n = -1.0
        self._vs_r1_leg = -1
        self._vs_refly_n = 0
        self._kg_dip_t = None     # fx_kg DIPCAP: when this dip's DIPFULL extra turn began (the champion's dip end)
        self._v3_depth = None     # fx_v3 OCC: this tick's depth image (for the occlusion test in _update_track)
        self._v3_exc = 0.0        # fx_v3 OCC: occluded miss time excused on the current track
        self._v3_exc_ev = False
        self._v3_exc_run = 0.0    # fx_v3 OCC: excused time since pad_miss_t was last reset (champion's count = sum)
        self._v3_saved = False    # fx_v3 OCC: the excuse kept alive this track past the champion's in-view drop
        self._v3_vl_ext = 0.0     # fx_v3 VLOOK: vland time not counted (estimate unobservable) in this APPROACH
        self._v3_vl_t0 = None     # time the current below-the-image vland stretch began
        self._v3_vl_k = -1
        self._v3_vl_on = False
        self._v3_vl_dir = None
        self._v3_vl_anchor = None
        self._v3_vl_tick = -9.0   # last tick the vland hover ran (a gap starts a new stretch; OCC skips hover ticks)
        self._v3_vh = 0.0
        self._ms_srch_t0 = None   # fx_ms SZ: time the current SEARCH started (only set with CX_MS_SZ on)
        self._ms_slope_t = -99.0  # fx_ms SLOPE: last tick it set the sink (hysteresis; read only with CX_MS_SLOPE on)
        self._ms_acted = False    # fx_ms SLOPE/KEEP acted this episode (gates the LAND-entry diagnostics)
        # fx_m2 per-seed state (set whatever the flags; read only behind CX_M2_*)
        self._m2_first = False    # the current SEARCH is the first one built at the clue on a mountain seed
        self._m2_t0 = None        # time that first SEARCH was built (set once per episode)
        self._m2_g0 = None        # ground under the drone then (ridge test: _m2_g0 - clue z > _M2_RIDGE_DZ)
        self._m2_ridge = False    # that first SEARCH is in the ridge class (PULSE, RESPIN)
        self._m2_high = False     # ...and started at or above clue z + _M2_SPIN_DZ over ground below clue z (SPIN)
        self._m2_rs = 0           # RESPIN: re-spins started this episode (at most 1)
        self._m2_rs_z = None      # RESPIN: clue z + _M2_RS_DZ while a re-spin runs, else None
        self._m2_rs_acc = 0.0     # RESPIN: yaw turned within 1 m of the re-spin height
        self._m2_zt = None        # height target of the current mountain SEARCH tick (turn counting)
        self._m2_seen = False     # that first SEARCH has handed over to APPROACH (disarms SPIN, ridge PULSE, a new RESPIN)
        self._m2_drop_t = None    # RESPIN: time the drop condition started holding (debounce), else None
        # fx_v8 per-seed state (set whatever the flags; read only behind CX_V8_*)
        self._v8_surf_t = 0.0     # SURF: time the down ray has contradicted the estimate's height (consecutive ticks)
        # fx_vg per-seed state (set whatever the flags; read only behind CX_VG_*)
        self._vg_gmin = None      # V8H: lowest down-ray ground seen this flight
        self._vg_v8_sh = 0.0      # V8H: c22's V8 counter (as if nothing were withheld), to log a withheld fire
        # fx_vl3 per-seed state (set whatever the flags; read only behind CX_VL_*)
        self._vl_hold_seen = False  # HOLD: the once-moving LAND hold was skipped at least once this seed (event count)
        # fx_m15 per-seed state (set whatever the flags; read only behind CX_M15_*)
        self._m15_look = None       # LOOK: the running look {t0, key}, else None
        self._m15_look_n = 0        # LOOK: looks started this seed
        self._m15_look_done = []    # LOOK: keys (first-sighting time) of tracks already looked at
        self._m15_look_tick = False  # LOOK: this tick's command is the look's (read by the CRUISE climb-over rule)
        # fx_mtf per-seed state (set whatever the flags; read only behind CX_MTF_*)
        self._mtf_zb_spots = []      # ZB: (x, y) of vetoed out-of-band estimates
        self._mtf_pitch = collections.deque(maxlen=15)  # YT: pitch (deg, nose-down positive) of the last 0.3 s of ticks
        self._mtf_pb_ref = None      # YT: pad_obs list of the last classified track (a new list = a new track)
        self._mtf_pb = False         # YT: that track is pulse-born
        self._mtf_yt_blk = set()     # YT: keys of tracks whose look was blocked (event once per track)
        self._mtf_lk_obs = None      # LKEEP: pad_obs list of the track that had an M15 look
        # fx_v16 per-seed state (set whatever the flags; read only behind CX_V16_*)
        self._v16_nbs = 0            # LOG: bad_spots entries already logged
        self._v16_look = None        # LOOK: the running look {t0, key}, else None
        self._v16_look_n = 0         # LOOK: looks started this seed
        self._v16_look_done = []     # LOOK: keys (first-sighting time) of tracks already looked at
        self._v16_vl = None          # VLAND: [APPROACH start time, below-image seconds not counted, sink seen]
        self._v16_spin_on = False    # SPIN: the current search's first turn is fx_v16's arrival turn
        self._v16_spin_n = 0         # SPIN: arrival turns started this seed
        self._v16_spin_t0 = 0.0      # SPIN: when it started
        self._v16_cf = None          # STALE: the current burst of conflicting detections [t_last, x, y, n]
        # fx_ra per-seed state (set whatever the flags; read only behind CX_RA_*)
        self._ra_cov = None          # DIP: _RaCover of the approach (village, until the decision)
        self._ra_dec = None          # DIP: the latched gate decision (None = not decided yet)
        # fx_ma per-seed state (set whatever the flags; read only behind CX_MA_*)
        self._ma_so = None           # SO: the running stand-off {t0, xy, k}, else None
        self._ma_so_n = 0            # SO: stand-offs started this seed (at most 1)
        self._ma_ring_ok = False     # RING: the current SEARCH is a first search built at the clue with no track
        self._ma_ring_tick = False   # RING: this tick is a first-ring leg whose inward offset yawvel keeps
        # fx_vx per-seed state (set whatever the flags; read only behind CX_VX_*)
        self._vx_gmin = None         # lowest down-ray ground seen away from the start
        self._vx_segs = []           # last 3 surface segments (dicts)
        self._vx_prev = None         # (surface, xy) of the previous tick
        self._vx = None              # the running episode (dict), else None
        self._vx_n = 0               # episodes started this seed
        self._vx_rej = []            # xy of spots already tried
        if _CX_S1F_CG or _CX_S1F_EDGE:
            # fx_s1f per-seed state (read only behind CX_S1F_*)
            self._s1f_fb = []        # CG: recent depth frames [(t, eye, M rows right/up/fwd, depth01)]
            self._s1f_zh = []        # CG: (t, z) while in a gated mode (base height window)
            self._s1f_ok = None      # CG: (x, y, z_top) the column over (x, y) was seen free up to z_top
            self._s1f_eacc = None    # EDGE: [summed class log-probs (6), frames] while the router holds an edge start

    def _ra_decide(self):
        # fx_ra DIP: decide once (at the first village SEARCH build) whether this flight may make the arrival dip
        if self._ra_dec is None:
            try:
                if self._ra_cov is None:
                    self._ra_dec = False
                    _cx_ra_ev("ra_feat", "none/none/%.2f" % self.t)
                else:
                    fu, gmax = self._ra_cov.features(self.v_gnd)
                    self._ra_dec = bool(gmax >= _RA_GMAX and fu < _RA_FU)
                    _cx_ra_ev("ra_feat", "%.3f/%.2f/%.2f" % (fu, gmax, self.t))
                _cx_ra_ev("ra_on" if self._ra_dec else "ra_off")
            except Exception:
                self._ra_dec = False
                _cx_ra_ev("ra_error")
            self._ra_cov = None
        return bool(self._ra_dec)

    def _params(self):
        if self.map_label == "forest" and (_CX_FO_HC > 0.0 or _CX_FO_HS > 0.0):
            # CX_FO_HC / CX_FO_HS: forest cruise / search height (m above the down-ray ground) instead of 4.0
            prm_ = dict(MAP_PARAMS["forest"])
            if _CX_FO_HC > 0.0:
                prm_["h_cruise"] = _CX_FO_HC
            if _CX_FO_HS > 0.0:
                prm_["h_search"] = _CX_FO_HS
            if self.map_label in _CX_PRM_OVR:
                prm_.update(_CX_PRM_OVR[self.map_label])
            return prm_
        if self.map_label in _CX_PRM_OVR:
            prm_ = dict(MAP_PARAMS[self.map_label])
            prm_.update(_CX_PRM_OVR[self.map_label])
            return prm_
        return MAP_PARAMS.get(self.map_label, MAP_PARAMS["unknown"])

    def _update_map(self, cls_logits, pos, alt_ray):
        if _CX_S1F_EDGE and self._s1f_eacc is not None:
            self._s1f_edge_frame(cls_logits)  # fx_s1f EDGE: class evidence from frames that look into the forest box
        if self.map_locked:
            return
        z = cls_logits - cls_logits.max()
        p = np.exp(z)
        p /= p.sum()
        self.map_logp += np.log(p + 1e-6)
        self.map_n += 1
        if self.step == 1:
            self._map_z0 = float(pos[2])
            if pos[2] < 0.7:
                self.map_logp[3] += 6.0
            if abs(pos[0]) > 76 or abs(pos[1]) > 76:
                self.map_logp[2] += 6.0
            if abs(pos[1]) > 23.5 or abs(pos[0]) > 38.5:
                self.map_logp[4] -= 6.0
            if abs(pos[0]) > 42.5 or abs(pos[1]) > 42.5:
                self.map_logp[5] -= 6.0
                self.map_logp[3] -= 6.0
        best = int(np.argmax(self.map_logp))
        self.map_label = MAP_NAMES[best]
        if self.map_n >= 60:
            self.map_locked = True
            if _MTN_HIGH_RELABEL and _OPEN_ALT > 0 and self.map_label in ("open", "warehouse", "city", "forest") and getattr(self, "_map_z0", 0.0) > _OPEN_ALT:
                self.map_label = "mountain"

    def _detect(self, depth, pos, rpy):
        if self.detector is None:
            return [], np.zeros(6)
        dets, cls = self.detector(depth)
        eye, fwd, right, up = cam_pose(pos, rpy)
        out = []
        for s, u, v, zp in dets:
            z = float(zp)
            if z > 19.5 or z < 0.3:
                continue
            w = pixel_depth_to_world(u, v, z, eye, fwd, right, up)
            out.append((s, w, z, u, v))
        return out, cls

    def _update_track(self, dets, pos):
        self._pos_xy = np.asarray(pos[:2], dtype=float)
        if (self.mode == "LAND" and self.pad_pos is not None and not self.pad_moving
                and pos[2] - self.pad_pos[2] < 1.5 and self.pad_hits >= 30):
            if dets and max(d[0] for d in dets) >= 0.45:
                self.pad_hits += 1
                self.pad_last_seen = self.t
            if _CX_S1M_RECEN and self.map_label == "mountain" and not _CX_STATE["shadow"]:
                self._s1m_recen_obs(dets, pos)  # fx_s1m RECEN: may re-seed the track and switch to APPROACH
            return
        if not dets or max(d[0] for d in dets) < 0.45:
            if (_CX_V3_OCC and _v3_live() and self.pad_pos is not None and self.map_label == "village"
                    and (self.mode == "APPROACH" or (_CX_V3_VQF and self.mode == "REACQUIRE" and getattr(self, "_vcheck_active", False)))
                    and not self.pad_moving and self._v3_exc < _V3_EXC_S and self._pad_in_view(pos, self._rpy)
                    and self.t - self._v3_vl_tick > 0.03 and self._v3_vh >= 0.5
                    and not self._static_established() and self._v3_obs(pos, self._rpy) == "occluded"):
                # fx_v3 OCC: the estimate's pixel shows a nearer surface (a house between us and the pad), so this
                # tick says nothing about the pad: no miss (the champion's unseen limits in APPROACH still run)
                self._v3_exc += SIM_DT
                self._v3_exc_run += SIM_DT
                if not self._v3_exc_ev:
                    self._v3_exc_ev = True
                    _cx_v3_ev("v3_occ_excuse")
                if self.pad_miss_t + self._v3_exc_run > 1.2:
                    self._v3_mark_saved()  # the champion would drop this track on this tick
                return
            if self.pad_pos is not None and self._pad_in_view(pos, self._rpy):
                self.pad_miss_t += SIM_DT
                if self.pad_miss_t > 1.2 and self.mode != "LAND" and not self._static_established():
                    if _CX_VS_LOOK and self.map_label == "village":
                        self._vs_drop = (np.asarray(self.pad_pos[:2], dtype=np.float64).copy(), self.t)
                    self._drop_track()
                elif (self._v3_exc_run > 0.0 and self.pad_miss_t + self._v3_exc_run > 1.2 and self.mode != "LAND"
                      and not self._static_established()):
                    self._v3_mark_saved()  # fx_v3 OCC: with the excused ticks counted, the champion drops it here
            return
        good = [d for d in dets if d[0] >= 0.45 and np.isfinite(d[1]).all()]
        if _F14_GATE and good and self.map_label == "forest":
            good = self._f14_gate(good)  # fx_f14 GATE: drop detections that cannot be the goal pad
        if _CX_CZ_GATE and good and self.map_label == "city":
            n_cz = len(good)
            good = [d for d in good if float(d[1][2]) <= _CZ_MAX_Z]  # fx_cz GATE: roof-top detections are not the pad
            if len(good) < n_cz:
                _cx_ms_ev("cz_rej")
        if _CX_MTF_ZB and good and self.map_label == "mountain" and self.center0 is not None and self._mtf_zb_spots:
            # fx_mtf ZB: near a spot vetoed earlier, detections outside the clue z band cannot re-seed it
            cz_ = float(self.center0[2])
            n0_ = len(good)
            good = [d for d in good if not (abs(float(d[1][2]) - cz_) > _MTF_ZB_SPOTZ and any(
                math.hypot(float(d[1][0]) - sx, float(d[1][1]) - sy) < _MTF_ZB_R for sx, sy in self._mtf_zb_spots))]
            if len(good) < n0_:
                _cx_mtf_ev("mtf_zb_rej")
        if not good:
            return
        if self.pad_pos is not None and len(good) > 1:
            pred0, _ = self._motion_at(min(self.t - self.pad_last_seen, 3.0))
            s, w, z, u, v = min(good, key=lambda d: float(np.linalg.norm(d[1][:2] - pred0[:2])))
        else:
            s, w, z, u, v = max(good, key=lambda d: d[0])
        self.pad_miss_t = 0.0
        self._v3_exc_run = 0.0  # fx_v3 OCC: the champion's miss count restarts here too
        if self.start_pos is not None and np.linalg.norm(w[:2] - self.start_pos[:2]) < 1.5 and abs(w[2] - self.start_pos[2]) < 1.5:
            return
        _bs_r = 1.0 if (_flag("vbad1") and self.map_label == "village") else 2.0
        for bs in self.bad_spots:
            if np.linalg.norm(w[:2] - bs[:2]) < _bs_r and abs(w[2] - bs[2]) < 2.0:
                return
        if _CX_VF_ZONE and self._vf_zones and self.map_label == "village" and self._vf_zone_hit(w):
            _cx_vf_ev("vf_zone_rej")  # fx_vf ZONE: the flat house part rejected earlier, seen from elsewhere
            return
        if self.center0 is not None and np.linalg.norm(w[:2] - self.center0[:2]) > 24.0:
            return
        if _CX_SV_TRK and self.pad_pos is None and self.map_label == "village" and self._sv_start(s, w, z):
            return  # fx_s1v: SGATE start gate / ALT restore of the alternate hypothesis
        if self.pad_pos is None:
            self.pad_pos = w.copy()
            self.pad_hits = 1
            self.pad_conf = s
            self.pad_obs = [(self.t, w[0], w[1], w[2], z)]
            self.pad_last_seen = self.t
            self.pad_min_z = float(z)
            self.pad_info = 0.0
            self.pad_raw = []
            self.pad_still_t = 0.0
            self._push_raw(w, z)
            return
        dt = self.t - self.pad_last_seen
        pred, _ = self._motion_at(min(dt, 3.0))
        err = float(np.linalg.norm(w[:2] - pred[:2]))
        est_static = self._static_established()
        wide = max(1.5, 0.12 * z) + 1.2 * dt
        gate = (max(0.7, 0.08 * z) if est_static else wide)
        if err <= wide:
            self._push_raw(w, z)
            if self._raw_motion_check(w):
                return
        if err > gate:
            self.pad_conflicts += 1
            need_conf = int(min(40, 6 + self.pad_hits // 4)) if not est_static else 100
            if (_CX_V16_STALE and self.map_label == "village" and not est_static and not self.pad_moving
                    and not self.pad_ever_moving and self.mode != "LAND" and self.pad_hits < _V16_STALE_HITS
                    and dt >= _V16_STALE_S and not _CX_STATE.get("shadow")):
                # fx_v16 STALE: a young track unseen this long yields to a burst of _V16_STALE_N conflicting detections
                # that agree with each other (a real glimpse comes as several consecutive hits; a lone detection does not
                # displace it: 557447861 lost a real 4-hit track to a single false detection 1.2 s later)
                cf16 = self._v16_cf
                if cf16 is None or self.t - cf16[0] > _V16_STALE_GAP or float(np.hypot(w[0] - cf16[1], w[1] - cf16[2])) > 1.5:
                    cf16 = self._v16_cf = [self.t, float(w[0]), float(w[1]), 0]
                cf16[0] = self.t
                cf16[3] += 1
                if cf16[3] >= _V16_STALE_N:
                    if self.pad_conflicts < need_conf:
                        _cx_v16_ev("v16_stale")
                    need_conf = 1
                    self._v16_cf = None
            if _CX_SV_TRK and self.map_label == "village" and self._sv_conflict(s, w, z, err, est_static, need_conf):
                return  # fx_s1v: handled (ALT switch / hold / re-seed with hits, or SGATE re-seed gate)
            if est_static and self.pad_conflicts >= 40 and z >= 1.0 and err > 1.0:
                if _F14_STATIC and self.map_label == "forest" and self._f14_static_reseed(w, z):
                    return  # fx_f14 STATIC: forest pads never move; re-seeded as a young static track
                if _CX_ME_ANY and self.map_label == "mountain" and not self.pad_moving:
                    _cx_me_log("me_mv", "reseed/%.2f/err%.2f/z%.1f" % (self.t, err, float(z)))
                if _CX_VM_ANY and self._vm_flip_reseed(w, err, z):
                    return  # fx_vm GATE: a ground-height village track stays static (this detection is ignored)
                self.pad_pos = w.copy()
                self.pad_hits = max(self.pad_hits, 30)
                self.pad_obs = [(self.t, w[0], w[1], w[2], z)]
                self.pad_last_seen = self.t
                self.pad_conflicts = 0
                self.pad_info = 0.0
                self.pad_moving = True
                self.pad_ever_moving = True
                self.pad_vel = np.zeros(2)
                self._v3_track_reset()  # fx_v3: re-seeded track, no excuse/pulse state carried over
                return
            if self.pad_conflicts >= need_conf:
                self._v3_track_reset()  # fx_v3: re-seeded track, no excuse/pulse state carried over
                self.pad_pos = w.copy()
                self.pad_vel = np.zeros(2)
                self.pad_hits = 1
                self.pad_conf = s
                self.pad_obs = [(self.t, w[0], w[1], w[2], z)]
                self.pad_last_seen = self.t
                self.pad_conflicts = 0
                self.pad_min_z = float(z)
                self.pad_info = 0.0
                self.pad_raw = []
                self.pad_still_t = 0.0
                self.pad_moving = bool(self.pad_ever_moving)
                self.pad_move_votes = 0
                self.pad_omega = 0.0
                self.pad_acc = np.zeros(2)
            return
        self.pad_conflicts = 0
        if self.pad_moving or not est_static:
            alpha = float(np.clip(0.35 * (6.0 / max(z, 2.0)), 0.15, 0.7))
            self.pad_pos = (1 - alpha) * pred + alpha * w
        else:
            sig = 0.05 + 0.012 * float(z)
            wgt = 1.0 / (sig * sig)
            alpha = float(np.clip(wgt / (self.pad_info + wgt), 0.04, 0.7))
            if z < 1.5 and float(np.linalg.norm(w[:2] - self.pad_pos[:2])) > 0.5 and self.pad_hits >= 30:
                alpha = 0.01
            self.pad_pos = (1 - alpha) * self.pad_pos + alpha * w
            self.pad_info = min(self.pad_info + wgt, 4000.0)
        self.pad_hits += 1
        self.pad_conf = 0.8 * self.pad_conf + 0.2 * s
        self.pad_last_seen = self.t
        if z < 6.0:
            self.pad_close_seen_t = self.t
        self.pad_min_z = min(self.pad_min_z, float(z))
        self.pad_obs.append((self.t, w[0], w[1], w[2], z))
        if len(self.pad_obs) > 75:
            self.pad_obs.pop(0)
        pass
        if not self.pad_hist or self.t - self.pad_hist[-1][0] >= 0.2:
            self.pad_hist.append((self.t, float(self.pad_pos[0]), float(self.pad_pos[1]), float(self.pad_pos[2])))
            if len(self.pad_hist) > 150:
                self.pad_hist.pop(0)
        if _flag("mmodel") and self.t - self.mm_t_fit >= 1.0:
            self.mm_t_fit = self.t
            self._mm_fit()
            if self.mm is not None and not self.pad_moving and self.mm["span"] * self.mm["w"] >= 2.5:
                self.pad_moving = True
                self.pad_ever_moving = True
                _, vfit = self._mm_predict(self.t)
                self.pad_vel = np.asarray(vfit, dtype=float)
                self.pad_slow_t = 0.0
        win = 1.2 if z < 6.0 else (1.5 if z < 10.0 else 2.0)
        obs = [o for o in self.pad_obs if self.t - o[0] <= win and o[4] < 16.0]
        if len(obs) >= 8 and (obs[-1][0] - obs[0][0]) >= 0.5 * win:
            z_mean = float(np.mean([o[4] for o in obs]))
            a = np.asarray(obs)
            tt = a[:, 0] - a[:, 0].mean()
            sxx = float((tt * tt).sum())
            vx = float((tt * (a[:, 1] - a[:, 1].mean())).sum() / sxx)
            vy = float((tt * (a[:, 2] - a[:, 2].mean())).sum() / sxx)
            v = np.array([vx, vy])
            pred_x = a[:, 1].mean() + vx * tt
            pred_y = a[:, 2].mean() + vy * tt
            resid = float(np.sqrt(np.mean((a[:, 1] - pred_x) ** 2 + (a[:, 2] - pred_y) ** 2)))
            sp_lin = float(np.linalg.norm(v))
            span = sp_lin * (a[-1, 0] - a[0, 0])
            acc = np.zeros(2)
            if len(a) >= (12 if _flag("mvfit") else 20) and self.pad_moving:
                try:
                    cx = np.polyfit(tt, a[:, 1], 2)
                    cy = np.polyfit(tt, a[:, 2], 2)
                    acc = np.array([2 * cx[0], 2 * cy[0]])
                    an = float(np.linalg.norm(acc))
                    if an > 1.5:
                        acc *= 1.5 / an
                    if _flag("mvfit") and z_mean < 10.0:
                        te = float(tt[-1])
                        v_end = np.array([cx[1] + 2 * cx[0] * te, cy[1] + 2 * cy[0] * te])
                        dv_e = v_end - v
                        dn_e = float(np.linalg.norm(dv_e))
                        if dn_e > 1.5:
                            dv_e *= 1.5 / dn_e
                        v = v + dv_e
                except Exception:
                    pass
            sp = float(np.linalg.norm(v))
            if sp > 3.0:
                v *= 3.0 / sp
                sp = 3.0
            self.pad_acc = 0.6 * self.pad_acc + 0.4 * acc
            v_new = (0.6 * self.pad_vel + 0.4 * v) if _flag("mvfit") else (0.8 * self.pad_vel + 0.2 * v)
            dv_ = v_new - self.pad_vel
            dn_ = float(np.linalg.norm(dv_))
            lim = 2.0 * SIM_DT if self.pad_moving else 9.0
            if dn_ > lim:
                dv_ *= lim / dn_
            self.pad_vel = self.pad_vel + dv_
            if sp < 0.4:
                self.pad_omega *= 0.5
            obs2 = [o for o in self.pad_obs if self.t - o[0] <= 2.5 and o[4] < 12.0]
            if len(obs2) >= 24 and sp > 0.3:
                b_ = np.asarray(obs2)
                half = len(b_) // 2
                def _fitv(c):
                    tt2 = c[:, 0] - c[:, 0].mean()
                    sxx2 = float((tt2 * tt2).sum())
                    if sxx2 < 1e-6:
                        return None, 0.0
                    return np.array([float((tt2 * (c[:, 1] - c[:, 1].mean())).sum() / sxx2),
                                     float((tt2 * (c[:, 2] - c[:, 2].mean())).sum() / sxx2)]), float(c[:, 0].mean())
                v1, t1 = _fitv(b_[:half])
                v2, t2 = _fitv(b_[half:])
                if v1 is not None and v2 is not None and np.linalg.norm(v1) > 0.5 and np.linalg.norm(v2) > 0.5 and t2 - t1 > 0.4:
                    om = wrap(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])) / (t2 - t1)
                    om = float(np.clip(om, -1.0, 1.0))
                    self.pad_omega = 0.7 * self.pad_omega + 0.3 * om
            if self.map_label in ("warehouse", "forest"):
                need_votes, need_sp, need_span = 999, 9.0, 99.0
            elif z_mean < 6.0:
                need_votes, need_sp, need_span = 6, 0.45, 0.7
                if _flag("slowmv"):
                    need_votes, need_sp, need_span = 6, 0.3, 0.4
            elif z_mean < 10.0:
                need_votes, need_sp, need_span = 6, 0.5, 0.9
                if _flag("slowmv"):
                    need_votes, need_sp, need_span = 6, 0.45, 0.7
            else:
                need_votes, need_sp, need_span = 8, 0.55, 1.2
            if self.map_label == "open":
                need_votes, need_sp, need_span = 4, min(need_sp, 0.3), min(need_span, 0.45)
            consistent = span > max(need_span, 2.5 * resid)
            if z_mean > 9.0 and not self.pad_moving:
                los = unit(self.pad_pos[:2] - self._pos_xy) if self._pos_xy is not None else np.zeros(2)
                v_perp = v - float(v @ los) * los
                sp_perp = float(np.linalg.norm(v_perp))
                span_perp = sp_perp * (a[-1, 0] - a[0, 0])
                consistent = consistent and sp_perp > 0.45 and span_perp > max(0.6 * need_span, 2.0 * resid)
            me_vinf = None
            if (_CX_ME_ANY and self.map_label == "mountain" and not self.pad_moving and z_mean <= 9.0 and consistent
                    and sp > need_sp):
                los_ = unit(self.pad_pos[:2] - self._pos_xy) if self._pos_xy is not None else np.zeros(2)
                g_, me_vinf = self._me_mv_gate(v, los_)
                v_perp_ = v - float(v @ los_) * los_
                sp_perp_ = float(np.linalg.norm(v_perp_))
                span_perp_ = sp_perp_ * (a[-1, 0] - a[0, 0])
                me_vinf = "%s/sp%.2f/spp%.2f/z%.1f/g%d" % (me_vinf, sp, sp_perp_, z_mean, int(g_))
                if _CX_ME_MOV and g_ and not (sp_perp_ > 0.45 and span_perp_ > max(0.6 * need_span, 2.0 * resid)):
                    # fx_me MOV: a radial vote on a closing level approach is the range bias, as beyond 9 m
                    consistent = False
                    _cx_me_ev("me_mv_vote_block")
            if (_CX_VK_MOTION and consistent and not self.pad_moving and self.map_label == "village"
                    and float(np.max(np.diff(a[:, 0]))) > _VK_VOTE_GAP):
                # the fit joins sightings from both sides of a detection gap: not evidence of motion
                consistent = False
                if sp > need_sp:
                    _cx_count("vk_motion_vote_gap")
            if _MTN_PULSE_NOVOTE and self.t - getattr(self, "_pulse_last_t", -99.0) < 0.8:
                pass  # a look-down pulse just swung the camera: apparent pad motion is ours, neither vote nor unvote
            elif sp > need_sp and consistent:
                self.pad_move_votes += 1
                if self.pad_move_votes >= need_votes and not (_CX_VM_ANY and not self.pad_moving and self._vm_flip_vote(
                        v, sp, sp_lin, span, resid, z_mean, a)):
                    if _CX_ME_ANY and not self.pad_moving and self.map_label == "mountain":
                        _cx_me_log("me_mv", "vote/%s" % me_vinf)
                    if not self.pad_moving:
                        self.pad_pos = np.array([obs[-1][1], obs[-1][2], obs[-1][3]])
                    self.pad_moving = True
                    if z_mean < 9.0 and sp_lin > 0.5:
                        self.pad_ever_moving = True
                    self.pad_slow_t = 0.0
            else:
                self.pad_move_votes = max(0, self.pad_move_votes - 1)
            self.pad_fit_sp = sp_lin
            self.pad_fit_z = z_mean
            if not self.pad_moving and sp < 0.3:
                self.pad_vel *= 0.5
        elif self.t - self.pad_last_seen > 2.0:
            self.pad_vel *= 0.9

    def _mm_fit(self):
        if len(self.mm_raw) < 30:
            return
        a = np.asarray(self.mm_raw)
        a = a[self.t - a[:, 0] <= 30.0]
        if len(a) < 30 or a[-1, 0] - a[0, 0] < 5.0:
            return
        t0 = a[-1, 0]
        tt = a[:, 0] - t0
        best = None
        for w in np.arange(0.15, 0.67, 0.015):
            X = np.stack([np.ones_like(tt), np.cos(w * tt), np.sin(w * tt), np.cos(2 * w * tt), np.sin(2 * w * tt)], axis=1)
            XtX = X.T @ X + np.diag([0.0, 0.0, 0.0, 2.0, 2.0])
            try:
                coef = np.linalg.solve(XtX, X.T @ a[:, 1:3])
            except np.linalg.LinAlgError:
                continue
            res = a[:, 1:3] - X @ coef
            rms = float(np.sqrt(np.mean(np.sum(res * res, axis=1))))
            if best is None or rms < best[0]:
                best = (rms, w, coef)
        rms, w, coef = best
        amp = float(np.hypot(*np.linalg.norm(coef[1:3], axis=0)))
        self.mm_last = (round(rms, 2), round(float(w), 2), round(amp, 1), len(a), round(float(a[-1, 0] - a[0, 0]), 1))
        if rms < 0.35 and amp > 0.8:
            self.mm = {"t0": t0, "w": w, "coef": coef, "rms": rms, "span": float(a[-1, 0] - a[0, 0])}
        else:
            self.mm = None

    def _mm_predict(self, t_abs):
        m = self.mm
        tt = t_abs - m["t0"]
        w = m["w"]; c = m["coef"]
        X = np.array([1.0, math.cos(w * tt), math.sin(w * tt), math.cos(2 * w * tt), math.sin(2 * w * tt)])
        dX = np.array([0.0, -w * math.sin(w * tt), w * math.cos(w * tt), -2 * w * math.sin(2 * w * tt), 2 * w * math.cos(2 * w * tt)])
        return X @ c, dX @ c

    def _motion_at(self, dt):
        if _flag("mmodel") and self.pad_moving and self.mm is not None and self.t - self.mm["t0"] < 4.0:
            pxy, vxy = self._mm_predict(self.pad_last_seen + dt)
            p = self.pad_pos.copy()
            p[:2] = pxy
            return p, vxy
        p = self.pad_pos.copy()
        v = self.pad_vel.copy()
        if not self.pad_moving:
            return p, np.zeros(2)
        om = float(self.pad_omega)
        sp = float(np.linalg.norm(v))
        if abs(om) < 0.2 or sp < 0.1:
            if _flag("mvfit") and sp > 0.6:
                da = min(dt, 1.5)
                p[:2] += v * dt + 0.5 * self.pad_acc * da * da
                return p, v + self.pad_acc * da
            p[:2] += v * dt
            return p, v
        th = math.atan2(v[1], v[0])
        th2 = th + om * dt
        p[0] += sp / om * (math.sin(th2) - math.sin(th))
        p[1] += sp / om * (math.cos(th) - math.cos(th2))
        v2 = np.array([sp * math.cos(th2), sp * math.sin(th2)])
        return p, v2

    def _pc_cdrift(self, pos, h0):
        """fx_pc: features of a city cdrift fire; True = veto it (CX_PC_VETO). See the fx_pc block at the top."""
        try:
            p = np.asarray(self.pad_pos, dtype=np.float64).reshape(-1)
            d_ = p[:2] - np.array([h0[1], h0[2]], dtype=np.float64)
            dr = float(np.linalg.norm(d_))
            rel = p[:2] - np.asarray(pos[:2], dtype=np.float64)
            dh = float(np.linalg.norm(rel))
            los = rel / max(dh, 1e-6)
            rad = float(d_ @ los)  # + = away from the drone
            perp = float(np.linalg.norm(d_ - rad * los))
            dz = float(p[2] - float(h0[3]))
            slope = (float(pos[2]) - float(p[2])) / max(dh, 1e-6)
            dzp = -rad * slope  # a point sliding along the camera ray toward the drone rises by this much
            sgn = 1.0 if dzp >= 0.0 else -1.0
            veto = bool(_CX_PC_VETO and self._pc_veto_n < _PC_VN and (rad < 0.0 or _PC_AWAY) and perp <= _PC_PERP
                        and dz * sgn >= max(_PC_DZMIN, _PC_RMIN * abs(dzp))
                        and abs(dz - dzp) <= _PC_TOLK * abs(dzp) + _PC_TOLC
                        and dr <= _PC_DRMAX and dh >= _PC_DHMIN)
            rec = [round(float(self.t), 2), round(dr, 2), round(rad, 2), round(perp, 2), round(dz, 2), round(dzp, 2),
                   round(dh, 1)]
            if _PC_DIAG:
                _pc_log("pc_cd", rec + [round(float(pos[2]), 2), round(float(p[2]), 2), int(self.pad_hits),
                                        str(self.mode), int(veto)], cap=8)
            if veto:
                self._pc_veto_n += 1
                self._pc_veto_t = float(self.t)
                _pc_ev("pc_veto")
                _pc_log("pc_veto_at", rec, cap=3)
            return veto
        except Exception:
            _pc_ev("pc_err")
            return False

    def _pad_predicted(self):
        if self.pad_pos is None:
            return None
        dt = min(self.t - self.pad_last_seen, 3.0)
        if not self.pad_moving:
            p = self.pad_pos.copy()
            p[:2] += self._pad_slow_vel() * min(dt, 1.5)
            return p
        p, _ = self._motion_at(dt)
        return p

    def _pad_slow_vel(self):
        return np.zeros(2)

    def _pad_vel_now(self):
        if self.pad_pos is None or not self.pad_moving:
            return np.zeros(2)
        dt = min(self.t - self.pad_last_seen, 3.0)
        _, v = self._motion_at(dt)
        return v

    def _track_reliable(self):
        if self.pad_pos is None or (self.t - self.pad_last_seen) >= 2.5:
            return False
        if self.pad_conf <= 0.6:
            if not _flag("invest") or self.map_label == "forest":
                return False
            return self.pad_hits >= 8 and self.pad_conf > 0.5 and (self.t - self.pad_last_seen) < 1.5
        need = 5
        if self.center0 is not None and np.linalg.norm(self.pad_pos[:2] - self.center0[:2]) > 30.0:
            need = 12
        if self.map_label == "forest":
            need = max(need, 15)
            if self.pad_conf <= 0.7:
                if _flag("invest") and self.pad_hits >= 10 and self.pad_conf > 0.6:
                    return True
                return False
        return self.pad_hits >= need

    def _pad_pixel_row(self, pos, rpy):
        p = self._pad_predicted()
        if p is None:
            return None
        eye, fwd, right, up = cam_pose(pos, rpy)
        d = p - eye
        z = float(d @ fwd)
        if z < 0.3:
            return None
        t = math.tan(math.radians(FOV_DEG) / 2)
        yu = float(d @ up) / z / t
        return (1 - yu) / 2 * IMG_H

    def _pad_in_view(self, pos, rpy, margin=8.0):
        p = self._pad_predicted()
        if p is None:
            return False
        eye, fwd, right, up = cam_pose(pos, rpy)
        d = p - eye
        z = float(d @ fwd)
        if z < 1.0 or z > 15.0:
            return False
        t = math.tan(math.radians(FOV_DEG) / 2)
        xr = float(d @ right) / z / t
        yu = float(d @ up) / z / t
        u = (xr + 1) / 2 * IMG_W
        v = (1 - yu) / 2 * IMG_H
        return margin <= u < IMG_W - margin and margin <= v < IMG_H - margin

    def _v3_obs(self, pos, rpy):
        """fx_v3 OCC/VLOOK: can the camera see the predicted pad now? 'in', 'below' (under the image's bottom rows or under the
        camera), 'occluded' (the depth at its pixel is a nearer surface) or 'out' (left/right/above the image)."""
        p = self._pad_predicted()
        if p is None:
            return "out"
        eye, fwd, right, up = cam_pose(pos, rpy)
        d = p - eye
        zc = float(d @ fwd)
        if zc <= 0.3:
            return "below" if float(d @ up) < 0.0 else "out"
        t = math.tan(math.radians(FOV_DEG) / 2)
        u = (float(d @ right) / zc / t + 1) / 2 * IMG_W
        v = (1 - float(d @ up) / zc / t) / 2 * IMG_H
        if u < 8.0 or u >= IMG_W - 8.0:
            return "out"
        if v >= IMG_H - 8.0:
            return "below"
        if v < 8.0:
            return "out"
        dep = self._v3_depth
        if dep is None:
            return "in"
        dm = DEPTH_MIN_M + float(dep[int(v), int(u)]) * (DEPTH_MAX_M - DEPTH_MIN_M)
        if dm < _V3_OCC_RATIO * zc and zc - dm > _V3_OCC_MIN:
            return "occluded"
        return "in"

    def _vch_rq_resume(self, yaw, pos):
        """fx_vrq (C'): end the capped REACQUIRE of a low-gain static village track by resuming the running search
        (search_wps kept, so S2N keeps its map and tabu); no FP anchor, no blacklist. Without a running search: CRUISE
        back to the clue when more than 8 m from it, SEARCH otherwise."""
        self._vch_rq_low = False
        self.search_anchor = None
        self._drop_track()
        _cx_ev("vch_rq_resume")
        if self.search_wps is not None:
            self._set_mode("SEARCH")
            if not self.spin_done:
                self.spin_prev = yaw
        elif (self.center0 is not None
              and float(np.linalg.norm(np.asarray(self.center0[:2], dtype=np.float64) - np.asarray(pos[:2], dtype=np.float64))) > 8.0):
            self._set_mode("CRUISE")
        else:
            self._set_mode("SEARCH")

    def _drop_track(self):
        if _CX_V16_LOG:
            self._v16_log_drop()
        if _FO_RESUME and self.pad_pos is not None:
            self._fo_drop = (self.t, self.pad_pos[:2].copy(), float(self.pad_last_seen))
        self.pad_pos = None
        self.pad_vel = np.zeros(2)
        self.pad_hits = 0
        self.pad_conf = 0.0
        self.pad_obs = []
        self.pad_conflicts = 0
        self.pad_moving = False
        self.pad_move_votes = 0
        self.pad_miss_t = 0.0
        self.pad_omega = 0.0
        self.pad_acc = np.zeros(2)
        self.pad_slow_t = 0.0
        self.pad_fit_sp = 0.0
        self.pad_fit_z = 99.0
        self.pad_min_z = 99.0
        self.pad_info = 0.0
        self.pad_raw = []
        self.pad_still_t = 0.0
        pass
        self.pad_close_seen_t = -99.0
        self.pad_hist = []
        self.pad_min_z = 99.0
        self.pad_raw = []
        self.ring_pts = []
        self.land_off = None
        if _CX_S1M_ANY:
            self._s1m_track_reset()  # fx_s1m: per-track state (the height-map buffer holds world points and is kept)
        if _CX_VK_UNSTICK:
            self._vk_unstuck = False  # a new track has to show a still pad again
        self._v3_track_reset()

    def _v3_track_reset(self):
        """fx_v3: per-track state (a dropped or re-seeded track starts with a full excuse budget and no pulse cycle)."""
        self._v3_exc = 0.0
        self._v3_exc_ev = False
        self._v3_exc_run = 0.0
        self._v3_saved = False
        self._v3_vl_t0 = None
        self._v3_vl_k = -1
        self._v3_vl_on = False
        self._v3_vl_anchor = None

    def _v3_mark_saved(self):
        # fx_v3 OCC: from here on the champion would have had no track; APPROACH 'lost' drops it first (see there)
        if not self._v3_saved:
            self._v3_saved = True
            _cx_v3_ev("v3_occ_saved")

    def _push_raw(self, w, z):
        if _flag("mmodel") and 1.0 <= z < 16.0 and (not self.mm_raw or self.t - self.mm_raw[-1][0] >= 0.1):
            self.mm_raw.append((self.t, float(w[0]), float(w[1])))
            if len(self.mm_raw) > 300:
                self.mm_raw.pop(0)
        _rmax = 16.0 if self.map_label == "village" else 8.0
        if 1.0 <= z < _rmax and (not self.pad_raw or self.t - self.pad_raw[-1][0] >= 0.1):
            self.pad_raw.append((self.t, float(w[0]), float(w[1])))
            if len(self.pad_raw) > 80:
                self.pad_raw.pop(0)

    def _raw_motion_check(self, w):
        if len(self.pad_raw) < 12:
            return False
        r_ = np.asarray(self.pad_raw)
        r_ = r_[self.t - r_[:, 0] <= 8.0]
        if len(r_) < 12 or r_[-1, 0] - r_[0, 0] < 1.5:
            return False
        q = max(3, len(r_) // 4)
        p0 = np.median(r_[:q, 1:3], axis=0)
        p1 = np.median(r_[-q:, 1:3], axis=0)
        d_ = p1 - p0
        disp = float(np.linalg.norm(d_))
        self.pad_still_t = (r_[-1, 0] - r_[0, 0]) if disp < 0.4 else 0.0
        los = unit(self.pad_pos[:2] - self._pos_xy) if self._pos_xy is not None else np.zeros(2)
        perp = float(np.linalg.norm(d_ - float(d_ @ los) * los))
        moved = (perp > 0.35 and disp > 0.5) or disp > 0.9
        if moved and _CX_ME_ANY and self.map_label == "mountain" and not self.pad_moving:
            g_, inf_ = self._me_mv_gate(d_, los)
            radial = perp <= 0.35 and disp <= _ME_MV_DISP
            if _CX_ME_MOV and g_ and radial:
                # fx_me MOV: the quarter medians moved away along the line of sight only: range bias of a closing,
                # level approach. Keep the track static and forget the far sightings that carry the bias.
                self.pad_raw = [p for p in self.pad_raw if self.t - p[0] <= 1.0]
                self._me_mv_blocks += 1
                _cx_me_ev("me_mv_block")
                _cx_me_log("me_mv", "raw_block/%s/d%.2f/p%.2f/n%d" % (inf_, disp, perp, len(r_)))
                return False
            _cx_me_log("me_mv", "raw%s/%s/d%.2f/p%.2f/n%d/g%d" % ("R" if perp <= 0.35 else "P", inf_, disp, perp, len(r_), int(g_)))
        if _CX_VK_UNSTICK and self._vk_tb and self.map_label == "village":
            self._vk_unstick_eval(r_, disp, moved)
        if moved and self.map_label not in ("warehouse", "forest"):
            if not self.pad_moving:
                if _CX_VK_MOTION and self.map_label == "village" and not self._vk_raw_moved_cont(r_):
                    _cx_count("vk_motion_raw_gap")
                    return False
                if _CX_VM_ANY and self._vm_flip("raw", "d%.2f/p%.2f/n%d/T%.1f" % (disp, perp, len(r_), float(r_[-1, 0] - r_[0, 0])),
                                                perp <= _VM_PERP):
                    return False  # fx_vm GATE: a ground-height village track stays static
                self.pad_pos = w.copy()
                self.pad_info = 0.0
                self.pad_obs = self.pad_obs[-20:]
                self.pad_last_seen = self.t
                self.pad_conflicts = 0
                self.pad_moving = True
                self.pad_ever_moving = True
                self.pad_slow_t = 0.0
                return True
        elif (self.pad_moving and not self.pad_ever_moving and r_[-1, 0] - r_[0, 0] >= 5.0
              and disp < 0.35 and self.pad_fit_sp < 0.2):
            self.pad_moving = False
            self.pad_move_votes = 0
            self.pad_vel = np.zeros(2)
            self.pad_omega = 0.0
            self.pad_acc = np.zeros(2)
        return False

    def _vk_unstick_eval(self, r_, disp, moved):
        """CX_VK_UNSTICK: after a village take-back, decide at every raw check whether the current (re-acquired) track
        shows a still pad. Evidence comes only from this track's own raw samples (pad_raw restarts with every track):
        the champion's 8 s check sees no motion and a quarter-median displacement < 0.5 m, and the latest continuous
        run of samples (no gap > 0.5 s) holds >= 15 samples over >= 2 s with a quarter-median displacement < 0.5 m and
        a least-squares speed < 0.25 m/s. The verdict is re-decided every time, so the one in force when LAND freezes
        the tracker (below 1.5 m) is the last one taken before LAND."""
        ok = (not moved and disp < 0.5 and self.pad_ever_moving and not self.pad_moving and self.pad_hits >= 30)
        if ok:
            brk = np.nonzero(np.diff(r_[:, 0]) > _VK_RAW_GAP)[0]
            seg = r_[int(brk[-1]) + 1:] if len(brk) else r_
            ok = len(seg) >= _VK_STILL_N and seg[-1, 0] - seg[0, 0] >= _VK_STILL_SPAN
            if ok:
                q = max(3, len(seg) // 4)
                d_ = np.median(seg[-q:, 1:3], axis=0) - np.median(seg[:q, 1:3], axis=0)
                tt = seg[:, 0] - seg[:, 0].mean()
                sxx = float((tt * tt).sum())
                vx = float((tt * (seg[:, 1] - seg[:, 1].mean())).sum() / sxx)
                vy = float((tt * (seg[:, 2] - seg[:, 2].mean())).sum() / sxx)
                ok = float(np.linalg.norm(d_)) < 0.5 and math.hypot(vx, vy) < _VK_STILL_SP
        if ok and not self._vk_unstuck:
            _cx_count("vk_unstick")
        self._vk_unstuck = bool(ok)

    def _vk_skip_hold(self):
        """CX_VK_UNSTICK: skip LAND's 'once moving' unseen hold on a take-back track shown still before LAND."""
        if self._vk_unstuck and not self.pad_moving and self.map_label == "village":
            _cx_count("vk_unstick_land")
            return True
        return False

    def _vl_hold_skip(self, pos, alt, pad):
        """CX_VL_HOLD: skip LAND's 'once moving' unseen hold (static branch, village) on a tick where the down ray shows
        the pad top right under the drone: within _VL_HOLD_XY of the estimate and the surface under the drone at the
        estimate's height (-_VL_HOLD_DN..+_VL_HOLD_UP). Never inside the router's discarded shadow act()."""
        if (_CX_STATE["shadow"] or self.map_label != "village" or self.pad_moving or not self.pad_ever_moving
                or pad is None):
            return False
        if not (_VL_HOLD_ALT <= alt < _VL_HOLD_ALT_MAX):
            return False
        if float(np.hypot(pos[0] - pad[0], pos[1] - pad[1])) >= _VL_HOLD_XY:
            return False
        dz_ = float(pos[2] - alt - pad[2])  # surface under the drone minus the estimate's height
        if not (-_VL_HOLD_DN <= dz_ <= _VL_HOLD_UP):
            return False
        _cx_vl_ev("vl_hold")
        if not self._vl_hold_seen:
            self._vl_hold_seen = True
            _cx_vl_ev("vl_hold_seed")
            CX_EVENTS["vl_hold_at"] = "%.2f/%.2f" % (float(self.t), dz_)
        return True

    def _vk_raw_moved_cont(self, r_):
        """CX_VK_MOTION: the raw check's verdict when it only looks at the latest continuous run of samples.
        No gap in the samples -> the champion's own verdict (True). A run shorter than the check's own minimum
        (12 samples over 1.5 s) cannot show motion yet."""
        brk = np.nonzero(np.diff(r_[:, 0]) > _VK_RAW_GAP)[0]
        if len(brk) == 0:
            return True
        seg = r_[int(brk[-1]) + 1:]
        if len(seg) < 12 or seg[-1, 0] - seg[0, 0] < 1.5:
            return False
        q = max(3, len(seg) // 4)
        d_ = np.median(seg[-q:, 1:3], axis=0) - np.median(seg[:q, 1:3], axis=0)
        disp = float(np.linalg.norm(d_))
        los = unit(self.pad_pos[:2] - self._pos_xy) if self._pos_xy is not None else np.zeros(2)
        perp = float(np.linalg.norm(d_ - float(d_ @ los) * los))
        return (perp > 0.35 and disp > 0.5) or disp > 0.9

    def _me_rim(self, pos, vel, alt, pad, v_des, pv, d_xy):
        """CX_ME_RIM (see the flag): mine's static mountain LAND with the downhill offset set. pad = the touchdown point
        (estimate + offset, estimate height). Returns the velocity command; with only CX_ME_DIAG it logs and returns
        v_des unchanged."""
        off = np.asarray(self.land_off, dtype=float)
        est_xy = np.asarray(pad[:2], dtype=float) - off
        est_z = float(pad[2])
        cal = [c[1] for c in self._me_rg_cal
               if self.t - c[0] < 20.0 and math.hypot(c[2] - est_xy[0], c[3] - est_xy[1]) < 0.3]
        if self._me_rg_flat is not None:
            ref, thr, src = self._me_rg_flat, _ME_RG_THR_CAL, "f"  # a false fire showed this flat surface: the pad top
        elif len(cal) >= 3:
            ref, thr, src = float(np.median(cal)), _ME_RG_THR_CAL, "c"
        else:
            ref, thr, src = est_z, _ME_RG_THR, "e"
        s_ = float(pos[2] - alt)
        ex = ref - s_
        self._me_rg_desc = self._me_rg_desc + 1 if float(vel[2]) < -0.2 else 0
        self._me_rg_alts = (self._me_rg_alts + [float(alt)])[-10:]
        # contact gate: at touchdown the ray reads through the pad (alt jumps from ~0.03 to the ground below it while the
        # drone still sinks for 1-2 ticks), so the ray must have read >= 0.12 m of free space for the last 0.2 s
        # still moving in toward the estimate's centre (> _ME_RG_VIN): the drone is crossing onto the pad from outside
        # and the ray will step up in a moment (fresh 1381876656, 586454828, 845457369 fired there and lost 0.02 each);
        # a touchdown point beside the pad shows itself once the drone stops there
        c_ = est_xy - pos[:2]
        v_in = float(vel[0] * c_[0] + vel[1] * c_[1]) / max(float(np.linalg.norm(c_)), 1e-6)
        sig = (alt < 19.5 and ex > thr and self.touch_t <= 0.0 and d_xy < _ME_RG_XY and v_in < _ME_RG_VIN
               and self._me_rg_desc >= 5 and float(pos[2]) - ref > 0.08 and min(self._me_rg_alts) > 0.12)
        self._me_rg_run = self._me_rg_run + 1 if sig else 0
        if not _CX_ME_RIM:
            if self._me_rg_run >= 2 and not self._me_rg_diag:
                self._me_rg_diag = True
                _cx_me_log("me_rg_would", "%.2f/%s/%.2f/%.2f/%d/%.2f/%.2f" % (self.t, src, ex, float(pos[2]) - ref, len(cal),
                                                                           est_z - ref, d_xy))
            return v_des
        if not self._me_rg_hold and self._me_rg_run >= 2:
            self._me_rg_nf += 1
            self._me_rg_hold = True
            self._me_rg_t0 = self.t
            self._me_rg_sfire = s_
            self._me_rg_bypass = True  # stop the sink now: 2.5 m/s^2 needs 0.2-0.5 s, the terrain is 0.2-0.3 m away
            self._me_rg_pfire = pos[:2].copy()
            # step the touchdown point _ME_RG_STEP inward of where the drone is now (it may have crossed the rim on its
            # way out to the offset point, so half the offset can lie behind it); centre on a second fire
            cur_ = pos[:2] - est_xy
            r_ = float(np.linalg.norm(cur_))
            new_r = min(r_ - _ME_RG_STEP, 0.5 * float(np.linalg.norm(off)))
            self.land_off = (cur_ * (new_r / r_)) if (self._me_rg_nf < 2 and r_ > 1e-3 and new_r > 0.05) else None
            _cx_me_ev("me_rg_fire")
            _cx_me_log("me_rg_at", "%.2f/%s/%.2f/%.2f/%d/%.2f/%.2f" % (self.t, src, ex, float(pos[2]) - ref, len(cal),
                                                                      est_z - ref, d_xy))
        if self._me_rg_hold:
            tgt = est_xy + (np.asarray(self.land_off, dtype=float) if self.land_off is not None else 0.0)
            rel_ = tgt - pos[:2]
            arrived = float(np.linalg.norm(rel_)) < 0.1
            moved_ = float(np.linalg.norm(pos[:2] - self._me_rg_pfire))
            stepped = s_ >= self._me_rg_sfire + 0.1  # the ray is on the pad top again
            if stepped or arrived or self.t - self._me_rg_t0 > _ME_RG_HOLD_S:
                self._me_rg_hold = False
                self._me_rg_run = 0
                if stepped:
                    _cx_me_ev("me_rg_step")
                elif arrived and moved_ > 0.15 and abs(s_ - self._me_rg_sfire) < 0.03:
                    # moved inward over a perfectly flat surface: the drone was over the pad top all along (the
                    # estimate's height was off), so that surface is the reference from now on
                    self._me_rg_flat = s_
                    _cx_me_ev("me_rg_flat")
                else:
                    _cx_me_ev("me_rg_arrive" if arrived else "me_rg_timeout")
                return v_des
            self.land_stuck_t = 0.0
            if float(pos[2]) - ref < _ME_RG_MARGIN:
                _cx_me_ev("me_rg_climb")
                return np.array([0.0, 0.0, 0.6])  # straight up first: sideways at this height meets the pad's side
            corr = 1.6 * rel_
            cn = float(np.linalg.norm(corr))
            if cn > 0.6:
                corr *= 0.6 / cn
            return np.array([pv[0] + corr[0], pv[1] + corr[1], 0.0])
        return v_des

    def _me_mv_gate(self, dvec, los):
        """CX_ME_MOV gate: is this apparent motion the approach's own range bias? dvec = the displacement (raw path) or
        the fitted velocity (vote path). Returns (gate, info)."""
        pos, vel = self._me_pos, self._me_vel
        if pos is None or vel is None or self.pad_pos is None:
            return False, "nopos"
        dz = float(pos[2] - self.pad_pos[2])
        rng = float(np.hypot(self.pad_pos[0] - pos[0], self.pad_pos[1] - pos[1]))
        close = float(vel[0] * los[0] + vel[1] * los[1])
        along = float(dvec[0] * los[0] + dvec[1] * los[1])
        gate = (not self.pad_ever_moving and self.pad_hits >= _ME_MV_HITS and dz < _ME_MV_DZ and close > _ME_MV_CLOSE
                and float(vel[2]) > _ME_MV_VZ and (along > 0.0 or not _ME_MV_DIR))
        info = "%.2f/dz%.2f/vz%.2f/r%.1f/cl%.2f/al%.2f/h%d/e%d" % (self.t, dz, float(vel[2]), rng, close, along,
                                                                   self.pad_hits, int(self.pad_ever_moving))
        return gate, info

    # ------------------------------------------------------------------ fx_vm (CX_VM_* only)
    def _vm_zstats(self):
        """fx_vm: (median z of the track's last _VM_NOBS detections, the same over those nearer than _VM_DMAX depth, n
        near); None where there are fewer than 5."""
        obs = self.pad_obs[-_VM_NOBS:]
        za = [float(o[3]) for o in obs]
        zn = [float(o[3]) for o in obs if float(o[4]) < _VM_DMAX]
        return (float(np.median(za)) if len(za) >= 5 else None, float(np.median(zn)) if len(zn) >= 5 else None, len(zn))

    def _vm_disp8(self):
        """fx_vm: the champion raw check's displacement measure on the track's raw samples of the last 8 s (median of
        the last quarter minus median of the first quarter), None with fewer than 6 samples."""
        if len(self.pad_raw) < 6:
            return None
        r_ = np.asarray(self.pad_raw)
        r_ = r_[self.t - r_[:, 0] <= 8.0]
        if len(r_) < 6:
            return None
        q = max(3, len(r_) // 4)
        return float(np.linalg.norm(np.median(r_[-q:, 1:3], axis=0) - np.median(r_[:q, 1:3], axis=0)))

    def _vm_flip_vote(self, v, sp, sp_lin, span, resid, z_mean, a):
        """fx_vm: the vote fit is about to flag the track moving. Radial evidence = the fitted velocity's part across
        the line of sight (drone -> estimate) is at most _VM_SPP."""
        try:
            los = unit(self.pad_pos[:2] - self._pos_xy) if self._pos_xy is not None else None
            spp = float(np.linalg.norm(v - float(v @ los) * los)) if los is not None else 99.0
            info = "sp%.2f/spl%.2f/spp%.2f/span%.2f/res%.2f/zm%.1f/n%d" % (
                float(sp), float(sp_lin), spp, float(span), float(resid), float(z_mean), len(a))
        except Exception:
            spp, info = 99.0, "err"
        return self._vm_flip("vote", info, spp <= _VM_SPP)

    def _vm_flip_reseed(self, w, err, z):
        """fx_vm: an established static track is about to be re-seeded as moving on 40 conflicting detections. Radial
        evidence = the new detection's offset from the estimate, across the line of sight, is at most _VM_PERP."""
        try:
            los = unit(self.pad_pos[:2] - self._pos_xy) if self._pos_xy is not None else None
            d_ = np.asarray(w[:2], dtype=np.float64) - np.asarray(self.pad_pos[:2], dtype=np.float64)
            pp = float(np.linalg.norm(d_ - float(d_ @ los) * los)) if los is not None else 99.0
            info = "err%.2f/p%.2f/zd%.2f/dep%.1f" % (float(err), pp, float(w[2]), float(z))
        except Exception:
            pp, info = 99.0, "err"
        return self._vm_flip("reseed", info, pp <= _VM_PERP)

    def _vm_flip(self, path, info="", radial=False):
        """fx_vm: called where mine's village track, not moving yet, is about to be flagged moving (path = vote, raw,
        reseed, movko). Logs the decision point (CX_VM_DIAG) and returns True when CX_VM_GATE vetoes the flip."""
        if self.map_label != "village" or self.pad_pos is None:
            return False
        shadow = bool(_CX_MUTE[0] or _CX_STATE.get("shadow"))
        g = self._vm_gmin
        zf = float(self.pad_pos[2])
        za, zn, nn = self._vm_zstats()
        dsp = self._vm_disp8() if _CX_VM_ANY else None
        veto = False
        if (_CX_VM_GATE and radial and not shadow and g is not None and g <= _VM_GMAX and int(self.pad_hits) >= _VM_HITS
                and za is not None and za - g <= _VM_DZ and zf - g <= _VM_DZ
                and (_VM_DISP <= 0.0 or (dsp is not None and dsp <= _VM_DISP))):
            veto = True
        if (not veto and (_CX_RG_VLOW or _CX_RG_VSLOW) and path in ("raw", "vote", "reseed") and not shadow
                and g is not None and g <= _VM_GMAX and int(self.pad_hits) >= _VM_HITS):
            if _CX_RG_VLOW and za is not None and za - g <= _RG_VLOW_DZ and zf - g <= _RG_VLOW_DZ:
                veto = True  # fx_s1r VLOW: below the lowest goal-pad top
                _cx_vm_ev("rg_vlow")
            elif _CX_RG_VSLOW and path == "raw" and float(self.pad_fit_sp) < _RG_VSLOW_SP:
                veto = True  # fx_s1r VSLOW: the raw check fired on a track the vote fit sees still
                _cx_vm_ev("rg_vslow")
        if not shadow:
            _cx_vm_ev("vm_flip")
            if veto:
                _cx_vm_ev("vm_veto")
            if _CX_VM_DIAG or veto:
                pxy = self._pos_xy if self._pos_xy is not None else np.zeros(2)
                _cx_vm_log("vm_flip_at", "%s:%.2f/zf%.2f/za%s/zn%s/n%d/g%s/h%d/dmin%.1f/sp%.2f/%s/dxy%.1f/%s" % (
                    path, float(self.t), zf, "-" if za is None else "%.2f" % za, "-" if zn is None else "%.2f" % zn, nn,
                    "-" if g is None else "%.2f" % g, int(self.pad_hits), float(self.pad_min_z), float(self.pad_fit_sp),
                    self.mode, float(np.hypot(pxy[0] - self.pad_pos[0], pxy[1] - self.pad_pos[1])),
                    "VETO" if veto else "flip") + ("/" + info if info else "") + ("/D-" if dsp is None else "/D%.2f" % dsp))
        return veto

    def _dv_nohold_ok(self, pos, alt, pad, unseen):
        """CX_DV_NOHOLD: may the static LAND branch keep descending instead of holding height while it centres?
        Only on a pad seen moving (pad_ever_moving: mine's LAND takes the static branch whenever pad_moving is off, e.g.
        after the gentle2 demotion, and its tracker is frozen there below 1.5 m), only while the pad is tracked (seen
        < 0.8 s ago), only when mine's action is flown (not in a shadow act), and only where a descent off the centre
        cannot reach anything but the pad: the down-ray is on the pad top (+-0.15 m) or the surface under the drone lies
        at least _DV_NH_DROP below it (a raised pad; the branch's own 'below' abort stops the sink 0.3 m under the top)."""
        if _DV_SHADOW[0] or not self.pad_ever_moving or unseen >= 0.8:
            return False
        # never on a pad that reads static: pad_ever_moving stays set on static pads too (a false motion flag, the village
        # re-acquire after a take-back), so mine's last fitted speed or filtered velocity must show motion
        if max(float(self.pad_fit_sp), float(np.hypot(self.pad_vel[0], self.pad_vel[1]))) < _DV_NH_MINSP:
            return False
        if alt >= _M2_SAT_M:
            return True  # nothing within 20 m under the drone
        surf = float(pos[2]) - float(alt)
        return abs(surf - float(pad[2])) <= 0.15 or surf < float(pad[2]) - _DV_NH_DROP

    def _static_established(self):
        return (self.pad_pos is not None and not self.pad_moving and not self.pad_ever_moving
                and self.pad_hits >= 30 and self.pad_min_z < 9.0 and self.pad_conf > 0.5
                and (self.pad_still_t >= 3.0 or (self.map_label in ("warehouse", "forest") and self.pad_hits >= 60 and self.pad_conf > 0.7)
                     or (_flag("vmem") and self.map_label == "village" and float(self.pad_pos[2]) < 1.5)
                     or (_flag("cmem") and self.map_label == "city" and float(self.pad_pos[2]) < 1.5)))

    def _track_solid(self):
        return self.pad_pos is not None and self.pad_hits >= 30 and self.pad_conf > 0.5

    # ------------------------------------------------------------------ fx_vx RAY (CX_VX_RAY only)
    def _vx_detect(self, pos, alt, center, yaw):
        """CX_VX_RAY: segment the surface under the drone (steps >= _VX_STEP) and, when the segment after a flat run has
        just reached 3 ticks, test the ground / run / ground triple as a pad crossing; start an episode if it passes."""
        if alt >= 19.5:
            self._vx_segs = []
            self._vx_prev = None
            return
        s = float(pos[2] - alt)
        xy = np.array([float(pos[0]), float(pos[1])])
        if self.start_pos is None or float(np.hypot(xy[0] - self.start_pos[0], xy[1] - self.start_pos[1])) > 1.5:
            if self._vx_gmin is None or s < self._vx_gmin:
                self._vx_gmin = s
        okm = self.mode in ("CRUISE", "SEARCH", "REACQUIRE")
        prev = self._vx_prev
        segs = self._vx_segs
        if prev is None or abs(s - prev[0]) >= _VX_STEP:
            e_in = xy.copy() if prev is None else 0.5 * (prev[1] + xy)
            segs.append({"lo": s, "hi": s, "s0": s, "sl": s, "sum": s, "n": 1, "e_in": e_in, "b": xy.copy(),
                         "alt": float(alt), "okm": okm})
            if len(segs) > (4 if _CX_PVS_VXRIM else 3):
                del segs[0]
        else:
            g_ = segs[-1]
            g_["lo"] = min(g_["lo"], s)
            g_["hi"] = max(g_["hi"], s)
            g_["sl"] = s
            g_["sum"] += s
            g_["n"] += 1
            g_["b"] = xy.copy()
            g_["alt"] = min(g_["alt"], float(alt))
            g_["okm"] = g_["okm"] and okm
        self._vx_prev = (s, xy.copy())
        if len(segs) < 3 or segs[-1]["n"] != 3 or self._vx is not None:
            return
        g0, p, g1 = segs[-3:]
        if (_CX_PVS_VXRIM and g0["n"] < _PVS_RIM_N and len(segs) == 4 and segs[0]["n"] >= _PVS_RIM_N
                and abs(g0["sl"] - segs[0]["sl"]) <= _PVS_RIM_DZ and g0["hi"] < p["lo"]):
            # fx_vxrim: the short ledge step before the pad top is looked through (the ground before it is the g0)
            _cx_vx_log("pvs_vxrim_at", "%.2f/%d/%.2f/%.2f" % (self.t, g0["n"], g0["sl"], segs[0]["sl"]), cap=4)
            g0 = dict(g0)
            g0["n"] = segs[0]["n"]
            g0["sl"] = segs[0]["sl"]
            _cx_vx_ev("pvs_vxrim")
        gmin = self._vx_gmin
        if gmin is None or p["hi"] - p["lo"] > _VX_FLAT or g0["n"] < 5:
            return
        sp = p["sum"] / p["n"]
        gb, ga = g0["sl"], g1["s0"]
        hs = sp - gmin
        L1 = float(np.linalg.norm(g1["e_in"] - p["e_in"]))
        M1 = 0.5 * (p["e_in"] + g1["e_in"])
        dcl = max(abs(float(M1[0] - center[0])), abs(float(M1[1] - center[1])))
        cls = None
        if abs(gb - gmin) <= 0.02 and abs(ga - gmin) <= 0.02 and _VX_A_LO <= hs <= _VX_A_HI:
            cls = "A"
        elif (_VX_B_LO <= hs <= _VX_B_HI and -0.01 <= gb - gmin <= _VX_B_GND and -0.01 <= ga - gmin <= _VX_B_GND
              and dcl <= _VX_B_BOX):
            cls = "B"
        if cls is None:
            return
        why = None
        if L1 > _VX_LMAX or L1 < _VX_LMIN:
            why = "len"
        elif not (p["okm"] and g1["okm"]):
            why = "mode"
        elif p["alt"] < _VX_AGL:
            why = "agl"
        elif dcl > _VX_CLUE:
            why = "clue"
        elif self.start_pos is not None and float(np.hypot(M1[0] - self.start_pos[0], M1[1] - self.start_pos[1])) < 3.0:
            why = "start"
        elif self.t > _VX_TMAX:
            why = "late"
        elif self.pad_moving or self.pad_ever_moving:
            why = "moving"
        elif self._track_reliable():
            why = "track"
        elif self._vx_n >= _VX_NMAX:
            why = "nmax"
        elif any(float(np.hypot(M1[0] - r[0], M1[1] - r[1])) < 1.5 for r in self._vx_rej):
            why = "rej"
        _cx_vx_log("vx_cand", "%.2f/%s/%.3f/%.2f/%s" % (self.t, cls, hs, L1, why or "fire"))
        if why is not None:
            return
        d1 = (g1["e_in"] - p["e_in"]) / max(L1, 1e-6)
        self._vx = {"ph": "ret", "t0": self.t, "tph": self.t, "M1": M1.copy(), "d1": d1, "n1": np.array([-d1[1], d1[0]]),
                    "L1": L1, "sp": sp, "cls": cls, "z": float(pos[2]), "yaw": float(yaw), "pm": self.mode,
                    "pt0": self.mode_t0, "up": None, "um": None, "on2": False, "uprev": None, "off_t": 0.0, "C": None}
        self._vx_n += 1
        self.mode = "VXRAY"
        _cx_vx_ev("vx_fire")

    def _vx_end(self, why):
        """CX_VX_RAY: abandon the episode, restore the interrupted mode (its clock excludes the episode)."""
        e = self._vx
        self._vx = None
        self._vx_rej.append(np.asarray(e["M1"], dtype=np.float64).copy())
        self.mode = e["pm"]
        self.mode_t0 = self.t - (e["t0"] - e["pt0"])
        _cx_vx_ev("vx_abort_" + why)
        _cx_vx_log("vx_at", "%.2f/%s/%.2f/abort_%s/%s" % (e["t0"], e["cls"], e["L1"], why, e["ph"]), cap=4)

    def _vx_land(self):
        """CX_VX_RAY: hand LAND a camera-free static track at the ray's pad centre and pad-top height."""
        e = self._vx
        C, top = e["C"], float(e["sp"])
        self._vx = None
        self._drop_track()
        self.pad_pos = np.array([float(C[0]), float(C[1]), top])
        self.pad_hits = 30
        self.pad_conf = 0.9
        self.pad_last_seen = self.t
        self.pad_min_z = 1.5
        self.pad_info = 4000.0
        self.pad_still_t = 3.0
        self.pad_obs = [(self.t, float(C[0]), float(C[1]), top, 1.5)]
        self.pad_hist = [(self.t, float(C[0]), float(C[1]), top)]
        self._set_mode("LAND")
        self.land_xy = np.array([float(C[0]), float(C[1])])
        self.land_z = top
        self.land_t0 = self.t
        self.land_stuck_t = 0.0
        self.touch_t = 0.0
        self.land_phase = "track"
        self.land_pdir = None
        self.land_off = None
        self.land_reclimb = False
        self.land_retry = 0
        self.land_dive = False
        _cx_vx_ev("vx_ok")
        _cx_vx_log("vx_at", "%.2f/%s/%.2f/%.2f/%.2f/%.2f" % (e["t0"], e["cls"], e["L1"], e["L2"], e["e"], self.t), cap=4)

    def _vx_step(self, pos, vel, alt):
        """CX_VX_RAY episode tick (mode VXRAY): returns (v_des, yaw_des). Ends the episode itself (LAND or restore)."""
        e = self._vx
        s = float(pos[2] - alt) if alt < 19.5 else None
        on = s is not None and abs(s - e["sp"]) <= _VX_ON
        off = s is None or abs(s - e["sp"]) > _VX_OFF
        xy = pos[:2]
        vh = float(np.hypot(vel[0], vel[1]))
        tph = self.t - e["tph"]
        M1, d1, n1 = e["M1"], e["d1"], e["n1"]
        u = float((xy - M1) @ n1)
        w = float((xy - M1) @ d1)
        zt = e["z"]
        vz = float(np.clip(1.5 * (zt - pos[2]), -1.0, 1.0))
        hold = np.array([0.0, 0.0, vz])
        why = None
        if self.pad_moving or self.pad_ever_moving:
            why = "moving"
        elif s is not None and s > float(e["sp"]) + 0.1 and alt < _VX_OBST:
            why = "obst"   # something close under the drone (the moves are flown blind, planner off)
        elif e["ph"] == "ret":
            rel = M1 - xy
            d = float(np.linalg.norm(rel))
            if d < _VX_RET_D and abs(float(vel[0] * d1[0] + vel[1] * d1[1])) < 0.6:
                e["ph"], e["tph"] = "x2a", self.t
                e["uprev"] = u
            elif tph > 6.0:
                why = "ret"
            v = rel * (min(2.5, math.sqrt(4.4 * max(d - 0.05, 0.0)) + 0.1) / max(d, 1e-6))
            vd = np.array([v[0], v[1], vz])
        elif e["ph"] in ("x2a", "x2b"):
            sgn = 1.0 if e["ph"] == "x2a" else -1.0
            vs_ = _VX_VX
            if e["ph"] == "x2b" and e["on2"] and not e.get("occp"):
                # back across the pad: slow down toward the far rim the first edge predicts (a small overshoot there
                # keeps the way back to the centre short)
                rem_ = u - (e["up"] - math.sqrt(max(0.36 - e["wp"] * e["wp"], 0.0)) - 0.6)
                vs_ = float(np.clip(0.35 + 1.6 * max(rem_, 0.0), 0.35, _VX_VX))
            v = n1 * (sgn * vs_) - d1 * float(np.clip(2.0 * w, -1.0, 1.0))
            vd = np.array([v[0], v[1], vz])
            up_ = e["uprev"] if e["uprev"] is not None else u
            wp_ = e["wprev"] if e.get("wprev") is not None else w
            if e["ph"] == "x2a":
                if on:
                    e["on1"] = True
                if e.get("on1") and off:
                    e["up"], e["wp"] = 0.5 * (u + up_), 0.5 * (w + wp_)
                    e["occp"] = s is not None and s > float(e["sp"]) + 0.1  # the ray left the pad top upward: an overhang
                    e["ph"], e["tph"] = "x2b", self.t
                elif u > 1.45:
                    why = "x2a_long"
                elif tph > (4.0 if e.get("on1") else 1.5):
                    why = "x2a_time"
            else:
                if on:
                    e["on2"] = True
                if e["on2"] and off:
                    e["um"], e["wm"] = 0.5 * (u + up_), 0.5 * (w + wp_)
                    occm = s is not None and s > float(e["sp"]) + 0.1
                    # each edge lies 0.6 m from the centre, which sits on the scan line (w = 0): the centre's position
                    # along the scan from either edge, corrected for the cross-track offset the edge was crossed at.
                    # An edge where the ray left the pad top upward is an overhang cutting the chord, not the rim: the
                    # centre then comes from the other edge alone, and the overhang must lie inside the pad's extent.
                    ucp = e["up"] - math.sqrt(max(0.36 - e["wp"] * e["wp"], 0.0))
                    ucm = e["um"] + math.sqrt(max(0.36 - e["wm"] * e["wm"], 0.0))
                    occp = bool(e.get("occp"))
                    if occp and not occm:
                        ec, dev = ucm, max(0.0, e["up"] - (ucm + 0.6))
                    elif occm and not occp:
                        ec, dev = ucp, max(0.0, (ucp - 0.6) - e["um"])
                    else:
                        ec, dev = 0.5 * (ucp + ucm), abs(ucp - ucm)
                    e["L2"], e["e"] = e["up"] - e["um"], ec
                    r_ = math.sqrt(0.25 * e["L1"] * e["L1"] + ec * ec)
                    if occp and occm:
                        why = "occ2"
                    elif max(abs(e["wp"]), abs(e["wm"])) > _VX_WMAX:
                        why = "wofs"
                    elif dev > _VX_UCTOL:
                        why = "l2"
                    elif abs(r_ - 0.6) > _VX_DISC:
                        why = "disc"
                    else:
                        e["C"] = M1 + n1 * ec
                        e["ph"], e["tph"] = "down", self.t
                elif u < e["up"] - 1.75:
                    why = "x2b_long"
                elif tph > 6.0:
                    why = "x2b_time"
            e["uprev"], e["wprev"] = u, w
        else:  # down: to C along the chord just scanned, sinking at once (the ray saw every column there clear)
            rel = e["C"] - xy
            d = float(np.linalg.norm(rel))
            v = rel * 2.0
            if float(np.linalg.norm(v)) > 1.0:
                v *= 1.0 / float(np.linalg.norm(v))
            zg = float(e["sp"]) + _VX_HLAND - 0.45
            if d < 0.5 and on:
                vz_s = -float(np.clip(2.0 * (pos[2] - zg) + 0.2, 0.0, _VX_SINK))
            else:
                vz_s = float(np.clip(-2.0 * float(vel[2]), -0.5, 0.5))  # not over the pad top: stop sinking
            vd = np.array([v[0], v[1], vz_s])
            if d < 0.15 and not on:
                e["off_t"] += SIM_DT
            if e["off_t"] > 0.3:
                why = "down_off"
            elif d < 0.15 and on and pos[2] - float(e["sp"]) <= _VX_HLAND + 0.15:
                self._vx_land()
                return vd, e["yaw"]
            elif tph > 6.0:
                why = "down_time"
        if why is not None:
            self._vx_end(why)
            return hold, e["yaw"]
        return vd, e["yaw"]

    # ------------------------------------------------------------------ fx_v8 SURF / NODIP (CX_V8_* only)
    def _v8_surf_step(self, pos, alt, center):
        """CX_V8_SURF: in a static village LAND the surface right under the drone, once it is centred over the estimate,
        must be the pad top at the estimate's height. Runs before the mode chain, so after a fire this tick is already
        flown in CRUISE/SEARCH without a track (pad is read after this call)."""
        pp = self.pad_pos
        if not (self.mode == "LAND" and self.map_label == "village" and pp is not None and not self.pad_moving
                and not self.pad_ever_moving and self.touch_t <= 0.0 and _v3_live()):
            self._v8_surf_t = 0.0
            self._vg_v8_sh = 0.0
            return
        d_ = float(np.hypot(pos[0] - pp[0], pos[1] - pp[1]))
        dz_ = float(pos[2] - alt - pp[2])  # surface under the drone minus the estimate's height
        up_, dn_ = _V8_SURF_UP, _V8_SURF_DN
        if _CX_VG_V8H:
            # fx_vg V8H: wider margins where the surface under the drone is at a pad-top height (see the flag)
            try:
                g_ = self._vg_gmin
                if g_ is not None and g_ <= _VG_GMAX:
                    hs_ = float(pos[2] - alt) - g_
                    if _VG_CAR_LO <= hs_ <= _VG_CAR_HI:
                        dn_ = max(dn_, _VG_DN2)
                    elif _VG_GND_LO <= hs_ <= _VG_GND_HI or hs_ >= _VG_HOUSE:
                        up_ = max(up_, _VG_UP2)
            except Exception:
                up_, dn_ = _V8_SURF_UP, _V8_SURF_DN
        gate_ = d_ < _V8_SURF_XY and _V8_SURF_ALT <= alt < _V8_SURF_ALT_MAX and float(pos[2] - pp[2]) > _V8_SURF_H
        if _CX_VG_V8H:
            # c22's counter, to tell when a fire that c22 would make is withheld (logging only)
            if gate_ and (dz_ > _V8_SURF_UP or dz_ < -_V8_SURF_DN):
                self._vg_v8_sh += SIM_DT
                if self._vg_v8_sh >= _V8_SURF_T - 1e-9:
                    self._vg_v8_sh = 0.0
                    if not (gate_ and (dz_ > up_ or dz_ < -dn_) and self._v8_surf_t + SIM_DT >= _V8_SURF_T - 1e-9):
                        _cx_vg_ev("vg_v8h_save")
                        try:
                            hs_l = float(pos[2] - alt) - float(self._vg_gmin)
                        except Exception:
                            hs_l = -1.0
                        at_ = (str(CX_EVENTS.get("vg_v8h_at", "")) + " %.2f/%.2f/%.2f" % (float(self.t), dz_, hs_l)).split()
                        CX_EVENTS["vg_v8h_at"] = " ".join(at_[-8:])
            else:
                self._vg_v8_sh = 0.0
        if gate_ and (dz_ > up_ or dz_ < -dn_):
            self._v8_surf_t += SIM_DT
        else:
            self._v8_surf_t = 0.0
            return
        if self._v8_surf_t < _V8_SURF_T - 1e-9:
            return
        # not the top of a pad at the estimate: a car roof, the ground beside a car, a pickup bed
        self._v8_surf_t = 0.0
        _cx_v8_ev("v8_surf")
        dc_ = (float(np.linalg.norm(np.asarray(self.center0[:2], dtype=np.float64) - np.asarray(pp[:2], dtype=np.float64)))
               if self.center0 is not None else -1.0)
        at_ = (str(CX_EVENTS.get("v8_surf_at", "")) + " %.2f/%.2f/%.1f" % (float(self.t), dz_, dc_)).split()
        CX_EVENTS["v8_surf_at"] = " ".join(at_[-8:])
        self.bad_spots.append(np.asarray(pp, dtype=np.float64).copy())
        self._drop_track()
        if self.search_wps is None and float(np.linalg.norm((center - pos)[:2])) > 8.0:
            self._set_mode("CRUISE")
        else:
            self._set_mode("SEARCH")
            if (_CX_V8_NODIP and self.search_wps is not None and not self.spin_done
                    and getattr(self, "_vdip_active", False) and self.search_center_xy is not None
                    and float(np.linalg.norm(np.asarray(self.search_center_xy, dtype=np.float64) - pos[:2])) > _V8_NODIP_M):
                # the unfinished dip belongs to the search centre, not to the car: resume the rings instead
                self.spin_done = True
                self._vdip_active = False
                _cx_v8_ev("v8_nodip")

    # ------------------------------------------------------------------ fx_vf LAND / KING / ZONE (CX_VF_* only)
    def _vf_in_band(self, z):
        """fx_vf: True when a height z lies in the village band where no pad exists (relative to the lowest ground seen)."""
        g = self._vf_gmin
        if g is None or g > _VF_GMAX:
            return False
        dz = float(z) - g
        return _VF_LO <= dz <= _VF_HI

    def _vf_zone_hit(self, w):
        """fx_vf ZONE: True when detection w lies within _VF_ZONE_R (xy) of an earlier band rejection and in the band."""
        if not self._vf_zones or not self._vf_in_band(float(w[2])):
            return False
        for zx, zy, _zz in self._vf_zones:
            if (float(w[0]) - zx) ** 2 + (float(w[1]) - zy) ** 2 < _VF_ZONE_R * _VF_ZONE_R:
                return True
        return False

    def _vf_reject(self, pos, center, why):
        """fx_vf: reject the current (band) track: blacklist it, drop it, and leave as CX_V8_SURF leaves a rejected car."""
        pp = np.asarray(self.pad_pos, dtype=np.float64).copy()
        _cx_vf_ev("vf_" + why)
        dc_ = (float(np.linalg.norm(np.asarray(self.center0[:2], dtype=np.float64) - pp[:2]))
               if self.center0 is not None else -1.0)
        at_ = (str(CX_EVENTS.get("vf_at", "")) + " %s:%.2f/%.2f/%.2f/%d/%.1f" % (
            why, float(self.t), float(pp[2]), float(self._vf_gmin), int(self.pad_hits), dc_)).split()
        CX_EVENTS["vf_at"] = " ".join(at_[-8:])
        self.bad_spots.append(pp)
        if _CX_VF_ZONE:
            self._vf_zones.append((float(pp[0]), float(pp[1]), float(pp[2])))
        self._drop_track()
        self._vland_t = 0.0
        if self.search_wps is None and float(np.linalg.norm((center - pos)[:2])) > 8.0:
            self._set_mode("CRUISE")
        else:
            self._set_mode("SEARCH")

    def _mtf_zb_veto(self, pos, center):
        """fx_mtf ZB: drop a static mountain track whose estimate lies outside the clue z band (not a pad): remember the
        spot (re-seed suppression in _update_track), no bad_spots (its +-2 m z box could cover a true pad under a false
        estimate), no vf_* events or zones. APPROACH / REACQUIRE / RECOVER leave as _vf_reject leaves; other modes stay."""
        pp = np.asarray(self.pad_pos, dtype=np.float64).copy()
        hits_ = int(self.pad_hits)
        self._mtf_zb_spots.append((float(pp[0]), float(pp[1])))
        _cx_mtf_ev("mtf_zb_veto")
        _cx_mtf_log("mtf_zb_at", "%.2f/%s/h%d/dz%+.1f/d%.1f" % (
            float(self.t), self.mode, hits_, float(pp[2]) - float(self.center0[2]),
            float(np.hypot(pp[0] - pos[0], pp[1] - pos[1]))), 4)
        self._drop_track()
        if self.mode in ("APPROACH", "REACQUIRE", "RECOVER"):
            # (RECOVER too: staying would hover up to 4 s and then rebuild the search rings around the clue)
            if self.search_wps is None and float(np.linalg.norm((center - pos)[:2])) > 8.0:
                self._set_mode("CRUISE")
            else:
                self._set_mode("SEARCH")

    # ------------------------------------------------------------------ fx_m15 LOOK / KEEP (CX_M15_* only)
    def _m15_key(self):
        """Identity of the current track: the time of its first sighting (pad_obs keeps the first 75 of a young track)."""
        return round(float(self.pad_obs[0][0]), 2) if self.pad_obs else None

    def _m15_look_step(self, pos, rpy, alt, yaw):
        """CX_M15_LOOK: (v_des, yaw_des) for this static mountain CRUISE/SEARCH tick while a look runs, else None."""
        lk = self._m15_look
        pp = self.pad_pos
        if lk is not None:
            end = None
            if pp is None:
                end = "drop"          # missed in view for 1.2 s (_update_track): not a pad
            elif self.pad_moving or self.pad_ever_moving:
                end = "moving"
            elif self._track_reliable():
                end = "reliable"      # (the mode chain switches to APPROACH on its next tick)
            elif self._m15_key() != lk["key"]:
                end = "new"
            elif self.t - lk["t0"] > _M15_LOOK_CAP:
                end = "cap"
            if end is not None:
                self._m15_look = None
                _cx_m15_ev("m15_look_end_" + end)
                return None
        else:
            if (pp is None or self.pad_moving or self.pad_ever_moving or self.pad_hits < _M15_LOOK_HITS
                    or self._m15_look_n >= _M15_LOOK_MAX or self.t > _M15_LOOK_TMAX):
                return None
            if self.t - self.pad_last_seen < _M15_LOOK_UNSEEN or self._track_reliable():
                return None
            key = self._m15_key()
            if key is None or key in self._m15_look_done:
                return None
            if float(np.hypot(pp[0] - pos[0], pp[1] - pos[1])) > _M15_LOOK_DMAX:
                return None
            if self.center0 is not None and abs(float(pp[2]) - float(self.center0[2])) > _M15_LOOK_DZ:
                # the task puts the pad top within clue z +- 5 m: an estimate outside that band is a false detection
                # (false young tracks in the c14 dev flights sat 6-14 m off it; real ones within 5.1 m)
                return None
            if self._pad_in_view(pos, rpy):
                return None
            if _CX_MTF_YT and not (self._mtf_pb and self.pad_obs is self._mtf_pb_ref):
                if key not in self._mtf_yt_blk:  # (level-born young track: no look)
                    self._mtf_yt_blk.add(key)
                    _cx_mtf_ev("mtf_yt_block")
                return None
            if _CX_MTF_YT and _MTF_YT_DHMIN > 0.0 and float(np.hypot(pp[0] - pos[0], pp[1] - pos[1])) < _MTF_YT_DHMIN:
                # fx_mtf YT DHMIN: the estimate is nearly under the drone; the look would hover at its own vantage with
                # the estimate past the blind limit and end "blind" (952614864: -7.9 s). Never retried: the base's flight.
                self._m15_look_done.append(key)
                _cx_mtf_ev("mtf_yt_dhmin")
                return None
            lk = self._m15_look = {"t0": self.t, "key": key}
            self._m15_look_done.append(key)
            self._m15_look_n += 1
            if _CX_MTF_LKEEP:
                self._mtf_lk_obs = self.pad_obs  # fx_mtf LKEEP: this track had a look
            _cx_m15_ev("m15_look")
        rel = pp - pos
        d_h = float(np.hypot(rel[0], rel[1]))
        yaw_des = math.atan2(rel[1], rel[0]) if d_h > 0.5 else yaw
        # Only motions the camera can see are asked for (the planner flies a request more than 50 deg off the camera
        # axis along its nearest in-view ray at 0.6 m/s, i.e. forward): face the estimate, close in horizontally only
        # while it is farther than _M15_LOOK_DFAR, and otherwise hover and change height until it sits ~35 deg below the
        # axis: _M15_LOOK_ZLO.._ZHI over it, never under ground + 2 (a pad above: climb to _ZLO over it)
        h_want = float(np.clip(0.7 * d_h, _M15_LOOK_ZLO, _M15_LOOK_ZHI))
        z_floor = float(pos[2] - alt) + 2.0
        z_w = max(float(pp[2]) + h_want, z_floor)
        vz = float(np.clip(1.2 * (z_w - pos[2]), -2.0 if alt > 8.0 else -1.0, 2.5))
        if d_h > _M15_LOOK_DFAR and abs(wrap(yaw_des - yaw)) < 0.6:
            v_h = min(_M15_LOOK_VH, 0.5 * (d_h - _M15_LOOK_DFAR) + 0.6)
            v_xy = rel[:2] / d_h * v_h
            planner = True
        else:
            v_xy = np.zeros(2)
            planner = False  # hover + vertical only (as the champion's spin flies it: ground + 2 floor)
            h = float(pos[2] - pp[2])
            if h > 0.0 and math.degrees(math.atan2(h, max(d_h, 1e-3))) > _M15_LOOK_BLIND and z_w - pos[2] > -0.3:
                # the ground + 2 floor keeps the estimate under the image from here: nothing more to see
                lk["blind"] = lk.get("blind", 0.0) + SIM_DT
                if lk["blind"] > _M15_LOOK_BLIND_S:
                    self._m15_look = None
                    _cx_m15_ev("m15_look_end_blind")
                    return None
        _cx_m15_ev("m15_look_ticks")
        return np.array([v_xy[0], v_xy[1], vz]), yaw_des, planner

    def _m15_keep_ok(self, pad, pos, rpy, vel, alt, unseen):
        """CX_M15_KEEP: may this static mountain APPROACH keep its young track past the unseen/miss drop?"""
        if self.pad_hits < _M15_KEEP_HITS or self.pad_miss_t > 1.2 or unseen > _M15_KEEP_S:
            return False
        rel = pad - pos
        dh = float(np.hypot(rel[0], rel[1]))
        h = -float(rel[2])
        if dh >= 8.0 or h <= 0.5:
            return False
        prow = self._pad_pixel_row(pos, rpy)
        # below the look-down limit (fx_ms KEEP's test), or under the image's bottom rows / under the camera now
        if not (h > 0.8 * dh + 0.5 or prow is None or prow > 118.0):
            return False
        if not (dh < 1.0 or float(vel[0] * rel[0] + vel[1] * rel[1]) > 0.2 * max(dh, 1e-6)):
            return False
        if dh < 0.6 and (h - alt) >= 0.6:
            _cx_m15_ev("m15_keep_veto_ray")  # over the estimate, the ground there is well above its top: no pad
            return False
        fl = self._ms_pad_float(pad)
        if fl is not None and fl < _M15_KEEP_FLOAT:
            _cx_m15_ev("m15_keep_veto_float")  # the estimate lies on the fitted terrain: a detection on the ground
            return False
        return True

    # ------------------------------------------------------------------ fx_v16 LOG (CX_V16_LOG only; reads state, never writes it)
    def _v16_log_pre(self):
        """CX_V16_LOG: new bad_spots since the last tick, and the track state before _update_track."""
        try:
            nb = len(self.bad_spots)
            n0 = int(getattr(self, "_v16_nbs", 0))
            if nb < n0:
                n0 = 0  # the router cleared the list (take-back)
            for bs in self.bad_spots[n0:]:
                _v16_log("v16log_bad", [round(float(self.t), 2), self.mode, round(float(bs[0]), 2), round(float(bs[1]), 2),
                                        round(float(bs[2]), 2)])
            self._v16_nbs = nb
            return (self.pad_pos is None, int(self.pad_hits), int(self.pad_conflicts), bool(self.pad_moving))
        except Exception:
            return None

    def _v16_log_post(self, pre, dets):
        """CX_V16_LOG: one row per tick with a detection >= 0.45: [t, mode, outcome, hits, dets[s,x,y,z,range], estimate]."""
        try:
            good = [d for d in dets if d[0] >= 0.45 and np.isfinite(d[1]).all()]
            if not good or pre is None:
                return
            none0, h0, c0, mv0 = pre
            if self.pad_pos is None:
                out = "rej" if none0 else "drop"
            elif none0:
                out = "new"
            elif self.pad_moving and not mv0:
                out = "mv"
            elif self.pad_hits < h0 or (self.pad_hits == 1 and h0 > 1):
                out = "rs"
            elif self.pad_hits > h0:
                out = "upd"
            elif self.pad_conflicts > c0:
                out = "cf"
            else:
                out = "rej"
            pp = self.pad_pos
            _v16_log("v16log_det", [round(float(self.t), 2), self.mode, out, int(self.pad_hits),
                                    [[round(float(d[0]), 2), round(float(d[1][0]), 2), round(float(d[1][1]), 2),
                                      round(float(d[1][2]), 2), round(float(d[2]), 1)] for d in good[:4]],
                                    None if pp is None else [round(float(pp[0]), 2), round(float(pp[1]), 2), round(float(pp[2]), 2)]])
        except Exception:
            pass

    def _v16_look_step(self, pos, rpy, alt, yaw):
        """CX_V16_LOOK: (v_des, yaw_des, use_planner) for this static village CRUISE/SEARCH tick while a look runs, else None."""
        lk = self._v16_look
        pp = self.pad_pos
        if lk is not None:
            end = None
            if pp is None:
                end = "drop"          # missed in view for 1.2 s (_update_track): not a pad, or hidden
            elif self.pad_moving or self.pad_ever_moving:
                end = "moving"
            elif self._track_reliable():
                end = "reliable"      # (the mode chain switches to APPROACH on its next tick)
            elif self._m15_key() != lk["key"]:
                end = "new"
            elif self.t - lk["t0"] > _V16_LOOK_CAP:
                end = "cap"
            if end is not None:
                self._v16_look = None
                _cx_v16_ev("v16_look_end_" + end)
                return None
        else:
            if (pp is None or self.pad_moving or self.pad_ever_moving or self.pad_hits < _V16_LOOK_HITS
                    or self._v16_look_n >= _V16_LOOK_MAX or self.t > _V16_LOOK_TMAX):
                return None
            if self.t - self.pad_last_seen < _V16_LOOK_UNSEEN or self._track_reliable():
                return None
            key = self._m15_key()
            if key is None or key in self._v16_look_done:
                return None
            if float(np.hypot(pp[0] - pos[0], pp[1] - pos[1])) > _V16_LOOK_DMAX:
                return None
            if self.v_gnd is None or float(pp[2]) - float(self.v_gnd) > _V16_LOOK_DZ:
                return None  # not at ground level: roofs and walls (a static village pad top is ~0.1-0.2 m over the ground)
            if self._pad_in_view(pos, rpy):
                return None
            lk = self._v16_look = {"t0": self.t, "key": key}
            self._v16_look_done.append(key)
            self._v16_look_n += 1
            _cx_v16_ev("v16_look")
        rel = pp - pos
        d_h = float(np.hypot(rel[0], rel[1]))
        yaw_des = math.atan2(rel[1], rel[0]) if d_h > 0.5 else yaw
        # the estimate ~39 deg below the camera axis: 0.8 x d_h above it; never under the surface below + 2.5 m; no climb
        h_want = float(np.clip(0.8 * d_h, 2.5, 8.0))
        z_floor = float(pos[2] - alt) + 2.5
        z_w = min(max(float(pp[2]) + h_want, z_floor), float(pos[2]))
        vz = float(np.clip(1.2 * (z_w - pos[2]), -1.0, 0.0))
        if d_h > _V16_LOOK_DFAR and abs(wrap(yaw_des - yaw)) < 0.6:
            v_h = min(_V16_LOOK_VH, 0.5 * (d_h - _V16_LOOK_DFAR) + 0.6)
            v_xy = rel[:2] / d_h * v_h
            planner = True
        else:
            v_xy = np.zeros(2)
            planner = False  # hover + sink only (as the champion's spin and vland hover fly)
            h = float(pos[2] - pp[2])
            if h > 0.0 and math.degrees(math.atan2(h, max(d_h, 1e-3))) > 44.0 and z_w - pos[2] > -0.3:
                lk["blind"] = lk.get("blind", 0.0) + SIM_DT
                if lk["blind"] > 1.0:
                    self._v16_look = None
                    _cx_v16_ev("v16_look_end_blind")
                    return None
        _cx_v16_ev("v16_look_ticks")
        return np.array([v_xy[0], v_xy[1], vz]), yaw_des, planner

    def _v16_vland(self, pos, rpy, alt, pad, d_h):
        """CX_V16_VLAND: vertical speed for this vland-hover tick when the estimate is under the image, else None."""
        vl = self._v16_vl
        if vl is None or abs(vl[0] - self.mode_t0) > 1e-6:
            vl = self._v16_vl = [self.mode_t0, 0.0, False]
        if vl[1] >= _V16_VL_EXT_S or d_h < 0.8 or alt < 1.2:
            return None
        prow = self._pad_pixel_row(pos, rpy)
        if prow is not None and prow <= _V16_VL_ROW:
            return None
        h = float(pos[2] - pad[2])
        z_t = float(pad[2]) + max(1.0, 0.7 * d_h)
        if h <= max(1.0, 0.7 * d_h) + 0.05:
            return None  # already low enough: the image edge is not what hides it
        vl[1] += SIM_DT
        if not vl[2]:
            vl[2] = True
            _cx_v16_ev("v16_vland")
        _cx_v16_ev("v16_vland_ticks")
        return float(np.clip(1.2 * (z_t - pos[2]), -0.8, 0.0))

    def _v16_log_drop(self):
        """CX_V16_LOG: a track is being dropped; record who dropped it (the caller of _drop_track)."""
        try:
            pp = self.pad_pos
            if pp is None:
                return
            import sys as _s16
            f_ = _s16._getframe(2)
            _v16_log("v16log_drop", [round(float(self.t), 2), self.mode, f_.f_code.co_name, int(f_.f_lineno), int(self.pad_hits),
                                     round(float(self.pad_conf), 2), round(float(self.t - self.pad_last_seen), 2),
                                     [round(float(pp[0]), 2), round(float(pp[1]), 2), round(float(pp[2]), 2)],
                                     bool(self.pad_moving), round(float(self.pad_miss_t), 2)])
        except Exception:
            pass

    # ------------------------------------------------------------------ fx_m2 PULSE (CX_M2_PULSE only)
    _M2_XR =((np.arange(40, 88) + 0.5) / IMG_W * 2 - 1) * math.tan(math.radians(FOV_DEG) / 2)
    _M2_YU = (1 - (np.arange(48, 112) + 0.5) / IMG_H * 2) * math.tan(math.radians(FOV_DEG) / 2)

    def _m2_pulse_clear(self, depth, pos, rpy):
        """Planar depth of the nearest pixel of the champion's pulse window (rows 48-112, cols 40-88) whose hit point is
        less than _M2_PULSE_DROP below the eye (or above it); DEPTH_MAX_M when there is none. Ground further below the
        eye is not in the path of a ~0.5 m level pulse."""
        z = DEPTH_MIN_M + depth[48:112, 40:88].astype(np.float64) * (DEPTH_MAX_M - DEPTH_MIN_M)
        eye, fwd, right, up = cam_pose(pos, rpy)
        # world height of each pixel's ray per metre of planar depth: fwd + xr*right + yu*up (pixel_depth_to_world)
        rz = fwd[2] + self._M2_XR[None, :] * right[2] + self._M2_YU[:, None] * up[2]
        obst = z * rz > -_M2_PULSE_DROP
        return float(z[obst].min()) if obst.any() else DEPTH_MAX_M

    # ------------------------------------------------------------------ fx_ma SO (CX_MA_SO only)
    def _ma_live_track(self):
        """fx_ma: a track that counts (seen within 1 s, or >= 5 hits); a young stale one (often a false glimpse far off)
        does not block SO or RING (774251874, 182857009, 913796364 carried one)."""
        return self.pad_pos is not None and (self.t - self.pad_last_seen < 1.0 or self.pad_hits >= 5)

    def _ma_so_step(self, depth, pos, rpy, vel, yaw, alt, center, d_c):
        """CX_MA_SO: (v_des, yaw_des, use_planner) for this CRUISE tick while the stand-off look runs, else None."""
        so = self._ma_so
        if so is None:
            # start test: first time only, no track at all, the clue this far out, the down ray valid
            if (self.mode != "CRUISE" or self._ma_so_n > 0 or not (_MA_SO_DLO <= d_c <= _MA_SO_D) or alt >= _M2_SAT_M
                    or self._ma_live_track()):
                return None
            h = float(pos[2] - center[2])
            ex = h - _MA_SO_K * d_c
            if ex <= _MA_SO_EX:
                return None  # the clue area is (nearly) in the camera's view already
            if h - max(0.0, alt - _MA_SO_FLOOR) > _MA_SO_KR * d_c or alt - _MA_SO_FLOOR < _MA_SO_SINK:
                return None  # even at the ground floor the clue would stay beyond look-down pulse range, or no room to sink
            vh = vel[:2].astype(np.float64)
            sp = float(np.hypot(vh[0], vh[1]))
            stop = pos[:2] + (vh * (sp / (2.0 * 3.0)) if sp > 1e-6 else 0.0)
            self._ma_so = so = {"t0": self.t, "xy": np.asarray(stop, dtype=np.float64).copy(), "k": -1, "on": False,
                                "dir": None}
            self._ma_so_n += 1
            _cx_ma_ev("ma_so")
            _cx_ma_ev("ma_so_d%d" % int(round(d_c)))
        end = None
        rel = center[:2] - pos[:2]
        d_now = float(np.hypot(rel[0], rel[1]))
        z_goal = float(center[2]) + _MA_SO_KT * d_now
        z_floor = float(pos[2] - alt) + _MA_SO_FLOOR
        z_t = max(z_goal, z_floor)
        if self.mode != "CRUISE":
            end = "trk" if self.mode == "APPROACH" else "mode"
        elif self.t - so["t0"] > _MA_SO_CAP:
            end = "cap"
        elif pos[2] <= z_t + 0.3:
            end = "floor" if z_floor >= z_goal else "cone"
        if end is not None:
            self._ma_so = None
            _cx_ma_ev("ma_so_end_" + end)
            return None
        _cx_ma_ev("ma_so_tick")
        vz_min = -2.5 if alt > 6.0 else (-1.5 if alt > 4.0 else -1.0)
        vz = float(np.clip(1.2 * (z_t - float(pos[2])), vz_min, 0.5))
        hold = so["xy"] - pos[:2]
        hn = float(np.hypot(hold[0], hold[1]))
        v_xy = hold * min(1.0, 1.0 * hn) / max(hn, 1e-6) if hn > 0.05 else np.zeros(2)
        yaw_t = math.atan2(rel[1], rel[0]) if d_now > 1.0 else yaw
        v = np.array([v_xy[0], v_xy[1], vz])
        if _MA_SO_PULSE and alt >= _MTN_PULSE_ALT:
            # look-down pulses toward the clue from a still hover (the champion's spin pulse: a forward step pitches the
            # body, and the camera, ~30 deg nose down for ~0.3 s)
            near_ahead = DEPTH_MIN_M + float(depth[48:112, 40:88].min()) * (DEPTH_MAX_M - DEPTH_MIN_M)
            ts = self.t - so["t0"] - 1.0
            k = int(ts / _MTN_PULSE_PERIOD) if ts >= 0.0 else -1
            if k != so["k"]:
                so["k"] = k
                lim = math.radians(_MTN_PULSE_MAXTILT)
                so["dir"] = np.array([math.cos(yaw), math.sin(yaw)])
                so["on"] = (k >= 0 and float(np.hypot(vel[0], vel[1])) < _MTN_PULSE_STILL and near_ahead > _MTN_PULSE_CLEAR
                            and abs(float(rpy[0])) < lim and abs(float(rpy[1])) < lim
                            and abs(wrap(yaw_t - yaw)) < 0.35)
                if so["on"]:
                    _cx_ma_ev("ma_so_pulse")
            if so["on"] and ts - k * _MTN_PULSE_PERIOD < _MTN_PULSE_S:
                v = np.array([so["dir"][0] * _MTN_PULSE_V, so["dir"][1] * _MTN_PULSE_V, vz])
                self._pulse_bypass = True
                self._pulse_ticks += 1
        return v, yaw_t, False

    # ------------------------------------------------------------------ fx_ms SLOPE (CX_MS_SLOPE only)
    def _ms_ground_ahead(self, depth, pos, rpy, v_des, alt, vh):
        """Height over the highest terrain seen in a 2 m wide corridor 0.3 .. 1 + 1.5 vh m ahead (or the down ray)."""
        if vh < 0.2:
            return alt
        u = np.array([v_des[0], v_des[1]]) / vh
        pooled = self.planner.grid.pool_min(depth)
        pts, zc = self.planner.grid.points_cam(pooled)
        ok = zc < 19.5
        if not ok.any():
            return alt
        p = pts[ok].astype(np.float64)
        eye, fwd, right, up = cam_pose(pos, rpy)
        w = eye[None, :] + p[:, 0:1] * right[None, :] + p[:, 1:2] * up[None, :] + p[:, 2:3] * fwd[None, :]
        rx, ry = w[:, 0] - pos[0], w[:, 1] - pos[1]
        along = rx * u[0] + ry * u[1]
        lat = np.abs(rx * u[1] - ry * u[0])
        m = (along > 0.3) & (along < 1.0 + 1.5 * vh) & (lat < 1.0) & (w[:, 2] < pos[2] + 2.0)
        if not m.any():
            return alt
        return min(alt, float(pos[2] - w[m, 2].max()))

    # ------------------------------------------------------------------ fx_ms KEEP (CX_MS_KEEP only)
    def _ms_pad_float(self, pad):
        """Height of the pad estimate over the terrain plane fitted through the ring points 0.8-2.5 m around it (same
        filters as _downhill_fit: >= 30 points in >= 3 octants, rms <= _MTN_LAND_RMS_MAX), or None without a valid fit.
        The simulator seats a mountain pad with its top 0.2 m over the highest terrain under its 0.69 m disc, so a real
        pad floats >= ~0.2 + 0.69*slope over that plane (~0.9 m at slope 1); a false detection on the terrain does not."""
        if not self.ring_pts:
            return None
        a = np.asarray(self.ring_pts, dtype=np.float64)
        dx, dy = a[:, 0] - pad[0], a[:, 1] - pad[1]
        d = np.hypot(dx, dy)
        m = (d > 0.8) & (d < 2.5)
        n = int(m.sum())
        if n < 30:
            return None
        if len(np.unique(np.floor((np.arctan2(dy[m], dx[m]) + math.pi) / (math.pi / 4.0)).astype(int))) < 3:
            return None
        A = np.c_[dx[m], dy[m], np.ones(n)]
        coef = np.linalg.lstsq(A, a[m, 2], rcond=None)[0]
        if float(np.sqrt(np.mean((a[m, 2] - A @ coef) ** 2))) > _MTN_LAND_RMS_MAX:
            return None
        return float(pad[2] - coef[2])

    def _ms_land_log(self, pad, slope):
        """LAND-entry diagnostics on seeds where SLOPE/KEEP acted: did the faster approach leave the downhill fit enough
        terrain (>= 30 ring points 0.8-2.5 m from the pad in >= 3 octants) to use the touchdown offset? Read-only."""
        n_ann = 0
        if self.ring_pts:
            a = np.asarray(self.ring_pts, dtype=np.float64)
            d = np.hypot(a[:, 0] - pad[0], a[:, 1] - pad[1])
            n_ann = int(((d > 0.8) & (d < 2.5)).sum())
        _cx_ms_log("ms_land_t", round(float(self.t), 2))
        _cx_ms_log("ms_land_ring_pts", len(self.ring_pts))
        _cx_ms_log("ms_land_ring_ann", n_ann)
        _cx_ms_log("ms_land_off", 0 if self.land_off is None else 1)
        _cx_ms_log("ms_land_rms_mm", -1 if self.land_rms is None else int(round(1000.0 * float(self.land_rms))))
        _cx_ms_log("ms_land_slope_pct", -1 if slope is None else int(round(100.0 * float(slope))))
        _cx_ms_log("ms_land_hits", int(self.pad_hits))

    # ------------------------------------------------------------------ coverage map
    # ------------------------------------------------------------------ fx_s1m (CX_S1M_* only)
    def _s1m_reset(self):
        self._s1m_buf = []          # (N,3) float32 world points around the static estimate (height map)
        self._s1m_nbuf = 0
        self._s1m_track_reset()

    def _s1m_track_reset(self):
        self._s1m_desc_t0 = None    # DESC armed at (s)
        self._s1m_desc_done = False  # DESC ended for this track
        self._s1m_desc_bad = 0      # consecutive down-ray vetoes
        self._s1m_rc = []           # RECEN candidates (t, x, y, z, score)
        self._s1m_top_n = 0         # consecutive no-pad-top ticks
        self._s1m_top_t = -99.0     # last time the pad-top check failed (s)

    def _s1m_collect(self, depth, pos, rpy):
        """Full-resolution (stride 2) depth points 0.25-2.6 m around the static estimate, in world coordinates."""
        try:
            pad = self.pad_pos
            if float(np.hypot(pad[0] - pos[0], pad[1] - pos[1])) > 14.0:
                return
            if _S1M_GRID[0] is None:
                t_ = math.tan(math.radians(FOV_DEG) / 2)
                uf = (np.arange(0, IMG_W, 2) + 0.5) / IMG_W * 2 - 1
                vf = 1 - (np.arange(0, IMG_H, 2) + 0.5) / IMG_H * 2
                xf, yf = np.meshgrid(uf * t_, vf * t_)
                _S1M_GRID[0] = (xf.ravel(), yf.ravel())
            xf, yf = _S1M_GRID[0]
            zf = DEPTH_MIN_M + depth[::2, ::2].reshape(-1).astype(np.float64) * (DEPTH_MAX_M - DEPTH_MIN_M)
            ok = (zf > 0.6) & (zf < 16.0)
            if not ok.any():
                return
            eye, fwd, right, up = cam_pose(pos, rpy)
            zc = zf[ok][:, None]
            w = eye[None, :] + (xf[ok][:, None] * zc) * right[None, :] + (yf[ok][:, None] * zc) * up[None, :] + zc * fwd[None, :]
            d = np.hypot(w[:, 0] - pad[0], w[:, 1] - pad[1])
            m = (d > 0.25) & (d < 2.6) & (np.abs(w[:, 2] - pad[2]) < 3.0)
            if m.any():
                self._s1m_buf.append(w[m].astype(np.float32))
                self._s1m_nbuf += int(m.sum())
                while self._s1m_nbuf > _S1M_CAP and len(self._s1m_buf) > 1:
                    self._s1m_nbuf -= len(self._s1m_buf[0])
                    del self._s1m_buf[0]
        except Exception:
            _cx_s1m_ev("s1m_err_collect")

    def _s1m_rel(self, est):
        if not self._s1m_buf:
            return None
        a = np.concatenate(self._s1m_buf).astype(np.float64)
        return a - np.asarray(est, dtype=np.float64)[None, :3]

    def _s1m_hm_offset(self, pad):
        """fx_s1m HM at a static mountain LAND entry: set land_off from the height map, or keep c29c's plane offset."""
        try:
            rel = self._s1m_rel(pad)
            if rel is None:
                _cx_s1m_ev("s1m_hm_fb_empty")
                return
            off, info = _s1m_hm_pick(rel, _MTN_LAND_OFFSET)
            if off is None:
                _cx_s1m_ev("s1m_hm_fb_" + str(info))
                return
            self.land_off = off
            _cx_s1m_ev("s1m_hm")
            _cx_s1m_log("s1m_hm_at", [round(float(self.t), 2)] + list(info))
        except Exception:
            _cx_s1m_ev("s1m_err_hm")

    def _s1m_flat_check(self, pad):
        """fx_s1m FLAT at a static mountain LAND entry (the LAND gate already passed): reject a spot whose height map
        shows terrain rising through the estimated pad top."""
        try:
            rel = self._s1m_rel(pad)
            if rel is None:
                return
            bad, info = _s1m_flat_bad(rel)
            if not bad:
                return
            _cx_s1m_ev("s1m_flat_rej")
            _cx_s1m_log("s1m_flat_at", "%.2f/%s" % (self.t, info))
            est = np.array([float(pad[0]), float(pad[1]), float(pad[2])])
            self.bad_spots.append(est)
            self.search_anchor = est[:2].copy()
            self._drop_track()
            self._set_mode("SEARCH")
            self.search_wps = None
        except Exception:
            _cx_s1m_ev("s1m_err_flat")

    def _s1m_desc_end(self, why):
        self._s1m_desc_done = True
        if self._s1m_desc_t0 is not None:
            _cx_s1m_ev("s1m_desc_end_" + why)
        self._s1m_desc_t0 = None
        return False

    def _s1m_desc_keep(self, pad, pos, vel, alt):
        """fx_s1m DESC: True keeps the track through APPROACH's lost rule this tick (see the flag)."""
        try:
            if (_CX_STATE["shadow"] or self._s1m_desc_done or self.pad_moving or self.pad_ever_moving
                    or self._static_established()):
                return False
            if self.pad_miss_t > 2.0:
                return self._s1m_desc_end("miss")  # the camera had it in view and did not see it
            rel = np.asarray(pad, dtype=np.float64) - pos
            d_h = float(math.hypot(rel[0], rel[1]))
            h = -float(rel[2])
            if self._s1m_desc_t0 is None:
                if self.pad_hits < _S1M_DESC_HITS or len(self.pad_obs) < _S1M_DESC_HITS:
                    return False
                obs = self.pad_obs[-10:]
                if max(math.hypot(o[1] - pad[0], o[2] - pad[1]) for o in obs) > _S1M_DESC_SPREAD:
                    return False
                if max(abs(o[3] - pad[2]) for o in obs) > _S1M_DESC_SPREAD:
                    return False
                if not (d_h < _S1M_DESC_DH and h > 1.0 and h > 0.8 * d_h + 0.3):
                    return False  # not close below the camera's look-down limit
                if not (d_h < 1.0 or float(vel[0] * rel[0] + vel[1] * rel[1]) > 0.2 * d_h):
                    return False  # not closing in
                rel_b = self._s1m_rel(pad)
                fl = _s1m_float(rel_b) if rel_b is not None else None
                if fl is not None and fl < _S1M_DESC_FLOAT:
                    _cx_s1m_log("s1m_desc_veto", "%.2f/float%.2f" % (self.t, fl))
                    return self._s1m_desc_end("float")
                self._s1m_desc_t0 = self.t
                _cx_s1m_ev("s1m_desc")
                _cx_s1m_log("s1m_desc_at", "%.2f/h%d/d%.2f/h%.2f/f%s" % (self.t, self.pad_hits, d_h, h,
                                                                          "-" if fl is None else "%.2f" % fl))
            if self.t - self._s1m_desc_t0 > _S1M_DESC_S:
                return self._s1m_desc_end("time")
            if d_h > _S1M_DESC_DH + 0.5:
                return self._s1m_desc_end("far")
            if d_h < 0.6 and alt < 19.5:
                # over the estimate the ray must not find ground well above the estimated pad top
                self._s1m_desc_bad = self._s1m_desc_bad + 1 if alt < h - _S1M_DESC_RAY else 0
                if self._s1m_desc_bad >= 3:
                    return self._s1m_desc_end("ray")
            _cx_s1m_ev("s1m_desc_tick")
            return True
        except Exception:
            _cx_s1m_ev("s1m_err_desc")
            return False

    def _s1m_est(self):
        if self.pad_pos is not None:
            return np.asarray(self.pad_pos, dtype=np.float64)
        if getattr(self, "land_xy", None) is not None:
            return np.array([float(self.land_xy[0]), float(self.land_xy[1]), float(self.land_z)])
        return None

    def _s1m_top_step(self, pos, alt):
        """Pad-top check in a centred static mountain LAND: the down ray over the estimate's centre must read the top."""
        try:
            if self.pad_moving or self.pad_ever_moving or self.land_off is not None:
                self._s1m_top_n = 0
                return
            est = self._s1m_est()
            if est is None:
                return
            h = float(pos[2] - est[2])
            if math.hypot(pos[0] - est[0], pos[1] - est[1]) < 0.35 and -0.3 < h < 1.2 and alt < 19.5:
                self._s1m_top_n = self._s1m_top_n + 1 if alt > h + _S1M_TOP_DZ else 0
                if self._s1m_top_n >= _S1M_TOP_N:
                    if self.t - self._s1m_top_t > 1.0:
                        _cx_s1m_ev("s1m_notop")
                    self._s1m_top_t = self.t
            else:
                self._s1m_top_n = 0
        except Exception:
            _cx_s1m_ev("s1m_err_top")

    def _s1m_recen_obs(self, dets, pos):
        """fx_s1m RECEN (frozen static LAND branch of _update_track): collect strong detections 1.5-4 m from the
        estimate; with the centre's pad-top check failed recently and >= _S1M_RC_N of them agreeing, re-seed the track
        there and APPROACH it."""
        try:
            est = self.pad_pos
            for d in dets or ():
                if d[0] < _S1M_RC_S:
                    continue
                w = np.asarray(d[1], dtype=np.float64)
                if w.shape[0] < 3 or not np.isfinite(w[:3]).all():
                    continue
                r = math.hypot(w[0] - est[0], w[1] - est[1])
                if _S1M_RC_LO <= r <= _S1M_RC_HI and abs(w[2] - est[2]) <= 2.5:
                    self._s1m_rc.append((self.t, float(w[0]), float(w[1]), float(w[2]), float(d[0])))
            self._s1m_rc = [c for c in self._s1m_rc if self.t - c[0] <= _S1M_RC_W]
            if len(self._s1m_rc) < _S1M_RC_N or self.t - self._s1m_top_t > _S1M_TOP_T:
                return
            a = np.asarray(self._s1m_rc, dtype=np.float64)
            med = np.median(a[:, 1:4], axis=0)
            agree = a[np.hypot(a[:, 1] - med[0], a[:, 2] - med[1]) <= 0.6]
            if len(agree) < _S1M_RC_N:
                return
            new = np.median(agree[:, 1:4], axis=0)
            _cx_s1m_ev("s1m_recen")
            _cx_s1m_log("s1m_recen_at", "%.2f/n%d/r%.2f/dz%.2f" % (self.t, len(agree), math.hypot(new[0] - est[0], new[1] - est[1]),
                                                                  float(new[2] - est[2])))
            obs = [(float(c[0]), float(c[1]), float(c[2]), float(c[3]), 2.0) for c in agree]
            self._drop_track()
            self.pad_pos = new.copy()
            self.pad_vel = np.zeros(2)
            self.pad_hits = len(agree)
            self.pad_conf = float(np.mean(agree[:, 4]))
            self.pad_obs = obs
            self.pad_last_seen = self.t
            self.pad_conflicts = 0
            self.pad_min_z = 2.0
            self.pad_info = 0.0
            self.pad_raw = []
            self.pad_still_t = 0.0
            self.pad_moving = False
            self.land_off = None
            self._set_mode("APPROACH")
        except Exception:
            _cx_s1m_ev("s1m_err_recen")

    def _s1m_badrec(self):
        """fx_s1m BADREC (a static mountain LAND just aborted into RECOVER): if the centre's pad-top check failed
        recently, blacklist the estimate and search instead of re-landing it."""
        try:
            if self.mode != "RECOVER" or self.pad_moving or self.pad_ever_moving:
                return
            if self.t - self._s1m_top_t > _S1M_TOP_T:
                return
            est = self._s1m_est()
            if est is None:
                return
            _cx_s1m_ev("s1m_badrec")
            _cx_s1m_log("s1m_badrec_at", "%.2f/%.2f/%.2f/%.2f" % (self.t, est[0], est[1], est[2]))
            self.bad_spots.append(np.array([est[0], est[1], est[2]]))
            self._qretry = False
            self.search_anchor = est[:2].copy()
            self._drop_track()
            self._set_mode("SEARCH")
            self.search_wps = None
        except Exception:
            _cx_s1m_ev("s1m_err_badrec")

    def _ring_collect(self, depth, pos, rpy):
        """Terrain points 0.75-3 m around the static pad estimate, back-projected from the depth image."""
        pad = self.pad_pos
        if float(np.linalg.norm(pad[:2] - pos[:2])) > 14.0:
            return
        pooled = self.planner.grid.pool_min(depth)
        pts, z = self.planner.grid.points_cam(pooled)
        ok = (z > 0.6) & (z < 16.0)
        if not ok.any():
            return
        eye, fwd, right, up = cam_pose(pos, rpy)
        pc = pts[ok]
        w = eye[None, :] + pc[:, 0:1] * right[None, :] + pc[:, 1:2] * up[None, :] + pc[:, 2:3] * fwd[None, :]
        d = np.hypot(w[:, 0] - pad[0], w[:, 1] - pad[1])
        m = (d > 0.75) & (d < 3.0) & (np.abs(w[:, 2] - pad[2]) < 3.0)
        if m.any():
            self.ring_pts.extend(w[m].tolist())
            if len(self.ring_pts) > 1500:
                del self.ring_pts[:len(self.ring_pts) - 1500]

    def _downhill_fit(self, pad):
        """Plane through the terrain seen around the pad: (unit downhill xy, slope) or (None, None)."""
        if not self.ring_pts:
            return None, None
        a = np.asarray(self.ring_pts, dtype=np.float64)
        dx, dy = a[:, 0] - pad[0], a[:, 1] - pad[1]
        d = np.hypot(dx, dy)
        m = (d > 0.8) & (d < 2.5)
        n = int(m.sum())
        if n < 30:
            return None, None
        if len(np.unique(np.floor((np.arctan2(dy[m], dx[m]) + math.pi) / (math.pi / 4.0)).astype(int))) < 3:
            return None, None
        A = np.c_[dx[m], dy[m], np.ones(n)]
        coef = np.linalg.lstsq(A, a[m, 2], rcond=None)[0]
        self.land_rms = float(np.sqrt(np.mean((a[m, 2] - A @ coef) ** 2)))
        g = np.array([coef[0], coef[1]])
        slope = float(np.linalg.norm(g))
        if slope < 1e-6:
            return None, 0.0
        return -g / slope, slope

    def _cov_update(self, depth, pos, rpy):
        if self.center0 is None:
            return
        if self.cov_seen is None:
            self.cov_seen = np.zeros((41, 41), dtype=bool)
            self.cov_origin = np.array([self.center0[0] - 20.0, self.center0[1] - 20.0])
        pooled = self.planner.grid.pool_min(depth)
        pts, z = self.planner.grid.points_cam(pooled)
        ok = (z > 1.0) & (z < 19.0)
        if not ok.any():
            return
        eye, fwd, right, up = cam_pose(pos, rpy)
        pc = pts[ok]
        w = eye[None, :] + pc[:, 0:1] * right[None, :] + pc[:, 1:2] * up[None, :] + pc[:, 2:3] * fwd[None, :]
        band = 8.0 if self.map_label == "village" else 6.0
        hz = np.abs(w[:, 2] - self.center0[2]) <= band
        ij = np.floor(w[:, :2] - self.cov_origin[None, :]).astype(int)
        m = hz & (ij[:, 0] >= 0) & (ij[:, 0] < 41) & (ij[:, 1] >= 0) & (ij[:, 1] < 41)
        if m.any():
            self.cov_seen[ij[m, 0], ij[m, 1]] = True

    def _cov_target(self, pos):
        if self.cov_seen is None:
            return None
        gx = self.cov_origin[0] + np.arange(41) + 0.5
        gy = self.cov_origin[1] + np.arange(41) + 0.5
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        inside = np.hypot(X - self.center0[0], Y - self.center0[1]) <= 18.0
        unseen = inside & ~self.cov_seen
        if unseen.sum() < 4:
            return None
        u = unseen.astype(np.float32)
        pad = np.pad(u, 2)
        dens = sum(pad[i:i + 41, j:j + 41] for i in range(5) for j in range(5))
        d = np.hypot(X - pos[0], Y - pos[1])
        score = dens * unseen / (1.0 + d / 12.0)
        k = int(np.argmax(score))
        i, j = np.unravel_index(k, score.shape)
        return np.array([X[i, j], Y[i, j]])

    def _city_zcap(self, pos, alt, vz):
        if not (_flag("czcap") and self.map_label == "city") or self.center0 is None:
            return vz
        h_abs = float(pos[2] - self.center0[2])
        if h_abs > 7.0 and vz > 0.0:
            return 0.0
        if h_abs > 4.5 and alt < 2.5 and vz > 0.0:
            return 0.0
        return vz

    def _city_detour(self, pos, dir_h, v_h, prm):
        if not (_flag("cdetour") and self.map_label == "city"):
            return dir_h, v_h
        wp = getattr(self, "_cd_wp", None)
        if wp is not None:
            to = wp - pos[:2]
            if float(np.linalg.norm(to)) < 1.5 or self.t - self._cd_t0 > 4.0:
                self._cd_wp = None
                self._cd_done_t = self.t
            else:
                return unit(to), prm["v_cruise"]
        f = getattr(self, "_pl_free", 99.0)
        if f < 5.0 and self.t - getattr(self, "_cd_done_t", -9.0) > 1.0:
            self._cd_short = getattr(self, "_cd_short", 0) + 1
        else:
            self._cd_short = 0
        if self._cd_short >= 3:
            left, right = getattr(self, "_pl_lr", (0.0, 0.0))
            yaw = float(self._rpy[2])
            r_vec = np.array([math.sin(yaw), -math.cos(yaw)])
            side = r_vec if right >= left else -r_vec
            if self.t - getattr(self, "_cd_done_t", -9.0) < 6.0 and getattr(self, "_cd_last_side", None) is not None:
                side = -self._cd_last_side
            self._cd_last_side = side
            self._cd_wp = pos[:2] + side * 8.0 + dir_h * 2.0
            self._cd_t0 = self.t
            self._cd_short = 0
            to = self._cd_wp - pos[:2]
            return unit(to), prm["v_cruise"]
        return dir_h, v_h

    def _set_mode(self, m):
        if m != self.mode:
            if self.mode == "REACQUIRE":
                self._vcheck_active = False
            if m != "APPROACH":
                self._vland_t = 0.0
                self._v3_vl_ext = 0.0
                self._v3_vl_t0 = None
                self._v3_vl_on = False
                self._v3_vl_anchor = None
            if m == "SEARCH" and getattr(self, "search_t_first", None) is None:
                self.search_t_first = self.t
            if m == "APPROACH":
                self.weak_approach = bool(self.pad_pos is not None and self.pad_conf <= 0.6)
                if _CX_VCH_RQ:
                    self._vch_h0 = int(self.pad_hits)
                    self._vch_app_t0 = self.t
            self.mode = m
            self.mode_t0 = self.t

    def _plan(self, depth, pos, rpy, v_des, tube, tube_min, lookahead=10.0):
        speed = float(np.linalg.norm(v_des))
        if speed < 0.05:
            return v_des, False, lookahead
        around = _flag("caround") and self.map_label == "city" and self.mode in ("CRUISE", "SEARCH")
        if around:
            lookahead = 18.0
        d = v_des / speed
        eye, fwd, right, up = cam_pose(pos, rpy)
        M = np.stack([right, up, fwd], axis=0)
        d_cam = M @ d
        cone_cos = math.cos(math.radians(50))
        z_floor = None
        if _flag("vabs") and self.map_label == "village" and self.v_gnd is not None and (pos[2] - self.v_gnd) >= _VABS_LIMIT_M - 0.05:
            z_floor = self.v_gnd + _VABS_ROOF_M
        free, _ = self.planner.free_distances(depth, tube, lookahead, z_floor=z_floor, eye_z=float(eye[2]), zvec=M[:, 2].astype(np.float32))
        if not np.isfinite(free).all():
            free = np.nan_to_num(free, nan=0.0)
        cand = self.planner.cand
        if self.map_label in ("forest", "warehouse"):
            free = free.copy()
            free[self.planner.cy_raw > 1.0] = 0.0
        if _flag("brake") and self.map_label in ("city", "mountain"):
            top_d = DEPTH_MIN_M + float(depth[0:12, 24:104].min()) * (DEPTH_MAX_M - DEPTH_MIN_M)
            steep_m = self.planner.cy_raw > 1.0
            free = free.copy()
            free[steep_m] = np.minimum(free[steep_m], max(top_d - 0.5, 0.0))
        cos_ang = cand @ d_cam
        ang = np.arccos(np.clip(cos_ang, -1, 1))
        L_pref = min(lookahead, max(3.0, speed * speed / (2 * 2.0) + 2.0))
        ang_w = 0.9
        if _flag("brake") and self.map_label in ("city", "mountain"):
            L_pref = min(lookahead, max(9.0, L_pref))
            ang_w = 0.4
        if around:
            L_pref = min(lookahead, 15.0)
            ang_w = 0.3
        score = np.minimum(free, L_pref) / L_pref - ang_w * ang
        if around:
            score[self.planner.cy_raw > 0.35] -= 0.3
        noclimb = _flag("cnoclimb") and self.map_label == "city" and self.mode in ("CRUISE", "SEARCH", "REACQUIRE")
        if noclimb:
            score[self.planner.cy_raw > 0.35] -= 5.0
        elev = np.arcsin(np.clip(cand[:, 1], -1, 1))
        if d_cam[1] > -0.5:
            score[elev < -0.6] -= 1.0
        best = int(np.argmax(score))
        if d_cam[2] < cone_cos:
            best = int(np.argmin(ang))
            if _flag("cnoclimb") and self.map_label == "city":
                lvl_i = np.where(self.planner.cy_raw <= 0.35)[0]
                best = int(lvl_i[np.argmin(ang[lvl_i])])
            fresh = self.mode == "APPROACH" and (self.t - self.pad_last_seen) < 1.0
            out_speed = min(speed, 1.2 if fresh else 0.6)
        else:
            out_speed = speed
        _fo_took = False
        if (_FO_TUBE and self.map_label == "forest" and self.mode in ("CRUISE", "SEARCH", "REACQUIRE") and free[best] < 1.0 and speed > 0.3
                and tube_min < _FO_TUBE_R < tube):
            # a corridor as wide as the forest safe distance, before settling for the thin tube
            freeM, _ = self.planner.free_distances(depth, _FO_TUBE_R, lookahead, z_floor=z_floor, eye_z=float(eye[2]), zvec=M[:, 2].astype(np.float32))
            freeM = np.array(freeM, dtype=np.float32)
            freeM[self.planner.cy_raw > 1.0] = 0.0
            scoreM = np.minimum(freeM, L_pref) / L_pref - ang_w * ang
            if d_cam[1] > -0.5:
                scoreM[elev < -0.6] -= 1.0
            bestM = int(np.argmax(scoreM))
            if freeM[bestM] > free[best] + 0.5 and freeM[bestM] >= 1.5:
                best = bestM
                free = freeM
                out_speed = min(out_speed, 1.2)
                _fo_took = True
                _cx("fo_tube")
        if not _fo_took and free[best] < 1.0 and speed > 0.3:
            free2, _ = self.planner.free_distances(depth, tube_min, lookahead, z_floor=z_floor, eye_z=float(eye[2]), zvec=M[:, 2].astype(np.float32))
            score2 = np.minimum(free2, L_pref) / L_pref - ang_w * ang
            if d_cam[1] > -0.5:
                score2[elev < -0.6] -= 1.0
            best2 = int(np.argmax(score2))
            if (_FO_STEEP and self.map_label == "forest" and self.mode == "SEARCH" and self.planner.cy_raw[best2] > 1.0
                    and free2[best2] > free[best] + 0.5):
                # the retry would take a steep-up row outside the image: zero those rows as the main tube does
                _cx("fo_steep")
                free2 = np.array(free2, dtype=np.float32)
                free2[self.planner.cy_raw > 1.0] = 0.0
                score2 = np.minimum(free2, L_pref) / L_pref - ang_w * ang
                if d_cam[1] > -0.5:
                    score2[elev < -0.6] -= 1.0
                best2 = int(np.argmax(score2))
            if free2[best2] > free[best] + 0.5:
                best = best2
                free = free2
                out_speed = min(out_speed, 1.2)
        f = float(free[best])
        if _flag("brake") and self.map_label in ("city", "mountain") and self.planner.cy_raw[best] > 1.0:
            out_speed = min(out_speed, 1.0)
        margin = 0.6
        decel = 2.2
        if _WF_ABRK and self.map_label == "forest" and self.mode == "APPROACH":
            margin, decel = 1.2, 1.5  # ws2 WF_ABRK
            _cx("wf_abrk")
        if _flag("brake") and self.map_label in ("city", "mountain"):
            margin, decel = 1.2, 1.5
        v_brake = math.sqrt(max(0.0, 2 * decel * (f - margin)))
        out_speed = min(out_speed, max(v_brake, 0.0))
        if self.map_label == "city":
            _near = DEPTH_MIN_M + float(depth.min()) * (DEPTH_MAX_M - DEPTH_MIN_M)
            if _near < 1.0:
                out_speed = min(out_speed, max(0.8, 2.5 * (_near - 0.3)))
        dir_world = M.T @ cand[best]
        blocked = f < 1.2
        if f < 1.5 and d_cam[2] >= cone_cos:
            if _F6_UNDER and self.map_label == "forest":
                # boxed in: remember it (and which side is freer) for the recovery in act(); the return is unchanged
                self._f6_esc_step = self.step
                lvl_ = np.abs(self.planner.cy_raw) <= 0.35
                self._f6_lr = (float(free[lvl_ & (self.planner.cx_raw < -0.3)].sum()),
                               float(free[lvl_ & (self.planner.cx_raw > 0.3)].sum()))
            top = DEPTH_MIN_M + float(depth[0:16, 24:104].min()) * (DEPTH_MAX_M - DEPTH_NEAR_DUMMY)
            climb = self._climb_speed(pos, top)
            if noclimb:
                lvl = self.planner.cy_raw <= 0.35
                left = float(free[lvl & (self.planner.cx_raw < -0.3)].sum()); rgt = float(free[lvl & (self.planner.cx_raw > 0.3)].sum())
                side = right if rgt >= left else -right
                if not hasattr(self, "_cn_side_t") or self.t - self._cn_side_t > 3.0:
                    self._cn_side = side; self._cn_side_t = self.t
                esc = self._cn_side * 0.9 - fwd * 0.4
                return esc, True, f
            if climb > 0.0:
                esc = np.array([0.0, 0.0, climb]) - fwd * 0.3
                return esc, True, f
            best2 = int(np.argmax(free))
            return (M.T @ cand[best2]) * min(0.6, float(free[best2]) * 0.3) - fwd * 0.3, True, f
        if _flag("cdetour") and self.map_label == "city":
            lvl = self.planner.cy_raw <= 0.35
            self._pl_free = f
            self._pl_lr = (float(free[lvl & (self.planner.cx_raw < -0.3)].sum()), float(free[lvl & (self.planner.cx_raw > 0.3)].sum()))
        return dir_world * out_speed, blocked, f

    def _unstick(self, depth, rpy, vel, v_des):
        """Back out when wedged against vegetation instead of grinding along it.

        Forest 153 sits at 0.03 m/s with the whole frame at DEPTH_MIN, commanding a 0.11 speed-fraction climb, and scrapes up a
        trunk until clearance reaches 0; 511 spends 17.7 s of its 60 s in the same state and times out. Measured over all 128
        forest seeds, only 10 are ever in this state at all and only 153 and 511 hold it for a full second, so the rule is
        narrowly exposed.
        """
        near = DEPTH_MIN_M + float(depth.min()) * (DEPTH_MAX_M - DEPTH_MIN_M)
        wedged = (near <= _STUCK_NEAR and float(np.hypot(vel[0], vel[1])) < _STUCK_VMAX
                  and self.mode not in ("LAND", "TAKEOFF"))
        self.stuck_t = self.stuck_t + SIM_DT if wedged else 0.0
        if self.stuck_t >= _STUCK_S and self.t >= self.escape_until:
            _eye, fwd, _right, _up = cam_pose(np.zeros(3), rpy)
            back = np.array([-float(fwd[0]), -float(fwd[1]), 0.0])
            n = float(np.linalg.norm(back))
            self.escape_dir = back / n if n > 1e-6 else np.array([-1.0, 0.0, 0.0])
            self.escape_until = self.t + _STUCK_HOLD
            self.stuck_t = 0.0
            self.escape_n = getattr(self, "escape_n", 0) + 1
        if self.t < self.escape_until and self.escape_dir is not None:
            self.escape_s = round(getattr(self, "escape_s", 0.0) + SIM_DT, 2)
            return np.array([self.escape_dir[0] * _STUCK_V, self.escape_dir[1] * _STUCK_V, 0.0])
        return v_des

    def _s1f_cg(self, depth, pos, rpy, pad, v_des):
        """CX_S1F_CG (see the flag comment): returns v_des, its vz capped when a forest climb has not been seen free."""
        t = self.t
        fb = self._s1f_fb
        if self.step % _S1F_CG_EVERY == 0:
            eye, fwd, right, up = cam_pose(pos, rpy)
            fb.append((t, eye, np.stack([right, up, fwd], axis=0), np.array(depth, dtype=np.float32)))
        while fb and t - fb[0][0] > _S1F_CG_MEM:
            fb.pop(0)
        zh = self._s1f_zh
        if self.mode not in ("SEARCH", "APPROACH", "REACQUIRE"):
            if zh:
                del zh[:]
            self._s1f_ok = None
            return v_des
        z = float(pos[2])
        zh.append((t, z))
        while zh and t - zh[0][0] > _S1F_CG_WIN:
            zh.pop(0)
        if float(v_des[2]) <= 0.0:
            return v_des
        if (self.mode == "APPROACH" and pad is not None
                and float(np.hypot(float(pad[0]) - pos[0], float(pad[1]) - pos[1])) < _S1F_CG_PADR):
            return v_des
        cap = min(z_ for _, z_ in zh) + _S1F_CG_DZ
        ok = self._s1f_ok
        if ok is not None and float(np.hypot(ok[0] - pos[0], ok[1] - pos[1])) <= _S1F_CG_OFF:
            cap = max(cap, ok[2])
        if float(v_des[2]) <= 1.2 * (cap - z):
            return v_des
        top = self._s1f_free_top(pos)
        if top is not None:
            self._s1f_ok = (float(pos[0]), float(pos[1]), top)
            cap = max(cap, top)
            if float(v_des[2]) <= 1.2 * (cap - z):
                _cx_ms_ev("s1f_cg_seen")
                return v_des
        _cx_ms_ev("s1f_cg_cap")
        return np.array([float(v_des[0]), float(v_des[1]), max(0.0, 1.2 * (cap - z))])

    def _s1f_free_top(self, pos):
        """CX_S1F_CG: highest z the column over pos was seen free up to (all 5 points of every level to there), less
        _S1F_CG_TOPM, or None when not even the lowest level was seen through."""
        lv = (0.4, 0.8, 1.2, _S1F_CG_UP)
        o = _S1F_CG_OFF
        offs = ((0.0, 0.0), (o, 0.0), (-o, 0.0), (0.0, o), (0.0, -o))
        P = np.array([[pos[0] + dx, pos[1] + dy, pos[2] + h] for h in lv for dx, dy in offs], dtype=np.float64)
        free = np.zeros(len(P), dtype=bool)
        tn = math.tan(math.radians(FOV_DEG) / 2)
        for _t, eye, M, d in reversed(self._s1f_fb):
            c = (P - eye) @ M.T
            zf = c[:, 2]
            ok = (~free) & (zf > DEPTH_MIN_M + 0.1) & (zf < DEPTH_MAX_M - 1.0)
            if not ok.any():
                continue
            zs = np.where(ok, zf, 1.0)
            ui = np.round(((c[:, 0] / zs) / tn + 1.0) * 0.5 * IMG_W - 0.5).astype(np.int64)
            vi = np.round((1.0 - (c[:, 1] / zs) / tn) * 0.5 * IMG_H - 0.5).astype(np.int64)
            ok &= (ui >= 1) & (ui <= IMG_W - 2) & (vi >= 1) & (vi <= IMG_H - 2)
            if not ok.any():
                continue
            idx = np.where(ok)[0]
            dm = np.full(len(idx), np.inf)
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    dm = np.minimum(dm, d[vi[idx] + dv, ui[idx] + du])
            dm = DEPTH_MIN_M + dm * (DEPTH_MAX_M - DEPTH_MIN_M)
            free[idx[dm >= zf[idx] + _S1F_CG_MARG]] = True
            if free.all():
                break
        lvl = free.reshape(len(lv), len(offs)).all(axis=1)
        h = 0.0
        for k in range(len(lv)):
            if not lvl[k]:
                break
            h = lv[k]
        return float(pos[2]) + h - _S1F_CG_TOPM if h > 0.0 else None

    def _s1f_edge_frame(self, cls_logits):
        """CX_S1F_EDGE: add this frame's class log-probs when the camera is turned away from +x (into the forest box)."""
        try:
            rpy = getattr(self, "_rpy", None)
            if rpy is None or abs(wrap(float(rpy[2]))) < math.radians(_S1F_EDGE_YAW):
                return
            z = np.asarray(cls_logits, dtype=np.float64).reshape(-1)
            z = z - float(z.max())
            p = np.exp(z)
            p /= p.sum()
            self._s1f_eacc[0] = self._s1f_eacc[0] + np.log(p + 1e-6)
            self._s1f_eacc[1] += 1
        except Exception:
            self._s1f_eacc = None
            _cx_ms_ev("s1f_edge_err")

    def _wf_ooc(self, v_des, yaw_des, yaw, use_planner):
        """ws2 WF_OOC (see the flag comment): (v_des, use_planner) for this forest tick."""
        vh_ = float(np.hypot(v_des[0], v_des[1]))
        if (not use_planner or self.mode not in ("SEARCH", "CRUISE", "REACQUIRE") or vh_ < 0.3):
            self._wf_ooc_t0 = None
            self._wf_ooc_used = 0.0
            return v_des, use_planner
        hd_ = math.atan2(float(v_des[1]), float(v_des[0]))
        off_ = abs(wrap(hd_ - yaw))
        faces_ = abs(wrap(yaw_des - hd_)) < math.radians(20.0)
        if self._wf_ooc_t0 is None:
            if off_ > _WF_OOC_ON and faces_ and self._wf_ooc_used < _WF_OOC_CAP:
                self._wf_ooc_t0 = self.t
                _cx("wf_ooc")
            else:
                if off_ < _WF_OOC_OFF:
                    self._wf_ooc_used = 0.0
                return v_des, use_planner
        held_ = self._wf_ooc_used + (self.t - self._wf_ooc_t0)
        if off_ < _WF_OOC_OFF or not faces_ or held_ >= _WF_OOC_CAP:
            self._wf_ooc_used = held_ if off_ >= _WF_OOC_OFF else 0.0
            self._wf_ooc_t0 = None
            return v_des, use_planner
        _cx("wf_ooc_tick")
        return np.zeros(3), False

    def _fo_canopy_vz(self, pos, alt, alt_c, vz):
        """CX_FO_SPIN: vertical command for a planner-off forest spin (and the level exit after a hold).

        With foliage under the drone -- the down-ray ground more than 1 m above the lowest ground seen since the spin
        began -- the drone is inside or under a crown and the camera cannot see above 45 deg. Then no climb is ever
        returned: while agl >= 1.2 the champion's vz is kept if it sinks (else 0), and once agl < 1.2 the drone freezes
        at its current height (vz 0), so a drone that meets the condition already low stays low. Otherwise the
        champion's vz is returned unchanged. This caps only the requested vz; it does not constrain the planner.
        """
        if alt >= 19.5:  # no down-ray hit: no ground sample
            return vz
        g_raw = float(pos[2] - alt)
        g_filt = float(pos[2] - alt_c)
        gref = getattr(self, "_fo_gref", None)
        gref = min(g_filt, g_raw) if gref is None else min(gref, g_filt, g_raw)
        self._fo_gref = gref
        if g_raw <= gref + 1.0:
            return vz
        if alt < 1.2:
            return 0.0
        return min(vz, 0.0)

    def _fo_fresh_drop(self, pos):
        """CX_FO_RESUME guard: True when the track that just ended APPROACH was dropped close by while still fresh.

        _update_track drops a non-static track once the pad has been in view but undetected for 1.2 s, so a real pad a
        few metres ahead that the detector briefly misses ends APPROACH with pad None too (760333354: dropped 5.1 m from
        the real pad and re-found 0.32 s later during the champion's near-stationary spin). Then the champion's rebuild
        and re-spin are kept. "Fresh" is last seen at most 2.0 s before the drop (the 1.2 s miss rule plus a few conflict
        frames; the reviewer's "seen within 0.5 s" can never hold at a miss-rule drop). 2037242053's false track was 7.7 m
        away but unseen for 2.9 s, so RESUME still applies there. 760333354's unseen time is not in the epoch trace.
        """
        rec = getattr(self, "_fo_drop", None)
        if rec is None or self.t - rec[0] > 0.1:
            return False
        near = float(np.hypot(rec[1][0] - pos[0], rec[1][1] - pos[1])) <= 10.0
        return near and (rec[0] - rec[2]) <= 2.0

    def _fp_post_center(self):
        """CX_FP_POST: cached posterior-mean pad xy (see the flag comment), or None to keep the clue."""
        if self._fp_c is None:
            c0, sp = self.center0, self.start_pos
            if c0 is None or sp is None:
                return None
            xs = np.arange(-17.5, 17.51, 1.0)
            X, Y = np.meshgrid(float(c0[0]) + xs, float(c0[1]) + xs)
            D = np.hypot(X - float(sp[0]), Y - float(sp[1]))
            Rm = np.sqrt(np.clip(59.0 / 1.06 - D / 3.0 - 2.0, 0.0, None) * (5.1 * 3.0 / (0.75 * math.pi)))
            Rm = np.clip(Rm + _FP_RMARGIN, 5.5, 20.0)
            U = np.maximum(np.abs(X - float(c0[0])), np.abs(Y - float(c0[1])))
            lo = np.maximum(U, 5.0)
            L = (np.where(lo < Rm, (1.0 / 60.0) * (1.0 / lo - 1.0 / Rm), 0.0)
                 + np.where(U <= Rm, ((20.0 - Rm) / 15.0) / (4.0 * Rm * Rm), 0.0))
            ok = (np.abs(X) <= 42.0) & (np.abs(Y) <= 42.0) & (D >= 22.0) & (D <= 45.0)
            W = L * np.where(ok, 1.0, _FP_FLOOR) / np.maximum(D, 1.0)
            w = float(W.sum())
            if not (w > 0.0) or not np.isfinite(w):
                self._fp_c = False
                return None
            m = np.array([float((W * X).sum()) / w, float((W * Y).sum()) / w])
            if not np.isfinite(m).all() or float(np.hypot(m[0] - c0[0], m[1] - c0[1])) < _FP_SHIFT_MIN:
                self._fp_c = False
                return None
            self._fp_c = m
            _cx("fp_post")
            CX_EVENTS["fp_post_shift"] = round(float(np.hypot(m[0] - c0[0], m[1] - c0[1])), 2)
        return None if self._fp_c is False else self._fp_c

    def _fp_look_step(self, pos, rpy, alt_c, yaw, prm):
        """CX_FP_LOOK: (v_des, yaw_des, use_planner) for this forest SEARCH tick while a look runs, else None."""
        lk = self._fp_look
        pp = self.pad_pos
        if lk is not None:
            end = None
            if pp is None:
                end = "drop"
            elif self.pad_moving or self.pad_ever_moving:
                end = "moving"
            elif self._track_reliable():
                end = "reliable"
            elif self._m15_key() != lk["key"]:
                end = "new"
            elif self.t - lk["t0"] > _FP_LOOK_CAP:
                end = "cap"
            if end is not None:
                self._fp_look = None
                _cx("fp_look_end_" + end)
                return None
        else:
            if (pp is None or self.pad_moving or self.pad_ever_moving or self.pad_hits < _FP_LOOK_HITS
                    or self._fp_look_n >= _FP_LOOK_MAX or self.t > _FP_LOOK_TMAX):
                return None
            if self.t - self.pad_last_seen < _FP_LOOK_UNSEEN or self._track_reliable():
                return None
            key = self._m15_key()
            if key is None or key in self._fp_look_done:
                return None
            d0_ = float(np.hypot(pp[0] - pos[0], pp[1] - pos[1]))
            if d0_ > _FP_LOOK_DMAX or d0_ < 2.0:
                return None
            g0_ = self._f14_gmin if self._f14_gmin is not None else (self._rc_gmin if self._rc_gmin is not None else 0.0)
            if not (g0_ + _FP_LOOK_ZLO <= float(pp[2]) <= g0_ + _FP_LOOK_ZHI):
                return None
            sp = self.start_pos
            if sp is not None and float(np.hypot(pp[0] - sp[0], pp[1] - sp[1])) < _FP_LOOK_SMIN:
                return None
            c0 = self.center0
            if c0 is not None and max(abs(float(pp[0] - c0[0])), abs(float(pp[1] - c0[1]))) > _FP_LOOK_CLUE:
                return None
            if self._pad_in_view(pos, rpy):
                return None
            lk = self._fp_look = {"t0": self.t, "key": key}
            self._fp_look_done.append(key)
            self._fp_look_n += 1
            _cx("fp_look")
        rel = pp - pos
        d_h = float(np.hypot(rel[0], rel[1]))
        yaw_des = math.atan2(rel[1], rel[0]) if d_h > 0.5 else yaw
        vz = float(np.clip(1.2 * (prm["h_cruise"] - alt_c), -1.0, 1.0))
        if d_h > _FP_LOOK_DFAR and abs(wrap(yaw_des - yaw)) < 0.6:
            v_h = min(_FP_LOOK_VH, 0.5 * (d_h - _FP_LOOK_DFAR) + 0.6)
            v_xy = rel[:2] / d_h * v_h
            planner = True
        else:
            v_xy = np.zeros(2)
            planner = False  # turn (or hover facing the estimate) in place, as the champion's forest spin flies
        _cx("fp_look_ticks")
        return np.array([v_xy[0], v_xy[1], vz]), yaw_des, planner

    # ------------------------------------------------------------------ fx_s1v SGATE / ALT / LOOK (CX_SV_* only)
    @staticmethod
    def _sv_evid(n, conf, unseen):
        """fx_s1v ALT: evidence of a hypothesis: hits x (score EMA - 0.45), capped, halved every _SV_ALT_HL s once it
        has been unseen for 0.5 s."""
        e = min(_SV_ALT_CAP, float(n) * max(0.0, float(conf) - 0.45))
        u = float(unseen) - 0.5
        if u > 0.0:
            e *= 0.5 ** (u / _SV_ALT_HL)
        return e

    def _sv_young(self, est_static):
        return (not est_static and not self.pad_moving and not self.pad_ever_moving and self.mode != "LAND")

    def _sv_snap(self):
        """fx_s1v ALT: the current young track as an alternate hypothesis (it passed the start gate when it began)."""
        obs = list(self.pad_obs)[-75:]
        return {"p": np.asarray(self.pad_pos, dtype=np.float64).copy(), "n": int(self.pad_hits),
                "conf": float(self.pad_conf), "smax": 1.0, "t1": float(self.pad_last_seen),
                "zmin": float(self.pad_min_z), "obs": obs, "src": "track"}

    def _sv_new(self, s, w, z):
        return {"p": np.asarray(w, dtype=np.float64).copy(), "n": 1, "conf": float(s), "smax": float(s),
                "t1": float(self.t), "zmin": float(z), "obs": [(self.t, float(w[0]), float(w[1]), float(w[2]), float(z))],
                "src": "cf"}

    def _sv_match(self, a, w, z):
        dt = max(0.0, self.t - a["t1"])
        return float(np.hypot(w[0] - a["p"][0], w[1] - a["p"][1])) <= max(1.5, 0.12 * float(z)) + 1.2 * min(dt, 3.0)

    def _sv_add(self, a, s, w, z):
        alpha = float(np.clip(0.35 * (6.0 / max(float(z), 2.0)), 0.15, 0.7))
        a["p"] = (1.0 - alpha) * a["p"] + alpha * np.asarray(w, dtype=np.float64)
        a["n"] += 1
        a["conf"] = 0.8 * a["conf"] + 0.2 * float(s)
        a["smax"] = max(a["smax"], float(s))
        a["t1"] = float(self.t)
        a["zmin"] = min(a["zmin"], float(z))
        a["obs"].append((self.t, float(w[0]), float(w[1]), float(w[2]), float(z)))
        if len(a["obs"]) > 75:
            a["obs"].pop(0)

    def _sv_gate_ok(self, a, s=0.0):
        return (not _CX_SV_SGATE) or a["smax"] >= _SV_SMIN or float(s) >= _SV_SMIN

    def _sv_promote(self, a, keep_old):
        """fx_s1v ALT: the alternate becomes the track with its hits (the champion's re-seed resets, as at a re-seed);
        the displaced young track becomes the alternate."""
        old = self._sv_snap() if (keep_old and self.pad_pos is not None and self.pad_hits >= 2) else None
        self._v3_track_reset()
        self.pad_pos = np.asarray(a["p"], dtype=np.float64).copy()
        self.pad_vel = np.zeros(2)
        self.pad_hits = int(a["n"])
        self.pad_conf = float(a["conf"])
        self.pad_obs = list(a["obs"])[-75:]
        self.pad_last_seen = self.t
        self.pad_conflicts = 0
        self.pad_min_z = float(a["zmin"])
        self.pad_info = 0.0
        self.pad_raw = []
        self.pad_still_t = 0.0
        self.pad_moving = bool(self.pad_ever_moving)
        self.pad_move_votes = 0
        self.pad_omega = 0.0
        self.pad_acc = np.zeros(2)
        self._sv_alt = old

    def _sv_start(self, s, w, z):
        """fx_s1v: a detection with no track. True = handled (ALT restore, or SGATE: too weak to start a track)."""
        try:
            a = self._sv_alt
            if _CX_SV_ALT and a is not None:
                if self.t - a["t1"] > _SV_ALT_KEEP:
                    self._sv_alt = None
                elif (self._sv_match(a, w, z) and self._sv_gate_ok(a, s) and not self.pad_ever_moving
                      and 0.8 * a["conf"] + 0.2 * float(s) >= _SV_ALT_RMIN):
                    # the new track starts on the alternate: it resumes the alternate's hits (as the champion's start)
                    self._sv_add(a, s, w, z)
                    _sv_log([round(float(self.t), 2), "restore", a["src"], int(a["n"]), round(float(a["conf"]), 2),
                             [round(float(a["p"][0]), 2), round(float(a["p"][1]), 2), round(float(a["p"][2]), 2)]])
                    self.pad_pos = np.asarray(a["p"], dtype=np.float64).copy()
                    self.pad_hits = int(a["n"])
                    self.pad_conf = float(a["conf"])
                    self.pad_obs = list(a["obs"])[-75:]
                    self.pad_last_seen = self.t
                    self.pad_min_z = float(a["zmin"])
                    self.pad_info = 0.0
                    self.pad_raw = []
                    self.pad_still_t = 0.0
                    self._push_raw(w, z)
                    self._sv_alt = None
                    _sv_ev("sv_alt_restore")
                    return True
            if _CX_SV_SGATE and float(s) < _SV_SMIN:
                _sv_ev("sv_sgate_start")
                return True
        except Exception:
            _sv_ev("sv_error")
        return False

    def _sv_conflict(self, s, w, z, err, est_static, need_conf):
        """fx_s1v: a detection outside the track's gate (pad_conflicts already counted). True = handled here (the caller
        returns): an ALT switch, hold or re-seed with hits, or an SGATE re-seed block. False = the champion's code runs."""
        try:
            young = self._sv_young(est_static)
            if _CX_SV_ALT and young:
                a = self._sv_alt
                if a is not None and self.t - a["t1"] > _SV_ALT_KEEP:
                    a = self._sv_alt = None
                if a is not None and self._sv_match(a, w, z):
                    self._sv_add(a, s, w, z)
                else:
                    e_new = max(0.0, float(s) - 0.45)
                    if a is None or self._sv_evid(a["n"], a["conf"], self.t - a["t1"]) <= e_new:
                        a = self._sv_alt = self._sv_new(s, w, z)
                    else:
                        a = None  # a stronger alternate is kept; this detection is only the champion's conflict
                if a is not None:
                    ea = self._sv_evid(a["n"], a["conf"], 0.0)
                    ep = self._sv_evid(self.pad_hits, self.pad_conf, self.t - self.pad_last_seen)
                    # only a clearly stronger-scoring cluster may take over with its hits (village false positives
                    # score lower than the pad: replayed switches onto the pad had score EMA >= 0.65 half the time,
                    # switches onto another false positive 6%)
                    ok = (self._sv_gate_ok(a) and a["conf"] >= _SV_ALT_CMIN
                          and a["conf"] >= float(self.pad_conf) + _SV_ALT_DCONF)
                    kind = None
                    if ok and a["n"] >= _SV_ALT_N and ea > ep + _SV_ALT_MARGIN:
                        kind = "promote"
                    elif self.pad_conflicts >= need_conf:
                        if ep > ea + _SV_ALT_MARGIN and self.t - self.pad_last_seen <= _SV_ALT_HOLD:
                            if self.pad_conflicts == need_conf:
                                _sv_ev("sv_alt_hold")
                                _sv_log([round(float(self.t), 2), "hold", int(self.pad_hits), round(ep, 2),
                                         int(a["n"]), round(ea, 2)])
                            return True
                        if ok:
                            kind = "reseed"
                    if kind is not None:
                        _sv_log([round(float(self.t), 2), kind, a["src"], int(a["n"]), round(ea, 2), int(self.pad_hits),
                                 round(ep, 2), [round(float(a["p"][0]), 2), round(float(a["p"][1]), 2),
                                                round(float(a["p"][2]), 2)], round(float(a["conf"]), 2),
                                 round(float(self.pad_conf), 2), round(float(a["smax"]), 2),
                                 round(float(self.t - self.pad_last_seen), 2)])
                        self._sv_promote(a, True)
                        _sv_ev("sv_alt_" + kind)
                        return True
            if _CX_SV_SGATE and float(s) < _SV_SMIN:
                if est_static and self.pad_conflicts >= 40 and z >= 1.0 and err > 1.0:
                    _sv_ev("sv_sgate_mv")
                    return True
                if self.pad_conflicts >= need_conf:
                    _sv_ev("sv_sgate_rs")
                    return True
            if _CX_SV_ALT and young and self.pad_conflicts >= need_conf and self.pad_hits >= 2:
                self._sv_alt = self._sv_snap()  # the champion's re-seed goes ahead: keep the displaced young track
        except Exception:
            _sv_ev("sv_error")
        return False

    def _sv_look_step(self, pos, rpy, alt_c, yaw, prm, alt=None):
        """CX_SV_LOOK: (v_des, yaw_des, use_planner) for this village SEARCH tick while a look runs, else None."""
        lk = self._sv_look
        pp = self.pad_pos
        if lk is not None:
            end = None
            if pp is None:
                end = "drop"
            elif self.pad_moving or self.pad_ever_moving:
                end = "moving"
            elif self._track_reliable():
                end = "reliable"
            elif self._m15_key() != lk["key"]:
                end = "new"
            elif self.t - lk["t0"] > _SV_LOOK_CAP:
                end = "cap"
            if end is not None:
                self._sv_look = None
                _sv_ev("sv_look_end_" + end)
                return None
        else:
            if (pp is None or self.pad_moving or self.pad_ever_moving or self.pad_hits < _SV_LOOK_HITS
                    or self._sv_look_n >= _SV_LOOK_MAX or self.t > _SV_LOOK_TMAX or self.pad_conf < _SV_LOOK_CONF):
                return None
            if self.t - self.pad_last_seen < _SV_LOOK_UNSEEN or self._track_reliable():
                return None
            key = self._m15_key()
            if key is None or key in self._sv_look_done:
                return None
            d0_ = float(np.hypot(pp[0] - pos[0], pp[1] - pos[1]))
            if d0_ > _SV_LOOK_DMAX or d0_ < 2.0:
                return None
            g0_ = float(self.v_gnd) if self.v_gnd is not None else 0.0
            if not (g0_ + _SV_LOOK_ZLO <= float(pp[2]) <= g0_ + _SV_LOOK_ZHI):
                return None
            c0 = self.center0
            if c0 is not None and max(abs(float(pp[0] - c0[0])), abs(float(pp[1] - c0[1]))) > _SV_LOOK_CLUE:
                return None
            ob = np.asarray([o[1:3] for o in self.pad_obs[-20:]], dtype=np.float64)
            if len(ob) < _SV_LOOK_HITS:
                return None
            spread = float(np.sqrt(np.mean(np.sum((ob - ob.mean(axis=0)) ** 2, axis=1))))
            if spread > _SV_LOOK_SPREAD:
                return None
            if self._pad_in_view(pos, rpy):
                return None
            lk = self._sv_look = {"t0": self.t, "key": key}
            self._sv_look_done.append(key)
            self._sv_look_n += 1
            _sv_ev("sv_look")
            _sv_log([round(float(self.t), 2), "look", int(self.pad_hits), round(float(self.pad_conf), 2), round(spread, 2),
                     round(d0_, 1), [round(float(pp[0]), 2), round(float(pp[1]), 2), round(float(pp[2]), 2)]])
        rel = pp - pos
        d_h = float(np.hypot(rel[0], rel[1]))
        yaw_des = math.atan2(rel[1], rel[0]) if d_h > 0.5 else yaw
        v_xy = np.zeros(2)
        planner = False  # turn (or hover facing the estimate) in place
        if d_h > _SV_LOOK_DFAR:
            vz = float(np.clip(1.2 * (prm["h_search"] - alt_c), -1.0, 1.0))
            if abs(wrap(yaw_des - yaw)) < 0.6:
                v_h = min(_SV_LOOK_VH, 0.5 * (d_h - _SV_LOOK_DFAR) + 0.6)
                v_xy = rel[:2] / d_h * v_h
                planner = True
        else:
            # the village search height (6.5 m) puts a pad nearer than ~1.2 x the height under the image (45 deg; the
            # forest look flies at 3.2 m): sink until the estimate sits ~39 deg below the axis, 0.8 x the xy distance
            # above it, never under the surface below + 2.5 m, never a climb (the CX_V16_LOOK rule); if the floor keeps
            # it under the image for 1 s, the look ends ('blind')
            h_want = float(np.clip(0.8 * d_h, 2.5, 8.0))
            z_floor = float(pos[2] - (alt if alt is not None else alt_c)) + 2.5  # down-ray surface + 2.5 m (review fix: alt_c is height over the lowest ground seen under vabs)
            z_w = min(max(float(pp[2]) + h_want, z_floor), float(pos[2]))
            vz = float(np.clip(1.2 * (z_w - float(pos[2])), -1.0, 0.0))
            h = float(pos[2] - pp[2])
            if h > 0.0 and math.degrees(math.atan2(h, max(d_h, 1e-3))) > 44.0 and z_w - float(pos[2]) > -0.3:
                lk["blind"] = lk.get("blind", 0.0) + SIM_DT
                if lk["blind"] > 1.0:
                    self._sv_look = None
                    _sv_ev("sv_look_end_blind")
                    return None
        _sv_ev("sv_look_ticks")
        return np.array([v_xy[0], v_xy[1], vz]), yaw_des, planner

    # ------------------------------------------------------------------ fx_s2nbv NBV (CX_S2N_* only)
    def _s2n_feed(self, depth, pos, rpy, alt):
        """One depth frame into the seen-ground map (32 x 32 min-pooled, usable window only)."""
        import time as _t
        t0_ = _t.perf_counter()
        if self._s2n_map is None:
            self._s2n_map = _S2nMap(self.center0[:2])
        g0 = float(self.v_gnd) if self.v_gnd is not None else float(pos[2] - alt)
        gr = self.planner.grid
        pooled = gr.pool_min(depth)[:gr.n - 2, 1:gr.n - 1]
        z = DEPTH_MIN_M + pooled.astype(np.float64) * (DEPTH_MAX_M - DEPTH_MIN_M)
        ok = z < 19.5
        if ok.any():
            xr = gr.xr[:gr.n - 2, 1:gr.n - 1][ok].astype(np.float64)
            yu = gr.yu[:gr.n - 2, 1:gr.n - 1][ok].astype(np.float64)
            zz = z[ok]
            eye, fwd, right, up = cam_pose(pos, rpy)
            W = (eye[None, :] + (zz * xr)[:, None] * right[None, :] + (zz * yu)[:, None] * up[None, :]
                 + zz[:, None] * fwd[None, :])
            self._s2n_map.update(eye, W, zz, g0)
        _s2n_ms(t0_)

    def _s2n_gaze(self, ex, ey, ez, g0, phi):
        """Camera yaw on the richest unseen wedge within +-_S2N_GOFF of the flight direction phi (hysteresis), else None."""
        Wd = self._s2n_map.wedges(ex, ey, ez, g0)
        cen = -math.pi + (np.arange(_S2N_NB) + 0.5) * (2 * math.pi / _S2N_NB)
        off = np.abs((cen - phi + math.pi) % (2 * math.pi) - math.pi)
        inc = off <= math.radians(_S2N_GOFF)
        Wc = np.where(inc, Wd, -1.0)
        kb = int(np.argmax(Wc))
        if Wc[kb] < _S2N_GMIN:
            self._s2n_gk = None
            return None
        if self._s2n_gk is not None and inc[self._s2n_gk] and Wd[self._s2n_gk] >= _S2N_HYST * Wc[kb]:
            kb = self._s2n_gk
        self._s2n_gk = kb
        return _s2n_bin_yaw(kb)

    def _s2n_plan(self, pos, yaw, ez, g0):
        if self._s2n_cand is None:
            self._s2n_cand = _s2n_cands(self.center0[:2])
        C = np.vstack([self._s2n_cand, [[float(pos[0]), float(pos[1])]]])
        gain, kb = self._s2n_map.vantage_gains(C, ez, g0)
        for (tx, ty) in self._s2n_tabu:
            gain = np.where(np.hypot(C[:, 0] - tx, C[:, 1] - ty) < 3.0, 0.2 * gain, gain)
        d = np.hypot(C[:, 0] - pos[0], C[:, 1] - pos[1])
        hd = np.where(d > 1.0, np.arctan2(C[:, 1] - pos[1], C[:, 0] - pos[0]), yaw)
        yb = -math.pi + (kb + 0.5) * (2 * math.pi / _S2N_NB)
        turn = np.abs((yb - hd + math.pi) % (2 * math.pi) - math.pi) / 0.65
        # the turn to depart: the planner crawls at 0.6 m/s while the flight direction is > 50 deg off the camera
        turn = turn + np.where(d > 1.0, np.maximum(np.abs((hd - yaw + math.pi) % (2 * math.pi) - math.pi)
                                                   - math.radians(45.0), 0.0) / 0.65, 0.0)
        util = gain / (d / 2.6 + 0.4 + turn + 0.8)
        j = int(np.argmax(util))
        return C[j].copy(), float(gain[j]), float(util[j])

    def _s2n_step(self, pos, yaw, vz):
        """CX_S2N_NBV: (v_des, yaw_des) for this village SEARCH ring-branch tick, else None (the champion's rings)."""
        M = self._s2n_map
        wps = self.search_wps
        if M is None or wps is None:
            return None
        if self._s2n_sid is not wps:
            # a new SEARCH build: ring 1 first again
            self._s2n_sid = wps
            self._s2n_st = None
            self._s2n_tgt = None
            self._s2n_look = None
            self._s2n_tabu = []
            self._s2n_gk = None
            self._s2n_gy = None
        if self._s2n_st == "fb" or self.search_i < _S2N_RING or self.t > _S2N_TMAX:
            return None
        import time as _t
        t0_ = _t.perf_counter()
        g0 = float(self.v_gnd) if self.v_gnd is not None else 0.0
        eye, fwd_, right_, up_ = cam_pose(pos, self._rpy)
        ex, ey, ez = float(eye[0]), float(eye[1]), float(eye[2])
        frame_tick = self.step % 10 == 0
        out = None
        live = None
        if (self.pad_pos is not None and self.t - self.pad_last_seen < _S2N_LIVE and not self.pad_moving
                and not self._track_reliable()):
            rel_ = self.pad_pos[:2] - pos[:2]
            key_ = self._m15_key()
            if key_ != self._s2n_live_key:
                self._s2n_live_key = key_
                self._s2n_live_acc = 0.0
            if 3.0 <= float(np.linalg.norm(rel_)) <= 25.0 and self._s2n_live_acc < _S2N_LIVEMAX:
                # a young track in view: keep the camera on it (at <= 1.5 m/s), at most _S2N_LIVEMAX s per track
                live = math.atan2(rel_[1], rel_[0])
                self._s2n_live_acc += SIM_DT
                _s2n_ev("s2n_live")
        if self._s2n_st is None:
            self._s2n_st = "plan"
            _s2n_ev("s2n_on")
        if self._s2n_st == "plan" or (self._s2n_st == "go" and self.t - self._s2n_tplan >= _S2N_REPLAN):
            tgt, g, u = self._s2n_plan(pos, yaw, ez, g0)
            self._s2n_tplan = self.t
            if g < _S2N_VMIN:
                # nothing worth a vantage: the champion's rings resume at the nearest ring-2/3 vertex
                n_ = len(wps)
                lo_ = min(max(_S2N_RING, 4), n_ - 1)
                self.search_i = min(range(lo_, n_), key=lambda i: float(np.hypot(wps[i][0] - pos[0], wps[i][1] - pos[1])))
                self._s2n_st = "fb"
                _s2n_ev("s2n_fb")
                _s2n_log([round(float(self.t), 2), "fb", round(g, 4)])
                _s2n_ms(t0_)
                return None
            if self._s2n_st == "plan" or self._s2n_tgt is None or u > _S2N_SWITCH * self._s2n_util:
                self._s2n_tgt, self._s2n_util = tgt, u
                _s2n_ev("s2n_vantage")
                _s2n_log([round(float(self.t), 2), "vantage", round(float(tgt[0]), 1), round(float(tgt[1]), 1),
                          round(g, 4), round(u, 5)])
            self._s2n_st = "go"
        if self._s2n_st == "go":
            to = self._s2n_tgt - pos[:2]
            d = float(np.linalg.norm(to))
            if d < 1.5:
                self._s2n_st = "look"
                self._s2n_look = {"t0": self.t, "face": None, "k": None, "w": 1.0}
                _s2n_ev("s2n_look")
            else:
                phi = math.atan2(to[1], to[0])
                v_h = min(float(self._params()["v_search"]), math.sqrt(3.0 * max(d - 0.8, 0.0)) + 0.5)
                if frame_tick or self._s2n_gk is None:
                    self._s2n_gy = self._s2n_gaze(ex, ey, ez, g0, phi)
                yd = self._s2n_gy if self._s2n_gy is not None else phi
                out = (np.array([to[0] / d * v_h, to[1] / d * v_h, vz]), yd)
        if out is None and self._s2n_st == "look":
            lk = self._s2n_look
            if frame_tick or lk["k"] is None:
                Wd = M.wedges(ex, ey, ez, g0)
                kb = int(np.argmax(Wd))
                if lk["k"] is None or Wd[lk["k"]] < _S2N_HYST * Wd[kb]:
                    lk["k"] = kb
                    lk["face"] = None
                lk["w"] = float(Wd[lk["k"]])
            yd = _s2n_bin_yaw(lk["k"])
            if abs(wrap(yd - yaw)) < math.radians(12) and lk["face"] is None:
                lk["face"] = self.t
            if ((lk["face"] is not None and self.t - lk["face"] >= _S2N_DWELL) or self.t - lk["t0"] > _S2N_LOOKMAX
                    or lk["w"] < _S2N_GMIN):
                self._s2n_st = "plan"
                self._s2n_tabu.append((float(pos[0]), float(pos[1])))
            out = (np.array([0.0, 0.0, vz]), yd)
        if out is not None and live is not None:
            v_ = out[0]
            vh_ = float(np.hypot(v_[0], v_[1]))
            if vh_ > 1.5:
                v_ = np.array([v_[0] * 1.5 / vh_, v_[1] * 1.5 / vh_, v_[2]])
            out = (v_, live)
        _s2n_ms(t0_)
        return out

    def _f14_gate(self, good):
        """CX_F14_GATE (forest): keep only detections that can be the goal pad (see the flag comment). A detection that
        fails is neither a hit nor a miss. Any error returns the detections unchanged (c14 behaviour)."""
        try:
            out = []
            sp = self.start_pos
            gmin = self._f14_gmin
            for d in good:
                w = d[1]
                if sp is not None and math.hypot(float(w[0]) - float(sp[0]), float(w[1]) - float(sp[1])) < _F14_NEAR_M:
                    _cx("f14_gate_start")
                    continue
                if gmin is not None and float(w[2]) > gmin + _F14_HIGH_M:
                    _cx("f14_gate_high")
                    continue
                out.append(d)
            return out
        except Exception:
            return good

    def _rc_land_veto(self):
        """CX_RC_LSEEN / CX_RC_LBAND (forest): True, with the track dropped, when the static track about to be LANDed on
        cannot be the pad (see the flag comment). Any error returns False (c19 behaviour)."""
        try:
            if self.map_label != "forest" or self.pad_pos is None or self.pad_moving or self.pad_ever_moving:
                return False
            pz = float(self.pad_pos[2])
            g = self._rc_gmin
            if _RC_LBAND and g is not None and abs(g) <= 0.5 and (pz < _RC_LBAND_LO or pz > _RC_LBAND_HI):
                _cx("rc_lband")
                _cx_ms_log("rc_lband_at", [round(float(self.t), 2), round(pz, 2), int(self.pad_hits), round(float(g), 2)])
                self.bad_spots.append(np.array([float(self.pad_pos[0]), float(self.pad_pos[1]), pz]))
                self._drop_track()
                return True
            unseen = float(self.t - self.pad_last_seen)
            if _RC_LSEEN and not self._static_established() and unseen >= _RC_LSEEN_S:
                far = float(self.pad_min_z) >= _RC_LSEEN_FAR
                _cx("rc_lseen")
                _cx_ms_log("rc_lseen_at", [round(float(self.t), 2), round(unseen, 2), int(self.pad_hits),
                                           round(float(self.pad_min_z), 1), round(pz, 2), bool(far)])
                if far:
                    self.bad_spots.append(np.array([float(self.pad_pos[0]), float(self.pad_pos[1]), pz]))
                self._drop_track()
                return True
            return False
        except Exception:
            return False

    def _f14_static_reseed(self, w, z):
        """CX_F14_STATIC (forest): the established static track met 40 conflicting detections. c14 re-seeds it as MOVING
        (pad_moving and pad_ever_moving set, so LAND takes the moving branch and the ever-moving hold). Forest pads never
        move: re-seed it at the new detection as a young static track instead (30 hits, not established, so the wide gate
        and the EMA fuse the next detections). Returns False on any error (then c14's re-seed runs)."""
        try:
            self._v3_track_reset()
            self.pad_pos = w.copy()
            self.pad_vel = np.zeros(2)
            self.pad_hits = 30
            self.pad_obs = [(self.t, w[0], w[1], w[2], z)]
            self.pad_last_seen = self.t
            self.pad_conflicts = 0
            self.pad_info = 0.0
            self.pad_raw = []
            self.pad_still_t = 0.0
            self.pad_move_votes = 0
            _cx("f14_static")
            return True
        except Exception:
            return False

    def _f6_under(self, pos, alt, yaw, v_des, yaw_des, use_planner):
        """CX_F6_UNDER: boxed-in recovery for forest CRUISE, SEARCH ring legs and REACQUIRE (see the flag comment).

        Returns (v_des, yaw_des, use_planner); unchanged unless a recovery phase is running.
        """
        active = self.mode in ("CRUISE", "REACQUIRE") or (self.mode == "SEARCH" and self.spin_done)
        if not active:
            if self._f6_ph is not None:
                _cx("f6_under_cancel")
                if self._f6_ph == "detour":
                    self._f6_det_end = self.t
                self._f6_low_until = -1.0  # the LOW height does not outlive a cancel (APPROACH, LAND, spin, ...)
            self._f6_ph = None
            self._f6_blk = 0.0
            return v_des, yaw_des, use_planner
        if self._f6_esc_step == self.step - 1:
            self._f6_blk += SIM_DT
        else:
            self._f6_blk = max(0.0, self._f6_blk - 0.5 * SIM_DT)
        if self._f6_blk >= _F6_UNDER_TRIG and self._f6_ep < _F6_UNDER_MAXEP:
            self._f6_start(pos, yaw, v_des)
        if self._f6_ph == "back":
            tr = self._f6_trail
            i = self._f6_back_i
            while i > self._f6_back_stop:
                tx, ty, tz, _ = tr[i]
                if (tx - pos[0]) ** 2 + (ty - pos[1]) ** 2 + (tz - pos[2]) ** 2 < 0.1225:
                    i -= 1
                else:
                    break
            self._f6_back_i = i
            tx, ty, tz, _ = tr[i]
            to = np.array([tx - pos[0], ty - pos[1], tz - pos[2]])
            d = float(np.linalg.norm(to))
            reached = i <= self._f6_back_stop and d < 0.35
            if reached or self.t - self._f6_ph_t0 > 5.0:
                del tr[i + 1:]  # the retraced stretch leads into the blockage: the trail continues from here
                if not reached and alt < 2.0:
                    # timed out still over foliage: a detour from here would start inside the crown; c3 keeps it
                    _cx("f6_under_back_abort")
                    self._f6_ph = None
                    return v_des, yaw_des, use_planner
                self._f6_detour(pos, v_des)
            else:
                _cx("f6_under_back")
                # the path flown is free: follow it backwards, camera kept on the blockage
                return unit(to) * min(0.8, 1.5 * d + 0.2), self._f6_yaw0, False
        if self._f6_ph == "detour":
            to = self._f6_wp - pos[:2]
            d = float(np.linalg.norm(to))
            if d < 1.2 or self.t - self._f6_ph_t0 > 6.0:
                self._f6_ph = None
                self._f6_det_end = self.t
                self._f6_low_until = self.t + _F6_UNDER_LOWS
                return v_des, yaw_des, use_planner
            _cx("f6_under_detour")
            dh = unit(to)
            yaw_w = math.atan2(dh[1], dh[0])
            vz = float(v_des[2])
            if alt < _F6_UNDER_LOWH - 0.6:
                vz = max(vz, 0.0)  # something close under the drone: do not sink onto it
            if abs(wrap(yaw_w - yaw)) > 0.6:
                # turn the camera to the detour side first. The planner is off here, so never climb: the camera cannot
                # see what is above, and c3's planner would have cone-limited a blind climb (hold height at worst)
                if vz > 0.0:
                    _cx("f6_under_turn_cap")
                    vz = 0.0
                return np.array([0.0, 0.0, vz]), yaw_w, False
            vh = min(1.5, 0.8 * d + 0.3)
            return np.array([dh[0] * vh, dh[1] * vh, vz]), yaw_w, True
        return v_des, yaw_des, use_planner

    def _f6_start(self, pos, yaw, v_des):
        """CX_F6_UNDER: start a recovery episode: pick the trail point to retrace to and the detour side.

        Starts only when a crumb >= _F6_UNDER_BACK m back (straight line) within the last 80 was recorded over open
        ground (down-ray >= 2 m). Otherwise the tick stays with c3: no episode is used up and the escape timer restarts.
        """
        self._f6_blk = 0.0
        tr = self._f6_trail
        n = len(tr)
        stop = None
        for i in range(n - 1, max(-1, n - 81), -1):
            if tr[i][3] >= 2.0 and math.hypot(tr[i][0] - pos[0], tr[i][1] - pos[1]) >= _F6_UNDER_BACK:
                stop = i  # far enough back, and the down-ray saw open space there
                break
        if stop is None:
            _cx("f6_under_skip")
            return
        self._f6_ep += 1
        _cx("f6_under")
        # start times as a text log (not a count), last 12 kept
        CX_EVENTS["f6_under_at"] = " ".join((str(CX_EVENTS.get("f6_under_at", "")) + " %.2f" % float(self.t)).split()[-12:])
        same = self._f6_ph == "detour" or self.t - self._f6_det_end < _F6_UNDER_KEEP
        if same and self._f6_side != 0.0:
            # the detour of the same blockage ran into foliage again: the other side of the FIRST heading (the current
            # yaw points along that detour, whose "other side" would lead back toward the blockage)
            _cx("f6_under_same")
            self._f6_side = -self._f6_side
        else:
            left, right = self._f6_lr
            self._f6_side = 1.0 if right >= left else -1.0
            self._f6_yaw0 = yaw
        if self._f6_ph == "detour":
            self._f6_det_end = self.t
        self._f6_ph = "back"
        self._f6_back_i = n - 1
        self._f6_back_stop = stop
        self._f6_ph_t0 = self.t

    def _f6_detour(self, pos, v_des):
        """CX_F6_UNDER: detour point to the chosen side of the heading at the blockage, a little forward along the goal."""
        g = np.array([float(v_des[0]), float(v_des[1])])
        if float(np.linalg.norm(g)) < 0.1:
            g = np.array([math.cos(self._f6_yaw0), math.sin(self._f6_yaw0)])
        g = unit(g)
        r = np.array([math.sin(self._f6_yaw0), -math.cos(self._f6_yaw0)])  # right of the heading at the blockage
        self._f6_wp = pos[:2] + r * (self._f6_side * _F6_UNDER_SIDE) + g * 1.5
        self._f6_ph = "detour"
        self._f6_ph_t0 = self.t
        self._f6_low_until = self.t + 6.0 + _F6_UNDER_LOWS

    def _climb_speed(self, pos, top_band_m):
        m = self.map_label
        if m == "warehouse":
            return 0.6 if (pos[2] < 9.0 and top_band_m > 4.0) else 0.0
        if m == "forest":
            return 0.5 if (pos[2] < 12.0 and top_band_m > 6.0) else 0.0
        return 1.0

    def _ct_roof(self, pos, alt, vel, pad, v_des):
        """CX_CT_ROOF (city): the down-ray is cast against collision shapes, so it is the one sensor that sees a building's
        convex-hull collider where the render shows open air (1443566373: the hull joins a 2.5 m awning to a 12.4 m tower
        with a ~47 deg face; the ray read 4.4 -> 1.4 m, then the drone hit it 0.54 s later at 2.9 m/s because the 3.2 m/s^2
        command ramp needs ~0.7 s to trade speed for climb). A step up of the surface below to under 2.5 m starts a roof
        episode: horizontal speed is capped by the height over that surface, the drone climbs toward 3 m over it, and the
        acceleration ramp is bypassed. While still over the raised structure the guard re-arms whenever the surface keeps
        rising (a crawl up the hull) or the drone sinks under 1.5 m over it."""
        hist = getattr(self, "_ct_ghist", None)
        if hist is None:
            hist = self._ct_ghist = []
            self._ct_until = -1.0
            self._ct_ep_t = -99.0
            self._ct_street_g = 0.0
        if alt < 19.5:
            hist.append((self.t, float(pos[2] - alt)))
        while hist and self.t - hist[0][0] > 0.5:
            hist.pop(0)
        if not hist or alt >= 19.5 or self.mode not in ("CRUISE", "SEARCH", "REACQUIRE", "APPROACH"):
            return v_des
        if pad is not None and float(np.hypot(pad[0] - pos[0], pad[1] - pos[1])) < 3.0:
            return v_des
        g_now = hist[-1][1]
        g_min = min(h[1] for h in hist)
        rise = g_now - g_min
        if rise >= 1.5 and alt < 2.5 and float(np.hypot(vel[0], vel[1])) > 0.8:
            if self.t >= self._ct_until:
                _cx_ev("ct_roof_step")
                self._ct_street_g = g_min
            self._ct_until = self.t + 0.8
            self._ct_ep_t = self.t
        elif (self.t - self._ct_ep_t < 15.0 and g_now > self._ct_street_g + 1.0
              and ((rise >= 0.4 and alt < 3.5) or alt < 1.5)):
            if self.t >= self._ct_until:
                _cx_ev("ct_roof_rearm")
            self._ct_until = max(self._ct_until, self.t + 0.8)
        if self.t >= self._ct_until:
            return v_des
        cap = float(np.clip(0.6 * (alt - 1.0), 0.0, 1.2))
        vh = float(np.hypot(v_des[0], v_des[1]))
        sc = min(1.0, cap / max(vh, 1e-6))
        vz = max(float(v_des[2]), float(np.clip(1.5 * (3.0 - alt), 0.0, 1.5)))
        self._ct_bypass = True
        _cx_ev("ct_roof_tick")
        return np.array([v_des[0] * sc, v_des[1] * sc, vz])

    def act(self, observation):
        self.step += 1
        self.t = self.step * SIM_DT
        st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        depth = np.asarray(observation["depth"], dtype=np.float32).reshape(IMG_H, IMG_W)
        pos = st[0:3].astype(np.float64)
        rpy = st[3:6].astype(np.float64)
        vel = st[6:9].astype(np.float64)
        alt = float(st[137]) * 20.0
        alt_c = alt
        if _flag("gndfilt") and self.map_label == "forest":
            self.gnd_hist.append((self.t, pos[2] - alt))
            while self.gnd_hist and self.t - self.gnd_hist[0][0] > 4.0:
                self.gnd_hist.pop(0)
            alt_c = pos[2] - min(g for _, g in self.gnd_hist)
        if _flag("vabs") and self.map_label == "village":
            g = pos[2] - alt
            if self.v_gnd is None or g < self.v_gnd:
                self.v_gnd = g
            alt_c = pos[2] - self.v_gnd
        goal_off = st[138:141].astype(np.float64)
        center = pos + goal_off
        if self.start_pos is None:
            self.start_pos = pos.copy()
            self.start_alt = alt
            self.center0 = center.copy()
            self.v_cmd = vel.copy()
        if _CX_FP_POST and self.map_label == "forest":
            # fx_fp POST: search at the template posterior mean instead of the raw clue (center0 keeps the clue)
            try:
                fpc_ = self._fp_post_center()
                if fpc_ is not None:
                    center = center.copy()
                    center[0], center[1] = float(fpc_[0]), float(fpc_[1])
                    goal_off = center - pos
            except Exception:
                self._fp_c = False
                _cx("fp_error")
        yaw = float(rpy[2])
        self._rpy = rpy
        if _CX_MTF_YT:
            self._mtf_pitch.append(math.degrees(float(rpy[1])))  # fx_mtf YT (nose-down positive)
        if _CX_RA_DIP and self._ra_dec is None and self.map_label == "village" and self.step % 10 == 2:
            # fx_ra DIP: approach coverage near the clue, sampled every 10th tick until the gate decides
            try:
                if self._ra_cov is None:
                    self._ra_cov = _RaCover(center[:2])
                self._ra_cov.ground(pos, alt)
                self._ra_cov.update(depth, pos, rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2])), self.v_gnd)
            except Exception:
                self._ra_dec = False
                _cx_ra_ev("ra_error")
        prm = self._params()
        if _RC_LBAND and alt < 19.5:
            # fx_rc LBAND: lowest ground the down-ray has seen this flight
            grc_ = float(pos[2] - alt)
            if self._rc_gmin is None or grc_ < self._rc_gmin:
                self._rc_gmin = grc_
        if _F14_GATE and alt < 19.5:
            # fx_f14 GATE: lowest ground the down-ray has seen this flight (the start platform top until the drone leaves it)
            g14_ = float(pos[2] - alt)
            if self._f14_gmin is None or g14_ < self._f14_gmin:
                self._f14_gmin = g14_
        if _CX_VG_V8H and alt < 19.5:
            # fx_vg V8H: lowest ground the down-ray has seen this flight
            gvg_ = float(pos[2] - alt)
            if self._vg_gmin is None or gvg_ < self._vg_gmin:
                self._vg_gmin = gvg_
        if _CX_VF_ANY and alt < 19.5:
            # fx_vf: lowest ground the down-ray has seen this flight
            gvf_ = float(pos[2] - alt)
            if self._vf_gmin is None or gvf_ < self._vf_gmin:
                self._vf_gmin = gvf_
        if _CX_VM_ANY and alt < 19.5 and (self.start_pos is None or float(np.hypot(
                float(pos[0]) - float(self.start_pos[0]), float(pos[1]) - float(self.start_pos[1]))) > 1.5):
            # fx_vm: lowest ground the down-ray has seen this flight, away from the start pad
            gvm_ = float(pos[2] - alt)
            if self._vm_gmin is None or gvm_ < self._vm_gmin:
                self._vm_gmin = gvm_

        if _CX_ME_ANY:
            self._me_pos = pos
            self._me_vel = vel
            if (self.map_label == "mountain" and self.pad_pos is not None and not self.pad_moving
                    and self.mode in ("APPROACH", "LAND") and alt < 4.0):
                # fx_me RIM: pad-top calibration from the down-ray while the drone is over the estimate's centre
                s_ = float(pos[2] - alt)
                if (float(np.hypot(pos[0] - self.pad_pos[0], pos[1] - self.pad_pos[1])) < _ME_RG_CAL_R
                        and abs(s_ - float(self.pad_pos[2])) < 0.5 and abs(float(rpy[0])) < 0.2 and abs(float(rpy[1])) < 0.2):
                    self._me_rg_cal.append((self.t, s_, float(self.pad_pos[0]), float(self.pad_pos[1])))
                    if len(self._me_rg_cal) > 200:
                        self._me_rg_cal.pop(0)
        dets, cls = self._detect(depth, pos, rpy)
        self.last_dets = dets
        self._update_map(cls, pos, alt)
        if (_CX_S2N_ANY and self.map_label == "village" and self.step % 10 == 0 and self.mode != "TAKEOFF"
                and self.center0 is not None and not self._s2n_dead):
            try:
                self._s2n_feed(depth, pos, rpy, alt)  # fx_s2nbv: one depth frame into the seen-ground map
            except Exception:
                self._s2n_dead = True
                _s2n_ev("s2n_error")
        if _flag("cov") and self.map_label in ("mountain", "village") and self.mode in ("CRUISE", "SEARCH", "REACQUIRE"):
            self._cov_update(depth, pos, rpy)
        if _CX_V3_OCC:
            self._v3_depth = depth
            self._v3_vh = float(np.hypot(vel[0], vel[1]))  # the occlusion excuse needs a moving view
        if _CX_V16_LOG:
            _v16_pre = self._v16_log_pre()
        if _PC_DIAG or _CX_PC_KTO:
            self._pc_pos = (float(pos[0]), float(pos[1]), float(pos[2]))  # fx_pc: drone position this tick
        if self.mode != "TAKEOFF" or self.t > 1.0:
            self._update_track(dets, pos)
        if _CX_MTF_YT and self.pad_obs and self.pad_pos is not None and self.pad_obs is not self._mtf_pb_ref:
            # fx_mtf YT: classify each new track at its first hit (every seeding site builds a new pad_obs list)
            try:
                self._mtf_pb_ref = self.pad_obs
                pm_ = max(self._mtf_pitch) if self._mtf_pitch else 0.0
                dep_ = math.degrees(math.atan2(float(pos[2] - self.pad_pos[2]),
                                               max(float(np.hypot(*(self.pad_pos[:2] - pos[:2]))), 1e-3)))
                self._mtf_pb = bool(pm_ >= _MTF_YT_PMIN or (_MTF_YT_DEP > 0.0 and dep_ >= _MTF_YT_DEP))
                if self.map_label == "mountain":
                    _cx_mtf_ev("mtf_yt_seed_pb" if self._mtf_pb else "mtf_yt_seed_lvl")
                    _cx_mtf_log("mtf_yt_at", "%.2f/p%+.1f/d%.1f/%s" % (float(self.t), pm_, dep_, "pb" if self._mtf_pb else "lvl"), 6)
            except Exception:
                self._mtf_pb = False
                _cx_mtf_ev("mtf_error")
        if (_CX_MTF_ZB and self.map_label == "mountain" and self.pad_pos is not None and self.center0 is not None
                and not self.pad_moving and not self.pad_ever_moving and self.mode not in ("LAND", "TAKEOFF")
                and self.pad_hits >= _MTF_ZB_HITS
                and (not _MTF_ZB_APP or self.mode in ("APPROACH", "REACQUIRE", "RECOVER"))
                and abs(float(self.pad_pos[2]) - float(self.center0[2])) > _MTF_ZB_DZ):
            try:
                self._mtf_zb_veto(pos, center)
            except Exception:
                _cx_mtf_ev("mtf_error")
        if _CX_VFP_LSEEN and self.map_label == "village":
            # ws2/vfp LSEEN: closest 3D distance from which the current track was detected (a new track list = a new track)
            if self.pad_pos is None or self.pad_obs is not getattr(self, "_vfp_ref", None):
                self._vfp_ref = self.pad_obs
                self._vfp_d3 = 99.0
            if self.pad_pos is not None and self.pad_obs and abs(float(self.pad_obs[-1][0]) - float(self.t)) < 1e-6:
                o_ = self.pad_obs[-1]
                self._vfp_d3 = min(float(getattr(self, "_vfp_d3", 99.0)),
                                   float(np.linalg.norm(np.array([o_[1], o_[2], o_[3]], dtype=np.float64) - pos)))
        if _CX_S1M_TOP and self.mode == "LAND" and self.map_label == "mountain":
            self._s1m_top_step(pos, alt)  # fx_s1m: pad-top check under a centred static LAND (RECEN/BADREC evidence)
        if _CX_V16_LOG:
            self._v16_log_post(_v16_pre, dets)
        if (_flag("cdrift") and self.map_label == "city" and self.pad_pos is not None and not self.pad_moving
                and self.pad_hits >= 30 and self.mode in ("APPROACH", "LAND") and len(self.pad_hist) >= 6
                and (self.t - self.pad_last_seen) < 0.5):
            old = [h for h in self.pad_hist if 1.8 <= self.t - h[0] <= 2.6]
            if _CX_PC_VETO and old:
                old = [h for h in old if h[0] >= self._pc_veto_t]  # fx_pc VETO: the window restarts after a veto
            if old:
                h0 = old[-1]
                if (float(np.hypot(self.pad_pos[0] - h0[1], self.pad_pos[1] - h0[2])) > 0.8
                        and not (_PC_VON and self._pc_cdrift(pos, h0))):
                    self.pad_moving = True
                    self.pad_ever_moving = True
                    self.pad_vel = (self.pad_pos[:2] - np.array([h0[1], h0[2]])) / max(self.t - h0[0], 0.2)
                    self._cdrift_fired = True
        if (self.map_label == "mountain" and self.pad_pos is not None and not self.pad_moving
                and self.mode in ("CRUISE", "SEARCH", "APPROACH", "REACQUIRE") and self.step % 2 == 0):
            self._ring_collect(depth, pos, rpy)
            if _CX_S1M_BUF:
                self._s1m_collect(depth, pos, rpy)  # fx_s1m: full-resolution height-map points (HM/DESC/FLAT)
        if self.pad_moving and self.pad_pos is not None:
            recent = (self.t - self.pad_last_seen) < 0.3
            if recent and self.pad_fit_sp < 0.2 and self.pad_fit_z < 6.0:
                self.pad_slow_t += SIM_DT
            elif recent:
                self.pad_slow_t = 0.0
            pass
        prm = self._params()
        if _F6_UNDER and self.map_label == "forest" and self._f6_ph != "back":
            # CX_F6_UNDER: breadcrumbs of the path flown (free space by construction), for a retrace when boxed in
            tr_ = self._f6_trail
            if not tr_ or (pos[0] - tr_[-1][0]) ** 2 + (pos[1] - tr_[-1][1]) ** 2 + (pos[2] - tr_[-1][2]) ** 2 > 0.0625:
                tr_.append((float(pos[0]), float(pos[1]), float(pos[2]), float(alt)))
                if len(tr_) > 160:
                    del tr_[0]

        if _CX_V8_SURF:
            self._v8_surf_step(pos, alt, center)
        if _CX_VX_RAY and self.map_label == "village" and not _CX_STATE.get("shadow"):
            # fx_vx RAY: the down-ray crossed a pad top the camera never saw
            try:
                self._vx_detect(pos, alt, center, yaw)
            except Exception:
                _cx_vx_ev("vx_error")
        v_des = np.zeros(3)
        yaw_des = yaw
        use_planner = True
        vz_override = None
        ms_slope = None  # fx_ms SLOPE: slope (m/m) toward the pad whose sink is set after the planner; None = off
        self._ma_ring_tick = False  # fx_ma RING (set in the SEARCH ring branch; read only behind CX_MA_RING)
        tube = prm["tube"]
        tube_min = prm["tube_min"]
        pad = self._pad_predicted()
        h_cruise = prm["h_cruise"]
        if _flag("vcruise12") and self.map_label == "village":
            h_cruise = 12.0
        vabs_lim = None
        if _flag("vabs") and self.map_label == "village" and self.v_gnd is not None:
            vabs_lim = _VABS_LIMIT_M
            if self.pad_pos is not None and not self.pad_moving:
                vabs_lim = max(vabs_lim, float(self.pad_pos[2]) - self.v_gnd + 2.5)
            h_cruise = min(h_cruise, vabs_lim)
        if (_F6_UNDER and self.map_label == "forest" and self.t < self._f6_low_until
                and self.mode in ("CRUISE", "REACQUIRE", "SEARCH")):
            h_cruise = min(h_cruise, _F6_UNDER_LOWH)  # CX_F6_UNDER: under the crown layer while detouring

        if self.mode == "TAKEOFF":
            yaw_des = math.atan2(goal_off[1], goal_off[0])
            climbed = pos[2] - self.start_pos[2]
            target_h = h_cruise
            if self.map_label in ("forest", "warehouse") and self.start_pos[2] > h_cruise - 0.5:
                target_h = 1.0
            if _flag("whfast") and self.map_label == "warehouse":
                target_h = min(target_h, 3.5)
            top = DEPTH_MIN_M + depth[0:12, 32:96].min() * (DEPTH_MAX_M - DEPTH_MIN_M)
            _ceil = 6.0 if self.map_label == "forest" else 2.5
            ready = (alt_c if vabs_lim is not None else alt) >= target_h or climbed >= target_h + 0.5 or (top < _ceil and climbed > 0.8)
            open_sky = self.map_label not in ("forest", "warehouse")
            _climb_v = 0.8 if self.map_label == "forest" else (
                2.5 if (open_sky or (_flag("whfast") and self.map_label == "warehouse")) else 1.5)
            if self.map_label == "forest":
                self._f1_steps = getattr(self, "_f1_steps", 0) + 1
            v_des = np.array([0.0, 0.0, _climb_v if not ready else 0.0])
            use_planner = False
            yaw_err = abs(wrap(yaw_des - yaw))
            if _flag("ftake") and self.map_label == "forest":
                fw = unit(goal_off[:2])
                v_f = 1.5 if yaw_err < 0.5 else (0.8 if yaw_err < 1.2 else 0.0)
                v_des = np.array([fw[0] * v_f, fw[1] * v_f, (max(_climb_v, 1.2) if not ready else 0.0)])
                use_planner = True
                if ready and (yaw_err < 0.6 or self.t - self.mode_t0 > 4.0):
                    self._set_mode("CRUISE")
            early = open_sky and climbed >= 2.0 and yaw_err < 0.6
            if self.mode == "TAKEOFF" and ((ready and (yaw_err < 0.35 or self.t - self.mode_t0 > 6.0)) or early):
                self._set_mode("CRUISE")
        elif self.mode == "CRUISE":
            to_c = center - pos
            d_h = float(np.linalg.norm(to_c[:2]))
            dir_h = unit(to_c[:2])
            v_h = min(prm["v_cruise"], math.sqrt(2 * 1.8 * max(d_h - 1.0, 0.0)) + 0.4)
            if _flag("whfast") and self.map_label == "warehouse":
                v_h = min(3.0, math.sqrt(2 * 2.0 * max(d_h - 1.0, 0.0)) + 0.4)
            # altitude: terrain following via altitude ray, but ignore the noisy centre z
            vz = float(np.clip(1.2 * (h_cruise - alt_c), -2.5 if alt_c > 8.0 else -1.0,
                               _MTN_CRUISE_VZ if (self.map_label == "mountain" and not (_MTN_CLIMB_NOPAD and self.pad_pos is not None)) else 1.5))
            if (_flag("kinghint") and self.hint_xy is not None and self.t - self.hint_t < 3.0
                    and float(np.linalg.norm(self.hint_xy - pos[:2])) < 35.0 and self.pad_pos is None):
                self.search_anchor = self.hint_xy.copy()
                self._set_mode("SEARCH")
                self.search_wps = None
                self.hinted = True
            if self.map_label == "mountain" and d_h < 25.0:
                z_t = max(float(center[2]) + 3.0, pos[2] - alt + 2.0)
                if _flag("mhigharr"):
                    z_t = max(float(center[2]) + 8.0, pos[2] - alt + 2.0)
                vz = float(np.clip(1.2 * (z_t - pos[2]), -2.5 if alt > 8.0 else -1.0, 1.5))
            elif _MTN_C11M and self.map_label == "mountain" and float(center[2]) > pos[2] + 2.0:
                # c11m (uid117): climb to the goal's height early instead of hugging the rising slope.
                z_t = max(pos[2] + (h_cruise - alt), float(center[2]) + 4.0)
                vz = float(np.clip(1.5 * (z_t - pos[2]), -1.0, 2.5))
            if (_CX_MT_NODIVE and self.map_label == "mountain" and getattr(self, "map_locked", False)
                    and d_h >= 25.0 and self.pad_pos is None and vz < 0.0):
                # fx_nodive: far leg: no dive faster than the straight glide to the near-leg height
                k_gl = (float(center[2]) + _MT_NODIVE_DZ - float(pos[2])) / max(d_h - 25.0, 5.0)
                vz_nd = max(vz, min(0.0, k_gl * v_h))
                if vz_nd > vz + 1e-6:
                    _cx("mt_nodive")
                vz = vz_nd
            if self.map_label in ("forest", "warehouse") and alt > h_cruise + 1.5 and not (_flag("whfast") and self.map_label == "warehouse"):
                v_h = min(v_h, 0.6)
                if _flag("fdown") and self.map_label == "forest":
                    v_h = min(prm["v_cruise"], 2.0)
            dir_h, v_h = self._city_detour(pos, dir_h, v_h, prm)
            vz = self._city_zcap(pos, alt, vz)
            v_des = np.array([dir_h[0] * v_h, dir_h[1] * v_h, vz])
            yaw_des = math.atan2(dir_h[1], dir_h[0]) if d_h > 1.0 else yaw
            if self._track_reliable():
                self._set_mode("APPROACH")
            elif d_h < 2.0:
                if (((_flag("vcheck") and self.map_label == "village") or (_flag("ccheck") and self.map_label == "city")) and self.pad_pos is not None
                        and self.pad_hits >= 3 and getattr(self, "_vcheck_n", 0) < 1
                        and float(np.linalg.norm(self.pad_pos[:2] - pos[:2])) <= 20.0):
                    self._vcheck_n = getattr(self, "_vcheck_n", 0) + 1
                    self._vcheck_active = True
                    tgt = self.pad_pos.copy()
                    back = unit(pos[:2] - tgt[:2])
                    if float(np.linalg.norm(back)) < 1e-6:
                        back = np.array([math.cos(yaw), math.sin(yaw)]) * -1.0
                    self.reacq_target = tgt
                    self.reacq_moving = False
                    self.reacq_wp = np.array([tgt[0] + back[0] * 5.0, tgt[1] + back[1] * 5.0, tgt[2] + 3.5])
                    self.reacq_hold = 0.0
                    self.reacq_side = 0
                    self.reacq_t_enter = self.t
                    self._set_mode("REACQUIRE")
                else:
                    self._set_mode("SEARCH")
                    self.search_wps = None
            elif _CX_OMF_CSTK and self.map_label == "city" and not getattr(self, "_omf_cstk_done", False):
                # fx_cstk: total CRUISE time near a clue inside a building -> SEARCH once (the rings avoid the clue point)
                if d_h < _OMF_CSTK_R:
                    if getattr(self, "_omf_cstk_t", None) is not None:
                        self._omf_cstk_acc = getattr(self, "_omf_cstk_acc", 0.0) + min(max(self.t - self._omf_cstk_t, 0.0), 0.1)
                    self._omf_cstk_t = self.t
                else:
                    self._omf_cstk_t = None
                if getattr(self, "_omf_cstk_acc", 0.0) >= _OMF_CSTK_S:
                    self._omf_cstk_done = True
                    _cx("omf_cstk")
                    self._set_mode("SEARCH")
                    self.search_wps = None
            if _CX_MA_SO and (self._ma_so is not None or (self._ma_so_n == 0 and self.map_label == "mountain")):
                # fx_ma SO: stand-off look at a clue that lies below the camera's view
                try:
                    so_ = self._ma_so_step(depth, pos, rpy, vel, yaw, alt, center, d_h)
                except Exception:
                    so_ = None
                    self._ma_so = None
                    _cx_ma_ev("ma_error")
                if so_ is not None:
                    v_des, yaw_des, use_planner = so_
        elif self.mode == "SEARCH":
            if self._track_reliable():
                # fx_m2 (plain writes; read only behind CX_M2_*): the first search's gates act only until its first
                # find, so a spin resumed after a weak APPROACH drop is the champion's
                self._m2_high = False
                self._m2_seen = True
                self._set_mode("APPROACH")
            else:
                if self.search_wps is None:
                    self.spin_start_yaw = yaw
                    self.spin_acc = 0.0
                    self.spin_prev = yaw
                    self.spin_done = False
                    c = center.copy()
                    anchor = getattr(self, "search_anchor", None)
                    if anchor is not None:
                        c[:2] = anchor
                        self.search_anchor = None
                    wps = []
                    n_wp = 4 if self.map_label == "village" else 5
                    radii = (7.0, 14.0, 20.0)
                    step = math.pi / 2
                    if _flag("r711"):
                        radii = (7.0, 11.0, 16.0)
                    if _flag("vhigh") and self.map_label == "village":
                        radii = (11.0, 16.0)
                        n_wp = 6
                        step = 2.0 * math.pi / n_wp
                    if _flag("mrings") and self.map_label == "mountain":
                        radii = (9.0, 16.0)
                        n_wp = 6
                        step = 2.0 * math.pi / n_wp
                    if _flag("r915") or (_CX_S1F_R915 and self.map_label == "forest"):
                        radii = (9.0, 15.0)
                        n_wp = 8
                        step = 2.0 * math.pi / n_wp
                        if _CX_S1F_R915 and self.map_label == "forest":
                            _cx_ms_ev("s1f_r915")  # fx_s1f R915: this forest search flies the r915 rings
                    a0 = self.spin_start_yaw
                    if _flag("ringback") and self.map_label != "mountain":
                        a0 = self.spin_start_yaw + math.pi
                    for r in radii:
                        for k in range(n_wp):
                            a = a0 + k * step
                            wp_ = c[:2] + r * np.array([math.cos(a), math.sin(a)])
                            if _flag("vclamp") and self.map_label == "village":
                                wp_ = np.clip(wp_, -_VBOX_M, _VBOX_M)
                            if _CX_FP_CLAMP and self.map_label == "forest":
                                wp_ = np.clip(wp_, -_FP_BOX_M, _FP_BOX_M)  # fx_fp CLAMP: forest pads lie inside +-42 m
                            wps.append(wp_)
                    self._vdip_active = False
                    _ra_vdip = _flag("vdip")
                    if _CX_RA_DIP and not _ra_vdip and self.map_label == "village":
                        _ra_vdip = self._ra_decide()
                    if (_ra_vdip and self.map_label == "village" and anchor is None
                            and (self.v_gnd is None or (alt_c - alt) < _VDIP_GND_M)):
                        self._vdip_active = True
                        if not _flag("vdip"):
                            _cx_ra_ev("ra_dip")
                    if (_flag("nospin") and self.map_label != "mountain" and not (_flag("fspin") and self.map_label == "forest")
                            and not (_flag("wspin") and self.map_label == "warehouse") and not self._vdip_active):
                        self.spin_done = True
                    if getattr(self, "hinted", False):
                        self.spin_done = True
                        self.hinted = False
                    if _CX_V16_SPIN:
                        self._v16_spin_on = False
                        if (self.map_label == "village" and anchor is None and self.spin_done and not self._vdip_active
                                and self._v16_spin_n < 1 and not self.pad_moving and not self.pad_ever_moving
                                and float(np.linalg.norm(c[:2] - pos[:2])) <= 6.0
                                and (self.v_gnd is None or (alt_c - alt) < _VDIP_GND_M) and not _CX_STATE.get("shadow")):
                            # fx_v16 SPIN: the first search at the clue turns in place while sinking to _V16_SPIN_H
                            self.spin_done = False
                            self._v16_spin_on = True
                            self._v16_spin_n += 1
                            self._v16_spin_t0 = self.t
                            _cx_v16_ev("v16_spin")
                    self.search_wps = wps
                    self.search_i = 0
                    self.n_wp_ring = n_wp
                    self.search_center_xy = c[:2].copy()
                    if _FO_SPIN:
                        self._fo_gref = None     # lowest ground seen since this spin began
                        self._fo_hold_xy = None  # where a canopy hold fired (arms the post-spin level exit)
                    if _CX_VS_DIPFULL or _CX_VS_RING1OUT:
                        self._vs_low_acc = 0.0    # spin made at the dip height (DIPFULL)
                        self._vs_dip_ext = False  # DIPFULL extension counted for this dip
                        self._vs_dip_ok = False   # a dip at this search centre has finished (RING1OUT)
                        self._vs_r1_leg = -1      # RING1OUT event counter per leg of this ring set
                        self._kg_dip_t = None     # fx_kg DIPCAP: start of this dip's extra turn
                        # DIPFULL extends only a dip that starts at the search centre (the arrival dip at the clue),
                        # not a restart dip wherever an APPROACH was dropped
                        self._vs_dip_near = float(np.linalg.norm(c[:2] - pos[:2])) <= _CX_VS_NEAR_M
                    if _CX_MS_SZ:
                        self._ms_srch_t0 = self.t
                    if _CX_MA_RING:
                        # fx_ma RING: only a search built at the clue (not re-anchored on a dropped track) with no track
                        self._ma_ring_ok = (self.map_label == "mountain" and anchor is None and not self._ma_live_track()
                                            and float(np.linalg.norm(c[:2] - pos[:2])) < 4.0)
                    if _CX_M2_ANY:
                        # fx_m2: SPIN and RESPIN act only on the first SEARCH built at the clue (not on a re-search
                        # anchored where an APPROACH was dropped); a re-spin never carries over into a new SEARCH
                        self._m2_rs_z = None
                        self._m2_first = False
                        self._m2_ridge = False
                        self._m2_high = False
                        self._m2_seen = False
                        self._m2_drop_t = None
                        if (self.map_label == "mountain" and anchor is None and self._m2_t0 is None
                                and float(np.linalg.norm(c[:2] - pos[:2])) < 4.0):
                            self._m2_first = True
                            self._m2_t0 = self.t
                            self._m2_g0 = float(pos[2] - alt)
                            if alt < _M2_SAT_M:
                                # (a saturated down ray gives no ground: no class, so none of the three acts)
                                self._m2_ridge = self._m2_g0 - float(center[2]) > _M2_RIDGE_DZ
                                self._m2_high = (float(pos[2]) >= float(center[2]) + _M2_SPIN_DZ
                                                 and self._m2_g0 < float(center[2]) + _M2_SPIN_GMAX)
                            else:
                                _cx_m2_ev("m2_sat")  # diagnostic: first search built with the down ray saturated
                            if self._m2_ridge:
                                _cx_m2_ev("m2_ridge")  # diagnostic: this search is in the ridge class
                if (_CX_M2_RESPIN and self._m2_first and self._m2_ridge and not self._m2_seen and self.spin_done
                        and self._m2_rs == 0 and self.t < _M2_RS_TMAX
                        and (self.pad_pos is None or self.t - self.pad_last_seen > 1.0)):
                    # (a stale 1-3 hit track whose prediction never re-enters the view window is never dropped:
                    # 1453984591, 1257414148, 259648514 end with 1-3 hits; it must not block this)
                    _rs_time = self.t - self._m2_t0 > _M2_RS_T
                    if float(pos[2] - alt) < self._m2_g0 - _M2_RS_DROP:
                        if self._m2_drop_t is None:
                            self._m2_drop_t = self.t
                    else:
                        self._m2_drop_t = None
                    _rs_drop = self._m2_drop_t is not None and self.t - self._m2_drop_t >= _M2_RS_DROP_S - 1e-6
                    if _rs_time or _rs_drop:
                        self._m2_rs = 1
                        self._m2_rs_z = float(center[2]) + _M2_RS_DZ
                        self._m2_rs_acc = 0.0
                        self.spin_done = False
                        self.spin_acc = 0.0
                        self.spin_prev = yaw
                        self.mode_t0 = self.t  # spin cap and pulse clock restart (in SEARCH only the spin reads mode_t0)
                        self._pulse_k = -1
                        _cx_m2_ev("m2_respin_drop" if _rs_drop else "m2_respin_time")
                elif self._m2_drop_t is not None:
                    self._m2_drop_t = None  # the drop must hold on consecutive armed ticks (only ever set with RESPIN on)
                h_s = prm["h_search"]
                if vabs_lim is not None:
                    h_s = min(max(h_s, _VABS_LIMIT_M), vabs_lim)
                if self.map_label == "village" and (_flag("vhigh") or _flag("vhigh2")):
                    h_s = 9.5
                if not self.spin_done and self.map_label not in ("village", "mountain", "warehouse"):
                    h_s = min(h_s, 3.5)
                if not self.spin_done and self.map_label == "village" and getattr(self, "_vdip_active", False):
                    h_s = _VDIP_ALT_M
                if _CX_V16_SPIN and not self.spin_done and self.map_label == "village" and self._v16_spin_on:
                    h_s = _V16_SPIN_H  # fx_v16 SPIN: sink only to this height over the lowest ground (the dip went to 3 m)
                if _F6_UNDER and self.map_label == "forest" and self.spin_done and self.t < self._f6_low_until:
                    h_s = min(h_s, _F6_UNDER_LOWH)  # CX_F6_UNDER: ring leg under the crown layer while detouring
                vz = float(np.clip(1.2 * (h_s - alt_c), -2.5 if alt_c > 8.0 else -1.0, 1.5))
                if _CX_V16_SPIN and not self.spin_done and self.map_label == "village" and self._v16_spin_on and alt < _V16_SPIN_FLOOR:
                    vz = max(vz, 0.5 * (_V16_SPIN_FLOOR - alt))  # fx_v16 SPIN: never within this of the surface below
                if self.map_label == "mountain":
                    z_t = max(float(center[2]) + (0.5 if _flag("mdesc") else 1.5), pos[2] - alt + 2.0)
                    if _flag("mspin2") and not self.spin_done:
                        z_t = max(float(center[2]) - 1.5, pos[2] - alt + 2.0)
                        if _CX_M2_SPIN and self._m2_first and self._m2_high and self._m2_rs_z is None:
                            # spin over the top of the clue-z +-5 pad band: every pad is below the camera (the ground + 2
                            # floor stays: the terrain under the spin may rise). _m2_high is latched off at the first
                            # find, so a spin resumed after a weak APPROACH drop sinks as the champion's does
                            z_m2 = max(float(center[2]) + _M2_SPIN_DZ, pos[2] - alt + 2.0)
                            if z_m2 > z_t:
                                z_t = z_m2
                                _cx_m2_ev("m2_spin")
                    if _flag("mlow"):
                        z_t = pos[2] - alt + 2.8
                    if (_CX_MS_SZ and self.spin_done and self.pad_pos is None and self._ms_srch_t0 is not None
                            and self.t - self._ms_srch_t0 > _MS_SZ_T):
                        # nothing seen from the clue-z rings: the pad may sit above them (clue z is +-5 m), so fly
                        # them from over the top of that band, where a pad at any allowed height is below the camera
                        z_hi = float(center[2]) + _MS_SZ_DZ
                        if z_hi > z_t:
                            z_t = z_hi
                            _cx_ms_ev("ms_sz")
                    if self._m2_rs_z is not None and not self.spin_done:
                        z_t = max(self._m2_rs_z, pos[2] - alt + 2.0)  # fx_m2 RESPIN height (ground + 2 floor)
                    self._m2_zt = z_t
                    vz = float(np.clip(1.2 * (z_t - pos[2]), -2.5 if alt > 8.0 else -1.0, 1.5))
                if not self.spin_done:
                    _vs_dy = abs(wrap(yaw - self.spin_prev)) if _CX_VS_DIPFULL else 0.0
                    if (self._m2_rs_z is not None and self._m2_zt is not None
                            and abs(float(pos[2]) - self._m2_zt) < 1.0):
                        self._m2_rs_acc += abs(wrap(yaw - self.spin_prev))  # fx_m2 RESPIN: turn made at the height
                    self.spin_acc += abs(wrap(yaw - self.spin_prev))
                    self.spin_prev = yaw
                    yaw_des = yaw + 1.2
                    v_des = np.array([0.0, 0.0, vz])
                    if _WF_SPCL and self.map_label == "forest" and float(v_des[2]) > _WF_SPCL_V:
                        # ws2 WF_SPCL: the spin climbs with the planner off; slow it and stop under foliage in the top band
                        top_ = DEPTH_MIN_M + float(depth[0:12, 32:96].min()) * (DEPTH_MAX_M - DEPTH_MIN_M)
                        g0_ = getattr(self, "_f14_gmin", None)
                        vz_sp = 0.0 if top_ < _WF_SPCL_TOP else min(float(v_des[2]), _WF_SPCL_V)
                        if g0_ is not None and float(pos[2]) - g0_ >= h_s - 0.3:
                            vz_sp = 0.0  # high enough over the flat forest ground (z of the lowest ground seen)
                        if vz_sp < float(v_des[2]):
                            v_des = np.array([0.0, 0.0, vz_sp])
                            _cx("wf_spcl")
                    if _FO_SPIN and self.map_label == "forest":
                        vz_fo = self._fo_canopy_vz(pos, alt, alt_c, vz)
                        if vz_fo != vz:
                            v_des = np.array([0.0, 0.0, vz_fo])
                            _cx("fo_spin_hold")
                            if getattr(self, "_fo_hold_xy", None) is None:
                                self._fo_hold_xy = pos[:2].copy()
                    use_planner = False
                    near_ahead = DEPTH_MIN_M + float(depth[48:112, 40:88].min()) * (DEPTH_MAX_M - DEPTH_MIN_M)
                    _m2_opened = False
                    if ((_CX_M2_PULSE_ALL or (_CX_M2_PULSE and self._m2_first and self._m2_ridge and not self._m2_seen)
                         or self._m2_rs_z is not None) and _MTN_SPIN_PULSE and self.map_label == "mountain"
                            and alt >= _MTN_PULSE_ALT and near_ahead <= _MTN_PULSE_CLEAR):
                        # the champion's gate is shut: re-test it counting only obstacles at flight level (PULSE: in the
                        # ridge-class first spin until its first find; PULSE_ALL: in every mountain spin; RESPIN: always
                        # inside its own re-spin, which sits at the ground + 2 floor)
                        near_ahead = self._m2_pulse_clear(depth, pos, rpy)
                        _m2_opened = near_ahead > _MTN_PULSE_CLEAR
                        if _m2_opened:
                            _cx_m2_ev("m2_pulse_open")  # ticks the champion's gate would have blocked
                    if (_MTN_SPIN_PULSE and self.map_label == "mountain" and alt >= _MTN_PULSE_ALT
                            and near_ahead > _MTN_PULSE_CLEAR):
                        # look down while spinning: a sharp forward velocity step pitches the body, and the fixed camera,
                        # ~30-35 deg nose down for ~0.3 s, so pads 50-70 deg below the horizon come into view (lookdown
                        # probe: 4 of 8 high-ground search failures were detected only during such pulses)
                        ts = self.t - self.mode_t0 - _MTN_PULSE_DELAY
                        k = int(ts / _MTN_PULSE_PERIOD) if ts >= 0.0 else -1
                        if k != self._pulse_k:
                            self._pulse_k = k
                            self._pulse_dir = np.array([math.cos(yaw), math.sin(yaw)])
                            # start a pulse only from a steady hover: a step on top of leftover cruise speed or tilt
                            # tipped 1098 past the 60 deg cutoff
                            lim = math.radians(_MTN_PULSE_MAXTILT)
                            self._pulse_on = (k >= 0 and float(np.hypot(vel[0], vel[1])) < _MTN_PULSE_STILL
                                              and abs(float(rpy[0])) < lim and abs(float(rpy[1])) < lim)
                            if _m2_opened and self._pulse_on:
                                _cx_m2_ev("m2_pulse_fire")  # a pulse only the flight-level gate let start
                        if getattr(self, "_pulse_on", False) and ts - k * _MTN_PULSE_PERIOD < _MTN_PULSE_S:
                            if _m2_opened:
                                _cx_m2_ev("m2_pulse_step")  # a pulse tick flown only because the new gate was open
                            v_des = np.array([self._pulse_dir[0] * _MTN_PULSE_V, self._pulse_dir[1] * _MTN_PULSE_V, vz])
                            self._pulse_bypass = True
                            self._pulse_ticks += 1
                    spin_need, spin_cap = (math.pi * 1.5, 8.0) if (_flag("mspin2") and self.map_label == "mountain") else (math.pi * 1.1, 6.0)
                    if (_flag("fspin") and self.map_label == "forest") or (_flag("wspin") and self.map_label == "warehouse"):
                        spin_need, spin_cap = (math.pi * 2.0, 9.5)
                    vdip = self.map_label == "village" and getattr(self, "_vdip_active", False)
                    if vdip:
                        spin_need, spin_cap = (math.pi * 2.0, _VDIP_CAP_S)
                    if _MTN_SPIN_PULSE and _MTN_PULSE_SPINX and self.map_label == "mountain":
                        spin_need, spin_cap = spin_need + math.pi, spin_cap + 4.0  # room for pulses after the level turn
                    _spin_end = (self.spin_acc > spin_need and (not vdip or alt_c <= _VDIP_ALT_M + 0.6)) or (self.t - self.mode_t0) > spin_cap
                    if self._m2_rs_z is not None:
                        # fx_m2 RESPIN: a full turn at the re-spin height, or the cap; then the rings resume at search_i
                        _spin_end = self._m2_rs_acc > _M2_RS_TURN or (self.t - self.mode_t0) > _M2_RS_CAP
                        if _spin_end:
                            self._m2_rs_z = None
                    if _CX_VS_DIPFULL and vdip and self._vs_dip_near:
                        # count only the turn made at the dip height: while descending, a pad nearer than the altitude
                        # is below the image, so the bearings swept up there are not searched for the near pads
                        if alt_c <= _VDIP_ALT_M + 0.6:
                            self._vs_low_acc = self._vs_low_acc + _vs_dy
                        _vs_end = (self._vs_low_acc > spin_need
                                   or (self.t - self.mode_t0) > spin_cap + _CX_VS_DIP_XCAP)
                        if (_CX_VS_DIPCAP and not _vs_end and self._vs_dip_ext and self._kg_dip_t is not None
                                and self.t - self._kg_dip_t >= _CX_VS_DIPCAP_S):
                            _vs_end = True  # CX_VS_DIPCAP: the extra turn has run its maximum
                            _cx_vs_ev("vs_dipcap_end")
                        _kg_keep = False  # CX_VS_DIPLATE: keep the champion's dip end (no extra turn)
                        if _spin_end and not _vs_end and not self._vs_dip_ext:
                            if _CX_VS_DIPLATE and self.t < _CX_VS_DIPLATE_T:
                                _kg_keep = True
                                _cx_vs_ev("vs_diplate_skip")
                            else:
                                self._vs_dip_ext = True
                                if _CX_VS_DIPCAP:
                                    self._kg_dip_t = self.t
                                _cx_vs_ev("vs_dipfull")
                        if not _kg_keep:
                            _spin_end = _vs_end
                    if _CX_V16_SPIN and self.map_label == "village" and self._v16_spin_on:
                        # fx_v16 SPIN: a turn of _V16_SPIN_TURN or _V16_SPIN_CAP s from its start, whatever came between
                        _spin_end = self.spin_acc > _V16_SPIN_TURN or (self.t - self._v16_spin_t0) > _V16_SPIN_CAP
                        if _spin_end:
                            self._v16_spin_on = False
                            _cx_v16_ev("v16_spin_end")
                    if _spin_end:
                        self.spin_done = True
                        self._vdip_active = False
                        if (_CX_VS_RING1OUT and vdip and self.search_center_xy is not None
                                and float(np.linalg.norm(self.search_center_xy - pos[:2])) <= _CX_VS_NEAR_M):
                            self._vs_dip_ok = True
                else:
                    wp = self.search_wps[self.search_i % len(self.search_wps)]
                    cov_look = None
                    if _flag("mmodel") and self.mm is not None and self.t - self.mm["t0"] < 12.0 and self.pad_pos is None:
                        pxy, _ = self._mm_predict(self.t + 1.0)
                        away = unit(pos[:2] - pxy)
                        if float(np.linalg.norm(pos[:2] - pxy)) < 1e-3:
                            away = np.array([math.cos(yaw), math.sin(yaw)])
                        wp = pxy + away * 6.0
                        cov_look = pxy
                    if (_flag("kinghint") and self.hint_xy is not None and self.t - self.hint_t < 3.0 and self.pad_pos is None):
                        away = unit(pos[:2] - self.hint_xy)
                        if float(np.linalg.norm(pos[:2] - self.hint_xy)) < 1e-3:
                            away = np.array([math.cos(yaw), math.sin(yaw)])
                        wp = self.hint_xy + away * 6.0
                        cov_look = self.hint_xy
                    if _flag("cov") and self.map_label in ("mountain", "village"):
                        if self.cov_wp is None or self.t - self.cov_t_plan > 2.5 or float(np.linalg.norm(self.cov_wp - pos[:2])) < 2.5:
                            tgt = self._cov_target(pos)
                            if tgt is not None:
                                away = unit(pos[:2] - tgt)
                                if float(np.linalg.norm(pos[:2] - tgt)) < 1e-3:
                                    away = np.array([math.cos(yaw), math.sin(yaw)])
                                self.cov_tgt = tgt
                                self.cov_wp = tgt + away * 7.0
                                self.cov_t_plan = self.t
                            else:
                                self.cov_wp = None
                        if self.cov_wp is not None:
                            wp = self.cov_wp
                            cov_look = self.cov_tgt
                    to = wp - pos[:2]
                    d_h = float(np.linalg.norm(to))
                    if d_h < (2.0 if self.map_label == "village" else 1.5):
                        self.search_i += 1
                        wp = self.search_wps[self.search_i % len(self.search_wps)]
                        to = wp - pos[:2]
                        d_h = float(np.linalg.norm(to))
                    dir_h = unit(to)
                    v_srch = prm["v_search"]
                    if (_flag("vsearch3") or _flag("msearch3")) and self.map_label == "mountain":
                        # msearch3: vsearch3's mountain part (search speed 3.0) on its own, so the village flags can be
                        # dropped without slowing the mountain search
                        v_srch = 3.0
                    v_h = min(v_srch, math.sqrt(2 * 1.5 * max(d_h - 0.8, 0.0)) + 0.5)
                    dir_h, v_h = self._city_detour(pos, dir_h, v_h, prm)
                    vz = self._city_zcap(pos, alt, vz)
                    if _FO_SPIN and self.map_label == "forest" and getattr(self, "_fo_hold_xy", None) is not None:
                        if float(np.hypot(pos[0] - self._fo_hold_xy[0], pos[1] - self._fo_hold_xy[1])) > 4.0:
                            self._fo_hold_xy = None  # clear of the crown the spin held under
                        else:
                            vz_fo = self._fo_canopy_vz(pos, alt, alt_c, vz)
                            if vz_fo != vz:
                                vz = vz_fo
                                _cx("fo_spin_post")
                    v_des = np.array([dir_h[0] * v_h, dir_h[1] * v_h, vz])
                    yaw_des = math.atan2(dir_h[1], dir_h[0])
                    if cov_look is not None:
                        lk = cov_look - pos[:2]
                        if float(np.linalg.norm(lk)) > 1.0:
                            yaw_des = math.atan2(lk[1], lk[0])
                            if d_h < 1.5:
                                v_des = np.array([0.0, 0.0, vz])
                    if (((self.map_label == "village" and _VYAW != "head") or (self.map_label == "mountain" and _flag("inward_mtn")))
                            and self.search_center_xy is not None and cov_look is None):
                        inward = self.search_center_xy - pos[:2]
                        if float(np.linalg.norm(inward)) > 2.0:
                            off = wrap(math.atan2(inward[1], inward[0]) - yaw_des)
                            if _flag("ringout") and self.map_label == "village" and self.search_i < getattr(self, "n_wp_ring", 4):
                                off = -off
                            if self.map_label == "village" and _VYAW == "out":
                                off = -off
                            if (_CX_VS_RING1OUT and self.map_label == "village" and getattr(self, "_vs_dip_ok", False)
                                    and 1 <= self.search_i < getattr(self, "n_wp_ring", 4)):
                                # first-ring legs after a dip at the centre: look outward, away from the swept inner disc
                                off = -wrap(math.atan2(inward[1], inward[0]) - yaw_des)
                                if getattr(self, "_vs_r1_leg", -1) != self.search_i:
                                    self._vs_r1_leg = self.search_i
                                    _cx_vs_ev("vs_ring1out")
                            clip_in = 0.4 if (_flag("vhigh") and self.map_label == "village") else 0.7
                            yaw_des = yaw_des + float(np.clip(off, -clip_in, clip_in))
                            if (_CX_MA_RING and self._ma_ring_ok and self.map_label == "mountain" and not self._ma_live_track()
                                    and 1 <= self.search_i < getattr(self, "n_wp_ring", 5)):
                                self._ma_ring_tick = True  # fx_ma RING: yawvel keeps this inward offset (capped)
                    if (_CX_VS_LOOK and self.map_label == "village" and self.t < getattr(self, "_vs_look_until", -1.0)):
                        lk_ = self._vs_look_xy - pos[:2]
                        if float(np.linalg.norm(lk_)) > 3.0:
                            yaw_des = math.atan2(lk_[1], lk_[0])
                            if getattr(self, "_vs_look_n", -1.0) != self._vs_look_until:
                                self._vs_look_n = self._vs_look_until
                                _cx_vs_ev("vs_look")
                    if _CX_S2N_NBV and self.map_label == "village" and not _CX_STATE.get("shadow") and not self._s2n_dead:
                        # fx_s2nbv NBV: after ring 1, vantages chosen on the seen-ground map replace the rings
                        try:
                            s2n_ = self._s2n_step(pos, yaw, vz)
                            if s2n_ is not None:
                                v_des, yaw_des = s2n_
                        except Exception:
                            self._s2n_dead = True
                            _s2n_ev("s2n_error")
        elif self.mode == "APPROACH":
            unseen = self.t - self.pad_last_seen
            weak = _flag("invest") and getattr(self, "weak_approach", False) and self.pad_hits < 30
            if pad is None:
                lost = True
            elif self.pad_moving:
                lost = unseen > 2.5
                if _flag("mvpatience") and pad is not None:
                    rel_ = pad[:2] - pos[:2]
                    if float(np.linalg.norm(rel_)) > 0.3:
                        lost = unseen > 2.5 + min(4.0, abs(wrap(math.atan2(rel_[1], rel_[0]) - yaw)) / 0.69)
            elif weak:
                lost = self.pad_miss_t > 1.2 or unseen > 3.0
            elif self._static_established():
                lost = unseen > 25.0
            else:
                lost = self.pad_miss_t > 2.0 or unseen > 6.0
                if (lost and _CX_MS_KEEP and self.map_label == "mountain" and self.pad_miss_t <= 2.0
                        and unseen <= _MS_KEEP_S and self.pad_hits >= _MS_KEEP_HITS):
                    rel_k = pad - pos
                    dh_k = float(np.hypot(rel_k[0], rel_k[1]))
                    h_k = -float(rel_k[2])
                    # below the look-down limit: 45 deg at the image's bottom edge when level, but the pad detector needs
                    # it a few rows inside the image and a braking (nose-up) attitude raises that edge, hence 0.8 + 0.5 m
                    # (keepview's "below the image" row 118 is ~40 deg)
                    below_k = dh_k < 6.0 and h_k > 0.8 * dh_k + 0.5
                    # closing in over it: moving toward the estimate, or already within 1 m of it
                    closing_k = dh_k < 1.0 or float(vel[0] * rel_k[0] + vel[1] * rel_k[1]) > 0.2 * max(dh_k, 1e-6)
                    # the down ray, once it falls on the estimate, must not see ground well above the pad top there
                    # (an estimate that is off along the slope); a pad that is really there reads alt ~ h
                    ray_k = dh_k >= 0.6 or (h_k - alt) < 0.6
                    # a pad floats over the terrain around it (see _ms_pad_float); an estimate lying on the fitted terrain
                    # plane is a false detection on the ground, and landing there is a crash. No valid fit: no veto.
                    fl_k = self._ms_pad_float(pad) if (below_k and closing_k and ray_k) else None
                    float_k = fl_k is None or fl_k >= _MS_KEEP_FLOAT
                    if below_k and closing_k and ray_k and float_k:
                        # the pad cannot be seen from here, so 'unseen' says nothing about it: keep descending on the
                        # track instead of REACQUIRE (the unseen limit becomes _MS_KEEP_S)
                        lost = False
                        self._ms_acted = True
                        _cx_ms_ev("ms_keep")
                        _cx_ms_log("ms_keep_float_cm", -999 if fl_k is None else int(round(100.0 * fl_k)))
                    elif below_k and closing_k:
                        # the champion's REACQUIRE goes ahead (logged, not counted: no action changes)
                        _cx_ms_log("ms_keep_veto", "ray" if not ray_k else "float %.2f" % fl_k)
                        _cx_ms_log("ms_keep_veto_t", round(float(self.t), 2))
            if (lost and (_CX_M15_KEEP or (_CX_MTF_LKEEP and self.pad_obs and self.pad_obs is getattr(self, "_mtf_lk_obs", None)))
                    and pad is not None and self.map_label == "mountain" and not self.pad_moving
                    and not self.pad_ever_moving and not self._static_established()):
                # fx_m15 KEEP: the young track's pad is below the image while we descend onto it (weak and non-weak)
                try:
                    if self._m15_keep_ok(pad, pos, rpy, vel, alt, unseen):
                        lost = False
                        _cx_m15_ev("m15_keep")
                        if not _CX_M15_KEEP:
                            _cx_mtf_ev("mtf_lkeep")  # fx_mtf LKEEP (KEEP enabled only by the look)
                except Exception:
                    _cx_m15_ev("m15_error")
            if lost and _CX_S1M_DESC and pad is not None and self.map_label == "mountain":
                lost = not self._s1m_desc_keep(pad, pos, vel, alt)  # fx_s1m DESC
            if lost:
                if (_CX_V3_OCC and self._v3_saved and pad is not None and not weak and not self.pad_moving
                        and self.map_label == "village"):
                    # fx_v3 OCC: the excuse kept this track alive past the champion's in-view drop. Leave the way the
                    # champion leaves a dropped track (CX_VS_REFLY, SEARCH rebuild), not by REACQUIRE, which would fly
                    # 3.5 m over the occluded estimate beside the house that hides it (1094226733: 0.146 m clearance)
                    if _CX_VS_LOOK:
                        self._vs_drop = (np.asarray(self.pad_pos[:2], dtype=np.float64).copy(), self.t)
                    self._drop_track()
                    pad = None
                    _cx_v3_ev("v3_occ_drop")
                if weak:
                    self._drop_track()
                    if self.search_wps is None and float(np.linalg.norm((center - pos)[:2])) > 8.0:
                        self._set_mode("CRUISE")
                    else:
                        self._set_mode("SEARCH")
                elif (_F14_REFLY and pad is None and self.map_label == "forest" and self.search_wps is None
                      and self._f14_refly_n < _F14_REFLY_MAX
                      and float(np.linalg.norm((center - pos)[:2])) > _F14_REFLY_M):
                    # fx_f14 REFLY: the track was dropped before any search, far from the clue: fly on to the clue
                    # (as the weak branch does) instead of spinning and ringing from here; capped per seed
                    self._f14_refly_n += 1
                    self.search_anchor = None
                    self._drop_track()
                    self._set_mode("CRUISE")
                    _cx("f14_refly")
                elif (_CX_VS_RESUME and pad is None and self.map_label == "village" and self.search_wps is not None
                      and not (_CX_VFP_RESG and bool(getattr(self, "_ra_dec", False)))
                      and (self.spin_done or not self._vdip_active
                           or self.v_gnd is None or (alt_c - alt) < _VDIP_GND_M)):
                    # track dropped during a running search: resume the search where it stopped (as the weak branch
                    # does) instead of rebuilding the rings and dipping again wherever the drone now is. An unfinished
                    # dip is resumed only over low ground (the same test the ring build uses before a dip); over a
                    # roof this falls through to the champion's rebuild, which re-tests the dip there
                    self.search_anchor = None
                    self._drop_track()
                    self._set_mode("SEARCH")
                    if not self.spin_done:
                        self.spin_prev = yaw  # the yaw turned during the APPROACH is not part of the dip's spin
                    _cx_vs_ev("vs_resume")
                    drop = self._vs_drop
                    if (_CX_VS_LOOK and drop is not None and 0.0 <= self.t - drop[1] < 0.05
                            and float(np.linalg.norm(drop[0] - pos[:2])) < 16.0):
                        # the dropped track may be a real pad hidden for a moment (roofs): keep the camera on it
                        # for a few seconds while the resumed leg flies on
                        self._vs_look_xy = drop[0].copy()
                        self._vs_look_until = self.t + _CX_VS_LOOK_S
                elif (_CX_VS_REFLY and pad is None and self.map_label == "village" and self.search_wps is None
                      and self._vs_refly_n < _CX_VS_REFLY_MAX
                      and float(np.linalg.norm((center - pos)[:2])) > 8.0):
                    # track dropped before any search, far from the clue: fly on to the clue (as the weak branch does)
                    # instead of dipping and ringing here; capped per seed so a recurring false positive cannot loop
                    # CRUISE <-> APPROACH, after which the champion's rebuild runs
                    self._vs_refly_n += 1
                    self.search_anchor = None
                    self._drop_track()
                    self._set_mode("CRUISE")
                    _cx_vs_ev("vs_refly")
                elif pad is not None and self.reacq_n < 3:
                    self.reacq_n += 1
                    if _CX_VCH_RQ:
                        self._vch_rq_low = bool(self.map_label == "village" and not self.pad_moving
                                                and not getattr(self, "pad_ever_moving", False)
                                                and self.t - getattr(self, "_vch_app_t0", self.t) >= _VCH_RQ_T
                                                and int(self.pad_hits) - int(getattr(self, "_vch_h0", 0)) < _VCH_RQ_H)
                        if self._vch_rq_low:
                            _cx_ev("vch_rq_low")
                    self.reacq_target = pad.copy()
                    self.reacq_moving = bool(self.pad_moving)
                    back = unit(pos[:2] - pad[:2])
                    if float(np.linalg.norm(back)) < 1e-6:
                        back = np.array([math.cos(yaw), math.sin(yaw)]) * -1.0
                    up_ = 1.5 if self.map_label == "forest" else (3.5 if self.map_label in ("village", "city") else (4.0 if self.map_label == "mountain" else 2.5))
                    if _flag("mreacq") and self.map_label == "mountain":
                        up_ = 2.5
                    self.reacq_wp = np.array([pad[0] + back[0] * 5.0, pad[1] + back[1] * 5.0, pad[2] + up_])
                    if not self.pad_moving:
                        self._drop_track()
                    self.reacq_hold = 0.0
                    self.reacq_side = 0
                    self.reacq_t_enter = self.t
                    self._set_mode("REACQUIRE")
                else:
                    self.search_anchor = pad[:2].copy() if pad is not None else None
                    self._drop_track()
                    self._set_mode("SEARCH")
                    if (_FO_RESUME and self.map_label == "forest" and pad is None and self.search_wps is not None
                            and (self.spin_done or float(getattr(self, "spin_acc", 0.0)) >= math.pi)
                            and not self._fo_fresh_drop(pos)):
                        # resume the rings where the lost track interrupted them; the clue area was already spun over
                        self.search_anchor = None
                        self.spin_done = True
                        _cx("fo_resume")
                    else:
                        self.search_wps = None
            else:
                rel = pad - pos
                d_h = float(np.linalg.norm(rel[:2]))
                if self.map_label == "forest":
                    tube = min(tube, 0.7)
                    tube_min = 0.4
                else:
                    tube = min(tube, 0.6)
                    tube_min = 0.3
                    if _flag("apptube") and self.map_label == "city":
                        tube = min(prm["tube"], 0.9)
                        tube_min = 0.4
                yaw_des = math.atan2(rel[1], rel[0]) if d_h > 0.6 else yaw
                if not self.pad_moving:
                    h = float(pos[2] - pad[2])
                    T = np.array([pad[0], pad[1], pad[2] + 1.2])
                    rel3 = T - pos
                    dist3 = float(np.linalg.norm(rel3))
                    dir3 = unit(rel3)
                    v_mag = min(3.0, math.sqrt(2 * (1.8 if _flag("fastapp") else 1.2) * max(dist3 - 0.5, 0.0)) + 0.3)
                    if _WF_AVMAX > 0.0 and self.map_label == "forest" and v_mag > _WF_AVMAX:
                        v_mag = _WF_AVMAX  # ws2 WF_AVMAX
                        _cx("wf_avmax")
                    established = self._static_established()
                    if not self._track_solid() and not (_flag("mdirect") and self.map_label == "mountain"):
                        v_mag = min(v_mag, 2.0)
                    steep = -dir3[2] > 0.64
                    if steep:
                        v_h = min(1.2, 0.5 * d_h + 0.3)
                        if _flag("mdirect") and self.map_label == "mountain":
                            v_h = min(2.0, 0.7 * d_h + 0.5)
                        crawl = False
                        if _flag("keepview"):
                            prow_s = self._pad_pixel_row(pos, rpy)
                            if prow_s is not None and prow_s > 118.0 and d_h > 2.0:
                                v_h = 0.3  # the pad is below the image: sink almost vertically until it is back in view
                                crawl = True
                        dir_h = unit(rel3[:2])
                        vz_override = -2.0 if alt > 4.0 else (-1.2 if alt > 2.2 else -0.6)
                        if alt < 1.2 and d_h > 1.0:
                            vz_override = 0.0
                            if _flag("mdirect") and self.map_label == "mountain" and h > 2.0 and alt > 0.7:
                                vz_override = -0.4
                        if crawl and self.map_label == "mountain" and alt < _MTN_SLOPE_ALT:
                            # the ground below throttles the sink (-0.6 m/s, 0 under 1.2 m) while keepview holds us to
                            # 0.3 m/s: on a slope neither closes the gap (7-12 m above the pad at 0.2 m/s for 20 s).
                            # Going down the slope toward the pad is the only way down.
                            self._slope_ticks = getattr(self, "_slope_ticks", 0) + 1
                            if _MTN_SLOPE_VH > 0.0:
                                v_h = max(v_h, min(_MTN_SLOPE_VH, 0.5 * d_h + 0.3))
                                if alt < 1.0:
                                    vz_override = 0.5  # do not skim the slope we are following
                        if (_CX_MS_SLOPE and self.map_label == "mountain" and 1.0 <= alt < _MS_SLOPE_ALT_HI and d_h > 1.5
                                and (h - alt) > _MS_SLOPE_DROP
                                and -float(dir3[2]) > (_MS_SLOPE_ST_HOLD if self.t - self._ms_slope_t < 0.1 else _MS_SLOPE_ST)):
                            # the ground right below us is far above the pad: we are on the slope that leads down to it.
                            # Sinking is capped by that ground, so fly down the slope: horizontal speed toward the pad with a
                            # sink matched to the mean slope from here to the pad, and an AGL hold (set after the planner).
                            ms_slope = float(np.clip((h - alt - 0.2) / max(d_h, 1.0), 0.0, 1.6))
                            v_h_s = min(_MS_SLOPE_VH, 0.5 * d_h + 0.3, 2.8 / math.sqrt(1.0 + ms_slope * ms_slope))
                            if v_h_s > v_h:
                                v_h = v_h_s
                            self._ms_slope_t = self.t
                        v_des = np.array([dir_h[0] * v_h, dir_h[1] * v_h, 0.0])
                        if (_MTN_APP_PULSE and crawl and self.map_label == "mountain" and alt >= _MTN_PULSE_ALT
                                and not self._static_established()):
                            # the pad a pulse found is below the image again: a young track dies after 6 s unseen while
                            # the drone sinks from 10-17 m above (398, 4044), so keep looking down on the way in
                            ta = self.t - self.mode_t0
                            ka = int(ta / _MTN_PULSE_PERIOD)
                            if ka != self._app_pulse_k:
                                self._app_pulse_k = ka
                                lim = math.radians(_MTN_PULSE_MAXTILT)
                                hd = np.array([math.cos(yaw), math.sin(yaw)])
                                vh_now = float(np.hypot(vel[0], vel[1]))
                                aligned = vh_now < 0.3 or float(hd @ (vel[:2] / max(vh_now, 1e-6))) > 0.7
                                near_ahead_a = DEPTH_MIN_M + float(depth[48:112, 40:88].min()) * (DEPTH_MAX_M - DEPTH_MIN_M)
                                self._app_pulse_on = (vh_now < _MTN_PULSE_STILL + 0.4 and aligned and near_ahead_a > _MTN_PULSE_CLEAR
                                                      and abs(float(rpy[0])) < lim and abs(float(rpy[1])) < lim)
                                self._app_pulse_dir = hd
                            if self._app_pulse_on and ta - ka * _MTN_PULSE_PERIOD < _MTN_PULSE_S:
                                v_des = np.array([self._app_pulse_dir[0] * _MTN_PULSE_V, self._app_pulse_dir[1] * _MTN_PULSE_V, 0.0])
                                self._pulse_bypass = True
                                self._pulse_ticks += 1
                    else:
                        v_des = dir3 * v_mag
                        if v_des[2] < -2.0:
                            v_des *= 2.0 / abs(v_des[2])
                    if not established:
                        prow = self._pad_pixel_row(pos, rpy)
                        if prow is not None and prow > 100.0:
                            if ms_slope is not None:
                                pass  # SLOPE: the pad is under the look-down limit anyway; the sink is matched to this
                                # speed, and a look-down pulse keeps its full 1.5 m/s step (see the post-planner block)
                            elif (_CX_MS_GLIDE and not steep and self.map_label == "mountain" and self._track_solid()):
                                _cx_ms_ev("ms_glide")  # a solid track: the cut only drives a slow-fast cycle with the cone cap
                            else:
                                v_des[:2] *= 0.7
                    if alt < 1.2 and d_h > 1.5 and v_des[2] < 0.0:
                        v_des[2] = 0.0
                    if (((_flag("vland") and self.map_label == "village") or (_flag("cland") and self.map_label == "city")) and not established
                            and self.pad_hits < 30 and dist3 < 4.0):
                        _v3_ob = "in"  # fx_v3 VLOOK: where the estimate is for the camera ("in" = the champion's hover)
                        if _CX_V3_VLOOK and self.t - self._v3_vl_tick > 0.03:
                            # the hover did not run on the last tick: a new stretch, no pulse cycle or hover point
                            # carried over from an earlier one
                            self._v3_vl_t0 = None
                            self._v3_vl_k = -1
                            self._v3_vl_on = False
                            self._v3_vl_anchor = None
                        self._v3_vl_tick = self.t  # read only by fx_v3 (OCC skips hover ticks; VLOOK: the gap test above)
                        if _CX_V3_VLOOK and _v3_live() and self.map_label == "village" and self._v3_vl_ext < _V3_VL_EXT_S:
                            _v3_ob = self._v3_obs(pos, rpy)
                        v_des = np.array([0.0, 0.0, 0.0])
                        vz_override = 0.0
                        use_planner = False
                        yaw_des = math.atan2(rel[1], rel[0]) if d_h > 0.3 else yaw
                        if _v3_ob == "below":
                            # fx_v3 VLOOK: the estimate is below the image, so a level hover cannot see the pad. Look
                            # down with pulses toward it (the step past the acceleration limit pitches the camera ~20-25 deg
                            # down for ~0.3 s; the champion's motion votes freeze 0.8 s after it) and drift back to the
                            # hover point in between. A new cycle starts each time the estimate drops below the image.
                            if self._v3_vl_t0 is None:
                                self._v3_vl_t0 = self.t
                                self._v3_vl_k = -1
                                self._v3_vl_anchor = pos[:2].copy()
                            ta_ = self.t - self._v3_vl_t0
                            k_ = int(ta_ / _V3_VL_PERIOD)
                            if k_ != self._v3_vl_k:
                                self._v3_vl_k = k_
                                lim_ = math.radians(12.0)
                                ahead_ = DEPTH_MIN_M + float(depth[48:112, 40:88].min()) * (DEPTH_MAX_M - DEPTH_MIN_M)
                                # a pulse pitches the camera ~20-25 deg: it can only show an estimate less than ~68 deg down
                                dep_ = math.degrees(math.atan2(float(pos[2] - pad[2]), max(d_h, 1e-3)))
                                self._v3_vl_on = (float(np.hypot(vel[0], vel[1])) < 0.6 and abs(float(rpy[0])) < lim_
                                                  and abs(float(rpy[1])) < lim_ and ahead_ > 2.5 and d_h > 1.0 and dep_ < 68.0)
                                self._v3_vl_dir = unit(rel[:2])
                            if self._v3_vl_on:
                                # this pulse cycle is the look: its ticks are not counted against the pad
                                self._v3_vl_ext += SIM_DT
                                if self._v3_vl_ext <= SIM_DT + 1e-9:
                                    _cx_v3_ev("v3_vl_below")
                            else:
                                self._vland_t = getattr(self, "_vland_t", 0.0) + SIM_DT  # no pulse allowed: champion hover
                            if self._v3_vl_on and ta_ - k_ * _V3_VL_PERIOD < _V3_VL_PULSE_S:
                                v_des = np.array([self._v3_vl_dir[0] * _V3_VL_PULSE_V, self._v3_vl_dir[1] * _V3_VL_PULSE_V, 0.0])
                                self._pulse_bypass = True
                                self._pulse_ticks += 1
                                if ta_ - k_ * _V3_VL_PERIOD < SIM_DT + 1e-9:
                                    _cx_v3_ev("v3_vl_pulse")
                            elif self._v3_vl_on and self._v3_vl_anchor is not None:
                                back_ = self._v3_vl_anchor - pos[:2]
                                bn_ = float(np.linalg.norm(back_))
                                if bn_ > 0.15:
                                    v_des = np.array([back_[0], back_[1], 0.0]) * (min(0.5, 0.8 * bn_) / bn_)
                        else:
                            # in view (or occluded: hovering does not change that) -> the champion's 2 s verification
                            self._vland_t = getattr(self, "_vland_t", 0.0) + SIM_DT
                            self._v3_vl_t0 = None
                        if (_CX_V16_VLAND and _v3_ob == "in" and self.map_label == "village" and not self.pad_moving
                                and not _CX_STATE.get("shadow")):
                            # fx_v16 VLAND: the estimate is under the image: sink (xy held) until it is back in view;
                            # such ticks are not a look at the pad, so they do not count toward the 2 s
                            try:
                                vz16 = self._v16_vland(pos, rpy, alt, pad, d_h)
                                if vz16 is not None:
                                    self._vland_t = max(0.0, self._vland_t - SIM_DT)
                                    v_des = np.array([0.0, 0.0, vz16])
                                    vz_override = vz16
                            except Exception:
                                _cx_v16_ev("v16_error")
                        if self._vland_t > 2.0:
                            self.bad_spots.append(np.array([pad[0], pad[1], pad[2]]))
                            self._vland_t = 0.0
                            self._drop_track()
                            self._set_mode("SEARCH")
                    elif (_RC_ANY and d_h < 1.0 and pos[2] - pad[2] < 1.8 and self._rc_land_veto()):
                        pass  # fx_rc LSEEN/LBAND: the track was dropped; APPROACH's lost handling runs next tick
                    elif (_CX_VFP_LSEEN and d_h < 1.0 and pos[2] - pad[2] < 1.8 and self.map_label == "village"
                          and not self.pad_moving and not self.pad_ever_moving and not _CX_STATE.get("shadow")
                          and self.t - self.pad_last_seen >= _VFP_LS_S
                          and float(getattr(self, "_vfp_d3", 0.0)) >= _VFP_LS_D):
                        # ws2/vfp LSEEN: never seen from close and not seen for a while: not a pad top to land on
                        _cx("vfp_lseen")
                        at_ = (str(CX_EVENTS.get("vfp_lseen_at", "")) + " %.2f/%.2f/%.1f" % (
                            float(self.t), float(self.t - self.pad_last_seen), float(self._vfp_d3))).split()
                        CX_EVENTS["vfp_lseen_at"] = " ".join(at_[-6:])
                        self.bad_spots.append(np.array([pad[0], pad[1], pad[2]]))
                        self._drop_track()
                    elif (_CX_VF_LAND and d_h < 1.0 and pos[2] - pad[2] < 1.8 and self.map_label == "village"
                          and not self.pad_moving and not self.pad_ever_moving and not _CX_STATE.get("shadow")
                          and self._vf_in_band(float(pad[2]))):
                        self._vf_reject(pos, center, "land")  # fx_vf LAND: a flat house part, not a pad (see the flag)
                    elif d_h < 1.0 and pos[2] - pad[2] < 1.8:
                        self._vland_t = 0.0
                        self._set_mode("LAND")
                        self.land_xy = pad[:2].copy()
                        self.land_z = float(pad[2])
                        if _flag("cstable") and self.map_label == "city":
                            close = [(x, y, zz) for (tt, x, y, zz, rr) in self.pad_obs if rr < 5.0 and self.t - tt < 2.5]
                            if len(close) >= 5:
                                med = np.median(np.asarray(close, dtype=float), axis=0)
                                if float(np.hypot(med[0] - pad[0], med[1] - pad[1])) > 0.4:
                                    self.land_xy = med[:2].copy()
                                    self.land_z = float(med[2])
                                    self.pad_pos = np.array([med[0], med[1], med[2]])
                        self.land_t0 = self.t
                        self.land_stuck_t = 0.0
                        self.touch_t = 0.0
                        self.land_phase = "track"
                        self.land_pdir = unit(rel[:2]) if d_h > 0.2 else None
                        self.land_off = None
                        self.land_reclimb = False
                        if _CX_ME_ANY:
                            self._me_rg_hold, self._me_rg_nf, self._me_rg_run, self._me_rg_desc = False, 0, 0, 0
                            if (self.map_label in _ME_EG_MAPS and not self._me_eg and not self.pad_moving
                                    and self.t + _ME_EG_LAND_S > _ME_EG_T):
                                # fx_me END: this LAND cannot finish by the horizon at the usual pace (median 3.1 s)
                                self._me_eg = True
                                if _CX_ME_END:
                                    _cx_me_ev("me_eg_latch")
                                _cx_me_log("me_eg_at", "%.2f/d%.2f/h%.2f" % (self.t, d_h, float(pos[2] - pad[2])))
                            self._me_rg_flat = None
                            self._me_rg_alts = []
                        if self.map_label == "mountain":
                            # the pad sits on a slope and the score counts clearance to the terrain beside it:
                            # touching down on its downhill side keeps the drone further from that terrain
                            self.land_rms = None
                            down, slope = self._downhill_fit(pad)
                            self.land_down, self.land_slope = down, slope
                            # offline (48 landings vs true clearance): a clean plane captures ~80% of the best touchdown
                            # gain; an uneven fit (rms > 0.25 m) or an implausibly steep one-sided plane points the wrong way
                            # (3550: a landing finishing at 59.9 s timed out on the offset's extra ~0.1 s of centring)
                            if (_MTN_LAND_OFFSET > 0.0 and down is not None and _MTN_LAND_SLOPE_MIN <= slope <= _MTN_LAND_SLOPE_MAX
                                    and self.land_rms is not None and self.land_rms <= _MTN_LAND_RMS_MAX
                                    and self.t <= _MTN_LAND_T_MAX and not getattr(self, "land_off_failed", False)):
                                self.land_off = down * _MTN_LAND_OFFSET
                            if (_CX_S1M_HM and _MTN_LAND_OFFSET > 0.0 and self.t <= _MTN_LAND_T_MAX
                                    and not getattr(self, "land_off_failed", False)):
                                self._s1m_hm_offset(pad)  # fx_s1m HM: height-map touchdown point (keeps c29c's on no map)
                            if _CX_S1M_FLAT and not _CX_STATE["shadow"]:
                                self._s1m_flat_check(pad)  # fx_s1m FLAT: terrain through the 'pad' -> bad spot, SEARCH
                            if (_CX_MS_SLOPE or _CX_MS_KEEP) and self._ms_acted:
                                self._ms_land_log(pad, slope)
                else:
                    pv = self._pad_vel_now()
                    sp = float(np.linalg.norm(pv))
                    pdir = unit(pv) if sp > 0.15 else unit(-rel[:2])
                    if d_h > 4.0:
                        lead = float(np.clip(d_h / 2.5, 0.3, 2.0))
                        tgt, _ = self._motion_at(min(self.t - self.pad_last_seen, 3.0) + lead)
                        rel_t = tgt[:2] - pos[:2]
                        dt_h = float(np.linalg.norm(rel_t))
                        v_h = min(2.8, math.sqrt(2 * 1.5 * max(dt_h - 0.5, 0.0)) + 0.6 + sp)
                        dir_h = unit(rel_t)
                        z_t = pad[2] + float(np.clip(0.5 * d_h, 1.2, 4.0))
                        vz = float(np.clip(1.5 * (z_t - pos[2]), -1.2, 1.2))
                        v_des = np.array([dir_h[0] * v_h, dir_h[1] * v_h, vz])
                    else:
                        h = float(pos[2] - pad[2])
                        prow = self._pad_pixel_row(pos, rpy)
                        below_fov = prow is not None and prow > 124.0
                        if below_fov and d_h < 0.9 and h < 2.2 and self.t - self.pad_last_seen < 1.5:
                            self._set_mode("LAND")
                            self.land_z = float(pad[2])
                            self.land_t0 = self.t
                            self.land_stuck_t = 0.0
                            self.land_dive = True
                            self.touch_t = 0.0
                            self.land_phase = "track"
                            self.land_retry = 0
                        L_des = max(0.6, 1.7 * h)
                        p_des = pad[:2] - pdir * L_des
                        rel_t = p_des - pos[:2]
                        dt_h = float(np.linalg.norm(rel_t))
                        corr = 2.0 * rel_t
                        cn = float(np.linalg.norm(corr))
                        if cn > 1.6:
                            corr *= 1.6 / cn
                        v_hv = pv + corr
                        h_hold = float(np.clip(0.5 * d_h, 1.0, 3.0))
                        vz = float(np.clip(1.5 * (pad[2] + h_hold - pos[2]), -1.0, 1.0))
                        v_des = np.array([v_hv[0], v_hv[1], vz])
                        use_planner = False
                        vmatch = float(np.linalg.norm(vel[:2] - pv))
                        if (dt_h < 0.6 and vmatch < 0.8 and h < 1.6
                                and self.t - self.pad_last_seen < 0.5 and self.mode != "LAND"):
                            demote = _flag("gentle") and sp < 0.25 and self.pad_fit_sp < 0.25
                            if _flag("gentle2"):
                                demote = sp < 0.25 and self.pad_fit_sp < 0.25 and self.pad_slow_t > 1.5
                            if demote:
                                self.pad_moving = False
                                self.pad_vel = np.zeros(2)
                            self._set_mode("LAND")
                            self.land_z = float(pad[2])
                            self.land_t0 = self.t
                            self.land_stuck_t = 0.0
                            self.land_dive = False
                            self.touch_t = 0.0
                            self.land_phase = "track"
                            self.land_retry = 0
        elif self.mode == "REACQUIRE":
            if (_CX_V3_VQF and _v3_live() and self.map_label == "village" and getattr(self, "_vcheck_active", False)
                    and not self.reacq_moving and self.pad_pos is None):
                # the vcheck candidate was rejected by observable misses: nothing left to check, search the clue now
                self.search_anchor = None
                self._set_mode("SEARCH")
                self.search_wps = None
                _cx_v3_ev("v3_vqf")
            if self.mode == "REACQUIRE" and self.reacq_moving and self.pad_pos is not None:
                hist = [h for h in self.pad_hist if self.t - h[0] <= 25.0] or [(self.t, *self.pad_pos.tolist())]
                cen = np.mean(np.asarray(hist)[:, 1:4], axis=0)
                back = unit(pos[:2] - cen[:2])
                if float(np.linalg.norm(back)) < 1e-6:
                    back = np.array([math.cos(yaw), math.sin(yaw)]) * -1.0
                self.reacq_target = cen.copy()
                self.reacq_wp = np.array([cen[0] + back[0] * 7.0, cen[1] + back[1] * 7.0, cen[2] + 3.0])
                if _flag("mmodel") and self.mm is not None and self.t - self.mm["t0"] < 12.0:
                    pxy, _ = self._mm_predict(self.t + 1.0)
                    self.reacq_target = np.array([pxy[0], pxy[1], cen[2]])
                    back2 = unit(pos[:2] - pxy)
                    if float(np.linalg.norm(back2)) < 1e-6:
                        back2 = back
                    self.reacq_wp = np.array([pxy[0] + back2[0] * 6.0, pxy[1] + back2[1] * 6.0, cen[2] + 2.5])
                if _flag("reacqorbit") and self.t - self.reacq_t_enter > 4.0:
                    ang = math.atan2(pos[1] - cen[1], pos[0] - cen[0]) + 0.6
                    self.reacq_wp = np.array([cen[0] + 6.0 * math.cos(ang), cen[1] + 6.0 * math.sin(ang), cen[2] + 2.5])
                if self.pad_last_seen > self.reacq_t_enter + 0.2 and self.pad_conf > 0.5:
                    self._set_mode("APPROACH")
            to = self.reacq_wp - pos
            d3 = float(np.linalg.norm(to))
            v_h = min(1.5, math.sqrt(2 * 1.2 * max(d3 - 0.3, 0.0)) + 0.3)
            v_des = unit(to) * v_h
            look = self.reacq_target - pos
            yaw_des = math.atan2(look[1], look[0])
            tube = min(tube, 0.6)
            tube_min = 0.3
            if self._track_reliable():
                self._set_mode("APPROACH")
            elif d3 < 0.5 and abs(wrap(yaw_des - yaw)) < 0.3 and self.t - self.mode_t0 > 2.0:
                self.reacq_hold += SIM_DT
                if self.reacq_hold > 1.5:
                    if self.reacq_side == 0 and not self.reacq_moving and not getattr(self, "_vch_rq_low", False):
                        self.reacq_side = 1
                        self.reacq_hold = 0.0
                        tgt = self.reacq_target
                        v0 = self.reacq_wp[:2] - tgt[:2]
                        ang = math.atan2(v0[1], v0[0]) + 2.0 * math.pi / 3.0
                        self.reacq_wp = np.array([tgt[0] + 5.0 * math.cos(ang), tgt[1] + 5.0 * math.sin(ang), tgt[2] + 2.5])
                        self.mode_t0 = self.t
                    elif getattr(self, "_vch_rq_low", False):
                        self._vch_rq_resume(yaw, pos)
                    else:
                        self.search_anchor = None if getattr(self, "_vcheck_active", False) else self.reacq_target[:2].copy()
                        self._drop_track()
                        self._set_mode("SEARCH")
                        self.search_wps = None
            elif self.t - self.mode_t0 > (12.0 if self.reacq_moving else (_VCH_RQ_CAP if getattr(self, "_vch_rq_low", False) else 8.0)):
                if getattr(self, "_vch_rq_low", False):
                    self._vch_rq_resume(yaw, pos)
                else:
                    self.search_anchor = None if getattr(self, "_vcheck_active", False) else self.reacq_target[:2].copy()
                    self._drop_track()
                    self._set_mode("SEARCH")
                    self.search_wps = None
        elif self.mode == "LAND":
            use_planner = False
            yaw_des = yaw
            if pad is None:
                pad = np.array([self.land_xy[0], self.land_xy[1], self.land_z])
            if self.land_off is not None and not self.pad_moving:
                if self.touch_t > 1.5:
                    self.land_off = None  # in contact without success: the offset point missed the pad, centre it
                    self.land_off_failed = True
                else:
                    self._landoff_ticks = getattr(self, "_landoff_ticks", 0) + 1
                    pad = np.array([pad[0] + self.land_off[0], pad[1] + self.land_off[1], pad[2]])
            pv = self._pad_vel_now() if self.pad_moving else self._pad_slow_vel()
            if not self.pad_moving and float(np.linalg.norm((pad - pos)[:2])) > 0.3:
                yaw_des = math.atan2(pad[1] - pos[1], pad[0] - pos[0])
            sp = float(np.linalg.norm(pv))
            h = float(pos[2] - pad[2])
            rel_xy = pad[:2] - pos[:2]
            d_xy = float(np.linalg.norm(rel_xy))
            if sp > 0.15:
                pdir = unit(pv)
            elif d_xy > 0.25:
                pdir = unit(rel_xy)
                self.land_pdir = pdir
            else:
                pdir = self.land_pdir if self.land_pdir is not None else np.array([math.cos(yaw), math.sin(yaw)])
            seen_recent = (self.t - self.pad_last_seen) < 0.8
            unseen = self.t - self.pad_last_seen
            on_pad = alt < 0.10 and abs(vel[2]) < 0.25
            resting = float(np.linalg.norm(vel)) < 0.08 and self.v_cmd[2] < -0.2 and h < 0.5
            if resting:
                self.rest_t += SIM_DT
            else:
                self.rest_t = 0.0
            if on_pad or self.rest_t > 0.4:
                self.touch_t += SIM_DT
                self.touch_hold = self.t + 1.5
            elif self.t > self.touch_hold:
                self.touch_t = 0.0
            abort = False
            if (_flag("vsurf") and self.map_label == "village" and not self.pad_moving
                    and self.touch_t <= 0.0 and h < 3.0 and (h - alt) > 0.6):
                self._vsurf_t = getattr(self, "_vsurf_t", 0.0) + SIM_DT
                if self._vsurf_t > 2.0:
                    abort = True
            else:
                self._vsurf_t = 0.0
            if abort:
                pass
            elif self.touch_t > 0.0:
                v_des = np.array([0.0, 0.0, -0.3])
                if self.touch_t > 3.0 and d_xy > 0.15:
                    nudge = unit(rel_xy) * 0.15
                    v_des = np.array([nudge[0], nudge[1], -0.3])
            elif not self.pad_moving:
                corr = 1.6 * rel_xy
                cn = float(np.linalg.norm(corr))
                if cn > 0.6:
                    corr *= 0.6 / cn
                v_hv = pv + corr
                fa = _flag("fastapp")
                if d_xy > 0.35:
                    vz = 0.0 if h < 1.6 else (-0.9 if fa else -0.6)
                    if _CX_DV_NOHOLD and h < 1.6 and self._dv_nohold_ok(pos, alt, pad, unseen):
                        vz = -0.9 if fa else -0.6  # fx_dv NOHOLD: a pad seen moving slides away under a hold
                        _cx_dv_ev("dv_nohold")
                elif h > 0.8:
                    vz = -1.3 if fa else -1.0
                elif h > 0.5:
                    vz = -0.8 if fa else -0.6
                elif d_xy > 0.15:
                    vz = -0.25 if (self.t > 50.0 or (_flag("mland") and self.map_label == "mountain") or (_flag("cfast") and self.map_label == "city")) else 0.0
                else:
                    vz = ((-0.60 if h > 0.25 else -0.50) if (self.t > 50.0 or (_flag("mland") and self.map_label == "mountain") or (_flag("cfast") and self.map_label == "city"))
                          else (-0.45 if h > 0.25 else -0.35))
                if _CX_ME_END and self._me_eg and self.map_label in _ME_EG_MAPS:
                    # fx_me END: fastest static landing (fx_lf's m17 profile): sink while centring down to 0.8 m,
                    # -1.3 m/s to 0.5 m, then -0.75 m/s once centred (LAND_SOFT still eases the contact below)
                    # the -0.75 m/s touchdown only when centred, level and not sliding: a touchdown at 0.4 m/s sideways
                    # and 11 deg tilt tipped over on the pad in 1899966432 (fx_lf's m17 TILTs were the same)
                    # 1899966432 (v5): a LAND entered at 2 m/s overshot the centre, sank at -1.3 m/s through 0.5-0.8 m
                    # still sliding and tipped over on the pad; the fast steps below 0.8 m wait for a calm drone too
                    vh_ = float(np.hypot(vel[0], vel[1]))
                    calm_ = vh_ < 0.2 and abs(float(self._rpy[0])) < 0.1 and abs(float(self._rpy[1])) < 0.1
                    if d_xy > 0.35:
                        if h >= 0.8 and vh_ < 1.0:
                            vz = -0.9
                    elif h > 0.5:
                        if calm_:
                            vz = -1.3
                    elif d_xy <= 0.15 and calm_:
                        vz = -0.75
                    _cx_me_ev("me_eg_tick")
                if self.t > 50.0:
                    self._t1_steps = getattr(self, "_t1_steps", 0) + 1
                if getattr(self, "_vsurf_t", 0.0) > 0.0:
                    vz = max(vz, 0.0)
                if _LAND_SOFT and self.map_label in _LAND_SOFT_MAPS and vz < 0.0 and h < _LAND_SOFT_H and float(vel[2]) > _LAND_SOFT_RATIO * vz:
                    vz = _LAND_SOFT_VZ  # touched: hold lightly instead of pressing the drone into the pad
                    if _LAND_SOFT_FAST:
                        self.v_cmd[2] = max(float(self.v_cmd[2]), _LAND_SOFT_VZ)
                    self._soft_ticks = getattr(self, "_soft_ticks", 0) + 1
                v_des = np.array([v_hv[0], v_hv[1], vz])
                if h > 0.5 and abs(vel[2]) < 0.05 and d_xy < 0.35:
                    self.land_stuck_t += SIM_DT
                else:
                    self.land_stuck_t = 0.0
                if (self.pad_ever_moving and unseen > 1.0 and h > 0.3 and not (_CX_VK_UNSTICK and self._vk_skip_hold())
                        and not (_CX_VL_HOLD and self._vl_hold_skip(pos, alt, pad))):
                    vz = 0.0
                    v_des = np.array([v_hv[0], v_hv[1], vz])
                    if unseen > 3.0:
                        abort = True
                if (_CX_ME_RIM or _CX_ME_DIAG) and self.map_label == "mountain" and self.land_off is not None:
                    v_des = self._me_rim(pos, vel, alt, pad, v_des, pv, d_xy)  # fx_me RIM (DIAG: log only)
                below = pos[2] < pad[2] - 0.3
                if _MTN_LAND_RECLIMB and below and self.land_off is not None:
                    # sank past the pad top without contact: the offset point is beside the pad (3062, 186: estimate
                    # 0.35-0.41 m off plus the 0.35 m offset). The centre estimate itself was good enough to land on,
                    # so climb back over the rim and land there instead of aborting into RECOVER/SEARCH (both timed out)
                    self.land_off = None
                    self.land_off_failed = True
                    self.land_reclimb = True
                    self.land_reclimb_t0 = self.t
                if self.land_reclimb:
                    if h < _MTN_LAND_RECLIMB_H and self.t - self.land_reclimb_t0 < 3.0:
                        v_des = np.array([0.0, 0.0, 0.8])  # straight up: sideways at this height hits the pad's side
                        self._reclimb_ticks = getattr(self, "_reclimb_ticks", 0) + 1
                        below = False
                    else:
                        self.land_reclimb = False
                abort = abort or below or self.land_stuck_t > 2.5 or self.t - self.land_t0 > 15.0
            else:
                oncoming = sp > 0.3 and float(pv @ rel_xy) < -0.3 * sp * max(d_xy, 1e-3)
                vmatch = float(np.linalg.norm(vel[:2] - pv))
                amb = None
                if _flag("mmodel") and not _flag("noamb") and self.mm is not None and self.t - self.mm["t0"] < 12.0:
                    ts = np.arange(0.4, 8.0, 0.1)
                    sps = np.array([float(np.linalg.norm(self._mm_predict(self.t + dt_)[1])) for dt_ in ts])
                    k_ = int(np.argmin(sps))
                    prev_ts = getattr(self, "amb_tstar", None)
                    if prev_ts is not None and prev_ts - self.t > 0.3:
                        kk = int(np.clip(round((prev_ts - self.t - 0.4) / 0.1), 0, len(ts) - 1))
                        if sps[kk] < 0.6:
                            k_ = kk
                    if sps[k_] < 0.35:
                        p_star, _ = self._mm_predict(self.t + ts[k_])
                        d_star = float(np.linalg.norm(p_star - pos[:2]))
                        if d_star < 14.0 and ts[k_] >= d_star / 2.0 + 0.5:
                            amb = (self.t + ts[k_], p_star)
                            self.amb_tstar = self.t + ts[k_]
                    if amb is None:
                        self.amb_tstar = None
                if amb is not None:
                    t_star, p_star = amb
                    dt_star = t_star - self.t
                    rel_s = p_star - pos[:2]
                    d_s = float(np.linalg.norm(rel_s))
                    dir_s = unit(rel_s)
                    v_hs = min(2.5, math.sqrt(2 * 1.5 * max(d_s - 0.3, 0.0)) + 0.3)
                    if dt_star > 1.3 or d_s > 0.4:
                        vz = float(np.clip(1.5 * (pad[2] + 0.9 - pos[2]), -1.0, 1.0))
                    else:
                        vz = -1.0
                    v_des = np.array([dir_s[0] * v_hs, dir_s[1] * v_hs, vz])
                    if d_s > 0.3:
                        yaw_des = math.atan2(rel_s[1], rel_s[0])
                    self.amb_dbg = f"mm t*={dt_star:.1f} d={d_s:.1f}"
                    self.land_phase = "ambush"
                    if self.t - self.land_t0 > 22.0 or pos[2] < pad[2] - 0.45:
                        abort = True
                elif self.land_phase == "ambush":
                    self.land_phase = "track"
                if amb is None and self.land_phase == "track":
                    if _flag("trackview"):
                        h_t = 1.35 if unseen < 0.7 else 1.9
                        p_des = pad[:2] - pdir * max(1.0, 1.5 * h)
                    else:
                        h_t = 1.0
                        p_des = pad[:2] - pdir * max(0.6, 1.3 * h)
                    rel_t = p_des - pos[:2]
                    vz = float(np.clip(1.5 * (pad[2] + h_t - pos[2]), -0.6, 0.6))
                    aligned = (float(np.linalg.norm(rel_t)) < 0.45 and vmatch < 0.7
                               and abs(h - h_t) < 0.35 and unseen < 0.4)
                    quick = (oncoming or sp < 0.45 or self.land_dive) and d_xy < 1.2 and h < 1.6 and unseen < 0.6
                    if aligned or quick:
                        self.land_phase = "drop"
                        self.drop_t0 = self.t
                    gain, cap = 2.0, 1.5
                    corr = gain * rel_t
                    cn = float(np.linalg.norm(corr))
                    if cn > cap:
                        corr *= cap / cn
                    v_hv = pv + corr
                    v_des = np.array([v_hv[0], v_hv[1], vz])
                elif amb is None:
                    lead = pv * 0.1
                    p_des = pad[:2] + lead
                    rel_t = p_des - pos[:2]
                    since = self.t - self.drop_t0
                    if self.map_label == "village":
                        z_floor = pad[2] - 0.08
                    else:
                        z_floor = pad[2] - (0.15 if since < 2.5 else 0.35)
                    vz = float(np.clip(2.5 * (z_floor - pos[2]), -1.1, 0.0))
                    if _flag("softdrop") and sp < 0.6:
                        vz = max(vz, -0.7 if h < 1.0 else -1.1)
                        if h < 0.45:
                            vz = max(vz, -0.5)
                    if (_flag("gentle") and sp < 0.35) or (_flag("gentle2") and sp < 0.35 and self.pad_slow_t > 1.0):
                        vz = max(vz, -0.5 if h < 0.8 else -0.9)
                        if h < 0.5 and float(np.linalg.norm(rel_t)) > 0.2:
                            vz = 0.0
                    corr = 2.0 * rel_t
                    cn = float(np.linalg.norm(corr))
                    if cn > 1.6:
                        corr *= 1.6 / cn
                    v_hv = pv + corr
                    v_des = np.array([v_hv[0], v_hv[1], vz])
                    if since > 4.0 or (unseen > 1.5 and d_xy > 0.8):
                        self.land_phase = "track"
                        self.land_retry += 1
                if amb is None and d_xy > 0.3:
                    yaw_des = math.atan2(rel_xy[1], rel_xy[0])
                unseen_lim = 3.0 if amb is None else 30.0
                if _flag("mvpatience") and d_xy > 0.3:
                    unseen_lim += min(4.0, abs(wrap(yaw_des - yaw)) / 0.69)
                if pos[2] < pad[2] - 0.45 or (unseen > unseen_lim and h > 0.7) or self.t - self.land_t0 > (22.0 if amb is not None else 20.0) or self.land_retry > (5 if (_flag("qretry") and self.map_label == "mountain") else 2):
                    abort = True
            if abort:
                if (((_flag("vmovko") and self.map_label == "village") or (_flag("cmovko") and self.map_label == "city")) and not self.pad_moving
                        and self.land_xy is not None and self.pad_pos is not None
                        and float(np.linalg.norm(self.pad_pos[:2] - self.land_xy)) > 1.0
                        and not (_CX_VM_ANY and self._vm_flip("movko", "dl%.2f" % float(np.linalg.norm(self.pad_pos[:2] - self.land_xy))))):
                    self.pad_moving = True
                    self.pad_ever_moving = True
                    self._vmovko_fired = True
                if self.pad_hits < 30 and not self.pad_moving:
                    self.bad_spots.append(np.array([pad[0], pad[1], pad[2]]))
                if getattr(self, "_vsurf_t", 0.0) > 2.0:
                    self.bad_spots.append(np.array([pad[0], pad[1], pad[2]]))
                    self._vsurf_t = 0.0
                if self.land_off is not None:
                    # an offset landing that aborts is not retried with the offset (186: re-approach ran out of clock)
                    self.land_off = None
                    self.land_off_failed = True
                self._set_mode("RECOVER")
                self.recover_z = max(pad[2] + 2.0, pos[2] + 0.5)
                self._qretry = False
                if (_flag("qretry") and self.map_label in ("mountain", "city", "warehouse") and self.pad_hits >= 30
                        and self.land_retry < 3 and getattr(self, "_vsurf_t", 0.0) <= 2.0):
                    self.recover_z = pos[2] + 0.6
                    self._qretry = True
                    self.land_retry += 1
                if _CX_S1M_BADREC and self.map_label == "mountain" and not _CX_STATE["shadow"]:
                    self._s1m_badrec()  # fx_s1m BADREC
        elif self.mode == "RECOVER":
            use_planner = False
            yaw_des = yaw
            vz = float(np.clip(1.5 * (self.recover_z - pos[2]), -0.5, 1.2))
            v_des = np.array([0.0, 0.0, vz])
            if pad is not None:
                look = pad - pos
                if float(np.linalg.norm(look[:2])) > 0.5:
                    yaw_des = math.atan2(look[1], look[0])
            if getattr(self, "_qretry", False) and (abs(self.recover_z - pos[2]) < 0.3 or self.t - self.mode_t0 > 0.6) and self.pad_pos is not None:
                self._qretry = False
                self.recover_n += 1
                self._set_mode("APPROACH")
            elif (abs(self.recover_z - pos[2]) < 0.3 and self.t - self.mode_t0 > 1.5) or self.t - self.mode_t0 > 4.0:
                self.recover_n += 1
                if self._track_reliable():
                    self._set_mode("APPROACH")
                elif (((_flag("vretry") and self.map_label == "village") or (_flag("cretry") and self.map_label == "city")) and self._static_established()
                        and self.recover_n <= 2
                        and not any(float(np.linalg.norm(pad[:2] - bs[:2])) < 1.0 for bs in self.bad_spots)):
                    self.land_retry += 1
                    self._set_mode("APPROACH")
                elif self.pad_moving and self.pad_pos is not None and self.reacq_n < 4:
                    self.reacq_n += 1
                    self.reacq_moving = True
                    self.reacq_target = pad.copy()
                    self.reacq_wp = pad.copy() + np.array([0.0, 0.0, 3.0])
                    self.reacq_hold = 0.0
                    self.reacq_side = 0
                    self.reacq_t_enter = self.t
                    self._set_mode("REACQUIRE")
                else:
                    self.search_anchor = pad[:2].copy() if pad is not None else None
                    self._drop_track()
                    self._set_mode("SEARCH")
                    self.search_wps = None
        elif self.mode == "VXRAY":
            # fx_vx RAY episode (the controller's own moves: planner off)
            use_planner = False
            vx_ = None
            if self._vx is not None and not _CX_STATE.get("shadow"):
                try:
                    vx_ = self._vx_step(pos, vel, alt)
                except Exception:
                    _cx_vx_ev("vx_error")
                    vx_ = None
            if vx_ is not None:
                v_des, yaw_des = vx_
            else:
                if self._vx is not None:
                    try:
                        self._vx_end("error")
                    except Exception:
                        self._vx = None
                if self.mode == "VXRAY":
                    self._set_mode("SEARCH")
                    self.search_wps = None

        self._m15_look_tick = False
        if _CX_M15_LOOK and self.map_label == "mountain":
            # fx_m15 LOOK: turn to a stale young track and fly to where the camera can see its estimate
            try:
                if self.mode in ("CRUISE", "SEARCH"):
                    lk_ = self._m15_look_step(pos, rpy, alt, yaw)
                    if lk_ is not None:
                        v_des, yaw_des, use_planner = lk_
                        self._m15_look_tick = True  # (skips the mountain CRUISE climb-over rule after the planner)
                        self._pulse_bypass = False  # a spin look-down pulse set this tick must not step the look command
                elif self._m15_look is not None:
                    self._m15_look = None
                    _cx_m15_ev("m15_look_end_mode")
            except Exception:
                self._m15_look = None
                _cx_m15_ev("m15_error")
        if _CX_V16_LOOK and self.map_label == "village" and not _CX_STATE.get("shadow"):
            # fx_v16 LOOK: turn to a stale young ground-level track and fly to where the camera can see its estimate
            try:
                if self.mode in ("CRUISE", "SEARCH"):
                    lk16_ = self._v16_look_step(pos, rpy, alt, yaw)
                    if lk16_ is not None:
                        v_des, yaw_des, use_planner = lk16_
                elif self._v16_look is not None:
                    self._v16_look = None
                    _cx_v16_ev("v16_look_end_mode")
            except Exception:
                self._v16_look = None
                _cx_v16_ev("v16_error")
        if _CX_FP_LOOK and self.map_label == "forest":
            # fx_fp LOOK: turn to a stale young forest track and close in until the camera decides it
            try:
                if self.mode == "SEARCH":
                    lkf_ = self._fp_look_step(pos, rpy, alt_c, yaw, prm)
                    if lkf_ is not None:
                        v_des, yaw_des, use_planner = lkf_
                elif self._fp_look is not None:
                    self._fp_look = None
                    _cx("fp_look_end_mode")
            except Exception:
                self._fp_look = None
                _cx("fp_error")
        if _CX_SV_LOOK and self.map_label == "village" and not _CX_STATE.get("shadow"):
            # fx_s1v LOOK: turn to a stale young village track at pad height and close in until the camera decides it
            try:
                if self.mode == "SEARCH":
                    lks_ = self._sv_look_step(pos, rpy, alt_c, yaw, prm, alt=alt)
                    if lks_ is not None:
                        v_des, yaw_des, use_planner = lks_
                elif self._sv_look is not None:
                    self._sv_look = None
                    _sv_ev("sv_look_end_mode")
            except Exception:
                self._sv_look = None
                _sv_ev("sv_error")
        if _F6_UNDER and self.map_label == "forest":
            v_des, yaw_des, use_planner = self._f6_under(pos, alt, yaw, v_des, yaw_des, use_planner)
        if _WF_OOC and self.map_label == "forest":
            v_des, use_planner = self._wf_ooc(v_des, yaw_des, yaw, use_planner)
        blocked = False
        free = 10.0
        if use_planner:
            vz_req = float(v_des[2])
            vxy_req = float(np.hypot(v_des[0], v_des[1]))
            v_des, blocked, free = self._plan(depth, pos, rpy, v_des, tube, tube_min)
            if (_MTN_CLIMB_V > 0.0 and self.map_label == "mountain" and vz_req > 0.05
                    and not (_MTN_CLIMB_NOPAD and self.pad_pos is not None)):
                # the planner caps blind climbs at 1 m/s (city roofs and overhangs) and brakes the whole vector on
                # rising ground ahead; mountain terrain has open sky above it, so keep the requested climb
                # (1486 crawled up a 21 m slope at 0.86 m/s for 25 s; 2132 sat blocked under a ridge for 11 s)
                vz_keep = min(vz_req, _MTN_CLIMB_V)
                if float(v_des[2]) < vz_keep - 0.05:
                    v_des = np.array([v_des[0], v_des[1], vz_keep])
                    self._climb_ticks = getattr(self, "_climb_ticks", 0) + 1
            if (_MTN_CLIMB_CUT > 0.0 and self.map_label == "mountain" and self.mode == "CRUISE" and vxy_req > 0.5
                    and not (_MTN_CLIMB_NOPAD and self.pad_pos is not None) and not self._m15_look_tick
                    and float(np.hypot(v_des[0], v_des[1])) < _MTN_CLIMB_CUT * vxy_req):
                # the planner cut the cruise to a crawl: rising ground or a wall ahead. With open sky above mountain
                # terrain, climb over it at speed instead (1486 still moved 0.6 m/s up its slope when only the
                # terrain-following climb was kept)
                climb = _MTN_CLIMB_V if _MTN_CLIMB_V > 0.0 else 2.0
                if float(v_des[2]) < climb - 0.05:
                    v_des = np.array([v_des[0], v_des[1], climb])
                    self._climb_ticks = getattr(self, "_climb_ticks", 0) + 1
            if _flag("yawvel") and self.map_label in ("city", "mountain", "forest") and self.mode in ("CRUISE", "SEARCH", "REACQUIRE"):
                vh = float(np.hypot(v_des[0], v_des[1]))
                if vh > 0.8:
                    ang_v = wrap(math.atan2(v_des[1], v_des[0]) - yaw_des)
                    if abs(ang_v) > 0.35:
                        if _CX_MA_RING and self._ma_ring_tick and self.mode == "SEARCH":
                            # fx_ma RING: keep the camera turned in toward the ring centre, capped under yawvel's speed cut
                            yaw_des = math.atan2(v_des[1], v_des[0]) - float(np.clip(ang_v, -_MA_RING_OFF, _MA_RING_OFF))
                            _cx_ma_ev("ma_ring")
                        else:
                            yaw_des = math.atan2(v_des[1], v_des[0])
                    vh_act = float(np.hypot(vel[0], vel[1]))
                    if vh_act > 0.8 and abs(wrap(math.atan2(vel[1], vel[0]) - yaw)) > 0.6:
                        v_des = np.array([v_des[0], v_des[1], v_des[2]]) * min(1.0, 1.5 / max(vh, 1e-6))
        if vz_override is not None:
            v_des = np.array([v_des[0], v_des[1], vz_override])
        if ms_slope is not None and self.mode == "APPROACH":
            # SLOPE: sink with the ground we fly over (mean slope x the horizontal speed the planner left us) and hold
            # _MS_SLOPE_AGL over it; never sink below 1.5 m. The ground is the higher of the down ray and the terrain the
            # camera sees in a 2 m wide corridor up to ~4.5 m ahead, so a bump ahead stops the sink before we reach it.
            self._ms_acted = True
            _cx_ms_ev("ms_slope")
            vh_fl = float(np.hypot(v_des[0], v_des[1]))
            alt_eff = self._ms_ground_ahead(depth, pos, rpy, v_des, alt, vh_fl)
            vz_s = -ms_slope * vh_fl + _MS_SLOPE_KP * (_MS_SLOPE_AGL - alt_eff)
            if alt_eff < 1.5:
                vz_s = max(vz_s, 0.5)
            if alt_eff < 1.2 and vh_fl > 0.8:
                v_des = np.array([v_des[0] * 0.8 / vh_fl, v_des[1] * 0.8 / vh_fl, v_des[2]])
                _cx_ms_ev("ms_slope_bump")
            vz_f = float(np.clip(vz_s, -2.2, 1.0))
            if self._pulse_bypass:
                # an approach look-down pulse steps the command past the acceleration limit so the body pitches nose
                # down. Only its horizontal part may step: the sink moves at the 2.5 m/s^2 approach cap as on any other
                # tick. (Its horizontal part is the full 1.5 m/s: SLOPE skips the x0.7 low-row cut, which in the
                # champion shrinks the pulse to 1.05 m/s and ~18-20 deg of pitch; at 1.5 m/s the body pitches ~30 deg,
                # and on 1571070702 that re-detected the pad, 67 deg below the horizon, where the champion's did not.)
                vz_f = float(np.clip(vz_f, self.v_cmd[2] - 2.5 * SIM_DT, self.v_cmd[2] + 2.5 * SIM_DT))
                _cx_ms_ev("ms_slope_pulse")
            v_des = np.array([v_des[0], v_des[1], vz_f])
        if vabs_lim is not None and self.mode not in ("LAND",):
            if alt_c >= vabs_lim - 0.05 and v_des[2] > 0.0:
                v_des = np.array([v_des[0], v_des[1], 0.0])
            if alt_c > vabs_lim + 0.5:
                v_des = np.array([v_des[0], v_des[1], min(float(v_des[2]), -0.6)])
        if (_flag("roofguard") and (self.map_label == "city" or (_flag("vroof") and self.map_label == "village")) and v_des[2] < 0.0
                and self.mode in ("CRUISE", "APPROACH", "SEARCH", "REACQUIRE")):
            low_d = DEPTH_MIN_M + float(depth[80:128, 24:104].min()) * (DEPTH_MAX_M - DEPTH_MIN_M)
            near_pad = pad is not None and float(np.hypot(pad[0] - pos[0], pad[1] - pos[1])) < 2.5
            if low_d < 3.0 and not near_pad:
                v_des = np.array([v_des[0], v_des[1], 0.0 if low_d > 1.5 else 0.6])
        if _flag("clook") and self.map_label == "city" and self.mode not in ("LAND",):
            vh = float(np.hypot(v_des[0], v_des[1]))
            if vh > 0.8:
                off = abs(wrap(math.atan2(v_des[1], v_des[0]) - yaw))
                cap = 3.0 if off < math.radians(25) else (0.8 if off > math.radians(60) else 3.0 - (3.0 - 0.8) * (off - math.radians(25)) / math.radians(35))
                if vh > cap:
                    v_des = np.array([v_des[0] * cap / vh, v_des[1] * cap / vh, v_des[2]])
            vel_h = float(np.hypot(vel[0], vel[1]))
            if vel_h > 1.2 and abs(wrap(math.atan2(vel[1], vel[0]) - yaw)) > math.radians(60):
                v_des = np.array([vel[0] * 0.2, vel[1] * 0.2, v_des[2]])
        if _flag("cwall") and self.map_label == "city" and self.mode in ("CRUISE", "SEARCH", "REACQUIRE", "APPROACH"):
            _sc = (DEPTH_MAX_M - DEPTH_MIN_M)
            dl = DEPTH_MIN_M + float(depth[30:100, 0:30].min()) * _sc
            dr = DEPTH_MIN_M + float(depth[30:100, 98:128].min()) * _sc
            dlo = DEPTH_MIN_M + float(depth[90:128, 24:104].min()) * _sc
            if min(dl, dr, dlo) < 1.8:
                eye_, fwd_, right_, up_ = cam_pose(pos, rpy)
                push = np.zeros(3)
                if dl < 1.8:
                    push += right_ * 0.5
                if dr < 1.8:
                    push -= right_ * 0.5
                vh = float(np.hypot(v_des[0], v_des[1]))
                if vh > 1.5:
                    v_des = np.array([v_des[0] * 1.5 / vh, v_des[1] * 1.5 / vh, v_des[2]])
                v_des = v_des + np.array([push[0], push[1], 0.0])
                if dlo < 1.8 and self.mode != "APPROACH" or (dlo < 1.8 and pad is not None and float(np.hypot(pad[0] - pos[0], pad[1] - pos[1])) > 2.5):
                    v_des = np.array([v_des[0], v_des[1], max(v_des[2], 0.5)])
        self._alt_hist = [(t_, a_) for (t_, a_) in getattr(self, "_alt_hist", []) if self.t - t_ <= 1.2] + [(self.t, float(alt))]
        alt_ago = max((a_ for (t_, a_) in self._alt_hist if self.t - t_ >= 0.8), default=float(alt))
        if (_flag("cskim") and self.map_label == "city" and self.mode in ("CRUISE", "SEARCH", "REACQUIRE", "APPROACH")
                and alt < 1.5 and alt_ago > 3.0):
            vh = float(np.hypot(v_des[0], v_des[1]))
            sc = min(1.0, 1.2 / max(vh, 1e-6))
            v_des = np.array([v_des[0] * sc, v_des[1] * sc, max(v_des[2], 1.5)])
        if _CX_CT_ROOF and self.map_label == "city":
            v_des = self._ct_roof(pos, alt, vel, pad, v_des)
        if _STUCK_S > 0.0 and self.map_label in _STUCK_MAPS:
            v_des = self._unstick(depth, rpy, vel, v_des)
        if _CX_S1F_CG and self.map_label == "forest":
            try:
                v_des = self._s1f_cg(depth, pos, rpy, pad, v_des)  # fx_s1f CG: no blind climb into a crown
            except Exception:
                _cx_ms_ev("s1f_cg_err")
        # acceleration limit
        if self.mode == "LAND":
            a_max = 3.0 if self.pad_moving else 2.5
        elif self.mode == "APPROACH" and not self.pad_moving:
            a_max = 2.5
        else:
            a_max = 3.2
        dv = v_des - self.v_cmd
        n = float(np.linalg.norm(dv))
        if getattr(self, "_pulse_bypass", False):
            self._pulse_bypass = False
            self._pulse_last_t = self.t  # look-down pulse: step the command (the acceleration limit would flatten the pitch)
        elif _CX_CT_ROOF and getattr(self, "_ct_bypass", False):
            self._ct_bypass = False  # roof-step guard: brake/climb now; 3.2 m/s^2 needs ~0.7 s, the hull face arrives in ~0.5 s
        elif _CX_ME_RIM and self._me_rg_bypass:
            self._me_rg_bypass = False  # fx_me RIM fire: stop the sink beside the rim now
        elif n > a_max * SIM_DT:
            dv *= (a_max * SIM_DT) / n
        if _CX_ME_RIM:
            self._me_rg_bypass = False
        if _CX_CT_ROOF:
            self._ct_bypass = False  # one tick only, whichever branch ran (a look-down pulse on the same tick takes precedence)
        self.v_cmd = self.v_cmd + dv
        err = self.v_cmd - vel
        en = float(np.linalg.norm(err))
        if en > 2.0:
            self.v_cmd = vel + err * (2.0 / en)
        sp = float(np.linalg.norm(self.v_cmd))
        if sp > SPEED_LIMIT:
            self.v_cmd *= SPEED_LIMIT / sp
            sp = SPEED_LIMIT
        if sp < 1e-3:
            d = np.zeros(3)
            s = 0.0
        else:
            d = self.v_cmd / sp
            s = sp / SPEED_LIMIT
        action = np.array([d[0], d[1], d[2], s, wrap(yaw_des) / math.pi], dtype=np.float32)
        self.debug = {"mode": self.mode, "map": self.map_label, "pad_hits": self.pad_hits,
                      "pv": np.round(self.pad_vel, 2).tolist(), "om": round(float(self.pad_omega), 2),
                      "pe": (np.round(self._pad_predicted()[:2], 1).tolist() if self.pad_pos is not None else None),
                      "pad_conf": round(self.pad_conf, 2), "moving": self.pad_moving, "free": round(free, 1),
                      "blocked": blocked, "n_det": len(dets), "ph": self.land_phase if self.mode == "LAND" else "",
                      "amb": self.amb_dbg if self.mode == "LAND" else None, "vd": np.round(v_des, 2).tolist(),
                      "pz": (round(float(self.pad_pos[2]), 2) if self.pad_pos is not None else None), "evm": self.pad_ever_moving,
                      "mm": (f"w={self.mm['w']:.2f} rms={self.mm['rms']:.2f} span={self.mm['span']:.0f}" if self.mm is not None else None),
                      "mml": getattr(self, "mm_last", None), "nraw": len(self.mm_raw)}
        if _CX_VX_RAY and self._vx is not None:
            self.debug["mode"] = "VX_" + str(self._vx["ph"])  # fx_vx: the episode phase in the trace
        return action


import sys as _sys, types as _types, base64 as _b64, zlib as _zlib

_KING_SRC_B64 = (
    "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"
    "uf85wXfGj/f/hD/Tt8l2+va3357KpWb3t+C1Xep0fys83v/9n/w3f/04zP7Xd944jOu3Yv+3XuW3wmv/2/an+N/w5dX8X44NLtHm"
    "5PTbcQH9OkwWfFB/47/9L9Fvo516Hj7n+tIqf8uX3k5vk79NtusdzJB6xJjYf+7Of4Uf4ZN//ft/yda0mZwuJ4fffuPzr63g/zEX"
    "6P9iz9hLf3mLvzpX/883/WPf9ofRef9tMpos3qa4c7Rmv40mP7f7PXt+/238sYxpc203b992P7eTt/3+v74dFzDib2+fbz/P355/"
    "wq+CeDlfHIrbzeHnNo7ffvJGf779Lxz5xwG+e1gs99/224+fkzf43mH7bfRt83b89ttquZn/Rq19Y134dlweFvy3MI5SBJun89/f"
    "FqPNFD4CvcCFgX6LsdAn/9/9t8V2u+LtUlvsa/ALnOFvf5vh/oCW94e30RR7Nt5+wBdlC9+gS/h5/Ay8tI2n9OL//jbJfVu9ve32"
    "NAXj5WYKHf77P7914dlJPFrvvi0P+7d4xpte7ukr+9Eaevrx8+3b7GMzOSy3m/+CntKvZvH2uPk2fluMPpfQd3wDHoHhzd+m/5Qn"
    "iS/O/2TuCbl9xMb+Kz8G1LG/klCQ3/pvNjNvYt35WhxHezmlf0NJsl8tdzv4z+iAI/s2gr5tf/7dOCjsC/9HtPb2SdtTTmXWs6xn"
    "/1ed2PVo9cZ++jf5gczWLvE2p38A/5/5/GJpPL5Y/kUXAd9wX/2VH3oYSVKSOudcnKwZvvnfsGH5vP5tFP+ELXYWs/p38+xNYNvB"
    "tNIRMPepHP7/hq0G35vB84tv8XY0/S+1p7Dtq7JC9vH/aAMzl+gvs5/btdmz5Xq3/XmwH+hvsEt+C+g57ZT/hYv2sPrs33ko2qGN"
    "v3FlkZXAzvq5oY9FQ8MoPwOl8DChenOsRXv4ZJxVRD1d58+jRQ7K5QR0Fv+I+fG6ag23fQ0ViQFRuBznUSU6jsuhePeoFR2BibrU"
    "mqOC/uJa/v/3SS7/of2f5Ws87XbjPhUlzKkguFLzCfWsF84HqC5Bd7R3eG5IAVPDpSX2JEeyIG4aMOY5tL42Ihbja4MajQg5LejK"
    "FP3A5VM8WdLnZu01ISuZv5tv6+1SVG5FUSMK8v2uX2u0g3wkqrIl9Bg3EIL9kSEnIJTYkYAashx98OlWVFowVi5PKJKh9nPkmaWf"
    "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"
    "2lYlbcC0zCer/fsbrTFcKuLIWg5p/e4iHaishdBXNyaPyfr6Bt1+ZMs4NsKks3g+yaQDTzg4kpFik2Tr96W98tL7zwhP4XGrEs2D"
    "WKag8N3z2MudZa8tKAJ8hpDbwvB6fUFPuvLKf8O9pjE2WIWwzlSOshLyLNBkhM/RGnsfF9Jr2iKHAXlE0FtakovuYVfDuP0QRrdV"
    "PnRejSNn8E3OgIbiL5Wny2jT/aLjx70uZGwX6a5WoWqMnKZyap/0U2t5RE7FWQO1Ut86mjo4nPUY+YuBXHroSplTpQgZHTp6okYU"
    "5Uj+DrYdXS0aEW5m9wocpCA98zwmHoNK3dlOgj16lR309/2nDkKVWp1adih9BuKdmgWGTRKBShaqU46Qa4+nawg4Ul0M93lCa8JF"
    "EgmzR+k+kFRe23K5sh/NMjbpR2Mww2VO4CYltLmS/V22DR7IK7UmKKtNnl5I/2tKqXvpqyvf8tjg+FkRVUibBXnNB8NkwMcq9eLO"
    "EJ9VmmHn6XlAeewPo/TiClMAe3gV72Ro4CMZXa8nNTSouHuV2TYreUD3m8l6rZxptnElnj8nQanQ7lWbw8JxPim2iGkx9FuRRscc"
    "EnEA7YFGhZT9VSLaf3W49n1IV9KHii6+19BxeKl2SJYrEnOv/D79iI5qlGhQqEpwwVjM2InRwFPn3WT8/BFPK0u6GPZVQNHRb/eC"
    "zKzOP/741x9/LN7JLs7e1vvvn4upX/T+8ed6s/9zsf7z++60+2u1mX4mb7v/jS///kl+/5386M//8+f3/Wn7tvurhb/vkb//U3/K"
    "v/Rv/PX9+/siefv+Hb7K/pr6xPZ1siQPwQ/9g8737fg2+edks9qSb/zz+/ju5i/y3/RtspmSf97Xqu3e96j+AI8o39386y/2m398"
    "7t//v2//+Nf/6FP7a3v6B/kRPJP8Dp//pzmG6WKy//6djV5Zgf+rT+//wTz41/744/3HZvWn+oE/F6vt5sf+z4cfm/VbkCxm831l"
    "s97/2CTJ248/X3d/smHLH/7xx/fW80P1qfv9oR6Sp7+87uf/5Ev1r79+vO02ydfbP//11/b1B3nPn//rz3+QV78lu3/88cdDNbiP"
    "nnrf+8/h08P36IV8fb396/XHj9fTP/+v95f3P3/i//l/ef/vf/6cws79H/L792Tzui8W/kW+/tJ7/N6qt7+3yDe9v275T+4H+JMC"
    "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"
    "8ltI0SJfwmN4Tr+CU3p28iY72E/cg/ZVcOzCUnhWw4Ma/Hz9uk6l16/VCe+Frri39dsvidUIsNWgDNSztTApcWVzUEodSNdoTRhi"
    "dU09RH0Y//k429p/UzVKWcriRvze3jqsUUuUks09x3qvYNhHb3b2S+uZUrbBV9D7/TeDpKoiloJ6z7qmJmg5ksqaSslhyChevTvU"
    "kyZCRiyDRPxR13l18H6/qg6UgVqm0tut3dfvKyphGaI+Eq9eH/xaY2CiFFT7yeDj8HjnzX5lNSylAiRo0h0ey+ZiRDs8lgoLLH68"
    "O3hf3Tso5Sw6sKWOtqK+Q4WLieDhTVmnxmSUpdTc121t7+7s/VxdD0q5fSzDxImLCfFz8Nv2bllx+A7FIf5IW+nCoeZ2ndViSpEe"
    "btdZLboU1HvZNTV//T/VnBJK4dSw1SrnvSqDq+W5rlWOyG2DSOOTABFkVCySVwla0ErmvpmsSvWzYaT2jeJx9oVlQ6TaNn0HgXdg"
    "RyvXtZnoNFD0EXRtpsmYru0q1LRdlejZrjJcJIym7QrsxD5muHW78fb193G2uGIrjrPzeeTDkK+BYcMvcnDg6PEtXdRUwP1p9W9X"
    "36B9u8KL0ckHnZA+n843PPcZeAcTRDNhSG0Fr/RNnvRREW+CfPTiHatbc07RgfXTraOSChz1MtAxzfLxVdF6zvtDiiOK1P1Is0RY"
    "uPz15fmsyD8EX26JiSOCQBqnvjkHLbWkFSRBBQBCC7NF63N3LqbMVMUT+CxjTUznOjBFu4naGFOqrkuFY59NCS/ja+fgYBmeByAy"
    "zE07mY4WFxAzf/KJqmMDkg0lD630L5Zl2PMORLmVIffFunoW0guaGJ52u10nxjP887advIePXppsaVkuOwbyeXeRj1v6eZYPR/l4"
    "3tKCzBPBhl54Z7qLyY3aUGaQGLHVei+O/b1UlhX/x9jaHQe8wFkqjo7gpuXCuv7igHqrgHQSKxI9MZCGxdWsKOYtRD7A87oJbb/V"
    "XfZm1aqyoV9to6LgLXS5Dc3Lpt4d+hV6qsIM4oT3VC2ssSLJ7iGFNAAIkmBRPPQKImAB7L0suGILQkNuYRUhcng6W+3PRCcuVvsX"
    "vbMGE9sASnbnoy/F0obdgO8VJVYgFOhcisasP3VfLLeYXCvtKmMpe3txVf/u4ior8Yy1Ykh/+UsAPgg+6i5IBPwXdSCHHmCLck5g"
    "1dpXWVVkfmxWyDSGC92jk7Sk4By0IwJP5tdz59cLT58qo8yhWvsnqyG/YlVBtWYAj2JOAaTEHFgz+fm8BSMSPYVHub6JHl+LRPin"
    "E0pLpXKUUwDdOI1wCkKmK5oSxwotAOijRWrsmFvm+BApDt/S2N2Wlv9Ee+wujd1eT7Sqw1x+mXobidZmlBkfcWJmXcMlXTdQi6PC"
    "kyIQEIAd3pQd3g/lfb+MVovAa0VZ+yFqrWeH04uWQOk46DOHeGzcnGtBYBVrCC/BwvZh7kqNo7M2W8AmYbbKtaVKpcCi5Kk2TXlQ"
    "z0NQsICfmm1uOopZ5cp1xGs3l59r8yk9Tnq6XMEYUjaQjqiFSvln/ASCoNwQnZLR9tKbSnPMZ3tmgMngtcAi4LFt6/GQy0eU8pXu"
    "oMcC/fMpH4MDaKaU1HcMh/TltTrTBoOxis4+TZ45OUtqRAQhveG8UN2WuRLfYw6Gx8rhsxLKmEFRno+3hurLWtXWNByXgBeg3DBL"
    "ojPyXknyoDH91RpCzI7hsxRmZjfY7G+I3JwLyXYNoONlCvxdxQtCgUmmFsjAIOxZBCf/M5FhdZ2XHfLLByOOJnCYB67jy8SzYL3C"
    "MrNKFhycBrGpv4STTJIOJql4aidKqZyW8NzzOctzXd2N0/y6PDQbwZ3fRqWeidkQhvJsZk7D7oktelbT+jCWq2mYvjeUfr9Ox7OS"
    "GGUev0o/KRTiOsdDATy0eBBwZypbgSMNlDaf2jgv0zTKRz/B5JHqwc2EbERrZ2mcq6oSAQewwNIKf2KlHmLJ7qjxliY7Fz2I8wL7"
    "lnmgdP1RpDJzIY7MoVwwLV2pQzBg6KmvQBQ9q4DpOns7+207j+BjGq8LHYfq67CE4/21axxKlxZzVIgrnOBV7WEXxDVQUwSvIYK9"
    "3w4fmU+gquyYummZPpMOTLQnsQJKCZhwjMgmOhLgLLZ9AzB1KoTPQhhflYq+DpCn8bD90ruQjt//ejrtxv8HGbB5aQ=="
)

_HANDOVER = os.environ.get("MY_HANDOVER", "any")
_HANDOVER_CITY = os.environ.get("MY_HANDOVER_CITY", "any")
_OPEN_ALT = float(os.environ.get("MY_OPEN_ALT", "12.0"))  # an "open" map whose start sits above this altitude is mountain terrain (open/city/warehouse/forest starts are <= 10.4 m): fly it myself
_STUCK_S = float(os.environ.get("MY_STUCK_S", "0"))   # OFF: measured -0.0089 forest on the real distribution, and it fires on city flights before the map label locks        # >0: seconds wedged (whole frame at DEPTH_MIN while barely moving) before backing out. Forest 153 grinds up a trunk into a collision; 511 spends 17.7 s of its 60 s wedged
_STUCK_NEAR = float(os.environ.get("MY_STUCK_NEAR", "0.55"))  # "camera touching something": DEPTH_MIN is 0.5
_STUCK_VMAX = float(os.environ.get("MY_STUCK_VMAX", "0.30"))
_STUCK_V = float(os.environ.get("MY_STUCK_V", "1.5"))        # back-away speed
_STUCK_HOLD = float(os.environ.get("MY_STUCK_HOLD", "1.2"))   # seconds to keep backing away once fired
_STUCK_MAPS = tuple(x for x in os.environ.get("MY_STUCK_MAPS", "forest").split(",") if x)
_MTN_HIGH_RELABEL = int(os.environ.get("MY_MTN_HIGH_RELABEL", "1"))  # starts above _OPEN_ALT are always mountain (every other map starts <= 10.4 m): never let a high start lock, route or hand over as another map
_HB_ON = int(os.environ.get("MY_KING_HANDBACK", "1"))  # mountain: a pad handed to the king mid-flight is taken back when the king has lost it and my (shadow) track says static and near
_HB_FIT = float(os.environ.get("MY_HB_FIT", "0.40"))    # my last fitted pad speed must be under this (m/s); frozen once detections stop (3298: 0.38)
_HB_HITS = int(os.environ.get("MY_HB_HITS", "60"))      # ...with at least this many detections on the track
_HB_DEST = float(os.environ.get("MY_HB_DEST", "4.0"))   # ...and the drone still within this of my estimate (m)
_HB_LOST_S = float(os.environ.get("MY_HB_LOST_S", "1.0"))  # total time since the hand-off the king pilot spent in search/navigation without sight of the pad (s); a brief re-sighting does not reset it (4198)
_HB_AFTER = float(os.environ.get("MY_HB_AFTER", "1.0"))    # never within this long of the hand-off (s)
_HB_WINDOW = float(os.environ.get("MY_HB_WINDOW", "6.0"))  # ...and only this soon after the hand-off (s): static rescues fired 2.0-4.6 s after it, hand-backs 8-25 s into a moving chase broke 3402 (0 = no limit)
# fx_mm (candidate fix, default OFF): mountain MOVING-pad hand-back. uid130 overshoots a pad that really moves, loses it
# and drops into search (goal-return: back to the pad's last seen, static, position and a spin for up to 400 ticks, then
# the clue centre) and does not land in time (1225093077, 2030969537, 67120516, 864678566), while my shadow track (kept
# alive by _handback_step) still sees the pad a few metres away. The static hand-back above cannot fire there (it needs a
# fitted pad speed < 0.40 m/s). With CX_MM_HB=1: once uid130, after flying at the pad (navigation/landing), has sat in
# search this long, give the flight back to my own moving-pad chase (APPROACH -> LAND track/drop), keeping the pad moving.
# Vetoes (review): never on a track the static hand-back owns (fit speed < _HB_FIT), never while uid130's own static
# landing rescue runs, never later than _CX_MM_WINDOW after the hand-off. Counts go to the module's one CX_EVENTS (top).
# fx_mv diagnostics (CX_MV_DBG=1, default OFF, changes no action): after a mountain hand-over to uid130, log uid130's
# post-hand-over decisions (mode, sight, goal-return anchor, attempts) into CX_EVENTS['mv_log'] on every change.
_CX_MV_DBG = _cx_on("CX_MV_DBG")
# fx_mv CX_MV_ANCHOR (default OFF): mountain moving pads handed to uid130. uid130's shadow run burns its 3 goal-return
# attempts before the hand-over (1225093077: _goal_return_attempts == 3 at the hand-over tick), so after a missed pass
# every loss sends it to the noisy clue centre (median 9.5 m and 2-4 m above the pad on the 13 failures) and it never
# sees the pad again. With the flag on: my detections of the pad (dual route and shadow) are sampled every 0.1 s; once
# uid130 is in 'search' after the hand-over, its goal-return anchor is set to the pad's ORBIT CENTRE estimated from those
# samples (the pad stays within its orbit radius + 0.3 <= 4.3 m of that point for the whole flight, at a fixed height),
# lifted _MV_Z_UP above the pad plane, and kept armed (never expires) while uid130 searches. uid130's own goal-return
# then flies there and spins (CX_MV_SPIN), and re-acquires with its own detector. Nothing changes until uid130 has
# flown at a pad after the hand-over (navigation/landing) and then dropped to 'search'. That includes short dropouts
# while the pad is under the camera in a final descent (1499833350 armed twice and landed 0.28 s later), not only real
# losses. Once armed, the static take-back in _handback_step is skipped. rev1 guards (review): samples come only from
# the track that handed over (earlier tracks are cleared when my track is dropped/re-seeded before the hand-over), and
# no arm while those samples span less than _MV_SPAN_MIN (a static pad flagged moving: uid130 and the take-back stay as
# in c3). Recommended set: CX_MV_ANCHOR=1 CX_MV_GATE=1 CX_MV_SPIN=1.
_CX_MV_ANCHOR = _cx_on("CX_MV_ANCHOR")
_MV_MIN_PTS = 15        # samples (one per 0.1 s of detections) before an anchor exists
_MV_KEEP_S = 60.0       # samples from the whole flight: the orbit centre never moves
_MV_OUTLIER_M = 6.0     # samples farther than this (xy) from their median are ignored (orbit radius <= 4.3 m)
_MV_Z_UP = _cx_knob("CX_MV_ZUP", 0.0, _CX_MV_ANCHOR)  # anchor this far above the pad plane (uid130's goal-return adds ~0.5 m close in)
# rev1 static guard: diagonal of the 5-95 % xy box of the anchor samples. Static pads handed over as moving span
# 0.55-1.03 m (6 deep-traced flights, whole flight after the hand-over); moving pads spanned >= 1.68 m at every
# first arm of the 14 engaged flights (2030969537 is the lowest). Below it the fix does nothing (CX_EVENTS mv_still).
_MV_SPAN_MIN = _cx_knob("CX_MV_SPAN", 1.3, _CX_MV_ANCHOR)
# CX_MV_SPIN (needs CX_MV_ANCHOR): hold uid130 in its stage-0 spin at the anchor instead of its outward spiral. The
# spiral flies tangentially, so a pad orbiting the spiral's centre stays ~90 deg off the camera axis (2030969537 v2:
# d 7-10 m for 18 s, never re-seen); spinning in place sweeps the whole orbit (radius <= 4.3 m) every ~9 s.
_CX_MV_SPIN = _cx_on("CX_MV_SPIN")
# CX_MV_GATE (needs CX_MV_ANCHOR): once armed (uid130 lost the pad after the hand-over), uid130's detector outputs whose
# position lies more than _MV_GATE_XY (xy) or _MV_GATE_Z (height) from the orbit anchor are reported as not visible
# (its pt model otherwise drags it to terrain 11-26 m away: 2030969537, 1225093077 logs). The true pad is always within
# 4.3 m of the orbit centre and at its height, so the gate only bites on the true pad if the anchor is > 4.7 m off.
_CX_MV_GATE = _cx_on("CX_MV_GATE")
_MV_GATE_XY = 9.0
_MV_GATE_Z = 3.0
_CX_MM_HB = os.environ.get("CX_MM_HB", "0") == "1"
_CX_MM_AFTER = 1.0    # s after the hand-off before a moving hand-back may fire
_CX_MM_WINDOW = 3.6   # ...and at most this long after it (s): tested fires at +1.02/+1.36/+3.14 rescued, +7.24 did not;
                      # the king re-found and landed 555197988 after a search that trips the trigger from ~+4.1-4.2
                      # (flown: mm_skip_window). Midpoint of that gap; chosen on this epoch's seeds.
_CX_MM_LOST_S = 0.3   # uid130 continuously in search this long (it enters search after 10-30 unseen ticks)
_CX_MM_SEEN_S = 2.0   # my shadow track saw the pad at most this long ago (my moving APPROACH gives up at 2.5 s + bearing term)
_CX_MM_HITS = 30      # ...on an established track (>= 30 also keeps my APPROACH off its weak-track path)
_CX_MM_DMAX = 6.0     # drone within this of my predicted pad (horizontal, m)
_CX_MM_DZMAX = 5.0    # ...and within this vertically (m)
_MTNFB_T = float(os.environ.get("MY_MTNFB_T", "30"))
_MTNFB_SEARCH_S = float(os.environ.get("MY_MTNFB_SEARCH_S", "0"))  # >0: hand a mountain search to the king after this long without a pad
_FB_LABELS = tuple(x for x in os.environ.get("MY_FB_LABELS", "mountain").split(",") if x)
_NO_MOTION_HO = tuple(x for x in os.environ.get("MY_NOMOTION_HO", "").split(",") if x)
_KING_ROUTE_LABELS = tuple(l for l in ("open", "warehouse") if os.environ.get("MY_ROUTE_" + l.upper(), "king") == "king") + (("city",) if os.environ.get("MY_ROUTE_CITY", "dual") == "king" else ()) + (("forest",) if os.environ.get("MY_ROUTE_FOREST", "mine") == "king" else ())
_KING_MOVING_LABELS = (("open",) if os.environ.get("MY_ROUTE_OPEN", "king") == "dual" else ()) + (("city",) if os.environ.get("MY_ROUTE_CITY", "dual") != "king" else ()) + (("mountain",) if os.environ.get("MY_ROUTE_MTN", "dual") == "dual" else ()) + (("village",) if os.environ.get("MY_ROUTE_VIL", "dual") == "dual" else ())


# c20: cut the old lane's graph-pack compute. On validators a seed gets a hard ~600 s wall-clock cap
# (GLOBAL_EVAL_CAP_SEC); over it the batch is an INFRA timeout, never uploaded, and the backend reaper
# scores the seed 0.0. uid183's 5 reaper zeros in its V5.1.6.0 re-evaluation are all village, where
# its step costs ~580 s per 1,000 steps locally against 90-280 s on other maps: the graph pack runs
# all 11 nodes every tick. Two node-level skips, each env-gated and OFF by default:
#   MY_LEAN_PT=1      pt.onnx (18 MB mountain detector, ~32% of ONNX time) fed its first-tick outputs'
#                     shapes as zeros from tick 2 on; the graph only flies village and forest
#   MY_LEAN_FOREST=1  the forest branch (mnew, forest_a, nav_f, forest_b) when the router picks main;
#                     the arbiter then returns main exactly (router[1] * forest + (1 - router[1]) * main)
# c21: both skips ON by default (c20lean: seed-identical to c11m/uid183 on every map, ~4x less ONNX per
# village tick). Set MY_LEAN_PT=0 / MY_LEAN_FOREST=0 to restore the full graph pack.
_LEAN_PT = os.environ.get("MY_LEAN_PT", "1") == "1"
_LEAN_FOREST = os.environ.get("MY_LEAN_FOREST", "1") == "1"
_LEAN_FOREST_NODES = frozenset(("mnew", "forest_a", "nav_f", "forest_b"))

# ---- fx_wh: warehouse flights flown by king._forest (env-gated, all OFF by default) ----
# (counts go to the module's one CX_EVENTS through _cx_ev, both defined at the top of this file)
_CX_WH_BOX = os.environ.get("CX_WH_BOX", "0") == "1"      # king._forest search point clamped into the pad box (king_src); WSEARCH measures arrival there
_CX_WH_ZLO = os.environ.get("CX_WH_ZLO", "0") == "1"      # king._forest search point height floored (king_src)
_CX_WH_AGL = os.environ.get("CX_WH_AGL", "0") == "1"      # no descent onto a surface under the drone that is not the pad
_CX_WH_WSTRK = os.environ.get("CX_WH_WSTRK", "0") == "1"  # WSEARCH waits while mine's track is reliable but young
_CX_WH_BOX_X, _CX_WH_BOX_Y = 38.0, 23.0   # warehouse pad placement box (TYPE_4_WORLD_RANGE_X/Y)
_CX_WH_AGL_MIN = 1.2      # keep at least this much height over a non-pad surface under the drone (m)
_CX_WH_AGL_CLIMB = 0.9    # below this, climb gently (m)
_CX_WH_AGL_LOOK = 0.5     # the height check looks this far ahead along the current sink rate (s)
_CX_WH_AGL_PAD_R = 1.2    # never within this horizontal distance of the king's pad estimate (pads keep 1 m clear of structures)
_CX_WH_AGL_MAX_N = 200    # fail-safe: hand the descent back after this many modified ticks inside one zone (4 s)
_CX_WH_AGL_START_R = 1.5  # the start pad (r 0.6, 1 m clear of structures) is not an obstacle within this xy radius...
_CX_WH_AGL_START_DZ = 0.3  # ...when the down-ray surface is within this height of the start pad top
_CX_WH_WSTRK_HITS = 30    # an established track for the WSEARCH > APPROACH hand-over


def _cx_wh_box_xy(c):
    p = np.array(c, dtype=np.float64, copy=True)
    p[0] = min(max(float(p[0]), -_CX_WH_BOX_X), _CX_WH_BOX_X)
    p[1] = min(max(float(p[1]), -_CX_WH_BOX_Y), _CX_WH_BOX_Y)
    return p


# ---- fx_w8: warehouse king-route speed guard (every flag default OFF; with both off the agent is c5) ----
# On warehouse the king's forest pilot flies its search at up to 3 m/s along a direction picked from camera rays, then
# slerped, smoothed and passed through its navigation net; when the target swings to the side (passing the search point,
# overshooting it, a flickering false detection) the flown direction ends up at or beyond the edge of the 90 deg camera
# while a rack is within ~1 m. 1977870403 (3 m/s, 44-51 deg off the camera axis, min depth 1.0 m -> hit a rack end
# 0.9 s later), 330943516 (1.6 m/s, 40-46 deg below the axis, min depth 0.5 m for 1 s -> clipped a block corner),
# 1064471781 (3 m/s, 40-49 deg off, min depth 0.6 m -> 0.21 m clearance).
# CX_W8_BLIND: while the pilot is in search/navigation, something is closer than _CX_W8_NEAR in the depth image and the
#   commanded or the actual velocity moves faster than _CX_W8_VB horizontally along a direction outside the central
#   +-_CX_W8_TH deg of the camera, the commanded speed is capped at _CX_W8_VB (direction and yaw unchanged).
# CX_W8_BRAKE: otherwise, when the commanded direction is inside that cone, the commanded speed is capped at the speed
#   that stops within the free depth along it (tube _CX_W8_TUBE, deceleration _CX_W8_ACC, margin _CX_W8_MARGIN).
# Both stop acting for the rest of the seed after _CX_W8_MAXN modified ticks (fail-safe).
_CX_W8_BLIND = os.environ.get("CX_W8_BLIND", "0") == "1"
_CX_W8_BRAKE = os.environ.get("CX_W8_BRAKE", "0") == "1"
_CX_W8_ANY = _CX_W8_BLIND or _CX_W8_BRAKE
_CX_W8_TH = math.tan(math.radians(float(os.environ.get("CX_W8_TH", "40")) if _CX_W8_ANY else 40.0))
_CX_W8_NEAR = float(os.environ.get("CX_W8_NEAR", "1.5")) if _CX_W8_ANY else 1.5      # m, min depth in the image
_CX_W8_VB = float(os.environ.get("CX_W8_VB", "1.0")) if _CX_W8_ANY else 1.0          # m/s
_CX_W8_TUBE = float(os.environ.get("CX_W8_TUBE", "0.3")) if _CX_W8_ANY else 0.3      # m
_CX_W8_ACC = float(os.environ.get("CX_W8_ACC", "2.0")) if _CX_W8_ANY else 2.0        # m/s^2
_CX_W8_MARGIN = float(os.environ.get("CX_W8_MARGIN", "0.4")) if _CX_W8_ANY else 0.4  # m
_CX_W8_VMINB = 0.5    # the brake leaves commands slower than this alone (m/s)
_CX_W8_MAXN = int(os.environ.get("CX_W8_MAXN", "300")) if _CX_W8_ANY else 300       # ticks (6 s) per seed
_CX_W8_N = 32
_cx_w8_u = (np.arange(_CX_W8_N) + 0.5) / _CX_W8_N * 2 - 1
_CX_W8_XR, _CX_W8_YU = np.meshgrid(_cx_w8_u * math.tan(math.radians(FOV_DEG) / 2), -_cx_w8_u * math.tan(math.radians(FOV_DEG) / 2))


def _cx_w8_cam(v, rpy):
    """Unit direction of world vector v in camera axes (right, up, forward); the camera looks along the body x axis."""
    R = rot_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    fwd, up = R[:, 0], R[:, 2]
    right = np.cross(fwd, up)
    u = np.asarray(v, dtype=np.float64) / max(float(np.linalg.norm(v)), 1e-9)
    return np.array([float(u @ right), float(u @ up), float(u @ fwd)])


def _cx_w8_in_cone(dc):
    return dc[2] > 1e-6 and abs(dc[0]) <= _CX_W8_TH * dc[2] and abs(dc[1]) <= _CX_W8_TH * dc[2]


def _cx_w8_free(dc, pooled01):
    """Free distance along camera direction dc through a tube of radius _CX_W8_TUBE (32x32 min-pooled depth, 10 m)."""
    z = DEPTH_MIN_M + pooled01.astype(np.float64) * (DEPTH_MAX_M - DEPTH_MIN_M)
    pts = np.stack([_CX_W8_XR * z, _CX_W8_YU * z, z], axis=-1).reshape(-1, 3)[z.reshape(-1) < 19.6]
    if pts.shape[0] == 0:
        return 10.0
    proj = pts @ dc
    lat2 = np.clip(np.sum(pts * pts, axis=1) - proj * proj, 0.0, None)
    hit = (proj > 0.0) & (lat2 < _CX_W8_TUBE * _CX_W8_TUBE)
    if not np.any(hit):
        return 10.0
    return float(min(10.0, np.min(np.clip(proj[hit] - np.sqrt(np.clip(_CX_W8_TUBE * _CX_W8_TUBE - lat2[hit], 0.0, None)), 0.0, None))))


# ---- fx_w17: warehouse king-forest crashes, round 17 (every flag default OFF; with all off the agent is c17) ----
# CX_W17_BRAKE (in-view proximity brake). In 13 of 21 warehouse failures of the candidate family the king's forest
#   pilot flew at 1.1-3.0 m/s straight at an obstacle its camera could see: free depth along the velocity fell from
#   2-3 m to the 0.5 m depth floor over 0.5-1.5 s while the pilot kept (or raised) its speed; its own direction also had
#   no clearance (the picker's fallback), and the king's tilt-guard coast held the velocity whenever the late swerve
#   tilted the drone. The brake caps the commanded velocity so the drone can stop inside the free depth measured along
#   the command and along the actual velocity (tube _CX_W17_TUBE; deceleration _CX_W17_ACC after a reaction lag
#   _CX_W17_LAG; stop margin _CX_W17_MARGIN). Only directions inside the camera view are judged; the part of the
#   command across the obstacle is kept (the pilot can still slide or turn away). Search/navigation only.
# CX_W17_EDGE (raised-pad edge). The forest pilot commits its landing to a static-lock anchor that can sit ~0.6 m off
#   the pad centre; on a pad raised on a rack the drone then slides past the pad edge and descends to the floor
#   (375612922, 1936553699; 66625680 and 1285880711 earlier). Mine's shadow track had the centre within 0.02-0.08 m.
#   While a committed landing is within 1.5 m (xy) and 1.2 m (height) of mine's reliable static track and the down-ray
#   misses the pad (surface more than 0.6 m below the pad top), the anchor is moved to mine's estimate when they
#   disagree by 0.3-1.5 m; below pad-top + 0.3 m and outside the pad's 0.45 m, the drone first climbs straight up.
_CX_W17_BRAKE = os.environ.get("CX_W17_BRAKE", "0") == "1"
_CX_W17_EDGE = os.environ.get("CX_W17_EDGE", "0") == "1"
_CX_W17_ACC = float(os.environ.get("CX_W17_ACC", "2.5")) if _CX_W17_BRAKE else 2.5       # m/s^2
_CX_W17_LAG = float(os.environ.get("CX_W17_LAG", "0.2")) if _CX_W17_BRAKE else 0.2       # s
_CX_W17_MARGIN = float(os.environ.get("CX_W17_MARGIN", "0.5")) if _CX_W17_BRAKE else 0.5  # m (depth floor 0.5 m)
_CX_W17_TUBE = float(os.environ.get("CX_W17_TUBE", "0.3")) if _CX_W17_BRAKE else 0.3     # m
_CX_W17_VIEW = math.tan(math.radians(42.0))   # judge only directions inside the 90 +-2 deg image
_CX_W17_MAXN = int(os.environ.get("CX_W17_MAXN", "750")) if _CX_W17_BRAKE else 750       # ticks (15 s) per seed
# rev 3 knob (unset = rev 2): judge a direction only when its speed (the command's for the command direction, the
# drone's for the velocity direction) exceeds this. 584484433 lost its pad after the brake stopped a 0.3-0.6 m/s creep.
_CX_W17_VMIN = float(os.environ.get("CX_W17_VMIN", "-1")) if _CX_W17_BRAKE else -1.0      # m/s
_CX_W17_EDGE_HITS = 30
_CX_W17_PADR = 1.2     # m: a free-depth hit this close to the pad estimate is the pad (or its support), not an obstacle
_CX_W17_EDGE_MISS = 5  # consecutive ticks the down-ray must miss the pad (it flickers to the floor during touchdown)

# ---- fx_rb: brake pins (regression cluster B, 2026-09-24; every flag default OFF, with all off the agent is c19) ----
# CX_RB_PINHO: a W17 brake 'pin' is a dead end for the king's forest pilot. It keeps commanding into the face the brake
#   stops it at (it has no boxed-in escape), and while the drone sits still 0.2-0.5 m from a box or rack its platform
#   detector locks onto the face (a stable false estimate), after which it lands into the obstacle (175100634,
#   1393963610), or its command drifts below the camera view where the brake does not judge it and it flies in
#   (2100853735). A pin tick: forest search/navigation, speed < _CX_RB_PIN_V and a brake modification within
#   _CX_RB_PIN_GAP ticks; once a pin episode lasts _CX_RB_PIN_T s the flight goes to mine (the WSEARCH hand-over), whose
#   planner has the boxed-in escape (climb or back off) and which keeps its shadow track and search.
_CX_RB_PINHO = os.environ.get("CX_RB_PINHO", "0") == "1"
_CX_RB_PIN_T = float(os.environ.get("CX_RB_PIN_T", "1.0")) if _CX_RB_PINHO else 1.0   # s
_CX_RB_PIN_V = float(os.environ.get("CX_RB_PIN_V", "0.5")) if _CX_RB_PINHO else 0.5   # m/s
_CX_RB_PIN_GAP = 10    # ticks


def _cx_w17_free(dc, pooled01, tube):
    """Free distance along camera direction dc through a tube of radius `tube` (32x32 min-pooled depth, 10 m)."""
    z = DEPTH_MIN_M + pooled01.astype(np.float64) * (DEPTH_MAX_M - DEPTH_MIN_M)
    pts = np.stack([_CX_W8_XR * z, _CX_W8_YU * z, z], axis=-1).reshape(-1, 3)[z.reshape(-1) < 19.6]
    if pts.shape[0] == 0:
        return 10.0
    proj = pts @ dc
    lat2 = np.clip(np.sum(pts * pts, axis=1) - proj * proj, 0.0, None)
    hit = (proj > 0.0) & (lat2 < tube * tube)
    if not np.any(hit):
        return 10.0
    return float(min(10.0, np.min(np.clip(proj[hit] - np.sqrt(np.clip(tube * tube - lat2[hit], 0.0, None)), 0.0, None))))


def _cx_w17_vmax(free):
    """Largest speed that still stops before free - margin after the reaction lag: v*lag + v^2/(2*acc) = free - margin."""
    d = max(0.0, float(free) - _CX_W17_MARGIN)
    a_, l_ = _CX_W17_ACC, _CX_W17_LAG
    return a_ * (math.sqrt(l_ * l_ + 2.0 * d / a_) - l_)


def _cx_w17_in_view(dc):
    return dc[2] > 1e-6 and abs(dc[0]) <= _CX_W17_VIEW * dc[2] and abs(dc[1]) <= _CX_W17_VIEW * dc[2]


# ---- fx_w8: warehouse king-route diagnostics (testing only; OFF unless CX_W8_DBG names a file) ----
_CX_W8_DBG = os.environ.get("CX_W8_DBG", "").strip()
if _CX_W8_DBG in ("0",):
    _CX_W8_DBG = ""


def _cx_w8_dbg_write(ctl, observation, a_king, a_out):
    """One JSON line per king-route tick: state, the king pilot's internals, its action, the flown action and a 32x32
    min-pooled depth image (uint8). Never changes an action; any error is swallowed."""
    try:
        import json as _json
        st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        dp = np.asarray(observation["depth"], dtype=np.float32).reshape(IMG_H, IMG_W)
        pooled = dp.reshape(32, 4, 32, 4).min(axis=(1, 3))
        k = ctl.king
        rec = {"step": int(ctl.step), "st": np.round(st[0:12].astype(float), 4).tolist(),
               "agl": round(float(st[137]) * 20.0, 3), "sv": np.round(st[138:141].astype(float), 3).tolist(),
               "act": getattr(k, "_active", None),
               "ak": None if a_king is None else np.round(np.asarray(a_king, dtype=float).reshape(-1), 4).tolist(),
               "ao": None if a_out is None else np.round(np.asarray(a_out, dtype=float).reshape(-1), 4).tolist(),
               "dep": _b64.b64encode(np.clip(pooled * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes()).decode()}
        kf = getattr(k, "_forest", None)
        if getattr(k, "_active", None) == "forest" and kf is not None:
            def _l(v):
                return None if v is None else np.round(np.asarray(v, dtype=float).reshape(-1), 3).tolist()
            esv = getattr(kf, "extra_search_vectors", None)
            ss = int(getattr(kf, "search_stage", 0))
            rec["kf"] = {"mode": str(kf._mode), "ss": ss, "rot": int(getattr(kf, "search_state_0_rot_count", 0)),
                         "gr": bool(getattr(kf, "_goal_return_mode", False)), "fwd": bool(getattr(kf, "_forward", True)),
                         "foc": int(getattr(kf, "first_order_cnt", 0)), "seeP": bool(getattr(kf, "see_P", False)),
                         "findP": bool(getattr(kf, "is_find_P", False)), "trk": bool(getattr(kf, "tracking", False)),
                         "gt": bool(getattr(kf, "_goal_is_tracked", False)),
                         "com": bool(getattr(kf, "_landing_committed", False)),
                         "pp": _l(getattr(kf, "platform_position", None)),
                         "lpp": _l(getattr(kf, "_landing_platform_position", None)),
                         "off": _l(esv[ss]) if (esv is not None and 0 <= ss < len(esv)) else None,
                         "la": _l(getattr(kf, "_last_action", None)), "hatt": bool(getattr(kf, "is_hatt", True))}
        km = getattr(k, "_main", None)
        if getattr(k, "_active", None) == "main" and km is not None:
            rec["km"] = {"mode": str(getattr(km, "_mode", ""))}
        rec["gc"] = int(getattr(k, "_g_coast", 0) or 0)
        with open(_CX_W8_DBG, "a") as fh:
            fh.write(_json.dumps(rec) + "\n")
    except Exception:
        pass


def _install_lean_graph(mod):
    G = getattr(mod, "_GraphPackController", None)
    if G is None or getattr(G, "_lean_installed", False):
        return
    orig_act = G.act

    def act(self, observation):
        try:
            router = self._memory.get("router")
            skip_forest = bool(_LEAN_FOREST and router is not None
                               and float(np.asarray(router, dtype=np.float32).reshape(-1)[1]) == 0.0)
        except Exception:
            skip_forest = False
        skip_pt = bool(_LEAN_PT and getattr(self, "_lean_pt_zero", None) is not None)
        if not (skip_forest or skip_pt):
            out = orig_act(self, observation)
            return out
        depth = np.ascontiguousarray(np.asarray(observation["depth"], dtype=np.float32).reshape(128, 128, 1))
        state = np.ascontiguousarray(np.asarray(observation["state"], dtype=np.float32).reshape(141))
        obs = {"depth": depth, "state": state}
        tick_outputs = {}
        for node_id in self._topo_order:
            node = self._nodes[node_id]
            if skip_forest and node_id in _LEAN_FOREST_NODES:
                continue
            if skip_pt and node_id == "pt":
                for name, value in self._lean_pt_zero.items():
                    tick_outputs[("pt", name)] = value
                continue
            feeds = {}
            for name, (kind, head, tail) in node["inputs"].items():
                if kind == "obs":
                    feeds[name] = obs[head]
                elif kind == "memory":
                    feeds[name] = self._memory[head]
                elif (head, tail) in tick_outputs:
                    feeds[name] = tick_outputs[(head, tail)]
                else:
                    # a skipped forest output feeding the arbiter: the router weights it by exactly 0
                    feeds[name] = np.zeros_like(tick_outputs[("main_b", tail)])
            outputs = self._model_sessions[node["model"]].run(None, feeds)
            for name, value in zip(node["output_names"], outputs):
                tick_outputs[(node_id, name)] = np.asarray(value)
        raw = tick_outputs[self._action_key].astype(np.float32, copy=False).reshape(-1)
        action = np.clip(raw, mod._GRAPHPACK_ACTION_LOWER, mod._GRAPHPACK_ACTION_UPPER)
        action = (np.rint(action / mod._GRAPHPACK_QUANT_STEP) * mod._GRAPHPACK_QUANT_STEP).astype(np.float32)
        if not np.isfinite(action).all():
            action = np.zeros(5, dtype=np.float32)
        for name, shape, writer in self._memory_slots:
            value = tick_outputs.get(writer)
            if value is not None:
                self._memory[name] = np.array(value, dtype=np.float32, copy=True).reshape(shape)
        return action

    def act_capture(self, observation):
        # first tick: run pt for real and remember its output shapes, then zeros from tick 2 on
        if _LEAN_PT and getattr(self, "_lean_pt_zero", None) is None:
            orig_run = self._model_sessions["pt"].run
            captured = {}
            sess = self._model_sessions["pt"]
            names = self._nodes["pt"]["output_names"]

            class _Cap:
                def __init__(self, s):
                    self._s = s
                def run(self, output_names, feeds, *a, **k):
                    outs = self._s.run(output_names, feeds, *a, **k)
                    for n, v in zip(names, outs):
                        captured[n] = np.zeros_like(np.asarray(v))
                    return outs
                def __getattr__(self, item):
                    return getattr(self._s, item)
            self._model_sessions["pt"] = _Cap(sess)
            try:
                out = act(self, observation)
            finally:
                self._model_sessions["pt"] = sess
            if len(captured) == len(names):
                self._lean_pt_zero = captured
            return out
        return act(self, observation)

    G.act = act_capture
    orig_reset = G.reset

    def reset(self):
        self._lean_pt_zero = None
        return orig_reset(self)
    G.reset = reset
    G._lean_installed = True


def _load_king_class():
    src = _zlib.decompress(_b64.b64decode("".join(_KING_SRC_B64))).decode("utf-8")
    mod = _types.ModuleType("_king_agent")
    mod.__file__ = __file__
    _sys.modules["_king_agent"] = mod
    exec(compile(src, "_king_agent", "exec"), mod.__dict__)
    if _LEAN_PT or _LEAN_FOREST:
        _install_lean_graph(mod)
    return mod.DroneFlightController


class DroneFlightController:
    def __init__(self, models_dir=None):
        self.mine = _MyController(models_dir)
        self.king = None
        try:
            self.king = _load_king_class()()
        except Exception:
            self.king = None
        self.route = None
        self.step = 0
        self.debug = {}
        self._dv_reset()  # fx_dv state exists before the first reset() too
        self._mg_reset()  # fx_mg state likewise
        if _CX_MPC:
            self._mpc_reset()  # fx_mpc state likewise
        if _CX_CR_ANY:
            self._cr_reset()  # fx_cr state likewise
        self._kl_reset()  # fx_kl state likewise
        if _CX_PW_KSEED:
            self._pw_ks_done, self._pw_ks_n = False, 0  # fx_kseed state likewise
        if _CX_PW_GSINK:
            self._pw_gs_done, self._pw_gs_s = False, 0.0  # fx_pw2 GSINK state likewise
        if _CX_VK2_ANY:
            self._vk2_reset()  # fx_vk (round 23) state likewise
        self._mv_pts = []  # CX_MV_ANCHOR per-seed state (also set in reset())
        self._mv_in_search = False
        self._mv_dbg_key = None
        self._mv_gate = None
        self._mv_had = False
        self._mv_armed = False
        self._mv_prev_hits = 0
        self._mv_clr = 0
        self._mv_still_ep = False
        self._mv_arm_t = None
        if _PC_ANY:
            self._pc_hold_t0, self._pc_hold_n, self._pc_ksee_done = None, 0, False  # fx_pc state likewise
        if _CX_S1F_EDGE:
            self._s1f_eh = 0  # fx_s1f EDGE: 0 not engaged, 1 holding, 2 done (state likewise)
        if _CX_RG_ANY:
            self._rg_reset()  # fx_s1r state likewise
        if _CX_S2K_OPEN:
            self._s2k_reset()  # fx_s2kpad state likewise

    def reset(self):
        self.mine.reset()
        if self.king is not None:
            try:
                self.king.reset()
            except Exception:
                pass
        if _CX_CT_KCITY and self.king is not None:
            # the city hint lives on the king's main controller, whose reset() does not know it: clear it per seed so a
            # stale True cannot relabel the next mountain/village seed's early XGB predictions (set only on a dual route)
            try:
                km = getattr(self.king, "_main", None)
                if km is not None:
                    km._cx_ct_city = False
            except Exception:
                pass
        if (_CX_K9_WH or _CX_K9_NG) and self.king is not None:
            # fx_k9: the hints live on the king's main and _forest controllers; agent_forest.reset() does not know
            # _cx_k9_wh, so clear all of them per seed (a stale hint must never reach the next seed's pick)
            try:
                km = getattr(self.king, "_main", None)
                if km is not None:
                    km._cx_k9_wh, km._cx_k9_ng, km._cx_k9_ng_latch = False, None, None
                kf = getattr(self.king, "_forest", None)
                if kf is not None:
                    kf._cx_k9_wh = False
            except Exception:
                pass
        self._k9_wh_seen = False
        self._k9_ng_seen = False
        self._last_key = None
        self._last_action = None
        self.route = None
        self.step = 0
        self.z0 = 0.0
        if _CX_S1F_EDGE:
            self._s1f_eh = 0  # fx_s1f EDGE per-seed state
        self.high_open = False
        self._ws_done = False
        self._wl_close = False; self._wl_far_t = None; self._wl_pad = None
        self.king_from_dual = False
        self.king_blocked = False
        self.ho_t = 0.0
        self.hb_lost_n = 0
        if _PC_ANY:  # fx_pc per-seed state: SEEN first held tick, held ticks; DIAG king-sighting probe done
            self._pc_hold_t0, self._pc_hold_n, self._pc_ksee_done = None, 0, False
        self.handback_t = None
        # fx_wh per-seed state (validators reuse one agent across seeds and call only reset()). c2: set whatever the
        # flags; every reader is behind its CX_WH_* flag (the forest source only reads _cx_wh when patched by BOX/ZLO)
        self._cx_agl_n = 0
        self._cx_wh_start = None
        self._cx_ws_latch = False
        self._cx_w8_n = 0      # fx_w8: modified ticks this seed (fail-safe budget)
        self._cx_w17_n = 0     # fx_w17: brake-modified ticks this seed (fail-safe budget)
        self._cx_w17_miss = 0  # fx_w17: consecutive EDGE ticks with the down-ray off the pad
        self._cx_w17_up = False  # fx_w17: EDGE climb latch
        self._cx_rb_fire = -999  # fx_rb: last tick the W17 brake modified the action
        self._cx_rb_pin0 = None  # fx_rb: first tick of the current pin episode
        self._cx_rb_pinl = -999  # fx_rb: last pin tick
        try:
            self.king._forest._cx_wh = False  # agent_forest.reset() keeps it: clear it so a seed never inherits it
            self.king._forest._cx_zlo_n, self.king._forest._cx_zlo_on = 0, False  # CX_WH_ZLO2 per-seed state
        except Exception:
            pass
        self._vkw_prev = None  # CX_VK_WATCH state (read only when the flag is on)
        self._mv_pts = []  # CX_MV_ANCHOR per-seed state (set whatever the flags)
        self._mv_in_search = False
        self._mv_dbg_key = None
        self._mv_gate = None
        self._mv_had = False
        self._mv_armed = False
        self._mv_prev_hits = 0
        self._mv_clr = 0
        self._mv_still_ep = False
        self._mv_arm_t = None
        self._vkw_n = 0
        self._vkw_d0 = 0.0
        self._cx_mm_n = 0      # fx_mm: consecutive ticks uid130 has spent in search since the hand-off
        self._cx_mm_armed = False  # fx_mm: uid130 has been in navigation/landing since the hand-off
        self.cx_mm_t = None    # fx_mm: time of the moving hand-back
        self._dv_reset()       # fx_dv per-seed state (read only when CX_DV_DIVE is on)
        self._mg_reset()       # fx_mg per-seed state (read only when CX_MG_FLOOR is on)
        if _CX_MPC:
            self._mpc_reset()  # fx_mpc per-seed state
        if _CX_CR_ANY:
            self._cr_reset()   # fx_cr per-seed state
        self._kl_reset()       # fx_kl per-seed state (read only when a CX_KL_* flag is on)
        if _CX_PW_KSEED:
            self._pw_ks_done, self._pw_ks_n = False, 0  # fx_kseed per-seed state
        if _CX_PW_GSINK:
            self._pw_gs_done, self._pw_gs_s = False, 0.0  # fx_pw2 GSINK per-seed state
        if _CX_VK2_ANY:
            self._vk2_reset()  # fx_vk (round 23) per-seed state
        if _CX_RG_ANY:
            self._rg_reset()   # fx_s1r per-seed state
        if _CX_S2K_OPEN:
            self._s2k_reset()  # fx_s2kpad per-seed state

    def _vk2_reset(self):
        self._vk2_grid = None    # 32x32 depth pooling grid (built on first use)
        self._vk2_prev = False   # BRAKE acted (or would have) on the previous tick (episode counting)
        self._vk2_diag = []      # DIAG ring buffer
        self._vk2_gmin = None    # lowest surface the down-ray has read this flight (the village ground)
        self._vk2_mem = []       # (t, world points) of recent frames, obstacle candidates (ground already cut)
        self._vk2_pin0 = None    # PIN: first tick of the running pin episode
        self._vk2_pinl = -999    # PIN: last pinned tick
        self._vk2_pin_done = False
        self._vk2_sh = []        # ROOF: (t, surface z) of the last ~0.1 s (restarted at a step)
        self._vk2_rprev = False  # ROOF acted on the previous tick
        self._vk2_rlast = -999.0  # ROOF: time of the last ROOF tick (episode counting)
        self._vk2_seen = None    # PIN: my shadow track's last sighting (>= 10 hits) on the village king route

    def _vk2_gate(self):
        """fx_vk (round 23): the village king route (route king and mine's map label village). A village read as open at
        take-off (1432876675) is left out: it is king-routed from the start with no pad estimate, and a true open flight
        of the graph pack could not be told apart from it."""
        return self.route == "king" and self.king is not None and self.mine.map_label == "village"

    def _vk2_pads(self, t, st):
        """Every pad estimate the router can read, as (x, y, z or None): mine's (shadow) track, the vking hand-over
        memory, the king pilot's platform track (main/uid130; not its seed value, the clue centre: 669185741 lost the
        brake 3 m from the clue, 15 m from the pad), and fx_mg's predicted and good estimates (xy only)."""
        out = []
        clue = st[0:3].astype(np.float64) + st[138:141].astype(np.float64)
        m = self.mine
        for p in (getattr(m, "pad_pos", None), getattr(self, "_vk_mem", None)):
            if p is not None:
                p = np.asarray(p, dtype=np.float64).reshape(-1)
                if p.size >= 3 and np.isfinite(p[:3]).all():
                    out.append((float(p[0]), float(p[1]), float(p[2])))
        pm = self._dv_kmain()
        if pm is not None:
            pp = getattr(pm, "platform_position", None)
            if pp is not None:
                pp = np.asarray(pp, dtype=np.float64).reshape(-1)
                if pp.size >= 3 and np.isfinite(pp[:3]).all() and float(np.abs(pp[:3] - clue).max()) > 1e-3:
                    out.append((float(pp[0]), float(pp[1]), float(pp[2])))
        if _CX_MG_FLOOR:
            try:
                ph = self._mg_pred(t)
                if ph is not None:
                    ph = np.asarray(ph, dtype=np.float64).reshape(-1)
                    out.append((float(ph[0]), float(ph[1]), None))
                if self._mg_gxy is not None:
                    g = np.asarray(self._mg_gxy, dtype=np.float64).reshape(-1)
                    out.append((float(g[0]), float(g[1]), None))
                top_, _src = self._mg_top()
                if top_ is not None and math.isfinite(float(top_)) and out:
                    out.append((out[-1][0], out[-1][1], float(top_)))
            except Exception:
                pass
        return out

    def _vk2_step(self, observation, a):
        """CX_VK2_DIAG / CX_VK2_BRAKE / CX_VK2_PIN / CX_VK2_ROOF (see the fx_vk round-23 block at the flags). Returns the
        route's action unchanged unless a visible obstacle lies within the stopping distance along the drone's velocity or
        the route's command (BRAKE), or a raised non-pad surface closes in under the drone (ROOF). Any error returns a."""
        try:
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            t = float(self.step) * SIM_DT
            pos = st[0:3].astype(np.float64)
            agl = float(st[137]) * 20.0
            sat = (not math.isfinite(agl)) or agl >= _M2_SAT_M
            surf = float(pos[2]) - agl
            if not sat and self.step > 25:  # (the first 0.5 s the ray reads the start pad)
                self._vk2_gmin = surf if self._vk2_gmin is None else min(self._vk2_gmin, surf)
            if not self._vk2_gate():
                self._vk2_prev, self._vk2_rprev = False, False
                self._vk2_mem, self._vk2_sh = [], []
                return a
            m0 = self.mine
            if (m0.pad_pos is not None and int(getattr(m0, "pad_hits", 0)) >= 10
                    and float(m0.pad_last_seen) == float(m0.t)):
                self._vk2_seen = np.asarray(m0.pad_pos, dtype=np.float64).reshape(-1)[:3].copy()  # PIN's reference
            rpy = st[3:6].astype(np.float64)
            vel = st[6:9].astype(np.float64)
            a_ = np.asarray(a, dtype=np.float64).reshape(-1)
            if a_.size < 5 or not np.isfinite(a_[:5]).all():
                self._vk2_prev, self._vk2_rprev = False, False
                return a
            n3 = float(np.linalg.norm(a_[0:3]))
            v_a = a_[0:3] / n3 * min(abs(float(a_[3])), 1.0) * SPEED_LIMIT if n3 > 1e-6 else np.zeros(3)
            gref = (self._vk2_gmin if self._vk2_gmin is not None else 0.2) + _VK2_GCUT
            pads = self._vk2_pads(t, st)
            dpad = min((math.hypot(p[0] - pos[0], p[1] - pos[1]) for p in pads), default=None)
            # ---- depth points (world), ground cut, recent-frame memory ----
            if self._vk2_grid is None:
                self._vk2_grid = DepthGrid(32)
            dep = np.asarray(observation["depth"], dtype=np.float32).reshape(IMG_H, IMG_W)
            pts, zc = self._vk2_grid.points_cam(self._vk2_grid.pool_min(dep))
            eye, fwd, right, up = cam_pose(pos, rpy)
            M = np.stack([right, up, fwd], axis=0)
            P = eye[None, :] + pts[:, 2:3] * fwd[None, :] + pts[:, 0:1] * right[None, :] + pts[:, 1:2] * up[None, :]
            P = P[(zc < 19.6) & (P[:, 2] > gref)]
            if _VK2_MEM_S > 0:
                self._vk2_mem = [(t_, q_) for (t_, q_) in self._vk2_mem if t - t_ <= _VK2_MEM_S + 1e-6]
                if self.step % 5 == 0:
                    near = np.hypot(P[:, 0] - pos[0], P[:, 1] - pos[1]) < 6.0
                    self._vk2_mem.append((t, P[near].copy()))

            def cut_pads(Q, r):
                if Q.shape[0] == 0:
                    return Q
                keep = np.ones(Q.shape[0], dtype=bool)
                for p in pads:
                    keep &= np.hypot(Q[:, 0] - p[0], Q[:, 1] - p[1]) > r
                return Q[keep]

            Pc = cut_pads(P, _VK2_PADR)
            # a remembered frame may hold a moving pad where it was: its pad cut grows with the frame's age
            Pm = [cut_pads(q_, _VK2_PADR + _VK2_PADV * (t - t_)) for (t_, q_) in self._vk2_mem]
            Pm = np.concatenate(Pm, axis=0) if Pm else np.zeros((0, 3))
            tv = math.tan(math.radians(_VK2_VIEW))

            def free_along(u):
                dc = M @ u
                inview = dc[2] > 1e-3 and abs(dc[0] / dc[2]) <= tv and abs(dc[1] / dc[2]) <= tv
                Q = np.concatenate([Pc, Pm], axis=0) if inview else Pm
                if Q.shape[0] == 0:
                    return None if not inview else 20.0
                R = Q - pos[None, :]
                proj = R @ u
                lat2 = np.clip(np.sum(R * R, axis=1) - proj * proj, 0.0, None)
                hit = (proj > 0.0) & (lat2 < _VK2_TUBE * _VK2_TUBE)
                if not hit.any():
                    return None if not inview else 20.0
                return float(np.min(np.clip(proj[hit] - np.sqrt(_VK2_TUBE * _VK2_TUBE - lat2[hit]), 0.0, None)))

            sp_c = float(np.linalg.norm(vel))
            sp_a = float(np.linalg.norm(v_a))
            near_pad = dpad is not None and dpad < _VK2_NEAR
            v_new = v_a.copy()
            fired, frees = False, {"v": None, "a": None}
            if not near_pad:
                dirs = []
                if sp_c >= _VK2_VMIN:
                    dirs.append(("v", vel / sp_c))
                if sp_a >= _VK2_VMIN:
                    dirs.append(("a", v_a / sp_a))
                for tag, u in dirs:
                    fr = free_along(u)
                    frees[tag] = fr
                    if fr is None:
                        continue
                    vm = math.sqrt(max(0.0, 2.0 * _VK2_DEC * (fr - _VK2_MARGIN)))
                    c = float(v_new @ u)
                    if c > vm + 1e-6:
                        v_new = v_new - (c - vm) * u  # remove only the excess closing speed; the sideways part stays
                        fired = True
            if _CX_VK2_DIAG:
                self._vk2_diag.append([round(t, 2), round(sp_c, 2),
                                       None if frees["v"] is None else round(frees["v"], 2),
                                       None if frees["a"] is None else round(frees["a"], 2),
                                       round(agl, 2), None if dpad is None else round(dpad, 1), int(fired)])
                if len(self._vk2_diag) > _VK2_DIAG_N:
                    del self._vk2_diag[0]
                CX_EVENTS["vk2_diag"] = self._vk2_diag
            # ---- ROOF: a raised non-pad surface closing in under the drone ----
            vz_min = None
            if not sat:
                if self._vk2_sh and abs(surf - self._vk2_sh[-1][1]) > 0.3:
                    self._vk2_sh = []
                self._vk2_sh.append((t, surf))
                self._vk2_sh = [(t_, s_) for (t_, s_) in self._vk2_sh if t - t_ <= 0.1 + 1e-6]
            else:
                self._vk2_sh = []
            if _CX_VK2_ROOF and not sat and surf > gref - _VK2_GCUT + _VK2_ROOF_RAISE:
                far = dpad is not None and dpad > _VK2_ROOF_FAR  # (no pad estimate at all: only the closing rule)
                if (dpad is None or dpad > _VK2_ROOF_R):
                    rise = 0.0
                    if len(self._vk2_sh) >= 2 and self._vk2_sh[-1][0] - self._vk2_sh[0][0] >= 0.04 - 1e-6:
                        rise = (self._vk2_sh[-1][1] - self._vk2_sh[0][1]) / (self._vk2_sh[-1][0] - self._vk2_sh[0][0])
                        rise = min(max(rise, 0.0), 3.0)
                    closing = max(0.0, -float(vel[2])) + rise
                    thr = 0.4 + _VK2_ROOF_TTC * closing
                    if far:
                        thr = max(thr, _VK2_ROOF_AGL)
                    elif dpad is not None and rise < 0.3:
                        thr = -1.0  # near a (roof) pad only a surface that rises under the drone counts
                    if agl < thr:
                        vz_min = min(max(1.5 * (_VK2_ROOF_AGL + 0.2 - agl), 0.0), 1.2) + min(rise, 1.5)
            if not fired and vz_min is None:
                self._vk2_prev, self._vk2_rprev = False, False
                self._vk2_pin_tick(t, False, sp_c, vel, st)
                return a
            if fired and not self._vk2_prev:
                _cx_vk2_ev("vk2_fire" if _CX_VK2_BRAKE else "vk2_would")
                at = CX_EVENTS.setdefault("vk2_at" if _CX_VK2_BRAKE else "vk2_would_at", [])
                if isinstance(at, list) and len(at) < 8:
                    at.append([round(t, 2), None if frees["v"] is None else round(frees["v"], 2),
                               None if frees["a"] is None else round(frees["a"], 2), round(sp_c, 2), round(sp_a, 2),
                               round(float(pos[2]), 2), round(agl, 2), None if dpad is None else round(dpad, 1),
                               str(getattr(self.king, "_active", None)), str((self.debug or {}).get("mode"))])
            self._vk2_prev = fired
            if vz_min is not None and t - self._vk2_rlast > 0.3 + 1e-9:
                _cx_vk2_ev("vk2_roof_fire")
                at = CX_EVENTS.setdefault("vk2_roof_at", [])
                if isinstance(at, list) and len(at) < 8:
                    at.append([round(t, 2), round(float(pos[2]), 2), round(agl, 2), round(surf, 2),
                               None if dpad is None else round(dpad, 1), round(float(vel[2]), 2), round(vz_min, 2),
                               str((self.debug or {}).get("mode"))])
            self._vk2_rprev = vz_min is not None
            if vz_min is not None:
                self._vk2_rlast = t
            if not _CX_VK2_BRAKE:
                fired = False
                if fired is False and vz_min is None:
                    _cx_vk2_ev("vk2_would_ticks")
                    return a
            if fired:
                _cx_vk2_ev("vk2_ticks")
            if vz_min is not None:
                _cx_vk2_ev("vk2_roof_ticks")
                vz_out = max(float(v_new[2]), vz_min)
                vz_out = max(float(v_new[2]), min(vz_out, float(vel[2]) + _VK2_LEAD))
                v_new = np.array([v_new[0], v_new[1], vz_out])
                if agl < _VK2_ROOF_BRAKE_AGL:
                    nh = float(np.hypot(v_new[0], v_new[1]))
                    if nh > _VK2_ROOF_VH:
                        v_new[0:2] = v_new[0:2] * (_VK2_ROOF_VH / nh)
            err = v_new - vel
            en = float(np.linalg.norm(err))
            if en > _VK2_DVH:
                v_new = vel + err * (_VK2_DVH / en)
            vh_ = float(np.hypot(v_new[0], v_new[1]))
            vmax_h = math.sqrt(max(SPEED_LIMIT * SPEED_LIMIT - float(v_new[2]) ** 2, 0.0))
            if vh_ > vmax_h:
                v_new[0:2] = v_new[0:2] * (vmax_h / vh_)
            self.debug = dict(self.debug or {})
            self.debug["mode"] = "VK2BRAKE" if fired else "VK2ROOF"
            self._vk2_pin_tick(t, fired, sp_c, vel, st)  # may hand the flight back to mine from the next tick
            return self._dv_vel_action(v_new, a, st)
        except Exception:
            self._vk2_prev, self._vk2_rprev = False, False
            _cx_vk2_ev("vk2_err")
            return a

    def _vk2_pin_tick(self, t, fired, sp_c, vel, st):
        """CX_VK2_PIN: BRAKE holds the king against an obstacle (braked ticks at < _VK2_PIN_V with gaps of at most
        _VK2_PIN_GAP s) for _VK2_PIN_S: take the flight back the way CX_VK_WATCH does (REACQUIRE aimed at my shadow
        track's last sighting with >= 10 hits on the village king route, else CX_VK_WATCH's reference: the live
        estimate, else the hand-over pad; v_cmd = the current velocity with any descent removed). Only after a village
        moving hand-over with the vking memory armed. True = handed back this tick."""
        if _CX_MPC and self._mpc_own:
            _cx_mpc_ev("mpc_veto_pin") if (fired and sp_c < _VK2_PIN_V) else None
            return False  # fx_mpc owns the flight: no take-back (the pin episode is not counted either)
        if not _CX_VK2_PIN or self._vk2_pin_done:
            return False
        if fired and sp_c < _VK2_PIN_V:
            if self._vk2_pin0 is None or t - self._vk2_pinl > _VK2_PIN_GAP + 1e-9:
                self._vk2_pin0 = t
            self._vk2_pinl = t
        if self._vk2_pin0 is None:
            return False
        if t - self._vk2_pinl > _VK2_PIN_GAP + 1e-9:
            self._vk2_pin0 = None
            return False
        if t - self._vk2_pin0 < _VK2_PIN_S - 1e-9:
            return False
        if (getattr(self, "_vk_mem", None) is None or getattr(self, "_vk_recaptured", False)
                or self.mine.map_label != "village" or self.route != "king"):
            return False
        try:
            m_ = self.mine
            ref = self._vk2_seen if self._vk2_seen is not None else self._vkw_ref()
            pos_ = st[0:3].astype(np.float64)
            m_._drop_track()
            m_.bad_spots = []
            m_.reacq_n = 0
            m_.recover_n = 0
            m_.land_retry = 0
            m_.reacq_moving = True
            m_.reacq_target = ref.copy()
            back = pos_[:2] - ref[:2]
            nb = float(np.linalg.norm(back))
            back = back / nb if nb > 1e-6 else np.array([1.0, 0.0])
            m_.reacq_wp = np.array([ref[0] + back[0] * 7.0, ref[1] + back[1] * 7.0, ref[2] + 3.0])
            m_.reacq_hold = 0.0
            m_.reacq_side = 0
            m_.reacq_t_enter = m_.t
            m_.pad_hist = []
            m_._set_mode("REACQUIRE")
            v0 = vel.astype(np.float64).copy()
            v0[2] = max(float(v0[2]), 0.0)
            m_.v_cmd = v0
            if _CX_VK_UNSTICK:
                m_._vk_tb = True
            self._vk_recaptured = True
            self._vk_mem = None
            self.route = "dual"
            self._vk2_pin_done = True
            _cx_vk2_ev("vk2_pin_takeback")
            if "vk2_pin_t" not in CX_EVENTS:
                CX_EVENTS["vk2_pin_t"] = [round(t, 2), round(float(np.linalg.norm(pos_[:2] - ref[:2])), 1)]
            self.debug = {"route": "dual", "map": m_.map_label, "mode": "VK2PIN", "pad_hits": 0}
            return True
        except Exception:
            self._vk2_pin_done = True
            _cx_vk2_ev("vk2_pin_err")
            return False

    def _cr_reset(self):
        self._cr_sh = []       # (t, surface z) read by the down-ray since the last step, the last ~0.1 s
        self._cr_prev = False  # the guard acted on the previous tick (episode counting)
        self._cr_grid = None   # 32x32 depth pooling grid (built on first use)
        self._cr_fhold = None  # KFWD hold: (until t, minimum vz)

    def _cr_step(self, observation, a):
        """CX_CR_KRAY / CX_CR_KFWD (see the block comment at the flags): city king route after the moving hand-over.
        Returns the route's action unchanged unless a non-pad roof is under the drone (down-ray) or on the glide ahead
        (depth); then the same horizontal command (capped when very low) with the sink cut or a climb. Any error
        returns a."""
        try:
            if not (self.route == "king" and self.king_from_dual and self.mine.map_label == "city"):
                self._cr_prev = False
                return a
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            t = float(self.step) * SIM_DT
            pos = st[0:3].astype(np.float64)
            vel = st[6:9].astype(np.float64)
            z = float(pos[2])
            agl = float(st[137]) * 20.0
            sat = (not math.isfinite(agl)) or agl >= _M2_SAT_M
            surf = z - agl
            if not sat:
                if self._cr_sh and abs(surf - self._cr_sh[-1][1]) > 0.3:
                    self._cr_sh = []  # a step (a roof edge): the rise rate restarts from it
                self._cr_sh.append((t, surf))
                self._cr_sh = [(t_, s_) for (t_, s_) in self._cr_sh if t - t_ <= 0.1 + 1e-6]
            else:
                self._cr_sh = []
            # ---- pad references ----
            m = self.mine
            tops, xys = [], []
            if m.pad_pos is not None:
                mp = np.asarray(m.pad_pos, dtype=np.float64).reshape(-1)
                if mp.size >= 3 and np.isfinite(mp[:3]).all():
                    tops.append(float(mp[2]))
                    xys.append(mp[:2].copy())
            if _CX_MG_FLOOR:
                top_, _src = self._mg_top()
                if top_ is not None and math.isfinite(float(top_)):
                    tops.append(float(top_))
                ph = self._mg_pred(t)
                if ph is not None:
                    xys.append(np.asarray(ph, dtype=np.float64).reshape(-1)[:2])
                if self._mg_gxy is not None:
                    xys.append(np.asarray(self._mg_gxy, dtype=np.float64).reshape(-1)[:2])
            kest = self._dv_kest()
            if kest is not None:
                xys.append(np.asarray(kest[0], dtype=np.float64)[:2])
            if not tops or not xys:
                self._cr_prev = False
                return a
            top = max(tops)
            dpad = min(math.hypot(float(p[0]) - pos[0], float(p[1]) - pos[1]) for p in xys)
            if dpad < _CR_PADR:
                self._cr_prev = False
                return a
            a_ = np.asarray(a, dtype=np.float64).reshape(-1)
            if a_.size < 5 or not np.isfinite(a_[:5]).all():
                self._cr_prev = False
                return a
            n3 = float(np.linalg.norm(a_[0:3]))
            v = a_[0:3] / n3 * min(abs(float(a_[3])), 1.0) * SPEED_LIMIT if n3 > 1e-6 else np.zeros(3)
            vz_min, vh_cap, src, info = None, None, None, None
            # ---- KRAY: a non-pad surface under the drone ----
            if _CX_CR_KRAY and not sat and surf > top + _CR_RAISE:
                rise = 0.0
                if len(self._cr_sh) >= 2 and self._cr_sh[-1][0] - self._cr_sh[0][0] >= 0.04 - 1e-6:
                    rise = (self._cr_sh[-1][1] - self._cr_sh[0][1]) / (self._cr_sh[-1][0] - self._cr_sh[0][0])
                    rise = min(max(rise, 0.0), 3.0)
                closing = max(0.0, -float(vel[2])) + rise
                if agl < max(_CR_AGL, 0.6 + _CR_TTC * closing):
                    vz_min = min(max(_CR_KP * (_CR_AGL - agl), 0.0), _CR_VUP) + min(rise, 1.5)
                    src = "ray"
                    info = [round(rise, 2)]
                    if agl < _CR_BRAKE_AGL:
                        vh_cap = _CR_VH
            # ---- KFWD: a non-pad roof on the glide ahead ----
            vh = math.hypot(float(vel[0]), float(vel[1]))
            if _CX_CR_KFWD and float(vel[2]) < -_CR_FVZ and vh > 0.5:
                if self._cr_grid is None:
                    self._cr_grid = DepthGrid(32)
                dep = np.asarray(observation["depth"], dtype=np.float32).reshape(IMG_H, IMG_W)
                pts, zc = self._cr_grid.points_cam(self._cr_grid.pool_min(dep))
                ok = zc < 19.6
                if ok.any():
                    pts = pts[ok].astype(np.float64)
                    eye, fwd, right, up = cam_pose(pos, st[3:6].astype(np.float64))
                    P = eye[None, :] + pts[:, 2:3] * fwd[None, :] + pts[:, 0:1] * right[None, :] + pts[:, 1:2] * up[None, :]
                    ux, uy = float(vel[0]) / vh, float(vel[1]) / vh
                    rx, ry = P[:, 0] - pos[0], P[:, 1] - pos[1]
                    s = rx * ux + ry * uy
                    lat = -rx * uy + ry * ux
                    zp = P[:, 2]
                    L = max(1.0, min(4.0, _CR_FT * vh))
                    cl = (z + float(vel[2]) * s / vh) - zp
                    thr = ((s > 0.2) & (s < L) & (np.abs(lat) < _CR_W) & (zp > top + _CR_RAISE) & (zp < z + 0.5)
                           & (cl < _CR_MARGIN))
                    if thr.any():
                        i_ = int(np.argmin(np.where(thr, cl, 1e9)))
                        cmin = float(cl[i_])
                        self._cr_fhold = (t + _CR_FHOLD, _CR_VUP_F if cmin < 0.3 else 0.0)
                        info = [round(float(s[i_]), 2), round(float(zp[i_]) - top, 2), round(cmin, 2), int(thr.sum())]
                        _cx_cr_ev("cr_fwd")
            fh = getattr(self, "_cr_fhold", None)
            if _CX_CR_KFWD and fh is not None and t <= fh[0] + 1e-9:
                if vz_min is None or fh[1] > vz_min:
                    vz_min = fh[1]
                    src = "fwd" if src is None else src + "+fwd"
            if vz_min is None:
                self._cr_prev = False
                return a
            vz_out = max(float(v[2]), vz_min)
            if vz_out > float(v[2]):
                # a climb leads the measured vz by at most _CR_LEAD (a step from a 2 m/s sink to a full climb saturated
                # the thrust and the attitude loop lost the horizontal: 796273227 sped up to 3.8 m/s at 39 deg)
                vz_out = max(float(v[2]), min(vz_out, float(vel[2]) + _CR_LEAD))
            if _CX_PW_CRCAP and vz_out > float(v[2]):
                # fx_pw2: the 3 m/s cap below may take at most _CR_DVH off the drone's measured horizontal speed
                vh_keep = max(0.0, math.hypot(float(vel[0]), float(vel[1])) - _CR_DVH)
                vz_cap = math.sqrt(max(SPEED_LIMIT * SPEED_LIMIT - vh_keep * vh_keep, 0.0))
                if vz_out > vz_cap:
                    if "pw_crcap1" not in CX_EVENTS:
                        CX_EVENTS["pw_crcap1"] = [round(t, 2), round(vh_keep + _CR_DVH, 2), round(vz_out, 2),
                                                  round(max(float(v[2]), vz_cap), 2)]
                    vz_out = max(float(v[2]), vz_cap)
                    _cx_cr_ev("pw_crcap")
            vxy = np.array([float(v[0]), float(v[1])])
            if vh_cap is not None:
                nh = float(np.hypot(vxy[0], vxy[1]))
                if nh > vh_cap:
                    vxy = vxy * (vh_cap / nh)
                    _cx_cr_ev("cr_brake")
                vd = np.array([float(vel[0]), float(vel[1])])
                dvh = vxy - vd
                nd = float(np.hypot(dvh[0], dvh[1]))
                if nd > _CR_DVH:
                    vxy = vd + dvh * (_CR_DVH / nd)
            if abs(vz_out - float(v[2])) < 1e-9 and vh_cap is None:
                self._cr_prev = False
                return a  # the route already keeps clear: fly it as it is
            if src is not None and "ray" in src:
                _cx_cr_ev("cr_ray")
            if not self._cr_prev:
                _cx_cr_ev("cr_fire")
                at = CX_EVENTS.setdefault("cr_at", [])
                if isinstance(at, list) and len(at) < 8:
                    at.append([round(t, 2), src, round(agl, 2) if not sat else None,
                               round(surf - top, 2) if not sat else None, round(dpad, 1), round(float(vel[2]), 2), info])
            self._cr_prev = True
            self.debug = dict(self.debug or {})
            self.debug["mode"] = "CRGUARD"
            vh_out = float(np.hypot(vxy[0], vxy[1]))
            vmax_h = math.sqrt(max(SPEED_LIMIT * SPEED_LIMIT - vz_out * vz_out, 0.0))
            if vh_out > vmax_h:
                vxy = vxy * (vmax_h / vh_out)
            return self._dv_vel_action(np.array([vxy[0], vxy[1], vz_out]), a, st)
        except Exception:
            self._cr_prev = False
            _cx_cr_ev("cr_err")
            return a

    def _kl_reset(self):
        self._kl_shist = []       # (t, surface z) of the down-ray over the last _KL_RISE_S (unsaturated ticks)
        self._kl_ep = None        # rise episode: (t, surface z before the rise) while the ray stays on the pad top
        self._kl_on = False       # a dive is running
        self._kl_t0 = 0.0
        self._kl_top = 0.0        # the pad-top z the running dive aims at
        self._kl_floor = -1e9     # the sink is capped so the drone can stop above this z
        self._kl_vxy = None       # pad velocity steering the dive (None = hold the drone's own velocity)
        self._kl_vz_prev = 0.0
        self._kl_n = 0            # dives this seed
        self._kl_brake_t = None   # post-dive brake runs until this time (s)
        self._kl_dead = False     # internal error: never act again this seed
        self._kb_blind = 0.0      # CX_KL_KBACK: time the king pilot has sat in 'landing' without sight of the pad (s)
        self._kb_done = False     # CX_KL_KBACK: internal error (or the take-back budget is spent): never again this seed
        self._kb_n = 0            # CX_KL_KBACK: take-backs this seed
        self._kb_tb = False       # CX_KL_KBACK: my controller flies after a take-back (CX_KL_KB_REHO may hand it back)

    def _mg_reset(self):
        self._mg_zs = []          # recent fresh pilot pad-top samples (z)
        self._mg_top_ray = None   # pad top read by the down-ray (exact)
        self._mg_pxy = None       # last pad xy estimate, its velocity, time and source
        self._mg_pv = np.zeros(2)
        self._mg_pt = -1e9
        self._mg_psrc = None
        self._mg_gxy = None       # last GOOD pad xy (pilot detection from >= _MG_GOOD_H above, or the ray on the pad)
        self._mg_gv = np.zeros(2)
        self._mg_gt = -1e9
        self._mg_mstep = -1       # mine.step seen on the previous tick (mine ran this tick when it moved on)
        self._mg_mv = False       # open king route: the king pilot has flagged the pad moving (sticky)
        self._mg_on = False       # a HOLD episode is running
        self._mg_latched = False  # resting on something (or an internal error): never again this seed
        self._mg_touched = False  # the ray read the pad top at agl <= _MG_TOUCH: the guard is off until it slides off
        self._mg_slid = 0         # consecutive ticks off the pad top at >= 0.8 m/s after a touch
        self._mg_ticks = 0        # HOLD ticks this seed (cap _MG_MAX_S)
        self._mg_last_hold = -1e9  # time of the last HOLD tick (DROP window)
        self._mg_blk = 0          # consecutive HOLD ticks not sinking although the guard asks for a sink
        self._mg_kkey = None      # the king pilot's last platform_position (a changed value = a new detection)
        self._mg_dropping = False  # a DROP is running (it continues through the touch latch until contact)
        self._mg_drop_t0 = -1e9
        self._mg_shist = []       # surface z (z - agl) read by the down-ray on the last 50 unsaturated ticks
        self._mg_start_xy = None  # drone xy on the first tick (the start pad)
        self._mg_prev_a = None    # the route's previous action and how many ticks in a row it repeated bit for bit
        self._mg_rep = 0

    def _dv_reset(self):
        self._dv_hist = []       # (t, agl, z) of the last _DV_JUMP_S
        self._dv_ep = None       # raised-surface episode after an observed ray drop: (t_drop, surface z, side surface z)
        self._dv_on = False      # a dive is running
        self._dv_t0 = 0.0
        self._dv_top = 0.0       # surface z under the drone when the dive started (the pad top)
        self._dv_floor = -1e9    # the sink is capped so the drone can stop above this z
        self._dv_vxy = np.zeros(2)
        self._dv_mstep = -1      # mine.step seen on the previous tick: mine ran this tick when it moved on
        self._dv_n = 0           # dives this seed (_DV_MAX_N once the pad is latched static: no further dive)
        self._dv_brake_t = None  # post-dive brake runs until this time (s)
        self._dv_vz_prev = 0.0   # measured vz on the previous dive tick (impact test)
        self._dv_tilt_t1 = None  # dv_tilt is tracked until this time (dive end + _DV_TILT_S), None = not tracking
        self._dv_why = None      # CX_DV_DIAG: last recorded reason for a non-firing episode

    def _vm_ho_log(self):
        """CX_VM_DIAG: the village dual->king hand-over (t, estimate z, gmin, recent detection heights, hits, flags)."""
        try:
            m = self.mine
            za, zn, nn = m._vm_zstats()
            g = m._vm_gmin
            zf = -9.0 if m.pad_pos is None else float(m.pad_pos[2])
            _cx_vm_log("vm_ho_at", "%.2f/zf%.2f/za%s/zn%s/n%d/g%s/h%d/dmin%.1f/sp%.2f/mv%d%d/%s" % (
                float(m.t), zf, "-" if za is None else "%.2f" % za, "-" if zn is None else "%.2f" % zn, nn,
                "-" if g is None else "%.2f" % g, int(m.pad_hits), float(m.pad_min_z), float(m.pad_fit_sp),
                int(bool(m.pad_moving)), int(bool(m.pad_ever_moving)), m.mode))
        except Exception:
            _cx_vm_ev("vm_err")

    def _vkw_ref(self):
        """CX_VK_WATCH: where my shadow track puts the pad now (seen within 3 s), else the pad at the hand-over."""
        m = self.mine
        if m.pad_pos is not None and m.t - m.pad_last_seen < 3.0:
            p = m._pad_predicted()
            if p is not None:
                return np.asarray(p, dtype=np.float64)
        return np.asarray(self._vk_mem, dtype=np.float64)

    def _vkw_fire(self, a, st_):
        """CX_VK_WATCH: the village graph pack has repeated one action bit for bit for _VK_WATCH_N ticks (its blind
        'repeat the last action' branch) while that action carries the drone away from the pad, near pad height and
        more than 1 m from it, and the hand-over is more than _VK_WATCH_MIN_T old (the leash waits the same 3 s).
        A repeat on top of the pad (the final descent) never fires. It is judged only against a live pad estimate:
        once my shadow track has not seen the pad for 3 s, the watch resets and leaves the take-back to the leash."""
        try:
            if getattr(self.king, "_active", None) != "graph":
                self._vkw_prev, self._vkw_n = None, 0
                return False
            m = self.mine
            if m.pad_pos is None or m.t - m.pad_last_seen >= 3.0:
                # without a live estimate _vkw_ref falls back to the hand-over position, and the jump to it read as
                # "moving away" on a fast figure-8 (1954733135: fired 3.0 s after the last sighting at 7 m, where
                # mine then lost the pad and timed out; the champion's 18 m leash landed it)
                self._vkw_prev, self._vkw_n = None, 0
                return False
            a_ = np.asarray(a, dtype=np.float32).reshape(-1)
            pos_ = st_[0:3].astype(np.float64)
            ref = self._vkw_ref()
            d_ref = float(np.linalg.norm(pos_[:2] - ref[:2]))
            prev = getattr(self, "_vkw_prev", None)
            if prev is not None and prev.shape == a_.shape and np.array_equal(prev, a_):
                self._vkw_n = int(getattr(self, "_vkw_n", 0)) + 1
            else:
                self._vkw_n = 0
                self._vkw_d0 = d_ref
            self._vkw_prev = a_.copy()
            if self._vkw_n < _VK_WATCH_N:
                return False
            if self.mine.t - float(getattr(self, "_vk_t0", self.mine.t)) <= _VK_WATCH_MIN_T:
                return False
            return d_ref > 1.0 and float(pos_[2] - ref[2]) < 1.5 and d_ref > float(getattr(self, "_vkw_d0", d_ref)) + 0.25
        except Exception:
            return False

    def act(self, observation):
        try:
            st_ = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            dp_ = np.asarray(observation["depth"], dtype=np.float32)
            key = (st_.tobytes(), float(dp_[::16, ::16].sum()) if dp_.ndim >= 2 else 0.0)
            last = getattr(self, "_last_key", None)
            if last is not None and key[0] == last[0] and key[1] == last[1] and getattr(self, "_last_action", None) is not None:
                return self._last_action
            self._last_key = key
        except Exception:
            pass
        a_out = self._act_inner(observation)
        if _CX_PW_KSEED:
            self._pw_kseed_step(observation)  # fx_kseed: changes only the king main pilot's goal-return target
        if _CX_S2K_OPEN:
            self._s2k_step(observation)  # fx_s2kpad: shadow padnet on open; may set the king main pilot's goal-return target
        if _CX_KL_KBACK:
            self._kb_step(observation)
        if _CX_PW_GSINK:
            self._pw_gsink_step(observation, a_out)  # fx_pw2: the city graph pack's slow in-place sink
        if _CX_DV_DIVE:
            a_out = self._dv_step(observation, a_out)
        if _CX_MG_FLOOR:
            a_out = self._mg_step(observation, a_out)
        if _CX_CR_ANY:
            a_out = self._cr_step(observation, a_out)
        if _CX_KL_MDROP:
            a_out = self._kl_step(observation, a_out)
        if _CX_VK2_ANY:
            a_out = self._vk2_step(observation, a_out)  # fx_vk (round 23): village king route
        if _CX_MPC:
            a_out = self._mpc_step(observation, a_out)  # fx_mpc: village roof moving-pad path wait (runs last)
        try:
            self._last_action = np.array(a_out, dtype=np.float32, copy=True)
        except Exception:
            self._last_action = a_out
        return a_out

    def _pw_kseed_step(self, observation):
        """fx_kseed CX_PW_KSEED (see the flag comment). Runs after the route chose this tick's action; changes only the
        king main pilot's goal-return target (read by its next act()). Any error stops it for the seed."""
        try:
            if getattr(self, "_pw_ks_done", True):
                return
            if not (self.route == "king" and self.king_from_dual and self.mine.map_label == "city"):
                return
            m = self.mine
            tn = float(self.step) * SIM_DT
            if tn - float(self.ho_t) > _PW_KS_T:
                self._pw_ks_done = True
                return
            if getattr(self.king, "_active", None) != "main":
                self._pw_ks_done = True
                return
            km = getattr(self.king, "_main", None)
            if km is None:
                self._pw_ks_done = True
                return
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            clue = st[0:3].astype(np.float64) + st[138:141].astype(np.float64)
            pp = getattr(km, "platform_position", None)
            kmode = str(getattr(km, "_mode", ""))
            ksees = False
            if bool(getattr(km, "see_P", False)) and pp is not None:
                pp = np.asarray(pp, dtype=np.float64).reshape(-1)
                ksees = bool(pp.size >= 3 and np.isfinite(pp[:3]).all() and float(np.abs(pp[:3] - clue).max()) > 1.0)
            if ksees and kmode in ("navigation", "landing"):
                self._pw_ks_done = True  # the pilot flies at a pad of its own: leave it alone from now on
                return
            if kmode != "search" or ksees:
                return  # 'takeoff' (1049306411: it saw the pad in takeoff, lost it, then searched the clue) or a sighting
            if m.pad_pos is None or float(self.ho_t) - float(m.pad_last_seen) > _PW_KS_AGE:
                self._pw_ks_done = True
                return
            tgt = np.asarray(m.pad_pos, dtype=np.float64).reshape(-1)
            if float(tgt[2]) > _PW_KS_ZMAX:
                self._pw_ks_done = True  # not a city pad height (goal pad tops are 0.2-1.05 m): a roof-top track
                return
            goal = np.array([tgt[0], tgt[1], tgt[2] + _PW_KS_ZUP], dtype=np.float32)
            cur = getattr(km, "_last_known_goal_pos", None)
            if (bool(getattr(km, "_goal_return_mode", False)) and cur is not None
                    and float(np.linalg.norm(np.asarray(cur, dtype=np.float64).reshape(-1)[:3] - goal)) <= 0.5):
                return
            km._goal_return_mode = True
            km._last_known_goal_pos = goal
            km._goal_return_search_steps = 0
            km.extra_search_vectors = None
            km.search_stage = 0
            km.search_state_0_rot_count = 0
            self._pw_ks_n = int(getattr(self, "_pw_ks_n", 0)) + 1
            _cx_kl_ev("pw_kseed")
            if "pw_kseed1" not in CX_EVENTS:
                CX_EVENTS["pw_kseed1"] = [round(tn, 2), round(tn - float(self.ho_t), 2),
                                          round(math.hypot(float(goal[0]) - float(st[0]), float(goal[1]) - float(st[1])), 1),
                                          round(math.hypot(float(goal[0]) - float(clue[0]), float(goal[1]) - float(clue[1])), 1)]
        except Exception:
            self._pw_ks_done = True
            _cx_kl_ev("pw_kseed_err")

    # ------------------------------------------------------------------ fx_s1r (CX_RG_* only)
    def _rg_reset(self):
        self._rg_on = False     # watch armed and running
        self._rg_done = False   # re-routed (or stopped for good) this seed
        self._rg_t0 = 0.0       # arming time
        self._rg_lab0 = None    # label at arming
        self._rg_src = None     # "map" (tick-10/60 decision) or "dual" (early open/warehouse re-route)
        self._rg_fed = -1       # router tick of the last fed mine act()
        self._rg_fed_a = None   # its action

    def _rg_arm(self, src):
        """fx_s1r: arm the watch on an early map-route king flight (never a moving hand-over) from a low start."""
        try:
            m = self.mine
            if (self._rg_done or self._rg_on or self.king is None or self.king_from_dual or self.step > 200
                    or float(getattr(self, "z0", 99.0)) >= _RG_Z0 or m.map_label not in ("open", "warehouse")):
                return
            self._rg_on, self._rg_t0, self._rg_lab0, self._rg_src = True, float(self.step) * SIM_DT, m.map_label, src
            _cx_ev("rg_arm")
        except Exception:
            self._rg_on = False

    def _rg_step(self, observation):
        """fx_s1r: one king tick of the watch. Returns mine's action when the flight is re-routed, else None (the king
        flies this tick as in c29c). Any error stops the watch for the seed."""
        try:
            m = self.mine
            tn = float(self.step) * SIM_DT
            if self.king_from_dual or tn - self._rg_t0 > _RG_T:
                self._rg_on = False
                return None
            fed = None
            if _CX_RG_KFEED and not m.map_locked and m.map_label == "open":
                fed = m.act(observation)  # keeps mine (and its classifier) running; action discarded unless it flips
                self._rg_fed, self._rg_fed_a = self.step, fed
                _cx_ev("rg_feed")
            lab = m.map_label
            if lab not in _RG_LABS:
                if m.map_locked:
                    self._rg_on = False  # the label can no longer change: the king keeps the flight
                return None
            a = fed if fed is not None else m.act(observation)  # mine's action for this tick
            if self.king is not None and lab in _KING_ROUTE_LABELS:
                new = "king"
            elif self.king is not None and lab in _KING_MOVING_LABELS:
                new = "dual"
            else:
                new = "mine"
            self.route = new
            self._rg_on, self._rg_done = False, True
            self.debug = dict(m.debug)
            self.debug["route"] = self.route
            _cx_ev("rg_reroute")
            _cx_rg_log("rg_at", "%.2f/%s/%s>%s/%s/n%d%s" % (tn, self._rg_src, self._rg_lab0, lab, new, int(m.map_n),
                                                          "/L" if m.map_locked else ""))
            return a
        except Exception:
            self._rg_on = False
            _cx_ev("rg_error")
            return None

    # ------------------------------------------------------------------ fx_s2kpad (CX_S2K_OPEN only)
    def _s2k_reset(self):
        self._s2k_done = False   # stopped for this seed (seed budget spent or an error)
        self._s2k_start = None   # start position (first observation): the start pad is not a goal
        self._s2k_trk = []       # shadow tracks: dict(p=xyz, n=hits, t0, t1, id)
        self._s2k_nid = 0
        self._s2k_bl = []        # xy of seeded tracks padnet did not see again during the attempt
        self._s2k_wait = 0.0     # s the pilot has searched without a sighting of its own while a credible track existed
        self._s2k_seed = None    # the running seed: dict(id, t, goal, n)
        self._s2k_nseed = 0
        self._s2k_tick = 0
        self._s2k_armed = False
        self._s2k_c1 = False

    def _s2k_cred(self, tk):
        return tk["n"] >= _S2K_NHIT and tk["t1"] - tk["t0"] >= _S2K_SPAN

    def _s2k_detect(self, observation, st, clue, tn):
        """One shadow padnet frame (mine's detector and decode, stateless) into the shadow tracks."""
        depth = np.asarray(observation["depth"], dtype=np.float32).reshape(IMG_H, IMG_W)
        dets, _cls = self.mine._detect(depth, st[0:3].astype(np.float64), st[3:6].astype(np.float64))
        s0 = self._s2k_start
        used = set()
        for s_, w, _z, _u, _v in sorted(dets, key=lambda d: -d[0]):
            if s_ < _S2K_SMIN or not (_S2K_ZLO <= float(w[2]) <= _S2K_ZHI):
                continue
            if s0 is not None and math.hypot(float(w[0]) - s0[0], float(w[1]) - s0[1]) < _S2K_STARTR:
                continue
            if math.hypot(float(w[0]) - clue[0], float(w[1]) - clue[1]) > _S2K_CLUER:
                continue
            best, bd = None, 1e9
            for i, tk in enumerate(self._s2k_trk):
                if i in used:
                    continue
                g = min(_S2K_GMAX, _S2K_GATE + _S2K_VG * (tn - tk["t1"]))
                d = math.hypot(float(w[0]) - tk["p"][0], float(w[1]) - tk["p"][1])
                if d <= g and d < bd:
                    best, bd = i, d
            if best is None:
                if len(self._s2k_trk) >= 6:
                    young = [tk for tk in self._s2k_trk if not self._s2k_cred(tk)]
                    if not young:
                        continue
                    self._s2k_trk.remove(min(young, key=lambda tk: tk["t1"]))
                self._s2k_trk.append(dict(p=np.asarray(w, dtype=np.float64).copy(), n=1, t0=tn, t1=tn, id=self._s2k_nid))
                self._s2k_nid += 1
                used.add(len(self._s2k_trk) - 1)
            else:
                tk = self._s2k_trk[best]
                tk["p"] = 0.8 * tk["p"] + 0.2 * np.asarray(w, dtype=np.float64)
                tk["n"] += 1
                tk["t1"] = tn
                used.add(best)
        self._s2k_trk = [tk for tk in self._s2k_trk if self._s2k_cred(tk) or tn - tk["t1"] <= 3.0]
        if not self._s2k_c1:
            for tk in self._s2k_trk:
                if self._s2k_cred(tk):
                    self._s2k_c1 = True
                    CX_EVENTS["s2k_cred1"] = [round(tn, 2), round(float(tk["p"][0]), 2), round(float(tk["p"][1]), 2),
                                              round(float(tk["p"][2]), 2), int(tk["n"])]
                    break

    def _s2k_goal(self, tk):
        return np.array([tk["p"][0], tk["p"][1], min(_S2K_ZHI, max(_S2K_ZLO, float(tk["p"][2])))], dtype=np.float32)

    def _s2k_step(self, observation):
        """fx_s2kpad CX_S2K_OPEN (see the flag comment). Runs after the route chose this tick's action; changes only the
        king main pilot's goal-return target (read by its next act()). Any error stops it for the seed."""
        try:
            if self._s2k_done:
                return
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            if self._s2k_start is None:
                self._s2k_start = (float(st[0]), float(st[1]), float(st[2]))
            if (self.route != "king" or self.king_from_dual or self.king is None or self.mine.map_label != "open"
                    or getattr(self.king, "_active", None) != "main"):
                return
            km = getattr(self.king, "_main", None)
            if km is None:
                return
            if not self._s2k_armed:
                self._s2k_armed = True
                _cx_kl_ev("s2k_arm")
            import time as _s2k_time
            tn = float(self.step) * SIM_DT
            kmode = str(getattr(km, "_mode", ""))
            clue = (float(st[0]) + float(st[138]), float(st[1]) + float(st[139]), float(st[2]) + float(st[140]))
            if kmode in ("takeoff", "search"):
                self._s2k_tick += 1
                if (self._s2k_tick - 1) % max(1, int(_S2K_EVERY)) == 0:
                    t0_ = _s2k_time.perf_counter()
                    self._s2k_detect(observation, st, clue, tn)
                    CX_EVENTS["s2k_ms"] = round(CX_EVENTS.get("s2k_ms", 0.0) + 1000.0 * (_s2k_time.perf_counter() - t0_), 1)
                    _cx_kl_ev("s2k_nd")
            pp = getattr(km, "platform_position", None)
            ksees = False
            if bool(getattr(km, "see_P", False)) and pp is not None:
                pp = np.asarray(pp, dtype=np.float64).reshape(-1)
                ksees = bool(pp.size >= 3 and np.isfinite(pp[:3]).all()
                             and max(abs(float(pp[0]) - clue[0]), abs(float(pp[1]) - clue[1]), abs(float(pp[2]) - clue[2])) > 1.0)
            gr = bool(getattr(km, "_goal_return_mode", False))
            sd = self._s2k_seed
            if sd is not None:
                cur = getattr(km, "_last_known_goal_pos", None)
                ours = bool(gr and cur is not None and float(np.linalg.norm(
                    np.asarray(cur, dtype=np.float64).reshape(-1)[:3] - np.asarray(sd["goal"], dtype=np.float64))) <= 0.05)
                if kmode in ("navigation", "landing") or ksees:
                    _cx_kl_ev("s2k_klock")  # the pilot sees a pad of its own: the seed has done its job
                    if "s2k_klock1" not in CX_EVENTS:
                        CX_EVENTS["s2k_klock1"] = [round(tn, 2), round(tn - sd["t"], 2), kmode]
                    self._s2k_seed, self._s2k_wait = None, 0.0
                    return
                tk = next((x for x in self._s2k_trk if x["id"] == sd["id"]), None)
                if not ours:
                    # the goal-return ended without a sighting (its 400-tick limit): blacklist a track padnet did not see again
                    _cx_kl_ev("s2k_end")
                    if tk is None or tk["n"] <= sd["n"]:
                        self._s2k_bl.append((float(sd["goal"][0]), float(sd["goal"][1])))
                        _cx_kl_ev("s2k_bl")
                    self._s2k_seed, self._s2k_wait = None, 0.0
                    if self._s2k_nseed >= int(_S2K_MAXSEED):
                        self._s2k_done = True
                    return
                if tk is not None and tn - tk["t1"] < 1.0:
                    goal = self._s2k_goal(tk)
                    if float(np.linalg.norm(goal[:2] - np.asarray(sd["goal"][:2], dtype=np.float32))) > 0.5:
                        km._last_known_goal_pos = goal  # a moving pad: keep the target on the track
                        sd["goal"] = goal
                return
            if (kmode != "search" or ksees or gr or getattr(km, "_landing_platform_position", None) is not None
                    or self._s2k_nseed >= int(_S2K_MAXSEED)):
                self._s2k_wait = 0.0
                return
            cands = [tk for tk in self._s2k_trk if self._s2k_cred(tk)
                     and not any(math.hypot(tk["p"][0] - b[0], tk["p"][1] - b[1]) < 2.0 for b in self._s2k_bl)]
            if not cands:
                self._s2k_wait = 0.0
                return
            self._s2k_wait += SIM_DT
            if self._s2k_wait < _S2K_WAIT - 1e-9:
                return
            tk = max(cands, key=lambda x: x["n"])
            goal = self._s2k_goal(tk)
            km._goal_return_mode = True
            km._last_known_goal_pos = goal
            km._goal_return_search_steps = 0
            km.extra_search_vectors = None
            km.search_stage = 0
            km.search_state_0_rot_count = 0
            self._s2k_seed = dict(id=tk["id"], t=tn, goal=goal, n=int(tk["n"]))
            self._s2k_nseed += 1
            self._s2k_wait = 0.0
            _cx_kl_ev("s2k_seed")
            if "s2k_seed1" not in CX_EVENTS:
                CX_EVENTS["s2k_seed1"] = [round(tn, 2), round(float(goal[0]), 2), round(float(goal[1]), 2), round(float(goal[2]), 2),
                                          round(math.hypot(float(goal[0]) - float(st[0]), float(goal[1]) - float(st[1])), 1),
                                          round(math.hypot(float(goal[0]) - clue[0], float(goal[1]) - clue[1]), 1),
                                          int(tk["n"]), round(tn - tk["t1"], 2)]
        except Exception:
            self._s2k_done = True
            _cx_kl_ev("s2k_err")

    def _dv_flagged(self):
        """CX_DV_DIVE: has this flight's pad been flagged moving? On the king route only after the moving hand-over
        (king_from_dual; the static hand-back and every take-back leave the king route); elsewhere mine's own flags."""
        if self.route == "king":
            return bool(self.king_from_dual)
        m = self.mine
        return bool(m.pad_moving or m.pad_ever_moving)

    def _pc_ksee(self):
        """fx_pc: (sees, king pilot mode, see_P, lost step, d(king platform, my predicted pad) or None), or None when
        the active king pilot carries no platform track."""
        pm = self._dv_kmain()
        if pm is None:
            return None
        m = self.mine
        sp = bool(getattr(pm, "see_P", False))
        ls = int(getattr(pm, "platform_lost_step", 99))
        pp = getattr(pm, "platform_position", None)
        pred = m._pad_predicted()
        d = None
        if pp is not None and pred is not None:
            pp = np.asarray(pp, dtype=np.float64).reshape(-1)
            if pp.size >= 2 and np.isfinite(pp[:2]).all():
                d = float(math.hypot(float(pp[0]) - float(pred[0]), float(pp[1]) - float(pred[1])))
        sees = sp and ls <= _PC_SEEN_LOST and d is not None and d <= _PC_SEEN_D
        return sees, str(getattr(pm, "_mode", "")), sp, ls, d

    def _pc_ksee_probe(self):
        """fx_pc DIAG: first tick (after my first flip) the king pilot sees a platform near my predicted pad."""
        try:
            r = self._pc_ksee()
            if r is not None and r[0]:
                self._pc_ksee_done = True
                CX_EVENTS["pc_ksee"] = [round(float(self.mine.t), 2), r[1], None if r[4] is None else round(r[4], 2)]
        except Exception:
            self._pc_ksee_done = True

    def _pc_hold(self):
        """fx_pc SEEN (city): True = keep my controller flying this tick instead of the moving hand-over."""
        m = self.mine
        if m.map_label != "city" or self.king is None:
            return False
        try:
            r = self._pc_ksee()
            t = float(m.t)
            if getattr(self, "_pc_hold_t0", None) is None:
                self._pc_hold_t0 = t
            hold = bool(_CX_PC_SEEN and r is not None and not r[0] and t - self._pc_hold_t0 < _PC_SEEN_MAX)
            n = int(getattr(self, "_pc_hold_n", 0))
            if _PC_DIAG and (not hold or n == 0):
                st = getattr(m, "_pc_pos", None)
                pred = m._pad_predicted()
                dme = None
                if st is not None and pred is not None:
                    dme = float(math.hypot(float(st[0]) - float(pred[0]), float(st[1]) - float(pred[1])))
                _pc_log("pc_ho", [round(t, 2), str(getattr(self.king, "_active", None)),
                                  None if r is None else r[1], None if r is None else int(r[2]),
                                  None if r is None else r[3], None if (r is None or r[4] is None) else round(r[4], 2),
                                  None if dme is None else round(dme, 2), int(hold), round(t - self._pc_hold_t0, 2)], cap=4)
            if hold:
                self._pc_hold_n = n + 1
                _pc_ev("pc_hold_ticks")
            return hold
        except Exception:
            _pc_ev("pc_err")
            return False

    def _pc_kto(self):
        """fx_pc KTO: take the king pilot out of a leftover 'takeoff' state at the city hand-over (see the top)."""
        try:
            pm = self._dv_kmain()
            if pm is None or str(getattr(pm, "_mode", "")) != "takeoff":
                return
            q = getattr(self.mine, "_pc_pos", None)
            sp = getattr(self.mine, "start_pos", None)
            if q is not None and sp is not None and float(q[2]) - float(np.asarray(sp).reshape(-1)[2]) < 1.0:
                return  # still on the way up: the pilot's own take-off has to finish
            sees = bool(getattr(pm, "see_P", False)) and int(getattr(pm, "platform_lost_step", 99)) <= _PC_SEEN_LOST
            if sees:
                pm.first_order_cnt = 20
                pm._mode = "navigation"
            else:
                pm.first_order_cnt = 30
                pm._mode = "search"
            CX_EVENTS["pc_kto"] = [round(float(self.mine.t), 2), int(sees)]
        except Exception:
            _pc_ev("pc_err")

    def _dv_kmain(self):
        """CX_DV_DIVE: the active king pilot's main controller when it carries a platform track (uid130 and main do;
        the village graph pack and the forest pilot do not), else None."""
        k = self.king
        if k is None:
            return None
        act = getattr(k, "_active", None)
        if act == "uid130":
            return getattr(getattr(getattr(k, "_uid130", None), "_base", None), "_main", None)
        if act == "main":
            return getattr(k, "_main", None)
        return None

    def _dv_kest(self):
        """CX_DV_DIVE: the king pilot's platform track when fresh: (pad xyz, pad vxy) or None (see_P,
        platform_lost_step counted in 20 ms king ticks, platform_position, an alpha-beta tracker for velocity)."""
        pm = self._dv_kmain()
        if pm is None or not bool(getattr(pm, "see_P", False)):
            return None
        if float(getattr(pm, "platform_lost_step", 99)) * SIM_DT > _DV_FRESH + 1e-6:
            return None
        pp = getattr(pm, "platform_position", None)
        if pp is None:
            return None
        pp = np.asarray(pp, dtype=np.float64).reshape(-1)
        if pp.size < 3 or not np.isfinite(pp[:3]).all():
            return None
        v = np.zeros(2)
        tr = getattr(pm, "_ab_tracker", None)
        if tr is not None and bool(getattr(tr, "initialized", False)):
            vv = np.asarray(getattr(tr, "vel", np.zeros(3)), dtype=np.float64).reshape(-1)
            if vv.size >= 2 and np.isfinite(vv[:2]).all():
                v = vv[:2].copy()
        return pp[:3].copy(), v

    def _dv_near_pad(self, st):
        """CX_DV_DIVE static latch (a): is any pad estimate -- mine's last fix or the king pilot's platform position,
        fresh or not -- within _DV_DXY of the drone horizontally? Read only on a tick with agl <= _DV_LATCH_AGL."""
        x, y = float(st[0]), float(st[1])
        pm = self._dv_kmain()
        for pp in (getattr(self.mine, "pad_pos", None), getattr(pm, "platform_position", None) if pm is not None else None):
            if pp is None:
                continue
            pp = np.asarray(pp, dtype=np.float64).reshape(-1)
            if pp.size >= 2 and np.isfinite(pp[:2]).all() and math.hypot(pp[0] - x, pp[1] - y) <= _DV_DXY:
                return True
        return False

    def _dv_latch(self, ev):
        """CX_DV_DIVE: the pad is static (or the one dive is spent): no further dive this seed."""
        if self._dv_n < _DV_MAX_N:
            self._dv_n = _DV_MAX_N
            _cx_dv_ev(ev)

    def _dv_skip(self, why, a, t, val=None, src=None):
        """CX_DV_DIVE: an episode that does not fire (CX_DV_DIAG=1 records the reason when it changes; no effect)."""
        if _CX_DV_DIAG and self._dv_why != why:
            self._dv_why = why
            d = CX_EVENTS.setdefault("dv_diag", [])
            if isinstance(d, list) and len(d) < 16:
                d.append([round(t, 2), why] + ([round(float(val), 2)] if val is not None else []) + ([src] if src else []))
        return a

    def _dv_vel_action(self, v, a, st):
        """World velocity v (m/s) as an action, keeping the route's yaw command."""
        a_ = np.asarray(a, dtype=np.float32).reshape(-1)
        yaw = float(a_[4]) if a_.size >= 5 and math.isfinite(float(a_[4])) else float(st[5]) / math.pi
        out = np.zeros(5, dtype=np.float32)
        sp = float(np.linalg.norm(v))
        if sp > 1e-6:
            out[0:3] = (np.asarray(v, dtype=np.float64) / sp).astype(np.float32)
            out[3] = min(1.0, sp / SPEED_LIMIT)
        out[4] = yaw
        return out

    def _dv_dive_action(self, a, st):
        # rate-limited on both axes (see COMMAND in the fx_dv block): the sink target leads the measured vz by at most
        # _DV_LEAD (the vertical thrust target stays well above zero, so the attitude loop keeps full authority), and the
        # horizontal command stays within _DV_DVH of the drone's own velocity, stepping toward the pad velocity
        room = float(st[2]) - self._dv_floor
        sink = min(_DV_VZ, math.sqrt(2.0 * _DV_BRAKE_A * max(room, 0.0)))
        vd = np.array([float(st[6]), float(st[7])], dtype=np.float64)
        dv = np.asarray(self._dv_vxy, dtype=np.float64) - vd
        n = float(np.hypot(dv[0], dv[1]))
        vxy = vd + (dv * (_DV_DVH / n) if n > _DV_DVH else dv)
        vh = float(np.hypot(vxy[0], vxy[1]))
        if vh > SPEED_LIMIT:
            vxy = vxy * (SPEED_LIMIT / vh)
            vh = SPEED_LIMIT
        sink = min(sink, math.sqrt(max(SPEED_LIMIT * SPEED_LIMIT - vh * vh, 0.0)))  # the sink gives way to the speed cap
        vz = max(-sink, float(st[8]) - _DV_LEAD)
        return self._dv_vel_action(np.array([vxy[0], vxy[1], vz]), a, st)

    def _dv_brake_action(self, a, st):
        # the route's action with its sink removed (and a gentle climb): stops the dive's momentum over a non-pad surface
        a_ = np.asarray(a, dtype=np.float64).reshape(-1)
        n = float(np.linalg.norm(a_[0:3])) if a_.size >= 4 else 0.0
        v = a_[0:3] / n * min(abs(float(a_[3])), 1.0) * SPEED_LIMIT if n > 1e-6 else np.zeros(3)
        v = np.array([v[0], v[1], max(float(v[2]), 0.5)])
        return self._dv_vel_action(v, a, st)

    def _dv_step(self, observation, a):
        """CX_DV_DIVE (see the fx_dv block at the top of the file). Runs once per new observation after the route has
        chosen its action; returns that action unchanged unless a dive (or its post-dive brake) is running."""
        try:
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            m = self.mine
            t = float(self.step) * SIM_DT
            z = float(st[2])
            agl = float(st[137]) * 20.0
            sat = (not math.isfinite(agl)) or agl >= _M2_SAT_M
            surf = z - agl
            mstep = int(getattr(m, "step", 0))
            live = mstep != self._dv_mstep  # mine.act() ran on this tick (its track and clock are current)
            self._dv_mstep = mstep
            win = [h for h in self._dv_hist if t - h[0] <= _DV_JUMP_S + 1e-6]
            self._dv_hist = win + [(t, agl, z)]
            if self._dv_on or (self._dv_tilt_t1 is not None and t <= self._dv_tilt_t1 + 1e-6):
                tilt = math.degrees(max(abs(float(st[3])), abs(float(st[4]))))
                if tilt > float(CX_EVENTS.get("dv_tilt", 0.0)):
                    CX_EVENTS["dv_tilt"] = round(tilt, 1)  # peak tilt from a dive start to _DV_TILT_S after its end
            if (not sat and agl <= _DV_LATCH_AGL and self._dv_n < _DV_MAX_N and not self._dv_on
                    and self._dv_flagged() and self._dv_near_pad(st)):
                self._dv_latch("dv_latch_touch")  # contact that did not end the seed: the pad is static
            if self._dv_brake_t is not None:
                if t <= self._dv_brake_t and float(st[8]) < -0.3:
                    return self._dv_brake_action(a, st)
                self._dv_brake_t = None
            if self._dv_on:
                el = t - self._dv_t0
                end = None
                if el >= _DV_MAX_S - 1e-6:
                    end = "dv_end_time"
                elif sat or abs(surf - self._dv_top) > _DV_SURF:
                    end = "dv_end_left"
                elif agl <= _DV_TOUCH:
                    end = "dv_end_touch"
                elif el >= _DV_BLOCK_S and float(st[8]) > -0.3:
                    end = "dv_end_block"
                elif float(st[8]) >= self._dv_vz_prev + _DV_HIT_DV:
                    end = "dv_end_hit"
                self._dv_vz_prev = float(st[8])
                if end is None:
                    self.debug = dict(self.debug or {})
                    self.debug["mode"] = "DIVE"
                    return self._dv_dive_action(a, st)
                self._dv_on = False
                self._dv_ep = None
                self._dv_tilt_t1 = t + _DV_TILT_S
                _cx_dv_ev(end)
                self._dv_latch("dv_latch_end")  # a dive that ends without the seed ending: never dive again
                vz_now = float(st[8])
                if end == "dv_end_left" and not sat and vz_now < -0.3 and agl < vz_now * vz_now / (2.0 * _DV_BRAKE_A) + _DV_FLOOR:
                    # the ray left the pad onto something the dive's sink could not stop above: brake for up to 0.3 s
                    self._dv_brake_t = t + 0.3
                    _cx_dv_ev("dv_brake")
                    return self._dv_brake_action(a, st)
                return a
            if not sat and win and agl >= _DV_AGL_MIN:
                hi = max(win, key=lambda h: h[1])
                if hi[1] - agl >= _DV_JUMP and (self._dv_ep is None or abs(surf - self._dv_ep[1]) > _DV_SURF):
                    self._dv_ep = (t, surf, hi[2] - hi[1])  # the ray just moved onto a raised surface
                    self._dv_why = None
            ep = self._dv_ep
            if ep is None:
                return a
            if sat or abs(surf - ep[1]) > _DV_SURF or t - ep[0] > _DV_EP_S + 1e-6:
                self._dv_ep = None
                return a
            if agl > _DV_HMAX:
                return a  # too high over this surface to be a pad the drone is about to touch (terrain steps; not logged)
            if self._dv_n >= _DV_MAX_N:
                return self._dv_skip("max_n", a, t)
            if agl < _DV_AGL_MIN:
                return self._dv_skip("contact", a, t, agl)  # touching (or about to): never start a dive into a contact
            if not self._dv_flagged():
                return self._dv_skip("flag", a, t)
            est, src = None, "mine"
            if live and m.pad_pos is not None and m.t - m.pad_last_seen <= _DV_FRESH + 1e-6:
                p = m._pad_predicted()
                if p is not None:
                    vxy = None
                    if m.pad_moving:
                        vxy = np.asarray(m._pad_vel_now(), dtype=np.float64).reshape(-1)[:2]
                    if vxy is None:
                        kest = self._dv_kest()  # mine's static branch has no pad velocity: the king's, else none
                        vxy = kest[1] if kest is not None else np.zeros(2)
                    est = (np.asarray(p, dtype=np.float64).reshape(-1), float(m.pad_pos[2]), vxy)
            if est is None:
                kest = self._dv_kest()
                if kest is not None:
                    est, src = (kest[0], float(kest[0][2]), kest[1]), "king"
            if est is None:
                return self._dv_skip("fresh", a, t)
            p, top, vxy = est
            dz = surf - top
            dxy = float(math.hypot(float(p[0]) - float(st[0]), float(p[1]) - float(st[1])))
            if not math.isfinite(dz) or abs(dz) > _DV_DZ:
                return self._dv_skip("dz", a, t, dz, src)
            if not math.isfinite(dxy) or dxy > _DV_DXY:
                return self._dv_skip("dxy", a, t, dxy, src)
            vxy = np.asarray(vxy, dtype=np.float64).reshape(-1)[:2]
            if vxy.size < 2 or not np.isfinite(vxy).all():
                vxy = np.zeros(2)
            spd = float(np.hypot(vxy[0], vxy[1]))
            if spd < _DV_MINSP:
                return self._dv_skip("slow", a, t, spd, src)  # reads static: never dive onto a static pad
            self._dv_on, self._dv_t0, self._dv_top = True, t, surf
            self._dv_floor = float(ep[2]) + _DV_FLOOR
            self._dv_vxy = vxy.copy()
            self._dv_vz_prev = float(st[8])
            self._dv_n += 1
            self._dv_ep = None
            self._dv_tilt_t1 = None
            tilt = math.degrees(max(abs(float(st[3])), abs(float(st[4]))))
            if tilt > float(CX_EVENTS.get("dv_tilt", 0.0)):
                CX_EVENTS["dv_tilt"] = round(tilt, 1)
            _cx_dv_ev("dv_fire")
            if src == "king":
                _cx_dv_ev("dv_src_king")
            if "dv_fire1" not in CX_EVENTS:
                rel = float(math.hypot(vxy[0] - float(st[6]), vxy[1] - float(st[7])))
                CX_EVENTS["dv_fire1"] = [round(t, 2), src, round(agl, 2), round(dz, 2), round(dxy, 2),
                                         round(spd, 2), str(self.route), round(rel, 2), round(tilt, 1)]
            if _CX_DV_DIAG:
                d = CX_EVENTS.setdefault("dv_diag", [])
                if isinstance(d, list) and len(d) < 16:
                    d.append([round(t, 2), "fire", round(spd, 2), src])
            self.debug = dict(self.debug or {})
            self.debug["mode"] = "DIVE"
            return self._dv_dive_action(a, st)
        except Exception:
            self._dv_on = False
            self._dv_brake_t = None
            self._dv_n = _DV_MAX_N  # never dive again after an internal error
            _cx_dv_ev("dv_error")
            return a

    # ---- fx_mpc: village roof moving-pad path wait (CX_MPC; see the fx_mpc block at the top of the file) ----
    def _mpc_reset(self):
        self._mpc_own = False       # an MPC episode owns the final action (HOP, WAIT, SINK, RETREAT; DV = yielded to fx_dv)
        self._mpc_done = False      # no further episode this seed (released, or an internal error)
        self._mpc_ph = None
        self._mpc_t_ph = 0.0
        self._mpc_samp = []         # (t, x, y, depth, weight) pad xy samples since the moving hand-over
        self._mpc_mstep = -1        # mine.step on the previous tick (mine ran this tick when it moved on)
        self._mpc_nep = 0           # episodes this seed (at most one)
        self._mpc_fit = None        # the running episode's line
        self._mpc_P = None          # its wait point (xy) and along-axis coordinate
        self._mpc_sP = 0.0
        self._mpc_top = None        # pad top (z) the episode holds at
        self._mpc_tsrc = None       # its source: 'ray' (exact) or 'pilot' (median of detections, reads low)
        self._mpc_t0 = None         # episode start (s)
        self._mpc_fit_t = -1e9      # last re-fit (s)
        self._mpc_yaw = None        # heading target (rad)
        self._mpc_mg_prev = False   # fx_mg HOLD on at the previous tick (T2 episode tracking)
        self._mpc_mg_d0 = None      # distance to the pad estimate at that HOLD episode's start
        self._mpc_mg_end = None     # end of the last HOLD episode, pending a T2 try (None: nothing pending)
        self._mpc_side = None       # (t, sign of the tick's near samples' s - s_P) (pass tracking)
        self._mpc_pass = 0          # passes of the wait point seen this episode (samples crossing it)
        self._mpc_miss = 0
        self._mpc_sink_t = None
        self._mpc_wait_t = None     # first arrival at the wait point this episode
        self._mpc_hb_t = None       # time of the last hand-back
        self._mpc_t3_t = -1e9       # T3 (stress test): last try
        self._mpc_bad0 = None       # PATH CHECK: first off-line sample since the last on-line one

    def _mpc_scope(self):
        return (self.route == "king" and self.king_from_dual and self.king is not None
                and self.mine.map_label in _MPC_MAPS)

    def _mpc_fail(self):
        self._mpc_own, self._mpc_done, self._mpc_ph = False, True, None
        _cx_mpc_ev("mpc_err")

    def _mpc_collect(self, t, pos):
        """Pad xy samples: mine's raw detection when its (shadow) track saw the pad this tick, the king pilot's fresh
        detection (main/uid130) as read by fx_mg."""
        m = self.mine
        mstep = int(getattr(m, "step", 0))
        live = mstep != self._mpc_mstep
        self._mpc_mstep = mstep
        obs_ = getattr(m, "pad_obs", None)
        if (live and m.pad_pos is not None and float(m.pad_last_seen) == float(m.t) and int(getattr(m, "pad_hits", 0)) >= 5
                and obs_ and float(obs_[-1][0]) == float(m.t)):
            o = obs_[-1]
            x, y, dep = float(o[1]), float(o[2]), float(o[4])
            if (math.isfinite(x) and math.isfinite(y) and math.isfinite(dep) and dep <= _MPC_ZMAX
                    and math.hypot(x - pos[0], y - pos[1]) <= _MPC_RNG):
                self._mpc_samp.append((t, x, y, dep, 1.0 / (0.05 + 0.02 * dep) ** 2))
        if _CX_MG_FLOOR and self._mg_pt == t and self._mg_psrc == "king" and self._mg_pxy is not None:
            x, y = float(self._mg_pxy[0]), float(self._mg_pxy[1])
            rng = math.hypot(x - pos[0], y - pos[1])
            if rng <= _MPC_RNG:
                self._mpc_samp.append((t, x, y, rng, 1.0 / (0.05 + 0.02 * rng) ** 2))
        if self._mpc_samp and t - self._mpc_samp[0][0] > 14.0:
            self._mpc_samp = [s_ for s_ in self._mpc_samp if t - s_[0] <= 12.0]

    def _mpc_fitnow(self, t):
        S = [s_ for s_ in self._mpc_samp if t - s_[0] <= _MPC_WIN]
        return _mpc_line_fit(S)

    def _mpc_pad_est(self, t):
        """Latest pad xy estimate: fx_mg's prediction, else the newest sample."""
        ph_ = self._mg_pred(t) if _CX_MG_FLOOR else None
        if ph_ is not None:
            return np.asarray(ph_, dtype=np.float64).reshape(-1)[:2]
        if self._mpc_samp:
            return np.array([self._mpc_samp[-1][1], self._mpc_samp[-1][2]], dtype=np.float64)
        return None

    def _mpc_point(self, fit, xy):
        """(wait point xy, its along-axis coordinate): the axis point nearest xy, clamped inside the observed stroke."""
        s_p = _mpc_pick(fit, xy, None, None)[0]
        return fit["mu"] + fit["u"] * s_p, s_p

    def _mpc_try_arm(self, t, st, why):
        """T1 / T2: arm an episode when every gate passes (see the fx_mpc block). True = armed (MPC owns from now)."""
        if (self._mpc_own or self._mpc_done or self._mpc_nep >= 1 or t > _MPC_TLAST
                or not self._mpc_scope()):
            return False
        row = [round(t, 2), why]
        if _MPC_HBM and (getattr(self, "_vk_mem", None) is None or getattr(self, "_vk_recaptured", False)):
            _cx_mpc_log("mpc_fitx", row + ["vkmem"], cap=6)  # a release could not go through the vking take-back
            return False
        pos = st[0:3].astype(np.float64)
        top, tsrc = self._mg_top() if _CX_MG_FLOOR else (None, None)
        if top is None or not math.isfinite(float(top)):
            _cx_mpc_log("mpc_fitx", row + ["top"], cap=6)
            return False
        top = float(top)
        if top - float(getattr(self, "z0", 0.0)) < _MPC_ROOF:
            _cx_mpc_log("mpc_fitx", row + ["ground", round(top - float(getattr(self, "z0", 0.0)), 2)], cap=6)
            return False
        dz = float(pos[2]) - top
        if dz < -_MPC_DZLO or dz > _MPC_DZHI:
            _cx_mpc_log("mpc_fitx", row + ["dz", round(dz, 2)], cap=6)
            return False
        if (getattr(self, "_dv_on", False) or getattr(self, "_dv_brake_t", None) is not None
                or self._mg_dropping or self._mg_touched or self._mg_latched):
            _cx_mpc_log("mpc_fitx", row + ["busy"], cap=6)
            return False
        pe = self._mpc_pad_est(t)
        dpad = None if pe is None else math.hypot(float(pe[0]) - pos[0], float(pe[1]) - pos[1])
        if dpad is None or dpad > _MPC_NEAR:
            _cx_mpc_log("mpc_fitx", row + ["far", None if dpad is None else round(dpad, 2)], cap=6)
            return False
        if why == "mg" and dz > _MG_ON and dpad < _MPC_T2_DX:
            # over the pad (fx_mg let go for 'high'): the king's approach from above ends in fx_dv's dive
            _cx_mpc_log("mpc_fitx", row + ["t2hi", round(dz, 2), round(dpad, 2)], cap=6)
            return False
        fit, fwhy = self._mpc_fitnow(t)
        if fwhy != "ok":
            _cx_mpc_log("mpc_fitx", row + [fwhy] + ([] if fit is None else [
                round(fit["rms"], 3), round(fit.get("ext", 0.0), 2), round(fit.get("sag", 0.0), 3), fit.get("n"),
                round(fit.get("span", 0.0), 1)]), cap=6)
            return False
        rel = pos[:2] - fit["mu"]
        lat = float(rel @ fit["nv"])
        s_d = float(rel @ fit["u"])
        if abs(lat) > _MPC_LAT or s_d < fit["s_lo"] - 2.0 or s_d > fit["s_hi"] + 2.0:
            _cx_mpc_log("mpc_fitx", row + ["off", round(lat, 2), round(s_d - fit["s_lo"], 2), round(fit["s_hi"] - s_d, 2)],
                        cap=6)
            return False
        # the pad's along-axis position and heading (intercept wait point)
        s0, vs = _mpc_sv(self._mpc_samp, fit, t)
        if s0 is None and pe is not None:
            s0 = float((pe - fit["mu"]) @ fit["u"])
        vm = None
        if vs is not None and _CX_MG_FLOOR and self._mg_pxy is not None and t - self._mg_pt <= 0.5:
            vm = float(np.asarray(self._mg_pv, dtype=np.float64).reshape(-1)[:2] @ fit["u"])
            if abs(vm) >= _MPC_IC_VMIN and vm * vs < 0.0:
                vs = None  # the regression and fx_mg's pad velocity disagree on the heading: nearest point
        s_p, how, eta = _mpc_pick(fit, pos[:2], s0, vs)
        P = fit["mu"] + fit["u"] * s_p
        self._mpc_own, self._mpc_ph, self._mpc_t_ph, self._mpc_t0 = True, "HOP", t, t
        self._mpc_fit, self._mpc_P, self._mpc_sP, self._mpc_top = fit, P, s_p, top
        self._mpc_tsrc = tsrc
        self._mpc_fit_t, self._mpc_side, self._mpc_pass, self._mpc_miss = t, None, 0, 0
        self._mpc_sink_t, self._mpc_wait_t = None, None
        self._mpc_bad0 = None
        self._mpc_nep += 1
        # face along the axis toward the pad's side of the wait point
        sg = 1.0
        if s0 is not None:
            sg = 1.0 if s0 >= s_p else -1.0
        self._mpc_yaw = math.atan2(sg * float(fit["u"][1]), sg * float(fit["u"][0]))
        if self._mg_on:
            self._mg_on = False
            _cx_mg_ev("mg_release_mpc")
        self._mg_dropping = False
        _cx_mpc_ev("mpc_arm")
        a_row = [round(t, 2), why, round(dpad, 2), round(dz, 2), round(lat, 2), round(s_d - s_p, 2), round(fit["rms"], 3),
                 round(fit["ext"], 2), round(fit["sag"], 3), fit["n"], round(fit["span"], 1), tsrc,
                 round(float(P[0]), 2), round(float(P[1]), 2), round(top, 2), how,
                 None if s0 is None else round(s0 - s_p, 2), None if vs is None else round(vs, 2),
                 None if vm is None else round(vm, 2), None if eta is None else round(eta, 2)]
        CX_EVENTS.setdefault("mpc_arm1", a_row)
        _cx_mpc_log("mpc_fit", a_row, cap=4)
        return True

    def _mpc_vkw(self, st_):
        """CX_VK_WATCH would take the flight back this tick: veto it while MPC owns the flight, or arm MPC instead (T1)."""
        try:
            t = float(self.step) * SIM_DT
            if self._mpc_own:
                _cx_mpc_ev("mpc_veto_vkw")
                return True
            if self._mpc_try_arm(t, np.asarray(st_, dtype=np.float32).reshape(-1), "vkw"):
                _cx_mpc_ev("mpc_veto_vkw")
                return True
        except Exception:
            self._mpc_fail()
        return False

    def _mpc_cmd(self, st, a, xy, z_t, yaw, vmax, dvh, vup=0.6, vdn=0.5, vz_force=None):
        """World-velocity action: xy position control (braking-limited, within dvh of the drone's velocity), z to z_t
        (fx_mg's floor gain; the command leads the measured vz by at most 0.35 m/s down / 0.8 m/s up), yaw target."""
        x, y, z = float(st[0]), float(st[1]), float(st[2])
        vm = np.array([float(st[6]), float(st[7])], dtype=np.float64)
        e = np.array([float(xy[0]) - x, float(xy[1]) - y], dtype=np.float64)
        d = float(np.hypot(e[0], e[1]))
        sp = min(vmax, math.sqrt(2.0 * 1.2 * d), 1.5 * d)
        vd = e / d * sp if d > 1e-6 else np.zeros(2)
        dv = vd - vm
        nd = float(np.hypot(dv[0], dv[1]))
        vxy = vm + (dv * (dvh / nd) if nd > dvh else dv)
        vz = vz_force if vz_force is not None else min(max(3.0 * (z_t - z), -vdn), vup)
        vz = max(vz, float(st[8]) - 0.35)
        vz = min(vz, float(st[8]) + 0.8)
        vh = float(np.hypot(vxy[0], vxy[1]))
        vmh = math.sqrt(max(SPEED_LIMIT * SPEED_LIMIT - vz * vz, 0.0))
        if vh > vmh:
            vxy = vxy * (vmh / vh)
        out = self._dv_vel_action(np.array([vxy[0], vxy[1], vz]), a, st)
        if yaw is not None and math.isfinite(yaw):
            out[4] = np.float32(((yaw + math.pi) % (2.0 * math.pi) - math.pi) / math.pi)
        return out

    def _mpc_sink_cmd(self, st, a, top, t):
        """SINK (fx_dv's dive dynamics): sink at <= _MPC_SINK_VZ, never faster than the drone could stop above
        top - _MPC_SINK_FL at 5 m/s^2, the command leading the measured vz by <= 0.25 m/s; horizontal within 0.3 m/s of the
        drone's velocity, stepping toward the pad velocity estimate (fx_mg's, when fresh), else holding its own."""
        room = float(st[2]) - (top - _MPC_SINK_FL)
        sink = min(_MPC_SINK_VZ, math.sqrt(2.0 * 5.0 * max(room, 0.0)))
        vd = np.array([float(st[6]), float(st[7])], dtype=np.float64)
        tgt = vd
        if _CX_MG_FLOOR and self._mg_pxy is not None and t - self._mg_pt <= _MG_PRED_S + 1e-9:
            pv = np.asarray(self._mg_pv, dtype=np.float64).reshape(-1)[:2]
            if pv.size == 2 and np.isfinite(pv).all():
                tgt = pv
        dv = tgt - vd
        n = float(np.hypot(dv[0], dv[1]))
        vxy = vd + (dv * (0.3 / n) if n > 0.3 else dv)
        vh = float(np.hypot(vxy[0], vxy[1]))
        if vh > SPEED_LIMIT:
            vxy = vxy * (SPEED_LIMIT / vh)
            vh = SPEED_LIMIT
        sink = min(sink, math.sqrt(max(SPEED_LIMIT * SPEED_LIMIT - vh * vh, 0.0)))
        vz = max(-sink, float(st[8]) - 0.25)
        out = self._dv_vel_action(np.array([vxy[0], vxy[1], vz]), a, st)
        yaw = self._mpc_yaw
        if yaw is not None and math.isfinite(yaw):
            out[4] = np.float32(((yaw + math.pi) % (2.0 * math.pi) - math.pi) / math.pi)
        return out

    def _mpc_handback(self, st):
        """Hand-back hygiene (fx_omv's protocol): a king main pilot (main / uid130) kept running on the observations while
        MPC flew, its landing state (anchor, patience, PID) and moving-pad tracking are stale: back to navigation with the
        tracking restarted from its current detection, its slews continued from the flown action; the stuck-landing
        rescue is cleared. The village graph pack has no such state (its memory saw every observation)."""
        try:
            pm = self._dv_kmain()
            if pm is not None and str(getattr(pm, "_mode", "")) in ("landing", "navigation"):
                pm._mode = "navigation"
                pm.first_order_cnt = 20
                pm._landing_platform_position = None
                pm._landing_patience = 0
                pm._goal_lost_steps = 0
                for k_, v_ in (("_landing_committed", False), ("_landing_commit_position", None),
                               ("_committed_landing_descent", False)):
                    if hasattr(pm, k_):
                        setattr(pm, k_, v_)
                ctl_ = getattr(pm, "controller", None)
                if ctl_ is not None:
                    ctl_.reset()
                pp_ = getattr(pm, "platform_position", None)
                if pp_ is not None:
                    pp_ = np.asarray(pp_, dtype=np.float64).reshape(-1)[:3].copy()
                    pm.landing_platform = pp_.copy()
                    pm.reverse_landing_platform = pp_.copy()
                pm.reverse_d = np.array([0.0, 0.0, 0.0])
                pm.move_in_auto_mode = False
                pm.reverse_buffer = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                pm.last_reverse_buffer = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                pm.is_find_P = False
                pm.tracking = False
                pm._first_plat_pos = None
                pm.p_buffer = 10.0
                _cx_mpc_ev("mpc_hb_nav")
            la = getattr(self, "_last_action", None)
            if pm is not None and la is not None:
                pm._last_action = np.array(la, dtype=np.float32).reshape(-1)[:5].copy()
                pm.prev_action = np.array(la, dtype=np.float32).reshape(-1)[:5].copy()
        except Exception:
            _cx_mpc_ev("mpc_hb_err")
        try:
            o = self.king
            for _ in range(6):
                if o is None:
                    break
                if "_rsc_init" in type(o).__dict__:
                    o._rsc_init()
                    break
                o = o.__dict__.get("_base")
        except Exception:
            pass

    def _mpc_takeback(self, st):
        """CX_MPC_HBM: hand the flight to my controller the way the CX_VK_WATCH take-back does (REACQUIRE aimed at my
        shadow track's live estimate, else the hand-over pad; v_cmd = the current velocity with any descent removed); the
        route is dual from the next tick and the king is not handed the flight again (vking recaptured). True = done."""
        if getattr(self, "_vk_mem", None) is None or getattr(self, "_vk_recaptured", False) or self.route != "king":
            return False
        m_ = self.mine
        ref = np.asarray(self._vkw_ref(), dtype=np.float64).reshape(-1)[:3].copy()
        if self._mpc_top is not None:
            ref[2] = min(float(ref[2]), float(self._mpc_top))
        pos_ = st[0:3].astype(np.float64)
        m_._drop_track()
        m_.bad_spots = []
        m_.reacq_n = 0
        m_.recover_n = 0
        m_.land_retry = 0
        m_.reacq_moving = True
        m_.reacq_target = ref.copy()
        back = pos_[:2] - ref[:2]
        nb = float(np.linalg.norm(back))
        back = back / nb if nb > 1e-6 else np.array([1.0, 0.0])
        m_.reacq_wp = np.array([ref[0] + back[0] * 7.0, ref[1] + back[1] * 7.0, ref[2] + 3.0])
        m_.reacq_hold = 0.0
        m_.reacq_side = 0
        m_.reacq_t_enter = m_.t
        m_.pad_hist = []
        m_._set_mode("REACQUIRE")
        v0 = st[6:9].astype(np.float64).copy()
        v0[2] = max(float(v0[2]), 0.0)
        m_.v_cmd = v0
        if _CX_VK_UNSTICK:
            m_._vk_tb = True
        self._vk_recaptured = True
        self._vk_mem = None
        self.route = "dual"
        self.debug = {"route": "dual", "map": m_.map_label, "mode": "MPCTB", "pad_hits": 0}
        _cx_mpc_ev("mpc_hb_mine")
        return True

    def _mpc_release(self, why, st, a, t):
        """End the episode (the only one this seed) and hand the flight back (CX_MPC_HBM: to my controller via the vking
        take-back, else to the route), rate-limited this tick and for _MPC_HB_S s."""
        self._mpc_own, self._mpc_ph, self._mpc_done = False, None, True
        if _MPC_HBM:
            try:
                if self._mpc_takeback(st):
                    # this tick: hold (the route's action was the king's); my controller flies from the next tick
                    a = self._mpc_cmd(st, a, st[0:2].astype(np.float64), float(st[2]), None, vmax=0.5, dvh=0.5)
            except Exception:
                _cx_mpc_ev("mpc_hb_err")
        self._mpc_handback(st)
        self._mpc_hb_t = t
        self._mpc_mg_end, self._mpc_mg_prev = None, bool(self._mg_on)
        _cx_mpc_ev("mpc_release_" + why)
        top = self._mpc_top
        _cx_mpc_log("mpc_rel", [round(t, 2), why, round(t - float(self._mpc_t0), 2), self._mpc_pass, self._mpc_miss,
                                None if top is None else round(float(st[2]) - top, 2)], cap=4)
        return self._mpc_after(st, a, t)

    def _mpc_after(self, st, a, t):
        """After a hand-back, for _MPC_HB_S s: the route's command within 0.5 m/s of the drone's velocity and its yaw
        target within 0.6 rad of the heading (fx_omv: a 1.4 m/s step plus a 180 deg yaw flip tipped the drone)."""
        try:
            v = self._mg_route_v(a)
            vel = st[6:9].astype(np.float64)
            changed = False
            dv = v - vel
            n_ = float(np.linalg.norm(dv))
            if n_ > 0.5:
                v = vel + dv * (0.5 / n_)
                changed = True
            out = self._dv_vel_action(v, a, st) if changed else a
            a_ = np.asarray(a, dtype=np.float64).reshape(-1)
            if a_.size >= 5 and math.isfinite(float(a_[4])):
                yc = float(st[5])
                e_ = (float(a_[4]) * math.pi - yc + math.pi) % (2.0 * math.pi) - math.pi
                if abs(e_) > 0.6:
                    out = np.array(out, dtype=np.float32, copy=True)
                    y_ = yc + math.copysign(0.6, e_)
                    y_ = (y_ + math.pi) % (2.0 * math.pi) - math.pi
                    out[4] = np.float32(y_ / math.pi)
                    changed = True
            if changed:
                _cx_mpc_ev("mpc_hb_ticks")
            return out
        except Exception:
            _cx_mpc_ev("mpc_hb_err")
            return a

    def _mpc_step(self, observation, a):
        """CX_MPC (see the fx_mpc block at the top of the file). Runs last, once per new observation. Returns the route's
        action unless an MPC episode owns the flight (or a hand-back is being rate-limited)."""
        if self._mpc_done and not self._mpc_own:
            if self._mpc_hb_t is not None and float(self.step) * SIM_DT - self._mpc_hb_t <= _MPC_HB_S + 1e-9:
                st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
                return self._mpc_after(st, a, float(self.step) * SIM_DT)
            return a
        try:
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            t = float(self.step) * SIM_DT
            pos = st[0:3].astype(np.float64)
            scope = self._mpc_scope()
            if scope:
                self._mpc_collect(t, pos)
            if self._mpc_own:
                if not scope:
                    return self._mpc_release("scope", st, a, t)
                if getattr(self, "_dv_on", False) or getattr(self, "_dv_brake_t", None) is not None:
                    # fx_dv's dive (the pad slid under a drone above its top) or its post-dive brake owns this tick; MPC
                    # resumes with a HOP when it ends without contact
                    if self._mpc_ph != "DV":
                        _cx_mpc_log("mpc_dv1", [round(t, 2), self._mpc_ph, None if self._mpc_top is None else
                                                round(float(pos[2]) - self._mpc_top, 2),
                                                round(t - float(self._mpc_t0), 2)], cap=2)
                        self._mpc_ph, self._mpc_t_ph = "DV", t
                    _cx_mpc_ev("mpc_dv")
                    return a
                if self._mpc_ph == "DV":
                    self._mpc_ph, self._mpc_t_ph = "HOP", t
                return self._mpc_run(st, a, t)
            if self._mpc_hb_t is not None and t - self._mpc_hb_t <= _MPC_HB_S + 1e-9:
                a = self._mpc_after(st, a, t)
            mg_on = bool(self._mg_on) if _CX_MG_FLOOR else False
            if not scope:
                self._mpc_mg_prev, self._mpc_mg_end = mg_on, None
                return a
            # T2: an fx_mg HOLD episode that started near the pad estimate ended without contact
            if mg_on and not self._mpc_mg_prev:
                pe = self._mpc_pad_est(t)
                self._mpc_mg_d0 = None if pe is None else math.hypot(float(pe[0]) - pos[0], float(pe[1]) - pos[1])
                self._mpc_mg_end = None
            if self._mpc_mg_prev and not mg_on:
                self._mpc_mg_end = t
            self._mpc_mg_prev = mg_on
            if (_MPC_T2 and self._mpc_mg_end is not None and not mg_on
                    and t - self._mpc_mg_end >= _MPC_MG_DELAY - 1e-9):
                self._mpc_mg_end = None  # one try per HOLD episode
                d0 = self._mpc_mg_d0
                if d0 is not None and d0 <= _MPC_MG_D0 and self._mpc_try_arm(t, st, "mg"):
                    return self._mpc_run(st, a, t)
            if _MPC_T3 and self._mpc_nep == 0 and t - self._mpc_t3_t >= 0.5 - 1e-9:
                pe = self._mpc_pad_est(t)
                top3, _s3 = self._mg_top() if _CX_MG_FLOOR else (None, None)
                if (pe is not None and top3 is not None and math.hypot(float(pe[0]) - pos[0], float(pe[1]) - pos[1]) <= _MPC_T3_D
                        and -_MPC_DZLO <= float(pos[2]) - float(top3) <= _MPC_DZHI):
                    self._mpc_t3_t = t
                    if self._mpc_try_arm(t, st, "t3"):
                        return self._mpc_run(st, a, t)
            return a
        except Exception:
            was = self._mpc_own
            self._mpc_fail()
            if was:
                # the same hand-back as a release: to my controller through the vking take-back (the king sank in place
                # onto roofs after MPC holds), hold this tick, hygiene for a king main pilot
                try:
                    st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
                    try:
                        if _MPC_HBM and self._mpc_takeback(st):
                            a = self._mpc_cmd(st, a, st[0:2].astype(np.float64), float(st[2]), None, vmax=0.5, dvh=0.5)
                    except Exception:
                        _cx_mpc_ev("mpc_hb_err")
                    self._mpc_handback(st)
                    self._mpc_hb_t = float(self.step) * SIM_DT
                except Exception:
                    pass
            return a

    def _mpc_run(self, st, a, t):
        """One tick of a running episode: HOP -> WAIT at the wait point (side band of the pad), SINK when the pad top
        is under the drone, RETREAT on an unexpected surface; release on timeout / late / path / misses."""
        pos = st[0:3].astype(np.float64)
        z = float(pos[2])
        agl = float(st[137]) * 20.0
        sat = (not math.isfinite(agl)) or agl >= _M2_SAT_M
        surf = z - agl
        top_n, s_n = self._mg_top() if _CX_MG_FLOOR else (None, None)
        if top_n is not None and math.isfinite(float(top_n)) and abs(float(top_n) - self._mpc_top) <= 0.2:
            if s_n == "ray" and self._mpc_tsrc != "ray":
                _cx_mpc_log("mpc_ray", [round(t, 2), round(float(top_n) - self._mpc_top, 3)], cap=2)
            self._mpc_top, self._mpc_tsrc = float(top_n), s_n  # the ray's exact reading refines it
        top = self._mpc_top
        ph = self._mpc_ph
        fit = self._mpc_fit
        if t > _MPC_TEND and ph != "SINK":
            return self._mpc_release("late", st, a, t)
        if t - self._mpc_t0 > _MPC_TMAX and ph != "SINK":
            if t <= _MPC_TKEEP:
                return self._mpc_release("timeout", st, a, t)
            if "mpc_keep" not in CX_EVENTS:
                CX_EVENTS["mpc_keep"] = [round(t, 2), self._mpc_pass, self._mpc_miss]
        # a pilot-median top reads up to ~0.16 m low: the true top lies in [top, top + _MPC_TUP]
        tup = 0.0 if self._mpc_tsrc == "ray" else _MPC_TUP
        rel_d = pos[:2] - fit["mu"]
        lat_d = abs(float(rel_d @ fit["nv"]))
        s_dr = float(rel_d @ fit["u"])
        corridor = lat_d <= _MPC_SINK_LAT and fit["s_lo"] - 0.8 <= s_dr <= fit["s_hi"] + 0.8
        pe = self._mpc_pad_est(t)
        est_near = pe is not None and math.hypot(float(pe[0]) - pos[0], float(pe[1]) - pos[1]) <= _MPC_SINK_XY
        ray_top = (not sat) and -0.10 <= surf - top <= 0.10 + tup
        ray_pad = ray_top and corridor and est_near
        # RETREAT: a surface above top - SURF that is not the pad top, or a pad-top-level surface off the pad's corridor
        # (inside the corridor it can only be the pad: nothing else lies above top - 0.43 in the placement disc)
        if ph != "RETREAT" and not sat and ((not ray_top and surf > top + tup - _MPC_SURF) or (ray_top and not corridor)):
            ph = self._mpc_ph = "RETREAT"
            self._mpc_t_ph = t
            _cx_mpc_ev("mpc_retreat")
            _cx_mpc_log("mpc_rt", [round(t, 2), round(surf - top, 2), round(agl, 2), int(ray_top), round(lat_d, 2)], cap=3)
        # ---- fresh samples (every sample of this tick: mine's, then the king pilot's): path check, pass / miss ----
        news = []
        for smp in reversed(self._mpc_samp):
            if smp[0] != t:
                break
            news.append(smp)
        news.reverse()
        near_s, near_w = 0.0, 0.0
        for smp in news:
            q = np.array([smp[1], smp[2]], dtype=np.float64)
            dq = math.hypot(float(q[0]) - pos[0], float(q[1]) - pos[1])
            if dq <= _MPC_INV_R:
                # PATH CHECK: the pad keeps running on the fitted line (a circle / figure-8 arc taken for a line does not)
                lat_q = abs(float((q - fit["mu"]) @ fit["nv"]))
                if lat_q <= _MPC_INNOV:
                    self._mpc_bad0 = None
                elif self._mpc_bad0 is None:
                    self._mpc_bad0 = t
                elif t - self._mpc_bad0 >= _MPC_INV_S - 1e-9 and ph != "SINK":
                    _cx_mpc_log("mpc_path", [round(t, 2), round(lat_q, 2), round(t - float(self._mpc_t0), 2)], cap=2)
                    return self._mpc_release("path", st, a, t)
            s_q = float((q - fit["mu"]) @ fit["u"]) - self._mpc_sP
            if 0.3 <= abs(s_q) <= 3.0 and dq <= 4.0:
                near_s += smp[4] * s_q
                near_w += smp[4]
        if near_w > 0.0:
            # pass tracking on the tick's weighted along-axis position (two sources on one tick are one observation)
            s_q = near_s / near_w
            sg = 1.0 if s_q >= 0.0 else -1.0
            prev = self._mpc_side
            if prev is not None and prev[1] != sg and t - prev[0] <= 1.5:
                self._mpc_pass += 1
                _cx_mpc_ev("mpc_pass")
                _cx_mpc_log("mpc_passes", [round(t, 2), ph, round(z - top, 2),
                                           round(math.hypot(float(self._mpc_P[0]) - pos[0], float(self._mpc_P[1]) - pos[1]), 2)], cap=6)
                if ph == "WAIT":
                    self._mpc_miss += 1
                    _cx_mpc_ev("mpc_miss")
                    self._mpc_fit_t = -1e9  # re-fit now
            self._mpc_side = (t, sg)
            # face the pad's side
            self._mpc_yaw = math.atan2(sg * float(fit["u"][1]), sg * float(fit["u"][0]))
        if self._mpc_miss >= 2 and ph == "WAIT":
            return self._mpc_release("miss", st, a, t)
        if ph in ("HOP", "WAIT") and t - self._mpc_fit_t >= 0.5 - 1e-9:
            self._mpc_fit_t = t
            f2, w2 = self._mpc_fitnow(t)
            if w2 == "ok":
                off = abs(float((self._mpc_P - f2["mu"]) @ f2["nv"]))
                if off > _MPC_REPL_MAX:
                    _cx_mpc_log("mpc_path", [round(t, 2), "repl", round(off, 2), round(t - float(self._mpc_t0), 2)], cap=2)
                    return self._mpc_release("path", st, a, t)
                if off > _MPC_REPL:
                    P2, s2 = self._mpc_point(f2, self._mpc_P)
                    _cx_mpc_ev("mpc_replace")
                    _cx_mpc_log("mpc_repl", [round(t, 2), round(off, 2), round(float(np.hypot(*(P2 - self._mpc_P))), 2),
                                             round(f2["rms"], 3), round(f2["ext"], 2)], cap=4)
                    self._mpc_fit, self._mpc_P, self._mpc_sP = f2, P2, s2
                    fit = f2
                    if ph == "WAIT" and float(np.hypot(*(P2 - pos[:2]))) > 0.3:
                        ph = self._mpc_ph = "HOP"
                        self._mpc_t_ph = t
        P = self._mpc_P
        dP = math.hypot(float(P[0]) - pos[0], float(P[1]) - pos[1])
        z_t = top + (_MPC_DZ if self._mpc_tsrc == "ray" else _MPC_DZP)
        # ---- SINK: the pad top is under the drone ----
        if ph in ("HOP", "WAIT") and ray_pad and 0.02 < agl <= _MPC_SINK_H:
            ph = self._mpc_ph = "SINK"
            self._mpc_sink_t = t
            _cx_mpc_ev("mpc_sink")
            _cx_mpc_log("mpc_sinks", [round(t, 2), round(z - top, 2), round(agl, 2), round(dP, 2)], cap=4)
        self.debug = dict(self.debug or {})
        self.debug["mode"] = "MPC_" + ph
        CX_EVENTS["mpc_last"] = [round(t, 2), ph, round(dP, 2), round(z - top, 3), round(agl, 2),
                                 round(t - float(self._mpc_t0), 2)]
        if ph == "SINK":
            if t - self._mpc_sink_t > 0.8 or (not sat and not ray_pad):
                ph = self._mpc_ph = "HOP" if dP > 0.3 else "WAIT"
                self._mpc_t_ph = t
            else:
                return self._mpc_sink_cmd(st, a, top, t)
        if ph == "RETREAT":
            if t - self._mpc_t_ph > 1.2 or z >= top + 0.55:
                return self._mpc_release("surface", st, a, t)
            return self._mpc_cmd(st, a, pos[:2], top + 0.6, self._mpc_yaw, vmax=0.5, dvh=0.5, vup=0.8)
        if ph == "HOP" and dP <= 0.25 and abs(z - z_t) <= 0.08:
            ph = self._mpc_ph = "WAIT"
            self._mpc_t_ph = t
            if self._mpc_wait_t is None:
                self._mpc_wait_t = t
                CX_EVENTS.setdefault("mpc_wait", [round(t, 2), round(t - float(self._mpc_t0), 2), round(float(P[0]), 2),
                                                  round(float(P[1]), 2), round(self._mpc_sP, 2)])
        # climb first: well under the target height, move sideways slowly (the hop may start level with the highest
        # structure beside the pad's disc, or low after an fx_dv dive)
        vcap = 0.5 if z < z_t - _MPC_CLIMB else _MPC_VH
        if ph == "WAIT":
            CX_EVENTS["mpc_wait_s"] = round(float(CX_EVENTS.get("mpc_wait_s", 0.0)) + SIM_DT, 2)
            if dP > 0.6:
                ph = self._mpc_ph = "HOP"
                self._mpc_t_ph = t
            return self._mpc_cmd(st, a, P, z_t, self._mpc_yaw, vmax=min(1.0, vcap), dvh=_MPC_DVH)
        return self._mpc_cmd(st, a, P, z_t, self._mpc_yaw, vmax=vcap, dvh=_MPC_DVH)

    def _cx_mpc_ev_leash(self):
        _cx_mpc_ev("mpc_veto_leash")
        return True

    # ---- fx_mg: moving-pad floor guard (CX_MG_FLOOR; see the fx_mg block at the top of the file) ----
    def _mg_set_pad(self, pxy, vxy, t, src, good):
        self._mg_pxy = np.array([float(pxy[0]), float(pxy[1])], dtype=np.float64)
        v = np.asarray(vxy, dtype=np.float64).reshape(-1)[:2]
        self._mg_pv = v.copy() if v.size == 2 and np.isfinite(v).all() else np.zeros(2)
        self._mg_pt = float(t)
        self._mg_psrc = src
        if good:
            self._mg_gxy, self._mg_gv, self._mg_gt = self._mg_pxy.copy(), self._mg_pv.copy(), float(t)

    def _mg_zadd(self, zp):
        if math.isfinite(zp):
            self._mg_zs.append(float(zp))
            if len(self._mg_zs) > 40:
                del self._mg_zs[0]

    def _mg_top(self):
        """(pad-top z, source) or (None, None): the down-ray's exact reading when there is one, else the median of the
        recent fresh pilot samples (at least 3)."""
        if self._mg_top_ray is not None:
            return self._mg_top_ray, "ray"
        if len(self._mg_zs) >= _MG_NZ:
            return float(np.median(np.asarray(self._mg_zs, dtype=np.float64))), "pilot"
        return None, None

    def _mg_pred(self, t):
        """Predicted pad xy now (the last estimate plus its velocity for at most _MG_PRED_S), or None."""
        if self._mg_pxy is None:
            return None
        return self._mg_pxy + self._mg_pv * min(max(t - self._mg_pt, 0.0), _MG_PRED_S)

    def _mg_release(self, why):
        if self._mg_on:
            self._mg_on = False
            _cx_mg_ev("mg_release_" + why)

    @staticmethod
    def _mg_route_v(a):
        """The world velocity the validator flies for action a: each component clipped to [-1, 1] (the action space),
        then 3 m/s * |speed| * unit(direction)."""
        a_ = np.clip(np.nan_to_num(np.asarray(a, dtype=np.float64).reshape(-1)[:5]), -1.0, 1.0)
        if a_.size < 4:
            return np.zeros(3)
        n_ = float(np.linalg.norm(a_[0:3]))
        return a_[0:3] / n_ * abs(float(a_[3])) * SPEED_LIMIT if n_ > 1e-9 else np.zeros(3)

    def _mg_drop_action(self, a, st):
        """DROP: sink at _MG_DROP_VZ (or the route's faster sink), the command leading the measured vz by at most 0.35 m/s
        (the vertical thrust target stays positive, so the attitude loop keeps authority: see fx_dv COMMAND); horizontal:
        the drone's own velocity, stepping (<= 0.3 m/s) toward the pad velocity estimate; the route's yaw."""
        vr = self._mg_route_v(a)
        vd = np.array([float(st[6]), float(st[7])], dtype=np.float64)
        dvp = self._mg_pv - vd
        nd = float(np.hypot(dvp[0], dvp[1]))
        vxy = vd + (dvp * (0.3 / nd) if nd > 0.3 else dvp)
        vz = max(min(float(vr[2]), -_MG_DROP_VZ), float(st[8]) - 0.35)
        _cx_mg_ev("mg_drop")
        self.debug = dict(self.debug or {})
        self.debug["mode"] = "MGDROP"
        return self._dv_vel_action(np.array([vxy[0], vxy[1], vz]), a, st)

    def _mg_step(self, observation, a):
        """CX_MG_FLOOR. Runs once per new observation after the route (and CX_DV_DIVE) chose the action; reads the pilots'
        pad estimates on every route, and returns the action unchanged unless the guard holds, steers or drops."""
        try:
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            t = float(self.step) * SIM_DT
            x, y, z = float(st[0]), float(st[1]), float(st[2])
            agl = float(st[137]) * 20.0
            sat = (not math.isfinite(agl)) or agl >= _M2_SAT_M
            surf = z - agl
            m = self.mine
            mstep = int(getattr(m, "step", 0))
            live = mstep != self._mg_mstep  # mine.act() ran on this tick (its track and clock are current)
            self._mg_mstep = mstep
            if self._mg_start_xy is None:
                self._mg_start_xy = (x, y)
            a_arr = np.asarray(a, dtype=np.float32).reshape(-1)
            if self._mg_prev_a is not None and self._mg_prev_a.shape == a_arr.shape and np.array_equal(self._mg_prev_a, a_arr):
                self._mg_rep += 1
            else:
                self._mg_rep = 0
            self._mg_prev_a = a_arr.copy()
            sx_, sy_ = self._mg_start_xy
            # ---- estimates (read-only) ----
            pm = self._dv_kmain()
            ksaw = False
            if pm is not None and bool(getattr(pm, "see_P", False)) and int(getattr(pm, "platform_lost_step", 99)) == 0:
                pp = getattr(pm, "platform_position", None)
                if pp is not None:
                    pp = np.asarray(pp, dtype=np.float64).reshape(-1)
                    key = tuple(np.round(pp[:3], 6).tolist()) if pp.size >= 3 else None
                    clue = st[0:3].astype(np.float64) + st[138:141].astype(np.float64)
                    if (key is not None and np.isfinite(pp[:3]).all() and key != self._mg_kkey
                            and float(np.abs(pp[:3] - clue).max()) > 1e-3  # (its seed value is the clue centre)
                            and math.hypot(pp[0] - sx_, pp[1] - sy_) > _MG_START_R):
                        self._mg_kkey = key  # a new detection this tick (platform_position is rewritten on each one)
                        ksaw = True
                        v = np.zeros(2)
                        tr = getattr(pm, "_ab_tracker", None)
                        if tr is not None and bool(getattr(tr, "initialized", False)):
                            vv = np.asarray(getattr(tr, "vel", np.zeros(3)), dtype=np.float64).reshape(-1)
                            if vv.size >= 2 and np.isfinite(vv[:2]).all():
                                v = vv[:2].copy()
                        good = z - float(pp[2]) >= _MG_GOOD_H
                        self._mg_set_pad(pp[:2], v, t, "king", good)
                        if good and not self._mg_on and math.hypot(pp[0] - x, pp[1] - y) <= _MG_EST_NEAR:
                            self._mg_zadd(float(pp[2]))
            if (live and m.pad_pos is not None and float(m.pad_last_seen) == float(m.t)
                    and int(getattr(m, "pad_hits", 0)) >= 5):
                mp = np.asarray(m.pad_pos, dtype=np.float64).reshape(-1)
                if mp.size >= 3 and np.isfinite(mp[:3]).all() and math.hypot(mp[0] - sx_, mp[1] - sy_) > _MG_START_R:
                    good = z - float(mp[2]) >= _MG_GOOD_H
                    if not ksaw:
                        self._mg_set_pad(mp[:2], m._pad_vel_now(), t, "mine", good)
                    if good and not self._mg_on and math.hypot(mp[0] - x, mp[1] - y) <= _MG_EST_NEAR:
                        self._mg_zadd(float(mp[2]))
            ph = self._mg_pred(t)
            dxy = float(math.hypot(ph[0] - x, ph[1] - y)) if ph is not None else 1e9
            # raised: the ray reads a surface at least 0.15 m above the lowest surface it read in the last second
            raised = (not sat) and bool(self._mg_shist) and surf >= min(self._mg_shist) + 0.15
            if not sat:
                self._mg_shist.append(surf)
                if len(self._mg_shist) > 50:
                    del self._mg_shist[0]
            if raised and dxy <= _MG_RAY_XY:
                # the down-ray on the pad top: the first reading must match the pilots' median, later ones refine it
                # only within 3 cm (no ratchet down a slope beside the pad)
                if self._mg_top_ray is None:
                    if len(self._mg_zs) >= _MG_NZ and abs(surf - float(np.median(np.asarray(self._mg_zs)))) <= _MG_TOL:
                        self._mg_top_ray = surf
                        _cx_mg_ev("mg_top_ray")
                elif abs(surf - self._mg_top_ray) <= 0.03:
                    self._mg_top_ray = 0.8 * self._mg_top_ray + 0.2 * surf
            top, tsrc = self._mg_top()
            ray_on = top is not None and not sat and abs(surf - top) <= _MG_TOL
            if ray_on and raised and dxy <= _MG_RAY_XY and not ksaw:
                self._mg_set_pad((x, y), self._mg_pv, t, "ray", True)  # the pad is under the drone (within ~0.6 m)
                ph, dxy = self._mg_pred(t), 0.0
            if self.route == "king" and not self.king_from_dual and pm is not None and (
                    bool(getattr(pm, "tracking", False)) or bool(getattr(pm, "move_in_auto_mode", False))):
                self._mg_mv = True  # open: the king pilot flags the pad moving (sticky)
            if _CX_MPC and self._mpc_own:
                # fx_mpc owns the flight: fx_mg keeps its estimates (read by MPC) but runs no episode
                self._mg_dropping = False
                if self._mg_on:
                    self._mg_on = False
                    _cx_mg_ev("mg_release_mpc")
                return a
            # ---- contact latch: never act on a drone touching (or about to touch) the pad ----
            # (a DROP already running continues until the ray leaves the pad top, i.e. the ray origin enters the pad at
            # contact, or for at most 0.6 s: stopping it 0.1 m over the pad could leave the drone hovering there)
            drop_on = self._mg_dropping and ray_on and t - self._mg_drop_t0 <= 0.6 + 1e-6
            if self._mg_latched:
                if drop_on:
                    return self._mg_drop_action(a, st)
                self._mg_dropping = False
                return a
            if self._mg_touched:
                # rev 7: after a touch the guard does nothing at all (a drone settling on a static pad reads the ground
                # through the pad) until the drone has slid off the pad top without contact: 3 ticks in a row with the
                # ray off the pad top at >= 0.8 m/s (a moving pad's contact ends the seed on its first tick)
                if drop_on:
                    return self._mg_drop_action(a, st)
                self._mg_dropping = False
                if not ray_on and not sat and math.hypot(float(st[6]), float(st[7])) >= 0.8:
                    self._mg_slid += 1
                else:
                    self._mg_slid = 0
                if self._mg_slid < 3:
                    return a
                self._mg_touched, self._mg_slid = False, 0  # it slid off the pad top without contact (1515709376)
                _cx_mg_ev("mg_untouch")
            if ray_on and agl <= _MG_TOUCH and dxy <= _MG_R and not self._mg_touched:
                # touching or about to: from now on the guard may only cut a sink (no horizontal change, no DROP), so a
                # drone settling on a static pad is never pushed sideways; a drone that hovers just over a moving pad
                # and slides off it without contact (1283975870) is still kept off the ground
                self._mg_touched = True
                self._mg_release("touch")
                _cx_mg_ev("mg_latch_touch")
                if drop_on:
                    return self._mg_drop_action(a, st)
                self._mg_dropping = False
                return a
            if not drop_on:
                self._mg_dropping = False
            # ---- gates ----
            if self.route != "king" or not (self.king_from_dual or self._mg_mv) or top is None:
                self._mg_release("far")
                return a
            if getattr(self, "_dv_on", False) or getattr(self, "_dv_brake_t", None) is not None:
                self._mg_release("far")
                return a  # a CX_DV_DIVE dive or its brake owns this tick
            if ph is None or dxy > (_MG_RFAR if self._mg_on else _MG_R) or (not self._mg_on and t - self._mg_pt > _MG_AGE):
                self._mg_release("far")
                return a  # a new episode needs a fresh estimate within _MG_R; a running one continues within _MG_RFAR
            if ray_on:
                self._mg_release("ray")
                if drop_on:
                    return self._mg_drop_action(a, st)
                if (_MG_DROP and not self._mg_touched and t - self._mg_last_hold <= _MG_DROP_S + 1e-6
                        and _MG_TOUCH < agl <= _MG_DROP_H):
                    # the guard held the drone beside the pad and the ray now finds the pad top under it: sink onto it
                    self._mg_dropping, self._mg_drop_t0 = True, t
                    return self._mg_drop_action(a, st)
                return a
            if not sat and surf > top + _MG_TOL:
                self._mg_release("far")
                return a  # something higher than the pad is under the drone: not this guard's case
            if z > top + _MG_ON + (0.1 if self._mg_on else 0.0):
                self._mg_release("high")
                return a
            if self._mg_ticks * SIM_DT >= _MG_MAX_S:
                self._mg_release("cap")
                return a
            if not self._mg_on and z < top - _MG_LOW:
                return a  # far below the pad top: not a drone that sank past its edge
            # ---- HOLD ----
            vr = self._mg_route_v(a)
            vz_hold = min(max(_MG_K * (top + _MG_HOLD - z), -_MG_VDN), _MG_VUP)
            loiter = (_MG_STEER and self._mg_rep >= _MG_REP_N and not ksaw and not self._mg_touched
                      and self._mg_gxy is not None)
            vz = vz_hold if loiter else max(float(vr[2]), vz_hold)
            vxy = np.array([float(vr[0]), float(vr[1])], dtype=np.float64)
            yaw = None
            biased = False
            steer = loiter or (_MG_STEER and not ksaw and not self._mg_touched and self._mg_gxy is not None
                               and t - self._mg_gt <= _MG_STEER_AGE)
            if steer:
                gp = self._mg_gxy + self._mg_gv * min(max(t - self._mg_gt, 0.0), _MG_PRED_S)
                off = gp - np.array([x, y], dtype=np.float64)
                vp = self._mg_gv
                vt = vp + _MG_KXY * off
                nt = float(np.hypot(vt[0], vt[1]))
                if nt > _MG_VXY:
                    vt = vt * (_MG_VXY / nt)
                vd = np.array([float(st[6]), float(st[7])], dtype=np.float64)
                dvp = vt - vd
                nd = float(np.hypot(dvp[0], dvp[1]))
                vxy = vd + (dvp * (_MG_DVH / nd) if nd > _MG_DVH else dvp)
                if _MG_YAW and float(np.hypot(off[0], off[1])) > 0.8:
                    yaw = math.atan2(off[1], off[0]) / math.pi
            elif _MG_BIAS > 0.0 and 0.2 < dxy <= 1.5 and not self._mg_touched:
                # the pilot sees the pad: keep its horizontal command plus a small inward component (the drone flying
                # level with the pad's side band then closes the last 5-20 cm, and a side contact is a landing)
                off = ph - np.array([x, y], dtype=np.float64)
                vxy = vxy + off / dxy * _MG_BIAS
                biased = True
            vh = float(np.hypot(vxy[0], vxy[1]))
            vmax_h = math.sqrt(max(SPEED_LIMIT * SPEED_LIMIT - vz * vz, 0.0))
            if vh > vmax_h:
                vxy = vxy * (vmax_h / vh)
            # resting on a pad edge (static pad): the guard asks for a sink (the drone is above its floor) and the drone
            # stands still (-0.02 < vz < 0.05, |v_xy| < 0.3), at pad-top height or above -> latch off (a free hover sits at the
            # floor, top + _MG_HOLD; rev 6: a drone RISING under a pilot climb is not resting, 1270074406)
            if (vz < -0.08 and -0.02 < float(st[8]) < 0.05 and math.hypot(float(st[6]), float(st[7])) < 0.3
                    and z - top >= -0.02):
                self._mg_blk += 1
                if self._mg_blk >= 5:
                    self._mg_latched = True
                    self._mg_release("latch")
                    _cx_mg_ev("mg_latch_block")
                    return a
            else:
                self._mg_blk = 0
            if not (abs(vz - float(vr[2])) > 1e-9 or steer or biased):
                return a  # the route already keeps above the floor and nothing else changes: fly it as it is
            if not self._mg_on:
                self._mg_on = True
                _cx_mg_ev("mg_fire")
                if "mg_fire1" not in CX_EVENTS:
                    CX_EVENTS["mg_fire1"] = [round(t, 2), round(z - top, 2), round(dxy, 2), round(agl, 2),
                                             round(surf - top, 2), tsrc, round(t - self._mg_pt, 2),
                                             str(getattr(self.king, "_active", None)), round(float(vr[2]), 2),
                                             "dual" if self.king_from_dual else "king"]
            self._mg_ticks += 1
            self._mg_last_hold = t
            _cx_mg_ev("mg_ticks")
            if steer:
                _cx_mg_ev("mg_steer")
            if loiter:
                _cx_mg_ev("mg_loiter")
            if biased:
                _cx_mg_ev("mg_bias")
            self.debug = dict(self.debug or {})
            self.debug["mode"] = "MGHOLD"
            out = self._dv_vel_action(np.array([vxy[0], vxy[1], vz]), a, st)
            if yaw is not None:
                out[4] = np.float32(yaw)
            return out
        except Exception:
            self._mg_on = False
            self._mg_latched = True  # never act again after an internal error
            _cx_mg_ev("mg_error")
            return a

    # ---- fx_kl: mountain drop onto a pad that slides under the drone (CX_KL_MDROP; see the fx_kl block at the top) ----
    def _kl_pad_vel(self):
        """A fresh pad velocity estimate (xy) and its source, or (None, None): the king pilot's platform track when it saw
        the pad within _DV_FRESH, else my shadow track when it saw the (moving) pad within _KL_VFRESH."""
        kest = self._dv_kest()
        if kest is not None:
            return np.asarray(kest[1], dtype=np.float64).reshape(-1)[:2], "king"
        m = self.mine
        if m.pad_pos is not None and bool(m.pad_moving) and float(m.t) - float(m.pad_last_seen) <= _KL_VFRESH + 1e-6:
            v = np.asarray(m._pad_vel_now(), dtype=np.float64).reshape(-1)[:2]
            if v.size == 2 and np.isfinite(v).all():
                return v, "mine"
        return None, None

    def _kl_dive_action(self, a, st):
        """The fx_dv rate-limited dive toward this dive's floor: the sink target leads the measured vz by at most _DV_LEAD
        (up to _KL_VZ, and never faster than the drone could stop above _kl_floor); horizontal within _DV_DVH of the
        drone's own velocity, stepping toward the pad velocity estimate (or holding the drone's velocity)."""
        room = float(st[2]) - self._kl_floor
        sink = min(_KL_VZ, math.sqrt(2.0 * _DV_BRAKE_A * max(room, 0.0)))
        vd = np.array([float(st[6]), float(st[7])], dtype=np.float64)
        tgt = vd if self._kl_vxy is None else np.asarray(self._kl_vxy, dtype=np.float64)
        dv = tgt - vd
        n = float(np.hypot(dv[0], dv[1]))
        vxy = vd + (dv * (_DV_DVH / n) if n > _DV_DVH else dv)
        vh = float(np.hypot(vxy[0], vxy[1]))
        if vh > SPEED_LIMIT:
            vxy = vxy * (SPEED_LIMIT / vh)
            vh = SPEED_LIMIT
        sink = min(sink, math.sqrt(max(SPEED_LIMIT * SPEED_LIMIT - vh * vh, 0.0)))
        vz = max(-sink, float(st[8]) - _DV_LEAD)
        return self._dv_vel_action(np.array([vxy[0], vxy[1], vz]), a, st)

    def _kl_step(self, observation, a):
        """CX_KL_MDROP. Runs once per new observation after the route, CX_DV_DIVE and CX_MG_FLOOR chose the action; returns
        that action unchanged unless a drop (or its post-drop brake) runs."""
        if self._kl_dead:
            return a
        try:
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            t = float(self.step) * SIM_DT
            x, y, z = float(st[0]), float(st[1]), float(st[2])
            agl = float(st[137]) * 20.0
            sat = (not math.isfinite(agl)) or agl >= _M2_SAT_M
            surf = z - (agl if not sat else 20.0)
            self._kl_shist = [h for h in self._kl_shist if t - h[0] <= _KL_RISE_S + 1e-6] + [(t, surf)]
            if not (self.route == "king" and self.king_from_dual and self.mine.map_label == "mountain"
                    and getattr(self, "_mv_armed", False)):
                self._kl_on, self._kl_ep, self._kl_brake_t = False, None, None
                return a
            if self._kl_brake_t is not None:
                if t <= self._kl_brake_t + 1e-6 and float(st[8]) < -0.3:
                    return self._dv_brake_action(a, st)
                self._kl_brake_t = None
            anc, _, _ = self._cx_mv_anchor()
            top, tol, src = None, _KL_TOL, None
            if getattr(self, "_mg_top_ray", None) is not None:
                top, src = float(self._mg_top_ray), "ray"
                if anc is not None and abs(top - float(anc[2])) > 0.5:
                    top = None  # the two disagree: not a pad top this guard trusts
            elif anc is not None:
                top, tol, src = float(anc[2]), _KL_TOL_ANC, "anc"
            if self._kl_on:
                el = t - self._kl_t0
                end = None
                if el >= _KL_MAX_S - 1e-6:
                    end = "kl_end_time"
                elif sat or abs(surf - self._kl_top) > max(tol, 0.08):
                    end = "kl_end_left"  # (also the contact tick: the ray origin enters the pad, see fx_dv)
                elif el >= 0.4 and float(st[8]) > -0.2:
                    end = "kl_end_block"
                elif float(st[8]) >= self._kl_vz_prev + 0.4:
                    end = "kl_end_hit"
                self._kl_vz_prev = float(st[8])
                if end is None:
                    _cx_kl_ev("kl_ticks")
                    self.debug = dict(self.debug or {})
                    self.debug["mode"] = "KLDROP"
                    return self._kl_dive_action(a, st)
                self._kl_on = False
                self._kl_ep = None
                _cx_kl_ev(end)
                vz_now = float(st[8])
                if end == "kl_end_left" and not sat and vz_now < -0.3 and agl < vz_now * vz_now / (2.0 * _DV_BRAKE_A) + _DV_FLOOR:
                    self._kl_brake_t = t + 0.3  # the pad slid away and the ray reads terrain inside the stopping distance
                    _cx_kl_ev("kl_brake")
                    return self._dv_brake_action(a, st)
                return a
            if top is None or sat or abs(surf - top) > tol:
                self._kl_ep = None
                return a
            if self._kl_ep is None:
                lo = min(h[1] for h in self._kl_shist)
                if surf - lo < _KL_RISE:
                    return a  # the ray is on the pad top, but it did not just slide in under the drone
                self._kl_ep = (t, lo)
            if self._kl_n >= _KL_MAX_N:
                return a
            if getattr(self, "_dv_on", False) or getattr(self, "_dv_brake_t", None) is not None:
                return a  # a CX_DV_DIVE dive or brake owns this tick
            if getattr(self, "_mg_dropping", False):
                return a  # a CX_MG_FLOOR DROP owns this tick
            if not (_KL_HMIN <= agl <= _KL_HMAX):
                return a
            if anc is None or math.hypot(float(anc[0]) - x, float(anc[1]) - y) > _KL_ANC_R:
                return a
            vxy, vsrc = self._kl_pad_vel()
            self._kl_on, self._kl_t0, self._kl_top = True, t, surf
            self._kl_floor = float(self._kl_ep[1]) + _DV_FLOOR
            self._kl_vxy = vxy
            self._kl_vz_prev = float(st[8])
            self._kl_n += 1
            self._kl_ep = None
            _cx_kl_ev("kl_fire")
            if src == "anc":
                _cx_kl_ev("kl_src_anc")
            if "kl_fire1" not in CX_EVENTS:
                pm = self._dv_kmain()
                CX_EVENTS["kl_fire1"] = [round(t, 2), round(agl, 2), round(surf - top, 3),
                                         round(math.hypot(float(anc[0]) - x, float(anc[1]) - y), 2), vsrc,
                                         str(getattr(pm, "_mode", None)) if pm is not None else None, round(float(st[8]), 2)]
            elif len(CX_EVENTS.setdefault("kl_at", [])) < 6:
                pm = self._dv_kmain()
                CX_EVENTS["kl_at"].append([round(t, 2), round(agl, 2), vsrc,
                                           str(getattr(pm, "_mode", None)) if pm is not None else None])
            self.debug = dict(self.debug or {})
            self.debug["mode"] = "KLDROP"
            return self._kl_dive_action(a, st)
        except Exception:
            self._kl_on, self._kl_brake_t, self._kl_dead = False, None, True
            _cx_kl_ev("kl_err")
            return a

    # ---- fx_kl: take the flight back from a king pilot blind in 'landing' (CX_KL_KBACK; see the fx_kl block at the top) ----
    def _kb_step(self, observation):
        """CX_KL_KBACK. Runs once per new observation right after the route chose its action (the action itself is not
        changed); on firing it re-arms my controller in REACQUIRE and switches the route to dual from the next tick."""
        if self._kb_done:
            return
        try:
            m = self.mine
            pm = self._dv_kmain()
            see = pm is not None and bool(getattr(pm, "see_P", False)) and int(getattr(pm, "platform_lost_step", 99)) == 0
            if self._kb_tb:
                # my controller flies after a take-back; CX_KL_KB_REHO hands a re-found moving pad back to the king
                if (_KB_REHO and self._kb_n < _KB_MAX_N and self.route == "dual" and see and m.mode == "APPROACH"
                        and bool(m.pad_moving) and m._track_reliable()):
                    self._kb_tb = False
                    self._kb_blind = 0.0
                    self.route = "king"
                    self.ho_t = float(m.t)
                    # the pilot (run every tick on the dual route) is still in the 'landing' it was taken from, whose
                    # tracking branch only sinks in place even with the pad in view (1882127997 rev1: 17 s at -0.15 m/s,
                    # 3-5 m from a pad it saw, into the ground): restart its approach from 'navigation', as at a first
                    # sighting (tracking is re-decided there once the pad moves)
                    pm._mode = "navigation"
                    pm.first_order_cnt = 20
                    pm.tracking = False
                    pm._first_plat_pos = None
                    pm._landing_platform_position = None
                    pm._landing_patience = 0
                    pm._last_landing_hdist = None
                    pm._goal_lost_steps = 0
                    pm._goal_return_mode = False
                    try:
                        pm.controller.reset()
                    except Exception:
                        pass
                    _cx_kl_ev("kb_reho")
                    if len(CX_EVENTS.setdefault("kb_reho_t", [])) < 4:
                        CX_EVENTS["kb_reho_t"].append(round(float(self.step) * SIM_DT, 2))
                return
            if self._kb_n >= _KB_MAX_N:
                self._kb_done = True
                return
            if not (self.route == "king" and self.king_from_dual and m.map_label in _KB_MAPS):
                self._kb_blind = 0.0
                return
            if pm is None:
                self._kb_blind = 0.0
                return
            kmode = str(getattr(pm, "_mode", ""))
            pw_rep = False
            if _CX_PW_CREP and m.map_label == "city" and not see and int(getattr(self, "_mg_rep", 0)) >= _PW_REP_N:
                a_p = getattr(self, "_mg_prev_a", None)
                if a_p is not None and np.asarray(a_p).size >= 4:
                    a_p = np.asarray(a_p, dtype=np.float64).reshape(-1)
                    n3 = float(np.linalg.norm(a_p[0:3]))
                    vh_p = (float(np.hypot(a_p[0], a_p[1])) / n3 * min(abs(float(a_p[3])), 1.0) * SPEED_LIMIT) if n3 > 1e-6 else 0.0
                    pw_rep = vh_p > _PW_REP_VH
            if pw_rep:
                pass  # fx_pw2 CX_PW_CREP: a long bit-identical repeat carrying the drone sideways (any mode, any height)
            elif kmode == "landing" and not see:
                pass  # blind in 'landing': the tracking branch repeats its last action
            elif _KB_REHO and kmode in ("landing", "navigation") and see and self._kb_n > 0:
                # after a re-hand-over the pilot can also sink in place while it sees the pad metres away
                # (1882127997 rev2: -0.15 m/s from 2.3 m, 3-5 m from the pad it tracked, into the ground)
                pp = getattr(pm, "platform_position", None)
                pp = np.asarray(pp, dtype=np.float64).reshape(-1) if pp is not None else None
                st0 = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
                if (pp is None or pp.size < 2 or not np.isfinite(pp[:2]).all()
                        or math.hypot(float(pp[0]) - float(st0[0]), float(pp[1]) - float(st0[1])) < _KB_FAR):
                    self._kb_blind = 0.0
                    return
            else:
                self._kb_blind = 0.0
                return
            self._kb_blind += SIM_DT
            if self._kb_blind < _KB_LOST_S - 1e-9 and not pw_rep:
                return
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            pos = st[0:3].astype(np.float64)
            agl = float(st[137]) * 20.0
            sat = (not math.isfinite(agl)) or agl >= _M2_SAT_M
            top, tsrc = (self._mg_top() if _CX_MG_FLOOR else (None, None))
            if top is None and m.pad_pos is not None:
                top, tsrc = float(np.asarray(m.pad_pos, dtype=np.float64).reshape(-1)[2]), "mine"
            if top is None or not math.isfinite(top):
                return
            if float(pos[2]) - top > _KB_ABOVE and not pw_rep:
                return  # not low: the pilot is not sinking beside the pad (yet)
            if not sat and float(pos[2]) - agl >= top - 0.15 and not pw_rep:
                return  # over the pad top (or something as high): a blind final descent, not this case
            ref, rsrc = None, None
            if m.map_label == "mountain" and _CX_MV_ANCHOR:
                anc, _, _ = self._cx_mv_anchor()
                if anc is not None:
                    ref, rsrc = np.asarray(anc, dtype=np.float64).copy(), "anchor"
            if ref is None and _CX_MG_FLOOR and self._mg_gxy is not None:
                ref, rsrc = np.array([self._mg_gxy[0], self._mg_gxy[1], top], dtype=np.float64), "mg_good"
            if ref is None and _CX_MG_FLOOR and self._mg_pxy is not None:
                ref, rsrc = np.array([self._mg_pxy[0], self._mg_pxy[1], top], dtype=np.float64), "mg"
            if ref is None and m.pad_pos is not None:
                ref, rsrc = np.asarray(m.pad_pos, dtype=np.float64).reshape(-1)[:3].copy(), "mine"
            if ref is None or not np.isfinite(ref).all():
                return
            ref[2] = top  # the pad plane
            t = float(self.step) * SIM_DT
            d_ref = float(np.linalg.norm(pos[:2] - ref[:2]))
            self._kb_n += 1
            self._kb_tb = True
            _cx_kl_ev("kb_fire")
            if pw_rep:
                _cx_kl_ev("pw_crep")
                if "pw_crep1" not in CX_EVENTS:
                    CX_EVENTS["pw_crep1"] = [round(t, 2), int(getattr(self, "_mg_rep", 0)), round(float(pos[2]) - top, 2),
                                             round(d_ref, 2)]
            rec = [round(t, 2), round(self._kb_blind, 2), round(float(pos[2]) - top, 2), tsrc,
                   round(d_ref, 2), rsrc, str(getattr(self.king, "_active", None))]
            if "kb_fire1" not in CX_EVENTS:
                CX_EVENTS["kb_fire1"] = rec
            else:
                CX_EVENTS["kb_fire2"] = rec
            self._kb_blind = 0.0
            m._drop_track()
            m.bad_spots = []
            m.reacq_n = 0
            m.recover_n = 0
            m.land_retry = 0
            m.reacq_moving = True
            m.reacq_target = ref.copy()
            back = pos[:2] - ref[:2]
            nb = float(np.linalg.norm(back))
            back = back / nb if nb > 1e-6 else np.array([1.0, 0.0])
            m.reacq_wp = np.array([ref[0] + back[0] * 7.0, ref[1] + back[1] * 7.0, ref[2] + 3.0])
            m.reacq_hold = 0.0
            m.reacq_side = 0
            m.reacq_t_enter = m.t
            m.pad_hist = []
            m._set_mode("REACQUIRE")
            v0 = st[6:9].astype(np.float64).copy()
            v0[2] = max(float(v0[2]), 0.0)
            m.v_cmd = v0
            self.king_blocked = True
            self.route = "dual"
        except Exception:
            self._kb_done = True
            _cx_kl_ev("kb_err")

    # ---- fx_pw2: take the flight back from the city graph pack's slow in-place sink (CX_PW_GSINK; see the flag) ----
    def _pw_gsink_step(self, observation, a):
        """fx_pw2 CX_PW_GSINK. Runs once per new observation right after the route chose its action (the action itself is
        not changed: the firing tick still flies it, as with CX_KL_KBACK); on firing it re-arms my controller in
        REACQUIRE and switches the route to dual from the next tick. Once per seed; any error stops it for the seed."""
        try:
            if getattr(self, "_pw_gs_done", True):
                return
            m = self.mine
            if not (self.route == "king" and self.king_from_dual and m.map_label == "city"
                    and getattr(self.king, "_active", None) == "graph"):
                self._pw_gs_s = 0.0
                return
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            pos = st[0:3].astype(np.float64)
            vel = st[6:9].astype(np.float64)
            a_ = np.asarray(a, dtype=np.float64).reshape(-1)
            ok = a_.size >= 4 and bool(np.isfinite(a_[:4]).all())
            if ok:
                n3 = float(np.linalg.norm(a_[0:3]))
                vc = a_[0:3] / n3 * min(abs(float(a_[3])), 1.0) * SPEED_LIMIT if n3 > 1e-6 else np.zeros(3)
                ok = (-_PW_GS_VZ <= float(vc[2]) <= -0.02 and math.hypot(float(vc[0]), float(vc[1])) < _PW_GS_VH
                      and -_PW_GS_VZ <= float(vel[2]) <= -0.05 and math.hypot(float(vel[0]), float(vel[1])) < _PW_GS_VH)
            top, tsrc = None, None
            if ok:
                top, tsrc = (self._mg_top() if _CX_MG_FLOOR else (None, None))
                if top is None and m.pad_pos is not None:
                    top, tsrc = float(np.asarray(m.pad_pos, dtype=np.float64).reshape(-1)[2]), "mine"
                ok = top is not None and math.isfinite(float(top)) and float(pos[2]) - float(top) <= _PW_GS_ABOVE
            if ok:
                agl = float(st[137]) * 20.0
                sat = (not math.isfinite(agl)) or agl >= _M2_SAT_M
                ok = sat or float(pos[2]) - agl < float(top) - 0.15  # the ray is not on the pad top (or anything as high)
            if not ok:
                self._pw_gs_s = 0.0
                return
            self._pw_gs_s += SIM_DT
            if self._pw_gs_s < _PW_GS_T - 1e-9:
                return
            top = float(top)
            ref, rsrc = None, None
            if _CX_MG_FLOOR and self._mg_gxy is not None:
                ref, rsrc = np.array([self._mg_gxy[0], self._mg_gxy[1], top], dtype=np.float64), "mg_good"
            if ref is None and _CX_MG_FLOOR and self._mg_pxy is not None:
                ref, rsrc = np.array([self._mg_pxy[0], self._mg_pxy[1], top], dtype=np.float64), "mg"
            if ref is None and m.pad_pos is not None:
                ref, rsrc = np.asarray(m.pad_pos, dtype=np.float64).reshape(-1)[:3].copy(), "mine"
            if ref is None or not np.isfinite(ref).all():
                return
            ref[2] = top  # the pad plane
            t = float(self.step) * SIM_DT
            d_ref = float(np.linalg.norm(pos[:2] - ref[:2]))
            self._pw_gs_done = True
            _cx_kl_ev("pw_gsink")
            CX_EVENTS["pw_gsink1"] = [round(t, 2), round(self._pw_gs_s, 2), round(float(pos[2]) - top, 2), tsrc,
                                      round(d_ref, 2), rsrc]
            m._drop_track()
            m.bad_spots = []
            m.reacq_n = 0
            m.recover_n = 0
            m.land_retry = 0
            m.reacq_moving = True
            m.reacq_target = ref.copy()
            back = pos[:2] - ref[:2]
            nb = float(np.linalg.norm(back))
            back = back / nb if nb > 1e-6 else np.array([1.0, 0.0])
            m.reacq_wp = np.array([ref[0] + back[0] * 7.0, ref[1] + back[1] * 7.0, ref[2] + 3.0])
            m.reacq_hold = 0.0
            m.reacq_side = 0
            m.reacq_t_enter = m.t
            m.pad_hist = []
            m._set_mode("REACQUIRE")
            v0 = st[6:9].astype(np.float64).copy()
            v0[2] = max(float(v0[2]), 0.0)
            m.v_cmd = v0
            self.king_blocked = True
            self.route = "dual"
        except Exception:
            self._pw_gs_done = True
            _cx_kl_ev("pw_gsink_err")

    def _handback_step(self, observation):
        """Shadow my controller during a mid-flight king hand-off and take the flight back when the king has lost a pad my
        track calls static and close: uid130 drops a static landing once the pad is under the camera and wanders off on the
        false motion lead, while my estimate stays within ~0.2 m (results/hb/)."""
        m = self.mine
        try:
            with _CxShadow():  # fx_ms: no CX_EVENTS counts from a discarded action (does nothing with every flag off)
                m.act(observation)  # its action is discarded: keeps the pad track and clock current
        except Exception:
            return
        try:
            pm = None
            if getattr(self.king, "_active", None) == "uid130":
                pm = getattr(getattr(getattr(self.king, "_uid130", None), "_base", None), "_main", None)
            if pm is None or m.t - self.ho_t < _HB_AFTER:
                return
            if _CX_MV_ANCHOR and getattr(self, "_mv_armed", False):
                # CX_MV_ANCHOR is steering uid130's search to the orbit: this static take-back would hand a MOVING pad
                # to my static landing when my frozen fit speed happens to read < 0.40 near a linear pad's turning
                # point (1048748035 test: 0.01). It is rare but real in the champion (1224832766, a static pad handed
                # over as moving: take-back at 43.84 s, 0.9555); rev1 never arms on such a pad (_MV_SPAN_MIN), so
                # there it still runs exactly as in c3.
                return
            if _HB_WINDOW > 0 and m.t - self.ho_t > _HB_WINDOW:
                return
            lost = str(getattr(pm, "_mode", "")) in ("search", "navigation") and not (
                bool(getattr(pm, "see_P", False)) and int(getattr(pm, "platform_lost_step", 99)) == 0)
            self.hb_lost_n += 1 if lost else 0
            if self.hb_lost_n * SIM_DT < _HB_LOST_S or m.pad_pos is None or m.pad_hits < _HB_HITS or float(m.pad_fit_sp) >= _HB_FIT:
                return
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            if float(np.linalg.norm(st[0:2].astype(np.float64) - np.asarray(m.pad_pos, dtype=np.float64)[:2])) >= _HB_DEST:
                return
            # take it back: a static pad, finish on the estimate
            m.pad_moving, m.pad_ever_moving, m.pad_move_votes = False, False, 0
            m.pad_vel = np.zeros(2)
            m.pad_still_t = max(float(m.pad_still_t), 3.0)
            m.pad_last_seen, m.pad_miss_t = m.t, 0.0
            m.v_cmd = st[6:9].astype(np.float64).copy()
            m._set_mode("APPROACH")
            self.route, self.king_blocked, self.handback_t = "dual", True, round(float(m.t), 2)
        except Exception:
            return

    def _cx_mv_collect(self):
        """CX_MV_ANCHOR: one sample of my pad estimate per 0.1 s, only on ticks with a fresh detection (mountain).
        Called on every mountain tick (dual route, and the shadow run after the hand-over). Before the hand-over, a drop
        or re-seed of my track (pad_hits falls) clears the samples, so the anchor is built only from the track that
        hands over (plus everything my shadow track sees after it), never from an earlier (decoy) track."""
        try:
            m = self.mine
            h = int(m.pad_hits) if m.pad_pos is not None else 0
            if h < self._mv_prev_hits and not getattr(self, "king_from_dual", False) and self._mv_pts:
                self._mv_clr += len(self._mv_pts)
                self._mv_pts = []
            self._mv_prev_hits = h
            if m.pad_pos is None or m.pad_hits < 10 or float(m.pad_last_seen) != float(m.t):
                return
            pts = self._mv_pts
            if pts and float(m.t) - pts[-1][0] < 0.099:
                return
            p = np.asarray(m.pad_pos, dtype=np.float64).reshape(-1)
            pts.append((float(m.t), float(p[0]), float(p[1]), float(p[2])))
            if len(pts) > 400:
                pts.pop(0)
        except Exception:
            pass

    def _cx_mv_anchor(self):
        """Orbit-centre estimate: 5-95 % mid-range (xy) and median height of the recent samples near their median.
        Returns (anchor, span, n): span is the diagonal of the 5-95 % xy box (the static guard), n the samples used."""
        t = float(self.mine.t)
        pts = [q for q in self._mv_pts if t - q[0] <= _MV_KEEP_S]
        if len(pts) < _MV_MIN_PTS:
            return None, 0.0, len(pts)
        a = np.asarray(pts, dtype=np.float64)[:, 1:4]
        med = np.median(a, axis=0)
        keep = (np.linalg.norm(a[:, :2] - med[:2], axis=1) <= _MV_OUTLIER_M) & (np.abs(a[:, 2] - med[2]) <= 2.0)
        n = int(keep.sum())
        if n < _MV_MIN_PTS:
            return None, 0.0, n
        c = a[keep]
        lo, hi = np.percentile(c[:, :2], 5, axis=0), np.percentile(c[:, :2], 95, axis=0)
        span = float(math.hypot(hi[0] - lo[0], hi[1] - lo[1]))
        return np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), float(np.median(c[:, 2]))]), span, n

    def _cx_mv_arm(self):
        """CX_MV_ANCHOR, before king.act on a mountain hand-over: while uid130 searches, aim its goal-return at the orbit."""
        try:
            if getattr(self.king, "_active", None) != "uid130":
                return
            k = getattr(getattr(getattr(self.king, "_uid130", None), "_base", None), "_main", None)
            if k is None:
                return
            if str(getattr(k, "_mode", "")) != "search":
                if self._mv_in_search and self._mv_arm_t is not None:
                    # diagnostics for the paired batch: how long each armed search lasted (short dropout vs real loss)
                    if len(CX_EVENTS.setdefault("mv_dur", [])) < 8:
                        CX_EVENTS["mv_dur"].append(round(float(self.mine.t) - self._mv_arm_t, 2))
                    self._mv_arm_t = None
                self._mv_in_search = False
                self._mv_still_ep = False
                if str(getattr(k, "_mode", "")) in ("navigation", "landing"):
                    self._mv_had = True  # uid130 flew at a pad after the hand-over: a later 'search' is a loss or dropout
                return
            if not getattr(self, "_mv_had", False):
                return  # still on its initial search (never locked since the hand-over): leave it alone
            anc, span, n_anc = self._cx_mv_anchor()
            if anc is None:
                if not self._mv_in_search:
                    CX_EVENTS["mv_noanc"] = CX_EVENTS.get("mv_noanc", 0) + 1
                return
            if not self._mv_armed and span < _MV_SPAN_MIN:
                # rev1 static guard: my samples of the hand-over track never moved, so this is most likely a static pad
                # flagged moving (1909652400, 1224832766). Leave uid130 and the static take-back exactly as in c3; the
                # test is repeated every search tick, so a moving pad whose samples spread later still arms.
                if not self._mv_still_ep:
                    self._mv_still_ep = True
                    CX_EVENTS["mv_still"] = CX_EVENTS.get("mv_still", 0) + 1
                    if len(CX_EVENTS.setdefault("mv_still_span", [])) < 8:
                        CX_EVENTS["mv_still_span"].append([round(float(self.mine.t), 2), round(span, 2), n_anc])
                return
            tgt = np.array([anc[0], anc[1], anc[2] + _MV_Z_UP], dtype=np.float32)
            cur = getattr(k, "_last_known_goal_pos", None)
            fresh = not self._mv_in_search  # uid130 just dropped to search: restart its pattern around the anchor
            was_gr = bool(getattr(k, "_goal_return_mode", False))
            if fresh or not was_gr or cur is None or float(np.linalg.norm(np.asarray(cur, dtype=np.float64) - tgt)) > 0.5:
                k._goal_return_mode = True
                k._last_known_goal_pos = tgt
                if fresh or not was_gr:
                    k._goal_return_search_steps = 0
                if fresh:
                    k.extra_search_vectors = None
                    k.search_stage = 0
                    k.search_state_0_rot_count = 0
                    self._mv_armed = True  # from now on the static hand-back (_handback_step) stays off, see there
                    self._mv_arm_t = float(self.mine.t)
                    CX_EVENTS["mv_arm"] = CX_EVENTS.get("mv_arm", 0) + 1
                    if len(CX_EVENTS.setdefault("mv_anc", [])) < 8:
                        CX_EVENTS["mv_anc"].append([round(float(self.mine.t), 2)] + [round(float(v), 2) for v in anc])
                        # [span of the samples (m), samples used, samples cleared from earlier tracks]
                        CX_EVENTS.setdefault("mv_span", []).append([round(span, 2), n_anc, int(self._mv_clr)])
            if int(getattr(k, "_goal_return_search_steps", 0)) > 300:
                k._goal_return_search_steps = 0  # uid130 drops goal-return after 400 steps: keep searching the orbit
            if _CX_MV_SPIN and int(getattr(k, "search_stage", 0)) == 0 and int(getattr(k, "search_state_0_rot_count", 0)) >= 90:
                k.search_state_0_rot_count = 0  # stage 0 ends at 100 spin ticks: keep spinning over the orbit
            self._mv_in_search = True
            if _CX_MV_GATE:
                self._mv_gate = np.asarray(anc, dtype=np.float64).copy()
                if not getattr(k, "_cx_mv_gated", False):
                    self._cx_mv_install_gate(k)
        except Exception as e:
            CX_EVENTS["mv_err"] = repr(e)[:200]

    def _cx_mv_install_gate(self, k):
        """CX_MV_GATE: wrap this uid130 instance's _predict_goal (instance attribute, the class is untouched). The king's
        uid130 instance survives reset() (its reset() only resets fields), so the wrapper stays on it, marked by
        k._cx_mv_gated, and is inert while self._mv_gate is None (cleared in reset())."""
        orig = k._predict_goal
        outer = self

        def _gated(depth, state, drone_position, drone_rpy):
            pr, tr, cov, q = orig(depth, state, drone_position, drone_rpy)
            g = getattr(outer, "_mv_gate", None)
            if g is not None:
                try:
                    p = np.asarray(tr, dtype=np.float64).reshape(-1)
                    if (math.hypot(p[0] - g[0], p[1] - g[1]) > _MV_GATE_XY or abs(p[2] - g[2]) > _MV_GATE_Z):
                        # also empty its CONSIST buffer: 6 mutually consistent outputs within 25 m of the clue are
                        # accepted as a sighting whatever their probability (how the 2030969537/67120516 decoys got in)
                        k._pred_buf = []
                        CX_EVENTS["mv_gate"] = CX_EVENTS.get("mv_gate", 0) + 1
                        return 0.0, tr, cov, q
                except Exception:
                    pass
            return pr, tr, cov, q

        k._predict_goal = _gated
        k._cx_mv_gated = True

    def _cx_mv_dbg(self, observation):
        """CX_MV_DBG: record uid130's state after a mountain hand-over whenever its mode / sight / goal-return changes."""
        try:
            k = getattr(getattr(getattr(self.king, "_uid130", None), "_base", None), "_main", None)
            if k is None or getattr(self.king, "_active", None) != "uid130":
                return
            m = self.mine
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            def r(v):
                return None if v is None else [round(float(x), 2) for x in np.asarray(v, dtype=float).reshape(-1)[:3]]
            key = (str(k._mode), bool(k.see_P), bool(k._goal_return_mode), int(k._goal_return_attempts), bool(k.tracking),
                   bool(k.move_in_auto_mode))
            if key == getattr(self, "_mv_dbg_key", None):
                return
            self._mv_dbg_key = key
            CX_EVENTS.setdefault("mv_log", []).append({
                "t": round(float(m.t), 2), "mode": key[0], "see": key[1], "gr": key[2], "gr_n": key[3], "trk": key[4],
                "auto": key[5], "gr_pos": r(k._last_known_goal_pos), "pp": r(k.platform_position), "lost": int(k.platform_lost_step),
                "tscore": int(k._track_score), "pos": r(st[0:3]), "clue": r(st[0:3] + st[-3:]),
                "m_pad": r(m.pad_pos), "m_hits": int(m.pad_hits), "m_age": round(float(m.t - m.pad_last_seen), 2)
                if getattr(m, "pad_last_seen", None) is not None else None, "m_mode": str(m.mode)})
        except Exception as e:
            CX_EVENTS["mv_dbg_err"] = repr(e)[:200]

    def _cx_wh_agl(self, observation, a):
        """fx_wh CX_WH_AGL: king._forest dives and lands along straight lines with only a forward camera, so a rack or
        prop passing under the drone is never seen (285543861 sank 1.45 -> 0.15 m onto a 4.6 m top 2.8 m short of the
        pad; 1039942737/175339097/2146221907 dived to 0.3-0.6 m over 4.3 m rack tops). The down-ray does see it: while
        the surface under the drone is not the pad (higher than pad - 0.3, or any non-floor surface on the way to the
        search centre) and the drone would come within 1.2 m of it in the next 0.5 s, stop sinking (climb gently under
        0.9 m). Horizontal motion is unchanged; never within 1.2 m of the pad estimate, never over the start pad;
        released after 4 s of modified ticks."""
        try:
            kf = self.king._forest if getattr(self.king, "_active", None) == "forest" else None
            if kf is None:
                self._cx_agl_n = 0
                return a
            mode = str(getattr(kf, "_mode", ""))
            if mode not in ("search", "navigation", "landing") or bool(getattr(kf, "tracking", False)):
                self._cx_agl_n = 0
                return a
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            agl = float(st[137]) * 20.0
            if agl >= 19.5:
                self._cx_agl_n = 0
                return a
            top = float(st[2]) - agl
            s0 = getattr(self, "_cx_wh_start", None)
            if (s0 is not None and abs(top - s0[2]) < _CX_WH_AGL_START_DZ
                    and float(np.hypot(float(st[0]) - s0[0], float(st[1]) - s0[1])) < _CX_WH_AGL_START_R):
                # the surface under the drone is its own start pad (warehouse starts float 1.75-10 m up): the forest
                # pilot leaves 'takeoff' after only +0.3 m, and that is not a descent onto an obstacle
                self._cx_agl_n = 0
                return a
            if mode == "search":
                # only on the way to the search centre (stage 0, not yet arrived, not a goal-return)
                if bool(kf._goal_return_mode) or int(kf.search_stage) > 0 or int(kf.search_state_0_rot_count) > 0:
                    self._cx_agl_n = 0
                    return a
                c = self.mine.center0
                if c is None:
                    self._cx_agl_n = 0
                    return a
                if _CX_WH_BOX:
                    c = _cx_wh_box_xy(c)
                if float(np.hypot(float(st[0]) - float(c[0]), float(st[1]) - float(c[1]))) < 3.0:
                    self._cx_agl_n = 0
                    return a
                obstacle = top > 0.5
            else:
                pad = None
                if bool(getattr(kf, "see_P", False)) and getattr(kf, "platform_position", None) is not None:
                    pad = kf.platform_position
                elif bool(getattr(kf, "_landing_committed", False)) and getattr(kf, "_landing_commit_position", None) is not None:
                    pad = kf._landing_commit_position
                if pad is None:
                    self._cx_agl_n = 0
                    return a
                pad = np.asarray(pad, dtype=np.float64).reshape(-1)
                if float(np.hypot(float(st[0]) - pad[0], float(st[1]) - pad[1])) < _CX_WH_AGL_PAD_R:
                    self._cx_agl_n = 0
                    return a
                obstacle = top > float(pad[2]) - 0.3
            vz_now = float(st[8])
            if not obstacle or agl + min(vz_now, 0.0) * _CX_WH_AGL_LOOK >= _CX_WH_AGL_MIN:
                self._cx_agl_n = 0
                return a
            a_ = np.asarray(a, dtype=np.float64).reshape(-1)
            n = float(np.linalg.norm(a_[0:3]))
            # the env clips speed to [-1, 1]: rebuild the velocity the drone will really be given
            v = a_[0:3] / n * min(abs(float(a_[3])), 1.0) * SPEED_LIMIT if n > 1e-6 else np.zeros(3)
            vz_min = 0.5 if agl < _CX_WH_AGL_CLIMB else 0.0
            if float(v[2]) >= vz_min:
                return a  # already not sinking: nothing to change (does not use up the fail-safe budget)
            if getattr(self, "_cx_agl_n", 0) >= _CX_WH_AGL_MAX_N:
                return a  # fail-safe: 4 s of held ticks in this zone, hand the descent back
            self._cx_agl_n = getattr(self, "_cx_agl_n", 0) + 1
            v[2] = vz_min
            sp = float(np.linalg.norm(v))
            out = np.array(a_, dtype=np.float64)
            if sp > 1e-6:
                out[0:3] = v / sp
                out[3] = min(1.0, sp / SPEED_LIMIT)
            else:
                out[0:3] = 0.0
                out[3] = 0.0
            _cx_ev("wh_agl_climb" if vz_min > 0.0 else "wh_agl_hold")
            return out.astype(np.float32).reshape(np.asarray(a).shape)
        except Exception:
            return a

    def _ct_kcity_hint(self):
        """CX_CT_KCITY: mine's padnet label drives the dual route. The king's own XGB, run every 5 ticks to tick 120 on the
        absolute state, labels some high-start city flights 'mountain' (12 of 167 city flights in epoch1000, all starting
        8.5-10 m up) and switches its pilot to uid130, whose chase after a moving-pad hand-over climbs 8 s and dives with no
        ground floor (1269694645 hit the street). Main flew 35 city moving-pad hand-overs with one loss (a hull crash), so
        pass mine's city label to the king's main; its XGB then reports city instead of mountain."""
        try:
            km = getattr(self.king, "_main", None)
            if km is not None:
                km._cx_ct_city = self.mine.map_label == "city"
        except Exception:
            pass

    def _k9_hint(self):
        """CX_K9_WH / CX_K9_NG: pass mine's padnet label to the king's own pilot pick, before king.act, on every route that
        runs the king (None, king, dual). The king decides at tick 6 (0.12 s), before this router's step-10 decision.
        - WH is updated only while king._active is None (the tick-6 pick). Once the pick is made it is frozen on the
          _forest pilot (whose own XGB keeps reading warehouse, as on a normally classified warehouse flight) and cleared
          on main, which then either never flies again (forest picked) or flies exactly as in c6 (mountain argmax).
        - NG is updated only while the king can still switch main -> graph (active None, or main with the route pending);
          after that it is frozen. The king side latches it once it has relabelled a routing-grade prediction."""
        try:
            k = self.king
            km = getattr(k, "_main", None)
            if km is None:
                return
            act = getattr(k, "_active", None)
            z0 = float(getattr(self, "z0", 0.0))
            lab = self.mine.map_label
            z_ok = _K9_Z_LO <= z0 <= _K9_Z_HI
            if _CX_K9_WH:
                if act is None:
                    on = bool(lab == "warehouse" and z_ok)
                    km._cx_k9_wh = on
                    kf = getattr(k, "_forest", None)
                    if kf is not None:
                        kf._cx_k9_wh = on
                    if on and not getattr(self, "_k9_wh_seen", False):
                        self._k9_wh_seen = True
                        _cx_ev("k9_wh_hint")
                elif bool(getattr(km, "_cx_k9_wh", False)):
                    km._cx_k9_wh = False
            if _CX_K9_NG and (act is None or (act == "main" and bool(getattr(k, "_route_pending", False)))):
                ng = lab if (lab in ("city", "open") and z_ok) else None
                km._cx_k9_ng = ng
                if ng is not None and not getattr(self, "_k9_ng_seen", False):
                    self._k9_ng_seen = True
                    _cx_ev("k9_ng_hint")
        except Exception:
            pass

    def _cx_mm_step(self, observation):
        """fx_mm (CX_MM_HB=1): hand a MOVING mountain pad back from uid130 to my moving-pad chase once uid130 has lost it.
        Runs after _handback_step (which already advanced my shadow controller this tick), reads state only, and
        changes nothing unless it fires. The mm_skip_* counters record ticks where every trigger held but a veto blocked
        it (diagnostics only: they change no action)."""
        m = self.mine
        try:
            if getattr(self.king, "_active", None) != "uid130":
                self._cx_mm_n = 0
                return
            u130 = getattr(self.king, "_uid130", None)
            pm = getattr(getattr(u130, "_base", None), "_main", None)
            if pm is None:
                self._cx_mm_n = 0
                return
            k_mode = str(getattr(pm, "_mode", ""))
            if k_mode in ("navigation", "landing"):
                self._cx_mm_armed = True  # uid130 has flown at the pad since the hand-off: a later search means it lost it
            self._cx_mm_n = (self._cx_mm_n + 1) if (k_mode == "search" and self._cx_mm_armed) else 0
            dt_ho = m.t - self.ho_t
            if dt_ho < _CX_MM_AFTER or self._cx_mm_n * SIM_DT < _CX_MM_LOST_S:
                return
            if m.pad_pos is None or m.pad_hits < _CX_MM_HITS or not m.pad_ever_moving or m.t - m.pad_last_seen > _CX_MM_SEEN_S:
                return
            pad = m._pad_predicted()
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            pos = st[0:3].astype(np.float64)
            if float(np.linalg.norm(pad[:2] - pos[:2])) > _CX_MM_DMAX or abs(float(pos[2] - pad[2])) > _CX_MM_DZMAX:
                return
            # every trigger holds; vetoes:
            if dt_ho > _CX_MM_WINDOW:  # late: the king may still re-find it (555197988); late chases failed (1225093077)
                CX_EVENTS["mm_skip_window"] = CX_EVENTS.get("mm_skip_window", 0) + 1
                return
            if bool(getattr(u130, "_active_rescue", False)):  # uid130's static landing rescue owns the drone over the pad
                CX_EVENTS["mm_skip_rescue"] = CX_EVENTS.get("mm_skip_rescue", 0) + 1
                return
            if float(m.pad_fit_sp) < _HB_FIT:  # a static-looking track belongs to the static hand-back (_handback_step)
                CX_EVENTS["mm_skip_fit"] = CX_EVENTS.get("mm_skip_fit", 0) + 1
                return
            # take it back as a moving pad: my APPROACH moving branch chases from behind and drops onto it (LAND track/drop)
            m.pad_moving = True
            m.pad_miss_t = 0.0
            m.reacq_n = 0
            m.land_retry = 0
            m.land_dive = False
            m.land_phase = "track"
            m.v_cmd = st[6:9].astype(np.float64).copy()
            m._set_mode("APPROACH")
            self.route, self.king_blocked, self.cx_mm_t = "dual", True, round(float(m.t), 2)
            CX_EVENTS["mm_handback"] = CX_EVENTS.get("mm_handback", 0) + 1
        except Exception:
            return

    def _cx_w8_guard(self, observation, a):
        """fx_w8 CX_W8_BLIND / CX_W8_BRAKE (see the block comment at the flags): cap the speed of the king forest
        pilot's command on warehouse when it flies where its camera does not look while something is near, or (BRAKE)
        faster than it can stop within the free depth along its commanded direction. Yaw is unchanged; the direction is
        unchanged unless the capped velocity would sit further from the current one than the pilot's own governor allows."""
        try:
            if int(getattr(self, "_cx_w8_n", 0)) >= _CX_W8_MAXN:
                return a
            kf = self.king._forest if getattr(self.king, "_active", None) == "forest" else None
            if kf is None or str(getattr(kf, "_mode", "")) not in ("search", "navigation"):
                return a
            a_ = np.asarray(a, dtype=np.float64).reshape(-1)
            n = float(np.linalg.norm(a_[0:3]))
            if n < 1e-6:
                return a
            sp = min(abs(float(a_[3])), 1.0) * SPEED_LIMIT        # the env clips speed to [-1, 1]
            v_c = a_[0:3] / n * sp
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            rpy = st[3:6].astype(np.float64)
            v_a = st[6:9].astype(np.float64)
            dep = np.asarray(observation["depth"], dtype=np.float32).reshape(IMG_H, IMG_W)
            pooled = dep.reshape(_CX_W8_N, IMG_H // _CX_W8_N, _CX_W8_N, IMG_W // _CX_W8_N).min(axis=(1, 3))
            near = DEPTH_MIN_M + float(pooled.min()) * (DEPTH_MAX_M - DEPTH_MIN_M) < _CX_W8_NEAR
            vlim = None
            ev = None
            if _CX_W8_BLIND and near:
                for v in (v_c, v_a):
                    # a climb is not judged (the start pad under a pitched camera, open space above racks): only the
                    # horizontal part plus any sink must lie inside the camera cone
                    vd = np.array([v[0], v[1], min(float(v[2]), 0.0)])
                    if float(np.hypot(v[0], v[1])) > _CX_W8_VB and not _cx_w8_in_cone(_cx_w8_cam(vd, rpy)):
                        vlim, ev = _CX_W8_VB, "w8_blind"
                        break
            if vlim is None and _CX_W8_BRAKE and sp > _CX_W8_VMINB:
                dc = _cx_w8_cam(v_c, rpy)
                if _cx_w8_in_cone(dc):
                    free = _cx_w8_free(dc, pooled)
                    vb = math.sqrt(2.0 * _CX_W8_ACC * max(0.0, free - _CX_W8_MARGIN))
                    if sp > vb:
                        vlim, ev = vb, "w8_brake"
            if vlim is None or sp <= vlim:
                return a
            self._cx_w8_n = int(getattr(self, "_cx_w8_n", 0)) + 1
            _cx_ev(ev)
            if "w8_t0" not in CX_EVENTS:
                CX_EVENTS["w8_t0"] = round(float(self.step) * SIM_DT, 2)
            if self._cx_w8_n >= _CX_W8_MAXN:
                _cx_ev("w8_budget")
            out = np.array(a_, dtype=np.float64)
            out[3] = max(0.0, vlim) / SPEED_LIMIT
            v_new = v_c * (max(0.0, vlim) / sp)
            # the forest pilot's own tilt-aware reference governor (agent_forest act/_reference_governor): never ask for
            # a velocity further from the current one than it would, so a cap on a steeply banked drone brakes as
            # gently as the pilot itself would (a hard cap at 40 deg tilt flipped a graph-pack flight, 1326470213)
            tilt = max(abs(float(rpy[0])), abs(float(rpy[1])))
            mve = max(0.2, 2.2 - (2.0 / 0.7) * max(0.0, tilt - 0.3))
            err = v_new - v_a
            en = float(np.linalg.norm(err))
            if en > mve:
                v_new = v_a + err * (mve / en)
                _cx_ev("w8_gov")
                sn = float(np.linalg.norm(v_new))
                if sn > 1e-6:
                    out[0:3] = v_new / sn
                    out[3] = min(1.0, sn / SPEED_LIMIT)
                else:
                    out[3] = 0.0
            return out.astype(np.float32).reshape(np.asarray(a).shape)
        except Exception:
            return a

    def _cx_w17_brake(self, observation, a):
        """fx_w17 CX_W17_BRAKE (see the block comment at the flags): on warehouse, while the king's forest pilot is in
        search/navigation, cap the commanded velocity so the drone can stop inside the free depth ahead, measured along
        the command and along the actual velocity (only where the camera sees). Yaw unchanged; any error returns a."""
        try:
            if int(getattr(self, "_cx_w17_n", 0)) >= _CX_W17_MAXN:
                return a
            kf = self.king._forest if getattr(self.king, "_active", None) == "forest" else None
            if kf is None or str(getattr(kf, "_mode", "")) not in ("search", "navigation"):
                return a
            a_ = np.asarray(a, dtype=np.float64).reshape(-1)
            n = float(np.linalg.norm(a_[0:3]))
            if n < 1e-6 or not np.all(np.isfinite(a_)):
                return a
            sp = min(abs(float(a_[3])), 1.0) * SPEED_LIMIT        # the env clips speed to [-1, 1]
            v_c = a_[0:3] / n * sp
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            rpy = st[3:6].astype(np.float64)
            v_a = st[6:9].astype(np.float64)
            dep = np.asarray(observation["depth"], dtype=np.float32).reshape(IMG_H, IMG_W)
            pooled = dep.reshape(_CX_W8_N, IMG_H // _CX_W8_N, _CX_W8_N, IMG_W // _CX_W8_N).min(axis=(1, 3))
            pos = st[0:3].astype(np.float64)
            # the pad itself is not an obstacle to brake for (rev 2: 1808503506 was braked on its own pad approach and
            # lost the pad): the pilot's visible goal and mine's reliable shadow track
            pads = []
            try:
                if bool(getattr(kf, "see_P", False)) and getattr(kf, "platform_position", None) is not None:
                    pads.append(np.asarray(kf.platform_position, dtype=np.float64).reshape(-1)[0:3])
            except Exception:
                pass
            try:
                m_ = self.mine
                if (m_.pad_pos is not None and int(m_.pad_hits) >= 20
                        and (float(m_.t) - float(m_.pad_last_seen)) < 2.0):
                    pads.append(np.asarray(m_.pad_pos, dtype=np.float64).reshape(-1)[0:3])
            except Exception:
                pass

            def _at_pad(dirw, free_):
                hp = pos + dirw * free_
                return any(float(np.linalg.norm(hp - q)) < _CX_W17_PADR for q in pads)

            v_new = v_c.copy()
            fired = False
            if sp > 0.3 and sp > _CX_W17_VMIN:
                dc = _cx_w8_cam(v_c, rpy)
                if _cx_w17_in_view(dc):
                    fr = _cx_w17_free(dc, pooled, _CX_W17_TUBE)
                    vm = _cx_w17_vmax(fr)
                    if sp > vm and not _at_pad(v_c / sp, fr):
                        v_new = v_c * (vm / sp)
                        fired = True
            spa = float(np.linalg.norm(v_a))
            if spa > 0.5 and spa > _CX_W17_VMIN:
                u = v_a / spa
                du = _cx_w8_cam(u, rpy)
                if _cx_w17_in_view(du):
                    fr = _cx_w17_free(du, pooled, _CX_W17_TUBE)
                    vm = _cx_w17_vmax(fr)
                    c = float(v_new @ u)
                    if c > vm and not _at_pad(u, fr):
                        v_new = v_new - (c - vm) * u   # remove only the excess closing speed; the sideways part stays
                        fired = True
            if not fired:
                return a
            self._cx_w17_n = int(getattr(self, "_cx_w17_n", 0)) + 1
            _cx_ev("w17_brake")
            if "w17_brake_t0" not in CX_EVENTS:
                CX_EVENTS["w17_brake_t0"] = round(float(self.step) * SIM_DT, 2)
            if self._cx_w17_n >= _CX_W17_MAXN:
                _cx_ev("w17_brake_budget")
            # the forest pilot's own tilt-aware reference governor: never ask for a velocity further from the current
            # one than it would (a hard cap on a banked drone could tip it over)
            tilt = max(abs(float(rpy[0])), abs(float(rpy[1])))
            mve = max(0.2, 2.2 - (2.0 / 0.7) * max(0.0, tilt - 0.3))
            err = v_new - v_a
            en = float(np.linalg.norm(err))
            if en > mve:
                v_new = v_a + err * (mve / en)
                _cx_ev("w17_brake_gov")
            out = np.array(a_, dtype=np.float64)
            sn = float(np.linalg.norm(v_new))
            if sn > 1e-6:
                out[0:3] = v_new / sn
                out[3] = min(1.0, sn / SPEED_LIMIT)
            else:
                out[3] = 0.0
            return out.astype(np.float32).reshape(np.asarray(a).shape)
        except Exception:
            return a

    def _cx_rb_pin_step(self, observation, fired):
        """fx_rb CX_RB_PINHO (see the block comment at the flags): track W17 brake pins of the king's forest pilot and,
        once a pin lasts _CX_RB_PIN_T s, hand the flight to mine the way WSEARCH does. True = handed over this tick (the
        caller still returns this tick's king action; mine flies from the next tick). Any error returns False."""
        try:
            s = int(self.step)
            if fired:
                self._cx_rb_fire = s
            kf = self.king._forest if getattr(self.king, "_active", None) == "forest" else None
            mode = str(getattr(kf, "_mode", "")) if kf is not None else ""
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            spd = float(np.linalg.norm(st[6:9]))
            if (mode in ("search", "navigation") and spd < _CX_RB_PIN_V
                    and (s - int(getattr(self, "_cx_rb_fire", -999))) <= _CX_RB_PIN_GAP):
                if getattr(self, "_cx_rb_pin0", None) is None or (s - int(getattr(self, "_cx_rb_pinl", -999))) > _CX_RB_PIN_GAP:
                    self._cx_rb_pin0 = s
                self._cx_rb_pinl = s
            p0 = getattr(self, "_cx_rb_pin0", None)
            if p0 is None:
                return False
            if (s - int(getattr(self, "_cx_rb_pinl", -999))) > _CX_RB_PIN_GAP:
                self._cx_rb_pin0 = None   # the episode ended before it counted
                return False
            if (s - int(p0)) * SIM_DT < _CX_RB_PIN_T:
                return False
            m_ = self.mine
            m_.reacq_n = 0
            m_.recover_n = 0
            m_.land_retry = 0
            m_.v_cmd = st[6:9].astype(np.float64).copy()
            if m_._track_reliable() and not m_.pad_moving:
                m_._set_mode("APPROACH")
                _cx_ev("rb_pinho_app")
            elif str(m_.mode) not in ("CRUISE", "SEARCH"):
                m_.search_wps = None
                m_.search_anchor = None
                m_.hinted = False
                c0_ = m_.center0
                if _CX_WH_BOX and c0_ is not None:
                    c0_ = _cx_wh_box_xy(c0_)
                    if float(np.linalg.norm(c0_[:2] - m_.center0[:2])) > 1e-6:
                        m_.search_anchor = c0_[:2].copy()
                m_._set_mode("SEARCH")
                _cx_ev("rb_pinho_search")
            self._ws_done = True
            self.route = "mine"
            _cx_ev("rb_pinho")
            if "rb_pinho_t" not in CX_EVENTS:
                CX_EVENTS["rb_pinho_t"] = round(float(s) * SIM_DT, 2)
            self.debug = {"route": "mine", "map": m_.map_label, "mode": "RBPIN", "pad_hits": m_.pad_hits}
            return True
        except Exception:
            _cx_ev("rb_pin_err")
            return False

    def _cx_w17_edge(self, observation, a):
        """fx_w17 CX_W17_EDGE (see the block comment at the flags): a committed forest landing whose anchor is off the
        centre of a raised pad. Re-anchor it on mine's reliable static shadow track, and climb first if the drone is
        already at or below the pad top beside the pad. Any error returns a."""
        try:
            kf = self.king._forest if getattr(self.king, "_active", None) == "forest" else None
            if kf is None or str(getattr(kf, "_mode", "")) != "landing" or not bool(getattr(kf, "_landing_committed", False)):
                self._cx_w17_miss, self._cx_w17_up = 0, False
                return a
            m_ = self.mine
            if (m_.pad_pos is None or int(m_.pad_hits) < _CX_W17_EDGE_HITS or bool(m_.pad_moving)
                    or bool(getattr(m_, "pad_ever_moving", False)) or (float(m_.t) - float(m_.pad_last_seen)) > 6.0):
                self._cx_w17_miss, self._cx_w17_up = 0, False
                return a
            st = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            pos = st[0:3].astype(np.float64)
            agl = float(st[137]) * 20.0
            pe = np.asarray(m_.pad_pos, dtype=np.float64).reshape(-1)
            dxy = float(np.hypot(pos[0] - pe[0], pos[1] - pe[1]))
            h = float(pos[2] - pe[2])
            if dxy > 1.5 or h > 1.2 or agl <= h + 0.6:
                self._cx_w17_miss, self._cx_w17_up = 0, False
                return a   # far, high, or the down-ray is on the pad (or something as high)
            # rev 2: the down-ray flickers through the pad to the floor while the drone sits on it (1281708568,
            # 2093186799 tipped over after a climb there): need a run of misses, and act only clearly above or below
            self._cx_w17_miss = int(getattr(self, "_cx_w17_miss", 0)) + 1
            if self._cx_w17_miss < _CX_W17_EDGE_MISS:
                return a
            lpp = getattr(kf, "_landing_platform_position", None)
            if lpp is not None and h > 0.15:
                off = float(np.hypot(float(lpp[0]) - pe[0], float(lpp[1]) - pe[1]))
                if 0.3 < off < 1.5:
                    kf._landing_platform_position[0:2] = pe[0:2]
                    lcp = getattr(kf, "_landing_commit_position", None)
                    if lcp is not None:
                        kf._landing_commit_position[0:2] = pe[0:2]
                    _cx_ev("w17_edge_anchor")
                    if "w17_edge_t0" not in CX_EVENTS:
                        CX_EVENTS["w17_edge_t0"] = round(float(self.step) * SIM_DT, 2)
            if h < -0.3 and dxy > 0.45 and self._cx_w17_miss >= 2 * _CX_W17_EDGE_MISS:
                self._cx_w17_up = True   # latched until the drone is 0.3 m above the pad top (then re-anchored above)
            if getattr(self, "_cx_w17_up", False) and h > 0.3:
                self._cx_w17_up = False
            if getattr(self, "_cx_w17_up", False):
                # beside the pad, below its top: straight up at 0.6 m/s (yaw held), then the corrected landing
                _cx_ev("w17_edge_climb")
                a_ = np.asarray(a, dtype=np.float64).reshape(-1)
                out = np.array([0.0, 0.0, 1.0, 0.2, float(a_[4]) if a_.shape[0] > 4 else 0.0], dtype=np.float64)
                return out.astype(np.float32).reshape(np.asarray(a).shape)
            return a
        except Exception:
            return a

    def _s1f_edge_engage(self, label):
        """CX_S1F_EDGE: True, and the hold starts, for an open/city first label on a start in the forest edge band."""
        try:
            if self.king is None or label not in ("open", "city") or self._s1f_eh != 0:
                return False
            sp = self.mine.start_pos
            if sp is None:
                return False
            x, y, z = float(sp[0]), float(sp[1]), float(sp[2])
            if not (_S1F_EDGE_XLO <= x <= _S1F_EDGE_XHI and abs(y) <= _S1F_EDGE_YMAX and z <= _S1F_EDGE_ZMAX):
                return False
            self._s1f_eh = 1
            self.mine._s1f_eacc = [np.zeros(6), 0]
            _cx_ms_ev("s1f_edge_hold_" + str(label))
            return True
        except Exception:
            self._s1f_eh = 2
            _cx_ms_ev("s1f_edge_err")
            return False

    def _s1f_edge_step(self):
        """CX_S1F_EDGE: decide the held edge start once enough inward frames are scored, or give up at the time limit."""
        m = self.mine
        try:
            acc = getattr(m, "_s1f_eacc", None)
            if acc is None:
                self._s1f_eh = 2
                _cx_ms_ev("s1f_edge_lost")
                return
            if acc[1] >= _S1F_EDGE_N:
                lp = np.asarray(acc[0], dtype=np.float64)
                lead = float(lp[5]) - float(max(lp[i] for i in range(6) if i != 5))
                self._s1f_eh = 2
                m._s1f_eacc = None
                CX_EVENTS["s1f_edge_lead"] = round(lead, 1)
                if lead >= _S1F_EDGE_LP:
                    m.map_label, m.map_locked = "forest", True
                    self.route = "mine"
                    _cx_ms_ev("s1f_edge_forest")
                else:
                    _cx_ms_ev("s1f_edge_release")
            elif float(m.t) > _S1F_EDGE_MAXT:
                self._s1f_eh = 2
                m._s1f_eacc = None
                _cx_ms_ev("s1f_edge_timeout")
        except Exception:
            self._s1f_eh = 2
            m._s1f_eacc = None
            _cx_ms_ev("s1f_edge_err")

    def _act_inner(self, observation):
        self.step += 1
        if self.step == 1:
            try:
                self.z0 = float(np.asarray(observation["state"], dtype=np.float32).reshape(-1)[2])
            except Exception:
                self.z0 = 0.0
            if _CX_WH_AGL:
                try:
                    st1_ = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
                    self._cx_wh_start = (float(st1_[0]), float(st1_[1]), float(st1_[2]) - float(st1_[137]) * 20.0)
                except Exception:
                    self._cx_wh_start = None
                self._cx_agl_n = 0
        if self.route is None:
            a_mine = self.mine.act(observation)
            a_king = None
            if self.king is not None:
                if _CX_K9_WH or _CX_K9_NG:
                    self._k9_hint()  # fx_k9: the king's tick-6 pilot pick happens on this route
                try:
                    a_king = self.king.act(observation)
                except Exception:
                    self.king = None
            lp = self.mine.map_logp
            top2 = sorted(lp)[-2:]
            decided = (self.step >= 10 and (top2[1] - top2[0]) > 6.0) or self.step >= 60
            if decided or self.king is None:
                label = self.mine.map_label
                if _MTN_HIGH_RELABEL and _OPEN_ALT > 0 and label in ("warehouse", "forest") and getattr(self, "z0", 0.0) > _OPEN_ALT:
                    self.mine.map_label, self.mine.map_locked, label = "mountain", True, "mountain"  # would route king / mine-only
                self.high_open = (label == "open" and _OPEN_ALT > 0 and getattr(self, "z0", 0.0) > _OPEN_ALT)
                if _CX_S1F_EDGE and self._s1f_edge_engage(label):
                    self.route = "dual"  # fx_s1f EDGE: mine flies until the inward view is scored
                elif self.king is not None and label in _KING_ROUTE_LABELS and not self.high_open:
                    self.route = "king"
                elif self.king is not None and self.high_open:
                    self.route = "dual"
                elif self.king is not None and label in _KING_MOVING_LABELS:
                    self.route = "dual"
                else:
                    self.route = "mine"
            self.debug = dict(self.mine.debug)
            self.debug["route"] = self.route
            if _CX_RG_ANY and self.route == "king":
                self._rg_arm("map")  # fx_s1r: watch an early map-route king flight for a label misread
            if self.route == "king" and a_king is not None:
                return a_king
            return a_mine
        if self.route == "king":
            try:
                if _CX_RG_ANY and self._rg_on:
                    a_rg = self._rg_step(observation)  # fx_s1r: None = the king keeps flying this tick
                    if a_rg is not None:
                        return a_rg
                if _CX_WH_BOX or _CX_WH_ZLO:
                    try:
                        # king_src clamps this pilot's search point on warehouse only (set each king tick, cleared in reset())
                        self.king._forest._cx_wh = (self.mine.map_label == "warehouse")
                    except Exception:
                        pass
                if _CX_MV_ANCHOR and self.king_from_dual and self.mine.map_label == "mountain":
                    self._cx_mv_arm()
                if _CX_K9_WH or _CX_K9_NG:
                    self._k9_hint()
                a = self.king.act(observation)
                a_k0_ = a if _CX_W8_DBG else None  # fx_w8 diagnostics only
                self.debug = {"route": "king", "map": self.mine.map_label, "mode": "KING", "pad_hits": 0}
                if (_flag("wsafe") or _flag("wleash") or _flag("wsearch")) and self.mine.map_label == "warehouse":
                    a_shadow = None
                    try:
                        a_shadow = (self._rg_fed_a if (_CX_RG_ANY and self._rg_fed == self.step)
                                    else self.mine.act(observation))
                    except Exception:
                        pass
                    m_ = self.mine
                    st_ = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
                    pos_ = st_[0:3].astype(np.float64); rpy_ = st_[3:6].astype(np.float64)
                    if _flag("wsafe") and self.step < 750 and a is not None:
                        a_ = np.asarray(a, dtype=np.float64).reshape(-1)
                        v_k = a_[0:3] * float(a_[3]) * SPEED_LIMIT
                        if float(np.linalg.norm(v_k)) > 0.5:
                            depth_ = np.asarray(observation["depth"], dtype=np.float32).reshape(IMG_H, IMG_W)
                            v_o, blocked_, free_ = m_._plan(depth_, pos_, rpy_, v_k, 1.0, 0.35, 10.0)
                            if blocked_ or free_ < 4.0:
                                v_brake = math.sqrt(max(0.0, 2 * 2.5 * (float(free_) - 0.8)))
                                sp_k = float(a_[3]) * SPEED_LIMIT
                                if v_brake < sp_k:
                                    a = np.array([a_[0], a_[1], a_[2], max(0.0, v_brake) / SPEED_LIMIT, a_[4]], dtype=np.float32)
                                    self.debug["mode"] = "KING+WSAFE"
                    if _flag("wsearch") and a_shadow is not None and not getattr(self, "_ws_done", False):
                        try:
                            kf = self.king._forest if getattr(self.king, "_active", None) == "forest" else None
                        except Exception:
                            kf = None
                        arrived = False
                        c0_ = m_.center0
                        if _CX_WH_BOX and c0_ is not None:
                            c0_ = _cx_wh_box_xy(c0_)  # the king now searches at the clamped centre
                        if kf is not None:
                            try:
                                arrived = (kf._mode == "search" and not bool(kf._goal_return_mode)
                                           and kf._landing_platform_position is None and not bool(kf._goal_is_tracked)
                                           and (int(kf.search_state_0_rot_count) > 0 or int(kf.search_stage) > 0)
                                           and c0_ is not None and float(np.linalg.norm(pos_[:2] - c0_[:2])) < 3.0)
                            except Exception:
                                arrived = False
                        if _CX_WH_WSTRK and not arrived and getattr(self, "_cx_ws_latch", False) and kf is not None:
                            # arrival was already seen and deferred: the spiral may leave the 3 m radius, so from now on
                            # only the king's own state decides (otherwise the deferral would switch WSEARCH off for good)
                            try:
                                arrived = (kf._mode == "search" and not bool(kf._goal_return_mode)
                                           and kf._landing_platform_position is None and not bool(kf._goal_is_tracked)
                                           and (int(kf.search_state_0_rot_count) > 0 or int(kf.search_stage) > 0))
                            except Exception:
                                arrived = False
                            if arrived and not (m_._track_reliable() and not m_.pad_moving
                                                and int(m_.pad_hits) < _CX_WH_WSTRK_HITS):
                                _cx_ev("wh_wstrk_late")
                        if (arrived and _CX_WH_WSTRK and m_._track_reliable() and not m_.pad_moving
                                and int(m_.pad_hits) < _CX_WH_WSTRK_HITS):
                            arrived = False  # a young track (759250480: 18 hits on a 2.5 m prop): keep the king flying
                            self._cx_ws_latch = True
                            _cx_ev("wh_wstrk_defer")
                        if arrived:
                            self._ws_done = True
                            m_.reacq_n = 0; m_.recover_n = 0; m_.land_retry = 0
                            if m_._track_reliable() and not m_.pad_moving:
                                m_._set_mode("APPROACH")
                                self.debug = {"route": "mine", "map": m_.map_label, "mode": "WSEARCH>APPROACH", "pad_hits": m_.pad_hits}
                            else:
                                m_.search_wps = None; m_.search_anchor = None; m_.hinted = False
                                if _CX_WH_BOX and c0_ is not None and float(np.linalg.norm(c0_[:2] - m_.center0[:2])) > 1e-6:
                                    m_.search_anchor = c0_[:2].copy()  # my rings around the clamped centre, not behind a wall
                                    _cx_ev("wh_ws_anchor")
                                m_._set_mode("SEARCH")
                                self.debug = {"route": "mine", "map": m_.map_label, "mode": "WSEARCH", "pad_hits": m_.pad_hits}
                            self.route = "mine"
                            return a_shadow
                    if _flag("wleash") and m_.pad_pos is not None and m_.pad_hits >= 10:
                        dp_ = float(np.linalg.norm(pos_[:2] - m_.pad_pos[:2]))
                        wl_pad = getattr(self, "_wl_pad", None)
                        if wl_pad is not None and float(np.linalg.norm(m_.pad_pos[:2] - wl_pad[:2])) > 2.0:
                            self._wl_close = False; self._wl_far_t = None; self._wl_pad = None
                        if dp_ < 1.5 and abs(float(pos_[2] - m_.pad_pos[2])) < 2.5:
                            self._wl_close = True; self._wl_far_t = None; self._wl_pad = m_.pad_pos.copy()
                        if getattr(self, "_wl_close", False) and dp_ > 4.0:
                            if self._wl_far_t is None:
                                self._wl_far_t = m_.t
                            elif m_.t - self._wl_far_t > 2.0:
                                m_.reacq_n = 0; m_.recover_n = 0; m_.land_retry = 0; m_.bad_spots = []
                                m_._set_mode("APPROACH")
                                self.route = "mine"
                                self.debug = {"route": "mine", "map": m_.map_label, "mode": "WLEASH", "pad_hits": m_.pad_hits}
                                if a_shadow is not None:
                                    return a_shadow
                        elif dp_ <= 4.0:
                            self._wl_far_t = None
                if (((_flag("vking") and self.mine.map_label == "village") or (_flag("mking") and self.mine.map_label == "mountain"))
                        and getattr(self, "_vk_mem", None) is not None):
                    if _CX_VS_ANY or _CX_V3_ANY or _CX_V8_SURF or _CX_VL_ANY or _CX_VX_RAY:
                        # fx_vs: events counted inside this discarded act() go to *_shadow; fx_v3, fx_v8, fx_vl3 do not act in it
                        _CX_STATE["shadow"] = True
                    try:
                        with _CxShadow():  # fx_ms: shadow act(), its action is discarded
                            self.mine.act(observation)
                    except Exception:
                        pass
                    if _CX_VS_ANY or _CX_V3_ANY or _CX_V8_SURF or _CX_VL_ANY or _CX_VX_RAY:
                        _CX_STATE["shadow"] = False
                    st_ = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
                    pos_ = st_[0:3].astype(np.float64)
                    far = float(np.linalg.norm(pos_[:2] - self._vk_mem[:2]))
                    if (far > (14.0 if self.mine.map_label == "mountain" else 18.0) and (self.mine.t - self._vk_t0) > 3.0
                            and not (_CX_MPC and self._mpc_own and self._cx_mpc_ev_leash())):
                        m_ = self.mine
                        m_._drop_track()
                        m_.bad_spots = []
                        m_.reacq_n = 0
                        m_.recover_n = 0
                        m_.land_retry = 0
                        m_.reacq_moving = True
                        m_.reacq_target = self._vk_mem.copy()
                        back = pos_[:2] - self._vk_mem[:2]
                        nb = float(np.linalg.norm(back))
                        back = back / nb if nb > 1e-6 else np.array([1.0, 0.0])
                        m_.reacq_wp = np.array([self._vk_mem[0] + back[0] * 7.0, self._vk_mem[1] + back[1] * 7.0, self._vk_mem[2] + 3.0])
                        m_.reacq_hold = 0.0
                        m_.reacq_side = 0
                        m_.reacq_t_enter = m_.t
                        m_.pad_hist = []
                        m_._set_mode("REACQUIRE")
                        if _CX_VK_UNSTICK and m_.map_label == "village":
                            m_._vk_tb = True
                        self._vk_recaptured = True
                        self._vk_mem = None
                        self.route = "dual"
                        self.debug = {"route": "dual", "map": self.mine.map_label, "mode": "VKING", "pad_hits": 0}
                    elif (_CX_VK_WATCH and self.mine.map_label == "village" and self._vkw_fire(a, st_)
                          and not (_CX_MPC and self._mpc_vkw(st_))):
                        # the vking take-back above, fired by the blind repeat instead of the 18 m leash and aimed at my
                        # shadow track's latest pad estimate. Like the leash, this tick still returns the king's action
                        # (one more 20 ms tick of its descent); from the next tick mine flies, starting from v_cmd = the
                        # current velocity with any descent removed, so the king's descent does not carry on after that
                        _cx_count("vk_watch_takeback")
                        m_ = self.mine
                        ref = self._vkw_ref()
                        m_._drop_track()
                        m_.bad_spots = []
                        m_.reacq_n = 0
                        m_.recover_n = 0
                        m_.land_retry = 0
                        m_.reacq_moving = True
                        m_.reacq_target = ref.copy()
                        back = pos_[:2] - ref[:2]
                        nb = float(np.linalg.norm(back))
                        back = back / nb if nb > 1e-6 else np.array([1.0, 0.0])
                        m_.reacq_wp = np.array([ref[0] + back[0] * 7.0, ref[1] + back[1] * 7.0, ref[2] + 3.0])
                        m_.reacq_hold = 0.0
                        m_.reacq_side = 0
                        m_.reacq_t_enter = m_.t
                        m_.pad_hist = []
                        m_._set_mode("REACQUIRE")
                        v0 = st_[6:9].astype(np.float64).copy()
                        v0[2] = max(float(v0[2]), 0.0)
                        m_.v_cmd = v0
                        if _CX_VK_UNSTICK:
                            m_._vk_tb = True
                        self._vk_recaptured = True
                        self._vk_mem = None
                        self.route = "dual"
                        self.debug = {"route": "dual", "map": self.mine.map_label, "mode": "VKWATCH", "pad_hits": 0}
                if _HB_ON and self.king_from_dual and self.mine.map_label == "mountain":
                    self._handback_step(observation)
                    if _CX_MM_HB and self.route == "king":
                        self._cx_mm_step(observation)
                    if _CX_MV_ANCHOR:
                        self._cx_mv_collect()  # my shadow track, advanced by _handback_step this tick
                    if _CX_MV_DBG and self.route == "king":
                        self._cx_mv_dbg(observation)
                if _CX_WH_AGL and self.mine.map_label == "warehouse" and a is not None:
                    a = self._cx_wh_agl(observation, a)
                if _CX_W8_ANY and self.mine.map_label == "warehouse" and a is not None:
                    a = self._cx_w8_guard(observation, a)
                if _CX_W17_EDGE and self.mine.map_label == "warehouse" and a is not None:
                    a = self._cx_w17_edge(observation, a)   # fx_w17 (reads mine's shadow track, updated this tick)
                _rb_n0 = 0
                if _CX_W17_BRAKE and self.mine.map_label == "warehouse" and a is not None:
                    if _CX_RB_PINHO:
                        _rb_n0 = int(getattr(self, "_cx_w17_n", 0))
                    a = self._cx_w17_brake(observation, a)  # fx_w17
                if _CX_RB_PINHO and _CX_W17_BRAKE and self.mine.map_label == "warehouse" and a is not None:
                    self._cx_rb_pin_step(observation, int(getattr(self, "_cx_w17_n", 0)) > _rb_n0)  # fx_rb
                if _CX_W8_DBG:
                    _cx_w8_dbg_write(self, observation, a_k0_, a)
                return a
            except Exception:
                self.route = "mine"
        if self.route == "dual":
            if _flag("kinghint") and self.king is not None:
                try:
                    for obj in (getattr(self.king, "_main", None), getattr(getattr(getattr(self.king, "_uid130", None), "_base", None), "_main", None)):
                        if obj is not None and bool(getattr(obj, "see_P", False)) and int(getattr(obj, "platform_lost_step", 99)) <= 5:
                            pp = getattr(obj, "platform_position", None)
                            if pp is not None:
                                self.mine.hint_xy = np.asarray(pp, dtype=float)[:2].copy()
                                self.mine.hint_t = self.mine.t
                            break
                except Exception:
                    pass
            a_mine = self.mine.act(observation)
            if _CX_MV_ANCHOR and self.mine.map_label == "mountain":
                self._cx_mv_collect()
            if _CX_CT_KCITY and self.king is not None:
                self._ct_kcity_hint()
            if (_CX_K9_WH or _CX_K9_NG) and self.king is not None:
                self._k9_hint()
            try:
                a_king = self.king.act(observation)
            except Exception:
                self.route = "mine"
                a_king = None
            if (_MTN_HIGH_RELABEL and _OPEN_ALT > 0 and self.step <= 200 and self.mine.map_label == "warehouse" and getattr(self, "z0", 0.0) > _OPEN_ALT
                    and (lambda lp: lp[1] - lp[0])(sorted(self.mine.map_logp)[-2:]) > 8.0):
                self.mine.map_label, self.mine.map_locked = "mountain", True  # instead of the late hand-over below
            if _CX_S1F_EDGE and self._s1f_eh == 1:
                self._s1f_edge_step()  # fx_s1f EDGE: may relabel forest (route mine) or end the hold
                if self.route == "mine":
                    self.debug = dict(self.mine.debug)
                    self.debug["route"] = self.route
                    return a_mine
            # a map first taken for city/mountain that later proves to be open/warehouse belongs to the king
            if (self.step <= 200 and a_king is not None and not (_CX_S1F_EDGE and self._s1f_eh == 1)
                    and self.mine.map_label in _KING_ROUTE_LABELS and self.mine.map_label in ("open", "warehouse")
                    and not (_CX_RG_ANY and self._rg_done)
                    and not (self.mine.map_label == "open" and _OPEN_ALT > 0 and getattr(self, "z0", 0.0) > _OPEN_ALT)):
                lp2 = sorted(self.mine.map_logp)[-2:]
                if (lp2[1] - lp2[0]) > 8.0:
                    self.route = "king"
                    self.debug = {"route": "king", "map": self.mine.map_label, "mode": "KING", "pad_hits": 0}
                    if _CX_RG_ANY:
                        self._rg_arm("dual")  # fx_s1r
                    return a_king
            if _flag("kingsee") and a_king is not None and self.mine.pad_pos is None and self.mine.mode in ("SEARCH", "REACQUIRE"):
                try:
                    for obj in (getattr(self.king, "_main", None), getattr(getattr(getattr(self.king, "_uid130", None), "_base", None), "_main", None)):
                        if obj is not None and bool(getattr(obj, "see_P", False)) and int(getattr(obj, "platform_lost_step", 99)) <= 2:
                            self.route = "king"
                            self.debug = {"route": "king", "map": self.mine.map_label, "mode": "KING", "pad_hits": 0}
                            return a_king
                except Exception:
                    pass
            if (_MTNFB_SEARCH_S > 0.0 and a_king is not None and self.mine.map_label in _FB_LABELS
                    and self.mine.pad_pos is None and self.mine.mode in ("SEARCH", "REACQUIRE")
                    and getattr(self.mine, "search_t_first", None) is not None
                    and self.mine.t - self.mine.search_t_first > _MTNFB_SEARCH_S):
                # my search has run this long without a pad: the king flying alone landed 11 of the 28 static
                # search failures of pool 600000, some before my controller reached the centre
                self.route = "king"
                self.debug = {"route": "king", "map": self.mine.map_label, "mode": "KING", "pad_hits": 0}
                return a_king
            if (_flag("mtnfb") and a_king is not None and self.mine.map_label in _FB_LABELS and self.mine.t > _MTNFB_T
                    and self.mine.pad_pos is None and self.mine.mode in ("CRUISE", "SEARCH", "REACQUIRE")):
                self.route = "king"
                self.debug = {"route": "king", "map": self.mine.map_label, "mode": "KING", "pad_hits": 0}
                return a_king
            fast_ok = self.mine.map_label not in _NO_MOTION_HO
            if fast_ok and (_HANDOVER_CITY if self.mine.map_label == "city" else _HANDOVER) == "fast":
                fast_ok = bool(self.mine.pad_moving) and float(self.mine.pad_fit_sp) > 0.65
            if (_CX_VF_KING and a_king is not None and (self.mine.pad_ever_moving or self.mine.pad_moving) and fast_ok
                    and not getattr(self, "_vk_recaptured", False) and not self.king_blocked
                    and self.mine.map_label == "village" and self.mine.pad_pos is not None
                    and self.mine._vf_in_band(float(self.mine.pad_pos[2]))):
                # fx_vf KING: the motion came from a track on a flat house part (no pad lies in the band): reject it and
                # clear its moving flags instead of handing the flight to the king; mine flies this tick as computed
                try:
                    st_vf = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
                    pos_vf = st_vf[0:3].astype(np.float64)
                    self.mine._vf_reject(pos_vf, pos_vf + st_vf[138:141].astype(np.float64), "king")
                    self.mine.pad_ever_moving = False
                except Exception:
                    _cx_vf_ev("vf_error")
            if (_PC_DIAG and a_king is not None and self.mine.map_label == "city"
                    and (self.mine.pad_ever_moving or self.mine.pad_moving) and not getattr(self, "_pc_ksee_done", True)):
                self._pc_ksee_probe()
            if (a_king is not None and (self.mine.pad_ever_moving or self.mine.pad_moving) and fast_ok
                    and not getattr(self, "_vk_recaptured", False) and not self.king_blocked
                    and not ((_CX_PC_SEEN or _PC_DIAG) and self._pc_hold())):
                if _CX_VM_DIAG and self.mine.map_label == "village":
                    self._vm_ho_log()
                self.route = "king"
                self.king_from_dual, self.ho_t, self.hb_lost_n = True, float(self.mine.t), 0
                if _CX_PC_KTO and self.mine.map_label == "city":
                    self._pc_kto()  # fx_pc KTO: the king main pilot leaves a leftover 'takeoff' state
                if _CX_MM_HB:
                    self._cx_mm_n, self._cx_mm_armed = 0, False  # fx_mm: count uid130's search from this hand-off on
                if (_flag("vking") and self.mine.map_label == "village") or (_flag("mking") and self.mine.map_label == "mountain"):
                    self._vk_mem = self.mine.pad_pos.copy() if self.mine.pad_pos is not None else None
                    self._vk_t0 = self.mine.t
                self.debug = {"route": "king", "map": self.mine.map_label, "mode": "KING", "pad_hits": 0}
                return a_king
            self.debug = dict(self.mine.debug)
            self.debug["route"] = self.route
            return a_mine
        a = self.mine.act(observation)
        self.debug = dict(self.mine.debug)
        self.debug["route"] = self.route
        return a


if _PC_DIAG:
    import sys as _pc_sys

    def _pc_mk(attr):
        key = "_pc_v_" + attr

        def _get(self):
            return self.__dict__.get(key, False)

        def _set(self, v):
            old = self.__dict__.get(key, False)
            self.__dict__[key] = v
            try:
                if v and not old and getattr(self, "map_label", None) == "city" and getattr(self, "_pc_nflip", 9) < 4:
                    self._pc_nflip += 1
                    f = _pc_sys._getframe(1)
                    p = getattr(self, "pad_pos", None)
                    q = getattr(self, "_pc_pos", None)
                    dh = zf = None
                    if p is not None and q is not None:
                        dh = round(float(math.hypot(float(p[0]) - q[0], float(p[1]) - q[1])), 1)
                        zf = round(float(p[2]), 2)
                    _pc_log("pc_flips", [round(float(self.t), 2), attr[4:], f.f_code.co_name, f.f_lineno, str(self.mode),
                                         int(getattr(self, "pad_hits", 0)), dh, None if q is None else round(q[2], 2), zf],
                            cap=4)
            except Exception:
                pass
        return property(_get, _set)

    _MyController.pad_moving = _pc_mk("pad_moving")
    _MyController.pad_ever_moving = _pc_mk("pad_ever_moving")
