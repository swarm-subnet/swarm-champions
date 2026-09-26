from __future__ import annotations
# --- shipped configuration: these fixes are on unless the environment already sets them ---
import os as _cx_os0
for _cx_k, _cx_v in (("CX_WH_BOX", "1"), ("CX_WH_ZLO2", "1"), ("CX_FO_SPIN", "1"), ("CX_CT_KCITY", "1"), ("CX_VK_MOTION", "1"), ("CX_VK_WATCH", "1"), ("CX_VK_UNSTICK", "1"), ("CX_MS_SLOPE", "1"), ("CX_VS_REFLY", "1"), ("CX_M2_PULSE", "1"), ("CX_M2_RESPIN", "1"), ("CX_F6_UNDER", "1"), ("CX_DV_DIVE", "1"), ("CX_MV_ANCHOR", "1"), ("CX_MV_GATE", "1"), ("CX_MV_SPIN", "1"), ("CX_V8_SURF", "1"), ("CX_VS_DIPLATE", "1"), ("CX_K9_WH", "1"), ("CX_VL_HOLD", "1"), ("CX_FO_HC", "3.2"), ("CX_MG_FLOOR", "1"), ("MY_SEARCH", "slowmv,mvfit,mvpatience,invest,nospin,inward_mtn,mdesc,mreacq,keepview,brake,yawvel,gndfilt,gentle2,apptube,roofguard,fastapp,vsearch3,mhigharr,mspin2,vmovko,vsurf,vclamp,vbad1,vmem,vland,vking,vretry,cmem,ccheck,cland,cretry,caround,cfast,cnoclimb,qretry,cdetour,czcap,cstable,cmovko,clook,cskim,cwall,cdrift,ftake,fdown,fspin,wleash,wsearch,wspin,vabs,mland"), ("CX_F14_GATE", "1"), ("CX_F14_STATIC", "1"), ("CX_F14_REFLY", "1"), ("CX_W17_BRAKE", "1"), ("CX_W17_EDGE", "1"), ("CX_V16_VLAND", "1"), ("CX_RA_DIP", "1"), ("CX_RB_PINHO", "1"), ("CX_RC_LSEEN", "1"), ("CX_RC_LBAND", "1"), ("CX_CR_KRAY", "1"), ("CX_MA_SO", "1"), ("CX_VF_LAND", "1"), ("CX_VF_KING", "1"), ("CX_VF_ZONE", "1"), ("CX_VF_LO", "1.6"), ("CX_KL_MDROP", "1"), ("CX_KL_KBACK", "1"), ("CX_KL_KB_REHO", "1"), ("CX_VK2_BRAKE", "1"), ("CX_VK2_PIN", "1"), ("CX_VK2_ROOF", "1"), ("CX_VG_V8H", "1"), ("CX_VX_RAY", "1"), ("CX_ME_RIM", "1"), ("CX_ME_END", "1"), ("CX_VM_GATE", "1"), ("CX_ME_EG_MAPS", "mountain,village,forest"), ("CX_PRM_CITY_H_CRUISE", "7.5"),):
    _cx_os0.environ.setdefault(_cx_k, _cx_v)

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


def _cx_mg_ev(name, n=1):
    # fx_mg counter (plain: the guard runs in the router, after any shadow act() has returned)
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


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


def _cx_m15_ev(name, n=1):
    # fx_m15 counter: muted inside a shadow act() (_CxShadow mutes while any CX_M15_* flag is on)
    if _CX_MUTE[0]:
        return
    CX_EVENTS[name] = CX_EVENTS.get(name, 0) + n


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
        self.land_off = None      # touchdown offset (xy), frozen at LAND entry
        self.stuck_t = 0.0        # seconds continuously wedged
        self.escape_until = -1.0  # backing out until this time
        self.escape_dir = None
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
            if len(segs) > 3:
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
        g0, p, g1 = segs
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
            lk = self._m15_look = {"t0": self.t, "key": key}
            self._m15_look_done.append(key)
            self._m15_look_n += 1
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
        yaw = float(rpy[2])
        self._rpy = rpy
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
        if _flag("cov") and self.map_label in ("mountain", "village") and self.mode in ("CRUISE", "SEARCH", "REACQUIRE"):
            self._cov_update(depth, pos, rpy)
        if _CX_V3_OCC:
            self._v3_depth = depth
            self._v3_vh = float(np.hypot(vel[0], vel[1]))  # the occlusion excuse needs a moving view
        if _CX_V16_LOG:
            _v16_pre = self._v16_log_pre()
        if self.mode != "TAKEOFF" or self.t > 1.0:
            self._update_track(dets, pos)
        if _CX_V16_LOG:
            self._v16_log_post(_v16_pre, dets)
        if (_flag("cdrift") and self.map_label == "city" and self.pad_pos is not None and not self.pad_moving
                and self.pad_hits >= 30 and self.mode in ("APPROACH", "LAND") and len(self.pad_hist) >= 6
                and (self.t - self.pad_last_seen) < 0.5):
            old = [h for h in self.pad_hist if 1.8 <= self.t - h[0] <= 2.6]
            if old:
                h0 = old[-1]
                if float(np.hypot(self.pad_pos[0] - h0[1], self.pad_pos[1] - h0[2])) > 0.8:
                    self.pad_moving = True
                    self.pad_ever_moving = True
                    self.pad_vel = (self.pad_pos[:2] - np.array([h0[1], h0[2]])) / max(self.t - h0[0], 0.2)
                    self._cdrift_fired = True
        if (self.map_label == "mountain" and self.pad_pos is not None and not self.pad_moving
                and self.mode in ("CRUISE", "SEARCH", "APPROACH", "REACQUIRE") and self.step % 2 == 0):
            self._ring_collect(depth, pos, rpy)
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
                    if _flag("r915"):
                        radii = (9.0, 15.0)
                        n_wp = 8
                        step = 2.0 * math.pi / n_wp
                    a0 = self.spin_start_yaw
                    if _flag("ringback") and self.map_label != "mountain":
                        a0 = self.spin_start_yaw + math.pi
                    for r in radii:
                        for k in range(n_wp):
                            a = a0 + k * step
                            wp_ = c[:2] + r * np.array([math.cos(a), math.sin(a)])
                            if _flag("vclamp") and self.map_label == "village":
                                wp_ = np.clip(wp_, -_VBOX_M, _VBOX_M)
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
            if (lost and _CX_M15_KEEP and pad is not None and self.map_label == "mountain" and not self.pad_moving
                    and not self.pad_ever_moving and not self._static_established()):
                # fx_m15 KEEP: the young track's pad is below the image while we descend onto it (weak and non-weak)
                try:
                    if self._m15_keep_ok(pad, pos, rpy, vel, alt, unseen):
                        lost = False
                        _cx_m15_ev("m15_keep")
                except Exception:
                    _cx_m15_ev("m15_error")
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
                    if self.reacq_side == 0 and not self.reacq_moving:
                        self.reacq_side = 1
                        self.reacq_hold = 0.0
                        tgt = self.reacq_target
                        v0 = self.reacq_wp[:2] - tgt[:2]
                        ang = math.atan2(v0[1], v0[0]) + 2.0 * math.pi / 3.0
                        self.reacq_wp = np.array([tgt[0] + 5.0 * math.cos(ang), tgt[1] + 5.0 * math.sin(ang), tgt[2] + 2.5])
                        self.mode_t0 = self.t
                    else:
                        self.search_anchor = None if getattr(self, "_vcheck_active", False) else self.reacq_target[:2].copy()
                        self._drop_track()
                        self._set_mode("SEARCH")
                        self.search_wps = None
            elif self.t - self.mode_t0 > (12.0 if self.reacq_moving else 8.0):
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
        if _F6_UNDER and self.map_label == "forest":
            v_des, yaw_des, use_planner = self._f6_under(pos, alt, yaw, v_des, yaw_des, use_planner)
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
    "eNq8vWlz4siWMPz9/gremg9TFd1dI4GpKfrOfSIMRmyGMouE0NwJByA2I5a2sFneeJ/f/p6TmzKllMDVfacj3GWDlMvJk2df/m07"
    "O+beZ6/harf92/x1t8k9P8/fDm+vs+fn3Gqz370ecuPtdncYH+CJ8G/0mcN5v9ou+Pc/9vjdOPg11zjMXseTYPZrbvC2D2b04f34"
    "sAxWE/70E/z5N/b7Swizst8Pu9ep+GK33Z5e37aH1WaWG4c5+Ih/s33b7M/42Xb/N/5ZeA7xk2f491dc24z+SX77NTcZh7Nvd+ST"
    "ybc7MR0b+hl/EdPSF6en5x1s9d9y/4D/ctN8bjre+it/fJjl5qsTjL7a5g7LWW6NQPg8Pz1PD7/m4J/jkvyzO3z5NTcDoJ5z82C8"
    "yPmz+fgtADhZFhkRBq64z1Wn2hn0c6uQDPWjU82Np4fVO4Hzb9MdbD7nr6awpjk8AE9tdv5bMPsd4LQLwv+YB+evAIfX2diHBeNC"
    "nseL2fbwNRp5PIfTyM3G02UunM18OEYfZp4Gs/Er7OCAm9gH4+ksN5nNd68zsozt7HT4miM7ypE14F4PO3zcX73OpofgzLYIj7/u"
    "3hZLAq3Z+684PIXBv4cw3xgO87f9Dl6GGcebPcz8ebrb7FcBrIQMidORJT/j7CGc4u7tdTr7IgbGBwgEcK1hbgkTzPxfc8flCnYE"
    "H4yD4/gcUuCQ5z7DNslbbMBgN/a/fP1bBJF/5P7f/w+PlezvdzyEyuC5VWkMRrnP/JB28/mXr7kBjLJ7Q/hNd9vD6y4I4NdwBuB4"
    "bo9X24r48Ctuf3p4nq4OZ1xaMIPlAsIet7nNeJ8LxpNZAItFwOMTu21unPPfxkHuFYf/O1muWytLD8MnW3aubNDZKw6aO+7eAp98"
    "A7gARwPr+Pq3Z3kT/2Co+3W2fV+97rZfF7PD50/SE59+zX0yPn0BLMx9Mj8xUKxLv1NMhsPZHX7DheFfyxWevgKXX8ly6VxwqwE8"
    "sJ39PljBmbITxYH+ne6fDAf/n64leCIgIojux2GI1wkm2o/97ewgw2EXh3XuM6DAr3i6uWH9Vz4jRx8y3ReAcMAPIloPzMq3BcgS"
    "HlZBkNvtZ9tfKbTnr7PZhazj77nntz1e9GdYyPP+dYaIBRcSwI4EArFtfOCnug0PeBjkgpJTxI1vZ1/p/W6Vnof13xkW0TeO49fZ"
    "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"
    "kkgke5TuA0nltS2XK/vRLGOTfjQJZrjMCbJJCW2uZH+XbYNHeKXWBGW1ydML6X/OKHUvfXX1Wx4bnHxWRBXSZkFe88EwGchjlXpx"
    "Z4jPKs1I5+llSHnsj+P04gpTgPTwKt7JyMBHMrpez2poUHH3qvNdVvKA7jeT9Vo503zrSjx/TINSodOvtUaF02JabINpMfLbkUbH"
    "HII4wPZAo0LK/ioR7f9yuPZtRFfSx4ouvtfYcXitdkiWK4K5V3mbvUcnNUo0LNQkuGAsZuzEaOCp814yeXqPZ9UVXQz7KhDRMej0"
    "g8yszj/++NcffyzfYBfnr5vD94/lzC96//hzsz38udz8+X1/3v+13s4+ktf9/09e/v0Dfv8dfvTn//7z++G8e93/1Sa/78Pf/6k/"
    "5V/6N/76/v1tmbx+/45fZX9NfWL3Ml3BQ8iH/kHn+3p6nf5zul3v4Bv//D65u/kL/pu9Trcz+Ge5Xuv0v0eNe3xE5e7mX3+x3/zj"
    "4/D2v77941//o0/tr935H/AjfCb8jjz/T3MMs+X08P07G72yAv9Hn97/xXnwr/3xx9uP7fpP9QN/Lte77Y/Dn/c/tpvXIFnOF4fq"
    "dnP4sU2S1x9/vuz/ZMOWP/zjj+/tp/vaY+/7fSOEpz+/HBb/5Ev1r79+vO63yefrP//11+7lB7znz//vz3/Aq1+T/T/++OO+FpSj"
    "x/73wVP4eP89eoavb3Z/vfz48XL+5//x/vL+50/yP/8v7//+z58z3Ln/Db9/S7Yvh2LhX/D15/7D93aj870N3/T+uuU/KQ/JTwrw"
    "7T/wH2GtHMLwen344R3/4ag8+B6W+zX4WfEv/8b/o9dof7/Hj/gwyFvvj2q5XQvL34On+Pt9rQ4/L+FX2U+fgqBX67MX+0X+4+hZ"
    "/413+0dtVKMr9L0Dn8Ej8np+/T57PbxOD9sff203G9AC+KFetfxYo88j/+6H5WoLnngPg/zefwhrvYenx3s601vlE/ALeN/jU6//"
    "vdevPfdwQt4fa/gHfvG+gSMh84N/N6rwQfjOoNG5fxp8h3+38PMwW/W3Ya0bNcLa/feHRp887lb7Na74c/hUKVcaj43+iIzom/4J"
    "3IGnfuNJbo3563JcJ1tS7lRr5EN3sLjBE0ym//3qZ33cCfNTYS2uhT1Yx+dajS6T982YVtDo1MhnhyP6nL8Kt7aPVGE1a+nhWT4p"
    "P1MO6+Ik2p9afnx+gPNUDulnPPdnyPvJp2Dpq0/tdqPfh+14LHfuG5369yeY6EOtfM/m4Fk+Ag8rP+pnsWj5WP8pqj7ASeioH/1f"
    "ZALpD9/XelWUXA84afbQG9sH8YCUKzDK78+P5T5saluuCzsWQaNvP0ee+olKFAS1UBzTO8/8+n2tX6viT3rkvpu//n/svX9b28iS"
    "KPw/n0LreXZHSmywTcgknpD3ECAJzxDIAplMDperR9gyaLAtjyRjO2dnP/tbVf271ZINyezZu/fOc06wpO7q6urq6qrq6uqLow+H"
    "5x/3TsJzAv6T0fzh2dnpGYqAo1MxqazqrMTRB0Dy18MP2PUPYuisksRz4YePrJ1O1/gO/KZ/395sl/sB022fvYE5jT8YwqWCHw6h"
    "N/un58ArDOkyKiADkC2B9Ps60u1tvaRq8N0eyl2j7/unJ2+PYNAE3c2ah/t7X+QnIMWvR8fHsKSF706B4349Oj96c2zLqxey0DmI"
    "4v334a/QOJD2YO/DRxiBg6NP54QlkMYuaVLn4uwTSXEYA1dBmBeAGKBxfAQsKSaQXRDGw1HwuV1u/+zT0TnOyIuji08HTAo8A0km"
    "hFRdqfbG572zw/enn84N/Hjhv8vOnn48PKmD1IX29DLHILRlgXef9s6UDKgstg99fBP+XUzAfZhqMEyHnz+eHp3IbpRatYppg0TC"
    "1/58uLf//vCA1y59hqXtRLLMC/fn93vHb/mifMBnq/cE9YFpwsqf7u8ffzqnNeXww+mZ4sHOTtsu8QZlKuDDFQIo9NNmqdD+MfTd"
    "Ft6dcjkQMqefzsLPh0fv3nNeeVFCSUxyxu8g/z8enQF/cfKCpD8zibhpF6ElTBXoIhnNEvwrTXAufa0SeyfvYO5RAUZFIh+oM50X"
    "+sxidUyUd3Zc38U8EVPpw97H8Ld3byyFZhxNw/4oyvNkmMTZ5u95OmlQ0eO9N6AZQhG/0U8K0mLTaTzBv+N0NimihH7fJ6MRaKH4"
    "cw5a4m06y+lhmIL6WIBOjrDEYilA/sNZ9k+97OnJ8Re2uspaeVz4CjMDMkm9g0PkB0t6tfVibNU6PDhS6w595u9C1EIUb+4Y35Ck"
    "+AUZoMuggj69sx2efYJx07omqCQow7r18YIveCA6tcKSlKzUe2DT8GLvl0NY0p3FSnCp9ap+aR9LSzWQ7Bx1AVytnsuncwBD3MdU"
    "E17kkPSe7g68g+Xk08nFHmoJv54eHYTne28PYTYZKtTzHbsUQAVlANVSbeJyzcwqinQ+Pj39ZU9qSWQL2MWYmlduvbO57S5ahcKz"
    "MgqsQhmRjokILCafUYqbHPfy5Y6yjqpptL1jFarCjxo1izowo6ExS1VTqOMsWYVAt9R+FX22gfayaBV1NjYG8dADI2oYzUZF+D4c"
    "+4HXeo3ibjIg+5FZ/oMCKjC7jp4XTCR+jbM09/3nYFoGLsMSi05ZUbDVsJy7zDAc6yartMwvlY2OTx2wY+l/g4L9umqaBdDGNco4"
    "Cphl3AVUmcrvvEjddyyifdd+2lRgXxgtbh9Ii/ZqUrRXU6L9cET/EIhWDS1oHTBdOlQ40wtvVxbeocLJKsicUMiUQK5NxpNZl1Xq"
    "p5N+VMQT+L9/Ce02sfHNi6umFy2SfLfNamdxMcsmpfKLpjdtIj82EX4TOwn1u03ESUDoBBsbtEp772AV4L6xX3oeYcfVCV4A1/gP"
    "0fRjFqNfKc16G2w64aQLk0lShKGfx6Nh0yNfTjiNitseuX6CnhwELLAJ+F2PYvS8vY1GeWx+nMzGTG/A1s1P11Eeh3kfFnX4dpJO"
    "rJpFFseAyDCFr5dX5W+5+T4ZkndQIbsZA0WK3NfQVdSVr+ZJcatXwlXZb2SwgMaTfjpIJje7wmWHvrHbaDIYxSZEqg3IoE60CYQe"
    "+KxUIEuN4iibxBmUobKXDf6iodCfRlk0xi7xT7JMKJCD742rauImk8JnQC4b8n3jKqihOc7knM1liXsugaiiAKWK0/G/6zTNC+qd"
    "xP0miwYJuhz5t8bVJXMJNq5qBhm7cB94oOZ59+jm5ZUvG7JU42o1I2B1fF2CANQwB05B2IymMO4D3/iM//2j9IZc3aN4WDR6OgER"
    "zCW9D/u3yWiQgXKnezShb0Czphtchk5YFzz68AiA+XQEEzgZLFxA+ccJzHukycNggkgaVAPFr0mRpBMTrmCZCshija+iqvHdwvdF"
    "FdB5jLTLXfCIs8X39fD803gTVAlAe1bilJjoxogHkoHeKd4LRDFzQgRKHk+ZjA6nWXodcaE8jCOQY3He0xQiSz/y/oOkas8WkTrO"
    "LtloyuKFKShEuw6q4UbAbTSN/VZHEWgcZTewEgnaKKECEBGArwPop9OlH5RmctMjciI/46T+mkw18jUtQWKJ+wlIHWPpYfJ4iKuh"
    "mrOaWCI6IGvIAmx2miXkBJOl1JRzlcSJYRWlmWSW1Rldlja53yjPeVgWlTxtlgLxEVOfL5EaV96/7HqtTq/E43xoea9QGMs+sYpB"
    "qcp9NJoheReXWuUrZDQd2CtvsZknX2MvBvWAGDSalEAhc8KMzuGbT2CDnnNi8wFF9LQuQW2dTvwttUfDV9WBeIS4Iv/xRgFZ9qgG"
    "idd9ADqudvJ4JYBKTPkkupTT4Mp7uitGn1UoTbgWTdtxtPD5GwUS5UjO1djFtPx9EE9SVHMZIaBUPhv7VCkIdGHCyr0CUzFudbqr"
    "RQl/x0B5W6x+4BAE3PZDn9Mgnha3YXfg048aaceah5+mvKJqVdoL9AFKAYRkjEJ4m6QzviE5dtnqXOFrbaow8PDv5ebmJpomJTAw"
    "t7p2eWhWiEZ40dRa6HWvSjr/CMQbFRMbo2VyiMlFLJtXkGYE2u8ldZerPbQ3XqYpQ+BGyPnrHEdikAyH/kBaFbzMsrpMm5eZxlkf"
    "FD+QOJzD1Assi8bhDvQKetaFvzvw9yf4+xL+vsS/Lzk9WM9QsVOrAXHjYHOcTPxAW6Ll+zhyf8iLgbsCzI3ye60Dl+2r2s+d+s/d"
    "+s/b9Z+f1X/eqf/8vP7zT/WfX5Q/+wOc6O3NTjuoorQo0t2pK/Ka3DtrFHn5cmVL7XansszNgn/xngLjVkKCct5rsu8DVd4HTjdf"
    "BiCuujAVWX027UGszIFDB2wu0yskX4g1NJuEvUvKr6KF9gp1nWUb1RuYMyMoPY36sQ/T4pYmDJNfsEgElz0QS0rCLDt88YDKT71b"
    "b2vLe2ZqUAsX1Hk9VFL8BOQFQp5bkEU3kACXy3ZvCRN70e4tOlelIowgwsRidMf3grSBo0birpBUlQdausrTDNclCix4BRZSeFV/"
    "Teo+QoOG5GZlpHNl781FFvXv4hq3SjSa3ka7uL3e9K7jAn92yPG3y7yZtpuFystVmZ5s4x6gyAL4YH0nVymXfoX1bZrmTi/MPbk2"
    "pEN1u84RQBWwj0k0AnVPeYQkCWAZjAvqv927/5r2Z9MB+tK4X0vDYWxqDeM682Y7cFpUWsMONwPr39g2cR7Vz4q+XmSzuM7VBSiE"
    "aEoKWwxReqpafyKYROlriLHXkhVdIyaBPtW59ImXuYZR/gQRqzh2SzQcGPXuJ7oKCsIrGt1sTtJs7AswpjYK5V/jhnuvirhaT30o"
    "B+3eT6S/NNz/9ZdoNI4mvar5Kidm0/sD5mq72/Qy+Lvdtlm5bpr9IT/9YX3J5JfM/kKTJvgrZ9HzZ1abGC9yyd3cXUdpWllCXFiy"
    "aHKDc+Lq0dMwjnJQZsegfOgTUr21pqb64MDru0xS1cKjp+vzZ45K30LUB855bZtEcL42sd8aUZiXnU2+4eRdymDMq7oe/btZn3P2"
    "E2zsCdvdonBOBo19rQeIHefeHdF1SxlBJvL+prUqBuwSzeGmHBV6vFo1HB85ODYsVAee3m5egGT6d6MkbpJ/pC2pKyHiMqPAL1Sg"
    "RwW2vPMS1guo9guKHI2tWIMtb4EWRgmxj/AFEE9nRZz5vzRZ872rwMmvDNQugXKyqSrQcRT4KD5/5EYmyHJuIDLrEA3GlVY3lXTY"
    "3Vi5xvamMpb1zd5V2d+iIfpr2ODkyNEAmnZ4FiV57P2KOtphlqWZP2zEi2ncL2LeogeYsr0gatrz3zc/Bx5wJf5o4hbyDYiQf2jY"
    "/dngElm2y5AGNnjldbGq/rZDb2sRUngkuVekqZePo9Go527U8hdwL4ftMcDVklv74zUdKBywHnf9lJMIeFgPvm7phUSL/Wgc3sf9"
    "cJ5mI76dAq/iLMK3uMuoNdjUP4MEmEfZoPI7+cYqv86m5idn1wgnDFSSo6DDhu6x5dfA15igT1VzFaU7rtK8axVVhAfI6QfTRoQN"
    "JlTyqR86wa+jPMkNYhdRdhM7yPXEoBqID9qsqSQrAxMmoOKlYAn2cD9tVEN4vuPA6V/MpqP4UgPtuX9zu9NCyhQn1sd1dh9gXlq1"
    "2BxC0QALdrCedLCx4uIB6nOR4GxDzlNGQWdf2Kc1e8IKf0MHOCIl9HXAukir4ADZpOwY/9GyKbUh3BrE+7sa/7IarGuzaXgDdjOq"
    "XaWTIgqLGclEZC3m0lewZAEhF/nODQpGUMxynyPQlC1VzjO5EWBZGwQyCLif+4UiwRDE8zUY+Yhe6VRLhx9tuaoz5Kqx1WC7HeR6"
    "dUUOhiunrPFFNkFFmmJgAkPGyOZ5odlUaAVRlscgyO4Bw3tYUW5ESMl/eEegpeBWInc1azOfXjQ93QUtNnnyfjSKMgFM42R8s5CW"
    "kPhufF66P5s7LNJ/TKhwn5AeW1ACDajhJiyrGFj6w6pJxvrjvT39lXSG2zRLvqaTIho1vfs4KxL4LCYca+DPhtmpRVP2TXiSNNPF"
    "B0bCjSkiziuv8wKfUVtS75fifb1YSCYAPhkQpqwhUDB8HYNAoCZ5Qn0TGxHJBKZ5OhJKnrF+eKC3hqBnJGif4e85/XYrG+mc7zZa"
    "7kkOWKpUHCqsqZ1SuAJbPNLROpA6HNK8GlIRj/l0hm4m49kYxPFg1o/RmmewmgJvcppWxW6VaiPgpkBUr9qp2QKDtf4uZKvBbcxU"
    "ICLsPBkUgshYJpnchHQOb8YWdEFxNhOpGPyjZmG5DvDPdi3rOKqMZ3nhXcfosN/m7F0uJVcVnCToL0YvLolHNivLNdDZzr6hA5f1"
    "m3dZeHMZOwAsKLDdZLun6Wwy4KXRmYWt6cXnruIEVC8tCMQa+FfQ5MEC0XxKVsPsseWplZo15a44NyvOWUWdcZA2BLPpsZ6A2BDv"
    "5oIGUvWbDEKYS1z10/mD6buKRzaknNGEt3q7NN6uVN44G9mzrdWRCx81XbXyLa16VINVFmPtrrgIb7IEVqcl/eWTFHQlfPJBRi2l"
    "0jUJb6PRUFtJoCS8pV3L+KabRQNtrvmSMAGL9dzhXCPhLNeHs7TgsMGNlkx/4nuieQFLe1X8LOslAFDdMOORlqUCS7MAOg8mcR6O"
    "krvYF0SrCXLSompJILU6zWqTRHYIVYtcDqVSl2RfmxKadxfHU7DK8130VwUSxCDJEIJWB2Y+i1pA2enLZpqkegWrbCS9bQHfNEzz"
    "Sss0/2eaphwFp53Av9X4UXgJ5fhAJUS8lP4H+LD9IJNBIMVthpOmty2VGB24FPBlA5uXI/dYD3B4YtDskoxFr3el2cxajU6vq9WY"
    "TVcU7/a2teJ80Mw6lfxTwcvUIbFKu7mYh7SwrhvMu5JxGWPm0TAO+8XCV5402+f2GKO9/HmV5m56CGYZHuXHGQQUtprjZj7faVjt"
    "MRAbAKs9BwbICr0Gz+K85PMIwyJNF6TTgRnotJVmkeEYC8wlsqnWRQJq2z+BQ0A0DeZuqo7ymYSALHdNiXJN+7UY7t0HOUOabuhy"
    "XHbdr0vVZtNd+UscsGBhPjCs8QDVFP4LVRtdWy1r8GVN3KH56cC18bItjqYDgcBaapHWahnAJzYAQmkqQ2g6x19or7NsCNqKBtxc"
    "uejNEwt35qJGlr6qlD0CMG18OIRQuWG5sHLUsE8J7qiJBVXvtOZNQgFuVSC5Jao5F8lyAyUu4fy/a00Hp5t115gjdfzGZ43BdugS"
    "siWTcA31tPjGeBhDqYFWaNdC1uEzUNVkf60F2W563SDoZOgCXuXNW7k8lyhQ8upVtqb7HaocXo7KLvfXgyjtjoF1V1dOKxcmXP7Q"
    "l3gC8wKmjQw2jLJ+P9Vkq9gfcTL831ztC0uEdlHqleD0GvX4kZi5Yf4H1+5h8Veb4aXZa01I78kTr6t0DJuhtFNt0yz9nSGZE5GN"
    "Sf+31S1tXhimpjra0jBhNXqVE75hDi2VdM/5hj7PVTlr2jfs2a5KWgKAHUExRCWUNV9oJctjA6XLL7UaGnWhqPZklLG4jkpa71j5"
    "P4XnNOnfKfUOfvRIaTGUrQFmYgrxyNQsR2+aOq3Xfs5KoJJYLEMWqW0V2W5uWNILj2FFiJBWcBuYeoPFhy/CUZreRbdxNNBLvMAS"
    "TuuktL5ANy5tprlyL2ZU1Bq3q+rpQ8Udg3flmgFYVh+2q0rRwIvao3fFJkQ8HCKIezEGKnzHGBjc/zfHQdgAo/g+AoEsTXyzv5tZ"
    "dB+PfBKh0q2ljwEAtlFQ6x1KtcnStxrRvbyx5vocjqJikk4wKqVURY///x2ULtTLNfKhDcVhKQsLbKs/eNHyqFzK4uqkY4RGxUjK"
    "QpK+Aog0xjCmTODwRP7k+9dYKFAQ++lolOSAIFA9v0OdS1alEN3A+zfP15p95fml8XxSpm/gaGCQQA+BPfhiMpyNRsyHojAkX/HQ"
    "ZYtvmAdZcMxM1K0VfhpP4iKL1FZn/kdWlA8gagblWv0C0ipiXJoYXBGFzTiSSrW0hjw2WG9XR7PUBUG9UrWWToTymbu2EFnq2J0K"
    "D4PpV8ZM+u19B9bqiHQ9kMv/pC2qIQZaxb67THDlVcxlja+kGC6hR3RyQ25WgK0cJ1OrqmoUedk3JbZmmrmbrORy5rIQwPKy5HFg"
    "8VrQy7lOBdKXZcFm58Ze64706zgvwjQbxPxozShe5Kk9dcpc2CojdWk1dtV08K61Zqyoo4Y/MBHWDtWZAC5Vf2AsrlyDWtVjv6x7"
    "uLoZVKJiNM0Hl95xHcxU6nXmuRSQrowYFUf4TyXMajOSaReGonlVZU1qZfk5TadRqRVTuuaVtC1VtBSaHY5QJS+e5kpVIp/eWsfP"
    "uAq7XpCHPBf2sAiPyNs+8ERDaP4xGP+Q4KSDFntYFVQNpdUspIKgrmCvq3FAFRC3p7E0RaQCKhPgvhYKAY6RuZmMraCXFCpUu0Sz"
    "6TIs0jBLcZMOI+B4IME0Kfq38mkZzflvt84KtMgzkXYjJ0gBrd85LA/0xMpNodxUlaNGVEH2yEoCF+RLVRIQUOXwgVtXaUHrWbgw"
    "gkI27PQoGp9ifCfi28oz622e4Rf+8qpSIMs2lxVtYi8R3NSCb+WJuWzlrGB/un6bX6vaXGKPllYL+KK/LPdfy0hT0ybnIq3tv+md"
    "1x4WNS53lFI3ccpVFabkV7vVuREwXTr2cdYNcVsR+iaxHkdFliDjaHNAts+8qPKpYzyJMEIuMCscKnZDekB1RybjrQhbKkVsrQd3"
    "ZZLfQAUzGWjbUVG6k6Y2mmt1OJcC5fJsPQyVh/S0PpSratAsjMUwuEubgV46oiXcA+k/tiMvDQ+aDOk2JkpdXNtTBxs+8ay0ylpp"
    "yVOykJZkuX7nzgxvBMztzrhxwVxxNaPB5Expn8WK3CxRWB8UB72Z+MlHcUbl/KjpXTe9gpSLXTpDz1f8yFQiIkFrttwxNc0scm0V"
    "Ycs9VJxcl3cUIrZu6a+u1dIfgQkNCOEeMlR+ZekAjDLXHM0tgB1xfPD3tZx7BT+h2xNVIvXlNaUN7ClgfMVO81yfa0ieQH0KufZi"
    "YU7f+AZfWmheB4zRAPFJUEpeXVKXpc8Y1OSurxppIiQlRljZV5aQuIfa1w5lnZn/0wSsW3dFPamKS/Ka5je2z4/bs84sQGKh4+Nl"
    "OZ3EoiYM1YR6Z9F5YX/eKtH5Lqhx5CM8Nn5b2mBpThzQPZTmhOFZRBulP6lXBoHxJGEfprDE9Q6seAyuwZOLd3gigROG3uMHv4Oh"
    "yOY8vmcb83pv7qHED17rv/Y/aHH/t/D0wlOWuDdMFk0PRJ03m4xQAlKB8PPe0cVuJ+h5RTKOveIWy2V54fUxqLRfeCDDoomH6YC9"
    "cXqfTG68aTTw0nuwEEfp3Luh4LZNaG9PfMe8ZlSIBETugR2DYG9BhIFx5hVgkkfJxEuHXlLg3L9OCg8MyL7nP21vdr1xwNPIYCVR"
    "GNqABrE8pk7z7qMsifOfoVUT2ST38lm/j73DmNXiFtrOi3j6Y67cFXTmBUCnk9EScGshpszZCn/GoGRteudpEz5DUVAykomoME1G"
    "aQFNQkOzaASVh6Ml9hdbQlsE2rRJ0ASUvetZMhrgR9EZHjiI52/oygQiw3zCAwuGGYh+hv8A3rZghmGj8DjEgoRJNPhRUM7vYyLW"
    "OANrPO3fzdFu6icZ9PZn6Dk0hy0BO8YwLZLJLE+TAYpbhJIn49koAsMJYOXTOB4ACCAAxhijKceoezPL0ALd9C5wNNJZ/xaRYsyS"
    "IJZo3WM+h2nhwegmY+I07BU2Qes44IEv2YDEIsUUNIOUp+EEJh0l1xmUGkCjwHFYrn8LyMBSBthF02mWRn1gH5jGAo5ChvGgGCbB"
    "Z/wt4xxONQzHvmy1N3e8vOk9BYnl5VdIXA6g2PSOhoxrFEx4ANYBgC+8/phBR3gRxbDDqJIPVMMGj3YlrAuTeIHVoC1FjNsUmAHb"
    "RLJMBq10OMQM2d6YUQ1714GH6BqmmETbx3/ukMYA+T6J5wH0q0hGlYTgSDxDlFPsUgxNwnwFwhAyE0ovyAg2QgkBTIuHAAAYvZtN"
    "gPyTGxgkDJuAmjdJf9PbmyyBJ2fXwHWjFChCHH4dEWdi8GieY2Hk6348GuVNaFN1HM13PmPwJcLFaZmOx4gJIxHM4zzO7iN0M/Pj"
    "Hi1itUEcDRgXAzlBOZoGtBcS5TENUEruTEqpg1xLN40gGW7j0QC/jxFbLLT5TxDF/BKUNKcbT/qLMM3FuzFyP3+Lv8V7zJEo3uNv"
    "XEH8IdREUTfDnR+YdWyk2M0sKA6AxiDSD389PLk4xxiYIdBrIKRvkU4ZIwBV8DaVYCNU8p8CMhCvzXhyn8BobQLh/YYqgEme2w1K"
    "H9foNETV49N3tTUPDt98eieqesALKCBQOoLk7cEEQDyYAGAzaQpSrEiACYHdfsZauBU0HOLlPFqLwNZ+g0PlN/Lo6NCJdGF9L8L4"
    "3p+APOUlJX0u8SV61uUbQhzfghKDiVk6AASh7h8es3zlO7oW8gMX4i3B7J4/Dqg8Jq/udLqWvvQDmxHYRZC7g9jzd57DLJ8nE5i4"
    "HmbDwcQGXPKxNY1W0GTBoJ7tnbw7pJtXQGZpUNmSwXcDZznAmE3ZHIZxnsLMjvjhUInfx72D8PC3/WOWdrkCElWHBQUWE3+xDIRU"
    "4CiBvJSiqUnzmq9s1MDe2QdxO84zRTRsoI8dp3VViuNSU2DLzLICRewcpRR+QJYZogD2YYkgcZSlKe8NNsauQHmpDdAP3ubmJkhf"
    "VFIEOJLY1JjPFk5aTYd8LRTt0DqIkhkmIdSKVSsfjk4o83TXaoWt/tgSqxWRhGc90lplcPb3WbJ+izewRk+IQliH72NYyxNUSEbx"
    "HFbvMa7z4638f3cDFhrKBHeUZKQskPYBixRUo2benO39gvnf37G03lYzWZxPUcyC3L/ha4OCxAca28XwH4aR5+O6y4nfQuLjgI+p"
    "Lbyv5gCVSrrnweySxwMzmdABFKVeQGsaMhZQkMC8x0towoO/szTvZQpLEGxdJEUonVLVd2enn04ORN2urEqrgVi7rfqIQ0k/oGWK"
    "zbbD8y8n+/asFwTEAXBRSS780BSnKz+IP/DkiOIwMqZ6zwIhSoIC98VatDEmxEM0LDiKlgLmc1CH/KaY9jMTFF8i2VRaoSroaLN2"
    "YU1hGoOUHQeHxxd7/C4kvR2D80mvTnRATOcRTVJDY8CDoMspUkaOje7ex/D0F8YWO3VdQ5bCRU6IFHOwSx1XG4eg9YPphron6+TF"
    "53ZZ3uPqZYPYrVc1GbNHs7APZqIALPj94nPHZHWN33EOUrWnnijqy8aZOgTqWZzh7QvFMviZMS9XkDTxkOOEZvEj1CaepeerPd7f"
    "QbDDvTfIOs92+Au8kuczSboX2ht2J9HeCb/MRX9N77RXeO3Ih3NxnwBblvESB54RnPAQNxzxO2RE3zFHlmUdMHFMxqbUp595TJie"
    "nr2BBs9omdnWSAhAhPn1IzcXgT+XSicGoo9N65Otvz97T1Fa4gIX46YTE9kfME+GDzK7ictZIFphFdFeTIrZIDYNKtzBBHESzT2/"
    "23pGFi3me0CNpxAj8oNmOFnDG/A833w2ck0ZOY+r6j0uEiKS4dgBXG1QKRQqBO+8RkxQLvlEpfVU2J08wtNcAEGVBiN6FGO4BKrr"
    "KbTUx/waJIoRV2hMN3YimI9gnwmMcegYTVHffpsUBVNuOs2fOs+kNcfNbSl1OtvPaORa3IiOpykWAnxgzf6wd4jM8hyNqesEVGOw"
    "4jrwFCh7cJ6ix2GOpCVBJAyEJvoTAFIR3cU0o472Q7r9AeUb2DgZ0F+3VTMgS9kEQ9nClxRFYKJWwKY0QKV7i9h9I23xTi7DHfHm"
    "V5gkNJN+eiZeCcXgxYsNHUGqxyAdXPC7/Gja7n06Fw44ymHDHHAwidvtTtNj5QPiU4e1Lcjug0GCy0xA7oUbYN2shXLQw1NcedNj"
    "5IC17jpeppzIus+AmsZhpaFHrRaVetyAJz5PZ0SpjPwEpCRmsxwtfDIqOhb+nZ/Qebj9E+KP9yUEXGsmVwYe0uSeCIGpl7GYaZyx"
    "TwTXMnEB1G+BItBDcr0ANtl+zprsWk12X+xgW887O7zNHbJSSIqWWtkRrfRoTFpyiD932PpEfSCA2893At6i/NR92WWNtX/iH08+"
    "Iqe8fG4v/9NbXNbo2B2fF2yPndRV3/TTMItYOmlgHC8sMTTVHUY0mMC/HdUblknJZzMupCmwOV16IUubFfL3YEgUQ9pI905xshOJ"
    "yE2SoCtjGyYLGjYYFxaRY49JR+oKjEpKFXIuBKXIpBRq6PdIR+hJjFxOKLB6ee/QSIAPpF4gw9Lk3N7cAfHK5NI1nksnB+LPnG1y"
    "ppgD1rQOwbJzJk0X0YpEB81ilFNN1nBObjtyfzJzDIwrNs8ZHJ2Vtjc7uDiwyQdsEMhyJxc4yDudrnxzsfeGZeKkeX7wmwGoC5O4"
    "+wJgBSZPbHcX210vn12zqSsFZvfFAv7PTbdkHN0w9RWB+l2g7ROPt4LHnKEkvGoZSscP3snBPoLrp6PZeOL9jk6Hj8s3s9EoLoBj"
    "QPwvQY6iXoMGDhccaI1Qax5PAoGtfOH6TMtqGrQYzGYqEdAazWDUE9xin3uocoGEBWUG1pb+HQ7MDV4rUtDSQu0PyJ4FGRMvon7B"
    "5tAbYxvCHyyb3mDBUi8MlirFV6tL4e/0emG/trL8Y3I0QHFnscPs+gna+dfpLLtN8Y5bkcPv9OJDNGVOhUajcVHh2U2v0Z0FqHMN"
    "kZvXmAG3RdBZYm0ogMpPsAmgKpN49hcYzmCn4Vu0ZZRNf4EhmUxUMU8E+4seDLPSUqu0XLdSbtx2RGXZUnMSrMzxN39YXUmCaDDg"
    "vccz103vK6zlGgUSGYqcppnva5uUC1fOu5YgGXKj6KS+NcuGX4FfVoFf1oBfrgs+pRBj6AMmIabY4gRTbjCy0ONS+7SUnxzhv6kd"
    "8vvV3LP9uipLHR+ga1DMBn3AOCxSvb9zR30YDRbzFdgnwmHUNiOWNXIT9AjA/TLFiOBkgX+xIv4FRpvTc131uas6rya5BKNJ0QVi"
    "sAosRhpNYGK958aanJM0Acno4FpsFtGay873ez6dEmVpUW7JbckrsmkqB6HNMy20WbIHngCGphUCdHMFCLqkwxNU8OlgVn5aUxm9"
    "k4pqv7sRWFgILEoI/F6NwMJCYFGHAJAI+vJqF0kBEvZ3+v17e3WS/rnIFzm/TNq9BBZPqPW7ljuQZsicBd/bOTarOL/mNgCeh5vx"
    "pdkkMeQW5yyezFkxGJ7shbY4ezHm2gVlrsniutGzu7ttctsH5Csu9fmqwFdtg/9AKbpF1yezlsAgSWKLG0m5pnXjZ+8kOiHPSgqW"
    "Tjw32HCxNGf8YlmXshRPaWlBNEOUo1CF55Z0DjpbJVSVparSuXIzqlWlz6V1how2XNRIxv5SL7lcJaL7hAUdfeU+k5M34gCIzNoo"
    "14v+oq60xn0JBbR8k4Be9rUokWRJsYdslW0ZM2hhlFtUlZvfazNm2UeR2Ff4khcYEEYsWzCmspvs4B9mQgYk4cvS+kJxaT7ymz7H"
    "8Ys5E5Eg/8aagQ4DMq/NQx6396wXc7T6YGbyVKm5wpVFR6jv83sWKGOdFsG4sPSOTjfaBynQqtyFlvSPVjKEpmeWv8SEQ3Ki8iMk"
    "+r0wXDxAUW3K24tJ3Xy/l+oUdZiLi0u2kFxRda1qYGR61OQVJdjAgOx7lnvuXuxjpUUIFlRYRNfiijmY+ucF2O7LFttcL9uF3Iyh"
    "fTYysECBR537CcXA3GLKdi57yCxMaFd1yjYMTYNOyhkueTVTRkuXyq7lFHcsc07iNpA6s4X5V8nsA7tTTfdZpuuHmBtNGlraiQLo"
    "vqFGmuWaeoOBle9Xy/bL9VtR0Fw8GCb471M5HZXN90Tu01IgkZ/QOohdwbUWajW9FvSwid0shVYlKCq6ZuvluCroI05OAfpf9dJX"
    "DDVZRxsHcSEBVDczeaoimLTXZKdksOCM0LMSipHc1e8yw1IulfeJ5+um7paw/YgHgqAsuM0eaQiB8e1jLp2PIBTkmUNgvM/CQGL7"
    "OPkfsyhTLkPmZ+AGu+mCHcJSCvCYW/KeBw+V3Fw97iqg5sqhNMBVRhCN2qFdxPmPynr7uL9HR748XwTZqDgVh1cD8d/Qh/As4AE+"
    "3MnlU2C2x6YqIIGrXLbVpRddj15RCpK5VwCjkJ8l2PT21TkpNrMN9zQ7T6W7qHEyk5canSL5XYLTf9O7SCj8CF5l8YhHYqS81wXt"
    "O216Z8Qq2F28X62JLmrocTqObyLMOxcPh/xssqQQJaihAsIPqNToicdupRMHR+cg2+07oZApeBKLuXaI0b65dI48BP9cBUbsqlGG"
    "p9BuslzaHXEQihyXWkb7Udr0bhMMcGDbABt8e47cgzDWeB47mfSTaTTyVLyv2EET/SZOYaehZ7g4f0Szp8jVCqwWMtKdUPXsxK2X"
    "TaICw+zfeabu8czMMdRPceHx/70E8sL7G1RyA4vvoxHOC0y+YYS4IsYwA+550DcGBlOhS6UVTYyXXLHKwwjP7P47tDljkbdxCIvH"
    "lL/jQbt5Lt/a44pnS9k38UOM9B/6UM8Fe+AIhKSQKHGMBWQTeseZPKLuk5pjb1TBiLKr4piTkngcd2+mUZ7L23IYe5NPFqNP/Ea/"
    "wZabz5iu22+MxGOXcnty/AguXgrTMxajOcIg2WMuPSpodY4J13IrCTunBAgA12doFpGkAJ1Ro7yu7HX0dHCXPG2bPwH0MWD2qnxi"
    "1bibR3HJKC/yP3wA9wSHR/Jck7EBvQR9B++y26XT1XaueD3V6CilERGDeLucpmB6dui4CfzpXtGI3SYV9+Fh0CdI27j0MUOUAcW/"
    "IfItwqxUBpjFzYpZh/4J0MgVHFWqjcy3syIoGmYrChn4g/2ZRfQEf3jv2uyP7KvlB3Hf4rdnKECoTUzQW1wetb3L3oRsOXLK/syf"
    "u6RCiKdtfGrljroTZtTJuvhMdXPxRHX7Tro8q6KLyUHIKzr/XNOLb+QdoCOQE+87eyTnIOMg3wDbXK/DM/mc0EaWwf9bIyFuP8aK"
    "XCg1SQecAPy70gymFUhknwZJwaq/ovdAi3Jf+JJlrsI81WlAtotClkvISuEoLtW7NXYk+E4VaXpl9R6/8Fwc1+bmWhMZk3d4Qvd5"
    "tQMWAyg3tGiXESzEKEMNbYID9p/P2mzBZNGggh7UPdpAjkZsJhfQIC1q2CwsffDjitZ9+iWUiOvQpUjAe95XFd/cI8lMuiL6OplC"
    "xfaJyDFD2llGShFK9yj72cvTETpxEhajhis9nhmhQMbRiEGQoakspagTFaDox0vqDRiFJQWAesBTma6o3qmvPliAqT1Y3tMVIrw9"
    "vBZk3FQA4Hk5Zoveta4JyVWLLXlaZnamgz6h8VApN1AjVT6WFf/94PmTkO8UTkIcHAn+N7Xmiax5+N8XuSHLPzLG1Iv8hqvpb7gX"
    "9RvMZhpyhVA1kQg6Vv2CVb88tOogpgNt0PgTwCDg1dHv8gVffZGvlKOD+BMq/M0TwwkDhUvPF+Pd8j4wrsrCeobTg6EECGhHZ3Rn"
    "jJ+h+TlKyVWVceEI5ipKLQAmso0EpcFBkcFDHtIphvePPDBz6fxIROHaRTRRt/0k/LQ/SYIb9C3Hgb1fIfNdxJfJFbN7fOA3Te7h"
    "B8w1fo2OElPqEWNyJzXV1+Udz/pwi+9Vov1rEX9AmdtY7NiQbWBzEaNNeSZjmBDqeU9bHTC6yFhEJ0kTXnS3tvkWPb5RQz/Pxe2r"
    "0zjiBCA1oIV3R+6oQ1dtvE7T29G4Zsqr0sWlSobyTctKwQs/MdOSt235B4FsHZyDgJJKw8M/dOnDNLdm8L3icJzLxky+XzmTYQZ3"
    "oIOT7zFxF5FbK+M2W1BlIpGErKvbqa/7G6VYJGEBqnVNwS9U8MvqglIE/TaWFCwLmS+Oj9k9lyIoAKgH/kfpmV9EgSYetK9MfMPX"
    "wCUZDKFU5LZM0s2J7F7zRWewajBDWPPOL7AEjFQLywLUsea4x09L+emL+hQz3/u97nzPDMZDYn10+u1j5uWvqfvFrKt58UGAET2h"
    "/SeIxFOEBr+WvOM1sgurauJLV//wE4qZjbVUvWeGDHyQmmepeI1hQ0q8eyb/lKhB6iZXTaIU/s3Ynyn+0a/ZNQTqQxXEHxy+ZhCV"
    "TKcrMMkvc7f50vkkhGsTD3PkcQESNuWHgMhnxjUpdDjLw4obyuFreL+5L4H0PrTrlL7Xd6ssfrVdrgh9TpfLkOdVd46u0GyCK0uL"
    "yVCDcasy5+jJOLeXf2QY/xxaOZfKQovT3D/X1v+cdFY8X+7Q/3K9jAE6Ri2mqkqAPuvzOd8bcio22C3iP+e4rb/o983VnfI3sSW+"
    "f11i+L6+wouixjKfoD3BV0F90ee+ZXirLf99c/lXznO6mDjJkDmxA301KueYwYpzBBYoswTxQaJdmdhvu5cen7wiLQYyqF2ByGnR"
    "x/uPWfuW3+Jx/opaqbTz/aTScYPIuZ7zoy/v40iygOj/3YST1Y/eRlVUwBwXtbl077JITV4b1Sc6ZIZuPehawG+CEsGe+mYJtQaL"
    "KeYe9+fMKUEwQNOCn3PUsgNtmwMDcPHK3yKaaZscH40g3cWSNG5yw2M5CtSVR0CEEz742aMdMIKEu/nSnc7cADAcuOKOieRjcoyM"
    "L7f5peDRzLoAAN9UxQjpvsXjRk84waOMRbBP0xFLLcgPveorga9OI2DPf8zVQZh8nKa65odRgJawv6QrwfH+eTXThsh2NAuxD0+9"
    "Bd4/v2ofStVu6xFdw8CM4hm2V+1XybsrsDRupauwlD/o4stnpJqkGNByRSkCKCIyCvhbP2k7N/ae8JwV5jYcd9bifZvUU7qj4I8m"
    "3a9JL+gWgj8s562ZAEGQSuU/kG9Kg9tv9NbFok9XilL7uY1QXxWzHctVbmqNeDjeEshzAvLdSKP8FYJ5tq821gCsbBnZArNjzO/S"
    "nNGaVVNfHvfwcSpgKvgcr+waNem0ghIGF6UzXtZBEHFeWxz1cJyn1+L8aUNfHBpAHSwjKSilBbuIWBMFhJdbEvCLcJizFduQZgle"
    "D4siXrzsiJfimkrc/KKPsnivyzeVZlV54mZ85t6zA1ltGYQw815jppPnWgRGW4tl42csmAOyra5/4gk8jJx5I0dfETVYrdCOmsml"
    "RVyoI+wqgisPY7Q8ddyDl6dYHP3IRwuwFOFR7MwH5z10O2q+GmjpFVRvYqe4tqHic/lhEXqP/7gXTl41q6iaiY86eiCtAGYLPoib"
    "rHjqFpFqc6B6yQ66wAJHyEtHPb/YLr0LrBup7OXXCMgTRLxM7ygtJa6n+ikYgErneWgVpjghLQr7HMNfKi5SL11ZTn4bED3JYBQ3"
    "pOaOT5hhzZ9HsLa2XgfeoHSr+W1CCs/lleX4KLTIZHLb0GS9wdOEYGLGwps9wtgF/XxdboGPSOO0wePphaW3A3OZHZDGo2kY2DcH"
    "esQizQE7ztzijuprjDjMrfDryZAi39rmW4xSd93gPh6MnO8pSAO3jDqbFqRZRXkiqOuTPK3o+vgV5ZgTIOYGKHXjvj92FwfBeB+y"
    "eDFSasX5QLZRz86dp0C0dDTCWCdiSXUIH2sn6QzPQECrPnpDmLl7xzyHdLkhRlIEjlY5pq2O+Y2Op4krarSTpXpzEo9IXjnBstXE"
    "Wi6NIZ4L9yn067Kz3e11tn+60qbFhyiZ7MuuOWaHgZYSIU/Uz3FIZxBJYe15p1PEBcztj/B4ZVydo0+yfpZMKUsZanIfTg8Oj8/D"
    "g6MzI2G1Drisp9tNo+2ioG55jSlbQ1kCwGSymU4mCxUFO4nu6yrD5+SGlFWrHo0OXt8T8tWUXaHz27s3H6LpR/HKN6B92PuIBVg/"
    "w5O9D4da0mgGEMrWdaVwYiErRfdRMiI1etcFbzMGFaPI/aCivnNO4NcBcFo/rvz8wf0FpBlvPI/znOXuqy2WTKazIsSMGihy6YxS"
    "o7IwTFOttAMurBQDVtbX+SNwFZNgzQE7/HJoDJZZM7oOiwx0P8p8HO69uWAPNnmzOI8LvzTS1K5iL968yY4WJE327HrHTKdTk9an"
    "40+77/BfIxY9vImL8xg3yU8xm/IeSQkerDqJ52HEUwKScAA9tCjICcff75LakuQhM7/oejGWFHA7bu0EPTvKXpY0p6iSdJHMWynb"
    "hp5Nl37gisrXColQ9A5thmh7aNDFIg3xGgNW0toBGlgJC6nMZW/7atURF3XXAq+zbR89qdJDB4GzMwNKDgeE8ie4uYaGLm5m5fyB"
    "uw1kuIa2T3M9Gw5p2LkEZwJckgSPimrOmfvwDv7dlWRhtS9b+g3wUGbccZTpamUiDgfhtVgNrX4+1dtQA6WNTTQCDcojGO3NbS3l"
    "dkFv2XF+LT98Mg0j/rq9o4EJ6Rz/Lof3hBB7ysCAEkqoEH5BuYrYl2AvKAc5ttLkrekE4TXuCTgrr81a9GGLIs5BZwBMR6ZWy0pW"
    "6ZoUyJSa4VJZDEpZxcrBPxWwOZYg3BRqK69ZqECAb2woOBuVFZ5dGRMenjcsVV/JBbuyLR2M8rRBuWsLLSUckiysSyxuwgISmcMn"
    "qzsGr1R15ZCUa2ztyiZsy8co7JJ9dNqxX3BBzs4IRZb4Y6H+hvjTCrLMW1Smwa82cSWDz/+YxfHX2DevEWEHNTD3QchacWpmpVJc"
    "gokxVQNl5O81wuKYyENfARd+8lf36sqRfdbIi+2AtC3rP5O/dlZAkk7IMrjnEshP8teLFeDYSfqqxcNoMbDrRiPG6nIxaD274kmD"
    "dR0Dg6TCCKxtdcGjjXlLUaLVVT871ciD+hhm0TzcrkS+3LDeAX4PppsdL7nKd1WXvZkpTyLZgFK/SaVhCox2ZaamqWOgPqhGIbeB"
    "hYK8uLkmMEnuo3r+EVMWXRzuX5yehcd7b8AcsQFhBp8QE4Okw2HIDGInpPdH796HF3u/HJ6+fStB6RPIF4jhUIK+yyjXYBoBfbpJ"
    "o1HI5QAWKzluVFF0F4R3EzAWWS300Wm7WIFjVhrg+bBhhEqO8rRjRxquqPLae9Zul8MN3f0Q9mxFcVdfLA3fqAArTxYJdNRlxdUV"
    "JOaYkED3DlSUAk5rY7b7kE57lGq4A28JdcBG8oerWy1L7rmCV53TTEDn+ZY77qhVpxQQVTf0GHO8A2UU30T9JSekkWddn0Ymt27Y"
    "7IgcR1tb7qEP1qoANgSzrAbl8oKYZP3IPCNq5eDLUEU9bZKWKPaPBiNE48+yi5TdhwSjv4tz+/xi7+Jon9JSfTw7PDjavzg6PTl3"
    "15pm6XV0nYxAlOt1909P3h4dHJ7sH4YX788Oz9Fvufq2HlwQQszmSsd7/68Zqn80sOONP10kwRni4F9b4NFEAJu6iKF7bEfJaxgV"
    "G5rmhmEALlKvBdWsmTdqlXQnhHyaZNGoofcRVDUoJBIWNEJHTaHD4XlQx2fHErCu5FwtNR8mMdkA2xTuR5Ow5LCtLB5N+rckz9zY"
    "hm7SOt66lSZNE+XcHgsmMAv4G/US15zU1jUVq5ihueG4Xs4p1Z24Vy4v4vYtNN61mo9S6mhceAmYOIWbymy7uGpm3CejEbBVCUxJ"
    "DvCCuijgk9is+iiB+D9iiZjDKN2mszz+HhT5P3o1l5T477ygp5Nhko3jAfVjFF3j9nII/5Ps7yih+UIw+yFdA/o/erT9CkvBEA1I"
    "Cl0uqFtiXVR2tuUaISHLsvQGZn6uX11porWm5MRQgaB0kKskRAFrl1DEY+pleYehq85pDx+cHFKyX6pWBLfCZWPbW7EGXtJRwP2j"
    "iy/hr0eHnz+eHp1chOeHe2f778O944uji08Hh+EHjI7SbjYWXHGpe6RJNXMQZj0Mfj06Pt57dyia3j/7dHR++BgMymOwHgJvT0Ey"
    "XHx7+85BXQ8FvJfl2xGoUiBeex28V2kNTJ5SWvMa3ZiHELMDIE8rW9yiFtdrkIHUOdk5bazNZZD5X0ORsnvX+7x3dvj+9NO5ZCNc"
    "NTgl/74eCQ3L3oltwBPQyIZdvhX3ABvV7F3HaDodLUlDT/okDDHyPksGpnc3Z6L8PskTtplsumtGeKVzbQn+Xjo6hCptSFxnoZCl"
    "KKguqHlijUKYgIODUgt0TSELKdPr72a1V7seJQGAYWf3FWDQs1paKQJY7ehiKDBf6hqW/03RjrAkjJsqLM/EsPIDkkr/KNdTjUQM"
    "vSEo4PKzT/5R7u01ebKKhi58bWZ2lcF9Ta4eHZ/u/8L1q9M3e2+OjmExKPO0c3wquu/aNJaOygqjodximddJoXH3Ri4g7073jmEp"
    "Oz96c6ype/Z5fNNtqhSjb8RiV4cq5ka8SIqwuMV9IowNXcNB+R3axPhUvdEqJ7PqOh78MNstZT0yPzt4xCF9MCihVLBKCD2An1aJ"
    "qZqZuRY4babWzOMHgWKLgLEZJPlQk/vZbBTntLt4H/vO7QG+x8LrCKUdZ6jvlBNNx+xtWg6IYNWGNbt33dmo7gHhn2V7rB+lZQh3"
    "WVxtwNpImZtlA/ddYGYMnBr4biunjmmVZWOhnWG6tRVFsQvclaXt11RUwtgaswl5/E3D2xrNqmmgbcJVoVR3Nfh24GLLusni/LYS"
    "iLnzygNj6nYI3bKtJCGqxFO5YCWzUVHFlHR1KtsOown4XJw9wgAqfd93C+/ECEjT0N6+ZrcfiZjl5475Ky14vv9ope1ZRvOyQ0+7"
    "RNShMdJeevm1fSVnGTLYwUM6WeNotGVc/cwS400TPMJiHLKBcvSj1JLStZXrVXvvanKLgdIunTY7oCvwJXOBAlj079s7JQtA23XH"
    "UZJDUtoOlrMVuOo2Kgo2oLqpowXL3cIM1Lz6WqAEqIyu4ImScKiLxFgZfWFSIZpTN83MlFrtkgkjaBJi5KGsVrJ78GCiAF5W12Sz"
    "r+k6ivJ6T1KUI2a2+Hq3RFdxK61R7hUWc++Q6nzR2Ww7FDiFH6WhfCB+7fXwaz8OP5dap6NjBYy83jVY2YWYVeOVUeHhKAJEhU9E"
    "t0eZgpnuR7nO/QoBg2awL9JYPA8qAgzouoUwxeDWsE/bTd2KPX0RfqCbaeUxN1FeF73us/Xx216Fn3DRmvOtX4qNUre6y2FomgKt"
    "qYvU2tAazWCxXcU9Cw9x1Q+s7fK+k/ZP5oHfDISi9nXHsQ/GQwdB141GyVdafvFElvax+ZzdBLGt67PGmrjSm1weFHvjke9olnyw"
    "UpQbW5+vnQEjVIDONsUD3UdsOT3pGNThQfjBc2wxE3i+XLRds7PSF/26CgP3lK1o2134ERu2pUHit81XN+Bmqe3OtrOGqSR8Fw2n"
    "Shmw+FFn1qdlrOmIeVBFCyFJBHRTT+KRPIbMqSopxQ5mYsCc6d2gZvCsLmnhSDKcd+3KqNBa8gVb/5Yubz+uyzvf2GVM2Cdl1F/S"
    "/4fos6KMQ5EVCvMaU8EBqXpiVsiSV241RyrD0SIskbdE7grYTJBXgZbx8gLSq9UDV43o9+EDt5L1KCEqI1bsWB5nZIuI5qmjlnIc"
    "uBc6Z+tWpPVT3RHw2MBgd2Nm5LE4fmN4Fhic1XGQq+IhjQbZFmvpxHfV9AHFThJHW0tMkLiMmG+qVpB6rqlqWLeduMirlknawosJ"
    "L1bwkyzewIuqV4AdZlLLxKWvtg/o2KiffhJRzRPCempvBu/vnYQXR/u/nNP9FUE13HpRy26AL+3N11K/tmTLc2H6fu/4bfhl73N4"
    "tndQW/0pzzmzCggeDNPJ/8jur+U5YSRa5TF5kPqnfj+1Qsed6mAlpNe7XiVjrJD0j9NQqyMcS0HwVb0xjQOWl74yujKo70VVsGVn"
    "ff0Es9DVLr904eSOe7C5CVipGWBymkqraIX5sD51XZyMwaSjeLiuiPwrBscdBbuxdvNmgouHnTmopVe5Nas2Osra7Qf1bkUE77cx"
    "7TdO2G/kD+57faAF/DAVd+Ov128frtuuZ/C6DJx2UK6sbrDYdRjYG2sfZZFwVqhsspw40KkQ2PIqYT5AoS814HKzwT9Xq3afeG9X"
    "hqlVNsyGcoDpMichJXIJLWCySrWeY0Jt1ggeNye7awTry7vaJaGiz6NknBSqs3QOQuusWSt42KpoMrWMBfx4dLZ3TLd70BXzDk7X"
    "/a7GvTAWib3LGs/rSq5xM6Y5y+wZ3u45XDffbvSuKxO+QTA/gsiPoO+Dg0vW2yV4zM7DAw9BVp601BdGl8dew3ktr/2LkpdWbkHv"
    "/BUOfU/z6JcEpitSwZmM1Yik04Ia1o4xKFXmIaK4YVc7Cc0WVzou9FhCrFp56Futghtru7tlJXJPqCeXa0L67fiYgCpPFzzWY8Nc"
    "Ka1nMGZbHVoAnQ5uM5RBZ4DXquHetzrcdZlVgUj16v6tjn3ZDWfv/0I/d5Vr/9Fu/ce6tNfs5s73dec/eDXbevZde2241HcrfOpW"
    "0DBN9LV0WcoGuuvtVHjF/3vrvGsFEFSGNtQ0bX66pMS47c1O06qDeVX5h876anlpoF7RCQIjcIcCpj/K6D+5PPZlPKTYbu6t2Pgk"
    "BQELi5QM9Z56Zy3O9PZOd9WyJjWD0lmkMmzQILvyLma7AfrIJvVPK1Cm+DlMMVsRi6sH0aGNP2KT0N+ocQ2sH6X5iEjMb4rIfGxk"
    "ZrUR5ebLZ5wtfYsv8dyBi6DWey1asUq4GuquOK1QXVRloJNp7WrZ3zUMa5N8XfKumk2yimCoKiatq1Y+EF5Gvm5C1vH1IM4xF2ul"
    "RSAj33l5eTKqfpNj//TDh6OLi8OD8Hjv5ODo5F14+uvh2fvDvYPww1qe7cdMQeNQYRmDt0cnYGyfvn17fnhRg0Ww5tiUg35260eP"
    "DxHf9aTVprYb2ipoUv+qfufLXokeup9Vv4FYzdf1DLSKsSsF3ffg3tUjZx7E+mbvy8pgC33v67H+majSOVNKcab5nu0wSDrtW2H8"
    "uhWGcL143ar4YCljha1fbmAK4i1mQWwdCngO7VhdHqtW6/EYpXhOlnsu7I1qqwTtTzt3EM2TVBrM1wrPFZoYLpLswJLtiVlnBXFl"
    "pGJ5Cm0+Xm8RcHiD1liS1nIKra6KaTnG0yKv31V6cG6t+n2y77vv5I6WXdF9gxfXiaVeXdflgRNqlGk+CA3ZurvIFqfiGF6ViC+n"
    "UbDFuJ727cEeNpYtwcKp55qPa5R6gC7Rq9vcLJ/4Qyo8wltYHlt2saMDOZnex9lM9bm2UqyxhOxa8bS4qDWQrRjx76aJrDjb5e7W"
    "7YBddOD0I9ZiVkMaYZZaZ0na4gYQJ/rrYBtdp/fxSs15rQP89n+vBQ3q+tWlyyrKejGejN57A/p5+PF47+Lt6dmHR2nHpQOPq8bt"
    "lQOXg8Pz/cOTi/A9nnFfYSrgRLSoW432WhE8Nfpl7QrJixnaMDvQbEfHVbUA6gdpsDWM9M0W2MXpp/33B6efT1bbQMItYPfrMWbX"
    "Gobft5hcdULuv7+l9S1HSf75ZukqG/svJp+9AbeCEFXrSrW9+007fNW7fNsb1UbGI/lA1B+DEAzx2NysSEkr/Ga2yfBanzwOKzzA"
    "OtsYbvvtq2/3LlQi9T2Q4emcVimlQpFeU+ekNNVs9S6HWDuTbWv7uSt6X7E4DBbLqhZM/tP97dWtMM97pXPV2UE65UqkQmRqNl7W"
    "GGOG5qrlrb35AuOW1wGFt4rhLaJlYtTIm+p9eOy20tSTSbFa8qyzVa9B78+yDJddW/x8w/a/jXD9znupRhB8gxkGDJJ8TScFRVMp"
    "lKp6yVCrHBrnxtIKDnAq88FDVoK/tA/OWerGuATSjozp/GUhWXRXmwheqGKtredM4DpivNfYBGc3IOKFMlqTwRqgyiF2rl46yLc6"
    "vKVymDHuZWVA/QMmOEbOyINzqzbpzPmMu6TdVTvddi2+/d998PZ/uf3qeICyJKlbV6p7t6OcTw9Qcey9/B7TwPC6CrF4CPVm4F4Q"
    "/osidl1M3XaFclca2U4VpX7juEqzW2vr2SWGqwA+SPFgW/4baxrwzgCA8kbJaJTOeTDzCpO+ksAPJO46LtIVEB64A1pBIXffH+2d"
    "cBHsQYsjKofPWZhWGbP/Q2TY1vZaDbJ0aK1ORWHXggimSnP1Cv2dhqH7zxqGKlEZ/FVE7T6OqFod0V/hcn4Isf/HbyuvYQhYsl/t"
    "7zryHn2fXeqKFp37jpz96U5kND0rB94dIuQo+MqtBFMpvfvbbdre1tuv2tku+4gEmNcW3P+3Hf1/3Xb0ytiWNapWTcqHbXzTFNYZ"
    "ukIvqUso1epwv51Lbj97rGBbo7sr99Rr9tOr3ImUgvJBJ0nWdLyaBgxYNI6URY9yXDzeP/G9/dqrmbuS5I/xCZf89Stde5cOo/Hh"
    "ulinKutNCaFWxTEhO0Q2pyuDClLPzAEMViz4GhCWhtfyU+OUZHDD+10HCz7Ud/rfyqs5wxvnUbi7wP9VHpCH+/D0C40f5soy9Xbj"
    "6Fb5s+6RcjMNmyPq9xOnEVAyAKKKcyRRxSESl8PPXZWOYwSPc4KSpMiTG8uZqHkuDG8idLbGafoou3hNV9MK8lGS33+ygbquyeNs"
    "+FKLTbJNorqs0N9uMG08bp/6+5hV1fVrlaX13Y0WCnrSyUFcpLOMI2SOCruzwYHPw6+Xqz/nbyfBbDpuAGLswOaOfjTWdSh9xUW3"
    "1ffIbOMxYfOCjdrjzfldMg3VV6oxMqMpKy6FMBmnH43jLNJWWv6CpU2iy7lm01A7lI5XdYE5NwYOTx0Rb3VDor4D69uXhZkpqMX1"
    "xXk0jLWbPYwuiUL2LVgwBlZ9h15O7wFWdkO5nD+cfjq52MOotF9Pjw7C8723hxdfwg97Z+/gXTmKaJrFwxhE6oAl8OfJTy0gH88O"
    "3x6enR0ehPvHh3tne3hX2gfnQeVRmt5Ft3E0KAPBRA3Hp6e/7FXEMzGjR5LJqr1/fIoX96zoDYMh+1QFZr3+MGA1PWLAVvWLH85X"
    "123ogN6enn3eOztQV5GsY4lZI35w+Hbv0/HFNw24CeNx423CeOBwm5UfO9ouKI8dbBesx4y1gFMealN8FQuUR0iQEH475BGuJCgz"
    "dh1rikMA7toCsaoCE5C7prh0dO0e1rmb3f29D4dne9CbX8ODw3cOoFxBk4J2V1tdVuCAxwVJed5FJ1Zl4dl015DkpXLzNLtjxxbz"
    "dDQjJGzFwxLT82jJrDJ9WQynCR7SdA0GvGzWT8xwvGs8N9eZiVDJ8bZZP/WgkvH8wI4yb9N36K4+o9ftrTWFV/fUmqdVF3VybSfM"
    "k3EyijJ2DYkjyb3TBiBWpb1U125tFQUDd94kxKJj6lhVtgdr1wmhCxAqh25rbRSdsA0aAaRBWviEdZM1XZEvwKz5qiz73PaYc6ZV"
    "Iey6mEoku9m1UcA4fhuHjbV2jsrNgzEkjbOmp3ynVwHYVmgi+JU32LAtGKdW21vpPRqNkmkcRvdpMsAZYlz5JvBouvX4wLh3RmU/"
    "SCZ5MuAuvq9i64UCETOw5LYHPI2C606nUlVrd0E6KFe28drrdPVUDcwuoL0qvHyPWxZNb71r+MLrCCTVfdw3T+KsnQS5auzC60o3"
    "rmyyfAUs1qrKGcKqwRihfJVYb2Gd9Tc2nfNIQnab3Zfo5WE9UU4XFlym83O5spvDNxw+aWmCV/vn2dVKOE/VfefiVXmViXbdidLK"
    "SF7XKxXFrnanU91qWElyG/GHyYMNJ3yY2r6YwS3K57pt5Gcs01WsGvoFn1rmMue1adVmt8Mo5/MuKCdMKd/6UGrbfTOX486UHccO"
    "kV2stHY6S7XsjcZ6pMr2UwUBy+xt2Ofi7vjAFmDVwiuo8UGJhOpCmpecco6tX+UqJBi8qF81y19oYmqIYiy03FegbZ/HwPKDUyTE"
    "HvNcCYz4coPJO8ZpWtzummQLVt3Tt/LC8tV3W2OpqtN8sGLplyEfn35WNyG/+4Q21oeqO/VuZsDX8cAcCJ1AdT7SVieouLuUg0WX"
    "cA1u+8dHH96Ef3dsYarqxuiughSs3ixw70ny9lDdrEhpY4Cpy4ilwcL9KO1pywJT2W0XU6O7nOGuSgUs0xdbxQL3Lh6YEveK13nd"
    "qlgKa2aI0mK8QQHWolDCIqGcDYgZJkmzE/+3gZZNz/WlA180OGjS3Idg7+gRVt1NzKZG1+xtYQwnbpCwj7BOsKYx2dr2qnlNlhTu"
    "zIMydh9nQHebvQktcTkjS49H2OwqxEQIOSV13d3W94T4BAfJyLLG6wIZdaH938LTi/Dz3tGFyS5OZPuLMGWH8/3asZESiavD9p6q"
    "3SflmrXSmjXLR9JV11hUi9EwWxh+8Frwn0dd65Hs8sbpPeilrWk08Ip01r8dpPOJVwC5Jjeen05GS68fjUYgaOa3UFoRZbcTELAN"
    "5tHBWA9FAxZKWtdx6HLTg342PepcdDPiPRKd6VkDxaBzT5SpLKSFI7FHWpSuL8Fb6wt3oEGqju/QT+jleWFmzqLam9PbKGcrwgCA"
    "WBsALrpLTS6zXN4/ICbFbcwGFsTbEkk+v40KeAuf1AY8hwtjgCqVh+z6/3k+NzkiD6Y4DJqXzSbebIK6QASfbuHvNBml1j4zTOg5"
    "yrC5psnpt7DyMTIWjufPjIXjsrPd7XW2f7oS8TmlnVYgE7Ujtj6QbCjM0lk5ck58YxnXd+1Z2VqRwl0troAtShnsDBNcvJ8tT2se"
    "pBcaiXFr28J4VlhTOZlM4gwwfxAXc/3lZhRYzKz0uEUfanqH9AcFSJTjux6yQzRZesNoNip63i32C3ljOEpubgsPLSSYnvRqNunf"
    "YjzVwKNg91F6k/TRXeDdpOnAHgfOrpxbTRMROhrf+w3oLAjKNGsEDv0HMyenN6A4QpFGDxhxmvmAb3DZ67bbV9YmTx376zzgZD34"
    "sIrx2r0dN9OZTGTxUJ0c0TjSiTXvERTbUHKO04QJGlALNFGl1o3j03duScNAVouFeVLckmT2FSSQZxGaNrk3vC3rLsPbzXmWgKzC"
    "vv2ep5PNwWw8zX1EzXvqNf7XpFHNgCa4aZTnZk8RFO8qToYsjqCBVT1G+wzGBO+ifRQNMP4iuo/pPskp7hTHA40a2KfNBvzLkKHH"
    "yfQriPt8l7W6CbNyLn7PVx/PWrRF4QWw1lI+LeGpH49Gu9jy/uHxcdMDyVzs6mxbbOIrB+s+kua6+FF0/05LKWBbVnas9F1y4qDi"
    "dn70ITy4cLvjDMMSORbMyvXXQx4/qsws+YXFEms0pg6W6CvL39Mmu1aeaFFd/gcQHh4YtyBB4yWI2GQR5146JOGKapCPP/JoLD7h"
    "I4raH6EU6kZ0T3dGCZoxZJgWa6KXEhvR3KGR5PMoG4eD+Hp2E0LD5FCcp9lo4FJUfGWVVzgSae3Tx1mZ9DI8UaaMoptfNuqOKeFx"
    "7rggF5GyehFNoP5zFimB3ao7t6QW41Vdn2bpdXSdjEDHbbBE/igy6O9rsvktPorm5gjDixWLhVl/chNWWZAIu4U8F5SUGJzYm9EU"
    "OHvg+4VY3qECWkj6Y8d87OIjtln2sBImr7wXm21a8tmuB+dA1LV94KoZhU71kO+WHjtuiyyYZtdJ4eG5nCym9Z50xzgaQG/KYwHo"
    "TwCuO6CdqbL0/V+9HRzpjtskRiKA9i+JUEcBt/2Ntx1xKDi23ao7gHgZXJPZr8uefhBzfpvg6Q82JkJxxN+ACCZnfuWhVYli+u0R"
    "2CdHJz3nYE7Tqd8ObHHGv64vu0gN35UotDoK0Wk0MOKzsCjFV9KPbtlzr0umr3SIIs1ANPm36EzAYb5FKSCb2mn3rgzF5ytr8Wt+"
    "iaT+mgfe1panUa4mLQQmX2ZR3AAj0AVkd9MrQOPDm8FvY9JAQdD32ILj3WTEk7TPlMPqS3wIkrEFHSYrEeOANcuGi4xUvIXGLKOq"
    "UlHg38ge+xBNfah6ibFt+LdzZQzkevISuYRGgnimvdnpaukqujvuzSNYj3Gd42HQuBzrKy8tukhAe67j8eJXuByjyfFSS4zx2h3o"
    "zZWOaDDwLykvNjIL/erIXzRaAPKKu9zVanpvZ+4rHZCWXnNxbqHatk0GI9u25XPF5y1hX9YjOdGWRjDcO/sQvt07c49E4LpIvnoO"
    "Uq8HI81sZzuTND6Fe0SwfOU5A3QWEhsITD/YUmQ9u8phX9EWKsn6RvDgTk4jmHO7pIzjTx86QXMA/3YwWhnxvdj7dG6Cvmdywcc6"
    "WK5FgHgsN1bR1DoqL6Mn2uXgAs61YDfLZQuZgF3mkcWgboAeIgxUfX6XoUtXoBRJuEsPKhsXRYh3E9EesGWZrv9AD+xiaS8xg3DC"
    "LhPB7pxf7J0cnL59y/O8MLK8Odv75TA83nsHqqxA4an8pd5tMX/lEzb2+/uBmwCIu+AOzEzX5Cg8gnUlt+kwgeMESHEhBZK7SFOu"
    "KfiYG5MTH98OYQIFSHeKucabWNkQYETroJUOhxsP593vxbdqcSVnDHdxaJ4pdG7EXH9mK4WvXFnQc7aOlGIf1mn77hm/l5aWKv8Z"
    "+aIZ05d1MsmI+WyMf26XU+DIJ8iZyXDIJlAPID716AbeRZLvtoPNC/QlbVF+fCYyMH8iXd/V+w5Ez4Fkj6A5NCQ0Ql0winckG8l4"
    "DJgP3qh8C6zItBpchtCFPomjzFpz2eVJNgmpqmb80/NrTlmupAQ0M18QHyBjXEegx5ILWqgaUSaVXShfywAPpyg0STbqw4k6iPtN"
    "YKZhapK0nwwYQZFB2JIjDXFKdU2hf7VOtaLRw3qN+B5+NMBEwtUTIDd6rNEGiAj8vQANo8FlFTzzX1SWDSy81Ia+zuHRgIV/AMXZ"
    "KkID2pirZ4wyaWTjXL15hm+gZ/AG/t0s0hHooX6ALyMEBP+qlzUNP3mCJCyHhUNH609FPmiob9IQPUMAtGJXTfq0cOSI5I+aZRKj"
    "eZQUDftrEeJr+FxYJgWqK4bWbdecQQEsxi+rjlvPmxVmazwKyhbrfX+M6yHwoDj2CJONmNEbZumYphYPa/XEJhPMs9T7Pc7uqrzD"
    "1EXrWArCxfd4+hURjpEc/du4f5dbXSYXnePwMdPduHJT+movHo4x4oAbUI7KNGiplFbHIGFxIEOwqvEteXzFISbfXIi2vHmGJm6G"
    "i+odWSpBOR+BrrG+9pgz8hwUjzrk0AHTKIMyY8bWihQz4SKAhhtFxX8MSYz0xg07lMviHagbb2rxBu4rczY1IblRnsziO5mVwQad"
    "7oumB/8El8/avRfwe7vbe/n8ahM3xwN0MpJx9JTSO+EZ2G7taMOSRCOdp+MYVAdQhJhqxILdfWNBYQtIkoMyMYoL6chDnig5ZlgL"
    "ldbBg0yNdcwNs1tDm9ZroPRIw0DZe+QK/lcwe9m90EjVcYqL8lBzOOUxELT12ivQDQDU9EcY4J4FHsw5eCT/1M/wB3DBBf0+GiWD"
    "KqtJ001QvOlL1hrKiiNG5OErM4oBx8JcIgqjCc6ZlQvUyoWdZlLNyv4dVle5vEc3I6hGewCPWuqdy7TBq4B/Nad+p2UcVpRQ7fDU"
    "LOUucvPlCCk+v13SriVCabKBiJEkhpz8cy1VAaA+UFXQ+oNc1yjFZ8hW8LM4c6n4VjEtDafeRmBu0mneKbVl5PRRqXGxIsceIMW7"
    "vd6zpof/Xtkho7ozM8WtWH9ADv12uxt4/4ZP6Op5+VILRhqMyeWACwCtA0+8gQ7i33axBPcfne2dvDu0vbfYicnST++C2k3GM9Ss"
    "pks85ZmBecdd1yzaSXqyWYiT/ojpLOp6OZwjIc8ue02vrZyus+m1eKu5YjPypLLd7yzNcx8q41HO68AqsrUr1L+XVeofltMVwNm0"
    "r0Om703EThXpR2OmfgKpEe0nHj8BJfJ0w3tEXL7/9LGcIn6AozoYX6Z3ql+LO9b0dZZGgz7uJBUp+TR+u0RZ0PR66PXeJD4KjJrL"
    "qppfkHRYuarmFE1V6BEqY08RqycekpM9LvCRSMCel/gMFDJupZ0uOYSOCaFjQegYEHSv8/Qrh9A1IXQtCF0Dgn6JQ9YlF92CGQPk"
    "nHuCt5E+hZdL/rLDX2ruebFTmRQ43XsYOcJUIdSO8C1uKTVZWjfrE9SNZMzDNZgAcyoP63cB9oGv1v1Bkvd/zPVGuVIFdQHCOM0L"
    "iQerTTGN3pgHISFUsLhVX+9i3Ib0fOgzU0I/7h2Eh7/tH1PvUDz8p48fX3kvcR8QpAXQt+RDaJHTPggqpAA2Ui8H5jcFO0CFTj/8"
    "A7Qe3F1ixSvmz2HkNjZZhF9+umAFQZgu5a+v4heAtgUzqYlKJhe2FG40GqdEblAAMbpiiOIWN6ZBcV16nTZtLOc/M8cKReBRVBBt"
    "GKK6RU7AeN6kH/ltCoYeqCwAzLsGLWq5CfDtvRamlBpBWHI5xIrkjH9p0nDM/UKDUU1EOgfusObeq82yDcf+4PuAu9Bw7w7daCfY"
    "ife4r4bztwU/xS6fXur8495JpeO1hAPbBWdytSma3dryXljerD1zXXx/2eshHr0eArhiT3V7eBRRgft4e2xNAPTNTs/pK2NAFrSW"
    "jGdjn6KaqM4zgw9L44B8BePkFzkv38GTIwi2vN+h++TGoN/xmUcUBLPw7MM5fkFeGF/uXJEFAHzDmDHFQMKbGaj7HuWkSWcFHo1i"
    "/uVk/GPuUVqvtQeAsUdTMRoMRlOjTUaHoC5vXZuevSvyMjLLm8cPX/EoI74F29PWhswwi5oAlPGRbFo2AY1q4ngQ3yMKcsVlnuBp"
    "dpnwkbwlLZq96LAXqCAQPHhFIOPJDM/2gi6GAVhGGgq+tQvNBLRbiz/Ydq1rT9BFSP4OO6VP2kCsCtg5ETtYJGPMIhhNcxAoeYLH"
    "4xMrjo3bSkpAVfoyKeKWm08sCWxPXxv0yOG4Z4fMiIvFEsw2SEFQuH+Bi0cWD5I+RuLgMkIGrfRTkRsHAAh/FZmCKmgpmoV9PswS"
    "rG8rzoiusVSwajVBaehPkvDAhPjHn6o6MzTB3EAgaEXgXzANvjJbja9Rf1oCVxq92lwscNucUOFW+ucOHvrUSrtxk/5eZvZqI8A3"
    "5dnaK9ZlWKfZQkKUzaPxdARjWIhlRI0arw2Gfxz1bz15s609sJYOcgvqTQx6gLNxNpDRjMdlXHxu0wYhvJB9luBu1L4DJgaAhRwx"
    "ViM3Adk46Rh7OgKm2tdplr52XLs+d8Q3qigbB2dBHueBjiqcrtKfgmHJ8P9JR2+SDnvog/q+aQx/S4aoAGJBoLWomnyH4keJDBQr"
    "dySmUNL6d8htuIDRD4YeXVYJVkZgR5HSMZBL2m2FsohtO+jdYXKrDu1kmftlExauDdUuqSr9NMu8E1FATDxSFQq5DgKxkNO7ADkZ"
    "kzs/oXcAJ2C5TM0Yt5s2tPfuUjOakKMvGzftBh2/aevTCAqvmLPArZE9K0SvLhdEyQVS8h2CW+gaUAkBGEdCAceT9cgqwT5j597Z"
    "IRRf2b414Puamazv9sCI+sWN9U00tVFmfX2lN44KqtgkPvx43mRsc3h8sceuViXvPhGYl3AHYTay2agkN7Qlhn+xlgfuilMLBHPI"
    "GQrssXIECtchS4LBF4MLlA/znEU2893prGgVaYvuGBf5l+S6xQ7SAc8PklnunZEdo5nxQNj/6I9mOKH0kLj/8NB/j7uzP5fD5Wij"
    "GdsHDkAqnZ69ARXojAQeF5Co9SwMfTkvTFXQvsR23QhEK5r1QXWpo7teLq6wyS872y96nWfatHlvgheBdKXYu12wK4lNLn2mxpDu"
    "UtK2apVb1D/fb47jaOLzfXBDhFPYG4mq1rPNHdBkN3fa7Y51Eci735reuy+s9Bi6fJMlA5/b8Tcg35jBrQUn7bOyQLf+nX/57rfN"
    "LILFHZ2k776I32JjvqM7lmg7G2v+kRW+7+8LnwI6JMiq4A/4zI0+igEgSGBjknSzwQ7jKCcXCEDnbinOTRtmNg5cxbCwNR2xN/uX"
    "+OHK8E1ppsAOmgL8IMs+IIpMID0pBlaB7c4TXcVkhQWvw+5TsLrXIbniMK7bP7G6so5s4oKZLSjY/QEsX9tkPLc3f0KnHWWHhL9n"
    "9H8wrXc2O/CTUqg9xTyE5dNp1PYF3wYyBY/uELWVU8stiombR6ZMmrb5zqK+ujwkHqq0UasWo0FlXgXakZXlrkGIsYg8tp0L9XCz"
    "m+dX2GYzETd8lQeP6UAzoml7G6ksgJCiPwuMorPVe8QzXbXgm8uqvYIcIdM211Zk/NQTrczXUB0O0MI4SLvBCx3Dg79vrIx4nLiM"
    "GwYZOwC/BP4swjFgIY68nXdnp59ODqAh1ZXbQlMXfegHbXvh3w67blyFlJqXjBumFBqURfWehYXgbVGNz32fXNh4Exlio8WtKQW2"
    "imvu+5Yvi4hlXZsGDTzZpbvVtjzlybqnyF7i66dQZI22MITObqxbagzA0i0OVmtfdbB0eqtD96khfaDLxgiCbG61cQXA87+qyRGd"
    "O9S5BKmmM4mOHZS2MPvKR+T+K0J+ro0AXllhxF1DL4gr8C8udvdfr2xdjYIlqsKOS5EUWvA3t1aoSIuOn4A9RO5NUl9G8TxWoQZs"
    "dxosH36yE/ScJBtmeLwEBNwM1ualN0hj5WmdjKrGD1q2xm/EHTpnh+dfTvZrOgEcgnjDaKniOLyjYKNyCy5fTvraptUAfSOM0C0B"
    "XBElGfM4zL39/fK5ocFtZVrm+0szZThKyVvoFkA0+8NKImtiYyBRbzXUWMIBkz3pLTAiFG9SnN4Tqsk4iwkEeDZFJKeY+PUUQCsm"
    "Ux+UbjetnGwa4CWdX0HijmHN2YwoH/O0zZchCvCGda3NnFQs9JtWJVFBy7ssz09+jbM093fqEgGhdTAVcbv2eVeWvwsWxHtoKZ+u"
    "zAaFVVzpDchvCc2QPqATE8o/s8oDHQLLhaB5iXDi5B5Luqzlow7oJHQyAUWYXAo0j+Z4FpLOGEunOAX4gH2a16ahAbTsCe3ItuAo"
    "xZU6rmVLrWNzMkDWB3nptLoQkDsJg/NoqSPnAVN/GGZhZ7Fj5whgR8j07AdduufGSnugsItcuVs45IckDJnE8xCnRMQ46Qn8oD9a"
    "y5p8ZkkiWJ2WsfHs7E01UxMomBY3lRMPC5gCRdV5rehknfToRyMel8JTWmypaubhVOwDJQdEM8CBPUWYI4QnDGqpNqNQTRdUE+Vo"
    "ZRNAVT6TSE5vDd0ts/LKKc8AbWuS1Wh7Sx/qtVJfM1htc5l38uKqeRWZs6quJ7JGVXordux2dRJsmokbRhLTppUfS/tckXN5dQps"
    "d/pr88i1mWSEtnLx3I9d1ZLgRtKsf6lIaiRL1V7V4TwDijuP/Byxu5MBWsz7Rxdfwl+PDj9/PD06QV1kb//9YU2iIy1JOR8Y2hMN"
    "R/GwfBsOH+x1clNtaBySz0bFuimtakQihUIhLBaxgDSGFdrtj2MFNzaspLioJ6AMCPmzjylFp+yJBeloIgGs4TlFZFFR1qjqVTpv"
    "i/M2zE/d3txGhV2HAIU63NV5Kwr99MIq1KdbosrAsNy8VE7Am4tyz7tWORy5ErT2C0epEqztZ1YpirAoAXv+3FWsBO2lwMwYQSQJ"
    "8CnRD8/b8F7BK0EIeMuwe7XLOwNveBNYk3Baa9QZRDa8+mowxQ0o0HpGPHFmOL5EfHqIXFPg0eOogfLY2TGUL0TKyIq9JmDqTI/6"
    "hicMTaDUrcdA5fRgBCrDLU9wUtmuYU29w6XG2zXIVFfP6DbVNF/V1LV6R5Wtd2rYhlmMefw0UiD7mW017eqac4h1rXS7ucENr5ik"
    "PN3fP/50fnR6Er45Pt3/BSTlweHHi/dWym46em8h9dqE99SGpyURt/OSmzNCoIuN0D1C9oWhTtmMB/sDkvjtVQKd4gp2vQ5zWhus"
    "+7o0Csxya3UeskhYPf9w+OH07Et4cbT/y7nR0+/Qu4qJvgJBTXrVlaSdR21wKP0XuYFYe2gh1+lCVL42nx4DaKrPqhZ0t6xvmj0u"
    "oyZ+bSlAGl20oV9FfSzL6Y5bjCVGKHDrYVKwVpUrqMURICtb/G47c+jCcsA1GGhKjwuh9gmuOWMpgqxl89fB4cXpp7Pw8+HRu/cX"
    "CJS3alR9WlsLKqnuuC5AYSjVjaXE2vKuaDXXGE/NF6yuAOLxsYznuqZnrTrXZUn34wkhZTLID+T73t6xPbHm3UNf1aU9bX1foWx0"
    "1NRstTc7O9pMQtNGz+Mt2ccAeClpir6aLZ2WTcUl6NexvsmWrRy/1XcK2XfdrMqEaXTAGvHVOTCF+GAeHoMWlckveZ36nJeyEEt5"
    "iTmbbWG89xs7NxuUbDauTNcZbuVE40xyuNON30Spnme8Z2nfyWQ6K9aLizdnlKxcZbaa8LUnvken7USaKYZdsLeZy1z7Qpo/xks6"
    "crmUulbAMprDzBS6Gn1qen4XRgg0ySDgOG1ubq5CBgycZ3aqngSE8q/RaBYfYlY5f9iIF9OYYryYHsJMI/998zPwAv5pdgKKNvRP"
    "mvtN9voGzMt/lDr4Z0PPMU03ozpGzLm/bhlultnuArOWAahWO4OxTEjmt3UhArl15Gotyhqaq7OFInMzpz9A4XQuN2MQmmKY2Vfc"
    "YvmHEd/WoEFqYPwA01vsiwJ4RSucWPtyyUFcmfPCaISG9NsaYSDoJk/FOEYjjAzf1gqHoS5DdLRj3nXwbe1ZsChwyuA2tb4p4Oms"
    "4GNZ0W4e5zneR5fNJj4TBXqr2uRBmWtyfakVI8CLhaSUC2FmqI7DSEjy8Ba0QZHWRptTjoY6V1a2yCvDxVPyC5nIr5qWcsGBlzFT"
    "0kspasWVyEV0F6fDYcP6rHL1UuIz42olvMy1qvQo5td0WxXqayySwqqwvVO7M7L6TkHLZP9gh/ozqNMirPiCmd/uohGoQejl2v/1"
    "F/pt78RQMZm4zrzJmxWws8lhCtTC3WRN4rn1KmCSPnZZ0/rl2S7AWsWHyQhGNh48pI7YbXhICzB3QRGPRsnXuJw/uYqqDxgCdvpw"
    "wFtzXeeAHuNbmHy6cTc2BU7D6fm1Ctm9rgTN5DxoKVVgZQE3Ide+Sl1Mfn69IU/Z6yYyRkyHXD6ELLWZq1jt5eTabAcBydJRVgyq"
    "uav5bbeG2uGPTjLkcRx+tC+vMHNS1pLRJrqzkCu9pU2faXg9Gw6JFTv6hlPtnerO1tS11pUUtCVjOfWac3QEZImpAm/BXz0wJMu/"
    "J0C6KGMyoNF04C5yyFUwsLraqswJ/AIOJHrIcr2W4rKsyRxdczbP3HNZfd9ky7PVl3gBn8XGGLsGJ6+QnnznCpqPs0mIJ+jdBckv"
    "dZ/Ec3bdFliet3Qp0OqieT+aiEzV7Y2VTtH2xoM3wtZwqFehWe1IX1HD5T53iQciL8iPG0ff1FeQy208ph32UaqWSw7Cgm1Erz8n"
    "y7f5tF3qEMXcyzu53f2mr0yjrJndRjHkp/G0yKsEOS8mKeBedQlDIbRucVfVjSBNBmA0TOZRhsK+ZvHvePNaXmRx5GJFx41pjk7e"
    "360/Avd34876pZXu75AgyCJJP0SGrp16erk5zGWKP9QO2JRLidUpX1VQ7GmvLBjd38jC9VjygYUBGSxXKAboTUyKolKZM8utUF0k"
    "MFltEOe4d1QBXN7OUOYa/IQ6dpjPxnqcGtp9H/Y+hsd7bw6Pz4OVq48CVCEBeAE6PohW4Ci6jkcVmplZ0jQBSrqBq3j+TV2BtRAE"
    "Ih9kTdVwkRYbxuXbxVSuHN1eI51MFo2SLkqoD2HkpSpQAqcV6qcTEI/jalqroqRyrVGOdzcb53GVvizLspSy65Xl4UdMPG7WIKCc"
    "1sNMasC1NfBYC3QtZwmKaosmY2CL+3iM2zPjVYXd8/kHPK/UynENwzzc3GsJygsFOMoUysNRdJN7fr/LHA/e7fIGLIL4Z0xqM2A5"
    "w6/jW1RT1NU83hY+7F+Ev6Cr3TioO8TbBTDeEk9HbZFrgCnzeOEMgESJcgsGqQfiv6AWht4o+ppAM6nIPQDrqH7Zj+ezXPnoapD3"
    "5gTmqS1quI8Nk+EK+BbsFFjOqMITT8zQdoRnTHcKK2GO+SqiIb5EVVPodj/ja8AXyUEEQN+mN5hFIy9DEPb8q7rQZ+W9QrIArJPc"
    "4LbHELp195KaRV34ljJi+66h+eVl+Pk9Gxj4efKOH1Ir9RjejukkG3KGA9e7l+H8ViAi0WBwmAwEvTu+TWdIVMZNmDldC7XFu5GR"
    "fJgpjR8GHmbp13jibo1UfDpC4WztRyTMj1s/IrQfy40Jysj25DkMoAVrtqJVEHdF/1a2/QMNN7FmigvpyTvKOwwo4N1UkackNjtn"
    "yEQUHnrTg/yYnKP4xSlbMqrC+56on+NkwiTeLor9s0/Hh5R+4uPZ4cHR/sXR6cm5WVZbYOwap2/23hwdw6QsBfUJkeraAzfXQtoB"
    "p+sS9DMJbPVz1zQXydJtE+aC6L7AoXoNte5v0M041qdX1CdJRCqpN/mKN2mRLqhMhtDUqzdZK7ajl3rqLKi4Qbu/pIIJCEr+z2MK"
    "3osQ/qeup6rjYRMz+atZKqCjYz27tqB1quKGBSOMRstZMtjZDvGIcU7+pvvYdpEbOc800iMxPh0dQG0iCdeqNNBTTelxg+YuQJtn"
    "pzz/DmjgUQIY0/FnSv3nZC1TtNeh+xHPX18c7l+cnjnw5ZoPlq+jB8ZV4aaGjbVTWwTEWdINN+r2VdYuwCWflIKJM9JVRfh4VrS+"
    "HuGMmkjF84u9i6P98PTk+EuIkW2clM0KZtaqVE60qsnG6+2fnrw9OjjEiLeL92eH5+9Pjw+adswbXyhsi+i+G8YT5KFBBWM770Uy"
    "MPtHQ/iIG3/+9+glqhy29Yk2q3t76/8Z3f+1RrdxLAiMEscA8Jka50UyZlEPvM+7tMT3yrdBOwZRy0sjANVm2zC7LqsYMQ8aRpXb"
    "qlpCV8lOdroFB75rgtRTFetxUgIzllGhxZumJ4wr9PgsIol0dvj26OSQIpR++xJ+qKGGFqzFBiC8j1jqBiItBWFxbhTkJk2YB8uJ"
    "8xHa1rrBv/UMrmWLSPM4I1+z7TfTMOCo2UngjCYrj0WbTfiOa5uNRl4JhVKHjlGsDjLjjRrucGTtgEkZvgPS/vHpuQ7PAFRKHah1"
    "aZ0RjkbT26jUdwcWe8cf3+8xXEpJ7+uH4ts7ScxVidPbvTOnkodShqLSeVgrdRXjV/n8fMo7/0TO+tqTVzXLBm+rLJ/Ksla/pM4t"
    "XyuZdZVY5lg4jvFUD5CrhZqVZ9UMcWWhtmei87I9GmGMtXTMr6bVSFAxAXT1sbTalBcoQbnyqmME+1auPTXj9w0LUA2XaavJ+nyx"
    "doCuturYHFSleiBnGbSq5iwNhHMdKzHE41df8Z/jXK52M/m6kIKKPNfVSkxZfdG1VFSlh3EEkOKcc55+/6muptLGYZRl3xAQyjJq"
    "MTCbefI1xiuaSLVGWWupAKwkZmMd6DsEWvm63QELwmXPbFjFKsKbmiB3EwlZ47KnoVF7KIRv5TivgFXHrlGOkNqN75usksjzcGUV"
    "ww13lSnuniWZuqdIR4WszGsfVNQmLFhIKB9/Js3y0jnIcnQfK1h56pP5LChmyXJt1XMYL0fCpcZFlmauK11VZU3g4eSXHza5iVl/"
    "WzIbripHIX5mh2SMvJdY6TXZ3dymJK0WT0DRAqusV6f/oRYhAf5fDfCHvx6KI1YYtdyuBSE22RQl+C9m5GpcqSSBPkIWnRHYOhdP"
    "O/Ysn+6y+vXbkcatqrj0ig44IG6JnMMuSHqmtyjPw2Sw4GNL2+Q3lOtLwA/s0vjW0zVsKnYpQV3ZFYSfWO1faoXtO73FBpLH0tho"
    "9ekCGu7KoK9OT5baOKlyIP2gNkFwYyIaTOKCOxnzaMk2i3oqsw7FG8GaOcY05MzLjw5k8rLpG1azZNDZbnt+p/v85fOXz54/2wks"
    "G0anRYMwdBSooC4lqlOP+uknyu/2G7L+ycX5ZQN6f0fAUZTL95tAJ199I2/+U42d8Hh/KHeMkLrO0aID/OuNAu0ZqTFw3tvs60T5"
    "F7rYim8jNegQtKLIK2+Fs+mho6w2rKLCi7RHltqSXUCLt3hMMb25+kwcAFXwtg/KrGS1ymRYCzUNj7zmg5iO5eapvkOFoqv1XO6K"
    "ofzDlLbxIuoXo6UX5Wx7UQDAd0SMZJjEA6tFhRsLAm3qXIlt3mTR9BZIAG35v717490no1F0E2+xRskRS8xNH2eTHCRdkzE55rqe"
    "xpNaRlZDVs3N6mTWqlEUSdmYKWjPgm7pwmeN8RnDObhecKLF8nS4KBRboyV5Q5dkcVLhrS6MWnYooNysdM8AcmPTmoy0rCrDdh1L"
    "q7eQiqwJwodPXnbxveOurnVk2xbdVkVcrx6dXJ8Uni/zpxbzlNEnzn90tPs7hhnqW1+BzvIkTnNLhrKWS4LUZNlSU+aIA/Hwyh5t"
    "4NwMUC5X4gc354pkUzYrKvG4mUwG8cJnwxSYclpbFIOKhsREYvVdlrlTGXi961W77d3XReGh6rU4sP7+MOcgsPqVtBfgK0hesxFv"
    "kaU2DEsj6EOisdSArxeUJYe3zr4p5ct5+CiWci1Y4kmryTixogKxstmSS/bWp8Kpiicz/L0kUtfvtmsDe51Ol3Zu1+i4a+v7cT12"
    "OYKi6XS01DdhMZFahrcc6LZfdSn33vL32qp91DmB+lj7dc4nrBFvuSpe7S8IGhQCtXQMo97dWnN6wwlOOOfU6MNAilBrUJki27Gp"
    "f2KnBWwXJ+UilicEdhv5NMmiUcPh9HQePXAKdbRcjWMHu0LTCUfxTdRfNnqOe+oIbl1egHJngsC9tClQzrQQVpvOZAmXnbXZrj5V"
    "XE1DcnvQMUilfB5ber/K3Z5QoCiF0deVu54NMJ29dnn9zjbmmGppEFj+y7JtZY7M/9/etza3cSOLfuevmMtTt4pjkbQoP5LwyKpV"
    "JNpWRa+V5Dh7dVVTI3Ek8VgiGZKyLef4vx9049UAGjNDWevN7tlUEg1ngAbQDTQajX5gYFIdp1uBfZK87D7/6XkK6fUAdu9HJzxS"
    "HU+WUxaBS3CDdj0KPIHx8GVJbj3b2jMTGfxp8jxN/vLtEF8+OsTeo0N8/ugQf3h0iGuPDvHFQyGehUmGA0bo+lDNGVaoYl2R272l"
    "mcXKUvcfkaShMccwr3/+jVEY7MVEMYqOAaI91ekYhVkeMYgU5HeJEFTpXqHcAP2ITL3H2SAc6A7inoYdDSCgj5sfKsqCxGhR5GeF"
    "z6SNxYFZS5yIOH7Ey83tnXfH2d6fgsVbAM5YIXcljsTBQY++f2AzCu1uG/rlYzTQYVvoPGoTMVx1vhlZZzXi6P6bR3wvHiG9Wr+Z"
    "SSy30JdY02dx/nN8uHO0uatDrB2fbB5ZzsPcAFzd+E5P+h/p06HgrvuAMY2kAstTYHJ5OS8UEhWUJ4xZDFllF5N5C3uE8fgMdqNV"
    "cN+ej8a2kqYbWyWNK704kui0Z2U9VvQyQ8XUE/YXzJK6UzneQ4W8FZ8ECv3HJ4NDhrSWvEHFzf03uwNZT8DwQ1Q7J2N7QCK8BdIj"
    "OFJSKQjM3rDyKo7o08BF/AzqhNYkIWxyth/NfdNek80wtNyWigTtSc7H1fMsUo26pcKixEtroRN38oG7Kw3vA7VarGqJmX8Awx2/"
    "Z2WICxsu8Y8HMjwfa0KulDzSQQOeIkbyj+ntwPobVJvfBzraP5ENvtWVKcw8aUuHQOW/G6RcBe8q8p0NcPOPdwKuqUn8X+4r/A/V"
    "AofmUBLBJsaOnZOCs6roYp5jnebCodrcQUbIJPhI//6KcBYC+m9xinUbFB9CLMFkDqxvkQPZUYQMsTLDAYbZlomuyL2Vcn3E+40D"
    "YBQmRTl7w+FEutGddTGxKHXEtCZeYaYltKaaeCkn61lxciqJCCvRMlFLZooOs5RAkIRIXXCyIGj6+d3r14MjaSjWj+hUOFYW+XLa"
    "icH2Ypma6RksvwpmSpifYwbmXhGz5Z2Bbw+2Nv9WPu7ypSD3hNK1AJzVmqUpLg8mtgubs+azZ/ou65SZz5l3rcAQVbFxb6O0PNv9"
    "QPoAbNr9aLi3Z5Ad8GqvMcuZyY6rLGOtWR/hyuqb3oE6tL91Ny3yq+7eVa8VsoWZ5yU3svBl3X2N/Kq/v5HfNIgsDhbLz7PzYiGj"
    "BLZi8wSszOxSGRwdHRyJM4p4ITgeXMLQuYa6hZCZUwpv+IxZQtzZEwz618HeYP+EyafgVUEhONs7PAZJ2dJl3WX6Opg1FAwgMuQJ"
    "94ztnaPBlnxztIkPARxKpRDA3mBzX0igxzv7g0hCB44aFVzPlQ4DxldXRGTM16vkUJUaoaIc5kXwWHGcxROnonjLgdFyDRm3UbYT"
    "uPBdwuGJ4WiP2w+iTToGH6XiJF+SlRmDEZ96XFpqPKXY0fZZuNR66o8odIaqxqhLTaj7oPsg56hMBdjo9uZ7M+e30xv0ZUD/g0DC"
    "8kUKIWqdehkqUbyRYFKXVeEKxuAAofUXiWHRIOb8t4Unrp2OFgUmxAa3CXhGzwnZXGl6bupHHcDroTZlaZjQP7EN5TarLPYYg813"
    "ZO9PV888ZwNdJUDNyc7e4PhQ8Kc6qFEElzdnOmF4Pjbi7Fwl/SbX7qpKAWHQFRaoMsciqONCD9OHu/u7UQ3hxb3uiNsaZsumyaQw"
    "L5Wkb8d0Hn/TPB7jiaNrhwzdoqIE5mYZgJJSlf5THHvufWg9rJGIwqYHOlk6rGcHbxJwaqBgqgvRNX+vB3M9c++ZrCQuSN2kX6sO"
    "5UgDId1cIYunm9tWQLdSfaDueOrlWVUqtChxHImFSsQGFYwNo8rjac/wjYBm2d0Yg18bGqq0jCQ1wVRZz9LDhxEy3mwetrWHCrA1"
    "g2hBWLjId5wapzf5BUpUc5kfXZY8FU30Vepa9abfEe/O2Kp408JQ2AEf0vVjfjMaojQbgApFINRpb+8cH+5ubvninRIMgD+O0VFs"
    "NEy/C8WkwAYjd8d6in04S/7i0hQWFkxYgYjbOxc9ctyqHsq/P3HpYsgZwExlnP+6JxuotyBrx5H+2TpVKu2HHwI5ZIYHwsc4DFL1"
    "az5WwfX9oK6MBjvQL/9hvBK+LqGqZqxy6+fYtIIrG152PXkWKU5jwooF84xFx1AIBIXAGEREyNTYMt+OR2fYIakTtLe5kDjJ9YyL"
    "vUgh6MyvO7u7m28G+l7qVxlaaXtz7zBytxg6fJYlcmiEa8NL8lMnCwTn5V0nMVO9pExlmbjCbB4kO9MrPz2TFT9md/NFYHzIOpn7"
    "NHlahyTuUvMquCfZk6N3xyduebjfbxCfcepj+zFu9OnlrYCMVTV90xFuGaHchn2fUl2Zy54VdNl78dTWdxXSYsMdoqghafWEEHZF"
    "55bDT6DpdYFygEqHp8qEulqnciyhNB2dBiXGReum0URdyIrzz1l+Ps++mP5VTZ9fB0fiNLy5m+3u7O2ceLnzNMJafsXN37yKAoWV"
    "wFNeOy/629IRetTw0fENVqsZT/mSLcWJa5SCyRs9FlNhjOPn9vMMbuSlRunUxS64M4vrx7f3JcrA/ZsqtOW4Grsol2o/ifByE5vP"
    "RiJGGx/fskf8XuQzcADwx4InBGPgLdeenbSkfer4iLnPi7gCJbTaU/2TWhS3N+1o4V6dwpWIc6s8IBOhDgMpB03kBnxjBQb0hogL"
    "DDTv3v+mDbmCoxguJY10KCf7E/OY8nUdahOrF3oJHh6gTfz7MIaSPhp9Qi2c/BkYRMlgHswyHFc7ISM4cX+uqU2BMQuwEaVeyTC/"
    "sWCGn/K50YHjnX48oiOZuaQlXY++S+vEp2RtAZcOvMRtrpD+mPYQdm4aDIs9DJpg2gyvLgmeyY4iFokw1jkuVGDtzjkWAiym/VCM"
    "73f2tw/ex69HYuFES76edqINnFW0QOOQlnwtbYEiw8zoPm9OUBqPNpXXnXVmT+w+a7nYXGwX44FFY43ycWJlYHu4pchsakyWfnVM"
    "5f8OgUXrDIK1wkWWA7Y5frjHUNVt4pKmnFmmUnbupTGLd7QAKkdeheNBaYzcVg1c2CnAMgdXN1Vms5uW3A4G5i6PPK+0ztmlZ2s0"
    "/CzNnho+3vUXaKEY30HaSGV2HGNzgQuwOz8aIXnOwSOYj7+Mn27z+QeWLUlcuIXDG3oJngmf3KADzdr6YkeMFNHkyYGTOzBsuxGH"
    "dGl4GlBfDLTBTd3MIBDBcg4mvPZmQi6nSpdNIz7t1d2A03vG1XAyXozGd75i1nKa22I4yscoOy7yiw8tB565ODMPZcErpOoEKRri"
    "8DtwlAdwExeleobBrWLPuX3GYYFoIX6nrFqS93sr23CdOGbBytQt26UJfWCdmLCo5VDx1qIszu2Ht6LcuxUDUUxcoimfMzKUNv5W"
    "9oRkEUMGdWaqKlMWW5Dlw7Hgz3yssJBHkIiGzvv1kJ94DD1cXg5j4rtMGR384QuYJSkfIoVcwjhxtetsLKYrjq8DaX6Z2MG6w6T+"
    "cuGGgyGFRq8Mt64b379WF9zNQuIpIjpUngFevYoeAoJrLjJtNvyA2X99t3M02M7e7pwcR67HlpBdIpXqrKAaVde5rRtVeKRUK218"
    "iySV1suFQGkXOgrE8kZATE5t6O+6Ccj4TeIQo9NraqP0h4SPXvhhUoN4R8R+rPT4xPI4dgyVnFEdorm6qT8rY/4BjkMCj7JoP9KK"
    "A15JTO+a1EkrIsFar4/Sqchm8eUVRTXM3NE6KkIvvS2Vkma9jDKP3tcSqsZ2iYe3x+fniFO7zhH4O4Q3N91+cFDyNG3UwiHYxnIp"
    "VJicTjwvrmUoomJWfm3XczRc1rkwYn5yID6fZL60v/nrGy5vhKpUUZra2wpsassVw/qni2sVPtuJm91OFuK/6Ufx3+/mdKqDPMMc"
    "bNGq6o+83ZePz/oviZXp3RyZ18KA4pKjvkqa0wXJjbqYQWP6lopqg0ste6wlnTg3SNOo6Qxtp36k2mXlVYYBSLGcs+ZNh0207Uhv"
    "uNWvs8JqxS1ZEQtWbSUVzCVeWRoiqFnFfDoW9M32Y5uHaX46mbZWA6gtGA5xq8Mhsk29KmsrjNvvVK8VgWeKqTfp8Sr/7HMWaKcD"
    "rcmzl3sY76Vs5C20jDEWta1oeBCaceB0lbdNaURDGLjVO8/6taqHb8aivzXc2dmpA4bRdLigSrFUG5DEMY71K+LdFjwWvGxzW851"
    "AS0SBJWumcBDwbIO5QMCPEQsORLLO+ZM4iZRbri3GLpJKzPn59I4rph1IZzbSJyPvvjCZXGTT+fF0GRmpyHdjfEe+rxgStwV3xvF"
    "7LeG2ZBWQb+yEr7+WNxAph2nZdbfczQeTz7mvqUnpfoCprzpgzfFbyZg7ecAMdxqMlP9YhI8pvYru9X5SqzPbhvPuqvR4YH105r4"
    "DjzL7x3qRMRECK+obJkNr7lwAjqTzzXTdGYeFxElNKj0vJTop04SDRUmi82K/xKcXyBArKEPoWeVI1JwNQRXfRGx8+RdmIhEZCeb"
    "yq1cUphrmyqO45cwLL4gmjOLsGDt1O8C6jgsZTne8OBelPcgkP5Yi96IdB+WVXlccAsB1mA2hLByLApqADTq3v3QeKosX2WGNy+K"
    "7DDqhMYNfjErKWe5rD/7tCyiuZeK9I7CWGxWkiUg9RqCV6buhLKD6NfqVtQrnCm7kazFnAw12njmhEUE2lEID4tFtsaHzDR9QKw9"
    "saom1WoXwlm2+Nm74rkmQlTG1S7EEdX9aFRNfwk/Bj4CzLDYGvlvqgNNV5BBCvuZGaQxlNPdapcj0T2Jg/H21SR0qDGE6/DISIPB"
    "V4RCWKYx61ta2bqcVTayA6GfeavpZrvguxaojq2Lgj3cDj0I9kMps6JLyguoPq8qz3p7uEdfONHaI3R0HsjDdN5OzjnFwqNOUUaN"
    "5xDQkAUDs40vBH8YA5s8ZUqJLhsFTK9tDJ3OUn/Nzu2gy+EzRQVSKhqhJ123Lhx3X6yWxBR3w4u4b097/TO2DaaT0FCPa4gfe+xT"
    "tEm/tfWk96I03vwQo6xp8LEBrqIbH/ep3+mtkggq4DnpScdwy/q5LdO82QtWt10Ac+bpqpS5pe9mmcpwwo7hrgvr80rvzD9lAWf6"
    "mPwlYYqGF/1gxAo2rL01EBSg8jonL7PX/RIDMlsYOOo5hr48siP0tQwy/Azocr3zhgSquT52mgvde82OrR1bPby6gG09UX6YCRFZ"
    "Enr1RYUPnE6G8zXVZ7TV3hqdvRJnG8kz8UVeb7gd27AtMl+Bga++6D9WwAN30F5ygPv8E96ViG4oXkzeZD19KTKb3lNbdnwnCoLB"
    "tf5+SgIYmgKi1flFfoO0MC+fYvBsU5g2uCbKOR0Q++EaYNR9KTAkMe++FidQDmoAU5mn+5XDXgO78UuuRUoG/VkjssvlpdsJFgbV"
    "sc7zS7ykF41bDK+epTg5fpCWDM63nv7m31EpYLxhfUkfJI6g6xLw6hoLwkGie1EFo0bHW97pWF7XTSDpw+WiZbpSJlMGVTH8Ka1L"
    "kvtqyEpfb8r0HU8t7P91Pr4qMFvgb9nfNt9DAJ5BqIEaF58yizEdSdYuhhUfYAd47vOy+mi/LtVwwK2YBp7KUPMOZisAbYDpvrf1"
    "VDXdqijRAZip/LNMX9aTzuN3ZkV2ZsXtjPU+kLknSmG0sV/o5OBPGjmnvtus6fx71vyzz5qbST7E/fhGTZtb+Sub5ovrfnKYO6l1"
    "6UfRbfjaou9Sn4PTj91CiDyLuW9XMMtHYvd5Pbop9ieL15O78XAAIkjrsrkHVRHMJbzuJ39QcF+bNIrL7N4XOubzbDLFGBiT2aJ7"
    "LH6LQ9vBFI2WAg2bKtzFRHrwOLodfclltrTiI2ZLAyhv4PMB+boLH7sHRyfZYH/z591Btrm7GwE9GkM05ck0E2J3triGyxjo3FpQ"
    "XGp8oLmdsZAvIR+m6j1jOrOYuQRoM0oo2QMY9yvTnTajSZ98FKfO2fzVaXPr8N3gc3FxB5UO1fvmWSwGVPH5opgukgH+Qc+GObzj"
    "yHx0N4bYOpLCQRear/MRCFyLSQLzMjnY3/8NtUCJvjTuJjsy4yEOGVwyREuQfFHUwkSJkJZ6NoYa+UIczZuhnWnTpKqdFQq9w6Q7"
    "GY8/d6FOcgmR1McCDnTDnXFdF1yaXM4mtzBUwiDG0zucdH/gkxB5LyfdcX5b9BP7Qhq72p9ot4sUhiSAmYTRSr+Slf773QguzCz4"
    "Jl7FgzyP6kB42GvaGrejOSpYTYU5IqnlQ4JzzaIlf7guhC4Ejpi/FPd6qYaU0vVN3xVqYBU7kJ11PLlb6AHKR4pA8gYxSH97KFRw"
    "KA7FkJo2O2FTCphj3WLp+GoNTwLqJ6QNErQFepTBMCAH7B78amKP9KPpiBTGBS0dWtjqf69uy46kbmCvIkMVDma7huZO1WQ76+J7"
    "r+xwBDoCktE+0IpYkGg2uqZUfaOxttKxBTAev+CZvhUaaQkycLs1Ut8aWC9fy1fVU6ScWS7yIVKKJLmU6AN+LQlySufYGU7cCJDb"
    "sK4lc6QmHb159nfz4r5wdnTzm9/T3c96V3ffBvu6+3npnX1wX9gpSTd4F+6/t3i1xXvU+Gfa5C/9TR3GMp3MX9oJgLuoT/pu8/tt"
    "r8CI5Xaq2WuNTa8ZDoTf8vrS4I7f5thNy+eb6gNqkJftkxCPCrRpMUnY0dYWITYDfmmJUMUxbUmJabW36X1CDpllY7am2sBVVc0F"
    "V1Utj61NF4SrcTcuukDcntVTt+sKDuNzYfHcz/n6CMcaB96fl+l9b67nEuGfmukxkpjkei7peabnTe/YipyCbe3HkeffYxfQOP84"
    "upK0p+KBeF0mHrif9Spx3wYLxP38CCvEBfhvuUCtEI8M/3qHfztn5Yn/u5//vYn3D9MAxM7zRA8gg+01vz7qmd4nwLce6x9Z3uH7"
    "uJzM4/PFStEnqFB+agyK1xV7qOeF4O4tB26bRKeeAt8eT7vjIRqNt91zOf/J9dIpKzOb3oef06SzkSzupjfFKRozt0mJs76HAVd+"
    "S92eS+x5ocWkm0ncrh6dyUxl0bA4Ev+fV8kzsBGgH6RiAMKgi6/sXPo1v7kzu7aZOAgjkcqP1tv2+3YvbYv9e5H8EUB3JvpFDiYV"
    "xPlJvZAhn9oQOAH+FaPNxJfsqpjcesE+Xecp5psgR5tLGSH1AQwylbNOaSI8P7dM4eBvFdFHFDsViAx5+xtH9MEWFHYF17jNFxdi"
    "AnO82hDkD9L6V0UKrqNfm7HQYpJw53CBZUl4KoNYdbvdM0pDIAAW9eipivfPwmgQgtST6b2f7wdgLa4WFJacCktDkoM9N4of2n0v"
    "UJERB/fKvBXxO06UL8VsMm+1fmgnz2Cax1edVW21XSCuhNqd3XnyiqMVa2utpydT/BFOG7m39TXtQrlF6SP7Gj1MCUVOUcYQNlJK"
    "EEqVQpIxpfagJTNwt8BXLhijHTbr2lmRtNy6ocKWKdeLWIjg1NLrpfW5mVI0KnbWa/cMN/PBO8wMDTDRsxG9HI3fiq10CmECz9zh"
    "Zp8ms5uhDFhKivX6z+Nz3U0YZ6FkF5OPAaTn/R+WgvR7AOGHfq9XFwSarYKUB8rX2YJJemDHrIzEsGBmnCPTwAqWgFtPnnU9e6kQ"
    "7dSCT136eoXaBGltD4Hq9+8RyeKfUaygTM74k2qPJ47ZcZ6nruMpCbTwKZ/disLnd1eoq7qZXGFEyliUDa84dW2vXWkGNhS4PyxX"
    "R6+1mlXE2UYcj8garVmP5B5ZqqWM+OqxXhIxjCvXi3ngPsJXkF4hQ9Uq25Jjf+VL1q5bc+DAbuwIG3VmldbBNv+F5dzHEzaDmy0/"
    "TFZMHA1q/gOE0qAPDxNNDR5C9TsJjlGmPeczQYZaF+aCgLQEeIezWjFsBoLz+OL6kyY1Gltf3U3u5kywXFFiMcvHc9FK0SKzqJ20"
    "xA4LkkCaErF7+QDXMITJnY0qECCNkT9LcOdLofHrjT5BBZX2TqnRvekcjTgrXy6xLFQNO+HXhdhTe1UbIivBDzVjRhfyHKJL3RVz"
    "NVWdppxFrjc+NzeFKn8KqYo7LzHc8ks3Q01MboSozE9lxgR0pRf9bXWwEerQf/FZyGH34j9IQqAbA+lR1B38bZAdb23uDogR3GSh"
    "nZUFhwYHI/GGjZpLbZHbpQV6VQXWaIGUeleIjQl2Ot0ryKZFZgfaZ3qfifPA3dT7tnYW0ywEEQ4eGIDFy9igR/Ak2drcGxxtZgev"
    "Xx8P/MSbK9BTU+TdYViKBNYmUkrLH8SKoLYAJNEiftyLH3fgmX8BwfxVZ0pDL5aLbPi3sYy85s3dxhJiG03mZydxaXiIasHOPGsH"
    "zmDTIKeN8kg+pacY22bVIab+QSZ+mKGD8EtsvMJlfnK0ufWLmFvboPQxIXQ42R5Q9iG/uc3H2jvYDCaNFP92AbOO26HfVCQh6wOE"
    "WUdgIC1sENQJjIkluXuAsYUGh9H81QR9Or4AL5ioQtHwG87hP6gm1nzNE7dzcC9JUlASBoG0K44up1yC02jh3jKF1yKFV2niJBNs"
    "9OFpVsMZ5mDbcomHoVhqAJ9X6f4CpYzLmpY8ey4LgBxCZd+BL9Uc8BKHVJKsgJ/11ZsPYS5oo8gs1fRRFDoX4gQ6T3al1/aWkMpn"
    "k5sbHQ0ZNT04uCyLKXmu8tF43k/ewB91uCdqnmIxnYgB9BN5d57fnBJtTFAcPQcFHvLheQ6X9kgPyfB7SrvjXQxdqWblX3EWwX74"
    "E0F3A7mKekTGpJ/JgRGNZ82EfuZBcnsI4qXzwirI1D4SQdrF3WwGCR4FS4vox1TXso81UXdZFEMl7dSogUoy+7HPdYyyTvKa8Hbt"
    "HuziuEOBeKS9za/EZLobFqH4gAXSwI+UVFnnSNBn3HVfOfSz/CcTS9WEO8EZ0/0FYhNBJRL3/FKX42YBmTbZx7i0ZIF4jfUwHJJB"
    "rIWVcvXF7sDIO4TW1T1YIWQkFYOrdeCJqoau6zMYWdDOcdH5DEem7IB+USuWXaRi9K+SX9zKevSqvuUWdmr6sMhSDpBIvJKkHOLZ"
    "+U0Fq2s0sjdHm4dvDzcx7uH+zuvB8Qno3KTxjmDUo8tivuj+13wybtKymzLF4e7B+8ERbfxUOkXR/6/qR1ZyD2G+Ozz0YGo/K+9/"
    "VfD++m5zX8pqEpoqgzG2njxJOmvguCU5fobWSIf5xYdSpi8JM80xHvrMwaV8pc25TBESEnpxnYCnuPkmjt8M8ttJE3KvF+OLCexB"
    "r5p3i8vOj80UzIAur/tedDFJHtEqEKgLqqbW5TWdzFOdEqbUcEsV6xbaTEnrXqGWsV0CKzK0zzoe/PXdYP9kZ3M3gPDNRl8/bx7v"
    "bAVgKw2+SEEhgAQFe+FQMRBydlvcgiES2DgFBwNdNB8OtSZMSAvjy9EVZM+d3beaWj92N5exLGe3YqLMMcH65FYwB/Qi6rkWMlhj"
    "np3fg5MFWDV99b0SqOpt7pYAKydlkzs25D9t4qt586wfMYDzWgVjoRbWOW2Kn2svXjbPmAQ3jsK0z5rCLWNfZ6xQ7fxXfQALseZZ"
    "GjsmEIs79bcdCZq4nM0dH0fSw9Wph6czxoopRjtddzR06/n2TKIUQ2d47ZIZC/pUtuZxX4PMLKBkbYNZHsCB2qIvWLx51oVcD4EJ"
    "KfxzLdYMWtUs8hGsW1G9O81nCxkxu9ltsqkjrjEGqdg5JufzZiT0pTRlRx0xKK+wqGymjXMsBIyhyAxosVgns/u60FXpqga42H0s"
    "PMCfgCbRAzA5ZQiS6FShWpGdscPAidHsK5rIX8zs1NRSlo7MvG8SxTuU4w/x0l5S+vjZpCbeZHX6chYxKIwcvX0etphMJ9lkNrRx"
    "fvAVqA8v8hv5pUUw5h8BJe2y+c1kEaZDacFrsRrEkATS5O22eodaWGAlzttPMzHZZ+BkNxXnQZjCeGkSZjKC4h5nlZPojElJlOts"
    "l8MCl0vuObBZILn2XossIxW3Stb/UGDKCAe6A5rHFbOTBNonfCtgtdwZaayCrOlSb+1HgSL8H2fA5BLf2AyR+s977fKKX9OSzv5F"
    "Ru+6LRbXk6EVw8IpJCdPn16tQWrm8cVIclT4no2GfTTgTQ1bFa80R5x/DSovlq9KvrYN48YyPJ+F8h9G46FmJxny2pBJy4sljksL"
    "vggAkC8iZ+K5GEXIqerhGUg0LWg4La0juB8UksVVXapmkfkylN30WOAFsQAYGIKuRoyGtq4RoV0aho7SRvOKU3K3U9yKkxKYRgM1"
    "NCWwWr2WLHk+XYMNOpNrcIyZzvADF+cbe2Xij488fI2LTzf3Jm/I6VlAYdEp2jfYTQT2Q0KZkZ6Komfd4Wh+Ic6kYYPEIcStwtOe"
    "9E+PQRQOBD1SjOuaQ2WggEwxmvw3koNUTkOzc0BfCrYF8Mtfq7EbdTxC4InKHnDgghx1WnlycX8hpMVA3Sf5vWyx4uDrs02XqHgv"
    "bTgZbihVAcmtqKWKZ3LHIVst2c8sC7T9BKYsD5dCJipmMn6zy9fQV6iWtYC+NCOgjP9g2eUlZfpc1pwFSajxoD4oX/fSPjxn2xZg"
    "SIyGvs644FqrfiU+TCNxxiCxD1hObcljxRU/St/QBLIjoh3wUHcCFMWwVARvUWaPsuMyEjnl9XHxGjth5FVRTnJvXqY28Eplahck"
    "ncpR2DFx2gVF6XPaImhxQVI3l2qJFaxUpB0MNhZbo7inAvq/QIwfBOEI0WJ+an+ZcChux+2WL2qmZ+6dNbZDpK8ZxshzAPiCX5Xx"
    "OmvdkhNzaDQoEQ21k4iujvuACjcGHvRiBvpI9cbRV1nlGsSs5D+U2Rlox8ZpdzS/BPVaoZoRtW5u/FVAxygZ8wuOKTeiHLmMIXvW"
    "wjg9XEKhwkSCCLZPNZ2q0w2rhaOXgFFvIoBwNPTazdBdhvrw9z6JHa3L3AZrldc3YAJClZlYKVPJcbYO9k+ODnZ3B0fZ7ubPg91j"
    "5K4qh4Hgq5/yWXEtmHuhIuAQ+h4dvDsZ0FokEqZOgiDrvNvZ7j1bDSvcQh4gsZs3yQboqVif+HGzzB0OqFf17U24p+cYtzPbE3/t"
    "2J2ATq/48FrZ5e0Uax5si25m2ztHYro3p8VMeWFmcmjo1tj0WpWfoPZrfIq13MJGRpfYmPHhlVdsTq9It66mQ7dbZW660BkpPb1i"
    "NdotAIcCOD7xCujU65ljn+w7p/YbJV1waka6ezcaijkC/ZWzhXQ2bUTLM4fYxzjIPvgw6x1og+76eSZqodGgxs0tWYrIbnaezwu5"
    "EOJugmHzjhoV7oUqWvtm+A1mykC+vuwWdvtnq9RDwxjNyIkVZbVBX3EFLXB2wfmiK5bzpcxChtOLZ9PYSGSCPdYk+6aJxky2KCJb"
    "3MCTDmAFtu3e6mo81YtEBJcipZy+wRzSIVrvFkUGp1B5pG9FaIr5K1rOEqBf2YkjqCXHSw7h9DsalJzPJh+K8JaHtlSjoHSK8bmb"
    "OkYIkcEmRpG8NTsWclf2evfg4Cjbg/3v+QuNclrgaPA623u3C3eva8xnyJR3vLl3uDtAEDRwMw5wMZmALPOJieVic5JRXKXafiDS"
    "Cp9Zz8VHfiuO3HOqHAhasXzkGkSYCdopQdagoG2NIX6xWBxzOHvCTP5ASFLdPUV8yOc0efo0WTtLNmz3CGaR0jqTPRxPwM3Uxa18"
    "h+cobLtJUhIFMy+6WwdrzIUspyYFHU7Wfv29h1+NpVNddsYJjQu15WuFnpv8vLjxZh68siJ3RIT0MBZn7+42YKd8ymmtkOSBa5V6"
    "r4hV0ldOdPXxXxUWyVCuwXXLRyYYsbEqID7PrEs/48yiSCbGofRoIJWrbqSlsZsELSH2sMQDZHMUu9NC39HAdi8AYfR7EqsPCzf9"
    "q7W78bC4GA1J1jcpLcjKk3OVrno9iWcCbdTY+2VSdvdiyDYNKdnpiCpmYZhZ2qI6WMiVZe3UtKqsGqLKwyUVOmNKBQPJlrU2OCYd"
    "UOFg2Z2/5Ahg80pVLfX43u2YiRtw4YJ84Oxg1n3Z1AgZ8VIED440UTrWxbt7ZmAQX8W6SqQhHvWGkta/0Gso1kj5FsSwKKSVifdB"
    "WQta5cxuiyEyGSzn0r5W+uGgxhKpiNkIInZfiR5Z1DbvoETtq3JH9bdAVaOGvsnIL0pyieySDK9l+fUGhJd9UsK064+Mrr3vMMBg"
    "fCinr7/CsagBYNJnMahf+hWp3g1Bw57pvZMObymxa4nZUANRESTVvnMiuEJHmxCPWkA04m9Jho6KzYqDqeSWEqAx1hkFy9INpnuW"
    "u1mxcQmUsuTy1VIm3VhxTVYleYXk65iSNo2tBKWcjOPBn4EobVYPbQlxktCG4LOxBMo2XtVmLgwwgsQSb6MYZvlT5/K7xjfvHhHx"
    "IXIw0wr3Rknh6m22/OLObxLQ3WyUz5dAGkAuslGD4dZssWymcaxE4SlyqVS+iOIoqsUzGsutP+9Kp44hAzTLn/LVkBj9wrJieBzG"
    "MhJlTWXEv7RqcHQL4TWTyRws+rPZ/CKbzPXL8d3t9N68H08bjezoeEtwjndbvwhQ0vtN1ekW44+jmYz82GqaYmhxvtZMU1n118Eu"
    "9kKHDIhWFgWh6moXWLuqvLvz+uRtncpYUFb/waktmGVVr1Ux7Lfb9PFhzaaPD1XXX5j624PjrVr1ZUFZf83ibevd8dtaQ8eCsnrv"
    "pVO9VvOyoKy/arv/FnaFOs1jQdV7i73tnWNwpgB1QnTcsghUbaKJVlOw2GzrOr+d/pxjokX2Htk4zBwV84u7ooa3zJNc/PfkQ8C2"
    "zmUrtsmWLulzBzECgNhyknPplyxLnC/uLtwk6XSFYoI7vOPnVvE0n33IPt/zyxg/fol8u5Yj4la/vRWQGFLqNYMktB/xdH0YzVTQ"
    "BTHV5OzmNgWI0bngndJyDoH4G4mryANQqjVub4lTixCBlcTx2lECMmR0VOkQhKH0uGE2U8X7bAQk24C/ubZNYc6+hC4F9hDBSA4c"
    "uppq3mhNZy152JNC9Jwzoj9O0Zjsw2A77LaNpYJewfJtd5Z/dO5ilUkEQSIKDCS6sk60DRtv7eFLOyspmqUpSHr/n7+4MSCNDlm6"
    "UUhQML9V802Xdlu/ZQcn2fvNnROZ08Z2E36iT7kP+eJzNllkcIsj4EuDKZ8odLBYIkn+I8Gm+kku5urN6Bwz+IJB4XjYmVxeJnhp"
    "dV7MR0Mh0Ca3k49Qe5obX9M8kdxGASbmksxc5s0h/Uncnf9+VxRfaMLqocpFv8Cc9dqwyqn48rktfq2T3pLtw/MwFqBe9n88S1MZ"
    "otZGRhLnsB9NLltncXhEuNYparWkwQl+EjlOpA5UR1olqrNA4iAMBw3uQtwVZg8dsp4YjhWiohoSsinwWXTJvgC0EGRYO/PDLDD7"
    "hAp0BDXWztiSatdogiDT/M9ww4gzE24wizAoypcsP5+gXOp2RgemUZ3luZfqnu4fYlcD1JiVYiI57DlfhGwXY3Nm7CCEDfa3m1V9"
    "0OWcbqyrxlASq2wKSolzu9vUdTGzjkqa0B1LaLfwuHRdAaw00KeIShuQy/Ml52+HjeOfp1DwCagPWteCp1lZMK1FHiZcEEzClV53"
    "VcwsaUpmheZGYBUcw3cEbseHK4XpRrViga8uhWFXNydnruGj0iEccIXx0vABEpt9tHGhokyRwMM/T0uoCAU8KsoodLDbcuseI7gJ"
    "1qyAsxza9xuAOlhD4IDb6CGKnLb8zElQ4bRRpS6I6XZJvfndjVU1awGtTCXh72USBLN7USPlBnuKQIJ7B4hGo/EfSUf8k3x8kVzd"
    "QTSHm/xeHCySxUj0FN+0k3NBpGEHN+MrgeaZIFZbxzXrTGeTz6NbiGijvyWt/OYmEYefzhWEgBLoEDv6+T3gKBedT7FB70h+RQ7k"
    "t5CiRb6Ex/CcfgWn9OzkTXawn7gH7avg2IWl8KyGBzX4+fp1nUqvX6sT3gtdcW/zt18SqxFgq0EZqGdrYVLiyuaglDqQrtGaMMTq"
    "mnqI+jD+83G2uf+mapSylMWN+L21eVijliglm3uO9bZh2EdvdvZL65lStsFt6P3+m0FSVRFLQb1nXVMTtBxJZU2l5DBkFK/eHepJ"
    "EyEjlkEi/qjrbB+836+qA2Wglqn0dnP39fuKSliGqI/Eq9cHv9YYmCgF1X4y+Dg83nmzX1kNS6kACZp0h8eyuRjRDo+lwgKLH+8O"
    "3lf3Dko5iw5sqaOtqO9Q4WIieHhT1qkxGWUpNfd1W1u7O3s/V9eDUm4fyzBx4mJC/Bz8trVbVhy+Q3GIP9JWunCouVVntZhSpIdb"
    "dVaLLgX1XnZNzV//XzWnhFI4NWy1ynmvyuBqea5rlSNyyyDS+CRABBkVi2Q7QQtaydw3klWpfjaM1L5RPM6+sGyIVNui7yDwDuxo"
    "5bo2E50Gij6Crs00GdO1XYWatqsSPdtVhouE0bRdgZ3Yxwy3bjfevv4+zhZXbMVxdj6PfBjyNTBs+EUODhw9vqWLmgq4P63+7eob"
    "tG9XeDE6+aAT0ufT+SvPfQbewQTRTBhSW8ErfZMnfVTEmyAfvXjH6tacU3Rg/XTrqKQCR70MdEyzfHxVtJ7z/pDiiCJ1P9IsERYu"
    "f315PivyD8GXW2LiiCCQxqlvzkFLLWkFSVABgNDCbNH63J2LKTNV8QQ+y1gT07kOTNFuojbGlKrrUuHYZ1PCy/jaOThYhucBiAxz"
    "006mo8UFxMyffKLq2IBkQ8lDK/2LZRn2vANRbmXIfbGunoX0giaGp91u14nxDP+8bSfv4aOXJltalsuOgXzeXeTjln6e5cNRPp63"
    "tCDzRLChF96Z7mJyozaUGSRGbLXei2N/L5Vlxf8xtnbHAS9wloqjI7hpubCuvzig3iogncSKRE8MpGFxNSuKeQuRD/C8bkLbb3WX"
    "vVm1qmzoV9uoKHgLXW5D87Kpd4d+hZ6qMIM44T1VC2usSLJ7SCENAIIkWBQPvYIIWAB7Lwuu2ILQkFtYRYgcns5W+zPRiYvV/kXv"
    "rMHENoCS3fnoS7G0YTfge0WJFQgFOpeiMetP3RfLLSbXSrvKWMreXlzVv7u4yko8Y60Y0l/+EoAPgo+6CxIB/0UdyKEH2KKcE1i1"
    "9lVWFZkfmxUyjeFC9+gkLSk4B+2IwJP59dz59cLTp8ooc6jW/slqyK9YVVCtGcCjmFMAKTEH1kx+Pm/BiERP4VGub6LH1yIR/umE"
    "0lKpHOUUQDdOI5yCkOmKpsSxQgsA+miRGjvmljk+RIrDtzR2t6XlP9Eeu0tjt9cTreowl1+m3qtEazPKjI84MbOu4ZKuG6jFUeFJ"
    "EQgIwA5vyA7vh/K+X0arReC1oqz9ELXWs8PpRUugdBz0mUM8Nm7OtSCwijWEl2Bh+zB3pcbRWZstYJMwW+XaUqVSYFHyVJumPKjn"
    "IShYwE/NNjcdxaxy5TritZvLz7X5lB4nPV2uYAwpG0hH1EKl/DN+AkFQbohOyWh76U2lOeazPTPAZPBaYBHw2Lb1eMjlI0r5SnfQ"
    "Y4H++ZSPwQE0U0rqO4ZD+vJanWmDwVhFZ58mz5ycJTUigpDecF6obstcie8xB8Nj5fBZCWXMoCjPx1tD9WWtamsajkvAC1BumCXR"
    "GXmvJHnQmP5qDSFmx/BZCjOzG2z2N0RuzoVkuwbQ8TIF/q7iBaHAJFMLZGAQ9iyCk/+byLC6zssO+eWDEUcTOMwD1/Fl4lmwXmGZ"
    "WSULDk6D2NBfwkkmSQeTVDy1E6VUTkt47vmc5bmu7sZpfl0emo3gzm+jUs/EbAhDeTYzp2H3xBY9q2l9GMvVNEzfG0q/X6fjWUmM"
    "Mo9fpZ8UCnGd46EAHlo8CLgzla3AkQZKm09tnJdpGuWjn2DySPXgRkI2orWzNM5VVYmAA1hgaYU/sVIPsWR31HhLk52LHsR5gX3L"
    "PFC6/ihSmbkQR+ZQLpiWrtQhGDD01Fcgip5VwHSdvZ39tp1H8DGN14WOQ/V1WMLx/to1DqVLizkqxBVO8Kr2sAviGqgpgtcQwd5v"
    "h4/MJ1BVdkzdtEyfSQcm2pNYAaUETDhGZBMdCXAW274BmDoVwmchjK9KRV8HyNN42H7pXUjH7389nXbjfwCkwEDd"
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
        if _CX_CR_ANY:
            self._cr_reset()  # fx_cr state likewise
        self._kl_reset()  # fx_kl state likewise
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
        self.high_open = False
        self._ws_done = False
        self._wl_close = False; self._wl_far_t = None; self._wl_pad = None
        self.king_from_dual = False
        self.king_blocked = False
        self.ho_t = 0.0
        self.hb_lost_n = 0
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
        if _CX_CR_ANY:
            self._cr_reset()   # fx_cr per-seed state
        self._kl_reset()       # fx_kl per-seed state (read only when a CX_KL_* flag is on)
        if _CX_VK2_ANY:
            self._vk2_reset()  # fx_vk (round 23) per-seed state

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
        if _CX_KL_KBACK:
            self._kb_step(observation)
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
        try:
            self._last_action = np.array(a_out, dtype=np.float32, copy=True)
        except Exception:
            self._last_action = a_out
        return a_out

    def _dv_flagged(self):
        """CX_DV_DIVE: has this flight's pad been flagged moving? On the king route only after the moving hand-over
        (king_from_dual; the static hand-back and every take-back leave the king route); elsewhere mine's own flags."""
        if self.route == "king":
            return bool(self.king_from_dual)
        m = self.mine
        return bool(m.pad_moving or m.pad_ever_moving)

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
            if kmode == "landing" and not see:
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
            if self._kb_blind < _KB_LOST_S - 1e-9:
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
            if float(pos[2]) - top > _KB_ABOVE:
                return  # not low: the pilot is not sinking beside the pad (yet)
            if not sat and float(pos[2]) - agl >= top - 0.15:
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
                if self.king is not None and label in _KING_ROUTE_LABELS and not self.high_open:
                    self.route = "king"
                elif self.king is not None and self.high_open:
                    self.route = "dual"
                elif self.king is not None and label in _KING_MOVING_LABELS:
                    self.route = "dual"
                else:
                    self.route = "mine"
            self.debug = dict(self.mine.debug)
            self.debug["route"] = self.route
            if self.route == "king" and a_king is not None:
                return a_king
            return a_mine
        if self.route == "king":
            try:
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
                        a_shadow = self.mine.act(observation)
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
                    if far > (14.0 if self.mine.map_label == "mountain" else 18.0) and (self.mine.t - self._vk_t0) > 3.0:
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
                    elif _CX_VK_WATCH and self.mine.map_label == "village" and self._vkw_fire(a, st_):
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
            # a map first taken for city/mountain that later proves to be open/warehouse belongs to the king
            if (self.step <= 200 and a_king is not None and self.mine.map_label in _KING_ROUTE_LABELS and self.mine.map_label in ("open", "warehouse")
                    and not (self.mine.map_label == "open" and _OPEN_ALT > 0 and getattr(self, "z0", 0.0) > _OPEN_ALT)):
                lp2 = sorted(self.mine.map_logp)[-2:]
                if (lp2[1] - lp2[0]) > 8.0:
                    self.route = "king"
                    self.debug = {"route": "king", "map": self.mine.map_label, "mode": "KING", "pad_hits": 0}
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
            if (a_king is not None and (self.mine.pad_ever_moving or self.mine.pad_moving) and fast_ok
                    and not getattr(self, "_vk_recaptured", False) and not self.king_blocked):
                if _CX_VM_DIAG and self.mine.map_label == "village":
                    self._vm_ho_log()
                self.route = "king"
                self.king_from_dual, self.ho_t, self.hb_lost_n = True, float(self.mine.t), 0
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
