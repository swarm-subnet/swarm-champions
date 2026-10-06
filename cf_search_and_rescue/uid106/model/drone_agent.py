import os
from pathlib import Path
import math
import numpy as np
import onnxruntime as ort
_H = Path(__file__).resolve().parent
ON = os.environ.get('KT_ON', '1') == '1'
AV_D = 4.0
AV_REL = 0.7
AV_ROT = 45.0
AV_HOLD = 15
AV_T0 = 100
AV_TYPES = ('forest', 'city')
AVF_ROT = 30.0
AVF_D = 4.0
VRV_ON = os.environ.get('KT_VRV', '1') == '1'
VRV_TYPES = tuple((x for x in os.environ.get('KT_VRV_TYPES', 'city').split(',') if x))
VRV_X = os.environ.get('KT_VRV_X', '1') == '1'
VRV_P = float(os.environ.get('KT_VRV_P', '0.9'))
VRV_R = float(os.environ.get('KT_VRV_R', '1.5'))
VRV_KR = float(os.environ.get('KT_VRV_KR', '3.0'))
VRV_NL = int(os.environ.get('KT_VRV_NL', '25'))
VRV_NH = int(os.environ.get('KT_VRV_NH', '15'))
VRV_F = float(os.environ.get('KT_VRV_F', '0.3'))
VRV_MAX = int(os.environ.get('KT_VRV_MAX', '2'))
VRV_SUP = int(os.environ.get('KT_VRV_SUP', '100'))
VRV_FB = os.environ.get('KT_VRV_FB', '1') == '1'
VRV_RGB = os.environ.get('KT_VRV_RGB', '1') == '1'
VRV_T = float(os.environ.get('KT_VRV_T', '0.5'))
VRV_UV = float(os.environ.get('KT_VRV_UV', '0.2'))
RLK_ON = os.environ.get('KT_RLK', '1') == '1'
RLK_NH = int(os.environ.get('KT_RLK_NH', '60'))
RLK_F = float(os.environ.get('KT_RLK_F', '0.3'))
RLK_V = float(os.environ.get('KT_RLK_V', '-0.5'))
RLK_DH = float(os.environ.get('KT_RLK_DH', '10.0'))
RLK_H = float(os.environ.get('KT_RLK_H', '3.3'))
RLK_T = int(os.environ.get('KT_RLK_T', '250'))
RLK_D0 = float(os.environ.get('KT_RLK_D0', '6.0'))
RLK_D1 = float(os.environ.get('KT_RLK_D1', '7.8'))
RLK_VZ = float(os.environ.get('KT_RLK_VZ', '0.8'))
RLK_AGL = float(os.environ.get('KT_RLK_AGL', '2.8'))
RLK_ACC = float(os.environ.get('KT_RLK_ACC', '0.06'))
RLK_YAW = float(os.environ.get('KT_RLK_YAW', '0.26'))
RLK_TYPES = tuple((x for x in os.environ.get('KT_RLK_TYPES', 'city').split(',') if x))
VRV_OOF = float(os.environ.get('KT_VRV_OOF', '0'))
TYX_ON = os.environ.get('KT_TYX', '1') == '1'
NRC_ON = os.environ.get('KT_NRC', '0') == '1'
NRC_TYPES = tuple((x for x in os.environ.get('KT_NRC_TYPES', 'city').split(',') if x))
NRC_HLO = float(os.environ.get('KT_NRC_HLO', '9.5'))
NRC_HHI = float(os.environ.get('KT_NRC_HHI', '14.0'))
NRC_ZLO = float(os.environ.get('KT_NRC_ZLO', '10.5'))
NRC_ZC = float(os.environ.get('KT_NRC_ZC', '21.5'))
NRC_ZH = float(os.environ.get('KT_NRC_ZH', '1.0'))
NRC_M = float(os.environ.get('KT_NRC_M', '1.5'))
NRC_VZ = float(os.environ.get('KT_NRC_VZ', '1.5'))
NRC_EIN = float(os.environ.get('KT_NRC_EIN', '0.15'))
NRC_RQ = float(os.environ.get('KT_NRC_RQ', '8.0'))
NRC_RING = float(os.environ.get('KT_NRC_RING', '0.5'))
NRC_DESC = float(os.environ.get('KT_NRC_DESC', '0.2'))
NRC_GL = os.environ.get('KT_NRC_GL', '1') == '1'
NRC_HV = float(os.environ.get('KT_NRC_HV', '2.0'))
NRC_HDIR = float(os.environ.get('KT_NRC_HDIR', '0.2'))
NRC_SH = os.environ.get('KT_NRC_SH', '0') == '1'
NRC_TX = float(os.environ.get('KT_NRC_TX', '1.0'))
NRC_TXZ = float(os.environ.get('KT_NRC_TXZ', '2.5'))
FDL_ON = os.environ.get('KT_FDL', '1') == '1'
FDL_TYPES = tuple((x for x in os.environ.get('KT_FDL_TYPES', 'city').split(',') if x))
FDL_K = int(os.environ.get('KT_FDL_K', '5'))
FDL_P = float(os.environ.get('KT_FDL_P', '0.80'))
FDL_RNG = float(os.environ.get('KT_FDL_RNG', '18'))
FDL_TOL = float(os.environ.get('KT_FDL_TOL', '2.5'))
FDL_SPAN = int(os.environ.get('KT_FDL_SPAN', '10'))
FDL_WIN = int(os.environ.get('KT_FDL_WIN', '200'))
FDL_CR = float(os.environ.get('KT_FDL_CR', '32'))
FDL_T0 = int(os.environ.get('KT_FDL_T0', '100'))
FDL_DUD = int(os.environ.get('KT_FDL_DUD', '100'))
FDL_TMAX = int(os.environ.get('KT_FDL_TMAX', '1000'))
FDL_BAN = float(os.environ.get('KT_FDL_BAN', '4.0'))
FDL_NEAR = float(os.environ.get('KT_FDL_NEAR', '6.0'))
FDL_LOCKR = float(os.environ.get('KT_FDL_LOCKR', '4.0'))
FDL_FMAX = float(os.environ.get('KT_FDL_FMAX', '0.2'))
FDL_LM = os.environ.get('KT_FDL_LM', '0') == '1'
FDL_LM_TYPES = tuple((x for x in os.environ.get('KT_FDL_LM_TYPES', 'mountain').split(',') if x))
FDL_LM_R = float(os.environ.get('KT_FDL_LM_R', '12'))
FDL_LA = os.environ.get('KT_FDL_LA', '0') == '1'
FG_ON = os.environ.get('KT_FG', '1') == '1'
SS_ON = os.environ.get('KT_SS', '1') == '1'
SS_Z = float(os.environ.get('KT_SS_Z', '0.3'))
SS_LIFT = float(os.environ.get('KT_SS_LIFT', '0.15'))
SS_VZ = float(os.environ.get('KT_SS_VZ', '0.2'))
SS_FRAC = float(os.environ.get('KT_SS_FRAC', '0.5'))
SS_MIN_HEAD = int(os.environ.get('KT_SS_MIN_HEAD', '4'))
SS_HOR = float(os.environ.get('KT_SS_HOR', '1.5'))
SS_OUT = float(os.environ.get('KT_SS_OUT', '1.3'))
SS_OUT_V = float(os.environ.get('KT_SS_OUT_V', '0.3'))
SS_MAXT = int(os.environ.get('KT_SS_MAXT', '450'))
SS_F0 = float(os.environ.get('KT_SS_F0', '0.3'))
SS_SIDE = float(os.environ.get('KT_SS_SIDE', '90'))
SS_TOP = float(os.environ.get('KT_SS_TOP', '0.55'))
SS_HMAX = float(os.environ.get('KT_SS_HMAX', '1.0'))
MX_ON = os.environ.get('KT_MX', '1') == '1'
MX_AT = float(os.environ.get('KT_MX_AT', '28'))
MX_EXT = int(os.environ.get('KT_MX_EXT', '70'))
MX_TYPES = tuple((x for x in os.environ.get('KT_MX_TYPES', 'warehouse,city').split(',') if x))
P3_ON = os.environ.get('KT_P3', '1') == '1'
P3_T = int(os.environ.get('KT_P3_T', '1000'))
P3_BAN_R = float(os.environ.get('KT_P3_BAN_R', '3.0'))
FPR_ON = os.environ.get('KT_FPR', '1') == '1'
FPR_PK = FPR_ON and os.environ.get('KT_FPR_PK', '1') == '1'
FPR_PKN = int(os.environ.get('KT_FPR_PKN', '450'))
FPR_PKR = float(os.environ.get('KT_FPR_PKR', '1.5'))
FPR_PKL = float(os.environ.get('KT_FPR_PKL', '3.0'))
FPR_PKW = int(os.environ.get('KT_FPR_PKW', '300'))
FPR_PKR2 = os.environ.get('KT_FPR_PKR2', '1') == '1'
FPR_PKZ = float(os.environ.get('KT_FPR_PKZ', '0.3'))
FPR_CG = FPR_ON and os.environ.get('KT_FPR_CG', '1') == '1'
FPR_CGR = float(os.environ.get('KT_FPR_CGR', '33.0'))
P3_PAUSE = int(os.environ.get('KT_P3_PAUSE', '350'))
FG_PH = tuple((int(x) for x in os.environ.get('KT_FG_PH', '3').split(',')))
FG_D = float(os.environ.get('KT_FG_D', '3.5'))
FG_D2 = float(os.environ.get('KT_FG_D2', '3.0'))
FG_W = float(os.environ.get('KT_FG_W', '0.45'))
FG_ROWS = int(os.environ.get('KT_FG_ROWS', '18'))
FG_MINH = float(os.environ.get('KT_FG_MINH', '1.5'))
FG_HOLD = int(os.environ.get('KT_FG_HOLD', '20'))
FG_V = float(os.environ.get('KT_FG_V', '1.5'))
FG_DN = float(os.environ.get('KT_FG_DN', '0.02'))
FG_TURN = float(os.environ.get('KT_FG_TURN', '5'))
FG_MAXB = float(os.environ.get('KT_FG_MAXB', '40'))
FG_ALAT = float(os.environ.get('KT_FG_ALAT', '3.0'))
FG_TILT = float(os.environ.get('KT_FG_TILT', '25'))
STALL_N = int(os.environ.get('KT_STALL_N', '150'))
STALL_V = float(os.environ.get('KT_STALL_V', '0.8'))
STALL_LOOK = float(os.environ.get('KT_STALL_LOOK', '15.0'))
STALL_HOLD = int(os.environ.get('KT_STALL_HOLD', '100'))
STALL_TYPES = tuple((t for t in os.environ.get('KT_STALL_TYPES', 'mountain').split(',') if t))
CFG = {'forest': (350, 15, 1000000000.0, 8.0, 5.0, 40, 120, 8.0, 0, 0.75), 'city': (450, 25, 10.0, 6.0, 5.0, 40, 250, 4.5, 1), 'mountain': (450, 15, 1000000000.0, 12.0, 8.0, 25, 120, 8.0, 0, 0.75), 'village': (450, 15, 10.0, 8.0, 5.0, 40, 120, 8.0, 0, 0.75), 'warehouse': (450, 15, 10.0, 6.0, 4.0, 40, 120, 6.0, 0, 0.75)}
MTN_STRONG = (450, 15, 1000000000.0, 12.0, 8.0, 25, 120, 8.0, 0, 0.75)

def _cfg_env(t, c):
    l = os.environ.get('KT_LANE_' + t.upper())
    st = os.environ.get('KT_STEP_' + t.upper())
    if l is None and st is None:
        return c
    c = list(c)
    if l is not None:
        c[3] = float(l)
    if st is not None:
        c[4] = float(st)
    return tuple(c)
CFG = {t: _cfg_env(t, c) for t, c in CFG.items()}
MTN_STRONG = _cfg_env('mountain', MTN_STRONG)
MTN_THR = 0.5
M_AGLF = os.environ.get('KT_M_AGLF', '1') == '1'
M_AGLF_LO = 1.4
M_AGLF_HI = 2.2
M_RPT = os.environ.get('KT_M_RPT', '1') == '1'
M_FARLOCK = os.environ.get('KT_M_FARLOCK', '1') == '1'
FARLOCK_R = 36.0
M_LANES = os.environ.get('KT_M_LANES', '0') == '1'
if M_LANES:
    CFG['mountain'] = CFG['mountain'][:9] + (1.0,)
    MTN_STRONG = MTN_STRONG[:9] + (1.0,)
LLT_ON = os.environ.get('KT_LLT', '1') == '1'
LLT_R = 33.0
LLT_BAN = 5.0
LLT_N = 10
LLT_V = -0.85
LLT_MAXT = 1750
LLT_RING = 1.5
LLT_ZHI = 5.0
LLT_ZLO = 2.3
LLT_VS = 0.4
LLT_HOLD = 75
LLT_AGL = 2.0

class _LLDone(Exception):
    pass
MP_ON = os.environ.get('KT_MP', '1') == '1'
MP_MOUT = float(os.environ.get('KT_MP_MOUT', '0.3'))
MP_ALL = os.environ.get('KT_MP_ALL', '0') == '1'
MP_GT = os.environ.get('KT_MP_GT', '1') == '1'
MP_PROG = os.environ.get('KT_MP_PROG', '1') == '1'
MP_BL = float(os.environ.get('KT_MP_BL', '1.0'))
MP_BC = int(os.environ.get('KT_MP_BC', '10'))
MP_TAIL = os.environ.get('KT_MP_TAIL', '1') == '1'
MP_DMIN = float(os.environ.get('KT_MP_DMIN', '6.0'))
MP_Q = 0.05
MP_RV = 80.0
MP_NCORE = 8
MP_RS = 9.0
MP_V = 2.6
MP_TP = 1.5
MP_C0 = 2.0
MP_STOP = 0.1
MP_PROG_M = 1.0
FF_ON = os.environ.get('KT_FF', '1') == '1'
FF_TYPES = tuple((t for t in os.environ.get('KT_FF_TYPES', 'forest').split(',') if t))
FF_FRACS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
RL_ON = os.environ.get('KT_RL', '1') == '1'
RL_N = int(os.environ.get('KT_RL_N', '40'))
RL_BAR = float(os.environ.get('KT_RL_BAR', '0.55'))
RL_TYPES = tuple((t for t in os.environ.get('KT_RL_TYPES', 'city,forest').split(',') if t))
LADDER = os.environ.get('KT_LADDER', '1') == '1'
TYPES = tuple((t for t in os.environ.get('KT_TYPES', 'forest,village,city,mountain,warehouse').split(',') if t))
LB = ('city', 'open', 'mountain', 'village', 'warehouse', 'forest')
SIM_DT = 0.02
VW = float(os.environ.get('KT_VETO_WARM', '0.02'))
RP_ON = os.environ.get('KT_RP', '1') == '1'
RP_T = float(os.environ.get('KT_RP_T', '0.713'))
RP_HMAX = float(os.environ.get('KT_RP_HMAX', '1.0'))
RP_T0 = 150
RP_EVERY = 75
RP_CAP = 40
RP_CHASE_MIN = 12
RP_CHASE_TICKS = 36
RP_RANGE_MAX = 20.0
RP_IMG_MAX = 29.0
RP_CLUE_R = 55.0
RP_LATCH_N = 2
RP_LATCH_TOL = 5.0
RP_EMA = 0.3
TCAP_ON = os.environ.get('KT_TCAP', '1') == '1'
TCAP_K = float(os.environ.get('KT_TCAP_K', '3.55'))
TCAP_R = float(os.environ.get('KT_TCAP_R', '2.5'))
TCAP_AGL = float(os.environ.get('KT_TCAP_AGL', '12.0'))
TCAP_M = float(os.environ.get('KT_TCAP_M', '0.7'))
TCAP_TYPES = tuple((x for x in os.environ.get('KT_TCAP_TYPES', 'village').split(',') if x))
RG_ON = os.environ.get('KT_RG', '1') == '1'
RG_TYPES = tuple((x for x in os.environ.get('KT_RG_TYPES', 'village,warehouse').split(',') if x))
RG_WAIT = int(os.environ.get('KT_RG_WAIT', '150'))
RG_D = float(os.environ.get('KT_RG_D', '1.0'))
RG_R = float(os.environ.get('KT_RG_R', '1.5'))
RG_V = float(os.environ.get('KT_RG_V', '0.45'))
RG_LAPS = float(os.environ.get('KT_RG_LAPS', '1.0'))
WHC_ON = os.environ.get('KT_WHC', '1') == '1'
WHC_MAX = float(os.environ.get('KT_WHC_MAX', '5.5'))
RP_TERM_R = 6.0
RP_HOVER_UP = 3.2
RP_DUD_TICKS = 300
RP_AGL_MIN = 2.0
RP_RING = 1.6
RP_TERM_MAX = 1200
TB_T = 50
TB_SEP = 6.0
TB_AGREE = 3.0
_DMIN, _DMAX = (0.5, 30.0)
_CAMF, _CAMU, _HT = (0.13, 0.05, 1.0)

def _sig(x):
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, float(x)))))

def _axes(rpy):
    r, p, y = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y))
    fwd = np.array([cy * cp, sy * cp, -sp], np.float64)
    up = np.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr], np.float64)
    return (fwd, up, np.cross(fwd, up))

def _mworld(u, v, cz, pos, rpy):
    fwd, up, right = _axes(rpy)
    cam = np.asarray(pos, np.float64) + fwd * _CAMF + up * _CAMU
    return cam + right * (u * _HT * cz) + up * (v * _HT * cz) + fwd * cz
CP3H = False
FB_P3 = os.environ.get('KT_FB_P3', '1') == '1'
FB_DUD = os.environ.get('KT_FB_DUD', '1') == '1'
FB_R2ON = os.environ.get('KT_FB_R2ON', '0') == '1'
FB_BX = os.environ.get('KT_FB_BX', '1') == '1'
FB_BOX = float(os.environ.get('KT_FB_BOX', '34.0'))
DG_ON = False
DG_R = float(os.environ.get('KT_DG_R', '34.0'))
FSL_ON = os.environ.get('KT_FSL', '1') == '1'
FSL_PH = tuple((int(x) for x in os.environ.get('KT_FSL_PH', '3').split(',') if x))
FSL_Z0 = 0.3
FSL_R = float(os.environ.get('KT_FSL_R', '0.40'))
FSL_A = float(os.environ.get('KT_FSL_A', '2.5'))
FSL_M = float(os.environ.get('KT_FSL_M', '0.70'))
FSL_NEAR = 5.0
FSL_MEMT = int(os.environ.get('KT_FSL_MEMT', '250'))
FSL_VOX = 0.15
FSL_BLIND = os.environ.get('KT_FSL_BLIND', '1') == '1'
FSL_BV = float(os.environ.get('KT_FSL_BV', '1.5'))
FSL_BFOV = math.radians(float(os.environ.get('KT_FSL_BFOV', '50')))
FSL_P3V = float(os.environ.get('KT_FSL_P3V', '1.2'))
FSL_P3C = float(os.environ.get('KT_FSL_P3C', '1.0'))
FSL_P3UP = float(os.environ.get('KT_FSL_P3UP', '0.02'))
FSL_AGL = os.environ.get('KT_FSL_AGL', '1') == '1'
FSL_AGLM = float(os.environ.get('KT_FSL_AGLM', '0.40'))
FSX_ON = os.environ.get('KT_FSL_SX', '1') == '1'
FSX_N = int(os.environ.get('KT_FSL_SXN', '50'))
FSX_V = 0.15
FSX_BACK = float(os.environ.get('KT_FSL_SXB', '1.5'))
FSX_SPD = 0.5
FSX_T = 150
FSX_AVH = 100
FSX_SEC = math.radians(45.0)
FSX_BLK = 300
FSX_MAX = 6
FSX_CHK = os.environ.get('KT_FSL_SXC', '1') == '1'
FSX_R2 = os.environ.get('KT_FSL_SXR', '1') == '1'
FSV_ON = os.environ.get('KT_FSL_VIA', '1') == '1'
FSV_R = 3.0
FSV_T = 200
TR_ON = os.environ.get('KT_TR', '1') == '1'
TR_N = int(os.environ.get('KT_TR_N', '50'))
TR_PROG = float(os.environ.get('KT_TR_PROG', '0.3'))
TR_STEP = float(os.environ.get('KT_TR_STEP', '2.0'))
TR_R = float(os.environ.get('KT_TR_R', '0.45'))
TR_M = float(os.environ.get('KT_TR_M', '0.7'))
TR_V = float(os.environ.get('KT_TR_V', '1.0'))
TR_HOLD = int(os.environ.get('KT_TR_HOLD', '60'))
TR_MAX = int(os.environ.get('KT_TR_MAX', '300'))
TR_AGREE = float(os.environ.get('KT_TR_AGREE', '2.5'))
TR_P2D = float(os.environ.get('KT_TR_P2D', '15.0'))
TR_P2 = os.environ.get('KT_TR_P2', '0') == '1'
TR_HZ = float(os.environ.get('KT_TR_HZ', '1.2'))
TR_RLV = float(os.environ.get('KT_TR_RLV', '0.25'))
TR_RLC = math.cos(math.radians(float(os.environ.get('KT_TR_RLA', '45'))))
TR_RLN = int(os.environ.get('KT_TR_RLN', '10'))
FAU_ON = os.environ.get('KT_FAU', '1') == '1'
FAU_N = int(os.environ.get('KT_FAU_N', '75'))
FAU_PROG = float(os.environ.get('KT_FAU_PROG', '0.3'))
FAU_HMIN = float(os.environ.get('KT_FAU_HMIN', '1.5'))
FAU_HMAX = float(os.environ.get('KT_FAU_HMAX', '6.0'))
FAU_C = float(os.environ.get('KT_FAU_C', '0.55'))
FAU_C2 = float(os.environ.get('KT_FAU_C2', '0.42'))
FAU_CP = float(os.environ.get('KT_FAU_CP', '0.85'))
FAU_SR = float(os.environ.get('KT_FAU_SR', '0.35'))
FAU_RS = float(os.environ.get('KT_FAU_RS', '0.35'))
FAU_V = float(os.environ.get('KT_FAU_V', '0.8'))
FAU_T = int(os.environ.get('KT_FAU_T', '400'))
FAU_MAXE = int(os.environ.get('KT_FAU_MAXE', '2'))
FAU_GR = float(os.environ.get('KT_FAU_GR', '1.1'))
FAU_GZ = float(os.environ.get('KT_FAU_GZ', '0.45'))
FAU_RP = int(os.environ.get('KT_FAU_RP', '25'))
FAU_COOL = int(os.environ.get('KT_FAU_COOL', '100'))
FAU_RES = float(os.environ.get('KT_FAU_RES', '0.3'))
FAU_PAD = float(os.environ.get('KT_FAU_PAD', '2.5'))
FAU_ZG = float(os.environ.get('KT_FAU_ZG', '1.2'))
FAU_ZD = float(os.environ.get('KT_FAU_ZD', '1.5'))
FAU_ZU = float(os.environ.get('KT_FAU_ZU', '0.6'))
FAU_LOOK = float(os.environ.get('KT_FAU_LOOK', '0.9'))
FAU_P3 = os.environ.get('KT_FAU_P3', '1') == '1'
FAU_STALL = int(os.environ.get('KT_FAU_STALL', '150'))
FAU_HOLD = int(os.environ.get('KT_FAU_HOLD', '150'))
FAU_DZ = float(os.environ.get('KT_FAU_DZ', '1.0'))
FAU_BDEG = math.radians(float(os.environ.get('KT_FAU_BDEG', '90')))
FAU_BV = float(os.environ.get('KT_FAU_BV', '0.5'))
_TR_AZ = np.radians(np.arange(-180.0, 180.0, 15.0))
_TR_EL = np.radians(np.array([-15.0, 0.0, 15.0, 30.0]))
R2A_ON = os.environ.get('KT_R2A', '1') == '1'
R2A_TOL = float(os.environ.get('KT_R2A_TOL', '2.5'))
R2NC_ON = os.environ.get('KT_R2NC', '1') == '1'
M3L_ON = os.environ.get('KT_M3L', '1') == '1'
M3L_DZ = float(os.environ.get('KT_M3L_DZ', '4.0'))
M3L_LOW = float(os.environ.get('KT_M3L_LOW', '5.0'))
M3L_XO = os.environ.get('KT_M3L_XO', '1') == '1'
VF_ON = os.environ.get('KT_VF', '1') == '1'
VF_LOW = float(os.environ.get('KT_VF_LOW', '5.0'))
VF_DZ = float(os.environ.get('KT_VF_DZ', '4.0'))
R2A_LK = os.environ.get('KT_R2A_LK', '1') == '1'
R2NC_D = float(os.environ.get('KT_R2NC_D', '8.5'))
DIB_ON = os.environ.get('KT_DIB', '1') == '1'
DIB_R = float(os.environ.get('KT_DIB_R', '1.5'))
DIB_HI = float(os.environ.get('KT_DIB_HI', '3.95'))
DIB_TGT = float(os.environ.get('KT_DIB_TGT', '3.65'))
DIB_VZ = float(os.environ.get('KT_DIB_VZ', '0.6'))
DIB_VMAX = float(os.environ.get('KT_DIB_VMAX', '0.9'))
DIB_HV = float(os.environ.get('KT_DIB_HV', '1.0'))
DIB_WAIT = int(os.environ.get('KT_DIB_WAIT', '110'))
DIB_DRY = os.environ.get('KT_DIB_DRY', '0') == '1'
HRZ_ON = os.environ.get('KT_HRZ', '1') == '1'
HRZ_DRY = os.environ.get('KT_HRZ_DRY', '0') == '1'
HRZ_R = float(os.environ.get('KT_HRZ_R', '1.6'))
HRZ_ELO = float(os.environ.get('KT_HRZ_ELO', '2.3'))
HRZ_EHI = float(os.environ.get('KT_HRZ_EHI', '3.45'))
HRZ_V = float(os.environ.get('KT_HRZ_V', '1.0'))
HRZ_VZ = float(os.environ.get('KT_HRZ_VZ', '0.3'))
HRZ_WAIT = int(os.environ.get('KT_HRZ_WAIT', '110'))
HRZ_DZ = float(os.environ.get('KT_HRZ_DZ', '3.7'))
HRZ_VUP = float(os.environ.get('KT_HRZ_VUP', '0.6'))
HRZ_VMAX = float(os.environ.get('KT_HRZ_VMAX', '0.9'))
HRZ_HOLD = int(os.environ.get('KT_HRZ_HOLD', '150'))
HRZ_TMAX = int(os.environ.get('KT_HRZ_TMAX', '300'))
HRZ_ZMAX = float(os.environ.get('KT_HRZ_ZMAX', '3.9'))
HRX_LOG = os.environ.get('KT_HRX_LOG', '1') == '1'
_FSL_B = (np.arange(64) * 4 + 2.0 - 128.0) / 128.0
_FSL_AZ, _FSL_EL = np.meshgrid(np.radians(np.arange(-40.0, 40.1, 8.0)), np.radians(np.array([0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0])), indexing='ij')
_FSL_AZ = _FSL_AZ.ravel()
_FSL_EL = _FSL_EL.ravel()

def _fsl_cloud(depth, pos, rpy):
    z = np.asarray(depth, np.float32).reshape(256, 256) * (_DMAX - _DMIN) + _DMIN
    zp = z.reshape(64, 4, 64, 4).min(axis=(1, 3))
    iy, ix = np.nonzero(zp < FSL_NEAR)
    if iy.size == 0:
        return np.zeros((0, 3))
    zz = zp[iy, ix].astype(np.float64)
    u = _FSL_B[ix]
    v = -_FSL_B[iy]
    fwd, up, right = _axes(rpy)
    cam = np.asarray(pos, np.float64) + fwd * _CAMF + up * _CAMU
    return cam[None, :] + zz[:, None] * fwd[None, :] + (u * zz)[:, None] * right[None, :] + (v * zz)[:, None] * up[None, :]

def _fsl_free(Prel, W, R):
    W = np.atleast_2d(W)
    if Prel.shape[0] == 0:
        return np.full(W.shape[0], 99.0)
    a = Prel @ W.T
    p2 = np.einsum('ij,ij->i', Prel, Prel)[:, None] - a * a
    return np.where((a > 0.0) & (p2 < R * R), a, 99.0).min(axis=0)

def _fsl_v(free):
    return np.sqrt(2.0 * FSL_A * np.maximum(0.0, np.asarray(free, np.float64) - FSL_M))

def _fau_dil(A):
    B = A.copy()
    B[1:] |= A[:-1]
    B[:-1] |= A[1:]
    C = B.copy()
    C[:, 1:] |= B[:, :-1]
    C[:, :-1] |= B[:, 1:]
    D = C.copy()
    D[:, :, 1:] |= C[:, :, :-1]
    D[:, :, :-1] |= C[:, :, 1:]
    return D

def _fau_plan(P, pos, T, gz):
    res = FAU_RES
    pos = np.asarray(pos, np.float64)
    T = np.asarray(T, np.float64)
    lo = np.minimum(pos, T) - FAU_PAD
    hi = np.maximum(pos, T) + FAU_PAD
    lo[2] = max(gz + FAU_ZG, min(float(pos[2]), float(T[2])) - FAU_ZD)
    hi[2] = max(float(T[2]) + FAU_ZU, float(pos[2]) + 0.15)
    if hi[2] - lo[2] < 2.0 * res:
        return (None, None)
    n = np.floor((hi - lo) / res).astype(np.int64) + 1
    if int(n.prod()) > 90000:
        return (None, None)
    nx, ny, nz = (int(n[0]), int(n[1]), int(n[2]))
    tiers = [c for c in (FAU_CP, FAU_C, FAU_C2) if c > 0.0]
    R = max(tiers)
    lev = np.zeros(nx * ny * nz, np.int8)
    occ = [np.zeros(nx * ny * nz, bool) for _ in tiers]
    if P is not None and len(P):
        m = np.all((P >= lo - R) & (P <= hi + R), axis=1)
        Q = P[m]
        if len(Q):
            if len(Q) > 400:
                iq = np.floor(Q / 0.1).astype(np.int64) + 100000
                Q = Q[np.unique((iq[:, 0] * 200003 + iq[:, 1]) * 200003 + iq[:, 2], return_index=True)[1]]
            k = int(math.ceil(R / res))
            ax = np.arange(-k, k + 1)
            O = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
            O = O[np.sqrt((O * O).sum(1)) * res <= R + res]
            b = np.floor((Q - lo) / res + 0.5).astype(np.int64)
            for s in range(0, len(Q), 600):
                I = b[s:s + 600, None, :] + O[None, :, :]
                ok = np.all((I >= 0) & (I < n), axis=2)
                d = np.sqrt(((lo + I * res - Q[s:s + 600, None, :]) ** 2).sum(2))
                F = (I[..., 0] * ny + I[..., 1]) * nz + I[..., 2]
                for ti, c in enumerate(tiers):
                    occ[ti][F[ok & (d < c)]] = True
    for ti in range(len(tiers)):
        lev += (~occ[ti]).astype(np.int8)
    lev = lev.reshape(nx, ny, nz)
    gi, gj, gk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    cx = lo[0] + gi * res
    cy = lo[1] + gj * res
    cz = lo[2] + gk * res
    near_s = np.sqrt((cx - pos[0]) ** 2 + (cy - pos[1]) ** 2 + (cz - pos[2]) ** 2) <= FAU_SR
    goal0 = (np.hypot(cx - T[0], cy - T[1]) <= FAU_GR) & (np.abs(cz - T[2]) <= FAU_GZ)
    si = np.clip(np.floor((pos - lo) / res + 0.5).astype(np.int64), 0, n - 1)
    si = (int(si[0]), int(si[1]), int(si[2]))
    for ti, c in enumerate(tiers):
        free = (lev >= len(tiers) - ti) | near_s
        goal = goal0 & free
        if not goal.any():
            continue
        D = np.full((nx, ny, nz), -1, np.int32)
        D[goal] = 0
        fr = goal
        step = 0
        while D[si] < 0:
            step += 1
            nb = _fau_dil(fr) & free & (D < 0)
            if not nb.any() or step > nx + ny + nz:
                break
            D[nb] = step
            fr = nb
        if D[si] < 0:
            continue
        path = [pos.copy()]
        cur = si
        while D[cur] > 0:
            best = None
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        a, b2, e = (cur[0] + di, cur[1] + dj, cur[2] + dk)
                        if (di or dj or dk) and 0 <= a < nx and (0 <= b2 < ny) and (0 <= e < nz) and (D[a, b2, e] == D[cur] - 1):
                            key = (-int(lev[a, b2, e]), di * di + dj * dj + dk * dk)
                            if best is None or key < best[0]:
                                best = (key, (a, b2, e))
            if best is None:
                break
            cur = best[1]
            path.append(lo + np.asarray(cur, np.float64) * res)
        if D[cur] != 0:
            continue
        return (np.asarray(path, np.float64), c)
    return (None, None)
OMB_ON = os.environ.get('KT_OMB', '1') == '1'
OMB_TYPES = tuple((x for x in os.environ.get('KT_OMB_TYPES', 'forest').split(',') if x))
CHL_ON = os.environ.get('KT_CHL', '1') == '1'
CHL_TYPES = tuple((x for x in os.environ.get('KT_CHL_TYPES', 'city,village').split(',') if x))
CHL_X = os.environ.get('KT_CHL_X', '1') == '1'
CHL_XP = float(os.environ.get('KT_CHL_XP', '0.15'))
CHL_XN = int(os.environ.get('KT_CHL_XN', '40'))
CHL_HC = float(os.environ.get('KT_CHL_HC', '0.4'))
CHL_N = int(os.environ.get('KT_CHL_N', '100'))
CHL_RSEE = float(os.environ.get('KT_CHL_RSEE', '20.0'))
CHL_HMIN = float(os.environ.get('KT_CHL_HMIN', '2.0'))
CHL_RW = float(os.environ.get('KT_CHL_RW', '12.0'))
CHL_RQ = float(os.environ.get('KT_CHL_RQ', '6.0'))
CHL_LINK = int(os.environ.get('KT_CHL_LINK', '1'))
CHL_EVERY = int(os.environ.get('KT_CHL_EVERY', '5'))
CHL_NOTCH = float(os.environ.get('KT_CHL_NOTCH', '0.5'))
CHL_XMAX = float(os.environ.get('KT_CHL_XMAX', '3.2'))
CHL_M = float(os.environ.get('KT_CHL_M', '0.3'))
CHL_TOPM = float(os.environ.get('KT_CHL_TOPM', '0.3'))
CHL_ACC = float(os.environ.get('KT_CHL_ACC', '3.0'))
CHL_TAU = float(os.environ.get('KT_CHL_TAU', '0.12'))
CHL_PUSH = float(os.environ.get('KT_CHL_PUSH', '0.3'))
CMK_ON = os.environ.get('KT_CMK', '1') == '1'
CMK_TYPES = tuple((x for x in os.environ.get('KT_CMK_TYPES', 'city,village').split(',') if x))
CMK_X = os.environ.get('KT_CMK_X', '1') == '1'
CMK_HC = float(os.environ.get('KT_CMK_HC', '8.0'))
CMK_HT = float(os.environ.get('KT_CMK_HT', '13.5'))
CMK_MT = float(os.environ.get('KT_CMK_MT', '3.4'))
CMK_ZT = float(os.environ.get('KT_CMK_ZT', '13.0'))
CMK_OF = os.environ.get('KT_CMK_OF', '0') == '1'
CMK_MO = float(os.environ.get('KT_CMK_MO', '1.5'))
CMK_ZC = float(os.environ.get('KT_CMK_ZC', '21.5'))
CMK_RQ = float(os.environ.get('KT_CMK_RQ', '8.0'))
CMK_ACC = float(os.environ.get('KT_CMK_ACC', '2.5'))
CMK_TAU = float(os.environ.get('KT_CMK_TAU', '0.15'))
CMK_PUSH = float(os.environ.get('KT_CMK_PUSH', '0.3'))
CMK_MNEAR = float(os.environ.get('KT_CMK_MNEAR', '1.0'))
CMK_TGAP = float(os.environ.get('KT_CMK_TGAP', '0.5'))
CMK_VMIN = float(os.environ.get('KT_CMK_VMIN', '0.3'))
CMK_MODES = tuple((int(x) for x in os.environ.get('KT_CMK_MODES', '1,2').split(',') if x))
CMK_SLV = float(os.environ.get('KT_CMK_SLV', '1.2'))
CMK_SLH = int(os.environ.get('KT_CMK_SLH', '50'))
CMK_G = float(os.environ.get('KT_CMK_G', '0.8'))
CMK_GW = int(os.environ.get('KT_CMK_GW', '250'))
CMK_GN = int(os.environ.get('KT_CMK_GN', '50'))
CMK_FMAX = float(os.environ.get('KT_CMK_FMAX', '0.35'))
CMK_DUTY = int(os.environ.get('KT_CMK_DUTY', '175'))
CMK_FREE = int(os.environ.get('KT_CMK_FREE', '250'))
CMK_MREL = float(os.environ.get('KT_CMK_MREL', '1.8'))
CHL_RT = float(os.environ.get('KT_CHL_RT', '6.0'))
CHL_VMIN = float(os.environ.get('KT_CHL_VMIN', '0.3'))
CHL_MODES = tuple((int(x) for x in os.environ.get('KT_CHL_MODES', '1,2').split(',') if x))
CHL_EMAX = float(os.environ.get('KT_CHL_EMAX', '1.2'))
OMB_R_SAFE = float(os.environ.get('KT_OMB_RSAFE', '1.0'))
VP_ON = os.environ.get('KT_VP', '1') == '1'
VP_T = int(os.environ.get('KT_VP_T', '250'))
VP_D = float(os.environ.get('KT_VP_D', '1.5'))
VP_ZLO = 0.3
VP_ZHI = 3.0
VP_TYPES = ('village', 'warehouse')
VP_NP = 500
VP_NZ = float(os.environ.get('KT_VP_NZ', '0.25'))
VP_SIG = tuple((float(x) for x in os.environ.get('KT_VP_SIG', '0.4855,0.491,0.551').split(',') if x))
VP_SIG_TOL = 0.0005
VWH_ON = os.environ.get('KT_VWH', '0') == '1'
VTO_ON = os.environ.get('KT_VTO', '0') == '1'
VPX_ON = os.environ.get('KT_VPX', '0') == '1'
VPC_ON = os.environ.get('KT_VPC', '0') == '1'
VPC_N = int(os.environ.get('KT_VPC_N', '100'))
VPC_D = float(os.environ.get('KT_VPC_D', '0.6'))
VPC_V = float(os.environ.get('KT_VPC_V', '1.5'))
VPC_DZ = float(os.environ.get('KT_VPC_DZ', '1.0'))
VPC_ZMAX = float(os.environ.get('KT_VPC_ZMAX', '8.0'))
VPC_T = int(os.environ.get('KT_VPC_T', '150'))
VTO_ALL = os.environ.get('KT_VTO_ALL', '0') == '1'
VTO_BLIND = os.environ.get('KT_VTO_BLIND', '1') == '1'
VTO_RS = tuple((float(x) for x in os.environ.get('KT_VTO_RS', '0.45,0.6').split(',') if x))
VTO_A = float(os.environ.get('KT_VTO_A', '45'))
VTO_FB = float(os.environ.get('KT_VTO_FB', '0.05'))
VTO_NEAR = float(os.environ.get('KT_VTO_NEAR', '1.2'))
VTO_CEXT = float(os.environ.get('KT_VTO_CEXT', '0.4'))
VTO_CORR = float(os.environ.get('KT_VTO_CORR', '0.35'))
VTO_ZLO = float(os.environ.get('KT_VTO_ZLO', '0.3'))
VTO_ZHI = float(os.environ.get('KT_VTO_ZHI', '5.0'))
VTO_V = float(os.environ.get('KT_VTO_V', '1.2'))
VTO_K = float(os.environ.get('KT_VTO_K', '2.5'))
VTO_VZ = float(os.environ.get('KT_VTO_VZ', '1.6'))
VTO_EXIT = float(os.environ.get('KT_VTO_EXIT', '0.97'))
VTO_ARR = float(os.environ.get('KT_VTO_ARR', '0.08'))
VTO_TMAX = int(os.environ.get('KT_VTO_TMAX', '60'))
VPF_ON = os.environ.get('KT_VPF', '1') == '1'
VPF_T1 = int(os.environ.get('KT_VPF_T1', '60'))
VPF_W = int(os.environ.get('KT_VPF_W', '150'))
VPF_DZ = float(os.environ.get('KT_VPF_DZ', '3.5'))
VPF_R = float(os.environ.get('KT_VPF_R', '2.0'))
VPF_VT = float(os.environ.get('KT_VPF_VT', '0.3'))
VPF_DC = float(os.environ.get('KT_VPF_DC', '3.0'))
VPF_TOPM = float(os.environ.get('KT_VPF_TOPM', '0.8'))
VPF_Z0MAX = float(os.environ.get('KT_VPF_Z0MAX', '7.0'))
VPF_V = float(os.environ.get('KT_VPF_V', '0.7'))
VPF_BACK = float(os.environ.get('KT_VPF_BACK', '0.8'))
VPF_RC = float(os.environ.get('KT_VPF_RC', '0.35'))
VPF_EH = float(os.environ.get('KT_VPF_EH', '1.5'))
VPF_EZ = float(os.environ.get('KT_VPF_EZ', '0.8'))
VPF_TILT = float(os.environ.get('KT_VPF_TILT', '0.5'))
VPF_ZTOP = float(os.environ.get('KT_VPF_ZTOP', '7.0'))
VPF_DZTOP = float(os.environ.get('KT_VPF_DZTOP', '3.0'))
VPF_TMAX = int(os.environ.get('KT_VPF_TMAX', '350'))
VPF_SHADOW = os.environ.get('KT_VPF_SHADOW', '0') == '1'

def _vto_pick(depth, pos, rpy, z0):
    fwd, up, right = _axes(rpy)
    cam = np.asarray(pos, np.float64) + fwd * _CAMF + up * _CAMU
    rows = np.arange(0, 256, 4)
    cols = np.arange(0, 256, 4)
    dm = np.asarray(depth, np.float32).reshape(256, 256)[np.ix_(rows, cols)].astype(np.float64) * (_DMAX - _DMIN) + _DMIN
    u = (cols + 0.5 - 128.0) / 128.0 * _HT
    v = (128.0 - rows - 0.5) / 128.0 * _HT
    ray = fwd[None, None, :] + u[None, :, None] * right[None, None, :] + v[:, None, None] * up[None, None, :]
    P = (cam[None, None, :] + dm[..., None] * ray)[dm < _DMAX - 0.2]
    q = P[(P[:, 2] >= z0 + VTO_ZLO) & (P[:, 2] <= z0 + VTO_ZHI)][:, 0:2] - np.asarray(pos[0:2], np.float64)[None, :]
    q = q[np.hypot(q[:, 0], q[:, 1]) < 3.0]
    if not VTO_BLIND and (not (len(q) and float(np.min(np.hypot(q[:, 0], q[:, 1]))) < VTO_NEAR)):
        return np.zeros(2)
    y0 = float(rpy[2])
    best = (np.zeros(2), -1.0)
    for a in np.arange(-VTO_A, VTO_A + 0.01, 15.0):
        ang = y0 + math.radians(float(a))
        uu = np.array([math.cos(ang), math.sin(ang)])
        if len(q):
            al = q @ uu
            lat = np.abs(q[:, 0] * uu[1] - q[:, 1] * uu[0])
        for r in VTO_RS:
            T = uu * r
            if len(q) and ((al > 0.0) & (al < r + VTO_CEXT) & (lat < VTO_CORR)).any():
                continue
            c = float(np.min(np.hypot(q[:, 0] - T[0], q[:, 1] - T[1]))) if len(q) else 9.0
            s = min(c, 1.5) + VTO_FB * r * math.cos(math.radians(float(a)))
            if s > best[1]:
                best = (T, s)
    return best[0]
OMB_R_SAFE_F = float(os.environ.get('KT_OMB_RSAFE_FOREST', '0.75'))
OMB_K = float(os.environ.get('KT_OMB_K', '1.5'))
OMB_SEE_M = float(os.environ.get('KT_OMB_SEE', '6.0'))
OMB_MEM_S = float(os.environ.get('KT_OMB_MEM', '10.0'))
OMB_INFL_M = float(os.environ.get('KT_OMB_INFL', '3.5'))
OMB_CAP_NEAR = float(os.environ.get('KT_OMB_CAP_NEAR', '2.0'))
OMB_CAP_MIN = float(os.environ.get('KT_OMB_CAP_MIN', '0.3'))
OMB_KEEP_M = float(os.environ.get('KT_OMB_KEEP', '15.0'))
OMB_BLIND_V = float(os.environ.get('KT_OMB_BLIND_V', '1.0'))
OMB_BLIND_DEG = float(os.environ.get('KT_OMB_BLIND_DEG', '120.0'))
OMB_BLIND_TYPES = tuple((x for x in os.environ.get('KT_OMB_BLIND_TYPES', 'forest').split(',') if x))
OMB_SLIDE_V = float(os.environ.get('KT_OMB_SLIDE_V', '1.2'))
OMB_SLIDE_HOLD = 50
OMB_STUCK_TICKS = int(os.environ.get('KT_OMB_STUCK', '150'))
OMB_STUCK_FREE = 100
OMB_GROUND = float(os.environ.get('KT_OMB_GROUND', '-1'))
OMB_VOX = 0.4
OMB_BLOCK = 8
OMB_SPEED = 3.0
OMB_HZ = 50.0
_OMB_G = None

def _omb_grid():
    global _OMB_G
    if _OMB_G is None:
        px = (np.arange(256, dtype=np.float64) + 0.5) / 256 * 2.0 - 1.0
        u = np.ascontiguousarray(np.broadcast_to(px[None, :], (256, 256)))
        v = np.ascontiguousarray(np.broadcast_to(-px[:, None], (256, 256)))
        _OMB_G = (256 // OMB_BLOCK, u, v)
    return _OMB_G

def _omb_vel(a):
    d = np.asarray(a[0:3], np.float64)
    n = float(np.linalg.norm(d))
    return d / n * abs(float(a[3])) * OMB_SPEED if n > 1e-09 else np.zeros(3, np.float64)

def _omb_store(a, v):
    s = float(np.linalg.norm(v))
    if s > 1e-09:
        a[0] = v[0] / s
        a[1] = v[1] / s
        a[2] = v[2] / s
        a[3] = min(s / OMB_SPEED, 1.0)
    else:
        a[0] = 0.0
        a[1] = 0.0
        a[2] = 0.0
        a[3] = 0.0

class _OMB:

    def __init__(self):
        self.reset()

    def reset(self):
        self.keys = np.zeros(0, np.int64)
        self.pts = np.zeros((0, 3), np.float64)
        self.seen_t = np.zeros(0, np.int64)
        self.tick = 0
        self.stats = {'ticks': 0, 'limited': 0, 'capped': 0, 'slide': 0, 'stuck': 0, 'blind': 0, 'min_r': 99.0}
        self.slide_side = 0.0
        self.slide_until = -1
        self.prog_pos = None
        self.prog_tick = 0
        self.free_until = -1
        self.rs_ovr = None

    def observe(self, depth, pos, rpy, ground_z=None):
        self.tick += 1
        nb, U, V = _omb_grid()
        dm = np.asarray(depth, np.float32).reshape(256, 256) * (_DMAX - _DMIN) + _DMIN
        blk = dm.reshape(nb, OMB_BLOCK, nb, OMB_BLOCK).transpose(0, 2, 1, 3).reshape(nb, nb, OMB_BLOCK * OMB_BLOCK)
        arg = blk.argmin(axis=2)
        mn = np.take_along_axis(blk, arg[:, :, None], axis=2)[:, :, 0]
        sel = mn <= OMB_SEE_M
        if sel.any():
            by, bx = np.nonzero(sel)
            a = arg[by, bx]
            py = by * OMB_BLOCK + a // OMB_BLOCK
            px = bx * OMB_BLOCK + a % OMB_BLOCK
            cz = mn[by, bx].astype(np.float64)
            u = U[py, px]
            v = V[py, px]
            fwd, up, right = _axes(rpy)
            cam = np.asarray(pos, np.float64) + fwd * _CAMF + up * _CAMU
            pts = cam[None, :] + right[None, :] * (u * _HT * cz)[:, None] + up[None, :] * (v * _HT * cz)[:, None] + fwd[None, :] * cz[:, None]
            if OMB_GROUND >= 0.0 and ground_z is not None:
                pts = pts[pts[:, 2] > ground_z + OMB_GROUND]
            if len(pts):
                ijk = np.floor(pts / OMB_VOX).astype(np.int64) + 500000
                keys = (ijk[:, 0] * 1000003 + ijk[:, 1]) * 1000003 + ijk[:, 2]
                allk = np.concatenate([keys, self.keys])
                allp = np.concatenate([pts, self.pts])
                allt = np.concatenate([np.full(keys.size, self.tick, np.int64), self.seen_t])
                _, first = np.unique(allk, return_index=True)
                self.keys, self.pts, self.seen_t = (allk[first], allp[first], allt[first])
        if self.tick % 25 == 0 and self.keys.size:
            keep = self.tick - self.seen_t <= int(OMB_MEM_S * OMB_HZ)
            keep &= np.hypot(self.pts[:, 0] - float(pos[0]), self.pts[:, 1] - float(pos[1])) <= OMB_KEEP_M
            if not keep.all():
                self.keys, self.pts, self.seen_t = (self.keys[keep], self.pts[keep], self.seen_t[keep])

    def constrain(self, a, pos, rpy, stype, capon=True):
        changed = False
        if OMB_BLIND_V > 0.0 and stype in OMB_BLIND_TYPES:
            v0 = _omb_vel(a)
            sh = math.hypot(v0[0], v0[1])
            if sh > OMB_BLIND_V:
                head = math.atan2(v0[1], v0[0])
                mis = abs((head - float(rpy[2]) + math.pi) % (2.0 * math.pi) - math.pi)
                if mis > math.radians(OMB_BLIND_DEG):
                    _omb_store(a, v0 * (OMB_BLIND_V / sh))
                    self.stats['blind'] += 1
                    changed = True
        if self.keys.size == 0:
            return changed
        P = self.pts
        x = np.asarray(pos, np.float64)
        D = x[None, :] - P
        R = np.sqrt((D * D).sum(axis=1))
        near = R <= OMB_INFL_M
        if not near.any():
            return changed
        D = D[near]
        R = R[near]
        o = np.argsort(R)
        D = D[o]
        R = R[o]
        v_cmd = _omb_vel(a)
        v = v_cmd.copy()
        r_safe = OMB_R_SAFE_F if stype == 'forest' else OMB_R_SAFE
        sp_cmd = float(math.hypot(v_cmd[0], v_cmd[1]))
        x2 = (float(x[0]), float(x[1]))
        if self.prog_pos is None or math.hypot(x2[0] - self.prog_pos[0], x2[1] - self.prog_pos[1]) > 1.0 or sp_cmd < 0.5:
            self.prog_pos = x2
            self.prog_tick = self.tick
        elif self.tick - self.prog_tick > OMB_STUCK_TICKS and self.tick > self.free_until:
            self.free_until = self.tick + OMB_STUCK_FREE
            self.prog_tick = self.tick
            self.stats['stuck'] += 1
        if self.tick <= self.free_until:
            r_safe = 0.35
        if self.rs_ovr is not None:
            r_safe = min(r_safe, self.rs_ovr)

        def _project(v):
            for _ in range(2):
                for i in range(min(len(R), 24)):
                    r = R[i]
                    n = D[i] / max(r, 1e-06)
                    lim = OMB_K * (r - r_safe)
                    vt = -float(v @ n)
                    if vt > lim:
                        v = v + (vt - lim) * n
            return v
        v = _project(v)
        if not np.allclose(v, v_cmd, atol=1e-06):
            changed = True
        sp_h = float(math.hypot(v[0], v[1]))
        if sp_cmd > 0.3 and sp_h < 0.35 * sp_cmd and (self.tick > self.free_until) and (self.rs_ovr is None):
            tw = -D[0][:2]
            tn = float(math.hypot(tw[0], tw[1]))
            if tn > 1e-06:
                tw = tw / tn
                tL = np.array([-tw[1], tw[0]])
                tR = -tL
                if self.tick <= self.slide_until and self.slide_side != 0.0:
                    side = self.slide_side
                else:
                    dot = float(v_cmd[0] * tL[0] + v_cmd[1] * tL[1]) / max(sp_cmd, 1e-06)
                    if abs(dot) > 0.2:
                        side = 1.0 if dot > 0 else -1.0
                    else:
                        pl = x[:2] + tL * 2.0
                        pr = x[:2] + tR * 2.0
                        dl = float(np.min(np.hypot(P[:, 0] - pl[0], P[:, 1] - pl[1])))
                        dr = float(np.min(np.hypot(P[:, 0] - pr[0], P[:, 1] - pr[1])))
                        side = 1.0 if dl >= dr else -1.0
                    self.slide_side = side
                    self.slide_until = self.tick + OMB_SLIDE_HOLD
                t = tL if side > 0 else tR
                vs = min(sp_cmd, OMB_SLIDE_V)
                v = _project(np.array([t[0] * vs, t[1] * vs, v_cmd[2]], np.float64))
                self.stats['slide'] += 1
                changed = True
        rmin = float(R[0])
        if rmin < self.stats['min_r']:
            self.stats['min_r'] = rmin
        sn = float(np.linalg.norm(v))
        cap = OMB_SPEED
        if capon and rmin < OMB_CAP_NEAR:
            cap = OMB_SPEED * max(OMB_CAP_MIN, min(1.0, (rmin - r_safe * 0.5) / (OMB_CAP_NEAR - r_safe * 0.5)))
        if sn > cap:
            v = v * (cap / sn)
            changed = True
            self.stats['capped'] += 1
        if changed:
            self.stats['limited'] += 1
            _omb_store(a, v)
        return changed
CHAMP_ROUTE = os.environ.get('KT_CHAMP_ROUTE', '0' if os.environ.get('KT_PLAN', '1') == '1' else '1') == '1'
ESC_TERR = os.environ.get('KT_ESC_TERR', '1') == '1'
ESC_ON = os.environ.get('KT_ESC', '1') == '1'
ESC_TRAP_M = float(os.environ.get('KT_ESC_TRAP', '1.0'))
ESC_FREE_M = 3.0
ESC_DIST = float(os.environ.get('KT_ESC_DIST', '2.0'))
WESC_ON = os.environ.get('KT_WESC', '1') == '1'
WESC_FRAC = float(os.environ.get('KT_WESC_FRAC', '0.5'))
WESC_H = float(os.environ.get('KT_WESC_H', '0.45'))
WESC_HOR = float(os.environ.get('KT_WESC_HOR', '1.0'))
WESC_Z = float(os.environ.get('KT_WESC_Z', '0.3'))
WESC_MODE = os.environ.get('KT_WESC_MODE', 'side')
WESC_BACK = float(os.environ.get('KT_WESC_BACK', '1.2'))
WESC_TOP = float(os.environ.get('KT_WESC_TOP', '1.2'))
WESC_SIDE_M = float(os.environ.get('KT_WESC_SIDE_M', '0.6'))
WESC_SIDE_SPEED = float(os.environ.get('KT_WESC_SIDE_SPEED', '0.2'))
WESC_SIDE_MAX = float(os.environ.get('KT_WESC_SIDE_MAX', '1.0'))
ESC_SPEED = 0.3
ESC_CLIMB = 0.6
ESC_AGL_TARGET = 2.5
ESC_MAX_CREEP = 300
ESC_MARGIN = 1.0
ESC_BODY_M = 1.2
ESC_MAX_CLIMB = 150
ESC_CLEAR_TICKS = 10
ESC_CREEP_AGL = 0.28
ESC_SCAN_TICKS = 45

class _Escape:

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = None
        self.dist = 0.0
        self.phase = None
        self.heading = None
        self.start = None
        self.clear = 0
        self.n = 0
        self.trap = False
        self.cycles = 0
        self.climb_start = None
        self.yaw0 = 0.0
        self.scan_i = 0
        self.scan_n = 0
        self.margin_start = None
        self.guard_hits = 0

    def opening(self, depth, yaw):
        d = np.asarray(depth, np.float32).reshape(256, 256) * (_DMAX - _DMIN) + _DMIN
        bands = [float(d[112:128, c0:c0 + 52].min()) for c0 in (0, 51, 102, 153, 204)]
        best = max(bands)
        k = min((i for i in range(5) if bands[i] >= best - 0.05), key=lambda i: abs(i - 2))
        if bands[k] < ESC_FREE_M:
            return None
        return float(yaw) - math.radians((k - 2) * 18.0)

    def _back(self):
        return self.mode == 'ceil' and WESC_MODE in ('back', 'side')

    def _yawcmd(self):
        h = self.yaw0 if self._back() or self.heading is None else self.heading
        return float(((h + math.pi) % (2 * math.pi) - math.pi) / math.pi)

    def _scan_seq(self):
        return (180.0, 90.0, -90.0) if self.mode == 'ceil' else (90.0, -90.0, 180.0)

    def _open(self, depth, yaw):
        if self.mode != 'ceil':
            return self.opening(depth, yaw)
        d = np.asarray(depth, np.float32).reshape(256, 256) * (_DMAX - _DMIN) + _DMIN
        bands = [(float(d[112:128, c0:c0 + 52].min()), float(d[0:64, c0:c0 + 52].min())) for c0 in (0, 51, 102, 153, 204)]
        ok = [i for i in range(5) if bands[i][0] >= ESC_FREE_M and bands[i][1] >= WESC_TOP]
        if not ok:
            return None
        k = min(ok, key=lambda i: (abs(i - 2), -bands[i][0]))
        return float(yaw) - math.radians((k - 2) * 18.0)

    def _side(self, d, yaw):
        v = (128.0 - np.arange(96) - 0.5) / 128.0
        u = (np.arange(256) + 0.5 - 128.0) / 128.0
        up = d[0:96]
        low = (up * v[:, None] < 1.0) & (up < 3.0)
        el = []
        er = []
        for r in range(0, 96, 4):
            row = low[r]
            if not row[124:132].any():
                continue
            jl = 128
            while jl > 0 and row[jl - 1]:
                jl -= 1
            jr = 127
            while jr < 255 and row[jr + 1]:
                jr += 1
            el.append(-float(u[jl]) * float(d[r, jl]))
            er.append(float(u[jr]) * float(d[r, jr]))
        if el:
            l = float(np.median(el))
            rr = float(np.median(er))
            e = min(l, rr)
        else:
            l = float(low[:, 0:96].sum())
            rr = float(low[:, 160:256].sum())
            e = 0.4
        right = rr < l
        dist = float(min(WESC_SIDE_MAX, max(0.6, e + WESC_SIDE_M)))
        return (float(yaw) + (-0.5 * math.pi if right else 0.5 * math.pi), dist)

    def ceiling(self, depth, pos, yaw):
        if float(pos[2]) > WESC_Z:
            return False
        d = np.asarray(depth, np.float32).reshape(256, 256) * (_DMAX - _DMIN) + _DMIN
        v = (128.0 - np.arange(96) - 0.5) / 128.0
        up = d[0:96, 96:160]
        hup = up * v[:, None]
        near = up < 3.0
        if not near.any():
            return False
        frac = float(((hup < 1.0) & near).mean())
        hmed = float(np.median(hup[near]))
        hor = float(np.median(d[96:128, 96:160]))
        if frac < WESC_FRAC or hmed > WESC_H or hor > WESC_HOR:
            return False
        self.trap = True
        self.mode = 'ceil'
        self.start = np.asarray(pos[0:2], np.float64).copy()
        self.yaw0 = float(yaw)
        if WESC_MODE == 'side':
            self.heading, self.dist = self._side(d, yaw)
            self.phase = 'creep'
            return True
        if WESC_MODE == 'back':
            self.heading = float(yaw) + math.pi
            self.phase = 'creep'
            return True
        h = self._open(depth, yaw)
        if h is not None:
            self.heading = h
            self.phase = 'creep'
            return True
        self.phase = 'scan'
        self.scan_i = 0
        self.scan_n = 0
        return True

    def decide(self, depth, pos, yaw):
        d = np.asarray(depth, np.float32).reshape(256, 256) * (_DMAX - _DMIN) + _DMIN
        if float((d[0:16] < 0.8).mean()) < 0.95:
            return False
        self.trap = True
        self.start = np.asarray(pos[0:2], np.float64).copy()
        self.yaw0 = float(yaw)
        h = self.opening(depth, yaw)
        if h is not None:
            self.heading = h
            self.phase = 'creep'
            return True
        self.phase = 'scan'
        self.scan_i = 0
        self.scan_n = 0
        return True

    def act(self, depth, pos, agl, pitch_deg, yaw):
        self.n += 1
        d = np.asarray(depth, np.float32).reshape(256, 256) * (_DMAX - _DMIN) + _DMIN
        top = float(d[0:64].min())
        if self.phase == 'scan':
            tgt = self.yaw0 + math.radians(self._scan_seq()[self.scan_i])
            self.scan_n += 1
            err = (yaw - tgt + math.pi) % (2 * math.pi) - math.pi
            if abs(err) < math.radians(8.0) or self.scan_n >= ESC_SCAN_TICKS:
                h = self._open(depth, yaw) if abs(err) < math.radians(12.0) else None
                if h is not None:
                    self.heading = h
                    self.phase = 'creep'
                    self.n = 0
                    self.clear = 0
                    self.start = np.asarray(pos[0:2], np.float64).copy()
                    return (np.array([math.cos(h), math.sin(h), 0.0], np.float32), ESC_SPEED, self._yawcmd() if self.mode == 'ceil' else None)
                self.scan_i += 1
                self.scan_n = 0
                if self.scan_i >= 3:
                    self.phase = 'done'
                    return None
                tgt = self.yaw0 + math.radians(self._scan_seq()[self.scan_i])
            yt = (tgt + math.pi) % (2 * math.pi) - math.pi
            return (np.zeros(3, np.float32), 0.0, float(yt / math.pi))
        if self.phase == 'creep':
            self.clear = self.clear + 1 if top >= 2.0 and abs(pitch_deg) < 6.0 else 0
            trav = float(np.linalg.norm(np.asarray(pos[0:2], np.float64) - self.start))
            if self.clear >= ESC_CLEAR_TICKS and self.margin_start is None:
                self.margin_start = np.asarray(pos[0:2], np.float64).copy()
            past = self.margin_start is not None and float(np.linalg.norm(np.asarray(pos[0:2], np.float64) - self.margin_start)) >= ESC_MARGIN
            rel = (self.heading - yaw + math.pi) % (2 * math.pi) - math.pi
            kb = min(4, max(0, int(round(2 - math.degrees(rel) / 18.0))))
            c0 = (0, 51, 102, 153, 204)[kb]
            if float(d[100:128, c0:c0 + 52].min()) < ESC_BODY_M and self.n > 15 and (self.guard_hits < 3) and (not self._back()):
                self.guard_hits += 1
                self.phase = 'scan'
                self.scan_i = 0
                self.scan_n = 0
                self.yaw0 = float(yaw)
                self.clear = 0
                self.margin_start = None
                return (np.zeros(3, np.float32), 0.0, None)
            if self.mode == 'ceil' and WESC_MODE == 'side' and (trav >= self.dist or self.n >= ESC_MAX_CREEP):
                self.phase = 'done'
                return None
            if past and (not (self.mode == 'ceil' and WESC_MODE == 'side')) or self.n >= ESC_MAX_CREEP or (WESC_MODE == 'back' and self._back() and (trav >= WESC_BACK)):
                self.phase = 'climb'
                self.n = 0
                self.climb_start = np.asarray(pos[0:2], np.float64).copy()
            else:
                vz = float(np.clip((ESC_CREEP_AGL - agl) * 3.0, -0.5, 0.5))
                v = np.array([math.cos(self.heading), math.sin(self.heading), vz], np.float64)
                v /= max(1e-06, float(np.linalg.norm(v)))
                return (v.astype(np.float32), WESC_SIDE_SPEED if self.mode == 'ceil' and WESC_MODE == 'side' else ESC_SPEED, self._yawcmd() if self.mode == 'ceil' else None)
        if self.phase == 'climb':
            if agl >= ESC_AGL_TARGET or self.n >= ESC_MAX_CLIMB:
                self.phase = 'done'
                return None
            if top < 0.9 and agl < 1.0 and (self.cycles < 2):
                self.cycles += 1
                self.phase = 'creep'
                self.n = 0
                self.clear = 0
                self.margin_start = None
                self.start = np.asarray(pos[0:2], np.float64).copy()
                return (np.array([math.cos(self.heading), math.sin(self.heading), 0.0], np.float32), ESC_SPEED, self._yawcmd() if self.mode == 'ceil' else None)
            return (np.array([0.0, 0.0, 1.0], np.float32), ESC_CLIMB, self._yawcmd() if self.mode == 'ceil' else None)
        return None
SPLIT_ON = os.environ.get('KT_SPLIT', '0') == '1'
RP_SPLIT = {'mountain': 'victim_rgb_mountain.onnx'} if SPLIT_ON else {}
R2_SPLIT = {t: 'victim_rgb_%s.onnx' % t for t in ('forest', 'village', 'warehouse')} if SPLIT_ON else {}
R2_ON = os.environ.get('KT_R2', '1') == '1'
R2_TYPES = tuple((t for t in os.environ.get('KT_R2_TYPES', 'forest,warehouse,village').split(',') if t in LB))
R2_T = float(os.environ.get('KT_R2_T', '0.65'))
R2_BAR = float(os.environ.get('KT_R2_BAR', '0.5'))
R2_HMAX = float(os.environ.get('KT_R2_HMAX', 'inf'))
R2_RANGE_MAX = float(os.environ.get('KT_R2_RANGE_MAX', '25.0'))
R2_CAP = int(os.environ.get('KT_R2_CAP', str(RP_CAP - 3)))
R2_TERM_R = float(os.environ.get('KT_R2_TERM_R', '6.0'))
R2_TERM_TYPES = tuple(os.environ.get('KT_R2_TERM_TYPES', 'forest,village,warehouse').split(','))
R2_FACE = os.environ.get('KT_R2_FACE', '1') == '1'
R2_FACE_TYPES = tuple(os.environ.get('KT_R2_FACE_TYPES', 'forest,warehouse').split(','))
R2_FACE_CLR = float(os.environ.get('KT_R2_FACE_CLR', '2.0'))
R2_TERM_P1 = os.environ.get('KT_R2_TERM_P1', '1') == '1'
R2_TERM_WAIT = int(os.environ.get('KT_R2_TERM_WAIT', '100'))
MTN_Z0 = float(os.environ.get('KT_MTN_Z0', '13.0'))
CM_ON = os.environ.get('KT_CM', '1') == '1'
CM_TYPES = tuple((t for t in os.environ.get('KT_CM_TYPES', 'village').split(',') if t))
CM_TH = 0.9
CM_NEED = int(os.environ.get('KT_CM_NEED', '4'))
CM_WIN = 50
CM_TOL = 2.5
CM_RMIN = 15.0
CM_T0 = 150
CM_NEAR = 8.0
CM_NEAR_T = 100
CM_TMAX = 750
CM_BAD_R = 6.0
LM_ON = os.environ.get('KT_LM', '1') == '1'
LM_D = int(os.environ.get('KT_LM_D', '300'))
LM_MISS = float(os.environ.get('KT_LM_MISS', '28.5'))
LM_SC = float(os.environ.get('KT_LM_SC', '31.0'))
LM_LCW = int(os.environ.get('KT_LM_LCW', '30'))
LM_LC = float(os.environ.get('KT_LM_LC', '0.6'))
LM_CH = float(os.environ.get('KT_LM_CH', '18.0'))
LM_CHG = int(os.environ.get('KT_LM_CHG', '300'))
LM_CHR = float(os.environ.get('KT_LM_CHR', '3.0'))
LM_NEAR = float(os.environ.get('KT_LM_NEAR', '8.0'))
LM_NEAR_T = int(os.environ.get('KT_LM_NEAR_T', '300'))
LM_TMAX = int(os.environ.get('KT_LM_TMAX', '750'))
LM_BAD_R = float(os.environ.get('KT_LM_BAD_R', '6.0'))

class _LM:

    def __init__(self):
        self.fbox = None
        self.reset()

    def reset(self):
        self.seg = None
        self.segs = []
        self.pend = None
        self.tgt = None
        self.sus = False
        self.run = 0
        self.near = None
        self.bad = []
        self.n_eng = 0
        self.n_rel = 0
        self.err = 0
        self.dead = False

    def _banned(self, e, p3):
        return any((float(np.hypot(*e - b)) < LM_BAD_R for b in self.bad)) or any((float(np.hypot(*e - np.asarray(b[0:2], np.float64))) < P3_BAN_R for b in p3))

    def _rel(self, ban):
        if ban:
            self.bad.append(self.tgt.copy())
        self.tgt = None
        self.sus = False
        self.run = 0
        self.near = None
        self.n_rel += 1

    def step(self, t, m, pos, forest, p3, c0):
        m = np.nan_to_num(np.asarray(m, np.float64).reshape(-1))
        if self.dead or m.size < 63:
            return None
        lk = bool(m[57] > 0.5)
        ph = int(np.rint(m[0]))
        p2 = np.asarray(pos[0:2], np.float64)
        dr = None
        if lk:
            if self.seg is None:
                self.seg = {'a': t, 'ds': float(np.hypot(*m[60:62] - p2)), 'c': [], 'ph': ph}
            s = self.seg
            s['c'] = (s['c'] + [bool(m[55] < 0.5)])[-LM_LCW:]
            s['ph'] = max(s['ph'], ph)
            s['e'] = m[60:62].copy()
            s['miss'] = float(m[59])
        elif self.seg is not None:
            s = self.seg
            self.seg = None
            e = s['e']
            ch = s['ds']
            j = len(self.segs)
            na = s['a']
            while j > 0 and na - self.segs[j - 1][1] < LM_CHG and (float(np.hypot(*self.segs[j - 1][2] - e)) < LM_CHR):
                j -= 1
                ch = self.segs[j][3]
                na = self.segs[j][0]
            self.segs.append((s['a'], t, e, s['ds']))
            dr = (e, bool(s['miss'] >= LM_MISS and s['ph'] < 3 and (ph < 3)), float(np.mean(s['c'])), ch)
        if ph >= 3:
            self.pend = None
            if self.tgt is not None:
                self.tgt = None
                self.sus = False
                self.run = 0
                self.near = None
            return None
        if self.tgt is not None:
            if lk:
                self.sus = True
                self.near = None
                return None
            if dr is not None and self.sus:
                if float(np.hypot(*dr[0] - self.tgt)) < LM_CHR:
                    if not dr[1] or self._banned(self.tgt, p3):
                        self._rel(True)
                        return None
                    self.sus = False
                    dr = None
                else:
                    self._rel(False)
            if self.tgt is not None:
                if not forest:
                    self._rel(False)
                    return None
                d = float(np.hypot(*self.tgt - p2))
                if d < LM_NEAR and self.near is None:
                    self.near = t
                if self.near is not None and t - self.near > LM_NEAR_T or self.run >= LM_TMAX:
                    self._rel(True)
                    return None
                if m[0] >= 1.5:
                    return None
                self.run += 1
                return self.tgt
        if lk:
            self.pend = None
            return None
        if dr is not None and forest and (self.fbox is None or max(abs(float(dr[0][0])), abs(float(dr[0][1]))) <= self.fbox) and dr[1] and (float(np.hypot(*dr[0] - np.asarray(c0, np.float64))) <= LM_SC) and (dr[2] >= LM_LC) and (dr[3] >= LM_CH) and (not self._banned(dr[0], p3)):
            self.pend = (dr[0].copy(), t)
        if self.pend is not None and t - self.pend[1] >= LM_D and (0.5 <= m[0] < 1.5) and forest:
            e = self.pend[0]
            self.pend = None
            if self._banned(e, p3):
                return None
            self.tgt = e
            self.sus = False
            self.run = 1
            self.near = None
            self.n_eng += 1
            if float(np.hypot(*e - p2)) < LM_NEAR:
                self.near = t
            return self.tgt
        return None
LA_ON = os.environ.get('KT_LA', '1') == '1'
LA_NONE_CITY = float(os.environ.get('KT_LA_NONE_CITY', '0.35'))
LA_W = int(os.environ.get('KT_LA_W', '50'))
LA_K = int(os.environ.get('KT_LA_K', '5'))
LA_R = float(os.environ.get('KT_LA_R', '1.5'))
LA_T0 = int(os.environ.get('KT_LA_T0', '150'))
LA_BAN = float(os.environ.get('KT_LA_BAN', '3.0'))
LA_TYPES = tuple((t for t in os.environ.get('KT_LA_TYPES', 'city').split(',') if t))
PG_ON = os.environ.get('KT_PG', '1') == '1'
PG_TYPES = tuple((t for t in os.environ.get('KT_PG_TYPES', 'city').split(',') if t))
PG_D = float(os.environ.get('KT_PG_D', '1.0'))
PG_DMIN = float(os.environ.get('KT_PG_DMIN', '0.5'))
PG_VMIN = float(os.environ.get('KT_PG_VMIN', '0.12'))
PG_HOLD = int(os.environ.get('KT_PG_HOLD', '25'))
PG_AGL = float(os.environ.get('KT_PG_AGL', '2.2'))
PG_CLIMB = float(os.environ.get('KT_PG_CLIMB', '1.2'))
PG_STOP = float(os.environ.get('KT_PG_STOP', '0'))
PG_REP = float(os.environ.get('KT_PG_REP', '0'))
PG_REPV = float(os.environ.get('KT_PG_REPV', '0.8'))
PG_MEM = int(os.environ.get('KT_PG_MEM', '75'))
NG_ON = os.environ.get('KT_NG', '1') == '1'
NG_MEMT = int(os.environ.get('KT_NG_MEMT', '75'))
NG_EVERY = int(os.environ.get('KT_NG_EVERY', '2'))
NG_NEAR = float(os.environ.get('KT_NG_NEAR', '20.0'))
NG_KEEP = float(os.environ.get('KT_NG_KEEP', '12.0'))
NG_Z0 = float(os.environ.get('KT_NG_Z0', '0.6'))
NG_ZROOF = float(os.environ.get('KT_NG_ZROOF', '1.2'))
NG_RR = float(os.environ.get('KT_NG_RR', '0.8'))
NG_RT = float(os.environ.get('KT_NG_RT', '3.0'))
NG_DZR = float(os.environ.get('KT_NG_DZR', '0.8'))
NG_DZT = float(os.environ.get('KT_NG_DZT', '0.3'))
NG_N = int(os.environ.get('KT_NG_N', '3'))
NG_DS = float(os.environ.get('KT_NG_DS', '0.4'))
NG_TAU = float(os.environ.get('KT_NG_TAU', '1.2'))
NG_LC = float(os.environ.get('KT_NG_LC', '1.5'))
NG_LMIN = float(os.environ.get('KT_NG_LMIN', '1.5'))
NG_SZ = float(os.environ.get('KT_NG_SZ', '0.6'))
NG_VMIN = float(os.environ.get('KT_NG_VMIN', '0.3'))
NG_HOLD = int(os.environ.get('KT_NG_HOLD', '20'))
NG_CLIMB = float(os.environ.get('KT_NG_CLIMB', '1.2'))
NG_ZMAX = float(os.environ.get('KT_NG_ZMAX', '14.0'))
NG_MAXT = int(os.environ.get('KT_NG_MAXT', '250'))
NG_COOL = int(os.environ.get('KT_NG_COOL', '100'))
NG_DV = float(os.environ.get('KT_NG_DV', '1.0'))
NG_VCL = float(os.environ.get('KT_NG_VCL', '0.8'))
NG_TILT = float(os.environ.get('KT_NG_TILT', '15.0'))
NG_EMAX = float(os.environ.get('KT_NG_EMAX', '1.2'))
NG_REL = int(os.environ.get('KT_NG_REL', '30'))
_NG_B = (np.arange(32) * 8 + 4.0 - 128.0) / 128.0

def _ng_cloud(depth, pos, rpy):
    z = np.asarray(depth, np.float32).reshape(256, 256) * (_DMAX - _DMIN) + _DMIN
    zp = z.reshape(32, 8, 32, 8).min(axis=(1, 3))
    iy, ix = np.nonzero(zp < NG_NEAR)
    if iy.size == 0:
        return np.zeros((0, 3))
    zz = zp[iy, ix].astype(np.float64)
    u = _NG_B[ix]
    v = -_NG_B[iy]
    fwd, up, right = _axes(rpy)
    cam = np.asarray(pos, np.float64) + fwd * _CAMF + up * _CAMU
    return cam[None, :] + zz[:, None] * fwd[None, :] + (u * zz)[:, None] * right[None, :] + (v * zz)[:, None] * up[None, :]
F86_ESC = False
F86_P3G = False
P3G_Z = 4.72
F86_CRN = False
CRN_Z = 2.3
CRN_M = 0.3
CRN_V = 0.15
CRN_SLOTS = ((-21.63, -20.92, -8.5, -7.5), (-21.63, -20.92, 7.55, 8.55), (20.92, 21.63, -8.5, -7.5), (20.92, 21.63, 7.55, 8.55), (-0.36, 0.36, -8.5, -7.5), (-0.36, 0.36, 7.55, 8.55), (-21.63, -20.92, -4.25, -3.85), (20.92, 21.63, -4.25, -3.85), (-21.48, -21.07, -14.41, -13.69), (-21.48, -21.07, 13.69, 14.41), (21.07, 21.48, -14.41, -13.69), (21.07, 21.48, 13.69, 14.41))

def _crn_exit(x, y):
    for x0, x1, y0, y1 in CRN_SLOTS:
        a0, a1, b0, b1 = (x0 - CRN_M, x1 + CRN_M, y0 - CRN_M, y1 + CRN_M)
        if a0 <= x <= a1 and b0 <= y <= b1:
            c = ((x - a0, (-1.0, 0.0)), (a1 - x, (1.0, 0.0)), (y - b0, (0.0, -1.0)), (b1 - y, (0.0, 1.0)))
            return min(c, key=lambda q: q[0])[1]
    return None
WH_Z0 = 0.191
WH_BOX = os.environ.get('KT_WH_BOX', '1') == '1'
WH_GATE = 15.5
WH_P3_T = 400
WH_HOLD = os.environ.get('KT_WH_HOLD', '1') == '1'
WH_HOLD_DZ = 5.6
P3F_ON = False
P3F_Z = 4.9
P3F_R = 1.5
WH_R2T = os.environ.get('KT_WH_R2T', '1') == '1'
WH_R2_TZ = 5.3
WH_SWEEP = os.environ.get('KT_WH_SWEEP', '1') == '1'
WH_NOPLAN = os.environ.get('KT_WH_NOPLAN', '1') == '1'
WH_SWFIX = os.environ.get('KT_WH_SWFIX', '1') == '1'
WH19 = os.environ.get('KT_WH19', '0') == '1'
if WH19:
    WH_SWEEP = False
    WH_NOPLAN = False
SW_ARR = 8.0
SW_LOOK = 9.0
SW_MIN = 8.5
SW_LANES = (12.0, 8.0)
SW_RUN = (-3.0, 0.0, 7.0)
SW_END = 6.0
SW_RC = 16.0
SW_RN = 6.5
SW_FOV = math.radians(40.0)
SW_TC = 1.5
SW_V = 2.8
SW_ZV = 1.0
SW_WMIN = 0.35
SW_STALL = 200

class _Sweep:

    def __init__(self, c0):
        g = np.arange(-12.0, 12.01, 1.0)
        X, Y = np.meshgrid(g, g)
        P = np.stack([X.ravel(), Y.ravel()], 1)
        dc = np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1])
        self.P = P[dc <= 30.0] if int((dc <= 30.0).sum()) >= 4 else P[np.argsort(dc)[:4]]
        self.E = np.where((self.P % 2.0 == 0.0).all(1))[0]
        if len(self.E) < 4:
            self.E = np.arange(len(self.P))
        self.w = np.ones(len(self.P))
        self.S = None
        self.cs = None
        self.V = None
        self.n_end = 0
        self.j = 0
        self.wp = None
        self.t_plan = -10 ** 9
        self.armed = False
        self.n_pass = 0
        self.n_obs = 0
        self.stall = 0
        self.n_stall = 0
        self.e = None
        self.t_arm = None
        self.st = 0

    def observe(self, pos, rpy, depth):
        fwd, up, right = _axes(rpy)
        cam = np.asarray(pos, np.float64) + fwd * _CAMF + up * _CAMU
        d = np.column_stack([self.P - cam[0:2], np.full(len(self.P), SW_ZV - cam[2])])
        zc = d @ fwd
        i = np.where((zc > 1.0) & (self.w > 0.001))[0]
        if not len(i):
            return
        u = d[i] @ right / zc[i]
        v = d[i] @ up / zc[i]
        rh = np.hypot(d[i, 0], d[i, 1])
        k = (np.abs(u) <= 0.9) & (np.abs(v) <= 0.9) & (rh <= SW_RC + 4.0)
        if not k.any():
            return
        i, u, v, rh = (i[k], u[k], v[k], rh[k])
        px = np.clip(np.rint((u + 1.0) * 128.0 - 0.5).astype(int), 0, 255)
        py = np.clip(np.rint((1.0 - v) * 128.0 - 0.5).astype(int), 0, 255)
        vis = np.asarray(depth, np.float32).reshape(256, 256)[py, px] * (_DMAX - _DMIN) + _DMIN >= zc[i] - 1.0
        self.w[i[vis]] *= np.exp(-SIM_DT / np.where(rh[vis] <= SW_RC, SW_TC, 2.0 * SW_TC))
        self.n_obs += 1

    def _cands(self, Q):
        out = []
        for a in (0, 1):
            l = Q[:, 1 - a]
            lo, hi = (float(l.min()), float(l.max()))
            for sp in SW_LANES:
                n = max(1, int(round((hi - lo) / sp)))
                wd = (hi - lo) / n
                ln = []
                for k in range(n):
                    c = lo + (k + 0.5) * wd
                    al = Q[np.abs(l - c) <= 0.5 * wd + 0.5, a]
                    if len(al):
                        ln.append((c, float(al.min()), float(al.max())))
                for B in SW_RUN:
                    for o in (ln, ln[::-1]):
                        for d0 in (0, 1):
                            w = []
                            d = d0
                            for c, a0, a1 in o:
                                x0, x1 = (a0 - B, max(a0 - B, a1 - SW_END)) if d == 0 else (a1 + B, min(a1 + B, a0 + SW_END))
                                for x in (x0, x1):
                                    w.append((x, c) if a == 0 else (c, x))
                                d ^= 1
                            out.append(np.clip(np.asarray(w, np.float64), -18.0, 18.0))
        return out

    def _eval(self, p, yaw, W):
        V = np.vstack([p, W])
        S = [V[0:1]]
        for k in range(len(V) - 1):
            sg = V[k + 1] - V[k]
            n = max(1, int(math.ceil(float(np.hypot(sg[0], sg[1])))))
            S.append(V[k] + sg * (np.arange(1, n + 1) / n)[:, None])
        S = np.vstack(S)
        ds = np.r_[0.0, np.hypot(*np.diff(S, axis=0).T)]
        cs = np.cumsum(ds)
        jj = np.minimum(np.searchsorted(cs, cs + SW_LOOK), len(S) - 1)
        h = S[jj] - S
        hd = np.arctan2(h[:, 1], h[:, 0])
        for k in range(1, len(S)):
            if jj[k] == len(S) - 1 and float(np.hypot(h[k, 0], h[k, 1])) < 1.0:
                hd[k] = hd[k - 1]
        t = max(0.0, abs(_wrap(float(hd[0]) - yaw)) - 0.41) / 1.1 + cs / SW_V
        E = self.E
        rel = self.P[E][None, :, :] - S[:, None, :]
        r = np.hypot(rel[..., 0], rel[..., 1])
        b = np.abs((np.arctan2(rel[..., 1], rel[..., 0]) - hd[:, None] + math.pi) % (2 * math.pi) - math.pi)
        acc = np.cumsum(((r >= SW_RN) & (r <= SW_RC) & (b <= SW_FOV)) * (ds / SW_V)[:, None], axis=0)
        cov = acc >= SW_TC
        tc = np.where(cov.any(0), t[np.argmax(cov, axis=0)], t[-1] + 20.0)
        return (float((self.w[E] * tc).sum() / max(float(self.w[E].sum()), 1e-09)), S, cs)

    def plan(self, pos, yaw):
        m = self.w >= SW_WMIN
        if int(m.sum()) < 12:
            self.w = np.maximum(self.w, 0.5)
            m = np.ones(len(self.P), bool)
        best = None
        for W in self._cands(self.P[m]):
            r = self._eval(np.asarray(pos[0:2], np.float64), float(yaw), W)
            if best is None or r[0] < best[0]:
                best = r
                self.V = W
        self.e, S, cs = best
        self.j = 0
        self.n_pass += 1
        self.n_end = len(S) - 1
        u = S[-1] - S[max(0, len(S) - 3)]
        u = u / max(float(np.hypot(u[0], u[1])), 1e-06)
        T = np.clip(S[-1] + u * np.arange(1.0, SW_LOOK + 1.0)[:, None], -18.0, 18.0)
        self.S = np.vstack([S, T])
        self.cs = np.r_[cs, cs[-1] + np.cumsum(np.hypot(*np.diff(np.vstack([S[-1:], T]), axis=0).T))]

    def carrot(self, pos):
        S = self.S
        p = np.asarray(pos[0:2], np.float64)
        dd = np.hypot(*(S[self.j:self.j + 25] - p).T)
        k = int(np.argmin(dd))
        if dd[k] > SW_LOOK + 3.0:
            return None
        if WH_SWFIX or dd[k] < 4.0:
            self.j += k
        if self.j >= self.n_end:
            return None
        q = min(len(S) - 1, int(np.searchsorted(self.cs, self.cs[self.j] + SW_LOOK)))
        if WH_SWFIX:
            while q < len(S) - 1 and float(np.hypot(*S[q] - p)) < SW_MIN:
                q += 1
        self.wp = S[q].copy()
        return self.wp

class _R2Off(Exception):
    pass

def _ss_frac(depth):
    d = np.asarray(depth, np.float32).reshape(256, 256) * (_DMAX - _DMIN) + _DMIN
    v = (1.0 - (np.arange(85) + 0.5) / 128.0)[:, None]
    top = d[0:85, 64:192]
    h = top * v
    frac = float(((h > 0.15) & (h < 1.0) & (top < 1.2)).mean())
    hor = float(np.median(d[112:136, 96:160]))
    return (frac, hor)

def _ss_top(depth):
    d = np.asarray(depth, np.float32).reshape(256, 256)[0:40, 64:192] * (_DMAX - _DMIN) + _DMIN
    return float(np.median(d))

def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
DT_ON = os.environ.get('KT_DEPTH', '0') == '1'
DT_EVERY = int(os.environ.get('KT_DT_EVERY', '4'))
DT_THR = float(os.environ.get('KT_DT_THR', '0.7'))
DT_T0 = int(os.environ.get('KT_DT_T0', '60'))
DT_RANGE = float(os.environ.get('KT_DT_RANGE', '28.0'))
DT_HMAX = float(os.environ.get('KT_DT_HMAX', '2.5'))
_G_HR, _G_G, _G_CELL, _G_WIN, HR, G, CELL, WIN = (128, 128, 0.8, 7, 128, 128, 0.8, 7)

def _axes_np(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw))
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    fwd = R[:, 0]
    y = R[:, 1]
    up = R[:, 2]
    right = -y
    return (fwd, up, right)
_uv = None

def geo_half_np(d, roll, pitch, yaw):
    u1 = (np.arange(HR, dtype=np.float32) * 2.0 - 127.5) / 127.5
    v1 = -u1
    fwd, up, right = _axes_np(roll, pitch, yaw)
    fwd, up, right = (a.astype(np.float32) for a in (fwd, up, right))
    d = np.clip(d.astype(np.float32), 0.0, 1.0)
    valid_full = ((d > 0.0) & (d < 1.0)).astype(np.float32)
    dh = d[::2, ::2]
    valid = valid_full[::2, ::2]
    cz = dh * 29.5 + 0.5
    cx = fwd[None, :] + right[None, :] * u1[:, None]
    ry_ = up[None, :] * v1[:, None]
    rx = cx[None, :, 0] + ry_[:, None, 0]
    ry = cx[None, :, 1] + ry_[:, None, 1]
    rz = cx[None, :, 2] + ry_[:, None, 2]
    nrm = np.sqrt(rx * rx + ry * ry + rz * rz)
    ray_sin = rz / nrm
    ray_cos = np.sqrt(np.maximum(1.0 - ray_sin * ray_sin, 0.0))
    px, py, pz = (cz * rx, cz * ry, cz * rz)
    z_rel = pz / 20.0
    log_range = np.log1p(cz * nrm) / math.log1p(30.0)
    person_px = np.minimum(1.0, 1.7 * 128.0 / np.maximum(cz, 0.5) / 64.0)
    ax, ay, az = (px[1:-1, 2:] - px[1:-1, :-2], py[1:-1, 2:] - py[1:-1, :-2], pz[1:-1, 2:] - pz[1:-1, :-2])
    bx, by, bz = (px[2:, 1:-1] - px[:-2, 1:-1], py[2:, 1:-1] - py[:-2, 1:-1], pz[2:, 1:-1] - pz[:-2, 1:-1])
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nzz = ax * by - ay * bx
    normal_z = np.zeros((HR, HR), np.float32)
    normal_z[1:-1, 1:-1] = np.abs(nzz) / np.sqrt(np.maximum(nx * nx + ny * ny + nzz * nzz, 1e-12))
    half = G * CELL / 2.0
    ci = np.floor((px + half) / CELL).astype(np.int32)
    cj = np.floor((py + half) / CELL).astype(np.int32)
    inside = (ci >= 0) & (ci < G) & (cj >= 0) & (cj < G) & (valid > 0)
    flat = np.where(inside, cj * G + ci, 0).ravel()
    grid = np.full(G * G, 1000000.0, np.float32)
    np.minimum.at(grid, flat, np.where(inside, pz, np.float32(1000000.0)).ravel().astype(np.float32))
    gp = np.pad(grid.reshape(G, G), WIN // 2, mode='constant', constant_values=1000000.0)
    from numpy.lib.stride_tricks import sliding_window_view
    pooled = sliding_window_view(sliding_window_view(gp, WIN, axis=0).min(-1), WIN, axis=1).min(-1)
    ground = pooled.reshape(-1)[flat].reshape(HR, HR)
    known = (ground < 500000.0) & inside
    height = np.where(known, (np.clip(pz - ground, -2.0, 8.0) + 2.0) / 10.0, 0.0).astype(np.float32)
    out = np.empty((9, HR, HR), np.float32)
    out[0] = ray_sin * valid
    out[1] = ray_cos * valid
    out[2] = z_rel * valid
    out[3] = log_range * valid
    out[4] = person_px * valid
    out[5] = normal_z * valid
    out[6] = height * valid
    out[7] = pitch * 57.29578 / 20.0
    out[8] = roll * 57.29578 / 20.0
    return out
_G4 = [0, 2, 5, 6]

class _DepthTrigger:

    def __init__(self):
        self.sess = None
        self.loaded_for = None
        self.mtype = None
        self.n_eval = 0
        self.n_hit = 0
        self.last = None
        self.last_hit_t = -10 ** 9

    def _load(self, mtype):
        self.loaded_for = mtype
        self.sess = None
        f = _H / ('victim_depth_%s.onnx' % mtype)
        if not f.exists():
            return
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(str(f), so, providers=['CPUExecutionProvider'])
        except Exception:
            self.sess = None

    def step(self, tick, obs, pos, rpy, mtype):
        if not mtype or tick < DT_T0 or tick % DT_EVERY:
            return None
        if mtype != self.loaded_for:
            self._load(mtype)
        if self.sess is None:
            return None
        d = np.asarray(obs['depth'], np.float32).reshape(256, 256)
        g = geo_half_np(d, float(rpy[0]), float(rpy[1]), float(rpy[2]))[_G4]
        o = np.asarray(self.sess.run(None, {'x': d[None, None], 'g': np.ascontiguousarray(g[None], np.float32)})[0], np.float32).reshape(5)
        self.n_eval += 1
        p = _sig(o[0])
        cz = math.exp(min(float(o[3]), 6.0))
        self.last = (p, float(o[1]), float(o[2]), cz, float(o[4]))
        if p < DT_THR or cz > DT_RANGE or float(o[4]) > DT_HMAX:
            return None
        q = _mworld(float(o[1]), float(o[2]), cz, pos, rpy)
        if not np.all(np.isfinite(q)):
            return None
        self.n_hit += 1
        self.last_hit_t = tick
        return (p, float(o[1]), float(o[2]), cz, q)
PLAN_ON = os.environ.get('KT_PLAN', '1') == '1'
PLAN_TYPES = tuple((t for t in os.environ.get('KT_PLAN_TYPES', 'forest,village,warehouse').split(',') if t))
PLAN_REFF = float(os.environ.get('KT_PLAN_REFF', '18.0'))
PLAN_RGAIN = float(os.environ.get('KT_PLAN_RGAIN', '11.0'))
PLAN_EVERY = int(os.environ.get('KT_PLAN_EVERY', '25'))
PLAN_ARR = float(os.environ.get('KT_PLAN_ARR', '6.0'))
PLAN_MIN = float(os.environ.get('KT_PLAN_MIN', '5'))
PLAN_TURN = float(os.environ.get('KT_PLAN_TURN', '0.5'))
PLAN_HINT = float(os.environ.get('KT_PLAN_HINT', '3.0'))
PLAN_STALL_N = int(os.environ.get('KT_PLAN_STALL_N', '100'))
PLAN_STALL_V = float(os.environ.get('KT_PLAN_STALL_V', '0.6'))
E3_ON = os.environ.get('KT_E3', '0') == '1'
RV_ON = os.environ.get('KT_RV', '1') == '1'
RV_R = float(os.environ.get('KT_RV_R', '40.0'))
RV_OFF = 22.0
_RV_ANG = (0.0, 0.0, 120.0, -120.0, -60.0, 180.0, 60.0, -60.0, 0.0)
E2A_ON = os.environ.get('KT_E2A', '0') == '1'
E1_TYPES = tuple((x for x in os.environ.get('KT_E1_TYPES', 'village,warehouse').split(',') if x))
E1P_ON = os.environ.get('KT_E1P', '1') == '1'
E1O_ON = os.environ.get('KT_E1O', '1') == '1'
E1O_TYPES = tuple((x for x in os.environ.get('KT_E1O_TYPES', 'village').split(',') if x))
E1G_ON = os.environ.get('KT_E1G', '1') == '1'
E1X_ON = os.environ.get('KT_E1X', '1') == '1'
E1B = float(os.environ.get('KT_E1B', '1.0'))
E1E_ON = os.environ.get('KT_E1E', '1') == '1'
E1_EVERY = int(os.environ.get('KT_E1_EVERY', '150'))
PLAN_GZ = float(os.environ.get('KT_PLAN_GZ', '0.05'))
PLAN_MARGIN = float(os.environ.get('KT_PLAN_MARGIN', '1.5'))
PLAN_QTICKS = float(os.environ.get('KT_PLAN_QTICKS', '30'))
PRUNE_H = float(os.environ.get('KT_PRUNE_H', '2.5'))
PRUNE_N = int(os.environ.get('KT_PRUNE_N', '3'))
E8X_ON = os.environ.get('KT_E8X', '0') == '1'
E8B = float(os.environ.get('KT_E8B', '-1'))
E8S_ON = os.environ.get('KT_E8S', '0') == '1'
E9A_ON = os.environ.get('KT_E9A', '0') == '1'
E9A_T = float(os.environ.get('KT_E9A_T', '0.65'))
E9A_RANGE = float(os.environ.get('KT_E9A_RANGE', '25.0'))
E6_ON = os.environ.get('KT_E6', '0') == '1'
E6_T0 = int(os.environ.get('KT_E6_T0', '0'))
E2P_ON = os.environ.get('KT_E2P', '0') == '1'
E2P_R = float(os.environ.get('KT_E2P_R', '40'))
E2P_EVERY = int(os.environ.get('KT_E2P_EVERY', '50'))
E2P_RESERVE = int(os.environ.get('KT_E2P_RESERVE', '6'))
E1RW_ON = os.environ.get('KT_E1RW', '0') == '1'

def _e1r(plan, tp):
    return (E1R_ON or (E1RW_ON and plan.mtype == 'warehouse' and (tp == 'village'))) and plan.mtype != tp
E1R_ON = os.environ.get('KT_E1R', '0') == '1'
E1X_TAIL = os.environ.get('KT_E1X_TAIL', '1') == '1'
E1OX_ON = os.environ.get('KT_E1OX', '0') == '1'
E1GD_ON = os.environ.get('KT_E1GD', '0') == '1'
PRUNE_ZREL = float(os.environ.get('KT_PRUNE_ZREL', '1e9'))
PRUNE_SUB = float(os.environ.get('KT_PRUNE_SUB', '1.0'))
E7D_ON = os.environ.get('KT_E7D', '1') == '1'
E7D_SWEEP = float(os.environ.get('KT_E7D_SWEEP', '0.8'))
E7D_N = int(os.environ.get('KT_E7D_N', '2'))
BAN_R = float(os.environ.get('KT_BAN_R', '2.5'))
E7S_ON = os.environ.get('KT_E7S', '1') == '1'
_TCUR = [None]

def _u19():
    return _TCUR[0] in ('village', 'warehouse')
SH_ON = os.environ.get('KT_SH', '0') == '1'
SH_LLT = os.environ.get('KT_SH_LLT', '1') == '1'
SH_MAPS = tuple((x for x in os.environ.get('KT_SH_MAPS', 'mountain').split(',') if x))
SH_MODES = tuple((int(x) for x in os.environ.get('KT_SH_MODES', '1,2,3').split(',') if x))
SH_DS = float(os.environ.get('KT_SH_DS', '1.0'))
SH_DSF = float(os.environ.get('KT_SH_DSF', '0.6'))
SH_DSN = float(os.environ.get('KT_SH_DSN', '0.35'))
SH_RN = float(os.environ.get('KT_SH_RN', '2.5'))
SH_ACC = float(os.environ.get('KT_SH_ACC', '3.5'))
SH_ACCV = float(os.environ.get('KT_SH_ACCV', '5.0'))
SH_TAU = float(os.environ.get('KT_SH_TAU', '0.12'))
SH_RATE = float(os.environ.get('KT_SH_RATE', '5.0'))
SH_RATEV = float(os.environ.get('KT_SH_RATEV', '8.0'))
SH_PUSH = float(os.environ.get('KT_SH_PUSH', '1.5'))
SH_VMAXP = float(os.environ.get('KT_SH_VMAXP', '0.6'))
SH_VOX = float(os.environ.get('KT_SH_VOX', '0.15'))
SH_RB = float(os.environ.get('KT_SH_RB', '0.12'))
SH_RI = float(os.environ.get('KT_SH_RI', '12.0'))
SH_RQ = float(os.environ.get('KT_SH_RQ', '3.5'))
SH_DRY = os.environ.get('KT_SH_DRY', '0') == '1'
SH_LOG = os.environ.get('KT_SH_LOG', '')
SH_DLK = int(os.environ.get('KT_SH_DLK', '20'))
SH_EPS = float(os.environ.get('KT_SH_EPS', '0.05'))
SH_STUCK = int(os.environ.get('KT_SH_STUCK', '120'))
SH_RLXN = int(os.environ.get('KT_SH_RLXN', '50'))
SH_RLXB = int(os.environ.get('KT_SH_RLXB', '150'))
SH_DSMIN = float(os.environ.get('KT_SH_DSMIN', '0.3'))
SH_VESC = float(os.environ.get('KT_SH_VESC', '0.8'))
SH_EHOLD = int(os.environ.get('KT_SH_EHOLD', '40'))
SH_EUP = float(os.environ.get('KT_SH_EUP', '0.5'))
SH_ELAT = int(os.environ.get('KT_SH_ELAT', '60'))
SH_ESIDE = os.environ.get('KT_SH_ESIDE', '0') == '1'
TK_ON = os.environ.get('KT_TK', '1') == '1'
TK_LOW = os.environ.get('KT_TK_LOW', '1') == '1'
TK_PLATE = os.environ.get('KT_TK_PLATE', '0') == '1'
TK_Z0 = float(os.environ.get('KT_TK_Z0', '0.191'))
TK_Z0_TOL = float(os.environ.get('KT_TK_Z0_TOL', '0.006'))
TK_F = float(os.environ.get('KT_TK_F', '0.9'))
TK_WALL = float(os.environ.get('KT_TK_WALL', '0.7'))
TK_OUT_MIN = float(os.environ.get('KT_TK_OUT_MIN', '0.7'))
TK_SIDE_D = float(os.environ.get('KT_TK_SIDE_D', '0.85'))
TK_OUTD = float(os.environ.get('KT_TK_OUTD', '-1'))
TK_HW = float(os.environ.get('KT_TK_HW', '0.35'))
TK_HZ2 = float(os.environ.get('KT_TK_HZ2', '1.5'))
TK_FTOP = float(os.environ.get('KT_TK_FTOP', '0.8'))
TK_PZ0 = float(os.environ.get('KT_TK_PZ0', '1.6'))
TK_PZ1 = float(os.environ.get('KT_TK_PZ1', '5.0'))
TK_GAP = float(os.environ.get('KT_TK_GAP', '1.05'))
TK_COL = float(os.environ.get('KT_TK_COL', '1.0'))
TK_REACH = float(os.environ.get('KT_TK_REACH', '0.2'))
TK_REL = int(os.environ.get('KT_TK_REL', '15'))
TK_TEND = int(os.environ.get('KT_TK_TEND', '700'))
TK_HOLD1 = int(os.environ.get('KT_TK_HOLD1', '10'))
TK_SPAWN = os.environ.get('KT_TK_SPAWN', '0') == '1'
RTY_ON = os.environ.get('KT_RTY', '1') == '1'
RTY_FILE = os.environ.get('KT_RTY_FILE', 'models/rgb_terrain_flat.npz')
RTY_T0 = int(os.environ.get('KT_RTY_T0', '150'))
RTY_DT = max(1, int(os.environ.get('KT_RTY_DT', '25')))
RTY_TEND = int(os.environ.get('KT_RTY_TEND', '900'))
RTY_N = int(os.environ.get('KT_RTY_N', '4'))
RTY_K = max(1, int(os.environ.get('KT_RTY_K', '2')))
RTY_P = float(os.environ.get('KT_RTY_P', '0.9'))
RTY_RES = int(os.environ.get('KT_RTY_RES', '6'))
RTY_FROM = tuple((x for x in os.environ.get('KT_RTY_FROM', 'None,city').split(',') if x))
RTY_TO = tuple((x for x in os.environ.get('KT_RTY_TO', 'city,village,forest').split(',') if x))
_RTY_LAB = ('city', 'open', 'mountain', 'village', 'warehouse', 'forest')
_RTY_M = [None]

def _rty_load():
    if _RTY_M[0] is None:
        z = np.load(str(_H / RTY_FILE))
        _RTY_M[0] = (z['feat'], z['thr'], z['lc'], z['rc'], z['leaf'], z['roots'], z['bias'].astype(np.float64), int(z['n_class']))
    return _RTY_M[0]

def _rty_hsv(f):
    r, g, b = (f[..., 0], f[..., 1], f[..., 2])
    mx = np.max(f, axis=-1)
    mn = np.min(f, axis=-1)
    d = mx - mn + 1e-09
    h = np.zeros_like(mx)
    m = mx == r
    h[m] = (g - b)[m] / d[m] % 6.0
    m = mx == g
    h[m] = (b - r)[m] / d[m] + 2.0
    m = mx == b
    h[m] = (r - g)[m] / d[m] + 4.0
    h /= 6.0
    s = d / (mx + 1e-09)
    return (h, s, mx)

def _rty_features(frame):
    f = np.clip(np.asarray(frame, np.float32), 0.0, 1.0)
    h, s, v = _rty_hsv(f)
    feats = []
    w = (s * v).ravel()
    hh, _ = np.histogram(h.ravel(), bins=12, range=(0, 1), weights=w)
    feats += list(hh / (w.sum() + 1e-06))
    feats += list(np.histogram(s.ravel(), bins=6, range=(0, 1))[0] / s.size)
    feats += list(np.histogram(v.ravel(), bins=6, range=(0, 1))[0] / v.size)
    feats += [((h > 0.18) & (h < 0.45) & (s > 0.15) & (v > 0.08)).mean(), (s < 0.12).mean(), ((h > 0.02) & (h < 0.13) & (s > 0.15)).mean(), ((h > 0.5) & (h < 0.75) & (s > 0.15)).mean(), ((v > 0.8) & (s < 0.15)).mean(), (v < 0.15).mean()]
    H = f.shape[0]
    for band in (slice(0, H // 3), slice(H // 3, 2 * H // 3), slice(2 * H // 3, H)):
        feats += [float(v[band].mean()), float(s[band].mean()), float(((h[band] > 0.18) & (h[band] < 0.45) & (s[band] > 0.15)).mean())]
    feats += [float(np.abs(np.diff(v, axis=1)).mean()), float(np.abs(np.diff(v, axis=0)).mean()), float(np.abs(np.diff(v[::4, ::4], axis=1)).mean()), float(v.std()), float(s.std())]
    return np.asarray(feats, np.float64)

def _rty_proba(frame):
    F, T, L, R, V, roots, bias, n_class = _rty_load()
    x = _rty_features(frame)
    m = np.zeros(n_class)
    for t in range(len(roots)):
        i = int(roots[t])
        while F[i] >= 0:
            i = int(L[i]) if x[F[i]] < T[i] else int(R[i])
        m[t % n_class] += V[i]
    m += bias
    e = np.exp(m - m.max())
    return e / e.sum()
TKX_ON = False

class _Planner:

    def __init__(self, sup, w0):
        self.sup = np.asarray(sup, np.float64)
        self.w0 = np.asarray(w0, np.float64)
        self.mass = self.w0.copy()
        self.wp = None
        self.t_wp = -10 ** 9
        self.n_obs = 0
        self.n_plan = 0
        self.n_hint = 0
        self.resets = 0
        self.stall = 0
        self.n_stall = 0
        self.seen_hits = {}
        self.blk = None
        d = self.sup[:, None, :] - self.sup[None, :, :]
        self.near = (d[:, :, 0] ** 2 + d[:, :, 1] ** 2 <= PLAN_RGAIN ** 2).astype(np.float64)
        self.mtype = None
        self.occ = np.zeros(len(self.sup), bool)
        self.occ_hits = np.zeros(len(self.sup), np.int64)
        self.n_occ = 0
        ix = np.round((self.sup[:, 0] + 48.0) / 2.0).astype(int)
        iy = np.round((self.sup[:, 1] + 48.0) / 2.0).astype(int)
        okg = (ix >= 0) & (ix < 49) & (iy >= 0) & (iy < 49)
        self.gidx = np.full((49, 49), -1, np.int64)
        self.gidx[iy[okg], ix[okg]] = np.arange(len(self.sup))[okg]

    def observe(self, pos, rpy, depth, agl):
        if E1P_ON and self.mtype in E1_TYPES:
            return self._observe_proj(pos, rpy, depth)
        d = self.sup - pos[0:2]
        r = np.hypot(d[:, 0], d[:, 1])
        yaw = float(rpy[2])
        pitch = float(rpy[1])
        b = np.arctan2(d[:, 1], d[:, 0]) - yaw
        b = (b + math.pi) % (2 * math.pi) - math.pi
        cand = (np.abs(b) <= math.radians(44.0)) & (r <= PLAN_REFF) & (r >= 1.5) & (self.mass > 0)
        if not cand.any():
            return
        i = np.where(cand)[0]
        bi = b[i]
        ri = r[i]
        agl = max(0.5, float(agl))
        u = -np.tan(bi)
        dep = np.arctan2(agl, ri)
        v = np.tan(-dep - pitch)
        ok = (np.abs(u) < 1.0) & (np.abs(v) < 1.0)
        if not ok.any():
            return
        i = i[ok]
        u = u[ok]
        v = v[ok]
        ri = ri[ok]
        px = np.clip(((u + 1.0) * 128.0).astype(int), 1, 254)
        py = np.clip(((1.0 - v) * 128.0).astype(int), 1, 254)
        dm = np.asarray(depth, np.float32).reshape(256, 256)
        dv = np.maximum.reduce([dm[py - 1, px], dm[py + 1, px], dm[py, px - 1], dm[py, px + 1], dm[py, px]]) * (_DMAX - _DMIN) + _DMIN
        rs = np.sqrt(ri * ri + agl * agl)
        seen = dv >= rs - 2.0
        q = np.where(ri <= 10.0, 0.85, np.where(ri <= 14.0, 0.6, 0.3))
        self.mass[i[seen]] *= 1.0 - q[seen]
        self.n_obs += 1

    def _observe_proj(self, pos, rpy, depth):
        fwd, up, right = _axes(rpy)
        cam = np.asarray(pos, np.float64) + fwd * _CAMF + up * _CAMU
        if E1O_ON and self.mtype in E1O_TYPES:
            self._occupancy(cam, fwd, up, right, depth)
        d = self.sup - pos[0:2]
        r = np.hypot(d[:, 0], d[:, 1])
        cand = (r <= PLAN_REFF) & (r >= 1.5) & (self.mass > 0)
        if not cand.any():
            return
        i = np.where(cand)[0]
        P = np.column_stack([self.sup[i], np.full(len(i), PLAN_GZ)]) - cam
        cz = P @ fwd
        ok = cz > _DMIN
        if not ok.any():
            return
        i = i[ok]
        P = P[ok]
        cz = cz[ok]
        u = P @ right / cz
        v = P @ up / cz
        ok = (np.abs(u) < 1.0) & (np.abs(v) < 1.0) & (cz < _DMAX)
        if not ok.any():
            return
        i = i[ok]
        u = u[ok]
        v = v[ok]
        cz = cz[ok]
        ri = r[i]
        px = np.clip(np.round((u + 1.0) * 128.0 - 0.5).astype(int), 1, 254)
        py = np.clip(np.round((1.0 - v) * 128.0 - 0.5).astype(int), 1, 254)
        dm = np.asarray(depth, np.float32).reshape(256, 256)
        dv = np.maximum.reduce([dm[py - 1, px], dm[py + 1, px], dm[py, px - 1], dm[py, px + 1], dm[py, px]]) * (_DMAX - _DMIN) + _DMIN
        seen = dv >= cz - PLAN_MARGIN
        q = np.where(ri <= 10.0, 0.85, np.where(ri <= 14.0, 0.6, 0.3))
        if PLAN_QTICKS > 1.0:
            q = 1.0 - (1.0 - q) ** (1.0 / PLAN_QTICKS)
        self.mass[i[seen]] *= 1.0 - q[seen]
        self.n_obs += 1

    def _occupancy(self, cam, fwd, up, right, depth):
        c = (np.arange(0, 256, 4) + 0.5 - 128.0) / 128.0
        dm = np.asarray(depth, np.float32).reshape(256, 256)[::4, ::4]
        z = dm * (_DMAX - _DMIN) + _DMIN
        ray = fwd[None, None, :] + c[None, :, None] * right[None, None, :] - c[:, None, None] * up[None, None, :]
        Pw = cam[None, None, :] + z[..., None] * ray
        hi = (dm < 0.999) & (Pw[..., 2] - PLAN_GZ > PRUNE_H) & (z <= PLAN_REFF + 4.0) & (Pw[..., 2] < cam[2] + PRUNE_ZREL)
        if not hi.any():
            return
        q = Pw[hi]
        ix = np.round((q[:, 0] + 48.0) / 2.0).astype(int)
        iy = np.round((q[:, 1] + 48.0) / 2.0).astype(int)
        ok = (ix >= 0) & (ix < 49) & (iy >= 0) & (iy < 49) & (np.abs(q[:, 0] - (ix * 2.0 - 48.0)) <= PRUNE_SUB) & (np.abs(q[:, 1] - (iy * 2.0 - 48.0)) <= PRUNE_SUB)
        k = self.gidx[iy[ok], ix[ok]]
        k = k[k >= 0]
        if k.size == 0:
            return
        np.add.at(self.occ_hits, k, 1)
        new = (self.occ_hits >= PRUNE_N) & ~self.occ
        if new.any():
            self.occ |= new
            self.n_occ += int(new.sum())
        self.mass[self.occ] = 0.0

    def hint(self, xy, amp):
        d = self.sup - np.asarray(xy, np.float64)[0:2]
        self.mass += amp * float(self.w0.max()) * np.exp(-(d[:, 0] ** 2 + d[:, 1] ** 2) / (2 * 4.0 ** 2))
        self.n_hint += 1

    def next(self, pos, yaw, tick, force=False):
        pe = E1_EVERY if E1_EVERY > 0 and self.mtype in E1_TYPES else PLAN_EVERY
        if not force and self.wp is not None and (tick - self.t_wp < pe) and (float(np.linalg.norm(self.wp - pos[0:2])) > PLAN_ARR):
            return self.wp
        if float(self.mass.sum()) < 0.05 * float(self.w0.sum()):
            self.mass = 0.5 * self.w0.copy()
            self.resets += 1
            self.mass[self.occ] = 0.0
        if E1OX_ON:
            self.mass[self.occ] = 0.0
        gain = self.near @ self.mass
        off = self.sup - pos[0:2]
        dist = np.hypot(off[:, 0], off[:, 1])
        ang = np.arctan2(off[:, 1], off[:, 0]) - yaw
        ang = np.abs((ang + math.pi) % (2 * math.pi) - math.pi)
        cost = dist / 3.0 + 1.0 + PLAN_TURN * ang
        score = gain / cost
        score[dist < (PLAN_MIN if _u19() else 8.0)] = -1.0
        if E1OX_ON:
            score[self.occ] = -1.0
        if self.blk is not None and tick <= self.blk[1]:
            bb = np.abs((np.arctan2(off[:, 1], off[:, 0]) - self.blk[0] + math.pi) % (2 * math.pi) - math.pi) < FSX_SEC
            if (~bb & (score > 0)).any():
                score[bb] = -1.0
        self.wp = self.sup[int(np.argmax(score))].copy()
        self.t_wp = tick
        self.n_plan += 1
        return self.wp

class _RgbPrimary:

    def __init__(self, model='victim_rgb_m.onnx', in_res=128, thr=None, hmax=None, rng_max=None, term_r=None, cap=None, model_by_type=None):
        self.sess = None
        self._tried = False
        self.model = model
        self.in_res = int(in_res)
        self.model_by_type = dict(model_by_type or {})
        self.mtype = None
        self._loaded_for = None
        self.model_used = None
        self.thr = thr if thr is None else float(thr)
        self.hmax = RP_HMAX if hmax is None else float(hmax)
        self.rng_max = RP_RANGE_MAX if rng_max is None else float(rng_max)
        self.term_r = RP_TERM_R if term_r is None else float(term_r)
        self.cap = RP_CAP if cap is None else int(cap)
        self.reset()

    def _pick(self):
        m = self.model_by_type.get(self.mtype)
        return m if m and (_H / m).exists() else self.model

    def set_type(self, mtype):
        if mtype and mtype != self.mtype:
            self.mtype = mtype
            if self._tried and self._loaded_for != self._pick():
                self._tried = False

    def _load(self):
        self._tried = True
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            m = self._pick()
            self.sess = ort.InferenceSession(str(_H / m), so, providers=['CPUExecutionProvider'])
            self.iname = self.sess.get_inputs()[0].name
            self._loaded_for = m
            self.model_used = m
        except Exception:
            self.sess = None

    def reset(self):
        self.n_req = 0
        self.last_req = -10 ** 9
        self.hit_t = -10 ** 9
        self.hits = []
        self.latch = None
        self.near_since = None
        self.bad = []
        self.clue0 = None
        self.n_fix = 0
        self.n_latch = 0
        self.n_dud = 0
        self.latch_t = None
        self.last_q = None
        self.last_q_t = None
        self.n_nc = 0
        self.n_sweep = 0
        self.rg_th = None
        self.rg_done = False
        self.rg_key = None
        self.rg_n = 0
        self.fast = getattr(self, 'fast', False)
        self.box = getattr(self, 'box', None)
        self.min_tz = getattr(self, 'min_tz', None)
        if RST47_ON:
            self.box = None
            self.min_tz = None
        self.whd = False

    def score(self, rgb):
        fr = np.asarray(rgb, np.float32)
        if self.sess is None or fr.size != 256 * 256 * 3 or float(np.mean(np.abs(fr))) < 0.005:
            return None
        if self.in_res == 256:
            r = np.ascontiguousarray(fr.reshape(256, 256, 3), np.float32)
        else:
            r = np.ascontiguousarray(fr.reshape(256, 256, 3).reshape(128, 2, 128, 2, 3).mean(axis=(1, 3)), np.float32)
        o = np.asarray(self.sess.run(None, {self.iname: r})[0], np.float32).reshape(5)
        return (_sig(o[0]), float(o[1]), float(o[2]), float(o[4]))

    def localise(self, u, v, depth, pos, rpy):
        d = np.asarray(depth, np.float32).reshape(256, 256)
        px = min(255, max(0, int(round((u + 1.0) * 0.5 * 256 - 0.5))))
        py = min(255, max(0, int(round((1.0 - v) * 0.5 * 256 - 0.5))))
        cz = float(np.median(d[max(0, py - 2):py + 3, max(0, px - 2):px + 3])) * (_DMAX - _DMIN) + _DMIN
        if cz > RP_IMG_MAX or cz > self.rng_max:
            return None
        q = _mworld(u, v, cz, pos, rpy)
        if not np.all(np.isfinite(q)):
            return None
        if self.clue0 is not None and float(np.linalg.norm(q[0:2] - self.clue0)) > RP_CLUE_R:
            return None
        if not -25.0 <= float(q[2] - pos[2]) <= 1.0:
            return None
        if self.box is not None and max(abs(float(q[0])), abs(float(q[1]))) > self.box:
            return None
        if getattr(self, 'fbox', None) is not None and max(abs(float(q[0])), abs(float(q[1]))) > self.fbox:
            self.n_fbx = getattr(self, 'n_fbx', 0) + 1
            return None
        for b in self.bad:
            if float(np.linalg.norm(q[0:2] - b)) < (BAN_R if _u19() else 6.0):
                return None
        return q

    def observe(self, tick, obs, pos, rpy):
        if self.sess is None and (not self._tried):
            self._load()
        rgb = obs.get('rgb') if hasattr(obs, 'get') else None
        if rgb is None:
            return
        sc = self.score(rgb)
        if sc is None:
            return
        p, u, v, h = sc
        if p < (RP_T if self.thr is None else self.thr) or h > self.hmax:
            return
        q = self.localise(u, v, obs['depth'], pos, rpy)
        if q is None:
            return
        self.hit_t = tick
        self.n_fix += 1
        self.last_q = np.asarray(q, np.float64).copy()
        self.last_q_t = tick
        if self.latch is not None:
            if float(np.linalg.norm(q[0:2] - self.latch[0:2])) <= 6.0:
                self.latch = (1.0 - RP_EMA) * self.latch + RP_EMA * q
            return
        self.hits = [(t, z) for t, z in self.hits if tick - t <= 400] + [(tick, q)]
        if len(self.hits) >= RP_LATCH_N:
            pts = np.asarray([z for _, z in self.hits[-RP_LATCH_N:]])
            c = pts.mean(0)
            if float(np.max(np.linalg.norm(pts[:, 0:2] - c[0:2], axis=1))) <= RP_LATCH_TOL:
                self.latch = c
                self.near_since = None
                self.n_latch += 1
                self.latch_t = tick

    def want_rgb(self, tick, nochase=False, paced_ok=True):
        if self.sess is None and (not self._tried):
            self._load()
        if self.sess is None or self.n_req >= self.cap or tick < RP_T0:
            return False
        chase = tick - self.hit_t <= RP_CHASE_TICKS and tick - self.last_req >= RP_CHASE_MIN
        paced = paced_ok and tick - self.last_req >= getattr(self, 'every', RP_EVERY)
        if nochase and chase:
            chase = False
            self.n_nc += int(not paced)
        if chase or paced:
            self.n_req += 1
            self.last_req = tick
            return True
        return False

    def tick_end(self, tick, pos):
        if self.latch is None:
            return
        _hb = bool(getattr(self, 'hb', False))
        _hok = not (self.fast or _hb) or abs(float(pos[2]) - (float(self.latch[2]) + RP_HOVER_UP)) < 1.0
        if getattr(self, 'whd', False):
            _tz = float(self.latch[2]) + RP_HOVER_UP
            if self.min_tz is not None:
                _tz = max(_tz, float(self.min_tz))
            _hok = abs(float(pos[2]) - _tz) < 1.0
            if not _hok and float(np.linalg.norm(self.latch[0:2] - pos[0:2])) <= RP_RING:
                self.n_wd = getattr(self, 'n_wd', 0) + 1
        if _hb and (not _hok) and (float(np.linalg.norm(self.latch[0:2] - pos[0:2])) <= RP_RING):
            self.n_hb = getattr(self, 'n_hb', 0) + 1
        if getattr(self, 'rg_th', None) is not None:
            self.near_since = None
        elif float(np.linalg.norm(self.latch[0:2] - pos[0:2])) <= RP_RING and _hok:
            if self.near_since is None:
                self.near_since = tick
            elif tick - self.near_since > RP_DUD_TICKS:
                if E7D_ON and _u19() and (self.n_sweep < E7D_N):
                    self.latch = self.latch.copy()
                    self.latch[2] += E7D_SWEEP if self.n_sweep == 0 else -2.0 * E7D_SWEEP
                    self.n_sweep += 1
                    self.near_since = None
                else:
                    self.bad.append(self.latch[0:2].copy())
                    self.latch = None
                    self.hits = []
                    self.near_since = None
                    self.n_dud += 1
        else:
            self.near_since = None
        if self.latch is not None and self.latch_t is not None and (tick - self.latch_t > RP_TERM_MAX):
            self.bad.append(self.latch[0:2].copy())
            self.latch = None
            self.hits = []
            self.near_since = None
            self.n_dud += 1

    def terminal(self, pos, agl_m):
        if self.latch is None:
            return None
        off = self.latch[0:2] - pos[0:2]
        d = float(np.linalg.norm(off))
        if d > self.term_r:
            return None
        tz = float(self.latch[2]) + RP_HOVER_UP
        if self.min_tz is not None:
            tz = max(tz, self.min_tz)
            if WHC_ON and tz > WHC_MAX:
                tz = WHC_MAX
                self.n_whc = getattr(self, 'n_whc', 0) + 1
        _tk = getattr(self, 'tcap_k', None)
        if TCAP_ON and _tk is not None and (d <= TCAP_R) and (0.3 < agl_m < TCAP_AGL):
            _cap = float(pos[2]) - float(agl_m) + _tk
            if _cap + TCAP_M < tz:
                tz = _cap
                self.n_tcap = getattr(self, 'n_tcap', 0) + 1
        dz = tz - float(pos[2])
        if dz < 0 and agl_m < RP_AGL_MIN:
            dz = 0.3
        dzc = max(-1.5, min(1.5, dz))
        if RG_ON and getattr(self, 'rg_ok', False) and (not getattr(self, 'rg_done', False)):
            _L = tuple(np.round(np.asarray(self.latch[0:2], np.float64), 1))
            if getattr(self, 'rg_key', None) != _L and getattr(self, 'rg_th', None) is None:
                self.rg_key = _L
                self.rg_n = 0
            if getattr(self, 'rg_th', None) is None:
                self.rg_n = getattr(self, 'rg_n', 0) + 1 if d < RG_D else 0
                if self.rg_n >= RG_WAIT:
                    self.rg_th0 = math.atan2(-float(off[1]), -float(off[0])) if d > 0.2 else 0.0
                    self.rg_th = self.rg_th0
                    self.n_ring = getattr(self, 'n_ring', 0) + 1
            if getattr(self, 'rg_th', None) is not None:
                self.rg_th += RG_V / (RG_R * 50.0)
                if self.rg_th - self.rg_th0 > 2.0 * math.pi * RG_LAPS:
                    self.rg_done = True
                    self.rg_th = None
                else:
                    _T = np.asarray(self.latch[0:2], np.float64) + RG_R * np.array([math.cos(self.rg_th), math.sin(self.rg_th)])
                    _o = _T - np.asarray(pos[0:2], np.float64)
                    _v = np.array([_o[0], _o[1], dzc], np.float64)
                    _n = float(np.linalg.norm(_v))
                    _sp = min(0.8, max(RG_V, 1.5 * float(np.hypot(_o[0], _o[1])))) / 3.0
                    return ((_v / _n if _n > 1e-06 else np.zeros(3)).astype(np.float32), float(_sp))
        vec = np.array([off[0], off[1], dzc], np.float64)
        n = float(np.linalg.norm(vec))
        dirn = vec / n if n > 1e-06 else np.zeros(3)
        if self.fast:
            d3 = math.hypot(d, dz)
            if d < 1.0 and abs(dz) < 0.7:
                speed = 0.04
            elif d < RP_RING and abs(dz) < 0.7:
                speed = 0.15
            else:
                speed = min(0.55, max(0.15, 0.06 * d3 + 0.1))
        elif E7S_ON and _u19():
            speed = 0.06 if d < 1.0 and abs(dz) < 0.7 else 0.3 if d < RP_RING else min(0.8, max(0.3, 0.12 * d + 0.05))
        elif d < 1.0 and abs(dz) < 0.7:
            speed = 0.04
        elif d < RP_RING:
            speed = 0.15
        else:
            speed = min(0.33, max(0.15, 0.05 * d + 0.1))
        return (dirn.astype(np.float32), float(speed))
DH_ON = os.environ.get('KT_DH', '1') == '1'
DH_TYPES = tuple((x for x in os.environ.get('KT_DH_TYPES', 'mountain').split(',') if x))
DH_EVERY = int(os.environ.get('KT_DH_EVERY', '2'))
DH_T0 = int(os.environ.get('KT_DH_T0', '30'))
DH_THR = float(os.environ.get('KT_DH_THR', '0.7'))
DH_RMAX = float(os.environ.get('KT_DH_RMAX', '30.0'))
DH_N = int(os.environ.get('KT_DH_N', '6'))
DH_TOL = float(os.environ.get('KT_DH_TOL', '3.0'))
DH_MTN = os.environ.get('KT_DH_MTN', '1') == '1'
DH_TERM = os.environ.get('KT_DH_TERM', '1') == '1'
DH_P1 = os.environ.get('KT_DH_P1', '1') == '1'
DH_STEER = os.environ.get('KT_DH_STEER', '1') == '1'
DH_TERM_TYPES = tuple((x for x in os.environ.get('KT_DH_TERM_TYPES', 'mountain').split(',') if x))
DH_PRIO_TYPES = tuple((x for x in os.environ.get('KT_DH_PRIO_TYPES', 'mountain').split(',') if x))
DH_PRIO = os.environ.get('KT_DH_PRIO', '1') == '1'
DH_UP = float(os.environ.get('KT_DH_UP', str(RP_HOVER_UP)))
DH_TERM_R = float(os.environ.get('KT_DH_TERM_R', str(RP_TERM_R)))
DH_RING = float(os.environ.get('KT_DH_RING', str(RP_RING)))
DH_SPD = float(os.environ.get('KT_DH_SPD', '0.33'))
DH_DIVE = os.environ.get('KT_DH_DIVE', '1') == '1'
DH_DIVE_V = float(os.environ.get('KT_DH_DIVE_V', '0.5'))
DH_NEAR_Z = float(os.environ.get('KT_DH_NEAR_Z', '1.0'))
DH_NEAR_H = float(os.environ.get('KT_DH_NEAR_H', '1.0'))
DH_DUD = int(os.environ.get('KT_DH_DUD', str(RP_DUD_TICKS)))
DH_STALE = int(os.environ.get('KT_DH_STALE', '0'))
DH_STALE_R = float(os.environ.get('KT_DH_STALE_R', '10.0'))
DH_BASE = float(os.environ.get('KT_DH_BASE', '3.0'))
DH_RAMP = int(os.environ.get('KT_DH_RAMP', '15'))
FCL_ON = os.environ.get('KT_FCL', '1') == '1'
FCL_CZ = 15.0
FCL_PK = 0.9
FCL_N = 2
FCL_TOL = 2.0
FCL_WIN = 50
FCL_WAIT = 20
FCL_T0 = 900
PF_ON = os.environ.get('KT_PF', '1') == '1'
PF_RGB_MAX = int(os.environ.get('KT_PF_RGB_MAX', '4'))
PF_BAN_R = 6.0
PF_MODEL = os.environ.get('KT_PF_MODEL', 'A')
_PF_LOG = os.environ.get('WPF_LOG')
_PF_DMIN, _PF_DMAX = (0.5, 30.0)
_PF_CZ, _PF_WIN, _PF_TOL = (15.0, 50, 2.0)

def _pf_dstats(d, px, py):
    ix, iy = (int(round(px)), int(round(py)))
    if not (0 <= ix < 256 and 0 <= iy < 256):
        return (np.nan,) * 4
    a5 = d[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]
    a9 = d[max(0, iy - 4):iy + 5, max(0, ix - 4):ix + 5]
    s = _PF_DMAX - _PF_DMIN
    return (float(d[iy, ix]) * s + _PF_DMIN, float(np.median(a5)) * s + _PF_DMIN, float(a9.min()) * s + _PF_DMIN, float(a9.max()) * s + _PF_DMIN)

def _pf_crop(img, px, py, n, fill):
    ix, iy = (int(round(px)), int(round(py)))
    h = n // 2
    out = np.full((n, n) + img.shape[2:], fill, img.dtype)
    y0, x0 = (iy - h, ix - h)
    ys, xs = (max(0, y0), max(0, x0))
    ye, xe = (min(img.shape[0], y0 + n), min(img.shape[1], x0 + n))
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return out

def _pf_pix(u, v):
    return ((u + 1.0) * 128.0 - 0.5, (1.0 - v) * 128.0 - 0.5)

def _pf_axes(rpy):
    r, p, y = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y))
    fwd = np.array([cy * cp, sy * cp, -sp])
    up = np.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr])
    return (fwd, up, np.cross(fwd, up))

def _pf_proj(q, pos, rpy):
    fwd, up, right = _pf_axes(rpy)
    d = np.asarray(q, float) - (np.asarray(pos, float) + fwd * 0.13 + up * 0.05)
    z = float(d @ fwd)
    if z <= 0.05:
        return None
    return (float(d @ right) / z, float(d @ up) / z, z)

def _pf_dcnn_in(crop16, cz):
    c = crop16.astype(np.float32)
    valid = (c >= 0).astype(np.float32)
    c0 = np.where(valid > 0, c, 0.0).astype(np.float32)
    czn = np.float32((min(cz, 30.0) - 0.5) / 29.5)
    rel = np.clip((c0 - czn) * 8.0, -1, 1) * valid
    return (np.stack([c0, valid, rel], 0)[None].astype(np.float32), np.array([[math.log(max(cz, 0.5)) / 3.0]], np.float32))

class _PFGBM:

    def __init__(self, z):
        self.feat = z['feat']
        self.thr = z['thr']
        self.left = z['left']
        self.right = z['right']
        self.dleft = z['dleft'].astype(bool)
        self.mnan = z['mnan'].astype(bool)
        self.mzero = z['mzero'].astype(bool)
        self.leaf = z['leaf']
        self.root = z['root']
        self.depth = int(z['depth'])

    def raw(self, x):
        x = np.asarray(x, np.float64)
        node = self.root.copy()
        for _ in range(self.depth + 1):
            act = node >= 0
            if not act.any():
                break
            n = node[act]
            f = self.feat[n]
            v = x[f]
            t = self.thr[n]
            isn = np.isnan(v)
            vz = np.where(isn & ~self.mnan[n], 0.0, v)
            miss = isn & self.mnan[n] | self.mzero[n] & ((vz == 0.0) | isn)
            go_left = np.where(miss, self.dleft[n], vz <= t)
            node[act] = np.where(go_left, self.left[n], self.right[n])
        return float(self.leaf[~node].sum())

class _PFilter:

    def __init__(self, d):
        import json
        self.meta = json.load(open(os.path.join(str(d), 'pf_meta.json')))
        self.models = {k: [_PFGBM(np.load(os.path.join(str(d), f))) for f in v['files']] for k, v in self.meta['models'].items()}
        self.feats = {k: v['features'] for k, v in self.meta['models'].items()}
        self.thr = {k: float(v['threshold']) for k, v in self.meta['models'].items()}
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sd = ort.InferenceSession(os.path.join(str(d), 'pf_cnn_depth.onnx'), so, providers=['CPUExecutionProvider'])
        self.sr = ort.InferenceSession(os.path.join(str(d), 'pf_cnn_rgb.onnx'), so, providers=['CPUExecutionProvider'])
        assert set(self.models) >= {'A', 'B'} and all((len(v) == 3 for v in self.models.values()))
        self.sd.run(None, {'x': np.zeros((1, 3, 64, 64), np.float32), 'z': np.zeros((1, 1), np.float32)})
        self.sr.run(None, {'x': np.zeros((1, 3, 96, 96), np.float32), 'z': np.zeros((1, 1), np.float32)})
        self.reset()

    def reset(self):
        self.acc = []

    def hit_record(self, tick, depth, o, m, use_m, cz, u, v, q, pos, rpy, vel, agl, clue0):
        d2 = np.asarray(depth, np.float32).reshape(256, 256)
        o = np.asarray(o, np.float64).reshape(5)
        m = np.asarray(m, np.float64).reshape(5)
        px, py = _pf_pix(u, v)
        dpx, dmed, dmin, dmax = _pf_dstats(d2, px, py)
        sig = lambda x: 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, float(x)))))
        p0 = sig(o[0])
        t_old, t_mtn = (0.657, 0.586)
        pmr = sig(m[0])
        lo = pmr * (t_old / t_mtn)
        hi = t_old + (pmr - t_mtn) * ((1.0 - t_old) / (1.0 - t_mtn))
        pm = float(min(max(min(lo, hi), 0.0), 1.0))
        lcz0, lczm = (min(o[3], 6.0), min(m[3], 6.0))
        q = np.asarray(q, np.float64).reshape(3)
        r = dict(tick=int(tick), q=q, cz=float(cz), p=float(pm if use_m else p0), p0=p0, pm=pm, use_m=float(bool(use_m)), u=float(u), v=float(v), pos=np.asarray(pos, np.float64).reshape(3)[:3].copy(), pitch=float(rpy[1]), roll=float(rpy[0]), spd=float(math.hypot(vel[0], vel[1])), vz=float(vel[2]), agl=float(agl), clue_d=float(np.linalg.norm(q[0:2] - np.asarray(clue0, float)[0:2])) if clue0 is not None else np.nan, d_px=dpx, d_med5=dmed, d_min9=dmin, d_max9=dmax, o4=float(o[4]), m4=float(m[4]), dl=float(lcz0 - lczm), duv=float(math.hypot(o[1] - m[1], o[2] - m[2]) * 128.0), sr0=dmed - math.exp(lcz0), srm=dmed - math.exp(lczm), cnn=np.nan)
        if cz <= _PF_CZ:
            cr = _pf_crop(d2, px, py, 64, -1.0).astype(np.float16)
            x, z = _pf_dcnn_in(cr, cz)
            r['cnn'] = float(np.asarray(self.sd.run(None, {'x': x, 'z': z})[0]).reshape(-1)[0])
        return r

    def add(self, rec):
        self.acc.append(rec)

    def rgb_features(self, rgb, pos, rpy, det, q_new, cz_new):
        pr = _pf_proj(q_new, pos, rpy)
        if pr is None or self.sr is None:
            return dict(rgb_p_last=np.nan, rgb_qd_last=np.nan, rgb_pixd_last=np.nan, cr_last=np.nan)
        px, py = _pf_pix(pr[0], pr[1])
        cr = (_pf_crop(np.asarray(rgb, np.float32).reshape(256, 256, 3), px, py, 96, 0.0) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        x = cr.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        z = np.array([[math.log(max(float(cz_new), 0.5)) / 3.0]], np.float32)
        s = float(np.asarray(self.sr.run(None, {'x': x, 'z': z})[0]).reshape(-1)[0])
        p, uu, vv, qq = det if det is not None else (np.nan, np.nan, np.nan, None)
        pixd = float(math.hypot(*np.subtract(_pf_pix(uu, vv), (px, py)))) if uu is not None and np.isfinite(uu) else np.nan
        qd = float(np.hypot(qq[0] - q_new[0], qq[1] - q_new[1])) if qq is not None else np.nan
        return dict(rgb_p_last=float(p), rgb_qd_last=qd, rgb_pixd_last=pixd, cr_last=s)

    def features(self):
        A = self.acc
        jj = len(A) - 1
        j = A[jj]
        T = np.array([a['tick'] for a in A])
        CZ = np.array([a['cz'] for a in A])
        Q = np.array([a['q'] for a in A])
        close = [x for x in range(jj + 1) if CZ[x] <= _PF_CZ and T[jj] - T[x] <= _PF_WIN]
        members = [jj]
        if len(close) >= 2:
            pair = close[-2:]
            mq = Q[pair].mean(0)
            if float(np.max(np.linalg.norm(Q[pair, :2] - mq[:2], axis=1))) <= _PF_TOL:
                members = pair
        M = [A[x] for x in members]
        mq = Q[members].mean(0)
        P = np.array([a['p'] for a in M])
        POS = np.array([a['pos'] for a in M])
        sr = np.array([a['d_med5'] - a['cz'] for a in M])
        cl = [x for x in range(jj + 1) if T[jj] - T[x] <= 400 and math.hypot(*Q[x, :2] - mq[:2]) <= 3.0]
        vp = np.array([A[x]['pos'][:2] for x in cl])
        f = dict(p_max=P.max(), p_min=P.min(), cz_min=min((a['cz'] for a in M)), cz_max=max((a['cz'] for a in M)), dq=float(np.linalg.norm(Q[members[0], :2] - Q[members[-1], :2])), dtick=float(T[members[-1]] - T[members[0]]), vp_d=float(np.linalg.norm(POS[0, :2] - POS[-1, :2])), dz=float(mq[2] - j['pos'][2]), rng_h=float(math.hypot(*mq[:2] - j['pos'][:2])), sr_absmax=float(np.nanmax(np.abs(sr))) if np.isfinite(sr).any() else np.nan, sr_absmin=float(np.nanmin(np.abs(sr))) if np.isfinite(sr).any() else np.nan, sr_last=float(sr[-1]), n_close_win=float(len(close)), n_clu400=float(len(cl)), vspan400=float(np.max(np.linalg.norm(vp - vp.mean(0), axis=1)) * 2) if len(cl) >= 2 else 0.0, n_acc_before=float(jj), n_acc_before_far=float(sum((1 for x in range(jj) if math.hypot(*Q[x, :2] - mq[:2]) > 6.0))), agl=j['agl'], spd=j['spd'], vz=j['vz'], pitch=j['pitch'], roll=j['roll'], clue_d=j['clue_d'], t=j['tick'] * 0.02, h_p0=j['p0'], h_pm=j['pm'], h_use_m=j['use_m'], h_u=j['u'], h_v=j['v'], h_dmin9=j['d_min9'] - j['cz'], h_dmax9=j['d_max9'] - j['cz'], h_dpx=j['d_px'] - j['cz'], h_span=float(j['d_min9'] <= j['cz'] + max(1, 0.1 * j['cz']) and j['d_max9'] >= j['cz'] - max(1, 0.1 * j['cz'])), hd_o4=j['o4'], hd_m4=j['m4'], hd_dlcz=j['dl'], hd_duv=j['duv'], hd_sr0=j['sr0'], hd_srm=j['srm'], hd_sr_hm=min(abs(j['sr0']), abs(j['srm'])), hd_p0m=min((a['p0'] for a in M)), hd_pmm=min((a['pm'] for a in M)), hd_dlcz_max=max((abs(a['dl']) for a in M)), hd_o4_min=min((a['o4'] for a in M)), hd_m4_min=min((a['m4'] for a in M)), cd_last=j['cnn'], cd_min=float(np.nanmin([a['cnn'] for a in M])) if np.isfinite([a['cnn'] for a in M]).any() else np.nan)
        f['sr_rel'] = f['sr_last'] / max(1.0, f['cz_min'])
        return (f, members)

    def score(self, model, f):
        x = np.array([f.get(k, np.nan) for k in self.feats[model]], np.float64)
        return float(np.mean([g.raw(x) for g in self.models[model]]))

def _pf_loc(u, v, depth, pos, rpy):
    d = np.asarray(depth, np.float32).reshape(256, 256)
    px = min(255, max(0, int(round((u + 1.0) * 0.5 * 256 - 0.5))))
    py = min(255, max(0, int(round((1.0 - v) * 0.5 * 256 - 0.5))))
    cz = float(np.median(d[max(0, py - 2):py + 3, max(0, px - 2):px + 3])) * (_DMAX - _DMIN) + _DMIN
    if cz > RP_IMG_MAX or cz > RP_RANGE_MAX:
        return None
    q = _mworld(u, v, cz, pos, rpy)
    if not np.all(np.isfinite(q)):
        return None
    if not -25.0 <= float(q[2] - pos[2]) <= 1.0:
        return None
    return q

def _pf_log(rec):
    if not _PF_LOG:
        return
    try:
        import json
        os.makedirs(_PF_LOG, exist_ok=True)
        with open(os.path.join(_PF_LOG, '%d.jsonl' % os.getpid()), 'a') as fh:
            fh.write(json.dumps(rec, default=float) + '\n')
    except Exception:
        pass
_T_OLD_DEPTH = 0.657
_T_MTN_DEPTH = 0.586

def _recal(p, t_own, t_target):
    lo = p * (t_target / t_own)
    hi = t_target + (p - t_own) * ((1.0 - t_target) / (1.0 - t_own))
    return float(min(max(min(lo, hi), 0.0), 1.0))

class _DHead(_RgbPrimary):

    def __init__(self):
        self.pf = None
        if FCL_ON and PF_ON:
            try:
                self.pf = _PFilter(_H)
            except Exception:
                self.pf = None
        super().__init__(model='victim_depth.onnx', term_r=RP_TERM_R)
        self.msess = None
        self._mtried = False

    def _load(self):
        self._tried = True
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(str(_H / 'victim_depth.onnx'), so, providers=['CPUExecutionProvider'])
            self.iname = self.sess.get_inputs()[0].name
        except Exception:
            self.sess = None

    def _mload(self):
        self._mtried = True
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            self.msess = ort.InferenceSession(str(_H / 'victim_depth_m.onnx'), so, providers=['CPUExecutionProvider'])
            self.miname = self.msess.get_inputs()[0].name
        except Exception:
            self.msess = None

    def reset(self):
        super().reset()
        self.n_eval = 0
        self.obs_xy = []
        self.fc = []
        self.fcl = None
        self.fcl_n = 0
        self.fcl_t = -1
        self.pf_ok = self.pf is not None
        self.pf_f = None
        self.pf_q = None
        self.pf_cz = None
        self.pf_rgb = False
        self.pf_want = False
        self.pf_t = -1
        self.pf_nrgb = 0
        self.pf_ban = []
        self.pf_ncand = 0
        self.pf_nkeep = 0
        self.pf_nrej = 0
        self.pf_err = 0
        if getattr(self, 'pf', None) is not None:
            self.pf.reset()

    def observe_depth(self, tick, obs, pos, rpy, mtn):
        if tick < DH_T0 or tick % DH_EVERY:
            return
        if self.sess is None and (not self._tried):
            self._load()
        if self.sess is None:
            return
        x = np.ascontiguousarray(np.asarray(obs['depth'], np.float32).reshape(256, 256, 1))
        o = np.asarray(self.sess.run(None, {self.iname: x})[0], np.float32).reshape(5)
        p = _sig(o[0])
        b = o
        _mraw = None
        if mtn and DH_MTN:
            if self.msess is None and (not self._mtried):
                self._mload()
            if self.msess is not None:
                m = np.asarray(self.msess.run(None, {self.miname: x})[0], np.float32).reshape(5)
                pm = _recal(_sig(m[0]), _T_MTN_DEPTH, _T_OLD_DEPTH)
                _mraw = m
                if pm > p:
                    p = pm
                    b = m
        self.n_eval += 1
        if p < DH_THR:
            return
        cz = math.exp(min(float(b[3]), 6.0))
        if cz > DH_RMAX:
            return
        q = _mworld(float(b[1]), float(b[2]), cz, pos, rpy)
        if not np.all(np.isfinite(q)):
            return
        if self.clue0 is not None and float(np.linalg.norm(q[0:2] - self.clue0)) > RP_CLUE_R:
            return
        if not -25.0 <= float(q[2] - pos[2]) <= 1.0:
            return
        for bb in self.bad:
            if float(np.linalg.norm(q[0:2] - bb)) < 6.0:
                return
        self.hit_t = tick
        self.n_fix += 1
        if mtn and FCL_ON and self.pf_ok and (self.n_latch == 0) and (self.fcl_n == 0):
            try:
                _s = np.asarray(obs['state'], np.float32).reshape(-1)
                self.pf.add(self.pf.hit_record(tick, obs['depth'], o, _mraw, b is _mraw, cz, float(b[1]), float(b[2]), q, pos, rpy, _s[6:9], float(_s[162]) * 20.0, self.clue0))
            except Exception:
                self.pf_ok = False
                self.pf_err += 1
        if self.latch is not None:
            if float(np.linalg.norm(q[0:2] - self.latch[0:2])) <= 6.0:
                self.latch = (1.0 - RP_EMA) * self.latch + RP_EMA * q
            return
        self.hits = [(t, z) for t, z in self.hits if tick - t <= 400] + [(tick, q)]
        self.obs_xy = [(t, p) for t, p in getattr(self, 'obs_xy', []) if tick - t <= 400] + [(tick, np.asarray(pos[0:2], np.float64).copy())]
        if len(self.hits) >= DH_N:
            pts = np.asarray([z for _, z in self.hits[-DH_N:]])
            c = pts.mean(0)
            if float(np.max(np.linalg.norm(pts[:, 0:2] - c[0:2], axis=1))) <= DH_TOL:
                ok = True
                if DH_BASE > 0.0:
                    ts = {t for t, z in self.hits if float(np.linalg.norm(np.asarray(z, np.float64)[0:2] - c[0:2])) <= DH_TOL}
                    vp = np.asarray([p for t, p in self.obs_xy if t in ts], np.float64)
                    ok = bool(len(vp) >= 2 and float(np.max(np.linalg.norm(vp - vp.mean(0), axis=1))) * 2.0 >= DH_BASE)
                if ok:
                    self.latch = c
                    self.near_since = None
                    self.n_latch += 1
                    self.latch_t = tick
        if FCL_ON and mtn and (self.latch is None) and (self.n_latch == 0) and (self.fcl is None) and (self.fcl_n == 0) and (tick >= FCL_T0):
            self.fc = [h for h in self.fc if tick - h[0] <= FCL_WIN]
            if cz <= FCL_CZ:
                self.fc.append((tick, np.asarray(q, np.float64).copy(), cz, p))
            if len(self.fc) >= FCL_N:
                L = self.fc[-FCL_N:]
                Q = np.asarray([h[1] for h in L])
                m = Q.mean(0)
                if max((h[3] for h in L)) >= FCL_PK and float(np.max(np.linalg.norm(Q[:, 0:2] - m[0:2], axis=1))) <= FCL_TOL:
                    self.fcl = (tick, m)
                    self.pf_f = None
                    self.pf_rgb = False
                    self.pf_want = False
                    self.pf_t = tick
                    self.pf_ncand += 1
                    if self.pf_ok:
                        try:
                            _f, _mem = self.pf.features()
                            _r = self.pf.acc[_mem[-1]]
                            self.pf_f = _f
                            self.pf_q = np.asarray(_r['q'], np.float64).copy()
                            self.pf_cz = float(_r['cz'])
                            self.pf_want = bool(PF_MODEL == 'B' and self.pf_nrgb < PF_RGB_MAX)
                        except Exception:
                            self.pf_f = None
                            self.pf_err += 1

    def pf_keep(self, m):
        if PF_MODEL == 'off':
            try:
                return (True, 'off', self.pf.score('A', self.pf_f) if self.pf is not None and self.pf_ok and (self.pf_f is not None) else float('nan'))
            except Exception:
                return (True, 'off', float('nan'))
        if self.pf is None or not self.pf_ok or self.pf_f is None:
            return (False, 'none', float('nan'))
        try:
            if self.pf_rgb:
                sc = self.pf.score('B', self.pf_f)
                k = bool(sc >= self.pf.thr['B'])
                if not k:
                    self.pf_ban.append(np.asarray(m[0:2], np.float64).copy())
                return (k, 'B', sc)
            if any((float(np.linalg.norm(np.asarray(m[0:2], np.float64) - bb)) < PF_BAN_R for bb in self.pf_ban)):
                return (False, 'banA', float('nan'))
            sc = self.pf.score('A', self.pf_f)
            return (bool(sc >= self.pf.thr['A']), 'A', sc)
        except Exception:
            self.pf_err += 1
            return (False, 'err', float('nan'))

    def terminal(self, pos, agl_m):
        if self.latch is None:
            return None
        off = self.latch[0:2] - pos[0:2]
        d = float(np.linalg.norm(off))
        if d > DH_TERM_R:
            return None
        tz = float(self.latch[2]) + DH_UP
        dz = tz - float(pos[2])
        if dz < 0 and agl_m < RP_AGL_MIN:
            dz = 0.3
        if DH_DIVE:
            if abs(dz) > DH_NEAR_Z or d > DH_NEAR_H:
                vec = np.array([off[0], off[1], max(-6.0, min(6.0, dz)) * (2.0 if abs(dz) > DH_NEAR_Z else 1.0)], np.float64)
                n = float(np.linalg.norm(vec))
                dirn = vec / n if n > 1e-06 else np.zeros(3)
                far = max(d, abs(dz))
                speed = min(DH_DIVE_V, max(0.12, 0.06 * far + 0.1))
                return (dirn.astype(np.float32), float(speed))
            vec = np.array([off[0], off[1], dz], np.float64)
            n = float(np.linalg.norm(vec))
            dirn = vec / n if n > 1e-06 else np.zeros(3)
            return (dirn.astype(np.float32), float(0.04 if n < 0.5 else 0.1))
        dzc = max(-1.5, min(1.5, dz))
        vec = np.array([off[0], off[1], dzc], np.float64)
        n = float(np.linalg.norm(vec))
        dirn = vec / n if n > 1e-06 else np.zeros(3)
        if d < 1.0 and abs(dz) < 0.7:
            speed = 0.04
        elif d < DH_RING:
            speed = 0.15
        else:
            speed = min(DH_SPD, max(0.15, 0.05 * d + 0.1))
        return (dirn.astype(np.float32), float(speed))

    def tick_end(self, tick, pos):
        if self.latch is None:
            return
        z_off = abs(float(self.latch[2]) + DH_UP - float(pos[2]))
        if float(np.linalg.norm(self.latch[0:2] - pos[0:2])) <= DH_RING and z_off <= DH_NEAR_Z + 0.5:
            if self.near_since is None:
                self.near_since = tick
            elif tick - self.near_since > DH_DUD:
                self.bad.append(self.latch[0:2].copy())
                self.latch = None
                self.hits = []
                self.near_since = None
                self.n_dud += 1
        else:
            self.near_since = None
        if self.latch is not None and self.latch_t is not None and (tick - self.latch_t > RP_TERM_MAX):
            self.bad.append(self.latch[0:2].copy())
            self.latch = None
            self.hits = []
            self.near_since = None
            self.n_dud += 1
        if self.latch is not None and DH_STALE > 0 and (tick - self.hit_t > DH_STALE) and (float(np.linalg.norm(self.latch[0:2] - pos[0:2])) <= DH_STALE_R):
            self.bad.append(self.latch[0:2].copy())
            self.latch = None
            self.hits = []
            self.near_since = None
            self.n_stale = getattr(self, 'n_stale', 0) + 1
import time as _nv_time
NV_ON = os.environ.get('KT_NV124', '1') == '1'
NV_T = float(os.environ.get('KT_NV124_T', '0.85'))
NV_EVERY = max(1, int(os.environ.get('KT_NV124_EVERY', '3')))
NV_T0 = int(os.environ.get('KT_NV124_T0', '30'))
NV_TYPES = tuple((x for x in os.environ.get('KT_NV124_TYPES', 'city').split(',') if x and x != 'mountain'))
NV_N = int(os.environ.get('KT_NV124_N', '6'))
NV_TOL = float(os.environ.get('KT_NV124_TOL', '3.0'))
NV_BASE = float(os.environ.get('KT_NV124_BASE', '3.0'))
NV_ZUP = float(os.environ.get('KT_NV124_ZUP', '1.0'))
NV_DC = float(os.environ.get('KT_NV124_DC', '4.0'))
NV_CLUE_R = float(os.environ.get('KT_NV124_CLUE_R', '33.0'))
NV_ZMAX = float(os.environ.get('KT_NV124_ZMAX', '3.0'))
NV_RZ = os.environ.get('KT_NV124_RZ', '1') == '1'
NV_CM = os.environ.get('KT_NV124_CM', '1') == '1'
NV_STALE = int(os.environ.get('KT_NV124_STALE', '250'))
NV_PROG = int(os.environ.get('KT_NV124_PROG', '250'))
NV_NEAR = float(os.environ.get('KT_NV124_NEAR', '8.0'))
NV_NEAR_T = int(os.environ.get('KT_NV124_NEAR_T', '150'))
NV_P3B = os.environ.get('KT_NV124_P3B', '1') == '1'
NV_TERM = os.environ.get('KT_NV124_TERM', '0') == '1'
NV_BAN_T = int(os.environ.get('KT_NV124_BAN_T', '0'))
NV_SBAN_T = int(os.environ.get('KT_NV124_SBAN_T', '600'))
NV_FILE = os.environ.get('KT_NV124_FILE', 'nv124_vil.onnx')
NV_RMAX = float(os.environ.get('KT_NV124_RMAX', '40'))
NV_PW = os.environ.get('KT_NV124_PW', '1') == '1'
_NV_ST = np.zeros(165, np.float32)
VNT_ON = os.environ.get('KT_VNT', '1') == '1'
VNT_TYPES = tuple((x for x in os.environ.get('KT_VNT_TYPES', 'village').split(',') if x and x != 'mountain'))
VNT_R = float(os.environ.get('KT_VNT_R', '25.0'))
VNT_MAX = int(os.environ.get('KT_VNT_MAX', '8'))
VNT_SPOT = int(os.environ.get('KT_VNT_SPOT', '2'))
VNT_CD = int(os.environ.get('KT_VNT_CD', '40'))
VNT_RES = int(os.environ.get('KT_VNT_RES', '6'))
VNT_GAP = int(os.environ.get('KT_VNT_GAP', '3'))
VNT_P1 = os.environ.get('KT_VNT_P1', '0') == '1'
VNT_TRIG = os.environ.get('KT_VNT_TRIG', '1') == '1'
VNT_AGREE = os.environ.get('KT_VNT_AGREE', '1') == '1'
VNT_AW = int(os.environ.get('KT_VNT_AW', '40'))
VNT_AD = float(os.environ.get('KT_VNT_AD', '2.5'))
VNT_SIG = os.environ.get('KT_VNT_SIG', '0') == '1'
VNT_KEEPCLK = os.environ.get('KT_VNT_KEEPCLK', '1') == '1'
_VNT_H = []

class _NVHead(_DHead):

    def __init__(self):
        self.pf = None
        self.n_err = 0
        self.ms = 0.0
        _RgbPrimary.__init__(self, model=NV_FILE, term_r=RP_TERM_R)
        self.msess = None
        self._mtried = True
        self._load()
        if self.sess is not None:
            try:
                self.sess.run([self.oname], {self.iname: np.zeros((256, 256, 1), np.float32), self.sname: _NV_ST})
            except Exception:
                self.sess = None

    def _load(self):
        self._tried = True
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            s = ort.InferenceSession(str(_H / NV_FILE), so, providers=['CPUExecutionProvider'])
            nm = [i.name for i in s.get_inputs()]
            self.iname, self.sname, self.oname = (nm[0], nm[1], s.get_outputs()[0].name)
            self.sess = s
            self.onames = [o.name for o in s.get_outputs()]
        except Exception:
            self.sess = None

    def reset(self):
        super().reset()
        self.n_hit = 0
        self.n_stale = 0
        self.n_prog = 0
        self.n_near = 0
        self.n_p3b = 0
        self.n_park = 0
        self.n_cap = 0
        self.n_dc = 0
        self.n_zmax = 0
        self.n_steer = 0
        self.ms = 0.0
        self.n_err = 0
        self.lh_t = -10 ** 9
        self.st_n = 0
        self.nr_n = 0
        self.lat_st = 0
        self.pd_min = 1000000000.0
        self.pd_n = 0
        self.go_t = -10 ** 9
        self.bad_t = []
        self.bad_s = []

    def ban(self, why, tick, soft=False):
        self.bad.append(self.latch[0:2].copy())
        self.bad_t.append(tick)
        self.bad_s.append(bool(soft))
        self.latch = None
        self.hits = []
        self.near_since = None
        self.n_dud += 1
        setattr(self, why, getattr(self, why) + 1)
        self.st_n = 0
        self.nr_n = 0
        self.lat_st = 0
        self.pd_min = 1000000000.0
        self.pd_n = 0

    def expire(self, tick):
        while len(self.bad_t) < len(self.bad):
            self.bad_t.append(tick)
        while len(self.bad_s) < len(self.bad):
            self.bad_s.append(False)
        if self.bad and (NV_BAN_T > 0 or NV_SBAN_T > 0):
            k = [i for i, (t, s) in enumerate(zip(self.bad_t, self.bad_s)) if not (NV_BAN_T > 0 and tick - t > NV_BAN_T or (s and NV_SBAN_T > 0 and (tick - t > NV_SBAN_T)))]
            self.bad = [self.bad[i] for i in k]
            self.bad_t = [self.bad_t[i] for i in k]
            self.bad_s = [self.bad_s[i] for i in k]

    def tick_end(self, tick, pos):
        if self.latch is None:
            return
        z_off = abs(float(self.latch[2]) + DH_UP - float(pos[2]))
        if float(np.linalg.norm(self.latch[0:2] - pos[0:2])) <= DH_RING and z_off <= DH_NEAR_Z + 0.5:
            if self.near_since is None:
                self.near_since = tick
            elif tick - self.near_since > DH_DUD:
                self.ban('n_park', tick)
                return
        else:
            self.near_since = None
        if self.lat_st > RP_TERM_MAX:
            self.ban('n_cap', tick)

    def observe_nv(self, tick, obs, pos, rpy, p3):
        if tick < NV_T0 or tick % NV_EVERY or self.sess is None:
            return
        x = np.ascontiguousarray(np.asarray(obs['depth'], np.float32).reshape(256, 256, 1))
        t0 = _nv_time.perf_counter()
        _pw = None
        if NV_PW and len(getattr(self, 'onames', ())) >= 2:
            _st = np.zeros(165, np.float32)
            _st[0:3] = np.asarray(pos, np.float32)[0:3]
            _st[3:6] = np.asarray(rpy, np.float32)[0:3]
            _oo = self.sess.run(self.onames[0:2], {self.iname: x, self.sname: _st})
            o = np.asarray(_oo[0], np.float32).reshape(-1)
            _pw = np.asarray(_oo[1], np.float64).reshape(-1)[0:3]
        else:
            o = np.asarray(self.sess.run([self.oname], {self.iname: x, self.sname: _NV_ST})[0], np.float32).reshape(-1)
        self.ms += 1000.0 * (_nv_time.perf_counter() - t0)
        self.n_eval += 1
        if _sig(o[0]) < NV_T:
            return
        cz = math.exp(min(float(o[3]), 6.0))
        if cz > (NV_RMAX if NV_RMAX > 0.0 else DH_RMAX):
            return
        q = _pw if _pw is not None else _mworld(float(o[1]), float(o[2]), cz, pos, rpy)
        if not np.all(np.isfinite(q)):
            return
        if NV_DC > 0.0:
            px = min(255, max(0, int(round((float(o[1]) + 1.0) * 0.5 * 256 - 0.5))))
            py = min(255, max(0, int(round((1.0 - float(o[2])) * 0.5 * 256 - 0.5))))
            dm = float(np.median(x[max(0, py - 2):py + 3, max(0, px - 2):px + 3, 0])) * (_DMAX - _DMIN) + _DMIN
            if abs(cz - dm) > NV_DC:
                self.n_dc += 1
                return
        if self.clue0 is not None and float(np.linalg.norm(q[0:2] - self.clue0)) > NV_CLUE_R:
            return
        if not -25.0 <= float(q[2] - pos[2]) <= NV_ZUP:
            return
        if NV_ZMAX > 0.0 and float(q[2]) > NV_ZMAX:
            self.n_zmax += 1
            return
        for bb in self.bad:
            if float(np.linalg.norm(q[0:2] - bb)) < 6.0:
                return
        for bb in p3:
            if float(np.linalg.norm(q[0:2] - np.asarray(bb[0:2], np.float64))) < P3_BAN_R:
                return
        self.hit_t = tick
        self.n_fix += 1
        self.n_hit += 1
        if self.latch is not None:
            if float(np.linalg.norm(q[0:2] - self.latch[0:2])) <= 6.0:
                self.latch = (1.0 - RP_EMA) * self.latch + RP_EMA * q
                self.lh_t = tick
                self.st_n = 0
            return
        self.hits = [(t, z) for t, z in self.hits if tick - t <= 400] + [(tick, q)]
        self.obs_xy = [(t, p) for t, p in self.obs_xy if tick - t <= 400] + [(tick, np.asarray(pos[0:2], np.float64).copy())]
        if len(self.hits) >= NV_N:
            pts = np.asarray([z for _, z in self.hits[-NV_N:]])
            c = pts.mean(0)
            if float(np.max(np.linalg.norm(pts[:, 0:2] - c[0:2], axis=1))) <= NV_TOL:
                ok = True
                if NV_BASE > 0.0:
                    ts = {t for t, z in self.hits if float(np.linalg.norm(np.asarray(z, np.float64)[0:2] - c[0:2])) <= NV_TOL}
                    vp = np.asarray([p for t, p in self.obs_xy if t in ts], np.float64)
                    ok = bool(len(vp) >= 2 and float(np.max(np.linalg.norm(vp - vp.mean(0), axis=1))) * 2.0 >= NV_BASE)
                if ok:
                    self.latch = c
                    self.near_since = None
                    self.n_latch += 1
                    self.latch_t = tick
                    self.lh_t = tick
                    self.st_n = 0
                    self.nr_n = 0
                    self.lat_st = 0
                    self.pd_min = 1000000000.0
                    self.pd_n = 0
NVF_ON = os.environ.get('KT_NVF', '1') == '1'
NVF_FILES = tuple((x for x in os.environ.get('KT_NVF_FILES', 'models/nvf_u56f.onnx,models/nvf_ft2e0.onnx').split(',') if x))
NVF_T = tuple((float(x) for x in os.environ.get('KT_NVF_T', '0.6,0.6').split(',') if x)) or (0.6,)
NVF_EVERY = max(1, int(os.environ.get('KT_NVF_EVERY', '2')))
NVF_T0 = int(os.environ.get('KT_NVF_T0', '30'))
NVF_TYPES = tuple((x for x in os.environ.get('KT_NVF_TYPES', 'forest').split(',') if x and x != 'mountain'))
NVF_N = int(os.environ.get('KT_NVF_N', '6'))
NVF_TOL = float(os.environ.get('KT_NVF_TOL', '3.0'))
NVF_BASE = float(os.environ.get('KT_NVF_BASE', '3.0'))
NVF_ZUP = float(os.environ.get('KT_NVF_ZUP', '1.0'))
NVF_DC = float(os.environ.get('KT_NVF_DC', '0'))
NVF_CLUE_R = float(os.environ.get('KT_NVF_CLUE_R', '33.0'))
NVF_ZMAX = float(os.environ.get('KT_NVF_ZMAX', '3.0'))
NVF_STALE = int(os.environ.get('KT_NVF_STALE', '1000'))
NVF_PROG = int(os.environ.get('KT_NVF_PROG', '250'))
NVF_NEAR = float(os.environ.get('KT_NVF_NEAR', '8.0'))
NVF_NEAR_T = int(os.environ.get('KT_NVF_NEAR_T', '150'))
NVF_SBAN_T = int(os.environ.get('KT_NVF_SBAN_T', '600'))
NVF_BAN_T = int(os.environ.get('KT_NVF_BAN_T', '0'))
NVF_P3B = os.environ.get('KT_NVF_P3B', '1') == '1'
NVF_RZ = os.environ.get('KT_NVF_RZ', '1') == '1'
NVF_TERM = os.environ.get('KT_NVF_TERM', '0') == '1'
NVF_SHADOW = os.environ.get('KT_NVF_SHADOW', '0') == '1'
NVF_LOG = os.environ.get('KT_NVF_LOG', '')
NVF_BOX = float(os.environ.get('KT_NVF_BOX', '34.0'))
NVF_AND = NVF_ON and os.environ.get('KT_NVF_AND', '1') == '1'
NVF_AND_HEADS = tuple((int(x) for x in os.environ.get('KT_NVF_AND_HEADS', '0,1').split(',') if x))
NVF_AND_T = tuple((float(x) for x in os.environ.get('KT_NVF_AND_T', '0.5').split(',') if x)) or (0.5,)
NVF_AND_PAIR = float(os.environ.get('KT_NVF_AND_PAIR', '3.0'))
NVF_AND_TOL = float(os.environ.get('KT_NVF_AND_TOL', '3.0'))
NVF_AND_N = max(1, int(os.environ.get('KT_NVF_AND_N', '3')))
NVF_AND_BASE = float(os.environ.get('KT_NVF_AND_BASE', '0.0'))
NVF_AND_SPAN = int(os.environ.get('KT_NVF_AND_SPAN', '0'))
NVF_AND_BOTH = os.environ.get('KT_NVF_AND_BOTH', '1') == '1'
NVF_AND_WIN = int(os.environ.get('KT_NVF_AND_WIN', '4'))

class _NVFHead(_NVHead):

    def __init__(self):
        self.pf = None
        self.n_err = 0
        self.ms = 0.0
        self.S = []
        _RgbPrimary.__init__(self, model=NVF_FILES[0] if NVF_FILES else 'none', term_r=RP_TERM_R)
        self.msess = None
        self._mtried = True
        self._load()

    def _load(self):
        self._tried = True
        self.S = []
        for i, f in enumerate(NVF_FILES):
            try:
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1
                s = ort.InferenceSession(str(_H / f), so, providers=['CPUExecutionProvider'])
                nm = [a.name for a in s.get_inputs()]
                on = s.get_outputs()[0].name
                s.run([on], {nm[0]: np.zeros((256, 256, 1), np.float32), nm[1]: _NV_ST})
                self.S.append((s, nm[0], nm[1], on, float(NVF_T[min(i, len(NVF_T) - 1)])))
            except Exception:
                pass
        self.sess = self.S[0][0] if self.S else None

    def reset(self):
        super().reset()
        self.k = 0
        self.t1 = -1
        self.log = []
        self.n_box = 0
        self.and_o = None
        self.and_hits = []
        self.and_last = {}
        self.n_and_ev = 0
        self.n_and_hit = 0
        self.n_and_lat = 0
        self.and_t1 = -1
        self.and_p1 = None
        self.and_ms = 0.0
        self.n_and_err = 0

    def expire(self, tick):
        while len(self.bad_t) < len(self.bad):
            self.bad_t.append(tick)
        while len(self.bad_s) < len(self.bad):
            self.bad_s.append(False)
        if self.bad and (NVF_BAN_T > 0 or NVF_SBAN_T > 0):
            k = [i for i, (t, s) in enumerate(zip(self.bad_t, self.bad_s)) if not (NVF_BAN_T > 0 and tick - t > NVF_BAN_T or (s and NVF_SBAN_T > 0 and (tick - t > NVF_SBAN_T)))]
            self.bad = [self.bad[i] for i in k]
            self.bad_t = [self.bad_t[i] for i in k]
            self.bad_s = [self.bad_s[i] for i in k]

    def observe_nvf(self, tick, obs, pos, rpy, p3):
        if tick < NVF_T0 or tick % NVF_EVERY or (not self.S):
            return
        s, iname, sname, oname, thr = self.S[self.k % len(self.S)]
        self.k += 1
        x = np.ascontiguousarray(np.asarray(obs['depth'], np.float32).reshape(256, 256, 1))
        t0 = _nv_time.perf_counter()
        o = np.asarray(s.run([oname], {iname: x, sname: _NV_ST})[0], np.float32).reshape(-1)
        self.ms += 1000.0 * (_nv_time.perf_counter() - t0)
        self.n_eval += 1
        if NVF_AND:
            self.and_o = ((self.k - 1) % len(self.S), o, x)
        if _sig(o[0]) < thr:
            return
        cz = math.exp(min(float(o[3]), 6.0))
        if cz > DH_RMAX:
            return
        q = _mworld(float(o[1]), float(o[2]), cz, pos, rpy)
        if not np.all(np.isfinite(q)):
            return
        if NVF_BOX > 0.0 and max(abs(float(q[0])), abs(float(q[1]))) > NVF_BOX:
            self.n_box += 1
            return
        if NVF_DC > 0.0:
            px = min(255, max(0, int(round((float(o[1]) + 1.0) * 0.5 * 256 - 0.5))))
            py = min(255, max(0, int(round((1.0 - float(o[2])) * 0.5 * 256 - 0.5))))
            dm = float(np.median(x[max(0, py - 2):py + 3, max(0, px - 2):px + 3, 0])) * (_DMAX - _DMIN) + _DMIN
            if abs(cz - dm) > NVF_DC:
                self.n_dc += 1
                return
        if self.clue0 is not None and float(np.linalg.norm(q[0:2] - self.clue0)) > NVF_CLUE_R:
            return
        if not -25.0 <= float(q[2] - pos[2]) <= NVF_ZUP:
            return
        if NVF_ZMAX > 0.0 and float(q[2]) > NVF_ZMAX:
            self.n_zmax += 1
            return
        for bb in self.bad:
            if float(np.linalg.norm(q[0:2] - bb)) < 6.0:
                return
        for bb in p3:
            if float(np.linalg.norm(q[0:2] - np.asarray(bb[0:2], np.float64))) < P3_BAN_R:
                return
        self.hit_t = tick
        self.n_fix += 1
        self.n_hit += 1
        if self.latch is not None:
            if float(np.linalg.norm(q[0:2] - self.latch[0:2])) <= 6.0:
                self.latch = (1.0 - RP_EMA) * self.latch + RP_EMA * q
                self.lh_t = tick
                self.st_n = 0
            return
        self.hits = [(t, z) for t, z in self.hits if tick - t <= 400] + [(tick, q)]
        self.obs_xy = [(t, p) for t, p in self.obs_xy if tick - t <= 400] + [(tick, np.asarray(pos[0:2], np.float64).copy())]
        if len(self.hits) >= NVF_N:
            pts = np.asarray([z for _, z in self.hits[-NVF_N:]])
            c = pts.mean(0)
            if float(np.max(np.linalg.norm(pts[:, 0:2] - c[0:2], axis=1))) <= NVF_TOL:
                ok = True
                if NVF_BASE > 0.0:
                    ts = {t for t, z in self.hits if float(np.linalg.norm(np.asarray(z, np.float64)[0:2] - c[0:2])) <= NVF_TOL}
                    vp = np.asarray([p for t, p in self.obs_xy if t in ts], np.float64)
                    ok = bool(len(vp) >= 2 and float(np.max(np.linalg.norm(vp - vp.mean(0), axis=1))) * 2.0 >= NVF_BASE)
                if ok:
                    self.latch = c
                    self.near_since = None
                    self.n_latch += 1
                    self.latch_t = tick
                    self.lh_t = tick
                    self.st_n = 0
                    self.nr_n = 0
                    self.lat_st = 0
                    self.pd_min = 1000000000.0
                    self.pd_n = 0
                    if self.t1 < 0:
                        self.t1 = tick
                    if NVF_LOG:
                        self.log.append(('latch', int(tick), [float(v) for v in c]))

    def _and_q(self, o, thr, x, pos, rpy, p3):
        if _sig(o[0]) < thr:
            return None
        cz = math.exp(min(float(o[3]), 6.0))
        if cz > DH_RMAX:
            return None
        q = _mworld(float(o[1]), float(o[2]), cz, pos, rpy)
        if not np.all(np.isfinite(q)):
            return None
        if NVF_BOX > 0.0 and max(abs(float(q[0])), abs(float(q[1]))) > NVF_BOX:
            return None
        if NVF_DC > 0.0:
            px = min(255, max(0, int(round((float(o[1]) + 1.0) * 0.5 * 256 - 0.5))))
            py = min(255, max(0, int(round((1.0 - float(o[2])) * 0.5 * 256 - 0.5))))
            dm = float(np.median(x[max(0, py - 2):py + 3, max(0, px - 2):px + 3, 0])) * (_DMAX - _DMIN) + _DMIN
            if abs(cz - dm) > NVF_DC:
                return None
        if self.clue0 is not None and float(np.linalg.norm(q[0:2] - self.clue0)) > NVF_CLUE_R:
            return None
        if not -25.0 <= float(q[2] - pos[2]) <= NVF_ZUP:
            return None
        if NVF_ZMAX > 0.0 and float(q[2]) > NVF_ZMAX:
            return None
        for bb in self.bad:
            if float(np.linalg.norm(q[0:2] - bb)) < 6.0:
                return None
        for bb in p3:
            if float(np.linalg.norm(q[0:2] - np.asarray(bb[0:2], np.float64))) < P3_BAN_R:
                return None
        return q

    def observe_and(self, tick, obs, pos, rpy, p3):
        st = self.and_o
        self.and_o = None
        if self.latch is not None:
            self.and_hits = []
            self.and_last = {}
            return
        if st is None or len(NVF_AND_HEADS) < 2:
            return
        hi, o, x = st
        ha, hb = (NVF_AND_HEADS[0], NVF_AND_HEADS[1])
        if hi not in (ha, hb) or max(ha, hb) >= len(self.S):
            return
        outs = {hi: o}
        if NVF_AND_BOTH:
            j = hb if hi == ha else ha
            s, iname, sname, oname, _thr = self.S[j]
            t0 = _nv_time.perf_counter()
            outs[j] = np.asarray(s.run([oname], {iname: x, sname: _NV_ST})[0], np.float32).reshape(-1)
            self.and_ms += 1000.0 * (_nv_time.perf_counter() - t0)
            self.n_and_ev += 1
        fq = {}
        for j, oj in outs.items():
            q = self._and_q(oj, NVF_AND_T[min(NVF_AND_HEADS.index(j), len(NVF_AND_T) - 1)], x, pos, rpy, p3)
            if q is not None:
                fq[j] = np.asarray(q, np.float64)
        z = None
        if NVF_AND_BOTH:
            if ha in fq and hb in fq and (float(np.linalg.norm(fq[ha][0:2] - fq[hb][0:2])) <= NVF_AND_PAIR):
                z = (fq[ha] + fq[hb]) / 2.0
        elif hi in fq:
            j = hb if hi == ha else ha
            lj = self.and_last.get(j)
            if lj is not None and tick - lj[0] <= NVF_AND_WIN and (float(np.linalg.norm(fq[hi][0:2] - lj[1][0:2])) <= NVF_AND_PAIR):
                z = (fq[hi] + lj[1]) / 2.0
            self.and_last[hi] = (tick, fq[hi])
        if z is None:
            return
        self.n_and_hit += 1
        self.and_hits = [(t, p, v) for t, p, v in self.and_hits if tick - t <= 400] + [(tick, z, np.asarray(pos[0:2], np.float64).copy())]
        if len(self.and_hits) < NVF_AND_N:
            return
        L = self.and_hits[-NVF_AND_N:]
        pts = np.asarray([p for _, p, _ in L])
        c = pts.mean(0)
        if float(np.max(np.linalg.norm(pts[:, 0:2] - c[0:2], axis=1))) > NVF_AND_TOL:
            return
        if NVF_AND_SPAN > 0:
            ts = [t for t, p, _ in self.and_hits if float(np.linalg.norm(p[0:2] - c[0:2])) <= NVF_AND_TOL]
            if max(ts) - min(ts) < NVF_AND_SPAN:
                return
        if NVF_AND_BASE > 0.0:
            vp = np.asarray([v for _, p, v in self.and_hits if float(np.linalg.norm(p[0:2] - c[0:2])) <= NVF_AND_TOL], np.float64)
            if not (len(vp) >= 2 and float(np.max(np.linalg.norm(vp - vp.mean(0), axis=1))) * 2.0 >= NVF_AND_BASE):
                return
        self.latch = c
        self.near_since = None
        self.n_latch += 1
        self.latch_t = tick
        self.lh_t = tick
        self.st_n = 0
        self.nr_n = 0
        self.lat_st = 0
        self.pd_min = 1000000000.0
        self.pd_n = 0
        if self.t1 < 0:
            self.t1 = tick
        self.n_and_lat += 1
        if self.and_t1 < 0:
            self.and_t1 = tick
            self.and_p1 = [round(float(c[0]), 2), round(float(c[1]), 2)]
        self.and_hits = []
        self.and_last = {}
        if NVF_LOG:
            self.log.append(('latch_and', int(tick), [float(v) for v in c]))

def _dfeat(depth):
    d = np.asarray(depth, np.float32)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        d = np.reshape(d, d.shape[:2])
    d = np.clip(d, 0.0, 1.0)
    gx = np.abs(np.diff(d, axis=1))
    gy = np.abs(np.diff(d, axis=0))
    pc = np.percentile(d, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    v = [float(d.min()), float(d.mean()), float(d.std()), float(d.max())]
    v += [float(p) for p in pc]
    v += [float((d <= 0.1).mean()), float((d <= 0.25).mean()), float((d >= 0.95).mean()), float((d >= 0.999).mean()), float((d <= 0.001).mean()), float(gx.mean() + gy.mean()), float(((gx > 0.05).mean() + (gy > 0.05).mean()) / 2.0)]
    h, w = d.shape
    tm, tn, tx = ([], [], [])
    for y0 in np.linspace(0, h, 5, dtype=int)[:-1]:
        y1 = int(y0 + h // 4)
        for x0 in np.linspace(0, w, 5, dtype=int)[:-1]:
            x1 = int(x0 + w // 4)
            t = d[y0:y1, x0:x1]
            tm.append(float(t.mean()))
            tn.append(float(t.min()))
            tx.append(float(t.max()))
    return v + tm + tn + tx

class _Net:

    def __init__(self, p):
        b = open(p, 'rb').read()
        d, h = (int(x) for x in np.frombuffer(b[:4], np.uint16))
        o = 4
        n1 = d * h
        n2 = h * 6
        self.w1 = np.frombuffer(b[o:o + n1], np.int8).reshape(d, h).astype(np.float32)
        o += n1
        self.w2 = np.frombuffer(b[o:o + n2], np.int8).reshape(h, 6).astype(np.float32)
        o += n2
        self.s1, self.s2 = np.frombuffer(b[o:o + 8], np.float32)
        o += 8
        self.b1 = np.frombuffer(b[o:o + 2 * h], np.float16).astype(np.float32)
        o += 2 * h
        self.b2 = np.frombuffer(b[o:o + 12], np.float16).astype(np.float32)
        o += 12
        self.mu = np.frombuffer(b[o:o + 2 * d], np.float16).astype(np.float32)
        o += 2 * d
        self.sd = np.frombuffer(b[o:o + 2 * d], np.float16).astype(np.float32)
        self.w1 *= self.s1
        self.w2 *= self.s2

    def probs(self, f):
        x = (f - self.mu) / self.sd
        hd = np.maximum(x @ self.w1 + self.b1, 0)
        z = hd @ self.w2 + self.b2
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

class _Cls:

    def __init__(self):
        self.net = _Net(_H / 'typenet.bin')
        self.reset()

    def reset(self):
        self.ps = np.zeros(6)
        self.n = 0
        self.latched = set()
        self.is_mountain = False
        self.p40 = None
        self.relatched = set()
        self.z0_mtn = False

    def update(self, tick, state, depth):
        if tick > 600 or tick % 5:
            return
        s = np.asarray(state, np.float32).reshape(-1)
        sf = np.zeros(141, np.float32)
        sf[:min(141, s.size)] = s[:141]
        f = np.asarray([float(tick), float(tick) * SIM_DT] + sf.tolist() + _dfeat(depth), np.float32)
        self.ps += self.net.probs(f)
        self.n += 1
        avg = self.ps / self.n
        if self.n == 40:
            self.p40 = float(avg[2])
        i = int(np.argmax(avg))
        lab = LB[i]
        bar = 0.5 if lab == 'warehouse' else 0.7
        if self.n >= 3 and avg[i] >= bar:
            self.latched.add(lab)
            if lab == 'mountain':
                self.is_mountain = True
        if self.n >= 40 and avg[4] >= 0.15:
            self.latched.add('warehouse')
        if RL_ON and self.n >= RL_N and (lab in RL_TYPES) and (avg[i] >= RL_BAR):
            self.relatched.add(lab)

    def settled(self, t):
        return t in self.latched

    def mtn_ok(self):
        if E2A_ON and self.is_mountain and self.z0_mtn:
            return True
        return bool(self.is_mountain and self.p40 is not None and (self.p40 >= MTN_THR))

def _warm(rgb):
    im = np.asarray(rgb, np.float32).reshape(256, 256, 3)[128:]
    r, g, b = (im[..., 0], im[..., 1], im[..., 2])
    return float(((r > g + 0.03) & (g >= b)).mean())
_BOX = np.array([25.0, 30.0, 30.0, 25.0, 12.0, 20.0])
CVR = 9.0

def _ccore(origin, pad, rpad=83.0, boxes=None):
    xs = np.arange(-48.0, 50.0, 2.0)
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    dc = np.linalg.norm(pts - origin, axis=1)
    dp = np.linalg.norm(pts - pad, axis=1)
    m = (dc <= 33.0) & (dp <= rpad)
    pts, dc, dp = (pts[m], dc[m], dp[m])
    ax = np.abs(pts).max(1)
    dens = np.zeros(len(pts))
    bx = _BOX if boxes is None else boxes
    for bb in bx:
        dens[ax <= bb] += 1.0 / (6.0 * (2 * bb) ** 2) if boxes is None else 1.0 / (len(bx) * (2 * bb) ** 2)
    w = dens / (1 + np.exp(-(30.0 - dc))) / (1 + np.exp(-(rpad - 3.0 - dp)))
    return (pts, w, dc)

def _bous(core, lane, spacing, cap, pos):
    core = np.asarray(core, np.float64)
    if core.shape[0] < 4:
        return []
    k = np.round(core[:, 0] / lane).astype(int)
    o = np.lexsort((np.where(k % 2 == 0, core[:, 1], -core[:, 1]), k))
    path = core[o]
    sel = [path[0]]
    for q in path[1:]:
        if float(np.linalg.norm(q - sel[-1])) >= spacing:
            sel.append(q)
    w = sel[:cap]
    if len(w) >= 2 and np.linalg.norm(w[-1] - pos) < np.linalg.norm(w[0] - pos):
        w = w[::-1]
    return [np.asarray(x, np.float64) for x in w]

def _bous_lanes(core, lane, spacing, cap, pos):
    core = np.asarray(core, np.float64)
    if core.shape[0] < 4:
        return []
    k = np.round(core[:, 0] / lane).astype(int)
    ks = sorted(set(k.tolist()))
    pos = np.asarray(pos, np.float64)[0:2]
    best = None
    for order in (ks, ks[::-1]):
        for first_up in (True, False):
            wps = []
            for n, kk in enumerate(order):
                m = k == kk
                y0, y1 = (float(core[m, 1].min()), float(core[m, 1].max()))
                x = float(np.clip(kk * lane, core[m, 0].min(), core[m, 0].max()))
                cnt = max(1, int((y1 - y0) // spacing) + 1)
                yy = np.linspace(y0, y1, cnt) if cnt > 1 else np.array([(y0 + y1) / 2.0])
                if (n % 2 == 0) != first_up:
                    yy = yy[::-1]
                wps += [np.array([x, y]) for y in yy]
            d0 = float(np.linalg.norm(wps[0] - pos))
            if best is None or d0 < best[0]:
                best = (d0, wps)
    return [np.asarray(x, np.float64) for x in best[1][:cap]]

def _mpost(origin, pad, q=MP_Q, rv=MP_RV):
    xs = np.arange(-48.0, 50.0, 2.0)
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    dc = np.linalg.norm(pts - origin, axis=1)
    dp = np.linalg.norm(pts - pad, axis=1)
    ax = np.abs(pts).max(1)
    i30 = ax <= 30.0
    i45 = ax <= 45.0
    inr = dp <= rv
    n30 = float((i30 & inr).sum())
    n45 = float((i45 & inr).sum())
    f1 = (1.0 - q * n30 * 4.0 / 3600.0) ** 100
    f2 = (1.0 - q * n45 * 4.0 / 8100.0) ** 50
    dens = np.zeros(len(pts))
    if n30 > 0:
        dens[i30 & inr] += (1.0 - f1) / n30
    if n45 > 0:
        dens[i45 & inr] += f1 * (1.0 - f2) / n45
    lam = q * 4.0 * (100.0 * i30 / 3600.0 + 50.0 * i45 / 8100.0) * ~inr
    o = np.argsort(dp, kind='stable')
    lo = lam[o]
    fb = np.zeros(len(pts))
    fb[o] = lo * np.exp(-(np.cumsum(lo) - lo))
    dens += f1 * f2 * fb
    m = dc <= 33.0
    w = dens / (1 + np.exp(-(30.0 - dc)))
    return (pts[m], w[m], dc[m])
_GK = [(a, b) for a in range(-4, 5) for b in range(-4, 5) if 4.0 * (a * a + b * b) <= MP_RS * MP_RS]

def _gtour(sup, mass, pos, hd, cap, tail=None):
    sup = np.asarray(sup, np.float64)
    m = np.asarray(mass, np.float64).copy()
    t = float(m.sum())
    tl = np.asarray(tail, np.float64).copy() if tail is not None else None
    if t <= 0 or not len(sup):
        return []
    ij = np.clip(np.rint((sup + 48.0) / 2.0).astype(int) + 4, 4, 52)
    cur = np.asarray(pos, np.float64)[0:2].copy()
    wps = []
    for _ in range(cap):
        if float(m.sum()) < MP_STOP * t:
            if tl is None or float(tl.sum()) <= 0:
                break
            m, tl = (tl, None)
            t = float(m.sum())
        M = np.zeros((57, 57))
        M[ij[:, 1], ij[:, 0]] = m
        G = np.zeros((57, 57))
        for a, b in _GK:
            G[4:53, 4:53] += M[4 + b:53 + b, 4 + a:53 + a]
        g = G[ij[:, 1], ij[:, 0]]
        dv = sup - cur
        dist = np.linalg.norm(dv, axis=1)
        if MP_DMIN > 0.0:
            g = np.where(dist < MP_DMIN, 0.0, g)
        ang = np.abs((np.arctan2(dv[:, 1], dv[:, 0]) - hd + math.pi) % (2 * math.pi) - math.pi)
        j = int(np.argmax(g / (dist / MP_V + MP_TP * ang / math.pi + MP_C0)))
        if g[j] <= 0:
            if tl is None or float(tl.sum()) <= 0:
                break
            m, tl = (tl, None)
            t = float(m.sum())
            continue
        _sw = np.linalg.norm(sup - sup[j], axis=1) <= MP_RS
        wps.append(sup[j].copy())
        m[_sw] = 0.0
        if tl is not None:
            tl[_sw] = 0.0
        if dist[j] > 1e-06:
            hd = math.atan2(float(dv[j, 1]), float(dv[j, 0]))
        cur = sup[j].copy()
    return wps

def _wps(origin, pad, pos, lane, spacing, cap):
    xs = np.arange(-48.0, 50.0, 2.0)
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    dc = np.linalg.norm(pts - origin, axis=1)
    dp = np.linalg.norm(pts - pad, axis=1)
    m = (dc <= 33.0) & (dp <= 83.0)
    pts, dc, dp = (pts[m], dc[m], dp[m])
    ax = np.abs(pts).max(1)
    dens = np.zeros(len(pts))
    for bb in _BOX:
        dens[ax <= bb] += 1.0 / (6.0 * (2 * bb) ** 2)
    w = dens / (1 + np.exp(-(30.0 - dc))) / (1 + np.exp(-(80.0 - dp)))
    core = pts
    if w.sum() > 0:
        o = np.argsort(-w)
        cw = np.cumsum(w[o]) / w.sum()
        core = pts[o[:int(np.searchsorted(cw, 0.9)) + 1]]
    if core.shape[0] < 4:
        core = pts[dc <= 30.0]
    if not core.shape[0]:
        return []
    k = np.round(core[:, 0] / lane).astype(int)
    o = np.lexsort((np.where(k % 2 == 0, core[:, 1], -core[:, 1]), k))
    path = core[o]
    sel = [path[0]]
    for q in path[1:]:
        if float(np.linalg.norm(q - sel[-1])) >= spacing:
            sel.append(q)
    w = sel[:cap]
    if len(w) >= 2 and np.linalg.norm(w[-1] - pos) < np.linalg.norm(w[0] - pos):
        w = w[::-1]
    return [np.asarray(x, np.float64) for x in w]

def _tk_pts(depth, pos, rpy, step=4, r0=0, r1=256):
    fwd, up, right = _axes(rpy)
    cam = np.asarray(pos, np.float64) + fwd * _CAMF + up * _CAMU
    rows = np.arange(r0, r1, step)
    cols = np.arange(0, 256, step)
    dm = np.asarray(depth, np.float32).reshape(256, 256)[np.ix_(rows, cols)].astype(np.float64) * (_DMAX - _DMIN) + _DMIN
    u = (cols + 0.5 - 128.0) / 128.0
    v = (128.0 - rows - 0.5) / 128.0
    ray = fwd[None, None, :] + u[None, :, None] * right[None, None, :] + v[:, None, None] * up[None, None, :]
    P = cam[None, None, :] + dm[..., None] * ray
    return (P, dm)

def _tk_free(depth, pos, rpy, head, half_w, z_lo, z_hi, min_al=0.0):
    P, dm = _tk_pts(depth, pos, rpy, 4, 0, 256)
    P = P.reshape(-1, 3)
    dm = dm.reshape(-1)
    ok = (dm < _DMAX - 0.2) & (P[:, 2] >= z_lo) & (P[:, 2] <= z_hi)
    if not ok.any():
        return 30.0
    q = P[ok, 0:2] - np.asarray(pos[0:2], np.float64)
    h = np.array([math.cos(head), math.sin(head)])
    al = q @ h
    lat = np.abs(q[:, 0] * h[1] - q[:, 1] * h[0])
    m = (al > min_al) & (lat <= half_w)
    return float(al[m].min()) if m.any() else 30.0

def _tk_dmax(depth, pos, rpy, head, z0):
    f_lo = _tk_free(depth, pos, rpy, head, TK_HW, z0 - 0.09, z0 + 0.4, 0.0)
    f_hi = _tk_free(depth, pos, rpy, head, TK_HW, z0 + 0.4, z0 + 1.1, 0.75)
    return min(f_lo, f_hi) - TK_WALL

def _tk_plates(depth, pos, rpy, z0):
    P, dm = _tk_pts(depth, pos, rpy, 4, 0, 64)
    z = float(pos[2])
    r = np.hypot(P[..., 0] - pos[0], P[..., 1] - pos[1])
    zz = P[..., 2]
    a, b = (slice(0, 15), slice(1, 16))
    okp = (dm[a] > 0.55) & (dm[b] > 0.55) & (dm[a] < _DMAX - 0.5) & (dm[b] < _DMAX - 0.5)
    flat = np.abs(zz[b] - zz[a]) < 0.04 + 0.015 * dm[a]
    outw = r[b] - r[a] > 0.02 + 0.01 * dm[a]
    zp = zz[a]
    hit = okp & flat & outw & (zp >= z0 + TK_PZ0) & (zp <= z0 + TK_PZ1) & (zp > z + 0.3)
    if not hit.any():
        return (None, None)
    q = P[a][hit]
    zq = zp[hit]
    rq = r[a][hit]
    reach = rq <= zq - (z + _CAMU) + TK_REACH
    return (q, float(zq[reach].min()) if reach.any() else None)
WL_ON = True
WL_ZLO = 0.35
WL_ZHI = 1.6
WL_DMIN = 0.12
WL_F0 = 0.35
WL_BIN = 0.2
WL_NBIN = 7
WL_NB = 3
WL_LMAX = 1.6
WL_TGT = 0.6
WL_MIN = 0.15
WL_MAXS = 0.6

def _wl_faces(depth, pos, rpy, z0):
    P, dm = _tk_pts(depth, pos, rpy, 4, 0, 256)
    P = P.reshape(-1, 3)
    dm = dm.reshape(-1)
    y = float(rpy[2])
    f = np.array([math.cos(y), math.sin(y)])
    r = np.array([math.sin(y), -math.cos(y)])
    q = P[:, 0:2] - np.asarray(pos[0:2], np.float64)
    fw = q @ f
    lt = q @ r
    ok = (dm > _DMIN + 0.01) & (dm < _DMAX - 0.2) & (P[:, 2] >= z0 + WL_ZLO) & (P[:, 2] <= z0 + WL_ZHI) & (fw >= WL_F0) & (fw < WL_F0 + WL_BIN * WL_NBIN) & (np.abs(lt) <= WL_LMAX)
    out = []
    for s in (-1.0, 1.0):
        m = ok & (lt * s > 0.0)
        if not m.any():
            out.append(float('inf'))
            continue
        b = np.floor((fw[m] - WL_F0) / WL_BIN).astype(int)
        a = np.abs(lt[m])
        mins = [float(a[b == k].min()) for k in range(WL_NBIN) if (b == k).any()]
        _d = float(np.median(mins)) if len(mins) >= WL_NB else float('inf')
        out.append(_d if _d >= WL_DMIN else float('inf'))
    return (out[0], out[1])
WX_ON = True
WX_PLAN = True
WX_P0CAP = 0.8
WX_CR = 0.9
WX_CEILP = 0.7
WX_STEP = 3
WX_RMAX = 0.6
WX_LAM = 0.1
WX_DELTA = 0.1
WX_SEC0 = -45.0
WX_SEC1 = 135.0
WX_V = 0.3
WX_VZ = 0.6
WX_ARR = 0.04
WX_EXIT = 1.1
WX_ZEX = 1.0
WX_KXY = 2.0
WX_CREEP = True
WX_CEIL = True
WX_RS = (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
WHD_ON = os.environ.get('KT_WHD', '0') == '1'
WHD_ZMIN = float(os.environ.get('KT_WHD_ZMIN', '5.0'))
WHD_DZLO = float(os.environ.get('KT_WHD_DZLO', '0.35'))
WHD_DZHI = float(os.environ.get('KT_WHD_DZHI', '0.2'))
WHD_LOOK = float(os.environ.get('KT_WHD_LOOK', '4.5'))
WHD_R = float(os.environ.get('KT_WHD_R', '1.05'))
WHD_VMIN = float(os.environ.get('KT_WHD_VMIN', '1.0'))
WHD_TGT = float(os.environ.get('KT_WHD_TGT', '2.5'))
WHD_MAXA = float(os.environ.get('KT_WHD_MAXA', '35'))
WHD_STEP = float(os.environ.get('KT_WHD_STEP', '5'))
WHD_RATE = float(os.environ.get('KT_WHD_RATE', '4'))
WHD_MEM = int(os.environ.get('KT_WHD_MEM', '75'))
WHD_TALL = float(os.environ.get('KT_WHD_TALL', '0'))
WHD_LOG = os.environ.get('KT_WHD_LOG', '0') == '1'
WHG_ON = os.environ.get('KT_WHG', '0') == '1'
WHG_DMIN = float(os.environ.get('KT_WHG_DMIN', '0.55'))
WHG_TAU = float(os.environ.get('KT_WHG_TAU', '0.25'))
WHG_BODY = float(os.environ.get('KT_WHG_BODY', '0.06'))
WHG_ZB = float(os.environ.get('KT_WHG_ZB', '0.037'))
WHG_M = float(os.environ.get('KT_WHG_M', '0.0'))
WHG_HEAD = float(os.environ.get('KT_WHG_HEAD', '12'))
WHG_EVERY = max(1, int(os.environ.get('KT_WHG_EVERY', '2')))
WXF_ON = os.environ.get('KT_WXF', '0') == '1'
WXF_REAR = os.environ.get('KT_WXF_REAR', '1') == '1'
WXF_TOL = float(os.environ.get('KT_WXF_TOL', '4'))
WXF_LEAD = float(os.environ.get('KT_WXF_LEAD', '90'))
WXF_TMAX = int(os.environ.get('KT_WXF_TMAX', '250'))
WXF_RS = tuple((float(x) for x in os.environ.get('KT_WXF_RS', '0.15,0.3,0.45,0.6').split(',') if x))
WXF_CAP = float(os.environ.get('KT_WXF_CAP', '1.05'))
WXF_DELTA = float(os.environ.get('KT_WXF_DELTA', '0.05'))
WXF_LAM = float(os.environ.get('KT_WXF_LAM', '0.1'))
WXF_RB = float(os.environ.get('KT_WXF_RB', '0.25'))
WXF_BODY = float(os.environ.get('KT_WXF_BODY', '0.06'))
WXF_ZFL = float(os.environ.get('KT_WXF_ZFL', '0.0325'))
WXF_TOPR = int(os.environ.get('KT_WXF_TOPR', '2'))
WXF_V = float(os.environ.get('KT_WXF_V', '0.5'))
WXF_ZEX = float(os.environ.get('KT_WXF_ZEX', '1.6'))
WXF_ARR = float(os.environ.get('KT_WXF_ARR', '0.1'))
WXF_XT = 500
WXF_VOX = 0.04
WXF_H = 2.5
WXF_CLS = os.environ.get('KT_WXF_CLS', '1') == '1'

def _wxf_frame(depth, pos, rpy, pad, z0):
    fwd, up, right = _axes(rpy)
    cam = np.asarray(pos, np.float64) + fwd * _CAMF + up * _CAMU
    rows = np.arange(0, 256, 4)
    cols = np.arange(0, 256, 4)
    dm = np.asarray(depth, np.float32).reshape(256, 256)[np.ix_(rows, cols)].astype(np.float64) * (_DMAX - _DMIN) + _DMIN
    u = (cols + 0.5 - 128.0) / 128.0
    v = (128.0 - rows - 0.5) / 128.0
    ray = fwd[None, None, :] + u[None, :, None] * right[None, None, :] + v[:, None, None] * up[None, None, :]
    top = np.zeros(dm.shape, bool)
    top[0:max(0, WXF_TOPR)] = True
    ok = dm < _DMAX - 0.2
    Ps = [(cam[None, None, :] + dm[..., None] * ray)[ok]]
    Ts = [top[ok]]
    cl = dm <= _DMIN + 0.005
    rz = ray[..., 2]
    zf = z0 - 0.091
    tf = np.where(rz < -1e-06, (cam[2] - zf) / np.maximum(-rz, 1e-06), 99.0)
    cl &= tf > _DMIN + 0.02
    if cl.any():
        for s in (0.2, 0.3, 0.4):
            Ps.append((cam[None, None, :] + s * ray)[cl])
            Ts.append(top[cl])
    P = np.concatenate(Ps, 0)
    tp = np.concatenate(Ts, 0)
    q = P[:, 0:2] - np.asarray(pad, np.float64)[None, :]
    h = np.hypot(q[:, 0], q[:, 1])
    m = (h < WXF_H) & (P[:, 2] > 0.1) & (P[:, 2] <= z0 + 4.6)
    return (np.column_stack([q[m], P[m, 2]]), q[m & tp & (P[:, 2] >= z0 + 0.35)])

def _wxf_dedup(Q):
    if len(Q) < 2:
        return Q
    K = np.round(np.asarray(Q, np.float64) / WXF_VOX).astype(np.int64) + 4096
    k = K[:, 0] * 67108864 + K[:, 1] * 8192
    if K.shape[1] > 2:
        k = k + K[:, 2]
    return Q[np.unique(k, return_index=True)[1]]

def _wxf_eval(Q, Qt, tx, ty, r, z0):
    ze = z0 + math.sqrt(max(0.0, 1.0 - r * r))
    c = ze - WXF_ZFL
    if len(Q):
        dh = np.hypot(Q[:, 0] - tx, Q[:, 1] - ty)
        if bool(((dh < WXF_RB) & (Q[:, 2] >= z0 + 0.05)).any()):
            return (0.0, True)
        c = min(c, float(np.min(np.sqrt(dh * dh + np.maximum(0.0, ze - Q[:, 2]) ** 2))) - WXF_BODY)
    if len(Qt):
        c = min(c, float(np.min(np.hypot(Qt[:, 0] - tx, Qt[:, 1] - ty))) - WXF_BODY)
    return (c, False)

def _wxf_path(Q, ux, uy, r, z0):
    if not len(Q):
        return False
    q = Q[(Q[:, 2] >= z0 - 0.06) & (Q[:, 2] <= z0 + 0.4)]
    if not len(q):
        return False
    al = q[:, 0] * ux + q[:, 1] * uy
    lat = np.abs(q[:, 0] * uy - q[:, 1] * ux)
    return bool(((al > 0.0) & (al < r + 0.2) & (lat < 0.2)).any())
WHK_ON = os.environ.get('KT_WHK', '0') == '1'
WHK_R = float(os.environ.get('KT_WHK_R', '1.2'))
WHK_V = float(os.environ.get('KT_WHK_V', '0.5'))
WHK_MEM = int(os.environ.get('KT_WHK_MEM', '60'))
WHK_EVERY = int(os.environ.get('KT_WHK_EVERY', '2'))
WHK_RNG = float(os.environ.get('KT_WHK_RNG', '4.0'))
WHK_DZ = float(os.environ.get('KT_WHK_DZ', '0.5'))
WHK_PH = float(os.environ.get('KT_WHK_PH', '1.5'))
WHK_CLIMB = os.environ.get('KT_WHK_CLIMB', '0') == '1'
WHK_ZC = float(os.environ.get('KT_WHK_ZC', '1.8'))
WCR_ON = os.environ.get('KT_WCR', '0') == '1'
WCR_RIBS = tuple((float(x) for x in os.environ.get('KT_WCR_RIBS', '-42.55,-21.275,0.0,21.275,42.55').split(',') if x))
WCR_BAND = float(os.environ.get('KT_WCR_BAND', '0.6'))
WCR_OUT = float(os.environ.get('KT_WCR_OUT', '1.0'))
WCR_ZL = float(os.environ.get('KT_WCR_ZL', '2.0'))
WCR_ZMAX = float(os.environ.get('KT_WCR_ZMAX', '2.4'))
WCR_V = float(os.environ.get('KT_WCR_V', '0.8'))
WCR_TMAX = int(os.environ.get('KT_WCR_TMAX', '500'))
WCR_HW = float(os.environ.get('KT_WCR_HW', '0.3'))
WCR_DZ = float(os.environ.get('KT_WCR_DZ', '0.3'))
WCR_DZU = float(os.environ.get('KT_WCR_DZU', '0.5'))
WCR_ZTOP = float(os.environ.get('KT_WCR_ZTOP', '2.85'))
WCR_M = float(os.environ.get('KT_WCR_M', '0.5'))
WCR_MG = float(os.environ.get('KT_WCR_MG', '0.3'))
WCR_YAWT = float(os.environ.get('KT_WCR_YAWT', '12'))
WCR_Y = os.environ.get('KT_WCR_Y', '0') == '1'
WCR_YB = float(os.environ.get('KT_WCR_YB', '1.0'))
WCR_YS = tuple((tuple((float(v) for v in s.split(':'))) for s in os.environ.get('KT_WCR_YS', '-15.4:-12.7,-8.4:-7.6,-4.1:-4.0,4.0:4.1,7.6:8.4,12.7:15.4').split(',') if s))

def _wcr_ydist(y):
    return min((0.0 if lo <= y <= hi else min(abs(y - lo), abs(y - hi)) for lo, hi in WCR_YS)) if WCR_YS else 0.0

def _whk_pts(depth, pos, rpy):
    fwd, up, right = _axes(rpy)
    cam = np.asarray(pos, np.float64) + fwd * _CAMF + up * _CAMU
    rows = np.arange(2, 256, 4)
    cols = np.arange(2, 256, 4)
    dm = np.asarray(depth, np.float32).reshape(256, 256)[np.ix_(rows, cols)].astype(np.float64) * (_DMAX - _DMIN) + _DMIN
    u = (cols + 0.5 - 128.0) / 128.0
    v = (128.0 - rows - 0.5) / 128.0
    ray = fwd[None, None, :] + u[None, :, None] * right[None, None, :] + v[:, None, None] * up[None, None, :]
    ok = (dm * np.linalg.norm(ray, axis=2) < WHK_RNG) & (dm < _DMAX - 0.2)
    P = (cam[None, None, :] + dm[..., None] * ray)[ok]
    return P[np.abs(P[:, 2] - float(pos[2])) <= WHK_DZ]

def _wx_frame(depth, pos, rpy, pad, z0):
    fwd, up, right = _axes(rpy)
    cam = np.asarray(pos, np.float64) + fwd * _CAMF + up * _CAMU
    rows = np.arange(0, 256, 4)
    cols = np.arange(0, 256, 4)
    dm = np.asarray(depth, np.float32).reshape(256, 256)[np.ix_(rows, cols)].astype(np.float64) * (_DMAX - _DMIN) + _DMIN
    u = (cols + 0.5 - 128.0) / 128.0
    v = (128.0 - rows - 0.5) / 128.0
    ray = fwd[None, None, :] + u[None, :, None] * right[None, None, :] + v[:, None, None] * up[None, None, :]
    P = [(cam[None, None, :] + dm[..., None] * ray)[dm < _DMAX - 0.2]]
    cl = dm <= _DMIN + 0.005
    rz = ray[..., 2]
    zf = z0 - 0.091
    tf = np.where(rz < -1e-06, (cam[2] - zf) / np.maximum(-rz, 1e-06), 99.0)
    cl &= tf > _DMIN + 0.02
    if cl.any():
        for s in (0.2, 0.3, 0.4):
            P.append((cam[None, None, :] + s * ray)[cl])
    P = np.concatenate(P, 0)
    q = P[:, 0:2] - np.asarray(pad, np.float64)[None, :]
    h = np.hypot(q[:, 0], q[:, 1])
    hi = (P[:, 2] >= z0 + 0.35) & (P[:, 2] <= z0 + 4.5) & (h < 2.2)
    lo = (P[:, 2] >= z0 - 0.12) & (P[:, 2] < z0 + 0.35) & (h < 1.5)
    return (q[hi], q[lo])

def _wx_pc(Qh, Ql, r, az):
    u = np.array([math.cos(az), math.sin(az)])
    T = u * r
    pc = float(np.min(np.hypot(Qh[:, 0] - T[0], Qh[:, 1] - T[1]))) if len(Qh) else 9.0
    blk = False
    if r > 0 and len(Ql):
        al = Ql @ u
        lat = np.abs(Ql[:, 0] * u[1] - Ql[:, 1] * u[0])
        blk = bool(((al > 0.0) & (al < r + 0.2) & (lat < 0.2)).any())
    return (min(1.0, pc), blk)

class _Shield:
    NX = 160
    NZ = 96

    def __init__(self):
        c = (np.arange(64) * 4 + 2.0 - 128.0) / 128.0
        self.U = np.tile(c[None, :], (64, 1))
        self.V = np.tile(-c[:, None], (1, 64))
        self.N = np.array([self.NX, self.NX, self.NZ], np.int64)
        self.half = self.N // 2
        self.thr = np.array([int(3.0 / SH_VOX), int(3.0 / SH_VOX), int(2.0 / SH_VOX)], np.int64)
        self.log = []
        self.reset()

    def reset(self):
        if SH_LOG and getattr(self, 'n_t', 0) > 0:
            self.log.append({'ep_end': 1, 'see_ms': round(self.t_see, 1), 'filt_ms': round(self.t_ms, 1), 'n': self.n_t, 'filt_max': round(self.t_max, 2), 'n_eng': self.n_eng, 'n_dlk': self.n_dlk, 'n_rlx': self.n_rlx})
        self.occ = np.zeros((self.NX, self.NX, self.NZ), bool)
        self.org = None
        self.stuck = 0
        self.relax = 0.0
        self.n_rlx = 0
        self.edir = None
        self.unb = 0
        self.elat = 0
        self.vlast = None
        self.n_eng = 0
        self.n_dlk = 0
        self.blk = 0
        self.t_ms = 0.0
        self.t_max = 0.0
        self.n_t = 0
        self.t_see = 0.0
        self.flush()
        if SH_LOG:
            self.log.append({'reset': 1})
            self.flush()

    def flush(self):
        if SH_LOG and self.log:
            try:
                import json as _json
                with open(SH_LOG, 'a') as f:
                    for r in self.log:
                        f.write(_json.dumps(r) + '\n')
            except Exception:
                pass
        self.log = []

    def _recenter(self, pos):
        new = np.floor(pos / SH_VOX).astype(np.int64) - self.half
        if self.org is not None:
            sh = new - self.org
            klo = np.maximum(0, -sh)
            khi = np.minimum(self.N, self.N - sh)
            occ = np.zeros_like(self.occ)
            if np.all(khi > klo):
                occ[klo[0]:khi[0], klo[1]:khi[1], klo[2]:khi[2]] = self.occ[klo[0] + sh[0]:khi[0] + sh[0], klo[1] + sh[1]:khi[1] + sh[1], klo[2] + sh[2]:khi[2] + sh[2]]
            self.occ = occ
        self.org = new

    def _put(self, P):
        k = np.floor(P / SH_VOX).astype(np.int64) - self.org
        ok = np.all((k >= 0) & (k < self.N), axis=1)
        if ok.any():
            k = k[ok]
            self.occ[k[:, 0], k[:, 1], k[:, 2]] = True

    def see(self, depth, st):
        import time as _time
        t0 = _time.perf_counter()
        self._see(depth, st)
        self.t_see += (_time.perf_counter() - t0) * 1000.0

    def _see(self, depth, st):
        pos = st[0:3].astype(np.float64)
        if self.org is None or np.any(np.abs(np.floor(pos / SH_VOX).astype(np.int64) - self.org - self.half) > self.thr):
            self._recenter(pos)
        m = np.asarray(depth, np.float32).reshape(64, 4, 64, 4).min(axis=(1, 3))
        z = m.astype(np.float64) * (_DMAX - _DMIN) + _DMIN
        sel = (m < 0.999) & (z <= SH_RI)
        if sel.any():
            fwd, up, right = _axes(st[3:6])
            cam = pos + fwd * _CAMF + up * _CAMU
            zs = z[sel][:, None]
            self._put(cam[None, :] + zs * (fwd[None, :] + self.U[sel][:, None] * right[None, :] + self.V[sel][:, None] * up[None, :]))
        if st.size > 162:
            agl = float(st[162]) * 20.0
            if agl < 19.5:
                self._put(np.array([[pos[0], pos[1], pos[2] - agl]]))

    def near(self, pos, R):
        c = np.floor(pos / SH_VOX).astype(np.int64) - self.org
        r = int(math.ceil(R / SH_VOX)) + 1
        lo = np.maximum(c - r, 0)
        hi = np.minimum(c + r + 1, self.N)
        if np.any(hi <= lo):
            return None
        ii = np.nonzero(self.occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        if ii[0].size == 0:
            return None
        D = (np.stack(ii, 1) + lo + self.org + 0.5) * SH_VOX - pos[None, :]
        dist = np.sqrt((D * D).sum(1))
        k = dist <= R
        if not k.any():
            return None
        return (D[k], dist[k])

    @staticmethod
    def _proj(v, n, lim, it=12):
        v = v.copy()
        w = 0.0
        for _ in range(it):
            viol = n @ v - lim
            j = int(np.argmax(viol))
            w = float(viol[j])
            if w <= 0.001:
                break
            v = v - w * n[j]
        return (v, w)

    def filt(self, out, st, ds, tick, info):
        o = np.asarray(out, np.float32).reshape(-1)
        dv = o[0:3].astype(np.float64)
        nd = float(np.linalg.norm(dv))
        vd = dv / nd * 3.0 * abs(float(o[3])) if nd > 1e-09 else np.zeros(3)
        pos = st[0:3].astype(np.float64)
        vs = vd
        dmin = 9.0
        nv = 0
        w = 0.0
        esc = 0
        bdir = False
        ds = min(ds, SH_DSMIN) + (ds - min(ds, SH_DSMIN)) * (1.0 - self.relax)
        nvd = float(np.linalg.norm(vd))
        vact = float(np.linalg.norm(st[6:9]))
        r = self.near(pos, SH_RQ)
        if r is not None:
            D, dist = r
            nv = int(dist.size)
            dmin = float(dist.min()) - SH_RB
            n = D / np.maximum(dist, 1e-06)[:, None]
            h = dist - SH_RB - ds
            acc = SH_ACC + (SH_ACCV - SH_ACC) * np.abs(n[:, 2])
            at = acc * SH_TAU
            lim = np.where(h >= 0.0, -at + np.sqrt(at * at + 2.0 * acc * np.maximum(h, 0.0)), np.maximum(-SH_VMAXP, SH_PUSH * h))
            lim0 = np.maximum(lim, 0.0)
            vs, w = self._proj(vd, n, lim)
            if w > 0.05:
                vs, w = self._proj(vd, n, lim0)
            if nvd > 0.02:
                u = vd / nvd
                vt, _w = self._proj(u * max(1.0, nvd), n, lim0)
                bdir = bool(float(vt @ u) < 0.5 * max(1.0, nvd))
            if bdir and vact < 0.5:
                self.stuck += 1
            elif not bdir:
                self.stuck = max(0, self.stuck - 2)
            if bdir:
                self.unb = 0
            else:
                self.unb += 1
            self.elat = max(0, self.elat - 1)
            if self.edir is not None and self.elat <= 0 and (self.unb > SH_EHOLD):
                self.edir = None
            if SH_DLK > 0 and nvd > 0.02 and (not info.get('nr', 0)) and (self.edir is not None or self.stuck >= SH_DLK):
                if self.edir is None:
                    self.edir = self._pick(vd / nvd, D, dist, ds)
                    if self.edir is not None:
                        self.n_dlk += 1
                        self.elat = SH_ELAT
                if self.edir is not None:
                    v2, w2 = self._proj(vs + (SH_VESC - float(vs @ self.edir)) * self.edir, n, lim0)
                    if w2 <= 0.05 and float(v2 @ self.edir) > 0.15:
                        vs = v2
                        esc = 1
                    else:
                        self.edir = None
                        self.elat = 0
            sp = float(np.linalg.norm(vs))
            if sp > 3.0:
                vs = vs * (3.0 / sp)
        else:
            self.stuck = max(0, self.stuck - 2)
            self.unb += 1
            if self.unb > SH_EHOLD:
                self.edir = None
        if self.stuck >= SH_STUCK:
            if self.relax <= 0.0:
                self.n_rlx += 1
            self.relax = min(1.0, self.relax + 1.0 / max(1, SH_RLXN))
        elif not bdir:
            self.relax = max(0.0, self.relax - 1.0 / max(1, SH_RLXB))
        eng = bool(float(np.linalg.norm(vs - vd)) > SH_EPS)
        if SH_LOG and (eng or tick % 25 == 0):
            fwd = _axes(st[3:6])[0]
            self.log.append({'t': tick, 'p': np.round(pos, 2).tolist(), 'v': np.round(st[6:9].astype(np.float64), 2).tolist(), 'yaw': round(float(st[5]), 3), 'pit': round(float(st[4]), 3), 'vd': np.round(vd, 2).tolist(), 'vs': np.round(vs, 2).tolist(), 'dmin': round(dmin, 2), 'nv': nv, 'ds': round(ds, 2), 'rlx': round(self.relax, 2), 'stk': self.stuck, 'esc': esc, 'bd': int(bdir), 'eng': int(eng), 'w': round(w, 3), 'cosf': round(float(vd @ fwd) / max(1e-06, nvd), 2), 'ts': round(self.t_see, 1), 'tf': round(self.t_ms, 1), 'nt': self.n_t, 'tmx': round(self.t_max, 2), **info})
            self.flush()
        if not eng and (self.vlast is None or float(np.linalg.norm(self.vlast - vd)) < 0.05):
            self.vlast = vd
            return out
        if eng:
            self.n_eng += 1
        if SH_DRY:
            self.vlast = vd
            return out
        vl = vd if self.vlast is None else self.vlast
        dl = vs - vl
        hm = SH_RATE * SIM_DT
        vm = SH_RATEV * SIM_DT
        dh = dl[0:2]
        nh = float(np.linalg.norm(dh))
        if nh > hm:
            dh = dh * (hm / nh)
        vo = np.array([vl[0] + dh[0], vl[1] + dh[1], vl[2] + max(-vm, min(vm, dl[2]))])
        self.vlast = vo
        sp = float(np.linalg.norm(vo))
        o = o.copy()
        if sp > 1e-06:
            o[0:3] = (vo / sp).astype(np.float32)
            o[3] = np.float32(min(1.0, sp / 3.0))
        else:
            o[0:3] = 0.0
            o[3] = np.float32(0.0)
        return o

    def _pick(self, u, D, dist, ds):
        hz = np.array([-u[1], u[0], 0.0])
        nh = float(np.linalg.norm(hz))
        hz = hz / nh if nh > 1e-06 else np.array([1.0, 0.0, 0.0])
        best = None
        bs = -1000000000.0
        R = SH_RB + 0.35
        cands = [(np.array([0.0, 0.0, 1.0]), SH_EUP), (np.array([0.0, 0.0, -1.0]), -0.3)]
        if SH_ESIDE:
            cands += [(hz, 0.0), (-hz, 0.0)]
        for e, pref in cands:
            al = D @ e
            lat2 = dist * dist - al * al
            hit = (al > 0.0) & (lat2 < R * R)
            room = float(al[hit].min()) if hit.any() else SH_RQ
            sc = room + pref
            if room >= SH_RB + ds + 0.4 and sc > bs:
                bs = sc
                best = e
        return best
PMAP_MAP = os.environ.get('KT_PMAP_MAP', 'forest:policy_t60f.onnx,city:policy_nr28.onnx')
RST47_ON = os.environ.get('KT_RST47', '1') == '1'

def _chl_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def _chl_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def _chl_seg(a, b):
    ab = _chl_sub(b, a)
    t = -_chl_dot(a, ab)
    den = _chl_dot(ab, ab)
    if t <= 0.0 or den <= 1e-18:
        return (a, [0])
    if t >= den:
        return (b, [1])
    t /= den
    return ((a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]), [0, 1])

def _chl_tri(a, b, c):
    ab = _chl_sub(b, a)
    ac = _chl_sub(c, a)
    ap = (-a[0], -a[1], -a[2])
    d1 = _chl_dot(ab, ap)
    d2 = _chl_dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return (a, [0])
    bp = (-b[0], -b[1], -b[2])
    d3 = _chl_dot(ab, bp)
    d4 = _chl_dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return (b, [1])
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and (d3 <= 0.0):
        v = d1 / (d1 - d3)
        return ((a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]), [0, 1])
    cp = (-c[0], -c[1], -c[2])
    d5 = _chl_dot(ab, cp)
    d6 = _chl_dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return (c, [2])
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and (d6 <= 0.0):
        w = d2 / (d2 - d6)
        return ((a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]), [0, 2])
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and (d5 - d6 >= 0.0):
        w = (d4 - d3) / (d4 - d3 + (d5 - d6))
        return ((b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])), [1, 2])
    den = va + vb + vc
    if abs(den) < 1e-18:
        best = None
        P = (a, b, c)
        for i, j in ((0, 1), (1, 2), (0, 2)):
            q, k = _chl_seg(P[i], P[j])
            n2 = _chl_dot(q, q)
            if best is None or n2 < best[0]:
                best = (n2, q, [(i, j)[x] for x in k])
        return (best[1], best[2])
    den = 1.0 / den
    v = vb * den
    w = vc * den
    return ((a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w), [0, 1, 2])

def _chl_out(p0, p1, p2, p3):
    e1 = _chl_sub(p1, p0)
    e2 = _chl_sub(p2, p0)
    n = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
    so = -_chl_dot(n, p0)
    sd = _chl_dot(n, _chl_sub(p3, p0))
    if sd * sd < 1e-24:
        return True
    return so * sd < 0.0

def _chl_closest(W):
    k = len(W)
    if k == 1:
        return (W[0], [0])
    if k == 2:
        return _chl_seg(W[0], W[1])
    if k == 3:
        return _chl_tri(W[0], W[1], W[2])
    best = None
    ins = True
    for i, j, k2, l in ((0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2), (1, 3, 2, 0)):
        if _chl_out(W[i], W[j], W[k2], W[l]):
            ins = False
            q, kk = _chl_tri(W[i], W[j], W[k2])
            n2 = _chl_dot(q, q)
            if best is None or n2 < best[0]:
                best = (n2, q, [(i, j, k2)[x] for x in kk])
    if ins:
        return ((0.0, 0.0, 0.0), [0, 1, 2, 3])
    return (best[1], best[2])

def _chl_gjk(S, p, iters=32, tol=0.0001):
    Q = S - p
    i = int(np.argmin(np.einsum('ij,ij->i', Q, Q)))
    W = [tuple((float(x) for x in Q[i]))]
    v = W[0]
    for _ in range(iters):
        vv = _chl_dot(v, v)
        if vv < 1e-12:
            return (0.0, p.copy(), 0)
        j = int(np.argmin(Q @ np.array(v)))
        w = tuple((float(x) for x in Q[j]))
        if vv - _chl_dot(v, w) <= tol * max(1.0, math.sqrt(vv)):
            break
        W.append(w)
        v, keep = _chl_closest(W)
        W = [W[x] for x in keep]
        if len(W) == 4:
            return (0.0, p.copy(), 0)
    return (math.sqrt(_chl_dot(v, v)), p + np.array(v), 1)

def _chl_label(occ, link):
    H, W = occ.shape
    N = H * W
    lab = np.where(occ, np.arange(N, dtype=np.int64).reshape(H, W), N)
    for _ in range(200):
        p = np.pad(lab, link, constant_values=N)
        m = lab
        for dx in range(-link, link + 1):
            for dy in range(-link, link + 1):
                if dx or dy:
                    m = np.minimum(m, p[link + dx:link + dx + H, link + dy:link + dy + W])
        m = np.where(occ, m, N)
        f = np.append(m.reshape(-1), N)
        for _j in range(64):
            g = f[f]
            if np.array_equal(g, f):
                break
            f = g
        m2 = f[:N].reshape(H, W)
        if np.array_equal(m2, lab):
            break
        lab = m2
    return np.where(occ, lab, -1)

class _ChullMem:

    def __init__(self):
        c = (np.arange(64) * 4 + 2.0 - 128.0) / 128.0
        self.U = np.tile(c[None, :], (64, 1))
        self.V = np.tile(-c[:, None], (1, 64))
        self.reset()

    def reset(self):
        self.zlo = np.full((CHL_N, CHL_N), np.inf, np.float32)
        self.zhi = np.full((CHL_N, CHL_N), -np.inf, np.float32)
        self.org = None
        self.g = None
        self.lab_t = -10 ** 9
        self.lab_org = None
        self.comps = []
        self.n_see = 0
        self.cmk_t = -10 ** 9
        self.cmk_org = None
        self.cmk_c = []
        self.nrc_t = -10 ** 9
        self.nrc_org = None
        self.nrc_c = []

    def _recenter(self, pos):
        new = np.floor(pos[0:2] / CHL_HC).astype(np.int64) - CHL_N // 2
        if self.org is not None:
            sh = new - self.org
            zl = np.full_like(self.zlo, np.inf)
            zh = np.full_like(self.zhi, -np.inf)
            klo = np.maximum(0, -sh)
            khi = np.minimum(CHL_N, CHL_N - sh)
            if np.all(khi > klo):
                zl[klo[0]:khi[0], klo[1]:khi[1]] = self.zlo[klo[0] + sh[0]:khi[0] + sh[0], klo[1] + sh[1]:khi[1] + sh[1]]
                zh[klo[0]:khi[0], klo[1]:khi[1]] = self.zhi[klo[0] + sh[0]:khi[0] + sh[0], klo[1] + sh[1]:khi[1] + sh[1]]
            self.zlo = zl
            self.zhi = zh
        self.org = new

    def see(self, depth, st):
        pos = st[0:3].astype(np.float64)
        self.n_see += 1
        if st.size > 162:
            agl = float(st[162]) * 20.0
            if agl < 19.5:
                gz = float(pos[2]) - agl
                self.g = gz if self.g is None else min(self.g, gz)
        if self.org is None or np.any(np.abs(np.floor(pos[0:2] / CHL_HC).astype(np.int64) - self.org - CHL_N // 2) > CHL_N // 4):
            self._recenter(pos)
        m = np.asarray(depth, np.float32).reshape(64, 4, 64, 4).min(axis=(1, 3))
        z = m.astype(np.float64) * (_DMAX - _DMIN) + _DMIN
        sel = (m < 0.999) & (z <= CHL_RSEE)
        if not sel.any():
            return
        fwd, up, right = _axes(st[3:6])
        cam = pos + fwd * _CAMF + up * _CAMU
        zs = z[sel][:, None]
        P = cam[None, :] + zs * (fwd[None, :] + self.U[sel][:, None] * right[None, :] + self.V[sel][:, None] * up[None, :])
        gm = float(P[:, 2].min())
        self.g = gm if self.g is None else min(self.g, gm)
        P = P[P[:, 2] >= self.g + CHL_HMIN]
        if not len(P):
            return
        k = np.floor(P[:, 0:2] / CHL_HC).astype(np.int64) - self.org
        ok = np.all((k >= 0) & (k < CHL_N), axis=1)
        if not ok.any():
            return
        k = k[ok]
        zz = P[ok, 2].astype(np.float32)
        np.minimum.at(self.zlo, (k[:, 0], k[:, 1]), zz)
        np.maximum.at(self.zhi, (k[:, 0], k[:, 1]), zz)

    def _label(self, pos):
        self.comps = []
        self.lab_org = self.org.copy()
        c = np.floor(pos[0:2] / CHL_HC).astype(np.int64) - self.org
        r = int(math.ceil(CHL_RW / CHL_HC))
        lo = np.maximum(c - r, 0)
        hi = np.minimum(c + r + 1, CHL_N)
        if np.any(hi <= lo):
            return
        occ = np.isfinite(self.zhi[lo[0]:hi[0], lo[1]:hi[1]])
        if not occ.any():
            return
        lab = _chl_label(occ, max(1, CHL_LINK))
        for lid in np.unique(lab[occ]):
            ii, jj = np.nonzero(lab == lid)
            self.comps.append((ii + lo[0], jj + lo[1]))

    def nrc_boxes(self, pos, tick):
        if self.org is None or self.g is None:
            return []
        if tick - self.nrc_t >= max(1, CHL_EVERY) or self.nrc_org is None or np.any(self.nrc_org != self.org):
            sv = (self.comps, self.lab_org)
            try:
                self._label(pos)
                self.nrc_c = self.comps
                self.nrc_org = self.lab_org
                self.nrc_t = tick
            finally:
                self.comps, self.lab_org = sv
        out = []
        px = float(pos[0])
        py = float(pos[1])
        for ii, jj in self.nrc_c:
            H = float(self.zhi[ii, jj].max()) - self.g
            if not NRC_HLO <= H < NRC_HHI:
                continue
            x0 = float(ii.min() + self.nrc_org[0]) * CHL_HC
            x1 = float(ii.max() + self.nrc_org[0] + 1) * CHL_HC
            y0 = float(jj.min() + self.nrc_org[1]) * CHL_HC
            y1 = float(jj.max() + self.nrc_org[1] + 1) * CHL_HC
            if math.hypot(max(x0 - px, 0.0, px - x1), max(y0 - py, 0.0, py - y1)) > NRC_RQ:
                continue
            out.append((x0, x1, y0, y1, H))
        return out

    def open_at(self, x, y):
        if self.org is None:
            return False
        k = np.floor(np.array([x, y], np.float64) / CHL_HC).astype(np.int64) - self.org
        if not (0 <= int(k[0]) < CHL_N and 0 <= int(k[1]) < CHL_N):
            return False
        return not bool(np.isfinite(self.zhi[int(k[0]), int(k[1])]))

    def cmk_feet(self, pos, tick, tgts):
        if self.org is None or self.g is None:
            return []
        if tick - self.cmk_t >= max(1, CHL_EVERY) or self.cmk_org is None or np.any(self.cmk_org != self.org):
            sv = (self.comps, self.lab_org)
            self._label(pos)
            self.cmk_c = self.comps
            self.cmk_org = self.lab_org
            self.cmk_t = tick
            self.comps, self.lab_org = sv
        out = []
        h = 0.5 * CHL_HC
        px = float(pos[0])
        py = float(pos[1])
        for ii, jj in self.cmk_c:
            H = float(self.zhi[ii, jj].max()) - self.g
            if H < CMK_HC:
                continue
            x = (ii + self.cmk_org[0] + 0.5) * CHL_HC
            y = (jj + self.cmk_org[1] + 0.5) * CHL_HC
            ex = np.maximum(np.abs(x - px) - h, 0.0)
            ey = np.maximum(np.abs(y - py) - h, 0.0)
            d = np.hypot(ex, ey)
            k = int(np.argmin(d))
            d2 = float(d[k])
            if d2 > CMK_RQ or d2 < 0.05:
                continue
            qx = min(max(px, float(x[k]) - h), float(x[k]) + h)
            qy = min(max(py, float(y[k]) - h), float(y[k]) + h)
            u = np.array([(qx - px) / d2, (qy - py) / d2, 0.0], np.float64)
            dt = [float(np.min(np.hypot(np.maximum(np.abs(x - tx) - h, 0.0), np.maximum(np.abs(y - ty) - h, 0.0)))) for tx, ty in tgts]
            out.append((u, d2, H, dt))
        return out

    def cons(self, pos, tick):
        if self.org is None:
            return []
        if tick - self.lab_t >= max(1, CHL_EVERY) or self.lab_org is None or np.any(self.lab_org != self.org):
            self._label(pos)
            self.lab_t = tick
        out = []
        for ii, jj in self.comps:
            x = (ii + self.org[0] + 0.5) * CHL_HC
            y = (jj + self.org[1] + 0.5) * CHL_HC
            if float(np.min((x - pos[0]) ** 2 + (y - pos[1]) ** 2)) > CHL_RQ * CHL_RQ:
                continue
            zl = self.zlo[ii, jj].astype(np.float64)
            zh = self.zhi[ii, jj].astype(np.float64)
            if float(pos[2]) >= float(zh.max()) - CHL_TOPM:
                continue
            S = np.concatenate([np.stack([x, y, zl], 1), np.stack([x, y, zh], 1)])
            d, cp, ok = _chl_gjk(S, pos)
            if not ok or d <= 0.001:
                continue
            dz = np.maximum(0.0, np.maximum(zl - cp[2], cp[2] - zh))
            nd = float(np.sqrt((x - cp[0]) ** 2 + (y - cp[1]) ** 2 + dz * dz).min())
            if nd < CHL_NOTCH or nd > CHL_XMAX:
                continue
            out.append(((cp - pos) / d, d))
        return out

class DroneFlightController:

    def __init__(self, *, model_path=None, providers=None):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(model_path or _H / 'policy.onnx'), so, providers=providers or ['CPUExecutionProvider'])
        self._psess = {}
        self._pmap_n = 0
        try:
            _pm = {}
            for _it in [x for x in PMAP_MAP.split(',') if ':' in x]:
                _t, _f = _it.split(':', 1)
                _pm[_t.strip()] = _f.strip()
            _loaded = {}
            for _t, _f in _pm.items():
                if _f not in _loaded and (_H / _f).exists():
                    _loaded[_f] = ort.InferenceSession(str(_H / _f), so, providers=providers or ['CPUExecutionProvider'])
                if _f in _loaded:
                    self._psess[_t] = _loaded[_f]
        except Exception:
            self._psess = {}
        self._ms = next((int(i.shape[0]) for i in self.session.get_inputs() if i.name == 'memory_tensor'))
        self._in = {i.name for i in self.session.get_inputs()}
        self._cls = _Cls() if ON else None
        if E9A_ON:
            self._rp = _RgbPrimary(model='victim_rgb256.onnx', in_res=256, thr=E9A_T, rng_max=E9A_RANGE) if ON and RP_ON else None
        else:
            self._rp = _RgbPrimary(model_by_type=RP_SPLIT) if ON and RP_ON else None
        if self._rp is not None and M_RPT:
            self._rp.fast = True
        self._r2 = _RgbPrimary(model='victim_rgb256.onnx', in_res=256, thr=R2_T, hmax=R2_HMAX, rng_max=R2_RANGE_MAX, term_r=R2_TERM_R, cap=R2_CAP, model_by_type=R2_SPLIT) if ON and R2_ON else None
        self._r2_on = False
        self._dt = _DepthTrigger() if ON and DT_ON else None
        self._dh = _DHead() if ON and DH_ON else None
        self._dh_term = None
        self._nv = _NVHead() if ON and (NV_ON or VNT_ON) else None
        if VNT_ON and self._r2 is not None:
            self._r2._vnt_r2 = True
        self._nvf = _NVFHead() if ON and NVF_ON else None
        self._plan = None
        self._esc = _Escape() if ON and ESC_ON else None
        self._sh = _Shield() if ON and SH_ON else None
        if E2P_ON:
            for _d in (self._rp, self._r2):
                if _d is not None:
                    _d.every = E2P_EVERY
        for _d, _warm in ((self._rp, 'rgb'), (self._r2, 'rgb'), (self._dh, 'depth')):
            if _d is None:
                continue
            try:
                _d._load()
                if _warm == 'depth' and getattr(_d, '_mload', None) is not None:
                    _d._mload()
                _z = np.zeros((256, 256, 1), np.float32)
                if getattr(_d, 'sess', None) is not None:
                    _d.sess.run(None, {_d.iname: _z if _warm == 'depth' else np.zeros((_d.in_res, _d.in_res, 3), np.float32)})
                if getattr(_d, 'msess', None) is not None:
                    _d.msess.run(None, {_d.miname: _z})
            except Exception:
                pass
        self.reset()

    def reset(self):
        self.memory_tensor = np.zeros(self._ms, np.float32)
        self.tick = 0
        self._rgb_n = 0
        self._h = {15: [], 25: []}
        self._pad = None
        self._z0 = None
        self._on = False
        self._cfg = None
        self._w = None
        self._i = 0
        self._adv = 0
        self._cv = None
        self._cvt = 0
        self._av = 0
        self._av_side = 1.0
        self._av_n = 0
        self._vrv_key = None
        self._vrv_nl = 0
        self._vrv_nh = 0
        self._vrv_spots = []
        self._vrv_sup = 0
        self._vrv_rev = 0
        self._vrv_acc = 0
        self._vrv_supn = 0
        self._vrv_err = 0
        self._tyx_n = 0
        self._vrv_m63 = None
        self._vrv_cf = 0
        self._nrc_n = 0
        self._nrc_cl = 0
        self._nrc_hold = 0
        self._nrc_err = 0
        self._nrc_g = False
        self._nrc_tx = 0
        self._vrv_rgbn = 0
        self._vrv_rgbok = 0
        self._vrv_last = None
        self._vrv_oof = 0
        self._m3l_n = 0
        self._rty_ovr = None
        self._rty_req = -1
        self._rty_only = False
        self._rty_n = 0
        self._rty_h = 0
        self._rty_done = False
        self._rty_last = -10 ** 9
        self._rty_hits = [0] * 6
        self._rty_t = -1
        self._rty_from = ''
        self._rty_pm = 0.0
        self._rty_err = 0
        if RTY_ON and getattr(self, '_r2', None) is not None:
            self._r2.cap = R2_CAP
        self._rlk = None
        self._rlk_spots = []
        self._rlk_n = 0
        self._rlk_end = {'pos': 0, 'lost': 0, 'time': 0}
        self._rlk_err = 0
        self._rlk_pts = []
        self._rlk_sup = False
        self._rlk_m63 = None
        self._fdl_h = []
        self._fdl_l = None
        self._fdl_t = 0
        self._fdl_t1 = -1
        self._fdl_near = 0
        self._fdl_bans = []
        self._fdl_lk = None
        self._fdl_c0 = None
        self._fdl_n = 0
        self._fdl_lat = 0
        self._fdl_bn = 0
        self._fdl_ok = 0
        self._fdl_lm = 0
        self._fdl_lp = None
        self._fdl_lw = False
        self._fdl_lan = 0
        self._fdl_lab = 0
        self._chull_mem = _ChullMem() if ON and (CHL_ON or CMK_ON) else None
        self._chull_n = 0
        self._cmk_n = 0
        self._cmk_st = {'t': 0, 'o': 0, 'slide': 0, 'relax': 0, 'tall': 0}
        self._cmk_side = 0.0
        self._cmk_su = -1
        self._cmk_free = -1
        self._cmk_lvl = 0
        self._cmk_gr = np.zeros(max(1, CMK_GW), np.uint8)
        self._cmk_gs = 0
        self._cmk_er = np.zeros(250, np.uint8)
        self._cmk_es = 0
        self._cmk_e1 = 0
        self._chl_lim = 0
        self._omb = _OMB() if OMB_ON else None
        self._omb_stats = self._omb.stats if self._omb is not None else {}
        self._vp = False
        self._vp_omb = None
        self._vp_n = 0
        self._vp_dm = None
        self._vp_nz = None
        self._vp_all = False
        self._vp_lite = False
        self._vpc_h = []
        self._vpc_ln = 0
        self._vpc_c = 0
        self._vpc_zc = 0.0
        self._vpc_n = 0
        self._vpc_k = 0
        self._vpc_o = None
        self._vpx = False
        self._vwh_n = 0
        self._vto_T = None
        self._vto_S = None
        self._vto_done = False
        self._vto_n = 0
        self._vto_arm = 0
        self._vto_r = 0
        self._vto_err = 0
        self._vpf_t1 = -1
        self._vpf_on = False
        self._vpf_done = False
        self._vpf_ph = 0
        self._vpf_c = 0
        self._vpf_blk = 0
        self._vpf_tt = -1
        self._vpf_n = 0
        self._vpf_why = 0
        self._vpf_nr = 0
        self._vpf_nt = 0
        self._vpf_nv = 0
        self._vpf_zt = 0
        self._vpf_dct = 0
        self._vpf_dr = 0
        self._vpf_lab = 0
        self._vpf_zr = 0
        self._vpf_zc = 0.0
        self._vpf_err = 0
        self._stall = 0
        self._push = 0
        self._stall_n = 0
        self.route = 'king'
        self._fg_on = 0
        self._fg_h = None
        self._fg_spd = None
        self._fg_n = 0
        self._fg_mem = None
        self._sup = None
        self._mass = None
        self._w0 = None
        self._rp_term = None
        self._mx_n = 0
        self._mx_used = 0
        self._c0 = None
        self._farlock_n = 0
        self._aglf_n = 0
        self._cm = None
        self._cm_t = 0
        self._cm_near = None
        self._cm_hits = []
        self._cm_bad = []
        self._cm_n = 0
        self._lm = _LM() if LM_ON else None
        self._cb = None
        self._wh_high = False
        self._whgate_n = 0
        self._sw = None
        self._swe = 0
        self._ll_est = None
        self._ll_n = 0
        self._ll_t = -10 ** 9
        self._ll_v = 0.0
        self._ll_on = False
        self._ll_t0 = 0
        self._ll_zt = None
        self._ll_ban = []
        self._ll_prevlk = False
        self._ll_hold = 0
        self._ll_trig = 0
        self._mp = False
        self._mp_n = 0
        self._mp_e = 0
        self._mp_tl = None
        self._mp_wp = None
        self._mp_bd = 0.0
        self._mp_hd = 0.0
        self._p3_n = 0
        self._p3_ban = []
        self._p3_resets = 0
        self._p3_pn = 0
        self._fpr_hv = None
        self._fpr_at = 0
        self._fpr_sup = 0
        self._fpr_pk = 0
        self._fpr_cg = 0
        self._fpr_t1 = -1
        self._fpr_err = 0
        self._la_hits = []
        self._la_ban = []
        self._la_spot = None
        self._la_n = 0
        self._la_rej = 0
        self._pg_cap = 1.0
        self._pg_t = -10 ** 9
        self._pg_n = 0
        self._pg_pts = []
        self._ng_mem = []
        self._ng_b = None
        self._ng_hold = 0
        self._ng_on = 0
        self._ng_cool = 0
        self._ng_n = 0
        self._ng_ticks = 0
        self._ng_rel = 0
        self._ng_dn = 0
        self._ss = None
        self._ss_k = 0
        self._ss_fr = []
        self._ss_hor = []
        self._ss_t = 0
        self._ss_head = None
        self._ss_start = None
        self._ss_z0 = None
        self._ss_yaw0 = 0.0
        self._ss_hit = False
        self._ss_sgn = -1.0
        self._ss_diag = False
        self._wx_h = []
        self._wx_l = []
        self._wx_T = None
        self._wx_leg = 0
        self._wx_dbg = None
        self._wx_n = 0
        self._wx_vetoed = False
        self._whg_h = []
        self._whg_n = 0
        self._whg_ev = 0
        self._whg_err = 0
        self._whg_rec = None
        self._wxf_on = False
        self._wxf_Q = []
        self._wxf_Tq = []
        self._wxf_t = 0
        self._wxf_rot = 0.0
        self._wxf_py = 0.0
        self._wxf_need = 0.0
        self._wxf_dbg = None
        self._wxf_err = 0
        self._wxf_cp = 0
        self._whk_q = []
        self._whk_n = 0
        self._whk_t = 0
        self._whk_min = 9.0
        self._whk_err = 0
        self._whd_q = []
        self._whd_th = 0.0
        self._whd_n = 0
        self._whd_ev = 0
        self._whd_t1 = 0
        self._whd_amax = 0.0
        self._whd_err = 0
        self._whd_lg = []
        self._wcr_ph = None
        self._wcr_rib = 0.0
        self._wcr_side = 1.0
        self._wcr_tried = 0
        self._wcr_t0 = 0
        self._wcr_dx0 = 0.0
        self._wcr_zl = 0.0
        self._wcr_n = 0
        self._wcr_nblk = 0
        self._wcr_free = -1.0
        self._wcr_why = ''
        self._wcr_te = -1
        self._wcr_err = 0
        self._wcr_y0 = 0.0
        self._wcr_z0 = 0.0
        self._wcr_f1 = -1.0
        self._wcr_f2 = -1.0
        self._wcr_s1 = 0.0
        self._wcr_tg = -1
        self._fsl_P = None
        self._fsl_T = None
        self._fsl_side = None
        self._fsl_last = -10 ** 9
        self._fsl_p3h = None
        self._fsl_tr = []
        self._fsx = None
        self._fsx_c = 0
        self._fsx_avt = -1
        self._fsx_rp = False
        self._fsx_n = 0
        self._tr = None
        self._tr_g = None
        self._tr_best = 0.0
        self._tr_t = 0
        self._tr_n = 0
        self._tr_k = 0
        self._tr_rk = 0
        self._tr_rel = 0
        self._tr_raw = None
        self._fau = None
        self._fau_w = None
        self._fau_n = 0
        self._fau_t = 0
        self._fau_t1 = -1
        self._fau_pl = 0
        self._fau_np = 0
        self._fau_ok = 0
        self._fau_why = ''
        self._fau_c = 0.0
        self._fau_ms = 0.0
        self._fau_p3 = 0
        self._fau_err = 0
        self._fau_cool = -1
        self._fau_pts = []
        self._fau_hd = 0
        self._fau_kp = 0
        self._dib_on = False
        self._dib_n = 0
        self._dib_t = 0
        self._dib_err = 0
        self._dib_L = None
        self._dib_w = 0
        self._dib_arm = False
        self._dib_first = None
        self._dib_fire = 0
        self._hrz_L = None
        self._hrz_w = 0
        self._hrz_arm = False
        self._hrz_act = 0
        self._hrz_hold = 0
        self._hrz_ban = []
        self._hrz_n = 0
        self._hrz_t = 0
        self._hrz_err = 0
        self._hrz_first = None
        self._hrz_u = 0
        self._hrz_ulog = False
        self._hrx_L = None
        self._hrx_w = 0
        self._hrb_w = 0
        self._hrx_done = False
        self._hrb_done = False
        self._fsl_n = 0
        self._fsl_nd = 0
        self._fsl_nb = 0
        self._fsl_n3 = 0
        self._fsl_na = 0
        self._fsx_nv = 0
        self._fsx_q = 0
        self._r2a_n = 0
        self._vf_n = 0
        self._fsl_err = 0
        self._fo = False
        self._fb_p3 = 0
        self._fb_bx = 0
        self._fb_r2n = 0
        self._ss_outd = SS_OUT
        self._tk_on = False
        self._tk_d1 = None
        self._tk_p1 = None
        self._tk_r1 = None
        self._tk_why = ''
        self._tk_phase = None
        self._tk_cells = {}
        self._tk_cap = 0
        self._tk_hold = 0
        self._tk_free_n = 0
        self._tk_gdone = False
        self._tk_forced = 0
        self._tk_zc = None
        self._tk_sp_done = False
        self._tk_zsp = None
        self._tk_spn = 0
        self._whh_n = 0
        self._esc_w = -1
        self._steer = False
        self._rv_ps = False
        self._rv_n = 0
        self._rv_t0 = -1
        if self._rp is not None:
            self._rp.reset()
        if getattr(self, '_r2', None) is not None:
            self._r2.reset()
        self._r2_on = False
        if getattr(self, '_dh', None) is not None:
            self._dh.reset()
        self._dh_term = None
        self._fcl_klk = False
        self._fcl_n = 0
        self._pf_rgb_t = None
        self._pf_only = False
        self._pf_nhide = 0
        if getattr(self, '_nv', None) is not None:
            self._nv.reset()
            self._nv_mirror()
        if VNT_ON:
            self._vnt_sp = []
            self._vnt_req = -10 ** 9
            self._vnt_last = -10 ** 9
            self._vnt_n = 0
            self._vnt_f = 0
            self._vnt_h = 0
            self._vnt_err = 0
            del _VNT_H[:]
            if getattr(self, '_r2', None) is not None:
                self._r2.n_vnt_al = 0
                self._r2.n_vnt_err = 0
        if getattr(self, '_nvf', None) is not None:
            self._nvf.reset()
        if getattr(self, '_dt', None) is not None:
            self._dt.n_eval = 0
            self._dt.n_hit = 0
            self._dt.last = None
            self._dt.last_hit_t = -10 ** 9
        self._plan = None
        if self._esc is not None:
            self._esc.reset()
        if getattr(self, '_sh', None) is not None:
            self._sh.reset()
        if self._cls:
            self._cls.reset()

    def _tour(self, core, lane, spacing, cap, pos):
        if M_LANES and self._cls is not None and self._cls.is_mountain:
            return _bous_lanes(core, lane, spacing, cap, pos)
        return _bous(core, lane, spacing, cap, pos)

    def _vbuild(self, pos, clue, c):
        if self._sup is None:
            _bx = [30.0] if E8X_ON and self._stype() == 'mountain' else None
            sup, w, dc = _ccore(pos[0:2] + clue, self._pad, 83.0, _bx)
            self._sup = sup
            self._w0 = w
            self._mass = w.copy()
            self._dc = dc
            if MP_ON and self._cls is not None and self._cls.is_mountain and ('open' not in self._cls.latched):
                try:
                    _nc = int(self._core(0.9).shape[0])
                    ps, pw, pd = _mpost(pos[0:2] + clue, self._pad)
                    _t = float(pw.sum())
                    _g = _nc < MP_BC if MP_BC > 0 else float(pw[np.linalg.norm(ps - self._pad, axis=1) > 83.0].sum()) >= MP_MOUT * _t or _nc < MP_NCORE
                    if _t > 0 and (MP_ALL or _g):
                        self._sup = ps
                        self._w0 = pw
                        self._mass = pw.copy()
                        self._dc = pd
                        self._mp = True
                        self._mp_tl = np.ones(len(ps)) if MP_TAIL else None
                except Exception:
                    self._mp = False
                    self._mp_tl = None
                    self._mp_e += 1
        if self._mp and MP_GT:
            try:
                wps = _gtour(self._sup, self._mass, pos[0:2], self._mp_hd, c[5], self._mp_tl)
                self._mp_n += 1
                if not wps:
                    self._mass = self._w0.copy()
                    wps = _gtour(self._sup, self._mass, pos[0:2], self._mp_hd, c[5], self._mp_tl)
                if wps:
                    return wps
            except Exception:
                self._mp_e += 1
        if FF_ON and self._stype() in FF_TYPES:
            wps = []
            for fr in FF_FRACS:
                wps = self._tour(self._core(fr), c[3], c[4], 10 ** 6, pos[0:2])
                if len(wps) <= c[5]:
                    break
            wps = wps[:c[5]]
        else:
            wps = self._tour(self._core(0.9), c[3], c[4], c[5], pos[0:2])
        if not wps:
            self._mass = self._w0.copy()
            wps = self._tour(self._core(0.9), c[3] * 0.65, c[4] * 0.7, c[5], pos[0:2])
        if not wps:
            wps = self._tour(self._sup[self._dc <= 30.0], c[3], c[4], c[5], pos[0:2])
        if not wps and LADDER:
            t = self._stype()
            b = float(_BOX[LB.index(t)]) if t in LB else 30.0
            dx = max(abs(float(self._pad[0])) - b, 0.0)
            dy = max(abs(float(self._pad[1])) - b, 0.0)
            rp = max(80.0, float(np.hypot(dx, dy)) + 30.0) + 3.0
            for rpad, bx in ((rp, None), (1000000000.0, [1.5 * b]), (1000000000.0, [1000.0])):
                sup, w, dc = _ccore(pos[0:2] + clue, self._pad, rpad, bx)
                if int((w > 0).sum()) >= 4:
                    self._sup = sup
                    self._w0 = w
                    self._mass = w.copy()
                    self._dc = dc
                    self._ladder = 1
                    self._mp = False
                    self._mp_tl = None
                    wps = self._tour(self._core(0.9), c[3], c[4], c[5], pos[0:2])
                    if wps:
                        break
        return wps

    def _mkplan(self, pos, st):
        tp = self._stype()
        bx = None
        if E1X_ON and _u19() and (tp in E1_TYPES) and (tp in LB):
            b = float(_BOX[LB.index(tp)])
            bx = [b, 1.5 * b] if E1X_TAIL else [b]
        _sup, _w0, _dc = _ccore(pos[0:2] + st[-2:].astype(np.float64), self._pad, 83.0, bx)
        p = _Planner(_sup, _w0)
        p.mtype = tp
        return p

    def _reprior(self, pos, st):
        old = self._plan
        new = self._mkplan(pos, st)
        if old is not None and new.mtype in E1_TYPES and (old.sup.shape == new.sup.shape) and np.array_equal(old.sup, new.sup):
            r = np.where(old.w0 > 0, np.minimum(1.0, old.mass / np.where(old.w0 > 0, old.w0, 1.0)), 1.0)
            new.mass = new.w0 * r
            new.occ = old.occ.copy()
            new.occ_hits = old.occ_hits.copy()
            new.n_occ = old.n_occ
            new.seen_hits = dict(old.seen_hits)
            new.n_obs = old.n_obs
            new.mass[new.occ] = 0.0
        return new

    def _core(self, frac):
        w = self._mass
        t = float(w.sum())
        if t <= 0:
            return np.zeros((0, 2))
        o = np.argsort(-w)
        cw = np.cumsum(w[o]) / t
        return self._sup[o[:int(np.searchsorted(cw, frac)) + 1]]

    def _swept(self, xy):
        if self._mass is not None:
            _s = np.linalg.norm(self._sup - np.asarray(xy, np.float64), axis=1) <= CVR
            self._mass[_s] = 0.0
            if self._mp_tl is not None:
                self._mp_tl[_s] = 0.0

    def _fdl_ban(self, q):
        q = np.asarray(q, np.float64).copy()
        self._fdl_bans.append(q)
        self._fdl_bn += 1
        self._fdl_h = [h for h in self._fdl_h if float(np.hypot(h[1] - float(q[0]), h[2] - float(q[1]))) >= FDL_BAN]

    def _fdl_step(self, st, obs_state):
        _t = 'mountain' if self._cls.is_mountain else self._stype()
        fl_on = _t in FDL_TYPES
        lm_on = FDL_LM and _t in FDL_LM_TYPES
        if not (fl_on or lm_on):
            return False
        if FDL_FMAX < 1.0 and float(self._cls.ps[LB.index('forest')]) / max(self._cls.n, 1) >= FDL_FMAX:
            return False
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        if mm.size < 63:
            return False
        pos = st[0:3].astype(np.float64)
        if self._fdl_c0 is None:
            self._fdl_c0 = pos[0:2] + np.asarray(obs_state, np.float32).reshape(-1)[-2:].astype(np.float64)
        _skip = False
        if FDL_LA:
            _lb = getattr(self, '_la_ban', None)
            if _lb is not None and len(_lb) > self._fdl_lan:
                for _q in _lb[self._fdl_lan:]:
                    _q = np.asarray(_q, np.float64).reshape(-1)[0:2].copy()
                    self._fdl_bans.append(_q)
                    self._fdl_lab += 1
                    self._fdl_h = [h for h in self._fdl_h if float(np.hypot(h[1] - float(_q[0]), h[2] - float(_q[1]))) >= FDL_BAN]
                self._fdl_lan = len(_lb)
                _skip = True
                if self._omb is not None:
                    self._omb.stats['fdl_lab'] = self._fdl_lab
        mode = float(mm[0])
        locked = float(mm[57]) > 0.5
        if self._fdl_lk is not None and (not locked) and (mode < 1.5):
            self._fdl_ban(self._fdl_lk)
            self._fdl_lk = None
        if locked:
            self._fdl_lp = np.asarray(mm[60:62], np.float64).copy()
            self._fdl_lw = True
            if self._fdl_l is not None and float(np.linalg.norm(self._fdl_lp - self._fdl_l)) <= FDL_LOCKR:
                self._fdl_ok += 1
                self._fdl_lk = None if lm_on else self._fdl_l.copy()
                self._fdl_l = None
                self._fdl_h = []
            return False
        if self._fdl_lw and mode < 1.5:
            self._fdl_lw = False
            if lm_on and self._fdl_l is None and (self._fdl_lp is not None) and (float(np.linalg.norm(self._fdl_lp - pos[0:2])) >= FDL_LM_R) and (not any((float(np.linalg.norm(self._fdl_lp - b)) < FDL_BAN for b in self._fdl_bans))):
                self._fdl_l = self._fdl_lp.copy()
                self._fdl_t = self.tick
                self._fdl_near = 0
                self._fdl_lat += 1
                self._fdl_lm += 1
                if self._fdl_t1 < 0:
                    self._fdl_t1 = self.tick
        if mode >= 1.5:
            return False
        if fl_on and self.tick >= FDL_T0 and (not _skip):
            p = float(mm[55])
            d = np.asarray(mm[43:45], np.float64)
            if p >= FDL_P and float(np.linalg.norm(d - pos[0:2])) >= FDL_RNG and (float(np.linalg.norm(d - self._fdl_c0)) <= FDL_CR) and (not any((float(np.linalg.norm(d - b)) < FDL_BAN for b in self._fdl_bans))):
                self._fdl_h.append((self.tick, float(d[0]), float(d[1])))
        if self._fdl_h and self.tick - self._fdl_h[0][0] > FDL_WIN:
            self._fdl_h = [h for h in self._fdl_h if self.tick - h[0] <= FDL_WIN]
        if fl_on and self._fdl_l is None and (len(self._fdl_h) >= FDL_K):
            a = np.asarray(self._fdl_h, np.float64)
            med = np.median(a[:, 1:3], axis=0)
            sel = a[np.linalg.norm(a[:, 1:3] - med, axis=1) <= FDL_TOL]
            if len(sel) >= FDL_K and float(sel[:, 0].max() - sel[:, 0].min()) >= FDL_SPAN:
                self._fdl_l = sel[:, 1:3].mean(axis=0)
                self._fdl_t = self.tick
                self._fdl_near = 0
                self._fdl_lat += 1
                if self._fdl_t1 < 0:
                    self._fdl_t1 = self.tick
        if self._fdl_l is None:
            return False
        if float(np.linalg.norm(self._fdl_l - pos[0:2])) < FDL_NEAR:
            self._fdl_near += 1
        if self._fdl_near >= FDL_DUD or self.tick - self._fdl_t >= FDL_TMAX:
            self._fdl_ban(self._fdl_l)
            self._fdl_l = None
            return False
        for _d in (getattr(self, '_rp', None), getattr(self, '_r2', None), getattr(self, '_dh', None)):
            if _d is not None and getattr(_d, 'latch', None) is not None:
                return False
        if getattr(self, '_cm', None) is not None or getattr(self, '_ll_on', False) or getattr(getattr(self, '_lm', None), 'tgt', None) is not None:
            return False
        lk = self._fdl_l - pos[0:2]
        st[-2] = np.float32(lk[0])
        st[-1] = np.float32(lk[1])
        if mm[1] != 0.0 or mm[2] != 0.0:
            mm = mm.copy()
            mm[1] = 0.0
            mm[2] = 0.0
            self.memory_tensor = mm.reshape(self._ms)
        self._fdl_n += 1
        self._steer = True
        if not self.route.endswith('+fdl'):
            self.route = self.route + '+fdl'
        return True

    def _vpf_step(self, observation, st, out):
        if self._vpf_done:
            return out
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        if mm.size < 1 or self._z0 is None or self._pad is None:
            return out
        md = float(mm[0])
        if self._vpf_t1 < 0:
            if md < 0.5:
                return out
            self._vpf_t1 = self.tick
            if VPF_T1 > 0 and self.tick > VPF_T1:
                return self._vpf_end(13, out)
        pos = st[0:3].astype(np.float64)
        busy = bool(any((getattr(self, k, None) is not None for k in ('_r2_term', '_rp_term', '_dh_term'))) or (self._esc is not None and self._esc.phase in ('scan', 'creep', 'climb')))
        if self._vpf_on:
            why = 0
            if not 0.5 <= md < 1.5:
                why = 2
            elif float(pos[2]) >= self._vpf_zc:
                why = 1
            elif self.tick - self._vpf_tt > VPF_TMAX:
                why = 3
            elif busy:
                why = 4
            if why:
                self._vpf_zr = int(round(float(pos[2]) * 100))
                return self._vpf_end(why, out)
            return self._vpf_cmd(observation, st, pos, out)
        if md >= 1.5:
            return self._vpf_end(12, out)
        if self.tick - self._vpf_t1 > VPF_W:
            return self._vpf_end(11, out)
        if self._cls.is_mountain or self._wh_spawn() or getattr(self, '_tk_on', False) or (float(self._z0) >= VPF_Z0MAX):
            return self._vpf_end(16, out)
        mi = float(self._fg_mem[0]) if self._fg_mem is not None and np.size(self._fg_mem) > 0 else md
        if not (0.5 <= md < 1.5 and 0.5 <= mi < 1.5):
            return out
        if float(pos[2]) - float(self._z0) >= VPF_DZ:
            return out
        dr = math.hypot(float(pos[0]) - float(self._pad[0]), float(pos[1]) - float(self._pad[1]))
        if dr > VPF_R:
            return out
        d = np.asarray(out[0:3], np.float64)
        n = float(np.linalg.norm(d))
        if n < 1e-09 or float(d[2]) / n * abs(float(out[3])) * OMB_SPEED >= -VPF_VT:
            return out
        self._vpf_c += 1
        _t = self._stype()
        lab = 1 if self._vp_all else 2 if _t == 'village' else 3 if _t == 'warehouse' else 0
        dc = float(np.asarray(observation['depth'], np.float32).reshape(256, 256)[96:160, 96:160].min()) * (_DMAX - _DMIN) + _DMIN
        blk = 0
        if lab == 0:
            blk |= 1
        if VPF_DC > 0.0 and dc >= VPF_DC:
            blk |= 2
        if busy:
            blk |= 8
        for _d in (getattr(self, '_nv', None), getattr(self, '_nvf', None)):
            if _d is not None and getattr(_d, 'latch', None) is not None:
                blk |= 16
        self._vpf_blk |= blk
        if blk:
            self._vpf_stat()
            return out
        self._vpf_tt = self.tick
        self._vpf_zt = int(round(float(pos[2]) * 100))
        self._vpf_dct = int(round(dc * 100))
        self._vpf_dr = int(round(dr * 100))
        self._vpf_lab = lab
        self._vpf_zc = max(VPF_ZTOP, float(self._z0) + VPF_DZTOP)
        if VPF_SHADOW:
            return self._vpf_end(20, out)
        self._vpf_on = True
        self._vpf_ph = 1
        return self._vpf_cmd(observation, st, pos, out)

    def _vpf_cmd(self, observation, st, pos, out):
        v = np.asarray(st[6:9], np.float64)
        rp = np.asarray(st[3:5], np.float64)
        h = np.asarray(self._pad[0:2], np.float64) - pos[0:2]
        dh = float(np.hypot(h[0], h[1]))
        if self._vpf_ph == 1 and dh <= VPF_RC:
            self._vpf_ph = 2
        if self._vpf_ph == 1:
            th = h / max(dh, 1e-06) * min(VPF_BACK, dh)
            tz = 0.0
            self._vpf_nr += 1
        else:
            th = h * min(1.0, 0.3 / max(dh, 1e-06))
            tz = VPF_V * OMB_SPEED
            if VPF_TOPM > 0.0:
                top = float(np.asarray(observation['depth'], np.float32).reshape(256, 256)[0:8, 64:192].min()) * (_DMAX - _DMIN) + _DMIN
                if top < VPF_TOPM:
                    tz = 0.0
                    self._vpf_nv += 1
        if float(np.max(np.abs(rp))) > VPF_TILT:
            c = np.array([v[0], v[1], max(float(v[2]), 0.0)], np.float64)
            self._vpf_nt += 1
        else:
            eh = th - v[0:2]
            ne = float(np.hypot(eh[0], eh[1]))
            if ne > VPF_EH:
                eh = eh * (VPF_EH / ne)
            ez = min(VPF_EZ, max(-VPF_EZ, tz - float(v[2])))
            c = np.array([v[0] + eh[0], v[1] + eh[1], float(v[2]) + ez], np.float64)
        s = float(np.linalg.norm(c))
        o = out.copy()
        if s < 1e-06:
            o[0] = np.float32(0.0)
            o[1] = np.float32(0.0)
            o[2] = np.float32(1.0)
            o[3] = np.float32(0.0)
        else:
            o[0:3] = (c / s).astype(np.float32)
            o[3] = np.float32(min(1.0, s / OMB_SPEED))
        self._vpf_n += 1
        if not self.route.endswith('+vpf'):
            self.route = self.route + '+vpf'
        self._vpf_stat()
        return o

    def _vpf_end(self, why, out):
        self._vpf_on = False
        self._vpf_done = True
        self._vpf_why = int(why)
        self._vpf_stat()
        return out

    def _vpf_stat(self):
        o = getattr(self, '_omb', None)
        if o is None or not isinstance(getattr(o, 'stats', None), dict):
            return
        for _k in ('t1', 'c', 'blk', 'tt', 'n', 'nr', 'nt', 'nv', 'why', 'zt', 'dct', 'dr', 'lab', 'zr', 'err'):
            o.stats['vpf_' + _k] = int(getattr(self, '_vpf_' + _k))

    def _vto_step(self, observation, st, out):
        if self._vto_S is None:
            self._vto_done = True
            if self.tick != 1 or self._cls is None or self._cls.is_mountain or (getattr(self, '_z0', None) is None):
                return out
            if self._wh_spawn() or self._z0 >= MTN_Z0 or (not (self._vp_all or VTO_ALL)):
                return out
            p = st[0:3].astype(np.float64)
            T = _vto_pick(observation['depth'], p, st[3:6].astype(np.float64), self._z0)
            r = float(np.hypot(T[0], T[1]))
            if r <= 0.0:
                return out
            self._vto_S = p.copy()
            self._vto_T = p[0:2] + T
            self._vto_arm = 1
            self._vto_r = int(round(r * 100))
            self._vto_done = False
        if self._vto_T is None:
            self._vto_done = True
            return out
        p = st[0:3].astype(np.float64)
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        dv = self._vto_T - p[0:2]
        dh = float(np.hypot(dv[0], dv[1]))
        d3 = float(np.linalg.norm(p - self._vto_S))
        if self.tick > VTO_TMAX or d3 >= VTO_EXIT or dh < VTO_ARR or (mm.size > 0 and float(mm[0]) >= 0.5):
            self._vto_T = None
            self._vto_done = True
            return out
        h = dv / max(dh, 1e-06) * min(VTO_V, VTO_K * dh)
        v = np.array([h[0], h[1], VTO_VZ], np.float64)
        sp = float(np.linalg.norm(v))
        o = out.copy()
        o[0:3] = (v / sp).astype(np.float32)
        o[3] = np.float32(min(1.0, sp / 3.0))
        self._vto_n += 1
        if not self.route.endswith('+vto'):
            self.route = self.route + '+vto'
        return o

    def _vpc_pre(self, st, out, p):
        o = out.copy()
        if self._vpc_c <= 0:
            return o
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        if float(p[2]) >= self._vpc_zc or self._vpc_c > VPC_T or (mm.size > 0 and (not 0.5 <= float(mm[0]) < 1.5)):
            self._vpc_c = 0
            self._vpc_h = []
            return o
        self._vpc_c += 1
        self._vpc_n += 1
        o[0] = np.float32(0.0)
        o[1] = np.float32(0.0)
        o[2] = np.float32(1.0)
        o[3] = np.float32(min(1.0, VPC_V / 3.0))
        if not self.route.endswith('+vpc'):
            self.route = self.route + '+vpc'
        self._vpc_o = o.copy()
        return o

    def _vpc_post(self, st, out, p):
        c = int(self._vp_n != self._vpc_ln)
        self._vpc_ln = self._vp_n
        if self._vpc_c > 0:
            o = self._vpc_o
            self._vpc_o = None
            return out if c or o is None else o
        self._vpc_h.append((float(p[0]), float(p[1]), c))
        if len(self._vpc_h) > VPC_N:
            self._vpc_h.pop(0)
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        if len(self._vpc_h) < VPC_N or mm.size == 0 or (not 0.5 <= float(mm[0]) < 1.5):
            return out
        h0 = self._vpc_h[0]
        if sum((x[2] for x in self._vpc_h)) < 0.8 * VPC_N or math.hypot(float(p[0]) - h0[0], float(p[1]) - h0[1]) >= VPC_D:
            return out
        P = self._vp_omb.pts if self._vp_omb is not None else np.zeros((0, 3))
        near = np.hypot(P[:, 0] - float(p[0]), P[:, 1] - float(p[1])) < 3.0 if len(P) else np.zeros(0, bool)
        if not near.any():
            return out
        gz = float(p[2]) - float(st[162]) * 20.0 if st.size > 162 else 0.0
        zc = min(float(P[near, 2].max()) + VPC_DZ, gz + VPC_ZMAX)
        if zc > float(p[2]) + 0.3:
            self._vpc_c = 1
            self._vpc_zc = zc
            self._vpc_k += 1
            self._vpc_h = []
        return out

    def _stype(self):
        _t = self._stype0()
        if VWH_ON and _t == 'warehouse' and self._vp_all:
            self._vwh_n += 1
            return 'village'
        return _t

    def _stype0(self):
        c = self._cls
        if c.is_mountain:
            return 'mountain' if 'mountain' in TYPES else None
        if RTY_ON and getattr(self, '_rty_ovr', None) is not None and (not self._wh_spawn()):
            return self._rty_ovr
        if RL_ON and c.relatched and (not (getattr(self, '_z0', None) is not None and self._z0 <= WESC_Z)):
            cands = [t for t in TYPES if t != 'mountain' and (c.settled(t) or t in c.relatched)]
            if cands:
                avg = c.ps / max(c.n, 1)
                return max(cands, key=lambda t: float(avg[LB.index(t)]))
        for t in TYPES:
            if t != 'mountain' and c.settled(t):
                return t
        return None

    def _r2_type(self):
        c = self._cls
        if c is None or c.is_mountain or c.n <= 0:
            return None
        if RTY_ON and getattr(self, '_rty_ovr', None) is not None and (not self._wh_spawn()):
            return self._rty_ovr if self._rty_ovr in R2_TYPES else None
        if FB_R2ON and self._fo:
            return 'forest'
        avg = c.ps / max(c.n, 1)
        i = int(np.argmax(avg))
        lab = LB[i]
        if lab not in R2_TYPES:
            return None
        if not (c.settled(lab) or (RL_ON and lab in c.relatched)):
            return None
        if VWH_ON and lab == 'warehouse' and self._vp_all and (float(avg[i]) >= R2_BAR):
            return 'village'
        return lab if float(avg[i]) >= R2_BAR else None

    def _r2_ok(self):
        return bool(self._r2 is not None and self._r2_type() is not None)

    def _champ(self):
        if not CHAMP_ROUTE:
            return False
        c = self._cls
        if c is None or c.is_mountain:
            return False
        return bool(c.settled('village') or c.settled('warehouse'))

    def act(self, observation):
        self.tick += 1
        _TCUR[0] = self._stype()
        rq = False
        self._fo = False
        self._steer = False
        self._rv_ps = False
        if (FPR_PK or FPR_CG) and '+fpr' in self.route:
            self.route = self.route.replace('+fpr', '')
        if '+fdl' in self.route:
            self.route = self.route.replace('+fdl', '')
        if '+dib' in self.route:
            self.route = self.route.replace('+dib', '')
        if '+hrz' in self.route:
            self.route = self.route.replace('+hrz', '')
        if '+chull' in self.route:
            self.route = self.route.replace('+chull', '')
        if '+cmk' in self.route:
            self.route = self.route.replace('+cmk', '')
        if '+whk' in self.route:
            self.route = self.route.replace('+whk', '')
        if '+vrv' in self.route:
            self.route = self.route.replace('+vrv', '')
        if '+rlk' in self.route:
            self.route = self.route.replace('+rlk', '')
        if '+nrc' in self.route:
            self.route = self.route.replace('+nrc', '')
        if '+vto' in self.route:
            self.route = self.route.replace('+vto', '')
        if '+vpc' in self.route:
            self.route = self.route.replace('+vpc', '')
        if VPF_ON and '+vpf' in self.route:
            self.route = self.route.replace('+vpf', '')
        if FAU_ON and '+fau' in self.route:
            self.route = self.route.replace('+fau', '')
        if WCR_ON and '+wcr' in self.route:
            self.route = self.route.replace('+wcr', '')
        if '+fs' in self.route or '+r2a' in self.route or '+r2nc' in self.route or ('+trt' in self.route):
            for _s in ('+fsv', '+fsx', '+fsq', '+fse', '+fsa', '+fsb', '+fsd', '+fs3', '+fsl', '+r2a', '+r2nc', '+trt'):
                self.route = self.route.replace(_s, '')
        if self.route[0:2] == 'sw':
            self.route = self.route.split('|', 1)[1]
        if self.route.endswith('+ng'):
            self.route = self.route[:-3]
        if self.route.endswith('+vp'):
            self.route = self.route[:-3]
        if WHD_ON and '+whd' in self.route:
            self.route = self.route.replace('+whd', '')
        if self.route.endswith('+omb'):
            self.route = self.route[:-4]
        if self.route.endswith('+sh'):
            self.route = self.route[:-3]
        if self.route.endswith('+p3r'):
            self.route = self.route[:-4]
        if self.route.endswith('+p3f'):
            self.route = self.route[:-4]
        if self.route.endswith('+av'):
            self.route = self.route[:-3]
        if self.route.endswith('+fg'):
            self.route = self.route[:-3]
        if self.route.endswith('+pg'):
            self.route = self.route[:-3]
        if self.route.endswith('+la'):
            self.route = self.route[:-3]
        if TK_ON and self.route.endswith('+tkp'):
            self.route = self.route[:-4]
        if self.route.endswith('+dht'):
            self.route = self.route[:-4]
        if self.route.endswith('+lm'):
            self.route = self.route[:-3]
        st = np.asarray(observation['state'], np.float32).reshape(-1).copy()
        if RTY_ON and self._rty_req >= 0:
            _rq = self._rty_req
            _ro = self._rty_only
            self._rty_req = -1
            self._rty_only = False
            if self.tick == _rq + 1:
                observation = self._rty_frame(observation, _ro)
        if self._pf_rgb_t is not None:
            _t = self._pf_rgb_t
            _only = self._pf_only
            self._pf_rgb_t = None
            self._pf_only = False
            if self.tick == _t + 1:
                try:
                    observation = self._pf_frame(observation, st, _only)
                except Exception:
                    if self._dh is not None:
                        self._dh.pf_rgb = False
                        self._dh.pf_err += 1
        if self._sh is not None:
            try:
                self._sh.see(observation['depth'], st)
            except Exception:
                pass
        if getattr(self, '_chull_mem', None) is not None and self._cls is not None and (not self._cls.is_mountain):
            try:
                self._chull_see(observation, st)
            except Exception:
                pass
        if ON and self._cls:
            pos = st[0:3].astype(np.float64)
            if self._pad is None:
                self._pad = pos[0:2].copy()
                self._z0 = float(pos[2])
                if VP_ON and VP_ZLO < self._z0 < VP_ZHI:
                    try:
                        _vd = np.asarray(observation['depth'], np.float32).reshape(256, 256) * (_DMAX - _DMIN) + _DMIN
                        self._vp_dm = float(np.median(_vd[64:192, 64:192]))
                        if self._vp_dm < VP_D:
                            _s = _vd[::2, ::2]
                            _m = (_s > _DMIN + 0.001) & (_s < VP_D)
                            if int(_m.sum()) >= VP_NP:
                                _px = (np.arange(0, 256, 2, dtype=np.float64) + 0.5) / 256 * 2.0 - 1.0
                                _z = _s[_m].astype(np.float64)
                                _u = np.broadcast_to(_px[None, :], _s.shape)[_m] * _HT * _z
                                _v = np.broadcast_to(-_px[:, None], _s.shape)[_m] * _HT * _z
                                _fw, _up, _rt = _axes(st[3:6].astype(np.float64))
                                _P = _u[:, None] * _rt[None, :] + _v[:, None] * _up[None, :] + _z[:, None] * _fw[None, :]
                                _P = _P - _P.mean(0)
                                self._vp_nz = float(abs(np.linalg.svd(_P, full_matrices=False)[2][2][2]))
                                if self._vp_nz <= VP_NZ:
                                    self._vp = True
                                    self._vp_omb = _OMB()
                    except Exception:
                        self._vp = False
                        self._vp_omb = None
                if VP_ON and VP_SIG and any((abs(self._z0 - _h) < VP_SIG_TOL for _h in VP_SIG)):
                    self._vp_all = True
                    if self._vp_omb is None:
                        self._vp = True
                        self._vp_omb = _OMB()
                        self._vp_lite = True
                if VPX_ON and VP_ON and (not self._wh_spawn()) and (self._z0 < MTN_Z0):
                    self._vpx = True
                    if self._vp_omb is None:
                        self._vp = True
                        self._vp_omb = _OMB()
                        self._vp_lite = True
                self._c0 = pos[0:2] + st[-2:].astype(np.float64)
                if self._wh_spawn():
                    _g = np.arange(-12.0, 12.01, 0.5)
                    _X, _Y = np.meshgrid(_g, _g)
                    _P = np.stack([_X.ravel(), _Y.ravel()], 1)
                    _m = np.hypot(_P[:, 0] - self._c0[0], _P[:, 1] - self._c0[1]) <= 30.0
                    self._cb = _P[_m].mean(0) if _m.any() else np.clip(self._c0, -12.0, 12.0)
                    for _d in (self._rp, self._r2):
                        if _d is not None:
                            if WH_BOX:
                                _d.box = WH_GATE
                            if WH_R2T:
                                _d.min_tz = WH_R2_TZ
                            _d.whd = False
                if MTN_Z0 > 0.0 and self._z0 >= MTN_Z0:
                    self._cls.is_mountain = True
                    self._cls.z0_mtn = True
                if TK_ON and abs(self._z0 - TK_Z0) <= TK_Z0_TOL:
                    self._tk_on = True
                    self._tk_d1 = np.asarray(observation['depth'], np.float32).reshape(256, 256).copy()
                    self._tk_p1 = pos.copy()
                    self._tk_r1 = st[3:6].astype(np.float64).copy()
            if self._ss == 'rscan' and WXF_CLS:
                self._wxf_cp += 1
            else:
                self._cls.update(self.tick - self._wxf_cp, st, np.asarray(observation['depth'], np.float32))
            self._fo = FSL_ON and self._fsl_gate()
            if self._r2 is not None:
                self._r2.hb = bool(FB_DUD and self._fo)
                self._r2.fbox = FB_BOX if FB_BX and self._fo else None
            if FB_R2ON and self._fo:
                try:
                    _fc = self._cls
                    _fa = _fc.ps / max(_fc.n, 1)
                    _fi = int(np.argmax(_fa))
                    _fl = LB[_fi]
                    if not (_fl in R2_TYPES and (_fc.settled(_fl) or (RL_ON and _fl in _fc.relatched)) and (float(_fa[_fi]) >= R2_BAR)):
                        self._fb_r2n += 1
                except Exception:
                    pass
            if not self._fo:
                self._fsx = None
                self._fsx_n = 0
                self._fsx_avt = -1
                self._fsx_rp = False
                if self._plan is not None:
                    self._plan.blk = None
                self._tr = None
                self._tr_g = None
                self._tr_rk = 0
            if self._dt is not None:
                try:
                    _mt = 'mountain' if self._cls.is_mountain else self._r2_type()
                    _hit = self._dt.step(self.tick, observation, pos, st[3:6].astype(np.float64), _mt)
                    if _hit is not None:
                        _tgt = self._rp if _mt == 'mountain' else self._r2
                        if _tgt is not None:
                            _tgt.hit_t = self.tick
                            if not self.route.endswith('+dt'):
                                self.route = self.route + '+dt'
                except Exception:
                    pass
            if WH_BOX and self._cb is not None and (not self._on) and (not self._cls.is_mountain):
                _o = self._cb - pos[0:2]
                st[-2] = np.float32(_o[0])
                st[-1] = np.float32(_o[1])
            for pd, hs in self._h.items():
                if self.tick % pd == 0:
                    hs.append(pos[0:2].copy())
                    if len(hs) > 20:
                        hs.pop(0)
            _wh = self._cb is not None and self._wh()
            _sw = WH_SWEEP and _wh
            _np = WH_NOPLAN and _wh
            if not self._on and (not _np):
                t = self._stype()
                c = CFG.get(t) if t else None
                if t == 'mountain' and self._cls.mtn_ok() and (not self._champ()):
                    c = MTN_STRONG
                if E6_ON and c and (t in E1_TYPES):
                    c = (E6_T0 if E6_T0 > 0 else c[0], c[1], 1000000000.0) + tuple(c[3:])
                if c and self.tick >= c[0]:
                    hs = self._h[c[1]]
                    if len(hs) >= 20:
                        hp = np.asarray(hs)
                        if float(np.max(np.linalg.norm(hp - hp[0], axis=1))) < c[2]:
                            if c[8] and self._cv != 'ok':
                                if self._cv != 'block':
                                    rgb = observation.get('rgb') if hasattr(observation, 'get') else None
                                    fr = np.asarray(rgb, np.float32) if rgb is not None else None
                                    if fr is not None and float(fr.max()) > 0.01:
                                        self._cv = 'block' if _warm(fr) >= VW else 'ok'
                                        if self._cv == 'ok':
                                            self._on = True
                                            self._cfg = c
                                    elif self._cvt < 3:
                                        rq = True
                                        self._cv = 'req'
                                        self._cvt += 1
                            else:
                                self._on = True
                                self._cfg = c
            if _sw:
                try:
                    self._swstep(pos, st, observation, _np)
                except Exception:
                    self._swe += 1
            _plan_steer = bool(not (_np or (_sw and self._sw is not None and self._sw.armed)) and self._on and PLAN_ON and (len(self._cfg) > 9) and (self._stype() in PLAN_TYPES))
            if E1E_ON and _u19() and PLAN_ON and (not _plan_steer) and (not _sw) and (not _np):
                try:
                    if isinstance(self._plan, _Planner) and _e1r(self._plan, self._stype()):
                        self._plan = self._reprior(pos, st) if self._stype() in E1_TYPES else None
                    if self._plan is None and self._stype() in PLAN_TYPES and (self._stype() in E1_TYPES):
                        self._plan = self._mkplan(pos, st)
                    if isinstance(self._plan, _Planner):
                        self._plan.observe(pos, st[3:6].astype(np.float64), observation['depth'], float(st[162]) * 20.0 if st.size > 162 else 5.0)
                except Exception:
                    self._plan = None
            if _plan_steer:
                c = self._cfg
                try:
                    if self._plan is None or isinstance(self._plan, _Sweep):
                        if WH_BOX and self._cb is not None and (not WH19):
                            _sup, _w0, _dc = _ccore(self._c0, self._pad, boxes=np.array([16.0]))
                            if int((_w0 > 0).sum()) < 4:
                                _sup, _w0, _dc = _ccore(pos[0:2] + st[-2:].astype(np.float64), self._pad)
                            self._plan = _Planner(_sup, _w0)
                            self._plan.mtype = self._stype()
                        else:
                            self._plan = self._mkplan(pos, st)
                    elif isinstance(self._plan, _Planner) and _e1r(self._plan, self._stype()):
                        self._plan = self._reprior(pos, st)
                    _agl = float(st[162]) * 20.0 if st.size > 162 else 5.0
                    self._plan.observe(pos, st[3:6].astype(np.float64), observation['depth'], _agl)
                    _m1 = True
                    if E1G_ON and _u19() and (self._stype() in E1_TYPES):
                        _mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                        _m1 = bool(int(round(float(_mm[0]))) == 1 and (not any((_d is not None and _d.latch is not None for _d in (self._rp, self._r2)))))
                    for _det in (self._rp, self._r2):
                        if _det is not None:
                            for _t, _q in _det.hits:
                                if (_t, id(_det)) not in self._plan.seen_hits:
                                    if _m1 or not E1GD_ON:
                                        self._plan.seen_hits[_t, id(_det)] = 1
                                    if _m1:
                                        self._plan.hint(_q, PLAN_HINT)
                    _vh = float(np.hypot(float(st[6]), float(st[7])))
                    self._plan.stall = self._plan.stall + 1 if _vh < PLAN_STALL_V and _m1 else 0
                    _force = False
                    if self._plan.stall >= PLAN_STALL_N and self._plan.wp is not None:
                        self._plan.mass[np.linalg.norm(self._plan.sup - self._plan.wp, axis=1) <= 6.0] = 0.0
                        self._plan.stall = 0
                        self._plan.n_stall += 1
                        _force = True
                    if self._fsx_rp:
                        self._fsx_rp = False
                        _force = True
                    _wp = self._plan.next(pos, float(st[5]), self.tick, _force)
                    off = _wp - pos[0:2]
                    bl = c[9] if len(c) > 9 else 1.0
                    if E1B >= 0.0 and self._stype() in E1_TYPES:
                        bl = E1B
                    if bl < 1.0:
                        off = bl * off + (1.0 - bl) * (self._cb - pos[0:2] if WH_BOX and self._cb is not None else st[-2:].astype(np.float64))
                    st[-2] = np.float32(off[0])
                    st[-1] = np.float32(off[1])
                    self.route = 'king+plan'
                    self._steer = True
                    self._rv_ps = True
                except Exception:
                    self._plan = None
            elif self._on:
                c = self._cfg
                self._mp_hd = float(st[5])
                if self._w is None:
                    if len(c) > 9:
                        self._w = self._vbuild(pos, st[-2:].astype(np.float64), c)
                    else:
                        self._w = _wps(pos[0:2] + st[-2:].astype(np.float64), self._pad, pos[0:2], c[3], c[4], c[5])
                    self._i = 0
                    self._adv = self.tick
                if len(self._w):
                    self._i = min(self._i, len(self._w) - 1)
                    wp = self._w[self._i]
                    _d = float(np.linalg.norm(wp - pos[0:2]))
                    if self._mp and MP_PROG:
                        if self._mp_wp is None or float(np.linalg.norm(self._mp_wp - wp)) > 1e-06:
                            self._mp_wp = np.asarray(wp, np.float64).copy()
                            self._mp_bd = _d
                        elif _d < self._mp_bd - MP_PROG_M:
                            self._mp_bd = _d
                            self._adv = self.tick
                    if _d < c[7] or self.tick - self._adv > c[6]:
                        if len(c) > 9 and (not (E8S_ON and self._stype() == 'mountain' and (_d >= c[7]))):
                            self._swept(wp)
                        nx = self._i + 1
                        if nx >= len(self._w):
                            if len(c) > 9:
                                rb = self._vbuild(pos, st[-2:].astype(np.float64), c)
                                if rb:
                                    self._w = rb
                            nx = 0
                        self._i = nx
                        self._adv = self.tick
                        wp = self._w[self._i]
                    off = wp - pos[0:2]
                    if STALL_N > 0 and self._stype() in STALL_TYPES and (self._rp is None or self._rp.latch is None):
                        _vh = float(np.hypot(float(st[6]), float(st[7])))
                        self._stall = self._stall + 1 if _vh < STALL_V else 0
                        if self._stall >= STALL_N:
                            if self._push == 0:
                                self._stall_n += 1
                            self._push = STALL_HOLD
                            self._stall = 0
                        if self._push > 0:
                            self._push -= 1
                            j = self._i
                            while j + 1 < len(self._w) and float(np.linalg.norm(self._w[j] - pos[0:2])) < STALL_LOOK:
                                j += 1
                            if j != self._i:
                                self._i = j
                                self._adv = self.tick
                            off = self._w[j] - pos[0:2]
                            _n = float(np.linalg.norm(off))
                            if 1e-06 < _n < STALL_LOOK:
                                off = off * (STALL_LOOK / _n)
                            self._stall_on = True
                    bl = c[9] if len(c) > 9 else 1.0
                    if E8B >= 0.0 and self._stype() == 'mountain':
                        bl = E8B
                    if self._mp:
                        bl = MP_BL
                    if bl < 1.0:
                        off = bl * off + (1.0 - bl) * st[-2:].astype(np.float64)
                    st[-2] = np.float32(off[0])
                    st[-1] = np.float32(off[1])
                    self._steer = True
                    self.route = 'king+tour+stall' if getattr(self, '_stall_on', False) else 'king+tour'
                    if self._mp:
                        self.route += '+mp'
                    elif self._mp_e:
                        self.route += '+mpe'
                    self._stall_on = False
            if CM_ON and self._cm is not None and (float(self.memory_tensor[0]) < 1.5):
                lk = self._cm - pos[0:2]
                st[-2] = np.float32(lk[0])
                st[-1] = np.float32(lk[1])
                self.route = 'king+cm'
                self._steer = True
                _mm = np.asarray(self.memory_tensor, np.float32).copy()
                _mm[1] = 0.0
                _mm[2] = 0.0
                self.memory_tensor = _mm
            if self._lm is not None:
                try:
                    self._lm.fbox = FB_BOX if FB_BX and self._fo else None
                    _lq = self._lm.step(self.tick, self.memory_tensor, pos, self._stype() == 'forest', self._p3_ban, self._c0)
                except Exception:
                    _lq = None
                    self._lm.err += 1
                    self._lm.dead = True
                    self._lm.tgt = None
                if _lq is not None:
                    _lo = _lq - pos[0:2]
                    st[-2] = np.float32(_lo[0])
                    st[-1] = np.float32(_lo[1])
                    self.route = 'king+lm'
                    self._steer = True
                    _mm = np.asarray(self.memory_tensor, np.float32).copy()
                    _mm[1] = 0.0
                    _mm[2] = 0.0
                    self.memory_tensor = _mm
            self._rp_term = None
            if self._rp is not None and (not self._champ()):
                try:
                    if self._rp.clue0 is None:
                        self._rp.clue0 = pos[0:2] + np.asarray(observation['state'], np.float32).reshape(-1)[-2:].astype(np.float64)
                    if self._cls.mtn_ok():
                        self._rp.set_type('mountain')
                        self._rp.observe(self.tick, observation, pos, st[3:6].astype(np.float64))
                        if self._rp.latch is not None:
                            lk = self._rp.latch[0:2] - pos[0:2]
                            st[-2] = np.float32(lk[0])
                            st[-1] = np.float32(lk[1])
                            self.route = 'king+rgb'
                            self._steer = True
                            self._rp_term = self._rp.terminal(pos, float(st[162]) * 20.0 if st.size > 162 else 20.0)
                        self._rp.tick_end(self.tick, pos)
                except Exception:
                    self._rp_term = None
            self._r2_on = False
            self._r2_term = None
            if self._r2 is not None:
                try:
                    if not self._r2_ok():
                        raise _R2Off
                    self._r2_on = True
                    self._r2.set_type(self._r2_type())
                    self._r2.rg_ok = bool(RG_ON and self._r2_type() in RG_TYPES)
                    self._r2.tcap_k = TCAP_K if TCAP_ON and self._r2_type() in TCAP_TYPES else None
                    if self._r2.clue0 is None:
                        self._r2.clue0 = pos[0:2] + np.asarray(observation['state'], np.float32).reshape(-1)[-2:].astype(np.float64)
                    self._r2.observe(self.tick, observation, pos, st[3:6].astype(np.float64))
                    if self._r2.latch is not None:
                        lk = self._r2.latch[0:2] - pos[0:2]
                        st[-2] = np.float32(lk[0])
                        st[-1] = np.float32(lk[1])
                        self.route = 'king+r2'
                        self._steer = True
                        if R2_TERM_R > 0.0 and self._r2_type() in R2_TERM_TYPES:
                            _mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                            _ph = float(_mm[0]) if _mm.size else 1.0
                            _lt = getattr(self._r2, 'latch_t', None)
                            if (not R2_TERM_P1 or 0.5 <= _ph < 1.5) and (_lt is None or self.tick - _lt >= R2_TERM_WAIT):
                                self._r2_term = self._r2.terminal(pos, float(st[162]) * 20.0 if st.size > 162 else 20.0)
                    self._r2.tick_end(self.tick, pos)
                except _R2Off:
                    pass
                except Exception:
                    self._r2_on = False
        if LLT_ON and self._ll_on and (self._cls is not None) and self._cls.is_mountain and (self._ll_est is not None):
            if self._rp is not None and self._rp.latch is not None:
                self._ll_on = False
            else:
                _lk = self._ll_est[0:2] - st[0:2].astype(np.float64)
                st[-2] = np.float32(_lk[0])
                st[-1] = np.float32(_lk[1])
                self.route = 'king+llt'
        self._dh_term = None
        if self._dh is not None:
            try:
                _mt = 'mountain' if self._cls.is_mountain else self._stype()
                if _mt in DH_TYPES:
                    if self._dh.clue0 is None:
                        self._dh.clue0 = pos[0:2] + np.asarray(observation['state'], np.float32).reshape(-1)[-2:].astype(np.float64)
                    self._dh.observe_depth(self.tick, observation, pos, st[3:6].astype(np.float64), _mt == 'mountain')
                    if FCL_ON and _mt == 'mountain':
                        _fm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                        if _fm.size > 57 and _fm[57] > 0.5:
                            self._fcl_klk = True
                        _f = self._dh.fcl
                        if _PF_LOG and _f is not None and (int(_f[0]) == self.tick):
                            try:
                                _sa = self._dh.pf.score('A', self._dh.pf_f) if self._dh.pf is not None and self._dh.pf_f is not None else None
                            except Exception:
                                _sa = None
                            self._pf_note('form', m=[float(x) for x in _f[1]], sA=_sa, want=bool(self._dh.pf_want), klk=bool(self._fcl_klk), ph=float(_fm[0]) if _fm.size else None, rgbn=self._rgb_n)
                        if _f is not None and (self._fcl_klk or self._dh.latch is not None or self._dh.n_latch > 0 or (self.tick - _f[0] > FCL_WIN)):
                            self._dh.fcl = None
                            self._dh.pf_want = False
                        elif _f is not None and self._dh.fcl_n == 0 and self._cls.mtn_ok() and (self.tick >= FCL_T0) and (self.tick - _f[0] >= FCL_WAIT) and (_fm.size > 57) and (0.5 <= float(_fm[0]) < 1.5) and (float(_fm[57]) < 0.5) and (self._rp is None or self._rp.latch is None) and (not (self._r2_on and self._r2 is not None and (self._r2.latch is not None))):
                            _k, _md, _sc = self._dh.pf_keep(_f[1])
                            self._pf_note('decide', keep=_k, model=_md, score=_sc, f0=int(_f[0]), m=[float(x) for x in _f[1]], nrgb=self._dh.pf_nrgb, rgbn=self._rgb_n)
                            if _k:
                                self._dh.latch = np.asarray(_f[1], np.float64).copy()
                                self._dh.latch_t = self.tick
                                self._dh.near_since = None
                                self._dh.n_latch += 1
                                self._dh.hits = []
                                self._dh.fcl = None
                                self._dh.fcl_n += 1
                                self._dh.fcl_t = self.tick
                                self._fcl_n += 1
                                self._dh.pf_nkeep += 1
                            else:
                                self._dh.fcl = None
                                self._dh.pf_nrej += 1
                        elif _f is not None and self._dh.pf_want and (not (self._cls.mtn_ok() and _fm.size > 57 and (0.5 <= float(_fm[0]) < 1.5) and (float(_fm[57]) < 0.5) and (self._rp is None or self._rp.latch is None) and (not (self._r2_on and self._r2 is not None and (self._r2.latch is not None))))):
                            self._dh.pf_want = False
                    _rgbl = bool(self._rp is not None and self._rp.latch is not None and self._cls.mtn_ok() and (not self._champ()) or (self._r2_on and self._r2 is not None and (self._r2.latch is not None)))
                    _prio = bool(DH_PRIO and _mt in DH_PRIO_TYPES)
                    if _prio and _rgbl and (_mt == 'mountain') and (self._dh.latch is not None) and (self._rp is not None) and (self._rp.latch is not None) and (self._ll_est is not None) and (self.tick - self._ll_t <= TB_T):
                        _rq = np.asarray(self._rp.latch[0:2], np.float64)
                        if float(np.linalg.norm(np.asarray(self._dh.latch[0:2], np.float64) - _rq)) > TB_SEP and float(np.hypot(float(self._ll_est[0]) - _rq[0], float(self._ll_est[1]) - _rq[1])) <= TB_AGREE:
                            _prio = False
                            self._tb_n = getattr(self, '_tb_n', 0) + 1
                    _term = bool(DH_TERM and _mt in DH_TERM_TYPES)
                    if self._dh.latch is not None and (not _rgbl or _prio):
                        _mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                        _ph = float(_mm[0]) if _mm.size else 1.0
                        if not DH_P1 or 0.5 <= _ph < 1.5 or (_prio and _rgbl):
                            if DH_STEER and (not _rgbl):
                                lk = self._dh.latch[0:2] - pos[0:2]
                                st[-2] = np.float32(lk[0])
                                st[-1] = np.float32(lk[1])
                                self.route = 'king+dh'
                            if _term:
                                self._dh_term = self._dh.terminal(pos, float(st[162]) * 20.0 if st.size > 162 else 20.0)
                    self._dh.tick_end(self.tick, pos)
            except Exception:
                self._dh_term = None
        if ON and FDL_ON and (self._cls is not None):
            try:
                self._fdl_step(st, observation['state'])
            except Exception:
                pass
        if getattr(self, '_nv', None) is not None and self._cls is not None:
            try:
                self._nv_step(observation, st, pos)
            except Exception:
                self._nv.n_err += 1
        if getattr(self, '_nvf', None) is not None and self._cls is not None:
            try:
                self._nvf_step(observation, st, pos)
            except Exception:
                self._nvf.n_err += 1
        if AV_D > 0.0 and self._cls and (self.tick >= AV_T0) and (self._stype() in AV_TYPES) and (self.tick > self._fsx_avt):
            dm = np.asarray(observation['depth'], np.float32).reshape(256, 256) * 29.5 + 0.5
            ctr = float(dm[96:160, 96:160].min())
            if ctr < (AVF_D if self._fo else AV_D):
                if self._av == 0:
                    self._av_side = 1.0 if float(dm[64:192, 128:224].min()) >= float(dm[64:192, 32:128].min()) else -1.0
                    self._av_n += 1
                self._av = AV_HOLD
            elif ctr > (AVF_D if self._fo else AV_D) + AV_REL and self._av > 0:
                self._av -= 1
            if self._av > 0:
                ang = -self._av_side * np.radians(AVF_ROT if self._fo else AV_ROT)
                cx, cy = (float(st[-2]), float(st[-1]))
                ca, sa = (float(np.cos(ang)), float(np.sin(ang)))
                st[-2] = np.float32(ca * cx - sa * cy)
                st[-1] = np.float32(sa * cx + ca * cy)
                if not self.route.endswith('+av'):
                    self.route = self.route + '+av'
        if MX_ON and self._cls is not None:
            try:
                mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                if mm.size >= 63:
                    if mm[59] < 0.5:
                        self._mx_n = 0
                    elif mm[57] > 0.5 and mm[59] >= MX_AT and (self._mx_n < MX_EXT) and ('warehouse' in MX_TYPES and getattr(self, '_z0', None) is not None and (self._z0 <= WESC_Z) or ((getattr(self, '_z0', None) is None or self._z0 > WESC_Z) and self._stype() in MX_TYPES)):
                        mm = mm.copy()
                        mm[59] = np.float32(MX_AT)
                        self._mx_n += 1
                        self._mx_used += 1
                        self.memory_tensor = mm.reshape(self._ms)
            except Exception:
                pass
        if FPR_PK:
            if self._fo:
                try:
                    self._fpr_park(st)
                except Exception:
                    self._fpr_err += 1
                    self._fpr_hv = None
                    self._fpr_at = 0
                    if self._omb is not None:
                        self._omb.stats['fpr_err'] = self._fpr_err
            else:
                self._fpr_hv = None
                self._fpr_at = 0
        if P3_ON:
            try:
                mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                if mm.size >= 63:
                    rs = False
                    if mm[0] >= 2.5:
                        _pz = self._fsx is not None and self._p3_pn < P3_PAUSE
                        self._p3_pn += int(_pz)
                        self._p3_n += int(not _pz)
                        if FAU_ON and FAU_P3 and (not _pz) and (getattr(self, '_fau', None) is not None):
                            self._p3_n -= 1
                            self._fau_p3 += 1
                        if self._p3_n > (450 if self._stype() == 'city' else WH_P3_T if WH_BOX and self._cb is not None else P3_T):
                            _pb = np.asarray(mm[8:10] if FB_P3 and self._fo or (CP3H and self._stype() == 'city') else mm[60:62], np.float64).copy()
                            if FB_P3 and self._fo and (float(np.linalg.norm(_pb - np.asarray(mm[60:62], np.float64))) >= P3_BAN_R):
                                self._fb_p3 += 1
                            self._p3_ban.append(_pb)
                            rs = True
                    else:
                        self._p3_n = 0
                    if not rs and self._p3_ban and (mm[0] >= 1.5) and (mm[57] > 0.5):
                        e = np.asarray(mm[60:62], np.float64)
                        if any((float(np.linalg.norm(e - b)) < P3_BAN_R for b in self._p3_ban)):
                            rs = True
                    if rs:
                        mm = mm.copy()
                        mm[0] = 1.0
                        mm[56] = 0.0
                        mm[57] = 0.0
                        mm[58] = 0.0
                        mm[59] = 0.0
                        self.memory_tensor = mm.reshape(self._ms)
                        self._p3_n = 0
                        self._p3_resets += 1
                        self.route = self.route + '+p3r'
            except Exception:
                pass
        _rv = False
        if RV_ON and self._steer and self._rv_ps and self.route.startswith('king+plan') and (self._cls is not None) and (self._c0 is not None):
            try:
                _rm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                _rk = int(round(float(_rm[1])))
                if _rm.size >= 16 and 1 <= _rk <= 8 and (0.5 <= float(_rm[0]) < 1.5) and (self._stype() == 'village'):
                    _ru = _rm[13:15].astype(np.float64)
                    _rn = float(np.hypot(_ru[0], _ru[1]))
                    if _rn > 1e-06:
                        _ra = math.radians(_RV_ANG[_rk])
                        _ca, _sa = (math.cos(_ra), math.sin(_ra))
                        _ru = _ru / _rn
                        _rt = st[0:2].astype(np.float64) + st[-2:].astype(np.float64) + RV_OFF * np.array([_ca * _ru[0] - _sa * _ru[1], _sa * _ru[0] + _ca * _ru[1]])
                        if float(np.hypot(_rt[0] - self._c0[0], _rt[1] - self._c0[1])) > RV_R:
                            _rv = True
                            self._rv_n += 1
                            if self._rv_t0 < 0:
                                self._rv_t0 = self.tick
            except Exception:
                _rv = False
        if (E3_ON or _rv) and self._steer:
            try:
                mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                if mm.size >= 3 and (mm[1] != 0.0 or mm[2] != 0.0):
                    mm = mm.copy()
                    mm[1] = 0.0
                    mm[2] = 0.0
                    self.memory_tensor = mm.reshape(self._ms)
            except Exception:
                pass
        feed = {'depth': np.asarray(observation['depth'], np.float32).reshape(256, 256, 1), 'state': st, 'memory_tensor': self.memory_tensor}
        if 'rgb' in self._in:
            rgb = observation.get('rgb') if hasattr(observation, 'get') else None
            feed['rgb'] = np.asarray(rgb, np.float32).reshape(256, 256, 3) if rgb is not None else np.zeros((256, 256, 3), np.float32)
        self._fg_mem = np.asarray(self.memory_tensor, np.float32).reshape(-1).copy()
        _pms = self.session
        if self._psess:
            try:
                if self._cls is not None:
                    _pt = self._stype()
                    if getattr(self, '_tk_on', False) and _pt != 'warehouse':
                        _pt = None
                    if _pt in self._psess:
                        _pms = self._psess[_pt]
                        self._pmap_n += 1
            except Exception:
                _pms = self.session
        a, m = _pms.run(['action', 'memory_tensor_out'], feed)
        self.memory_tensor = np.asarray(m, np.float32).reshape(self._ms)
        out = np.asarray(a, np.float32).reshape(-1)
        self._tr_raw = out.copy()
        if VRV_ON and self._cls is not None and (self._fg_mem is not None):
            try:
                self._vrv(observation)
            except Exception:
                self._vrv_err += 1
        if RLK_ON and self._rlk is not None and (out.shape[0] >= 5):
            try:
                out = self._rlk_step(st, out)
            except Exception:
                self._rlk = None
                self._rlk_sup = False
                self._rlk_err += 1
        if M3L_ON and self._cls is not None and (self._fg_mem is not None):
            try:
                _vo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                if _vo.size >= 65 and float(self._fg_mem[0]) < 2.5 and (float(_vo[0]) >= 2.5) and (not self._cls.is_mountain) and (not self._chull_wh()) and (not (M3L_XO and 'open' in self._cls.latched)) and (self._stype() == 'city'):
                    _hz = float(_vo[16:46].reshape(10, 3)[:, 2].astype(np.float64).mean())
                    if float(_vo[10]) - _hz < M3L_LOW:
                        _vo = _vo.copy()
                        _vo[10] = np.float32(_hz + M3L_DZ)
                        self.memory_tensor = _vo.reshape(self._ms)
                        self._m3l_n += 1
                        if self._omb is not None:
                            self._omb.stats['m3l_n'] = self._m3l_n
            except Exception:
                pass
        if VF_ON and self._cls is not None and (self._fg_mem is not None):
            try:
                _vo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                if _vo.size >= 65 and float(self._fg_mem[0]) < 2.5 and (float(_vo[0]) >= 2.5):
                    _hz = float(_vo[16:46].reshape(10, 3)[:, 2].astype(np.float64).mean())
                    if float(_vo[10]) - _hz < VF_LOW and self._stype() == 'village' and self._vp_all:
                        _vo = _vo.copy()
                        _vo[10] = np.float32(_hz + VF_DZ)
                        self.memory_tensor = _vo.reshape(self._ms)
                        self._vf_n += 1
            except Exception:
                pass
        if LA_ON and self._cls is not None:
            try:
                self._lassist(observation)
            except Exception:
                pass
        if CM_ON and self._cls is not None:
            try:
                self._cm_update(st)
            except Exception:
                self._cm = None
        if WH_BOX and self._cb is not None:
            try:
                _mm = self.memory_tensor
                _far = _mm[57] > 0.5 and max(abs(float(_mm[60])), abs(float(_mm[61]))) > WH_GATE or (_mm[0] >= 2.5 and max(abs(float(_mm[8])), abs(float(_mm[9]))) > WH_GATE)
                _r2l = self._r2.latch if self._r2 is not None and self._r2_on else None
                if not _far and _r2l is not None and (_mm[57] > 0.5) and (_mm[0] < 2.5):
                    _p2 = st[0:2].astype(np.float64)
                    _far = float(np.hypot(*np.asarray(_r2l[0:2], np.float64) - _p2)) <= R2_TERM_R + 2.0 and float(np.hypot(float(_mm[60]) - float(_r2l[0]), float(_mm[61]) - float(_r2l[1]))) > 6.0
                if _far:
                    _mm = _mm.copy()
                    self._p3_ban.append(np.asarray(_mm[60:62], np.float64).copy())
                    _mm[56:60] = 0.0
                    if _mm[0] >= 2.5:
                        _mm[0] = 1.0
                    elif _mm[0] >= 1.5:
                        _mm[0] = 0.0 if self._fg_mem is not None and float(self._fg_mem[0]) < 0.5 else 1.0
                    self.memory_tensor = _mm
                    self._whgate_n += 1
                    self._p3_n = 0
            except Exception:
                pass
        if rq and out.shape[0] >= 6:
            out = out.copy()
            out[5] = 1.0
        if DG_ON and self._cls is not None and (not self._cls.is_mountain) and (getattr(self, '_c0', None) is not None) and (self._stype() == 'city'):
            try:
                _gm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                if _gm.size >= 63:
                    _gc = np.asarray(self._c0, np.float64)
                    if _gm[0] < 2.5 and _gm[57] > 0.5 and (float(np.hypot(float(_gm[60]) - _gc[0], float(_gm[61]) - _gc[1])) > DG_R) or (_gm[0] >= 2.5 and float(np.hypot(float(_gm[8]) - _gc[0], float(_gm[9]) - _gc[1])) > DG_R):
                        _gm = _gm.copy()
                        _gm[56:60] = 0.0
                        if _gm[0] >= 2.5:
                            _gm[0] = 1.0
                        elif _gm[0] >= 1.5:
                            _gm[0] = 0.0 if self._fg_mem is not None and float(self._fg_mem[0]) < 0.5 else 1.0
                        self.memory_tensor = _gm.reshape(self._ms)
                        self._f86_dg = getattr(self, '_f86_dg', 0) + 1
            except Exception:
                pass
        if FB_BX and self._fo:
            try:
                _fm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                if _fm.size >= 63 and (_fm[57] > 0.5 and max(abs(float(_fm[60])), abs(float(_fm[61]))) > FB_BOX or (_fm[0] >= 2.5 and max(abs(float(_fm[8])), abs(float(_fm[9]))) > FB_BOX)):
                    _fm = _fm.copy()
                    _fm[56:60] = 0.0
                    if _fm[0] >= 2.5:
                        _fm[0] = 1.0
                    elif _fm[0] >= 1.5:
                        _fm[0] = 0.0 if self._fg_mem is not None and float(self._fg_mem[0]) < 0.5 else 1.0
                    self.memory_tensor = _fm.reshape(self._ms)
                    self._fb_bx += 1
            except Exception:
                pass
        if FPR_CG and self._fo and (self._c0 is not None):
            try:
                _fm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                if _fm.size >= 63 and (_fm[57] > 0.5 and _fm[0] < 2.5 and (float(np.hypot(float(_fm[60]) - self._c0[0], float(_fm[61]) - self._c0[1])) > FPR_CGR) or (_fm[0] >= 2.5 and float(np.hypot(float(_fm[8]) - self._c0[0], float(_fm[9]) - self._c0[1])) > FPR_CGR)):
                    _fm = _fm.copy()
                    _fm[56:60] = 0.0
                    if _fm[0] >= 2.5:
                        _fm[0] = 1.0
                    elif _fm[0] >= 1.5:
                        _fm[0] = 0.0 if self._fg_mem is not None and float(self._fg_mem[0]) < 0.5 else 1.0
                    self.memory_tensor = _fm.reshape(self._ms)
                    self._fpr_cg += 1
                    if not self.route.endswith('+fpr'):
                        self.route = self.route + '+fpr'
                    if self._omb is not None:
                        self._omb.stats['fpr_cg'] = self._fpr_cg
            except Exception:
                self._fpr_err += 1
                if self._omb is not None:
                    self._omb.stats['fpr_err'] = self._fpr_err
        if M_FARLOCK and self._cls is not None and self._cls.is_mountain and (self._c0 is not None):
            try:
                _mm = self.memory_tensor
                if _mm[57] > 0.5 and float(np.hypot(float(_mm[60]) - self._c0[0], float(_mm[61]) - self._c0[1])) > FARLOCK_R and (_mm[0] < 2.5):
                    _mm = _mm.copy()
                    _mm[56:60] = 0.0
                    if 1.5 <= _mm[0] < 2.5:
                        _mm[0] = 0.0 if self._fg_mem is not None and float(self._fg_mem[0]) < 0.5 else 1.0
                    self.memory_tensor = _mm
                    self._farlock_n += 1
                    if not self.route.endswith('+far'):
                        self.route = self.route + '+far'
            except Exception:
                pass
        if LLT_ON and self._cls is not None and self._cls.is_mountain and (self._c0 is not None):
            try:
                _m = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                _lkd = bool(_m[57] > 0.5)
                _p = st[0:3].astype(np.float64)
                if _lkd:
                    _e = np.asarray(_m[60:63], np.float64)
                    if float(np.hypot(_e[0] - self._c0[0], _e[1] - self._c0[1])) <= LLT_R and (not any((float(np.hypot(_e[0] - b[0], _e[1] - b[1])) < LLT_BAN for b in self._ll_ban))):
                        if self._ll_est is None or float(np.hypot(_e[0] - self._ll_est[0], _e[1] - self._ll_est[1])) > 4.0:
                            self._ll_est = _e.copy()
                            self._ll_n = 1
                        else:
                            _k = min(self._ll_n, 50)
                            self._ll_est = (self._ll_est * _k + _e) / (_k + 1)
                            self._ll_n += 1
                        self._ll_t = self.tick
                        _fw, _up, _rt = _axes(st[3:6])
                        _cam = _p + _fw * _CAMF + _up * _CAMU
                        _d = self._ll_est - _cam
                        _x = float(_d @ _fw)
                        self._ll_v = float(_d @ _up) / _x if _x > 0.5 else -9.0
                        if self._ll_on:
                            self._ll_on = False
                elif self._ll_prevlk and (not self._ll_on) and (self._ll_est is not None) and (self._ll_n >= LLT_N) and (0.5 <= float(_m[0]) < 1.5) and (self.tick - self._ll_t <= 2) and (self._ll_v < LLT_V):
                    self._ll_on = True
                    self._ll_t0 = self.tick
                    self._ll_zt = None
                    self._ll_hold = 0
                    self._ll_trig += 1
                self._ll_prevlk = _lkd
            except Exception:
                self._ll_on = False
        _fo = self._fo
        if _fo and R2A_ON:
            try:
                self._r2a(st)
            except Exception:
                self._fsl_err += 1
                self.route = self.route + '+fse'
        if rq and out.shape[0] >= 6:
            out = out.copy()
            out[5] = 1.0
        if self._esc is not None and out.shape[0] >= 4 and (self._esc.mode == 'ceil' or not self._champ()):
            try:
                _p = st[0:3].astype(np.float64)
                _agl = float(st[162]) * 20.0 if st.size > 162 else 20.0
                if self.tick == 1:
                    _cv = bool(WESC_ON and self._esc.ceiling(observation['depth'], _p, float(st[5])))
                    if _cv and WX_ON and WX_CEIL and self._tk_on and self._wx_veto(observation, st):
                        self._esc.reset()
                        self._wx_vetoed = True
                    elif not _cv:
                        self._esc.decide(observation['depth'], _p, float(st[5]))
                _eok = True
                if ESC_TERR and self._esc.mode != 'ceil':
                    _c2 = self._cls
                    _eok = bool(_c2 is not None and (_c2.is_mountain or _c2.settled('forest')))
                if _eok and self._esc.phase in ('scan', 'creep', 'climb'):
                    _e = self._esc.act(observation['depth'], _p, _agl, math.degrees(float(st[4])), float(st[5]))
                    if _e is not None:
                        out = out.copy()
                        out[0:3] = _e[0]
                        out[3] = np.float32(_e[1])
                        out[4] = np.float32(0.0 if _e[2] is None else _e[2])
                        if SH_ON:
                            self._esc_w = self.tick
                        if self._esc.mode == 'ceil':
                            self.route = 'king+wesc'
            except Exception:
                pass
        if SS_ON and out.shape[0] >= 5:
            try:
                _o = self._sscan(observation, st, out)
                if _o is not None:
                    return _o
            except Exception:
                self._ss = 'done'
        if self._rp is not None and out.shape[0] >= 6 and (self._cls is not None) and self._cls.mtn_ok() and (not self._champ()):
            out = out.copy()
            if self._rp_term is not None:
                out[0:3] = self._rp_term[0]
                out[3] = np.float32(self._rp_term[1])
                out[4] = np.float32(0.0)
            if self._rp.want_rgb(self.tick, paced_ok=self._paced_ok(self._rp, st)):
                out[5] = np.float32(1.0)
        if self._r2 is not None and out.shape[0] >= 6 and self._r2_on:
            if getattr(self, '_r2_term', None) is not None:
                _use = True
                _spd = float(self._r2_term[1])
                _yt = 0.0
                _dirv = np.asarray(self._r2_term[0], np.float32).copy()
                try:
                    if R2_FACE and (self._r2_type() in R2_FACE_TYPES or (WH_R2T and self._cb is not None)) and (self._r2.latch is not None):
                        _lx = float(self._r2.latch[0]) - float(st[0])
                        _ly = float(self._r2.latch[1]) - float(st[1])
                        _dh = math.hypot(_lx, _ly)
                        if _dh > 1.0:
                            _hd = math.atan2(_ly, _lx)
                            _yt = _hd
                            if abs(_wrap(_hd - float(st[5]))) > math.radians(30.0):
                                _spd = min(_spd, 0.03)
                            else:
                                _dm = np.asarray(observation['depth'], np.float32).reshape(256, 256) * 29.5 + 0.5
                                if float(_dm[110:146, 112:144].min()) < min(_dh, R2_FACE_CLR):
                                    _use = False
                        else:
                            _yt = float(st[5])
                except Exception:
                    _use = True
                    _spd = float(self._r2_term[1])
                    _yt = 0.0
                if _use and WH_R2T and (self._cb is not None) and (st.size > 162):
                    _spd = max(_spd, float(st[159]) - 0.05)
                if _use:
                    out = out.copy()
                    out[0:3] = _dirv
                    out[3] = np.float32(_spd)
                    out[4] = np.float32(_wrap(_yt) / math.pi if _yt != 0.0 else 0.0)
                    if not self.route.endswith('+r2t'):
                        self.route = self.route + '+r2t'
            _nc = bool(_fo and R2NC_ON and self._r2nc(st))
            _nc0 = self._r2.n_nc
            if self._r2.want_rgb(self.tick, nochase=_nc, paced_ok=self._paced_ok(self._r2, st)):
                out = out.copy()
                out[5] = np.float32(1.0)
                if VNT_ON:
                    self._vnt_req = -10 ** 9
            elif VNT_ON and self.tick - getattr(self, '_vnt_req', -10 ** 9) <= 2 and (self._r2.sess is not None) and (self._r2.latch is None) and (self._r2.n_req < self._r2.cap) and (self.tick - self._r2.last_req >= VNT_GAP) and (self.tick - getattr(self, '_vnt_last', -10 ** 9) >= VNT_GAP):
                self._r2.n_req += 1
                self._vnt_req = -10 ** 9
                self._vnt_last = self.tick
                self._vnt_f += 1
                if not VNT_KEEPCLK:
                    self._r2.last_req = self.tick
                out = out.copy()
                out[5] = np.float32(1.0)
                if not self.route.endswith('+vnt'):
                    self.route = self.route + '+vnt'
            if self._r2.n_nc != _nc0:
                self.route = self.route + '+r2nc'
        if getattr(self, '_dh_term', None) is not None and out.shape[0] >= 4:
            self._dh_n = getattr(self, '_dh_n', 0) + 1
            a = 1.0 if DH_RAMP <= 0 else min(1.0, float(self._dh_n) / float(DH_RAMP))
            _d = np.asarray(self._dh_term[0], np.float64)
            _s = float(self._dh_term[1])
            _pd = np.asarray(out[0:3], np.float64)
            _pn = float(np.linalg.norm(_pd))
            if _pn > 1e-06 and a < 1.0:
                _d = (1.0 - a) * (_pd / _pn) + a * _d
                _n = float(np.linalg.norm(_d))
                _d = _d / _n if _n > 1e-06 else np.asarray(self._dh_term[0], np.float64)
                _s = (1.0 - a) * float(out[3]) + a * _s
            out = out.copy()
            out[0:3] = _d.astype(np.float32)
            out[3] = np.float32(_s)
            out[4] = np.float32(0.0)
            if not self.route.endswith('+dht'):
                self.route = self.route + '+dht'
        else:
            self._dh_n = 0
        if FG_ON and self._cls is not None and (out.shape[0] >= 4) and (self._fg_mem is not None) and (self._stype() == 'forest') and (not (_fo and 3 in FSL_PH)):
            try:
                out = self._fguard(observation, st, out)
            except Exception:
                self._fg_on = 0
                self._fg_h = None
        if LLT_ON and self._ll_on and (self._ll_est is not None) and (out.shape[0] >= 5):
            try:
                _p = st[0:3].astype(np.float64)
                _agl = float(st[162]) * 20.0 if st.size > 162 else 20.0
                _e = self._ll_est
                _dx = float(_e[0] - _p[0])
                _dy = float(_e[1] - _p[1])
                _dh = math.hypot(_dx, _dy)
                if self.tick - self._ll_t0 > LLT_MAXT:
                    raise _LLDone
                if self._ll_zt is None and _dh < LLT_RING:
                    self._ll_zt = float(_e[2]) + LLT_ZHI
                if self._ll_zt is None:
                    _zt = float(_e[2]) + LLT_ZHI
                    _dz = _zt - float(_p[2])
                    if _dz < 0 and _agl < LLT_AGL:
                        _dz = 0.3
                    _dm = np.asarray(observation['depth'], np.float32).reshape(256, 256) * 29.5 + 0.5
                    _dzc = max(-max(1.5, 0.5 * (_agl - 4.0)), min(1.5, _dz))
                    if float(_dm[96:160, 96:160].min()) < 4.0:
                        _dzc = 1.5
                    _vec = np.array([_dx, _dy, _dzc], np.float64)
                    _spd = min(0.6, max(0.15, 0.08 * math.hypot(_dh, _dz) + 0.1))
                else:
                    if float(_p[2]) <= self._ll_zt + 0.6:
                        self._ll_zt = max(float(_e[2]) + LLT_ZLO, self._ll_zt - LLT_VS * SIM_DT)
                    if self._ll_zt <= float(_e[2]) + LLT_ZLO + 1e-06:
                        self._ll_hold += 1
                        if self._ll_hold > LLT_HOLD:
                            self._ll_ban.append(np.asarray(_e[0:2], np.float64).copy())
                            raise _LLDone
                    _dz = self._ll_zt - float(_p[2])
                    if _dz < 0 and _agl < LLT_AGL:
                        _dz = 0.3
                    _vec = np.array([_dx, _dy, _dz], np.float64)
                    _spd = min(0.6, max(0.03, 0.3 * float(np.linalg.norm(_vec))))
                _n = float(np.linalg.norm(_vec))
                _dir = _vec / _n if _n > 1e-06 else np.zeros(3)
                out = out.copy()
                out[0:3] = _dir.astype(np.float32)
                out[3] = np.float32(_spd)
                out[4] = np.float32(math.atan2(_dy, _dx) / math.pi) if _dh > 1.5 else np.float32(float(st[5]) / math.pi)
            except _LLDone:
                self._ll_on = False
                self._ll_est = None
                self._ll_n = 0
            except Exception:
                self._ll_on = False
        if M_AGLF and self._cls is not None and self._cls.is_mountain and (out.shape[0] >= 4) and (self._fg_mem is not None):
            try:
                if int(round(float(self._fg_mem[0]))) == 2 or int(round(float(self.memory_tensor[0]))) == 2:
                    _agl = float(st[162]) * 20.0 if st.size > 162 else 20.0
                    if _agl < M_AGLF_HI and float(out[2]) < 0.0:
                        out = out.copy()
                        out[2] = np.float32(float(out[2]) * max(0.0, (_agl - M_AGLF_LO) / (M_AGLF_HI - M_AGLF_LO)))
                        self._aglf_n += 1
                    if _agl < M_AGLF_LO:
                        out = out.copy()
                        out[2] = np.float32(max(float(out[2]), 0.35))
            except Exception:
                pass
        if F86_ESC and self._esc is not None and (self._esc.mode != 'ceil') and (self._esc.phase in ('scan', 'creep', 'climb')) and (self._esc.n == 0) and (self._ss in ('off', 'done')):
            try:
                if self._wh_spawn() and self._cls is not None and (not self._cls.is_mountain) and (not self._cls.settled('forest')):
                    self._esc.phase = 'done'
                    self._f86_a = getattr(self, '_f86_a', 0) + 1
            except Exception:
                pass
        _whh = False
        if WH_HOLD and TKX_ON and self._tk_on and (not self._wh()):
            try:
                _whh = bool(self._cls is not None and (not self._cls.is_mountain) and (self._stype() is None))
            except Exception:
                _whh = False
        if WH_HOLD and (self._wh() or _whh) and (out.shape[0] >= 4):
            try:
                if _whh:
                    self._whh_n += 1
                _busy = self._ss not in ('off', 'done') or (self._esc is not None and self._esc.phase in ('scan', 'creep', 'climb'))
                _zt = self._z0 + WH_HOLD_DZ
                if not _busy:
                    if not self._wh_high:
                        if float(st[2]) >= _zt:
                            self._wh_high = True
                        else:
                            out = out.copy()
                            out[0:3] = np.array([0.0, 0.0, 1.0], np.float32)
                            out[3] = np.float32(0.7)
                            if WCR_ON and self._tk_on and (self._wcr_ph not in ('done', 'abort')):
                                try:
                                    _o = self._wcr(observation, st, out)
                                    if _o is not None:
                                        out = _o
                                except Exception:
                                    self._wcr_err += 1
                                    self._wcr_ph = 'abort'
                                    self._wcr_why = 'err'
                                    self._wcr_te = self.tick
                                    if self._omb is not None:
                                        self._omb.stats['wcr_err'] = int(self._wcr_err)
                                        self._omb.stats['wcr_why'] = 'err'
                            if F86_CRN and self._tk_on and (float(st[2]) >= CRN_Z - 0.8):
                                _ce = _crn_exit(float(st[0]), float(st[1]))
                                if _ce is not None:
                                    self._f86_d = getattr(self, '_f86_d', 0) + 1
                                    if float(st[2]) >= CRN_Z:
                                        out[0:3] = np.array([_ce[0], _ce[1], 0.0], np.float32)
                                        out[3] = np.float32(CRN_V)
                                    else:
                                        out[3] = np.float32(0.25)
                    elif int(round(float(self.memory_tensor[0]))) == 1 and float(st[2]) < _zt and (float(out[2]) < 0.25):
                        out = out.copy()
                        out[2] = np.float32(0.25)
            except Exception:
                pass
        if WHK_ON and out.shape[0] >= 4 and (self._cb is not None) and self._wh_spawn() and (self._cls is not None) and (not self._cls.is_mountain):
            try:
                out = self._whk(observation, st, out)
            except Exception:
                self._whk_err += 1
                self._whk_q = []
                if self._omb is not None:
                    self._omb.stats['whk_err'] = int(self._whk_err)
        if WHD_ON and self._tk_on and (out.shape[0] >= 4) and (self._cls is not None) and (not self._cls.is_mountain):
            try:
                out = self._whd(observation, st, out)
            except Exception:
                self._whd_err += 1
                self._whd_q = []
                self._whd_th = 0.0
            if self._omb is not None:
                self._omb.stats['whd_n'] = int(self._whd_n)
                self._omb.stats['whd_ev'] = int(self._whd_ev)
                self._omb.stats['whd_t1'] = int(self._whd_t1)
                self._omb.stats['whd_amax'] = int(round(self._whd_amax))
                self._omb.stats['whd_err'] = int(self._whd_err)
                if WHD_LOG:
                    self._omb.stats['whd_log'] = self._whd_lg
        _chl_o0 = out if CHL_EMAX > 0.0 and getattr(self, '_chull_mem', None) is not None and (out.shape[0] >= 4) else None
        if CHL_ON and getattr(self, '_chull_mem', None) is not None and (self._cls is not None) and (out.shape[0] >= 4):
            try:
                out = self._chull(st, out)
            except Exception:
                pass
        if CMK_ON and getattr(self, '_chull_mem', None) is not None and (self._cls is not None) and (out.shape[0] >= 4):
            try:
                out = self._cmk(st, out)
            except Exception:
                pass
        if NRC_ON and CMK_ON and (getattr(self, '_chull_mem', None) is not None) and (self._cls is not None) and (out.shape[0] >= 4):
            try:
                out = self._nrc(st, out)
            except Exception:
                self._nrc_err += 1
        if _chl_o0 is not None and out is not _chl_o0:
            try:
                out = self._chl_emax(st, _chl_o0, out)
            except Exception:
                pass
        if P3F_ON and self._cb is not None and self._wh_spawn() and (out.shape[0] >= 4) and (self._r2 is not None) and (self._r2.latch is not None):
            try:
                _pm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                if _pm.size and float(_pm[0]) >= 1.5 and (float(st[2]) < P3F_Z) and (float(out[2]) < 0.0):
                    if float(np.hypot(float(self._r2.latch[0]) - float(st[0]), float(self._r2.latch[1]) - float(st[1]))) > P3F_R:
                        out = out.copy()
                        out[2] = np.float32(0.0)
                        self._p3f_n = getattr(self, '_p3f_n', 0) + 1
                        if not self.route.endswith('+p3f'):
                            self.route = self.route + '+p3f'
            except Exception:
                pass
        if PG_ON and self._cls is not None and (out.shape[0] >= 4):
            try:
                out = self._pguard(observation, st, out)
            except Exception:
                pass
        if NG_ON and self._cls is not None and (out.shape[0] >= 4) and (self._stype() == 'city'):
            try:
                out = self._nguard(observation, st, out)
            except Exception:
                self._ng_b = None
                self._ng_hold = 0
        _fau_o = None
        if FAU_ON:
            if self._omb is not None:
                self._omb.rs_ovr = None
            if _fo and out.shape[0] >= 5:
                try:
                    _fau_o = self._fau_step(st, out)
                except Exception:
                    self._fau = None
                    self._fau_w = None
                    self._fau_err += 1
                    _fau_o = None
                    self._fau_cool = self.tick + FAU_COOL
                if _fau_o is not None:
                    out = _fau_o
                    if self._omb is not None:
                        self._omb.rs_ovr = FAU_RS
                    try:
                        self._fsl_update(observation['depth'], st[0:3].astype(np.float64), st[3:6].astype(np.float64))
                        self._fsx = None
                        self._fsx_n = 0
                    except Exception:
                        self._fsl_err += 1
                self._fau_stat()
            elif getattr(self, '_fau', None) is not None:
                self._fau_end('gate')
                self._fau_stat()
        if _fo and TR_ON and (out.shape[0] >= 5) and (_fau_o is None):
            try:
                out = self._trt(st, out)
            except Exception:
                self._tr = None
                self._tr_g = None
                self._fsl_err += 1
        if _fo and out.shape[0] >= 5 and (_fau_o is None):
            try:
                out = self._fsl(observation, st, out)
            except Exception:
                self._fsx = None
                self._fsl_err += 1
                self.route = self.route + '+fse'
        if VPF_ON and (not self._vpf_done) and (out.shape[0] >= 4) and (self._cls is not None):
            try:
                out = self._vpf_step(observation, st, out)
            except Exception:
                self._vpf_on = False
                self._vpf_done = True
                self._vpf_why = 9
                self._vpf_err += 1
                try:
                    self._vpf_stat()
                except Exception:
                    pass
        if DIB_ON and _fo and (out.shape[0] >= 4) and (self._r2 is not None) and self._r2_on and (self._r2.latch is not None):
            try:
                out = self._dib(st, out)
            except Exception:
                self._dib_on = False
                self._dib_err += 1
        elif self._dib_L is not None:
            self._dib_on = False
            self._dib_L = None
            self._dib_w = 0
            self._dib_arm = False
        if HRZ_ON and _fo and (out.shape[0] >= 4) and (self._r2 is not None) and self._r2_on and (self._r2.latch is not None):
            try:
                out = self._hrz(st, out)
            except Exception:
                self._hrz_arm = False
                self._hrz_w = 0
                self._hrz_err += 1
            if HRX_LOG and (not (self._hrx_done and self._hrb_done)):
                try:
                    self._hrx_log(st)
                except Exception:
                    self._hrx_done = True
                    self._hrb_done = True
        elif self._hrz_L is not None:
            self._hrz_u = 0
            self._hrz_L = None
            self._hrz_w = 0
            self._hrz_arm = False
            self._hrz_act = 0
            self._hrz_hold = 0
            self._hrx_L = None
            self._hrx_w = 0
            self._hrb_w = 0
        if self._omb is not None and self._cls is not None and (out.shape[0] >= 4):
            try:
                _t = self._stype()
                if _t in OMB_TYPES:
                    _p = st[0:3].astype(np.float64)
                    _r = st[3:6].astype(np.float64)
                    _gz = float(_p[2]) - float(st[162]) * 20.0 if st.size > 162 else None
                    self._omb.observe(observation['depth'], _p, _r, _gz)
                    self._omb.stats['ticks'] += 1
                    _esc = self._esc is not None and self._esc.phase in ('scan', 'creep', 'climb')
                    if not _esc:
                        _o = out.copy()
                        if self._omb.constrain(_o, _p, _r, _t):
                            out = _o
                            if not self.route.endswith('+omb'):
                                self.route = self.route + '+omb'
            except Exception:
                pass
        if VTO_ON and (not self._vto_done) and (out.shape[0] >= 4):
            try:
                out = self._vto_step(observation, st, out)
            except Exception:
                self._vto_done = True
                self._vto_T = None
                self._vto_err += 1
            if self._omb is not None:
                self._omb.stats['vto_arm'] = self._vto_arm
                self._omb.stats['vto_n'] = self._vto_n
                self._omb.stats['vto_r'] = self._vto_r
                self._omb.stats['vto_err'] = self._vto_err
        if VNT_ON and self._omb is not None:
            self._omb.stats['vnt_n'] = getattr(self, '_vnt_n', 0)
            self._omb.stats['vnt_f'] = getattr(self, '_vnt_f', 0)
            self._omb.stats['vnt_h'] = getattr(self, '_vnt_h', 0)
            self._omb.stats['vnt_al'] = int(getattr(self._r2, 'n_vnt_al', 0)) if self._r2 is not None else 0
            self._omb.stats['vnt_err'] = getattr(self, '_vnt_err', 0) + (int(getattr(self._r2, 'n_vnt_err', 0)) if self._r2 is not None else 0)
        if (VWH_ON or VPX_ON) and self._omb is not None:
            self._omb.stats['vwh_n'] = self._vwh_n
            self._omb.stats['vpx'] = int(self._vpx)
            self._omb.stats['vp_n'] = self._vp_n
        if VPC_ON and self._omb is not None:
            self._omb.stats['vpc_n'] = self._vpc_n
            self._omb.stats['vpc_k'] = self._vpc_k
        if self._vp and self._vp_omb is not None and (out.shape[0] >= 4):
            try:
                if self.tick > VP_T and (not self._vp_all) and (not self._vpx):
                    self._vp = False
                    self._vp_omb = None
                elif self._stype() in VP_TYPES:
                    _p = st[0:3].astype(np.float64)
                    _r = st[3:6].astype(np.float64)
                    _gz = float(_p[2]) - float(st[162]) * 20.0 if st.size > 162 else None
                    self._vp_omb.observe(observation['depth'], _p, _r, _gz)
                    _o = self._vpc_pre(st, out, _p) if VPC_ON else out.copy()
                    if (self.tick <= VP_T and (not self._vp_lite) or self._stype() == 'village') and self._vp_omb.constrain(_o, _p, _r, 'village', not (self._vp_lite or self.tick > VP_T)):
                        out = _o
                        self._vp_n += 1
                        if not self.route.endswith('+vp'):
                            self.route = self.route + '+vp'
                    if VPC_ON:
                        out = self._vpc_post(st, out, _p)
            except Exception:
                self._vp = False
                self._vp_omb = None
        if self._tk_on and TK_PLATE and (not self._tk_gdone) and (out.shape[0] >= 4):
            try:
                out = self._tk_guard(observation, st, out)
            except Exception:
                self._tk_gdone = True
        if self._sh is not None and out.shape[0] >= 4:
            try:
                out = self._shield(st, out)
            except Exception:
                pass
        if F86_P3G and out.shape[0] >= 4 and getattr(self, '_wh_high', False):
            try:
                if self._wh_spawn() and self._cls is not None and (not self._cls.is_mountain) and (float(st[2]) < P3G_Z) and (float(out[2]) < 0.0):
                    _pm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                    if _pm.size and float(_pm[0]) >= 1.5:
                        out = out.copy()
                        out[2] = np.float32(0.0)
                        self._f86_c = getattr(self, '_f86_c', 0) + 1
            except Exception:
                pass
        if self._dh is not None and self._dh.pf_want:
            self._dh.pf_want = False
            try:
                if out.shape[0] >= 6 and self._rgb_n < 40 and (self._dh.fcl is not None) and (self.tick - self._dh.pf_t <= 2):
                    self._pf_only = bool(float(out[5]) <= 0.5)
                    out = out.copy()
                    out[5] = np.float32(1.0)
                    self._pf_rgb_t = self.tick
                    self._dh.pf_nrgb += 1
                    self._pf_note('rgbreq', only=self._pf_only, rgbn=self._rgb_n, f0=int(self._dh.fcl[0]))
            except Exception:
                self._pf_rgb_t = None
                self._pf_only = False
        if VRV_ON and self._vrv_sup > 0 and (out.shape[0] >= 6):
            try:
                self._vrv_sup -= 1
                _vm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                if not (1.5 <= float(_vm[0]) < 2.5 and float(_vm[57]) > 0.5) or self._vrv_sup <= 0:
                    if float(_vm[0]) >= 2.5:
                        self._vrv_cf += 1
                    self._vrv_sup = 0
                    if self._vrv_m63 is not None:
                        _vm = _vm.copy()
                        _vm[63] = np.float32(self._vrv_m63)
                        self.memory_tensor = _vm.reshape(self._ms)
                        self._vrv_m63 = None
                elif float(out[5]) > 0.5:
                    out = out.copy()
                    out[5] = np.float32(0.0)
                    self._vrv_supn += 1
                if self._omb is not None:
                    self._omb.stats['vrv_sup'] = self._vrv_supn
                    self._omb.stats['vrv_cf'] = self._vrv_cf
            except Exception:
                self._vrv_sup = 0
                self._vrv_err += 1
                try:
                    if self._vrv_m63 is not None:
                        _vm = np.asarray(self.memory_tensor, np.float32).reshape(-1).copy()
                        _vm[63] = np.float32(self._vrv_m63)
                        self.memory_tensor = _vm.reshape(self._ms)
                except Exception:
                    pass
                self._vrv_m63 = None
        if RLK_ON and self._rlk_sup:
            self._rlk_sup = False
            try:
                if out.shape[0] >= 6:
                    out = out.copy()
                    out[5] = np.float32(0.0)
                if self._rlk_m63 is not None:
                    _rm = np.asarray(self.memory_tensor, np.float32).reshape(-1).copy()
                    _rm[63] = np.float32(self._rlk_m63)
                    self.memory_tensor = _rm.reshape(self._ms)
            except Exception:
                self._rlk_err += 1
        if RTY_ON and (not self._rty_done) and (out.shape[0] >= 6):
            try:
                out = self._rty_want(out)
            except Exception:
                self._rty_err += 1
                self._rty_done = True
                self._rty_req = -1
                self._rty_only = False
        if out.shape[0] >= 6 and float(out[5]) > 0.5 and (self._rgb_n < 40):
            self._rgb_n += 1
        _s = self._sw
        if _s is not None or self._swe:
            self.route = 'sw%d%d_%d_%d_%d|' % (_s is not None and _s.armed, _s is not None and _s.st, 0 if _s is None else _s.n_pass, -1 if _s is None or _s.t_arm is None else _s.t_arm, self._swe) + self.route
        return out

    def _rty_want(self, out):
        if self.tick > RTY_TEND or self._rty_n >= RTY_N:
            self._rty_done = True
            self._rty_stat()
            return out
        if self.tick < RTY_T0 or self.tick - self._rty_last < RTY_DT or self._rgb_n >= 40 - RTY_RES:
            return out
        c = self._cls
        if c is None or c.is_mountain or self._pad is None or self._wh_spawn() or (self._vrv_sup > 0) or self._rlk_sup:
            return out
        if (self._stype() or 'None') not in RTY_FROM:
            return out
        self._rty_req = self.tick
        self._rty_last = self.tick
        self._rty_n += 1
        if float(out[5]) > 0.5:
            self._rty_only = False
            self._rty_stat()
            return out
        self._rty_only = True
        o = out.copy()
        o[5] = np.float32(1.0)
        self._rty_stat()
        return o

    def _rty_frame(self, observation, only):
        rgb = observation.get('rgb') if hasattr(observation, 'get') else None
        ob = observation
        if only and rgb is not None:
            ob = dict(observation)
            ob['rgb'] = np.zeros_like(np.asarray(rgb, np.float32))
            self._rty_h += 1
        try:
            fr = np.asarray(rgb, np.float32) if rgb is not None else None
            if fr is None or fr.size != 256 * 256 * 3 or float(np.mean(np.abs(fr))) < 0.005:
                self._rty_stat()
                return ob
            p = _rty_proba(fr.reshape(256, 256, 3))
            k = int(np.argmax(p))
            pk = float(p[k])
            self._rty_pm = round(pk, 3)
            if pk >= RTY_P:
                self._rty_hits[k] += 1
                if self._rty_hits[k] >= RTY_K:
                    lab = _RTY_LAB[k]
                    cur = self._stype() or 'None'
                    self._rty_done = True
                    self._rty_t = self.tick
                    self._rty_from = cur
                    if lab in RTY_TO and lab != cur and (not (self._cls is not None and self._cls.is_mountain)) and (not self._wh_spawn()):
                        self._rty_ovr = lab
                        if lab in R2_TYPES and getattr(self, '_r2', None) is not None:
                            self._r2.cap = max(0, int(self._r2.cap) - int(self._rty_n))
        except Exception:
            self._rty_err += 1
            self._rty_done = True
        self._rty_stat()
        return ob

    def _rty_stat(self):
        o = getattr(self, '_omb', None)
        if o is None or not isinstance(getattr(o, 'stats', None), dict):
            return
        o.stats['rty_n'] = int(self._rty_n)
        o.stats['rty_h'] = int(self._rty_h)
        o.stats['rty_t'] = int(self._rty_t)
        o.stats['rty_err'] = int(self._rty_err)
        o.stats['rty_ovr'] = self._rty_ovr or ''
        o.stats['rty_from'] = self._rty_from
        o.stats['rty_pm'] = float(self._rty_pm)

    def _nvf_flush(self):
        nv = self._nvf
        if not NVF_LOG or nv is None or (not getattr(nv, 'log', None)):
            return
        try:
            import json
            os.makedirs(NVF_LOG, exist_ok=True)
            with open(os.path.join(NVF_LOG, '%d.jsonl' % os.getpid()), 'a') as fh:
                fh.write(json.dumps(dict(pad=None if self._pad is None else [float(self._pad[0]), float(self._pad[1])], z0=None if self._z0 is None else float(self._z0), ev=nv.log)) + '\n')
        except Exception:
            pass

    def _nvf_stats(self):
        nv = self._nvf
        o = getattr(self, '_omb', None)
        if o is None or not isinstance(getattr(o, 'stats', None), dict):
            return
        o.stats['nvf_ev'] = int(nv.n_eval)
        o.stats['nvf_hit'] = int(nv.n_hit)
        o.stats['nvf_lat'] = int(nv.n_latch)
        o.stats['nvf_st'] = int(nv.n_steer)
        o.stats['nvf_ban'] = int(nv.n_dud)
        o.stats['nvf_t1'] = int(nv.t1)
        o.stats['nvf_ms'] = int(nv.ms + 0.5)
        o.stats['nvf_err'] = int(nv.n_err)
        o.stats['nvf_box'] = int(getattr(nv, 'n_box', 0))
        if NVF_AND:
            o.stats['nvf_and_ev'] = int(nv.n_and_ev)
            o.stats['nvf_and_hit'] = int(nv.n_and_hit)
            o.stats['nvf_and_lat'] = int(nv.n_and_lat)
            o.stats['nvf_and_t1'] = int(nv.and_t1)
            o.stats['nvf_and_ms'] = int(nv.and_ms + 0.5)
            o.stats['nvf_and_err'] = int(nv.n_and_err)
            o.stats['nvf_and_p1'] = nv.and_p1

    def _nvf_step(self, observation, st, pos):
        nv = self._nvf
        _mt = 'mountain' if self._cls.is_mountain else self._stype()
        if _mt is None or _mt == 'mountain' or _mt not in NVF_TYPES or (_mt in DH_TYPES) or self._wh_spawn():
            return
        if nv.clue0 is None:
            nv.clue0 = pos[0:2] + np.asarray(observation['state'], np.float32).reshape(-1)[-2:].astype(np.float64)
        _p3 = self._p3_ban if NVF_P3B else []
        if nv.latch is not None and any((float(np.linalg.norm(nv.latch[0:2] - np.asarray(b[0:2], np.float64))) < P3_BAN_R for b in _p3)):
            nv.ban('n_p3b', self.tick)
        nv.expire(self.tick)
        nv.observe_nvf(self.tick, observation, pos, st[3:6].astype(np.float64), _p3)
        if NVF_AND:
            try:
                nv.observe_and(self.tick, observation, pos, st[3:6].astype(np.float64), _p3)
            except Exception:
                nv.n_and_err += 1
                nv.and_o = None
        if NVF_LOG and nv.log:
            self._nvf_flush()
            nv.log = []
        if NVF_SHADOW:
            self._nvf_stats()
            return
        _mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        _ph = float(_mm[0]) if _mm.size else 1.0
        _rgbl = bool(self._rp is not None and self._rp.latch is not None and self._cls.mtn_ok() and (not self._champ()) or (self._r2_on and self._r2 is not None and (self._r2.latch is not None)))
        _go = bool(nv.latch is not None and (not _rgbl) and (0.5 <= _ph < 1.5))
        if _go:
            _d = float(np.linalg.norm(nv.latch[0:2] - pos[0:2]))
            if nv.go_t != self.tick - 1:
                nv.pd_min = 1000000000.0
                nv.pd_n = 0
            if not (NVF_TERM and _d <= DH_TERM_R):
                nv.st_n += int(nv.lh_t != self.tick)
            if _d < nv.pd_min - 1.0 or _d < 8.0:
                nv.pd_min = min(nv.pd_min, _d)
                nv.pd_n = 0
            else:
                nv.pd_n += 1
            if NVF_STALE > 0 and nv.st_n > NVF_STALE:
                nv.ban('n_stale', self.tick, True)
                _go = False
            elif NVF_PROG > 0 and nv.pd_n > NVF_PROG:
                nv.ban('n_prog', self.tick, True)
                _go = False
            elif not NVF_TERM and NVF_NEAR_T > 0 and (_d < NVF_NEAR) and (_mm.size > 57) and (float(_mm[57]) < 0.5):
                nv.nr_n += 1
                if nv.nr_n > NVF_NEAR_T:
                    nv.ban('n_near', self.tick)
                    _go = False
        _term = None
        if _go and NVF_TERM:
            _term = nv.terminal(pos, float(st[162]) * 20.0 if st.size > 162 else 20.0)
        if _go:
            lk = nv.latch[0:2] - pos[0:2]
            st[-2] = np.float32(lk[0])
            st[-1] = np.float32(lk[1])
            self.route = 'king+nvf'
            self._steer = True
            nv.n_steer += 1
            nv.lat_st += 1
            nv.go_t = self.tick
            if isinstance(self._plan, _Planner):
                self._plan.stall = 0
            if NVF_RZ and _mm.size >= 3 and (_mm[1] != 0.0 or _mm[2] != 0.0):
                _mm = _mm.copy()
                _mm[1] = 0.0
                _mm[2] = 0.0
                self.memory_tensor = _mm.reshape(self._ms)
            if _term is not None:
                self._dh_term = _term
        nv.tick_end(self.tick, pos)
        nv.expire(self.tick)
        self._nvf_stats()

    def _nv_mirror(self):
        nv = self._nv
        d = getattr(self, '_dh', None)
        if nv is None or d is None:
            return
        d.n_dhn = nv.n_eval
        d.n_dhn_hit = nv.n_hit
        d.n_dhn_lat = nv.n_latch
        d.n_dhn_only = nv.n_steer
        d.n_zup = nv.n_dud
        d.dhn_ms = int(nv.ms + 0.5)
        d.n_dhn_err = nv.n_err

    def _vnt_step(self, observation, st, pos):
        nv = self._nv
        _mt = 'mountain' if self._cls.is_mountain else self._stype()
        if _mt is None or _mt == 'mountain' or _mt not in VNT_TYPES or self._wh_spawn():
            return
        if VNT_SIG and (not bool(getattr(self, '_vp_all', False))):
            return
        if not (self._r2_on and self._r2 is not None) or self._r2.latch is not None:
            return
        if nv.clue0 is None:
            nv.clue0 = pos[0:2] + np.asarray(observation['state'], np.float32).reshape(-1)[-2:].astype(np.float64)
        _h0 = nv.hit_t
        nv.observe_nv(self.tick, observation, pos, st[3:6].astype(np.float64), self._p3_ban if NV_P3B else [])
        nv.latch = None
        self._nv_mirror()
        if nv.hit_t != self.tick or _h0 == self.tick or (not nv.hits) or (nv.hits[-1][0] != self.tick):
            return
        self._vnt_h += 1
        q = np.asarray(nv.hits[-1][1], np.float64).reshape(-1)[0:3]
        _VNT_H.append((self.tick, q.copy()))
        while _VNT_H and self.tick - _VNT_H[0][0] > max(VNT_AW, 1):
            _VNT_H.pop(0)
        if not VNT_TRIG or float(np.linalg.norm(q - np.asarray(pos[0:3], np.float64))) > VNT_R:
            return
        if VNT_P1:
            _mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
            if _mm.size < 3 or not 0.5 <= float(_mm[0]) < 1.5:
                return
        if any((float(np.linalg.norm(q[0:2] - np.asarray(b[0:2], np.float64))) < 4.0 for b in self._r2.bad)):
            return
        if self._vnt_n >= VNT_MAX or self._r2.n_req >= self._r2.cap - VNT_RES:
            return
        sp = None
        for s in self._vnt_sp:
            if float(np.linalg.norm(s[0] - q[0:2])) < 4.0:
                sp = s
                break
        if sp is not None and (sp[1] >= VNT_SPOT or self.tick - sp[2] < VNT_CD):
            return
        if sp is None:
            sp = [q[0:2].copy(), 0, -10 ** 9]
            self._vnt_sp.append(sp)
        sp[1] += 1
        sp[2] = self.tick
        self._vnt_n += 1
        self._vnt_req = self.tick

    def _nv_step(self, observation, st, pos):
        if VNT_ON and (not (NV_ON and ('mountain' if self._cls.is_mountain else self._stype()) in NV_TYPES)):
            try:
                return self._vnt_step(observation, st, pos)
            except Exception:
                self._vnt_err = getattr(self, '_vnt_err', 0) + 1
                return
        nv = self._nv
        _mt = 'mountain' if self._cls.is_mountain else self._stype()
        if _mt is None or _mt == 'mountain' or _mt not in NV_TYPES or (_mt in DH_TYPES):
            return
        if self._wh_spawn():
            return
        if _mt == 'warehouse' and (not self._vp_all):
            return
        if nv.clue0 is None:
            nv.clue0 = pos[0:2] + np.asarray(observation['state'], np.float32).reshape(-1)[-2:].astype(np.float64)
        _p3 = self._p3_ban if NV_P3B else []
        if nv.latch is not None and any((float(np.linalg.norm(nv.latch[0:2] - np.asarray(b[0:2], np.float64))) < P3_BAN_R for b in _p3)):
            nv.ban('n_p3b', self.tick)
        nv.expire(self.tick)
        nv.observe_nv(self.tick, observation, pos, st[3:6].astype(np.float64), _p3)
        _mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        _ph = float(_mm[0]) if _mm.size else 1.0
        _rgbl = bool(self._rp is not None and self._rp.latch is not None and self._cls.mtn_ok() and (not self._champ()) or (self._r2_on and self._r2 is not None and (self._r2.latch is not None)))
        _cml = bool(CM_ON and self._cm is not None and (_ph < 1.5))
        _go = bool(nv.latch is not None and (not _rgbl) and (not (NV_CM and _cml)) and (0.5 <= _ph < 1.5))
        if _go:
            _d = float(np.linalg.norm(nv.latch[0:2] - pos[0:2]))
            if nv.go_t != self.tick - 1:
                nv.pd_min = 1000000000.0
                nv.pd_n = 0
            if not (NV_TERM and _d <= DH_TERM_R):
                nv.st_n += int(nv.lh_t != self.tick)
            if _d < nv.pd_min - 1.0 or _d < 8.0:
                nv.pd_min = min(nv.pd_min, _d)
                nv.pd_n = 0
            else:
                nv.pd_n += 1
            if NV_STALE > 0 and nv.st_n > NV_STALE:
                nv.ban('n_stale', self.tick, True)
                _go = False
            elif NV_PROG > 0 and nv.pd_n > NV_PROG:
                nv.ban('n_prog', self.tick, True)
                _go = False
            elif not NV_TERM and NV_NEAR_T > 0 and (_d < NV_NEAR) and (_mm.size > 57) and (float(_mm[57]) < 0.5):
                nv.nr_n += 1
                if nv.nr_n > NV_NEAR_T:
                    nv.ban('n_near', self.tick)
                    _go = False
        _term = None
        if _go and NV_TERM:
            _term = nv.terminal(pos, float(st[162]) * 20.0 if st.size > 162 else 20.0)
        if _go:
            lk = nv.latch[0:2] - pos[0:2]
            st[-2] = np.float32(lk[0])
            st[-1] = np.float32(lk[1])
            self.route = 'king+nv'
            self._steer = True
            nv.n_steer += 1
            nv.lat_st += 1
            nv.go_t = self.tick
            if isinstance(self._plan, _Planner):
                self._plan.stall = 0
            if _cml:
                self._cm_t += 1
                if self._cm_near is not None:
                    self._cm_near += 1
            if NV_RZ and _mm.size >= 3 and (_mm[1] != 0.0 or _mm[2] != 0.0):
                _mm = _mm.copy()
                _mm[1] = 0.0
                _mm[2] = 0.0
                self.memory_tensor = _mm.reshape(self._ms)
            if _term is not None:
                self._dh_term = _term
        nv.tick_end(self.tick, pos)
        nv.expire(self.tick)
        self._nv_mirror()

    def _pf_frame(self, observation, st, only):
        dh = self._dh
        rgb = observation.get('rgb') if hasattr(observation, 'get') else None
        fr = np.array(rgb, np.float32).reshape(256, 256, 3) if rgb is not None else None
        if only and rgb is not None:
            ob = dict(observation)
            ob['rgb'] = np.zeros_like(np.asarray(rgb, np.float32))
            observation = ob
            self._pf_nhide += 1
        if dh is None or dh.fcl is None or dh.pf_f is None or (fr is None) or (float(np.mean(np.abs(fr))) < 0.005):
            return observation
        pos = st[0:3].astype(np.float64)
        rpy = st[3:6].astype(np.float64)
        det = None
        if self._rp is not None:
            if self._rp.sess is None and (not self._rp._tried):
                self._rp._load()
            sc = self._rp.score(fr)
            if sc is not None:
                p, u, v, h = sc
                q = _pf_loc(u, v, observation['depth'], pos, rpy)
                det = (float(p), float(u), float(v), None if q is None else np.asarray(q, np.float64))
        if det is None:
            return observation
        dh.pf_f.update(dh.pf.rgb_features(fr, pos, rpy, det, dh.pf_q, dh.pf_cz))
        dh.pf_rgb = True
        self._pf_note('rgb', p=det[0], f0=int(dh.fcl[0]), cr=dh.pf_f.get('cr_last'), qd=dh.pf_f.get('rgb_qd_last'))
        return observation

    def set_oracle(self, vc, vtop):
        self._pf_vc = [float(x) for x in np.asarray(vc, np.float64).reshape(-1)[0:3]] if _PF_LOG else None

    def _pf_note(self, ev, **kw):
        if _PF_LOG:
            _pf_log(dict(ev=ev, tick=self.tick, pid=os.getpid(), pad=None if self._pad is None else [float(self._pad[0]), float(self._pad[1])], z0=self._z0, pm=PF_MODEL, vc=getattr(self, '_pf_vc', None), **kw))

    def _swstep(self, pos, st, obs, npl):
        sw = self._sw
        if sw is None:
            sw = self._sw = _Sweep(self._c0)
            self._plan = sw
        sw.st = 0
        if float(pos[2]) > self._z0 + 1.5:
            sw.observe(pos, st[3:6].astype(np.float64), obs['depth'])
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        p1 = 0.5 <= float(mm[0]) < 1.5
        lat = self._r2 is not None and self._r2.latch is not None and self._r2_ok()
        sw.stall = sw.stall + 1 if p1 and (not lat) and (float(np.hypot(float(st[6]), float(st[7]))) < 0.8) else 0
        if not sw.armed:
            _o = self._cb - pos[0:2]
            if npl:
                st[-2] = np.float32(_o[0])
                st[-1] = np.float32(_o[1])
            if not (p1 and float(pos[2]) > self._z0 + 4.5 and (float(np.hypot(_o[0], _o[1])) <= SW_ARR or sw.stall >= SW_STALL)):
                return
            sw.plan(pos, float(st[5]))
            sw.armed = True
            sw.t_arm = self.tick
            sw.stall = 0
            sw.t_plan = self.tick
            self._plan = sw
        if sw.S is None:
            return
        wp = sw.wp
        if p1 and (not lat):
            if sw.stall >= SW_STALL:
                sw.j = min(len(sw.S) - 1, sw.j + 10)
                sw.stall = 0
                sw.n_stall += 1
            wp = sw.carrot(pos)
            if wp is None and self.tick - sw.t_plan >= 50:
                sw.plan(pos, float(st[5]))
                sw.t_plan = self.tick
                wp = sw.carrot(pos)
            if wp is None:
                wp = sw.wp = sw.S[-1].copy()
        if p1 and (not (WH_SWFIX and lat)) and (mm[1] != 0.0 or mm[2] != 0.0):
            mm = mm.copy()
            mm[1] = 0.0
            mm[2] = 0.0
            self.memory_tensor = mm.reshape(self._ms)
        if wp is not None and bool(np.all(np.isfinite(wp))):
            _o = wp - pos[0:2]
            st[-2] = np.float32(_o[0])
            st[-1] = np.float32(_o[1])
            sw.st = 1
        self.route = 'king+sw'
        self._steer = True

    def _wh_spawn(self):
        return self._z0 is not None and abs(self._z0 - WH_Z0) < 0.05

    def _wh(self):
        c = self._cls
        return bool(self._wh_spawn() and c is not None and (not c.is_mountain) and c.settled('warehouse'))

    def _cm_update(self, st):
        if self._stype() not in CM_TYPES:
            return
        mo = self.memory_tensor
        mi = self._fg_mem
        p2 = st[0:2].astype(np.float64)
        if mo[0] >= 2.5:
            self._cm = None
            self._cm_hits = []
            return
        if self.tick >= CM_T0 and mi is not None and (mi[57] < 0.5) and (mo[57] < 0.5) and (mo[0] < 1.5) and (mo[55] > CM_TH):
            q = np.asarray(mo[43:45], np.float64)
            if all((float(np.hypot(*q - b)) > CM_BAD_R for b in self._cm_bad)):
                self._cm_hits = [(k, z) for k, z in self._cm_hits if self.tick - k <= CM_WIN] + [(self.tick, q)]
                nb = [z for _, z in self._cm_hits if float(np.hypot(*z - q)) <= CM_TOL]
                if self._cm is None and len(nb) >= CM_NEED:
                    c = np.mean(nb, 0)
                    if float(np.hypot(*c - p2)) >= CM_RMIN:
                        self._cm = c
                        self._cm_t = self.tick
                        self._cm_near = None
                        self._cm_n += 1
        if self._cm is not None:
            d = float(np.hypot(*self._cm - p2))
            if d < CM_NEAR and self._cm_near is None:
                self._cm_near = self.tick
            if self._cm_near is not None and self.tick - self._cm_near > CM_NEAR_T and (mo[57] < 0.5) or self.tick - self._cm_t > CM_TMAX:
                r2l = getattr(self._r2, 'latch', None) if self._r2 is not None else None
                if r2l is None or float(np.hypot(*np.asarray(r2l[0:2], np.float64) - self._cm)) > 6.0:
                    self._cm_bad.append(self._cm.copy())
                self._cm = None
                self._cm_hits = []

    def _lassist(self, obs):
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        mi = self._fg_mem
        if mm.size < 63 or mi is None:
            return
        rgb = obs.get('rgb') if hasattr(obs, 'get') else None
        had = rgb is not None and float(np.mean(np.abs(np.asarray(rgb, np.float32)))) > 0.005
        if had and 1.5 <= float(mi[0]) < 2.5 and (mi[57] > 0.5) and (mm[0] < 1.5) and (mm[57] < 0.5):
            spot = np.asarray(mi[60:62], np.float64).copy()
            self._la_ban.append(spot)
            self._la_rej += 1
            if self._la_spot is not None and float(np.linalg.norm(spot - self._la_spot)) < LA_BAN:
                p = np.asarray(obs['state'], np.float32).reshape(-1)[0:3].astype(np.float32)
                off = np.array([[30, 0, 0], [-30, 0, 0], [0, 30, 0], [0, -30, 0], [21, 21, 0], [-21, -21, 0], [21, -21, 0], [-21, 21, 0], [30, 30, 0], [-30, -30, 0]], np.float32)
                mm = mm.copy()
                mm[16:46] = (p[None, :] + off).reshape(-1)
                self.memory_tensor = mm.reshape(self._ms)
                self._la_spot = None
            self._la_hits = []
            return
        _lt = self._stype()
        if _lt is None and LA_NONE_CITY > 0.0 and (self._cls.n > 0) and (not self._cls.is_mountain) and (float(self._cls.ps[0]) / self._cls.n >= LA_NONE_CITY):
            _lt = 'city'
        if (_lt if _lt is not None else 'none') not in LA_TYPES or self.tick < LA_T0 or mm[57] > 0.5 or (not 0.5 <= float(mm[0]) < 1.5):
            self._la_hits = []
            return
        if mm[58] > 0.5:
            self._la_hits.append((self.tick, np.asarray(mm[43:46], np.float64).copy()))
        self._la_hits = [(k, q) for k, q in self._la_hits if self.tick - k < LA_W]
        if len(self._la_hits) < LA_K:
            return
        P = np.asarray([q for _, q in self._la_hits])
        med = np.median(P[:, 0:2], axis=0)
        near = np.linalg.norm(P[:, 0:2] - med, axis=1) <= LA_R
        if int(near.sum()) < LA_K:
            return
        c = P[near].mean(axis=0)
        if any((float(np.linalg.norm(c[0:2] - b)) < LA_BAN for b in self._la_ban)):
            return
        mm = mm.copy()
        mm[16:46] = np.tile(c.astype(np.float32), 10)
        mm[57] = 1.0
        mm[58] = 0.0
        mm[59] = 0.0
        mm[60:63] = c.astype(np.float32)
        self.memory_tensor = mm.reshape(self._ms)
        self._la_spot = c[0:2].copy()
        self._la_n += 1
        self._la_hits = []
        if not self.route.endswith('+la'):
            self.route = self.route + '+la'

    def _pguard(self, obs, st, out):
        mm = self._fg_mem
        ph = float(mm[0]) if mm is not None else 0.0
        if ph < 0.5 or self._stype() not in PG_TYPES:
            return out
        agl = float(st[162]) * 20.0 if st.size > 162 else 20.0
        d = np.asarray(obs['depth'], np.float32).reshape(256, 256)[::2, ::2] * 29.5 + 0.5
        fwd, up, right = _axes(st[3:6])
        g = (np.arange(128) * 2 + 1.0 - 128.0) / 128.0
        U = g[None, :]
        V = -g[:, None]
        rx = fwd[0] + U * right[0] + V * up[0]
        ry = fwd[1] + U * right[1] + V * up[1]
        rz = fwd[2] + U * right[2] + V * up[2]
        gd = np.where(rz < -0.02, (agl + 0.05) / np.maximum(-rz, 0.001), 1000000000.0)
        ob = (d < 0.8 * gd) & (d < 29.0)
        dmin = float(d[ob].min()) if ob.any() else 30.0
        cap = PG_VMIN + (1.0 - PG_VMIN) * min(1.0, max(0.0, (dmin - PG_DMIN) / max(PG_D - PG_DMIN, 0.001)))
        if agl < PG_AGL and ph < 2.5:
            cap = min(cap, PG_VMIN + (1.0 - PG_VMIN) * min(1.0, max(0.0, (agl - 0.6) / max(PG_AGL - 0.6, 0.001))))
        if cap < 0.999 and self._pg_cap >= 0.999:
            self._pg_n += 1
        if cap <= self._pg_cap:
            self._pg_cap = cap
            self._pg_t = self.tick
        elif self.tick - self._pg_t > PG_HOLD:
            self._pg_cap = min(cap, self._pg_cap + 0.02)
        dv = np.asarray(out[0:3], np.float64)
        n = float(np.linalg.norm(dv))
        if n < 1e-06:
            return out
        vel = dv / n * float(out[3]) * 3.0
        vh = float(np.hypot(vel[0], vel[1]))
        capv = 3.0 * self._pg_cap
        if vh > 1e-06:
            hx, hy = (vel[0] / vh, vel[1] / vh)
            rn = np.sqrt(rx * rx + ry * ry + rz * rz)
            cosm = (rx * hx + ry * hy) / rn
            near = ob & (d * rn < PG_STOP) & (cosm > 0.82)
            if near.any():
                capv = 0.0
        hit = False
        if vh > capv:
            vel[0:2] *= capv / max(vh, 1e-06)
            hit = True
        if PG_REP > 0.0:
            pos = np.asarray(st[0:3], np.float64)
            cam = pos + fwd * _CAMF + up * _CAMU
            rng = d * np.sqrt(rx * rx + ry * ry + rz * rz)
            sel = ob & (rng < 6.0)
            sel[1::2, :] = False
            sel[:, 1::2] = False
            if sel.any():
                P = np.stack([cam[0] + (d * rx)[sel], cam[1] + (d * ry)[sel], cam[2] + (d * rz)[sel]], 1)
            else:
                P = np.zeros((0, 3))
            self._pg_pts.append(P)
            if len(self._pg_pts) > PG_MEM:
                self._pg_pts.pop(0)
            Q = np.concatenate(self._pg_pts, 0) if self._pg_pts else np.zeros((0, 3))
            if Q.shape[0]:
                D = Q - pos[None, :]
                dd = np.linalg.norm(D, axis=1)
                m = dd < PG_REP
                if m.any():
                    w = (PG_REP - dd[m]) / PG_REP
                    nh = (D[m] / dd[m][:, None] * w[:, None]).sum(0)
                    nn = float(np.linalg.norm(nh))
                    if nn > 1e-06:
                        nh /= nn
                        dist = float(dd[m].min())
                        tow = float(vel @ nh)
                        if tow > 0.0:
                            vel = vel - tow * nh
                            hit = True
                        push = PG_REPV * min(1.0, max(0.0, (PG_REP - dist) / max(PG_REP - 0.4, 0.001)))
                        if push > 0.0:
                            vel = vel - push * nh
                            hit = True
        if agl < PG_AGL and ph < 2.5:
            vz = PG_CLIMB * min(1.0, (PG_AGL - agl) / 0.5)
            if vel[2] < vz:
                vel[2] = vz
                hit = True
        if not hit:
            return out
        o = out.copy()
        sp = float(np.linalg.norm(vel))
        if sp < 0.001:
            o[3] = np.float32(0.0)
        else:
            o[0:3] = (vel / sp).astype(np.float32)
            o[3] = np.float32(min(1.0, sp / 3.0))
        if not self.route.endswith('+pg'):
            self.route = self.route + '+pg'
        return o

    def _nguard(self, obs, st, out):
        mm = self._fg_mem
        ph = float(mm[0]) if mm is not None else 0.0
        if not 0.5 <= ph < 2.5:
            self._ng_b = None
            self._ng_hold = 0
            self._ng_on = 0
            self._ng_rel = 0
            return out
        pos = st[0:3].astype(np.float64)
        rpy = st[3:6].astype(np.float64)
        if self.tick % NG_EVERY == 0:
            P = _ng_cloud(obs['depth'], pos, rpy)
            if P.shape[0]:
                P = P[(P[:, 2] > NG_Z0) & (np.hypot(P[:, 0] - pos[0], P[:, 1] - pos[1]) < NG_KEEP)]
            self._ng_mem.append(P)
            if len(self._ng_mem) > NG_MEMT:
                self._ng_mem.pop(0)
        if self._ng_cool > 0:
            self._ng_cool -= 1
            self._ng_b = None
            self._ng_hold = 0
            self._ng_on = 0
            self._ng_rel = 0
            return out
        vel = st[6:9].astype(np.float64)
        vh = float(np.hypot(vel[0], vel[1]))
        dv = np.asarray(out[0:3], np.float64)
        n = float(np.linalg.norm(dv))
        vc = dv / n * float(out[3]) * 3.0 if n > 1e-06 else np.zeros(3)
        vch = float(np.hypot(vc[0], vc[1]))
        dirs = []
        if vh > NG_VMIN:
            dirs.append((vel[0] / vh, vel[1] / vh, max(NG_LMIN, vh * NG_TAU), float(vel[2]) / vh))
        if vch > NG_VMIN:
            dirs.append((vc[0] / vch, vc[1] / vch, NG_LC, float(vc[2]) / vch))
        hit = None
        dhit = False
        if dirs and self._ng_mem:
            Q = np.concatenate(self._ng_mem, 0)
            if Q.shape[0]:
                Lm = max((d[2] for d in dirs)) + NG_RT
                Q = Q[np.hypot(Q[:, 0] - pos[0], Q[:, 1] - pos[1]) < Lm]
            z0 = float(pos[2])
            for bx, by, L, sz in dirs:
                if hit is not None or Q.shape[0] == 0:
                    break
                sz = max(-NG_SZ, min(NG_SZ, sz))
                for s in np.arange(NG_DS, L + 1e-06, NG_DS):
                    px = pos[0] + bx * s
                    py = pos[1] + by * s
                    hd = np.hypot(Q[:, 0] - px, Q[:, 1] - py)
                    nr = hd < NG_RR
                    nt = hd < NG_RT
                    zq = Q[:, 2]
                    for pz in (z0, z0 + sz * s) if sz < 0.0 else (z0,):
                        if int((nr & (zq > NG_ZROOF) & (zq < pz - NG_DZR)).sum()) >= NG_N and int((nt & (zq > pz + NG_DZT)).sum()) >= NG_N:
                            if pz == z0:
                                hit = (bx, by)
                            else:
                                dhit = True
                            break
                    if hit is not None:
                        break
        if hit is not None:
            if self._ng_b is None:
                self._ng_n += 1
            self._ng_b = np.array(hit, np.float64)
            self._ng_hold = NG_HOLD
        elif self._ng_hold > 0:
            self._ng_hold -= 1
        else:
            if self._ng_b is not None:
                self._ng_rel = NG_REL
            self._ng_b = None
            self._ng_on = 0
        if self._ng_b is not None:
            self._ng_on += 1
            self._ng_ticks += 1
            if self._ng_on > NG_MAXT:
                self._ng_cool = NG_COOL
                self._ng_b = None
                self._ng_hold = 0
                self._ng_on = 0
                self._ng_rel = NG_REL
        v = vc.copy()
        act = False
        if self._ng_b is not None:
            b = self._ng_b
            act = True
            va = float(vel[0] * b[0] + vel[1] * b[1])
            cap = max(0.0, va - NG_DV)
            tow = float(v[0] * b[0] + v[1] * b[1])
            if tow > cap:
                v[0] -= (tow - cap) * b[0]
                v[1] -= (tow - cap) * b[1]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(float(st[3])) * math.cos(float(st[4]))))))
            if va <= NG_VCL and tilt <= NG_TILT and (float(pos[2]) < NG_ZMAX):
                v[2] = max(float(v[2]), NG_CLIMB)
            else:
                v[2] = max(float(v[2]), 0.0)
        elif dhit:
            act = True
            self._ng_dn += 1
            v[2] = max(float(v[2]), 0.0)
        if self._ng_b is not None or self._ng_rel > 0:
            if self._ng_b is None:
                self._ng_rel -= 1
            ex = v[0] - vel[0]
            ey = v[1] - vel[1]
            en = math.hypot(ex, ey)
            slow = float(v[0] * vel[0] + v[1] * vel[1]) >= 0.0 and math.hypot(v[0], v[1]) <= vh
            if en > NG_EMAX and (self._ng_b is not None or not slow):
                v[0] = vel[0] + ex / en * NG_EMAX
                v[1] = vel[1] + ey / en * NG_EMAX
                act = True
        if not act:
            return out
        sp = float(np.linalg.norm(v))
        o = out.copy()
        if sp < 0.001:
            o[3] = np.float32(0.0)
        else:
            o[0:3] = (v / sp).astype(np.float32)
            o[3] = np.float32(min(1.0, sp / 3.0))
        if not self.route.endswith('+ng'):
            self.route = self.route + '+ng'
        return o

    def _shield(self, st, out):
        import time as _time
        t0 = _time.perf_counter()
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        mode = int(round(float(mm[0]))) if mm.size else 1
        if self._cls is not None and self._cls.is_mountain:
            mt = 'mountain'
        else:
            mt = (self._stype() if self._cls is not None else None) or 'unk'
        if mode not in SH_MODES or mt not in SH_MAPS or getattr(self, '_esc_w', -1) == self.tick:
            self._sh.vlast = None
            return out
        ds = SH_DSF if mt == 'forest' else SH_DS
        pos = st[0:3].astype(np.float64)
        near = False
        if mode in (2, 3) and mm.size >= 63:
            tg = mm[8:10] if mode == 3 else mm[60:62]
            near = bool(math.hypot(float(tg[0]) - pos[0], float(tg[1]) - pos[1]) < SH_RN)
        if getattr(self, '_r2_term', None) is not None or getattr(self, '_rp_term', None) is not None:
            near = True
        if SH_LLT and getattr(self, '_ll_on', False):
            near = True
        if near:
            ds = min(ds, SH_DSN)
        o = self._sh.filt(out, st, ds, self.tick, {'m': mode, 'mt': mt, 'nr': int(near)})
        if o is not out and (not self.route.endswith('+sh')):
            self.route = self.route + '+sh'
        dt = (_time.perf_counter() - t0) * 1000.0
        self._sh.t_ms += dt
        self._sh.n_t += 1
        self._sh.t_max = max(self._sh.t_max, dt)
        return o

    def _cmk(self, st, out):
        _k = self.tick % self._cmk_gr.size
        _agl = float(st[162]) * 20.0 if st.size > 162 else 99.0
        _gb = 1 if _agl < 19.5 and abs(float(st[2]) - _agl) <= CMK_G else 0
        self._cmk_gs += _gb - int(self._cmk_gr[_k])
        self._cmk_gr[_k] = _gb
        _k = self.tick % 250
        self._cmk_es += self._cmk_e1 - int(self._cmk_er[_k])
        self._cmk_er[_k] = self._cmk_e1
        self._cmk_e1 = 0
        if self._cls.is_mountain or self._chull_wh():
            return out
        if self._cmk_gs < CMK_GN:
            return out
        _t = self._stype()
        if not (_t in CMK_TYPES or (CMK_X and self._chull_x(_t))):
            return out
        if float(self._cls.ps[LB.index('forest')]) / max(self._cls.n, 1) >= CMK_FMAX:
            return out
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        mode = int(round(float(mm[0]))) if mm.size else -1
        if mode not in CMK_MODES:
            return out
        if any((getattr(self, k, None) is not None for k in ('_r2_term', '_rp_term', '_dh_term'))):
            return out
        if self._esc is not None and self._esc.phase in ('scan', 'creep', 'climb'):
            return out
        pos = st[0:3].astype(np.float64)
        g = self._chull_mem.g
        if g is None:
            return out
        zr = float(pos[2]) - float(g)
        if zr >= max(CMK_ZT, CMK_ZC if CMK_OF else 0.0):
            return out
        v_cmd = _omb_vel(out)
        sp_cmd = math.hypot(float(v_cmd[0]), float(v_cmd[1]))
        if float(np.linalg.norm(v_cmd)) < CMK_VMIN:
            return out
        tg = []
        if mode == 2 and mm.size >= 63:
            tg.append((float(mm[60]), float(mm[61])))
        if mode == 1:
            tg.append((float(pos[0]) + float(st[-2]), float(pos[1]) + float(st[-1])))
        for _d in (getattr(self, '_rp', None), getattr(self, '_r2', None), getattr(self, '_dh', None)):
            _l = getattr(_d, 'latch', None) if _d is not None else None
            if _l is not None:
                tg.append((float(_l[0]), float(_l[1])))
        if self._cmk_es >= CMK_DUTY and self.tick > self._cmk_free:
            self._cmk_lvl = 2 if self._cmk_lvl > 0 and self.tick - self._cmk_free < 500 else 1
            self._cmk_free = self.tick + CMK_FREE
            self._cmk_st['relax'] += 1
        rel = self._cmk_lvl if self.tick <= self._cmk_free else 0
        feet = self._chull_mem.cmk_feet(pos, self.tick, tg)
        if not feet:
            return out
        at = CMK_ACC * CMK_TAU
        C = []
        tall = False
        for u, d2, H, dt in feet:
            M = 0.0
            kind = None
            if H >= CMK_HT and zr < CMK_ZT:
                M = (CMK_MT, CMK_MREL, CMK_MNEAR)[rel]
                kind = 't'
                tall = True
            if CMK_OF and zr < CMK_ZC:
                mo = CMK_MO if not rel else min(CMK_MO, CMK_MNEAR)
                if mo > M:
                    M = mo
                    kind = 'o'
            for q in dt:
                M = min(M, max(CMK_MNEAR, q - CMK_TGAP))
            if M <= 0.0:
                continue
            hh = d2 - SH_RB - M
            C.append((u, -at + math.sqrt(at * at + 2.0 * CMK_ACC * hh) if hh >= 0.0 else -min(CMK_PUSH, -hh), kind))
        if not C:
            return out

        def _proj(v):
            hit = None
            for _ in range(6):
                w = None
                for u, l, kd in C:
                    e = float(v @ u) - l
                    if e > 0.001 and (w is None or e > w[0]):
                        w = (e, u, kd)
                if w is None:
                    break
                v = v - w[0] * w[1]
                hit = w if hit is None else hit
            return (v, hit)
        v, hit = _proj(v_cmd.copy())
        if hit is None:
            return out
        self._cmk_e1 = 1
        sp_h = math.hypot(float(v[0]), float(v[1]))
        if sp_cmd >= 0.5 and sp_h < 0.35 * sp_cmd:
            n = hit[1]
            tL = np.array([-n[1], n[0], 0.0])
            tR = -tL
            if self.tick <= self._cmk_su and self._cmk_side != 0.0:
                side = self._cmk_side
            else:
                dot = float(v_cmd[0] * tL[0] + v_cmd[1] * tL[1]) / max(sp_cmd, 1e-06)
                if abs(dot) > 0.2:
                    side = 1.0 if dot > 0 else -1.0
                else:
                    al = max((float(tL @ u) for u, l, kd in C))
                    ar = max((float(tR @ u) for u, l, kd in C))
                    side = 1.0 if al <= ar else -1.0
                self._cmk_side = side
                self._cmk_su = self.tick + CMK_SLH
            tt = tL if side > 0 else tR
            vs = min(sp_cmd, CMK_SLV)
            v, _h2 = _proj(np.array([tt[0] * vs, tt[1] * vs, float(v_cmd[2])], np.float64))
            self._cmk_st['slide'] += 1
        o = out.copy()
        _omb_store(o, v)
        self._cmk_n += 1
        self._cmk_st['t' if hit[2] == 't' else 'o'] += 1
        if tall:
            self._cmk_st['tall'] += 1
        if self._omb is not None:
            self._omb.stats['cmk_n'] = self._cmk_n
            for _kk, _vv in self._cmk_st.items():
                self._omb.stats['cmk_' + _kk] = _vv
        if not self.route.endswith('+cmk'):
            self.route = self.route + '+cmk'
        return o

    def _vrv(self, obs):
        mi = self._fg_mem
        mo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        if mi.size < 64 or mo.size < 64:
            return
        if self._chull_wh():
            return
        _t = self._stype()
        if not (_t in VRV_TYPES or (VRV_X and self._chull_x(_t))):
            return
        lk_in = bool(1.5 <= float(mi[0]) < 2.5 and float(mi[57]) > 0.5)
        if not lk_in:
            if float(mo[0]) >= 2.5 or float(mi[0]) >= 2.5:
                self._vrv_key = None
                self._vrv_nl = 0
                self._vrv_nh = 0
            return
        key = np.asarray(mi[60:62], np.float64)
        if self._vrv_key is None or float(np.linalg.norm(key - self._vrv_key)) > VRV_KR:
            self._vrv_key = key.copy()
            self._vrv_nl = 0
            self._vrv_nh = 0
        self._vrv_nl += 1
        if float(mo[55]) >= VRV_P and float(np.linalg.norm(np.asarray(mo[43:45], np.float64) - key)) <= VRV_R:
            self._vrv_nh += 1
        rgb = obs.get('rgb') if hasattr(obs, 'get') else None
        had = rgb is not None and float(np.mean(np.abs(np.asarray(rgb, np.float32)))) > 0.005
        if not (had and float(mo[0]) < 1.5 and (float(mo[57]) < 0.5)):
            return
        sp = None
        for _s in self._vrv_spots:
            if float(np.linalg.norm(_s[0] - key)) <= VRV_KR:
                sp = _s
                break
        ok = bool(self._vrv_nl >= VRV_NL and self._vrv_nh >= VRV_NH and (self._vrv_nh >= VRV_F * self._vrv_nl) and (sp is None or sp[1] < VRV_MAX))
        _rdok = ok
        _oof = False
        if ok and VRV_RGB and (VRV_OOF > 0.0):
            _st = np.asarray(obs['state'], np.float32).reshape(-1)
            _fw, _up, _rt = _axes(_st[3:6].astype(np.float64))
            _cam = _st[0:3].astype(np.float64) + _fw * _CAMF + _up * _CAMU
            _d = np.asarray(mi[60:63], np.float64) - _cam
            _z = float(_d @ _fw)
            _oof = bool(_z <= 0.5 or abs(float(_d @ _rt) / _z) > VRV_OOF or abs(float(_d @ _up) / _z) > VRV_OOF)
            if _oof:
                self._vrv_oof += 1
                if self._omb is not None:
                    self._omb.stats['vrv_oof'] = self._vrv_oof
        if ok and VRV_RGB and (not _oof):
            ok = False
            self._vrv_rgbn += 1
            if getattr(self, '_vrv_sess', None) is None:
                _so = ort.SessionOptions()
                _so.intra_op_num_threads = 1
                _so.inter_op_num_threads = 1
                self._vrv_sess = ort.InferenceSession(str(_H / 'victim_rgb256.onnx'), _so, providers=['CPUExecutionProvider'])
                self._vrv_in = self._vrv_sess.get_inputs()[0].name
            _o = np.asarray(self._vrv_sess.run(None, {self._vrv_in: np.ascontiguousarray(np.asarray(rgb, np.float32).reshape(256, 256, 3))})[0], np.float32).reshape(-1)
            _st = np.asarray(obs['state'], np.float32).reshape(-1)
            _fw, _up, _rt = _axes(_st[3:6].astype(np.float64))
            _cam = _st[0:3].astype(np.float64) + _fw * _CAMF + _up * _CAMU
            _d = np.asarray(mi[60:63], np.float64) - _cam
            _z = float(_d @ _fw)
            _p = _sig(float(_o[0]))
            self._vrv_last = (round(_p, 3), round(float(_o[1]), 2), round(float(_o[2]), 2))
            if _z > 0.5 and _p >= VRV_T and (abs(float(_o[1]) - float(_d @ _rt) / _z) <= VRV_UV) and (abs(float(_o[2]) - float(_d @ _up) / _z) <= VRV_UV):
                ok = True
                self._vrv_rgbok += 1
            if self._omb is not None:
                self._omb.stats['vrv_rgbn'] = self._vrv_rgbn
                self._omb.stats['vrv_rgbok'] = self._vrv_rgbok
        if RLK_ON and _rdok and (not ok) and (_t not in VRV_TYPES or _t in RLK_TYPES):
            try:
                if self._rlk_try(obs, mi, mo):
                    return
            except Exception:
                self._rlk = None
                self._rlk_err += 1
                try:
                    self.memory_tensor = mo.reshape(self._ms)
                except Exception:
                    pass
        if not ok:
            self._vrv_acc += 1
            if self._omb is not None:
                self._omb.stats['vrv_acc'] = self._vrv_acc
            return
        if sp is None:
            self._vrv_spots.append([key.copy(), 1])
        else:
            sp[1] += 1
        mm = mo.copy()
        mm[0] = mi[0]
        mm[56:63] = mi[56:63]
        mm[59] = 0.0
        if VRV_FB:
            self._vrv_m63 = float(mo[63]) if self._vrv_m63 is None else self._vrv_m63
            mm[63] = np.float32(max(float(mo[63]), 40.0))
        self.memory_tensor = mm.reshape(self._ms)
        self._vrv_sup = VRV_SUP
        self._vrv_rev += 1
        if self._omb is not None:
            self._omb.stats['vrv_rev'] = self._vrv_rev
        if not self.route.endswith('+vrv'):
            self.route = self.route + '+vrv'

    def _rlk_stats(self):
        if self._omb is not None:
            _s = self._omb.stats
            _s['rlk_n'] = self._rlk_n
            _s['rlk_err'] = self._rlk_err
            _s['rlk_pts'] = [list(_q) for _q in self._rlk_pts]
            for _k, _v in self._rlk_end.items():
                _s['rlk_' + _k] = _v

    def _rlk_try(self, obs, mi, mo):
        if self._rlk is not None:
            return False
        key = np.asarray(mi[60:62], np.float64)
        if any((float(np.linalg.norm(key - _s)) <= VRV_KR for _s in self._rlk_spots)):
            return False
        if not (self._vrv_nh >= RLK_NH and self._vrv_nh >= RLK_F * self._vrv_nl):
            return False
        _st = np.asarray(obs['state'], np.float64).reshape(-1)
        _dh = float(np.hypot(float(key[0]) - _st[0], float(key[1]) - _st[1]))
        if _dh > RLK_DH:
            return False
        _fw, _up, _rt = _axes(_st[3:6])
        _d = np.asarray(mi[60:63], np.float64) - (_st[0:3] + _fw * _CAMF + _up * _CAMU)
        _z = float(_d @ _fw)
        _u, _v = (float(_d @ _rt) / _z, float(_d @ _up) / _z) if _z > 0.5 else (9.9, -9.9)
        if not (_v < RLK_V or abs(_u) > 1.0):
            return False
        mm = mo.copy()
        mm[0] = mi[0]
        mm[56:63] = mi[56:63]
        mm[59] = 0.0
        self.memory_tensor = mm.reshape(self._ms)
        self._rlk_spots.append(key.copy())
        self._rlk_n += 1
        self._rlk_m63 = float(mo[63])
        self._rlk = {'t0': int(self.tick), 'zt': float(mi[62]) + RLK_H, 'v': np.asarray(_st[6:9], np.float64).copy()}
        if len(self._rlk_pts) < 6:
            self._rlk_pts.append([round(float(key[0]), 2), round(float(key[1]), 2), int(self.tick), int(self._vrv_nh), int(self._vrv_nl), round(_dh, 2), round(max(-9.9, min(9.9, _u)), 2), round(max(-9.9, min(9.9, _v)), 2)])
        self._rlk_stats()
        return True

    def _rlk_step(self, st, out):
        r = self._rlk
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        self._rlk_sup = False
        if not (1.5 <= float(mm[0]) < 2.5 and float(mm[57]) > 0.5):
            self._rlk_end['lost'] += 1
            self._rlk = None
            self._rlk_stats()
            return out
        if self.tick - r['t0'] > RLK_T:
            self._rlk_end['time'] += 1
            self._rlk = None
            self._rlk_stats()
            return out
        p = st[0:3].astype(np.float64)
        dx = float(mm[60]) - p[0]
        dy = float(mm[61]) - p[1]
        D = math.hypot(dx, dy)
        agl = float(st[162]) * 20.0 if st.size > 162 else 20.0
        zt = max(r['zt'], float(p[2]) - agl + RLK_AGL)
        ez = zt - float(p[2])
        _ye = math.atan2(math.sin(math.atan2(dy, dx) - float(st[5])), math.cos(math.atan2(dy, dx) - float(st[5])))
        if abs(ez) < 0.3 and float(np.linalg.norm(st[6:9].astype(np.float64))) < 0.6 and (RLK_D0 - 0.3 <= D <= RLK_D1 + 0.3) and (abs(_ye) < RLK_YAW):
            self._rlk_end['pos'] += 1
            self._rlk = None
            self._rlk_stats()
            return out
        vr = 0.0
        if D > RLK_D1:
            vr = min(1.0, 0.8 * (D - RLK_D1) + 0.2)
        elif D < RLK_D0:
            vr = -min(1.0, 0.8 * (RLK_D0 - D) + 0.2)
        ux, uy = (dx / D, dy / D) if D > 1e-06 else (0.0, 0.0)
        vd = np.array([ux * vr, uy * vr, max(-RLK_VZ, min(0.8, 1.2 * ez))], np.float64)
        v = r['v'] + np.clip(vd - r['v'], -RLK_ACC, RLK_ACC)
        r['v'] = v
        sp = float(np.linalg.norm(v))
        o = out.copy()
        if sp < 0.001:
            o[3] = np.float32(0.0)
        else:
            o[0:3] = (v / sp).astype(np.float32)
            o[3] = np.float32(min(1.0, sp / 3.0))
        o[4] = np.float32(math.atan2(dy, dx) / math.pi if D > 1.0 else float(st[5]) / math.pi)
        self._rlk_sup = True
        if not self.route.endswith('+rlk'):
            self.route = self.route + '+rlk'
        return o

    def _nrc(self, st, out):
        if self._cls.is_mountain or self._chull_wh():
            return out
        if self._cmk_gs >= CMK_GN:
            self._nrc_g = True
        if not (self._nrc_g if NRC_GL else self._cmk_gs >= CMK_GN):
            return out
        _t = self._stype()
        if not (_t in NRC_TYPES or (CMK_X and self._chull_x(_t))):
            return out
        if float(self._cls.ps[LB.index('forest')]) / max(self._cls.n, 1) >= CMK_FMAX:
            return out
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        mode = int(round(float(mm[0]))) if mm.size else -1
        if mode not in CMK_MODES:
            return out
        if any((getattr(self, k, None) is not None for k in ('_r2_term', '_rp_term', '_dh_term'))):
            return out
        if self._esc is not None and self._esc.phase in ('scan', 'creep', 'climb'):
            return out
        g = self._chull_mem.g
        if g is None:
            return out
        pos = st[0:3].astype(np.float64)
        zr = float(pos[2]) - float(g)
        if zr < NRC_ZLO or zr >= NRC_ZC + NRC_ZH:
            return out
        bx = self._chull_mem.nrc_boxes(pos, self.tick)
        if not bx:
            return out
        tg = []
        if mode == 2 and mm.size >= 63:
            tg.append((float(mm[60]), float(mm[61])))
        if mode == 1:
            tg.append((float(pos[0]) + float(st[-2]), float(pos[1]) + float(st[-1])))
        for _d in (getattr(self, '_rp', None), getattr(self, '_r2', None), getattr(self, '_dh', None)):
            _l = getattr(_d, 'latch', None) if _d is not None else None
            if _l is not None:
                tg.append((float(_l[0]), float(_l[1])))
        tgz = []
        if NRC_TX > 0.0:
            _gt = []
            if mode == 2 and mm.size >= 63:
                _gt.append((float(mm[60]), float(mm[61]), float(mm[62])))
            for _d in (getattr(self, '_rp', None), getattr(self, '_r2', None), getattr(self, '_dh', None)):
                _l = getattr(_d, 'latch', None) if _d is not None else None
                if _l is not None and len(_l) >= 3:
                    _gt.append((float(_l[0]), float(_l[1]), float(_l[2])))
            for tx, ty, tz in _gt:
                if tz - float(g) < NRC_TXZ and math.hypot(tx - float(pos[0]), ty - float(pos[1])) <= NRC_TX and self._chull_mem.open_at(tx, ty):
                    tgz.append((tx, ty))
        v0 = _omb_vel(out)
        v = v0.copy()
        at = CMK_ACC * CMK_TAU
        px = float(pos[0])
        py = float(pos[1])
        climb = False
        hold = False
        txn = False
        for x0, x1, y0, y1, H in bx:
            if tgz and any((x0 <= tx <= x1 and y0 <= ty <= y1 for tx, ty in tgz)):
                txn = True
                continue
            qx = min(max(px, x0), x1)
            qy = min(max(py, y0), y1)
            dx = qx - px
            dy = qy - py
            d = math.hypot(dx, dy)
            if d < 1e-06:
                if zr < NRC_ZC:
                    cx = 0.5 * (x0 + x1) - px
                    cy = 0.5 * (y0 + y1) - py
                    cn = math.hypot(cx, cy)
                    if cn > 1e-06:
                        e = (float(v[0]) * cx + float(v[1]) * cy) / cn
                        if e > 0.0:
                            v[0] -= e * cx / cn
                            v[1] -= e * cy / cn
                    climb = True
                else:
                    hold = True
                continue
            M = NRC_M
            for tx, ty in tg:
                M = min(M, max(0.0, math.hypot(max(x0 - tx, 0.0, tx - x1), max(y0 - ty, 0.0, ty - y1)) - CMK_TGAP))
            if zr >= NRC_ZC:
                if NRC_SH and d - SH_RB - M < NRC_RING:
                    hold = True
                continue
            ux = dx / d
            uy = dy / d
            hh = d - SH_RB - M
            l = -at + math.sqrt(at * at + 2.0 * CMK_ACC * hh) if hh >= 0.0 else -min(CMK_PUSH, -hh)
            e = float(v[0]) * ux + float(v[1]) * uy - l
            if e > 0.001:
                v[0] -= e * ux
                v[1] -= e * uy
                if float(v0[0]) * ux + float(v0[1]) * uy >= NRC_EIN:
                    climb = True
            if hh < NRC_RING and float(v0[2]) >= -NRC_DESC:
                climb = True
        if climb:
            v[2] = max(float(v[2]), NRC_VZ)
            self._nrc_cl += 1
        elif hold and float(v[2]) < 0.0:
            v[2] = 0.0
            self._nrc_hold += 1
            if NRC_HV > 0.0 and float(v0[2]) < -NRC_DESC:
                s0 = float(np.linalg.norm(v0))
                hs = math.hypot(float(v[0]), float(v[1]))
                sp = max(hs, NRC_HV)
                ux = None
                if hs > 1e-06 and hs >= NRC_HDIR * s0:
                    ux = float(v[0]) / hs
                    uy = float(v[1]) / hs
                elif tg:
                    _dx = tg[0][0] - px
                    _dy = tg[0][1] - py
                    _dn = math.hypot(_dx, _dy)
                    if _dn > 1e-06:
                        ux = _dx / _dn
                        uy = _dy / _dn
                if ux is not None and sp > hs:
                    v[0] = ux * sp
                    v[1] = uy * sp
        if txn:
            self._nrc_tx += 1
            if self._omb is not None:
                self._omb.stats['nrc_tx'] = self._nrc_tx
        if float(np.max(np.abs(v - v0))) < 1e-06:
            return out
        o = out.copy()
        _omb_store(o, v)
        self._nrc_n += 1
        if self._omb is not None:
            self._omb.stats['nrc_n'] = self._nrc_n
            self._omb.stats['nrc_cl'] = self._nrc_cl
            self._omb.stats['nrc_hold'] = self._nrc_hold
        if not self.route.endswith('+nrc'):
            self.route = self.route + '+nrc'
        return o

    def _chull_wh(self):
        return bool(getattr(self, '_z0', None) is not None and self._z0 <= WESC_Z)

    def _chull_see(self, observation, st):
        if self._chull_wh():
            return
        _t = self._stype()
        if _t is None or _t in CHL_TYPES or (CMK_ON and _t in CMK_TYPES) or (TYX_ON and _t == 'warehouse'):
            self._chull_mem.see(observation['depth'], st)

    def _chull_x(self, t):
        c = self._cls
        if TYX_ON and t == 'warehouse' and (not self._chull_wh()) and (c is not None) and (not c.is_mountain) and (c.n >= max(1, CHL_XN)):
            _x = bool(float(c.ps[0]) / max(c.n, 1) >= CHL_XP)
            if _x:
                self._tyx_n += 1
                if self._omb is not None:
                    self._omb.stats['tyx_n'] = self._tyx_n
            return _x
        if t is not None or c is None or c.is_mountain or (c.n < max(1, CHL_XN)):
            return False
        return bool(float(c.ps[0]) / max(c.n, 1) >= CHL_XP)

    def _chull(self, st, out):
        if self._cls.is_mountain or self._chull_wh():
            return out
        _t = self._stype()
        if not (_t in CHL_TYPES or (CHL_X and self._chull_x(_t))):
            return out
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        mode = int(round(float(mm[0]))) if mm.size else -1
        if mode not in CHL_MODES:
            return out
        if any((getattr(self, k, None) is not None for k in ('_r2_term', '_rp_term', '_dh_term'))):
            return out
        if self._esc is not None and self._esc.phase in ('scan', 'creep', 'climb'):
            return out
        pos = st[0:3].astype(np.float64)
        if mode == 2 and mm.size >= 63 and (math.hypot(float(mm[60]) - pos[0], float(mm[61]) - pos[1]) < CHL_RT):
            return out
        for _d in (getattr(self, '_rp', None), getattr(self, '_r2', None), getattr(self, '_dh', None)):
            _l = getattr(_d, 'latch', None) if _d is not None else None
            if _l is not None and math.hypot(float(_l[0]) - pos[0], float(_l[1]) - pos[1]) < CHL_RT:
                return out
        for _q in (getattr(self, '_cm', None), getattr(getattr(self, '_lm', None), 'tgt', None)):
            if _q is not None and math.hypot(float(_q[0]) - pos[0], float(_q[1]) - pos[1]) < CHL_RT:
                return out
        v = _omb_vel(out)
        if float(np.linalg.norm(v)) < CHL_VMIN:
            return out
        cons = self._chull_mem.cons(pos, self.tick)
        if not cons:
            return out
        at = CHL_ACC * CHL_TAU
        L = []
        for u, d in cons:
            h = d - SH_RB - CHL_M
            L.append(-at + math.sqrt(at * at + 2.0 * CHL_ACC * h) if h >= 0.0 else -min(CHL_PUSH, -h))
        hit = False
        for _ in range(6):
            w = None
            for (u, d), l in zip(cons, L):
                e = float(v @ u) - l
                if e > 0.001 and (w is None or e > w[0]):
                    w = (e, u)
            if w is None:
                break
            v = v - w[0] * w[1]
            hit = True
        if not hit:
            return out
        o = out.copy()
        _omb_store(o, v)
        self._chull_n += 1
        if self._omb is not None:
            self._omb.stats['chull_n'] = self._chull_n
        if not self.route.endswith('+chull'):
            self.route = self.route + '+chull'
        return o

    def _chl_emax(self, st, o0, o1):
        vel = np.asarray(st[6:9], np.float64)
        v0 = _omb_vel(o0)
        v1 = _omb_vel(o1)
        ex = float(v1[0] - vel[0])
        ey = float(v1[1] - vel[1])
        en = math.hypot(ex, ey)
        cap = max(CHL_EMAX, math.hypot(float(v0[0] - vel[0]), float(v0[1] - vel[1])))
        if not en > cap + 1e-06 or not math.isfinite(en):
            return o1
        v = v1.copy()
        v[0] = float(vel[0]) + ex / en * cap
        v[1] = float(vel[1]) + ey / en * cap
        o = o1.copy()
        _omb_store(o, v)
        self._chl_lim += 1
        if self._omb is not None:
            self._omb.stats['chl_lim'] = self._chl_lim
        return o

    def _paced_ok(self, det, st):
        if not E2P_ON:
            return True
        if self._rgb_n >= 40 - E2P_RESERVE:
            return False
        c0 = getattr(det, 'clue0', None)
        if c0 is None:
            return True
        return float(np.linalg.norm(np.asarray(st[0:2], np.float64) - np.asarray(c0, np.float64))) <= E2P_R

    def _tk_decide(self, obs, st, tp, hz):
        fr0 = float(self._ss_fr[0])
        S = bool(tp <= SS_TOP)
        Ffwd = bool(fr0 > SS_F0)
        F = bool(fr0 >= TK_F)
        yaw0 = float(self._ss_yaw0)
        yaw = float(st[5])
        pos = st[0:3].astype(np.float64)
        rpy = st[3:6].astype(np.float64)
        z0 = float(self._z0)
        od = TK_OUTD if TK_OUTD > 0.0 else SS_OUT

        def _fwd():
            d = min(od, _tk_dmax(self._tk_d1, self._tk_p1, self._tk_r1, yaw0, z0))
            return (yaw0, d) if d >= TK_OUT_MIN else None
        _c = np.asarray(self._tk_d1, np.float32).reshape(256, 256)[0:8, 16:240] * (_DMAX - _DMIN) + _DMIN
        ftop = float((_c <= _DMIN + 0.005).mean())
        self._tk_dbg = (round(fr0, 2), round(tp, 2), round(hz, 2), round(ftop, 2))
        if S and hz < SS_HMAX:
            if Ffwd:
                return None
            r = _fwd()
            return (yaw0 + math.pi, od, 'back2') if r is None else (r[0], r[1], 'fwd')
        if S and (not Ffwd) and (hz < TK_HZ2):
            r = _fwd()
            return None if r is None else (r[0], r[1], 'fwd2')
        if S and F and (ftop >= TK_FTOP):
            return (yaw + math.pi, TK_SIDE_D, 'oside')
        if F and (not S) and (ftop >= TK_FTOP):
            d = min(TK_SIDE_D, _tk_dmax(obs['depth'], pos, rpy, yaw, z0))
            if d >= TK_OUT_MIN:
                return (yaw, d, 'side')
        return None

    def _tk_guard(self, obs, st, out):
        if self.tick > TK_TEND:
            self._tk_gdone = True
            return out
        pos = st[0:3].astype(np.float64)
        rpy = st[3:6].astype(np.float64)
        z = float(pos[2])
        z0 = float(self._z0)
        if not self._tk_sp_done:
            self._tk_sp_done = True
            q0, zr0 = _tk_plates(self._tk_d1, self._tk_p1, self._tk_r1, z0)
            self._tk_add(q0)
            self._tk_zsp = zr0 if TK_SPAWN else None
        q, zr = _tk_plates(obs['depth'], pos, rpy, z0)
        self._tk_add(q)
        conf = zr
        if self._tk_cells:
            K = np.array(list(self._tk_cells.keys()), np.float64) * 0.25
            Z = np.array(list(self._tk_cells.values()))
            near = np.hypot(K[:, 0] - pos[0], K[:, 1] - pos[1]) <= TK_COL
            if near.any():
                zn = float(Z[near].min())
                conf = zn if conf is None else min(conf, zn)
        if conf is not None and conf < z - 0.05:
            conf = None
        zc = conf
        if self._tk_zsp is not None:
            if float(np.hypot(pos[0] - self._tk_p1[0], pos[1] - self._tk_p1[1])) <= 0.35 and self._tk_zsp >= z - 0.05:
                zc = self._tk_zsp if zc is None else min(zc, self._tk_zsp)
                if z >= self._tk_zsp - TK_GAP - 0.3:
                    self._tk_spn += 1
                    if self._tk_spn > TK_REL:
                        self._tk_zsp = None
            else:
                self._tk_zsp = None
        if zc is None:
            self._tk_free_n += 1
            if self._tk_cap and self._tk_free_n >= TK_REL:
                self._tk_cap = 0
                self._tk_phase = 'rel'
            if not self._tk_cap and (z >= z0 + 4.2 or float(np.hypot(pos[0] - self._pad[0], pos[1] - self._pad[1])) > 8.0):
                self._tk_gdone = True
            if not self._tk_cap:
                return out
            zc = self._tk_zc
        else:
            self._tk_free_n = 0
            self._tk_zc = zc
        zt = float(zc) - TK_GAP
        vzmax = float(np.clip(2.0 * (zt - z), -0.8, 2.2))
        dv = np.asarray(out[0:3], np.float64)
        n = float(np.linalg.norm(dv))
        sp = float(out[3]) * 3.0
        v = dv / n * sp if n > 1e-06 else np.zeros(3)
        if v[2] > vzmax:
            self._tk_cap += 1
            self._tk_phase = 'cap'
            v[2] = vzmax
            nv = float(np.linalg.norm(v))
            out = out.copy()
            if nv > 1e-06:
                out[0:3] = (v / nv).astype(np.float32)
                out[3] = np.float32(min(1.0, nv / 3.0))
            else:
                out[0:3] = np.array([0.0, 0.0, 1.0], np.float32)
                out[3] = np.float32(0.0)
            if not self.route.endswith('+tkp'):
                self.route = self.route + '+tkp'
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        if self._tk_cap and conf is not None and mm.size and (int(round(float(mm[0]))) == 0) and (z >= zt - 0.25):
            self._tk_hold += 1
            if self._tk_hold >= TK_HOLD1:
                mm = mm.copy()
                mm[0] = np.float32(1.0)
                self.memory_tensor = mm.reshape(self._ms)
                self._tk_forced += 1
                self._tk_phase = 'm1'
        return out

    def _tk_add(self, q):
        if q is None or len(q) == 0:
            return
        ix = np.round(q[:, 0] / 0.25).astype(np.int64)
        iy = np.round(q[:, 1] / 0.25).astype(np.int64)
        for kx, ky, kz in zip(ix.tolist(), iy.tolist(), q[:, 2].tolist()):
            k = (kx, ky)
            o = self._tk_cells.get(k)
            if o is None or kz < o:
                self._tk_cells[k] = kz

    def _wxf_add(self, obs, st):
        Q, T = _wxf_frame(obs['depth'], st[0:3].astype(np.float64), st[3:6].astype(np.float64), self._pad, float(self._z0))
        self._wxf_Q.append(_wxf_dedup(Q).astype(np.float32))
        self._wxf_Tq.append(_wxf_dedup(T).astype(np.float32))

    def _wxf_begin(self, obs, st, out):
        yaw = float(st[5])
        sg = float(self._ss_sgn)
        side = self._ss_yaw0 + sg * math.radians(SS_SIDE)
        self._wxf_on = True
        self._ss = 'rscan'
        self._wxf_t = 0
        self._wxf_rot = 0.0
        self._wxf_py = yaw
        self._wxf_need = 180.0 - math.degrees(sg * _wrap(yaw - side)) if WXF_REAR else 0.0
        self._tk_why = 'wxf'
        self._tk_phase = 'wxf'
        return self._wxf_hover(st, out, self._wxf_need)

    def _wxf_hover(self, st, out, rem):
        z = float(st[2])
        yaw = float(st[5])
        sg = float(self._ss_sgn)
        o = out.copy()
        o[5] = np.float32(0.0)
        zt = self._ss_z0 + SS_LIFT
        vz = float(np.clip(2.0 * (zt - z), -SS_VZ, SS_VZ))
        o[0:3] = np.asarray([0.0, 0.0, 1.0 if vz >= 0 else -1.0], np.float32)
        o[3] = np.float32(abs(vz) / 3.0)
        o[4] = np.float32(_wrap(yaw + sg * math.radians(max(0.0, min(WXF_LEAD, rem)))) / math.pi)
        self.route = 'king+ssf'
        return o

    def _wxf_rscan(self, obs, st, out):
        z = float(st[2])
        yaw = float(st[5])
        sg = float(self._ss_sgn)
        self._wxf_rot += math.degrees(sg * _wrap(yaw - self._wxf_py))
        self._wxf_py = yaw
        self._wxf_t += 1
        rem = self._wxf_need - self._wxf_rot
        if rem <= WXF_TOL or self._wxf_t >= WXF_TMAX:
            try:
                self._wxf_add(obs, st)
            except Exception:
                self._wxf_err += 1
            T = None
            try:
                T = self._wxf_plan()
            except Exception:
                self._wxf_err += 1
                T = None
            self._wxf_stats()
            self._ss_hit = True
            self._ss_start = st[0:2].astype(np.float64).copy()
            self._ss = 'xit'
            self._wx_n += 1
            if T is None:
                self._wx_T = np.asarray(self._pad, np.float64).copy()
                self._wx_leg = 2
            else:
                self._wx_T = T
                self._wx_leg = 1
            return None
        return self._wxf_hover(st, out, rem)

    def _wxf_plan(self):
        z0 = float(self._z0)
        Q = np.concatenate(self._wxf_Q, 0).astype(np.float64) if self._wxf_Q else np.zeros((0, 3))
        Qt = np.concatenate(self._wxf_Tq, 0).astype(np.float64) if self._wxf_Tq else np.zeros((0, 2))
        rlim = max(WXF_RS + (0.0,)) + WXF_CAP + WXF_BODY + 0.1
        if len(Q):
            Q = Q[np.hypot(Q[:, 0], Q[:, 1]) < rlim]
            Q = _wxf_dedup(Q)
        if len(Qt):
            Qt = Qt[np.hypot(Qt[:, 0], Qt[:, 1]) < rlim]
            Qt = _wxf_dedup(Qt)
        p0, b0 = _wxf_eval(Q, Qt, 0.0, 0.0, 0.0, z0)
        p0 = 0.0 if b0 else min(p0, WXF_CAP)
        y0 = float(self._ss_yaw0)
        sg = float(self._ss_sgn)
        best = None
        for r in WXF_RS:
            for k in range(24):
                az = math.radians(15.0 * k)
                if not WXF_REAR:
                    rel = math.degrees(_wrap((az - y0) * sg))
                    if rel < WX_SEC0 or rel > WX_SEC1:
                        continue
                ux, uy = (math.cos(az), math.sin(az))
                if _wxf_path(Q, ux, uy, r, z0):
                    continue
                c, b = _wxf_eval(Q, Qt, r * ux, r * uy, r, z0)
                if b:
                    continue
                c = min(c, WXF_CAP)
                sc = c - WXF_LAM * r
                if best is None or sc > best[0]:
                    best = (sc, r, az, c)
        self._wxf_dbg = (round(p0, 3), None if best is None else best[1], None if best is None else int(round(math.degrees(best[2]))), None if best is None else round(best[3], 3), int(len(Q)), int(len(Qt)), int(self._wxf_t))
        if best is None or best[3] - p0 < WXF_DELTA:
            return None
        return np.array([self._pad[0] + best[1] * math.cos(best[2]), self._pad[1] + best[1] * math.sin(best[2])], np.float64)

    def _wxf_stats(self):
        if self._omb is None or self._wxf_dbg is None:
            return
        d = self._wxf_dbg
        s = self._omb.stats
        s['wxf_p0'] = float(d[0])
        s['wxf_r'] = -1.0 if d[1] is None else float(d[1])
        s['wxf_az'] = -1 if d[2] is None else int(d[2])
        s['wxf_c'] = -1.0 if d[3] is None else float(d[3])
        s['wxf_nq'] = int(d[4])
        s['wxf_nt'] = int(d[5])
        s['wxf_t'] = int(d[6])
        s['wxf_err'] = int(self._wxf_err)
        s['wxf_cp'] = int(self._wxf_cp)

    def _wcr_stats(self):
        o = self._omb
        if o is None:
            return
        s = o.stats
        s['wcr_t1'] = int(self._wcr_t0)
        s['wcr_dx'] = int(round(100.0 * self._wcr_dx0))
        s['wcr_rib'] = float(self._wcr_rib)
        s['wcr_n'] = int(self._wcr_n)
        s['wcr_blk'] = int(self._wcr_nblk)
        s['wcr_free'] = int(round(100.0 * min(30.0, self._wcr_free)))
        s['wcr_side'] = int(self._wcr_side)
        s['wcr_why'] = str(self._wcr_why)
        s['wcr_te'] = int(self._wcr_te)
        s['wcr_err'] = int(self._wcr_err)
        s['wcr_y'] = int(round(100.0 * self._wcr_y0))
        s['wcr_z'] = int(round(100.0 * self._wcr_z0))
        s['wcr_zl'] = int(round(100.0 * self._wcr_zl))
        s['wcr_f1'] = int(round(100.0 * min(30.0, self._wcr_f1)))
        s['wcr_f2'] = int(round(100.0 * min(30.0, self._wcr_f2)))
        s['wcr_s1'] = int(self._wcr_s1)
        s['wcr_tg'] = int(self._wcr_tg)

    def _wcr_end(self, why):
        self._wcr_ph = 'done' if why == 'done' else 'abort'
        self._wcr_why = why
        self._wcr_te = self.tick
        self._wcr_stats()
        return None

    def _wcr(self, obs, st, out):
        pos = st[0:3].astype(np.float64)
        x = float(pos[0])
        y = float(pos[1])
        z = float(pos[2])
        if self._wcr_ph is None:
            if not WCR_RIBS or z >= WCR_ZMAX or self._z0 is None:
                return None
            rib = min(WCR_RIBS, key=lambda r: abs(x - r))
            dx = x - rib
            if abs(dx) >= WCR_BAND or (WCR_Y and _wcr_ydist(y) >= WCR_YB):
                return None
            sd = 1.0 if dx > 0.0 else -1.0
            if abs(dx) < 0.05 and self._cb is not None:
                sd = 1.0 if float(self._cb[0]) >= rib else -1.0
            self._wcr_ph = 'up'
            self._wcr_rib = rib
            self._wcr_side = sd
            self._wcr_s1 = sd
            self._wcr_tried = 1
            self._wcr_t0 = self.tick
            self._wcr_dx0 = dx
            self._wcr_y0 = y
            self._wcr_z0 = z
            self._wcr_zl = max(float(self._z0) + WCR_ZL, z)
        if self.tick - self._wcr_t0 > WCR_TMAX:
            return self._wcr_end('tmax')
        rib = self._wcr_rib
        zl = self._wcr_zl
        yaw = float(st[5])
        vzn = float(st[8])
        rpy = st[3:6].astype(np.float64)
        sd = self._wcr_side
        hd = 0.0 if sd > 0.0 else math.pi
        rem = WCR_OUT - (x - rib) * sd
        if rem <= 0.0:
            return self._wcr_end('done')
        zhi = min(z + WCR_DZU, WCR_ZTOP)
        if self._wcr_ph == 'up' and abs(zl - z) < 0.12 and (abs(vzn) < 0.4) and (abs(_wrap(yaw - hd)) < math.radians(WCR_YAWT)):
            fr = _tk_free(obs['depth'], pos, rpy, hd, WCR_HW, z - WCR_DZ, zhi, 0.0)
            self._wcr_free = fr
            if self._wcr_tried < 2:
                self._wcr_f1 = fr
            else:
                self._wcr_f2 = fr
            if fr >= rem + WCR_M:
                self._wcr_ph = 'go'
                self._wcr_tg = self.tick
            else:
                self._wcr_nblk += 1
                if self._wcr_tried >= 2:
                    return self._wcr_end('blk')
                self._wcr_tried = 2
                sd = -sd
                self._wcr_side = sd
                hd = 0.0 if sd > 0.0 else math.pi
                rem = WCR_OUT - (x - rib) * sd
        o = out.copy()
        if self._wcr_ph == 'go':
            fr = _tk_free(obs['depth'], pos, rpy, hd, WCR_HW, z - WCR_DZ, zhi, 0.0)
            self._wcr_free = fr
            if fr < rem + WCR_MG:
                self._wcr_nblk += 1
                return self._wcr_end('blk2')
            vh = min(WCR_V, 1.2 * rem + 0.15)
            vz = float(np.clip(2.0 * (zl - z), -0.5, 0.5))
            v = np.array([sd * vh, 0.0, vz], np.float64)
            n = float(np.linalg.norm(v))
            o[0:3] = (v / n).astype(np.float32)
            o[3] = np.float32(min(1.0, n / 3.0))
        else:
            vz = float(np.clip(2.0 * (zl - z), -1.0, 1.5))
            o[0:3] = np.array([0.0, 0.0, 1.0 if vz >= 0.0 else -1.0], np.float32)
            o[3] = np.float32(abs(vz) / 3.0)
        o[4] = np.float32(_wrap(hd) / math.pi)
        self._wcr_n += 1
        if not self.route.endswith('+wcr'):
            self.route = self.route + '+wcr'
        self._wcr_stats()
        return o

    def _whk(self, obs, st, out):
        if self._ss not in ('off', 'done') or (self._esc is not None and self._esc.phase in ('scan', 'creep', 'climb')):
            self._whk_q = []
            return out
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        ph = float(mm[0]) if mm.size else 0.0
        pos = st[0:3].astype(np.float64)
        z = float(pos[2])
        r2 = bool(self._r2 is not None and self._r2_on and (self._r2.latch is not None))
        cl = bool(not self._wh_high and float(np.hypot(pos[0] - self._pad[0], pos[1] - self._pad[1])) < 3.0)
        if WHK_CLIMB:
            g = z >= self._z0 + WHK_ZC and (cl or ph >= WHK_PH or r2)
        else:
            g = z >= self._z0 + WHK_ZC and (not cl) and (ph >= WHK_PH or r2)
        if not g:
            self._whk_q = []
            return out
        if self.tick % WHK_EVERY == 0:
            self._whk_q.append(_whk_pts(obs['depth'], pos, st[3:6].astype(np.float64)).astype(np.float32))
            if len(self._whk_q) > WHK_MEM:
                self._whk_q.pop(0)
        Q = np.concatenate(self._whk_q, 0).astype(np.float64) if self._whk_q else np.zeros((0, 3))
        if not len(Q):
            return out
        D = Q - pos[None, :]
        D = D[np.abs(D[:, 2]) <= WHK_DZ]
        if not len(D):
            return out
        dd = np.hypot(D[:, 0], D[:, 1])
        m = dd < WHK_R
        if not m.any():
            return out
        w = (WHK_R - dd[m]) / WHK_R
        nh = (D[m][:, 0:2] / np.maximum(dd[m], 0.001)[:, None] * w[:, None]).sum(0)
        nn = float(np.hypot(nh[0], nh[1]))
        if nn < 1e-06:
            return out
        nh = nh / nn
        dist = float(dd[m].min())
        self._whk_min = min(self._whk_min, dist)
        dv = np.asarray(out[0:3], np.float64)
        n = float(np.linalg.norm(dv))
        vel = dv / n * float(out[3]) * 3.0 if n > 1e-06 else np.zeros(3)
        hit = False
        tow = float(vel[0] * nh[0] + vel[1] * nh[1])
        if tow > 0.0:
            vel[0:2] -= tow * nh
            hit = True
        push = WHK_V * min(1.0, max(0.0, (WHK_R - dist) / max(WHK_R - 0.4, 0.001)))
        if push > 0.0:
            vel[0:2] -= push * nh
            hit = True
        if not hit:
            return out
        o = out.copy()
        sp = float(np.linalg.norm(vel))
        if sp < 0.001:
            o[3] = np.float32(0.0)
        else:
            o[0:3] = (vel / sp).astype(np.float32)
            o[3] = np.float32(min(1.0, sp / 3.0))
        self._whk_n += 1
        if self._whk_t == 0:
            self._whk_t = self.tick
        if self._omb is not None:
            self._omb.stats['whk_n'] = int(self._whk_n)
            self._omb.stats['whk_t'] = int(self._whk_t)
            self._omb.stats['whk_min'] = float(self._whk_min)
        if not self.route.endswith('+whk'):
            self.route = self.route + '+whk'
        return o

    def _whd(self, obs, st, out):
        if self._ss not in ('off', 'done') or (self._esc is not None and self._esc.phase in ('scan', 'creep', 'climb')):
            self._whd_q = []
            self._whd_th = 0.0
            return out
        pos = st[0:3].astype(np.float64)
        z = float(pos[2])
        if z < WHD_ZMIN:
            self._whd_q = []
            self._whd_th = 0.0
            return out
        P, dm = _tk_pts(obs['depth'], pos, st[3:6].astype(np.float64), 4, 0, 256)
        P = P.reshape(-1, 3)
        dm = dm.reshape(-1)
        zc = z + max(WHD_DZHI, WHD_TALL + 0.3, 0.0)
        m = (dm < _DMAX - 0.2) & (dm > _DMIN + 0.005) & (P[:, 2] >= z - WHD_DZLO) & (P[:, 2] <= (z + 3.0 if WHD_LOG else zc))
        P = P[m]
        if len(P):
            P = P[np.hypot(P[:, 0] - pos[0], P[:, 1] - pos[1]) < WHD_LOOK]
        Pl = None
        if WHD_LOG:
            Pl = P[P[:, 2] > zc].copy()
            P = P[P[:, 2] <= zc]
        self._whd_q.append((self.tick, P.copy(), Pl))
        while self._whd_q and self.tick - self._whd_q[0][0] > WHD_MEM:
            self._whd_q.pop(0)
        dv = np.asarray(out[0:3], np.float64)
        n = float(np.linalg.norm(dv))
        sp = float(out[3]) * 3.0
        vh = dv[0:2] / n * sp if n > 1e-06 else np.zeros(2)
        vn = float(np.hypot(vh[0], vh[1]))
        th = 0.0
        if vn >= WHD_VMIN and getattr(self, '_r2_term', None) is None:
            mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
            md = int(round(float(mm[0]))) if mm.size else 1
            tg = None
            if md >= 3 and mm.size >= 11:
                tg = mm[8:10].astype(np.float64)
            elif md == 2 and mm.size >= 63 and (float(mm[57]) > 0.5):
                tg = mm[60:62].astype(np.float64)
            elif self._r2 is not None and self._r2_on and (self._r2.latch is not None):
                tg = np.asarray(self._r2.latch[0:2], np.float64)
            lim = WHD_LOOK
            near = False
            if tg is not None:
                dt = float(np.hypot(tg[0] - pos[0], tg[1] - pos[1]))
                near = dt < WHD_TGT
                lim = min(lim, dt)
            Qs = [q for _, q, _ in self._whd_q if len(q)]
            if not near and Qs:
                Q = np.concatenate(Qs, 0)
                R = Q[:, 0:2] - pos[0:2]
                kb = (np.hypot(R[:, 0], R[:, 1]) < WHD_LOOK) & (Q[:, 2] >= z - WHD_DZLO) & (Q[:, 2] <= z + WHD_DZHI)
                if WHD_TALL > 0.0 and kb.any():
                    Ht = Q[Q[:, 2] >= z + WHD_TALL, 0:2]
                    if len(Ht):
                        cs = set(map(tuple, np.floor(Ht / 0.3).astype(np.int64).tolist()))
                        cb = np.floor(Q[kb, 0:2] / 0.3).astype(np.int64)
                        ok_t = np.array([any(((a + i, b + j) in cs for i in (-1, 0, 1) for j in (-1, 0, 1))) for a, b in cb.tolist()], bool)
                        ii = np.where(kb)[0]
                        kb[ii[~ok_t]] = False
                    else:
                        kb[:] = False
                R = R[kb]
                if len(R):
                    uh = vh / vn
                    ks = np.radians(np.arange(-WHD_MAXA, WHD_MAXA + 1e-06, WHD_STEP))
                    ks = ks[np.argsort(np.abs(ks), kind='stable')]
                    U = np.stack([uh[0] * np.cos(ks) - uh[1] * np.sin(ks), uh[0] * np.sin(ks) + uh[1] * np.cos(ks)], 1)
                    A = R @ U.T
                    L = R @ np.stack([-U[:, 1], U[:, 0]], 1).T
                    inc = (A > 0.05) & (A < lim) & (np.abs(L) < WHD_R)
                    if bool(inc[:, int(np.argmin(np.abs(ks)))].any()):
                        self._whd_ev += 1
                        ok = ~inc.any(axis=0)
                        if WHD_LOG and len(self._whd_lg) < 400:
                            _c = inc[:, int(np.argmin(np.abs(ks)))]
                            _Qc = Q[kb][_c][:80]
                            _QA = np.concatenate([Q] + [a for _, _, a in self._whd_q if a is not None and len(a)], 0)
                            _Qa = _QA[(np.hypot(_QA[:, 0] - pos[0], _QA[:, 1] - pos[1]) < WHD_LOOK) & (_QA[:, 2] > z + WHD_DZHI)][:20000]
                            _near = _Qa[np.min(np.hypot(_Qa[:, 0:1] - _Qc[:, 0][None, :], _Qa[:, 1:2] - _Qc[:, 1][None, :]), axis=1) < 0.3] if len(_Qc) else _Qa[:0]
                            self._whd_lg.append([int(self.tick), round(z, 2), int(md), round(float(_Qc[:, 2].min()), 2), round(float(_Qc[:, 2].max()), 2), round(float(_near[:, 2].max()), 2) if len(_near) else None, int(len(_Qc)), round(float(np.min(np.hypot(_Qc[:, 0] - pos[0], _Qc[:, 1] - pos[1]))), 2)])
                        if ok.any():
                            th = float(ks[int(np.argmax(ok))])
                        else:
                            W = np.where((A > 0.05) & (A < lim), np.abs(L), 9.0)
                            th = float(ks[int(np.argmax(W.min(axis=0)))])
        st_ = math.radians(WHD_RATE)
        self._whd_th += max(-st_, min(st_, th - self._whd_th))
        if abs(self._whd_th) < 1e-09:
            self._whd_th = 0.0
            return out
        c = math.cos(self._whd_th)
        s = math.sin(self._whd_th)
        o = out.copy()
        o[0] = np.float32(c * float(out[0]) - s * float(out[1]))
        o[1] = np.float32(s * float(out[0]) + c * float(out[1]))
        self._whd_n += 1
        self._whd_amax = max(self._whd_amax, abs(math.degrees(self._whd_th)))
        if self._whd_t1 == 0:
            self._whd_t1 = self.tick
        if not self.route.endswith('+whd'):
            self.route = self.route + '+whd'
        return o

    def _wx_veto(self, obs, st):
        try:
            Qh, Ql = _wx_frame(obs['depth'], st[0:3].astype(np.float64), st[3:6].astype(np.float64), self._pad, float(self._z0))
            p0, _ = _wx_pc(Qh, Ql, 0.0, 0.0)
        except Exception:
            return False
        self._wx_dbg = ('veto', round(p0, 2))
        return p0 >= WX_CEILP

    def _wx_plan(self, st, lat):
        if not self._wx_h:
            return None
        Qh = np.concatenate(self._wx_h, 0)
        Ql = np.concatenate(self._wx_l, 0)
        if len(Qh) > 40000:
            Qh = Qh[np.unique(np.round(Qh / 0.03).astype(np.int64), axis=0, return_index=True)[1]]
        if len(Ql) > 40000:
            Ql = Ql[np.unique(np.round(Ql / 0.03).astype(np.int64), axis=0, return_index=True)[1]]
        y0 = float(self._ss_yaw0)
        sg = float(self._ss_sgn)
        if lat is None:
            p0, _ = _wx_pc(Qh, Ql, 0.0, 0.0)
        else:
            p0, _ = _wx_pc(Qh, Ql, float(lat[1]), float(lat[0]))
        best = None
        for r in WX_RS:
            if r > WX_RMAX + 1e-09:
                continue
            for k in range(24):
                az = math.radians(15.0 * k)
                rel = math.degrees(_wrap((az - y0) * sg))
                if rel < WX_SEC0 or rel > WX_SEC1:
                    continue
                pc, blk = _wx_pc(Qh, Ql, r, az)
                if blk:
                    continue
                sc = pc - WX_LAM * r
                if best is None or sc > best[0]:
                    best = (sc, r, az, pc)
        self._wx_n += 1
        if best is None:
            self._wx_dbg = (round(p0, 2), None)
            return None
        _, r, az, pc = best
        self._wx_dbg = (round(p0, 2), r, round(math.degrees(az)), round(pc, 2))
        if pc - p0 < WX_DELTA or p0 >= WX_P0CAP:
            return None
        return np.array([self._pad[0] + r * math.cos(az), self._pad[1] + r * math.sin(az)], np.float64)

    def _whg_step(self, obs, st):
        pos = st[0:3].astype(np.float64)
        z0 = float(self._z0)
        if self.tick % WHG_EVERY == 0:
            _h, _l = _wx_frame(obs['depth'], pos, st[3:6].astype(np.float64), self._pad, z0)
            if len(_h):
                self._whg_h.append(_h)
        else:
            return False
        if abs(_wrap(float(self._ss_head) - float(self._ss_yaw0))) > math.radians(WHG_HEAD):
            return False
        qp = pos[0:2] - np.asarray(self._pad, np.float64) + st[6:8].astype(np.float64) * WHG_TAU
        d = float(np.hypot(qp[0], qp[1]))
        if d < WHG_DMIN or d >= 0.999:
            return False
        H = [a for a in self._wx_h if len(a)] + self._whg_h
        s = 9.0
        if H:
            Q = np.concatenate(H, 0)
            s = float(np.min(np.hypot(Q[:, 0] - qp[0], Q[:, 1] - qp[1])))
        f = z0 + math.sqrt(max(0.0, 1.0 - d * d)) - WHG_ZB
        self._whg_ev += 1
        if s - WHG_BODY >= f + WHG_M:
            self._whg_rec = (d, s, f)
            return True
        return False

    def _sscan(self, obs, st, out):
        z = float(st[2])
        yaw = float(st[5])
        if self._ss is None:
            if self.tick != 1 or z > SS_Z or (self._esc is not None and self._esc.mode == 'ceil'):
                self._ss = 'off'
                return None
            self._ss = 'scan'
            self._ss_z0 = z
            self._ss_yaw0 = yaw
            self._ss_t = 0
            f0, h0 = _ss_frac(obs['depth'])
            self._ss_fr = [f0]
            self._ss_hor = [h0]
            _c = _wrap(float(out[4]) * math.pi - yaw)
            self._ss_sgn = 1.0 if _c > 0.0001 else -1.0
        if self._ss in ('off', 'done'):
            if self.route in ('king+ss', 'king+ssf'):
                self.route = 'king'
            return None
        self._ss_t += 1
        if WXF_ON and self._tk_on and (self._ss in ('scan', 'rscan')) and (self.tick % WX_STEP == 1):
            try:
                self._wxf_add(obs, st)
            except Exception:
                self._wxf_err += 1
        if self._ss == 'rscan':
            _o = self._wxf_rscan(obs, st, out)
            if _o is not None:
                return _o
        if self._ss == 'scan' and WX_ON and WX_PLAN and self._tk_on and (self.tick % WX_STEP == 1):
            try:
                _h, _l = _wx_frame(obs['depth'], st[0:3].astype(np.float64), st[3:6].astype(np.float64), self._pad, float(self._z0))
                self._wx_h.append(_h)
                self._wx_l.append(_l)
            except Exception:
                pass
        if self._ss_t > SS_MAXT + (WXF_XT if self._wxf_on else 0):
            self._ss = 'done'
            self.route = 'king'
            return None
        o = out.copy()
        o[5] = np.float32(0.0)
        zt = self._ss_z0 + SS_LIFT
        vz = float(np.clip(2.0 * (zt - z), -SS_VZ, SS_VZ))

        def _cmd(dirv, spd, hd):
            o[0:3] = np.asarray(dirv, np.float32)
            o[3] = np.float32(spd)
            o[4] = np.float32(_wrap(hd) / math.pi)
            self.route = 'king+ss'
            return o
        side = self._ss_yaw0 + self._ss_sgn * math.radians(SS_SIDE)
        if self._ss == 'scan':
            air = z > self._ss_z0 + 0.04
            if air and z > self._ss_z0 + 0.08 and (abs(_wrap(yaw - side)) < math.radians(6.0)):
                f, hz = _ss_frac(obs['depth'])
                self._ss_fr.append(f)
                self._ss_hor.append(hz)
                tp = _ss_top(obs['depth'])
                if not self._ss_diag:
                    self._ss_diag = True
                    self._ss_fr.pop()
                    self._ss_hor.pop()
                    o[0:3] = np.asarray([0.0, 0.0, 1.0 if vz >= 0 else -1.0], np.float32)
                    o[3] = np.float32(abs(vz) / 3.0)
                    o[4] = np.float32(_wrap(side) / math.pi)
                    self.route = 'ssm%+d_%02d_%03d_%03d' % (int(self._ss_sgn), min(99, int(self._ss_fr[0] * 100)), min(999, int(tp * 100)), min(999, int(hz * 100)))
                    return o
                _tkd = None
                if self._tk_on and TK_LOW:
                    try:
                        _tkd = self._tk_decide(obs, st, tp, hz)
                    except Exception:
                        _tkd = None
                if _tkd is not None:
                    self._ss_head = _tkd[0]
                    self._ss_outd = _tkd[1]
                    self._tk_why = _tkd[2]
                    self._tk_phase = _tkd[2]
                    self._ss = 'out'
                    self._ss_hit = True
                    self._ss_start = st[0:2].astype(np.float64).copy()
                elif tp <= SS_TOP and hz < SS_HMAX:
                    self._ss_head = self._ss_yaw0 + (math.pi if self._ss_fr[0] > SS_F0 else 0.0)
                    self._ss = 'out'
                    self._ss_hit = True
                    self._ss_start = st[0:2].astype(np.float64).copy()
                else:
                    _wxf = self._wxf_begin(obs, st, out) if WXF_ON and WX_ON and self._tk_on and (not self._wx_vetoed) else None
                    if _wxf is not None:
                        return _wxf
                    _wl = None
                    if WL_ON and self._tk_on and (self._cls is not None) and (not self._cls.is_mountain) and (self._stype() in ('warehouse', 'forest', None)):
                        try:
                            _dl, _dr = _wl_faces(obs['depth'], st[0:3].astype(np.float64), st[3:6].astype(np.float64), float(self._z0))
                            _dn, _df, _sg = (_dl, _dr, 1.0) if _dl <= _dr else (_dr, _dl, -1.0)
                            if math.isfinite(_dn):
                                _tg = min(0.5 * (_dn + _df), 1.1) if math.isfinite(_df) else WL_TGT
                                _sh = min(WL_MAXS, _tg - _dn)
                                self._wl_dbg = (round(_dl, 2), round(_dr, 2), round(_sh, 2))
                                if _sh >= WL_MIN:
                                    _wl = (yaw + (-0.5 * math.pi if _sg > 0 else 0.5 * math.pi), _sh)
                        except Exception:
                            _wl = None
                    _wx = None
                    if WX_ON and WX_PLAN and self._tk_on and (not self._wx_vetoed):
                        try:
                            _wx = self._wx_plan(st, None if _wl is None else (_wl[0], _wl[1]))
                        except Exception:
                            _wx = None
                    if _wx is not None:
                        self._wx_T = _wx
                        self._ss = 'xit'
                        self._wx_leg = 1
                        self._tk_why = 'wx'
                        self._tk_phase = 'wx'
                        self._ss_hit = True
                        self._ss_start = st[0:2].astype(np.float64).copy()
                    elif _wl is None:
                        self._ss = 'done'
                        self.route = 'king'
                        return None
                    if _wx is None:
                        self._ss_head = _wl[0]
                        self._ss_outd = _wl[1]
                        self._tk_why = 'lat'
                        self._tk_phase = 'lat'
                        self._ss = 'out'
                        self._ss_hit = True
                        self._ss_start = st[0:2].astype(np.float64).copy()
            else:
                return _cmd([0.0, 0.0, 1.0 if vz >= 0 else -1.0], abs(vz) / 3.0, side if air else yaw)
        if WHG_ON and self._ss == 'out' and WX_ON and WX_CREEP and self._tk_on and (self._tk_why != 'lat') and (self._ss_outd > WX_CR + 0.05):
            try:
                if self._whg_step(obs, st):
                    self._wx_T = st[0:2].astype(np.float64).copy()
                    self._ss = 'xit'
                    self._wx_leg = 2
                    self._tk_phase = 'whg'
                    self._wx_n += 1
                    self._whg_n += 1
            except Exception:
                self._whg_err += 1
            if self._omb is not None:
                self._omb.stats['whg_n'] = int(self._whg_n)
                self._omb.stats['whg_ev'] = int(self._whg_ev)
                self._omb.stats['whg_err'] = int(self._whg_err)
                if self._whg_rec is not None:
                    self._omb.stats['whg_d'] = int(round(100 * self._whg_rec[0]))
                    self._omb.stats['whg_s'] = int(round(100 * min(9.0, self._whg_rec[1])))
                    self._omb.stats['whg_f'] = int(round(100 * self._whg_rec[2]))
        if self._ss == 'out' and WX_ON and WX_CREEP and self._tk_on and (self._tk_why != 'lat') and (self._ss_outd > WX_CR + 0.05):
            if float(np.linalg.norm(st[0:2].astype(np.float64) - self._ss_start)) >= WX_CR:
                self._wx_T = st[0:2].astype(np.float64).copy()
                self._ss = 'xit'
                self._wx_leg = 2
                self._tk_phase = 'wxc'
                self._wx_n += 1
        if self._ss == 'xit':
            _p = st[0:3].astype(np.float64)
            _T = self._wx_T
            _dx = float(_T[0] - _p[0])
            _dy = float(_T[1] - _p[1])
            _dh = math.hypot(_dx, _dy)
            if self._wx_leg == 1 and (_dh <= (WXF_ARR if self._wxf_on else WX_ARR) or float(np.linalg.norm(_p[0:2] - self._ss_start)) >= float(np.hypot(_T[0] - self._ss_start[0], _T[1] - self._ss_start[1]))):
                self._wx_leg = 2
            if self._wx_leg == 1:
                h = math.atan2(_dy, _dx)
                dv = np.array([math.cos(h), math.sin(h), 0.4 * vz], np.float32)
                dv /= float(np.linalg.norm(dv))
                return _cmd(dv, min(WXF_V if self._wxf_on else WX_V, max(0.05, 2.0 * _dh)) / 3.0, yaw)
            _d3 = math.sqrt((_p[0] - self._pad[0]) ** 2 + (_p[1] - self._pad[1]) ** 2 + (_p[2] - self._z0) ** 2)
            if _d3 >= WX_EXIT and _p[2] >= self._z0 + (WXF_ZEX if self._wxf_on else WX_ZEX):
                self._ss = 'done'
                self.route = 'king'
                return None
            dv = np.array([WX_KXY * _dx, WX_KXY * _dy, 1.0], np.float32)
            dv /= float(np.linalg.norm(dv))
            return _cmd(dv, WX_VZ, yaw)
        if self._ss == 'out':
            trav = float(np.linalg.norm(st[0:2].astype(np.float64) - self._ss_start))
            if trav >= self._ss_outd:
                self._ss = 'done'
                self.route = 'king'
                return None
            h = self._ss_head
            dv = np.array([math.cos(h), math.sin(h), 0.4 * vz], np.float32)
            dv /= float(np.linalg.norm(dv))
            return _cmd(dv, SS_OUT_V / 3.0, yaw)
        self._ss = 'done'
        self.route = 'king'
        return None

    def _fguard(self, obs, st, out):
        m = self._fg_mem
        if int(round(float(m[0]))) not in FG_PH:
            self._fg_on = 0
            self._fg_h = None
            self._fg_spd = None
            return out
        pos = st[0:3].astype(np.float64)
        rel = m[8:11].astype(np.float64) - pos
        dh = float(np.hypot(rel[0], rel[1]))
        vel = st[6:9].astype(np.float64)
        if dh < FG_MINH:
            self._fg_on = 0
            self._fg_h = None
            self._fg_spd = None
            return out
        yaw = float(st[5])
        if abs(float(st[3])) > math.radians(FG_TILT) or abs(float(st[4])) > math.radians(FG_TILT):
            self._fg_h = None
            return out
        bt = math.atan2(math.sin(math.atan2(rel[1], rel[0]) - yaw), math.cos(math.atan2(rel[1], rel[0]) - yaw))
        if abs(bt) > math.radians(FG_MAXB):
            self._fg_on = 0
            self._fg_h = None
            self._fg_spd = None
            return out
        dm = np.asarray(obs['depth'], np.float32).reshape(256, 256) * 29.5 + 0.5
        col = dm[128 - FG_ROWS:128 + FG_ROWS].min(axis=0)

        def cix(b):
            return int(round((1.0 - math.tan(b)) * 128.0 - 0.5))

        def free(c, need):
            w = int(math.ceil(128.0 * FG_W / max(need, 0.5)))
            lo, hi = (c - w, c + w + 1)
            if lo < 0 or hi > 256:
                return False
            return bool(col[lo:hi].min() >= need)
        need_t = min(dh - 0.3, FG_D)
        blocked = not free(cix(bt), need_t)
        if blocked:
            if self._fg_on == 0:
                self._fg_n += 1
            self._fg_on = FG_HOLD
        elif self._fg_on > 0:
            self._fg_on -= 1
        if self._fg_on <= 0:
            self._fg_h = None
            self._fg_spd = None
            return out
        out = out.copy()
        need2 = min(dh, FG_D2)
        best = None
        for k in range(0, int(FG_MAXB) + 1, 2):
            for sg in (1, -1) if k else (1,):
                b = bt + sg * math.radians(k)
                if abs(b) > math.radians(FG_MAXB):
                    continue
                if free(cix(b), need2):
                    best = b
                    break
            if best is not None:
                break
        sp = float(np.linalg.norm(vel)) / 3.0
        if self._fg_spd is None:
            self._fg_spd = min(1.0, sp)
        if best is None:
            cap = 0.05
            hwant = self._fg_h if self._fg_h is not None else math.atan2(vel[1], vel[0]) if sp > 0.05 else yaw
        else:
            cap = FG_V / 3.0
            hwant = yaw + best
        hcur = self._fg_h if self._fg_h is not None else math.atan2(float(out[1]), float(out[0])) if abs(float(out[0])) + abs(float(out[1])) > 1e-06 else yaw
        dhd = math.atan2(math.sin(hwant - hcur), math.cos(hwant - hcur))
        lim = min(math.radians(FG_TURN), FG_ALAT / max(float(np.linalg.norm(vel[0:2])), 0.5) / 50.0)
        hnew = hcur + max(-lim, min(lim, dhd))
        self._fg_h = hnew
        k = float(np.hypot(float(out[0]), float(out[1])))
        v = np.array([math.cos(hnew) * max(k, 0.3), math.sin(hnew) * max(k, 0.3), float(out[2])])
        v = v / max(float(np.linalg.norm(v)), 1e-06)
        self._fg_spd = max(min(float(out[3]), cap), self._fg_spd - FG_DN) if float(out[3]) > cap else float(out[3])
        self._fg_spd = min(self._fg_spd, float(out[3]))
        out[0:3] = v.astype(np.float32)
        out[3] = np.float32(self._fg_spd)
        if not self.route.endswith('+fg'):
            self.route = self.route + '+fg'
        return out

    def _fau_stat(self):
        o = getattr(self, '_omb', None)
        if o is None or not isinstance(getattr(o, 'stats', None), dict):
            return
        s = o.stats
        s['fau_n'] = int(self._fau_n)
        s['fau_t'] = int(self._fau_t)
        s['fau_t1'] = int(self._fau_t1)
        s['fau_pl'] = int(self._fau_pl)
        s['fau_np'] = int(self._fau_np)
        s['fau_ok'] = int(self._fau_ok)
        s['fau_why'] = self._fau_why
        s['fau_c'] = round(float(self._fau_c), 2)
        s['fau_ms'] = int(self._fau_ms + 0.5)
        s['fau_p3'] = int(self._fau_p3)
        s['fau_err'] = int(self._fau_err)
        s['fau_hd'] = int(self._fau_hd)
        s['fau_kp'] = int(self._fau_kp)
        s['fau_pts'] = list(self._fau_pts)

    def _fau_end(self, why):
        self._fau = None
        self._fau_why = why
        self._fau_ok += int(why == 'done')
        self._fau_w = None
        if why != 'done':
            self._fau_cool = self.tick + FAU_COOL

    def _fau_mem(self):
        L = []
        if self._omb is not None and getattr(self._omb, 'pts', None) is not None and len(self._omb.pts):
            L.append(np.asarray(self._omb.pts, np.float64))
        if self._fsl_P is not None and len(self._fsl_P):
            L.append(np.asarray(self._fsl_P, np.float64))
        return np.concatenate(L) if L else np.zeros((0, 3))

    def _fau_keep(self, E, pos, T):
        path = E['path']
        q = path[-1]
        if float(math.hypot(float(q[0] - T[0]), float(q[1] - T[1]))) > FAU_GR + 0.3 or abs(float(q[2]) - float(T[2])) > FAU_GZ + 0.3:
            return False
        i0 = int(np.argmin(np.linalg.norm(path - pos[None, :], axis=1)))
        R = np.concatenate([pos[None, :], path[i0:]])
        S = [R[0:1]]
        for a, b in zip(R[:-1], R[1:]):
            k = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / 0.15)))
            S.append(a + (b - a) * (np.arange(1, k + 1)[:, None] / k))
        S = np.concatenate(S)
        P = self._fau_mem()
        if not len(P):
            return True
        lo = S.min(0) - 1.0
        hi = S.max(0) + 1.0
        P = P[np.all((P >= lo) & (P <= hi), axis=1)]
        if not len(P):
            return True
        d = np.sqrt(((S[:, None, :] - P[None, :, :]) ** 2).sum(2)).min(0) if len(S) * len(P) < 400000 else None
        if d is None:
            return False
        return bool(float(d.min()) >= max(float(E.get('c', FAU_C)) - 0.12, FAU_C2 - 0.05))

    def _fau_pts_plan(self, pos, T, st):
        import time as _tm
        _t0 = _tm.perf_counter()
        P = self._fau_mem()
        gz = float(pos[2]) - (float(st[162]) * 20.0 if st.size > 162 else 5.0)
        path, c = _fau_plan(P, pos, T, gz)
        self._fau_pl += 1
        self._fau_ms = max(self._fau_ms, (_tm.perf_counter() - _t0) * 1000.0)
        if path is not None:
            self._fau_c = c
        return path

    def _fau_step(self, st, out):
        mo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        mi = self._fg_mem
        E = self._fau
        if mo.size < 65 or mi is None or mi.size < 65 or (int(round(float(mo[0]))) != 3) or (int(round(float(mi[0]))) != 3):
            if E is not None:
                self._fau_end('mode')
            self._fau_w = None
            return None
        pos = st[0:3].astype(np.float64)
        T = mo[8:11].astype(np.float64)
        if not self._tr_r2ok(T):
            if E is not None:
                self._fau_end('agree')
            self._fau_w = None
            return None
        dh = float(math.hypot(float(T[0] - pos[0]), float(T[1] - pos[1])))
        W = self._fau_w
        if W is None or float(math.hypot(float(T[0] - W[0][0]), float(T[1] - W[0][1]))) > 1.0:
            W = self._fau_w = [T.copy(), dh, self.tick]
        if dh < W[1] - FAU_PROG:
            W[1] = dh
            W[2] = self.tick
        if E is not None:
            if float(np.linalg.norm(T[0:2] - E['T'][0:2])) > 1.5:
                self._fau_end('tgt')
                return None
            if E['hold'] < 0 and dh <= FAU_GR + 0.15 and (abs(float(pos[2]) - float(T[2])) <= FAU_GZ + 0.15):
                E['hold'] = self.tick
            if E['hold'] >= 0:
                if self.tick - E['hold'] >= FAU_HOLD:
                    self._fau_end('done')
                    return None
                if dh <= FAU_GR + 0.6:
                    hv = T[0:2] - pos[0:2]
                    sh = min(0.25, 0.5 * dh)
                    dz = float(T[2]) - float(pos[2])
                    v = np.array([hv[0] / max(dh, 1e-06) * sh, hv[1] / max(dh, 1e-06) * sh, max(-0.3, min(0.3, 0.8 * dz))])
                    vn = float(np.linalg.norm(v))
                    out = out.copy()
                    if vn > 0.001:
                        out[0:3] = (v / vn).astype(np.float32)
                    out[3] = np.float32(vn / 3.0)
                    if dh > 0.3:
                        out[4] = np.float32(_wrap(math.atan2(float(hv[1]), float(hv[0]))) / math.pi)
                    self._fau_t += 1
                    self._fau_hd += 1
                    if '+fau' not in self.route:
                        self.route = self.route + '+fau'
                    return out
                E['hold'] = -1
                E['path'] = None
                E['best'] = dh + abs(float(pos[2]) - float(T[2]))
                E['bt'] = self.tick
            if self.tick - E['t0'] > FAU_T and E['hold'] < 0:
                self._fau_end('time')
                return None
            _g = dh + abs(float(pos[2]) - float(T[2]))
            if _g < E['best'] - 0.3:
                E['best'] = _g
                E['bt'] = self.tick
            if self.tick - E['bt'] > FAU_STALL and E['hold'] < 0:
                self._fau_end('stall')
                return None
            if E['path'] is None or self.tick - E['tp'] >= FAU_RP:
                E['tp'] = self.tick
                E['T'] = T.copy()
                if E['path'] is not None and self._fau_keep(E, pos, T):
                    self._fau_kp += 1
                    p = E['path']
                else:
                    p = self._fau_pts_plan(pos, T, st)
                    if p is not None:
                        E['c'] = self._fau_c
                if p is None:
                    E['np'] += 1
                    self._fau_np += 1
                    if E['np'] >= 3:
                        self._fau_end('nopath')
                        return None
                else:
                    E['path'] = p
            path = E['path']
            if path is None:
                return None
        else:
            if self._fau_n >= FAU_MAXE or self.tick < self._fau_cool:
                return None
            if not FAU_HMIN <= dh <= FAU_HMAX or self.tick - W[2] < FAU_N:
                return None
            gz = float(pos[2]) - (float(st[162]) * 20.0 if st.size > 162 else 5.0)
            if float(T[2]) < gz + 1.5 or abs(float(T[2]) - float(pos[2])) > FAU_DZ:
                return None
            path = self._fau_pts_plan(pos, T, st)
            if path is None:
                self._fau_np += 1
                self._fau_cool = self.tick + FAU_COOL
                self._fau_why = 'nopath0'
                return None
            E = self._fau = {'T': T.copy(), 't0': self.tick, 'tp': self.tick, 'path': path, 'np': 0, 'best': dh + abs(float(pos[2]) - float(T[2])), 'bt': self.tick, 'hold': -1, 'c': self._fau_c}
            self._fau_n += 1
            if self._fau_t1 < 0:
                self._fau_t1 = self.tick
            if len(self._fau_pts) < 4:
                self._fau_pts.append([int(self.tick)] + [round(float(x), 2) for x in pos] + [round(float(x), 2) for x in T] + [round(dh, 2)])
        d = np.linalg.norm(path - pos[None, :], axis=1)
        i0 = int(np.argmin(d))
        seg = np.linalg.norm(np.diff(path[i0:], axis=0), axis=1) if len(path) - i0 > 1 else np.zeros(0)
        acc = np.concatenate([[0.0], np.cumsum(seg)]) + float(d[i0])
        j = i0 + int(np.searchsorted(acc, FAU_LOOK))
        j = min(j, len(path) - 1)
        q = path[j]
        rem = float(acc[-1])
        rel = q - pos
        dn = float(np.linalg.norm(rel))
        if dn < 0.05:
            return None
        sp = min(FAU_V, 0.25 + 0.6 * rem)
        if math.hypot(float(rel[0]), float(rel[1])) > 0.1 and abs(_wrap(math.atan2(float(rel[1]), float(rel[0])) - float(st[5]))) > FAU_BDEG:
            sp = min(sp, FAU_BV)
        out = out.copy()
        out[0:3] = (rel / dn).astype(np.float32)
        out[3] = np.float32(sp / 3.0)
        if math.hypot(float(rel[0]), float(rel[1])) > 0.3:
            out[4] = np.float32(_wrap(math.atan2(float(rel[1]), float(rel[0]))) / math.pi)
        self._fau_t += 1
        if '+fau' not in self.route:
            self.route = self.route + '+fau'
        return out

    def _trt(self, st, out):
        mo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        if mo.size < 65 or self._tr_n >= TR_MAX:
            self._tr = None
            self._tr_g = None
            self._tr_rk = 0
            return out
        ph = int(round(float(mo[0])))
        pos = st[0:3].astype(np.float64)
        G = None
        if ph == 3:
            if self._tr_r2ok(mo[8:11]):
                G = mo[8:11].astype(np.float64)
        elif ph == 2 and TR_P2 and (float(mo[57]) > 0.5) and (self._r2 is not None) and self._r2_on and (self._r2.latch is not None):
            if float(np.hypot(float(self._r2.latch[0]) - float(mo[60]), float(self._r2.latch[1]) - float(mo[61]))) <= TR_AGREE:
                G = mo[16:46].reshape(10, 3).astype(np.float64).mean(0) + np.array([0.0, 0.0, 4.0])
        if G is None:
            self._tr = None
            self._tr_g = None
            self._tr_rk = 0
            return out
        dv = G - pos
        dG = float(np.linalg.norm(dv))
        dh = float(np.hypot(float(dv[0]), float(dv[1])))
        if ph == 3 and dh <= TR_HZ or (ph == 2 and (dG <= 3.0 or dG > TR_P2D)):
            self._tr = None
            self._tr_g = None
            self._tr_rk = 0
            return out
        if self._tr_g is None or float(np.linalg.norm(self._tr_g - G)) > 1.0:
            self._tr_g = G.copy()
            self._tr_best = dG
            self._tr_t = self.tick
            self._tr = None
            self._tr_rk = 0
        if dG < self._tr_best - TR_PROG:
            self._tr_best = dG
            self._tr_t = self.tick
        if self._tr is not None:
            raw = self._tr_raw
            rd = None if raw is None or raw.size < 4 else raw[0:3].astype(np.float64)
            rn = 0.0 if rd is None else float(np.linalg.norm(rd))
            if rn > 1e-06 and float(raw[3]) > TR_RLV and (float(rd @ dv) >= TR_RLC * rn * dG):
                self._tr_rk += 1
            else:
                self._tr_rk = 0
            if self._tr_rk >= TR_RLN:
                self._tr = None
                self._tr_rk = 0
                self._tr_best = dG
                self._tr_t = self.tick
                self._tr_rel += 1
                return out
            rel = self._tr[0] - pos
            if float(np.linalg.norm(rel)) < 0.35 or self.tick > self._tr[1]:
                self._tr = self._tr_plan(pos, G, st)
        elif self.tick - self._tr_t >= TR_N:
            self._tr = self._tr_plan(pos, G, st)
            if self._tr is not None:
                self._tr_k += 1
        if self._tr is None:
            return out
        rel = self._tr[0] - pos
        dn = float(np.linalg.norm(rel))
        if dn < 1e-06:
            return out
        out = out.copy()
        out[0:3] = (rel / dn).astype(np.float32)
        out[3] = np.float32(min(TR_V, 0.3 + 0.5 * dn) / 3.0)
        if math.hypot(float(rel[0]), float(rel[1])) > 0.3:
            out[4] = np.float32(_wrap(math.atan2(float(rel[1]), float(rel[0]))) / math.pi)
        self._tr_n += 1
        self._fsx = None
        self._fsx_n = 0
        if '+trt' not in self.route:
            self.route = self.route + '+trt'
        return out

    def _tr_r2ok(self, T):
        r = self._r2
        if r is None:
            return False
        Q = ([r.latch[0:2]] if r.latch is not None else []) + ([r.last_q[0:2]] if r.last_q is not None else []) + list(r.bad)
        return any((float(np.hypot(float(q[0]) - float(T[0]), float(q[1]) - float(T[1]))) <= TR_AGREE for q in Q))

    def _tr_plan(self, pos, G, st):
        P = self._fsl_P if self._fsl_P is not None else np.zeros((0, 3))
        Prel = P - pos[None, :]
        d = G - pos
        dG = float(np.linalg.norm(d))
        if dG < 1e-06:
            return None
        if float(_fsl_free(Prel, d / dG, TR_R)[0]) >= dG - 0.3:
            return (G.copy(), self.tick + TR_HOLD)
        gz = float(pos[2]) - (float(st[162]) * 20.0 if st.size > 162 else 5.0)
        yaw = float(st[5])
        A, E = np.meshgrid(math.atan2(float(d[1]), float(d[0])) + _TR_AZ, _TR_EL, indexing='ij')
        A = A.ravel()
        E = E.ravel()
        W = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], 1)
        s = np.minimum(_fsl_free(Prel, W, TR_R) - TR_M, TR_STEP)
        best = None
        for j in np.where(s >= 0.5)[0]:
            V = pos + W[j] * float(s[j])
            if float(V[2]) < gz + 1.2 or float(V[2]) > float(G[2]) + 2.5:
                continue
            r = G - V
            L = float(np.linalg.norm(r))
            if L > dG + 1.0:
                continue
            fv = float(_fsl_free(P - V[None, :], r / max(L, 1e-06), TR_R)[0])
            c = float(s[j]) + L + (0.0 if fv >= L - 0.3 else 3.0) + (0.5 if abs(_wrap(float(A[j]) - yaw)) > math.radians(60.0) else 0.0)
            if best is None or c < best[0]:
                best = (c, V)
        return None if best is None else (best[1], self.tick + TR_HOLD)

    def _fpr_park(self, st):
        mm = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        if mm.size < 63 or float(mm[0]) < 2.5:
            self._fpr_hv = None
            self._fpr_at = 0
            return
        hv = np.asarray(mm[8:10], np.float64)
        if self._fpr_hv is None or float(np.hypot(float(hv[0] - self._fpr_hv[0]), float(hv[1] - self._fpr_hv[1]))) > 1.0:
            self._fpr_hv = hv.copy()
            self._fpr_at = 0
            self._fpr_sup = self.tick
        if float(mm[57]) > 0.5 and float(np.hypot(float(mm[60]) - float(hv[0]), float(mm[61]) - float(hv[1]))) <= FPR_PKL:
            self._fpr_sup = self.tick
        _r2 = getattr(self, '_r2', None)
        _rl = getattr(_r2, 'latch', None) if _r2 is not None else None
        if FPR_PKR2 and _rl is not None and (float(np.hypot(float(_rl[0]) - float(hv[0]), float(_rl[1]) - float(hv[1]))) <= FPR_PKL):
            self._fpr_sup = self.tick
        if float(np.hypot(float(st[0]) - float(hv[0]), float(st[1]) - float(hv[1]))) < FPR_PKR and float(st[2]) <= float(mm[10]) + FPR_PKZ:
            self._fpr_at += 1
        if self._fpr_at >= FPR_PKN and self.tick - self._fpr_sup > FPR_PKW:
            self._p3_ban.append(hv.copy())
            mm = mm.copy()
            mm[0] = 1.0
            mm[56] = 0.0
            mm[57] = 0.0
            mm[58] = 0.0
            mm[59] = 0.0
            self.memory_tensor = mm.reshape(self._ms)
            self._p3_n = 0
            self._fpr_pk += 1
            self._fpr_hv = None
            self._fpr_at = 0
            if self._fpr_t1 < 0:
                self._fpr_t1 = self.tick
            self.route = self.route + '+fpr'
            if self._omb is not None:
                self._omb.stats['fpr_pk'] = self._fpr_pk
                self._omb.stats['fpr_t1'] = self._fpr_t1

    def _fsl_gate(self):
        return bool(self._cls is not None and self._z0 is not None and (self._z0 > FSL_Z0) and (self._stype() == 'forest'))

    def _r2nc(self, st):
        mo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        if mo.size < 65 or int(round(float(mo[0]))) != 2 or float(mo[57]) < 0.5:
            return False
        A = mo[16:46].reshape(10, 3).astype(np.float64).mean(0) + np.array([0.0, 0.0, 4.0])
        return float(np.linalg.norm(st[0:3].astype(np.float64) - A)) > R2NC_D

    def _dib(self, st, out):
        L = np.asarray(self._r2.latch, np.float64).reshape(-1)
        p = st[0:3].astype(np.float64)
        v = st[6:9].astype(np.float64)
        if self._dib_L is None or math.hypot(float(L[0] - self._dib_L[0]), float(L[1] - self._dib_L[1])) > 3.0:
            self._dib_L = L[0:2].copy()
            self._dib_w = 0
            self._dib_arm = DIB_WAIT <= 0
            self._dib_on = False
        ok = math.hypot(float(L[0] - p[0]), float(L[1] - p[1])) <= DIB_R and math.hypot(float(v[0]), float(v[1])) <= DIB_HV
        e = float(p[2] - L[2])
        if not self._dib_arm:
            self._dib_w = self._dib_w + 1 if ok and e > DIB_HI else 0
            if self._dib_w < DIB_WAIT:
                return out
            self._dib_arm = True
            self._dib_fire += 1
            if self._dib_first is None:
                self._dib_first = int(self.tick)
                try:
                    self.a3_route_log = tuple(self.a3_route_log or ()) + (('dibw', int(self.tick)),)
                except Exception:
                    pass
        if DIB_DRY:
            return out
        if not ok:
            self._dib_on = False
            return out
        if not self._dib_on:
            if e <= DIB_HI:
                return out
            self._dib_on = True
            self._dib_n += 1
        if e <= DIB_TGT:
            self._dib_on = False
            return out
        d = np.asarray(out[0:3], np.float64)
        n = float(np.linalg.norm(d))
        s = float(out[3]) * 3.0
        vc = d / n * s if n > 1e-06 else np.zeros(3)
        vz = -min(DIB_VZ, max(0.2, 1.2 * (e - DIB_TGT)))
        vc[2] = min(float(vc[2]), vz)
        hn = math.hypot(float(vc[0]), float(vc[1]))
        hmax = math.sqrt(max(0.0, DIB_VMAX * DIB_VMAX - vc[2] * vc[2]))
        if hn > hmax:
            vc[0:2] = vc[0:2] * (hmax / hn)
        m = float(np.linalg.norm(vc))
        out = out.copy()
        out[0:3] = (vc / m).astype(np.float32)
        out[3] = np.float32(m / 3.0)
        self._dib_t += 1
        if not self.route.endswith('+dib'):
            self.route = self.route + '+dib'
        return out

    def _hrz_logev(self, tag, L, p, extra=()):
        try:
            self.a3_route_log = tuple(self.a3_route_log or ()) + ((tag, int(self.tick), [round(float(L[0]), 2), round(float(L[1]), 2), round(float(L[2]), 2)], [round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2)]) + tuple(extra),)
        except Exception:
            pass

    def _hrz(self, st, out):
        L = np.asarray(self._r2.latch, np.float64).reshape(-1)
        p = st[0:3].astype(np.float64)
        v = st[6:9].astype(np.float64)
        if self._hrz_L is None or math.hypot(float(L[0] - self._hrz_L[0]), float(L[1] - self._hrz_L[1])) > 1.0:
            self._hrz_L = L[0:2].copy()
            self._hrz_w = 0
            self._hrz_arm = False
            self._hrz_act = 0
            self._hrz_hold = 0
            self._hrz_u = 0
        if any((math.hypot(float(L[0] - b[0]), float(L[1] - b[1])) <= 1.0 for b in self._hrz_ban)):
            self._hrz_w = 0
            self._hrz_arm = False
            return out
        dh = math.hypot(float(L[0] - p[0]), float(L[1] - p[1]))
        e = float(p[2] - L[2])
        if not self._hrz_arm:
            mo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
            ph3 = bool(mo.size and float(mo[0]) >= 2.5)
            r2t = getattr(self, '_r2_term', None) is not None
            ok = (ph3 or r2t) and dh <= HRZ_R and (HRZ_ELO <= e <= HRZ_EHI) and (float(np.linalg.norm(v)) <= HRZ_V) and (abs(float(v[2])) <= HRZ_VZ)
            self._hrz_u = self._hrz_u + 1 if ok else 0
            if self._hrz_u >= HRZ_WAIT and (not self._hrz_ulog):
                self._hrz_ulog = True
                self._hrz_logev('hrzu', L, p, (round(dh, 2), round(e, 2), int(r2t)))
            ok = ok and float(p[2]) < HRZ_ZMAX
            self._hrz_w = self._hrz_w + 1 if ok else 0
            if self._hrz_w < HRZ_WAIT:
                return out
            self._hrz_arm = True
            self._hrz_n += 1
            if self._hrz_first is None:
                self._hrz_first = int(self.tick)
                self._hrz_logev('hrzw', L, p, (round(dh, 2), round(e, 2), int(r2t)))
        if HRZ_DRY:
            return out
        if dh > HRZ_R + 1.0 or self._hrz_act >= HRZ_TMAX or self._hrz_hold >= HRZ_HOLD:
            self._hrz_ban.append(L[0:2].copy())
            self._hrz_arm = False
            self._hrz_w = 0
            return out
        self._hrz_act += 1
        if abs(e - HRZ_DZ) <= 0.15 and dh <= HRZ_R:
            self._hrz_hold += 1
        d = np.asarray(out[0:3], np.float64)
        n = float(np.linalg.norm(d))
        s = float(out[3]) * 3.0
        vc = d / n * s if n > 1e-06 else np.zeros(3)
        vc[2] = float(np.clip(1.2 * (HRZ_DZ - e), -0.3, HRZ_VUP))
        hn = math.hypot(float(vc[0]), float(vc[1]))
        hmax = math.sqrt(max(0.0, HRZ_VMAX * HRZ_VMAX - vc[2] * vc[2]))
        if hn > hmax:
            vc[0:2] = vc[0:2] * (hmax / hn)
        m = float(np.linalg.norm(vc))
        if m < 1e-06:
            return out
        out = out.copy()
        out[0:3] = (vc / m).astype(np.float32)
        out[3] = np.float32(m / 3.0)
        self._hrz_t += 1
        if not self.route.endswith('+hrz'):
            self.route = self.route + '+hrz'
        return out

    def _hrx_log(self, st):
        L = np.asarray(self._r2.latch, np.float64).reshape(-1)
        p = st[0:3].astype(np.float64)
        if self._hrx_L is None or math.hypot(float(L[0] - self._hrx_L[0]), float(L[1] - self._hrx_L[1])) > 1.0:
            self._hrx_L = L[0:2].copy()
            self._hrx_w = 0
            self._hrb_w = 0
        mo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        if mo.size < 11 or float(mo[0]) < 2.5:
            self._hrx_w = 0
            self._hrb_w = 0
            return
        T = mo[8:10].astype(np.float64)
        dh = math.hypot(float(L[0] - p[0]), float(L[1] - p[1]))
        tl = math.hypot(float(T[0] - L[0]), float(T[1] - L[1]))
        tp = math.hypot(float(T[0] - p[0]), float(T[1] - p[1]))
        sp = float(np.linalg.norm(st[6:9]))
        self._hrx_w = self._hrx_w + 1 if tl >= 2.3 and 2.3 <= dh <= 4.0 and (tp <= 1.5) and (sp <= 1.0) else 0
        self._hrb_w = self._hrb_w + 1 if tl < 2.3 and 2.3 <= dh <= 4.0 and (sp <= 1.0) else 0
        if self._hrx_w >= 250 and (not self._hrx_done):
            self._hrx_done = True
            self._hrz_logev('hrxw', L, p, (round(dh, 2), round(tl, 2)))
        if self._hrb_w >= 250 and (not self._hrb_done):
            self._hrb_done = True
            self._hrz_logev('hrbw', L, p, (round(dh, 2), round(tl, 2)))

    def _r2a(self, st):
        mi = self._fg_mem
        mo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        if mi is None or mi.size < 65 or mo.size < 65 or (not 1.5 <= float(mi[0]) < 2.5) or (float(mo[0]) >= 2.5):
            return
        if R2A_LK and (float(mi[57]) < 0.5 or float(mi[59]) >= 29.0):
            return
        if self._r2 is None or not self._r2_on or self._r2.last_q_t != self.tick:
            return
        pos = st[0:3].astype(np.float64)
        m10 = mo[16:46].reshape(10, 3).astype(np.float64).mean(0)
        q = self._r2.last_q
        if float(np.linalg.norm(pos - m10 - np.array([0.0, 0.0, 4.0]))) < 8.0 and float(np.hypot(q[0] - m10[0], q[1] - m10[1])) < R2A_TOL:
            mo = mo.copy()
            mo[0] = 3.0
            mo[8:11] = (m10 + np.array([0.0, 0.0, 4.5])).astype(np.float32)
            mo[64] = 0.0
            self.memory_tensor = mo.reshape(self._ms)
            self._r2a_n += 1
            self.route = self.route + '+r2a'

    def _fsl_update(self, depth, pos, rpy):
        P = _fsl_cloud(depth, pos, rpy)
        M = self._fsl_P
        T = self._fsl_T
        if M is None:
            M = np.zeros((0, 3))
            T = np.zeros(0, np.int64)
        if M.shape[0]:
            dd = M - pos[None, :]
            keep = (self.tick - T <= FSL_MEMT) & (np.einsum('ij,ij->i', dd, dd) < 36.0)
            M = M[keep]
            T = T[keep]
        if P.shape[0]:
            M = np.concatenate([P, M])
            T = np.concatenate([np.full(P.shape[0], self.tick, np.int64), T])
            k = np.round(M / FSL_VOX).astype(np.int64) + (1 << 20)
            _, iu = np.unique(k[:, 0] << 42 | k[:, 1] << 21 | k[:, 2], return_index=True)
            M = M[iu]
            T = T[iu]
        self._fsl_P = M
        self._fsl_T = T

    def _fsx_start(self, pos, ph, mo):
        L = 0.0
        tgt = None
        prv = pos
        for q in reversed(self._fsl_tr):
            L += float(np.linalg.norm(q - prv))
            prv = q
            if L >= FSX_BACK:
                tgt = q.copy()
                break
        if tgt is None and L >= 0.5:
            tgt = prv.copy()
        self._fsx_c += 1
        self._fsx_n = 0
        self._av = 0
        self._fsx_avt = self.tick + FSX_T + FSX_AVH
        if self._plan is not None and self._plan.wp is not None:
            w = self._plan.wp - pos[0:2]
            self._plan.blk = (math.atan2(float(w[1]), float(w[0])), self.tick + FSX_BLK)
            self._fsx_rp = True
        via = bool(FSV_ON and ph == 3)
        self._fsx = ('back', tgt, self.tick + FSX_T, via) if tgt is not None else self._fsv(pos, mo) if via else None

    def _fsv(self, pos, mo):
        T = mo[8:11].astype(np.float64)
        r = pos[0:2] - T[0:2]
        rr = float(np.hypot(r[0], r[1]))
        if rr < 0.5:
            return None
        u = r / rr
        rad = min(max(rr, FSV_R), 1.5 * FSV_R)
        Prel = self._fsl_P - pos[None, :]
        best = None
        for sg in (1.0, -1.0):
            a = sg * math.pi / 3.0
            w = np.array([math.cos(a) * u[0] - math.sin(a) * u[1], math.sin(a) * u[0] + math.cos(a) * u[1]])
            V = np.array([T[0] + w[0] * rad, T[1] + w[1] * rad, T[2]])
            dv = V - pos
            L = float(np.linalg.norm(dv))
            f = min(L, float(_fsl_free(Prel, dv / max(L, 1e-06), FSL_R)[0]))
            if best is None or f > best[0] + 0.3:
                best = (f, V)
        self._fsx_nv += 1
        return ('via', best[1], self.tick + FSV_T, False)

    def _fsl(self, obs, st, out):
        pos = st[0:3].astype(np.float64)
        rpy = st[3:6].astype(np.float64)
        vel = st[6:9].astype(np.float64)
        vn = float(np.linalg.norm(vel))
        self._fsl_update(obs['depth'], pos, rpy)
        if self.tick % 2 == 0:
            self._fsl_tr.append(pos.copy())
            self._fsl_tr = self._fsl_tr[-150:]
        mi = self._fg_mem
        mo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
        phi = int(round(float(mi[0]))) if mi is not None and mi.size else 0
        ph = int(round(float(mo[0])))
        if ph != 3:
            self._fsl_p3h = None
        if phi < 1 or ph < 1 or ph not in FSL_PH:
            self._fsx = None
            self._fsx_n = 0
            return out
        tag = ''
        if self._fsx is not None and self._fsx[0] == 'back':
            e = self._fsx
            rel = e[1] - pos
            dn = float(np.linalg.norm(rel))
            bk = bool(dn >= 0.3 and self.tick <= e[2])
            bc = float(_fsl_v(_fsl_free(self._fsl_P - pos[None, :], rel / dn, FSL_R))[0]) if bk and FSX_CHK else 99.0
            if bk and bc >= 0.1:
                out = out.copy()
                out[0:3] = (rel / dn).astype(np.float32)
                out[3] = np.float32(min(FSX_SPD, 0.3 + dn, bc) / 3.0)
                self._fsx_n = 0
                self.route = self.route + '+fsx'
                return out
            if bk:
                self._fsx_q += 1
                tag += '+fsq'
            self._fsx = self._fsv(pos, mo) if e[3] and ph == 3 else None
        if self._fsx is not None and self._fsx[0] == 'via':
            rel = self._fsx[1] - pos
            dh = math.hypot(float(rel[0]), float(rel[1]))
            if ph != 3 or dh < 0.8 or self.tick > self._fsx[2]:
                self._fsx = None
            else:
                dn = float(np.linalg.norm(rel))
                out = out.copy()
                out[0:3] = (rel / dn).astype(np.float32)
                out[3] = np.float32(min(1.0, 0.3 + 0.5 * dn) / 3.0)
                out[4] = np.float32(_wrap(math.atan2(float(rel[1]), float(rel[0]))) / math.pi)
                tag += '+fsv'
        d = np.asarray(out[0:3], np.float64)
        n = float(np.linalg.norm(d))
        spd = float(out[3]) * 3.0
        if n < 1e-06 or spd < 0.05:
            self._fsl_stall(pos, vn, ph, mo)
            self.route = self.route + tag
            return out
        d = d / n
        Prel = self._fsl_P - pos[None, :]
        yaw = float(rpy[2])
        la = 99.0
        if FSL_AGL and st.size > 162:
            va = math.sqrt(2.0 * FSL_A * max(0.0, float(st[162]) * 20.0 - FSL_AGLM))
            dh = math.hypot(float(d[0]), float(d[1]))
            if -float(d[2]) * spd > va:
                if dh > 1e-06:
                    _z = -va / spd
                    _s = math.sqrt(max(0.0, 1.0 - _z * _z)) / dh
                    d = np.array([float(d[0]) * _s, float(d[1]) * _s, _z])
                    out = out.copy()
                    out[0:3] = d.astype(np.float32)
                else:
                    la = va
                self._fsl_na += 1
                tag += '+fsa'
        vmax = float(_fsl_v(_fsl_free(Prel, d, FSL_R))[0])
        if vn > 0.3:
            vmax = min(vmax, max(0.3, float(_fsl_v(_fsl_free(Prel, vel / vn, FSL_R))[0])))
        lb = 99.0
        l3 = 99.0
        if FSL_BLIND:
            hv = math.hypot(float(vel[0]), float(vel[1]))
            hc = spd * math.hypot(float(d[0]), float(d[1]))
            if hv > FSL_BV and abs(_wrap(math.atan2(float(vel[1]), float(vel[0])) - yaw)) > FSL_BFOV or (hc > FSL_BV and abs(_wrap(math.atan2(float(d[1]), float(d[0])) - yaw)) > FSL_BFOV):
                lb = FSL_BV
        if ph == 3:
            rel = mo[8:11].astype(np.float64) - pos
            if math.hypot(float(rel[0]), float(rel[1])) > 1.5 and self._fsl_clut(Prel, rel):
                l3 = FSL_P3V
            if self._fsl_p3h is not None:
                l3 = min(l3, self._fsl_p3h + FSL_P3UP)
        lim = min(lb, l3, la)
        vc = min(vmax, lim)
        if spd <= vc + 1e-06:
            self._fsl_side = None
            if ph == 3 and self._fsl_p3h is not None:
                self._fsl_p3h = spd
            self._fsl_stall(pos, vn, ph, mo)
            self.route = self.route + tag
            return out
        best = None
        if vmax < 0.5 * spd:
            fwd, up, right = _axes(rpy)
            W = (np.cos(_FSL_EL) * np.cos(_FSL_AZ))[:, None] * fwd[None, :] + (np.cos(_FSL_EL) * np.sin(_FSL_AZ))[:, None] * right[None, :] + np.sin(_FSL_EL)[:, None] * up[None, :]
            cw = _fsl_v(_fsl_free(Prel, W, FSL_R))
            ang = np.arccos(np.clip(W @ d, -1.0, 1.0))
            az0 = math.atan2(float(d @ right), float(d @ fwd))
            sd = np.sign(_FSL_AZ - az0)
            pen = np.where((self._fsl_side is None) | (sd == (self._fsl_side or 0.0)), 0.0, math.radians(20.0))
            ok = (cw >= min(spd, 1.0)) & (ang <= math.radians(75.0))
            if ok.any():
                j = int(np.argmin(np.where(ok, ang + pen, 1000000000.0)))
                best = (W[j], float(cw[j]))
                self._fsl_side = float(sd[j]) or 1.0
        out = out.copy()
        if best is not None:
            sp = min(spd, best[1], max(vmax, 0.6), lim)
            out[0:3] = best[0].astype(np.float32)
            self._fsl_nd += 1
            tag += '+fsd'
        else:
            sp = max(0.0, vc)
        out[3] = np.float32(sp / 3.0)
        if ph == 3:
            self._fsl_p3h = sp
        self._fsl_last = self.tick
        self._fsl_n += 1
        if vmax < spd:
            tag += '+fsl'
        if lb < spd and lb <= min(vmax, l3, la):
            self._fsl_nb += 1
            tag += '+fsb'
        if l3 < spd and l3 < min(vmax, lb, la):
            self._fsl_n3 += 1
            tag += '+fs3'
        self._fsl_stall(pos, vn, ph, mo)
        self.route = self.route + tag
        return out

    def _fsl_clut(self, Prel, rel):
        L = float(np.linalg.norm(rel))
        if L < 1e-06 or Prel.shape[0] == 0:
            return False
        u = rel / L
        a = Prel @ u
        p2 = np.einsum('ij,ij->i', Prel, Prel) - a * a
        return int(((a > 0.0) & (a < L) & (p2 < FSL_P3C * FSL_P3C)).sum()) >= 2

    def _fsl_stall(self, pos, vn, ph, mo):
        if not FSX_ON:
            return
        if FSX_R2 and getattr(self, '_r2_term', None) is not None:
            self._fsx_n = 0
            return
        if ph == 3:
            rel = mo[8:11].astype(np.float64) - pos
            if math.hypot(float(rel[0]), float(rel[1])) <= 2.0:
                self._fsx_n = 0
                return
        self._fsx_n = self._fsx_n + 1 if vn < FSX_V and self.tick - self._fsl_last <= 10 else 0
        if self._fsx_n >= FSX_N and self._fsx_c < FSX_MAX and (self._fsx is None or self._fsx[0] == 'via'):
            self._fsx_start(pos, ph, mo)
A3_Z0 = 13.0
_C032Controller = DroneFlightController
LA3_ON = os.environ.get('KT_LA3', '1') == '1'
LA3_Z = float(os.environ.get('KT_LA3_Z', '6.6'))
LA3_T = float(os.environ.get('KT_LA3_T', '12'))
LA3_TYPES = ('village', 'forest', 'warehouse')
LA3_Z0 = float(os.environ.get('KT_LA3_Z0', '10.3'))

def _a3_load(parent):
    import importlib.util as _ilu
    _p = _H / 'a3_stack.py'
    if not _p.exists():
        return None
    _spec = _ilu.spec_from_file_location('mtn_a3_stack_' + str(abs(hash(str(_p)))), str(_p))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.A3Stack(parent)

class DroneFlightController(_C032Controller):

    def __init__(self, *, model_path=None, providers=None):
        self._a3 = None
        self._a3_route = None
        self.a3_route_log = None
        self._la3_pad = None
        self._la3_z0 = None
        self._la3_rgb = 0
        self._la3_off = False
        self._la3_t = None
        _C032Controller.__init__(self, model_path=model_path, providers=providers)
        try:
            self._a3 = _a3_load(self)
        except Exception:
            self._a3 = None

    def reset(self):
        _C032Controller.reset(self)
        self._a3_route = None
        self._la3_pad = None
        self._la3_z0 = None
        self._la3_rgb = 0
        self._la3_off = False
        self._la3_t = None
        if getattr(self, '_a3', None) is not None:
            try:
                self._a3.reset()
            except Exception:
                self._a3 = None

    def act(self, observation):
        if self._a3_route is None:
            try:
                _z0 = float(np.asarray(observation['state'], np.float32).reshape(-1)[2])
            except Exception:
                _z0 = float('-inf')
            self._a3_route = 'a3' if _z0 >= (LA3_Z0 if LA3_Z0 > 0.0 else A3_Z0) else 'c032'
            self.a3_route_log = (self._a3_route, _z0, self._a3 is not None)
            if LA3_ON:
                try:
                    self._la3_pad = np.asarray(observation['state'], np.float64).reshape(-1)[0:3].copy()
                    self._la3_z0 = _z0
                except Exception:
                    self._la3_off = True
        if self._a3_route == 'a3' and self._a3 is not None:
            return self._a3.act(observation)
        if LA3_ON and (not self._la3_off) and (self._a3_route == 'c032') and (self._a3 is not None) and self._la3_gate():
            if self._la3_go(observation):
                return self._a3.act(observation)
        a = _C032Controller.act(self, observation)
        if LA3_ON:
            try:
                if float(np.asarray(a, np.float32).reshape(-1)[5]) > 0.5:
                    self._la3_rgb += 1
            except Exception:
                pass
        return a

    def _la3_gate(self):
        try:
            c = getattr(self, '_cls', None)
            if c is None or self._la3_pad is None or self._la3_z0 is None:
                return False
            if float(getattr(self, 'tick', 0)) * SIM_DT > LA3_T:
                self._la3_off = True
                return False
            if c.is_mountain:
                return True
            if LA3_Z > 0.0 and self._la3_z0 >= LA3_Z and ('open' not in c.latched) and any((x in c.latched for x in LA3_TYPES)):
                return True
            return False
        except Exception:
            self._la3_off = True
            return False

    def _la3_go(self, observation):
        try:
            A = self._a3
            st = np.asarray(observation['state'], np.float64).reshape(-1)
            B = A.brain
            if B.B is not None or A.k != 0:
                raise RuntimeError('a3 stack already used this episode')
            k0 = int(getattr(self, 'tick', 0))
            B._init(st)
            pad = np.asarray(self._la3_pad, np.float64).copy()
            B.pad = pad
            B.B = A.C.Belief(pad[0:2], B.clue, B.par)
            B._update_val(force=True)
            B.b0 = B.B.b.sum(0).copy()
            B.reg0 = B.B.reg.copy()
            B.dsc = float(np.hypot(*B.clue - pad[0:2]))
            B.frames = int(min(self._la3_rgb, 40))
            B.k = k0
            self._a3_route = 'a3'
            self._la3_t = round(k0 * SIM_DT, 2)
            self.a3_route_log = ('a3', self._la3_z0, True, 'la3', self._la3_t)
            self.route = 'king+la3'
            return True
        except Exception:
            self._la3_off = True
            self._a3_route = 'c032'
            try:
                self._a3.reset()
            except Exception:
                self._a3 = None
            return False
R2_SPLIT = {'village': 'victim_rgb_village.onnx'}
_FV_FILES = set(R2_SPLIT.values())
_FV_SESS = {}
_FV_FAR = []
_fv_orig_load = _RgbPrimary._load
_fv_orig_loc = _RgbPrimary.localise
_fv_orig_pobs = _Planner.observe

def _fv_session(m):
    if m not in _FV_SESS:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        s = ort.InferenceSession(str(_H / m), so, providers=['CPUExecutionProvider'])
        s.run(None, {s.get_inputs()[0].name: np.zeros((256, 256, 3), np.float32)})
        _FV_SESS[m] = s
    return _FV_SESS[m]

def _fv_load(self):
    m = self._pick()
    if m in _FV_FILES and (_H / m).exists():
        self._tried = True
        try:
            self.sess = _fv_session(m)
            self.iname = self.sess.get_inputs()[0].name
            self._loaded_for = m
            self.model_used = m
        except Exception:
            self.sess = None
        return
    return _fv_orig_load(self)

def _fv_localise(self, u, v, depth, pos, rpy):
    if getattr(self, 'model_used', None) not in _FV_FILES:
        return _fv_orig_loc(self, u, v, depth, pos, rpy)
    d = np.asarray(depth, np.float32).reshape(256, 256)
    px = min(255, max(0, int(round((u + 1.0) * 0.5 * 256 - 0.5))))
    py = min(255, max(0, int(round((1.0 - v) * 0.5 * 256 - 0.5))))
    cz = float(d[max(0, py - 1):py + 2, max(0, px - 1):px + 2].min()) * (_DMAX - _DMIN) + _DMIN
    if cz > RP_IMG_MAX:
        return None
    far = cz > self.rng_max
    q = _mworld(u, v, cz, pos, rpy)
    if not np.all(np.isfinite(q)):
        return None
    if self.clue0 is not None and float(np.linalg.norm(q[0:2] - self.clue0)) > RP_CLUE_R:
        return None
    if not -25.0 <= float(q[2] - pos[2]) <= 1.0:
        return None
    if self.box is not None and max(abs(float(q[0])), abs(float(q[1]))) > self.box:
        return None
    if getattr(self, 'fbox', None) is not None and max(abs(float(q[0])), abs(float(q[1]))) > self.fbox:
        self.n_fbx = getattr(self, 'n_fbx', 0) + 1
        return None
    for b in self.bad:
        if float(np.linalg.norm(q[0:2] - b)) < (BAN_R if _u19() else 6.0):
            return None
    if far:
        _FV_FAR.append(np.asarray(q, np.float64).copy())
        self.n_fvfar = getattr(self, 'n_fvfar', 0) + 1
        return None
    return q

def _fv_pobs(self, *a, **k):
    r = _fv_orig_pobs(self, *a, **k)
    while _FV_FAR:
        self.hint(_FV_FAR.pop(0), PLAN_HINT)
    return r
_RgbPrimary._load = _fv_load
_RgbPrimary.localise = _fv_localise
_Planner.observe = _fv_pobs
_FVBase = DroneFlightController

class DroneFlightController(_FVBase):

    def __init__(self, *, model_path=None, providers=None):
        for _m in _FV_FILES:
            try:
                _fv_session(_m)
            except Exception:
                pass
        _FVBase.__init__(self, model_path=model_path, providers=providers)

    def reset(self):
        del _FV_FAR[:]
        _FVBase.reset(self)
_fv_orig_set_type = _RgbPrimary.set_type

def _fv_set_type(self, mtype):
    _fv_orig_set_type(self, mtype)
    if self.sess is not None and self._loaded_for is not None:
        m = self._pick()
        if m != self._loaded_for and (m in _FV_FILES or self._loaded_for in _FV_FILES):
            self._load()
_RgbPrimary.set_type = _fv_set_type
SFL_P = 0.95
SFL_R = 12.0
_sfl_orig_observe = _RgbPrimary.observe

def _sfl_observe(self, tick, obs, pos, rpy):
    if self.mtype != 'village' or self.latch is not None:
        self._sfl = None
        return _sfl_orig_observe(self, tick, obs, pos, rpy)
    cap = {}
    prev = self.__dict__.get('score', None)
    base = self.score

    def _sc(rgb, _b=base):
        r = _b(rgb)
        cap['r'] = r
        return r
    self.score = _sc
    n0 = self.n_fix
    try:
        _sfl_orig_observe(self, tick, obs, pos, rpy)
    finally:
        if prev is None:
            self.__dict__.pop('score', None)
        else:
            self.score = prev
    r = cap.get('r')
    if r is None:
        return
    try:
        if self.n_fix > n0:
            if self.latch is None and float(r[0]) >= SFL_P and (self.last_q is not None) and (float(np.hypot(float(self.last_q[0]) - float(pos[0]), float(self.last_q[1]) - float(pos[1]))) <= SFL_R):
                self._sfl = (tick, np.asarray(self.last_q, np.float64).copy())
            else:
                self._sfl = None
            return
        s = getattr(self, '_sfl', None)
        self._sfl = None
        if s is not None and self.latch is None and (tick - s[0] <= RP_CHASE_TICKS + RP_CHASE_MIN):
            self.latch = s[1].copy()
            self.near_since = None
            self.n_latch += 1
            self.latch_t = tick
            self.n_sfl = getattr(self, 'n_sfl', 0) + 1
    except Exception:
        self._sfl = None
_RgbPrimary.observe = _sfl_observe
C1_T0 = 600
C1_VIDX = 3
_c1_orig_r2_type = _C032Controller._r2_type

def _c1_r2_type(self):
    r = _c1_orig_r2_type(self)
    if r is None:
        try:
            c = self._cls
            if c is not None and (not c.is_mountain) and (c.n >= 60) and (int(self.tick) >= C1_T0) and (int(np.argmax(np.asarray(c.ps, np.float64))) == C1_VIDX):
                self._vpr_n = getattr(self, '_vpr_n', 0) + 1
                return 'village'
        except Exception:
            return r
    return r
_c2_orig_stype = _C032Controller._stype
_c2_orig_r2_type = _C032Controller._r2_type

def _c2_fix(self, r):
    if r == 'warehouse' and bool(getattr(self, '_vp_all', False)):
        self._vrl_n = getattr(self, '_vrl_n', 0) + 1
        return 'village'
    return r

def _c2_stype(self):
    return _c2_fix(self, _c2_orig_stype(self))

def _c2_r2_type(self):
    return _c2_fix(self, _c2_orig_r2_type(self))
_C032Controller._stype = _c2_stype
_C032Controller._r2_type = _c2_r2_type
FV_FOREST = 'victim_rgb_forest.onnx'
R2_SPLIT['forest'] = FV_FOREST
_FV_FILES.add(FV_FOREST)
FV_Z0MIN = 0.3
FZ_ON = True
FZ_LAT = 1.75
_FVZ_Z0 = [None]
_fvz_orig_pick = _RgbPrimary._pick
_fvz_orig_observe = _RgbPrimary.observe
_fvz_orig_reset = _RgbPrimary.reset
_fvz_orig_set_type = _RgbPrimary.set_type

def _fvz_pick(self):
    m = _fvz_orig_pick(self)
    if m == FV_FOREST and (not (_FVZ_Z0[0] is not None and _FVZ_Z0[0] > FV_Z0MIN)):
        return self.model
    return m

def _fvz_observe(self, tick, obs, pos, rpy):
    if not (FZ_ON and getattr(self, 'model_used', None) == FV_FOREST and (getattr(self, '_loaded_for', None) == FV_FOREST)):
        return _fvz_orig_observe(self, tick, obs, pos, rpy)
    if self.sess is None and (not self._tried):
        self._load()
    rgb = obs.get('rgb') if hasattr(obs, 'get') else None
    if rgb is None:
        return
    sc = self.score(rgb)
    if sc is None:
        return
    p, u, v, h = sc
    if p < (RP_T if self.thr is None else self.thr) or h > self.hmax:
        return
    q = self.localise(u, v, obs['depth'], pos, rpy)
    if q is None:
        return
    self.hit_t = tick
    self.n_fix += 1
    self.last_q = np.asarray(q, np.float64).copy()
    self.last_q_t = tick
    if self.latch is not None:
        if float(np.linalg.norm(q[0:2] - self.latch[0:2])) <= 6.0:
            self.latch = (1.0 - RP_EMA) * self.latch + RP_EMA * q
        return
    self.hits = [(t, z) for t, z in self.hits if tick - t <= 400] + [(tick, q)]
    if len(self.hits) >= RP_LATCH_N:
        pts = np.asarray([z for _, z in self.hits[-RP_LATCH_N:]])
        c = pts.mean(0)
        if float(np.max(np.linalg.norm(pts[:, 0:2] - c[0:2], axis=1))) <= RP_LATCH_TOL:
            if float(c[2]) > FZ_LAT:
                self.hits = [(t, z) for t, z in self.hits if float(z[2]) <= FZ_LAT]
                self.n_fz = getattr(self, 'n_fz', 0) + 1
                return
            self.latch = c
            self.near_since = None
            self.n_latch += 1
            self.latch_t = tick

def _fvz_reset(self):
    _fvz_orig_reset(self)
    self.n_fz = 0

def _fvz_set_type(self, mtype):
    _fvz_orig_set_type(self, mtype)
    if self.sess is not None and self._loaded_for is not None:
        m = self._pick()
        if m != self._loaded_for and FV_FOREST in (m, self._loaded_for):
            self._load()
            self.n_fvzre = getattr(self, 'n_fvzre', 0) + 1
_RgbPrimary._pick = _fvz_pick
_RgbPrimary.observe = _fvz_observe
_RgbPrimary.reset = _fvz_reset
_RgbPrimary.set_type = _fvz_set_type
_FVZBase = DroneFlightController

class DroneFlightController(_FVZBase):

    def __init__(self, *, model_path=None, providers=None):
        self._fvz_z0 = None
        _FVZ_Z0[0] = None
        _FVZBase.__init__(self, model_path=model_path, providers=providers)

    def reset(self):
        self._fvz_z0 = None
        _FVZ_Z0[0] = None
        _FVZBase.reset(self)

    def act(self, observation):
        if getattr(self, '_fvz_z0', None) is None:
            try:
                self._fvz_z0 = float(np.asarray(observation['state'], np.float32).reshape(-1)[2])
            except Exception:
                self._fvz_z0 = None
        _FVZ_Z0[0] = self._fvz_z0
        return _FVZBase.act(self, observation)
import collections as _tmx_col
TMX_DRY = False
TMX_Q = 1.0
TMX_RN = 2.5
TMX_RF = 8.0
TMX_CL = 1.0
TMX_W = 100
TMX_EXT = 500
_tmx_orig_tick_end = _RgbPrimary.tick_end
_tmx_orig_reset = _RgbPrimary.reset

def _tmx_reset(self):
    _tmx_orig_reset(self)
    self.n_tmax = 0
    self.n_tmx = 0
    self.n_tmx_err = 0
    self.tmx_log = []
    self._tmx_buf = _tmx_col.deque(maxlen=TMX_W + 1)
    self._tmx_ext = None

def _tmx_decide(self, tick, pos):
    buf = self.__dict__.get('_tmx_buf')
    if buf is None:
        buf = self._tmx_buf = _tmx_col.deque(maxlen=TMX_W + 1)
    buf.append((int(tick), float(pos[0]), float(pos[1])))
    L = self.latch
    lt = self.latch_t
    if L is None or lt is None or (not tick - lt > RP_TERM_MAX):
        return None
    ext = self.__dict__.get('_tmx_ext')
    if ext is not None and ext[0] == lt:
        return lt + ext[1]
    if not (getattr(self, 'model_used', None) == FV_FOREST and getattr(self, '_loaded_for', None) == FV_FOREST):
        return None
    self.n_tmax = getattr(self, 'n_tmax', 0) + 1
    dl = float(np.hypot(float(pos[0]) - float(L[0]), float(pos[1]) - float(L[1])))
    old = next((b for b in buf if b[0] >= tick - TMX_W), None)
    dl0 = float(np.hypot(old[1] - float(L[0]), old[2] - float(L[1]))) if old is not None else dl
    q = self.last_q
    dq = float(np.hypot(float(q[0]) - float(L[0]), float(q[1]) - float(L[1]))) if q is not None else 99.0
    fire = bool(dq <= TMX_Q and (dl <= TMX_RN or (dl <= TMX_RF and dl0 - dl >= TMX_CL)))
    if not isinstance(self.__dict__.get('tmx_log'), list):
        self.tmx_log = []
    self.tmx_log.append([int(tick), round(dl, 2), round(dl0 - dl, 2), round(dq, 2), int(fire)] + [round(float(x), 2) for x in L[:3]])
    if fire and (not TMX_DRY):
        self.n_tmx = getattr(self, 'n_tmx', 0) + 1
        self._tmx_ext = (lt, TMX_EXT)
        return lt + TMX_EXT
    return None

def _tmx_tick_end(self, tick, pos):
    lt = self.latch_t
    try:
        eff = _tmx_decide(self, tick, pos)
    except Exception:
        self.n_tmx_err = getattr(self, 'n_tmx_err', 0) + 1
        eff = None
    if eff is None:
        return _tmx_orig_tick_end(self, tick, pos)
    self.latch_t = eff
    try:
        return _tmx_orig_tick_end(self, tick, pos)
    finally:
        self.latch_t = lt
_RgbPrimary.reset = _tmx_reset
_RgbPrimary.tick_end = _tmx_tick_end
OV_DRY = False
OV_MODE = 'handoff'
OV_NL = 15
OV_NH = 4
OV_F = 0.0
OV_MAX = 2
OV_KR = 3.0
OV_R = 1.5
OV_P = 0.9
OV_OOF = 1.0
_ov_orig_vrv = _C032Controller._vrv

def _ov_state(self):
    self._ov_key = None
    self._ov_nl = 0
    self._ov_nh = 0
    self._ov_spots = []
    self._ov_n = 0
    self._ov_nd = 0
    self._ov_err = 0
    self._ov_log = []

def _ov_review(self, obs):
    if not hasattr(self, '_ov_spots'):
        _ov_state(self)
    c = self._cls
    if c is None or c.is_mountain or self._stype() != 'village':
        self._ov_key = None
        self._ov_nl = 0
        self._ov_nh = 0
        return
    mi = self._fg_mem
    mo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
    if mi is None or mi.size < 64 or mo.size < 64:
        return
    lk_in = bool(1.5 <= float(mi[0]) < 2.5 and float(mi[57]) > 0.5)
    if not lk_in:
        if float(mo[0]) >= 2.5 or float(mi[0]) >= 2.5:
            self._ov_key = None
            self._ov_nl = 0
            self._ov_nh = 0
        return
    key = np.asarray(mi[60:62], np.float64)
    if self._ov_key is None or float(np.linalg.norm(key - self._ov_key)) > OV_KR:
        self._ov_key = key.copy()
        self._ov_nl = 0
        self._ov_nh = 0
    self._ov_nl += 1
    if float(mo[55]) >= OV_P and float(np.linalg.norm(np.asarray(mo[43:45], np.float64) - key)) <= OV_R:
        self._ov_nh += 1
    rgb = obs.get('rgb') if hasattr(obs, 'get') else None
    had = rgb is not None and float(np.mean(np.abs(np.asarray(rgb, np.float32)))) > 0.005
    if not (had and float(mo[0]) < 1.5 and (float(mo[57]) < 0.5)):
        return
    r2 = self._r2
    if r2 is not None and r2.latch is not None:
        return
    self._ov_nd += 1
    sp = None
    for _s in self._ov_spots:
        if float(np.linalg.norm(_s[0] - key)) <= OV_KR:
            sp = _s
            break
    ok = bool(self._ov_nl >= OV_NL and self._ov_nh >= OV_NH and (self._ov_nh >= OV_F * self._ov_nl) and (sp is None or sp[1] < OV_MAX))
    _st = np.asarray(obs['state'], np.float32).reshape(-1)
    _fw, _up, _rt = _axes(_st[3:6].astype(np.float64))
    _cam = _st[0:3].astype(np.float64) + _fw * _CAMF + _up * _CAMU
    _d = np.asarray(mi[60:63], np.float64) - _cam
    _z = float(_d @ _fw)
    _u = float(_d @ _rt) / _z if _z > 0.5 else 9.0
    _v = float(_d @ _up) / _z if _z > 0.5 else 9.0
    oof = bool(_z <= 0.5 or abs(_u) > OV_OOF or abs(_v) > OV_OOF)
    if len(self._ov_log) < 50:
        self._ov_log.append([int(self.tick), int(ok), int(oof), int(self._ov_nl), int(self._ov_nh), round(_u, 2), round(_v, 2)] + [round(float(x), 2) for x in mi[60:63]] + [round(float(x), 2) for x in _st[0:3]])
    if not (ok and oof):
        return
    if OV_MODE == 'handoff':
        tq = np.asarray(mi[60:63], np.float64)
        if r2 is None or not self._r2_on or any((float(np.linalg.norm(tq[0:2] - np.asarray(b, np.float64)[0:2])) < BAN_R for b in r2.bad)) or (sp is not None and sp[1] >= 1):
            return
    self._ov_n += 1
    if OV_DRY:
        return
    if OV_MODE == 'handoff':
        if sp is None:
            self._ov_spots.append([key.copy(), 1])
        else:
            sp[1] += 1
        r2.latch = tq.copy()
        r2.near_since = None
        r2.latch_t = int(self.tick)
        r2.n_latch += 1
        r2.hits = []
        r2.n_ovh = getattr(r2, 'n_ovh', 0) + 1
        if not self.route.endswith('+ovh'):
            self.route = self.route + '+ovh'
        return
    if sp is None:
        self._ov_spots.append([key.copy(), 1])
    else:
        sp[1] += 1
    mm = mo.copy()
    mm[0] = mi[0]
    mm[56:63] = mi[56:63]
    mm[59] = 0.0
    self._vrv_m63 = float(mo[63]) if self._vrv_m63 is None else self._vrv_m63
    mm[63] = np.float32(max(float(mo[63]), 40.0))
    self.memory_tensor = mm.reshape(self._ms)
    self._vrv_sup = VRV_SUP
    if not self.route.endswith('+ov'):
        self.route = self.route + '+ov'

def _ov_vrv(self, obs):
    try:
        _ov_orig_vrv(self, obs)
    finally:
        try:
            _ov_review(self, obs)
        except Exception:
            self._ov_err = getattr(self, '_ov_err', 0) + 1
_C032Controller._vrv = _ov_vrv
_OVBase = DroneFlightController

class DroneFlightController(_OVBase):

    def __init__(self, *, model_path=None, providers=None):
        _ov_state(self)
        _OVBase.__init__(self, model_path=model_path, providers=providers)

    def reset(self):
        _OVBase.reset(self)
        _ov_state(self)
EZ_ON = True
_ez_orig_observe = _RgbPrimary.observe

def _ez_observe(self, tick, obs, pos, rpy):
    if not (EZ_ON and self.latch is not None and (getattr(self, 'model_used', None) == FV_FOREST) and (getattr(self, '_loaded_for', None) == FV_FOREST)):
        return _ez_orig_observe(self, tick, obs, pos, rpy)
    L0 = self.latch.copy()
    r = _ez_orig_observe(self, tick, obs, pos, rpy)
    q = self.last_q
    if self.latch is not None and self.last_q_t == tick and (q is not None) and (float(q[2]) > FZ_LAT) and (not np.array_equal(self.latch, L0)):
        self.latch = L0
        self.n_ez = getattr(self, 'n_ez', 0) + 1
    return r
_RgbPrimary.observe = _ez_observe
VSIG_ON = True
_VSIG = [False]
_vsig_prev_pick = _RgbPrimary._pick

def _vsig_pick(self):
    m = _vsig_prev_pick(self)
    if VSIG_ON and m == R2_SPLIT.get('village') and (not _VSIG[0]):
        self.x43_vsig_pick = getattr(self, 'x43_vsig_pick', 0) + 1
        return self.model
    return m

def _vsig_sfl_gate(self, tick, obs, pos, rpy):
    if VSIG_ON and (not _VSIG[0]):
        return _sfl_orig_observe(self, tick, obs, pos, rpy)
    return _sfl_observe(self, tick, obs, pos, rpy)
_vsig_prev_ov_review = _ov_review

def _vsig_ov_review(self, obs):
    if VSIG_ON and (not bool(getattr(self, '_vp_all', False))):
        self._ov_key = None
        self._ov_nl = 0
        self._ov_nh = 0
        return
    return _vsig_prev_ov_review(self, obs)
_RgbPrimary._pick = _vsig_pick
_fvz_orig_observe = _vsig_sfl_gate
_ov_review = _vsig_ov_review
_VSIGBase = DroneFlightController

class DroneFlightController(_VSIGBase):

    def __init__(self, *, model_path=None, providers=None):
        _VSIG[0] = False
        _VSIGBase.__init__(self, model_path=model_path, providers=providers)

    def reset(self):
        _VSIG[0] = False
        _VSIGBase.reset(self)

    def act(self, observation):
        _VSIG[0] = bool(getattr(self, '_vp_all', False))
        try:
            return _VSIGBase.act(self, observation)
        finally:
            _VSIG[0] = bool(getattr(self, '_vp_all', False))
RA_ON = os.environ.get('KT_X_RA', '1') == '1'
RA_Z1, RA_AX1 = (10.3, 42.5)
RA_AX2 = 75.5
RA_AX3, RA_Z3 = (60.5, 5.3)
_RABase = DroneFlightController

def _ra_rule(st):
    x0, y0, z0 = (float(st[0]), float(st[1]), float(st[2]))
    ax = max(abs(x0), abs(y0))
    if not (z0 == z0 and ax == ax) or z0 >= A3_Z0:
        return 0
    if z0 >= RA_Z1 and ax > RA_AX1:
        return 1
    if ax > RA_AX2:
        return 2
    if ax > RA_AX3 and z0 >= RA_Z3:
        return 3
    return 0

class DroneFlightController(_RABase):

    def __init__(self, *, model_path=None, providers=None):
        self._x43_ra = 0
        _RABase.__init__(self, model_path=model_path, providers=providers)

    def reset(self):
        _RABase.reset(self)
        self._x43_ra = 0

    def act(self, observation):
        if RA_ON and getattr(self, '_a3_route', None) is None and (getattr(self, '_a3', None) is not None):
            try:
                _st = np.asarray(observation['state'], np.float32).reshape(-1)
                _r = _ra_rule(_st)
                if _r:
                    self._a3_route = 'a3'
                    self.a3_route_log = ('a3', float(_st[2]), True, 'A%d' % _r)
                    self._x43_ra = int(_r)
            except Exception:
                pass
        return _RABase.act(self, observation)
import collections as _x43_col
X43_MODE = 'on'
LAR_ON = True
FD_ON = True
LAR_AGE = 150
LAR_DMAX = 10.0
LAR_W = 150
LAR_PROG = 0.4
LAR_HV = 1.0
LAR_STAT = 1.5
LAR_AWAY = 0.3
LAR_C = 0.3
LAR_INF = 0.55
LAR_GR = 1.2
LAR_ZG0 = 3.4
LAR_ZG1 = 4.0
LAR_ZH = 3.7
LAR_ZLO = 2.3
LAR_V = 0.8
LAR_HOLD = 150
LAR_TMAX = 700
LAR_RP = 100
LAR_NOPROG = 250
LAR_DEV = 1.0
LAR_LA = 3
FD_Z = 6.5
FD_NEED = 0.35
FD_R = 6.0
FD_VZ = 1.0
FD_VMAX = 1.5
FD_TGT = 3.2
_x43_orig_hrz = _C032Controller._hrz
_X43_R = int(math.ceil((LAR_INF + 0.5 * LAR_C * math.sqrt(3.0)) / LAR_C))
_X43_OFF = np.array([(i, j, k) for i in range(-_X43_R, _X43_R + 1) for j in range(-_X43_R, _X43_R + 1) for k in range(-_X43_R, _X43_R + 1) if math.sqrt(i * i + j * j + k * k) * LAR_C <= LAR_INF + 0.5 * LAR_C * math.sqrt(3.0)], np.int64)

def _x43_state(self):
    self._x43_lar = []
    self._x43_fd = []
    self._x43_lan = 0
    self._x43_fdn = 0
    self._x43_err = 0
    self._x43_plan = 0
    self._x43_nop = 0
    self._x43_act = 0
    self._x43_hold = 0
    self._x43_rel = []
    self._l43_L = None
    self._l43_h = _x43_col.deque(maxlen=400)
    self._l43_done = False
    self._l43_act = None
    self._l43_fd = False
    self._l43_fdlog = False
    self._l43_v = {}

def _x43_dil(f):
    g = f.copy()
    g[1:, :, :] |= f[:-1, :, :]
    g[:-1, :, :] |= f[1:, :, :]
    h = g.copy()
    h[:, 1:, :] |= g[:, :-1, :]
    h[:, :-1, :] |= g[:, 1:, :]
    q = h.copy()
    q[:, :, 1:] |= h[:, :, :-1]
    q[:, :, :-1] |= h[:, :, 1:]
    return q

def _x43_plan(self, p, L, agl):
    om = self._omb
    P = om.pts if om is not None and getattr(om, 'pts', None) is not None else np.zeros((0, 3))
    C = LAR_C
    lz = float(L[2])
    zlo = lz + LAR_ZLO
    zhi = max(float(p[2]), lz + LAR_ZG1) + 1.2
    if zhi - zlo < 2 * C:
        return None
    o = np.array([min(p[0], L[0]) - 3.0, min(p[1], L[1]) - 3.0, zlo], np.float64)
    top = np.array([max(p[0], L[0]) + 3.0, max(p[1], L[1]) + 3.0, zhi], np.float64)
    n = np.ceil((top - o) / C).astype(np.int64) + 1
    occ = np.zeros(tuple((int(x) for x in n)), bool)
    if P.shape[0]:
        m = np.all((P >= (o - LAR_INF - C)[None, :]) & (P <= (top + LAR_INF + C)[None, :]), axis=1)
        Q = P[m]
        if Q.shape[0]:
            ci = np.floor((Q - o[None, :]) / C + 0.5).astype(np.int64)
            I = ci[:, None, :] + _X43_OFF[None, :, :]
            cen = o[None, None, :] + I * C
            ok = (np.linalg.norm(cen - Q[:, None, :], axis=2) <= LAR_INF) & np.all((I >= 0) & (I < n[None, None, :]), axis=2)
            I = I[ok]
            occ[I[:, 0], I[:, 1], I[:, 2]] = True
    free = ~occ
    gx = o[0] + np.arange(n[0]) * C
    gy = o[1] + np.arange(n[1]) * C
    gzz = o[2] + np.arange(n[2]) * C
    hd = np.hypot(gx[:, None] - float(L[0]), gy[None, :] - float(L[1])) <= LAR_GR
    zm = (gzz >= lz + LAR_ZG0) & (gzz <= lz + LAR_ZG1)
    goal = hd[:, :, None] & zm[None, None, :] & free
    if not goal.any():
        return None
    D = np.full(occ.shape, -1, np.int32)
    D[goal] = 0
    fr = goal
    it = 0
    s = np.clip(np.floor((p - o) / C + 0.5).astype(np.int64), 0, n - 1)
    while it < 250:
        it += 1
        nw = _x43_dil(fr) & free & (D < 0)
        if not nw.any():
            break
        D[nw] = it
        fr = nw
        if D[s[0], s[1], s[2]] >= 0:
            break
    c = tuple((int(x) for x in s))
    if D[c] < 0:
        best = None
        for off in _X43_OFF:
            q = (c[0] + off[0], c[1] + off[1], c[2] + off[2])
            if all((0 <= q[i] < n[i] for i in range(3))) and D[q] >= 0:
                key = (abs(off[0]) + abs(off[1]) + abs(off[2]), D[q])
                if best is None or key < best[0]:
                    best = (key, q)
        if best is None:
            return None
        c = best[1]
    path = [c]
    while D[c] > 0 and len(path) < 300:
        best = None
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    q = (c[0] + i, c[1] + j, c[2] + k)
                    if (i or j or k) and all((0 <= q[t] < n[t] for t in range(3))) and (0 <= D[q] < D[c]):
                        key = (D[q], i * i + j * j + k * k)
                        if best is None or key < best[0]:
                            best = (key, q)
        if best is None:
            break
        c = best[1]
        path.append(c)
    if D[c] != 0:
        return None
    return o[None, :] + np.asarray(path, np.float64) * C

def _x43_blocked(self, path):
    om = self._omb
    P = om.pts if om is not None and getattr(om, 'pts', None) is not None else None
    if P is None or not len(P) or path is None or (not len(path)):
        return False
    lo = path.min(0) - LAR_INF
    hi = path.max(0) + LAR_INF
    Q = P[np.all((P >= lo[None, :]) & (P <= hi[None, :]), axis=1)]
    if not len(Q):
        return False
    d = np.linalg.norm(path[:, None, :] - Q[None, :, :], axis=2).min(1)
    return bool((d < LAR_INF - 0.1).any())

def _x43_lar_step(self, st, out, L, p):
    A = self._l43_act
    k = int(self.tick)
    A['n'] += 1
    self._x43_act += 1
    lz = float(L[2])
    dl = math.hypot(float(L[0]) - float(p[0]), float(L[1]) - float(p[1]))
    G = A['goal'] if A['goal'] is not None else np.array([float(L[0]), float(L[1]), lz + LAR_ZH])
    dg = float(np.linalg.norm(G - p))
    if dg < A['best'] - 0.3:
        A['best'] = dg
        A['bt'] = k
    why = None
    if A['n'] > LAR_TMAX:
        why = 'tmax'
    elif A['hold'] >= LAR_HOLD:
        why = 'hold'
    elif A['fail'] >= 2:
        why = 'nopath'
    elif not A['holding'] and k - A['bt'] > LAR_NOPROG:
        why = 'noprog'
    if why is not None:
        self._l43_act = None
        if len(self._x43_rel) < 20:
            self._x43_rel.append([k, why, round(dl, 2), round(float(p[2]) - lz, 2), A['n'], A['hold']])
        return out
    agl = float(st[162]) * 20.0 if st.size > 162 else 20.0
    if not A['holding'] and A['goal'] is not None and (float(np.linalg.norm(A['goal'] - p)) <= 0.4):
        A['holding'] = True
    if A['holding']:
        A['hold'] += 1
        self._x43_hold += 1
        rel = A['goal'] - p
        dn = float(np.linalg.norm(rel))
        out = out.copy()
        if dn > 1e-06:
            out[0:3] = (rel / dn).astype(np.float32)
            out[3] = np.float32(min(0.4, 0.8 * dn) / 3.0)
        else:
            out[0:3] = 0.0
            out[3] = np.float32(0.0)
        out[4] = np.float32(_wrap(float(st[5])) / math.pi)
        if not self.route.endswith('+lar'):
            self.route = self.route + '+lar'
        return out
    path = A['path']
    need = path is None or k - A['pt'] >= LAR_RP
    if not need:
        dev = float(np.linalg.norm(path[A['i']:] - p[None, :], axis=1).min())
        need = dev > LAR_DEV or (k - A['pt'] >= 10 and (k - A['pt']) % 10 == 0 and _x43_blocked(self, path[A['i']:]))
    if need:
        A['pt'] = k
        self._x43_plan += 1
        path = _x43_plan(self, p, L, agl)
        if path is None:
            A['fail'] += 1
            self._x43_nop += 1
            A['path'] = None
            return out
        A['path'] = path
        A['i'] = 0
        A['goal'] = path[-1].copy()
    d = np.linalg.norm(path[A['i']:] - p[None, :], axis=1)
    i = A['i'] + int(np.argmin(d))
    A['i'] = i
    j = min(len(path) - 1, i + LAR_LA)
    tgt = path[j]
    rel = tgt - p
    dn = float(np.linalg.norm(rel))
    if dn < 1e-06:
        return out
    out = out.copy()
    out[0:3] = (rel / dn).astype(np.float32)
    out[3] = np.float32(min(LAR_V, 0.3 + 0.5 * float(np.linalg.norm(A['goal'] - p))) / 3.0)
    if math.hypot(float(rel[0]), float(rel[1])) > 0.3:
        out[4] = np.float32(_wrap(math.atan2(float(rel[1]), float(rel[0]))) / math.pi)
    else:
        out[4] = np.float32(_wrap(float(st[5])) / math.pi)
    if not self.route.endswith('+lar'):
        self.route = self.route + '+lar'
    return out

def _x43_post(self, st, out):
    if not hasattr(self, '_x43_lar'):
        _x43_state(self)
    r2 = self._r2
    if r2 is None or r2.latch is None or getattr(r2, 'model_used', None) != FV_FOREST or (getattr(r2, '_loaded_for', None) != FV_FOREST):
        return out
    if self._r2_type() == 'village':
        return out
    L = np.asarray(r2.latch, np.float64).reshape(-1)
    p = st[0:3].astype(np.float64)
    v = st[6:9].astype(np.float64)
    k = int(self.tick)
    if self._l43_L is None or math.hypot(float(L[0]) - float(self._l43_L[0]), float(L[1]) - float(self._l43_L[1])) > 1.5:
        self._l43_L = L[0:2].copy()
        self._l43_h.clear()
        self._l43_done = False
        self._l43_act = None
        self._l43_fd = False
        self._l43_fdlog = False
        self._l43_v = {}
    dl = math.hypot(float(L[0]) - float(p[0]), float(L[1]) - float(p[1]))
    e = float(p[2]) - float(L[2])
    self._l43_h.append((k, dl, float(p[0]), float(p[1])))
    lt = r2.latch_t
    age = k - int(lt) if lt is not None else 0
    hv = math.hypot(float(v[0]), float(v[1]))
    mo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
    ph = round(float(mo[0]), 1) if mo.size else -1.0
    if FD_ON:
        tl = max(0.1, 58.0 - 0.02 * k)
        need = (e - 4.0) / tl
        if not self._l43_fd:
            if dl <= FD_R and float(p[2]) > FD_Z and (need > FD_NEED):
                self._l43_fd = True
                self._x43_fdn += 1
                if not self._l43_fdlog and len(self._x43_fd) < 20:
                    self._l43_fdlog = True
                    self._x43_fd.append([k, round(dl, 2), round(e, 2), round(need, 3), round(float(p[2]), 2)])
        elif dl > FD_R + 0.5 or e <= FD_TGT + 0.3:
            self._l43_fd = False
        if self._l43_fd and X43_MODE == 'on' and (self._l43_act is None):
            agl = float(st[162]) * 20.0 if st.size > 162 else 20.0
            if agl >= RP_AGL_MIN:
                dd = np.asarray(out[0:3], np.float64)
                nn = float(np.linalg.norm(dd))
                sp = float(out[3]) * 3.0
                vc = dd / nn * sp if nn > 1e-06 else np.zeros(3)
                vz = -min(FD_VZ, max(0.2, 1.2 * (e - FD_TGT)))
                vc[2] = min(float(vc[2]), vz)
                hn = math.hypot(float(vc[0]), float(vc[1]))
                hmax = math.sqrt(max(0.0, FD_VMAX * FD_VMAX - vc[2] * vc[2]))
                if hn > hmax:
                    vc[0:2] = vc[0:2] * (hmax / hn)
                m = float(np.linalg.norm(vc))
                if m > 1e-06:
                    out = out.copy()
                    out[0:3] = (vc / m).astype(np.float32)
                    out[3] = np.float32(m / 3.0)
                    self._x43_fdt = getattr(self, '_x43_fdt', 0) + 1
                    if not self.route.endswith('+fd'):
                        self.route = self.route + '+fd'
    if LAR_ON:
        H = self._l43_h
        for W in (100, 150, 250):
            if W in self._l43_v or age < LAR_AGE or (not RP_RING < dl <= LAR_DMAX) or (hv > LAR_HV) or (len(H) <= W):
                continue
            win = list(H)[-(W + 1):]
            if win[0][0] != k - W or max((x[1] for x in win)) > LAR_DMAX or (not all((x[1] > RP_RING for x in win))):
                continue
            if dl > win[0][1] + LAR_AWAY:
                continue
            if max((math.hypot(x[2] - win[0][2], x[3] - win[0][3]) for x in win)) > LAR_STAT:
                continue
            if win[0][1] - dl < LAR_PROG:
                self._l43_v[W] = k
                if len(self._x43_lar) < 30:
                    self._x43_lar.append([W, k, round(dl, 2), round(e, 2), ph, round(hv, 2), age, round(win[0][1] - dl, 2), round(min((x[1] for x in win)), 2)])
                if W == LAR_W and (not self._l43_done):
                    self._l43_done = True
                    self._x43_lan += 1
                    if X43_MODE == 'on':
                        self._l43_act = {'n': 0, 'hold': 0, 'holding': False, 'fail': 0, 'path': None, 'pt': -10 ** 9, 'best': 1000000000.0, 'bt': k, 'i': 0, 'goal': None}
        if self._l43_act is not None and X43_MODE == 'on':
            out = _x43_lar_step(self, st, out, L, p)
    return out

def _x43_hrz(self, st, out):
    out = _x43_orig_hrz(self, st, out)
    try:
        return _x43_post(self, st, out)
    except Exception:
        self._x43_err = getattr(self, '_x43_err', 0) + 1
        self._l43_act = None
        return out
_C032Controller._hrz = _x43_hrz
_X43Base = DroneFlightController

class DroneFlightController(_X43Base):

    def __init__(self, *, model_path=None, providers=None):
        _x43_state(self)
        _X43Base.__init__(self, model_path=model_path, providers=providers)

    def reset(self):
        _X43Base.reset(self)
        _x43_state(self)

    def act(self, observation):
        if '+lar' in self.route or '+fd' in self.route:
            self.route = self.route.replace('+lar', '').replace('+fd', '')
        return _X43Base.act(self, observation)
PV_ON = True
PAD_TOL = 5e-05

def _x43_pad(self):
    z0 = getattr(self, '_z0', None)
    return bool(getattr(self, '_vp_all', False)) and z0 is not None and (min((abs(float(z0) - h) for h in VP_SIG)) < PAD_TOL)
_pv_prev_stype = _C032Controller._stype
_pv_prev_r2_type = _C032Controller._r2_type

def _pv_hit(self):
    if not (PV_ON and _x43_pad(self)):
        return None
    t = _pv_prev_stype(self)
    return t if t in ('city', 'forest') else None

def _pv_stype(self):
    t = _pv_prev_stype(self)
    if PV_ON and t in ('city', 'forest') and _x43_pad(self):
        self._x43_pv = getattr(self, '_x43_pv', 0) + 1
        if getattr(self, '_x43_pv_t0', -1) < 0:
            self._x43_pv_t0 = int(self.tick)
            self._x43_pv_lab = t
        return 'village'
    return t

def _pv_r2_type(self):
    r = _pv_prev_r2_type(self)
    if r != 'village':
        c = self._cls
        if c is not None and (not c.is_mountain) and (c.n > 0) and (_pv_hit(self) is not None):
            return 'village'
    return r
_C032Controller._stype = _pv_stype
_C032Controller._r2_type = _pv_r2_type
RF_ON = True
_rf_prev_r2_type = _C032Controller._r2_type

def _rf_r2_type(self):
    if getattr(self, '_x43_rf_force', False):
        return 'village'
    return _rf_prev_r2_type(self)
_C032Controller._r2_type = _rf_r2_type
_RFBase = DroneFlightController

def _rf_state(self):
    self._x43_rf_force = False
    self._x43_rf = 0
    self._x43_rf_f = 0
    self._x43_rf_fix = 0
    self._x43_rf_lat = -1
    self._x43_rf_q = []
    self._x43_rf_err = 0
    self._rf_sh = None

def _rf_shadow(self, obs, st):
    rgb = obs.get('rgb') if hasattr(obs, 'get') else None
    if rgb is None:
        return
    fr = np.asarray(rgb, np.float32)
    if fr.size != 256 * 256 * 3 or float(np.mean(np.abs(fr))) < 0.005:
        return
    self._x43_rf_f += 1
    sh = self._rf_sh
    if sh is None:
        sh = self._rf_sh = _RgbPrimary(model='victim_rgb256.onnx', in_res=256, thr=R2_T, hmax=R2_HMAX, rng_max=R2_RANGE_MAX, term_r=R2_TERM_R, cap=R2_CAP, model_by_type=R2_SPLIT)
        sh.set_type('village')
        sh.clue0 = np.asarray(self._c0, np.float64).copy() if getattr(self, '_c0', None) is not None else None
    r2 = self._r2
    sh.bad = [np.asarray(b, np.float64).copy() for b in r2.bad]
    n0 = len(_FV_FAR)
    try:
        sh.observe(int(self.tick), obs, st[0:3].astype(np.float64), st[3:6].astype(np.float64))
    finally:
        del _FV_FAR[n0:]
    self._x43_rf_fix = int(sh.n_fix)
    if sh.latch is not None and r2.latch is None:
        r2.set_type('village')
        r2.latch = np.asarray(sh.latch, np.float64).copy()
        r2.near_since = None
        r2.latch_t = int(self.tick)
        r2.n_latch += 1
        r2.hits = []
        r2.n_rf = getattr(r2, 'n_rf', 0) + 1
        self._x43_rf_force = True
        self._x43_rf_lat = int(self.tick)
        self._x43_rf_q = [round(float(x), 2) for x in r2.latch[:3]]

class DroneFlightController(_RFBase):

    def __init__(self, *, model_path=None, providers=None):
        _rf_state(self)
        _RFBase.__init__(self, model_path=model_path, providers=providers)

    def reset(self):
        _RFBase.reset(self)
        _rf_state(self)

    def act(self, observation):
        out = _RFBase.act(self, observation)
        if RF_ON and (not self._x43_rf_force):
            try:
                if _x43_pad(self) and self._cls is not None and (not self._cls.is_mountain) and (self._r2 is not None) and (self._stype() == 'village') and (self._r2_type() is None):
                    self._x43_rf += 1
                    _rf_shadow(self, observation, np.asarray(observation['state'], np.float32).reshape(-1))
            except Exception:
                self._x43_rf_err += 1
        return out
VB_ON = True
VB_BOX = 29.0
_VB_PAD = [False]
_vb_orig_observe = _RgbPrimary.observe

def _vb_observe(self, tick, obs, pos, rpy):
    had = self.latch is not None
    r = _vb_orig_observe(self, tick, obs, pos, rpy)
    try:
        if VB_ON and _VB_PAD[0] and (not had) and (self.latch is not None) and (self.mtype == 'village') and (getattr(self, 'model_used', None) == R2_SPLIT.get('village')):
            L = np.asarray(self.latch, np.float64)
            if max(abs(float(L[0])), abs(float(L[1]))) > VB_BOX:
                self.latch = None
                self.near_since = None
                self.n_latch = max(0, self.n_latch - 1)
                self.hits = []
                self._sfl = None
                self.bad.append(L[0:2].copy())
                self.x43_vb = getattr(self, 'x43_vb', 0) + 1
                lg = self.__dict__.setdefault('x43_vb_log', [])
                if len(lg) < 4:
                    lg.append([int(tick)] + [round(float(x), 2) for x in L[:3]])
    except Exception:
        self.x43_vb_err = getattr(self, 'x43_vb_err', 0) + 1
    return r
_RgbPrimary.observe = _vb_observe
_VBBase = DroneFlightController

class DroneFlightController(_VBBase):

    def act(self, observation):
        try:
            _VB_PAD[0] = _x43_pad(self)
        except Exception:
            _VB_PAD[0] = False
        try:
            return _VBBase.act(self, observation)
        finally:
            _VB_PAD[0] = False
VZ_ON = True
VZ_LAT = 3.0
_vz_orig_observe = _RgbPrimary.observe

def _vz_observe(self, tick, obs, pos, rpy):
    had = self.latch is not None
    r = _vz_orig_observe(self, tick, obs, pos, rpy)
    try:
        if VZ_ON and _VB_PAD[0] and (not had) and (self.latch is not None) and (self.mtype == 'village') and (getattr(self, 'model_used', None) == R2_SPLIT.get('village')):
            L = np.asarray(self.latch, np.float64)
            if float(L[2]) > VZ_LAT:
                self.latch = None
                self.near_since = None
                self.n_latch = max(0, self.n_latch - 1)
                self._sfl = None
                self.hits = [(t, z) for t, z in self.hits if float(z[2]) <= VZ_LAT]
                self.x43_vz = getattr(self, 'x43_vz', 0) + 1
                lg = self.__dict__.setdefault('x43_vz_log', [])
                if len(lg) < 4:
                    lg.append([int(tick)] + [round(float(x), 2) for x in L[:3]])
    except Exception:
        self.x43_vz_err = getattr(self, 'x43_vz_err', 0) + 1
    return r
_RgbPrimary.observe = _vz_observe
VRV_OOF = float(os.environ.get('KT_X_OOF', '0'))
_oof_prev_vrv = _C032Controller._vrv

def _oof_vrv(self, obs):
    o0 = int(getattr(self, '_vrv_oof', 0))
    r0 = int(getattr(self, '_vrv_rev', 0))
    try:
        return _oof_prev_vrv(self, obs)
    finally:
        o1 = int(getattr(self, '_vrv_oof', 0))
        r1 = int(getattr(self, '_vrv_rev', 0))
        self._x43_oof = o1
        if o1 > o0 and r1 > r0:
            self._x43_oofrev = getattr(self, '_x43_oofrev', 0) + 1
_C032Controller._vrv = _oof_vrv
CP3H = os.environ.get('KT_X_P3H', '0') == '1'
_P3HBase = DroneFlightController

class DroneFlightController(_P3HBase):

    def act(self, observation):
        n0 = len(getattr(self, '_p3_ban', None) or [])
        out = _P3HBase.act(self, observation)
        try:
            pb = getattr(self, '_p3_ban', None) or []
            if len(pb) > n0 and self._stype() == 'city':
                mo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                if mo.size >= 63 and float(np.linalg.norm(np.asarray(pb[-1], np.float64)[0:2] - mo[60:62].astype(np.float64))) >= P3_BAN_R:
                    self._x43_p3h = getattr(self, '_x43_p3h', 0) + 1
        except Exception:
            self._x43_p3h_err = getattr(self, '_x43_p3h_err', 0) + 1
        return out
TPW_ON = True
TPW_T0 = 2900 - 110 - 60
_tpw_orig_dib = _C032Controller._dib
_tpw_orig_hrz = _C032Controller._hrz

def _tpw_log(self, kind, dl, e):
    lg = self.__dict__.setdefault('_x43_tpw', [])
    if not any((x[1] == kind for x in lg)) and len(lg) < 6:
        lg.append([int(self.tick), kind, round(dl, 2), round(e, 2)])

def _tpw_dib(self, st, out):
    try:
        if TPW_ON and int(self.tick) >= TPW_T0 and (not self._dib_arm) and (self._dib_L is not None):
            L = np.asarray(self._r2.latch, np.float64).reshape(-1)
            p = st[0:3].astype(np.float64)
            v = st[6:9].astype(np.float64)
            if math.hypot(float(L[0] - self._dib_L[0]), float(L[1] - self._dib_L[1])) <= 3.0:
                dl = math.hypot(float(L[0] - p[0]), float(L[1] - p[1]))
                e = float(p[2] - L[2])
                if dl <= DIB_R and math.hypot(float(v[0]), float(v[1])) <= DIB_HV and (e > DIB_HI) and (self._dib_w < DIB_WAIT - 1):
                    self._dib_w = DIB_WAIT - 1
                    self._x43_tpwn = getattr(self, '_x43_tpwn', 0) + 1
                    _tpw_log(self, 'dib', dl, e)
    except Exception:
        self._x43_tpw_err = getattr(self, '_x43_tpw_err', 0) + 1
    return _tpw_orig_dib(self, st, out)

def _tpw_hrz(self, st, out):
    try:
        if TPW_ON and int(self.tick) >= TPW_T0 and (not self._hrz_arm) and (self._hrz_L is not None):
            L = np.asarray(self._r2.latch, np.float64).reshape(-1)
            p = st[0:3].astype(np.float64)
            v = st[6:9].astype(np.float64)
            if math.hypot(float(L[0] - self._hrz_L[0]), float(L[1] - self._hrz_L[1])) <= 1.0 and (not any((math.hypot(float(L[0] - b[0]), float(L[1] - b[1])) <= 1.0 for b in self._hrz_ban))):
                dh = math.hypot(float(L[0] - p[0]), float(L[1] - p[1]))
                e = float(p[2] - L[2])
                mo = np.asarray(self.memory_tensor, np.float32).reshape(-1)
                ph3 = bool(mo.size and float(mo[0]) >= 2.5)
                r2t = getattr(self, '_r2_term', None) is not None
                ok = (ph3 or r2t) and dh <= HRZ_R and (HRZ_ELO <= e <= HRZ_EHI) and (float(np.linalg.norm(v)) <= HRZ_V) and (abs(float(v[2])) <= HRZ_VZ) and (float(p[2]) < HRZ_ZMAX)
                if ok and self._hrz_w < HRZ_WAIT - 1:
                    self._hrz_w = HRZ_WAIT - 1
                    self._x43_tpwn = getattr(self, '_x43_tpwn', 0) + 1
                    _tpw_log(self, 'hrz', dh, e)
    except Exception:
        self._x43_tpw_err = getattr(self, '_x43_tpw_err', 0) + 1
    return _tpw_orig_hrz(self, st, out)
_C032Controller._dib = _tpw_dib
_C032Controller._hrz = _tpw_hrz
_TPWBase = DroneFlightController

class DroneFlightController(_TPWBase):

    def reset(self):
        _TPWBase.reset(self)
        for _k in ('_x43_tpw', '_x43_tpwn', '_x43_tpw_err'):
            self.__dict__.pop(_k, None)
_vnt_prev_observe = _RgbPrimary.observe

def _vnt_observe(self, tick, obs, pos, rpy):
    if not (VNT_ON and VNT_AGREE and getattr(self, '_vnt_r2', False)) or self.latch is not None:
        return _vnt_prev_observe(self, tick, obs, pos, rpy)
    n0 = self.n_fix
    r = _vnt_prev_observe(self, tick, obs, pos, rpy)
    try:
        if self.latch is None and self.n_fix > n0 and (self.last_q is not None) and (self.last_q_t == tick):
            q = np.asarray(self.last_q, np.float64)
            if float(q[2]) <= 3.0 and max(abs(float(q[0])), abs(float(q[1]))) <= 29.0:
                for t, h in _VNT_H:
                    if tick - t <= VNT_AW and float(np.hypot(float(h[0]) - q[0], float(h[1]) - q[1])) <= VNT_AD:
                        self.latch = q.copy()
                        self.near_since = None
                        self.n_latch += 1
                        self.latch_t = tick
                        self._sfl = None
                        self.n_vnt_al = getattr(self, 'n_vnt_al', 0) + 1
                        break
    except Exception:
        self.n_vnt_err = getattr(self, 'n_vnt_err', 0) + 1
    return r
_RgbPrimary.observe = _vnt_observe
