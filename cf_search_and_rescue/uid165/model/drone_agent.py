import os
os.environ.setdefault('KT_AV_D', '4.0')
from pathlib import Path
import math
import numpy as np
import onnxruntime as ort
_H=Path(__file__).resolve().parent
ON=os.environ.get('KT_ON','1')=='1'
AV_D=float(os.environ.get('KT_AV_D','3.0'))
AV_REL=float(os.environ.get('KT_AV_REL','0.7'))
AV_ROT=float(os.environ.get('KT_AV_ROT','45.0'))
AV_HOLD=int(os.environ.get('KT_AV_HOLD','15'))
AV_T0=int(os.environ.get('KT_AV_T0','100'))
AV_WIN=int(os.environ.get('KT_AV_WIN','32'))
AV_TTC=float(os.environ.get('KT_AV_TTC','0'))
AV_TYPES=tuple(t for t in os.environ.get('KT_AV_TYPES','forest,city').split(',') if t)
CFG={
 'forest':   (350,15,1000000000.0,8.0,5.0,40,120,8.0,0,0.75),
 'city':     (450,25,10.0,6.0,5.0,40,250,4.5,1),
 'mountain': (450, 15, 1000000000.0, 12.0, 8.0, 25, 120, 8.0, 0, 0.75),
 'village':  (450,15,10.0,8.0,5.0,40,120,8.0,0,0.75),
}
MTN_STRONG=(450,15,1000000000.0,12.0,8.0,25,120,8.0,0,0.75)
MTN_THR=0.5
RL_ON=os.environ.get('KT_RL','1')=='1'; RL_N=int(os.environ.get('KT_RL_N','40')); RL_BAR=float(os.environ.get('KT_RL_BAR','0.55'))
RL_TYPES=tuple(t for t in os.environ.get('KT_RL_TYPES','city,forest').split(',') if t)
LADDER=os.environ.get('KT_LADDER','1')=='1'
TYPES=tuple(t for t in os.environ.get('KT_TYPES','forest,village,city,mountain').split(',') if t)
LB=('city','open','mountain','village','warehouse','forest')
SIM_DT=0.02
VW=float(os.environ.get('KT_VETO_WARM','0.02'))
RP_ON=os.environ.get('KT_RP','1')=='1'
RP_T=float(os.environ.get('KT_RP_T','0.713'))
RP_HMAX=float(os.environ.get('KT_RP_HMAX','1.0'))
RP_TYPES=tuple(t for t in os.environ.get('KT_RP_TYPES','').split(',') if t)
RP_TDEC=float(os.environ.get('KT_RP_TDEC','-1'))
RP_TDEC_T=float(os.environ.get('KT_RP_TDEC_T','1200'))
def _rpbar(tick):
 if RP_TDEC < 0.0 or tick <= RP_T0:
  return RP_T
 f=min(1.0,(float(tick)-RP_T0)/max(1.0,RP_TDEC_T-RP_T0))
 return RP_T+(RP_TDEC-RP_T)*f
RP_T0=150; RP_EVERY=75; RP_CAP=40; RP_CHASE_MIN=12; RP_CHASE_TICKS=36
RP_RANGE_MAX=float(os.environ.get('KT_RP_RANGE','20.0')); RP_IMG_MAX=29.0; RP_CLUE_R=55.0; RP_LATCH_N=int(os.environ.get('KT_RP_LATCH_N','2')); RP_LATCH_TOL=5.0; RP_EMA=0.3
RP_TERM_R=6.0; RP_HOVER_UP=3.2; RP_DUD_TICKS=300; RP_AGL_MIN=2.0; RP_RING=1.6; RP_TERM_MAX=1200
R2_ON=os.environ.get('KT_R2','1')=='1'
R2_TYPES=tuple(t for t in os.environ.get('KT_R2_TYPES','forest,warehouse,village').split(',') if t in LB)
R2_T=float(os.environ.get('KT_R2_T','0.65'))
R2_BAR=float(os.environ.get('KT_R2_BAR','0.5'))
R2_HMAX=float(os.environ.get('KT_R2_HMAX','inf'))
R2_RANGE_MAX=float(os.environ.get('KT_R2_RANGE_MAX','25.0'))
R2_CAP=int(os.environ.get('KT_R2_CAP',str(RP_CAP-3)))
MTN_Z0=float(os.environ.get('KT_MTN_Z0','13.0'))
VIL_AGL=float(os.environ.get('KT_VIL_AGL','0.0'))
VIL_REL=float(os.environ.get('KT_VIL_REL','15.0'))
VIL_KZ=float(os.environ.get('KT_VIL_KZ','0.35'))
VIL_DB=float(os.environ.get('KT_VIL_DB','0.3'))
class _R2Off(Exception):
 pass
_DMIN,_DMAX=0.5,30.0; _CAMF,_CAMU,_HT=0.13,0.05,1.0
def _sig(x):
 return 1.0/(1.0+math.exp(-max(-40.0,min(40.0,float(x)))))
def _axes(rpy):
 r,p,y=(float(rpy[0]),float(rpy[1]),float(rpy[2]))
 cr,sr,cp,sp,cy,sy=(math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y))
 fwd=np.array([cy*cp,sy*cp,-sp],np.float64); up=np.array([cy*sp*cr+sy*sr,sy*sp*cr-cy*sr,cp*cr],np.float64)
 return fwd,up,np.cross(fwd,up)
def _mworld(u,v,cz,pos,rpy):
 fwd,up,right=_axes(rpy); cam=np.asarray(pos,np.float64)+fwd*_CAMF+up*_CAMU
 return cam+right*(u*_HT*cz)+up*(v*_HT*cz)+fwd*cz
CHAMP_ROUTE=os.environ.get('KT_CHAMP_ROUTE','1')=='1'
ESC_TERR=os.environ.get('KT_ESC_TERR','1')=='1'
ESC_ON=os.environ.get('KT_ESC','1')=='1'; ESC_TRAP_M=float(os.environ.get('KT_ESC_TRAP','1.0')); ESC_FREE_M=3.0; ESC_DIST=float(os.environ.get('KT_ESC_DIST','2.0'))
WESC_ON=os.environ.get('KT_WESC','1')=='1'; WESC_FRAC=float(os.environ.get('KT_WESC_FRAC','0.5'))
WESC_H=float(os.environ.get('KT_WESC_H','0.45')); WESC_HOR=float(os.environ.get('KT_WESC_HOR','1.0'))
WESC_Z=float(os.environ.get('KT_WESC_Z','0.3')); WESC_MODE=os.environ.get('KT_WESC_MODE','side')
WESC_BACK=float(os.environ.get('KT_WESC_BACK','1.2')); WESC_TOP=float(os.environ.get('KT_WESC_TOP','1.2'))
WESC_SIDE_M=float(os.environ.get('KT_WESC_SIDE_M','0.6')); WESC_SIDE_SPEED=float(os.environ.get('KT_WESC_SIDE_SPEED','0.2')); WESC_SIDE_MAX=float(os.environ.get('KT_WESC_SIDE_MAX','1.0'))
ESC_SPEED=0.3; ESC_CLIMB=0.6; ESC_AGL_TARGET=2.5; ESC_MAX_CREEP=300; ESC_MARGIN=1.0; ESC_BODY_M=1.2; ESC_MAX_CLIMB=150; ESC_CLEAR_TICKS=10; ESC_CREEP_AGL=0.28; ESC_SCAN_TICKS=45
class _Escape:
 def __init__(self):
  self.reset()
 def reset(self):
  self.mode=None; self.dist=0.0; self.phase=None; self.heading=None; self.start=None; self.clear=0; self.n=0; self.trap=False; self.cycles=0; self.climb_start=None; self.yaw0=0.0; self.scan_i=0; self.scan_n=0; self.margin_start=None; self.guard_hits=0
 def opening(self,depth,yaw):
  d=np.asarray(depth,np.float32).reshape(256,256)*(_DMAX-_DMIN)+_DMIN
  bands=[float(d[112:128,c0:c0+52].min()) for c0 in (0,51,102,153,204)]
  best=max(bands); k=min((i for i in range(5) if bands[i]>=best-0.05), key=lambda i: abs(i-2))
  if bands[k]<ESC_FREE_M:
   return None
  return float(yaw)-math.radians((k-2)*18.0)
 def _back(self):
  return self.mode=='ceil' and WESC_MODE in ('back','side')
 def _yawcmd(self):
  h=self.yaw0 if (self._back() or self.heading is None) else self.heading
  return float(((h+math.pi)%(2*math.pi)-math.pi)/math.pi)
 def _scan_seq(self):
  return (180.0,90.0,-90.0) if self.mode=='ceil' else (90.0,-90.0,180.0)
 def _open(self,depth,yaw):
  if self.mode!='ceil':
   return self.opening(depth,yaw)
  d=np.asarray(depth,np.float32).reshape(256,256)*(_DMAX-_DMIN)+_DMIN
  bands=[(float(d[112:128,c0:c0+52].min()),float(d[0:64,c0:c0+52].min())) for c0 in (0,51,102,153,204)]
  ok=[i for i in range(5) if bands[i][0]>=ESC_FREE_M and bands[i][1]>=WESC_TOP]
  if not ok:
   return None
  k=min(ok,key=lambda i:(abs(i-2),-bands[i][0]))
  return float(yaw)-math.radians((k-2)*18.0)
 def _side(self,d,yaw):
  v=(128.0-np.arange(96)-0.5)/128.0; u=(np.arange(256)+0.5-128.0)/128.0
  up=d[0:96]; low=((up*v[:,None])<1.0)&(up<3.0)
  el=[]; er=[]
  for r in range(0,96,4):
   row=low[r]
   if not row[124:132].any():
    continue
   jl=128
   while jl>0 and row[jl-1]:
    jl-=1
   jr=127
   while jr<255 and row[jr+1]:
    jr+=1
   el.append(-float(u[jl])*float(d[r,jl])); er.append(float(u[jr])*float(d[r,jr]))
  if el:
   l=float(np.median(el)); rr=float(np.median(er)); e=min(l,rr)
  else:
   l=float(low[:,0:96].sum()); rr=float(low[:,160:256].sum()); e=0.4
  right=rr<l
  dist=float(min(WESC_SIDE_MAX,max(0.6,e+WESC_SIDE_M)))
  return float(yaw)+(-0.5*math.pi if right else 0.5*math.pi),dist
 def ceiling(self,depth,pos,yaw):
  if float(pos[2])>WESC_Z:
   return False
  d=np.asarray(depth,np.float32).reshape(256,256)*(_DMAX-_DMIN)+_DMIN
  v=(128.0-np.arange(96)-0.5)/128.0
  up=d[0:96,96:160]; hup=up*v[:,None]; near=up<3.0
  if not near.any():
   return False
  frac=float(((hup<1.0)&near).mean()); hmed=float(np.median(hup[near])); hor=float(np.median(d[96:128,96:160]))
  if frac<WESC_FRAC or hmed>WESC_H or hor>WESC_HOR:
   return False
  self.trap=True; self.mode='ceil'; self.start=np.asarray(pos[0:2],np.float64).copy(); self.yaw0=float(yaw)
  if WESC_MODE=='side':
   self.heading,self.dist=self._side(d,yaw); self.phase='creep'; return True
  if WESC_MODE=='back':
   self.heading=float(yaw)+math.pi; self.phase='creep'; return True
  h=self._open(depth,yaw)
  if h is not None:
   self.heading=h; self.phase='creep'; return True
  self.phase='scan'; self.scan_i=0; self.scan_n=0; return True
 def decide(self,depth,pos,yaw):
  d=np.asarray(depth,np.float32).reshape(256,256)*(_DMAX-_DMIN)+_DMIN
  if float((d[0:16]<0.8).mean())<0.95:
   return False
  self.trap=True; self.start=np.asarray(pos[0:2],np.float64).copy(); self.yaw0=float(yaw)
  h=self.opening(depth,yaw)
  if h is not None:
   self.heading=h; self.phase='creep'; return True
  self.phase='scan'; self.scan_i=0; self.scan_n=0; return True
 def act(self,depth,pos,agl,pitch_deg,yaw):
  self.n+=1
  d=np.asarray(depth,np.float32).reshape(256,256)*(_DMAX-_DMIN)+_DMIN
  top=float(d[0:64].min())
  if self.phase=='scan':
   tgt=self.yaw0+math.radians(self._scan_seq()[self.scan_i]); self.scan_n+=1
   err=(yaw-tgt+math.pi)%(2*math.pi)-math.pi
   if abs(err)<math.radians(8.0) or self.scan_n>=ESC_SCAN_TICKS:
    h=self._open(depth,yaw) if abs(err)<math.radians(12.0) else None
    if h is not None:
     self.heading=h; self.phase='creep'; self.n=0; self.clear=0; self.start=np.asarray(pos[0:2],np.float64).copy()
     return np.array([math.cos(h),math.sin(h),0.0],np.float32),ESC_SPEED,(self._yawcmd() if self.mode=='ceil' else None)
    self.scan_i+=1; self.scan_n=0
    if self.scan_i>=3:
     self.phase='done'; return None
    tgt=self.yaw0+math.radians(self._scan_seq()[self.scan_i])
   yt=(tgt+math.pi)%(2*math.pi)-math.pi
   return np.zeros(3,np.float32),0.0,float(yt/math.pi)
  if self.phase=='creep':
   self.clear=self.clear+1 if (top>=2.0 and abs(pitch_deg)<6.0) else 0
   trav=float(np.linalg.norm(np.asarray(pos[0:2],np.float64)-self.start))
   if self.clear>=ESC_CLEAR_TICKS and self.margin_start is None:
    self.margin_start=np.asarray(pos[0:2],np.float64).copy()
   past=(self.margin_start is not None and float(np.linalg.norm(np.asarray(pos[0:2],np.float64)-self.margin_start))>=ESC_MARGIN)
   rel=(self.heading-yaw+math.pi)%(2*math.pi)-math.pi
   kb=min(4,max(0,int(round(2-math.degrees(rel)/18.0)))); c0=(0,51,102,153,204)[kb]
   if float(d[100:128,c0:c0+52].min())<ESC_BODY_M and self.n>15 and self.guard_hits<3 and not self._back():
    self.guard_hits+=1; self.phase='scan'; self.scan_i=0; self.scan_n=0; self.yaw0=float(yaw); self.clear=0; self.margin_start=None
    return np.zeros(3,np.float32),0.0,None
   if self.mode=='ceil' and WESC_MODE=='side' and (trav>=self.dist or self.n>=ESC_MAX_CREEP):
    self.phase='done'; return None
   if (past and not (self.mode=='ceil' and WESC_MODE=='side')) or self.n>=ESC_MAX_CREEP or (WESC_MODE=='back' and self._back() and trav>=WESC_BACK):
    self.phase='climb'; self.n=0; self.climb_start=np.asarray(pos[0:2],np.float64).copy()
   else:
    vz=float(np.clip((ESC_CREEP_AGL-agl)*3.0,-0.5,0.5))
    v=np.array([math.cos(self.heading),math.sin(self.heading),vz],np.float64); v/=max(1e-6,float(np.linalg.norm(v)))
    return v.astype(np.float32),(WESC_SIDE_SPEED if (self.mode=='ceil' and WESC_MODE=='side') else ESC_SPEED),(self._yawcmd() if self.mode=='ceil' else None)
  if self.phase=='climb':
   if agl>=ESC_AGL_TARGET or self.n>=ESC_MAX_CLIMB:
    self.phase='done'; return None
   if top<0.9 and agl<1.0 and self.cycles<2:
    self.cycles+=1; self.phase='creep'; self.n=0; self.clear=0; self.margin_start=None; self.start=np.asarray(pos[0:2],np.float64).copy()
    return np.array([math.cos(self.heading),math.sin(self.heading),0.0],np.float32),ESC_SPEED,(self._yawcmd() if self.mode=='ceil' else None)
   return np.array([0.0,0.0,1.0],np.float32),ESC_CLIMB,(self._yawcmd() if self.mode=='ceil' else None)
  return None
class _RgbPrimary:
 def __init__(self,model='victim_rgb_m.onnx',in_res=128,thr=None,hmax=None,rng_max=None,term_r=None,cap=None):
  self.sess=None; self._tried=False
  self.model=model; self.in_res=int(in_res)
  self.thr=thr if thr is None else float(thr)
  self.hmax=RP_HMAX if hmax is None else float(hmax)
  self.rng_max=RP_RANGE_MAX if rng_max is None else float(rng_max)
  self.term_r=RP_TERM_R if term_r is None else float(term_r)
  self.cap=RP_CAP if cap is None else int(cap)
  self.reset()
 def _load(self):
  self._tried=True
  try:
   so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
   self.sess=ort.InferenceSession(str(_H/self.model),so,providers=['CPUExecutionProvider']); self.iname=self.sess.get_inputs()[0].name
  except Exception:
   self.sess=None
 def reset(self):
  self.n_req=0; self.last_req=-10**9; self.hit_t=-10**9; self.hits=[]; self.latch=None; self.near_since=None; self.bad=[]; self.clue0=None
  self.n_fix=0; self.n_latch=0; self.n_dud=0; self.latch_t=None
 def score(self,rgb):
  fr=np.asarray(rgb,np.float32)
  if self.sess is None or fr.size!=256*256*3 or float(np.mean(np.abs(fr)))<0.005:
   return None
  if self.in_res==256:
   r=np.ascontiguousarray(fr.reshape(256,256,3),np.float32)
  else:
   r=np.ascontiguousarray(fr.reshape(256,256,3).reshape(128,2,128,2,3).mean(axis=(1,3)),np.float32)
  o=np.asarray(self.sess.run(None,{self.iname:r})[0],np.float32).reshape(5)
  return _sig(o[0]),float(o[1]),float(o[2]),float(o[4])
 def localise(self,u,v,depth,pos,rpy):
  d=np.asarray(depth,np.float32).reshape(256,256)
  px=min(255,max(0,int(round((u+1.0)*0.5*256-0.5)))); py=min(255,max(0,int(round((1.0-v)*0.5*256-0.5))))
  cz=float(np.median(d[max(0,py-2):py+3,max(0,px-2):px+3]))*(_DMAX-_DMIN)+_DMIN
  if cz>RP_IMG_MAX or cz>self.rng_max:
   return None
  q=_mworld(u,v,cz,pos,rpy)
  if not np.all(np.isfinite(q)):
   return None
  if self.clue0 is not None and float(np.linalg.norm(q[0:2]-self.clue0))>RP_CLUE_R:
   return None
  if not (-25.0<=float(q[2]-pos[2])<=1.0):
   return None
  for b in self.bad:
   if float(np.linalg.norm(q[0:2]-b))<6.0:
    return None
  return q
 def observe(self,tick,obs,pos,rpy):
  if self.sess is None and not self._tried:
   self._load()
  rgb=obs.get('rgb') if hasattr(obs,'get') else None
  if rgb is None:
   return
  sc=self.score(rgb)
  if sc is None:
   return
  p,u,v,h=sc
  if p<(_rpbar(tick) if self.thr is None else self.thr) or h>self.hmax:
   return
  q=self.localise(u,v,obs['depth'],pos,rpy)
  if q is None:
   return
  self.hit_t=tick; self.n_fix+=1
  if self.latch is not None:
   if float(np.linalg.norm(q[0:2]-self.latch[0:2]))<=6.0:
    self.latch=(1.0-RP_EMA)*self.latch+RP_EMA*q
   return
  self.hits=[(t,z) for t,z in self.hits if tick-t<=400]+[(tick,q)]
  if len(self.hits)>=RP_LATCH_N:
   pts=np.asarray([z for _,z in self.hits[-RP_LATCH_N:]]); c=pts.mean(0)
   if float(np.max(np.linalg.norm(pts[:,0:2]-c[0:2],axis=1)))<=RP_LATCH_TOL:
    self.latch=c; self.near_since=None; self.n_latch+=1; self.latch_t=tick
 def want_rgb(self,tick):
  if self.sess is None and not self._tried:
   self._load()
  if self.sess is None or self.n_req>=self.cap or tick<RP_T0:
   return False
  chase=(tick-self.hit_t<=RP_CHASE_TICKS) and (tick-self.last_req>=RP_CHASE_MIN)
  paced=(tick-self.last_req>=RP_EVERY)
  if chase or paced:
   self.n_req+=1; self.last_req=tick; return True
  return False
 def tick_end(self,tick,pos):
  if self.latch is None:
   return
  if float(np.linalg.norm(self.latch[0:2]-pos[0:2]))<=RP_RING:
   if self.near_since is None:
    self.near_since=tick
   elif tick-self.near_since>RP_DUD_TICKS:
    self.bad.append(self.latch[0:2].copy()); self.latch=None; self.hits=[]; self.near_since=None; self.n_dud+=1
  else:
   self.near_since=None
  if self.latch is not None and self.latch_t is not None and tick-self.latch_t>RP_TERM_MAX:
   self.bad.append(self.latch[0:2].copy()); self.latch=None; self.hits=[]; self.near_since=None; self.n_dud+=1
 def terminal(self,pos,agl_m):
  if self.latch is None:
   return None
  off=self.latch[0:2]-pos[0:2]; d=float(np.linalg.norm(off))
  if d>self.term_r:
   return None
  tz=float(self.latch[2])+RP_HOVER_UP; dz=tz-float(pos[2])
  if dz<0 and agl_m<RP_AGL_MIN:
   dz=0.3
  dzc=max(-1.5,min(1.5,dz))
  vec=np.array([off[0],off[1],dzc],np.float64); n=float(np.linalg.norm(vec))
  dirn=vec/n if n>1e-6 else np.zeros(3)
  if d<1.0 and abs(dz)<0.7:
   speed=0.04
  elif d<RP_RING:
   speed=0.15
  else:
   speed=min(0.33,max(0.15,0.05*d+0.10))
  return dirn.astype(np.float32),float(speed)
def _dfeat(depth):
 d=np.asarray(depth,np.float32)
 if d.ndim==3 and d.shape[-1]==1:
  d=d[...,0]
 if d.ndim!=2:
  d=np.reshape(d,d.shape[:2])
 d=np.clip(d,0.0,1.0)
 gx=np.abs(np.diff(d,axis=1))
 gy=np.abs(np.diff(d,axis=0))
 pc=np.percentile(d,[1,5,10,25,50,75,90,95,99])
 v=[float(d.min()),float(d.mean()),float(d.std()),float(d.max())]
 v += [float(p) for p in pc]
 v += [float((d<=0.1).mean()),float((d<=0.25).mean()),float((d>=0.95).mean()),
  float((d>=0.999).mean()),float((d<=0.001).mean()),
  float(gx.mean()+gy.mean()),float(((gx>0.05).mean()+(gy>0.05).mean())/2.0)]
 h,w=d.shape
 tm,tn,tx=[],[],[]
 for y0 in np.linspace(0,h,5,dtype=int)[:-1]:
  y1=int(y0+h // 4)
  for x0 in np.linspace(0,w,5,dtype=int)[:-1]:
   x1=int(x0+w // 4)
   t=d[y0:y1,x0:x1]
   tm.append(float(t.mean())); tn.append(float(t.min())); tx.append(float(t.max()))
 return v+tm+tn+tx
class _Net:
 def __init__(self,p):
  b=open(p,'rb').read()
  d,h=(int(x) for x in np.frombuffer(b[:4],np.uint16)); o=4
  n1=d*h; n2=h*6
  self.w1=np.frombuffer(b[o:o+n1],np.int8).reshape(d,h).astype(np.float32); o += n1
  self.w2=np.frombuffer(b[o:o+n2],np.int8).reshape(h,6).astype(np.float32); o += n2
  self.s1,self.s2=np.frombuffer(b[o:o+8],np.float32); o += 8
  self.b1=np.frombuffer(b[o:o+2*h],np.float16).astype(np.float32); o += 2*h
  self.b2=np.frombuffer(b[o:o+12],np.float16).astype(np.float32); o += 12
  self.mu=np.frombuffer(b[o:o+2*d],np.float16).astype(np.float32); o += 2*d
  self.sd=np.frombuffer(b[o:o+2*d],np.float16).astype(np.float32)
  self.w1 *= self.s1; self.w2 *= self.s2
 def probs(self,f):
  x=(f-self.mu)/self.sd
  hd=np.maximum(x@self.w1+self.b1,0)
  z=hd@self.w2+self.b2
  z -= z.max()
  e=np.exp(z)
  return e/e.sum()
class _Cls:
 def __init__(self):
  self.net=_Net(_H/'typenet.bin')
  self.reset()
 def reset(self):
  self.ps=np.zeros(6); self.n=0
  self.latched=set(); self.is_mountain=False; self.p40=None; self.relatched=set()
 def update(self,tick,state,depth):
  if tick>600 or tick%5:
   return
  s=np.asarray(state,np.float32).reshape(-1)
  sf=np.zeros(141,np.float32)
  sf[:min(141,s.size)]=s[:141]
  f=np.asarray([float(tick),float(tick)*SIM_DT]+sf.tolist()+_dfeat(depth),np.float32)
  self.ps += self.net.probs(f); self.n += 1
  avg=self.ps/self.n
  if self.n==40:
   self.p40=float(avg[2])
  i=int(np.argmax(avg)); lab=LB[i]
  bar=0.5 if lab=='warehouse' else 0.7
  if self.n>=3 and avg[i]>=bar:
   self.latched.add(lab)
   if lab=='mountain':
    self.is_mountain=True
  if self.n>=40 and avg[4]>=0.15:
   self.latched.add('warehouse')
  if RL_ON and self.n>=RL_N and lab in RL_TYPES and avg[i]>=RL_BAR:
   self.relatched.add(lab)
 def settled(self,t):
  return t in self.latched
 def mtn_ok(self):
  return bool(self.is_mountain and self.p40 is not None and self.p40>=MTN_THR)
def _warm(rgb):
 im=np.asarray(rgb,np.float32).reshape(256,256,3)[128:]
 r,g,b=im[...,0],im[...,1],im[...,2]
 return float(((r>g+0.03) & (g>=b)).mean())
_BOX=np.array([25.0,30.0,30.0,25.0,12.0,20.0])
CVR=9.0
def _ccore(origin,pad,rpad=83.0,boxes=None):
 xs=np.arange(-48.0,50.0,2.0)
 X,Y=np.meshgrid(xs,xs)
 pts=np.stack([X.ravel(),Y.ravel()],1)
 dc=np.linalg.norm(pts-origin,axis=1)
 dp=np.linalg.norm(pts-pad,axis=1)
 m=(dc<=33.0)&(dp<=rpad)
 pts,dc,dp=pts[m],dc[m],dp[m]
 ax=np.abs(pts).max(1)
 dens=np.zeros(len(pts))
 bx=_BOX if boxes is None else boxes
 for bb in bx:
  dens[ax<=bb] += 1.0/(6.0*(2*bb) ** 2) if boxes is None else 1.0/(len(bx)*(2*bb) ** 2)
 w=dens/(1+np.exp(-(30.0-dc)))/(1+np.exp(-((rpad-3.0)-dp)))
 return pts,w,dc
def _bous(core,lane,spacing,cap,pos):
 core=np.asarray(core,np.float64)
 if core.shape[0]<4:
  return []
 k=np.round(core[:,0]/lane).astype(int)
 o=np.lexsort((np.where(k%2==0,core[:,1],-core[:,1]),k))
 path=core[o]
 sel=[path[0]]
 for q in path[1:]:
  if float(np.linalg.norm(q-sel[-1]))>=spacing:
   sel.append(q)
 w=sel[:cap]
 if len(w)>=2 and np.linalg.norm(w[-1]-pos)<np.linalg.norm(w[0]-pos):
  w=w[::-1]
 return [np.asarray(x,np.float64) for x in w]
def _wps(origin,pad,pos,lane,spacing,cap):
 xs=np.arange(-48.0,50.0,2.0)
 X,Y=np.meshgrid(xs,xs)
 pts=np.stack([X.ravel(),Y.ravel()],1)
 dc=np.linalg.norm(pts-origin,axis=1)
 dp=np.linalg.norm(pts-pad,axis=1)
 m=(dc<=33.0) & (dp<=83.0)
 pts,dc,dp=pts[m],dc[m],dp[m]
 ax=np.abs(pts).max(1)
 dens=np.zeros(len(pts))
 for bb in _BOX:
  dens[ax<=bb] += 1.0/(6.0*(2*bb) ** 2)
 w=dens/(1+np.exp(-(30.0-dc)))/(1+np.exp(-(80.0-dp)))
 core=pts
 if w.sum()>0:
  o=np.argsort(-w)
  cw=np.cumsum(w[o])/w.sum()
  core=pts[o[:int(np.searchsorted(cw,0.9))+1]]
 if core.shape[0]<4:
  core=pts[dc<=30.0]
 if not core.shape[0]:
  return []
 k=np.round(core[:,0]/lane).astype(int)
 o=np.lexsort((np.where(k%2==0,core[:,1],-core[:,1]),k))
 path=core[o]
 sel=[path[0]]
 for q in path[1:]:
  if float(np.linalg.norm(q-sel[-1]))>=spacing:
   sel.append(q)
 w=sel[:cap]
 if len(w)>=2 and np.linalg.norm(w[-1]-pos)<np.linalg.norm(w[0]-pos):
  w=w[::-1]
 return [np.asarray(x,np.float64) for x in w]
class DroneFlightController:
 def __init__(self,*,model_path=None,providers=None):
  so=ort.SessionOptions()
  so.intra_op_num_threads=2
  so.inter_op_num_threads=1
  so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
  so.add_session_config_entry("session.intra_op.allow_spinning","0")
  self.session=ort.InferenceSession(str(model_path or _H/'policy.onnx'),so,
           providers=providers or ["CPUExecutionProvider"])
  self._ms=next(int(i.shape[0]) for i in self.session.get_inputs()
      if i.name=="memory_tensor")
  self._in={i.name for i in self.session.get_inputs()}
  self._cls=_Cls() if ON else None
  self._rp=_RgbPrimary() if (ON and RP_ON) else None
  self._r2=(_RgbPrimary(model='victim_rgb256.onnx',in_res=256,thr=R2_T,
                        hmax=R2_HMAX,rng_max=R2_RANGE_MAX,term_r=0.0,cap=R2_CAP)
            if (ON and R2_ON) else None)
  self._r2_on=False
  self._esc=_Escape() if (ON and ESC_ON) else None
  self.reset()
 def reset(self):
  self.memory_tensor=np.zeros(self._ms,np.float32)
  self.tick=0
  self._h={15: [],25: []}
  self._pad=None; self._z0=None
  self._on=False
  self._cfg=None
  self._w=None
  self._i=0
  self._adv=0
  self._cv=None
  self._cvt=0
  self._av=0; self._av_side=1.0; self._av_n=0
  self.route='king'
  self._sup=None; self._mass=None; self._w0=None
  self._rp_term=None
  if self._rp is not None:
   self._rp.reset()
  if self._r2 is not None:
   self._r2.reset()
  self._r2_on=False
  if self._esc is not None:
   self._esc.reset()
  if self._cls:
   self._cls.reset()
 def _vbuild(self,pos,clue,c):
  if self._sup is None:
   sup,w,dc=_ccore(pos[0:2]+clue,self._pad)
   self._sup=sup; self._w0=w; self._mass=w.copy(); self._dc=dc
  wps=_bous(self._core(0.9),c[3],c[4],c[5],pos[0:2])
  if not wps:
   self._mass=self._w0.copy()
   wps=_bous(self._core(0.9),c[3]*0.65,c[4]*0.7,c[5],pos[0:2])
  if not wps:
   wps=_bous(self._sup[self._dc<=30.0],c[3],c[4],c[5],pos[0:2])
  if not wps and LADDER:
   t=self._stype()
   b=float(_BOX[LB.index(t)]) if t in LB else 30.0
   dx=max(abs(float(self._pad[0]))-b,0.0); dy=max(abs(float(self._pad[1]))-b,0.0)
   rp=max(80.0,float(np.hypot(dx,dy))+30.0)+3.0
   for rpad,bx in ((rp,None),(1e9,[1.5*b]),(1e9,[1e3])):
    sup,w,dc=_ccore(pos[0:2]+clue,self._pad,rpad,bx)
    if int((w>0).sum())>=4:
     self._sup=sup; self._w0=w; self._mass=w.copy(); self._dc=dc; self._ladder=1
     wps=_bous(self._core(0.9),c[3],c[4],c[5],pos[0:2])
     if wps:
      break
  return wps
 def _core(self,frac):
  w=self._mass; t=float(w.sum())
  if t<=0:
   return np.zeros((0,2))
  o=np.argsort(-w)
  cw=np.cumsum(w[o])/t
  return self._sup[o[:int(np.searchsorted(cw,frac))+1]]
 def _swept(self,xy):
  if self._mass is not None:
   self._mass[np.linalg.norm(self._sup-np.asarray(xy,np.float64),axis=1)<=CVR]=0.0
 def _stype(self):
  c=self._cls
  if c.is_mountain:
   return 'mountain' if 'mountain' in TYPES else None
  if RL_ON and c.relatched and not (getattr(self,'_z0',None) is not None and self._z0<=WESC_Z):
   cands=[t for t in TYPES if t!='mountain' and (c.settled(t) or t in c.relatched)]
   if cands:
    avg=c.ps/max(c.n,1)
    return max(cands,key=lambda t: float(avg[LB.index(t)]))
  for t in TYPES:
   if t!='mountain' and c.settled(t):
    return t
  return None

 def _r2_type(self):
  c=self._cls
  if c is None or c.is_mountain or c.n<=0:
   return None
  avg=c.ps/max(c.n,1)
  i=int(np.argmax(avg)); lab=LB[i]
  if lab not in R2_TYPES:
   return None
  if not (c.settled(lab) or (RL_ON and lab in c.relatched)):
   return None
  return lab if float(avg[i])>=R2_BAR else None
 def _r2_ok(self):
  return bool(self._r2 is not None and self._r2_type() is not None)
 def _champ(self):
  if not CHAMP_ROUTE:
   return False
  c=self._cls
  if c is None or c.is_mountain:
   return False
  return bool(c.settled('village') or c.settled('warehouse'))
 def act(self,observation):
  self.tick += 1
  _rpx=False
  rq=False
  if self.route.endswith('+av'): self.route=self.route[:-3]
  st=np.asarray(observation["state"],np.float32).reshape(-1).copy()
  if ON and self._cls:
   pos=st[0:3].astype(np.float64)
   if self._pad is None:
    self._pad=pos[0:2].copy(); self._z0=float(pos[2])
    if MTN_Z0>0.0 and self._z0>=MTN_Z0:
     self._cls.is_mountain=True
   self._cls.update(self.tick,st,np.asarray(observation["depth"],np.float32))
   for pd,hs in self._h.items():
    if self.tick%pd==0:
     hs.append(pos[0:2].copy())
     if len(hs)>20:
      hs.pop(0)
   if not self._on:
    t=self._stype()
    c=CFG.get(t) if t else None
    if t=='mountain' and self._cls.mtn_ok() and not self._champ():
     c=MTN_STRONG
    if c and self.tick>=c[0]:
     hs=self._h[c[1]]
     if len(hs)>=20:
      hp=np.asarray(hs)
      if float(np.max(np.linalg.norm(hp-hp[0],axis=1)))<c[2]:
       if c[8] and self._cv!='ok':
        if self._cv!='block':
         rgb=observation.get('rgb') if hasattr(observation,'get') else None
         fr=np.asarray(rgb,np.float32) if rgb is not None else None
         if fr is not None and float(fr.max())>0.01:
          self._cv='block' if _warm(fr)>=VW else 'ok'
          if self._cv=='ok':
           self._on=True; self._cfg=c
         elif self._cvt<3:
          rq=True; self._cv='req'; self._cvt += 1
       else:
        self._on=True; self._cfg=c
   if self._on:
    c=self._cfg
    if self._w is None:
     if len(c)>9:
      self._w=self._vbuild(pos,st[-2:].astype(np.float64),c)
     else:
      self._w=_wps(pos[0:2]+st[-2:].astype(np.float64),self._pad,
        pos[0:2],c[3],c[4],c[5])
     self._i=0; self._adv=self.tick
    if len(self._w):
     self._i=min(self._i,len(self._w)-1)
     wp=self._w[self._i]
     if float(np.linalg.norm(wp-pos[0:2]))<c[7] or self.tick-self._adv>c[6]:
      if len(c)>9:
       self._swept(wp)
      nx=self._i+1
      if nx>=len(self._w):
       if len(c)>9:
        rb=self._vbuild(pos,st[-2:].astype(np.float64),c)
        if rb:
         self._w=rb
       nx=0
      self._i=nx
      self._adv=self.tick
      wp=self._w[self._i]
     off=wp-pos[0:2]
     bl=c[9] if len(c)>9 else 1.0
     if bl<1.0:
      off=bl*off+(1.0-bl)*st[-2:].astype(np.float64)
     st[-2]=np.float32(off[0]); st[-1]=np.float32(off[1])
     self.route='king+tour'
   self._rp_term=None
   _rpx=bool(RP_TYPES) and (self._stype() in RP_TYPES)
   if self._rp is not None and (_rpx or not self._champ()):
    try:
     if self._rp.clue0 is None:
      self._rp.clue0=pos[0:2]+np.asarray(observation['state'],np.float32).reshape(-1)[-2:].astype(np.float64)
     if self._cls.mtn_ok() or _rpx:
      self._rp.observe(self.tick,observation,pos,st[3:6].astype(np.float64))
      if self._rp.latch is not None:
       lk=self._rp.latch[0:2]-pos[0:2]; st[-2]=np.float32(lk[0]); st[-1]=np.float32(lk[1]); self.route='king+rgb'
       self._rp_term=self._rp.terminal(pos,float(st[162])*20.0 if st.size>162 else 20.0)
      self._rp.tick_end(self.tick,pos)
    except Exception:
     self._rp_term=None
   self._r2_on=False
   if self._r2 is not None:
    try:
     if not self._r2_ok():
      raise _R2Off
     self._r2_on=True
     if self._r2.clue0 is None:
      self._r2.clue0=pos[0:2]+np.asarray(observation['state'],np.float32).reshape(-1)[-2:].astype(np.float64)
     self._r2.observe(self.tick,observation,pos,st[3:6].astype(np.float64))
     if self._r2.latch is not None:
      lk=self._r2.latch[0:2]-pos[0:2]; st[-2]=np.float32(lk[0]); st[-1]=np.float32(lk[1]); self.route='king+r2'
     self._r2.tick_end(self.tick,pos)
    except _R2Off:
     pass
    except Exception:
     self._r2_on=False
  if AV_D>0.0 and self._cls and self.tick>=AV_T0 and self._stype() in AV_TYPES:
   dm=np.asarray(observation['depth'],np.float32).reshape(256,256)*29.5+0.5
   ctr=float(dm[128-AV_WIN:128+AV_WIN,128-AV_WIN:128+AV_WIN].min())
   if AV_TTC>0.0:
    _sp=float(np.linalg.norm(st[6:9])) if st.size>8 else 0.0
    ctr=ctr/max(_sp,0.25) if _sp>0.25 else 1e9
   if ctr<AV_D:
    if self._av==0:
     self._av_side=1.0 if float(dm[64:192,128:224].min())>=float(dm[64:192,32:128].min()) else -1.0
     self._av_n+=1
    self._av=AV_HOLD
   elif ctr>AV_D+AV_REL and self._av>0:
    self._av-=1
   if self._av>0:
    ang=-self._av_side*np.radians(AV_ROT)
    cx,cy=float(st[-2]),float(st[-1]); ca,sa=float(np.cos(ang)),float(np.sin(ang))
    st[-2]=np.float32(ca*cx-sa*cy); st[-1]=np.float32(sa*cx+ca*cy)
    if not self.route.endswith('+av'): self.route=self.route+'+av'
  feed={"depth": np.asarray(observation["depth"],np.float32).reshape(256,256,1),
    "state": st,"memory_tensor": self.memory_tensor}
  if "rgb" in self._in:
   rgb=observation.get("rgb") if hasattr(observation,"get") else None
   feed["rgb"]=(np.asarray(rgb,np.float32).reshape(256,256,3)
      if rgb is not None else np.zeros((256,256,3),np.float32))
  a,m=self.session.run(["action","memory_tensor_out"],feed)
  self.memory_tensor=np.asarray(m,np.float32).reshape(self._ms)
  out=np.asarray(a,np.float32).reshape(-1)
  if rq and out.shape[0]>=6:
   out=out.copy(); out[5]=1.0
  if self._esc is not None and out.shape[0]>=4 and (self._esc.mode=='ceil' or not self._champ()):
   try:
    _p=st[0:3].astype(np.float64); _agl=float(st[162])*20.0 if st.size>162 else 20.0
    if self.tick==1:
     if not (WESC_ON and self._esc.ceiling(observation['depth'],_p,float(st[5]))):
      self._esc.decide(observation['depth'],_p,float(st[5]))
    _eok=True
    if ESC_TERR and self._esc.mode!='ceil':
     _c2=self._cls
     _eok=bool(_c2 is not None and (_c2.is_mountain or _c2.settled('forest')))
    if _eok and self._esc.phase in ('scan','creep','climb'):
     _e=self._esc.act(observation['depth'],_p,_agl,math.degrees(float(st[4])),float(st[5]))
     if _e is not None:
      out=out.copy(); out[0:3]=_e[0]; out[3]=np.float32(_e[1]); out[4]=np.float32(0.0 if _e[2] is None else _e[2])
      if self._esc.mode=='ceil': self.route='king+wesc'
   except Exception:
    pass
  if self._rp is not None and out.shape[0]>=6 and self._cls is not None and (_rpx or (self._cls.mtn_ok() and not self._champ())):
   out=out.copy()
   if self._rp_term is not None:
    out[0:3]=self._rp_term[0]; out[3]=np.float32(self._rp_term[1]); out[4]=np.float32(0.0)
   if self._rp.want_rgb(self.tick):
    out[5]=np.float32(1.0)
  if self._r2 is not None and out.shape[0]>=6 and self._r2_on:
   if self._r2.want_rgb(self.tick):
    out=out.copy(); out[5]=np.float32(1.0)
  if VIL_AGL>0.0 and out.shape[0]>=4 and self._cls is not None:
   try:
    if self._stype()=='village' and not self._cls.is_mountain:
     _lat=(self._r2.latch is not None) if self._r2 is not None else False
     _rw=np.asarray(observation['state'],np.float32).reshape(-1)[-2:].astype(np.float64)
     _cd=float(np.linalg.norm(_rw))
     _ag=float(st[162])*20.0 if st.size>162 else 99.0
     if (not _lat) and _cd>VIL_REL and _ag<VIL_AGL-VIL_DB:
      _v=out[0:3].astype(np.float64); _n=float(np.linalg.norm(_v))
      if _n>1e-6:
       _v=_v/_n
       if _v[2]<VIL_KZ:
        _v[2]=VIL_KZ
        _n2=float(np.linalg.norm(_v))
        if _n2>1e-6:
         out=out.copy(); out[0:3]=(_v/_n2).astype(np.float32)
         self.route='king+vilz'
   except Exception:
    pass
  return out

