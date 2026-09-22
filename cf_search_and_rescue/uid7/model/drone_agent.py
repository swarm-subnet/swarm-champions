import os
from pathlib import Path
import math
import numpy as np
import onnxruntime as ort
_H=Path(__file__).resolve().parent
ON=os.environ.get('KT_ON','1')=='1'
AV_D=4.0
AV_REL=0.7
AV_ROT=45.0
AV_HOLD=15
AV_T0=100
AV_TYPES=('forest','city')
FG_ON=os.environ.get('KT_FG','1')=='1'
SS_ON=os.environ.get('KT_SS','1')=='1'; SS_Z=float(os.environ.get('KT_SS_Z','0.3'))
SS_LIFT=float(os.environ.get('KT_SS_LIFT','0.15')); SS_VZ=float(os.environ.get('KT_SS_VZ','0.2'))
SS_FRAC=float(os.environ.get('KT_SS_FRAC','0.5')); SS_MIN_HEAD=int(os.environ.get('KT_SS_MIN_HEAD','4'))
SS_HOR=float(os.environ.get('KT_SS_HOR','1.5')); SS_OUT=float(os.environ.get('KT_SS_OUT','1.3'))
SS_OUT_V=float(os.environ.get('KT_SS_OUT_V','0.3')); SS_MAXT=int(os.environ.get('KT_SS_MAXT','450'))
SS_F0=float(os.environ.get('KT_SS_F0','0.3')); SS_SIDE=float(os.environ.get('KT_SS_SIDE','90'))
SS_TOP=float(os.environ.get('KT_SS_TOP','0.55')); SS_HMAX=float(os.environ.get('KT_SS_HMAX','1.0'))
MX_ON=os.environ.get('KT_MX','1')=='1'; MX_AT=float(os.environ.get('KT_MX_AT','28')); MX_EXT=int(os.environ.get('KT_MX_EXT','70'))
MX_TYPES=tuple(x for x in os.environ.get('KT_MX_TYPES','warehouse,city').split(',') if x)
P3_ON=os.environ.get('KT_P3','1')=='1'; P3_T=int(os.environ.get('KT_P3_T','1000')); P3_BAN_R=float(os.environ.get('KT_P3_BAN_R','3.0'))
FG_PH=tuple(int(x) for x in os.environ.get('KT_FG_PH','3').split(','))
FG_D=float(os.environ.get('KT_FG_D','3.5')); FG_D2=float(os.environ.get('KT_FG_D2','3.0'))
FG_W=float(os.environ.get('KT_FG_W','0.45')); FG_ROWS=int(os.environ.get('KT_FG_ROWS','18'))
FG_MINH=float(os.environ.get('KT_FG_MINH','1.5')); FG_HOLD=int(os.environ.get('KT_FG_HOLD','20'))
FG_V=float(os.environ.get('KT_FG_V','1.5')); FG_DN=float(os.environ.get('KT_FG_DN','0.02')); FG_TURN=float(os.environ.get('KT_FG_TURN','5'))
FG_MAXB=float(os.environ.get('KT_FG_MAXB','40'))
FG_ALAT=float(os.environ.get('KT_FG_ALAT','3.0')); FG_TILT=float(os.environ.get('KT_FG_TILT','25'))
STALL_N=int(os.environ.get('KT_STALL_N','150'))        # stalled ticks before the lookahead fires (0 = off)
STALL_V=float(os.environ.get('KT_STALL_V','0.8'))   # horizontal speed under which a tick counts as stalled
STALL_LOOK=float(os.environ.get('KT_STALL_LOOK','15.0'))  # push the clue to the first waypoint this far ahead
STALL_HOLD=int(os.environ.get('KT_STALL_HOLD','100'))     # ticks the push stays engaged
STALL_TYPES=tuple(t for t in os.environ.get('KT_STALL_TYPES','mountain').split(',') if t)
CFG={
 'forest':   (350,15,1000000000.0,8.0,5.0,40,120,8.0,0,0.75),
 'city':     (450,25,10.0,6.0,5.0,40,250,4.5,1),
 'mountain': (450, 15, 1000000000.0, 12.0, 8.0, 25, 120, 8.0, 0, 0.75),
 'village':  (450,15,10.0,8.0,5.0,40,120,8.0,0,0.75),
 'warehouse':(450,15,10.0,6.0,4.0,40,120,6.0,0,0.75),
}
MTN_STRONG=(450,15,1000000000.0,12.0,8.0,25,120,8.0,0,0.75)
# kt_v267: the tour lane/step were sized for the policy's own ~9-15 m detector; uid255's depth heads see 25-30 m,
# so a wider lane covers more ground per pass. KT_LANE_<type> / KT_STEP_<type> scale c[3] / c[4] for that type.
def _cfg_env(t,c):
 l=os.environ.get('KT_LANE_'+t.upper()); st=os.environ.get('KT_STEP_'+t.upper())
 if l is None and st is None:
  return c
 c=list(c)
 if l is not None: c[3]=float(l)
 if st is not None: c[4]=float(st)
 return tuple(c)
CFG={t:_cfg_env(t,c) for t,c in CFG.items()}
MTN_STRONG=_cfg_env('mountain',MTN_STRONG)
MTN_THR=0.5
FF_ON=os.environ.get('KT_FF','1')=='1'
FF_TYPES=tuple(t for t in os.environ.get('KT_FF_TYPES','forest').split(',') if t)
FF_FRACS=(0.9,0.8,0.7,0.6,0.5,0.4,0.3)
RL_ON=os.environ.get('KT_RL','1')=='1'; RL_N=int(os.environ.get('KT_RL_N','40')); RL_BAR=float(os.environ.get('KT_RL_BAR','0.55'))
RL_TYPES=tuple(t for t in os.environ.get('KT_RL_TYPES','city,forest').split(',') if t)
LADDER=os.environ.get('KT_LADDER','1')=='1'
TYPES=tuple(t for t in os.environ.get('KT_TYPES','forest,village,city,mountain,warehouse').split(',') if t)
LB=('city','open','mountain','village','warehouse','forest')
SIM_DT=0.02
VW=float(os.environ.get('KT_VETO_WARM','0.02'))
RP_ON=os.environ.get('KT_RP','1')=='1'
RP_T=float(os.environ.get('KT_RP_T','0.713'))
RP_HMAX=float(os.environ.get('KT_RP_HMAX','1.0'))
RP_T0=150; RP_EVERY=75; RP_CAP=40; RP_CHASE_MIN=12; RP_CHASE_TICKS=36
RP_RANGE_MAX=20.0; RP_IMG_MAX=29.0; RP_CLUE_R=55.0; RP_LATCH_N=2; RP_LATCH_TOL=5.0; RP_EMA=0.3
RP_TERM_R=6.0; RP_HOVER_UP=3.2; RP_DUD_TICKS=300; RP_AGL_MIN=2.0; RP_RING=1.6; RP_TERM_MAX=1200
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
# --- obstacle-memory barrier (kt_v260, ported from cf_swarm_sar uid224 _ObstacleMemory) ---
OMB_ON=os.environ.get('KT_OMB','1')=='1'
OMB_TYPES=tuple(x for x in os.environ.get('KT_OMB_TYPES','forest,city').split(',') if x)
OMB_R_SAFE=float(os.environ.get('KT_OMB_RSAFE','1.0'))
OMB_R_SAFE_F=float(os.environ.get('KT_OMB_RSAFE_FOREST','0.75'))
OMB_K=float(os.environ.get('KT_OMB_K','1.5'))
OMB_SEE_M=float(os.environ.get('KT_OMB_SEE','6.0'))
OMB_MEM_S=float(os.environ.get('KT_OMB_MEM','10.0'))
OMB_INFL_M=float(os.environ.get('KT_OMB_INFL','3.5'))
OMB_CAP_NEAR=float(os.environ.get('KT_OMB_CAP_NEAR','2.0'))
OMB_CAP_MIN=float(os.environ.get('KT_OMB_CAP_MIN','0.3'))
OMB_KEEP_M=float(os.environ.get('KT_OMB_KEEP','15.0'))
OMB_BLIND_V=float(os.environ.get('KT_OMB_BLIND_V','1.0'))
OMB_BLIND_DEG=float(os.environ.get('KT_OMB_BLIND_DEG','120.0'))
OMB_BLIND_TYPES=tuple(x for x in os.environ.get('KT_OMB_BLIND_TYPES','forest').split(',') if x)
OMB_SLIDE_V=float(os.environ.get('KT_OMB_SLIDE_V','1.2'))
OMB_SLIDE_HOLD=50
OMB_STUCK_TICKS=int(os.environ.get('KT_OMB_STUCK','150'))
OMB_STUCK_FREE=100
OMB_GROUND=float(os.environ.get('KT_OMB_GROUND','-1'))   # >=0: drop voxels below ground_z+this (off by default)
OMB_VOX=0.4
OMB_BLOCK=8
OMB_SPEED=3.0
OMB_HZ=50.0
_OMB_G=None
def _omb_grid():
 global _OMB_G
 if _OMB_G is None:
  px=(np.arange(256,dtype=np.float64)+0.5)/256*2.0-1.0
  u=np.ascontiguousarray(np.broadcast_to(px[None,:],(256,256)))
  v=np.ascontiguousarray(np.broadcast_to(-px[:,None],(256,256)))
  _OMB_G=(256//OMB_BLOCK,u,v)
 return _OMB_G
def _omb_vel(a):
 d=np.asarray(a[0:3],np.float64); n=float(np.linalg.norm(d))
 return (d/n)*abs(float(a[3]))*OMB_SPEED if n>1e-9 else np.zeros(3,np.float64)
def _omb_store(a,v):
 s=float(np.linalg.norm(v))
 if s>1e-9:
  a[0]=v[0]/s; a[1]=v[1]/s; a[2]=v[2]/s; a[3]=min(s/OMB_SPEED,1.0)
 else:
  a[0]=0.0; a[1]=0.0; a[2]=0.0; a[3]=0.0
class _OMB:
 def __init__(self):
  self.reset()
 def reset(self):
  self.keys=np.zeros(0,np.int64); self.pts=np.zeros((0,3),np.float64); self.seen_t=np.zeros(0,np.int64)
  self.tick=0
  self.stats={'ticks':0,'limited':0,'capped':0,'slide':0,'stuck':0,'blind':0,'min_r':99.0}
  self.slide_side=0.0; self.slide_until=-1; self.prog_pos=None; self.prog_tick=0; self.free_until=-1
 def observe(self,depth,pos,rpy,ground_z=None):
  self.tick+=1
  nb,U,V=_omb_grid()
  dm=np.asarray(depth,np.float32).reshape(256,256)*(_DMAX-_DMIN)+_DMIN
  blk=dm.reshape(nb,OMB_BLOCK,nb,OMB_BLOCK).transpose(0,2,1,3).reshape(nb,nb,OMB_BLOCK*OMB_BLOCK)
  arg=blk.argmin(axis=2)
  mn=np.take_along_axis(blk,arg[:,:,None],axis=2)[:,:,0]
  sel=mn<=OMB_SEE_M
  if sel.any():
   by,bx=np.nonzero(sel)
   a=arg[by,bx]
   py=by*OMB_BLOCK+a//OMB_BLOCK; px=bx*OMB_BLOCK+a%OMB_BLOCK
   cz=mn[by,bx].astype(np.float64)
   u=U[py,px]; v=V[py,px]
   fwd,up,right=_axes(rpy)
   cam=np.asarray(pos,np.float64)+fwd*_CAMF+up*_CAMU
   pts=cam[None,:]+right[None,:]*(u*_HT*cz)[:,None]+up[None,:]*(v*_HT*cz)[:,None]+fwd[None,:]*cz[:,None]
   if OMB_GROUND>=0.0 and ground_z is not None:
    pts=pts[pts[:,2]>ground_z+OMB_GROUND]
   if len(pts):
    ijk=np.floor(pts/OMB_VOX).astype(np.int64)+500000
    keys=(ijk[:,0]*1000003+ijk[:,1])*1000003+ijk[:,2]
    allk=np.concatenate([keys,self.keys]); allp=np.concatenate([pts,self.pts])
    allt=np.concatenate([np.full(keys.size,self.tick,np.int64),self.seen_t])
    _,first=np.unique(allk,return_index=True)
    self.keys,self.pts,self.seen_t=allk[first],allp[first],allt[first]
  if self.tick%25==0 and self.keys.size:
   keep=(self.tick-self.seen_t)<=int(OMB_MEM_S*OMB_HZ)
   keep&=np.hypot(self.pts[:,0]-float(pos[0]),self.pts[:,1]-float(pos[1]))<=OMB_KEEP_M
   if not keep.all():
    self.keys,self.pts,self.seen_t=self.keys[keep],self.pts[keep],self.seen_t[keep]
 def constrain(self,a,pos,rpy,stype):
  changed=False
  if OMB_BLIND_V>0.0 and stype in OMB_BLIND_TYPES:
   v0=_omb_vel(a); sh=math.hypot(v0[0],v0[1])
   if sh>OMB_BLIND_V:
    head=math.atan2(v0[1],v0[0])
    mis=abs((head-float(rpy[2])+math.pi)%(2.0*math.pi)-math.pi)
    if mis>math.radians(OMB_BLIND_DEG):
     _omb_store(a,v0*(OMB_BLIND_V/sh)); self.stats['blind']+=1; changed=True
  if self.keys.size==0:
   return changed
  P=self.pts; x=np.asarray(pos,np.float64)
  D=x[None,:]-P; R=np.sqrt((D*D).sum(axis=1))
  near=R<=OMB_INFL_M
  if not near.any():
   return changed
  D=D[near]; R=R[near]; o=np.argsort(R); D=D[o]; R=R[o]
  v_cmd=_omb_vel(a); v=v_cmd.copy()
  r_safe=OMB_R_SAFE_F if stype=='forest' else OMB_R_SAFE
  sp_cmd=float(math.hypot(v_cmd[0],v_cmd[1])); x2=(float(x[0]),float(x[1]))
  if self.prog_pos is None or math.hypot(x2[0]-self.prog_pos[0],x2[1]-self.prog_pos[1])>1.0 or sp_cmd<0.5:
   self.prog_pos=x2; self.prog_tick=self.tick
  elif self.tick-self.prog_tick>OMB_STUCK_TICKS and self.tick>self.free_until:
   self.free_until=self.tick+OMB_STUCK_FREE; self.prog_tick=self.tick; self.stats['stuck']+=1
  if self.tick<=self.free_until:
   r_safe=0.35
  def _project(v):
   for _ in range(2):
    for i in range(min(len(R),24)):
     r=R[i]; n=D[i]/max(r,1e-6); lim=OMB_K*(r-r_safe); vt=-float(v@n)
     if vt>lim:
      v=v+(vt-lim)*n
   return v
  v=_project(v)
  if not np.allclose(v,v_cmd,atol=1e-6):
   changed=True
  sp_h=float(math.hypot(v[0],v[1]))
  if sp_cmd>0.3 and sp_h<0.35*sp_cmd and self.tick>self.free_until:
   tw=-D[0][:2]; tn=float(math.hypot(tw[0],tw[1]))
   if tn>1e-6:
    tw=tw/tn; tL=np.array([-tw[1],tw[0]]); tR=-tL
    if self.tick<=self.slide_until and self.slide_side!=0.0:
     side=self.slide_side
    else:
     dot=float(v_cmd[0]*tL[0]+v_cmd[1]*tL[1])/max(sp_cmd,1e-6)
     if abs(dot)>0.2:
      side=1.0 if dot>0 else -1.0
     else:
      pl=x[:2]+tL*2.0; pr=x[:2]+tR*2.0
      dl=float(np.min(np.hypot(P[:,0]-pl[0],P[:,1]-pl[1]))); dr=float(np.min(np.hypot(P[:,0]-pr[0],P[:,1]-pr[1])))
      side=1.0 if dl>=dr else -1.0
     self.slide_side=side; self.slide_until=self.tick+OMB_SLIDE_HOLD
    t=tL if side>0 else tR; vs=min(sp_cmd,OMB_SLIDE_V)
    v=_project(np.array([t[0]*vs,t[1]*vs,v_cmd[2]],np.float64)); self.stats['slide']+=1; changed=True
  rmin=float(R[0])
  if rmin<self.stats['min_r']:
   self.stats['min_r']=rmin
  sn=float(np.linalg.norm(v)); cap=OMB_SPEED
  if rmin<OMB_CAP_NEAR:
   cap=OMB_SPEED*max(OMB_CAP_MIN,min(1.0,(rmin-r_safe*0.5)/(OMB_CAP_NEAR-r_safe*0.5)))
  if sn>cap:
   v=v*(cap/sn); changed=True; self.stats['capped']+=1
  if changed:
   self.stats['limited']+=1; _omb_store(a,v)
  return changed
CHAMP_ROUTE=os.environ.get('KT_CHAMP_ROUTE','0' if os.environ.get('KT_PLAN','1')=='1' else '1')=='1'  # village/warehouse -> UID_169's process end to end (off when the planner runs there)
ESC_TERR=os.environ.get('KT_ESC_TERR','1')=='1'        # escape acts ONLY on positively-latched mountain/forest
ESC_ON=os.environ.get('KT_ESC','1')=='1'; ESC_TRAP_M=float(os.environ.get('KT_ESC_TRAP','1.0')); ESC_FREE_M=3.0; ESC_DIST=float(os.environ.get('KT_ESC_DIST','2.0'))
# Low-shelf spawn escape. The sim's start-pad check ignores bodies under 1 m tall and probes only 0.45 m up, so the
# drone can spawn under a rack's lowest shelf; the policy's blind vertical climb hits it within ~16 ticks.
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
  # ceiling mode: an opening must also be clear overhead in the same columns (not just along the shelf)
  d=np.asarray(depth,np.float32).reshape(256,256)*(_DMAX-_DMIN)+_DMIN
  bands=[(float(d[112:128,c0:c0+52].min()),float(d[0:64,c0:c0+52].min())) for c0 in (0,51,102,153,204)]
  ok=[i for i in range(5) if bands[i][0]>=ESC_FREE_M and bands[i][1]>=WESC_TOP]
  if not ok:
   return None
  k=min(ok,key=lambda i:(abs(i-2),-bands[i][0]))
  return float(yaw)-math.radians((k-2)*18.0)
 def _side(self,d,yaw):
  # Ground truth (ovh_probe): the trap is a ~0.4-0.8 m rack beam 0.64 m up, directly overhead and running along the view,
  # so its underside fills the centre columns of the upper image. Step sideways toward the nearer beam edge, taken as the
  # lateral offset y=u*depth of the outermost contiguous low-ceiling pixel (5/5 correct vs ray ground truth). No yaw: on the
  # floor the drone turns ~25 deg per second, which is what ran the scan mode out of time.
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
SPLIT_ON=os.environ.get('KT_SPLIT','0')=='1'      # per-map victim checkers (victim_rgb_<map>.onnx next to this file); missing file -> the shared net
RP_SPLIT={'mountain':'victim_rgb_mountain.onnx'} if SPLIT_ON else {}
R2_SPLIT={t:'victim_rgb_%s.onnx'%t for t in ('forest','village','warehouse')} if SPLIT_ON else {}
R2_ON=os.environ.get('KT_R2','1')=='1'
R2_TYPES=tuple(t for t in os.environ.get('KT_R2_TYPES','forest,warehouse,village').split(',') if t in LB)
R2_T=float(os.environ.get('KT_R2_T','0.65'))
R2_BAR=float(os.environ.get('KT_R2_BAR','0.5'))
R2_HMAX=float(os.environ.get('KT_R2_HMAX','inf'))
R2_RANGE_MAX=float(os.environ.get('KT_R2_RANGE_MAX','25.0'))
R2_CAP=int(os.environ.get('KT_R2_CAP',str(RP_CAP-3)))
R2_TERM_R=float(os.environ.get('KT_R2_TERM_R','6.0'))
R2_TERM_TYPES=tuple(os.environ.get('KT_R2_TERM_TYPES','forest,village,warehouse').split(','))
R2_FACE=os.environ.get('KT_R2_FACE','1')=='1'; R2_FACE_TYPES=tuple(os.environ.get('KT_R2_FACE_TYPES','forest').split(','))
R2_FACE_CLR=float(os.environ.get('KT_R2_FACE_CLR','2.0'))
R2_TERM_P1=os.environ.get('KT_R2_TERM_P1','1')=='1'; R2_TERM_WAIT=int(os.environ.get('KT_R2_TERM_WAIT','100'))
MTN_Z0=float(os.environ.get('KT_MTN_Z0','13.0'))
class _R2Off(Exception):
 pass
def _ss_frac(depth):
 d=np.asarray(depth,np.float32).reshape(256,256)*(_DMAX-_DMIN)+_DMIN
 v=(1.0-(np.arange(85)+0.5)/128.0)[:,None]
 top=d[0:85,64:192]; h=top*v
 frac=float(((h>0.15)&(h<1.0)&(top<1.2)).mean())
 hor=float(np.median(d[112:136,96:160]))
 return frac,hor
def _ss_top(depth):
 d=np.asarray(depth,np.float32).reshape(256,256)[0:40,64:192]*(_DMAX-_DMIN)+_DMIN
 return float(np.median(d))
def _wrap(a):
 return (a+math.pi)%(2*math.pi)-math.pi

# ---------------- depth expert trigger (KT_DEPTH=1): victim_depth_<map>.onnx on the always-on depth frame every DT_EVERY ticks ----------------
# A hit pulls the next RGB request forward (the existing chase logic), so the RGB budget is spent confirming what depth saw rather than on blind pacing.
DT_ON=os.environ.get('KT_DEPTH','0')=='1'; DT_EVERY=int(os.environ.get('KT_DT_EVERY','4')); DT_THR=float(os.environ.get('KT_DT_THR','0.7')); DT_T0=int(os.environ.get('KT_DT_T0','60'))
DT_RANGE=float(os.environ.get('KT_DT_RANGE','28.0')); DT_HMAX=float(os.environ.get('KT_DT_HMAX','2.5'))
_G_HR,_G_G,_G_CELL,_G_WIN,HR,G,CELL,WIN=128,128,0.8,7,128,128,0.8,7
def _axes_np(roll, pitch, yaw):
    """fwd, up, right world unit vectors of the camera for rpy (rad); identical to the agent's camera_axes."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    fwd = R[:, 0]; y = R[:, 1]; up = R[:, 2]; right = -y
    return fwd, up, right
_uv = None
def geo_half_np(d, roll, pitch, yaw):
    """d: (256,256) normalised depth; angles in rad. Returns (9,128,128) float32 computed from every 2nd pixel: ray_sin, ray_cos, z_rel, log_range, person_px, normal_z, height, pitch plane, roll plane."""
    u1 = (np.arange(HR, dtype=np.float32) * 2.0 - 127.5) / 127.5; v1 = -u1
    fwd, up, right = _axes_np(roll, pitch, yaw)
    fwd, up, right = (a.astype(np.float32) for a in (fwd, up, right))
    d = np.clip(d.astype(np.float32), 0.0, 1.0); valid_full = ((d > 0.0) & (d < 1.0)).astype(np.float32)
    dh = d[::2, ::2]; valid = valid_full[::2, ::2]
    cz = dh * 29.5 + 0.5
    cx = fwd[None, :] + right[None, :] * u1[:, None]; ry_ = up[None, :] * v1[:, None]
    rx = cx[None, :, 0] + ry_[:, None, 0]; ry = cx[None, :, 1] + ry_[:, None, 1]; rz = cx[None, :, 2] + ry_[:, None, 2]
    nrm = np.sqrt(rx * rx + ry * ry + rz * rz)
    ray_sin = rz / nrm; ray_cos = np.sqrt(np.maximum(1.0 - ray_sin * ray_sin, 0.0))
    px, py, pz = cz * rx, cz * ry, cz * rz
    z_rel = pz / 20.0; log_range = np.log1p(cz * nrm) / math.log1p(30.0)
    person_px = np.minimum(1.0, (1.7 * 128.0 / np.maximum(cz, 0.5)) / 64.0)
    ax, ay, az = px[1:-1, 2:] - px[1:-1, :-2], py[1:-1, 2:] - py[1:-1, :-2], pz[1:-1, 2:] - pz[1:-1, :-2]
    bx, by, bz = px[2:, 1:-1] - px[:-2, 1:-1], py[2:, 1:-1] - py[:-2, 1:-1], pz[2:, 1:-1] - pz[:-2, 1:-1]
    nx = ay * bz - az * by; ny = az * bx - ax * bz; nzz = ax * by - ay * bx
    normal_z = np.zeros((HR, HR), np.float32); normal_z[1:-1, 1:-1] = np.abs(nzz) / np.sqrt(np.maximum(nx * nx + ny * ny + nzz * nzz, 1e-12))
    half = G * CELL / 2.0
    ci = np.floor((px + half) / CELL).astype(np.int32); cj = np.floor((py + half) / CELL).astype(np.int32)
    inside = (ci >= 0) & (ci < G) & (cj >= 0) & (cj < G) & (valid > 0)
    flat = np.where(inside, cj * G + ci, 0).ravel()
    grid = np.full(G * G, 1e6, np.float32); np.minimum.at(grid, flat, np.where(inside, pz, np.float32(1e6)).ravel().astype(np.float32))
    gp = np.pad(grid.reshape(G, G), WIN // 2, mode='constant', constant_values=1e6)
    from numpy.lib.stride_tricks import sliding_window_view
    pooled = sliding_window_view(sliding_window_view(gp, WIN, axis=0).min(-1), WIN, axis=1).min(-1)
    ground = pooled.reshape(-1)[flat].reshape(HR, HR); known = (ground < 5e5) & inside
    height = np.where(known, (np.clip(pz - ground, -2.0, 8.0) + 2.0) / 10.0, 0.0).astype(np.float32)
    out = np.empty((9, HR, HR), np.float32)
    out[0] = ray_sin * valid; out[1] = ray_cos * valid; out[2] = z_rel * valid; out[3] = log_range * valid; out[4] = person_px * valid; out[5] = normal_z * valid; out[6] = height * valid
    out[7] = pitch * 57.29578 / 20.0; out[8] = roll * 57.29578 / 20.0
    return out


_G4=[0,2,5,6]
class _DepthTrigger:
 def __init__(self):
  self.sess=None; self.loaded_for=None; self.mtype=None; self.n_eval=0; self.n_hit=0; self.last=None; self.last_hit_t=-10**9
 def _load(self,mtype):
  self.loaded_for=mtype; self.sess=None
  f=_H/('victim_depth_%s.onnx'%mtype)
  if not f.exists(): return
  try:
   so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
   self.sess=ort.InferenceSession(str(f),so,providers=['CPUExecutionProvider'])
  except Exception:
   self.sess=None
 def step(self,tick,obs,pos,rpy,mtype):
  """Returns a hit (p,u,v,cz,world_xyz) or None."""
  if not mtype or tick<DT_T0 or tick%DT_EVERY: return None
  if mtype!=self.loaded_for: self._load(mtype)
  if self.sess is None: return None
  d=np.asarray(obs['depth'],np.float32).reshape(256,256)
  g=geo_half_np(d,float(rpy[0]),float(rpy[1]),float(rpy[2]))[_G4]
  o=np.asarray(self.sess.run(None,{'x':d[None,None],'g':np.ascontiguousarray(g[None],np.float32)})[0],np.float32).reshape(5); self.n_eval+=1
  p=_sig(o[0]); cz=math.exp(min(float(o[3]),6.0)); self.last=(p,float(o[1]),float(o[2]),cz,float(o[4]))
  if p<DT_THR or cz>DT_RANGE or float(o[4])>DT_HMAX: return None
  q=_mworld(float(o[1]),float(o[2]),cz,pos,rpy)
  if not np.all(np.isfinite(q)): return None
  self.n_hit+=1; self.last_hit_t=tick
  return (p,float(o[1]),float(o[2]),cz,q)


# ---------------- coverage planner (KT_PLAN=1): next-best-view search instead of the fixed boustrophedon tour ----------------
# The champion's tour marks 9 m around every reached waypoint as covered no matter where the camera looked. The planner keeps the same
# prior support (_ccore) but removes probability mass only where the camera has actually seen the ground: inside the horizontal FOV, within
# PLAN_REFF, and not occluded according to the depth image; it then flies to the point whose neighbourhood holds the most unseen mass per
# second of travel. Checker hits (RGB fixes) add mass around their world estimate so suspicious spots get a second look.
PLAN_ON=os.environ.get('KT_PLAN','1')=='1'; PLAN_TYPES=tuple(t for t in os.environ.get('KT_PLAN_TYPES','forest,village,warehouse').split(',') if t)
PLAN_REFF=float(os.environ.get('KT_PLAN_REFF','18.0')); PLAN_RGAIN=float(os.environ.get('KT_PLAN_RGAIN','11.0')); PLAN_EVERY=int(os.environ.get('KT_PLAN_EVERY','25'))
PLAN_ARR=float(os.environ.get('KT_PLAN_ARR','6.0')); PLAN_MIN=float(os.environ.get('KT_PLAN_MIN','8.0')); PLAN_TURN=float(os.environ.get('KT_PLAN_TURN','0.5')); PLAN_HINT=float(os.environ.get('KT_PLAN_HINT','3.0'))
PLAN_STALL_N=int(os.environ.get('KT_PLAN_STALL_N','100')); PLAN_STALL_V=float(os.environ.get('KT_PLAN_STALL_V','0.6'))
class _Planner:
 def __init__(self,sup,w0):
  self.sup=np.asarray(sup,np.float64); self.w0=np.asarray(w0,np.float64); self.mass=self.w0.copy(); self.wp=None; self.t_wp=-10**9
  self.n_obs=0; self.n_plan=0; self.n_hint=0; self.resets=0; self.stall=0; self.n_stall=0; self.seen_hits={}
  d=self.sup[:,None,:]-self.sup[None,:,:]; self.near=((d[:,:,0]**2+d[:,:,1]**2)<=PLAN_RGAIN**2).astype(np.float64)
 def observe(self,pos,rpy,depth,agl):
  """Remove mass where the camera has really seen the ground this tick (2D support points projected into the depth image)."""
  d=self.sup-pos[0:2]; r=np.hypot(d[:,0],d[:,1]); yaw=float(rpy[2]); pitch=float(rpy[1])
  b=np.arctan2(d[:,1],d[:,0])-yaw; b=(b+math.pi)%(2*math.pi)-math.pi
  cand=(np.abs(b)<=math.radians(44.0))&(r<=PLAN_REFF)&(r>=1.5)&(self.mass>0)
  if not cand.any(): return
  i=np.where(cand)[0]; bi=b[i]; ri=r[i]; agl=max(0.5,float(agl))
  u=-np.tan(bi); dep=np.arctan2(agl,ri); v=np.tan(-dep-pitch)
  ok=(np.abs(u)<1.0)&(np.abs(v)<1.0)
  if not ok.any(): return
  i=i[ok]; u=u[ok]; v=v[ok]; ri=ri[ok]
  px=np.clip(((u+1.0)*128.0).astype(int),1,254); py=np.clip(((1.0-v)*128.0).astype(int),1,254)
  dm=np.asarray(depth,np.float32).reshape(256,256)
  dv=np.maximum.reduce([dm[py-1,px],dm[py+1,px],dm[py,px-1],dm[py,px+1],dm[py,px]])*(_DMAX-_DMIN)+_DMIN     # thin occluders (branches) must not hide a whole cell
  rs=np.sqrt(ri*ri+agl*agl); seen=dv>=rs-2.0
  q=np.where(ri<=10.0,0.85,np.where(ri<=14.0,0.6,0.3))
  self.mass[i[seen]]*=(1.0-q[seen]); self.n_obs+=1
 def hint(self,xy,amp):
  d=self.sup-np.asarray(xy,np.float64)[0:2]; self.mass+=amp*float(self.w0.max())*np.exp(-(d[:,0]**2+d[:,1]**2)/(2*4.0**2)); self.n_hint+=1
 def next(self,pos,yaw,tick,force=False):
  if not force and self.wp is not None and tick-self.t_wp<PLAN_EVERY and float(np.linalg.norm(self.wp-pos[0:2]))>PLAN_ARR: return self.wp
  if float(self.mass.sum())<0.05*float(self.w0.sum()):
   self.mass=0.5*self.w0.copy(); self.resets+=1
  gain=self.near@self.mass
  off=self.sup-pos[0:2]; dist=np.hypot(off[:,0],off[:,1]); ang=np.arctan2(off[:,1],off[:,0])-yaw; ang=np.abs((ang+math.pi)%(2*math.pi)-math.pi)
  cost=dist/3.0+1.0+PLAN_TURN*ang
  score=gain/cost; score[dist<PLAN_MIN]=-1.0          # the camera already covers the near field: always aim beyond it
  self.wp=self.sup[int(np.argmax(score))].copy(); self.t_wp=tick; self.n_plan+=1
  return self.wp

class _RgbPrimary:
 def __init__(self,model='victim_rgb_m.onnx',in_res=128,thr=None,hmax=None,rng_max=None,term_r=None,cap=None,model_by_type=None):
  self.sess=None; self._tried=False
  self.model=model; self.in_res=int(in_res)
  self.model_by_type=dict(model_by_type or {}); self.mtype=None; self._loaded_for=None; self.model_used=None
  self.thr=thr if thr is None else float(thr)
  self.hmax=RP_HMAX if hmax is None else float(hmax)
  self.rng_max=RP_RANGE_MAX if rng_max is None else float(rng_max)
  self.term_r=RP_TERM_R if term_r is None else float(term_r)
  self.cap=RP_CAP if cap is None else int(cap)
  self.reset()
 def _pick(self):
  m=self.model_by_type.get(self.mtype)
  return m if (m and (_H/m).exists()) else self.model
 def set_type(self,mtype):
  if mtype and mtype!=self.mtype:
   self.mtype=mtype
   if self._tried and self._loaded_for!=self._pick(): self._tried=False   # map type settled after the first load: swap to that map's checker
 def _load(self):
  self._tried=True
  try:
   so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
   m=self._pick()
   self.sess=ort.InferenceSession(str(_H/m),so,providers=['CPUExecutionProvider']); self.iname=self.sess.get_inputs()[0].name
   self._loaded_for=m; self.model_used=m
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
  if p<(RP_T if self.thr is None else self.thr) or h>self.hmax:
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
# ---------------- depth-head latch (kt_v265; heads from cf_swarm_sar uid255 VictimDetector) ----------------
# uid255's depth victim heads (victim_depth.onnx, victim_depth_m.onnx on mountain) see a SAR victim at 25-35 m in the always-on
# depth frame with ~0 false fires on mountain/village (probe sardet). Consistent strong hits latch a world point; the latch drives
# the clue channel and the RGB checkers' slow terminal while the policy is still searching (phase 1) and no RGB latch holds.
DH_ON=os.environ.get('KT_DH','1')=='1'
DH_TYPES=tuple(x for x in os.environ.get('KT_DH_TYPES','mountain').split(',') if x)
DH_EVERY=int(os.environ.get('KT_DH_EVERY','2')); DH_T0=int(os.environ.get('KT_DH_T0','30'))
DH_THR=float(os.environ.get('KT_DH_THR','0.7')); DH_RMAX=float(os.environ.get('KT_DH_RMAX','30.0'))
DH_N=int(os.environ.get('KT_DH_N','6')); DH_TOL=float(os.environ.get('KT_DH_TOL','3.0'))
DH_MTN=os.environ.get('KT_DH_MTN','1')=='1'; DH_TERM=os.environ.get('KT_DH_TERM','1')=='1'
DH_P1=os.environ.get('KT_DH_P1','1')=='1'; DH_STEER=os.environ.get('KT_DH_STEER','1')=='1'
DH_TERM_TYPES=tuple(x for x in os.environ.get('KT_DH_TERM_TYPES','mountain').split(',') if x)   # the dive terminal is mountain-only: in village it cost two successes (608820969, 1517797866) for one conversion
DH_PRIO_TYPES=tuple(x for x in os.environ.get('KT_DH_PRIO_TYPES','mountain').split(',') if x)
DH_PRIO=os.environ.get('KT_DH_PRIO','1')=='1'   # the depth latch outranks an RGB latch for the terminal hover (mountain RGB localises on the slope below the victim)
DH_UP=float(os.environ.get('KT_DH_UP',str(RP_HOVER_UP))); DH_TERM_R=float(os.environ.get('KT_DH_TERM_R',str(RP_TERM_R)))
DH_RING=float(os.environ.get('KT_DH_RING',str(RP_RING))); DH_SPD=float(os.environ.get('KT_DH_SPD','0.33'))
DH_DIVE=os.environ.get('KT_DH_DIVE','1')=='1'      # vertical-first terminal: the RGB profile drops 0.35 m/s and times out 10 m above a correct latch
DH_DIVE_V=float(os.environ.get('KT_DH_DIVE_V','0.5'))   # normalised speed while more than DH_NEAR_Z off the hover point (x3 m/s)
DH_NEAR_Z=float(os.environ.get('KT_DH_NEAR_Z','1.0')); DH_NEAR_H=float(os.environ.get('KT_DH_NEAR_H','1.0'))
DH_DUD=int(os.environ.get('KT_DH_DUD',str(RP_DUD_TICKS)))
DH_STALE=int(os.environ.get('KT_DH_STALE','0'))      # OFF by default: while the terminal descends the victim sits below the camera, so no hits arrive and this banned a CORRECT latch (3337902153). The parallax gate handles phantoms instead.
DH_STALE_R=float(os.environ.get('KT_DH_STALE_R','10.0'))
DH_BASE=float(os.environ.get('KT_DH_BASE','3.0'))   # parallax gate: the latch's hits must come from viewpoints this far apart.
# A wrong range estimate puts the world point somewhere that MOVES as the drone moves (village 608820969 latched a phantom 33 m out).
DH_RAMP=int(os.environ.get('KT_DH_RAMP','15'))      # blend into the terminal over this many ticks: a hard override at cruise TILTs the drone (seed 847729925)
_T_OLD_DEPTH=0.657; _T_MTN_DEPTH=0.586
def _recal(p,t_own,t_target):
 lo=p*(t_target/t_own); hi=t_target+(p-t_own)*((1.0-t_target)/(1.0-t_own))
 return float(min(max(min(lo,hi),0.0),1.0))
class _DHead(_RgbPrimary):
 """The RGB checkers' latch / dud / terminal machinery, fed by uid255's depth heads every DH_EVERY ticks."""
 def __init__(self):
  super().__init__(model='victim_depth.onnx',term_r=RP_TERM_R)
  self.msess=None; self._mtried=False
 def _load(self):
  self._tried=True
  try:
   so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
   self.sess=ort.InferenceSession(str(_H/'victim_depth.onnx'),so,providers=['CPUExecutionProvider']); self.iname=self.sess.get_inputs()[0].name
  except Exception:
   self.sess=None
 def _mload(self):
  self._mtried=True
  try:
   so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
   self.msess=ort.InferenceSession(str(_H/'victim_depth_m.onnx'),so,providers=['CPUExecutionProvider']); self.miname=self.msess.get_inputs()[0].name
  except Exception:
   self.msess=None
 def reset(self):
  super().reset(); self.n_eval=0; self.obs_xy=[]
 def observe_depth(self,tick,obs,pos,rpy,mtn):
  if tick<DH_T0 or tick%DH_EVERY:
   return
  if self.sess is None and not self._tried:
   self._load()
  if self.sess is None:
   return
  x=np.ascontiguousarray(np.asarray(obs['depth'],np.float32).reshape(256,256,1))
  o=np.asarray(self.sess.run(None,{self.iname:x})[0],np.float32).reshape(5); p=_sig(o[0]); b=o
  if mtn and DH_MTN:
   if self.msess is None and not self._mtried:
    self._mload()
   if self.msess is not None:
    m=np.asarray(self.msess.run(None,{self.miname:x})[0],np.float32).reshape(5); pm=_recal(_sig(m[0]),_T_MTN_DEPTH,_T_OLD_DEPTH)
    if pm>p:
     p=pm; b=m
  self.n_eval+=1
  if p<DH_THR:
   return
  cz=math.exp(min(float(b[3]),6.0))
  if cz>DH_RMAX:
   return
  q=_mworld(float(b[1]),float(b[2]),cz,pos,rpy)
  if not np.all(np.isfinite(q)):
   return
  if self.clue0 is not None and float(np.linalg.norm(q[0:2]-self.clue0))>RP_CLUE_R:
   return
  if not (-25.0<=float(q[2]-pos[2])<=1.0):
   return
  for bb in self.bad:
   if float(np.linalg.norm(q[0:2]-bb))<6.0:
    return
  self.hit_t=tick; self.n_fix+=1
  if self.latch is not None:
   if float(np.linalg.norm(q[0:2]-self.latch[0:2]))<=6.0:
    self.latch=(1.0-RP_EMA)*self.latch+RP_EMA*q
   return
  self.hits=[(t,z) for t,z in self.hits if tick-t<=400]+[(tick,q)]
  self.obs_xy=[(t,p) for t,p in getattr(self,'obs_xy',[]) if tick-t<=400]+[(tick,np.asarray(pos[0:2],np.float64).copy())]
  if len(self.hits)>=DH_N:
   pts=np.asarray([z for _,z in self.hits[-DH_N:]]); c=pts.mean(0)
   if float(np.max(np.linalg.norm(pts[:,0:2]-c[0:2],axis=1)))<=DH_TOL:
    ok=True
    if DH_BASE>0.0:
     ts={t for t,_ in self.hits[-DH_N:]}
     vp=np.asarray([p for t,p in self.obs_xy if t in ts],np.float64)
     ok=bool(len(vp)>=2 and float(np.max(np.linalg.norm(vp-vp.mean(0),axis=1)))*2.0>=DH_BASE)
    if ok:
     self.latch=c; self.near_since=None; self.n_latch+=1; self.latch_t=tick

 def terminal(self,pos,agl_m):
  """Same shape as the RGB terminal but with the depth latch's own hover height and speed cap."""
  if self.latch is None:
   return None
  off=self.latch[0:2]-pos[0:2]; d=float(np.linalg.norm(off))
  if d>DH_TERM_R:
   return None
  tz=float(self.latch[2])+DH_UP; dz=tz-float(pos[2])
  if dz<0 and agl_m<RP_AGL_MIN:
   dz=0.3
  if DH_DIVE:
   # the hover point is typically 6-13 m BELOW the drone when the latch forms: go vertical first at a real rate,
   # then settle. The RGB profile normalises [off_x,off_y,clip(dz,+-1.5)] and crawls down at ~0.35 m/s.
   if abs(dz)>DH_NEAR_Z or d>DH_NEAR_H:
    vec=np.array([off[0],off[1],max(-6.0,min(6.0,dz))*(2.0 if abs(dz)>DH_NEAR_Z else 1.0)],np.float64)
    n=float(np.linalg.norm(vec)); dirn=vec/n if n>1e-6 else np.zeros(3)
    far=max(d,abs(dz))
    speed=min(DH_DIVE_V,max(0.12,0.06*far+0.10))
    return dirn.astype(np.float32),float(speed)
   vec=np.array([off[0],off[1],dz],np.float64); n=float(np.linalg.norm(vec))
   dirn=vec/n if n>1e-6 else np.zeros(3)
   return dirn.astype(np.float32),float(0.04 if n<0.5 else 0.10)
  dzc=max(-1.5,min(1.5,dz))
  vec=np.array([off[0],off[1],dzc],np.float64); n=float(np.linalg.norm(vec))
  dirn=vec/n if n>1e-6 else np.zeros(3)
  if d<1.0 and abs(dz)<0.7:
   speed=0.04
  elif d<DH_RING:
   speed=0.15
  else:
   speed=min(DH_SPD,max(0.15,0.05*d+0.10))
  return dirn.astype(np.float32),float(speed)

 def tick_end(self,tick,pos):
  """Dud ban only while parked AT the hover point: the RGB timer counts ticks within 1.6 m horizontally, which
  banned a correct latch mid-descent (seed 3337902153, 10 m above the victim)."""
  if self.latch is None:
   return
  z_off=abs(float(self.latch[2])+DH_UP-float(pos[2]))
  if float(np.linalg.norm(self.latch[0:2]-pos[0:2]))<=DH_RING and z_off<=DH_NEAR_Z+0.5:
   if self.near_since is None:
    self.near_since=tick
   elif tick-self.near_since>DH_DUD:
    self.bad.append(self.latch[0:2].copy()); self.latch=None; self.hits=[]; self.near_since=None; self.n_dud+=1
  else:
   self.near_since=None
  if self.latch is not None and self.latch_t is not None and tick-self.latch_t>RP_TERM_MAX:
   self.bad.append(self.latch[0:2].copy()); self.latch=None; self.hits=[]; self.near_since=None; self.n_dud+=1
  if (self.latch is not None and DH_STALE>0 and tick-self.hit_t>DH_STALE
      and float(np.linalg.norm(self.latch[0:2]-pos[0:2]))<=DH_STALE_R):
   self.bad.append(self.latch[0:2].copy()); self.latch=None; self.hits=[]; self.near_since=None
   self.n_stale=getattr(self,'n_stale',0)+1


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
  self._rp=_RgbPrimary(model_by_type=RP_SPLIT) if (ON and RP_ON) else None
  self._r2=(_RgbPrimary(model='victim_rgb256.onnx',in_res=256,thr=R2_T,
                        hmax=R2_HMAX,rng_max=R2_RANGE_MAX,term_r=R2_TERM_R,cap=R2_CAP,model_by_type=R2_SPLIT)
            if (ON and R2_ON) else None)
  self._r2_on=False
  self._dt=_DepthTrigger() if (ON and DT_ON) else None
  self._dh=_DHead() if (ON and DH_ON) else None; self._dh_term=None
  self._plan=None
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
  self._omb=_OMB() if OMB_ON else None; self._omb_stats=self._omb.stats if self._omb is not None else {}
  self._stall=0; self._push=0; self._stall_n=0
  self.route='king'
  self._fg_on=0; self._fg_h=None; self._fg_spd=None; self._fg_n=0; self._fg_mem=None
  self._sup=None; self._mass=None; self._w0=None
  self._rp_term=None
  self._mx_n=0; self._mx_used=0
  self._p3_n=0; self._p3_ban=[]; self._p3_resets=0
  self._ss=None; self._ss_k=0; self._ss_fr=[]; self._ss_hor=[]; self._ss_t=0; self._ss_head=None; self._ss_start=None; self._ss_z0=None; self._ss_yaw0=0.0; self._ss_hit=False; self._ss_sgn=-1.0; self._ss_diag=False
  if self._rp is not None:
   self._rp.reset()
  if getattr(self,'_r2',None) is not None:
   self._r2.reset()
  self._r2_on=False
  if getattr(self,'_dh',None) is not None:
   self._dh.reset()
  self._dh_term=None
  if getattr(self,'_dt',None) is not None:
   self._dt.n_eval=0; self._dt.n_hit=0; self._dt.last=None; self._dt.last_hit_t=-10**9
  self._plan=None
  if self._esc is not None:
   self._esc.reset()
  if self._cls:
   self._cls.reset()
 def _vbuild(self,pos,clue,c):
  if self._sup is None:
   sup,w,dc=_ccore(pos[0:2]+clue,self._pad)
   self._sup=sup; self._w0=w; self._mass=w.copy(); self._dc=dc
  if FF_ON and self._stype() in FF_TYPES:
   wps=[]
   for fr in FF_FRACS:
    wps=_bous(self._core(fr),c[3],c[4],10**6,pos[0:2])
    if len(wps)<=c[5]:
     break
   wps=wps[:c[5]]
  else:
   wps=_bous(self._core(0.9),c[3],c[4],c[5],pos[0:2])
  if not wps:
   self._mass=self._w0.copy()
   wps=_bous(self._core(0.9),c[3]*0.65,c[4]*0.7,c[5],pos[0:2])
  if not wps:
   wps=_bous(self._sup[self._dc<=30.0],c[3],c[4],c[5],pos[0:2])
  if not wps and LADDER:
   # Port of the cf_swarm_sar king (uid173) posterior relaxation ladder. Runs ONLY when every champion
   # fallback produced no tour (start far from the clue: nothing within 33 m of the clue AND 83 m of the
   # pad), so any seed that already forms a tour is untouched. Pad radius grows to reach the type box
   # (max(80, dist(pad->box edge)+30) + 3 m margin), then drops the pad with the box x1.5, then clue only.
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
  """True on village/warehouse: bypass every added subsystem and run the champion's
  process end to end. Mountain wins the tie (is_mountain is sticky and set first)."""
  if not CHAMP_ROUTE:
   return False
  c=self._cls
  if c is None or c.is_mountain:
   return False
  return bool(c.settled('village') or c.settled('warehouse'))
 def act(self,observation):
  self.tick += 1
  rq=False
  if self.route.endswith('+omb'): self.route=self.route[:-4]
  if self.route.endswith('+p3r'): self.route=self.route[:-4]
  if self.route.endswith('+av'): self.route=self.route[:-3]
  if self.route.endswith('+fg'): self.route=self.route[:-3]
  if self.route.endswith('+dht'): self.route=self.route[:-4]
  st=np.asarray(observation["state"],np.float32).reshape(-1).copy()
  if ON and self._cls:
   pos=st[0:3].astype(np.float64)
   if self._pad is None:
    self._pad=pos[0:2].copy(); self._z0=float(pos[2])
    if MTN_Z0>0.0 and self._z0>=MTN_Z0:
     self._cls.is_mountain=True
   self._cls.update(self.tick,st,np.asarray(observation["depth"],np.float32))
   if self._dt is not None:
    try:
     _mt='mountain' if self._cls.is_mountain else self._r2_type()
     _hit=self._dt.step(self.tick,observation,pos,st[3:6].astype(np.float64),_mt)
     if _hit is not None:
      _tgt=self._rp if _mt=='mountain' else self._r2
      if _tgt is not None:
       _tgt.hit_t=self.tick          # chase: the next RGB request comes within RP_CHASE_MIN ticks instead of the RP_EVERY pacing
       if not self.route.endswith('+dt'): self.route=self.route+'+dt'
    except Exception:
     pass
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
   if self._on and PLAN_ON and len(self._cfg)>9 and self._stype() in PLAN_TYPES:
    c=self._cfg
    try:
     if self._plan is None:
      _sup,_w0,_dc=_ccore(pos[0:2]+st[-2:].astype(np.float64),self._pad); self._plan=_Planner(_sup,_w0)
     _agl=float(st[162])*20.0 if st.size>162 else 5.0
     self._plan.observe(pos,st[3:6].astype(np.float64),observation['depth'],_agl)
     for _det in (self._rp,self._r2):
      if _det is not None:
       for _t,_q in _det.hits:
        if (_t,id(_det)) not in self._plan.seen_hits: self._plan.seen_hits[(_t,id(_det))]=1; self._plan.hint(_q,PLAN_HINT)
     _vh=float(np.hypot(float(st[6]),float(st[7]))); self._plan.stall=self._plan.stall+1 if _vh<PLAN_STALL_V else 0; _force=False
     if self._plan.stall>=PLAN_STALL_N and self._plan.wp is not None:      # blocked on the way: drop that spot and pick another
      self._plan.mass[np.linalg.norm(self._plan.sup-self._plan.wp,axis=1)<=6.0]=0.0; self._plan.stall=0; self._plan.n_stall+=1; _force=True
     _wp=self._plan.next(pos,float(st[5]),self.tick,_force)
     off=_wp-pos[0:2]; bl=c[9] if len(c)>9 else 1.0
     if bl<1.0: off=bl*off+(1.0-bl)*st[-2:].astype(np.float64)
     st[-2]=np.float32(off[0]); st[-1]=np.float32(off[1]); self.route='king+plan'
    except Exception:
     self._plan=None
   elif self._on:
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
     if STALL_N>0 and self._stype() in STALL_TYPES and (self._rp is None or self._rp.latch is None):
      _vh=float(np.hypot(float(st[6]),float(st[7])))
      self._stall=self._stall+1 if _vh<STALL_V else 0
      if self._stall>=STALL_N:
       if self._push==0:
        self._stall_n+=1
       self._push=STALL_HOLD; self._stall=0
      if self._push>0:
       self._push-=1
       j=self._i
       while j+1<len(self._w) and float(np.linalg.norm(self._w[j]-pos[0:2]))<STALL_LOOK:
        j+=1
       if j!=self._i:
        self._i=j; self._adv=self.tick
       off=self._w[j]-pos[0:2]; _n=float(np.linalg.norm(off))
       if 1e-6<_n<STALL_LOOK:
        off=off*(STALL_LOOK/_n)
       self._stall_on=True
     bl=c[9] if len(c)>9 else 1.0
     if bl<1.0:
      off=bl*off+(1.0-bl)*st[-2:].astype(np.float64)
     st[-2]=np.float32(off[0]); st[-1]=np.float32(off[1])
     self.route='king+tour+stall' if getattr(self,'_stall_on',False) else 'king+tour'
     self._stall_on=False
   self._rp_term=None
   if self._rp is not None and not self._champ():
    try:
     if self._rp.clue0 is None:
      self._rp.clue0=pos[0:2]+np.asarray(observation['state'],np.float32).reshape(-1)[-2:].astype(np.float64)
     if self._cls.mtn_ok():
      self._rp.set_type('mountain')
      self._rp.observe(self.tick,observation,pos,st[3:6].astype(np.float64))
      if self._rp.latch is not None:
       lk=self._rp.latch[0:2]-pos[0:2]; st[-2]=np.float32(lk[0]); st[-1]=np.float32(lk[1]); self.route='king+rgb'
       self._rp_term=self._rp.terminal(pos,float(st[162])*20.0 if st.size>162 else 20.0)
      self._rp.tick_end(self.tick,pos)
    except Exception:
     self._rp_term=None
   self._r2_on=False; self._r2_term=None
   if self._r2 is not None:
    try:
     if not self._r2_ok():
      raise _R2Off
     self._r2_on=True
     self._r2.set_type(self._r2_type())
     if self._r2.clue0 is None:
      self._r2.clue0=pos[0:2]+np.asarray(observation['state'],np.float32).reshape(-1)[-2:].astype(np.float64)
     self._r2.observe(self.tick,observation,pos,st[3:6].astype(np.float64))
     if self._r2.latch is not None:
      lk=self._r2.latch[0:2]-pos[0:2]; st[-2]=np.float32(lk[0]); st[-1]=np.float32(lk[1]); self.route='king+r2'
      if R2_TERM_R>0.0 and self._r2_type() in R2_TERM_TYPES:
       _mm=np.asarray(self.memory_tensor,np.float32).reshape(-1); _ph=float(_mm[0]) if _mm.size else 1.0
       _lt=getattr(self._r2,'latch_t',None)
       if (not R2_TERM_P1 or 0.5<=_ph<1.5) and (_lt is None or self.tick-_lt>=R2_TERM_WAIT):
        self._r2_term=self._r2.terminal(pos,float(st[162])*20.0 if st.size>162 else 20.0)
     self._r2.tick_end(self.tick,pos)
    except _R2Off:
     pass
    except Exception:
     self._r2_on=False
   self._dh_term=None
   if self._dh is not None:
    try:
     _mt='mountain' if self._cls.is_mountain else self._stype()
     if _mt in DH_TYPES:
      if self._dh.clue0 is None:
       self._dh.clue0=pos[0:2]+np.asarray(observation['state'],np.float32).reshape(-1)[-2:].astype(np.float64)
      self._dh.observe_depth(self.tick,observation,pos,st[3:6].astype(np.float64),_mt=='mountain')
      _rgbl=bool((self._rp is not None and self._rp.latch is not None and self._cls.mtn_ok() and not self._champ()) or
                 (self._r2_on and self._r2 is not None and self._r2.latch is not None))
      _prio=bool(DH_PRIO and _mt in DH_PRIO_TYPES)
      _term=bool(DH_TERM and _mt in DH_TERM_TYPES)
      if self._dh.latch is not None and (not _rgbl or _prio):
       _mm=np.asarray(self.memory_tensor,np.float32).reshape(-1); _ph=float(_mm[0]) if _mm.size else 1.0
       if (not DH_P1) or 0.5<=_ph<1.5 or (_prio and _rgbl):
        if DH_STEER and not _rgbl:
         lk=self._dh.latch[0:2]-pos[0:2]; st[-2]=np.float32(lk[0]); st[-1]=np.float32(lk[1]); self.route='king+dh'
        if _term:
         self._dh_term=self._dh.terminal(pos,float(st[162])*20.0 if st.size>162 else 20.0)
      self._dh.tick_end(self.tick,pos)
    except Exception:
     self._dh_term=None
  if AV_D>0.0 and self._cls and self.tick>=AV_T0 and self._stype() in AV_TYPES:
   dm=np.asarray(observation['depth'],np.float32).reshape(256,256)*29.5+0.5
   ctr=float(dm[96:160,96:160].min())
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
  if MX_ON and self._cls is not None:
   try:
    mm=np.asarray(self.memory_tensor,np.float32).reshape(-1)
    if mm.size>=63:
     if mm[59]<0.5:
      self._mx_n=0
     elif mm[57]>0.5 and mm[59]>=MX_AT and self._mx_n<MX_EXT and (
       ('warehouse' in MX_TYPES and getattr(self,'_z0',None) is not None and self._z0<=WESC_Z) or
       ((getattr(self,'_z0',None) is None or self._z0>WESC_Z) and self._stype() in MX_TYPES)):
      mm=mm.copy(); mm[59]=np.float32(MX_AT); self._mx_n+=1; self._mx_used+=1
      self.memory_tensor=mm.reshape(self._ms)
   except Exception:
    pass
  if P3_ON:
   try:
    mm=np.asarray(self.memory_tensor,np.float32).reshape(-1)
    if mm.size>=63:
     rs=False
     if mm[0]>=2.5:
      self._p3_n+=1
      if self._p3_n>P3_T:
       self._p3_ban.append(np.asarray(mm[60:62],np.float64).copy()); rs=True
     else:
      self._p3_n=0
     if not rs and self._p3_ban and mm[0]>=1.5 and mm[57]>0.5:
      e=np.asarray(mm[60:62],np.float64)
      if any(float(np.linalg.norm(e-b))<P3_BAN_R for b in self._p3_ban): rs=True
     if rs:
      mm=mm.copy(); mm[0]=1.0; mm[56]=0.0; mm[57]=0.0; mm[58]=0.0; mm[59]=0.0
      self.memory_tensor=mm.reshape(self._ms); self._p3_n=0; self._p3_resets+=1; self.route=self.route+'+p3r'
   except Exception:
    pass
  feed={"depth": np.asarray(observation["depth"],np.float32).reshape(256,256,1),
    "state": st,"memory_tensor": self.memory_tensor}
  if "rgb" in self._in:
   rgb=observation.get("rgb") if hasattr(observation,"get") else None
   feed["rgb"]=(np.asarray(rgb,np.float32).reshape(256,256,3)
      if rgb is not None else np.zeros((256,256,3),np.float32))
  self._fg_mem=np.asarray(self.memory_tensor,np.float32).reshape(-1).copy()
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
  if SS_ON and out.shape[0]>=5:
   try:
    _o=self._sscan(observation,st,out)
    if _o is not None:
     return _o
   except Exception:
    self._ss='done'
  if self._rp is not None and out.shape[0]>=6 and self._cls is not None and self._cls.mtn_ok() and not self._champ():
   out=out.copy()
   if self._rp_term is not None:
    out[0:3]=self._rp_term[0]; out[3]=np.float32(self._rp_term[1]); out[4]=np.float32(0.0)
   if self._rp.want_rgb(self.tick):
    out[5]=np.float32(1.0)
  if self._r2 is not None and out.shape[0]>=6 and self._r2_on:
   if getattr(self,'_r2_term',None) is not None:
    _use=True; _spd=float(self._r2_term[1]); _yt=0.0; _dirv=np.asarray(self._r2_term[0],np.float32).copy()
    try:
     if R2_FACE and self._r2_type() in R2_FACE_TYPES and self._r2.latch is not None:
      _lx=float(self._r2.latch[0])-float(st[0]); _ly=float(self._r2.latch[1])-float(st[1]); _dh=math.hypot(_lx,_ly)
      if _dh>1.0:
       _hd=math.atan2(_ly,_lx); _yt=_hd
       if abs(_wrap(_hd-float(st[5])))>math.radians(30.0):
        _spd=min(_spd,0.03)
       else:
        _dm=np.asarray(observation['depth'],np.float32).reshape(256,256)*29.5+0.5
        if float(_dm[110:146,112:144].min())<min(_dh,R2_FACE_CLR): _use=False
      else:
       _yt=float(st[5])
    except Exception:
     _use=True; _spd=float(self._r2_term[1]); _yt=0.0
    if _use:
     out=out.copy(); out[0:3]=_dirv; out[3]=np.float32(_spd); out[4]=np.float32(_wrap(_yt)/math.pi if _yt!=0.0 else 0.0)
     if not self.route.endswith('+r2t'): self.route=self.route+'+r2t'
   if self._r2.want_rgb(self.tick):
    out=out.copy(); out[5]=np.float32(1.0)
  if getattr(self,'_dh_term',None) is not None and out.shape[0]>=4:
   self._dh_n=getattr(self,'_dh_n',0)+1
   a=1.0 if DH_RAMP<=0 else min(1.0,float(self._dh_n)/float(DH_RAMP))
   _d=np.asarray(self._dh_term[0],np.float64); _s=float(self._dh_term[1])
   _pd=np.asarray(out[0:3],np.float64); _pn=float(np.linalg.norm(_pd))
   if _pn>1e-6 and a<1.0:
    _d=(1.0-a)*(_pd/_pn)+a*_d; _n=float(np.linalg.norm(_d))
    _d=_d/_n if _n>1e-6 else np.asarray(self._dh_term[0],np.float64)
    _s=(1.0-a)*float(out[3])+a*_s
   out=out.copy(); out[0:3]=_d.astype(np.float32); out[3]=np.float32(_s); out[4]=np.float32(0.0)
   if not self.route.endswith('+dht'): self.route=self.route+'+dht'
  else:
   self._dh_n=0
  if FG_ON and self._cls is not None and out.shape[0]>=4 and self._fg_mem is not None and self._stype()=='forest':
   try:
    out=self._fguard(observation,st,out)
   except Exception:
    self._fg_on=0; self._fg_h=None
  if self._omb is not None and self._cls is not None and out.shape[0]>=4:
   try:
    _t=self._stype()
    if _t in OMB_TYPES:
     _p=st[0:3].astype(np.float64); _r=st[3:6].astype(np.float64)
     _gz=float(_p[2])-float(st[162])*20.0 if st.size>162 else None
     self._omb.observe(observation['depth'],_p,_r,_gz); self._omb.stats['ticks']+=1
     _esc=self._esc is not None and self._esc.phase in ('scan','creep','climb')
     if not _esc:
      _o=out.copy()
      if self._omb.constrain(_o,_p,_r,_t):
       out=_o
       if not self.route.endswith('+omb'): self.route=self.route+'+omb'
   except Exception:
    pass
  return out
 def _sscan(self,obs,st,out):
  z=float(st[2]); yaw=float(st[5])
  if self._ss is None:
   if self.tick!=1 or z>SS_Z or (self._esc is not None and self._esc.mode=='ceil'):
    self._ss='off'; return None
   self._ss='scan'; self._ss_z0=z; self._ss_yaw0=yaw; self._ss_t=0
   f0,h0=_ss_frac(obs['depth']); self._ss_fr=[f0]; self._ss_hor=[h0]
   _c=_wrap(float(out[4])*math.pi-yaw); self._ss_sgn=1.0 if _c>1e-4 else -1.0
  if self._ss in ('off','done'):
   if self.route=='king+ss': self.route='king'
   return None
  self._ss_t+=1
  if self._ss_t>SS_MAXT:
   self._ss='done'; self.route='king'; return None
  o=out.copy(); o[5]=np.float32(0.0)
  zt=self._ss_z0+SS_LIFT; vz=float(np.clip(2.0*(zt-z),-SS_VZ,SS_VZ))
  def _cmd(dirv,spd,hd):
   o[0:3]=np.asarray(dirv,np.float32); o[3]=np.float32(spd); o[4]=np.float32(_wrap(hd)/math.pi); self.route='king+ss'; return o
  side=self._ss_yaw0+self._ss_sgn*math.radians(SS_SIDE)
  if self._ss=='scan':
   air=z>self._ss_z0+0.04
   if air and z>self._ss_z0+0.08 and abs(_wrap(yaw-side))<math.radians(6.0):
    f,hz=_ss_frac(obs['depth']); self._ss_fr.append(f); self._ss_hor.append(hz); tp=_ss_top(obs['depth'])
    if not self._ss_diag:
     self._ss_diag=True; self._ss_fr.pop(); self._ss_hor.pop()
     o[0:3]=np.asarray([0.0,0.0,1.0 if vz>=0 else -1.0],np.float32); o[3]=np.float32(abs(vz)/3.0); o[4]=np.float32(_wrap(side)/math.pi)
     self.route='ssm%+d_%02d_%03d_%03d'%(int(self._ss_sgn),min(99,int(self._ss_fr[0]*100)),min(999,int(tp*100)),min(999,int(hz*100))); return o
    if tp<=SS_TOP and hz<SS_HMAX:
     self._ss_head=self._ss_yaw0+(math.pi if self._ss_fr[0]>SS_F0 else 0.0)
     self._ss='out'; self._ss_hit=True; self._ss_start=st[0:2].astype(np.float64).copy()
    else:
     self._ss='done'; self.route='king'; return None
   else:
    return _cmd([0.0,0.0,1.0 if vz>=0 else -1.0],abs(vz)/3.0,side if air else yaw)
  if self._ss=='out':
   trav=float(np.linalg.norm(st[0:2].astype(np.float64)-self._ss_start))
   if trav>=SS_OUT:
    self._ss='done'; self.route='king'; return None
   h=self._ss_head; dv=np.array([math.cos(h),math.sin(h),0.4*vz],np.float32); dv/=float(np.linalg.norm(dv))
   return _cmd(dv,SS_OUT_V/3.0,yaw)
  self._ss='done'; self.route='king'; return None
 def _fguard(self,obs,st,out):
  m=self._fg_mem
  if int(round(float(m[0]))) not in FG_PH:
   self._fg_on=0; self._fg_h=None; self._fg_spd=None; return out
  pos=st[0:3].astype(np.float64); rel=m[8:11].astype(np.float64)-pos; dh=float(np.hypot(rel[0],rel[1]))
  vel=st[6:9].astype(np.float64)
  if dh<FG_MINH:
   self._fg_on=0; self._fg_h=None; self._fg_spd=None; return out
  yaw=float(st[5])
  if abs(float(st[3]))>math.radians(FG_TILT) or abs(float(st[4]))>math.radians(FG_TILT):
   self._fg_h=None; return out
  bt=math.atan2(math.sin(math.atan2(rel[1],rel[0])-yaw),math.cos(math.atan2(rel[1],rel[0])-yaw))
  if abs(bt)>math.radians(FG_MAXB):
   self._fg_on=0; self._fg_h=None; self._fg_spd=None; return out
  dm=np.asarray(obs['depth'],np.float32).reshape(256,256)*29.5+0.5
  col=dm[128-FG_ROWS:128+FG_ROWS].min(axis=0)
  def cix(b):
   return int(round((1.0-math.tan(b))*128.0-0.5))
  def free(c,need):
   w=int(math.ceil(128.0*FG_W/max(need,0.5)))
   lo,hi=c-w,c+w+1
   if lo<0 or hi>256: return False
   return bool(col[lo:hi].min()>=need)
  need_t=min(dh-0.3,FG_D)
  blocked=not free(cix(bt),need_t)
  if blocked:
   if self._fg_on==0: self._fg_n+=1
   self._fg_on=FG_HOLD
  elif self._fg_on>0:
   self._fg_on-=1
  if self._fg_on<=0:
   self._fg_h=None; self._fg_spd=None; return out
  out=out.copy()
  need2=min(dh,FG_D2)
  best=None
  for k in range(0,int(FG_MAXB)+1,2):
   for sg in ((1,-1) if k else (1,)):
    b=bt+sg*math.radians(k)
    if abs(b)>math.radians(FG_MAXB): continue
    if free(cix(b),need2):
     best=b; break
   if best is not None: break
  sp=float(np.linalg.norm(vel))/3.0
  if self._fg_spd is None: self._fg_spd=min(1.0,sp)
  if best is None:
   cap=0.05; hwant=self._fg_h if self._fg_h is not None else math.atan2(vel[1],vel[0]) if sp>0.05 else yaw
  else:
   cap=FG_V/3.0; hwant=yaw+best
  hcur=self._fg_h if self._fg_h is not None else (math.atan2(float(out[1]),float(out[0])) if abs(float(out[0]))+abs(float(out[1]))>1e-6 else yaw)
  dhd=math.atan2(math.sin(hwant-hcur),math.cos(hwant-hcur)); lim=min(math.radians(FG_TURN),FG_ALAT/max(float(np.linalg.norm(vel[0:2])),0.5)/50.0)
  hnew=hcur+max(-lim,min(lim,dhd)); self._fg_h=hnew
  k=float(np.hypot(float(out[0]),float(out[1])))
  v=np.array([math.cos(hnew)*max(k,0.3),math.sin(hnew)*max(k,0.3),float(out[2])])
  v=v/max(float(np.linalg.norm(v)),1e-6)
  self._fg_spd=max(min(float(out[3]),cap),self._fg_spd-FG_DN) if float(out[3])>cap else float(out[3])
  self._fg_spd=min(self._fg_spd,float(out[3]))
  out[0:3]=v.astype(np.float32); out[3]=np.float32(self._fg_spd)
  if not self.route.endswith('+fg'): self.route=self.route+'+fg'
  return out

# r20
