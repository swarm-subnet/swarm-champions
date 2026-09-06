import os
from pathlib import Path
import numpy as np
import onnxruntime as ort
_H=Path(__file__).resolve().parent
ON=os.environ.get('KT_ON','1')=='1'
CFG={
 'forest':   (350,15,10.0,8.0,5.0,40,120,8.0,0),
 'city':     (450,25,10.0,6.0,5.0,40,250,4.5,1),
 'mountain': (750,25,10.0,6.0,5.0,40,250,4.5,0),
}
TYPES=tuple(t for t in os.environ.get('KT_TYPES','forest,city,mountain').split(',') if t)
LB=('city','open','mountain','village','warehouse','forest')
SIM_DT=0.02
VW=float(os.environ.get('KT_VETO_WARM','0.02'))
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
  self.latched=set(); self.is_mountain=False
 def update(self,tick,state,depth):
  if tick>600 or tick%5:
   return
  s=np.asarray(state,np.float32).reshape(-1)
  sf=np.zeros(141,np.float32)
  sf[:min(141,s.size)]=s[:141]
  f=np.asarray([float(tick),float(tick)*SIM_DT]+sf.tolist()+_dfeat(depth),np.float32)
  self.ps += self.net.probs(f); self.n += 1
  avg=self.ps/self.n
  i=int(np.argmax(avg)); lab=LB[i]
  bar=0.5 if lab=='warehouse' else 0.7
  if self.n>=3 and avg[i]>=bar:
   self.latched.add(lab)
   if lab=='mountain':
    self.is_mountain=True
  if self.n>=40 and avg[4]>=0.15:
   self.latched.add('warehouse')
 def settled(self,t):
  return t in self.latched
def _warm(rgb):
 im=np.asarray(rgb,np.float32).reshape(256,256,3)[128:]
 r,g,b=im[...,0],im[...,1],im[...,2]
 return float(((r>g+0.03) & (g>=b)).mean())
_BOX=np.array([25.0,30.0,30.0,25.0,12.0,20.0])
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
  self.reset()
 def reset(self):
  self.memory_tensor=np.zeros(self._ms,np.float32)
  self.tick=0
  self._h={15: [],25: []}
  self._pad=None
  self._on=False
  self._cfg=None
  self._w=None
  self._i=0
  self._adv=0
  self._cv=None
  self._cvt=0
  if self._cls:
   self._cls.reset()
 def _stype(self):
  c=self._cls
  if c.is_mountain:
   return 'mountain' if 'mountain' in TYPES else None
  for t in TYPES:
   if t!='mountain' and c.settled(t):
    return t
  return None
 def act(self,observation):
  self.tick += 1
  rq=False
  st=np.asarray(observation["state"],np.float32).reshape(-1).copy()
  if ON and self._cls:
   pos=st[0:3].astype(np.float64)
   if self._pad is None:
    self._pad=pos[0:2].copy()
   self._cls.update(self.tick,st,np.asarray(observation["depth"],np.float32))
   for pd,hs in self._h.items():
    if self.tick%pd==0:
     hs.append(pos[0:2].copy())
     if len(hs)>20:
      hs.pop(0)
   if not self._on:
    t=self._stype()
    c=CFG.get(t) if t else None
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
     self._w=_wps(pos[0:2]+st[-2:].astype(np.float64),self._pad,
        pos[0:2],c[3],c[4],c[5])
     self._i=0; self._adv=self.tick
    wp=self._w[self._i]
    if float(np.linalg.norm(wp-pos[0:2]))<c[7] or self.tick-self._adv>c[6]:
     self._i=(self._i+1)%len(self._w)
     self._adv=self.tick
     wp=self._w[self._i]
    off=wp-pos[0:2]
    st[-2]=np.float32(off[0]); st[-1]=np.float32(off[1])
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
  return out
