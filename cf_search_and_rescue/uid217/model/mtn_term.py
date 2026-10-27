import math
import types
from collections import Counter

import numpy as np


def adverse_velocity(direction, velocity, command):
    for v in (velocity[:2],command[:2]):
        speed=float(np.linalg.norm(v))
        if speed>.7 and float(np.dot(v,direction)) < -.15*speed:
            return True
    return False


def capture_path(position,velocity,target,acceleration=2.):
    speed=float(np.linalg.norm(velocity[:2]));delta=target[:2]-position[:2]
    distance=float(np.linalg.norm(delta))
    if not (1.<speed<3.5 and .8<distance<2.5 and abs(velocity[2])<.4
            and abs(target[2]-position[2])<.55):return None
    direction=velocity[:2]/speed
    if float(delta@direction)<.95*distance:return None
    duration=(speed-.8)/acceleration
    travel=.15*speed+.5*(speed+.8)*duration
    end=position.copy();end[:2]+=direction*travel
    if np.linalg.norm(end[:2]-target[:2])>1.2:return None
    end[2]+=velocity[2]*(duration+.15)
    if abs(end[2]-target[2])>.6:return None
    return position+np.linspace(0,1,7)[:,None]*(end-position)


def install(brain,FL):
    f=brain.fl;original=f.cmd_approach
    f.mtn_term_stats=Counter();f.mtn_term_dbg={}
    f.mtn_term_braking=False

    def approach(self,x,y,vtop):
        st=self.st;p=st[:3];vel=st[6:9];e=np.array([x,y])-p[:2]
        dh=float(np.linalg.norm(e));direction=e/max(dh,1e-9)
        if dh>2.5 and adverse_velocity(direction,vel,self.sh.cmd):
            self.mtn_term_braking=True
        if self.mtn_term_braking:
            if np.linalg.norm(vel[:2])<.4 and np.linalg.norm(self.sh.cmd[:2])<.25:
                self.mtn_term_braking=False
            else:
                self.dp_hist.append(np.asarray(p,dtype=float).copy())
                floor=self.ground+2. if self.ground is not None else p[2]
                speed=float(np.linalg.norm(vel[:2]))
                if speed>.2 and self.grid is not None:
                    stop=speed*speed/(2*self.sh.A)+.6*speed+1.
                    _,h,_=self.grid.corridor(p[:2],vel[:2]/speed,min(6.,max(2.,stop)),self.half_w)
                    if np.any(h>-1e8):floor=max(floor,float(h.max())+2.)
                vz=float(np.clip(1.2*(max(vtop+self.up,floor)-p[2]),-1.,self.vz_max))
                self._set_phase('appr')
                c=self.sh([0.,0.,vz],vel)
                self.dbg=dict(ph='appr',dh=dh,z_des=floor,vcap=0.)
                self.mtn_term_dbg=dict(mode='brake',floor=floor)
                self.mtn_term_stats['brake_ticks']+=1
                return FL.vel_action(c,math.atan2(e[1],e[0]))
        estimate=brain.est
        if (estimate is not None and estimate.get('rgb_ok') and estimate.get('n',0)>=2
                and self.fb_i==0 and np.linalg.norm(self.hoff)<.1 and self.lmap is not None):
            target=np.array([x,y,vtop+self.up])
            path=capture_path(p,vel,target)
            L=self.lmap
            known=(L.h>-1e8)&(np.hypot(L.GX-x,L.GY-y)<2.5)
            if (path is not None and int(known.sum())>=9
                    and float(L.clearance(path).min())>=self.hover_clr):
                self.dp_hist.append(np.asarray(p,dtype=float).copy())
                vz=float(np.clip(self.kz_hold*(target[2]-p[2]),-.6,.6))
                c=self.sh([0.,0.,vz],vel)
                self._set_phase('hold')
                self.dbg=dict(ph='hold',dh=dh,z_des=target[2],vcap=0.)
                self.mtn_term_dbg=dict(mode='capture',predicted_end=path[-1].tolist())
                self.mtn_term_stats['capture_ticks']+=1
                return FL.vel_action(c,float(st[5]))
        self.mtn_term_dbg=dict(mode='original_approach')
        return original(x,y,vtop)

    f.cmd_approach=types.MethodType(approach,f)
