import math
import os
import types
from collections import Counter

import numpy as np

FLY_DESC = os.environ.get('KTM_FLY_DESC', '0').strip()


def tangent_profile(z, distances, floors, end_distance, end_height, buffer=3.):
    valid = np.isfinite(floors) & (floors > -1e8)
    x = np.maximum(np.asarray(distances)[valid] - buffer, 1.)
    y = np.asarray(floors)[valid]
    slopes = np.r_[(y-z)/x, (end_height-z)/max(end_distance,1.)]
    k = int(np.argmax(slopes))
    return float(slopes[k]), float(np.r_[y,end_height][k])


def install(brain, FL):
    f = brain.fl
    original_search, original_far = f.cmd_search, f.cmd_far
    f.mtn_fly_stats = Counter()
    f.mtn_fly_dbg = {}

    def command(self, hd, vmag, yaw, H, zgoal, dist, fallback):
        st = self.st; pos, vel = st[:3], st[6:9]
        hd = np.asarray(hd, dtype=float)
        vn = float(np.linalg.norm(vel[:2]))
        aligned = vn < .5 or float(vel[:2]@hd)/vn > .94
        if brain.stage not in ('transit','search') or dist < 5 or not aligned:
            self.mtn_fly_dbg = {'active':False,'reason':'stage/near/turn'}
            return fallback()
        margin = max(2.5,.6*H)
        length = min(28.,dist+1.)
        al, hm, unseen = self.grid.corridor(pos[:2],hd,length,self.half_w)
        valid = hm > -1e8
        if valid.sum()<3 or unseen>.6 or self.ground is None:
            self.mtn_fly_dbg = {'active':False,'reason':'unmapped'}
            return fallback()
        need = hm+margin
        old_al, old_h, old_unseen = self.grid.corridor(pos[:2],hd,self.look_s,self.half_w)
        old_need = old_h+margin
        base_z = zgoal if zgoal is not None else self.ground+H
        old_z = max(base_z,float(old_need.max()),self.ground+margin)
        old_vz = float(np.clip(1.2*(old_z-pos[2]),-2.,2.5))
        goal = brain.goal
        if brain.stage=='search' and goal is not None and len(goal)>2:
            target_z = float(goal[2])
            if zgoal is not None:
                target_z = float(zgoal)
            end_d = min(dist,length)
            if dist>length:
                target_z = pos[2]+(target_z-pos[2])*length/dist
        else:
            end_d = length
            far = valid & (al >= .6*length)
            target_z = float(np.max(hm[far])+H) if far.any() else self.ground+H
        slope, band = tangent_profile(float(pos[2]),al,need,end_d,target_z)
        if vn>.5 and float(vel[:2]@hd)/vn < .999:
            av,hv,_ = self.grid.corridor(pos[:2],vel[:2]/vn,min(16.,length),self.half_w)
            sv,_ = tangent_profile(float(pos[2]),av,hv+margin,length,target_z)
            slope=max(slope,sv)
        speed = min(3.,float(vmag))
        if dist>12 and vmag>=brain.par['V']-1e-8 and old_unseen<.2:
            speed=3.
        vz = speed*slope/math.sqrt(1+slope*slope)
        vz = max(vz,1.2*(self.ground+margin-pos[2]))
        if (FLY_DESC == '1' or (FLY_DESC == 'transit' and brain.stage == 'transit')) and old_vz < 0. and old_vz < vz:
            vz = max(old_vz,1.2*(self.ground+margin-pos[2]))
            self.mtn_fly_stats['desc_ticks']+=1
        vz = float(np.clip(vz,-2.,self.vz_max))
        horizontal = min(float(vmag),math.sqrt(max(0.,speed*speed-vz*vz)))
        if speed==3. and vmag>=brain.par['V']-1e-8:
            horizontal=math.sqrt(max(0.,speed*speed-vz*vz))
        horizontal = self._climb_cap(al,need,float(pos[2]),horizontal)
        self._set_phase('search')
        c=self.sh([hd[0]*horizontal,hd[1]*horizontal,vz],vel)
        self.dbg={'ph':'search','dh':dist,'z_des':float(pos[2]+vz/1.2),'vcap':horizontal}
        self.mtn_fly_dbg=dict(active=True,band=band,target_z=target_z,slope=slope,
            old_z=old_z,old_vz=old_vz,vz=vz,unseen=unseen,look=length)
        self.mtn_fly_stats[brain.stage+'_ticks']+=1
        self.mtn_fly_stats['vertical_demand_saved']+=.02*(abs(old_vz)-abs(vz))
        return FL.vel_action(c,yaw)

    def search(self,hd,vmag,yaw,H,zgoal,dist):
        return command(self,hd,vmag,yaw,H,zgoal,dist,
                       lambda: original_search(hd,vmag,yaw,H,zgoal,dist))

    def far(self,hd,vmag,yaw,H,dist,V,cfg):
        self.mtn_fly_dbg={'active':False,'reason':'existing_far_profile'}
        return original_far(hd,vmag,yaw,H,dist,V,cfg)

    f.cmd_search=types.MethodType(search,f)
    f.cmd_far=types.MethodType(far,f)
