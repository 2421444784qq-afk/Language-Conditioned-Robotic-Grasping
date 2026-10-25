from pathlib import Path
import csv, cv2, mujoco, numpy as np
from envs.custom_pick_place_env import PandaPickPlaceGymEnv
ROOT=Path(__file__).resolve().parents[1]
XML=ROOT/'envs/assets/scene_two_graspable_blocks.xml'
OUT=ROOT/'results/random_position_results.csv'
MASTER_SEED=20261025; N_TRIALS=50
MAX_DELTA=0.002; XY_TOL=0.004; Z_TOL=0.004
GRASP_OFFSET_Z=0.025; LIFT_HEIGHT=0.06
OPEN_COMMAND=-1.0; CLOSE_COMMAND=1.0
PROMPTS=['left','right','purple','blue']
# Random unseen positions: same safe x=0.50 plane, broad Y range, >=8 cm separation.
Y_MIN=-0.16; Y_MAX=0.16; MIN_SEP=0.08

def largest(mask):
 n,lab,stats,cent=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8); cand=[]
 for i in range(1,n):
  x,y,w,h,a=stats[i]
  if a>=20: cand.append((float(cent[i][0]),float(cent[i][1]),int(a)))
 return max(cand,key=lambda z:z[2]) if cand else None

def detect(env):
 im=np.asarray(env._compute_observation()['pixels']['front']); hsv=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
 roi=np.zeros(im.shape[:2],bool); roi[55:127,10:120]=True
 p=largest((hsv[:,:,0]>=125)&(hsv[:,:,0]<=175)&(hsv[:,:,1]>=70)&(hsv[:,:,2]>=45)&roi)
 b=largest((hsv[:,:,0]>=78)&(hsv[:,:,0]<=112)&(hsv[:,:,1]>=70)&(hsv[:,:,2]>=45)&roi)
 return p,b

def route(prompt,p,b):
 if not p or not b: return None
 objs=[('purple',p),('blue',b)]
 if prompt=='purple': return 'purple'
 if prompt=='blue': return 'blue'
 if prompt=='left': return min(objs,key=lambda z:z[1][0])[0]
 if prompt=='right': return max(objs,key=lambda z:z[1][0])[0]
 return None

def sample_layout(rng):
 while True:
  py=float(rng.uniform(Y_MIN,Y_MAX)); by=float(rng.uniform(Y_MIN,Y_MAX))
  if abs(py-by)>=MIN_SEP: return py,by

def run_trial(i,py,by,prompt,seed):
 env=PandaPickPlaceGymEnv(xml_path=XML,image_obs=True,render_mode='rgb_array',reward_type='dense',random_block_position=False,random_target_position=False,seed=seed)
 row={'trial':i,'purple_y':round(py,5),'blue_y':round(by,5),'separation':round(abs(py-by),5),'prompt':prompt,'seed':seed}
 try:
  env.reset(seed=seed); m=env._model; d=env._data
  d.jnt('block').qpos[:3]=(0.5,py,env._block_z); d.jnt('blue_block').qpos[:3]=(0.5,by,env._block_z); mujoco.mj_forward(m,d)
  p,b=detect(env); row['purple_detected']=bool(p); row['blue_detected']=bool(b)
  if not p or not b:
   row.update(selected=None,expected=None,routing_ok=False,grasp_ok=False,lift_m=0.0,status='DETECT_FAIL'); return row
  selected=route(prompt,p,b)
  # Front-camera x increases with world Y in this scene, so image-left is lower world Y.
  left_color='purple' if p[0]<b[0] else 'blue'; right_color='blue' if left_color=='purple' else 'purple'
  expected={'purple':'purple','blue':'blue','left':left_color,'right':right_color}[prompt]
  row.update(purple_cx=round(p[0],2),blue_cx=round(b[0],2),selected=selected,expected=expected)
  if selected!=expected:
   row.update(routing_ok=False,grasp_ok=False,lift_m=0.0,status='ROUTE_FAIL'); return row
  row['routing_ok']=True
  body_name='block' if selected=='purple' else 'blue_block'; body_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,body_name); pinch_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'pinch')
  def obj(): return d.xpos[body_id].copy()
  def eef(): return d.site_xpos[pinch_id].copy()
  def move_to(target,gcmd,max_steps):
   target=np.asarray(target,float)
   for _ in range(max_steps):
    mocap=d.mocap_pos[0].copy(); err=target-mocap
    if np.linalg.norm(err[:2])<XY_TOL and abs(err[2])<Z_TOL:
     ae=target-eef()
     if np.linalg.norm(ae[:2])<0.01 and abs(ae[2])<0.01: return True
    a=np.zeros(7,np.float32); a[:3]=np.clip(err,-MAX_DELTA,MAX_DELTA); a[6]=gcmd
    _,_,term,trunc,_=env.step(a)
    if term or trunc: return False
   return False
  initial=obj().copy()
  if not move_to([initial[0],initial[1],d.mocap_pos[0][2]],OPEN_COMMAND,220): row.update(grasp_ok=False,lift_m=0.0,status='ALIGN_FAIL'); return row
  if not move_to([initial[0],initial[1],initial[2]+GRASP_OFFSET_Z],OPEN_COMMAND,150): row.update(grasp_ok=False,lift_m=0.0,status='DESCEND_FAIL'); return row
  for _ in range(3):
   a=np.zeros(7,np.float32); a[6]=CLOSE_COMMAND; env.step(a)
  lt=eef().copy(); lt[2]+=LIFT_HEIGHT
  if not move_to(lt,0.0,150): row.update(grasp_ok=False,lift_m=0.0,status='LIFT_MOVE_FAIL'); return row
  lift=float(obj()[2]-initial[2]); ok=lift>0.03; row.update(grasp_ok=ok,lift_m=round(lift,5),status='PASS' if ok else 'NO_LIFT'); return row
 except Exception as e:
  row.update(routing_ok=False,grasp_ok=False,lift_m=0.0,status='EXCEPTION:'+repr(e)); return row
 finally: env.close()

def main():
 rng=np.random.default_rng(MASTER_SEED); rows=[]
 trials=[]
 for i in range(1,N_TRIALS+1):
  py,by=sample_layout(rng); prompt=PROMPTS[int(rng.integers(0,len(PROMPTS)))]; trials.append((i,py,by,prompt,MASTER_SEED+i))
 print('RANDOM UNSEEN BATCH START trials=',N_TRIALS,'Y=',(Y_MIN,Y_MAX),'min_sep=',MIN_SEP,flush=True)
 for i,py,by,prompt,seed in trials:
  r=run_trial(i,py,by,prompt,seed); rows.append(r); print(f'[{i:02d}/{N_TRIALS}]',r,flush=True)
 fields=['trial','purple_y','blue_y','separation','prompt','seed','purple_detected','blue_detected','purple_cx','blue_cx','selected','expected','routing_ok','grasp_ok','lift_m','status']
 with OUT.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 print('\n===== RANDOM UNSEEN SUMMARY =====')
 for label,key in [('detection','purple_detected'),('routing','routing_ok'),('grasp','grasp_ok')]:
  if label=='detection': n=sum(bool(r.get('purple_detected')) and bool(r.get('blue_detected')) for r in rows)
  else: n=sum(bool(r.get(key)) for r in rows)
  print(label+':',n,'/',N_TRIALS,'=',round(n/N_TRIALS,3))
 statuses={s:sum(r.get('status')==s for r in rows) for s in sorted(set(r.get('status') for r in rows))}; print('status_counts:',statuses)
 for prompt in PROMPTS:
  q=[r for r in rows if r['prompt']==prompt]; print(prompt,'n=',len(q),'route=',sum(bool(r.get('routing_ok')) for r in q),'grasp=',sum(bool(r.get('grasp_ok')) for r in q))
 print('CSV:',OUT)
if __name__=='__main__': main()
