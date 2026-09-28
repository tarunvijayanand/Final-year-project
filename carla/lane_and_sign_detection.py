import sys
sys.path.append(r"C:\Carla\PythonAPI\carla")

import carla
import random
import numpy as np
import cv2
import time
import torch
from collections import deque
from ultralytics import YOLO
from agents.navigation.global_route_planner import GlobalRoutePlanner
from attention_unet import AttentionUNet


# ================= CONFIG =================
YOLO_MODEL = r"C:\Users\tarun pv\OneDrive\Desktop\Final Yeat Project\training\runs\detect\traffic_training\yolo_model\weights\best.pt"
LANE_MODEL  = r"C:\Users\tarun pv\OneDrive\Desktop\Final Yeat Project\training\best_attention_unet_lane.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SPEED        = 35    # km/h cruise speed
LIDAR_SLOW_DIST  = 12.0  # meters — start slowing
LIDAR_STOP_DIST  = 4.0   # meters — hard brake
RADAR_CLOSE_MS   = 8.0   # m/s closing speed threshold
KP = 1.0
KI = 0.02
KD = 0.25
# ==========================================


# ================= PID ====================
class PID:
    def __init__(self,kp,ki,kd):
        self.kp=kp; self.ki=ki; self.kd=kd
        self.prev=0; self.sum=0

    def step(self,error):
        self.sum=np.clip(self.sum+error,-10,10)
        d=error-self.prev
        self.prev=error
        return self.kp*error+self.ki*self.sum+self.kd*d

pid=PID(KP,KI,KD)
prev_steer=0.0


def compute_route_steer(vehicle, waypoint):
    t=vehicle.get_transform()
    loc=t.location
    f=t.get_forward_vector()
    target=waypoint.transform.location-loc
    dot=f.x*target.x+f.y*target.y
    det=f.x*target.y-f.y*target.x
    return float(np.clip(np.arctan2(det,dot),-1,1))


# ================= LOAD MODELS =================
print("Loading models...")

yolo=YOLO(YOLO_MODEL)

lane_model=AttentionUNet(num_classes=3).to(DEVICE)
checkpoint=torch.load(LANE_MODEL,map_location=DEVICE)
if isinstance(checkpoint,dict):
    if "model_state_dict" in checkpoint:
        checkpoint=checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        checkpoint=checkpoint["state_dict"]
new_state={k.replace("module.",""):v for k,v in checkpoint.items()}
lane_model.load_state_dict(new_state,strict=False)
lane_model.eval()

print("Models loaded.")


# ================= CONNECT CARLA =================
print("Connecting to CARLA...")
client=carla.Client("localhost",2000)
client.set_timeout(60.0)
world=client.get_world()

settings=world.get_settings()
settings.synchronous_mode=False
world.apply_settings(settings)

world.set_weather(carla.WeatherParameters(
    cloudiness=5, precipitation=0, sun_altitude_angle=85
))

bp=world.get_blueprint_library()
world_map=world.get_map()


# ================= SPAWN VEHICLE =================
vehicle_bp=random.choice(bp.filter("vehicle.*"))
spawn=random.choice(world_map.get_spawn_points())
vehicle=world.spawn_actor(vehicle_bp,spawn)
vehicle.set_autopilot(False)


# ================= ROUTE =================
grp=GlobalRoutePlanner(world_map,2.0)
end=random.choice(world_map.get_spawn_points())
route=grp.trace_route(spawn.location,end.location)


# ================= SENSOR TRANSFORMS =================
bbox=vehicle.bounding_box.extent
cam_tf=carla.Transform(
    carla.Location(x=bbox.x+1,z=bbox.z+0.8),
    carla.Rotation(pitch=-10)
)
front_tf=carla.Transform(
    carla.Location(x=bbox.x+0.5,z=bbox.z+0.5),
    carla.Rotation(pitch=0)
)


# ================= RGB CAMERA =================
cam_bp=bp.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x","800")
cam_bp.set_attribute("image_size_y","450")
camera=world.spawn_actor(cam_bp,cam_tf,attach_to=vehicle)

rgb_queue=deque(maxlen=1)
def rgb_cb(img):
    arr=np.reshape(np.copy(img.raw_data),(img.height,img.width,4))
    rgb_queue.append(np.ascontiguousarray(arr[:,:,:3]))
camera.listen(rgb_cb)


# ================= SEMANTIC CAMERA =================
sem_bp=bp.find("sensor.camera.semantic_segmentation")
sem_bp.set_attribute("image_size_x","800")
sem_bp.set_attribute("image_size_y","450")
sem_cam=world.spawn_actor(sem_bp,cam_tf,attach_to=vehicle)

sem_queue=deque(maxlen=1)
def sem_cb(img):
    arr=np.reshape(np.copy(img.raw_data),(img.height,img.width,4))
    sem_queue.append(np.ascontiguousarray(arr[:,:,:3]))
sem_cam.listen(sem_cb)


# =====================================================================
# SENSOR 1 — LiDAR
# FIX: Three filters added to stop false positives from trees/benches/curbs:
#   1. Narrow cone from 40° to 15° — only looks directly ahead on the road
#   2. Height filter z > -1.0 AND z < 2.0 — ignores ground AND tall trees
#   3. Minimum point cluster of 5 — single stray reflections ignored
# =====================================================================
lidar_bp=bp.find("sensor.lidar.ray_cast")
lidar_bp.set_attribute("range","50")
lidar_bp.set_attribute("points_per_second","100000")
lidar_bp.set_attribute("channels","32")
lidar_bp.set_attribute("rotation_frequency","20")
lidar_bp.set_attribute("upper_fov","5")    # FIX: was 10, now 5 — ignores high objects
lidar_bp.set_attribute("lower_fov","-20")  # FIX: was -30, now -20 — less ground noise

lidar_tf=carla.Transform(
    carla.Location(x=bbox.x,z=bbox.z*2+0.3)
)
lidar_sensor=world.spawn_actor(lidar_bp,lidar_tf,attach_to=vehicle)

lidar_dist=deque(maxlen=1)
def lidar_cb(data):
    pts=np.frombuffer(data.raw_data,dtype=np.float32).reshape(-1,4)

    # FIX 1: Only forward points (x>0.5 avoids vehicle body reflections)
    fwd=pts[(pts[:,0]>0.5)]

    # FIX 2: Height filter — ignore ground (z<-1.0) AND trees/signs (z>2.0)
    fwd=fwd[(fwd[:,2]>-1.0) & (fwd[:,2]<2.0)]

    if len(fwd)==0:
        lidar_dist.append(999.0)
        return

    # FIX 3: Narrow to 15-degree cone (was 20) — road only, not sidewalk
    angles=np.abs(np.arctan2(fwd[:,1],fwd[:,0]))*180/np.pi
    cone=fwd[angles<15]

    # FIX 4: Require at least 5 points in cone — ignore single stray reflections
    if len(cone)<5:
        lidar_dist.append(999.0)
        return

    dists=np.sqrt(cone[:,0]**2+cone[:,1]**2)
    lidar_dist.append(float(np.min(dists)))

lidar_sensor.listen(lidar_cb)


radar_bp=bp.find("sensor.other.radar")
radar_bp.set_attribute("horizontal_fov","20")  # FIX: narrowed from 30 to 20
radar_bp.set_attribute("vertical_fov","10")
radar_bp.set_attribute("range","30")
radar_bp.set_attribute("points_per_second","1500")
radar_sensor=world.spawn_actor(radar_bp,front_tf,attach_to=vehicle)

radar_closing=deque(maxlen=1)
def radar_cb(data):
    max_closing=0.0
    for det in data:
        if det.velocity<0:
            max_closing=max(max_closing,abs(det.velocity))
    radar_closing.append(max_closing)
radar_sensor.listen(radar_cb)


# =====================================================================
# SENSOR 3 — COLLISION SENSOR
# =====================================================================
col_bp=bp.find("sensor.other.collision")
col_sensor=world.spawn_actor(col_bp,carla.Transform(),attach_to=vehicle)

collision_flag={"active":False,"timer":0.0}
def col_cb(event):
    collision_flag["active"]=True
    collision_flag["timer"]=time.time()
    print(f"[COLLISION] Hit: {event.other_actor.type_id}")
col_sensor.listen(col_cb)


# =====================================================================
# SENSOR 4 — LANE INVASION SENSOR
# =====================================================================
lane_bp=bp.find("sensor.other.lane_invasion")
lane_sensor=world.spawn_actor(lane_bp,carla.Transform(),attach_to=vehicle)

lane_invasion={"active":False,"timer":0.0,"side":"none"}
def lane_cb(event):
    for marking in event.crossed_lane_markings:
        mtype=str(marking.type)
        if "Solid" in mtype or "SolidSolid" in mtype:
            lane_invasion.update({"active":True,"timer":time.time(),"side":"solid"})
        elif "Broken" in mtype:
            lane_invasion.update({"active":True,"timer":time.time(),"side":"broken"})
lane_sensor.listen(lane_cb)


time.sleep(1.5)
print("Autonomous Running — All Sensors Active...")


# ================= LANE VISUALIZATION =================
def draw_lane_overlay(frame,lane_mask,road_mask):
    overlay=frame.copy()
    road_colored=np.zeros_like(frame)
    road_colored[road_mask>0]=[180,210,0]
    overlay=cv2.addWeighted(overlay,1.0,road_colored,0.35,0)
    lane_colored=np.zeros_like(frame)
    lane_colored[lane_mask>0]=[0,0,255]
    overlay=cv2.addWeighted(overlay,1.0,lane_colored,0.85,0)
    lane_thick=cv2.dilate(lane_mask,np.ones((3,3),np.uint8),iterations=1)
    overlay[lane_thick>0]=[0,0,220]
    return overlay


# ================= LOOP =================
idx=0
frame_count=0
last_obstacle_detected=False
last_pothole_detected=False

# FIX: Stuck recovery — tracks how long car has been near-zero speed
stuck_timer=time.time()
stuck_speed_threshold=1.0   # km/h — below this we consider it possibly stuck
stuck_timeout=4.0            # seconds — if stuck longer than this, force move

try:
    while True:

        if not rgb_queue:
            continue

        frame=rgb_queue.popleft()
        frame_count+=1

        if idx>=len(route):
            vehicle.apply_control(carla.VehicleControl(brake=1))
            break

        wp=route[idx][0]
        if vehicle.get_location().distance(wp.transform.location)<3:
            idx+=1
            continue

        raw_steer=compute_route_steer(vehicle,wp)
        steer=0.5*prev_steer+0.5*raw_steer
        steer=np.clip(steer,-1,1)
        prev_steer=steer

        v=vehicle.get_velocity()
        speed=3.6*np.sqrt(v.x**2+v.y**2+v.z**2)
        speed_error=MAX_SPEED-speed
        throttle=float(np.clip(speed_error/MAX_SPEED,0.0,0.7))
        brake=0.0


        # ===== TURN SPEED CONTROL =====
        # FIX: Removed brake on turns — only reduce throttle.
        # Previously brake=0.3 on sharp turns was stopping the car mid-corner.
        turn_strength=abs(steer)
        if turn_strength>0.3:
            throttle=min(throttle,0.35)
            brake=0.0
        if turn_strength>0.6:
            throttle=min(throttle,0.25)
            brake=0.0    # FIX: was brake=0.3, now 0.0 — no braking on turns
        # ==============================


        # ===== TRAFFIC LIGHT =====
        traffic_light_stop=False
        if vehicle.is_at_traffic_light():
            tl=vehicle.get_traffic_light()
            tl_state=tl.get_state()
            if tl_state==carla.TrafficLightState.Red:
                traffic_light_stop=True
                tl_label="RED - STOP"; tl_color=(0,0,255)
            elif tl_state==carla.TrafficLightState.Yellow:
                traffic_light_stop=True
                tl_label="YELLOW - READY"; tl_color=(0,165,255)
            else:
                tl_label="GREEN - GO"; tl_color=(0,255,0)
            cv2.putText(frame,tl_label,(20,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,tl_color,3)
        if traffic_light_stop:
            throttle=0.0; brake=1.0
        # =========================


        # ===== COLLISION — Emergency brake =====
        if collision_flag["active"]:
            if time.time()-collision_flag["timer"]<1.5:
                throttle=0.0; brake=1.0
                cv2.putText(frame,"!! COLLISION !!",(20,130),
                            cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)
            else:
                collision_flag["active"]=False
        # =======================================


        # ===== LIDAR — Distance-based braking =====
        # FIX: Only trigger LiDAR brake when NOT in a turn.
        # On turns (steer>0.3), LiDAR was detecting walls/barriers on the
        # side as forward obstacles and stopping the car mid-corner.
        lidar_override=False
        nearest=lidar_dist[-1] if lidar_dist else 999.0

        if not traffic_light_stop and not collision_flag["active"] and turn_strength<0.3:
            if nearest<LIDAR_STOP_DIST:
                throttle=0.0; brake=0.9
                lidar_override=True
                cv2.putText(frame,f"LIDAR: STOP {nearest:.1f}m",(20,170),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            elif nearest<LIDAR_SLOW_DIST:
                scale=(nearest-LIDAR_STOP_DIST)/(LIDAR_SLOW_DIST-LIDAR_STOP_DIST)
                throttle=float(np.clip(throttle*scale,0.0,0.5))
                brake=float(np.clip((1.0-scale)*0.3,0.0,0.3))
                lidar_override=True
                cv2.putText(frame,f"LIDAR: SLOW {nearest:.1f}m",(20,170),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,165,255),2)
        # ==========================================


        # ===== RADAR — Closing speed brake =====
        if not lidar_override and not traffic_light_stop and not collision_flag["active"]:
            closing=radar_closing[-1] if radar_closing else 0.0
            if closing>RADAR_CLOSE_MS:
                radar_scale=np.clip(1.0-(closing-RADAR_CLOSE_MS)/10.0,0.1,1.0)
                throttle=float(throttle*radar_scale)
                brake=float(np.clip((1.0-radar_scale)*0.5,0.0,0.5))
                cv2.putText(frame,f"RADAR: {closing:.1f}m/s closing",(20,210),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,100,255),2)
        # ========================================


        # ===== YOLO OBSTACLE + POTHOLE =====
        if frame_count%3==0:
            last_obstacle_detected=False
            last_pothole_detected=False
            res=yolo(frame,device=0,verbose=False)[0]
            for b in res.boxes:
                x1,y1,x2,y2=[int(v) for v in b.xyxy[0]]
                cls_id  =int(b.cls[0])
                cls_name=res.names[cls_id]
                conf    =float(b.conf[0])
                label   =f"{cls_name} {conf:.2f}"
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,label,(x1,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)
                area    =(x2-x1)*(y2-y1)
                center_x=(x1+x2)/2
                if area>25000 and abs(center_x-frame.shape[1]/2)<150:
                    last_obstacle_detected=True
                if "pothole" in cls_name.lower() and abs(center_x-frame.shape[1]/2)<200:
                    last_pothole_detected=True

        if last_obstacle_detected:
            throttle=0; brake=0.8

        if last_pothole_detected and not last_obstacle_detected:
            pothole_speed_error=15-speed
            throttle=float(np.clip(pothole_speed_error/15,0.0,0.4))
            brake=0.1 if speed>15 else 0.0
            cv2.putText(frame,"POTHOLE - SLOWING",(20,90),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,165,255),2)
        # ====================================


        # ===== LANE INVASION — Corrective steer =====
        if lane_invasion["active"]:
            if time.time()-lane_invasion["timer"]<0.8:
                correction=0.08 if lane_invasion["side"]=="broken" else 0.15
                steer=float(np.clip(steer+correction,-1,1))
                lc=(0,200,255) if lane_invasion["side"]=="broken" else (0,0,255)
                cv2.putText(frame,"LANE INVASION - CORRECTING",(20,250),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,lc,2)
            else:
                lane_invasion["active"]=False
        # ============================================


        # ===== FIX: STUCK RECOVERY =====
        # If car has been near-zero speed for 4 seconds with no valid reason,
        # it means LiDAR/radar false-positive stopped it — force it to move.
        # Resets timer whenever car is actually moving normally.
        real_stop_reason=(traffic_light_stop or
                          collision_flag["active"] or
                          last_obstacle_detected or
                          (nearest<LIDAR_STOP_DIST and turn_strength<0.3))

        if speed>stuck_speed_threshold:
            stuck_timer=time.time()   # reset timer — car is moving fine

        if not real_stop_reason and speed<stuck_speed_threshold:
            if time.time()-stuck_timer>stuck_timeout:
                # Car is stuck with no real reason — override and move
                throttle=0.5
                brake=0.0
                cv2.putText(frame,"STUCK RECOVERY",(20,290),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
        # ================================


        # ===== LANE KEEPING via Semantic Camera =====
        if sem_queue:
            sem_raw=sem_queue[-1]
            class_map=sem_raw[:,:,2]

            lane_mask=(class_map==6).astype(np.uint8)*255
            road_mask=(class_map==7).astype(np.uint8)*255

            frame=draw_lane_overlay(frame,lane_mask,road_mask)

            h=lane_mask.shape[0]
            lane_bottom=lane_mask[h//2:,:]
            road_bottom=road_mask[h//2:,:]

            lane_cols=np.where(lane_bottom>0)[1]
            road_cols=np.where(road_bottom>0)[1]

            sem_width        =sem_raw.shape[1]
            sem_center       =sem_width/2
            right_lane_target=sem_width*0.75

            if len(lane_cols)>30:
                right_lane_cols=lane_cols[lane_cols>=sem_center]
                if len(right_lane_cols)>10:
                    right_boundary=(sem_center+np.mean(right_lane_cols))/2
                    error=(right_boundary-right_lane_target)/sem_width
                elif len(road_cols)>50:
                    error=(np.mean(road_cols)-right_lane_target)/sem_width
                else:
                    error=0.0
            elif len(road_cols)>50:
                error=(np.mean(road_cols)-right_lane_target)/sem_width
            else:
                error=0.0

            if abs(steer)<0.15 and abs(error)>0.02:
                steer+=error*0.15
                steer=np.clip(steer,-1,1)

            cx=int(frame.shape[1]*0.75)
            cv2.line(frame,(cx,frame.shape[0]),(cx,frame.shape[0]//2),(0,255,255),2)
        # ============================================


        vehicle.apply_control(carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        ))


        # ===== HUD =====
        cv2.putText(frame,f"Speed: {speed:.1f} km/h",
                    (20,frame.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.putText(frame,f"LiDAR: {nearest:.1f}m",
                    (620,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,200),2)
        closing_disp=radar_closing[-1] if radar_closing else 0.0
        cv2.putText(frame,f"Radar: {closing_disp:.1f}m/s",
                    (620,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2)
        # ===============

        cv2.imshow("RGB + All Sensors + Lane",frame)
        if sem_queue:
            cv2.imshow("Semantic View",sem_queue[-1].copy())
        if cv2.waitKey(1)==ord("q"):
            break


finally:
    print("Cleaning up...")
    for s in [lidar_sensor,radar_sensor,col_sensor,lane_sensor,camera,sem_cam]:
        s.stop(); s.destroy()
    vehicle.destroy()
    cv2.destroyAllWindows()
    print("Done.")