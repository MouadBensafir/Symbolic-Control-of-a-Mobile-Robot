import pybullet as p
import pybullet_data
import numpy as np
import pickle
import math
import time
import os

# -----------------------------------------
# Load trajectory + regions from pickle
# -----------------------------------------
with open("sim-traj.pkl", "rb") as f:
    data = pickle.load(f)
traj = data 

with open("sim-regions.pkl", "rb") as f:
    data = pickle.load(f)
regions = data 


print(f"Loaded {len(traj)} trajectory points.")
print(f"Loaded {len(regions)} regions.")

# -----------------------------------------
# Setup PyBullet
# -----------------------------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0, 0, -9.81)

# Ground
plane = p.loadURDF("plane.urdf")

# Floor texture
texture_path = os.path.abspath("Assets/floor.png")
if os.path.exists(texture_path):
    texture_id = p.loadTexture(texture_path)
    p.changeVisualShape(plane, -1, textureUniqueId=texture_id)

# Camera
p.resetDebugVisualizerCamera(
    cameraDistance=7,
    cameraYaw=0,
    cameraPitch=-75,
    cameraTargetPosition=[5, 4, 0]
)

# -----------------------------------------
# Create regions
# -----------------------------------------
def hex_to_rgba(hex_color, alpha=0.5):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return [r, g, b, alpha]

def create_region_box(x_range, y_range, name, color="#ffaaaa"):
    x_min, x_max = x_range
    y_min, y_max = y_range
    cx = (x_min + x_max)/2
    cy = (y_min + y_max)/2
    size_x = (x_max - x_min)/2
    size_y = (y_max - y_min)/2
    size_z = 0.05

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size_x, size_y, size_z])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size_x, size_y, size_z],
                              rgbaColor=hex_to_rgba(color))
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                      basePosition=[cx, cy, size_z])

for i, (name, bounds) in enumerate(regions.items()):
    x_range = bounds[0]
    y_range = bounds[1]
    # Pick a nice color (cycling)
    color = ["#fdf7bb", "#d8f2f9", "#dbf5b9", "#ffb99d"][i % 4]
    create_region_box(x_range, y_range, name, color)

# -----------------------------------------
# Create walls
# -----------------------------------------
def create_wall(x, y, length, thickness, height):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, thickness/2, height/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, thickness/2, height/2],
                              rgbaColor=[0.1,0.1,0.1,1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                      basePosition=[x, y, height/2])
    
def create_point(x, y, radius, height):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                              rgbaColor=[0.1,0.1,0.7,1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                      basePosition=[x, y, height/2])
    

MAP_MIN = -0.5
MAP_MAX = 10.5
WALL_T = 0.05
H = 0.25

create_wall((MAP_MIN + MAP_MAX)/2, MAP_MAX + WALL_T/2, MAP_MAX - MAP_MIN, WALL_T, H)
create_wall((MAP_MIN + MAP_MAX)/2, MAP_MIN - WALL_T/2, MAP_MAX - MAP_MIN, WALL_T, H)
create_wall(MAP_MIN - WALL_T/2, (MAP_MIN + MAP_MAX)/2, WALL_T, MAP_MAX - MAP_MIN, H)
create_wall(MAP_MAX + WALL_T/2, (MAP_MIN + MAP_MAX)/2, WALL_T, MAP_MAX - MAP_MIN, H)

# -----------------------------------------
# Load Husky robot
# -----------------------------------------
robot_path = "husky/husky.urdf"
robot = p.loadURDF(robot_path, basePosition=[0,0,0], useFixedBase=False, globalScaling=0.5)

# -----------------------------------------
# Animation parameters
# -----------------------------------------
FPS = 60
dt = 1.0 / FPS

# Initialize robot
x, y, a = traj[0]
p.resetBasePositionAndOrientation(robot, [x, y, 0], p.getQuaternionFromEuler([0, 0, a]))
p.stepSimulation()

# -----------------------------------------
# Animation loop (smooth rotation near goal)
# -----------------------------------------
for i in range(len(traj)-1):
    x1, y1, a1 = traj[i]
    x2, y2, a2 = traj[i+1]
    
    create_point(x1, y1, radius=0.05, height=0.1)  # mark goal point

    # Start rotation a few steps before reaching goal
    da = ((a2 - a1 + math.pi) % (2*math.pi)) - math.pi

    steps = 20
    blend_start = 0.65  

    for alpha in np.linspace(0, 1, steps):
        x = x1*(1-alpha) + x2*alpha
        y = y1*(1-alpha) + y2*alpha

        if alpha < blend_start:
            ang = a1
        else:
            t = (alpha - blend_start)/(1 - blend_start)
            ang = a1 + t*da

        quat = p.getQuaternionFromEuler([0,0,ang])
        p.resetBasePositionAndOrientation(robot, [x,y,0], quat)
        p.stepSimulation()
        time.sleep(dt)

print("Simulation finished!")

# Keep window open
while True:
    p.stepSimulation()
    time.sleep(0.1)
