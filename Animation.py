import pybullet as p
import pybullet_data
import numpy as np
import math
import time
import os
import imageio


def hex_to_rgba(hex_color, alpha=0.5):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return [r, g, b, alpha]


def create_region_box(x_range, y_range, color, p_client):
    x_min, x_max = x_range
    y_min, y_max = y_range
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    size_x = (x_max - x_min) / 2
    size_y = (y_max - y_min) / 2
    size_z = 0.05

    col = p_client.createCollisionShape(p.GEOM_BOX, halfExtents=[size_x, size_y, size_z])
    vis = p_client.createVisualShape(
        p.GEOM_BOX, halfExtents=[size_x, size_y, size_z], rgbaColor=hex_to_rgba(color)
    )
    p_client.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[cx, cy, size_z]
    )


def create_wall(x, y, length, thickness, height, p_client):
    col = p_client.createCollisionShape(
        p.GEOM_BOX, halfExtents=[length / 2, thickness / 2, height / 2]
    )
    vis = p_client.createVisualShape(
        p.GEOM_BOX, halfExtents=[length / 2, thickness / 2, height / 2], rgbaColor=[0.1, 0.1, 0.1, 1]
    )
    p_client.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[x, y, height / 2]
    )


def create_point(x, y, radius, height, p_client):
    col = p_client.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p_client.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=[0.1, 0.1, 0.7, 1])
    p_client.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[x, y, height / 2]
    )


def animate(traj, regions, assets_dir="Assets", robot_urdf="husky/husky.urdf", fps=60):
    """Animate a trajectory using PyBullet.

    traj: list of (x,y,angle)
    regions: dict mapping name->[(x_min,x_max),(y_min,y_max),...]
    """
    if len(traj) == 0:
        return
    
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    except Exception:
        pass
    p.setGravity(0, 0, -9.81)

    # Ground
    plane = p.loadURDF("plane.urdf")

    # Floor texture
    texture_path = os.path.join(assets_dir, "floor.png")
    if os.path.exists(texture_path):
        try:
            texture_id = p.loadTexture(texture_path)
            p.changeVisualShape(plane, -1, textureUniqueId=texture_id)
        except Exception:
            pass

    p.resetDebugVisualizerCamera(cameraDistance=7, cameraYaw=0, cameraPitch=-75, cameraTargetPosition=[5, 4, 0])

    # Create regions
    for i, (name, bounds) in enumerate(regions.items()):
        x_range = bounds[0]
        y_range = bounds[1]
        color = ["#fdf7bb", "#d8f2f9", "#dbf5b9", "#ffb99d"][i % 4]
        create_region_box(x_range, y_range, color, p)

    # Walls
    MAP_MIN = -0.5
    MAP_MAX = 10.5
    WALL_T = 0.05
    H = 0.25

    create_wall((MAP_MIN + MAP_MAX) / 2, MAP_MAX + WALL_T / 2, MAP_MAX - MAP_MIN, WALL_T, H, p)
    create_wall((MAP_MIN + MAP_MAX) / 2, MAP_MIN - WALL_T / 2, MAP_MAX - MAP_MIN, WALL_T, H, p)
    create_wall(MAP_MIN - WALL_T / 2, (MAP_MIN + MAP_MAX) / 2, WALL_T, MAP_MAX - MAP_MIN, H, p)
    create_wall(MAP_MAX + WALL_T / 2, (MAP_MIN + MAP_MAX) / 2, WALL_T, MAP_MAX - MAP_MIN, H, p)

    # Load robot
    robot = p.loadURDF(robot_urdf, basePosition=[0, 0, 0], useFixedBase=False, globalScaling=0.5)

    dt = 1.0 / fps

    x0, y0, a0 = traj[0]
    p.resetBasePositionAndOrientation(robot, [x0, y0, 0], p.getQuaternionFromEuler([0, 0, a0]))
    p.stepSimulation()

    # Animation loop
    for i in range(len(traj) - 1):
        x1, y1, a1 = traj[i]
        x2, y2, a2 = traj[i + 1]

        create_point(x1, y1, radius=0.05, height=0.1, p_client=p)

        da = ((a2 - a1 + math.pi) % (2 * math.pi)) - math.pi

        steps = 20
        blend_start = 0.65

        for alpha in np.linspace(0, 1, steps):
            x = x1 * (1 - alpha) + x2 * alpha
            y = y1 * (1 - alpha) + y2 * alpha

            if alpha < blend_start:
                ang = a1
            else:
                t = (alpha - blend_start) / (1 - blend_start)
                ang = a1 + t * da

            if robot is not None:
                quat = p.getQuaternionFromEuler([0, 0, ang])
                p.resetBasePositionAndOrientation(robot, [x, y, 0], quat)
            p.stepSimulation()
            time.sleep(dt)

    print("Animation finished.")

    try:
        while True:
            p.stepSimulation()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    # Disconnect
    try:
        p.disconnect()
    except Exception:
        pass


#==================================
#       Animation for GUI
#==================================
def animate_gui(traj, regions,save_path, assets_dir="Assets", robot_urdf="husky/husky.urdf", fps=60):
    if p.isConnected():
        p.disconnect()
    
    p.connect(p.DIRECT) # no window

    # Initialization

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    except Exception:
        pass
    p.setGravity(0, 0, -9.81)

    # Ground
    plane = p.loadURDF("plane.urdf")

    # Floor texture
    texture_path = os.path.join(assets_dir, "floor.png")
    if os.path.exists(texture_path):
        try:
            texture_id = p.loadTexture(texture_path)
            p.changeVisualShape(plane, -1, textureUniqueId=texture_id)
        except Exception:
            pass

    p.resetDebugVisualizerCamera(cameraDistance=7, cameraYaw=0, cameraPitch=-75, cameraTargetPosition=[5, 4, 0])

    # Create regions
    for i, (name, bounds) in enumerate(regions.items()):
        x_range = bounds[0]
        y_range = bounds[1]
        color = ["#fdf7bb", "#d8f2f9", "#dbf5b9", "#ffb99d"][i % 4]
        create_region_box(x_range, y_range, color, p)

    # Walls
    MAP_MIN = -0.5
    MAP_MAX = 10.5
    WALL_T = 0.05
    H = 0.25

    create_wall((MAP_MIN + MAP_MAX) / 2, MAP_MAX + WALL_T / 2, MAP_MAX - MAP_MIN, WALL_T, H, p)
    create_wall((MAP_MIN + MAP_MAX) / 2, MAP_MIN - WALL_T / 2, MAP_MAX - MAP_MIN, WALL_T, H, p)
    create_wall(MAP_MIN - WALL_T / 2, (MAP_MIN + MAP_MAX) / 2, WALL_T, MAP_MAX - MAP_MIN, H, p)
    create_wall(MAP_MAX + WALL_T / 2, (MAP_MIN + MAP_MAX) / 2, WALL_T, MAP_MAX - MAP_MIN, H, p)

    # Load robot
    robot = p.loadURDF(robot_urdf, basePosition=[0, 0, 0], useFixedBase=False, globalScaling=0.5)

    dt = 1.0 / fps

    x0, y0, a0 = traj[0]
    p.resetBasePositionAndOrientation(robot, [x0, y0, 0], p.getQuaternionFromEuler([0, 0, a0]))
    p.stepSimulation()

    # Animation loop
    frames = []

    for i in range(len(traj) - 1):
        x1, y1, a1 = traj[i]
        x2, y2, a2 = traj[i + 1]

        create_point(x1, y1, radius=0.05, height=0.1, p_client=p)

        da = ((a2 - a1 + math.pi) % (2 * math.pi)) - math.pi

        steps = 10
        blend_start = 0.65

        for alpha in np.linspace(0, 1, steps):
            x = x1 * (1 - alpha) + x2 * alpha
            y = y1 * (1 - alpha) + y2 * alpha

            if alpha < blend_start:
                ang = a1
            else:
                t = (alpha - blend_start) / (1 - blend_start)
                ang = a1 + t * da

            if robot is not None:
                quat = p.getQuaternionFromEuler([0, 0, ang])
                p.resetBasePositionAndOrientation(robot, [x, y, 0], quat)
            
            p.stepSimulation()

            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[5, 5, 14], # Adjust based on arena size
                cameraTargetPosition=[5, 5, 0],
                cameraUpVector=[0, 1, 0]
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
            )
        
            width, height, rgbImg, _, _ = p.getCameraImage(
                width=320, height=320, # Keep small for speed
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_TINY_RENDERER # CPU-based renderer (works everywhere)
            )
            frame = np.reshape(rgbImg, (height, width, 4))[:,:,:3].astype(np.uint8)
            frames.append(frame)

    # Disconnect
    try:
        p.disconnect()
    except Exception:
        pass

    # Save GIF
    imageio.mimsave(save_path, frames, fps=30, loop=0)
    return save_path


