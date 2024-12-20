import os
from pathlib import Path
import numpy as np
from utils.helpers import mesh_data, get_usd_and_h5_paths

# Set up Home Directory
HOME = str(Path.cwd())

def initialize_world(world, ground_plane=False):
    world_settings = {
        "stage_units_in_meters": 1.0,
        "physics_dt": 1.0 / 40.0,
        "rendering_dt": 1.0 / 40.0,
    }
    my_world = world(**world_settings)
    if ground_plane:
        my_world.scene.add_default_ground_plane()
    return my_world

def initialize_robot(my_world, robot, add_reference_to_stage):
    robot_path = os.path.join(HOME, "bc_files/fmm.usd")
    add_reference_to_stage(usd_path=robot_path, prim_path="/World/FMM")
    robot_sim = my_world.scene.add(robot(prim_path="/World/FMM", name="fmm", position=[0, 0, 0]))
    return robot_sim

def initialize_environment(my_world, add_reference_to_stage, XFormPrim):
    hospital_usd_path = os.path.join(HOME, "bc_files/grasp_scene.usd")
    add_reference_to_stage(usd_path=hospital_usd_path, prim_path="/World/Hospital")
    my_world.scene.add(XFormPrim(prim_path="/World/Hospital", name="hospital", position=[0, 0, 0]))

def initialize_object_grasps(my_world, add_reference_to_stage, RigidPrim, XFormPrim, omni, utils):
    mesh_path = os.path.join(HOME, "bc_files/bowl.h5")
    suc_grasps, object_position, object_scale, object_mass = mesh_data(mesh_path)
    object_path = os.path.join(HOME, "bc_files/simple_bowl.usd")
    add_reference_to_stage(usd_path=object_path, prim_path="/World/Hospital/object")
    my_object = my_world.scene.add(RigidPrim(prim_path="/World/Hospital/object", name="fancy_bucket", position=(0, 0, 0), mass=object_mass * 100))
    my_object = my_world.scene.add(XFormPrim(prim_path="/World/Hospital/object", name="fancy_bowl", position=object_position, scale=(object_scale, object_scale, object_scale), visible=True))
    # Add PHYSICS to ShapeNet model
    stage = omni.usd.get_context().get_stage()
    prim = stage.DefinePrim("/World/Hospital/object", "Xform")
    shape_approximation = "convexDecomposition"
    utils.setRigidBody(prim, shape_approximation, False)
    return my_object, suc_grasps

def initialize_specific_object_grasps(obj_name, my_world, add_reference_to_stage, RigidPrim, XFormPrim, omni, utils):
    data_dir = os.path.join(HOME, "ValidGraspObjects", obj_name)
    usd_file, h5_file = get_usd_and_h5_paths(data_dir)
    suc_grasps, object_position, object_scale, object_mass = mesh_data(h5_file)
    add_reference_to_stage(
        usd_path=usd_file, prim_path="/World/Hospital/object"
    )
    my_object = my_world.scene.add(
        RigidPrim(
            prim_path="/World/Hospital/object",
            name="rigid_bowl",
            position=(0, 0, 0),
            mass=object_mass * 100,
        )
    )
    my_object = my_world.scene.add(
        XFormPrim(
            prim_path="/World/Hospital/object",
            name="fancy_bowl",
            position=object_position + np.array([0, 0, 0]),
            scale=(object_scale * 100, object_scale * 100, object_scale * 100),
            visible=True,
        )
    )
    # Add PHYSICS to ShapeNet model
    stage = omni.usd.get_context().get_stage()
    prim = stage.DefinePrim("/World/Hospital/object", "Xform")
    shape_approximation = "convexDecomposition"
    utils.setRigidBody(prim, shape_approximation, False)
    return my_object, suc_grasps    

def initialize_multiple_objects_grasps(my_world, add_reference_to_stage, RigidPrim, XFormPrim, omni, utils):
    # Initialize Objects
    obj_dir = os.path.join(HOME, "bc_files/ValidGraspObjects")
    obj_paths = [os.path.join(obj_dir, d) for d in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, d))]
    success_grasps = {}
    objects = []
    for i, obj_path in enumerate(obj_paths):
        x_ax = i + 60
        obj_usd, obj_h5 = get_usd_and_h5_paths(obj_path)
        suc_grasps, object_position, object_scale, object_mass = mesh_data(
            obj_h5
        )
        success_grasps[i] = suc_grasps
        add_reference_to_stage(
            obj_usd, prim_path="/World/Hospital/object{}".format(i)
        )
        object_name = "my_object{}".format(i)
        object_name = my_world.scene.add(
            RigidPrim(
                prim_path="/World/Hospital/object{}".format(i),
                name="fancy_obj{}".format(i),
                position=(0, 0, 0),
                mass=object_mass * 100,
            )
        )

        object_name = my_world.scene.add(
            XFormPrim(
                prim_path="/World/Hospital/object{}".format(i),
                name="fancy_object{}".format(i),
                position=object_position + np.array([x_ax, 1, 1]),
                scale=(object_scale * 100, object_scale * 100, object_scale * 100),
                visible=True,
            )
        )
        # Add PHYSICS to ShapeNet model
        stage = omni.usd.get_context().get_stage()
        prim = stage.DefinePrim("/World/Hospital/object{}".format(i), "Xform")
        shape_approximation = "convexDecomposition"
        utils.setRigidBody(prim, shape_approximation, False)
        objects.append(object_name)
    return objects, success_grasps





def start_simulation(my_world):
    my_world.reset()
    my_world.initialize_physics()
    my_world.play()

