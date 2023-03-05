import open3d as o3d
import h5py
from pathlib import Path

HOME = str(Path.home())
print("HOME: ", HOME)

config = {
    "mesh_dir": HOME
    + "/Documents/isaac-codes/Grasping_task/imitation_learning/cup/cup.h5",
    "obj_dir": HOME
    + "/Documents/isaac-codes/Grasping_task/imitation_learning/cup/cup.obj",
}


def mesh_data(mesh_dir):
    mesh = h5py.File(mesh_dir, "r")
    success_idcs = mesh["grasps"]["qualities"]["flex"]["object_in_gripper"][()]
    success_idcs == 1
    grasp_pts = mesh["grasps"]["transforms"][()]
    success_grasps = grasp_pts[success_idcs.nonzero()]
    obj_pos = mesh["object"]["com"][()]
    obj_scale = mesh["object"]["scale"][()]
    obj_mass = mesh["object"]["mass"][()]
    return success_grasps, obj_pos, obj_scale, obj_mass


suc_grasps, object_position, object_scale, object_mass = mesh_data(config["mesh_dir"])
print("object position: ", object_position)
textured_mesh = o3d.io.read_triangle_mesh(
    "/home/mokhtars/Documents/isaac-codes/Grasping_task/imitation_learning/cup/cup.obj"
)

textured_mesh.compute_vertex_normals()

# Scale and Translation

print("Initial center: ", textured_mesh.get_center())
initial_pos = textured_mesh.get_center()
initial_transl = -initial_pos
print("Initial translation: ", initial_transl)
textured_mesh.translate(initial_transl)
textured_mesh.scale(object_scale, center=(0, 0, 0))
print("Second center (supposed to be zero): ", textured_mesh.get_center())
textured_mesh.translate(
    object_position
)  # comment this line if you want to put the mesh at (0, 0, 0)
print("Final center: ", textured_mesh.get_center())

# Grasps
l = [textured_mesh]

for i in range(30):
    grasp = o3d.geometry.TriangleMesh.create_coordinate_frame()
    initial_grasp_pos = grasp.get_center()
    initial_grasp_transl = -initial_grasp_pos
    grasp.translate(initial_grasp_transl)
    grasp.scale(2 * object_scale, center=(0, 0, 0))
    grasp.transform(suc_grasps[i * 20])
    l.append(grasp)


# Visualization
o3d.visualization.draw_geometries(l)
