import trimesh
import pyrender
from os import path, listdir
path_root = "/home/luo/workspace/thesis_ws/SoftRas/data/YCB"
for obj in listdir(path_root):
    print(obj)
    # if obj in ("019_pitcher_base","025_mug","051_large_clamp","024_bowl","037_scissors","052_extra_large_clamp"):
    scene = pyrender.Scene()
    path_obj = path.join(path_root,obj,"textured_simple.obj")
    fuze_trimesh = trimesh.load(path_obj)
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)