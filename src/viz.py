import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import trimesh
import pyrender
from PIL import Image
from pathlib import Path
import json
from trimesh.transformations import quaternion_matrix, translation_matrix
from dotenv import load_dotenv
import pdb
from tqdm import tqdm
import time
import re
import traceback
from collections import defaultdict
from src.sample import AssetRetrievalModule
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import copy
import seaborn as sns
import glob
from scipy import stats
import pandas as pd
import cv2
import imageio.v2 as imageio
import math
import shutil
import subprocess

from src.utils import get_pth_mesh, create_floor_plan_polygon, remove_and_recreate_folder, precompute_fid_scores_for_caching, get_pths_dataset_split, get_model, get_test_instrs_all
from src.dataset import load_train_val_test_datasets, create_full_scene_from_before_and_added, create_instruction_from_scene, process_scene_sample

# Add this before your rendering code
import ctypes
from OpenGL.GL import glGenTextures
from OpenGL.GL import GLuint

# Monkey patch the problematic function
def patched_glGenTextures(count, textures):
    textures_array = (GLuint * count)()
    glGenTextures(count, textures_array)
    return textures_array[0]

# Replace the original function
import OpenGL.GL
OpenGL.GL.glGenTextures = patched_glGenTextures

def _to_rgba_uint8(img_like, tex_name="texture"):
    arr = np.asarray(img_like)

    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        if np.nanmax(arr_f) <= 1.0 + 1e-6:
            arr_f *= 255.0
        arr = np.clip(arr_f, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        # 灰度 -> RGBA
        rgba = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)

    elif arr.ndim == 3:
        h, w, c = arr.shape

        if c == 1:
            gray = arr[:, :, 0]
            rgba = np.stack([gray, gray, gray, np.full_like(gray, 255)], axis=-1)

        elif c == 2:
            # 更稳妥：按 LA 理解
            gray = arr[:, :, 0]
            alpha = arr[:, :, 1]
            rgba = np.stack([gray, gray, gray, alpha], axis=-1)

        elif c == 3:
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            rgba = np.concatenate([arr, alpha], axis=-1)

        elif c == 4:
            rgba = arr

        else:
            raise ValueError(f"{tex_name}: unsupported channel count {c}")
    else:
        raise ValueError(f"{tex_name}: unsupported ndim {arr.ndim}")

    return rgba

def fix_textures(mesh, mesh_path):
    if not hasattr(mesh, "visual"):
        return

    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return

    texture_fields = [
        "image",
        "baseColorTexture",
        "metallicRoughnessTexture",
        "normalTexture",
        "emissiveTexture",
        "occlusionTexture",
    ]

    for field in texture_fields:
        tex = getattr(mat, field, None)
        if tex is None:
            continue

        try:
            # 有些字段可能是 PIL.Image，有些是 numpy array
            rgba = _to_rgba_uint8(tex, tex_name=field)
            setattr(mat, field, rgba)
        except Exception as e:
            print(f"[viz] WARN Failed to normalize {field} for {mesh_path}: {e}")

    # 尽量保留材质，只补配置
    try:
        mat.doubleSided = True
    except Exception:
        pass

    try:
        mat.alphaMode = 'BLEND'
    except Exception:
        pass

def load_mesh_with_transform(mesh_path, position=None, rotation=None, scale=None):
    if mesh_path is None:
        return None
    if isinstance(mesh_path, (str, Path)) and not Path(str(mesh_path)).exists():
        print(f"[viz] WARN mesh file not found: {mesh_path}")
        return None

    mesh = trimesh.load(mesh_path)

    # Convert any 2-channel textures to RGBA and make materials double-sided
    if isinstance(mesh, trimesh.Scene):
        for m in mesh.geometry.values():
            fix_textures(m, mesh_path)
            if hasattr(m, 'visual') and hasattr(m.visual, 'material'):
                m.visual.material.doubleSided = True
                if hasattr(m.visual.material, 'alphaMode'):
                    m.visual.material.alphaMode = 'BLEND'
        # ★ 关键修复：把 Scene 内部所有 node transform 烘焙到顶点，合并为单一 Trimesh
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception as e:
            print(f"[viz] WARN dump(concatenate=True) failed for {mesh_path}: {e}")
            # fallback: 尝试只取最大的 geometry
            geoms = list(mesh.geometry.values())
            if geoms:
                mesh = max(geoms, key=lambda g: g.vertices.shape[0] if hasattr(g, 'vertices') else 0)
            else:
                return None
    else:
        fix_textures(mesh, mesh_path)
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
            mesh.visual.material.doubleSided = True
            if hasattr(mesh.visual.material, 'alphaMode'):
                mesh.visual.material.alphaMode = 'BLEND'

    # scale -> rotation -> translation（现在 mesh 一定是单个 Trimesh，apply 直接改顶点）
    if scale is not None:
        mesh.apply_scale(scale)

    if rotation is not None:
        # scene 约定: rot = [x, y, z, w]
        # quaternion_matrix 期望 [w, x, y, z]
        quat_wxyz = [rotation[3], rotation[0], rotation[1], rotation[2]]
        rot_mat = quaternion_matrix(quat_wxyz)
        mesh.apply_transform(rot_mat)

    if position is not None:
        mesh.apply_transform(translation_matrix(position))

    return mesh

def setup_camera(scene, resolution, view_type, use_dynamic_zoom, camera_height, scene_span, look_at_target=None):
    """Set up camera for rendering.
    
    look_at_target: (x, y, z) world position the camera should look at.
                    Defaults to (0, 0, 0) for backward compatibility.
    """
    fov = np.pi / 4.0
    camera = pyrender.PerspectiveCamera(yfov=np.pi/4.0, znear=0.05, zfar=100.0)
    scene_x, scene_y, scene_z = scene_span
    scene_aspect = scene_x / max(scene_z, 1e-5)
    
    if scene_aspect > 1.0:
        limiting_span = scene_x
    else:
        limiting_span = scene_z
    
    # Default look-at target
    if look_at_target is None:
        target = np.array([0.0, 0.0, 0.0])
    else:
        target = np.array(look_at_target, dtype=float)
    
    if view_type == "top":
        if camera_height == None:
            if use_dynamic_zoom:
                required_distance = (limiting_span/2) / np.tan(fov/2)
                camera_height = max(2.0, required_distance + 2.5)
            else:
                camera_height = 13.0

        # Camera positioned directly above the target point
        camera_pose = np.array([
            [1.0, 0.0, 0.0, target[0]],
            [0.0, 0.0, 1.0, camera_height],
            [0.0, -1.0, 0.0, target[2]],
            [0.0, 0.0, 0.0, 1.0]
        ])

    elif view_type == "diag":
        if camera_height == None:
            if use_dynamic_zoom:
                diagonal_length = np.sqrt(scene_x**2 + scene_y**2 + scene_z**2)
                required_distance = (diagonal_length/2) / np.tan(fov/2)
                camera_height = max(2.0, required_distance*0.8)
            else:
                camera_height = 10.0

        # Position relative to target
        position = np.array([
            target[0] + camera_height,
            camera_height,
            target[2] + camera_height,
        ])
        
        forward = target - position
        forward = forward / np.linalg.norm(forward)
        
        world_up = np.array([0.0, 1.0, 0.0])
        
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        camera_pose = np.eye(4)
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = up
        camera_pose[:3, 2] = -forward
        camera_pose[:3, 3] = position

    scene.add(camera, pose=camera_pose)
    return camera_pose

def setup_lighting(scene, camera_pose):
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    #light_pose = np.eye(4)
    #light_pose[:3, :3] = camera_pose[:3, :3]  # Keep camera orientation
    #light_pose[1, 3] = 3.0  # Move up by 3 units
    # light_pose = camera_pose.copy()
    # light_pose[1, 3] = 3.0
    light_pose = camera_pose.copy()
    light_pose[1, 3] = 3.0

    # If we're using OSMesa, adjust the transform to match Pyglet behavior
    # if os.environ.get('PYOPENGL_PLATFORM') == 'osmesa':
    #     # Try inverting certain rotations to match Pyglet's interpretation
    #     correction = np.array([
    #         [ 1,  0,  0,  0],
    #         [ 0,  0, -1,  3],  # Flip the forward direction
    #         [ 0,  1,  0,  0],
    #         [ 0,  0,  0,  1]
    #     ])
    #     light_pose = correction @ light_pose

    # print("Light pose matrix:")
    # print(light_pose)
    
    scene.add(light, pose=light_pose)
    
    scene.ambient_light = np.array([0.6, 0.6, 0.6, 1.0])

def create_bbox(size, pos, rot, color=[0.0, 0.0, 1.0, 0.7]):
    bbox = trimesh.creation.box(extents=size)

    material = trimesh.visual.material.PBRMaterial(baseColorFactor=color, alphaMode='BLEND', doubleSided=False, metallicFactor=0.0, roughnessFactor=1.0)
    bbox.visual = trimesh.visual.TextureVisuals(material=material, uv=bbox.vertices[:, [0, 1]])
    bbox.fix_normals(multibody=True)

    bottom_center_transform = np.eye(4)
    bottom_center_transform[1, 3] = size[1] / 2
    bbox.apply_transform(bottom_center_transform)
    
    if rot is not None:
        # Convert [x,y,z,w] to [w,x,y,z] for trimesh
        quat_wxyz = [rot[3], rot[0], rot[1], rot[2]]
        rotation_matrix = quaternion_matrix(quat_wxyz)
        bbox.apply_transform(rotation_matrix)
    
    if pos is not None:
        translation = translation_matrix(pos)
        bbox.apply_transform(translation)
    
    return bbox

def create_floor_slab(bounds_bottom):
    # bounds_bottom = [[-5, 0, -5], [5, 0, -5], [5, 0, 5], [-5, 0, 5]]
    floor_plan_polygon = create_floor_plan_polygon(bounds_bottom)
    
    floor_mesh = trimesh.creation.extrude_polygon(
        polygon=floor_plan_polygon,
        height=0.15
    )

    rotation = trimesh.transformations.rotation_matrix(
        angle=np.pi/2,
        direction=[1, 0, 0]
    )
    floor_mesh.apply_transform(rotation)
    
    try:
        img = Image.open('src/frontend/public/texture.png')
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=img
        )
        
        vertices = floor_mesh.vertices
        bounds = floor_mesh.bounds
        bounds_range = bounds[1] - bounds[0]
        dims = np.argsort(bounds_range)[-2:]
        
        uvs = np.zeros((len(vertices), 2))
        uvs[:, 0] = (vertices[:, dims[0]] - bounds[0][dims[0]]) / bounds_range[dims[0]]
        uvs[:, 1] = (vertices[:, dims[1]] - bounds[0][dims[1]]) / bounds_range[dims[1]]
        
        floor_mesh.visual = trimesh.visual.TextureVisuals(
            uv=uvs,
            material=material
        )
    except Exception as e:
        print(f"Failed to load texture: {e}")
        floor_mesh.visual.face_colors = [245, 222, 179, 178]
    
    return floor_mesh

def create_pyrender_scene_from_trimesh(trimesh_scene, bg_color=None):
    pyrender_scene = pyrender.Scene(bg_color=bg_color)

    # 用于去重 warning（避免每帧刷屏）
    warned = set()

    for node_name in trimesh_scene.graph.nodes_geometry:
        transform, geom_name = trimesh_scene.graph.get(node_name)
        geom = trimesh_scene.geometry[geom_name]

        try:
            pyrender_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
        except ValueError as e:
            msg = str(e)
            if "2-channel texture" in msg or "Cannot reformat 2-channel texture into RGBA" in msg:
                key = (geom_name, "2ch")
                if key not in warned:
                    print(f"[viz] WARN Mesh.from_trimesh failed for {geom_name}: {e} (fallback to flat color)")
                    warned.add(key)
                # 降级：移除纹理相关 visual，改为纯色，再尝试一次
                try:
                    geom2 = geom.copy()
                    geom2.visual = trimesh.visual.ColorVisuals(
                        mesh=geom2, vertex_colors=[200, 200, 200, 255]
                    )
                    pyrender_mesh = pyrender.Mesh.from_trimesh(geom2, smooth=False)
                except Exception as e2:
                    key2 = (geom_name, "fallback_failed")
                    if key2 not in warned:
                        print(f"[viz] WARN fallback flat-color failed for {geom_name}: {e2}")
                        warned.add(key2)
                    continue
            else:
                # 其他错误照旧跳过（但也别每帧刷屏）
                key = (geom_name, msg)
                if key not in warned:
                    print(f"[viz] WARN Mesh.from_trimesh failed for {geom_name}: {e}")
                    warned.add(key)
                continue

        if transform is None:
            transform = np.eye(4)
        else:
            transform = np.asarray(transform, dtype=float)
            if transform.shape != (4, 4):
                transform = np.eye(4)

        pyrender_scene.add(pyrender_mesh, pose=transform)

    return pyrender_scene

def add_objects_to_trimesh_scene(
    trimesh_scene,
    objects,
    show_bboxes: bool = False,
    show_assets: bool = True,
    show_assets_voxelized: bool = False,
    show_bounds: bool = False,
    bounds_bottom=None,
    bounds_top=None,
):
    """
    将 objects 添加到 trimesh.Scene：
    - show_assets: 加载 glb 资产
    - show_assets_voxelized: 体素化显示资产（若你工程已有 voxelize_mesh）
    - show_bboxes: 绘制 3D bbox 线框
    - show_bounds: 绘制房间 bounds_bottom/bounds_top
    """
    if objects is None:
        return

    # 1) 添加物体
    for i, obj in enumerate(objects):
        try:
            if not isinstance(obj, dict):
                continue

            # 注意：scene json 里通常是 jid；有的流程会用 sampled_asset_jid
            jid = obj.get("jid") if obj.get("jid") is not None else obj.get("sampled_asset_jid")
            pos = obj.get("pos")
            rot = obj.get("rot")
            scale = obj.get("scale")
            size = obj.get("size")

            # --- 资产 mesh ---
            mesh = None
            if show_assets or show_assets_voxelized:
                mesh_path = get_pth_mesh(jid)
                mesh = load_mesh_with_transform(mesh_path, pos, rot, scale)

                if mesh is not None:
                    # load_mesh_with_transform 现在保证返回单个 Trimesh（不是 Scene）
                    trimesh_scene.add_geometry(mesh, geom_name=f"obj{i}")

            # --- bbox 线框（不依赖资产）---
            if show_bboxes:
                try:
                    # 优先用 size+pos+rot 画 OBB；如果你工程里已有 get_xz_bbox_from_obj 等也可替换
                    if isinstance(size, (list, tuple)) and len(size) >= 3 and isinstance(pos, (list, tuple)) and len(pos) >= 3:
                        bbox = _make_bbox_lines_from_size_pos_rot(size=size, pos=pos, rot=rot)
                        if bbox is not None:
                            trimesh_scene.add_geometry(bbox, geom_name=f"bbox_{i}")
                except Exception as e:
                    print(f"[viz] WARN failed to add bbox for obj {i}: {e}")

        except Exception as e:
            print(f"Failed to add object {i}: {e}")
            traceback.print_exc()
            continue

    # 2) 添加房间边界（可选）
    if show_bounds:
        try:
            if bounds_bottom:
                b = _make_bounds_polyline(bounds_bottom, y=0.0, color=(0, 0, 0, 255))
                if b is not None:
                    trimesh_scene.add_geometry(b, geom_name="bounds_bottom")
            if bounds_top:
                t = _make_bounds_polyline(bounds_top, y=0.0, color=(0, 0, 0, 255))
                if t is not None:
                    trimesh_scene.add_geometry(t, geom_name="bounds_top")
        except Exception as e:
            print(f"[viz] WARN failed to add bounds: {e}")

def _save_rendered_image(color, output_file):
    """Save rendered RGB/RGBA result to disk."""
    output_file = str(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if output_file.lower().endswith(".png"):
        file_type = "PNG"
        if color.ndim == 3 and color.shape[2] == 3:
            alpha = (color.sum(axis=2) > 0).astype(np.uint8) * 255
            rgba = np.dstack((color, alpha))
            img = Image.fromarray(rgba, "RGBA")
        elif color.ndim == 3 and color.shape[2] == 4:
            img = Image.fromarray(color, "RGBA")
        else:
            img = Image.fromarray(color)
    else:
        file_type = "JPEG"
        if color.ndim == 3 and color.shape[2] == 4:
            img = Image.fromarray(color[:, :, :3], "RGB")
        else:
            img = Image.fromarray(color)

    if file_type == "JPEG":
        img.save(output_file, file_type, quality=95)
    else:
        img.save(output_file, file_type)


def _build_fresh_pyrender_scene_for_view(
    trimesh_scene,
    resolution,
    view_type,
    use_dynamic_zoom,
    camera_height,
    scene_span,
    bg_color=None,
    look_at_target=None,
):
    """
    Always build a brand-new pyrender.Scene from trimesh.Scene.
    This avoids reusing old Mesh objects across different OpenGL contexts.
    """
    pyrender_scene = create_pyrender_scene_from_trimesh(trimesh_scene, bg_color=bg_color)

    camera_pose = setup_camera(
        pyrender_scene,
        resolution,
        view_type,
        use_dynamic_zoom,
        camera_height,
        scene_span,
        look_at_target=look_at_target,
    )
    setup_lighting(pyrender_scene, camera_pose)
    return pyrender_scene, camera_pose


def _render_fresh_scene_once(
    trimesh_scene,
    resolution,
    view_type,
    use_dynamic_zoom,
    camera_height,
    scene_span,
    bg_color=None,
    look_at_target=None,
    flags=None,
):
    """Render one frame using a fresh scene + fresh renderer."""
    if flags is None:
        flags = (
            pyrender.RenderFlags.SKIP_CULL_FACES
            | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
            | pyrender.RenderFlags.RGBA
        )

    renderer = None
    try:
        pyrender_scene, camera_pose = _build_fresh_pyrender_scene_for_view(
            trimesh_scene=trimesh_scene,
            resolution=resolution,
            view_type=view_type,
            use_dynamic_zoom=use_dynamic_zoom,
            camera_height=camera_height,
            scene_span=scene_span,
            bg_color=bg_color,
            look_at_target=look_at_target,
        )
        renderer = pyrender.OffscreenRenderer(*resolution)
        color, depth = renderer.render(pyrender_scene, flags=flags)
        return color, depth, camera_pose
    finally:
        if renderer is not None:
            try:
                renderer.delete()
            except Exception:
                pass


def render_single_frame(
    trimesh_scene,
    resolution,
    view_type,
    use_dynamic_zoom,
    camera_height,
    scene_span,
    bg_color=None,
    look_at_target=None,
    max_attempts=3,
):
    """
    Render a single frame and return the image array.
    NOTE: input is trimesh_scene, not pyrender_scene.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            color, depth, camera_pose = _render_fresh_scene_once(
                trimesh_scene=trimesh_scene,
                resolution=resolution,
                view_type=view_type,
                use_dynamic_zoom=use_dynamic_zoom,
                camera_height=camera_height,
                scene_span=scene_span,
                bg_color=bg_color,
                look_at_target=look_at_target,
            )
            return color
        except Exception as e:
            last_error = e
            print(f"Render attempt {attempt} failed: {e}")
            traceback.print_exc()
            time.sleep(0.5)
    raise RuntimeError(f"Failed to render frame after {max_attempts} attempts: {last_error}")


def render_with_retry(
    trimesh_scene,
    resolution,
    pth_output,
    filename,
    view_type,
    use_dynamic_zoom,
    camera_height,
    scene_span,
    bg_color=None,
    look_at_target=None,
    max_attempts=3,
):
    """
    Render one specified view and save to disk.
    Each attempt rebuilds a fresh pyrender scene.
    """
    last_error = None
    pth_output = Path(pth_output)
    pth_output.mkdir(parents=True, exist_ok=True)
    output_file = pth_output / filename

    for attempt in range(1, max_attempts + 1):
        try:
            color, depth, camera_pose = _render_fresh_scene_once(
                trimesh_scene=trimesh_scene,
                resolution=resolution,
                view_type=view_type,
                use_dynamic_zoom=use_dynamic_zoom,
                camera_height=camera_height,
                scene_span=scene_span,
                bg_color=bg_color,
                look_at_target=look_at_target,
            )
            _save_rendered_image(color, output_file)
            return
        except Exception as e:
            last_error = e
            print(f"Render attempt {attempt} failed: {e}")
            traceback.print_exc()
            time.sleep(0.5)

    raise RuntimeError(f"Failed to render {view_type} after {max_attempts} attempts: {last_error}")


def remove_pyrender_nodes(pyrender_scene):
    """
    Kept only for backward compatibility.
    Not needed in the new fresh-scene rendering pipeline.
    """
    return


def render_both_views(
    trimesh_scene,
    resolution,
    pth_output,
    base_filename,
    use_dynamic_zoom,
    camera_height,
    scene_span,
    bg_color=None,
):
    """Render top and diagonal views separately with fresh scenes."""
    render_with_retry(
        trimesh_scene=trimesh_scene,
        resolution=resolution,
        pth_output=Path(pth_output) / "top",
        filename=f"{base_filename}.jpg",
        view_type="top",
        use_dynamic_zoom=use_dynamic_zoom,
        camera_height=camera_height,
        scene_span=scene_span,
        bg_color=bg_color,
        max_attempts=3,
    )

    render_with_retry(
        trimesh_scene=trimesh_scene,
        resolution=resolution,
        pth_output=Path(pth_output) / "diag",
        filename=f"{base_filename}.jpg",
        view_type="diag",
        use_dynamic_zoom=use_dynamic_zoom,
        camera_height=camera_height,
        scene_span=scene_span,
        bg_color=bg_color,
        max_attempts=3,
    )


def render_scene_and_export(
    scene_with_assets,
    filename,
    pth_output,
    resolution=(1024, 1024),
    show_bboxes=False,
    show_assets=True,
    show_assets_voxelized=False,
    show_bounds=False,
    use_dynamic_zoom=True,
    camera_height=None,
    bg_color=None,
):
    bounds_bottom = scene_with_assets["bounds_bottom"]
    trimesh_scene, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)

    add_objects_to_trimesh_scene(
        trimesh_scene,
        scene_with_assets["objects"],
        show_bboxes,
        show_assets,
        show_assets_voxelized,
        show_bounds,
        scene_with_assets.get("bounds_bottom"),
        scene_with_assets.get("bounds_top"),
    )

    render_both_views(
        trimesh_scene=trimesh_scene,
        resolution=resolution,
        pth_output=pth_output,
        base_filename=filename,
        use_dynamic_zoom=use_dynamic_zoom,
        camera_height=camera_height,
        scene_span=scene_span,
        bg_color=bg_color,
    )


def render_scene_to_frame(
    trimesh_scene,
    resolution,
    view_type,
    use_dynamic_zoom,
    camera_height,
    scene_span,
    bg_color=None,
    return_camera_params=False,
    look_at_target=None,
):
    """
    Render one frame from a trimesh scene.
    We do NOT create pyrender_scene outside and pass it around anymore.
    """
    color, depth, camera_pose = _render_fresh_scene_once(
        trimesh_scene=trimesh_scene,
        resolution=resolution,
        view_type=view_type,
        use_dynamic_zoom=use_dynamic_zoom,
        camera_height=camera_height,
        scene_span=scene_span,
        bg_color=bg_color,
        look_at_target=look_at_target,
    )

    if return_camera_params:
        fov = np.pi / 4.0
        if camera_height is None:
            scene_x, scene_y, scene_z = scene_span
            scene_aspect = scene_x / max(scene_z, 1e-5)
            limiting_span = scene_x if scene_aspect > 1.0 else scene_z
            if use_dynamic_zoom:
                required_distance = (limiting_span / 2) / np.tan(fov / 2)
                if view_type == "top":
                    actual_camera_height = max(2.0, required_distance + 2.5)
                else:
                    actual_camera_height = max(2.0, required_distance)
            else:
                actual_camera_height = 13.0 if view_type == "top" else 10.0
        else:
            actual_camera_height = camera_height

        cam_params = {
            "fov": fov,
            "camera_height": actual_camera_height,
            "camera_pose": camera_pose,
            "resolution": resolution,
        }
        return color, cam_params

    return color

def setup_trimesh_scene_with_floor(bounds_bottom):
    trimesh_scene = trimesh.Scene()
    floor_slab = create_floor_slab(bounds_bottom)
    trimesh_scene.add_geometry(floor_slab)

    x_span = np.array(bounds_bottom)[:, 0].max() - np.array(bounds_bottom)[:, 0].min()
    y_span = np.array(bounds_bottom)[:, 1].max() - np.array(bounds_bottom)[:, 1].min()
    z_span = np.array(bounds_bottom)[:, 2].max() - np.array(bounds_bottom)[:, 2].min()
    scene_span = (x_span, y_span, z_span)

    return trimesh_scene, scene_span

# ─────────────────────────────────────────────────────────────────────────────
# Model info lookup for super-category labels
# ─────────────────────────────────────────────────────────────────────────────

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)

_MODEL_INFO_CACHE = None  # lazy-loaded singleton


def _extract_model_id_from_jid(jid):
    """Extract UUID model_id from a 3D-FUTURE jid string."""
    if not isinstance(jid, str) or not jid:
        return None
    head = jid[:36]
    if _UUID_RE.match(head):
        return head
    if _UUID_RE.match(jid):
        return jid
    return None


def _get_model_info_mapping():
    """Lazy-load and cache model_info.json → {model_id: (super_category, category)}."""
    global _MODEL_INFO_CACHE
    if _MODEL_INFO_CACHE is not None:
        return _MODEL_INFO_CACHE

    model_info_path = os.environ.get(
        "PTH_MODEL_INFO_JSON",
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "dataset", "3D-FUTURE-model", "model_info.json"),
    )

    mapping = {}
    try:
        with open(model_info_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                items = data["data"]
            else:
                for k, v in data.items():
                    if isinstance(v, dict):
                        sc = v.get("super-category", v.get("super_category"))
                        cat = v.get("category")
                        mapping[str(k)] = (sc, cat)
                _MODEL_INFO_CACHE = mapping
                return mapping

        for it in items:
            if not isinstance(it, dict):
                continue
            mid = it.get("model_id") or it.get("id") or it.get("uid")
            if not mid:
                continue
            sc = it.get("super-category", it.get("super_category"))
            cat = it.get("category")
            mapping[str(mid)] = (sc, cat)

    except Exception as e:
        print(f"[anno] WARNING: failed to load model_info.json: {e}")

    _MODEL_INFO_CACHE = mapping
    return mapping


def _get_super_category_for_obj(obj):
    """Return super-category string for an object, falling back to desc/category."""
    # 1) Check if already present on the object
    sc = obj.get("super-category") or obj.get("super_category")
    if sc:
        return sc

    # 2) Look up from model_info.json via jid
    jid = obj.get("jid") or obj.get("sampled_asset_jid") or ""
    model_id = _extract_model_id_from_jid(jid)
    if model_id:
        mapping = _get_model_info_mapping()
        entry = mapping.get(model_id)
        if entry and entry[0]:
            return entry[0]

    # 3) Fallback
    return obj.get("category") or obj.get("desc") or ""

# ─────────────────────────────────────────────────────────────────────────────
# Annotated top-view rendering: coordinate grid, axis arrows, bbox wireframes,
# direction arrows, and text label overlay.
# ─────────────────────────────────────────────────────────────────────────────

def _anno_set_color(mesh, color):
    """Apply a solid RGBA color to a trimesh mesh via PBRMaterial.
    
    Using PBRMaterial ensures the color survives the trimesh→pyrender conversion,
    unlike face_colors which gets dropped by pyrender.Mesh.from_trimesh().
    """
    r, g, b, a = [c / 255.0 for c in color]
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[r, g, b, a],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
    )

def _anno_create_cylinder(start, end, radius=0.02, sections=12, color=None):
    """Create a trimesh cylinder between two 3D points."""
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-8:
        return None

    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)

    z_axis = np.array([0.0, 0.0, 1.0])
    d_norm = direction / length
    cross = np.cross(z_axis, d_norm)
    dot = np.dot(z_axis, d_norm)

    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            R = np.eye(4)
        else:
            R = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    else:
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        R = trimesh.transformations.rotation_matrix(angle, cross)

    mid = (start + end) / 2.0
    T = np.eye(4)
    T[:3, 3] = mid

    cyl.apply_transform(R)
    cyl.apply_transform(T)

    if color is not None:
        _anno_set_color(cyl, color)   # ← 改用 PBRMaterial
    return cyl


def _anno_create_cone(base_center, direction, height=0.1, radius=0.05, sections=12, color=None):
    """Create a trimesh cone pointing in *direction* from *base_center*."""
    direction = np.array(direction, dtype=float)
    d_len = np.linalg.norm(direction)
    if d_len < 1e-8:
        return None
    d_norm = direction / d_len

    cone = trimesh.creation.cone(radius=radius, height=height, sections=sections)

    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, d_norm)
    dot = np.dot(z_axis, d_norm)

    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            R = np.eye(4)
        else:
            R = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    else:
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        R = trimesh.transformations.rotation_matrix(angle, cross)

    center = np.array(base_center, dtype=float) + d_norm * (height / 2.0)
    T = np.eye(4)
    T[:3, 3] = center

    cone.apply_transform(R)
    cone.apply_transform(T)

    if color is not None:
        _anno_set_color(cone, color)   # ← 改用 PBRMaterial
    return cone

def _anno_create_arrow(start, direction, shaft_length=1.0, shaft_radius=0.02,
                       head_length=0.1, head_radius=0.06, color=(255, 0, 0, 255)):
    """Shaft cylinder + cone head → single mesh."""
    start = np.array(start, dtype=float)
    direction = np.array(direction, dtype=float)
    d_len = np.linalg.norm(direction)
    if d_len < 1e-8:
        return None
    d_norm = direction / d_len

    shaft_end = start + d_norm * shaft_length
    shaft = _anno_create_cylinder(start, shaft_end, radius=shaft_radius, color=color)
    head = _anno_create_cone(shaft_end, d_norm, height=head_length, radius=head_radius, color=color)

    parts = [m for m in [shaft, head] if m is not None]
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


def _anno_quat_to_rotmat(rot):
    """[x, y, z, w] quaternion → 3×3 rotation matrix."""
    x, y, z, w = rot
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ])


def _anno_create_bbox_wireframe(center, size, rotation_matrix=None,
                                color=(0, 80, 255, 255), line_radius=0.015):
    """12-edge wireframe box built from thin cylinders."""
    cx, cy, cz = center
    hx, hy, hz = [s / 2.0 for s in size]

    corners_local = np.array([
        [-hx, -hy, -hz], [+hx, -hy, -hz], [+hx, +hy, -hz], [-hx, +hy, -hz],
        [-hx, -hy, +hz], [+hx, -hy, +hz], [+hx, +hy, +hz], [-hx, +hy, +hz],
    ])

    if rotation_matrix is not None:
        corners_local = (np.array(rotation_matrix) @ corners_local.T).T

    corners = corners_local + np.array([cx, cy, cz])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    meshes = []
    for i, j in edges:
        cyl = _anno_create_cylinder(corners[i], corners[j], radius=line_radius, color=color)
        if cyl is not None:
            meshes.append(cyl)

    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def _anno_create_circle_marker(center, radius=0.05, color=(255, 0, 0, 255), height=0.005):
    """Small flat cylinder → circle marker on the ground."""
    disk = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    disk.apply_translation(center)
    _anno_set_color(disk, color)       # ← 改用 PBRMaterial（替换原 face_colors）
    return disk


def anno_add_coordinate_grid(trimesh_scene, room_size_x, room_size_z, y=0.002):
    """Red dot grid on the floor plane (XZ in the scene's coordinate system).

    The scene uses  X = right, Y = up, Z = forward  (OpenGL-ish).
    Grid lives at y ≈ 0 (floor).
    """
    markers = []
    for ix in range(int(np.ceil(room_size_x)) + 1):
        for iz in range(int(np.ceil(room_size_z)) + 1):
            marker = _anno_create_circle_marker(
                [float(ix), y, float(iz)], radius=0.04, color=(255, 0, 0, 255)
            )
            markers.append(marker)
    if markers:
        trimesh_scene.add_geometry(trimesh.util.concatenate(markers))


def anno_add_axis_arrows(trimesh_scene, origin=(0, 0.005, 0), length=1.0):
    """XYZ arrows at origin.  X = red, Y = green, Z = blue."""
    specs = [
        ([1, 0, 0], (255, 0,   0,   255)),  # +X  red
        ([0, 1, 0], (0,   200, 0,   255)),  # +Y  green
        ([0, 0, 1], (0,   80,  255, 255)),  # +Z  blue
    ]
    for direction, color in specs:
        arrow = _anno_create_arrow(
            origin, direction,
            shaft_length=length,
            shaft_radius=0.025,
            head_length=0.15,
            head_radius=0.07,
            color=color,
        )
        if arrow is not None:
            trimesh_scene.add_geometry(arrow)
            
def _anno_normalize_scene(scene):
    """Return a deep-copied scene with all object positions translated so that
    the room's XZ bounding box starts at (0, *, 0).
    
    Also returns the offset so annotations can be placed correctly.
    Original scene dict is NOT modified.
    """
    import copy
    scene_copy = copy.deepcopy(scene)

    bounds_bottom = scene_copy.get("bounds_bottom", [])
    if not bounds_bottom:
        return scene_copy, 0.0, 0.0

    bp = np.array(bounds_bottom)
    x_min = float(bp[:, 0].min())
    z_min = float(bp[:, 2].min())

    # 平移使左下角对齐到 (0, *, 0)
    x_offset = x_min
    z_offset = z_min

    # Translate bounds_bottom
    for pt in scene_copy["bounds_bottom"]:
        pt[0] -= x_offset
        pt[2] -= z_offset

    # Translate bounds_top
    for pt in scene_copy.get("bounds_top", []):
        pt[0] -= x_offset
        pt[2] -= z_offset

    # Translate all object positions
    for obj in scene_copy.get("objects", []):
        pos = obj.get("pos")
        if pos and len(pos) >= 3:
            obj["pos"] = [pos[0] - x_offset, pos[1], pos[2] - z_offset]

    return scene_copy, x_offset, z_offset


def anno_add_coordinate_grid(trimesh_scene, room_x_max, room_z_max, y=0.002):
    """Integer-coordinate dot grid on the floor plane."""
    markers = []
    nx = int(np.ceil(room_x_max)) + 1
    nz = int(np.ceil(room_z_max)) + 1
    for ix in range(nx):
        for iz in range(nz):
            marker = _anno_create_circle_marker(
                [float(ix), y, float(iz)], radius=0.04, color=(255, 0, 0, 255)
            )
            markers.append(marker)
    if markers:
        trimesh_scene.add_geometry(trimesh.util.concatenate(markers))


def anno_add_object_bboxes(trimesh_scene, objects_list):
    """Add wireframe bbox + front-direction arrow for every object.

    Returns *label_infos* list for the later 2-D text overlay step.
    """
    BBOX_COLOR  = (30, 144, 255, 255)   # dodger blue
    ARROW_COLOR = (255, 220, 0,   255)  # yellow

    label_infos = []

    for obj in objects_list:
        pos  = obj.get("pos")
        rot  = obj.get("rot")
        size = obj.get("size")
        if not pos or not size:
            continue

        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])

        # pos 是底面中心（bottom-center），bbox 的几何中心需要上移 sy/2
        bbox_cy = cy + sy / 2.0

        R = None
        if rot and len(rot) == 4:
            R = _anno_quat_to_rotmat(rot)

        # --- wireframe bbox ---
        bbox_mesh = _anno_create_bbox_wireframe(
            center=[cx, bbox_cy, cz],   # 用几何中心
            size=[sx, sy, sz],
            rotation_matrix=R,
            color=BBOX_COLOR,
            line_radius=0.015,
        )
        if bbox_mesh is not None:
            trimesh_scene.add_geometry(bbox_mesh)

        # --- front direction arrow ---
        if R is not None:
            front_dir = R[:, 2].copy()
            front_dir[1] = 0.0
            fd_len = np.linalg.norm(front_dir)
            if fd_len > 1e-6:
                front_dir = front_dir / fd_len
                # 箭头起点放在 bbox 顶面中心正上方一点
                arrow_origin = np.array([cx, bbox_cy + sy / 2.0 + 0.02, cz])
                arrow_len = max(sx, sz) * 0.6
                arrow = _anno_create_arrow(
                    arrow_origin, front_dir,
                    shaft_length=arrow_len,
                    shaft_radius=0.020,
                    head_length=0.12,
                    head_radius=0.06,
                    color=ARROW_COLOR,
                )
                if arrow is not None:
                    trimesh_scene.add_geometry(arrow)

        # --- label info ---
        super_cat = _get_super_category_for_obj(obj)
        label = super_cat if super_cat else ""

        label_infos.append({
            "label":     label,
            "center_3d": [cx, bbox_cy, cz],   # 同样用几何中心
            "size":      [sx, sy, sz],
        })

    return label_infos

def _anno_world_to_pixel(point_3d, camera_pose, fov, resolution):
    """Project a 3D world point to 2D pixel coordinates using the actual camera.
    
    camera_pose: 4x4 camera-to-world matrix (as set in setup_camera)
    fov: vertical field of view in radians
    resolution: (W, H)
    
    Returns (px, py) in pixel coords, or None if behind camera.
    """
    W, H = resolution
    
    # World-to-camera transform = inverse of camera_pose
    cam_inv = np.linalg.inv(camera_pose)
    
    # Transform point to camera space
    p_world = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    p_cam = cam_inv @ p_world
    
    # In OpenGL camera convention: camera looks along -Z
    # p_cam = [right, up, -forward]
    x_cam, y_cam, z_cam = p_cam[0], p_cam[1], p_cam[2]
    
    # Point is behind camera if z_cam >= 0 (OpenGL: forward = -Z)
    if z_cam >= 0:
        return None
    
    # Perspective projection
    aspect = W / H
    # fov is yfov
    fy = 1.0 / np.tan(fov / 2.0)
    fx = fy / aspect
    
    # Normalized device coordinates
    ndc_x = fx * x_cam / (-z_cam)
    ndc_y = fy * y_cam / (-z_cam)
    
    # NDC [-1, 1] → pixel [0, W], [0, H]
    px = (ndc_x + 1.0) * 0.5 * W
    py = (1.0 - ndc_y) * 0.5 * H  # Y 翻转：NDC +Y = 图像上方
    
    return int(round(px)), int(round(py))


# =========================
# Improved annotated top-view rendering
# Replace your existing annotation-related functions with this block.
# Reuses these existing functions from your file:
#   - setup_trimesh_scene_with_floor
#   - add_objects_to_trimesh_scene
#   - render_scene_to_frame
#   - anno_add_coordinate_grid
#   - anno_add_axis_arrows
#   - _anno_create_bbox_wireframe
#   - _anno_create_arrow
#   - _anno_quat_to_rotmat
# =========================

# -----------------------------------------------------------------------------
# Label strategy
# -----------------------------------------------------------------------------

_ANNO_LABEL_SPECIAL_MAP = {
    "Lounge Chair / Cafe Chair / Office Chair": "Lounge Chair",
    "Bookcase / jewelry Armoire": "Bookcase",
    "Corner/Side Table": "Side Table",
    "Corner / Side Table": "Side Table",
    "King-size Bed": "King Bed",
    "Single bed": "Single Bed",
    "Dining table": "Dining Table",
}

_ANNO_DESC_RULES = [
    (r"\bpendant lamp\b", "Pendant Lamp"),
    (r"\bceiling lamp\b", "Ceiling Lamp"),
    (r"\bfloor lamp\b", "Floor Lamp"),
    (r"\btable lamp\b", "Table Lamp"),
    (r"\blamp\b", "Lamp"),
    (r"\bplant\b", "Plant"),
    (r"\bwardrobe\b", "Wardrobe"),
    (r"\bnightstand\b", "Nightstand"),
    (r"\bcoffee table\b", "Coffee Table"),
    (r"\bdining table\b", "Dining Table"),
    (r"\bdesk\b", "Desk"),
    (r"\btv stand\b", "TV Stand"),
    (r"\bbookcase\b", "Bookcase"),
    (r"\bsofa\b", "Sofa"),
    (r"\bbed\b", "Bed"),
    (r"\bchair\b", "Chair"),
    (r"\bmirror\b", "Mirror"),
    (r"\brug\b", "Rug"),
    (r"\bvase\b", "Vase"),
    (r"\bclock\b", "Clock"),
    (r"\bpainting\b", "Painting"),
    (r"\bartwork\b", "Artwork"),
]

_ANNO_OTHER_LIKE = {"others", "other", "misc", "unknown"}


def _anno_simplify_label(label: str) -> str:
    """Make labels shorter and more VLM-friendly."""
    if label is None:
        return ""
    label = str(label).strip()
    if not label:
        return ""

    label = label.replace("_", " ")
    label = re.sub(r"\s+", " ", label)

    if label in _ANNO_LABEL_SPECIAL_MAP:
        return _ANNO_LABEL_SPECIAL_MAP[label]

    # For labels like "Lounge Chair / Cafe Chair / Office Chair"
    if " / " in label:
        parts = [p.strip() for p in label.split(" / ") if p.strip()]
        if parts:
            label = parts[0]

    return _ANNO_LABEL_SPECIAL_MAP.get(label, label)


def _anno_guess_label_from_desc(desc: str) -> str:
    """Fallback for category=None or super-category=Others."""
    if not desc:
        return ""
    desc_l = str(desc).lower()
    for pattern, mapped in _ANNO_DESC_RULES:
        if re.search(pattern, desc_l):
            return mapped
    return ""


def _anno_get_display_base_label(obj: dict, label_mode: str = "category") -> str:
    """
    label_mode:
      - 'category': prefer category, fallback to super-category, then desc
      - 'super-category': use super-category only
      - 'hybrid': e.g. 'Nightstand (Cabinet)'
    """
    category = _anno_simplify_label(obj.get("category"))
    super_cat = _anno_simplify_label(obj.get("super-category") or obj.get("super_category"))
    desc_guess = _anno_guess_label_from_desc(obj.get("desc", ""))

    super_is_other = super_cat.lower() in _ANNO_OTHER_LIKE if super_cat else True

    if label_mode == "super-category":
        if super_cat and not super_is_other:
            return super_cat
        return desc_guess or category or super_cat or "Object"

    if label_mode == "hybrid":
        if category:
            if super_cat and (not super_is_other) and (super_cat.lower() not in category.lower()):
                # make hybrid concise
                short_super = super_cat
                if short_super == "Cabinet/Shelf/Desk":
                    short_super = "Cabinet"
                return f"{category} ({short_super})"
            return category
        if super_cat and not super_is_other:
            return super_cat
        return desc_guess or "Object"

    # default: category-first
    if category:
        return category
    if super_cat and not super_is_other:
        return super_cat
    if desc_guess:
        return desc_guess
    if super_cat:
        return super_cat
    return "Object"


def _anno_assign_instance_labels(objects_list, label_mode="category", add_instance_id=True):
    """Generate stable display labels with optional numbering."""
    base_labels = [_anno_get_display_base_label(obj, label_mode=label_mode) for obj in objects_list]

    if not add_instance_id:
        return base_labels

    totals = defaultdict(int)
    for lb in base_labels:
        totals[lb] += 1

    current = defaultdict(int)
    out = []
    for lb in base_labels:
        current[lb] += 1
        if totals[lb] > 1:
            out.append(f"{lb}#{current[lb]}")
        else:
            out.append(lb)
    return out


# -----------------------------------------------------------------------------
# Geometry helpers for better label placement
# -----------------------------------------------------------------------------

def _anno_get_planar_basis(rotation_matrix=None):
    """Return normalized planar right/front vectors on XZ plane."""
    if rotation_matrix is None:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
        front = np.array([0.0, 0.0, 1.0], dtype=float)
        return right, front

    right = np.array(rotation_matrix[:, 0], dtype=float)
    front = np.array(rotation_matrix[:, 2], dtype=float)

    right[1] = 0.0
    front[1] = 0.0

    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        right = right / np.linalg.norm(right)

    if np.linalg.norm(front) < 1e-8:
        front = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        front = front / np.linalg.norm(front)

    return right, front


def _anno_make_footprint_corners(cx, floor_y, cz, sx, sz, rotation_matrix=None):
    """World-space footprint corners on the floor plane."""
    local = np.array([
        [-sx / 2.0, 0.0, -sz / 2.0],
        [ sx / 2.0, 0.0, -sz / 2.0],
        [ sx / 2.0, 0.0,  sz / 2.0],
        [-sx / 2.0, 0.0,  sz / 2.0],
    ], dtype=float)

    if rotation_matrix is not None:
        local = (np.array(rotation_matrix, dtype=float) @ local.T).T

    local[:, 0] += cx
    local[:, 1] += floor_y
    local[:, 2] += cz
    return local.tolist()


def _anno_make_label_candidates(center_floor, size, rotation_matrix=None):
    """
    Put label candidates around the floor footprint, not above the object.
    This avoids perspective drift for tall objects/lights.
    """
    cx, floor_y, cz = center_floor
    sx, _, sz = size

    right, front = _anno_get_planar_basis(rotation_matrix)

    base = np.array([cx, floor_y, cz], dtype=float)

    margin_main = max(0.16, 0.12 * max(sx, sz) + 0.08)
    margin_diag = margin_main * 0.75

    dx = sx / 2.0 + margin_main
    dz = sz / 2.0 + margin_main
    ddx = sx / 2.0 + margin_diag
    ddz = sz / 2.0 + margin_diag

    candidates = [
        base + front * dz,
        base - front * dz,
        base + right * dx,
        base - right * dx,

        base + right * ddx + front * ddz,
        base + right * ddx - front * ddz,
        base - right * ddx + front * ddz,
        base - right * ddx - front * ddz,

        base,  # last-resort fallback
    ]

    out = []
    seen = set()
    for c in candidates:
        key = tuple(np.round(c, 4).tolist())
        if key not in seen:
            out.append(c.tolist())
            seen.add(key)
    return out


# -----------------------------------------------------------------------------
# Scene normalization
# -----------------------------------------------------------------------------

def _anno_normalize_scene(scene):
    """
    Return a deep-copied scene with all object positions translated so that
    room XZ min becomes (0, 0). Original scene is not modified.
    """
    scene_copy = copy.deepcopy(scene)

    bounds_bottom = scene_copy.get("bounds_bottom", [])
    if not bounds_bottom:
        return scene_copy, 0.0, 0.0

    bp = np.array(bounds_bottom, dtype=float)
    x_min = float(bp[:, 0].min())
    z_min = float(bp[:, 2].min())

    x_offset = x_min
    z_offset = z_min

    for pt in scene_copy["bounds_bottom"]:
        pt[0] -= x_offset
        pt[2] -= z_offset

    for pt in scene_copy.get("bounds_top", []):
        pt[0] -= x_offset
        pt[2] -= z_offset

    for obj in scene_copy.get("objects", []):
        pos = obj.get("pos")
        if pos and len(pos) >= 3:
            obj["pos"] = [pos[0] - x_offset, pos[1], pos[2] - z_offset]

    return scene_copy, x_offset, z_offset


# -----------------------------------------------------------------------------
# Add bboxes + collect label infos
# -----------------------------------------------------------------------------

def anno_add_object_bboxes(
    trimesh_scene,
    objects_list,
    label_mode="hybrid",
    add_instance_id=True,
    show_bboxes=True,   # 新增
):
    label_infos = []

    for i, obj in enumerate(objects_list):
        # 这里是你原来的几何计算逻辑
        # 比如拿到 bbox corners / center / yaw / label_text 等
        # ------------------------------------------------
        # center = ...
        # corners = ...
        # label_text = ...
        # ------------------------------------------------

        # 1) 只有在 show_bboxes=True 时才画 bbox
        if show_bboxes:
            # 这里保留你原来的 bbox wireframe 添加逻辑
            # e.g. add_bbox_wireframe(trimesh_scene, corners, ...)
            pass

        # 2) 方向箭头仍然保留（如果你希望隐藏 bbox 但仍保留朝向）
        # e.g. add_direction_arrow(trimesh_scene, center, yaw, ...)
        pass

        # 3) label info 仍然保留
        label_infos.append({
            # "text": label_text,
            # "anchor": center,
            # 其他你原来需要的字段
        })

    return label_infos


# -----------------------------------------------------------------------------
# Projection
# -----------------------------------------------------------------------------

def _anno_world_to_pixel(point_3d, camera_pose, fov, resolution):
    """
    Project a 3D world point to 2D pixel coordinates using the actual camera.
    Returns (px, py) or None if behind camera.
    """
    W, H = resolution

    cam_inv = np.linalg.inv(camera_pose)

    p_world = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=float)
    p_cam = cam_inv @ p_world

    x_cam, y_cam, z_cam = p_cam[0], p_cam[1], p_cam[2]

    # OpenGL camera convention: camera looks along -Z
    if z_cam >= 0:
        return None

    aspect = W / H
    fy = 1.0 / np.tan(fov / 2.0)
    fx = fy / aspect

    ndc_x = fx * x_cam / (-z_cam)
    ndc_y = fy * y_cam / (-z_cam)

    px = (ndc_x + 1.0) * 0.5 * W
    py = (1.0 - ndc_y) * 0.5 * H

    return int(round(px)), int(round(py))


# -----------------------------------------------------------------------------
# 2D overlay layout helpers
# -----------------------------------------------------------------------------

def _anno_rect_area(rect):
    x1, y1, x2, y2 = rect
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _anno_rect_intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _anno_shift_rect_into_bounds(rect, W, H, margin=2):
    """Shift rect into image bounds without resizing."""
    x1, y1, x2, y2 = rect
    dx = 0.0
    dy = 0.0

    if x1 < margin:
        dx = margin - x1
    elif x2 > W - margin:
        dx = (W - margin) - x2

    if y1 < margin:
        dy = margin - y1
    elif y2 > H - margin:
        dy = (H - margin) - y2

    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]


def _anno_bbox_from_projected_points(points_2d):
    pts = [p for p in points_2d if p is not None]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _anno_choose_label_layout(info, draw, font, used_rects, w2p, W, H, pad=4):
    """
    Greedy selection among multiple candidate anchors:
    - avoid overlap with previous labels
    - avoid covering object footprint
    - keep label near its object
    """
    label = info["label"]
    bb = draw.textbbox((0, 0), label, font=font)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]

    obj_center_px = w2p(*info["center_floor_3d"])
    if obj_center_px is None:
        return None

    footprint_px = [w2p(*p) for p in info["footprint_corners_3d"]]
    obj_rect = _anno_bbox_from_projected_points(footprint_px)

    best = None
    best_score = float("inf")

    candidates = info["anchor_candidates_3d"]
    for cand_idx, cand_world in enumerate(candidates):
        cand_px = w2p(*cand_world)
        if cand_px is None:
            continue

        cx, cy = cand_px
        rect = [
            cx - tw / 2.0 - pad,
            cy - th / 2.0 - pad,
            cx + tw / 2.0 + pad,
            cy + th / 2.0 + pad,
        ]
        rect = _anno_shift_rect_into_bounds(rect, W, H, margin=2)

        label_center = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)

        score = 0.0

        # Overlap with previous labels: strong penalty
        for used in used_rects:
            score += 2500.0 * _anno_rect_intersection_area(rect, used)

        # Covering the object itself: medium penalty
        if obj_rect is not None:
            score += 400.0 * _anno_rect_intersection_area(rect, obj_rect)

        # Prefer closer placements, but not too strong
        score += 0.18 * np.hypot(label_center[0] - obj_center_px[0], label_center[1] - obj_center_px[1])

        # Slight preference for earlier candidates
        score += 3.0 * cand_idx

        if score < best_score:
            best_score = score
            best = {
                "rect": rect,
                "label_center": label_center,
                "obj_center_px": obj_center_px,
            }

    if best is None:
        # fallback: place around projected center
        cx, cy = obj_center_px
        rect = [
            cx - tw / 2.0 - pad,
            cy - th / 2.0 - pad,
            cx + tw / 2.0 + pad,
            cy + th / 2.0 + pad,
        ]
        rect = _anno_shift_rect_into_bounds(rect, W, H, margin=2)
        best = {
            "rect": rect,
            "label_center": ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0),
            "obj_center_px": obj_center_px,
        }

    return best


# -----------------------------------------------------------------------------
# 2D overlay
# -----------------------------------------------------------------------------

def _anno_overlay_labels(
    image,
    label_infos,
    coord_grid_x_max,
    coord_grid_z_max,
    resolution,
    font_size=14,
    camera_params=None,
):
    """
    Draw coordinate text + object labels on a PIL RGBA image.
    Uses:
      - category-first labels
      - floor-anchored labels
      - greedy collision avoidance
      - leader lines

    Robust version:
      - supports both `label` and `text`
      - safely skips malformed label infos
      - avoids KeyError when some items do not contain `label`
    """
    from PIL import ImageDraw, ImageFont
    import numpy as np

    draw = ImageDraw.Draw(image)

    def _try_load_font(size):
        for p in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]:
            try:
                return ImageFont.truetype(p, size)
            except (IOError, OSError):
                continue
        return ImageFont.load_default()

    font = _try_load_font(font_size)
    coord_font = _try_load_font(max(10, font_size - 4))

    W, H = resolution

    if camera_params is None:
        print("[WARNING] _anno_overlay_labels: no camera_params, skipping overlay")
        return image

    cam_pose = camera_params["camera_pose"]
    fov = camera_params["fov"]

    def w2p(wx, wy, wz):
        return _anno_world_to_pixel([wx, wy, wz], cam_pose, fov, resolution)

    # ------------------------------------------------------------------
    # normalize / sanitize label_infos
    # ------------------------------------------------------------------
    if label_infos is None:
        label_infos = []

    sanitized_infos = []
    for info in label_infos:
        if not isinstance(info, dict):
            continue

        # 兼容两种命名：label / text
        label = info.get("label", info.get("text", ""))
        if label is None:
            label = ""
        label = str(label).strip()

        # 拷贝一份，避免修改原对象
        new_info = dict(info)
        new_info["label"] = label
        new_info["priority"] = float(new_info.get("priority", 0.0))

        sanitized_infos.append(new_info)

    label_infos = sanitized_infos

    # --- coordinate grid labels (on floor) ---
    for ix in range(int(np.ceil(coord_grid_x_max)) + 1):
        for iz in range(int(np.ceil(coord_grid_z_max)) + 1):
            result = w2p(float(ix), 0.0, float(iz))
            if result is not None:
                px, py = result
                if 0 <= px < W and 0 <= py < H:
                    r = 4
                    draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 0, 0, 255))
                    draw.text(
                        (px + 6, py - 6),
                        f"({ix},{iz})",
                        fill=(255, 0, 0, 255),
                        font=coord_font,
                    )

    # --- axis labels ---
    axis_font = _try_load_font(font_size)
    arrow_label_len = 0.5

    result_x = w2p(arrow_label_len, 0.0, 0.0)
    if result_x is not None:
        draw.text(
            (result_x[0] + 4, result_x[1] + 2),
            "+X",
            fill=(255, 0, 0, 255),
            font=axis_font,
        )

    result_z = w2p(0.0, 0.0, arrow_label_len)
    if result_z is not None:
        draw.text(
            (result_z[0] + 4, result_z[1] - 14),
            "+Z",
            fill=(0, 80, 255, 255),
            font=axis_font,
        )

    # --- object labels ---
    LABEL_BG = (30, 144, 255, 220)
    TEXT_FG = (255, 255, 255, 255)
    LEADER_FG = (30, 144, 255, 235)
    LEADER_DOT = (255, 220, 0, 255)

    used_rects = []

    # Larger footprints first = slightly better layout stability
    label_infos_sorted = sorted(
        label_infos,
        key=lambda x: (
            -x.get("priority", 0.0),
            x.get("label", "")
        )
    )

    for info in label_infos_sorted:
        label = info.get("label", "")
        if not label:
            continue

        # 如果 info 缺少 _anno_choose_label_layout 所需的关键字段，
        # 这里直接跳过，避免内部报错
        try:
            chosen = _anno_choose_label_layout(
                info=info,
                draw=draw,
                font=font,
                used_rects=used_rects,
                w2p=w2p,
                W=W,
                H=H,
                pad=4,
            )
        except Exception as e:
            print(f"[WARNING] _anno_overlay_labels: skip one label بسبب bad info: {e}")
            continue

        if chosen is None:
            continue

        rect = chosen["rect"]
        obj_center_px = chosen["obj_center_px"]
        label_center = chosen["label_center"]

        x1, y1, x2, y2 = rect
        text_bb = draw.textbbox((0, 0), label, font=font)
        tw = text_bb[2] - text_bb[0]
        th = text_bb[3] - text_bb[1]
        text_x = int(round((x1 + x2 - tw) / 2.0))
        text_y = int(round((y1 + y2 - th) / 2.0))

        # leader line
        if np.hypot(label_center[0] - obj_center_px[0], label_center[1] - obj_center_px[1]) > 10:
            draw.line(
                [obj_center_px[0], obj_center_px[1], label_center[0], label_center[1]],
                fill=LEADER_FG,
                width=2,
            )
            r = 3
            draw.ellipse(
                [
                    obj_center_px[0] - r,
                    obj_center_px[1] - r,
                    obj_center_px[0] + r,
                    obj_center_px[1] + r,
                ],
                fill=LEADER_DOT,
            )

        draw.rectangle([x1, y1, x2, y2], fill=LABEL_BG)
        draw.text((text_x, text_y), label, fill=TEXT_FG, font=font)

        used_rects.append(rect)

    return image


# -----------------------------------------------------------------------------
# Main render entry
# -----------------------------------------------------------------------------

def render_annotated_top_view(
    scene,
    filename,
    pth_viz_output,
    resolution=(1024, 1024),
    use_dynamic_zoom=True,
    camera_height=None,
    show_assets=True,
    font_size=14,
    bg_color=None,
    label_mode="hybrid",   # "category" | "super-category" | "hybrid"
    add_instance_id=True,
    show_bboxes=True,      # 新增：是否显示 bounding box
):
    """
    Recommended defaults:
      label_mode='category'
      add_instance_id=True
      show_bboxes=True

    This version is better for VLM use because:
      - category is more discriminative than super-category
      - labels are floor-anchored, so tall objects/lights don't drift visually
      - repeated instances are explicitly numbered
    """
    pth_viz_output = Path(pth_viz_output) if not isinstance(pth_viz_output, Path) else pth_viz_output

    # ---- normalize scene so room XZ min = (0, 0) ----
    scene_norm, _x_off, _z_off = _anno_normalize_scene(scene)

    bounds_bottom = scene_norm.get("bounds_bottom", [])
    objects_list = scene_norm.get("objects", [])

    # ---- base trimesh scene (floor) ----
    trimesh_scene, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)

    # ---- add actual asset meshes ----
    if show_assets:
        add_objects_to_trimesh_scene(trimesh_scene, objects_list)

    # ---- room bounds ----
    if bounds_bottom:
        bp = np.array(bounds_bottom, dtype=float)
        room_x_min = float(bp[:, 0].min())
        room_x_max = float(bp[:, 0].max())
        room_z_min = float(bp[:, 2].min())
        room_z_max = float(bp[:, 2].max())
    else:
        room_x_min, room_x_max = 0.0, 5.0
        room_z_min, room_z_max = 0.0, 5.0

    scene_center_x = (room_x_min + room_x_max) / 2.0
    scene_center_z = (room_z_min + room_z_max) / 2.0

    # ---- coordinate grid markers ----
    anno_add_coordinate_grid(trimesh_scene, room_x_max, room_z_max, y=0.002)

    # ---- axis arrows at origin ----
    room_size_x = room_x_max - room_x_min
    room_size_z = room_z_max - room_z_min
    arrow_len = min(room_size_x, room_size_z) * 0.15
    arrow_len = max(arrow_len, 0.3)
    anno_add_axis_arrows(trimesh_scene, origin=(0.0, 0.005, 0.0), length=arrow_len)

    # ---- bbox wireframes + direction arrows + label infos ----
    label_infos = anno_add_object_bboxes(
        trimesh_scene,
        objects_list,
        label_mode=label_mode,
        add_instance_id=add_instance_id,
        show_bboxes=show_bboxes,   # 新增
    )

    # ---- render with camera looking at scene center ----
    look_at = [scene_center_x, 0.0, scene_center_z]
    frame, cam_params = render_scene_to_frame(
        trimesh_scene,
        resolution,
        "top",
        use_dynamic_zoom,
        camera_height,
        scene_span,
        bg_color=bg_color,
        return_camera_params=True,
        look_at_target=look_at,
    )

    # ---- PIL overlay ----
    if isinstance(frame, np.ndarray):
        pil_image = Image.fromarray(frame).convert("RGBA")
    else:
        pil_image = frame.convert("RGBA")

    pil_image = _anno_overlay_labels(
        pil_image,
        label_infos,
        coord_grid_x_max=room_x_max,
        coord_grid_z_max=room_z_max,
        resolution=resolution,
        font_size=font_size,
        camera_params=cam_params,
    )

    # ---- save ----
    output_dir = pth_viz_output / "top-annotated"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f"{filename}.jpg"
    pil_image.convert("RGB").save(str(output_path), quality=95)
    print(f"[annotated top] saved to {output_path}")

    return output_path

def create_progressive_gif(scene_with_assets, filename, pth_output, view_type, resolution=(1024, 1024), use_dynamic_zoom=True, camera_height=None, duration=0.8):
    
    # Create output directory
    gif_output_dir = pth_output / f"{view_type}-gif"
    os.makedirs(gif_output_dir, exist_ok=True)
    
    # Add floor
    bounds_bottom = scene_with_assets["bounds_bottom"]
    trimesh_scene, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)
    
    # Collect frames
    frames = []
    
    # First frame with just the floor
    try:
        base_frame = render_scene_to_frame(trimesh_scene, resolution, view_type, use_dynamic_zoom, camera_height, scene_span)
        frames.append(base_frame)
    except Exception as e:
        print(f"Failed to render base frame: {e}")
        return
    
    # Add objects one by one
    for i, obj in enumerate(scene_with_assets["objects"]):
        try:
            # Add this object to the scene
            jid = obj["jid"] if obj.get("jid") is not None else obj["sampled_asset_jid"]
            mesh = load_mesh_with_transform(get_pth_mesh(jid), obj.get("pos"), obj.get("rot"), obj.get("scale"))
            trimesh_scene.add_geometry(mesh)
            
            # Render current state
            frame = render_scene_to_frame(trimesh_scene, resolution, view_type, use_dynamic_zoom, camera_height, scene_span)
            frames.append(frame)
            
        except Exception as e:
            print(f"Failed to add object {i}: {e}")
            traceback.print_exc()
            continue
    
    # Save as GIF if we have at least two frames
    if len(frames) >= 2:
        gif_path = os.path.join(gif_output_dir, f"{filename}.gif")
        durations = [duration*1000] * len(frames)
        imageio.mimsave(gif_path, frames, duration=durations, loop=0)

def render_full_scene_and_export_with_gif(scene_with_assets, filename, pth_output, resolution=(1024, 1024), show_bboxes=False, show_assets=True, show_assets_voxelized=True, show_bounds=False, use_dynamic_zoom=True, camera_height=None, create_gif=True, gif_duration=0.6, show_bboxes_also=False, bg_color=None):
    # render assets only
    render_scene_and_export(scene_with_assets, filename, pth_output, resolution, show_bboxes, show_assets, show_assets_voxelized, show_bounds, use_dynamic_zoom, camera_height, bg_color=bg_color)
    
    # if show_bboxes_also:
        # render_scene_and_export(scene_with_assets, f"{filename}-bboxes", pth_output, resolution, show_bboxes=True, show_assets=False, use_dynamic_zoom=use_dynamic_zoom, camera_height=camera_height)

    if create_gif:
        create_progressive_gif(scene_with_assets, filename, pth_output, "top", resolution, use_dynamic_zoom, camera_height, gif_duration)
        create_progressive_gif(scene_with_assets, filename, pth_output, "diag", resolution, use_dynamic_zoom, camera_height, gif_duration)

def create_instr_before_after_gif(scene_after, filename, pth_output, view_type, resolution=(1024, 1024), use_dynamic_zoom=True, camera_height=None, duration=0.8):
    # Create output directory
    gif_output_dir = pth_output / f"{view_type}-gif"
    os.makedirs(gif_output_dir, exist_ok=True)
    
    # setup scene
    bounds_bottom = scene_after["bounds_bottom"]
    trimesh_scene, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)
    
    # Collect frames
    frames = []

    # get frame for "before" scene
    add_objects_to_trimesh_scene(trimesh_scene, scene_after["objects"][:-1])
    before_frame = render_scene_to_frame(trimesh_scene, resolution, view_type, use_dynamic_zoom, camera_height, scene_span)
    frames.append(before_frame)
    
    # get frame for "after" scene
    last_objects = [scene_after["objects"][-1]]
    add_objects_to_trimesh_scene(trimesh_scene, last_objects)
    after_frame = render_scene_to_frame(trimesh_scene, resolution, view_type, use_dynamic_zoom, camera_height, scene_span)
    frames.append(after_frame)

    # Save as GIF with two frames
    if len(frames) == 2:
        gif_path = os.path.join(gif_output_dir, f"{filename}.gif")
        durations = [duration*1000] * len(frames)
        imageio.mimsave(gif_path, frames, duration=durations, loop=0)

def render_frame_at_angle(trimesh_scene, angle_degrees, resolution, camera_height, scene_span, bg_color=None):
    """
    Render a single frame of the scene from a specific angle by rotating the
    trimesh scene, then rendering via the fresh-scene pipeline.
    """
    scene_copy = copy.deepcopy(trimesh_scene)

    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=math.radians(angle_degrees),
        direction=[0, 1, 0],
        point=[0, 0, 0],
    )
    scene_copy.apply_transform(rotation_matrix)

    return render_scene_to_frame(
        trimesh_scene=scene_copy,
        resolution=resolution,
        view_type="diag",
        use_dynamic_zoom=False,
        camera_height=camera_height,
        scene_span=scene_span,
        bg_color=bg_color,
    )

def _ffmpeg_transcode_to_vscode_compatible(input_path: str, output_path: str) -> None:
    """
    Transcode to H.264 + yuv420p (+aac if audio exists) for better VS Code compatibility.
    Requires ffmpeg in PATH.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")

    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed:\n{proc.stderr}")

def create_360_video_full(
    scene_with_assets,
    filename,
    pth_output,
    room_type=None,
    resolution=(1536, 1024),  # 3:2 aspect ratio
    camera_height=None,
    fps=30,
    video_duration=4.0,
    step_time=0.5,
    bg_color=None,
    vscode_compatible: bool = True,
    mode: str = "full",          # "full" | "blink_last"
    visibility_time: float = 0.5 # used when mode == "blink_last"
):
    """
    Generate a 360° mp4 video.

    Modes:
    - mode="full": progressively place objects over multiple rotations, then one final full rotation.
    - mode="blink_last": show 'before' (all except last) and 'after' (with last) blinking during one rotation.

    If vscode_compatible=True:
    - try to write H.264 directly (avc1/H264) if OpenCV supports it;
    - else write mp4v to a temp file and transcode to H.264 yuv420p with ffmpeg.
    """
    # Create output directory for video
    video_output_dir = pth_output
    os.makedirs(video_output_dir, exist_ok=True)

    # Output paths
    final_video_path = os.path.join(video_output_dir, f"{filename}_360.mp4")
    tmp_video_path = os.path.join(video_output_dir, f"{filename}_360_tmp.mp4")

    # Setup scene basics
    bounds_bottom = scene_with_assets["bounds_bottom"]

    # Helper: open writer with best codec available
    def _open_writer():
        # Try H.264 first for compatibility, then fallback to mp4v temp.
        for fourcc_str, out_path in [("avc1", final_video_path), ("H264", final_video_path), ("mp4v", tmp_video_path)]:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            w = cv2.VideoWriter(out_path, fourcc, fps, resolution)
            if w is not None and w.isOpened():
                return w, out_path
        return None, None

    writer, video_path_written = _open_writer()
    if writer is None:
        raise RuntimeError("Failed to open cv2.VideoWriter with available codecs (avc1/H264/mp4v).")

    # Render according to mode
    if mode == "blink_last":
        # Build before/after scenes
        trimesh_scene_before, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)

        objects = scene_with_assets.get("objects", [])
        if len(objects) == 0:
            raise ValueError("scene_with_assets['objects'] is empty.")
        if len(objects) == 1:
            # before: empty; after: one object
            pass
        else:
            add_objects_to_trimesh_scene(trimesh_scene_before, objects[:-1])

        trimesh_scene_after = copy.deepcopy(trimesh_scene_before)
        add_objects_to_trimesh_scene(trimesh_scene_after, [objects[-1]])

        total_frames = int(fps * video_duration)
        frames_visible = max(1, int(fps * visibility_time))

        print(f"Creating {total_frames} frames for {video_duration}s 360° instruction video (blink_last)...")

        for frame_idx in tqdm(range(total_frames), desc="Rendering frames"):
            angle_progress = frame_idx / max(1, total_frames)
            angle_degrees = angle_progress * 360.0

            cycle_position = frame_idx % (frames_visible * 2)
            show_after = cycle_position >= frames_visible
            current_scene = trimesh_scene_after if show_after else trimesh_scene_before

            frame = render_frame_at_angle(
                current_scene,
                angle_degrees,
                resolution,
                camera_height,
                scene_span,
                bg_color
            )

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

    elif mode == "full":
        trimesh_scene_base, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)

        num_objects = len(scene_with_assets["objects"])
        if num_objects == 0:
            raise ValueError("scene_with_assets['objects'] is empty.")

        objects_per_rotation = max(1, int(video_duration / step_time))
        min_rotations_needed = math.ceil(num_objects / objects_per_rotation)

        # Add one extra rotation at the end to show complete scene
        total_rotations = min_rotations_needed + 1
        total_video_duration = total_rotations * video_duration

        total_frames = int(fps * total_video_duration)
        frames_per_step = max(1, int(fps * step_time))
        frames_per_rotation = max(1, int(fps * video_duration))

        print("Creating 360° full scene video:")
        print(f"- {num_objects} objects to place")
        print(f"- {objects_per_rotation} objects per {video_duration}s rotation")
        print(f"- {min_rotations_needed} rotations needed for placement")
        print(f"- {total_rotations} total rotations (including final complete scene)")
        print(f"- {total_video_duration}s total duration")
        print(f"- {total_frames} total frames")

        all_objects = copy.deepcopy(scene_with_assets["objects"])

        for frame_idx in tqdm(range(total_frames), desc="Rendering frames"):
            current_rotation = frame_idx // frames_per_rotation
            frame_in_rotation = frame_idx % frames_per_rotation

            angle_progress = frame_in_rotation / frames_per_rotation
            angle_degrees = angle_progress * 360.0

            if current_rotation < min_rotations_needed:
                objects_shown_from_prev_rotations = current_rotation * objects_per_rotation
                additional_objects_this_rotation = min(
                    frame_in_rotation // frames_per_step,
                    objects_per_rotation
                )
                total_objects_to_show = min(
                    objects_shown_from_prev_rotations + additional_objects_this_rotation,
                    num_objects
                )
            else:
                total_objects_to_show = num_objects

            current_scene = copy.deepcopy(trimesh_scene_base)
            objects_to_add = all_objects[:total_objects_to_show]
            add_objects_to_trimesh_scene(current_scene, objects_to_add)

            frame = render_frame_at_angle(
                current_scene,
                angle_degrees,
                resolution,
                camera_height,
                scene_span,
                bg_color
            )

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

    else:
        writer.release()
        raise ValueError(f"Unknown mode: {mode}. Expected 'full' or 'blink_last'.")

    writer.release()

    # If we wrote mp4v to temp and vscode_compatible is requested, transcode via ffmpeg.
    if vscode_compatible and video_path_written == tmp_video_path:
        try:
            _ffmpeg_transcode_to_vscode_compatible(tmp_video_path, final_video_path)
            try:
                os.remove(tmp_video_path)
            except OSError:
                pass
            video_path_written = final_video_path
        except Exception as e:
            print(f"[WARN] ffmpeg transcode skipped/failed: {e}")
            # Keep tmp as final if transcode fails
            video_path_written = tmp_video_path

    print(f"Created 360° video at {video_path_written}")
    return video_path_written

def render_instr_scene_and_export_with_gif(scene_after, filename, pth_output, resolution=(1024, 1024), show_bboxes=False, show_assets=True, show_assets_voxelized=False, show_bounds=False, use_dynamic_zoom=True, camera_height=None, create_gif=True, gif_duration=0.8, bg_color=None):
    render_scene_and_export(scene_after, filename, pth_output, resolution, show_bboxes, show_assets, show_assets_voxelized, show_bounds, use_dynamic_zoom, camera_height, bg_color=bg_color)
    # render_scene_and_export(scene_after, f"{filename}-bboxes", pth_output, resolution, show_bboxes=True, show_assets=False, use_dynamic_zoom=use_dynamic_zoom, camera_height=camera_height)
    if create_gif:
        create_instr_before_after_gif(scene_after, filename, pth_output, "top", resolution, use_dynamic_zoom, camera_height, gif_duration)
        create_instr_before_after_gif(scene_after, filename, pth_output, "diag", resolution, use_dynamic_zoom, camera_height, gif_duration)

def render_full_scenes_for_room_type(room_type, pth_root, pth_folder_prefix, pth_output):

    folder_name = f"{pth_folder_prefix}-{room_type}"
    pth_output_full = pth_output / folder_name
    remove_and_recreate_folder(pth_output_full)

    # we take the full train split if less than 5K, otherwise we sample 5K
    all_pths = get_pths_dataset_split(room_type, "train")
    if len(all_pths) > 5000:
        all_pths = np.random.choice(all_pths, 5000, replace=False)
    
    # test only
    # all_pths = all_pths[:5]

    cnt = 0
    pbar = tqdm(all_pths)
    for pth in pbar:
        scene = json.load(open(os.path.join(pth_root, pth), "r"))
        scene_id = pth.split(".")[0]
        render_full_scene_and_export_with_gif(scene, filename=scene_id, pth_output=pth_output_full, create_gif=False)
        cnt += 1
        pbar.set_description(f"Rendering scenes (# {cnt})")

    precompute_fid_scores_for_caching(f"{folder_name}-top", str(pth_output_full / "top"))
    precompute_fid_scores_for_caching(f"{folder_name}-diag", str(pth_output_full / "diag"))
    
    print(f"rendered all scenes for room type: {room_type}, total: {cnt}")

def get_assets_from_gt_for_scene(scene, scene_id):
    # for each object in scene, get the asset from full_scene_with_assets via "jid" that matches "desc"
    full_scene_with_assets = json.load(open(f"{os.getenv('PTH_STAGE_2_DEDUP')}/{scene_id}.json"))
    for obj in scene.get("objects"):
        desc = obj.get("desc")
        for asset in full_scene_with_assets.get("objects"):
            if asset.get("desc") == desc:
                obj["jid"] = asset.get("jid")
                break

def render_instr_scenes_for_room_type(room_type, pth_root, pth_folder_prefix, pth_output_base):
    print("=== Starting file reading phase ===")
    
    import gc
    gc.collect()
    
    folder_name = f"{pth_folder_prefix}-{room_type}"
    pth_output_full = pth_output_base / folder_name
    remove_and_recreate_folder(pth_output_full)
    
    dataset_train, _, _ = load_train_val_test_datasets(lambda_instr_exp=None, use_cached_dataset=True, room_type=room_type, do_sanity_check=False, seed=42)

    all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
    all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))
    
    # Track failures for reporting
    failed_renders = []

    # Add a buffer to account for potential failures
    np.random.seed(42)
    target_size = min(5000, len(dataset_train))
    dataset_train = dataset_train.select(range(target_size))

    model, tokenizer, max_seq_length = get_model("meta-llama/Llama-3.2-1B-Instruct", use_gpu=True, accelerator=None)
    
    print("=== Starting rendering phase ===")
    cnt = 0
    batch_size = 100
    for i in range(0, len(dataset_train), batch_size):
        batch = dataset_train.select(range(i, min(i + batch_size, len(dataset_train))))
        for sample in tqdm(batch, desc=f"Rendering batch {i//batch_size + 1}/{len(dataset_train)//batch_size}"):
            try:
                
                _, _, _, instr_sample = process_scene_sample(sample, tokenizer, max_seq_length, all_prompts, all_assets_metadata_simple_descs, do_simple_descs=False, do_augm=False, do_full_sg_outputs=False)
                # instr_sample = create_instruction_from_scene(sample, all_prompts, all_assets_metadata_simple_descs, do_simple_descs=False)
                
                scene_id = sample["pth_orig_file"].split(".")[0]
                scene_after = create_full_scene_from_before_and_added(json.loads(instr_sample.get("sg_input")), json.loads(instr_sample.get("sg_output_add")))
                get_assets_from_gt_for_scene(scene_after, scene_id)

                render_instr_scene_and_export_with_gif(scene_after, filename=scene_id, pth_output=pth_output_full, create_gif=False)

                # Verify both files were actually created
                top_file = pth_output_full / "top" / f"{scene_id}.jpg"
                diag_file = pth_output_full / "diag" / f"{scene_id}.jpg"
                if top_file.exists() and diag_file.exists():
                    cnt += 1
                else:
                    failed_renders.append((scene_id, "Files not created after render"))
                    print(f"Failed to create files for {scene_id} after seemingly successful render")
                    if not top_file.exists():
                        print(f"Missing top view: {top_file}")
                    if not diag_file.exists():
                        print(f"Missing diag view: {diag_file}")
                
            except Exception as exc:
                traceback.print_exc()
                print(f"Failed to render {scene_id}: {exc}")
                continue

            gc.collect()
        gc.collect()
    
    print(f"=== Completed rendering phase. Rendered {cnt} scenes ===")
    print(f"Failed to render {len(failed_renders)} scenes")
    
    # Verify output files exist
    top_files = set(os.listdir(pth_output_full / "top"))
    diag_files = set(os.listdir(pth_output_full / "diag"))
    
    print(f"Files in top directory: {len(top_files)}")
    print(f"Files in diag directory: {len(diag_files)}")
    
    # Only proceed with FID computation if we have files
    if len(top_files) > 0 and len(diag_files) > 0:
        precompute_fid_scores_for_caching(f"{folder_name}-top", str(pth_output_full / "top"))
        precompute_fid_scores_for_caching(f"{folder_name}-diag", str(pth_output_full / "diag"))
    
    print(f"Completed processing for room type: {room_type}")

def create_360_video_voxelization(scene_teaser, pth_folder_fig):
    resolution = (1536, 1024)  # 3:2 aspect ratio
    fps = 30
    camera_height = 5.5
    bg_color = np.array([240, 240, 240]) / 255.0
    step_time = 0.8  # Time between object additions/replacements
    still_time = 4.0  # Time to hold still between phases
    
    # Calculate timing parameters
    num_objects = len(scene_teaser["objects"])
    empty_scene_duration = 2.0  # 2 seconds empty scene
    bounds_only_duration = 4.0  # 4 seconds with bounds only
    bbox_placement_duration = num_objects * step_time  # 0.8s per bbox addition
    bbox_still_duration = still_time  # 4s still with all bboxes
    asset_replacement_duration = num_objects * step_time  # 0.8s per bbox->asset replacement
    asset_still_duration = still_time  # 4s still with all assets
    voxel_replacement_duration = num_objects * step_time  # 0.8s per asset->voxel replacement
    
    # Calculate how much extra time needed to complete full rotation
    base_duration = (empty_scene_duration + bounds_only_duration + 
                    bbox_placement_duration + bbox_still_duration +
                    asset_replacement_duration + asset_still_duration + 
                    voxel_replacement_duration)
    
    # Add time to complete the rotation (so we end where we started for looping)
    # We want at least 4 seconds of voxels, plus enough to complete the circle
    voxel_still_minimum = still_time
    total_for_full_rotation = base_duration + voxel_still_minimum
    
    # Calculate how much more time needed to complete exactly one full rotation
    # If we're past 360°, add time until we reach the next full rotation
    extra_time_for_loop = 0
    if total_for_full_rotation % 360 != 0:
        # Add time to reach next "clean" rotation point for seamless looping
        extra_time_for_loop = 1.0  # Add 1 second buffer for clean loop
    
    voxel_still_duration = voxel_still_minimum + extra_time_for_loop
    
    total_video_duration = base_duration + voxel_still_duration
    total_frames = int(fps * total_video_duration)
    
    # Calculate frame ranges for each phase
    empty_frames = int(fps * empty_scene_duration)
    bounds_frames = int(fps * bounds_only_duration)
    bbox_placement_frames = int(fps * bbox_placement_duration)
    bbox_still_frames = int(fps * bbox_still_duration)
    asset_replacement_frames = int(fps * asset_replacement_duration)
    asset_still_frames = int(fps * asset_still_duration)
    voxel_replacement_frames = int(fps * voxel_replacement_duration)
    voxel_still_frames = int(fps * voxel_still_duration)
    frames_per_step = int(fps * step_time)
    
    print(f"Creating voxelization 360° video:")
    print(f"- Phase 1 (Empty scene): {empty_frames} frames ({empty_scene_duration}s)")
    print(f"- Phase 2 (Bounds only): {bounds_frames} frames ({bounds_only_duration}s)")
    print(f"- Phase 3 (Bbox placement): {bbox_placement_frames} frames ({bbox_placement_duration}s)")
    print(f"- Phase 3b (Bbox still): {bbox_still_frames} frames ({bbox_still_duration}s)")
    print(f"- Phase 4 (Asset replacement): {asset_replacement_frames} frames ({asset_replacement_duration}s)")
    print(f"- Phase 4b (Asset still): {asset_still_frames} frames ({asset_still_duration}s)")
    print(f"- Phase 5 (Voxel replacement): {voxel_replacement_frames} frames ({voxel_replacement_duration}s)")
    print(f"- Phase 5b (Voxel still): {voxel_still_frames} frames ({voxel_still_duration}s)")
    print(f"- {num_objects} objects, {step_time}s per step")
    print(f"- Total: {total_frames} frames ({total_video_duration}s)")
    
    # Setup scene basics
    bounds_bottom = scene_teaser["bounds_bottom"]
    bounds_top = scene_teaser["bounds_top"]
    all_objects = copy.deepcopy(scene_teaser["objects"])
    
    # Fix flickering for selected samples (weird mesh issue)
    # for obj in all_objects:
    #     if "lamp" in obj.get("desc", "").lower() or "plant" in obj.get("desc", "").lower():
    #         obj["pos"][1] -= 0.01
    
    # Pre-compute voxelized objects for caching (expensive operation)
    print("Pre-computing voxelized objects for caching...")
    voxelized_objects_cache = []
    for i, obj in enumerate(tqdm(all_objects, desc="Voxelizing objects")):
        try:
            jid = obj["jid"] if obj.get("jid") is not None else obj["sampled_asset_jid"]
            mesh = load_mesh_with_transform(get_pth_mesh(jid), obj.get("pos"), obj.get("rot"), obj.get("scale"))
            
            voxel_size = 0.05
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.to_geometry()
            
            voxelized = mesh.voxelized(pitch=voxel_size).fill()
            voxel_points = voxelized.points
            voxel_mesh = trimesh.Trimesh()
            
            for point in voxel_points:
                cube = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size])
                transform = np.eye(4)
                transform[:3, 3] = point
                cube.apply_transform(transform)
                voxel_mesh = trimesh.util.concatenate([voxel_mesh, cube])
            
            voxelized_objects_cache.append(voxel_mesh)
        except Exception as e:
            print(f"Failed to voxelize object {i}: {e}")
            # Fallback to empty mesh
            voxelized_objects_cache.append(trimesh.Trimesh())
    
    print(f"Pre-computed {len(voxelized_objects_cache)} voxelized objects")
    
    # Create output directory
    pth_folder_fig = Path("./eval/viz/360videos-voxelization")
    remove_and_recreate_folder(pth_folder_fig)
    
    # Prepare video writer
    video_path = pth_folder_fig / "voxelization_360_demo.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
    
    # Setup base scene with floor
    bounds_bottom = scene_teaser["bounds_bottom"]
    trimesh_scene_base, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)
    
    # Helper function to add objects with mixed visualization (including voxels)
    def add_objects_mixed_visualization(trimesh_scene, objects, bbox_indices, asset_indices, voxel_indices, bounds_bottom=None, bounds_top=None, voxelized_cache=None):
        """Add objects with some as bboxes, some as assets, and some as voxels"""
        # Add bounds if specified
        if bounds_bottom is not None or bounds_top is not None:
            add_objects_to_trimesh_scene(
                trimesh_scene, [], 
                show_bboxes=False, 
                show_assets=False, 
                show_bounds=True,
                show_assets_voxelized=False,
                bounds_bottom=bounds_bottom,
                bounds_top=bounds_top,
                voxelized_objects_cache=voxelized_cache
            )
        
        # Add bounding boxes for specified indices
        if bbox_indices:
            bbox_objects = [objects[i] for i in bbox_indices]
            add_objects_to_trimesh_scene(
                trimesh_scene, bbox_objects,
                show_bboxes=True,
                show_assets=False,
                show_bounds=False,
                show_assets_voxelized=False,
                bounds_bottom=None,
                bounds_top=None,
                voxelized_objects_cache=voxelized_cache
            )
        
        # Add assets for specified indices
        if asset_indices:
            asset_objects = [objects[i] for i in asset_indices]
            add_objects_to_trimesh_scene(
                trimesh_scene, asset_objects,
                show_bboxes=False,
                show_assets=True,
                show_bounds=False,
                show_assets_voxelized=False,
                bounds_bottom=None,
                bounds_top=None,
                voxelized_objects_cache=voxelized_cache
            )
        
        # Add voxelized assets for specified indices
        if voxel_indices:
            voxel_objects = [objects[i] for i in voxel_indices]
            add_objects_to_trimesh_scene(
                trimesh_scene, voxel_objects,
                show_bboxes=False,
                show_assets=True,
                show_bounds=False,
                show_assets_voxelized=True,
                bounds_bottom=None,
                bounds_top=None,
                voxelized_objects_cache=voxelized_cache
            )
    
    # Generate frames with progress bar
    for frame_idx in tqdm(range(total_frames), desc="Rendering frames"):
        # Calculate angle for continuous rotation through entire video
        angle_progress = frame_idx / total_frames
        angle_degrees = angle_progress * 360
        
        # Determine which phase we're in
        if frame_idx < empty_frames:
            # Phase 1: Empty scene (just floor)
            current_scene = copy.deepcopy(trimesh_scene_base)
            
        elif frame_idx < empty_frames + bounds_frames:
            # Phase 2: Bounds only
            current_scene = copy.deepcopy(trimesh_scene_base)
            add_objects_to_trimesh_scene(
                current_scene, [], 
                show_bboxes=False, 
                show_assets=False, 
                show_bounds=True,
                show_assets_voxelized=False,
                bounds_bottom=bounds_bottom,
                bounds_top=bounds_top
            )
            
        elif frame_idx < empty_frames + bounds_frames + bbox_placement_frames:
            # Phase 3: Incremental bbox placement
            placement_frame = frame_idx - empty_frames - bounds_frames
            bboxes_to_show = min(placement_frame // frames_per_step + 1, num_objects)
            
            current_scene = copy.deepcopy(trimesh_scene_base)
            bbox_indices = list(range(bboxes_to_show))
            asset_indices = []
            voxel_indices = []
            
            add_objects_mixed_visualization(
                current_scene, all_objects, bbox_indices, asset_indices, voxel_indices,
                bounds_bottom, bounds_top
            )
            
        elif frame_idx < empty_frames + bounds_frames + bbox_placement_frames + bbox_still_frames:
            # Phase 3b: All bboxes still
            current_scene = copy.deepcopy(trimesh_scene_base)
            bbox_indices = list(range(num_objects))
            asset_indices = []
            voxel_indices = []
            
            add_objects_mixed_visualization(
                current_scene, all_objects, bbox_indices, asset_indices, voxel_indices,
                bounds_bottom, bounds_top
            )
            
        elif frame_idx < empty_frames + bounds_frames + bbox_placement_frames + bbox_still_frames + asset_replacement_frames:
            # Phase 4: Incremental bbox->asset replacement
            replacement_frame = frame_idx - empty_frames - bounds_frames - bbox_placement_frames - bbox_still_frames
            assets_to_show = min(replacement_frame // frames_per_step + 1, num_objects)
            
            current_scene = copy.deepcopy(trimesh_scene_base)
            bbox_indices = list(range(assets_to_show, num_objects))  # Remaining bboxes
            asset_indices = list(range(assets_to_show))  # Already replaced with assets
            voxel_indices = []
            
            add_objects_mixed_visualization(
                current_scene, all_objects, bbox_indices, asset_indices, voxel_indices,
                bounds_bottom, bounds_top, voxelized_objects_cache
            )
            
        elif frame_idx < empty_frames + bounds_frames + bbox_placement_frames + bbox_still_frames + asset_replacement_frames + asset_still_frames:
            # Phase 4b: All assets still
            current_scene = copy.deepcopy(trimesh_scene_base)
            bbox_indices = []
            asset_indices = list(range(num_objects))
            voxel_indices = []
            
            add_objects_mixed_visualization(
                current_scene, all_objects, bbox_indices, asset_indices, voxel_indices,
                bounds_bottom, bounds_top, voxelized_objects_cache
            )
            
        elif frame_idx < empty_frames + bounds_frames + bbox_placement_frames + bbox_still_frames + asset_replacement_frames + asset_still_frames + voxel_replacement_frames:
            # Phase 5: Incremental asset->voxel replacement
            replacement_frame = frame_idx - empty_frames - bounds_frames - bbox_placement_frames - bbox_still_frames - asset_replacement_frames - asset_still_frames
            voxels_to_show = min(replacement_frame // frames_per_step + 1, num_objects)
            
            current_scene = copy.deepcopy(trimesh_scene_base)
            bbox_indices = []
            asset_indices = list(range(voxels_to_show, num_objects))  # Remaining assets
            voxel_indices = list(range(voxels_to_show))  # Already replaced with voxels
            
            add_objects_mixed_visualization(
                current_scene, all_objects, bbox_indices, asset_indices, voxel_indices,
                bounds_bottom, bounds_top, voxelized_objects_cache
            )
            
        else:
            # Phase 5b: All voxels still (until we complete rotation for looping)
            current_scene = copy.deepcopy(trimesh_scene_base)
            bbox_indices = []
            asset_indices = []
            voxel_indices = list(range(num_objects))
            
            add_objects_mixed_visualization(
                current_scene, all_objects, bbox_indices, asset_indices, voxel_indices,
                bounds_bottom, bounds_top, voxelized_objects_cache
            )
        
        # Render the frame from the current angle
        frame = render_frame_at_angle(
            current_scene, 
            angle_degrees, 
            resolution, 
            camera_height, 
            scene_span, 
            bg_color
        )
        
        # Convert from RGB to BGR (OpenCV uses BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()
    print(f"Created voxelization demonstration video at {video_path}")
    return video_path

def create_360_videos_assets(scene_example, camera_height, pth_folder_fig):
    # Setup video parameters
    resolution = (1536, 1024)  # 3:2 aspect ratio
    fps = 30
    camera_height = 5.5
    bg_color = np.array([240, 240, 240]) / 255.0
    step_time = 0.8  # Time per asset sample
    
    # Calculate timing parameters
    scene_before_duration = 4.0  # 4 seconds showing scene before
    bbox_duration = 4.0  # 4 seconds showing blue bounding box
    num_asset_samples = 10  # Number of different assets to sample
    asset_sampling_duration = num_asset_samples * step_time  # 0.8s per asset sample
    
    total_video_duration = scene_before_duration + bbox_duration + asset_sampling_duration
    total_frames = int(fps * total_video_duration)
    
    # Calculate frame ranges for each phase
    before_frames = int(fps * scene_before_duration)
    bbox_frames = int(fps * bbox_duration)
    sampling_frames = int(fps * asset_sampling_duration)
    frames_per_step = int(fps * step_time)
    
    print(f"Creating asset sampling 360° video:")
    print(f"- Phase 1 (Scene before): {before_frames} frames ({scene_before_duration}s)")
    print(f"- Phase 2 (Blue bbox): {bbox_frames} frames ({bbox_duration}s)")
    print(f"- Phase 3 (Asset sampling): {sampling_frames} frames ({asset_sampling_duration}s)")
    print(f"- {num_asset_samples} asset samples, {step_time}s per sample")
    print(f"- Total: {total_frames} frames ({total_video_duration}s)")
    print(f"- Total rotations: {total_video_duration / 8.0:.1f} (2 full rotations)")
    
    # Extract scene components
    bounds_bottom = scene_example["bounds_bottom"]
    bounds_top = scene_example["bounds_top"]
    all_objects = copy.deepcopy(scene_example["objects"])
    
    # Fix flickering for selected samples (weird mesh issue)
    # for obj in all_objects:
    #     if "lamp" in obj.get("desc", "").lower() or "plant" in obj.get("desc", "").lower() or "vase" in obj.get("desc", "").lower():
    #         obj["pos"][1] -= 0.01
    
    # Separate the scene before (all objects except last) and the target object
    scene_before_objects = all_objects[:-1] if len(all_objects) > 0 else []
    target_object = all_objects[-1] if len(all_objects) > 0 else None
    
    if target_object is None:
        print("Error: No target object found for asset sampling")
        return
    
    # Initialize sampling engine
    sampling_engine = AssetRetrievalModule(
        lambd=0.5, sigma=0.05, temp=0.2, top_p=0.95, top_k=20, 
        asset_size_threshold=0.5, rand_seed=1234, do_print=False
    )
    
    # Pre-sample different assets for the target object
    print("Pre-sampling different assets for target object...")
    sampled_assets = []
    for i in tqdm(range(num_asset_samples), desc="Sampling assets"):
        # Create a temporary scene with the target object
        temp_scene = {
            "room_type": scene_example["room_type"],
            "bounds_bottom": bounds_bottom,
            "bounds_top": bounds_top,
            "objects": scene_before_objects + [copy.deepcopy(target_object)]
        }
        
        # Sample a new asset for the last object
        try:
            sampled_scene = sampling_engine.sample_last_asset(temp_scene, is_greedy_sampling=False)
            sampled_target_object = sampled_scene["objects"][-1]
            sampled_assets.append(sampled_target_object)
        except Exception as e:
            print(f"Failed to sample asset {i}: {e}")
            # Fallback to original object
            sampled_assets.append(copy.deepcopy(target_object))
    
    print(f"Pre-sampled {len(sampled_assets)} different assets")
    
    # Prepare video writer
    video_path = pth_folder_fig / "asset_sampling_360_demo.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, resolution)
    
    # Setup base scene with floor
    trimesh_scene_base, scene_span = setup_trimesh_scene_with_floor(bounds_bottom)
    
    # Generate frames with progress bar
    for frame_idx in tqdm(range(total_frames), desc="Rendering frames"):
        # Calculate angle for continuous rotation through entire video
        angle_progress = frame_idx / total_frames
        angle_degrees = angle_progress * 360
        
        # Determine which phase we're in
        if frame_idx < before_frames:
            # Phase 1: Scene before (without target object)
            current_scene = copy.deepcopy(trimesh_scene_base)
            if scene_before_objects:
                add_objects_to_trimesh_scene(
                    current_scene, scene_before_objects,
                    show_bboxes=False,
                    show_assets=True,
                    show_bounds=False,
                    show_assets_voxelized=False,
                    bounds_bottom=None,
                    bounds_top=None
                )
            
        elif frame_idx < before_frames + bbox_frames:
            # Phase 2: Scene before + blue bounding box for target object
            current_scene = copy.deepcopy(trimesh_scene_base)
            
            # Add scene before objects
            if scene_before_objects:
                add_objects_to_trimesh_scene(
                    current_scene, scene_before_objects,
                    show_bboxes=False,
                    show_assets=True,
                    show_bounds=False,
                    show_assets_voxelized=False,
                    bounds_bottom=None,
                    bounds_top=None
                )
            
            # Add blue bounding box for target object
            add_objects_to_trimesh_scene(
                current_scene, [target_object],
                show_bboxes=True,
                show_assets=False,
                show_bounds=False,
                show_assets_voxelized=False,
                bounds_bottom=None,
                bounds_top=None
            )
            
        else:
            # Phase 3: Asset sampling - show different sampled assets
            sampling_frame = frame_idx - before_frames - bbox_frames
            current_asset_index = min(sampling_frame // frames_per_step, num_asset_samples - 1)
            
            current_scene = copy.deepcopy(trimesh_scene_base)
            
            # Add scene before objects
            if scene_before_objects:
                add_objects_to_trimesh_scene(
                    current_scene, scene_before_objects,
                    show_bboxes=False,
                    show_assets=True,
                    show_bounds=False,
                    show_assets_voxelized=False,
                    bounds_bottom=None,
                    bounds_top=None
                )
            
            # Add current sampled asset
            current_sampled_object = sampled_assets[current_asset_index]
            add_objects_to_trimesh_scene(
                current_scene, [current_sampled_object],
                show_bboxes=False,
                show_assets=True,
                show_bounds=False,
                show_assets_voxelized=False,
                bounds_bottom=None,
                bounds_top=None
            )
        
        # Render the frame from the current angle
        frame = render_frame_at_angle(
            current_scene, 
            angle_degrees, 
            resolution, 
            camera_height, 
            scene_span, 
            bg_color
        )
        
        # Convert from RGB to BGR (OpenCV uses BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()
    print(f"Created asset sampling demonstration video at {video_path}")
    return video_path

def run_viz_for_full_training_dataset():

    # load_dotenv(".env.local")
    load_dotenv(".env.stanley")
    # load_dotenv(".env.sherlock")

    # dataset_train, dataset_val, dataset_test = load_train_val_test_datasets()

    # sampling_engine = AssetRetrievalModule(lambd=0.5, sigma=0.05, temp=0.2, top_p=0.95, top_k=20, asset_size_threshold=0.5, rand_seed=1234, do_print=False)

    # pth_output = Path("./eval/viz/3d-front-train/")
    # remove_and_recreate_folder(pth_output)
    # for idx in tqdm(range(len(dataset_train))):
    #     sample = dataset_train.select([idx])[0]
    #     scene = json.loads(sample.get("sg_input"))
    #     scene_with_assets = sampling_engine.sample_all_assets(scene)
    #     render_scene_and_export(scene_with_assets, idx, pth_output=pth_output)

    # pth_output = Path("./eval/viz/3d-front-train-full-scenes-v2/")
    # remove_and_recreate_folder(pth_output)
    # max_obj_cnt = {}
    # for idx in tqdm(range(len(dataset_train))):
    #     sample = dataset_train.select([idx])[0]
    #     scene = json.loads(sample.get("sg_output"))
    #     scene_id = sample.get("pth_orig_file").split("/")[-2]
    #     n_objects = len(scene.get("objects"))
    #     if max_obj_cnt.get(scene_id) is not None and n_objects <= max_obj_cnt[scene_id]:
    #         print("skipping scene as not more objects")
    #         continue
    #     max_obj_cnt[scene_id] = n_objects
    #     scene_with_assets = sampling_engine.sample_all_assets(scene)
    #     render_scene_and_export(scene_with_assets, scene_id, pth_output=pth_output)

    pth_output_base = Path(os.getenv("PTH_EVAL_VIZ_CACHE"))
    pth_root = os.getenv("PTH_STAGE_2_DEDUP")

    # render just a single image into PTH_EVAL_VIZ_CACHE
    # get random scene from PTH_STAGE_2_DEDUP if json file
    # all_pths = [f for f in os.listdir(pth_root) if f.endswith(".json")]
    # pth = all_pths[0]
    # pth = "9bf7779c-3afd-474d-8343-05df08fda70c-6838264d-6da5-4aae-bc11-b539d0042e14.json"
    # scene_id = pth.split(".")[0]
    # scene = json.load(open(os.path.join(pth_root, pth), "r"))
    # render_scene_and_export(scene, filename=scene_id, pth_output=pth_output_base)

    # render full scenes
    # pth_folder_prefix = "3d-front-train-full-scenes"
    # render_full_scenes_for_room_type("bedroom", pth_root, pth_folder_prefix, pth_output_base)
    # render_full_scenes_for_room_type("livingroom", pth_root, pth_folder_prefix, pth_output_base)
    # render_full_scenes_for_room_type("all", pth_root, pth_folder_prefix, pth_output_base)

    # # render instr scenes
    pth_folder_prefix = "3d-front-train-instr-scenes"
    render_instr_scenes_for_room_type("bedroom", pth_root, pth_folder_prefix, pth_output_base)
    render_instr_scenes_for_room_type("livingroom", pth_root, pth_folder_prefix, pth_output_base)
    render_instr_scenes_for_room_type("all", pth_root, pth_folder_prefix, pth_output_base)
        
if __name__ == "__main__":

    # load_dotenv(".env.local")
    load_dotenv(".env.stanley")
    
    # run_viz_for_full_training_dataset()
    # xvfb-run -a python src/viz.py

    # metrics_raw = json.load(open("/home/martinbucher/git/stan-24-sgllm/eval/metrics-raw/eval_samples_respace_instr_bedroom_qwen1.5B_raw.json"))
    # metrics_raw = json.load(open("/home/martinbucher/git/stan-24-sgllm/eval/metrics-raw/eval_samples_respace_instr_livingroom_qwen1.5B_raw.json"))
    # for seed in range(3):
    #     for i, elem in enumerate(metrics_raw[seed]):
    #         if elem.get("txt_pms_score") == float('inf') or elem.get("txt_pms_score") is float("nan") or elem.get("txt_pms_score") is None or not isinstance(elem.get("txt_pms_score"), float):
    #             print(seed, i)
    #             break
