"""
根据 scene_json 程序化生成 scene_summary 文本。
支持：房间类型与尺寸、物体短名与分类、相对锚点距离/方向、朝向描述、边界/碰撞/墙体/物体关系等。
物体类别优先从 dataset/3D-FUTURE-model/model_info.json 的 category 解析（通过 jid/sampled_asset_jid），
否则回退到根据 desc 关键词匹配。

运行方式（在 respace 项目根目录下）：
  # 单文件
  python scripts/scene_summary.py /path/to/scene.json -o /path/to/out.txt --json-out /path/to/out.json

  # 目录：遍历所有 step_*.json，用 tqdm 显示进度
  python scripts/scene_summary.py /path/to/scenes_growth3 -o /path/to/scene_summary --json-out /path/to/summary

  # 指定 model_info（可选）
  python scripts/scene_summary.py /path/to/scenes_growth3 -o ./scene_summary --json-out ./summary --model-info ./dataset/3D-FUTURE-model/model_info.json

依赖：tqdm（可选，用于目录模式进度条） pip install tqdm
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
    _progress_write = tqdm.write
except ImportError:
    def tqdm(iterable, desc=None, unit=None, **kwargs):
        return iterable
    _progress_write = print

# 假定场景单位为米
UNIT_LABEL = "m"

# model_info.json 默认路径（相对 respace 项目根目录）
DEFAULT_MODEL_INFO_PATH = Path(__file__).resolve().parent.parent / "dataset" / "3D-FUTURE-model" / "model_info.json"

# model_info 中 category 字符串 -> 短类名 slug（与 3D-FUTURE 官方类别一致）
# 未在此列出的 category 会按 category.lower().replace(" / ", "_").replace(" ", "_").replace("/", "_") 生成 slug
CATEGORY_TO_SLUG_OVERRIDES: Dict[str, str] = {
    "Lounge Chair / Cafe Chair / Office Chair": "lounge_chair",
    "Lounge Chair / Book-chair / Computer Chair": "lounge_chair",
    "Three-Seat / Multi-seat Sofa": "sofa",
    "Three-Seat / Multi-person sofa": "sofa",
    "Loveseat Sofa": "sofa",
    "Two-seat Sofa": "sofa",
    "L-shaped Sofa": "sofa",
    "Lazy Sofa": "sofa",
    "Chaise Longue Sofa": "sofa",
    "Corner/Side Table": "side_table",
    "Round End Table": "round_end_table",
    "Bookcase / jewelry Armoire": "bookcase",
    "Sideboard / Side Cabinet / Console Table": "sideboard",
    "Sideboard / Side Cabinet / Console": "sideboard",
    "Drawer Chest / Corner cabinet": "drawer_chest",
    "Footstool / Sofastool / Bed End Stool / Stool": "stool",
    "King-size Bed": "bed",
    "Double Bed": "bed",
    "Single bed": "bed",
    "Bed Frame": "bed_frame",
    "Bunk Bed": "bunk_bed",
    "Kids Bed": "kids_bed",
    "Couch Bed": "couch_bed",
    "Tea Table": "tea_table",
}

# 当无法从 model_info 解析时，用 desc 关键词匹配（与 model_info 类别名/描述一致）
DESC_CATEGORY_PATTERNS = [
    (r"dining\s+table", "dining_table"),
    (r"coffee\s+table", "coffee_table"),
    (r"tea\s+table", "tea_table"),
    (r"dining\s+chair", "dining_chair"),
    (r"lounge\s+chair|cafe\s+chair|office\s+chair", "lounge_chair"),
    (r"armchair|arm\s+chair", "armchair"),
    (r"sofa|loveseat|couch|three-seat|multi-seat|two-seat|l-shaped|lazy\s+sofa|chaise", "sofa"),
    (r"nightstand|bedside\s+table", "nightstand"),
    (r"floor\s+lamp", "floor_lamp"),
    (r"pendant\s+lamp|chandelier", "pendant_lamp"),
    (r"ceiling\s+lamp", "ceiling_lamp"),
    (r"wall\s+lamp", "wall_lamp"),
    (r"table\s+lamp", "table_lamp"),
    (r"wardrobe|closet", "wardrobe"),
    (r"king-size\s+bed|double\s+bed|single\s+bed|bed\s+frame|bunk\s+bed|kids\s+bed|couch\s+bed", "bed"),
    (r"desk", "desk"),
    (r"bookcase|bookshelf|shelf", "bookcase"),
    (r"tv\s+stand", "tv_stand"),
    (r"corner\s+table|side\s+table", "side_table"),
    (r"round\s+end\s+table", "round_end_table"),
    (r"barstool", "barstool"),
    (r"bar\b", "bar"),
    (r"stool|footstool|sofastool", "stool"),
    (r"sideboard|console\s+table", "sideboard"),
    (r"drawer\s+chest|corner\s+cabinet", "drawer_chest"),
    (r"table", "table"),
    (r"chair", "chair"),
]

# 可作为“锚点”的类别（椅子等会相对其描述）
ANCHOR_CATEGORIES = {"dining_table", "coffee_table", "tea_table", "desk", "table", "side_table", "round_end_table"}

# 椅子相对桌子的合理距离阈值（米）
CHAIR_TABLE_MAX_ACCEPTABLE_M = 1.0
CHAIR_TABLE_TOO_FAR_M = 1.2

# 方向名称（从锚点看向物体的 8 方向）
DIRECTION_NAMES = [
    "front", "front-right", "right", "back-right",
    "back", "back-left", "left", "front-left",
]

# 边界 / 墙体 / 关系 阈值（米）
WALL_VERY_CLOSE_M = 0.35       # 贴墙判定
FLOOR_Y_EPS = 0.05             # on_floor: pos.y <= this
COLLISION_OVERLAP_EPS = 0.001  # AABB 重叠即碰撞
RELATION_VERY_CLOSE_M = 0.35   # front_against 等“非常近”
ON_TOP_VERTICAL_EPS = 0.08     # 子物体底 >= 父物体顶 - eps
INSIDE_MARGIN = 0.02           # inside: 子 bbox 在父 bbox 内留边距
RELATION_FACE_ANGLE_DEG = 40   # 朝向一致/相对的角度容差（度）


def _point_in_polygon_xy(px: float, pz: float, polygon_xy: List[Tuple[float, float]]) -> bool:
    """射线法判断 (px, pz) 是否在 polygon_xy 内。"""
    n = len(polygon_xy)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, zi = polygon_xy[i]
        xj, zj = polygon_xy[j]
        if ((zi > pz) != (zj > pz)) and (px < (xj - xi) * (pz - zi) / (zj - zi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _object_aabb(obj: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    """物体 AABB：pos 为底中心，返回 (min_x, max_x, min_y, max_y, min_z, max_z)。"""
    pos = obj.get("pos") or [0, 0, 0]
    size = obj.get("size") or [0.1, 0.1, 0.1]
    px, py, pz = pos[0], pos[1], pos[2]
    sx, sy, sz = size[0], size[1], size[2]
    hx, hz = sx / 2, sz / 2
    return (px - hx, px + hx, py, py + sy, pz - hz, pz + hz)


def _aabbs_overlap(
    a: Tuple[float, float, float, float, float, float],
    b: Tuple[float, float, float, float, float, float],
    eps: float = COLLISION_OVERLAP_EPS,
) -> bool:
    """两个 AABB 是否重叠。"""
    return not (
        a[1] < b[0] - eps or b[1] < a[0] - eps
        or a[3] < b[2] - eps or b[3] < a[2] - eps
        or a[5] < b[4] - eps or b[5] < a[4] - eps
    )


def _point_to_segment_dist_2d(
    px: float, pz: float,
    x0: float, z0: float, x1: float, z1: float,
) -> Tuple[float, float, float]:
    """点到线段 (x0,z0)-(x1,z1) 的距离，以及线段单位方向 (dx,dz)。返回 (dist, dx, dz)。"""
    dx = x1 - x0
    dz = z1 - z0
    L = math.hypot(dx, dz)
    if L < 1e-9:
        d = math.hypot(px - x0, pz - z0)
        return (d, 0.0, 0.0)
    ux = dx / L
    uz = dz / L
    t = (px - x0) * ux + (pz - z0) * uz
    t = max(0.0, min(1.0, t))
    nx = x0 + t * dx
    nz = z0 + t * dz
    dist = math.hypot(px - nx, pz - nz)
    return (dist, ux, uz)


def _room_wall_edges(room: Dict[str, Any]) -> List[Tuple[float, float, float, float]]:
    """房间边界 (x,z) 的边列表，每条 (x0,z0,x1,z1)。"""
    pts = _room_bounds_xy(room)
    if len(pts) < 2:
        return []
    edges = []
    for i in range(len(pts)):
        x0, z0 = pts[i]
        x1, z1 = pts[(i + 1) % len(pts)]
        edges.append((x0, z0, x1, z1))
    return edges


def _min_distance_to_walls(
    px: float, pz: float,
    edges: List[Tuple[float, float, float, float]],
) -> Tuple[float, float, float]:
    """到最近墙的距离及该墙的单位方向 (从墙指向室内近似用边的法向)。返回 (dist, wall_dx, wall_dz)。"""
    best_d = float("inf")
    best_wx, best_wz = 0.0, 0.0
    for (x0, z0, x1, z1) in edges:
        d, ux, uz = _point_to_segment_dist_2d(px, pz, x0, z0, x1, z1)
        if d < best_d:
            best_d = d
            # 墙方向 (x1-x0, z1-z0)；室内法向取垂直于墙向左（逆时针）
            best_wx = -uz
            best_wz = ux
    return (best_d, best_wx, best_wz)


def _room_bounds_xy(room: Dict[str, Any]) -> List[Tuple[float, float]]:
    """从 bounds_bottom 取 (x, z) 作为平面顶点。"""
    bottom = room.get("bounds_bottom") or room.get("bounds_top")
    if not bottom:
        return []
    return [(float(p[0]), float(p[2])) for p in bottom]


def _room_bbox_and_centroid(room: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    """返回 (min_x, max_x, min_z, max_z, cx, cz)。"""
    pts = _room_bounds_xy(room)
    if not pts:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    cx = (min_x + max_x) / 2
    cz = (min_z + max_z) / 2
    return min_x, max_x, min_z, max_z, cx, cz


def _room_size_category(width: float, depth: float) -> str:
    area = width * depth
    if area < 15:
        return "small"
    if area < 35:
        return "medium"
    return "large"


def _format_room_type(room_type: str) -> str:
    if not room_type:
        return "unknown"
    s = room_type.strip().lower().replace("_", " ")
    return s.title()


def _category_str_to_slug(category: str) -> str:
    """将 model_info 的 category 字符串转为短类名 slug。"""
    if not category or not category.strip():
        return "object"
    c = category.strip()
    if c in CATEGORY_TO_SLUG_OVERRIDES:
        return CATEGORY_TO_SLUG_OVERRIDES[c]
    slug = (
        c.lower()
        .replace(" / ", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )
    return slug or "object"


def _load_model_info(path: Optional[Path] = None) -> Dict[str, str]:
    """加载 model_info.json，返回 model_id -> slug 映射。"""
    p = path or DEFAULT_MODEL_INFO_PATH
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for item in data:
        mid = item.get("model_id")
        cat = item.get("category")
        if mid and cat:
            out[mid] = _category_str_to_slug(cat)
    return out


def _extract_model_id(jid: Optional[str]) -> Optional[str]:
    """从 jid（如 uuid 或 uuid-(a)-(b)-(c)）提取前 36 位 model_id。"""
    if not jid:
        return None
    s = jid.strip()
    if len(s) >= 36 and s[8] == "-" and s[13] == "-" and s[18] == "-" and s[23] == "-":
        return s[:36]
    return None


def _get_object_category(
    obj: Dict[str, Any],
    model_id_to_slug: Optional[Dict[str, str]] = None,
) -> str:
    """优先用 jid/sampled_asset_jid 查 model_info 得 slug，否则用 desc 匹配。"""
    if model_id_to_slug:
        jid = obj.get("jid") or obj.get("sampled_asset_jid")
        mid = _extract_model_id(jid)
        if mid and mid in model_id_to_slug:
            return model_id_to_slug[mid]
    desc = obj.get("desc") or ""
    return _category_from_desc(desc)


def _category_from_desc(desc: str) -> str:
    if not desc:
        return "object"
    d = desc.lower()
    for pattern, name in DESC_CATEGORY_PATTERNS:
        if re.search(pattern, d):
            return name
    return "object"


def _quat_to_yaw_rad(rot: List[float]) -> float:
    """四元数 [x,y,z,w] -> 绕 Y 轴旋转弧度（yaw）。Y 向上，前方为 +Z 时 yaw=0。"""
    if len(rot) < 4:
        return 0.0
    x, y, z, w = rot[0], rot[1], rot[2], rot[3]
    # 标准 yaw 公式
    siny = 2.0 * (w * y + z * x)
    cosy = 1.0 - 2.0 * (x * x + y * y)
    return math.atan2(siny, cosy)


def _yaw_to_direction_name(yaw_rad: float) -> str:
    """yaw 弧度 -> 8 方向名。"""
    deg = math.degrees(yaw_rad) % 360
    idx = round(deg / 45) % 8
    return DIRECTION_NAMES[idx]


def _angle_between_dirs_rad(ax: float, az: float, bx: float, bz: float) -> float:
    """向量 (ax,az) 与 (bx,bz) 的夹角 [0, pi]。"""
    dot = ax * bx + az * bz
    na = math.hypot(ax, az)
    nb = math.hypot(bx, bz)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (na * nb)))
    return math.acos(cos)


def _direction_from_to(ox: float, oz: float, tx: float, tz: float) -> str:
    """从 (ox,oz) 指向 (tx,tz) 的 8 方向（以 +Z 为 front）。"""
    dx = tx - ox
    dz = tz - oz
    dist = math.hypot(dx, dz)
    if dist < 1e-6:
        return "center"
    # 世界坐标系：+Z 为“前”，+X 为“右”
    yaw = math.atan2(dx, dz)
    return _yaw_to_direction_name(yaw)


def _assign_short_names(
    objects: List[Dict[str, Any]],
    model_id_to_slug: Optional[Dict[str, str]] = None,
) -> List[Tuple[Dict, str]]:
    """为每个物体分配短名 category_1, category_2, ... 优先用 model_info 类别。"""
    counts: Dict[str, int] = {}
    result: List[Tuple[Dict, str]] = []
    for obj in objects:
        cat = _get_object_category(obj, model_id_to_slug)
        counts[cat] = counts.get(cat, 0) + 1
        short = f"{cat}_{counts[cat]}"
        result.append((obj, short))
    return result


def _position_in_room_label(
    px: float, pz: float,
    min_x: float, max_x: float, min_z: float, max_z: float,
    cx: float, cz: float,
) -> str:
    """物体在房间内的位置描述。"""
    tol = 0.5
    if abs(px - cx) <= tol and abs(pz - cz) <= tol:
        return "center of room"
    labels = []
    if px <= min_x + tol:
        labels.append("left")
    elif px >= max_x - tol:
        labels.append("right")
    if pz <= min_z + tol:
        labels.append("back")
    elif pz >= max_z - tol:
        labels.append("front")
    if not labels:
        return "mid area"
    return " ".join(labels) + " side of room"


def _orientation_label(yaw_rad: float, room_approx_axis_rad: float = 0.0) -> str:
    """粗略朝向：与房间轴线对齐或四向。"""
    deg = math.degrees(yaw_rad) % 180
    if deg <= 22.5 or deg >= 157.5:
        return "aligned with room axis"
    if 67.5 <= deg <= 112.5:
        return "perpendicular to room axis"
    return f"facing {_yaw_to_direction_name(yaw_rad)}"


def _is_facing_point(
    ox: float, oz: float, yaw_rad: float,
    tx: float, tz: float,
    max_angle_deg: float = 55,
) -> bool:
    """物体在 (ox,oz) 朝向 yaw，是否大致面向 (tx,tz)。"""
    dx = tx - ox
    dz = tz - oz
    dist = math.hypot(dx, dz)
    if dist < 1e-6:
        return True
    # 物体前方方向 (Y-up, 前为 +Z)
    fx = math.sin(yaw_rad)
    fz = math.cos(yaw_rad)
    angle = math.degrees(_angle_between_dirs_rad(fx, fz, dx, dz))
    return angle <= max_angle_deg


def _distance_assessment(dist_m: float, max_acceptable: float = CHAIR_TABLE_MAX_ACCEPTABLE_M) -> str:
    if dist_m <= max_acceptable:
        return "acceptable"
    if dist_m >= CHAIR_TABLE_TOO_FAR_M:
        return f"too far (expected < {max_acceptable:.1f}m)"
    return "slightly far"


def _object_front_back_left_right(yaw_rad: float) -> Tuple[float, float, float, float, float, float, float, float]:
    """物体前/后/左/右方向 (x,z)。返回 (fx, fz, bx, bz, lx, lz, rx, rz)。前=+Z 时 yaw=0。"""
    fx = math.sin(yaw_rad)
    fz = math.cos(yaw_rad)
    bx, bz = -fx, -fz
    lx, lz = -fz, fx
    rx, rz = fz, -fx
    return (fx, fz, bx, bz, lx, lz, rx, rz)


def _wall_relation(
    px: float, pz: float, yaw_rad: float,
    dist_to_wall: float, wall_dx: float, wall_dz: float,
) -> Optional[str]:
    """根据物体朝向与墙法向判断 against_wall 或 side_against_wall。墙法向 (wall_dx, wall_dz) 指向室内。"""
    if dist_to_wall > WALL_VERY_CLOSE_M:
        return None
    _, _, bx, bz, lx, lz, rx, rz = _object_front_back_left_right(yaw_rad)
    # 物体背对墙：back 指向墙 = back 与 (-wall_dx, -wall_dz) 同向
    dot_back = bx * (-wall_dx) + bz * (-wall_dz)
    dot_left = lx * (-wall_dx) + lz * (-wall_dz)
    dot_right = rx * (-wall_dx) + rz * (-wall_dz)
    dot_front = (math.sin(yaw_rad)) * (-wall_dx) + (math.cos(yaw_rad)) * (-wall_dz)
    cos_lim = math.cos(math.radians(RELATION_FACE_ANGLE_DEG))
    if dot_back >= cos_lim:
        return "against_wall"
    if dot_left >= cos_lim or dot_right >= cos_lim or dot_front >= cos_lim:
        return "side_against_wall"
    return None


def _center_dist_2d(ax: float, az: float, bx: float, bz: float) -> float:
    return math.hypot(ax - bx, az - bz)


def _relation_two_objects(
    child_pos: List[float],
    child_size: List[float],
    child_yaw: float,
    parent_pos: List[float],
    parent_size: List[float],
    parent_yaw: float,
) -> Optional[str]:
    """判定 child 相对 parent 的关系：front_against, front_to_front, leftright_to_leftright, side_by_side, back_to_back, on_top_of, inside。"""
    cx, cy, cz = child_pos[0], child_pos[1], child_pos[2]
    csx, csy, csz = child_size[0], child_size[1], child_size[2]
    px, py, pz = parent_pos[0], parent_pos[1], parent_pos[2]
    psx, psy, psz = parent_size[0], parent_size[1], parent_size[2]
    # AABB (底中心)
    c_min_x = cx - csx / 2
    c_max_x = cx + csx / 2
    c_min_y = cy
    c_max_y = cy + csy
    c_min_z = cz - csz / 2
    c_max_z = cz + csz / 2
    p_min_x = px - psx / 2
    p_max_x = px + psx / 2
    p_min_y = py
    p_max_y = py + psy
    p_min_z = pz - psz / 2
    p_max_z = pz + psz / 2
    dist_2d = _center_dist_2d(cx, cz, px, pz)
    # on_top_of: 子底 >= 父顶 - eps，且 xz 有重叠
    if c_min_y >= p_max_y - ON_TOP_VERTICAL_EPS:
        if not (c_max_x < p_min_x or c_min_x > p_max_x or c_max_z < p_min_z or c_min_z > p_max_z):
            return "on_top_of"
    # inside: 子 bbox 在父 bbox 内（留边距）
    margin = INSIDE_MARGIN
    if (
        c_min_x >= p_min_x + margin and c_max_x <= p_max_x - margin
        and c_min_z >= p_min_z + margin and c_max_z <= p_max_z - margin
        and c_min_y >= p_min_y + margin and c_max_y <= p_max_y - margin
    ):
        return "inside"
    if dist_2d > RELATION_VERY_CLOSE_M * 2:
        return None
    cfx, cfz, cbx, cbz, clx, clz, crx, crz = _object_front_back_left_right(child_yaw)
    pfx, pfz, pbx, pbz, plx, plz, prx, prz = _object_front_back_left_right(parent_yaw)
    # 从 child 指向 parent 的单位向量
    dx = px - cx
    dz = pz - cz
    d_len = math.hypot(dx, dz)
    if d_len < 1e-9:
        return None
    ux, uz = dx / d_len, dz / d_len
    cos_deg = math.cos(math.radians(RELATION_FACE_ANGLE_DEG))
    # child front 指向 parent
    child_faces_parent = cfx * ux + cfz * uz >= cos_deg
    # parent front 指向 child（从 parent 指向 child 的单位向量点乘 parent_forward）
    p_to_c_x = cx - px
    p_to_c_z = cz - pz
    p_to_c_len = math.hypot(p_to_c_x, p_to_c_z)
    parent_faces_child = False
    if p_to_c_len >= 1e-9:
        pucx = p_to_c_x / p_to_c_len
        pucz = p_to_c_z / p_to_c_len
        parent_faces_child = pfx * pucx + pfz * pucz >= cos_deg
    if child_faces_parent and parent_faces_child:
        return "front_to_front"
    if child_faces_parent:
        return "front_against"
    # back_to_back: 两者 back 都背对对方
    if cbx * (-ux) + cbz * (-uz) >= cos_deg and pbx * ux + pbz * uz >= cos_deg:
        return "back_to_back"
    # leftright_to_leftright: 一侧对一侧
    if (clx * ux + clz * uz >= cos_deg or crx * ux + crz * uz >= cos_deg) and (
        plx * (-ux) + plz * (-uz) >= cos_deg or prx * (-ux) + prz * (-uz) >= cos_deg
    ):
        return "leftright_to_leftright"
    if abs(clx * ux + clz * uz) <= math.sin(math.radians(RELATION_FACE_ANGLE_DEG)) or abs(crx * ux + crz * uz) <= math.sin(math.radians(RELATION_FACE_ANGLE_DEG)):
        return "side_by_side"
    return None


def build_room_summary(room: Dict[str, Any]) -> Dict[str, Any]:
    """从 scene 提取房间信息。"""
    min_x, max_x, min_z, max_z, cx, cz = _room_bbox_and_centroid(room)
    width = round(max_x - min_x, 2)
    depth = round(max_z - min_z, 2)
    room_type = room.get("room_type") or "unknown"
    polygon_xy = _room_bounds_xy(room)
    return {
        "room_type": _format_room_type(room_type),
        "room_type_raw": room_type,
        "width_m": width,
        "depth_m": depth,
        "size_category": _room_size_category(width, depth),
        "centroid_xz": (cx, cz),
        "bounds_xz": (min_x, max_x, min_z, max_z),
        "polygon_xy": polygon_xy,
        "wall_edges": _room_wall_edges(room),
    }


def build_object_summaries(
    room: Dict[str, Any],
    room_info: Dict[str, Any],
    model_info_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """为每个物体构建摘要：短名、位置、朝向、相对锚点、边界/墙体/碰撞/物体关系等。"""
    objects = room.get("objects") or []
    model_id_to_slug = _load_model_info(model_info_path) if model_info_path is not None else _load_model_info()
    named = _assign_short_names(objects, model_id_to_slug)
    min_x, max_x, min_z, max_z = room_info["bounds_xz"]
    cx, cz = room_info["centroid_xz"]
    polygon_xy = room_info.get("polygon_xy") or []
    wall_edges = room_info.get("wall_edges") or []

    # 预计算每个物体的 AABB
    aabbs = [_object_aabb(obj) for obj in objects]

    # 找出锚点物体（桌子等）
    anchor_indices: Dict[str, int] = {}
    for i, (obj, short) in enumerate(named):
        cat = short.rsplit("_", 1)[0]
        if cat in ANCHOR_CATEGORIES:
            anchor_indices[short] = i

    result: List[Dict[str, Any]] = []
    for i, (obj, short) in enumerate(named):
        pos = obj.get("pos") or [0, 0, 0]
        rot = obj.get("rot") or [0, 0, 0, 1]
        size = obj.get("size") or [0.1, 0.1, 0.1]
        px, py, pz = pos[0], pos[1], pos[2]
        yaw = _quat_to_yaw_rad(rot)

        cat = short.rsplit("_", 1)[0]
        position_label = _position_in_room_label(
            px, pz, min_x, max_x, min_z, max_z, cx, cz
        )
        orientation_label = _orientation_label(yaw)

        entry: Dict[str, Any] = {
            "index": i + 1,
            "short_name": short,
            "category": cat,
            "position_room": position_label,
            "orientation": orientation_label,
            "pos_xz": (round(px, 3), round(pz, 3)),
            "yaw_rad": yaw,
        }

        # out of boundary
        out_of_boundary = bool(polygon_xy and not _point_in_polygon_xy(px, pz, polygon_xy))
        entry["out_of_boundary"] = out_of_boundary

        # on_floor
        entry["on_floor"] = py <= FLOOR_Y_EPS

        # against_wall / side_against_wall（仅当到最近墙距离 <= WALL_VERY_CLOSE_M=0.35m 且背/侧对墙时为非 null）
        wall_relation: Optional[str] = None
        dist_wall_m: Optional[float] = None
        if wall_edges:
            dist_wall, wdx, wdz = _min_distance_to_walls(px, pz, wall_edges)
            dist_wall_m = round(dist_wall, 3)
            wall_relation = _wall_relation(px, pz, yaw, dist_wall, wdx, wdz)
        entry["against_wall"] = wall_relation
        entry["distance_to_nearest_wall_m"] = dist_wall_m

        # 碰撞：与其它物体 AABB 重叠
        collisions: List[str] = []
        for j in range(len(objects)):
            if j == i:
                continue
            if _aabbs_overlap(aabbs[i], aabbs[j]):
                collisions.append(named[j][1])
        entry["collisions"] = collisions

        # 物体间关系：对每个其它物体判定 relation
        relations_to_objects: List[Dict[str, str]] = []
        for j in range(len(objects)):
            if j == i:
                continue
            other_obj = objects[j]
            other_short = named[j][1]
            other_pos = other_obj.get("pos") or [0, 0, 0]
            other_size = other_obj.get("size") or [0.1, 0.1, 0.1]
            other_rot = other_obj.get("rot") or [0, 0, 0, 1]
            other_yaw = _quat_to_yaw_rad(other_rot)
            rel = _relation_two_objects(pos, size, yaw, other_pos, other_size, other_yaw)
            if rel:
                relations_to_objects.append({"target": other_short, "relation": rel})
        entry["relations_to_objects"] = relations_to_objects

        # 若是椅子类，找最近的桌子锚点并计算相对信息
        if cat in ("dining_chair", "lounge_chair", "chair", "armchair"):
            best_anchor: Optional[str] = None
            best_dist: Optional[float] = None
            best_anchor_idx: Optional[int] = None
            for aname, aidx in anchor_indices.items():
                aobj = objects[aidx]
                ap = aobj.get("pos") or [0, 0, 0]
                ax, az = ap[0], ap[2]
                dist = math.hypot(px - ax, pz - az)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_anchor = aname
                    best_anchor_idx = aidx
            if best_anchor is not None and best_dist is not None:
                aobj = objects[best_anchor_idx]
                ap = aobj.get("pos") or [0, 0, 0]
                ax, az = ap[0], ap[2]
                direction = _direction_from_to(ax, az, px, pz)
                assessment = _distance_assessment(best_dist)
                facing = _is_facing_point(px, pz, yaw, ax, az)
                entry["relative_to_anchor"] = {
                    "anchor": best_anchor,
                    "distance_m": round(best_dist, 2),
                    "direction": direction,
                    "distance_assessment": assessment,
                    "facing_anchor": facing,
                }
        result.append(entry)
    return result


def scene_summary_dict(
    scene: Dict[str, Any],
    model_info_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """从 scene_json 生成结构化 summary。物体类别优先从 model_info 解析。"""
    room_info = build_room_summary(scene)
    object_summaries = build_object_summaries(scene, room_info, model_info_path)
    return {
        "room": room_info,
        "objects": object_summaries,
    }


def format_scene_summary(summary: Dict[str, Any]) -> str:
    """将结构化 summary 格式化为可读文本。"""
    lines: List[str] = ["Scene Summary:", ""]
    room = summary["room"]
    lines.append("Room:")
    lines.append(f"- Type: {room['room_type']}")
    lines.append(f"- Size: {room['size_category']} ({room['width_m']}{UNIT_LABEL} x {room['depth_m']}{UNIT_LABEL})")
    lines.append("")

    for obj in summary["objects"]:
        idx = obj["index"]
        short = obj["short_name"]
        lines.append(f"{idx}. {short}")
        lines.append(f"   - Position: {obj['position_room']}")

        if obj.get("out_of_boundary"):
            lines.append("   - **out_of_boundary**: object is outside room bounds")
        if obj.get("on_floor"):
            lines.append("   - **on_floor**: object stands on the ground")
        aw = obj.get("against_wall")
        if aw:
            lines.append(f"   - **{aw}**: object's back/side faces the wall, very close")

        if obj.get("collisions"):
            lines.append(f"   - **Collisions**: {', '.join(obj['collisions'])}")
        for ro in obj.get("relations_to_objects") or []:
            lines.append(f"   - **{ro['relation']}** with {ro['target']}")

        rel = obj.get("relative_to_anchor")
        if rel:
            dist = rel["distance_m"]
            assessment = rel["distance_assessment"]
            if assessment == "acceptable":
                dist_note = f"{dist}{UNIT_LABEL} ({assessment})"
            else:
                dist_note = f"{dist}{UNIT_LABEL} ({assessment})"
            lines.append(f"   - Relative to {rel['anchor']}:")
            lines.append(f"     - Distance: {dist_note}")
            lines.append(f"     - Direction: {rel['direction']}")
            anchor_label = rel["anchor"].rsplit("_", 1)[0].replace("_", " ")
            if rel.get("facing_anchor"):
                lines.append(f"   - Orientation: facing {anchor_label} (correct)")
            else:
                lines.append(f"   - Orientation: not facing {anchor_label}")
        else:
            lines.append(f"   - Orientation: {obj['orientation']}")

        lines.append("")

    return "\n".join(lines).strip()


def generate_scene_summary(
    scene: Dict[str, Any],
    model_info_path: Optional[Path] = None,
) -> str:
    """一步：从 scene 字典生成 summary 文本。"""
    summary = scene_summary_dict(scene, model_info_path)
    return format_scene_summary(summary)


def _to_serializable(obj: Any) -> Any:
    """将 dict/list 中的 tuple 转为 list，便于 JSON 序列化。"""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def _collect_scene_jsons(root: Path) -> List[Path]:
    """递归收集目录下所有 scene JSON 文件（step_*.json），保持稳定顺序。"""
    out: List[Path] = []
    for p in sorted(root.rglob("*.json")):
        if p.is_file() and p.name.startswith("step_") and p.name.endswith(".json"):
            out.append(p)
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate scene_summary from scene JSON file(s). "
        "If input is a directory, iterates over all step_*.json under it."
    )
    parser.add_argument(
        "scene_json",
        type=Path,
        help="Path to a single scene JSON file, or a directory (e.g. scenes_growth3) to process all step_*.json",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path: for a single file, the .txt path; for a directory, the folder for all .txt (default: same dir or scene_summary)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional: folder to write structured summary JSONs (when input is dir, same relative path under this folder)",
    )
    parser.add_argument(
        "--model-info",
        type=Path,
        default="/home2/zhangjiawei/respace/dataset/3D-FUTURE-model/model_info.json",
        help="Path to 3D-FUTURE model_info.json (default: dataset/3D-FUTURE-model/model_info.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files; by default skip already processed files",
    )
    args = parser.parse_args()

    path = args.scene_json.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        # 单文件模式
        scene = json.loads(path.read_text(encoding="utf-8"))
        summary_dict = scene_summary_dict(scene, model_info_path=args.model_info)
        text = format_scene_summary(summary_dict)
        out_txt = args.output or path.parent / "scene_summary.txt"
        out_txt = out_txt.resolve()
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(text, encoding="utf-8")
        print(f"Wrote: {out_txt}")
        if args.json_out:
            args.json_out = args.json_out.resolve()
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(_to_serializable(summary_dict), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote: {args.json_out}")
        return

    # 目录模式：迭代所有 step_*.json
    if not path.is_dir():
        raise NotADirectoryError(path)
    json_files = _collect_scene_jsons(path)
    if not json_files:
        print(f"No step_*.json found under {path}")
        return

    output_dir = (args.output or path.parent / "scene_summary").resolve()
    json_out_dir = args.json_out.resolve() if args.json_out else None
    if json_out_dir and not json_out_dir.is_dir():
        json_out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skip = 0
    for json_path in tqdm(json_files, desc="scene_summary", unit="file"):
        # 输出路径（与下方写入逻辑一致）
        try:
            rel = json_path.resolve().relative_to(path.resolve())
        except ValueError:
            rel = Path(json_path.name)
        rel_stem = rel.with_suffix("")
        out_txt = output_dir / rel.parent / f"{rel_stem.name}.txt"
        out_json = (json_out_dir / rel.parent / json_path.name) if json_out_dir else None

        # 已解析过则跳过（以 .txt 为准；除非 --force 则覆盖）
        if not args.force and out_txt.exists():
            n_skip += 1
            continue

        try:
            scene = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            _progress_write(f"Skip {json_path}: {e}")
            continue
        summary_dict = scene_summary_dict(scene, model_info_path=args.model_info)
        text = format_scene_summary(summary_dict)

        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(text, encoding="utf-8")

        if json_out_dir and out_json is not None:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(
                json.dumps(_to_serializable(summary_dict), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        n_ok += 1

    print(f"Done: {n_ok} written, {n_skip} skipped (existing), {len(json_files)} total -> {output_dir}")
    if json_out_dir:
        print(f"       JSON summaries -> {json_out_dir}")


if __name__ == "__main__":
    main()
