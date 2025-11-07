# -*- coding: utf-8 -*-
"""
OfficialPosition 原子属性全量写入 + 地名验证（对 Place.历史名称 的包含匹配）
- 加载：沿用稳健加载（rdflib -> 临时 .owl -> owlready2）
- 遍历：OfficialPosition 实例，读取「原始称谓」（若无则回退「官职名称」）
- 解析：parse_title(raw_title)
- 地名：仅当 解析出的“地名候选” 与 任一 Place 的“历史名称”存在【双向包含】关系时才写入
- 其他：核心职称、层级、机构、机构缩写、职系、修饰_*、对齐码系 一律按原逻辑写入
- 输出：*_positions_full.nt / *_positions_full.ttl
"""

from owlready2 import *
from rdflib import Graph
from pathlib import Path
import tempfile
import re
import os, argparse, datetime
def _parse_args():
    ap = argparse.ArgumentParser(description="脚本8：官职解析")
    # 支持 --src（首选）与 --onto（兼容上游习惯），都映射到 dest="src"
    ap.add_argument(
        "--src", "--onto", dest="src", required=False,
        default=os.getenv(
            "ONTO_FILE",  # 若上游用 ONTO_FILE 传入则优先
            os.path.join(os.getenv("OUT_DIR", str(Path.cwd() / "本体结构")), "ontology_dedup_positions_full.nt")
        ),
        help="输入 NT 文件"
    )
    ap.add_argument(
        "--out-dir", dest="out_dir", required=False,
        default=os.getenv("OUT_DIR", str(Path.cwd() / "本体结构")),
        help="输出目录（OUT_DIR）"
    )

    return ap.parse_args()

_args = _parse_args()
SRC = _args.src
OUT_DIR = _args.out_dir

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

OUT_TTL = str(Path(OUT_DIR) / "ontology_dedup_updated.ttl")
OUT_NT  = str(Path(OUT_DIR) / "ontology_dedup_positions_full.nt")
# ========= 源与目标路径 =========


# ========= 引入你的解析器（改成你本地模块名） =========
from 官职解析器 import parse_title

# ========= 机构缩写（用于补齐“机构缩写”） =========
INSTITUTION_ABBR = {
    "都察院": "JCY", "吏部": "LIBU", "户部": "HUBU", "礼部": "LIPU",
    "兵部": "BINGBU", "刑部": "XINGBU", "工部": "GONGBU",
    "太常寺": "TAICHANGSI", "光禄寺": "GUANGLUSI", "太仆寺": "TAIPUSI",
    "大理寺": "DALISI", "鸿胪寺": "HONGLUSI", "通政司": "TONGZHENGSI",
    "宗人府": "ZONGRENFU", "詹事府": "ZHANSHIFU", "国子监": "GUOZIJIAN",
    "布政使司": "BUZHENGSI", "按察使司": "ANCHASI",
    "五军都督府": "WJDD", "都督府": "DUDUFU",
    "都指挥使司": "DUZHIHUISI", "卫所": "WEISUO",
    "锦衣卫": "JINYIWEI",
    "散官": "SANGUAN",
}

# ========= 稳健加载（rdflib → RDF/XML → owlready2） =========
def load_ontology_robust(src_path: str):
    p = Path(src_path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    fmt = {".nt":"nt",".ttl":"turtle",".rdf":"xml",".owl":"xml",".xml":"xml"}.get(p.suffix.lower(), None)
    if fmt is None:
        raise ValueError(f"不支持的输入格式：{p.suffix}")
    g = Graph()
    g.parse(str(p), format=fmt)
    td = tempfile.TemporaryDirectory()
    owl_path = Path(td.name) / (p.stem + "_tmp.owl")
    g.serialize(destination=str(owl_path), format="xml", encoding="utf-8")
    world = World()
    onto  = world.get_ontology(str(owl_path)).load()
    return world, onto, td

world, onto, _tmpdir = load_ontology_robust(SRC)

# ========= 工具：跨命名空间查找 =========
def find_class(names):
    names = set(names)
    for c in list(onto.classes()) + list(world.classes()):
        if c.name in names: return c
        try:
            iri = c.iri
            if any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in names):
                return c
        except Exception:
            pass
    return None

def all_dataprops():
    seen, res = set(), []
    for dp in list(onto.data_properties()) + list(world.data_properties()):
        if dp not in seen:
            seen.add(dp); res.append(dp)
    return res

def find_dataprop_by_names(names):
    names = set(names)
    for dp in all_dataprops():
        if dp.name in names:
            return dp
        try:
            iri = dp.iri
            if any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in names):
                return dp
        except Exception:
            pass
    return None

# ========= 绑定类 =========
OfficialPosition = find_class(["OfficialPosition"])
Place            = find_class(["Place"])
if OfficialPosition is None:
    raise RuntimeError("未找到类：OfficialPosition")
if Place is None:
    raise RuntimeError("未找到类：Place")

# ========= 确保数据属性存在（没有则创建为 FunctionalProperty） =========
def ensure_dp(name: str, domain_cls):
    dp = find_dataprop_by_names([name])
    if dp is None:
        with onto:
            cls = type(name, (DataProperty, FunctionalProperty), {})
            cls.domain = [domain_cls]
            cls.range  = [str]
        dp = find_dataprop_by_names([name])
    return dp

# OfficialPosition 要写的属性
DP_RAW_TITLE     = find_dataprop_by_names(["原始称谓"]) or find_dataprop_by_names(["官职名称"])
if DP_RAW_TITLE is None:
    # 若没有“原始称谓/官职名称”，创建“原始称谓”
    DP_RAW_TITLE = ensure_dp("原始称谓", OfficialPosition)

DP_CORE          = ensure_dp("核心职称", OfficialPosition)
DP_TIER          = ensure_dp("层级", OfficialPosition)
DP_INST          = ensure_dp("机构", OfficialPosition)
DP_INST_ABBR     = ensure_dp("机构缩写", OfficialPosition)
DP_FAMILY        = ensure_dp("职系", OfficialPosition)
DP_DIRMOD        = ensure_dp("修饰_方位", OfficialPosition)
DP_DEPMOD        = ensure_dp("修饰_副", OfficialPosition)
DP_PLACE         = ensure_dp("地名", OfficialPosition)

DP_ALIGN_CORE    = ensure_dp("对齐码_core", OfficialPosition)
DP_ALIGN_INST    = ensure_dp("对齐码_inst", OfficialPosition)
DP_ALIGN_TIER    = ensure_dp("对齐码_tier", OfficialPosition)
DP_ALIGN_LOC_CORE= ensure_dp("对齐码_loc_core", OfficialPosition)
DP_ALIGN_LOC_INST= ensure_dp("对齐码_loc_inst", OfficialPosition)
DP_ALIGN_LOC_FULL= ensure_dp("对齐码_loc_full", OfficialPosition)

# Place 的“历史名称”属性（必须存在；若没有则创建，便于后续填充）
DP_PLACE_HISTNAME = find_dataprop_by_names(["历史名称", "历史名", "古称"])
if DP_PLACE_HISTNAME is None:
    DP_PLACE_HISTNAME = ensure_dp("历史名称", Place)

# ========= 小工具 =========
def get_first_str(dp, inst):
    try:
        vals = list(dp[inst])
        return str(vals[0]).strip() if vals else None
    except Exception:
        return None

def set_single(dp, inst, value: str):
    """将非空字符串写入；value 为空则清空该属性。"""
    try:
        if value is None or str(value).strip() == "":
            dp[inst] = []
        else:
            dp[inst] = [str(value).strip()]
    except Exception:
        pass

def inst_abbr(name: str) -> str:
    if not name: return ""
    return INSTITUTION_ABBR.get(name, "")

# 规范化：去空白、统一标点、去常见后缀空格
def norm_text(s: str) -> str:
    if s is None: return ""
    s = str(s)
    # 去空白
    s = re.sub(r"[\s\u3000]+", "", s)
    # 半全角与标点归一（只做一些常见的）
    s = (s.replace("（","(").replace("）",")")
           .replace("，",",").replace("．",".").replace("・","·")
           .replace("：",":").replace("；",";").replace("—","-")
           .replace("－","-").replace("–","-").replace("—","-"))
    return s

# 预索引：收集 Place 的“历史名称”（已规范化），便于快速匹配
def index_place_histnames():
    histnames = []  # List[str]
    for pl in Place.instances():
        try:
            vals = list(DP_PLACE_HISTNAME[pl]) if DP_PLACE_HISTNAME else []
        except Exception:
            vals = []
        for v in vals:
            nv = norm_text(v)
            if nv:
                histnames.append(nv)
    return histnames

PLACE_HISTNAME_INDEX = index_place_histnames()

def place_name_valid(candidate: str) -> bool:
    """候选地名 与 任一 Place.历史名称 存在双向包含关系则返回 True"""
    c = norm_text(candidate)
    if not c:
        return False
    for hn in PLACE_HISTNAME_INDEX:
        if not hn:
            continue
        # 双向包含：候选 ⊆ 历史名  或  历史名 ⊆ 候选
        if (c in hn) or (hn in c):
            return True
    return False

# ========= 主处理 =========
updated, skipped_raw, failed = 0, 0, 0
place_written, place_skipped = 0, 0

for pos in OfficialPosition.instances():
    # 1) 读取原始称谓（若新属性为空，回退旧字段“官职名称”）
    raw = get_first_str(DP_RAW_TITLE, pos)
    if not raw:
        dp_old = find_dataprop_by_names(["官职名称"])
        raw = get_first_str(dp_old, pos) if dp_old else None
    if not raw:
        skipped_raw += 1
        continue

    # 2) 解析
    try:
        parsed = parse_title(raw)
        d = parsed.__dict__ if hasattr(parsed, "__dict__") else dict(parsed)
    except Exception:
        failed += 1
        continue

    # 3) 写入除“地名”外的所有原子属性
    inst = (d.get("机构") or "").strip()
    abbr = (d.get("机构缩写") or "").strip()
    if not abbr:
        abbr = inst_abbr(inst)

    set_single(DP_CORE,           pos, (d.get("核心职称") or "").strip())
    set_single(DP_TIER,           pos, (d.get("通用层级") or "").strip())
    set_single(DP_INST,           pos, inst)
    set_single(DP_INST_ABBR,      pos, abbr)
    set_single(DP_FAMILY,         pos, (d.get("职系") or "").strip())
    set_single(DP_DIRMOD,         pos, (d.get("修饰_方位") or "").strip())
    set_single(DP_DEPMOD,         pos, (d.get("修饰_副") or "").strip())

    set_single(DP_ALIGN_CORE,     pos, (d.get("对齐码_core") or "").strip())
    set_single(DP_ALIGN_INST,     pos, (d.get("对齐码_inst") or "").strip())
    set_single(DP_ALIGN_TIER,     pos, (d.get("对齐码_tier") or "").strip())
    set_single(DP_ALIGN_LOC_CORE, pos, (d.get("对齐码_loc_core") or "").strip())
    set_single(DP_ALIGN_LOC_INST, pos, (d.get("对齐码_loc_inst") or "").strip())
    set_single(DP_ALIGN_LOC_FULL, pos, (d.get("对齐码_loc_full") or "").strip())

    # 4) 地名：只有验证通过才写入，否则清空（保持严格）
    candidate_place = (d.get("地名") or "").strip()
    if candidate_place and place_name_valid(candidate_place):
        set_single(DP_PLACE, pos, candidate_place)
        place_written += 1
    else:
        # 未通过验证：清空“地名”，避免写入噪声
        set_single(DP_PLACE, pos, "")
        place_skipped += 1

    updated += 1

print(f"[DONE] 原子属性写入：updated={updated}, skipped_raw={skipped_raw}, failed={failed}")
print(f"[PLACE] 地名验证：written={place_written}, skipped_by_check={place_skipped}")

# ========= 保存 =========
g_out = world.as_rdflib_graph()
g_out.serialize(destination=str(OUT_NT),  format="nt")
g_out.serialize(destination=str(OUT_TTL), format="turtle")
print(f"[SAVED] NT  -> {OUT_NT}")
print(f"[SAVED] TTL -> {OUT_TTL}")
