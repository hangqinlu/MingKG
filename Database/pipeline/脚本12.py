# -*- coding: utf-8 -*-
"""
Place 合并脚本（按“现代名称”同名合并）
- 现代名称来源：DataProperty / AnnotationProperty / PropAssertion(about=Place, prop in {现代名称, 今名})
- 迁移（loser → survivor）：
    1) 所有指向 loser 的对象属性引用，重定向到 survivor（保持去重）
    2) PropAssertion：about 从 loser 改为 survivor；重复（prop+value_norm/value）删除
       且补 TextProvenance.contains：添加 survivor，移除 loser
    3) 数据/注释属性：仅当 survivor 缺失时，用 loser 的值补齐（冲突保留 survivor）
- 幸存者选择：信息度最高（属性更全 + 引用更多 + PA 更多）；并列按 IRI
- 输出：合并日志 + 保存为 NT/TTL
"""

import re
from pathlib import Path
import tempfile

import rdflib as rd
import owlready2 as ow


# ========= 路径 =========
import argparse
import os

# ====== 路径配置 ======
def _parse_args():
    ap = argparse.ArgumentParser(description="脚本12：区划层级补齐")
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
SRC      = Path(_args.src).expanduser().resolve()
OUT_DIR  = Path(_args.out_dir).expanduser().resolve()

# 确保输出目录存在
OUT_DIR.mkdir(parents=True, exist_ok=True)



OUT_NT  = Path(SRC).with_name("ontology_places_merged.nt")
OUT_TTL = Path(SRC).with_name("ontology_places_merged.ttl")


# ========= 稳健加载（rdflib → RDF/XML → owlready2） =========
def load_ontology_robust(src_path: str):
    p = Path(src_path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    fmt = {
        ".nt": "nt",
        ".ttl": "turtle",
        ".rdf": "xml",
        ".owl": "xml",
        ".xml": "xml",
    }.get(p.suffix.lower(), None)

    g = rd.Graph()
    g.parse(str(p), format=(fmt or "turtle"))

    td = tempfile.TemporaryDirectory()
    owl_path = Path(td.name) / (p.stem + "_tmp.owl")
    g.serialize(destination=str(owl_path), format="xml", encoding="utf-8")

    world = ow.World()
    onto  = world.get_ontology(str(owl_path)).load()
    return world, onto, td


world, onto, _tmpdir = load_ontology_robust(SRC)


# ========= 跨命名空间工具 =========
def all_objprops():
    seen, res = set(), []
    for op in list(onto.object_properties()) + list(world.object_properties()):
        if op not in seen:
            seen.add(op); res.append(op)
    return res

def all_dataprops():
    seen, res = set(), []
    for dp in list(onto.data_properties()) + list(world.data_properties()):
        if dp not in seen:
            seen.add(dp); res.append(dp)
    return res

def all_annprops():
    seen, res = set(), []
    for ap in list(onto.annotation_properties()) + list(world.annotation_properties()):
        if ap not in seen:
            seen.add(ap); res.append(ap)
    return res

def find_class(names):
    names = set(names)
    for c in list(onto.classes()) + list(world.classes()):
        try:
            if c.name in names: return c
            iri = c.iri
            if any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in names):
                return c
        except Exception:
            pass
    raise RuntimeError(f"未找到类：{names}")

def find_objprop_by_name(name):
    for op in all_objprops():
        if op.name == name:
            return op
    return None

def find_dataprop_by_names(names):
    names = set(names)
    for dp in all_dataprops():
        try:
            if dp.name in names: return dp
            iri = dp.iri
            if any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in names):
                return dp
        except Exception:
            pass
    return None

def find_annprop_by_names(names):
    names = set(names)
    for ap in all_annprops():
        try:
            if ap.name in names: return ap
            iri = ap.iri
            if any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in names):
                return ap
        except Exception:
            pass
    return None


# ========= 常用类/属性绑定 =========
Place          = find_class(["Place"])
PropAssertion  = find_class(["PropAssertion"])
TextProvenance = find_class(["TextProvenance"])

about       = find_objprop_by_name("about")
derivedFrom = find_objprop_by_name("derivedFrom")
contains    = find_objprop_by_name("contains")

dp_pa_prop       = find_dataprop_by_names(["prop"])
dp_pa_value      = find_dataprop_by_names(["value"])
dp_pa_value_norm = find_dataprop_by_names(["value_norm"])


# ========= 小工具 =========
def list_vals(prop, inst):
    try:
        return list(prop[inst]) or []
    except Exception:
        return []

def set_stripped(vals):
    out = set()
    for v in vals:
        s = str(v).strip()
        if s:
            out.add(s)
    return out

def normalize_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\s\u3000]+", "", s)     # 去所有空白
    s = s.replace("，", ",").replace("．",".").replace("・","·")
    return s

def get_dp_values(inst, names):
    vals = set()
    for nm in names:
        dp = find_dataprop_by_names([nm])
        if dp is not None:
            vals |= set_stripped(list_vals(dp, inst))
    return vals

def get_ap_values(inst, names):
    vals = set()
    for nm in names:
        ap = find_annprop_by_names([nm])
        if ap is not None:
            try:
                vals |= set_stripped(list(ap[inst]))
            except Exception:
                pass
    return vals

def get_pa_values_about(inst, names):
    vals = set()
    if dp_pa_prop is None:
        return vals
    for pa in list(PropAssertion.instances()):
        try:
            if inst in about[pa]:
                p = (list(dp_pa_prop[pa]) or [""])[0]
                if str(p) in names:
                    # value_norm 优先；无则 value
                    vn = None
                    if dp_pa_value_norm is not None:
                        lst = list(dp_pa_value_norm[pa]); vn = lst[0] if lst else None
                    if vn is None and dp_pa_value is not None:
                        lst = list(dp_pa_value[pa]); vn = lst[0] if lst else None
                    if vn is not None:
                        s = str(vn).strip()
                        if s:
                            vals.add(s)
        except Exception:
            pass
    return vals

# 现代名称的取值（DP/AP/PA）
MODERN_NAME_ALIASES = {
    "dp": ["现代名称", "今名"],
    "ap": ["现代名称", "今名"],
    "pa": ["现代名称", "今名"],
}

def modern_names_of(pl) -> set:
    vals = set()
    vals |= get_dp_values(pl, MODERN_NAME_ALIASES["dp"])
    vals |= get_ap_values(pl, MODERN_NAME_ALIASES["ap"])
    vals |= get_pa_values_about(pl, MODERN_NAME_ALIASES["pa"])
    return set(normalize_name(v) for v in vals if v and normalize_name(v))


# ========= 信息度 & 幸存者选择 =========
def info_score_place(pl) -> int:
    score = 0
    # 数据/注释属性是否有值
    for dp in all_dataprops():
        try:
            if dp[pl]: score += 1
        except Exception:
            pass
    for ap in all_annprops():
        try:
            if ap[pl]: score += 1
        except Exception:
            pass

    # 被对象属性引用的次数（粗略统计：遍历主体）
    refs = 0
    inds = list(world.individuals())
    for op in all_objprops():
        for subj in inds:
            try:
                if pl in op[subj]:
                    refs += 1
            except Exception:
                pass
    score += refs

    # 被 PropAssertion.about 的数量
    pa_count = 0
    for pa in list(PropAssertion.instances()):
        try:
            if pl in about[pa]:
                pa_count += 1
        except Exception:
            pass
    score += pa_count
    return score

def pick_survivor(places):
    ranked = sorted(places, key=lambda x: (info_score_place(x), getattr(x, "iri", "")), reverse=True)
    return ranked[0], ranked[1:]


# ========= 迁移（loser → survivor） =========
def migrate_place(loser, survivor):
    if loser == survivor:
        return

    # 1) 对象属性：所有主体的该对象属性值里，用 survivor 替换 loser（并去重）
    inds = list(world.individuals())
    for op in all_objprops():
        for subj in inds:
            try:
                lst = list(op[subj]) or []
            except Exception:
                continue
            if not lst:
                continue
            if loser not in lst:
                continue
            # 替换并去重
            nlst = [survivor if x == loser else x for x in lst]
            dedup, seen = [], set()
            for x in nlst:
                if x not in seen:
                    seen.add(x); dedup.append(x)
            try:
                op[subj] = dedup
            except Exception:
                pass

    # 2) PropAssertion：about(loser) → about(survivor)
    #    若 (prop, value_norm/value) 已在 survivor 上存在，则删除重复的 PA
    existing = set()
    for pa in list(PropAssertion.instances()):
        try:
            if survivor in about[pa]:
                p = (list(dp_pa_prop[pa]) or [""])[0] if dp_pa_prop else ""
                vn = None
                if dp_pa_value_norm is not None:
                    lst = list(dp_pa_value_norm[pa]); vn = lst[0] if lst else None
                if vn is None and dp_pa_value is not None:
                    lst = list(dp_pa_value[pa]); vn = lst[0] if lst else None
                existing.add((str(p), str(vn)))
        except Exception:
            pass

    for pa in list(PropAssertion.instances()):
        try:
            if loser in about[pa]:
                p = (list(dp_pa_prop[pa]) or [""])[0] if dp_pa_prop else ""
                vn = None
                if dp_pa_value_norm is not None:
                    lst = list(dp_pa_value_norm[pa]); vn = lst[0] if lst else None
                if vn is None and dp_pa_value is not None:
                    lst = list(dp_pa_value[pa]); vn = lst[0] if lst else None
                key = (str(p), str(vn))

                if key in existing:
                    # 与 survivor 重复，直接删除
                    try:
                        ow.destroy_entity(pa)
                    except Exception:
                        pass
                    continue

                # 改 about：loser -> survivor
                try:
                    about[pa].remove(loser)
                except Exception:
                    pass
                if survivor not in about[pa]:
                    about[pa].append(survivor)

                # 修补 contains：把 survivor 加入，移除 loser
                for tp in list(derivedFrom[pa]):
                    try:
                        cur = list(contains[tp]) if contains[tp] else []
                    except Exception:
                        cur = []
                    try:
                        if survivor not in cur:
                            contains[tp] = cur + [survivor]
                        if loser in contains[tp]:
                            contains[tp] = [x for x in contains[tp] if x != loser]
                    except Exception:
                        pass
        except Exception:
            pass

    # 3) 数据/注释属性：仅当 survivor 缺失时，从 loser 补充
    for dp in all_dataprops():
        try:
            s_vals = list(dp[survivor]) or []
            l_vals = list(dp[loser]) or []
            if (not s_vals) and l_vals:
                dp[survivor] = l_vals
        except Exception:
            pass

    for ap in all_annprops():
        try:
            s_vals = list(ap[survivor]) or []
            l_vals = list(ap[loser]) or []
            if (not s_vals) and l_vals:
                ap[survivor] = l_vals
        except Exception:
            pass


# ========= 合并主流程 =========
def merge_places_by_modern_name():
    # 1) 按“现代名称”分桶
    name2places = {}
    for pl in list(Place.instances()):
        names = modern_names_of(pl)
        for nm in names:
            name2places.setdefault(nm, []).append(pl)

    # 2) 桶内合并
    total_merged = 0
    for nm, bucket in name2places.items():
        uniq = list({pl for pl in bucket})
        if len(uniq) <= 1:
            continue

        survivor, losers = pick_survivor(uniq)
        print(f"[MERGE] 现代名称='{nm}'  →  幸存者={getattr(survivor, 'name', '')} ({getattr(survivor, 'iri', '')})")
        if losers:
            print("        删除：")
            for l in losers:
                print(f"        - {getattr(l, 'name', '')} ({getattr(l, 'iri', '')})")

        for l in losers:
            migrate_place(l, survivor)
            try:
                ow.destroy_entity(l)
                total_merged += 1
            except Exception as e:
                print(f"    [WARN] 删除失败：{getattr(l,'name','')} ({getattr(l,'iri','')})  err={e}")

    print(f"[DONE] Place 合并完成：删除重复 {total_merged} 个。")


# ========= 执行 & 保存 =========
with onto:
    merge_places_by_modern_name()

g = world.as_rdflib_graph()
g.serialize(destination=str(OUT_NT), format="nt")
g.serialize(destination=str(OUT_TTL), format="turtle")
_tmpdir.cleanup()

print(f"[SAVED] NT  -> {OUT_NT}")
print(f"[SAVED] TTL -> {OUT_TTL}")
