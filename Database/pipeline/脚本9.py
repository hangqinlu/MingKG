# -*- coding: utf-8 -*-
"""
按人合并任命事件：同一官职 -> 合并为同一 AppointmentEvent
- 仅在同一 Person 的 appointedIn 链内进行（不会跨人误并）
- 幸存者：伪随机（可复现）挑选
- 继承：失败者的一切信息都迁移到幸存者
  * 对象属性：把所有指向 loser 的引用改为指向 survivor（全量对象属性，不限 hasPosition）
  * 数据/注释属性：并集（去重）到 survivor（即“继承全部信息”）
  * PropAssertion：about 从 loser 改为 survivor；重复断言去重；derivedFrom.contains 修补
- 删除：destroy_entity(loser)
- 输出：合并日志 + 保存 NT / TTL
"""

from owlready2 import *
from rdflib import Graph
from pathlib import Path
import tempfile
import hashlib
import csv
import os, argparse, datetime
def _parse_args():
    ap = argparse.ArgumentParser(description="脚本10：官职事件合并")
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
# ========= 路径 =========
OUT_NT  = Path(SRC).with_name("ontology_appt_merged.nt")
OUT_TTL = Path(SRC).with_name("ontology_appt_merged.ttl")

# ========= 稳健加载 =========
def load_ontology_robust(src_path: str):
    p = Path(src_path)
    fmt = {".nt":"nt", ".ttl":"turtle", ".rdf":"xml", ".owl":"xml", ".xml":"xml"}.get(p.suffix.lower(), None)
    g = Graph()
    g.parse(str(p), format=fmt)
    td = tempfile.TemporaryDirectory()
    owl_path = Path(td.name) / (p.stem + "_tmp.owl")
    g.serialize(destination=str(owl_path), format="xml", encoding="utf-8")
    world = World()
    onto  = world.get_ontology(str(owl_path)).load()
    return world, onto, td

world, onto, _tmpdir = load_ontology_robust(SRC)

# ========= 跨命名空间收集 =========
def all_classes():  return list(onto.classes()) + list(world.classes())
def all_objprops(): return list(onto.object_properties()) + list(world.object_properties())
def all_dataprops():return list(onto.data_properties()) + list(world.data_properties())
def all_annprops(): return list(onto.annotation_properties()) + list(world.annotation_properties())

def find_class(names):
    names = set(names)
    for c in all_classes():
        if c.name in names: return c
        try:
            iri = c.iri
            if any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in names): return c
        except: pass
    raise RuntimeError(f"未找到类：{names}")

def find_objprop_one(names):
    names = set(names)
    for p in all_objprops():
        if p.name in names: return p
        try:
            iri = p.iri
            if any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in names): return p
        except: pass
    raise RuntimeError(f"未找到对象属性：{names}")

def find_dataprop_one(names):
    names = set(names)
    for p in all_dataprops():
        if p.name in names: return p
        try:
            iri = p.iri
            if any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in names): return p
        except: pass
    return None

# ========= 绑定核心类/属性 =========
Person           = find_class(["Person"])
AppointmentEvent = find_class(["AppointmentEvent"])
OfficialPosition = find_class(["OfficialPosition"])
PropAssertion    = find_class(["PropAssertion"])
TextProvenance  = find_class(["TextProvenance"])

appointedIn   = find_objprop_one(["appointedIn"])   # Person -> AppointmentEvent
hasPosition   = find_objprop_one(["hasPosition"])   # AppointmentEvent -> OfficialPosition
about         = find_objprop_one(["about"])         # PropAssertion -> owl:Thing
derivedFrom   = find_objprop_one(["derivedFrom"])
contains      = find_objprop_one(["contains"])

# 常见数据属性（可缺）
dp_person_name = find_dataprop_one(["姓名"])
dp_pa_prop       = find_dataprop_one(["prop"])
dp_pa_value      = find_dataprop_one(["value"])
dp_pa_value_norm = find_dataprop_one(["value_norm"])

# ========= 小工具 =========
def list_vals(prop, inst):
    try: return list(prop[inst])
    except Exception: return []

def person_display_name(p):
    try:
        vals = list(dp_person_name[p]) if dp_person_name else []
        return (vals[0] if vals else getattr(p, "name", "Unknown")).strip()
    except Exception:
        return getattr(p, "name", "Unknown")

def deterministic_survivor(items):
    """可复现“随机”：对每个 IRI 取 blake2b，取最小哈希者为幸存者"""
    def h(x):
        return hashlib.blake2b(str(getattr(x, "iri", x)).encode("utf-8"), digest_size=8).hexdigest()
    return sorted(items, key=lambda x: (h(x), getattr(x, "iri", "")))[0]

# ========= 迁移逻辑 =========
def migrate_all_references_and_data(loser, survivor):
    if loser == survivor: return

    # 1) 对象属性：把所有指向 loser 的引用改为 survivor（全量对象属性 & 全部主体）
    for op in all_objprops():
        for subj in list(world.individuals()):
            try:
                cur = list(op[subj])
            except Exception:
                continue
            if not cur: continue
            if loser in cur:
                nlst = [survivor if x == loser else x for x in cur]
                # 去重
                seen, dedup = set(), []
                for x in nlst:
                    if x not in seen:
                        seen.add(x); dedup.append(x)
                try:
                    op[subj] = dedup
                except Exception:
                    pass

    # 2) 数据 / 注释属性：并集（保留全部信息）
    for dp in all_dataprops():
        try:
            s_vals = list(dp[survivor])
            l_vals = list(dp[loser])
            if l_vals:
                # 合并去重
                merged = list(dict.fromkeys([*s_vals, *l_vals]))
                dp[survivor] = merged
        except Exception:
            pass
    for ap in all_annprops():
        try:
            s_vals = list(ap[survivor])
            l_vals = list(ap[loser])
            if l_vals:
                merged = list(dict.fromkeys([*s_vals, *l_vals]))
                ap[survivor] = merged
        except Exception:
            pass

    # 3) PropAssertion：about 从 loser 改到 survivor；重复断言去重；修补 contains
    # 先收集 survivor 现有断言键（prop, value_norm/value 字符串）
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
                    # 与 survivor 重复，删掉这条
                    try: destroy_entity(pa)
                    except Exception: pass
                    continue
                # 改 about
                try:
                    about[pa].remove(loser)
                except Exception: pass
                if survivor not in about[pa]:
                    about[pa].append(survivor)
                # 修补 contains
                for tp in list(derivedFrom[pa]):
                    try:
                        if survivor not in contains[tp]:
                            contains[tp].append(survivor)
                        if loser in contains[tp]:
                            nlst = [x for x in list(contains[tp]) if x != loser]
                            contains[tp] = nlst
                    except Exception:
                        pass
        except Exception:
            pass

# ========= 合并主流程 =========
def merge_appointments_per_person():
    merged_cnt = 0
    persons = list(Person.instances())
    print(f"[INFO] 人物数：{len(persons)}")

    for p in persons:
        pname = person_display_name(p)
        appts = list(appointedIn[p])
        if len(appts) <= 1:
            continue

        # 建桶：按“官职集合键”分组（事件可挂多个官职；用集合键避免错并/漏并）
        key_to_events = {}
        for ae in appts:
            pos = list(hasPosition[ae])
            if not pos:
                # 没挂官职的不合并（也可另起一桶）
                continue
            key = tuple(sorted(getattr(po, "iri", "") for po in pos))
            key_to_events.setdefault(key, []).append(ae)

        for key, bucket in key_to_events.items():
            if len(bucket) <= 1:
                continue
            survivor = deterministic_survivor(bucket)
            losers = [x for x in bucket if x is not survivor]

            print(f"[MERGE] 人物：{pname} | 官职组={len(key)}个职位 | 幸存者={getattr(survivor,'name','')} | 失败者={len(losers)}")
            # 迁移 + 删除
            for l in losers:
                migrate_all_references_and_data(l, survivor)
                # 确保 Person.appointedIn 不丢：指向 survivor（并集去重），去掉 l
                cur = list(appointedIn[p])
                if l in cur:
                    cur = [x for x in cur if x != l]
                    if survivor not in cur:
                        cur.append(survivor)
                    try:
                        appointedIn[p] = cur
                    except Exception:
                        pass
                try:
                    destroy_entity(l)
                    merged_cnt += 1
                except Exception as e:
                    print(f"  [WARN] 删除失败：{getattr(l,'name','')} ({l.iri}) err={e}")

    print(f"[DONE] 合并完成：删除任命事件 {merged_cnt} 个。")

# ========= 执行 & 保存 =========
with onto:
    merge_appointments_per_person()

g = world.as_rdflib_graph()
g.serialize(destination=str(OUT_NT), format="nt")
g.serialize(destination=str(OUT_TTL), format="turtle")
print(f"[SAVED] NT  -> {OUT_NT}")
print(f"[SAVED] TTL -> {OUT_TTL}")
