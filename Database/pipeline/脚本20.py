# -*- coding: utf-8 -*-
"""
两个人物之间“属性相同”的社会关系事件合并（随机幸存者）
+ 预处理：删除“无对方人物”的社会关系事件（参与人数 < 2）
- 针对每一对人物 (A, B)，找出同时连接 A 且 B 的社会关系事件
- 若其中存在“属性签名”完全一致的多条事件：随机选择一个事件作为幸存者，其余合并并删除
- 属性签名包含：事件类名 + 关系类型 + 事件上的所有数据属性（值经字符串化、去重、排序）

写回：
- 人物的 socialRelationEvent：把对受害者事件的链接改为幸存者
- PropAssertion.about：把受害者事件改指向幸存者（并保持去重）
- TextProvenance.contains：若包含受害者，则改为包含幸存者（并去重）
- 删除受害者事件实例

输出：
- ontology_people_relations_merged.ttl / .nt
"""

import random
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
from collections import defaultdict

from rdflib import Graph
from owlready2 import World, ThingClass, destroy_entity

import argparse
import os


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



OUT_TTL = str(Path(SRC).with_name("ontology_people_relations_merged.ttl"))
OUT_NT  = str(Path(SRC).with_name("ontology_people_relations_merged.nt"))

random.seed(20250913)

# ========= 稳健加载（用 fileobj + 干净 IRI，避免 Windows file:// 问题）=========
def robust_load(src_path: str):
    """
    1) 用 rdflib 读取源本体
    2) 序列化为 Turtle（优先）或 NT 中转
    3) 用一个“干净”的 HTTP IRI 构建 onto，并通过 load(fileobj=...) 载入中转文件
       —— 不传文件路径/URI，彻底规避 Windows 下 file:// 的解析问题
    """
    p = Path(src_path)
    fmt = {".nt": "nt", ".ttl": "turtle", ".rdf": "xml", ".owl": "xml", ".xml": "xml"}.get(p.suffix.lower())
    if fmt is None:
        raise RuntimeError(f"不支持的本体后缀：{p.suffix}")

    # 1) rdflib 读源
    g0 = Graph()
    g0.parse(str(p), format=fmt)

    world = World()
    base_iri = "http://mingkg.org/temp#"  # 干净 IRI，避免把本地路径当成 IRI

    # 2) 优先 Turtle 中转
    tmp_ttl = Path(p.parent) / (p.stem + "_tmp_for_owlready2.ttl")
    try:
        g0.serialize(destination=str(tmp_ttl), format="turtle", encoding="utf-8")
        with tmp_ttl.open("rb") as fh:
            onto = world.get_ontology(base_iri).load(fileobj=fh)
    except Exception:
        # 3) 退回 NT 中转
        tmp_nt = Path(p.parent) / (p.stem + "_tmp_for_owlready2.nt")
        g0.serialize(destination=str(tmp_nt), format="nt", encoding="utf-8")
        with tmp_nt.open("rb") as fh:
            onto = world.get_ontology(base_iri).load(fileobj=fh)

    # 明确设置 base_iri，避免后续新建实体继承错误 IRI
    try:
        onto.base_iri = base_iri
    except:
        pass

    # 4) 取 rdflib 视图
    g = world.as_rdflib_graph()
    return world, onto, g

world, onto, g = robust_load(SRC)
print(f"[OK] 加载：{SRC}  三元组≈{len(g)}")

# ========= 工具 =========
def find_class(names) -> Optional[ThingClass]:
    targets = set(names)
    try:
        for c in list(onto.classes()):
            try:
                if c.name in targets: return c
                iri = getattr(c, "iri", "")
                if iri and any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in targets):
                    return c
            except: pass
    except: pass
    try:
        for c in list(world.classes()):
            try:
                if c.name in targets: return c
                iri = getattr(c, "iri", "")
                if iri and any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in targets):
                    return c
            except: pass
    except: pass
    return None

def dp_get_all(inst, dp_name: str) -> List[Any]:
    try:
        v = getattr(inst, dp_name)
        if isinstance(v, list): return list(v)
        return [v] if v is not None else []
    except:
        try:
            prop = getattr(onto, dp_name); return list(prop[inst])
        except:
            return []

def dp_get_one(inst, dp_name: str):
    vs = dp_get_all(inst, dp_name)
    return vs[0] if vs else None

def op_get_all(inst, op_name: str) -> List[Any]:
    try:
        v = getattr(inst, op_name, [])
        return list(v) if isinstance(v, list) else ([v] if v else [])
    except:
        try:
            prop = getattr(onto, op_name); return list(prop[inst])
        except:
            return []

def person_name(p) -> str:
    return (dp_get_one(p, "姓名") or getattr(p, "name", "") or "").strip() or getattr(p, "iri", "")

def event_class_name(e) -> str:
    try:
        return e.__class__.__name__
    except Exception:
        return "SocialRelationEvent"

RELTYPE_KEYS_TEXT = {"关系类型","關係類型","关系","RelationType"}

def prop_assertions_about(inst) -> List[Any]:
    out = []
    try:
        for pa in onto.PropAssertion.instances():
            if inst in op_get_all(pa, "about"):
                out.append(pa)
    except Exception:
        pass
    return out

def extract_relation_type(evt) -> str:
    # 1) 若是 SubordinationEvent，优先数据属性 关系类型
    try:
        Subor = find_class(["SubordinationEvent"])
        if Subor and isinstance(evt, Subor):
            val = dp_get_one(evt, "关系类型")
            if val: return str(val).strip()
    except Exception:
        pass
    # 2) PropAssertion(prop=关系类型)
    for pa in prop_assertions_about(evt):
        p = (dp_get_one(pa, "prop") or "").strip()
        if p in RELTYPE_KEYS_TEXT:
            v = dp_get_one(pa, "value_norm") or dp_get_one(pa, "value")
            if v: return str(v).strip()
    return ""

def person_key(p) -> str:
    return getattr(p, "iri", None) or getattr(p, "name", "") or str(id(p))

# ========= 事件属性签名 =========
def event_signature(evt) -> Tuple:
    """
    签名 = (类名, 关系类型, [(dp_key, [vals...]), ...])
    """
    cls = event_class_name(evt)
    rel = extract_relation_type(evt)

    dp_items: List[Tuple[str, Tuple[str, ...]]] = []
    try:
        for dp in onto.data_properties():
            try:
                vals = list(dp[evt])
            except Exception:
                vals = []
            if not vals: continue
            key = getattr(dp, "name", None)
            if not key:
                iri = str(getattr(dp, "iri", ""))
                key = iri.rsplit("#", 1)[-1] if "#" in iri else iri.rsplit("/", 1)[-1]
            svals = sorted({str(v) for v in vals})
            dp_items.append((key, tuple(svals)))
    except Exception:
        pass

    dp_items.sort(key=lambda x: x[0])
    return (cls, rel, tuple(dp_items))

# ========= 实体准备 =========
Person = find_class(["Person","人物"])
SREventClass = find_class(["SocialRelationEvent","社会关系事件"])
if not Person:
    raise SystemExit("[FATAL] 未找到 Person/人物 类。")
if not SREventClass:
    print("[WARN] 未找到 SocialRelationEvent 类；‘无对方人物’预清理将仅依据人物->事件连接。")

all_people = list(Person.instances())
print("[INFO] 人物数：", len(all_people))

# ========= 预处理：删除“无对方人物”的社会关系事件（参与人数 < 2） =========
def purge_missing_counterpart_sre():
    # 1) 构建 事件 -> 人物（仅用 socialRelationEvent 边）
    event_to_people: Dict[Any, List[Any]] = defaultdict(list)
    for p in all_people:
        for evt in op_get_all(p, "socialRelationEvent"):
            event_to_people[evt].append(p)

    # 2) 加入“0 人参与”的事件（若能识别 SRE 类）
    if SREventClass:
        for evt in list(SREventClass.instances()):
            event_to_people.setdefault(evt, [])  # 确保孤立事件也在键里

    # 3) 判定并删除（<2 人）
    deleted = 0
    for evt, plist in sorted(event_to_people.items(), key=lambda kv: str(getattr(kv[0], "iri", kv[0]))):
        # 去重
        uniq, seen = [], set()
        for x in plist:
            k = person_key(x)
            if k in seen: continue
            seen.add(k); uniq.append(x)

        if len(uniq) >= 2:
            continue

        names = [person_name(p) for p in uniq]
        if len(uniq) == 1:
            counterpart_note = "（无对方人物）"
        else:
            counterpart_note = "（无参与人物）"

        print(f"[DELETE-SRE] 事件={getattr(evt,'name',getattr(evt,'iri',evt))} | 参与人物={names} | 对方={counterpart_note}")

        try:
            destroy_entity(evt)  # 精准删除：仅删该事件实例及其直接三元组
            deleted += 1
        except Exception as e:
            print(f"[WARN] 删除社会关系事件失败：{getattr(evt,'name',evt)} | {e}")

    print(f"[STATS] 预处理：删除无对方人物的社会关系事件 = {deleted}")
    return deleted

_ = purge_missing_counterpart_sre()

# ========= 预处理后：重新获取人物（以防个别链接更新） =========
all_people = list(Person.instances())

# ========= 构建：事件 → 人物 =========
event_to_people: Dict[Any, List[Any]] = defaultdict(list)
for p in all_people:
    for evt in op_get_all(p, "socialRelationEvent"):
        event_to_people[evt].append(p)

# ========= 构建：人物对 → 事件列表 =========
pair2events: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
for evt, plist in event_to_people.items():
    uniq, seen = [], set()
    for x in plist:
        k = person_key(x)
        if k in seen: continue
        seen.add(k); uniq.append(x)
    n = len(uniq)
    if n < 2: continue
    for i in range(n):
        for j in range(i+1, n):
            a, b = uniq[i], uniq[j]
            ka, kb = person_key(a), person_key(b)
            key = (ka, kb) if ka < kb else (kb, ka)
            pair2events[key].append(evt)

# ========= 合并（按人物对内签名分组） =========
total_pairs_scanned = 0
total_groups_merged = 0
total_events_deleted = 0

for key, evts in pair2events.items():
    if len(evts) < 2: continue
    total_pairs_scanned += 1

    bucket: Dict[Tuple, List[Any]] = defaultdict(list)
    for e in evts:
        try:
            sig = event_signature(e)
        except Exception:
            sig = (event_class_name(e), "", ())
        bucket[sig].append(e)

    for sig, group in bucket.items():
        if len(group) <= 1: continue

        survivor = random.choice(group)
        victims  = [e for e in group if e is not survivor]

        with onto:
            # 1) 人物->事件：统一指向 survivor
            for p in all_people:
                try:
                    lst = list(getattr(p, "socialRelationEvent", []))
                except:
                    lst = []
                if not lst: continue
                changed = False
                new_lst = []
                for e in lst:
                    if e in victims:
                        if survivor not in new_lst:
                            new_lst.append(survivor)
                        changed = True
                    else:
                        new_lst.append(e)
                if changed:
                    try:
                        setattr(p, "socialRelationEvent", list(dict.fromkeys(new_lst)))
                    except:
                        try:
                            getattr(onto, "socialRelationEvent")[p] = list(dict.fromkeys(new_lst))
                        except:
                            pass

            # 2) 事件出边迁移（如 hasPlace 等）
            for v in victims:
                for op in onto.object_properties():
                    try:
                        targets = list(op[v])
                    except Exception:
                        targets = []
                    if not targets: continue
                    try:
                        cur = list(op[survivor])
                    except Exception:
                        cur = []
                    merged = list(dict.fromkeys(list(cur) + targets))
                    try:
                        op[survivor] = merged
                    except Exception:
                        pass
                    try:
                        op[v] = []
                    except Exception:
                        pass

            # 3) PropAssertion.about 改指向
            try:
                for pa in list(onto.PropAssertion.instances()):
                    abouts = []
                    try:
                        abouts = list(getattr(pa, "about", []))
                    except:
                        pass
                    if not abouts: continue
                    hit = False
                    new_about = []
                    for x in abouts:
                        if x in victims:
                            hit = True
                            if survivor not in new_about:
                                new_about.append(survivor)
                        else:
                            new_about.append(x)
                    if hit:
                        try:
                            setattr(pa, "about", list(dict.fromkeys(new_about)))
                        except:
                            try:
                                getattr(onto, "about")[pa] = list(dict.fromkeys(new_about))
                            except:
                                pass
            except Exception:
                pass

            # 4) TextProvenance.contains 改指向
            try:
                if hasattr(onto, "TextProvenance"):
                    for tp in list(onto.TextProvenance.instances()):
                        cont = []
                        try:
                            cont = list(getattr(tp, "contains", []))
                        except:
                            pass
                        if not cont: continue
                        changed = False
                        new_cont = []
                        for x in cont:
                            if x in victims:
                                changed = True
                                if survivor not in new_cont:
                                    new_cont.append(survivor)
                            else:
                                new_cont.append(x)
                        if changed:
                            try:
                                setattr(tp, "contains", list(dict.fromkeys(new_cont)))
                            except:
                                try:
                                    getattr(onto, "contains")[tp] = list(dict.fromkeys(new_cont))
                                except:
                                    pass
            except Exception:
                pass

            # 5) 删除受害者事件
            for v in victims:
                try:
                    destroy_entity(v)
                    total_events_deleted += 1
                except Exception as e:
                    print(f"[WARN] 删除事件失败：{getattr(v,'name',v)} | {e}")

        total_groups_merged += 1
        print(f"[MERGED] 人物对 {key} | 签名合并 {len(group)}→1 | 幸存者={getattr(survivor,'name','')}")

print(f"[STATS] 扫描人物对：{total_pairs_scanned}  | 合并组数：{total_groups_merged}  | 删除事件：{total_events_deleted}")

# ========= 保存（rdflib 直接序列化 Turtle/NT）=========
try:
    g.serialize(destination=OUT_TTL, format="turtle")
    g.serialize(destination=OUT_NT,  format="nt")
    print(f"[SAVED] {OUT_TTL}")
    print(f"[SAVED] {OUT_NT}")
except Exception as e:
    print("[WARN] 导出失败：", e)

print("[DONE]")
