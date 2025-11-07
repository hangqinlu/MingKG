# -*- coding: utf-8 -*-
"""
force_merge_events_by_level.py

逐人逐【考试等级】将 ParticipationEvent 强制合并为 1 条：
- 不再以“甲第等级/名次是否冲突”决定是否合并——一律合并；
- 数据属性做并集；若双方都有值且不相等 -> 在幸存者上写 PropAssertion 标记冲突：
    * prop="属性冲突"
    * value=属性名
    * value_norm="值1｜值2｜...（去重后）"
- 迁移所有溯源：PropAssertion.about 改指向幸存者；其 derivedFrom 的 TextProvenance.contains 也补挂幸存者
- 对象属性并集：hasExam / hasPlace
- 稳健加载与去重打印
- 导出 NT/TTL
"""

from owlready2 import *
from pathlib import Path
from rdflib import Graph
import tempfile, hashlib, os
from typing import Any, List, Dict, Set, Tuple

import os, argparse, datetime

def _parse_args():
    ap = argparse.ArgumentParser(description="脚本7：考试事件合并")
    # 支持 --src（首选）与 --onto（兼容上游习惯），都映射到 dest="src"
    ap.add_argument(
        "--src", "--onto", dest="src", required=False,
        default=os.getenv(
            "ONTO_FILE",  # 若上游用 ONTO_FILE 传入则优先
            os.path.join(os.getenv("OUT_DIR", str(Path.cwd() / "本体结构")), "ontology_dedup_乡会殿处理.nt")
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

OUT_OWL = str(Path(OUT_DIR) / "ontology_dedup_updated.owl")
OUT_TTL = str(Path(OUT_DIR) / "ontology_dedup_updated.ttl")
OUT_NT  = str(Path(OUT_DIR) / "ontology_dedup_考试事件合并.nt")





# ===== 2) 稳健加载（rdflib -> RDF/XML -> owlready2）=====
def load_ontology_robust(src_path: str):
    p = Path(src_path)
    if not p.exists():
        raise FileNotFoundError(f"本体文件不存在：{src_path}")

    fmt_map = {".nt": "nt", ".ttl": "turtle", ".rdf": "xml", ".owl": "xml", ".xml": "xml"}
    in_fmt = fmt_map.get(p.suffix.lower(), None)

    g = Graph()
    if in_fmt:
        g.parse(str(p), format=in_fmt)
    else:
        g.parse(str(p))

    td = tempfile.TemporaryDirectory()
    owl_path = Path(td.name) / (p.stem + "_tmp.owl")
    g.serialize(destination=str(owl_path), format="xml", encoding="utf-8")

    world = World()
    onto  = world.get_ontology(str(owl_path)).load()
    return world, onto, td

world, onto, _tmpdir = load_ontology_robust(SRC)

# ===== 3) 绑定类/属性 =====
def _ends_with_any(iri: str, names: List[str]) -> bool:
    iri = iri or ""
    return any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in names)

def must_get_class(names):
    for c in list(onto.classes()):
        try:
            if c.name in names or _ends_with_any(getattr(c,"iri",""), names): return c
        except: pass
    for c in list(world.classes()):
        try:
            if c.name in names or _ends_with_any(getattr(c,"iri",""), names): return c
        except: pass
    raise RuntimeError(f"未找到类 {names}")

def must_get_objprop(names):
    for p in list(onto.object_properties()):
        try:
            if p.name in names or _ends_with_any(getattr(p,"iri",""), names): return p
        except: pass
    for p in list(world.object_properties()):
        try:
            if p.name in names or _ends_with_any(getattr(p,"iri",""), names): return p
        except: pass
    raise RuntimeError(f"未找到对象属性 {names}")

def must_get_dataprop(names):
    for p in list(onto.data_properties()):
        try:
            if p.name in names or _ends_with_any(getattr(p,"iri",""), names): return p
        except: pass
    for p in list(world.data_properties()):
        try:
            if p.name in names or _ends_with_any(getattr(p,"iri",""), names): return p
        except: pass
    raise RuntimeError(f"未找到数据属性 {names}")

Person             = must_get_class(["Person","人物"])
ParticipationEvent = must_get_class(["ParticipationEvent","参与事件"])
ImperialExam       = must_get_class(["ImperialExam","科举考试","考试"])
PropAssertion      = must_get_class(["PropAssertion"])
TextProvenance     = must_get_class(["TextProvenance"])
Place              = must_get_class(["Place"])

participatesIn = must_get_objprop(["participatesIn"])  # Person -> ParticipationEvent
hasExam        = must_get_objprop(["hasExam"])         # Event -> ImperialExam
hasPlace       = must_get_objprop(["hasPlace"])        # Event -> Place
about          = must_get_objprop(["about"])           # PropAssertion -> owl:Thing
derivedFrom    = must_get_objprop(["derivedFrom"])     # PropAssertion -> TextProvenance
contains       = must_get_objprop(["contains"])        # TextProvenance -> owl:Thing

dp_exam_level  = must_get_dataprop(["考试等级"])

# PropAssertion 的数据属性
dp_pa_prop       = must_get_dataprop(["prop"])
dp_pa_value      = must_get_dataprop(["value"])
dp_pa_value_norm = must_get_dataprop(["value_norm"])

print(f"[OK] 载入成功：类={len(list(onto.classes()))}，个体≈{len(list(onto.individuals()))}。")
print(f"[OK] Person 个体：{len(list(Person.instances()))}")

# ===== 4) 通用工具 =====
def list_vals(dp, inst) -> List[Any]:
    try:
        return list(dp[inst])
    except Exception:
        return []

def set_vals_union(dp, inst, union_vals: List[Any]):
    # 覆盖为去重后的并集（字符串化）
    uniq = []
    seen = set()
    for v in union_vals:
        sv = str(v)
        if sv not in seen:
            seen.add(sv); uniq.append(sv)
    try:
        dp[inst] = uniq
    except Exception:
        try:
            cur = []
            try: cur = list(dp[inst])
            except: pass
            cur = [x for x in cur if x is not None]
            cur += [x for x in uniq if x not in cur]
            dp[inst] = cur
        except Exception:
            pass  # 容错：个别属性可能是功能化或写入受限

def safe_first(vals):
    try:
        return vals[0] if vals else None
    except Exception:
        return None

def person_name(p) -> str:
    try:
        nm_prop = getattr(onto, "姓名", None)
        if nm_prop:
            v = list_vals(nm_prop, p)
            if v and str(v[0]).strip():
                return str(v[0]).strip()
    except: pass
    # 回退 PropAssertion(about=p, prop=姓名/name)
    try:
        cands = []
        for pa in PropAssertion.instances():
            if p in about[pa]:
                pr = safe_first(list_vals(dp_pa_prop, pa))
                if str(pr) in ("姓名","name"):
                    vv = safe_first(list_vals(dp_pa_value_norm, pa)) or safe_first(list_vals(dp_pa_value, pa))
                    if vv and str(vv).strip(): cands.append(str(vv).strip())
        if cands:
            cands.sort(key=len)
            return cands[0]
    except: pass
    return getattr(p, "name", "Unknown")

def _propassert_has_level(inst, level_text: str) -> bool:
    for pa in PropAssertion.instances():
        try:
            if inst in about[pa]:
                p  = safe_first(list_vals(dp_pa_prop, pa)) or ""
                v1 = safe_first(list_vals(dp_pa_value_norm, pa))
                v2 = safe_first(list_vals(dp_pa_value, pa)) if v1 is None else None
                val = str(v1 if v1 is not None else (v2 if v2 is not None else ""))
                if str(p) in ("考试等级","考试级别") and level_text in val:
                    return True
        except: pass
    return False

def exam_level_of_event(pe) -> str:
    # 优先 exam 节点
    exs = list(hasExam[pe])
    for ex in exs:
        levels = list_vals(dp_exam_level, ex)
        for lv in levels:
            s = str(lv).strip()
            if s: return s
        if _propassert_has_level(ex, "殿试"): return "殿试"
        if _propassert_has_level(ex, "会试"): return "会试"
        if _propassert_has_level(ex, "乡试"): return "乡试"
    # 事件自身
    levels_pe = list_vals(dp_exam_level, pe)
    for lv in levels_pe:
        s = str(lv).strip()
        if s: return s
    # 事件 PA
    if _propassert_has_level(pe, "殿试"): return "殿试"
    if _propassert_has_level(pe, "会试"): return "会试"
    if _propassert_has_level(pe, "乡试"): return "乡试"
    return ""  # 未知

def info_score(pe) -> int:
    score = 0
    # 核心指标：是否有数据/连边/溯源数量（ capped ）
    try:
        if list(hasExam[pe]): score += 1
    except: pass
    try:
        if list(hasPlace[pe]): score += 1
    except: pass
    # 溯源断言计数
    n = 0
    for pa in PropAssertion.instances():
        try:
            if pe in about[pa]:
                n += 1
        except: pass
    score += min(n, 5)
    return score

def deterministic_pick(a, b):
    seed = int(hashlib.blake2b((a.iri + "|" + b.iri).encode("utf-8"), digest_size=4).hexdigest(), 16)
    return a if (seed % 2 == 0) else b

def pick_stronger(a, b):
    sa, sb = info_score(a), info_score(b)
    if sa > sb: return a, b
    if sb > sa: return b, a
    return (a, b) if a.iri < b.iri else (b, a)

# ===== 5) 冲突标记 & 溯源迁移 =====
def create_conflict_assertion(survivor, dp_name: str, all_vals: List[str]):
    """在幸存者上写 1 条冲突标记：prop='属性冲突'，value=属性名，value_norm='v1｜v2｜...'（去重）"""
    # 先检测是否已存在同样的冲突记录，避免重复
    merged = "｜".join(sorted(set(str(v) for v in all_vals if str(v).strip())))
    for pa in PropAssertion.instances():
        try:
            if survivor in about[pa]:
                p = safe_first(list_vals(dp_pa_prop, pa))
                vn = safe_first(list_vals(dp_pa_value_norm, pa))
                v  = safe_first(list_vals(dp_pa_value, pa))
                if str(p) == "属性冲突" and str(v) == dp_name and str(vn) == merged:
                    return  # 已有同条
        except: pass
    pa = PropAssertion(f"PropAssertion_conflict_{hashlib.blake2b((survivor.iri+'|'+dp_name+'|'+merged).encode('utf-8'),digest_size=6).hexdigest()}")
    try: pa.prop = ["属性冲突"]
    except: setattr(pa, "prop", ["属性冲突"])
    try: pa.value = [dp_name]
    except: setattr(pa, "value", [dp_name])
    try: pa.value_norm = [merged]
    except: setattr(pa, "value_norm", [merged])
    try:
        about[pa].append(survivor)
    except Exception:
        try:
            pa.about = [survivor]
        except: pass

def migrate_propassertions_and_provenance(loser, survivor):
    """把关于 loser 的所有 PropAssertion 挂到 survivor；derivedFrom 的 TextProvenance 也补 contains=survivor"""
    for pa in list(PropAssertion.instances()):
        try:
            if loser in about[pa]:
                # 改 about
                cur = [x for x in about[pa] if x != loser]
                if survivor not in cur: cur.append(survivor)
                try: about[pa] = cur
                except: pass
                # 补 contains
                for tp in list(derivedFrom[pa]):
                    try:
                        if survivor not in contains[tp]:
                            contains[tp].append(survivor)
                    except: pass
        except: pass

# ===== 6) 强制合并：逐人 x 等级 =====
_print_once_seen: Set[str] = set()
def print_once(msg: str):
    if msg not in _print_once_seen:
        _print_once_seen.add(msg); print(msg)

def all_persons() -> List[Any]:
    try: return list(Person.instances())
    except: return []

def merge_bucket_force(person, level: str, events: List[Any]):
    pname = person_name(person)
    level_disp = level if level else "(未知等级)"
    print(f"\n[FORCE] 人物：{pname} | 等级：{level_disp} | 事件数={len(events)}")

    # 1) 选幸存者：信息度最高；完全一致则伪随机；
    events = sorted(events, key=lambda x: x.iri)
    survivor = events[0]
    for e in events[1:]:
        s, l = pick_stronger(survivor, e)
        survivor = s

    # 2) 合并其余事件到幸存者
    # 准备一个“候选数据属性集合”：动态发现——凡是在任意事件上出现过值的数据属性都纳入
    dataprops = []
    seen_dp_iri = set()
    for e in events:
        for dp in list(onto.data_properties()) + list(world.data_properties()):
            try:
                vals = list(dp[e])
                if vals:
                    iri = getattr(dp, "iri", dp.name)
                    if iri not in seen_dp_iri:
                        dataprops.append(dp); seen_dp_iri.add(iri)
            except: pass

    # 对象属性合并（事先记录，逐个 loser 并入）
    for e in events:
        if e is survivor: continue
        # 对象属性并集
        try:
            for ex in list(hasExam[e]):
                if ex not in list(hasExam[survivor]):
                    hasExam[survivor].append(ex)
        except: pass
        try:
            for pl in list(hasPlace[e]):
                if pl not in list(hasPlace[survivor]):
                    hasPlace[survivor].append(pl)
        except: pass

    # 数据属性并集 + 冲突检测
    for dp in dataprops:
        # 收集所有事件在该 dp 的值
        bucket_vals: List[str] = []
        per_event_vals: List[Set[str]] = []
        for e in events:
            try:
                vals = [str(v) for v in list(dp[e]) if v is not None and str(v).strip() != ""]
            except:
                vals = []
            per_event_vals.append(set(vals))
            for v in vals:
                if v not in bucket_vals:
                    bucket_vals.append(v)

        # 并到幸存者
        try:
            cur = [str(v) for v in list(dp[survivor])]
        except:
            cur = []
        union_vals = list(dict.fromkeys(cur + bucket_vals))
        try:
            dp[survivor] = union_vals
        except:
            set_vals_union(dp, survivor, union_vals)

        # 冲突：至少两个不同非空集合 & 不是完全一致
        # 例如：{三甲} vs {二甲} / {十一} vs {十二}
        non_empty_sets = [s for s in per_event_vals if s]
        if len(non_empty_sets) >= 2:
            # 如果这些非空集合的并集大小 > 1，说明存在不同值
            merged_all = set().union(*non_empty_sets)
            if len(merged_all) > 1:
                dp_name = getattr(dp, "name", getattr(dp, "iri", "未知数据属性"))
                create_conflict_assertion(survivor, dp_name, sorted(merged_all))

    # 3) 迁移溯源 & 删除败者
    for e in events:
        if e is survivor: continue
        migrate_propassertions_and_provenance(e, survivor)
        try:
            destroy_entity(e)
        except Exception:
            pass

    print_once(f"[OK] 合并完成：人物={pname} 等级={level_disp} 幸存者={getattr(survivor,'name','(pe)')}  数据/对象属性已并集，冲突已标记。")

def run_all():
    for p in all_persons():
        # 分桶：同一人物按“考试等级”聚合 ParticipationEvent
        evs = list(participatesIn[p])
        if len(evs) <= 1:
            continue
        buckets: Dict[str, List[Any]] = {}
        for pe in evs:
            lvl = exam_level_of_event(pe)  # 可能是 殿试/会试/乡试/""(未知)
            buckets.setdefault(lvl, []).append(pe)

        for lvl, lst in buckets.items():
            if len(lst) >= 2:
                merge_bucket_force(p, lvl, lst)

# ===== 7) 执行 & 导出 =====
with onto:
    run_all()

g = world.as_rdflib_graph()
g.serialize(destination=str(OUT_NT), format="nt")
g.serialize(destination=str(OUT_TTL), format="turtle")
print(f"\n[SAVED] NT  -> {OUT_NT}")
print(f"[SAVED] TTL -> {OUT_TTL}")
