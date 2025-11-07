# -*- coding: utf-8 -*-
"""
人物清洗批处理（殿试判定版：仅“殿试”才视为科举人物；补姓避免“张张伟”）
规则更新：
1) 删除姓名为两字且包含“氏”的 Person（如“张氏”、“李氏”）
2) “科举人物”严格判定：存在 participatesIn → ParticipationEvent → hasExam → ImperialExam，
   且该 ImperialExam 的“考试等级”包含“殿试”，才视为科举人物；否则视为非科举人物，需要补充姓。
   在 SocialRelationEvent 中，科举人物与非科举人物有关联时：
   - 若该“名”尚未被继承过：在原实例前置姓（如 张 + 熊 -> 张熊）
   - 若该“名”已被继承过且出现新的姓：克隆一个 Person，新实例姓名= 姓 + 原始名，仅把【当前事件】挂到新实例，
     并且从原实例上【移除】该事件（克隆不继承原实例的其它关系）
   - 同一“名”的同一姓若已克隆过：仅把【当前事件】追加到该克隆实例，并且从原实例上【移除】该事件
   - 姓名构造时，若“姓”与“名”的第一个字相同，避免出现“张张伟”这类重复：直接用“原始名”（视作已具该姓）
3) 同名合并：姓名相同的 Person 合并，弱信息并入强信息；完整迁移对象属性与溯源
输出：ontology_people_cleaned.ttl / .nt
"""

import uuid
from pathlib import Path
from typing import Any, List, Dict, Optional

from rdflib import Graph
from owlready2 import World, ThingClass, destroy_entity

# ========= 路径配置（按需修改） =========
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

OUT_TTL = str(Path(SRC).with_name("ontology_people_cleaned.ttl"))
OUT_NT  = str(Path(SRC).with_name("ontology_people_cleaned.nt"))

# ========= 稳健加载 =========
def robust_load(src_path: str):
    p = Path(src_path)
    fmt = {".nt":"nt",".ttl":"turtle",".rdf":"xml",".owl":"xml",".xml":"xml"}.get(p.suffix.lower())
    if fmt is None:
        raise RuntimeError(f"不支持的本体后缀：{p.suffix}")
    g0 = Graph(); g0.parse(str(p), format=fmt)
    tmp_owl = Path(p.parent) / (p.stem + "_tmp_for_owlready2.owl")
    g0.serialize(destination=str(tmp_owl), format="xml", encoding="utf-8")
    world = World()
    onto  = world.get_ontology(str(tmp_owl)).load()
    g = world.as_rdflib_graph()
    return world, onto, g

world, onto, g = robust_load(SRC)
print(f"[OK] 加载：{SRC}  三元组≈{len(g)}")

# ========= 小工具 =========
def find_class(names) -> Optional[ThingClass]:
    targets = set(names)
    for c in list(onto.classes()):
        try:
            if c.name in targets: return c
            iri = getattr(c, "iri", "")
            if iri and any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in targets):
                return c
        except: pass
    for c in list(world.classes()):
        try:
            if c.name in targets: return c
            iri = getattr(c, "iri", "")
            if iri and any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in targets):
                return c
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
    vs = dp_get_all(inst, dp_name); return vs[0] if vs else None

def dp_set_single(inst, dp_name: str, value):
    if isinstance(value, (list, tuple, set)): value = next(iter(value), None)
    with onto:
        try:
            setattr(inst, dp_name, value); return
        except: pass
        try:
            prop = getattr(onto, dp_name); prop[inst] = [value]; return
        except: pass
        try:
            setattr(inst, dp_name, None); setattr(inst, dp_name, value)
        except: pass

def op_get_all(inst, op_name: str) -> List[Any]:
    try:
        v = getattr(inst, op_name, [])
        return list(v) if isinstance(v, list) else ([v] if v else [])
    except:
        try:
            prop = getattr(onto, op_name); return list(prop[inst])
        except:
            return []

def op_add_unique(subj, op_name: str, obj):
    cur = op_get_all(subj, op_name)
    if obj in cur: return
    cur.append(obj)
    with onto:
        try:
            setattr(subj, op_name, cur); return
        except: pass
        try:
            prop = getattr(onto, op_name); prop[subj] = cur
        except: pass

def op_remove_one(subj, op_name: str, obj):
    cur = op_get_all(subj, op_name)
    if not cur: return
    new_cur = [x for x in cur if x != obj]
    with onto:
        try:
            setattr(subj, op_name, new_cur); return
        except: pass
        try:
            prop = getattr(onto, op_name); prop[subj] = new_cur
        except: pass

def person_name(p) -> str:
    return (dp_get_one(p, "姓名") or getattr(p, "name", "") or "").strip()

def first_char_chinese(s: str) -> Optional[str]:
    if not s: return None
    c = s[0]
    return c if '\u4e00' <= c <= '\u9fff' else None

def build_name_with_surname(surname: str, base_orig: str) -> str:
    """构造 ‘姓+名’，若名首字与姓相同，避免‘张张伟’——直接返回原始名"""
    if not surname or not base_orig: return (surname or "") + (base_orig or "")
    return base_orig if base_orig.startswith(surname) else (surname + base_orig)

def uid(prefix="Person"):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def new_person_with_name(PersonCls, name_text: str):
    with onto:
        inst = PersonCls(uid())
    dp_set_single(inst, "姓名", name_text)
    return inst

# ====== “殿试”相关判定 ======
def is_dianshi_exam(exam) -> bool:
    """ImperialExam 且 考试等级 含 ‘殿试’"""
    try:
        # 类名判断尽量宽松：有此类即用 isinstance；否则只要有“考试等级”且含“殿试”
        if hasattr(onto, "ImperialExam"):
            try:
                if not isinstance(exam, onto.ImperialExam):
                    return False
            except:
                pass
        lvl = dp_get_one(exam, "考试等级")
        return isinstance(lvl, str) and ("殿试" in lvl)
    except:
        return False

def is_kju_person_strict(p) -> bool:
    """严格科举人物：participatesIn → hasExam → ImperialExam 且考试等级含‘殿试’"""
    for evt in op_get_all(p, "participatesIn"):
        for ex in op_get_all(evt, "hasExam"):
            if is_dianshi_exam(ex):
                return True
    return False

# ========= 入口处理 =========
Person = find_class(["Person","人物"])
if not Person:
    raise SystemExit("[FATAL] 未找到 Person/人物 类。")

# 1) 删除“二字且含氏”
people = list(Person.instances())
to_delete = []
for p in people:
    nm = person_name(p)
    if nm and (len(nm) == 2) and ("氏" in nm):
        to_delete.append(p)

if to_delete:
    print(f"[STEP1] 删除二字且含“氏”的人物：{len(to_delete)}")
    for p in to_delete:
        print("  -", person_name(p))
        with onto:
            try:
                destroy_entity(p)
            except Exception as e:
                print(f"[WARN] 删除失败：{getattr(p,'name',p)} | {e}")
else:
    print("[STEP1] 无需删除：未发现“二字且含氏”的人物")

# 重新抓取
people = list(Person.instances())

# 2) 殿试-严格判定的科举 / 非科举；继承姓（首次前置；新姓克隆；同姓复用克隆，克隆只连目标事件）
is_kju = {p: is_kju_person_strict(p) for p in people}

# 聚合：事件 -> 人物（仅 SocialRelationEvent）
event2persons: Dict[Any, List[Any]] = {}
for p in people:
    for evt in op_get_all(p, "socialRelationEvent"):
        event2persons.setdefault(evt, []).append(p)

# 原始名缓存（用于构造“姓+名”，始终以未改名前的最初文本为基）
original_name: Dict[Any, str] = {p: person_name(p) for p in people}
# 每个“名”（基准实例）已继承过的姓集合
inherited_surnames: Dict[Any, set] = {p: set() for p in people}
# 避免同姓重复克隆：基准 -> {姓: 克隆实例}
clone_by_surname: Dict[Any, Dict[str, Any]] = {p: {} for p in people}

changed_cnt = 0

for evt, plist in event2persons.items():
    if len(plist) < 2:
        continue

    kju_list  = [p for p in plist if is_kju.get(p, False)]
    non_list  = [p for p in plist if not is_kju.get(p, False)]
    if not kju_list or not non_list:
        continue

    # 从每位科举人物（殿试）向非科举人物“补姓”
    for kp in kju_list:
        kname = person_name(kp)
        surname = first_char_chinese(kname)
        if not surname:
            continue

        for np in non_list:
            nm = person_name(np)
            if not nm:
                continue

            base = np  # 作为“名”的基准实例
            base_orig = original_name.get(base, nm) or nm

            # 姓与名首字相同 → 视为已具该姓：不做前置重复
            already_has_same_lead = base_orig.startswith(surname)

            # 已经继承过该姓
            if surname in inherited_surnames.setdefault(base, set()):
                # 若已有该姓的克隆：把【当前事件】追加到克隆，并从基准移除该事件
                if surname in clone_by_surname.setdefault(base, {}):
                    tgt = clone_by_surname[base][surname]
                    op_add_unique(tgt, "socialRelationEvent", evt)
                    op_remove_one(base, "socialRelationEvent", evt)
                else:
                    # 之前可能首次前置在基准实例；遇到同姓新事件 → 克隆承接该事件
                    new_name = build_name_with_surname(surname, base_orig)
                    newp = new_person_with_name(Person, new_name)
                    op_add_unique(newp, "socialRelationEvent", evt)
                    op_remove_one(base, "socialRelationEvent", evt)
                    clone_by_surname[base][surname] = newp
                changed_cnt += 1
                continue

            used_before = bool(inherited_surnames.get(base))
            if not used_before:
                # 第一次继承
                if already_has_same_lead:
                    # 基准名首字与姓相同：不做“姓+名”的重复前置，仅登记该姓已继承
                    inherited_surnames[base].add(surname)
                else:
                    new_name = build_name_with_surname(surname, base_orig)
                    if person_name(base) != new_name:
                        dp_set_single(base, "姓名", new_name)
                    inherited_surnames[base].add(surname)
                changed_cnt += 1
            else:
                # 已经继承过其他姓 → 创建克隆仅连当前事件，并从基准移除此事件
                new_name = build_name_with_surname(surname, base_orig)
                newp = new_person_with_name(Person, new_name)
                op_add_unique(newp, "socialRelationEvent", evt)
                op_remove_one(base, "socialRelationEvent", evt)
                clone_by_surname.setdefault(base, {})[surname] = newp
                inherited_surnames[base].add(surname)
                changed_cnt += 1

print(f"[STEP2] 姓氏继承/克隆处理完成：改动≈{changed_cnt}")

# 3) 同名合并：信息量强者为幸存者
def info_score_person(p) -> int:
    sc = 0
    try:
        for dp in onto.data_properties():
            try: sc += len(list(dp[p]))
            except: pass
    except: pass
    try:
        for op in onto.object_properties():
            try: sc += len(list(op[p]))
            except: pass
    except: pass
    try:
        for pa in onto.PropAssertion.instances():
            if p in op_get_all(pa, "about"): sc += 1
    except: pass
    return sc

def migrate_person_edges(src, dst):
    if src == dst: return
    with onto:
        for opn in ["participatesIn","appointedIn","heldOfficeAtEvent","performedTaskIn","socialRelationEvent"]:
            lst = op_get_all(src, opn)
            if lst:
                try: setattr(src, opn, [])
                except: pass
                cur = op_get_all(dst, opn)
                for obj in lst:
                    if obj not in cur: cur.append(obj)
                try:
                    setattr(dst, opn, cur)
                except:
                    try:
                        prop = getattr(onto, opn); prop[dst] = cur
                    except: pass
        try:
            for pa in list(onto.PropAssertion.instances()):
                abouts = op_get_all(pa, "about")
                if src in abouts:
                    new_about = [dst if x == src else x for x in abouts]
                    try: setattr(pa, "about", list(dict.fromkeys(new_about)))
                    except:
                        try: getattr(onto, "about")[pa] = list(dict.fromkeys(new_about))
                        except: pass
        except: pass
        try:
            for tp in list(onto.TextProvenance.instances()):
                cont = op_get_all(tp, "contains")
                if src in cont:
                    new_cont = [dst if x == src else x for x in cont]
                    try: setattr(tp, "contains", list(dict.fromkeys(new_cont)))
                    except:
                        try: getattr(onto, "contains")[tp] = list(dict.fromkeys(new_cont))
                        except: pass
        except: pass

def safe_destroy(inst):
    with onto:
        try:
            destroy_entity(inst)
        except Exception as e:
            print(f"[WARN] 删除失败：{getattr(inst,'name',inst)}  | {e}")

people = list(Person.instances())
name_buckets: Dict[str, List[Any]] = {}
for p in people:
    nm = person_name(p)
    if nm: name_buckets.setdefault(nm, []).append(p)

merged_cnt = 0
for nm, bucket in name_buckets.items():
    if len(bucket) <= 1: continue
    bucket_sorted = sorted(bucket, key=lambda x: info_score_person(x), reverse=True)
    survivor, victims = bucket_sorted[0], bucket_sorted[1:]
    print(f"[STEP3] 合并同名『{nm}』：{len(bucket)} → 1  | 幸存者={getattr(survivor,'name','')}")
    for v in victims:
        migrate_person_edges(v, survivor)
        safe_destroy(v)
        merged_cnt += 1

print(f"[STEP3] 合并完成：合并 {merged_cnt} 个重复人物实例")

# ========= 保存 =========
try:
    g.serialize(destination=OUT_TTL, format="turtle")
    g.serialize(destination=OUT_NT,  format="nt")
    print(f"[SAVED] {OUT_TTL}")
    print(f"[SAVED] {OUT_NT}")
except Exception as e:
    print("[WARN] 导出失败：", e)

print("[DONE]")
