# -*- coding: utf-8 -*-
"""
批量解析“学籍” → 拆为三组 PropAssertion（类型/层级/机构名），并导出追踪CSV
- 鲁棒加载：NT -> RDF/XML 临时转换 -> owlready2.World
- 调用：parse_academic_affiliation(text) -> {"raw","类型","层级","机构名"}
- 写回：
  · 对每条学籍原文，分别写 3 组断言（每组 1 条 PA）：
      prop="学籍.类型"   / record_id=<edurec_xxx:type>   / 独立 TextProvenance(学籍解析:类型)
      prop="学籍.层级"   / record_id=<edurec_xxx:level>  / 独立 TextProvenance(学籍解析:层级)
      prop="学籍.机构名" / record_id=<edurec_xxx:org>    / 独立 TextProvenance(学籍解析:机构名)
  · about → Person；derivedFrom → TextProvenance；TextProvenance.contains → Person
- 去重：同(about, prop, value_norm, value_raw) 不重复写
- 导出 CSV（均为字符字段，不含 ID）
"""

import os
import re
import csv
import uuid
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple

from rdflib import Graph
from owlready2 import World, ThingClass, destroy_entity, DataPropertyClass

# ========= 路径配置 =========
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


OUT_TTL  = str(Path(SRC).with_name("ontology_academic_parsed_split3.ttl"))
OUT_NT   = str(Path(SRC).with_name("ontology_academic_parsed_split3.nt"))
OUT_CSV  = str(Path(SRC).with_name("academic_parse_trace.csv"))

# ========= 学籍解析函数（按你提供的规则） =========
SI_LIST = ["宣慰司", "宣抚司", "宣德司"]

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", "", str(s or ""))

def detect_central(s: str) -> Optional[Tuple[str, str, str]]:
    if re.search(r"(国子(监)?生|太学生)", s):
        return ("中央官学", "", "")
    return None

def detect_ethnic(s: str) -> Optional[Tuple[str, str, str]]:
    for si in SI_LIST:
        if si in s:
            m = re.search(r"([^\s府州县卫司]+)?(" + re.escape(si) + r")(?:学?生)", s)
            name = ((m.group(1) or "") + si) if m else si
            return ("民族地区官学", "司", name)
    if "儒学生" in s:
        return ("民族地区官学", "", "")
    return None

def detect_military(s: str) -> Optional[Tuple[str, str, str]]:
    m = re.search(r"([^\s府州县司]+?)卫学生", s)
    if m:
        return ("军事系统官学", "卫", m.group(1))
    if re.search(r"(卫学生|卫学(军生|官生))", s):
        return ("军事系统官学", "卫", "")
    return None

def detect_local(s: str) -> Optional[Tuple[str, str, str]]:
    m = re.search(r"([^\s卫州县司]+?)府(学生|附生)", s)
    if m: return ("地方官学", "府", m.group(1))
    m = re.search(r"([^\s府卫县司]+?)州学生", s)
    if m: return ("地方官学", "州", m.group(1))
    m = re.search(r"([^\s府州卫司]+?)县(学生|附生)", s)
    if m: return ("地方官学", "县", m.group(1))
    if re.search(r"(增广生|附学生|学生)$", s):
        return ("地方官学", "", "")
    return None

def parse_academic_affiliation(text: str) -> dict:
    s = normalize_text(text)
    for detector in (detect_central, detect_ethnic, detect_military, detect_local):
        hit = detector(s)
        if hit:
            t, lv, nm = hit
            return {"raw": text, "类型": t, "层级": lv, "机构名": nm}
    return {"raw": text, "类型": "地方官学", "层级": "", "机构名": ""}

# ========= 鲁棒加载 =========
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

# ========= 命名空间容错 =========
def find_class(names) -> Optional[ThingClass]:
    target = set(names)
    for c in list(onto.classes()):
        try:
            if c.name in target: return c
            iri = getattr(c, "iri", "")
            if iri and any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in target): return c
        except: pass
    for c in list(world.classes()):
        try:
            if c.name in target: return c
            iri = getattr(c, "iri", "")
            if iri and any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in target): return c
        except: pass
    return None

def find_data_property(names) -> Optional[DataPropertyClass]:
    target = set(names)
    for dp in list(onto.data_properties()):
        try:
            if dp.name in target: return dp
            iri = getattr(dp, "iri", "")
            if iri and any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in target): return dp
        except: pass
    for dp in list(world.data_properties()):
        try:
            if dp.name in target: return dp
            iri = getattr(dp, "iri", "")
            if iri and any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in target): return dp
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

# ========= 关键类/属性 =========
Person        = find_class(["Person","人物"])
PropAssertion = find_class(["PropAssertion"])
TextProv      = find_class(["TextProvenance","Text_Provenance"])
if not Person or not PropAssertion or not TextProv:
    raise SystemExit("[FATAL] 缺少 Person/PropAssertion/TextProvenance 类")

DP_XUEJI = find_data_property(["学籍"])
if not DP_XUEJI:
    raise SystemExit("[FATAL] 未找到数据属性“学籍”")

XUEJI_DP_NAME = DP_XUEJI.name

# 若不存在 record_id 数据属性，则创建到 PropAssertion
try:
    _ = getattr(onto, "record_id")
except Exception:
    with onto:
        class record_id(onto.DataProperty):
            domain = [PropAssertion]
            range  = [str]

# ========= 工具 =========
def uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"

def has_same_pa(person, prop_key: str, norm_val: str, raw_val: str) -> bool:
    """ 同(about=person, prop, value_norm, value_raw) 视为已存在 """
    try:
        for pa in PropAssertion.instances():
            if person not in op_get_all(pa, "about"):
                continue
            prop = (getattr(pa, "prop", []) or [""])[0] if hasattr(pa,"prop") else ""
            if prop != prop_key:
                continue
            vnorm = (getattr(pa, "value_norm", []) or [""])[0]
            vraw  = (getattr(pa, "value", []) or [""])[0]
            if str(vnorm) == str(norm_val) and str(vraw) == str(raw_val):
                return True
    except Exception:
        pass
    return False

def new_textprov(person, raw_text: str, facet: str):
    """ 为三组分别创建独立 TextProvenance，Text_Source 标注学籍解析:facet """
    with onto:
        tp = TextProv(uid("TP"))
    try:
        dp_set_single(tp, "record_confidence", "auto")
        dp_set_single(tp, "Text_Source", f"学籍解析:{facet}")
        dp_set_single(tp, "Text_body", str(raw_text or ""))
    except Exception:
        pass
    op_add_unique(tp, "contains", person)
    return tp

# ========= 主流程 =========
people = list(Person.instances())
print(f"[INFO] Person 数：{len(people)}")

rows = []  # CSV 追踪

created_cnt = 0
skipped_cnt = 0
touched_person = 0

for idx, p in enumerate(people, 1):
    xueji_list = [str(v).strip() for v in dp_get_all(p, XUEJI_DP_NAME) if str(v).strip()]
    if not xueji_list:
        continue
    touched_person += 1
    pname = (dp_get_one(p, "姓名") or getattr(p, "name", "") or "").strip()

    for raw in xueji_list:
        parsed = parse_academic_affiliation(raw)
        t_val  = parsed.get("类型","")
        l_val  = parsed.get("层级","")
        n_val  = parsed.get("机构名","")

        # 三组（独立 record_id + 独立 TP）
        rec_base = uid("edurec")
        groups = [
            ("学籍.类型",   t_val, "type",  f"{rec_base}:type"),
            ("学籍.层级",   l_val, "level", f"{rec_base}:level"),
            ("学籍.机构名", n_val, "org",   f"{rec_base}:org"),
        ]

        created_flags = {"type": "否", "level": "否", "org": "否"}

        for prop_key, norm_val, facet, recid in groups:
            # 判重
            if has_same_pa(p, prop_key, str(norm_val), raw):
                skipped_cnt += 1
                continue

            tp = new_textprov(p, raw, facet)

            with onto:
                pa = PropAssertion(uid("PA"))
            try: setattr(pa, "prop",       [prop_key])
            except: pass
            try: setattr(pa, "value",      [raw])               # 原文
            except: pass
            try: setattr(pa, "value_norm", [str(norm_val)])     # 规范值
            except: pass
            try: getattr(onto, "record_id")[pa] = [recid]
            except Exception:
                try: setattr(pa, "record_id", [recid])
                except: pass

            op_add_unique(pa, "about", p)
            op_add_unique(pa, "derivedFrom", tp)
            op_add_unique(tp, "contains", p)

            created_cnt += 1
            created_flags[facet] = "是"

        # 追踪行（字符字段）
        rows.append({
            "person_name": pname or "",
            "xueji_raw": raw,
            "parsed_type": t_val,
            "parsed_level": l_val,
            "parsed_orgname": n_val,
            "created_type_pa": created_flags["type"],
            "created_level_pa": created_flags["level"],
            "created_org_pa": created_flags["org"],
        })

print(f"[DONE] 新建PA={created_cnt} 跳过(已存在)={skipped_cnt} 涉及人物={touched_person}")

# ========= 保存 =========
try:
    g.serialize(destination=OUT_TTL, format="turtle")
    g.serialize(destination=OUT_NT,  format="nt")
    print(f"[SAVED] {OUT_TTL}")
    print(f"[SAVED] {OUT_NT}")
except Exception as e:
    print("[WARN] 导出失败：", e)

# ========= 导出 CSV（全为字符段）=========
if rows:
    os.makedirs(str(Path(OUT_CSV).parent), exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "person_name","xueji_raw","parsed_type","parsed_level","parsed_orgname",
            "created_type_pa","created_level_pa","created_org_pa"
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] CSV 导出：{OUT_CSV}（{len(rows)} 行）")
else:
    print("[INFO] 无学籍数据，未生成CSV。")
