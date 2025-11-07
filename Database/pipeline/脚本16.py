# -*- coding: utf-8 -*-
"""
Person.户籍地 —— 双通道读取 → 原值去重+被包含删除 → 先规范后合并 → 清空重写 → Axiom注释 → 详细日志（适配最新解析脚本）
- 仅操作 数据属性名==「户籍地」
- 读取：rdflib 与 owlready2 双通道合并；在“规范化前”做同值去重 + 『被包含』删除（保留更长项）
- 规范：调用你“最新版本”的户籍地解析脚本 place_parser.normalize_all_systems(raw_text, {}, civil_idx, mil_idx, tusi_idx)
- 合并：规范结果同值去重 + 被包含合并（保留信息量更长者）
- 写回：清空旧值（含 Axiom）→ 只写回新值（全部到「户籍地」）
- Axiom 注释：是否规范 / 原始户籍地 / 存在歧义 / 需进一步审核
- 详细日志打印到端口
"""

import re
import importlib
import importlib.util
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional, Set

from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, OWL, XSD
from owlready2 import World, ThingClass

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
OUT_TTL = str(Path(SRC).with_name("ontology_huji_dual_prenorm_containment_latestparser.ttl"))
OUT_NT  = str(Path(SRC).with_name("ontology_huji_dual_prenorm_containment_latestparser.nt"))

# 如无法直接 import place_parser，请填你的解析脚本绝对路径
PARSER_FILE = r"C:\Users\卢航青\PycharmProjects\pythonProject11\图属性数据——OWL数据\外部知识解析类\户籍地解析规则.py"

# ========= 解析器导入 =========
parser = None
def _try_import_parser():
    global parser
    try:
        parser = importlib.import_module("place_parser")
        print("[OK] 解析器：import place_parser")
        return
    except Exception as e:
        print(f"[WARN] 直接 import place_parser 失败：{e}")

    if PARSER_FILE and Path(PARSER_FILE).exists():
        try:
            spec = importlib.util.spec_from_file_location("place_parser_ext", PARSER_FILE)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)  # type: ignore
            parser = mod
            print(f"[OK] 解析器：from file {PARSER_FILE}")
            return
        except Exception as e:
            print(f"[WARN] 通过 PARSER_FILE 加载解析器失败：{e}")

    print("[WARN] 未能导入外部解析器；将仅保留原值（不做规范化）。")
    class _Fallback:
        def normalize_text(self, s: str) -> str:
            s = (s or "").strip()
            s = re.sub(r"\s+", "", s)
            return s
        def normalize_all_systems(self, raw_text: str, _unused_min_units, *idx):
            return {"民政": [], "军事卫所": [], "土司": []}
        CIVIL_XLSX   = ""
        MIL_XLSX     = ""
        TUSI_XLSX    = ""
        CIVIL_SHEET  = None
        MIL_SHEET    = None
        TUSI_SHEET   = None
        def build_generic_index(self, *args, **kwargs):
            raise RuntimeError("no-index")
    parser = _Fallback()

_try_import_parser()

# ========= 鲁棒加载 =========
def robust_load(src_path: str):
    p = Path(src_path)
    fmt = {".nt":"nt", ".ttl":"turtle", ".rdf":"xml", ".owl":"xml", ".xml":"xml"}.get(p.suffix.lower())
    if fmt is None: raise RuntimeError(f"不支持的本体后缀：{p.suffix}")
    g0 = Graph(); g0.parse(str(p), format=fmt)
    tmp_owl = Path(p.parent) / (p.stem + "_tmp_for_owlready2.owl")
    g0.serialize(destination=str(tmp_owl), format="xml", encoding="utf-8")
    world = World()
    onto  = world.get_ontology(str(tmp_owl)).load()
    g = world.as_rdflib_graph()
    return world, onto, g

world, onto, g = robust_load(SRC)
print(f"[OK] 加载：{SRC}  三元组≈{len(g)}")

# ========= 工具 =========
def find_class(names) -> Optional[ThingClass]:
    targets = set(names)
    for c in list(onto.classes()):
        try:
            if c.name in targets: return c
            iri = getattr(c, "iri", "")
            if iri and any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in targets): return c
        except: pass
    for c in list(world.classes()):
        try:
            if c.name in targets: return c
            iri = getattr(c, "iri", "")
            if iri and any(iri.endswith("#"+n) or iri.endswith("/"+n) for n in targets): return c
        except: pass
    return None

def dps_named(name_exact: str) -> List[Any]:
    hits, seen = [], set()
    for dp in list(onto.data_properties()) + list(world.data_properties()):
        try:
            nm = getattr(dp, "name", "")
            iri = str(getattr(dp, "iri", ""))
            if nm == name_exact and iri not in seen:
                hits.append(dp); seen.add(iri)
        except: pass
    return hits

def iri_of(x) -> str:
    try: return str(getattr(x, "iri"))
    except Exception: return str(x)

def person_name(p) -> str:
    try:
        v = getattr(p, "姓名")
        if isinstance(v, list): v = v[0] if v else ""
        return (v or getattr(p, "name","") or "").strip()
    except:
        return (getattr(p, "name","") or "").strip()

def normalize_text(s: str) -> str:
    try:
        return parser.normalize_text(s)
    except Exception:
        s = (s or "").strip()
        s = re.sub(r"\s+", "", s)
        return s

# ========= 仅操作户籍地 =========
Person = find_class(["Person","人物"])
if not Person: raise SystemExit("[FATAL] 未找到 Person/人物 类。")

HUJI_DPS = dps_named("户籍地")
if not HUJI_DPS: raise SystemExit("[FATAL] 未找到数据属性『户籍地』")
PREF_DP = HUJI_DPS[0]
print("[INFO] 户籍地 DPs:", [iri_of(dp) for dp in HUJI_DPS])
print("[INFO] 写回目标 DP：", iri_of(PREF_DP))

# ========= AnnotationProperty =========
def base_ns(iri: str) -> str:
    return iri.rsplit("#",1)[0] + "#" if "#" in iri else iri.rsplit("/",1)[0] + "#"

BASE = base_ns(iri_of(PREF_DP))
AP_IS_NORM  = URIRef(BASE + "是否规范")
AP_RAW      = URIRef(BASE + "原始户籍地")
AP_AMBIG    = URIRef(BASE + "存在歧义")
AP_REVIEW   = URIRef(BASE + "需进一步审核")

with onto:
    for ap in (AP_IS_NORM, AP_RAW, AP_AMBIG, AP_REVIEW):
        if (ap, RDF.type, OWL.AnnotationProperty) not in g:
            g.add((ap, RDF.type, OWL.AnnotationProperty))

# ========= 读取（双通道） =========
def read_values_dual_for_dp(p, dp) -> Tuple[List[str], List[str]]:
    """返回：rdflib值列表, owlready2值列表（原文）"""
    s = URIRef(iri_of(p))
    dp_uri = URIRef(iri_of(dp))
    rd_vals, ow_vals = [], []

    # rdflib 通道
    for o in g.objects(s, dp_uri):
        if isinstance(o, Literal):
            txt = str(o).strip()
            if txt: rd_vals.append(txt)

    # owlready2 通道
    try:
        for v in list(dp[p]):
            if v is None: continue
            txt = str(v).strip()
            if txt: ow_vals.append(txt)
    except Exception:
        try:
            for v in list(getattr(onto, dp.name)[p]):
                if v is None: continue
                txt = str(v).strip()
                if txt: ow_vals.append(txt)
        except Exception:
            pass

    return rd_vals, ow_vals

def merge_raw_values_with_containment(values: List[str]) -> List[str]:
    """
    原值层面：
      - 去空
      - 按 normalize_text 判重（相同规范串保留更长原文）
      - 『被包含』删除：若 A 的规范串是 B 的子串且 len(B) > len(A)，删除 A，保留 B
        （例如 '贵州卫' 与 '贵州贵州卫' → 仅保留 '贵州贵州卫'）
    """
    vals = [v for v in values if (v or "").strip()]
    if not vals: return []

    # 同值去重：相同规范串保留更长原文
    best_by_norm: Dict[str, str] = {}
    for v in vals:
        nv = normalize_text(v)
        if not nv: continue
        if nv not in best_by_norm or len(v) > len(best_by_norm[nv]):
            best_by_norm[nv] = v
    uniq = list(best_by_norm.values())

    # 被包含删除（按规范串）
    norm_map = {v: normalize_text(v) for v in uniq}
    drop: Set[str] = set()
    for a in uniq:
        na = norm_map[a]
        for b in uniq:
            if a == b: continue
            nb = norm_map[b]
            if len(nb) > len(na) and nb.find(na) != -1:
                drop.add(a); break

    survivors = [v for v in uniq if v not in drop]
    return survivors

# ========= 删除现有值 + Axiom =========
def remove_axioms_for_dp(graph: Graph, subj: URIRef, dp_uri: URIRef):
    to_kill = []
    for ax in graph.subjects(predicate=OWL.annotatedSource, object=subj):
        if (ax, OWL.annotatedProperty, dp_uri) in graph:
            to_kill.append(ax)
    for ax in to_kill:
        graph.remove((ax, None, None))

def hard_delete_all_for_dp(inst, dp) -> int:
    s = URIRef(iri_of(inst)); dp_uri = URIRef(iri_of(dp))
    cnt = sum(1 for _ in g.triples((s, dp_uri, None)))
    with onto:
        # 清 owlready2 值
        try: dp[inst] = []
        except:
            try: getattr(onto, dp.name)[inst] = []
            except: pass
        # 删 Axiom
        remove_axioms_for_dp(g, s, dp_uri)
        # 删 rdflib 三元组
        for (ss, pp, oo) in list(g.triples((s, dp_uri, None))):
            g.remove((ss, pp, oo))
    return cnt

# ========= Axiom 写入 =========
def write_axiom(graph: Graph, subj: URIRef, dp_uri: URIRef, lit: Literal,
                *, is_norm: bool, raw_joined: str, ambiguous: bool):
    ax = BNode()
    with onto:
        graph.add((ax, RDF.type, OWL.Axiom))
        graph.add((ax, OWL.annotatedSource, subj))
        graph.add((ax, OWL.annotatedProperty, dp_uri))
        graph.add((ax, OWL.annotatedTarget, lit))
        graph.add((ax, AP_IS_NORM, Literal("是" if is_norm else "否", datatype=XSD.string)))
        graph.add((ax, AP_RAW,     Literal(raw_joined or "", datatype=XSD.string)))
        graph.add((ax, AP_AMBIG,   Literal("是" if ambiguous else "否", datatype=XSD.string)))
        graph.add((ax, AP_REVIEW,  Literal("是" if ambiguous else "否", datatype=XSD.string)))

def write_values_with_axioms(inst, dp, values: List[str], *, is_norm_flags: List[bool],
                             raw_joined_for_norm: str, ambiguous: bool, raw_for_fail: Dict[str,str]):
    s = URIRef(iri_of(inst)); dp_uri = URIRef(iri_of(dp))
    with onto:
        # 批量设值（失败则逐条兜底）
        try:
            dp[inst] = list(values)
        except Exception:
            try:
                getattr(onto, dp.name)[inst] = list(values)
            except Exception:
                cur = []
                try: cur = list(dp[inst])
                except: pass
                cur = [x for x in cur if x is not None] + list(values)
                try: dp[inst] = cur
                except: pass

        # 为每个值写注释
        for v, is_ok in zip(values, is_norm_flags):
            lit = Literal(v, datatype=XSD.string)
            raw_join = raw_joined_for_norm if is_ok else raw_for_fail.get(v, v)
            write_axiom(g, s, dp_uri, lit, is_norm=is_ok, raw_joined=raw_join, ambiguous=ambiguous)

# ========= 规范（严格按“最新解析脚本”的调用方式） =========
def normalize_one_raw(text: str, civil_idx, mil_idx, tusi_idx) -> List[str]:
    """
    返回单个原值的规范候选（可能多值）。
    —— 依据最新解析器：normalize_all_systems(raw_text, {}, civil_idx, mil_idx, tusi_idx)
    """
    try:
        canon = parser.normalize_all_systems(text, {}, civil_idx, mil_idx, tusi_idx)
        out = []
        for sys_name in ("民政", "军事卫所", "土司"):
            for item in canon.get(sys_name, []):
                if item.get("matched") and item.get("value"):
                    out.append(item["value"])
        # 以规范串去重
        seen = set(); res = []
        for v in out:
            nv = normalize_text(v)
            if nv and nv not in seen:
                res.append(v); seen.add(nv)
        return res
    except Exception:
        return []

def merge_same_and_containment(values: List[str]) -> List[str]:
    """规范后的同值去重 + 被包含合并（保留更长信息量）"""
    vals = [v for v in values if (v or "").strip()]
    if not vals: return []
    # 同值去重（按规范串保留更长）
    best_by_norm: Dict[str, str] = {}
    for v in vals:
        nv = normalize_text(v)
        if not nv: continue
        if nv not in best_by_norm or len(v) > len(best_by_norm[nv]):
            best_by_norm[nv] = v
    uniq = list(best_by_norm.values())

    # 被包含合并
    norm_map = {v: normalize_text(v) for v in uniq}
    keep: List[str] = []
    for v in uniq:
        nv = norm_map[v]
        drop = False
        for u in uniq:
            if u == v: continue
            nu = norm_map[u]
            if len(nu) > len(nv) and nu.find(nv) != -1:
                drop = True; break
        if not drop:
            keep.append(v)
    return keep

# ========= 读 Excel 索引（解析器） =========
civil_idx = mil_idx = tusi_idx = None
try:
    civil_idx = parser.build_generic_index(parser.CIVIL_XLSX, sheet_name=getattr(parser, "CIVIL_SHEET", None))
    mil_idx   = parser.build_generic_index(parser.MIL_XLSX,   sheet_name=getattr(parser, "MIL_SHEET", None))
    tusi_idx  = parser.build_generic_index(parser.TUSI_XLSX,  sheet_name=getattr(parser, "TUSI_SHEET", None))
    print("[OK] 三表索引构建完成")
except Exception as e:
    print(f"[WARN] 三表索引不可用：{e}；将保留原值，不做规范化。")
    civil_idx = mil_idx = tusi_idx = None

# ========= 主流程 =========
people = list(Person.instances())
print(f"[INFO] 人物数：{len(people)}")

changed = 0
for idx, p in enumerate(people, 1):
    pname = person_name(p) or "(无名)"
    piri  = iri_of(p)
    print(f"\n[PERSON] {idx}/{len(people)}  姓名: {pname}  IRI: {piri}")

    # 1) 双通道：把所有『户籍地』值全部捞出
    raw_all: List[str] = []
    for dp in HUJI_DPS:
        rd_vals, ow_vals = read_values_dual_for_dp(p, dp)
        print(f"  - 读取（DP=户籍地）rdflib={len(rd_vals)}: {rd_vals}")
        print(f"  - 读取（DP=户籍地）owlready2={len(ow_vals)}: {ow_vals}")
        raw_all.extend(rd_vals); raw_all.extend(ow_vals)

    if not raw_all:
        print("  - 无户籍地原值，跳过")
        continue

    # 2) 原值合并：同值去重 + 『被包含』删除
    pre_dedup = merge_raw_values_with_containment(raw_all)
    print(f"  - 原值合并去重+被包含后={len(pre_dedup)}：{pre_dedup}")

    # 3) 清空旧『户籍地』（含 Axiom）
    total_deleted = 0
    for dp in HUJI_DPS:
        total_deleted += hard_delete_all_for_dp(p, dp)
    print(f"  - 清空旧『户籍地』：删除值 {total_deleted} 条（含其注释Axiom）")

    # 4) 逐条规范（严格按最新解析器的 normalize_all_systems 调用）
    per_raw_log: List[Tuple[str, List[str]]] = []
    canon_all: List[str] = []
    fail_kept: List[str] = []

    if civil_idx and mil_idx and tusi_idx:
        for raw in pre_dedup:
            hits = normalize_one_raw(raw, civil_idx, mil_idx, tusi_idx)
            per_raw_log.append((raw, hits))
            if hits:
                canon_all.extend(hits)
            else:
                fail_kept.append(raw)
    else:
        for raw in pre_dedup:
            per_raw_log.append((raw, []))
            fail_kept.append(raw)

    # 打印逐条规范结果
    for raw, hits in per_raw_log:
        print(f"  · 原值: {raw} -> 规范候选: {hits if hits else '（未命中，保留原值）'}")

    # 5) 规范结果合并（同值去重 + 被包含合并）
    canon_merged = merge_same_and_containment(canon_all)
    print(f"  - 合并后规范值：{canon_merged if canon_merged else '（无）'}")
    if fail_kept:
        print(f"  - 保留未解析原值：{fail_kept}")

    # 6) 歧义：最终写回条数（规范 + 失败）≥ 2
    final_count = len(canon_merged) + len(fail_kept)
    ambiguous = final_count >= 2
    print(f"  - 歧义：{'是' if ambiguous else '否'}（最终条数={final_count}）")

    # 7) 写回 + Axiom
    to_write: List[str] = []
    is_norm_flags: List[bool] = []
    raw_for_fail: Dict[str, str] = {}

    # 命中的规范值
    raw_joined = ""
    if canon_merged:
        raw_joined = "｜".join(sorted(set(pre_dedup), key=lambda x: normalize_text(x)))
        for v in canon_merged:
            to_write.append(v); is_norm_flags.append(True)

    # 规范失败保留
    for v in fail_kept:
        to_write.append(v); is_norm_flags.append(False)
        raw_for_fail[v] = v

    if to_write:
        write_values_with_axioms(
            p, PREF_DP, to_write,
            is_norm_flags=is_norm_flags,
            raw_joined_for_norm=raw_joined,
            ambiguous=ambiguous,
            raw_for_fail=raw_for_fail
        )
        print(f"  - 写回『户籍地』{len(to_write)} 条；其中规范={sum(is_norm_flags)}，非规范={len(to_write)-sum(is_norm_flags)}")
        changed += 1
    else:
        print("  - 本人无可写回值（异常）")

# ========= 保存 =========
g.serialize(destination=OUT_TTL, format="turtle")
g.serialize(destination=OUT_NT,  format="nt")
print(f"\n[SAVED] {OUT_TTL}")
print(f"[SAVED] {OUT_NT}")
print(f"[DONE] 完成：改动人数≈{changed}")
