# -*- coding: utf-8 -*-
"""
一体化调试：先补齐溯源链，再逐个官职检验与导出
- 步骤1（执行）：全局补齐 contains，使 TextProvenance.contains 覆盖 PropAssertion.about 的全部目标
- 步骤2（检验）：逐个 OfficialPosition 追踪其溯源字符串，输出明细/汇总 CSV，并打印缺失样本
"""

from owlready2 import *
from rdflib import Graph
from pathlib import Path
import tempfile
import csv
import os, argparse, datetime
def _parse_args():
    ap = argparse.ArgumentParser(description="脚本9")
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


# ========= 输出路径 =========
OUT_DETAIL  = Path(SRC).with_name("pos_prov_trace.csv")                 # 检验明细
OUT_SUMMARY = Path(SRC).with_name("pos_prov_trace_summary.csv")         # 检验汇总
OUT_NT_FIXED   = Path(SRC).with_name(Path(SRC).stem + "_fixed.nt")      # 补齐后保存（便于留存）
OUT_TTL_FIXED  = Path(SRC).with_name(Path(SRC).stem + "_fixed.ttl")

# ========= 稳健加载（rdflib → RDF/XML → owlready2）=========
def load_ontology_robust(src_path: str):
    p = Path(src_path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    fmt = {".nt":"nt",".ttl":"turtle",".rdf":"xml",".owl":"xml",".xml":"xml"}.get(p.suffix.lower(), None)
    g = Graph()
    g.parse(str(p), format=fmt)
    td = tempfile.TemporaryDirectory()
    owl_path = Path(td.name) / (p.stem + "_tmp.owl")
    g.serialize(destination=str(owl_path), format="xml", encoding="utf-8")
    world = World()
    onto  = world.get_ontology(str(owl_path)).load()
    return world, onto, td

world, onto, _tmpdir = load_ontology_robust(SRC)

# ========= 基础查找 =========
def all_classes():  return list(onto.classes()) + list(world.classes())
def all_objprops(): return list(onto.object_properties()) + list(world.object_properties())
def all_dataprops():return list(onto.data_properties()) + list(world.data_properties())

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
    return None

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
OfficialPosition = find_class(["OfficialPosition"])
PropAssertion    = find_class(["PropAssertion"])
TextProvenance   = find_class(["TextProvenance"])

# 对象属性
about        = find_objprop_one(["about"])          # PropAssertion -> Thing
derivedFrom  = find_objprop_one(["derivedFrom"])    # PropAssertion -> TextProvenance
contains     = find_objprop_one(["contains"])       # TextProvenance -> Thing

# 数据属性：PropAssertion
dp_prop      = find_dataprop_one(["prop"])
dp_val       = find_dataprop_one(["value"])
dp_valn      = find_dataprop_one(["value_norm"])

# 数据属性：TextProvenance
dp_tp_src    = find_dataprop_one(["Text_Source", "Text_source", "TextSource"])
dp_tp_body   = find_dataprop_one(["Text_body", "TextBody", "Text_body"])
dp_tp_conf   = find_dataprop_one(["record_confidence", "recordConfidence"])

# 展示：官职关键字段（缺则空串）
dp_pos_core  = find_dataprop_one(["核心职称"])
dp_pos_inst  = find_dataprop_one(["机构"])
dp_pos_tier  = find_dataprop_one(["层级", "通用层级"])
dp_pos_raw   = find_dataprop_one(["原始称谓", "官职名称"])
dp_pos_grade = find_dataprop_one(["官阶"])
dp_pos_place = find_dataprop_one(["地名"])

# ========= 小工具 =========
def iri_of(x):
    try:
        return x.iri
    except Exception:
        return str(x)

def get_first(dp, inst):
    if dp is None: return ""
    try:
        vals = list(dp[inst])
        return str(vals[0]).strip() if vals else ""
    except Exception:
        return ""

def one_line(s, limit=600):
    if not s: return ""
    t = " ".join(str(s).split())
    return t if len(t) <= limit else (t[:limit] + "…")

def pa_key_text(pa):
    p = get_first(dp_prop, pa)
    v = get_first(dp_valn, pa) or get_first(dp_val, pa)
    return p, v

# ========= 步骤1：执行补齐（先做！）=========
def repair_provenance_chain_globally():
    """
    把所有 pa 的 derivedFrom(tp) 的 contains(tp) 补齐为 about(pa) 的全集。
    返回新增边数量。
    """
    if not (about and derivedFrom and contains):
        return 0
    patched = 0
    pas = list(PropAssertion.instances())
    for pa in pas:
        try:
            targets = list(about[pa])
        except Exception:
            targets = []
        if not targets:
            continue
        try:
            tps = list(derivedFrom[pa])
        except Exception:
            tps = []
        if not tps:
            continue
        for tp in tps:
            for t in targets:
                try:
                    if t not in contains[tp]:
                        contains[tp].append(t)
                        patched += 1
                except Exception:
                    pass
    return patched

patched_cnt = repair_provenance_chain_globally()
print(f"[REPAIR] 新增 contains 边数量：{patched_cnt}")

# （可选）把补齐后的图保存一份，便于复核
g_fixed = world.as_rdflib_graph()
g_fixed.serialize(destination=str(OUT_NT_FIXED),  format="nt")
g_fixed.serialize(destination=str(OUT_TTL_FIXED), format="turtle")
print(f"[SAVED] 保存补齐后的本体：\n  NT  -> {OUT_NT_FIXED}\n  TTL -> {OUT_TTL_FIXED}")

# ========= 步骤2：执行检验（逐个官职追踪并导出 CSV）=========
detail_rows = []
summary_rows = []

positions = list(OfficialPosition.instances())
positions.sort(key=lambda x: iri_of(x))

total_missing = 0
total_pa = 0
total_tp = 0

for pos in positions:
    pos_iri   = iri_of(pos)
    pos_core  = get_first(dp_pos_core, pos)
    pos_inst  = get_first(dp_pos_inst, pos)
    pos_tier  = get_first(dp_pos_tier, pos)
    pos_raw   = get_first(dp_pos_raw, pos)
    pos_grade = get_first(dp_pos_grade, pos)
    pos_place = get_first(dp_pos_place, pos)

    # 找到 about 到该官职的所有 PropAssertion
    pas = []
    for pa in list(PropAssertion.instances()):
        try:
            if about and pos in about[pa]:
                pas.append(pa)
        except Exception:
            pass

    pa_cnt = len(pas)
    missing_cnt = 0

    if pa_cnt == 0:
        summary_rows.append([
            pos_iri, pos_core, pos_inst, pos_tier, pos_raw, pos_grade, pos_place,
            pa_cnt, 0, 0
        ])
        continue

    for pa in pas:
        total_pa += 1
        ptxt, vtxt = pa_key_text(pa)

        try:
            tps = list(derivedFrom[pa]) if derivedFrom else []
        except Exception:
            tps = []

        if not tps:
            # 无 TextProvenance 也记录一行（便于追踪）
            detail_rows.append([
                pos_iri, pos_core, pos_inst, pos_tier, pos_raw, pos_grade, pos_place,
                iri_of(pa), ptxt, vtxt,
                "", "", "", "", "",  # tp/文本/可信度/计数/contains检查
            ])
            continue

        for tp in tps:
            total_tp += 1
            tp_iri  = iri_of(tp)
            tp_src  = get_first(dp_tp_src, tp)
            tp_body = get_first(dp_tp_body, tp)
            tp_conf = get_first(dp_tp_conf, tp)

            # 校验 contains 是否含该官职（应为 True，因已补齐）
            has_pos = ""
            try:
                has_pos = (pos in contains[tp]) if contains else False
            except Exception:
                has_pos = False
            if not has_pos:
                missing_cnt += 1
                total_missing += 1

            # 附带计数，帮助排查
            try:
                about_targets_count = len(list(about[pa]))
            except Exception:
                about_targets_count = 0
            try:
                contains_targets_count = len(list(contains[tp])) if contains else 0
            except Exception:
                contains_targets_count = 0

            detail_rows.append([
                pos_iri, pos_core, pos_inst, pos_tier, pos_raw, pos_grade, pos_place,
                iri_of(pa), ptxt, vtxt,
                tp_iri, tp_src, one_line(tp_body, 600), tp_conf,
                str(about_targets_count), str(contains_targets_count),
                "TRUE" if has_pos else "FALSE"
            ])

    tp_cnt_for_pos = sum(1 for r in detail_rows if r[0] == pos_iri and r[10])  # 本官职关联的 tp 行
    summary_rows.append([
        pos_iri, pos_core, pos_inst, pos_tier, pos_raw, pos_grade, pos_place,
        pa_cnt, tp_cnt_for_pos, missing_cnt
    ])

# ========= 导出 CSV =========
with OUT_DETAIL.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "pos_iri","核心职称","机构","层级","原始称谓","官阶","地名",
        "pa_iri","prop","value_or_norm",
        "tp_iri","Text_Source","Text_body(snippet)","record_confidence",
        "about_targets_count","contains_targets_count","contains_has_position"
    ])
    w.writerows(detail_rows)

with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "pos_iri","核心职称","机构","层级","原始称谓","官阶","地名",
        "prop_assertions_count","textprovenances_count","missing_contains_for_this_position"
    ])
    w.writerows(summary_rows)

# ========= 控制台打印 =========
print(f"[STATS] OfficialPosition 总数：{len(positions)}")
print(f"[STATS] 涉及的 PropAssertion 总数：{total_pa}")
print(f"[STATS] 涉及的 TextProvenance 总数：{total_tp}")
print(f"[STATS] contains 缺失总计（补齐后仍为 FALSE 的计数）：{total_missing}")
print(f"[CSV] 明细 -> {OUT_DETAIL}")
print(f"[CSV] 汇总 -> {OUT_SUMMARY}")

# 打印若干缺失样本（最多 10 条）
missing_samples = [r for r in detail_rows if r and len(r) >= 17 and r[-1] == "FALSE"][:10]
if missing_samples:
    print("\n[WARN] contains 缺失样本（前 10 条）:")
    for r in missing_samples:
        print(f"  pos={r[0]} | pa={r[7]} | prop={r[8]} | val={r[9]} | tp={r[10]} | src={r[11]}")
else:
    print("\n✅ 检验通过：所有 TextProvenance.contains 均包含对应官职。")
