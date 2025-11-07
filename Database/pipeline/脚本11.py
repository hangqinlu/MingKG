# -*- coding: utf-8 -*-
"""
后处理：检查并纠正 Place.现代区划层级（输出调试明细 CSV）
- 为每个 Place 写出一行调试信息到 CSV：Place_URI | 历史名称 | 现代名称 | 解析层级_规范化 | 现有层级_原值 | 现有层级_规范化 | 动作
- 同时保留原逻辑：覆盖写回本体中的“现代区划层级”，输出 NT + 变更清单 CSV
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from rdflib import Graph, URIRef, RDF, RDFS, Literal
from rdflib.namespace import OWL, XSD
import csv, time, re
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



_PREFIX = f"{SRC.stem}_postcheck_admin"
NT_OUT = SRC.with_name(f"{_PREFIX}.nt")
CSV_LOG = SRC.with_name(f"{_PREFIX}_changes.csv")        # 变更清单（filled/corrected/kept）
CSV_DBG = SRC.with_name(f"{_PREFIX}_debug.csv")          # 新增：调试明细（逐条输出解析信息）

# ====== 本体元素（按局部名匹配）======
CLASS_PLACE = "Place"
DP_MOD_NAME = "现代名称"
DP_HIS_NAME = "历史名称"
DP_ADMIN_LV = "现代区划层级"   # 若不存在则自动创建

# ====== 解析规则 ======
MUNICIPALITIES = {"北京市", "天津市", "上海市", "重庆市"}
SARS = {"香港特别行政区", "澳门特别行政区"}

LEVEL_SUFFIXES = {
    "province":  ["特别行政区", "自治区", "省"],
    "prefecture":["自治州", "地区", "盟", "州", "市"],
    "county":    ["自治旗", "自治县", "林区", "特区", "新区", "市辖区", "旗", "县", "区", "市"],
    "township":  ["民族乡", "苏木", "乡", "镇", "街道", "地区办事处"],
    "village":   ["居委会", "村委会", "嘎查村", "社区", "嘎查", "村"],
}
ALL_SUFFIXES: List[str] = sorted(set(sum(LEVEL_SUFFIXES.values(), [])), key=len, reverse=True)
PUNCT_RE = re.compile(r"[·•·\s、，,；;]+")

# —— 规范化映射
CANON_MAP = {
    "市辖区": "区",
    "自治县": "县",
    "自治州": "州",
    "自治旗": "旗",
    "嘎查村": "村",
}
def canon_level_suffix(s: str) -> str:
    return CANON_MAP.get(s or "", s or "")

# ====== 解析工具 ======
def _clean(s: str) -> str:
    return PUNCT_RE.sub("", (s or "").strip())

def _consume_province_prefix(s: str) -> Tuple[Optional[str], str]:
    for special in sorted(MUNICIPALITIES | SARS, key=len, reverse=True):
        if s.startswith(special):
            return special, s[len(special):]
    m = re.match(r"^(.+?(?:自治区|省))", s)
    if m:
        return m.group(1), s[m.end():]
    return None, s

def _allowed_suffixes_for_stage(stage: str) -> List[Tuple[str, str]]:
    if stage == "start":
        lvls = ["province", "prefecture", "county"]
    elif stage == "province":
        lvls = ["prefecture", "county"]
    elif stage == "prefecture":
        lvls = ["county"]
    elif stage == "county":
        lvls = ["township"]
    elif stage == "township":
        lvls = ["village"]
    else:
        lvls = []
    sufs: List[Tuple[str,str]] = []
    for lvl in lvls:
        for suf in LEVEL_SUFFIXES.get(lvl, []):
            sufs.append((lvl, suf))
    sufs.sort(key=lambda x: len(x[1]), reverse=True)
    return sufs

def _next_starts_with_suffix(s: str, i: int) -> bool:
    if i >= len(s):
        return False
    tail = s[i:]
    for suf in ALL_SUFFIXES:
        if tail.startswith(suf):
            return True
    return False

def _tokenize_hier(name: str) -> List[Dict[str, str]]:
    s = _clean(name)
    tokens: List[Dict[str,str]] = []

    def push(raw: str, level: str, suffix: str):
        base = raw[:-len(suffix)] if suffix and len(raw) > len(suffix) else raw
        tokens.append({"raw": raw, "level": level, "suffix": suffix, "name_base": base})

    prov, rest = _consume_province_prefix(s)
    stage = "start"
    if prov:
        suf = "市" if prov in MUNICIPALITIES else ("特别行政区" if prov in SARS else ("自治区" if prov.endswith("自治区") else "省"))
        push(prov, "province", suf)
        s = rest
        stage = "province"

    buf = ""
    i = 0
    while i < len(s):
        buf += s[i]
        i += 1
        matched = False
        for lvl, suf in _allowed_suffixes_for_stage(stage):
            if buf.endswith(suf):
                if suf == "州" and lvl == "prefecture" and _next_starts_with_suffix(s, i):
                    continue  # 等待“州市”等更长片段
                push(buf, lvl, suf)
                buf = ""
                stage = lvl
                matched = True
                break
        if matched:
            continue
    if buf:
        push(buf, "unknown", "")
    return tokens

def last_unit_suffix_from_full_name(fullname: str) -> str:
    segs = _tokenize_hier(fullname)
    for seg in reversed(segs):
        if seg["level"] in ("province","prefecture","county","township","village"):
            return canon_level_suffix(seg["suffix"] or "")
    return ""

# ====== rdflib 工具 ======
def localname(u: URIRef) -> str:
    s = str(u)
    for sep in ["#", "/", ":"]:
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s

def split_base_and_local(uri: URIRef) -> Tuple[str, str]:
    s = str(uri)
    for sep in ["#", "/"]:
        if sep in s:
            base, loc = s.rsplit(sep, 1)
            return base + sep, loc
    return s, ""

def find_predicates_by_local(g: Graph, target_local: str) -> List[URIRef]:
    uris: Set[URIRef] = set()
    for s, p, o in g:
        if localname(p) == target_local:
            uris.add(p)
    return list(uris)

def get_place_class_uri(g: Graph) -> Optional[URIRef]:
    for s, t in g.subject_objects(RDF.type):
        if localname(t) == CLASS_PLACE:
            return t
    return None

def guess_base_from_node_props(g: Graph, node: URIRef, prefer_locals: List[str]) -> Optional[str]:
    for p, o in g.predicate_objects(node):
        if localname(p) in prefer_locals:
            base, _ = split_base_and_local(p)
            return base
    for p, o in g.predicate_objects(node):
        if isinstance(o, Literal):
            base, _ = split_base_and_local(p)
            return base
    return None

def get_or_create_dataprop(g: Graph, node: URIRef, pred_local: str) -> URIRef:
    cands = find_predicates_by_local(g, pred_local)
    if cands:
        return cands[0]
    base = guess_base_from_node_props(g, node, [DP_MOD_NAME, DP_ADMIN_LV, DP_HIS_NAME])
    if base is None:
        cls = get_place_class_uri(g)
        if cls is not None:
            base, _ = split_base_and_local(cls)
        else:
            base = "http://example.org/ontology#"
    p = URIRef(base + pred_local)
    g.add((p, RDF.type, OWL.DatatypeProperty))
    g.add((p, RDFS.label, Literal(pred_local)))
    cls = get_place_class_uri(g)
    if cls is not None:
        g.add((p, RDFS.domain, cls))
    g.add((p, RDFS.range, XSD.string))
    return p

def set_literal_overwrite(g: Graph, s: URIRef, pred_local: str, value: str):
    if value is None:
        return False
    p_uri = None
    for p, o in g.predicate_objects(s):
        if localname(p) == pred_local:
            p_uri = p; break
    if p_uri is None:
        p_uri = get_or_create_dataprop(g, s, pred_local)
    for o in list(g.objects(s, p_uri)):
        if isinstance(o, Literal):
            g.remove((s, p_uri, o))
    g.add((s, p_uri, Literal(value)))
    return True

# ====== 主流程（输出 CSV 调试明细）======
def main():
    if not SRC.exists():
        raise FileNotFoundError(f"找不到输入本体：{SRC}")

    print(f"[LOAD] 读取：{SRC}")
    g = Graph()
    g.parse(location=str(SRC), format="nt")

    places: List[URIRef] = [s for s, t in g.subject_objects(RDF.type) if localname(t) == CLASS_PLACE]

    changes = []
    debug_rows = []
    filled = 0
    corrected = 0
    kept_ok = 0
    skipped_no_mod = 0

    for pl in places:
        mod = ""
        his_list: List[str] = []
        exist_lvls: List[str] = []

        for p, o in g.predicate_objects(pl):
            if isinstance(o, Literal):
                lp = localname(p)
                if lp == DP_MOD_NAME and not mod:
                    mod = str(o).strip()
                elif lp == DP_HIS_NAME:
                    v = str(o).strip()
                    if v:
                        his_list.append(v)
                elif lp == DP_ADMIN_LV:
                    v = str(o).strip()
                    if v:
                        exist_lvls.append(v)

        hist_display = " | ".join(dict.fromkeys(his_list)) if his_list else ""
        existing_raw = exist_lvls[0] if exist_lvls else ""
        existing_canon = canon_level_suffix(existing_raw) if existing_raw else ""

        if not mod:
            skipped_no_mod += 1
            debug_rows.append({
                "Place_URI": str(pl),
                "历史名称": hist_display,
                "现代名称": "",
                "解析层级_规范化": "",
                "现有层级_原值": existing_raw,
                "现有层级_规范化": existing_canon,
                "动作": "skipped_no_mod",
            })
            continue

        computed_lvl = last_unit_suffix_from_full_name(mod)

        if not existing_raw:
            set_literal_overwrite(g, pl, DP_ADMIN_LV, computed_lvl)
            filled += 1
            action = "filled"
            changes.append({"Place_URI": str(pl), "现代名称": mod, "动作": "filled", "旧值": "", "新值": computed_lvl})
        else:
            if existing_canon != computed_lvl:
                set_literal_overwrite(g, pl, DP_ADMIN_LV, computed_lvl)
                corrected += 1
                action = "corrected"
                changes.append({"Place_URI": str(pl), "现代名称": mod, "动作": "corrected", "旧值": existing_raw, "新值": computed_lvl})
            else:
                kept_ok += 1
                action = "kept"
                changes.append({"Place_URI": str(pl), "现代名称": mod, "动作": "kept", "旧值": existing_raw, "新值": existing_raw})

        debug_rows.append({
            "Place_URI": str(pl),
            "历史名称": hist_display,
            "现代名称": mod,
            "解析层级_规范化": computed_lvl,
            "现有层级_原值": existing_raw,
            "现有层级_规范化": existing_canon,
            "动作": action,
        })

    # 保存 NT
    g.serialize(destination=str(NT_OUT), format="nt")
    print(f"[SAVE] 已写出：{NT_OUT}（填充 {filled}；纠正 {corrected}；保持 {kept_ok}；无现代名 {skipped_no_mod}）")

    # 变更 CSV
    with CSV_LOG.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Place_URI","现代名称","动作","旧值","新值"])
        w.writeheader()
        for r in changes:
            w.writerow(r)
    print(f"[CSV] 变更清单：{CSV_LOG}（{len(changes)} 条）")

    # 调试明细 CSV（你要的“输出为CSV”）
    with CSV_DBG.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Place_URI","历史名称","现代名称",
                "解析层级_规范化","现有层级_原值","现有层级_规范化","动作"
            ]
        )
        w.writeheader()
        for r in debug_rows:
            w.writerow(r)
    print(f"[CSV] 调试明细：{CSV_DBG}（{len(debug_rows)} 条）")

if __name__ == "__main__":
    main()
