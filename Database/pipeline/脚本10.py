# -*- coding: utf-8 -*-
"""
Place 层级链构建（仅使用现代名称；禁止历史名称参与匹配）
- 构建对象属性：isSubPlaceOf（Place→Place）
- 父级查找：只用现代名称做“精确匹配”，并做层级严格校验；不使用历史名称；不做模糊/包含
- 父级缺失：自动创建（现代名称 = 父级全称；现代区划层级 = 父名的最小单位后缀）
- 无现代名称的实例：跳过（不构链、不补父）
- 逐实例端口打印 + 两个 CSV：
  · place_hierarchy_build_summary.csv —— 每个处理过实例的一次结果摘要
  · place_hierarchy_chains.csv —— 每个实例自下而上的现代名称链与 URI 链 http://mingkg.org/ontology/imperial_exam.owl#Place_4_8a004936_6b6e_4d1d_81b0_757e1b17d648_71

依赖：rdflib
pip install rdflib
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from rdflib import Graph, URIRef, RDF, RDFS, Literal
from rdflib.namespace import OWL, XSD
import csv, re, uuid
from collections import deque
import os

# ========= 开关 =========
VERBOSE = True  # 打开逐实例端口打印

def log(*args):
    if VERBOSE:
        print(*args)

import argparse


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
SRC      = Path(_args.src).expanduser().resolve()
OUT_DIR  = Path(_args.out_dir).expanduser().resolve()

# 确保输出目录存在
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_CHAIN   = Path(SRC).with_name("place_hierarchy_chains.csv")
CSV_SUMMARY = Path(SRC).with_name("place_hierarchy_build_summary.csv")
NT_OUT  = Path(SRC).with_name("创建地点层级链.nt")


# ========= 本体元素（按局部名匹配）=========
CLASS_PLACE    = "Place"
DP_HIS_NAME    = "历史名称"         # 仅输出到 CSV，算法完全不用
DP_MOD_NAME    = "现代名称"
DP_ADMIN_LV_1  = "现代区划层级"     # 优先写回/读取
DP_ADMIN_LV_2  = "区划层级"         # 兼容旧字段
OP_SUB_PLACE   = "isSubPlaceOf"

# ========= 名称解析词典与规则 =========
MUNICIPALITIES = {"北京市", "天津市", "上海市", "重庆市"}      # 省级
SARS = {"香港特别行政区", "澳门特别行政区"}                  # 省级

LEVEL_SUFFIXES = {
    "province":  ["特别行政区", "自治区", "省"],
    "prefecture":["自治州", "地区", "盟", "州", "市"],
    "county":    ["自治旗", "自治县", "林区", "特区", "新区", "市辖区", "旗", "县", "区", "市"],
    "township":  ["民族乡", "苏木", "乡", "镇", "街道", "地区办事处"],
    "village":   ["居委会", "村委会", "嘎查村", "社区", "嘎查", "村"],
}
ALL_SUFFIXES: List[str] = sorted(set(sum(LEVEL_SUFFIXES.values(), [])), key=len, reverse=True)
PUNCT_RE = re.compile(r"[·•·\s、，,；;]+")

CANON_MAP = {"市辖区":"区","自治县":"县","自治州":"州","自治旗":"旗","嘎查村":"村"}
def canon_level_suffix(s: str) -> str:
    return CANON_MAP.get(s or "", s or "")

# ========= 基础工具 =========
def localname(u: URIRef) -> str:
    s = str(u)
    for sep in ["#", "/", ":"]:
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s

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
    if stage == "start":        lvls = ["province", "prefecture", "county"]
    elif stage == "province":   lvls = ["prefecture", "county"]
    elif stage == "prefecture": lvls = ["county"]
    elif stage == "county":     lvls = ["township"]
    elif stage == "township":   lvls = ["village"]
    else:                       lvls = []
    sufs: List[Tuple[str,str]] = []
    for lvl in lvls:
        for suf in LEVEL_SUFFIXES.get(lvl, []):
            sufs.append((lvl, suf))
    sufs.sort(key=lambda x: len(x[1]), reverse=True)
    return sufs

def _next_starts_with_suffix(s: str, i: int) -> bool:
    if i >= len(s): return False
    tail = s[i:]
    for suf in ALL_SUFFIXES:
        if tail.startswith(suf):
            return True
    return False

def _tokenize_hier(name: str) -> List[Dict[str, str]]:
    """层级门控切分 + “州后缀”前瞻：避免把“漳州+市”切成“漳州|市”"""
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
                    continue
                push(buf, lvl, suf)
                buf = ""
                stage = lvl
                matched = True
                break
        if matched: continue
    if buf:
        push(buf, "unknown", "")
    return tokens

def parse_modern_place(full_name: str) -> Dict[str, Optional[str]]:
    segs = _tokenize_hier(full_name)
    out = {"province": None, "prefecture": None, "county": None, "township": None, "village": None, "segments": segs}
    for seg in segs:
        lv, raw = seg["level"], seg["raw"]
        if lv in out and out[lv] is None:
            out[lv] = raw
    return out

def last_unit_suffix_from_full_name(fullname: str) -> str:
    segs = _tokenize_hier(fullname)
    for seg in reversed(segs):
        if seg["level"] in ("province","prefecture","county","township","village"):
            return canon_level_suffix(seg["suffix"] or "")
    return ""

def parent_info_from_parsed(parsed: Dict[str, Optional[str]]) -> Optional[Dict[str, str]]:
    segs = parsed["segments"]
    idxs = [i for i, seg in enumerate(segs) if seg["level"] in ("province","prefecture","county","township","village")]
    if len(idxs) < 2: return None
    parent_idx = idxs[-2]
    parent_full = "".join(seg["raw"] for seg in segs[:parent_idx+1])
    return {
        "parent_full": parent_full,
        "parent_level_suffix": segs[parent_idx]["suffix"] or ""
    }

# ========= rdflib 帮手 =========
def get_place_class_uri(g: Graph) -> Optional[URIRef]:
    for s, t in g.subject_objects(RDF.type):
        if localname(t) == CLASS_PLACE:
            return t
    return None

def get_graph_base_for_place(g: Graph) -> str:
    cls = get_place_class_uri(g)
    if cls is not None:
        s = str(cls)
        for sep in ["#", "/"]:
            if sep in s:
                return s.rsplit(sep, 1)[0] + sep
    for s, t in g.subject_objects(RDF.type):
        if localname(t) == CLASS_PLACE:
            s_uri = str(s)
            for sep in ["#", "/"]:
                if sep in s_uri:
                    return s_uri.rsplit(sep, 1)[0] + sep
    return "http://example.org/ontology#"

def find_predicates_by_local(g: Graph, target_local: str) -> List[URIRef]:
    uris = set()
    for s, p, o in g:
        if localname(p) == target_local:
            uris.add(p)
    return list(uris)

def get_or_create_objprop(g: Graph, pred_local: str) -> URIRef:
    cands = find_predicates_by_local(g, pred_local)
    if cands: return cands[0]
    base = get_graph_base_for_place(g)
    p = URIRef(base + pred_local)
    g.add((p, RDF.type, OWL.ObjectProperty))
    cls = get_place_class_uri(g)
    if cls is not None:
        g.add((p, RDFS.domain, cls))
        g.add((p, RDFS.range,  cls))
    return p

def get_or_create_dataprop(g: Graph, pred_local: str) -> URIRef:
    cands = find_predicates_by_local(g, pred_local)
    if cands: return cands[0]
    base = get_graph_base_for_place(g)
    p = URIRef(base + pred_local)
    g.add((p, RDF.type, OWL.DatatypeProperty))
    g.add((p, RDFS.label, Literal(pred_local)))
    cls = get_place_class_uri(g)
    if cls is not None:
        g.add((p, RDFS.domain, cls))
    g.add((p, RDFS.range, XSD.string))
    return p

def add_literal_unique(g: Graph, s: URIRef, pred_local: str, value: str):
    if not value: return
    p_uri = None
    for p, o in g.predicate_objects(s):
        if localname(p) == pred_local:
            p_uri = p; break
    if p_uri is None:
        p_uri = get_or_create_dataprop(g, pred_local)
    for o in g.objects(s, p_uri):
        if str(o) == value:
            return
    g.add((s, p_uri, Literal(value)))

# ========= 名称索引（仅现代名！历史名完全不用） =========
def norm_mod_key(s: str) -> str:
    return _clean(s).lower()

def build_place_indexes_modonly(g: Graph, places: List[URIRef]):
    idx_mod: Dict[str, URIRef] = {}     # 仅现代名索引
    meta: Dict[URIRef, Dict[str, str]] = {}
    for pl in places:
        his, mod, lvl = "", "", ""
        for p, o in g.predicate_objects(pl):
            if isinstance(o, Literal):
                lp = localname(p)
                if lp == DP_MOD_NAME and not mod:
                    mod = str(o).strip()
                elif lp == DP_HIS_NAME and not his:  # 仅记录，不参与任何匹配
                    his = str(o).strip()
                elif lp in (DP_ADMIN_LV_1, DP_ADMIN_LV_2) and not lvl:
                    lvl = str(o).strip()
        if mod:
            idx_mod[norm_mod_key(mod)] = pl
        meta[pl] = {"his": his, "mod": mod, "lvl": lvl}
    return idx_mod, meta

# ========= 父级匹配（仅现代名精确匹配 + 层级严格校验） =========
def match_parent_by_modern_exact(idx_mod, name: str,
                                 *, exclude_uri: Optional[URIRef],
                                 require_level: Optional[str],
                                 level_getter) -> Tuple[Optional[URIRef], str]:
    if not name:
        return None, "none"
    q_mod = norm_mod_key(name)
    uri = idx_mod.get(q_mod)
    if uri and uri != exclude_uri:
        if require_level and level_getter and level_getter(uri) != require_level:
            return None, "exact_mod_level_mismatch"
        return uri, "exact_mod"
    return None, "none"

# ========= 主流程 =========
def main():
    if not SRC.exists():
        raise FileNotFoundError(f"找不到本体文件：{SRC}")

    log(f"[LOAD] 读取：{SRC}")
    g = Graph()
    g.parse(location=str(SRC), format="nt")

    # 初始 Place 实例
    init_places = [s for s, t in g.subject_objects(RDF.type) if isinstance(s, URIRef) and localname(t) == CLASS_PLACE]
    log(f"[INFO] 初始 Place 实例数：{len(init_places)}")

    # 索引、谓词
    idx_mod, meta = build_place_indexes_modonly(g, init_places)
    p_sub = get_or_create_objprop(g, OP_SUB_PLACE)
    cls_place = get_place_class_uri(g)
    if cls_place is None:
        base = get_graph_base_for_place(g)
        cls_place = URIRef(base + CLASS_PLACE)
        g.add((cls_place, RDF.type, OWL.Class))

    def level_of(uri: URIRef) -> str:
        row = meta.get(uri, {})
        lv = (row.get("lvl") or "").strip()
        if not lv:
            modn = (row.get("mod") or "").strip()
            if modn:
                lv = last_unit_suffix_from_full_name(modn)
        return canon_level_suffix(lv)

    # CSV 汇总
    sum_rows: List[Dict[str,str]] = []
    add_rel, create_parent = 0, 0

    # 全闭包
    queue = deque(init_places)
    processed: Set[URIRef] = set()

    while queue:
        child = queue.popleft()
        if child in processed:
            continue
        processed.add(child)

        his = meta.get(child, {}).get("his", "")
        mod = meta.get(child, {}).get("mod", "")
        log("="*88)
        log(f"[PLACE] 子地：{child}")
        log(f"  历史名称(仅展示) = {his or '(空)'}")
        log(f"  现代名称 = {mod or '(空)'}")

        # ★★ 关键：无现代名称 → 直接跳过（历史名称不允许参与任何匹配）
        if not mod:
            log("  → 跳过：无现代名称（禁用历史名称匹配）")
            sum_rows.append({
                "子地_URI": str(child),"子地_现代名称":"", "子地_历史名称": his,
                "父地_URI":"", "父地_现代名称":"", "父级全称":"", "匹配方式":"no-modern-name", "结果":"skipped"
            })
            continue

        parsed = parse_modern_place(mod)
        seg_view = [f"{seg['raw']}<{seg['level']}>" for seg in parsed["segments"]]
        start_level = last_unit_suffix_from_full_name(mod)
        log(f"  解析片段 = {seg_view}")
        log(f"  起点层级 = {start_level or '(未识别)'}")

        pinfo = parent_info_from_parsed(parsed)
        if not pinfo:
            log("  → 未得到父级信息（单段或未识别），视作顶级/数据待补")
            sum_rows.append({
                "子地_URI": str(child),"子地_现代名称": mod,"子地_历史名称": his,
                "父地_URI":"", "父地_现代名称":"", "父级全称":"", "匹配方式":"no-parent", "结果":"root-or-stub"
            })
            continue

        parent_full = pinfo["parent_full"]
        expected_parent_suffix = canon_level_suffix(pinfo["parent_level_suffix"])
        log(f"  期望父级全称 = {parent_full}")
        log(f"  期望父级层级 = {expected_parent_suffix or '(未识别)'}")

        # 自指防护：父全称与子现代名相同（规范化后）
        if norm_mod_key(parent_full) == norm_mod_key(mod):
            log("  !! 自指防护：parent_full 与 子地现代名相同 → 阻断")
            sum_rows.append({
                "子地_URI": str(child),"子地_现代名称": mod,"子地_历史名称": his,
                "父地_URI":"", "父地_现代名称":"", "父级全称": parent_full, "匹配方式":"self-parent-prefix-equal", "结果":"skipped"
            })
            continue

        # ★★ 仅现代名精确匹配 + 层级校验 + 排除 self
        def _level_getter(u: URIRef) -> str: return level_of(u)

        parent_uri, mode = match_parent_by_modern_exact(
            idx_mod, parent_full,
            exclude_uri=child,
            require_level=expected_parent_suffix,
            level_getter=_level_getter
        )
        if mode == "exact_mod_level_mismatch":
            log(f"  × 命中实例层级不符（mode={mode}）")

        parent_was_created = False
        if parent_uri is None:
            # 创建父级（仍只写现代字段）
            base = get_graph_base_for_place(g)
            parent_uri = URIRef(base + "Place_auto_" + uuid.uuid4().hex)
            g.add((parent_uri, RDF.type, cls_place))
            add_literal_unique(g, parent_uri, DP_MOD_NAME, parent_full)
            lvl_label = last_unit_suffix_from_full_name(parent_full)
            if lvl_label:
                add_literal_unique(g, parent_uri, DP_ADMIN_LV_1, lvl_label)

            # 更新索引/缓存（仅现代名）
            idx_mod[norm_mod_key(parent_full)] = parent_uri
            meta[parent_uri] = {"his":"", "mod":parent_full, "lvl":lvl_label}

            create_parent += 1
            mode = "created"
            parent_was_created = True
            log(f"  + 父级不存在，已创建：{parent_uri}（现代名称='{parent_full}'，层级='{lvl_label}'）")
        else:
            log(f"  √ 父级命中：{parent_uri}（mode={mode}，父级层级='{level_of(parent_uri)}'）")

        # 双保险：仍等于自身则放弃连边
        if parent_uri is child:
            log("  !! 双保险：parent_uri 等于 child，放弃连边")
            sum_rows.append({
                "子地_URI": str(child),"子地_现代名称": mod,"子地_历史名称": his,
                "父地_URI": str(parent_uri),"父地_现代名称": meta.get(parent_uri,{}).get("mod",""),
                "父级全称": parent_full,"匹配方式":"self-uri-after-create","结果":"skipped"
            })
            continue

        # 建边（若不存在）
        exists = any(True for _ in g.triples((child, p_sub, parent_uri)))
        if not exists:
            g.add((child, p_sub, parent_uri))
            add_rel += 1
            result = "added"
            log("  → 已新增 isSubPlaceOf 关系")
        else:
            result = "exists"
            log("  → 关系已存在")

        sum_rows.append({
            "子地_URI": str(child),
            "子地_现代名称": mod,
            "子地_历史名称": his,  # 仅供 CSV 参考
            "父地_URI": str(parent_uri),
            "父地_现代名称": meta.get(parent_uri,{}).get("mod",""),
            "父级全称": parent_full,
            "匹配方式": mode,
            "结果": f"{result}{'(parent_created)' if parent_was_created else ''}"
        })

        # 闭包：父级入队
        if parent_uri not in processed:
            queue.append(parent_uri)
            log("  → 父级加入队列，继续向上构建")

    # 保存新图
    g.serialize(destination=str(NT_OUT), format="nt")
    log(f"[SAVE] 写出：{NT_OUT}（新增关系 {add_rel}；新建父级 {create_parent}；处理 {len(processed)}）")

    # 写出构建摘要（含历史名仅供查看）
    with CSV_SUMMARY.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "子地_URI","子地_现代名称","子地_历史名称",
            "父地_URI","父地_现代名称","父级全称","匹配方式","结果"
        ])
        w.writeheader()
        for r in sum_rows:
            w.writerow(r)
    log(f"[CSV] 构建摘要：{CSV_SUMMARY}（{len(sum_rows)} 行）")

    # ========= 导出“层级链”CSV（现代名称链 + URI 链；历史名仅展示）=========
    places_all = [s for s, t in g.subject_objects(RDF.type) if isinstance(s, URIRef) and localname(t) == CLASS_PLACE]

    def first_lit(node: URIRef, key_local: str) -> str:
        for p, o in g.predicate_objects(node):
            if isinstance(o, Literal) and localname(p) == key_local:
                v = str(o).strip()
                if v: return v
        return ""

    def climb_chain(start: URIRef) -> Tuple[List[URIRef], List[str]]:
        chain_nodes: List[URIRef] = [start]
        chain_mods:  List[str]    = [first_lit(start, DP_MOD_NAME)]
        seen: Set[URIRef] = {start}
        cur = start
        while True:
            parents = [o for o in g.objects(cur, URIRef(next(iter(find_predicates_by_local(g, OP_SUB_PLACE)), None))) if isinstance(o, URIRef)]
            if not parents: break
            par = parents[0]
            if par in seen:  # 防环
                break
            seen.add(par)
            chain_nodes.append(par)
            chain_mods.append(first_lit(par, DP_MOD_NAME))
            cur = par
        return chain_nodes, chain_mods

    rows_chain: List[Dict[str,str]] = []
    p_sub = next(iter(find_predicates_by_local(g, OP_SUB_PLACE)), None)
    for pl in places_all:
        mod = first_lit(pl, DP_MOD_NAME)
        his = first_lit(pl, DP_HIS_NAME)  # 仅输出参考
        lvl = first_lit(pl, DP_ADMIN_LV_1) or first_lit(pl, DP_ADMIN_LV_2) or last_unit_suffix_from_full_name(mod or "")
        # 向上爬链
        # 这里直接用上面的 climb_chain（已基于 isSubPlaceOf）
        chain_nodes, chain_mods = [], []
        # 修正：调用时需传入 p_sub
        def climb_chain2(start: URIRef, p_sub_pred: URIRef):
            nodes = [start]; mods = [first_lit(start, DP_MOD_NAME)]
            seen = {start}; cur = start
            while True:
                parents = [o for o in g.objects(cur, p_sub_pred) if isinstance(o, URIRef)]
                if not parents: break
                par = parents[0]
                if par in seen: break
                seen.add(par); nodes.append(par); mods.append(first_lit(par, DP_MOD_NAME)); cur = par
            return nodes, mods

        if p_sub is not None:
            chain_nodes, chain_mods = climb_chain2(pl, p_sub)
        mods_clean = [m for m in chain_mods if m]
        uris_chain = [str(u) for u in chain_nodes]

        rows_chain.append({
            "Place_URI": str(pl),
            "现代名称": mod,
            "历史名称": his,  # 仅显示，不参与算法
            "起点层级": lvl,
            "链_现代名称": " → ".join(mods_clean),
            "顶级现代名称": (mods_clean[-1] if mods_clean else ""),
            "链_URI": " → ".join(uris_chain),
        })

    with CSV_CHAIN.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "Place_URI","现代名称","历史名称","起点层级",
            "链_现代名称","顶级现代名称","链_URI"
        ])
        w.writeheader()
        for r in rows_chain:
            w.writerow(r)
    log(f"[CSV] 层级链：{CSV_CHAIN}（{len(rows_chain)} 行）")

if __name__ == "__main__":
    main()
