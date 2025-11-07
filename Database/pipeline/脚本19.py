# -*- coding: utf-8 -*-
"""
关系类型批量规范（仅事件的数据属性，rdflib 版，无 owlready2 依赖）+ 逐人日志（修复版）
- 人物识别：满足其有 (p, socialRelationEvent, ?evt) 出边，或拥有数据属性“姓名”，或 rdf:type 的本地名为 Person/人物。
- 事件识别：由 (p, socialRelationEvent, evt) 找到的对象节点均视为事件（不强依赖 rdf:type）
- 关系键发现：按“本地名”匹配，自动涵盖多种写法（关系类型/關係類型/RelationType/relationshipType/relation_type 等）
- 核心修复：
  * 读取原 Literal（含语言标签），统一删除 (evt, pred, Literal(*)) 后再写回，避免删不掉的情况
  * 旧值与映射键统一做 NFKC+strip 规范化，提高匹配命中率
  * 写回时沿用第一条原值的语言标签（若存在）
- 仅改事件节点上的 **数据属性**；不碰 PropAssertion
- 导出：TTL + NT，UTF-8
"""

import logging
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from rdflib import Graph, URIRef, RDF, Literal

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
OUT_TTL = str(Path(SRC).with_name("ontology_people_cleaned_reltype_fixed_with_logs.ttl"))
OUT_NT  = str(Path(SRC).with_name("ontology_people_cleaned_reltype_fixed_with_logs.nt"))

# ========== 日志 ==========
LOG_LEVEL = logging.INFO  # 可改为 logging.DEBUG
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="~%H:%M:%S",
)
log = logging.getLogger("RelTypeFix-rdflib")

# ========== 工具 ==========
def localname(u: Union[URIRef, str]) -> str:
    s = str(u or "")
    for sep in ("#", "/", ":"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s

def norm_txt(s: Optional[str]) -> str:
    if s is None: return ""
    return unicodedata.normalize("NFKC", str(s)).strip()

def guess_format_by_suffix(path: str) -> str:
    suf = Path(path).suffix.lower()
    return {
        ".nt": "nt",
        ".ttl": "turtle",
        ".rdf": "xml",
        ".owl": "xml",
        ".xml": "xml",
    }.get(suf, "turtle")  # 默认尝试 turtle

def load_graph_any(path: str) -> Graph:
    g = Graph()
    fmt = guess_format_by_suffix(path)
    try:
        g.parse(path, format=fmt)
        return g
    except Exception:
        # 两步兜底：先尝试 turtle，再尝试 nt
        try:
            g = Graph(); g.parse(path, format="turtle"); return g
        except Exception:
            g = Graph(); g.parse(path, format="nt"); return g

def predicates_by_local(g: Graph, want_locals: Set[str]) -> Set[URIRef]:
    out: Set[URIRef] = set()
    for _, p, _ in g.triples((None, None, None)):
        lp = localname(p)
        if lp in want_locals:
            out.add(p)
    return out

def all_literal_objs_of(g: Graph, s: URIRef, p: URIRef) -> List[Literal]:
    """返回 (s,p,Literal) 的所有 Literal 对象（保留语言/数据类型）"""
    vals: List[Literal] = []
    for o in g.objects(s, p):
        if isinstance(o, Literal):
            vals.append(o)
    return vals

def literal_text(o: Literal) -> str:
    """将 Literal 转为规范化文本用于匹配映射"""
    try:
        return norm_txt(str(o))
    except Exception:
        return ""

def person_name(g: Graph, person: URIRef) -> str:
    # 取“姓名”本地名的数据属性；若无则尝试 label/name 常用键
    name_keys = {"姓名", "name", "label", "rdfs_label", "标题", "title"}
    ps = predicates_by_local(g, name_keys)
    for p in ps:
        for o in g.objects(person, p):
            if isinstance(o, Literal):
                t = norm_txt(str(o))
                if t:
                    return t
    # 兜底：IRI 尾巴
    return localname(person)

def get_person_candidates(g: Graph) -> List[URIRef]:
    persons: Set[URIRef] = set()

    # 1) 有 socialRelationEvent 出边的主体
    sre_keys = {"socialRelationEvent"}
    sre_preds = predicates_by_local(g, sre_keys)
    for s, p, o in g.triples((None, None, None)):
        if p in sre_preds and isinstance(o, URIRef):
            persons.add(s)

    # 2) 有“姓名”数据属性者也视作人物
    name_keys = {"姓名"}
    name_preds = predicates_by_local(g, name_keys)
    for s, p, o in g.triples((None, None, None)):
        if p in name_preds and isinstance(o, Literal):
            persons.add(s)

    # 3) rdf:type 的本地名为 Person/人物 的
    type_candidates = {"Person", "人物"}
    for s, _, t in g.triples((None, RDF.type, None)):
        if localname(t) in type_candidates:
            persons.add(s)

    return sorted(persons, key=lambda x: str(x))

def events_of_person(g: Graph, person: URIRef) -> List[URIRef]:
    sre_preds = predicates_by_local(g, {"socialRelationEvent"})
    evts: List[URIRef] = []
    seen: Set[URIRef] = set()
    for p in sre_preds:
        for o in g.objects(person, p):
            if isinstance(o, URIRef) and o not in seen:
                seen.add(o); evts.append(o)
    return evts

def replace_event_reltype_values(
    g: Graph,
    evt: URIRef,
    rel_pred: URIRef,
    mapping_norm: Dict[str, str]
) -> Tuple[bool, List[str], List[str], str]:
    """
    对事件 evt 的某个“关系类型”谓词 rel_pred：
      - 读取所有 Literal（保留语言标签）
      - 统一规范化文本后做映射；未命中的保留原文本
      - 去重保序
      - 若发生变化：删除该键的**全部 Literal**，再写回新值（沿用第一条原值的语言标签）
    返回：(是否变更, 旧值文本列表, 新值文本列表, 备注原因)
    """
    old_objs = all_literal_objs_of(g, evt, rel_pred)
    if not old_objs:
        return (False, [], [], "无文字值")

    old_vals_text = [literal_text(o) for o in old_objs]

    # 规范化 + 映射
    new_vals_raw: List[str] = []
    for t in old_vals_text:
        t_norm = norm_txt(t)
        new_vals_raw.append(mapping_norm.get(t_norm, t))  # 未命中保留原文本

    # 去重保序
    new_vals: List[str] = []
    for v in new_vals_raw:
        if v not in new_vals:
            new_vals.append(v)

    if new_vals == old_vals_text:
        return (False, old_vals_text, new_vals, "无变化（值未命中映射或与映射结果一致）")

    # —— 执行删除（删除该键下的所有 Literal 值）——
    for o in list(g.objects(evt, rel_pred)):
        if isinstance(o, Literal):
            g.remove((evt, rel_pred, o))

    # —— 写回新值（沿用第一条原值的语言标签；否则无语言标签）——
    lang = old_objs[0].language if old_objs and isinstance(old_objs[0], Literal) else None
    for v in new_vals:
        if lang:
            g.add((evt, rel_pred, Literal(v, lang=lang)))
        else:
            g.add((evt, rel_pred, Literal(v)))

    return (True, old_vals_text, new_vals, "已替换并写回")

# ========== 规则 & 键 ==========
# 映射表（写你需要的规范化规则）；此处会做 NFKC+strip 规范化后比较
REL_MAP = {
    "曾孙": "曾祖孙",
    "曾孫": "曾祖孙",
    "子":   "父子",
    "兄":   "兄弟",
    "弟":   "兄弟",
    "孙":   "祖孙",
    "孫":   "祖孙",
}
# 支持更广的键本地名（自动发现）
REL_KEYS_LOCAL_CANON = {
    "关系类型", "關係類型", "RelationType", "relationshipType", "relationType",
    "relation_type", "rel_type", "relType"
}

def build_normalized_map(raw_map: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in raw_map.items():
        out[norm_txt(k)] = norm_txt(v)
    return out

# ========== 主流程 ==========
def main():
    g = load_graph_any(SRC)
    log.info("[OK] 载入：%s  三元组≈%d", SRC, len(g))

    persons = get_person_candidates(g)
    if not persons:
        log.warning("[WARN] 没找到疑似人物的主体（请确认文件是否包含人物数据）")

    # 发现所有候选键中实际出现在图里的谓词
    reltype_preds = predicates_by_local(g, REL_KEYS_LOCAL_CANON)
    if not reltype_preds:
        log.warning("[WARN] 未在图中发现任何本地名为 %r 的谓词，可能文件里没有这些键。", list(REL_KEYS_LOCAL_CANON))

    MAP_NORM = build_normalized_map(REL_MAP)

    scan_evt_keys = 0
    touch_events  = 0
    change_cnt    = 0
    skipped_no_key = 0

    for idx, p in enumerate(persons, 1):
        pname = person_name(g, p)
        evts = events_of_person(g, p)
        log.info("—— 人物(%d/%d)：%s ｜ 关系事件=%d", idx, len(persons), pname, len(evts))

        for evt in evts:
            if not reltype_preds:
                skipped_no_key += 1
                log.debug("    [SKIP] 事件[%s]：图中未发现关系类型键（本地名集合：%r）",
                          localname(evt), REL_KEYS_LOCAL_CANON)
                continue

            changed_this_evt = False
            had_any_key_vals = False

            for rp in reltype_preds:
                old_objs = all_literal_objs_of(g, evt, rp)
                if old_objs:
                    had_any_key_vals = True
                changed, old_vals, new_vals, reason = replace_event_reltype_values(g, evt, rp, MAP_NORM)
                if old_vals:
                    scan_evt_keys += 1
                if changed:
                    change_cnt += 1
                    changed_this_evt = True
                    log.info("    [FIX] 事件[%s] 键[%s]：%r → %r（%s）",
                             localname(evt), localname(rp), old_vals, new_vals, reason)
                else:
                    # 仅在该键存在但无变化时打印
                    if old_objs:
                        log.debug("    [KEEP] 事件[%s] 键[%s]：%r（%s）",
                                  localname(evt), localname(rp), old_vals, reason)

            if changed_this_evt:
                touch_events += 1
            else:
                # 没有任何键值（或都无变化）也记录一下，便于排查
                if not had_any_key_vals:
                    log.debug("    [NOTE] 事件[%s]：未找到任何关系类型值", localname(evt))

    log.info("[SUMMARY] 扫描键次数=%d ｜ 涉及事件(有变更)=%d ｜ 修改次数=%d ｜ 无键跳过=%d",
             scan_evt_keys, touch_events, change_cnt, skipped_no_key)

    # 导出
    try:
        g.serialize(destination=OUT_TTL, format="turtle")
        g.serialize(destination=OUT_NT,  format="nt")
        log.info("[SAVED] %s", OUT_TTL)
        log.info("[SAVED] %s", OUT_NT)
    except Exception as e:
        log.warning("[WARN] 导出失败：%s", e)

    log.info("[DONE]")

if __name__ == "__main__":
    main()
