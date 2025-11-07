# -*- coding: utf-8 -*-
"""
基于“现代名称”的地点事件消歧与合并（同类内合并版）
- 范围：BirthEvent / OfficeAtPlaceEvent / TaskEvent 三类事件
- 规则：对同一人物 & 同一事件类别（严格同类）
    * 若两个事件的地点“现代名称”相同 → 合并
    * 若一个“现代名称”是另一个的真包含（A in B 或 B in A）→ 合并到更具体（更长/更多行政词段）的那条
- 幸存者选择：信息量评分（文字属性数 + 边数）最高者，并列随机
- 合并：迁移所有非 rdf:type 的边与文字属性到幸存者；删除败者
- 输出：CSV（人名、关系类型、地点历史名称、地点现代名称），以及合并后的 .nt

依赖：rdflib
pip install rdflib
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from rdflib import Graph, URIRef, RDF, Literal
import re
import random
import csv
import time

# ======= 路径配置 =======
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
NT_OUT = SRC.with_name(f"{SRC.stem}_merged_by_modname_sameclass.nt")
CSV_OUT = SRC.with_name(f"person_place_after_merge_by_modname_sameclass.csv")

# ======= 名称（局部名匹配，鲁棒） =======
CLASS_PERSON = "Person"
CLASS_PLACE  = "Place"
CLASS_BIRTH  = "BirthEvent"
CLASS_OFFICE = "OfficeAtPlaceEvent"
CLASS_TASK   = "TaskEvent"

OP_HAS_PLACE = "hasPlace"

PERSON_NAME_KEYS = ["姓名", "name", "label", "rdfs_label", "标题", "title"]
PLACE_HIS_NAME   = "历史名称"
PLACE_MOD_NAME   = "现代名称"

# ======= 工具函数 =======
def localname(u: URIRef) -> str:
    s = str(u)
    for sep in ["#", "/", ":"]:
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s

def is_instance_of(g: Graph, inst: URIRef, class_local: str) -> bool:
    for t in g.objects(inst, RDF.type):
        if localname(t) == class_local:
            return True
    return False

def find_instances(g: Graph, class_local: str) -> List[URIRef]:
    out = []
    for s, t in g.subject_objects(RDF.type):
        if isinstance(s, URIRef) and localname(t) == class_local:
            out.append(s)
    return out

def get_literals(g: Graph, node: URIRef) -> Dict[str, List[str]]:
    res: Dict[str, List[str]] = {}
    for p, o in g.predicate_objects(node):
        if isinstance(o, Literal):
            k = localname(p)
            res.setdefault(k, []).append(str(o))
    return res

def get_first_display(props: Dict[str, List[str]], keys: List[str]) -> str:
    for k in keys:
        if k in props and props[k]:
            return props[k][0]
    for vs in props.values():
        for v in vs:
            if v.strip():
                return v
    return ""

def get_place_for_event(g: Graph, evt: URIRef) -> Optional[URIRef]:
    # 优先 hasPlace
    for p, o in g.predicate_objects(evt):
        if isinstance(o, URIRef) and localname(p) == OP_HAS_PLACE and is_instance_of(g, o, CLASS_PLACE):
            return o
    # 回退：任一指向 Place 的边
    for p, o in g.predicate_objects(evt):
        if isinstance(o, URIRef) and is_instance_of(g, o, CLASS_PLACE):
            return o
    return None

def score_event_info(g: Graph, evt: URIRef) -> int:
    """信息量评分：文字属性数 + 出/入边数（不含 rdf:type）；文字额外+1"""
    score = 0
    for p, o in g.predicate_objects(evt):
        if localname(p) == "type":
            continue
        score += 1
        if isinstance(o, Literal):
            score += 1
    for s, p in g.subject_predicates(evt):
        if localname(p) == "type":
            continue
        score += 1
    return score

def copy_literal_props(g: Graph, src: URIRef, dst: URIRef):
    """把 src 的文字属性合并到 dst（不复制 rdf:type）"""
    to_add = []
    for p, o in g.predicate_objects(src):
        if isinstance(o, Literal) and localname(p) != "type":
            exists = any((p2 == p and str(o2) == str(o)) for p2, o2 in g.predicate_objects(dst))
            if not exists:
                to_add.append((dst, p, o))
    for t in to_add:
        g.add(t)

def rewire_node(g: Graph, loser: URIRef, winner: URIRef):
    """把图中所有指向/来自 loser 的边改指向 winner；不迁移 rdf:type"""
    to_add, to_del = [], []

    for p, o in list(g.predicate_objects(loser)):
        if localname(p) == "type":
            to_del.append((loser, p, o))
            continue
        to_add.append((winner, p, o))
        to_del.append((loser, p, o))

    for s, p in list(g.subject_predicates(loser)):
        to_add.append((s, p, winner))
        to_del.append((s, p, loser))

    for t in to_del:
        g.remove(t)
    for t in to_add:
        g.add(t)

def relation_label_by_event(g: Graph, evt: URIRef) -> str:
    if is_instance_of(g, evt, CLASS_BIRTH):
        return "生"
    if is_instance_of(g, evt, CLASS_OFFICE):
        return "职任"
    if is_instance_of(g, evt, CLASS_TASK):
        return "任务执行"
    return "事件"

# —— 现代名称归一化，用于“相等/包含”判断（去空白、破折号、中文顿号等）
_punct_re = re.compile(r"[\s·・,，、;；—\-–——]+")
def norm_modname(s: str) -> str:
    s2 = _punct_re.sub("", (s or "").strip())
    return s2

def admin_segments(s: str) -> List[str]:
    """粗略按行政后缀切分，段数越多视为越具体"""
    suffixes = ["自治区", "自治州", "自治县", "地区", "盟",
                "省", "市", "州", "县", "区", "旗", "乡", "镇", "街道", "村", "组"]
    segs = []
    buf = ""
    ns = s or ""
    for ch in ns:
        buf += ch
        for suf in suffixes:
            if buf.endswith(suf):
                segs.append(buf)
                buf = ""
                break
    if buf:
        segs.append(buf)
    return [x for x in segs if x]

def specificity_key(modname_raw: str) -> Tuple[int, int]:
    """用于比较谁更具体：优先段数，其次长度"""
    m = modname_raw or ""
    segs = admin_segments(m)
    return (len(segs), len(m))

# ======= 主流程 =======
def main():
    random.seed(42)  # 并列随机的可复现性

    if not SRC.exists():
        raise FileNotFoundError(f"找不到输入本体：{SRC}")

    print(f"[LOAD] 读取：{SRC}")
    g = Graph()
    g.parse(location=str(SRC), format="nt")

    persons = find_instances(g, CLASS_PERSON)
    print(f"[INFO] 人物实例：{len(persons)}")

    total_merged = 0
    log_merge_by_person: List[str] = []

    # ========== 辅助：取事件类别标签 ==========
    def evt_class_key(evt: URIRef) -> Optional[str]:
        if is_instance_of(g, evt, CLASS_BIRTH):  return CLASS_BIRTH
        if is_instance_of(g, evt, CLASS_OFFICE): return CLASS_OFFICE
        if is_instance_of(g, evt, CLASS_TASK):   return CLASS_TASK
        return None

    for person in persons:
        # —— 先收集此人的三类事件，**分开三个桶**（同类内合并）
        events_by_class: Dict[str, Set[URIRef]] = {
            CLASS_BIRTH: set(), CLASS_OFFICE: set(), CLASS_TASK: set()
        }

        # 正向边（人物 → 事件）
        for p, e in g.predicate_objects(person):
            if isinstance(e, URIRef):
                k = evt_class_key(e)
                if k: events_by_class[k].add(e)
        # 反向边（事件 → 人物）
        for e, p in g.subject_predicates(person):
            if isinstance(e, URIRef):
                k = evt_class_key(e)
                if k: events_by_class[k].add(e)

        # —— 在每个“同类桶”内做现代名称相等/包含聚簇与合并
        for cls_name, evts_sameclass in events_by_class.items():
            if not evts_sameclass:
                continue

            # 收集：event → (place_uri, modname_raw, modname_norm)
            evt2place: Dict[URIRef, Tuple[Optional[URIRef], str, str]] = {}
            for evt in evts_sameclass:
                pl = get_place_for_event(g, evt)
                if pl is None:
                    continue
                pl_props = get_literals(g, pl)
                mod_raw = (pl_props.get(PLACE_MOD_NAME, [""])[0]).strip() if PLACE_MOD_NAME in pl_props else ""
                if not mod_raw:
                    # 没有现代名称的地点，不参与“现代名称消歧”
                    continue
                evt2place[evt] = (pl, mod_raw, norm_modname(mod_raw))

            if not evt2place:
                continue

            # 构造“现代名称相等/包含”的聚簇（**同类内部**）
            evts_sorted = sorted(evt2place.keys(), key=lambda e: specificity_key(evt2place[e][1]), reverse=True)

            used: Set[URIRef] = set()
            clusters: List[List[URIRef]] = []

            for e in evts_sorted:
                if e in used: continue
                _, mraw_e, _ = evt2place[e]
                cluster = [e]
                used.add(e)
                ne = norm_modname(mraw_e)
                for f in evts_sorted:
                    if f in used: continue
                    _, mraw_f, _ = evt2place[f]
                    nf = norm_modname(mraw_f)
                    if ne == nf or (ne in nf) or (nf in ne):
                        cluster.append(f)
                        used.add(f)
                clusters.append(cluster)

            # —— 对每个聚簇：选幸存者并合并（同类）
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                # 先按“更具体（段数、长度）”排序，再按信息量评分；并列随机
                cluster_sorted = sorted(
                    cluster,
                    key=lambda e: (specificity_key(evt2place[e][1]), score_event_info(g, e)),
                    reverse=True
                )
                top_spec = specificity_key(evt2place[cluster_sorted[0]][1])
                top_score = score_event_info(g, cluster_sorted[0])
                bests = [e for e in cluster_sorted
                         if specificity_key(evt2place[e][1]) == top_spec and score_event_info(g, e) == top_score]
                winner = random.choice(bests)
                losers = [e for e in cluster if e != winner]

                # 仅同类聚簇才会走到这里，因此不再做类别检查
                for loser in losers:
                    copy_literal_props(g, loser, winner)
                    rewire_node(g, loser, winner)
                    # 清理 loser 遗留
                    for t in list(g.triples((loser, None, None))):
                        g.remove(t)
                    for t in list(g.triples((None, None, loser))):
                        g.remove(t)
                    total_merged += 1

                # 调试日志（简洁）
                pid = str(person)
                wr_mod = evt2place[winner][1]
                losers_mods = [evt2place[l][1] for l in losers]
                log_merge_by_person.append(
                    f"[MERGE:{cls_name}] person={pid} winner_mod='{wr_mod}' losers_mods={losers_mods} size={1+len(losers)}"
                )

    print(f"[MERGE] 基于现代名称的地点事件消歧与合并完成（同类内）：{total_merged} 次")
    # （如需详细日志，可写文件）
    # LOG_PATH = NT_IN.with_name(f"{NT_IN.stem}_merge_log_sameclass.txt")
    # Path(LOG_PATH).write_text("\n".join(log_merge_by_person), encoding="utf-8")
    # print(f"[LOG] {LOG_PATH}")

    # —— 保存新的 .nt
    g.serialize(destination=str(NT_OUT), format="nt")
    print(f"[SAVE] 已写入：{NT_OUT}")

    # —— 导出 CSV：仅 人名、关系类型、地点历史名称、地点现代名称
    rows: List[Dict[str, str]] = []
    persons = find_instances(g, CLASS_PERSON)

    seen: Set[Tuple[URIRef, URIRef, URIRef]] = set()
    for person in persons:
        person_props = get_literals(g, person)
        person_name  = get_first_display(person_props, PERSON_NAME_KEYS)

        # 三类事件（双向）—— 这里只是导出，不会合并
        person_events: Set[URIRef] = set()
        for p, e in g.predicate_objects(person):
            if isinstance(e, URIRef) and (
                is_instance_of(g, e, CLASS_BIRTH) or
                is_instance_of(g, e, CLASS_OFFICE) or
                is_instance_of(g, e, CLASS_TASK)
            ):
                person_events.add(e)
        for e, p in g.subject_predicates(person):
            if isinstance(e, URIRef) and (
                is_instance_of(g, e, CLASS_BIRTH) or
                is_instance_of(g, e, CLASS_OFFICE) or
                is_instance_of(g, e, CLASS_TASK)
            ):
                person_events.add(e)

        for evt in person_events:
            place = get_place_for_event(g, evt)
            if place is None:
                continue
            key = (person, evt, place)
            if key in seen:
                continue
            seen.add(key)

            place_props = get_literals(g, place)
            his_name = place_props.get(PLACE_HIS_NAME, [""])[0] if PLACE_HIS_NAME in place_props else ""
            mod_name = place_props.get(PLACE_MOD_NAME, [""])[0] if PLACE_MOD_NAME in place_props else ""
            rel = relation_label_by_event(g, evt)

            rows.append({
                "人名": person_name,
                "关系类型": rel,
                "地点_历史名称": his_name,
                "地点_现代名称": mod_name,
            })

    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["人名", "关系类型", "地点_历史名称", "地点_现代名称"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[CSV] 已输出：{CSV_OUT}（{len(rows)} 行）")

if __name__ == "__main__":
    main()
