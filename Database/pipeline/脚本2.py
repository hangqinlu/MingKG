# -*- coding: utf-8 -*-
"""
脚本2：JSON → Neo4j 导入（无交互；外部参数或环境变量注入）
- 严格沿用你给出的核心逻辑（Word 溯源 + TextProvenance + 节点 + 关系 + PropAssertion）
- 支持 JSON 数组 / 单对象 / JSONL / UTF-8(BOM)；失败时给出清晰诊断
- 不再调用 input()，通过命令行或环境变量获得所有路径与 Neo4j 连接信息
"""

import os
import io
import sys
import re
import json
import uuid
import hashlib
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
from collections import defaultdict

# ==== 依赖 ====
# pip install python-docx py2neo rapidfuzz
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

try:
    from rapidfuzz import fuzz
    HAS_RF = True
except Exception:
    import difflib
    HAS_RF = False

from py2neo import Graph

import argparse

# ===================== 参数解析（无交互） =====================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="脚本2：JSON→Neo4j 导入（外部配置，非交互）")

    ap.add_argument("--docx", dest="docx_path",
                    default=os.getenv("DOCX_PATH", "").strip(),
                    help="Word 溯源文件路径（DOCX_PATH）")

    ap.add_argument("--json", dest="json_file",
                    default=os.getenv("JSON_FILE", "").strip(),
                    help="输入 JSON/JSONL 路径（JSON_FILE）")

    ap.add_argument("--save-aug-json", dest="save_aug_json",
                    default=os.getenv("SAVE_AUG_JSON", "").strip(),
                    help="保存带 Text_source 的中间 JSON（可留空）")

    ap.add_argument("--neo4j-uri", dest="neo4j_uri",
                    default=os.getenv("NEO4J_URI", "bolt://localhost:7687").strip(),
                    help="Neo4j URI（NEO4J_URI）")

    ap.add_argument("--neo4j-user", dest="neo4j_user",
                    default=os.getenv("NEO4J_USER", "neo4j").strip(),
                    help="Neo4j 用户名（NEO4J_USER）")

    ap.add_argument("--neo4j-pwd", dest="neo4j_pwd",
                    default=os.getenv("NEO4J_PWD", "").strip(),
                    help="Neo4j 密码（NEO4J_PWD）")

    ap.add_argument("--neo4j-db", dest="neo4j_db",
                    default=os.getenv("NEO4J_DB", "").strip(),
                    help="Neo4j 数据库名（留空使用 Neo4j 默认 dbms.default_database）")

    return ap.parse_args()

_args = parse_args()
DOCX_PATH = _args.docx_path
JSON_FILE = _args.json_file
NEO4J_URI = _args.neo4j_uri
NEO4J_PWD = _args.neo4j_pwd
NEO4J_USER = _args.neo4j_user


# ===================== Label 规范 =====================
# ========= Label 规范 =========
LABEL_ALIAS = {
    "Official Position": "OfficialPosition",
    "OfficialPosition":  "OfficialPosition",
    "ImperialExam":      "ImperialExam",
    "Person":            "Person",
    "Place":             "Place",
}

def norm_label(label: str) -> str:
    return LABEL_ALIAS.get(label, label)

def clean_props(d: Dict[str, Any]) -> Dict[str, Any]:
    if not d:
        return {}
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}

# ========= A. Word 溯源 =========
def load_docx_sources(docx_path: str) -> List[Dict[str, str]]:
    """
    识别 Word 中“居中段落”为来源标题，直到下一个居中段落之前的段落合并为正文。
    返回 [{'title': '《xxx》', 'text': '整段正文...'}, ...]
    """
    doc = Document(docx_path)
    sources = []
    cur = None
    for p in doc.paragraphs:
        raw = (p.text or "").strip()
        if not raw:
            continue
        is_center = (p.alignment == WD_ALIGN_PARAGRAPH.CENTER)
        is_heading = (p.style and p.style.name and ("Title" in p.style.name or "Heading" in p.style.name))

        if is_center or is_heading:
            if cur and (cur['title'] or cur['paras']):
                cur['text'] = "\n".join(cur['paras'])
                sources.append({'title': cur['title'], 'text': cur['text']})
            cur = {'title': raw, 'paras': []}
        else:
            if cur is None:
                cur = {'title': '（未知来源）', 'paras': []}
            cur['paras'].append(raw)

    if cur and (cur['title'] or cur['paras']):
        cur['text'] = "\n".join(cur['paras'])
        sources.append({'title': cur['title'], 'text': cur['text']})
    return sources

def normalize(s: str) -> str:
    if not s:
        return ""
    # 最小化处理：小写 + 去空白（不要做太激进以免误伤短片段匹配）
    return "".join(s.lower().split())

def sim(a: str, b: str) -> float:
    if HAS_RF:
        # 兼顾部分摘取
        s1 = fuzz.partial_ratio(a, b)
        s2 = fuzz.token_set_ratio(a, b)
        return (s1 + s2) / 2
    else:
        return 100.0 * (difflib.SequenceMatcher(None, a, b).ratio())

def find_best_title_for_text(query_text: str, sources: List[Dict[str, str]]) -> Tuple[Optional[str], float]:
    """
    仅基于 Title+Text，返回最佳标题与分数（若低于阈值则返回 None）。
    """
    if not query_text:
        return None, 0.0
    qn = normalize(query_text)
    best = (None, 0.0)
    for s in sources:
        tn = normalize(s.get("text", ""))
        if not tn:
            continue
        score = sim(qn, tn)
        if score > best[1]:
            best = (s.get("title"), score)
    # 经验阈值：60 分以下不可信
    if best[1] < 60.0:
        return None, best[1]
    return best

def text_record_id(text: str) -> str:
    """稳定 record_id：对 Text 做 sha1，便于跨次运行复用（同一文本→同一 record_id）"""
    h = hashlib.sha1((text or "").encode("utf-8")).hexdigest()
    return f"REC-{h[:16]}"

# NEW —— 用于属性值规范化（断言去重键的一部分）
_punct_re = re.compile(r"[\s\u3000\u00A0·\.\-_:、，,。；;（）\(\)\[\]{}“”\"'‘’/\\]+")
def canon_val(s: Any) -> str:
    if s is None:
        return ""
    x = unicodedata.normalize("NFKC", str(s)).strip()
    x = _punct_re.sub(" ", x).strip().lower()
    return x

def load_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    # 兼容 JSONL
    if "\n" in text and text.lstrip().startswith("{") and "\n{" in text:
        records = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            records.append(json.loads(s))
        return records
    obj = json.loads(text)
    return obj if isinstance(obj, list) else [obj]

def augment_records_with_provenance(records: List[Dict[str, Any]], sources: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for rec in records:
        text = rec.get("Text") or ""
        best_title, score = find_best_title_for_text(text, sources)
        rec["Text_source"] = best_title  # 只存标题
        rec["record_id"]   = text_record_id(text)
        rec["record_confidence"] = round(score, 2)
        out.append(rec)
    return out

# ========= B. 导入 Neo4j（单一溯源实体 TextProvenance + PropAssertion） =========

def _key_exact(label: str, props: Dict[str, Any]) -> Tuple:
    return ("EXACT", norm_label(label), tuple(sorted(clean_props(props).items())))

def _key_person(props: Dict[str, Any]) -> Optional[Tuple]:
    n = clean_props(props).get("姓名")
    return ("PERSON", n) if n else None

def _key_place(props: Dict[str, Any]) -> Optional[Tuple]:
    h = clean_props(props).get("歷史名稱")
    return ("PLACE", h) if h else None

def _key_offpos(props: Dict[str, Any]) -> Optional[Tuple]:
    t = clean_props(props).get("官職名稱")
    return ("OFFPOS", t) if t else None

def _key_exam_full(props: Dict[str, Any]) -> Optional[Tuple]:
    p = clean_props(props)
    lvl, when = p.get("考試等級"), p.get("考試時間")
    if lvl and when:
        return ("EXAM_FULL", lvl, when)
    return None

def _key_exam_level(props: Dict[str, Any]) -> Optional[Tuple]:
    p = clean_props(props)
    lvl = p.get("考試等級")
    if lvl:
        return ("EXAM_LVL", lvl)
    return None

def build_record_node_index(rec: Dict[str, Any]) -> Dict[Tuple, str]:
    index: Dict[Tuple, str] = {}
    for n in (rec.get("nodes") or []):
        if (n or {}).get("type") != "node":
            continue
        label = norm_label(n.get("label"))
        props = clean_props(n.get("properties") or {})
        node_uid = str(uuid.uuid4())
        n["_node_uid"] = node_uid

        keys = set()
        keys.add(_key_exact(label, props))
        if label == "Person":
            k = _key_person(props)
            if k: keys.add(k)
        elif label == "Place":
            k = _key_place(props)
            if k: keys.add(k)
        elif label == "OfficialPosition":
            k = _key_offpos(props)
            if k: keys.add(k)
        elif label == "ImperialExam":
            k1 = _key_exam_full(props)
            k2 = _key_exam_level(props)
            if k1: keys.add(k1)
            if k2: keys.add(k2)

        for k in keys:
            if k is None:
                continue
            if k in index and index[k] != node_uid:
                print(f"[WARN] 本记录内键 {k} 命中多节点，已忽略后续重复。")
            else:
                index[k] = node_uid
    return index

def resolve_endpoint_uid(endpoint_map: Dict[str, Any], node_index: Dict[Tuple, str],
                         fallback_by_label: Dict[str, List[str]]) -> Optional[str]:
    label = norm_label(endpoint_map.get("label"))
    em = {k: v for k, v in endpoint_map.items() if k != "label"}
    em_clean = clean_props(em)

    uid = node_index.get(_key_exact(label, em_clean))
    if uid:
        return uid

    if label == "Person":
        uid = node_index.get(_key_person(em_clean))
    elif label == "Place":
        uid = node_index.get(_key_place(em_clean))
    elif label == "OfficialPosition":
        uid = node_index.get(_key_offpos(em_clean))
    elif label == "ImperialExam":
        uid = node_index.get(_key_exam_full(em_clean)) or node_index.get(_key_exam_level(em_clean))
    else:
        uid = None
    if uid:
        return uid

    candidates = fallback_by_label.get(label, [])
    if len(candidates) >= 1:
        return candidates[0]
    return None

# ---------- Cypher（统一 TextProvenance） ----------
def create_textprov_tx(tx, record_id: str, text: str, text_source_title: Optional[str], rec_conf: float) -> None:
    """
    MERGE 单一溯源实体：(:TextProvenance {record_id})
    """
    tx.run("""
        MERGE (p:TextProvenance {record_id:$rid})
        ON CREATE SET
          p.Text_source = $title,
          p.record_confidence = $conf,
          p.Text = $text,
          p.created_at = timestamp()
        SET p.last_seen = timestamp()
        """, rid=record_id, title=text_source_title or "", conf=float(rec_conf or 0.0), text=text or "")

def create_node_tx(tx, label: str, props: Dict[str, Any], batch_id: str, node_uid: str, record_id: str) -> None:
    label = norm_label(label)
    all_props = clean_props(props)
    all_props["batch_id"] = batch_id
    all_props["node_uid"] = node_uid
    all_props["record_id"] = record_id
    tx.run(f"CREATE (n:`{label}`) SET n += $props", props=all_props)

def link_textprov_contains_tx(tx, label: str, node_uid: str, record_id: str, batch_id: str) -> None:
    tx.run(f"""
        MATCH (p:TextProvenance {{record_id:$rid}})
        MATCH (n:`{norm_label(label)}` {{record_id:$rid, batch_id:$bid, node_uid:$uid}})
        MERGE (p)-[:CONTAINS]->(n)
        """, rid=record_id, bid=batch_id, uid=node_uid)

def create_rel_tx(tx, rel: Dict[str, Any], batch_id: str,
                  node_index: Dict[Tuple, str],
                  fallback_by_label: Dict[str, List[str]],
                  record_id: str) -> None:
    r_type = rel.get("label")
    start  = rel.get("start", {}) or {}
    end    = rel.get("end", {}) or {}
    rprops = clean_props(rel.get("properties", {}))
    s_label = norm_label(start.get("label"))
    e_label = norm_label(end.get("label"))

    s_uid = resolve_endpoint_uid(start, node_index, fallback_by_label)
    e_uid = resolve_endpoint_uid(end,   node_index, fallback_by_label)
    if not s_uid or not e_uid:
        print(f"[WARN] 跳过关系 '{r_type}'：端点未解析 (s_uid={s_uid}, e_uid={e_uid})")
        return

    rprops["record_id"] = record_id
    tx.run(f"""
        MATCH (a:`{s_label}` {{batch_id:$bid, node_uid:$suid}})
        MATCH (b:`{e_label}` {{batch_id:$bid, node_uid:$euid}})
        CREATE (a)-[r:`{r_type}`]->(b)
        SET r += $rprops
        """, bid=batch_id, suid=s_uid, euid=e_uid, rprops=rprops)

# NEW —— 属性断言：为“节点属性”生成 PropAssertion，并连回实体与 TextProvenance
def create_prop_assertion_tx(tx, label: str, node_uid: str, batch_id: str,
                             record_id: str, prop_key: str, prop_val) -> None:
    if prop_val in (None, "", []):
        return
    val_str = str(prop_val)
    val_norm = canon_val(val_str)

    # 用 (node_uid, record_id, prop, value_norm) 幂等去重
    tx.run("""
        MERGE (a:PropAssertion {
            node_uid:$uid, record_id:$rid, prop:$prop, value_norm:$valn
        })
        ON CREATE SET a.value=$val, a.batch_id=$bid, a.created_at=timestamp()
        SET a.last_seen=timestamp()
        """, uid=node_uid, rid=record_id, prop=prop_key, valn=val_norm, val=val_str, bid=batch_id)

    # 连接 ABOUT 与 DERIVED_FROM 到 TextProvenance
    tx.run(f"""
        MATCH (n:`{norm_label(label)}` {{batch_id:$bid, node_uid:$uid}})
        MATCH (p:TextProvenance {{record_id:$rid}})
        MATCH (a:PropAssertion {{node_uid:$uid, record_id:$rid, prop:$prop, value_norm:$valn}})
        MERGE (a)-[:ABOUT]->(n)
        MERGE (a)-[:DERIVED_FROM]->(p)
        """, bid=batch_id, uid=node_uid, rid=record_id, prop=prop_key, valn=val_norm)

def import_record(graph: Graph, rec: Dict[str, Any]) -> None:
    batch_id = str(uuid.uuid4())
    # 溯源信息（来自 A 步）
    record_id = rec.get("record_id") or text_record_id(rec.get("Text",""))
    text_source_title = rec.get("Text_source")
    rec_conf = float(rec.get("record_confidence") or 0.0)

    node_index = build_record_node_index(rec)
    fallback_by_label: Dict[str, List[str]] = defaultdict(list)
    for n in (rec.get("nodes") or []):
        if (n or {}).get("type") != "node":
            continue
        label = norm_label(n.get("label"))
        uid = n.get("_node_uid")
        if uid:
            fallback_by_label[label].append(uid)

    tx = graph.begin()
    try:
        # 0) MERGE 单一溯源实体 TextProvenance
        create_textprov_tx(tx, record_id, rec.get("Text",""), text_source_title, rec_conf)

        # 1) 创建节点
        for n in (rec.get("nodes") or []):
            if (n or {}).get("type") != "node":
                continue
            label = n.get("label")
            props = n.get("properties") or {}
            node_uid = n.get("_node_uid") or str(uuid.uuid4())
            create_node_tx(tx, label, props, batch_id, node_uid, record_id)
            link_textprov_contains_tx(tx, label, node_uid, record_id, batch_id)

        # 2) 创建关系（只连本记录内）
        for r in (rec.get("relationships") or []):
            if (r or {}).get("type") != "relationship":
                continue
            create_rel_tx(tx, r, batch_id, node_index, fallback_by_label, record_id)

        # 3) NEW —— 属性断言（为每个节点属性写 PropAssertion）
        total_asserts = 0
        for n in (rec.get("nodes") or []):
            if (n or {}).get("type") != "node":
                continue
            label = norm_label(n.get("label"))
            node_uid = n.get("_node_uid")
            props = clean_props(n.get("properties") or {})
            for k, v in props.items():
                create_prop_assertion_tx(tx, label, node_uid, batch_id, record_id, k, v)
                total_asserts += 1

        tx.commit()
        # 适度输出，便于你追踪
        print(f"[OK] record_id={record_id} | nodes={len(rec.get('nodes') or [])} "
              f"| rels={len(rec.get('relationships') or [])} | prop_asserts={total_asserts}")
    except Exception as e:
        try:
            tx.rollback()
        except Exception:
            pass
        raise e

# =================== 主流程 ===================

def main():
    # A) Word 溯源
    srcs = load_docx_sources(DOCX_PATH)
    if not srcs:
        print("[WARN] Word 中未识别到来源（居中段落）")
    records = load_records(JSON_FILE)
    records = augment_records_with_provenance(records, srcs)

    # 可选：保存带 Text_source 的中间文件

    # B) 导入 Neo4j
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    print(f"[INFO] 读取到 {len(records)} 条记录，开始导入 Neo4j（TextProvenance + 属性断言）...")

    for i, rec in enumerate(records, 1):
        import_record(graph, rec)
        if i % 50 == 0 or i == len(records):
            print(f"[INFO] 已导入 {i}/{len(records)}")

    print("[DONE] 全部导入完成。")

if __name__ == "__main__":
    main()
