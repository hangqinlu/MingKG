# -*- coding: utf-8 -*-
"""
学术专长标准化（纯 rdflib 版；必清空→再写回，旧值必删除）
目标标准集合：书经 / 易经 / 春秋 / 诗经 / 礼记
- 支持补齐后缀：春→春秋，易/周易→易经，诗→诗经，礼→礼记，书/尚书→书经；含繁体
- 支持一值多项分隔：、，,；;/｜| 和空格
- 每人先清空旧值（删除原三元组），再整体写回规范值；规范为空则保持清空
- 只处理 Person 的数据属性：学术专长（不碰 PropAssertion）
- 不使用 owlready2；直接 rdflib 读/写 NT/TTL
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
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


OUT_TTL = str(Path(SRC).with_name("ontology_academic_fixed.ttl"))
OUT_NT  = str(Path(SRC).with_name("ontology_academic_fixed.nt"))

# ===== 读入 =====
p = Path(SRC)
fmt = {".nt":"nt", ".ttl":"turtle", ".rdf":"xml", ".owl":"xml", ".xml":"xml"}.get(p.suffix.lower())
if fmt is None:
    raise RuntimeError(f"不支持的本体后缀：{p.suffix}")

g = Graph()
g.parse(str(p), format=fmt)
print(f"[OK] 加载：{SRC}  三元组≈{len(g)}")

# ===== 小工具（纯 rdflib）=====
def localname(u: URIRef) -> str:
    s = str(u)
    for sep in ("#", "/", ":"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s

def class_uris_by_localnames(g: Graph, wants: Set[str]) -> Set[URIRef]:
    """返回所有本地名在 wants 内的类 URI（遍历 rdf:type 的宾语）"""
    out: Set[URIRef] = set()
    for _, t in g.subject_objects(RDF.type):
        if isinstance(t, URIRef) and localname(t) in wants:
            out.add(t)
    # 也从 RDFS/OWL 声明里捞（可选，稳妥一些）
    for t in g.all_nodes():
        if isinstance(t, URIRef) and localname(t) in wants:
            out.add(t)
    return out

def instances_of_classes(g: Graph, class_uris: Set[URIRef]) -> List[URIRef]:
    insts = []
    for s, t in g.subject_objects(RDF.type):
        if isinstance(s, URIRef) and isinstance(t, URIRef) and t in class_uris:
            insts.append(s)
    return insts

def predicate_uris_by_local(g: Graph, subj: Optional[URIRef], pred_local: str) -> Set[URIRef]:
    """在全图（或限定主体）中查找本地名=pred_local 的谓词 URI"""
    preds: Set[URIRef] = set()
    if subj is None:
        it = g.triples((None, None, None))
    else:
        it = g.triples((subj, None, None))
    for s, p, o in it:
        if isinstance(p, URIRef) and localname(p) == pred_local:
            preds.add(p)
    return preds

def get_data_prop_values(g: Graph, s: URIRef, prop_local: str) -> List[str]:
    vals = []
    # 该主体下，所有本地名=prop_local 的谓词
    preds = predicate_uris_by_local(g, s, prop_local)
    for p in preds:
        for o in g.objects(s, p):
            if isinstance(o, Literal):
                txt = str(o).strip()
                if txt:
                    vals.append(txt)
    return vals

def clear_data_prop(g: Graph, s: URIRef, prop_local: str):
    """删除该主体上本地名=prop_local 的所有数据属性三元组"""
    to_del = []
    preds = predicate_uris_by_local(g, s, prop_local)
    for p in preds:
        for o in g.objects(s, p):
            if isinstance(o, Literal):
                to_del.append((s, p, o))
    for t in to_del:
        g.remove(t)

def add_data_prop_values(g: Graph, s: URIRef, prop_local: str, values: List[str]):
    """把一组值追加到 s 的某个本地名数据属性上：
       - 首选：如果图中已有某个命名空间下的该属性，沿用那个谓词 URI
       - 否则：创建一个匿名/临时命名空间的同名谓词（不建议，但可兜底）
    """
    if not values:
        return
    preds = list(predicate_uris_by_local(g, s, prop_local))
    if preds:
        p_use = preds[0]
    else:
        # 兜底：复用图中其他主体已经用过的同名谓词
        global_preds = list(predicate_uris_by_local(g, None, prop_local))
        if global_preds:
            p_use = list(global_preds)[0]
        else:
            # 再兜底：构造一个匿名谓词（尽量避免）
            p_use = URIRef(f"http://anonymous#{prop_local}")
    for v in values:
        g.add((s, p_use, Literal(v)))

def get_first_display(g: Graph, s: URIRef, keys_local: List[str]) -> str:
    """按一串候选本地名取第一个非空的显示值（数据属性）"""
    for k in keys_local:
        vals = get_data_prop_values(g, s, k)
        if vals:
            return vals[0]
    # 兜底：随便拿一个数据属性的文本
    for p, o in g.predicate_objects(s):
        if isinstance(o, Literal):
            txt = str(o).strip()
            if txt:
                return txt
    return ""

# ===== 规范策略 =====
CANON = ["书经", "易经", "春秋", "诗经", "礼记"]
ORDER_INDEX = {k: i for i, k in enumerate(CANON)}

ALIASES = {
    # 书经
    "書經":"书经", "尚书":"书经", "尚書":"书经", "书":"书经",
    # 易经
    "易經":"易经", "周易":"易经", "易":"易经",
    # 春秋
    "春":"春秋",
    # 诗经
    "詩經":"诗经", "诗":"诗经",
    # 礼记
    "禮記":"礼记", "礼":"礼记",
}

SEP_RE = re.compile(r"[、，,；;／/｜\|\s]+")

def normalize_token(t: str) -> Optional[str]:
    s = (t or "").strip()
    if not s: return None
    if s in CANON: return s
    if s in ALIASES:
        s2 = ALIASES[s]
        return s2 if s2 in CANON else None
    s2 = s.replace("經","经").replace("記","记")
    return s2 if s2 in CANON else None

def normalize_values(raw_values: List[str]) -> List[str]:
    tokens: List[str] = []
    for v in raw_values:
        if v is None: continue
        txt = str(v).strip()
        if not txt: continue
        parts = SEP_RE.split(txt)
        tokens.extend([p for p in parts if p])
    normalized, seen = [], set()
    for t in tokens:
        canon = normalize_token(t)
        if canon and canon not in seen:
            seen.add(canon); normalized.append(canon)
    normalized.sort(key=lambda x: ORDER_INDEX.get(x, 999))
    return normalized

# ===== 主流程（纯 rdflib）=====
PERSON_CLASS_NAMES = {"Person", "人物"}           # 类本地名候选
PERSON_NAME_KEYS    = ["姓名", "name", "label", "rdfs_label", "标题", "title"]  # 显示名候选
TARGET_DP_LOCAL     = "学术专长"                  # 仅处理这个数据属性

# 找到 Person 类的 URI
person_classes = class_uris_by_localnames(g, PERSON_CLASS_NAMES)
if not person_classes:
    raise SystemExit("[FATAL] 未找到 Person/人物 类（按本地名匹配）")

# 列出所有 Person 实例
people = instances_of_classes(g, person_classes)
print("[INFO] 人物数：", len(people))

touched = changed = removed_only = 0

for i, s in enumerate(people, 1):
    # 人名用于打印
    pname = get_first_display(g, s, PERSON_NAME_KEYS) or localname(s)

    # 读原值
    old_vals = get_data_prop_values(g, s, TARGET_DP_LOCAL)
    old_vals = [x for x in old_vals if str(x).strip()]
    if not old_vals:
        continue

    # 规范化
    new_vals = normalize_values(old_vals)

    # 必清空
    clear_data_prop(g, s, TARGET_DP_LOCAL)

    # 写回（若有）
    if new_vals:
        add_data_prop_values(g, s, TARGET_DP_LOCAL, new_vals)
        print(f"[FIX] ({i}/{len(people)}) {pname} | 原值={old_vals} → 规范={new_vals}（已清空旧值并写回）")
        changed += 1
    else:
        print(f"[REMOVE] ({i}/{len(people)}) {pname} | 原值={old_vals} → 规范后为空（已清空旧值）")
        removed_only += 1
    touched += 1

print(f"[SUMMARY] 处理人物={touched} | 写回规范={changed} | 仅删除旧值(无可保留)={removed_only}")

# ===== 保存 =====
try:
    g.serialize(destination=OUT_TTL, format="turtle")
    g.serialize(destination=OUT_NT,  format="nt")
    print(f"[SAVED] {OUT_TTL}")
    print(f"[SAVED] {OUT_NT}")
except Exception as e:
    print("[WARN] 导出失败：", e)
