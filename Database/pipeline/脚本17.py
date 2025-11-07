# -*- coding: utf-8 -*-
"""
批处理：由人物“不规范时间表达”解析出【干支/年号/月日】→ 结合每场殿试逐场推断出生年
— 若出生年唯一 => 判定为“确定”，否则写入全部候选并标记“不确定”
— 计算登科年龄（周岁= 殿试年 - 生年）
— 以 UI 可见的方式写回：Person 的 DatatypeProperty 三元组 + owl:Axiom 注释（JSON 解析/溯源）

⚠️ 重要改动：所有 g.add / g.remove 写操作都封装为 g_add/g_remove，并在 with onto: 中执行，
   以避免 Owlready2 报错 “Cannot add triples outside a 'with' block.”
"""

import os
import re
import uuid
import json
from typing import List, Tuple, Dict, Optional, Any, Set

from owlready2 import *
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, OWL, XSD, RDFS
from pathlib import Path
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

OUT_DIR  = Path(_args.out_dir).expanduser().resolve()

# 确保输出目录存在
OUT_DIR.mkdir(parents=True, exist_ok=True)
_args = _parse_args()


# 确保输出目录存在
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TTL = str(Path(SRC).with_name("ontology_huji_dual_prenorm_containment_latestparser.ttl"))
OUT_NT  = str(Path(SRC).with_name("ontology_huji_dual_prenorm_containment_latestparser.nt"))

# ======== 明/南明 年号区间 ========
REIGNS: List[Tuple[str, int, int]] = [
    ("洪武", 1368, 1398), ("建文", 1399, 1402), ("永乐", 1403, 1424), ("洪熙", 1425, 1425),
    ("宣德", 1426, 1435), ("正统", 1436, 1449), ("景泰", 1450, 1456), ("天顺", 1457, 1464),
    ("成化", 1465, 1487), ("弘治", 1488, 1505), ("正德", 1506, 1521), ("嘉靖", 1522, 1566),
    ("隆庆", 1567, 1572), ("万历", 1573, 1620), ("泰昌", 1620, 1620), ("天启", 1621, 1627),
    ("崇祯", 1628, 1644),
    ("弘光", 1645, 1645), ("隆武", 1645, 1646), ("绍武", 1647, 1647), ("永历", 1646, 1662),
]
ERA2SPAN: Dict[str, Tuple[int, int]] = {nm: (s, e) for nm, s, e in REIGNS}
ERA_ORDER = [nm for nm, _, _ in REIGNS]
ERA_INDEX = {nm: i for i, nm in enumerate(ERA_ORDER)}

HEAVENLY = list("甲乙丙丁戊己庚辛壬癸")
EARTHLY  = list("子丑寅卯辰巳午未申酉戌亥")

# 年号别名（解析原文时使用）
ERA_ALIAS = {
    "永樂":"永乐","正統":"正统","天順":"天顺","龍庆":"隆庆","隆慶":"隆庆",
    "萬曆":"万历","萬歷":"万历","泰昌":"泰昌","天啟":"天启","崇禎":"崇祯",
    "紹武":"绍武","永曆":"永历",
}

# 干支容错（仅对两字候选局部纠错，不做全局替换）
STEM_CONFUSION   = {"已": "己", "巳": "己", "王": "壬", "攵": "戊"}
BRANCH_CONFUSION = {"成": "辰", "戍": "戌", "扌": "戌"}

# ========= UI 可见写回：候选属性名 =========
BIRTH_DP_CANDIDATES   = ["生年（判定）","生年_判定","出生年（判定）","出生年_判定"]
STATUS_DP_CANDIDATES  = ["出生年判定状态","生年判定状态"]
AGE_DP_CANDIDATES     = ["登科年龄","登科年齿","登科歲"]

# ========= 全局 onto/world/graph =========
onto: Optional[Ontology] = None
world: Optional[World]   = None
g: Optional[Graph]       = None


# ========== 通用/辅助 ==========
def _safe_get_class(name: str):
    global onto
    if onto is None: return None
    try:
        cls = getattr(onto, name, None)
        if isinstance(cls, ThingClass): return cls
    except: pass
    try:
        cls = onto.search_one(iri=f"*#{name}") or onto.search_one(iri=f"*{name}")
        if isinstance(cls, ThingClass): return cls
    except: pass
    try:
        cls = default_world.search_one(iri=f"*#{name}") or default_world.search_one(iri=f"*{name}")
        if isinstance(cls, ThingClass): return cls
    except: pass
    return None

def instances_of(name: str):
    cls = _safe_get_class(name)
    try:
        return list(cls.instances()) if cls else []
    except:
        return []

def dp_get_all(inst, dp_name: str) -> List[Any]:
    global onto
    try:
        v = getattr(inst, dp_name)
        if isinstance(v, list): return list(v)
        return [v] if v is not None else []
    except:
        try:
            prop = getattr(onto, dp_name)
            return list(prop[inst])
        except:
            return []

def dp_get_one(inst, dp_name: str):
    vs = dp_get_all(inst, dp_name)
    return vs[0] if vs else None

def op_get_all(inst, op_name: str) -> List[Any]:
    global onto
    try:
        v = getattr(inst, op_name, [])
        return list(v) if isinstance(v, list) else ([v] if v else [])
    except:
        try:
            prop = getattr(onto, op_name)
            return list(prop[inst])
        except:
            return []

def gz_of_year(year: int) -> str:
    return HEAVENLY[(year - 4) % 10] + EARTHLY[(year - 4) % 12]

def years_matching_gz_in_span(gz: str, start: int, end: int) -> List[int]:
    return [y for y in range(start, end + 1) if gz_of_year(y) == gz]

def era_of_year(y: int) -> Optional[str]:
    for nm,(s,e) in ERA2SPAN.items():
        if s <= y <= e: return nm
    return None

def is_dianshi_exam(exam) -> bool:
    level = dp_get_one(exam, "考试等级")
    return (level == "殿试") or (isinstance(level, str) and "殿试" in level)

def make_uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def base_ns_from_iri(iri: str) -> str:
    return iri.rsplit("#",1)[0] + "#" if "#" in iri else iri.rsplit("/",1)[0] + "#"


# ========== 解析“不规范表达” → 年号/干支/月日 ==========
CN_NUM = {
    "零":0, "〇":0, "○":0, "Ｏ":0,
    "一":1, "元":1, "正":1, "两":2, "二":2, "三":3, "四":4, "五":5, "六":6, "七":7, "八":8, "九":9,
    "十":10, "廿":20, "卅":30, "朔":1,
    "壹":1, "贰":2, "貳":2, "叁":3, "參":3, "肆":4, "伍":5, "陆":6, "陸":6, "柒":7, "捌":8, "玖":9, "拾":10
}

def _canon_stem(ch: str) -> str:
    return ch if ch in HEAVENLY else STEM_CONFUSION.get(ch, ch)

def _canon_branch(ch: str) -> str:
    return ch if ch in EARTHLY else BRANCH_CONFUSION.get(ch, ch)

def _cn_to_int_frag(s: str) -> Optional[int]:
    s = s.strip()
    if re.fullmatch(r"\d{1,3}", s): return int(s)
    if s.startswith("廿"):
        rest = s[1:] or "零"
        return 20 + (CN_NUM.get(rest, 0) if rest in CN_NUM else (_cn_to_int_frag(rest) or 0))
    if s.startswith("卅"):
        rest = s[1:] or "零"
        return 30 + (CN_NUM.get(rest, 0) if rest in CN_NUM else (_cn_to_int_frag(rest) or 0))
    if "十" in s or "拾" in s:
        s = s.replace("拾", "十")
        left, _, right = s.partition("十")
        ten = 10 if left in ("","零") else CN_NUM.get(left, 0) * 10
        if ten == 0 and left not in ("","零"): return None
        return ten + (CN_NUM.get(right, 0) if right in CN_NUM else (_cn_to_int_frag(right) or 0))
    acc = 0
    for ch in s:
        if ch in CN_NUM: acc = acc*10 + CN_NUM[ch]
        else: return None
    return acc if acc>0 else None

def parse_unstructured_time(s: str) -> Dict[str, Any]:
    """返回 dict: era, gz, month, day, notes(list)"""
    raw = re.sub(r"\s+", "", str(s or ""))

    # 年号
    era = None
    for nm in ERA2SPAN.keys():
        if nm in raw:
            era = nm; break
    if not era:
        for a,c in ERA_ALIAS.items():
            if a in raw:
                era = c; break

    # 干支（就地纠错）
    gz = None; note = []
    pairs = {raw[i:i+2] for i in range(max(0, len(raw)-1))}
    for token in pairs:
        if len(token) < 2: continue
        h,b = token[0], token[1]
        if h in HEAVENLY and b in EARTHLY:
            gz = h+b; break
        hc, bc = _canon_stem(h), _canon_branch(b)
        if (hc!=h or bc!=b) and hc in HEAVENLY and bc in EARTHLY:
            gz = hc+bc
            note.append(f"纠正: '{h}{b}' → '{gz}'")
            break

    # 月日
    m = d = None
    m1 = re.search(r"([元正一二三四五六七八九十廿卅壹贰貳叁參肆伍陆陸柒捌玖拾\d]{1,4})月", raw)
    d1 = re.search(r"([朔元正初一二三四五六七八九十廿卅壹贰貳叁參肆伍陆陸柒捌玖拾\d]{1,4})(日|号)", raw)
    if m1:
        mv = _cn_to_int_frag(m1.group(1))
        if mv and 1 <= mv <= 12:
            m = mv
    if d1:
        dd = d1.group(1).replace("初","")
        dv = _cn_to_int_frag(dd)
        if dv and 1 <= dv <= 31:
            d = dv

    return {"era": era, "gz": gz, "month": m, "day": d, "notes": note}


# ========== 从人物收集“不规范表达” ==========
BIRTH_RAW_PROPS = [
    "生年_原文","生年原文","出生_原文","出生时间_原文","出生原文",
    "出生纪年","出生纪年_原文","生年干支","生年表述","出生表述",
]

def collect_unstructured_texts_for_person(person) -> List[str]:
    texts: List[str] = []
    # 人物自身数据属性里可能就有
    for dpn in ["生年","生年干支","出生时间","出生纪年","生年原文","出生原文"]:
        v = dp_get_one(person, dpn)
        if isinstance(v, str) and v.strip():
            texts.append(v)
    # PropAssertion 值里也可能有
    try:
        for pa in onto.PropAssertion.instances():
            if person in op_get_all(pa, "about"):
                p = dp_get_one(pa, "prop") or ""
                val = dp_get_one(pa,"value") or dp_get_one(pa,"value_norm")
                if not isinstance(val,str): continue
                if (p in BIRTH_RAW_PROPS) or ("生" in p and ("年" in p or "干支" in p or "出生" in p)):
                    texts.append(val)
                else:
                    if re.search(r"[甲乙丙丁戊己庚辛壬癸].?[子丑寅卯辰巳午未申酉戌亥]", val):
                        texts.append(val)
                    elif re.search(r"[月日号]", val):
                        texts.append(val)
    except Exception:
        pass
    # 去重
    uniq, seen = [], set()
    for t in texts:
        k = re.sub(r"\s+","",t)
        if k and k not in seen:
            uniq.append(t); seen.add(k)
    return uniq


# ========== 由“单场殿试 + 出生干支”推断出生年 ==========
def extract_exam_years(exam) -> List[int]:
    cand_years: List[int] = []

    # 直接字段
    y = dp_get_one(exam, "考试时间")
    if y and str(y).isdigit():
        cand_years.append(int(str(y)))

    # PropAssertion 中的考试时间/规范
    try:
        for pa in onto.PropAssertion.instances():
            if exam in op_get_all(pa, "about"):
                p = dp_get_one(pa, "prop")
                vnorm = dp_get_one(pa, "value_norm") or dp_get_one(pa, "value")
                if not vnorm: continue
                s = re.sub(r"\s+","",str(vnorm))
                if p in ("考试时间_规范","考试时间") and s.isdigit():
                    cand_years.append(int(s))
    except Exception:
        pass

    # 年号+序数
    era = None; ordv = None
    try:
        for pa in onto.PropAssertion.instances():
            if exam in op_get_all(pa, "about"):
                p = dp_get_one(pa, "prop")
                v = dp_get_one(pa, "value_norm") or dp_get_one(pa, "value")
                if not v: continue
                sv = re.sub(r"\s+","",str(v))
                if p == "年号":
                    sv = ERA_ALIAS.get(sv, sv)
                    if sv in ERA2SPAN: era = sv
                elif p == "序数":
                    t = re.sub(r"[年秊年]$", "", sv)
                    t = t.replace("元","一").replace("正","一").replace("廿","二十").replace("卅","三十")
                    table = str.maketrans("一二三四五六七八九零〇", "12345678900")
                    tn = t.translate(table)
                    if tn.isdigit(): ordv = int(tn)
    except Exception:
        pass

    if era in ERA2SPAN and isinstance(ordv,int):
        s,e = ERA2SPAN[era]; y3 = s + ordv - 1
        if s <= y3 <= e: cand_years.append(y3)

    # 年号+干支
    gz = None
    try:
        for pa in onto.PropAssertion.instances():
            if exam in op_get_all(pa, "about"):
                if dp_get_one(pa, "prop") == "干支":
                    v = dp_get_one(pa,"value_norm") or dp_get_one(pa,"value")
                    if v and len(str(v))==2 and str(v)[0] in HEAVENLY and str(v)[1] in EARTHLY:
                        gz = str(v); break
    except Exception:
        pass

    if gz and era in ERA2SPAN:
        s,e = ERA2SPAN[era]
        cand_years += years_matching_gz_in_span(gz, s, e)

    return sorted(list(dict.fromkeys(cand_years)))

def infer_birth_years_from_exam(exam, gz_birth: str) -> Tuple[Optional[str], List[int], Optional[str]]:
    if not gz_birth or len(gz_birth)!=2 or gz_birth[0] not in HEAVENLY or gz_birth[1] not in EARTHLY:
        return None, [], None
    # 取考试年号（优先 PropAssertion；否则由考试年反推）
    era = None
    try:
        for pa in onto.PropAssertion.instances():
            if exam in op_get_all(pa, "about") and dp_get_one(pa,"prop")=="年号":
                era = (dp_get_one(pa,"value_norm") or dp_get_one(pa,"value") or "").strip()
                era = ERA_ALIAS.get(era, era)
                break
    except Exception:
        pass
    if era not in ERA2SPAN:
        years = extract_exam_years(exam)
        if years:
            era = era_of_year(years[0])
    if era not in ERA2SPAN:
        return None, [], None

    exam_years = extract_exam_years(exam)
    ub = max(exam_years) if exam_years else None

    start_idx = ERA_INDEX[era]
    used_era = era
    cand: List[int] = []
    back_to_era = None
    for i in range(start_idx, -1, -1):
        era_i = ERA_ORDER[i]
        s,e = ERA2SPAN[era_i]
        ys = years_matching_gz_in_span(gz_birth, s, e)
        if ub is not None:
            ys = [y for y in ys if y <= ub]
        if ys:
            used_era = era
            back_to_era = None if era_i == era else era_i
            cand = sorted(list(dict.fromkeys(ys)))
            break
    return used_era, cand, back_to_era


# ========== Owlready2 写操作封装（必须 with onto:） ==========
def g_add(triple):
    with onto:
        g.add(triple)

def g_remove(triple):
    with onto:
        g.remove(triple)

def ensure_annotation_property(ap_iri: URIRef):
    with onto:
        if (ap_iri, RDF.type, OWL.AnnotationProperty) not in g:
            g.add((ap_iri, RDF.type, OWL.AnnotationProperty))

def remove_axiom_exact(subj: URIRef, dp: URIRef, lit: Literal):
    with onto:
        for ax in list(g.subjects(OWL.annotatedSource, subj)):
            if (ax, RDF.type, OWL.Axiom) in g and \
               (ax, OWL.annotatedProperty, dp) in g and \
               (ax, OWL.annotatedTarget, lit) in g:
                g.remove((ax, None, None))

def write_with_axiom(subj: URIRef, dp: URIRef, lit: Literal,
                     ap_iri: URIRef, payload: dict):
    ensure_annotation_property(ap_iri)
    # 先删同值（避免重复）
    remove_axiom_exact(subj, dp, lit)
    g_remove((subj, dp, lit))
    # 新三元组
    g_add((subj, dp, lit))
    # Axiom 注释（JSON）
    ax = BNode()
    with onto:
        g.add((ax, RDF.type, OWL.Axiom))
        g.add((ax, OWL.annotatedSource,  subj))
        g.add((ax, OWL.annotatedProperty, dp))
        g.add((ax, OWL.annotatedTarget,  lit))
        payload_json = Literal(
            json.dumps(payload, ensure_ascii=False, separators=(",",":")),
            datatype=XSD.string
        )
        g.add((ap_iri, RDF.type, OWL.AnnotationProperty))  # 再保险
        g.add((ax, ap_iri, payload_json))

def localname(u: str) -> str:
    return u.rsplit("#",1)[-1] if "#" in u else u.rsplit("/",1)[-1]

def find_existing_dp_uris_by_names(name_candidates: List[str]) -> List[URIRef]:
    """在图中查找以候选名结尾的 DatatypeProperty 或任意谓词（再补声明为 DatatypeProperty）"""
    hits: List[URIRef] = []
    # 1) 已声明为 DatatypeProperty 的
    for dp in g.subjects(RDF.type, OWL.DatatypeProperty):
        s = str(dp)
        if any(s.endswith("#"+nm) or s.endswith("/"+nm) or localname(s)==nm for nm in name_candidates):
            hits.append(URIRef(s))
    # 2) 其它谓词也纳入
    for s,p,o in g.triples((None, None, None)):
        if isinstance(p, URIRef):
            ps = str(p)
            if any(ps.endswith("#"+nm) or ps.endswith("/"+nm) or localname(ps)==nm for nm in name_candidates):
                if URIRef(ps) not in hits:
                    hits.append(URIRef(ps))
    return hits

def pick_preferred_dp_uri(all_found: List[URIRef], preferred_names: List[str],
                          fallback_ns: Optional[str], fallback_name: str) -> URIRef:
    if all_found:
        # 精确按优先名挑选
        for want in preferred_names:
            for uri in all_found:
                s = str(uri)
                if s.endswith("#"+want) or s.endswith("/"+want) or localname(s)==want:
                    return uri
        return all_found[0]
    # 不存在 → 创建一个新的 DP IRI（声明为 DatatypeProperty）
    ns = fallback_ns or "http://example.org/onto#"
    new_uri = URIRef(ns + fallback_name)
    g_add((new_uri, RDF.type, OWL.DatatypeProperty))
    return new_uri

def ensure_dp_domain_person(dp_uri: URIRef, person_class_iri: Optional[str]):
    """可选：给 DP 声明 domain=Person，提升 UI 显示概率"""
    if not person_class_iri: return
    po = URIRef(person_class_iri)
    has_domain = False
    for o in g.objects(dp_uri, RDFS.domain):
        if str(o) == str(po) or str(o).endswith("#Person") or str(o).endswith("/Person") or "人物" in str(o):
            has_domain = True; break
    if not has_domain:
        g_add((dp_uri, RDFS.domain, po))

def label_maker_for_year(by: int) -> str:
    era = era_of_year(by)
    gz  = gz_of_year(by)
    return f"{era}{gz}" if era else gz


# ========== 主流程 ==========
def main():
    global onto, world, g
    if not os.path.exists(SRC):
        print("[FATAL] 本体文件不存在：", SRC); return

    # 加载
    onto = get_ontology(SRC).load()
    world = default_world
    g = world.as_rdflib_graph()
    print(f"[OK] 已加载：{SRC}")

    # Person 类 IRI（用于 DP.domain）
    Person = _safe_get_class("Person") or _safe_get_class("人物")
    person_class_iri = str(getattr(Person, "iri", "")) if Person else None

    # 基础命名空间推断（优先用某个现有 DP 的命名空间）
    base_ns = None
    for s,p,o in g.triples((None, RDF.type, OWL.DatatypeProperty)):
        base_ns = base_ns or base_ns_from_iri(str(s))
        break
    if not base_ns and person_class_iri:
        base_ns = base_ns_from_iri(person_class_iri)

    # 找到/创建 三个首选 DP
    birth_dp_candidates  = find_existing_dp_uris_by_names(BIRTH_DP_CANDIDATES)
    status_dp_candidates = find_existing_dp_uris_by_names(STATUS_DP_CANDIDATES)
    age_dp_candidates    = find_existing_dp_uris_by_names(AGE_DP_CANDIDATES)

    dp_birth_uri  = pick_preferred_dp_uri(birth_dp_candidates,  BIRTH_DP_CANDIDATES,  base_ns, BIRTH_DP_CANDIDATES[0])
    dp_status_uri = pick_preferred_dp_uri(status_dp_candidates, STATUS_DP_CANDIDATES, base_ns, STATUS_DP_CANDIDATES[0])
    dp_age_uri    = pick_preferred_dp_uri(age_dp_candidates,    AGE_DP_CANDIDATES,    base_ns, AGE_DP_CANDIDATES[0])

    # 确保它们是 DatatypeProperty 且 domain=Person（增加 UI 可见概率）
    for uri in [dp_birth_uri, dp_status_uri, dp_age_uri]:
        g_add((uri, RDF.type, OWL.DatatypeProperty))
        ensure_dp_domain_person(uri, person_class_iri)

    # AnnotationProperty “解析”
    AP_PARSE = URIRef(base_ns + "解析")
    ensure_annotation_property(AP_PARSE)

    persons = instances_of("Person")
    print(f"[INFO] 人物数：{len(persons)}")

    changed_cnt = 0

    for idx, person in enumerate(persons, 1):
        name = dp_get_one(person, "姓名") or getattr(person,"name","(无名)")
        # 该人所有殿试（去重：按实例 id/iri）
        exams = []
        for evt in op_get_all(person, "participatesIn"):
            for ex in op_get_all(evt, "hasExam"):
                if is_dianshi_exam(ex):
                    exams.append(ex)
        ex_keys, uniq_exams = set(), []
        for ex in exams:
            key = getattr(ex,"name",None) or getattr(ex,"iri",None) or f"id{ id(ex) }"
            if key in ex_keys: continue
            ex_keys.add(key); uniq_exams.append(ex)
        if not uniq_exams:
            continue

        # 逐人收集“不规范表达”并解析出干支候选
        raws = collect_unstructured_texts_for_person(person)
        if not raws:
            continue
        gz_candidates: Set[str] = set()
        for t in raws:
            parsed = parse_unstructured_time(t)
            if parsed.get("gz"):
                gz_candidates.add(parsed["gz"])
        if not gz_candidates:
            continue

        # 逐场殿试 推断出生年（合并去重）
        birth_years: Set[int] = set()
        exam_years_all: Set[int] = set()
        for ex in uniq_exams:
            yrs = extract_exam_years(ex)
            exam_years_all.update(yrs)
        for gz in gz_candidates:
            for ex in uniq_exams:
                _, cand, _ = infer_birth_years_from_exam(ex, gz)
                birth_years.update(cand)

        if not birth_years:
            continue

        birth_years_sorted = sorted(birth_years)
        status = "确定" if len(birth_years_sorted) == 1 else "不确定"

        # 计算登科年龄：对每个殿试年 × 每个出生年候选
        exam_ages = []
        for y in sorted(exam_years_all):
            for by in birth_years_sorted:
                age = y - by
                if 0 <= age <= 120:
                    exam_ages.append({"exam_year": y, "age": age})

        # 写回（UI 可见）
        person_iri = URIRef(getattr(person, "iri"))
        # 2.1 状态
        write_with_axiom(
            person_iri, dp_status_uri, Literal(status, datatype=XSD.string),
            AP_PARSE, {"系统":"birth-infer","字段":"出生年判定状态"}
        )
        # 2.2 生年（判定）：多候选
        for by in birth_years_sorted:
            label = label_maker_for_year(by) or str(by)  # “年号+干支”优先，退化为年份字符串
            write_with_axiom(
                person_iri, dp_birth_uri, Literal(label, datatype=XSD.string),
                AP_PARSE, {"系统":"birth-infer","字段":"生年（判定）","公元年":by}
            )
        # 2.3 登科年龄
        for item in exam_ages:
            y, ag = int(item["exam_year"]), int(item["age"])
            write_with_axiom(
                person_iri, dp_age_uri, Literal(f"{y}: {ag}", datatype=XSD.string),
                AP_PARSE, {"系统":"birth-infer","字段":"登科年龄","殿试年":y,"周岁":ag}
            )

        changed_cnt += 1
        print(f"[{idx}/{len(persons)}] {name} | 出生年候选={birth_years_sorted} | 状态={status} | 殿试数={len(uniq_exams)}")

    # 保存
    try:
        g.serialize(destination=OUT_TTL, format="turtle")
        g.serialize(destination=OUT_NT,  format="nt")
        print(f"\n[SAVED] {OUT_TTL}")
        print(f"[SAVED] {OUT_NT}")
        print(f"[DONE] 完成写回（变更人数≈{changed_cnt}）")
    except Exception as e:
        print("[WARN] 导出失败：", e)


if __name__ == "__main__":
    main()
