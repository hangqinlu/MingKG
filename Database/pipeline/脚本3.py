# -*- coding: utf-8 -*-
"""
Neo4j → OWL：实体/事件消歧 + 溯源（TextProvenance/PropAssertion）+ 简繁统一 + 可视化 + 导出
- 读取：关系 (s)-[r]->(t)、(:PropAssertion)-[:ABOUT]->(实体)、(:TextProvenance)
- 事件化：参与/担任/社会关系/生/任务执行/职任/隶属（把关系属性挂到事件个体）
- 溯源：
    * 导入 Neo4j 的 TextProvenance / PropAssertion（保持原值；prop 做简繁映射）
    * 事件属性由本脚本生成 PropAssertion（about→事件；derivedFrom→record_id）
    * contains(TextProvenance→实体/事件) 自动补齐
- OWL 公理（不删除数据、仅表达唯一性/同一性语义）：
    * FunctionalProperty：
        - Person：姓名、户籍类型、户籍地、家庭排行
        - ImperialExam：考试时间
        - ParticipationEvent：甲第等级、功名、考中年龄、名次
    * HasKey：
        - Person(姓名)、Place(历史名称)、OfficialPosition(官职名称)、ImperialExam(考试等级+考试时间)
    * 事件类互斥（保留）；⚠ 不再对“事件→人物”施加 exactly(1) 约束
- 消歧规则：
    * Person：键=姓名
    * Place：键=历史名称
    * OfficialPosition：键=官职名称
    * ImperialExam：仅当(考试等级+考试时间)齐全时全局合并
    * 人域“无考试时间”科举合并：同一人+同一等级；若该人已有同等级“有时间”场次，无时间并到该场次
- 事件去重：
    * 参与：同一人+同一场考试
    * 担任：同一人+同一官职（不写“授予类型”，也不溯源该字段）
- 导出：TTL/RDF/NT、CSV（实体/事件）、违规清单、两张 PNG（全量/投影）+ GraphML、仪表盘 HTML

本版关键变更：
- 删除对象属性：hasPerson、hasOtherPerson（事件→人物）
- 仅保留“人物→事件”对象属性：participatesIn / appointedIn / socialRelationEvent /
  bornInEvent / performedTaskIn / heldOfficeAtEvent（Domain=Person，Range=各事件子类）
- 所有用到人物参与关系的逻辑（验证/去重/导出/可视化）改用人物→事件 + 内存索引 EVENT_PERSONS
"""

import os, re, csv, json, hashlib, datetime, unicodedata
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

from neo4j import GraphDatabase
from owlready2 import *
import networkx as nx
import matplotlib.pyplot as plt

# ================== 基本配置 ==================
# ================== 基本配置 ==================
# NEO4J_URI  = "bolt://localhost:7687"
# NEO4J_USER = "neo4j"
# NEO4J_PWD  = "xxx"
# RUN_REASONER_BEFORE = True
# RUN_REASONER_AFTER  = False
# OUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", f"owl_inspect_{...}")
# ONTO_IRI = "http://mingkg.org/ontology/imperial_exam.owl#"

import os, argparse, datetime

def _parse_args():
    ap = argparse.ArgumentParser(description="脚本3：Neo4j→OWL/NT（外部配置）")
    ap.add_argument("--neo4j-uri",  default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--neo4j-pwd",  default=os.getenv("NEO4J_PWD",  "lhq18385458795"))
    ap.add_argument("--out-dir", default=os.getenv("OUT_DIR",
        os.path.join(os.path.expanduser("~"), "Desktop", f"owl_inspect_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")),
        help="输出目录（OUT_DIR）")
    ap.add_argument("--reasoner-before", action="store_true", default=os.getenv("RUN_REASONER_BEFORE","True").lower()=="true",
                    help="是否在合并前运行推理（RUN_REASONER_BEFORE）")
    ap.add_argument("--reasoner-after", action="store_true", default=os.getenv("RUN_REASONER_AFTER","False").lower()=="true",
                    help="是否在合并后运行推理（RUN_REASONER_AFTER）")
    return ap.parse_args()

_args = _parse_args()
NEO4J_URI  = _args.neo4j_uri
NEO4J_USER = _args.neo4j_user
NEO4J_PWD  = _args.neo4j_pwd
OUT_DIR    = _args.out_dir
RUN_REASONER_BEFORE = _args.reasoner_before
RUN_REASONER_AFTER  = _args.reasoner_after
os.makedirs(OUT_DIR, exist_ok=True)

# =============== 简繁统一 & 规范化 ===============

ONTO_IRI = "http://mingkg.org/ontology/imperial_exam.owl#"

# =============== 简繁统一 & 规范化 ===============
try:
    from opencc import OpenCC
    _cc = OpenCC("t2s")
except Exception:
    _cc = None

COMMON_T2S_SINGLE = str.maketrans({
    "臺":"台","灣":"湾","廣":"广","東":"东","蘇":"苏","陝":"陕","齊":"齐","經":"经","緯":"纬",
    "歷":"历","縣":"县","鄉":"乡","會":"会","試":"试","殿":"殿","階":"阶","職":"职","遷":"迁",
    "關":"关","係":"系","類":"类","錄":"录","點":"点","數":"数","專":"专","術":"术","戶":"户",
    "當":"当","與":"与","於":"于","學":"学","官":"官","稱":"称",
})
COMMON_T2S_MULTI = {
    "會試":"会试","殿試":"殿试","鄉試":"乡试","童試":"童试",
    "社會關係":"社会关系","參與":"参与","擔任":"担任","隸屬":"隶属","任務執行":"任务执行","職任":"职任",
    "名稱":"名称","時間":"时间","地點":"地点","政策":"政策","甲第等級":"甲第等级","年齡":"年龄",
    "經緯座標":"经纬坐标","歷史名稱":"历史名称","現代名稱":"现代名称","官職名稱":"官职名称","官階":"官阶",
    "考試等級":"考试等级","考試時間":"考试时间","考試地點":"考试地点","授予類型":"授予类型","遷轉":"迁转","授予時間":"授予时间",
}

_punct_re = re.compile(r"[\s\u3000\u00A0·\.\-_:、，,。；;（）\(\)\[\]{}“”\"'‘’/\\]+")

def t2s(s: Any) -> str:
    if s is None: return ""
    x = str(s)
    if _cc:
        try: x = _cc.convert(x)
        except Exception: pass
    x = x.translate(COMMON_T2S_SINGLE)
    for k,v in COMMON_T2S_MULTI.items():
        if k in x: x = x.replace(k, v)
    return x

def canon_text(s: Any) -> str:
    x = t2s(s)
    x = unicodedata.normalize("NFKC", x).strip()
    x = _punct_re.sub(" ", x).strip().lower()
    return x

def safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s or "")

# =============== Label/属性映射（可扩） ===============
LABEL_ALIASES = {
    "Person":{"Person","人物"},
    "Place":{"Place","地點","地点"},
    "ImperialExam":{"ImperialExam","科舉","科举","考试"},
    "OfficialPosition":{"OfficialPosition","Official Position","職官","官職","官员"},
    "TextProvenance":{"TextProvenance"},
    "PropAssertion":{"PropAssertion"},
}
REL_ALIASES = {
    "参与":{"参与","參與"},
    "担任":{"担任","擔任"},
    "社会关系":{"社会关系","社會關係"},
    "生":{"生"},
    "任务执行":{"任务执行","任務執行"},
    "职任":{"职任","職任"},
    "隶属":{"隶属","隸屬"},
}
PROP_ALIASES = {
    # Person
    "姓名":{"姓名"}, "字":{"字"}, "生年":{"生年"}, "户籍类型":{"户籍类型","戶籍類型"},
    "户籍地":{"户籍地","戶籍地"}, "学籍":{"学籍","學籍"}, "家庭排行":{"家庭排行"}, "学术专长":{"学术专长","學術專長"},
    # Place
    "历史名称":{"历史名称","歷史名稱"}, "现代名称":{"现代名称","現代名稱"},
    "现代区划层级":{"现代区划层级","現代區劃層級"}, "经纬坐标":{"经纬坐标","經緯座標"},
    # OfficialPosition
    "官职名称":{"官职名称","官職名稱"}, "官阶":{"官阶","官階"},
    # ImperialExam
    "考试等级":{"考试等级","考試等級"}, "考试时间":{"考试时间","考試時間"}, "考试地点":{"考试地点","考試地點"},
    "主考官":{"主考官"}, "科举政策":{"科举政策","科舉政策"},
    # 关系属性
    "甲第等级":{"甲第等级","甲第等級","甲第"}, "考中年龄":{"考中年龄","考中年齡","登科年龄","登科年齡"},
    "名次":{"名次"}, "功名":{"功名"},
    "授予类型":{"授予类型","授予類型"}, "迁转":{"迁转","遷轉"}, "授予时间":{"授予时间","授予時間"},
    "关系类型":{"关系类型","關係類型"}, "职事名目":{"职事名目","職事名目"}, "任务时间":{"任务时间","任務時間"},
    "任职时间":{"任职时间","任職時間"},
    # 溯源元字段
    "record_id":{"record_id"}, "batch_id":{"batch_id"}, "node_uid":{"node_uid"},
    "Text_source":{"Text_source"}, "record_confidence":{"record_confidence"}, "Text":{"Text"},
}

def _rev(d: Dict[str,set]) -> Dict[str,str]:
    out={}
    for k,alts in d.items():
        out[canon_text(k)] = k
        for a in alts: out[canon_text(a)] = k
    return out

LABEL_INDEX = _rev(LABEL_ALIASES)
REL_INDEX   = _rev(REL_ALIASES)
PROP_INDEX  = _rev(PROP_ALIASES)

def norm_label(s:str)->str: return LABEL_INDEX.get(canon_text(s), s)
def norm_rel(s:str)->str:   return REL_INDEX.get(canon_text(s), s)
def norm_prop(s:str)->str:  return PROP_INDEX.get(canon_text(s), s)

def norm_labels(lbls)->set: return {norm_label(l) for l in (lbls or [])}
def norm_props(props: Dict[str,Any]) -> Dict[str,Any]:
    out={}
    for k,v in (props or {}).items():
        nk = norm_prop(k)
        if isinstance(v, str): out[nk] = t2s(v).strip()
        else: out[nk] = v
    return out

REL_TYPES = {"参与","担任","社会关系","生","任务执行","职任","隶属"}

# =============== 溯源时跳过的属性（不生成 PropAssertion） ===============
PROV_SKIP_KEYS = {
    "授予类型","官阶","现代名称","现代区划层级","经纬坐标"
}

# ================== Neo4j 查询 ==================
Q_FETCH_EDGES = """
MATCH (s)-[r]->(t)
RETURN
  type(r) AS rel_type,
  properties(r) AS rprops,
  labels(s) AS s_labels, properties(s) AS s_props, elementId(s) AS sid,
  labels(t) AS t_labels, properties(t) AS t_props, elementId(t) AS tid
"""
Q_FETCH_TP = """
MATCH (p:TextProvenance)
RETURN p.record_id AS record_id,
       p.Text_source AS Text_source,
       p.record_confidence AS record_confidence,
       p.Text AS Text
"""
Q_FETCH_PA = """
MATCH (a:PropAssertion)-[:ABOUT]->(n)
OPTIONAL MATCH (a)-[:DERIVED_FROM]->(p:TextProvenance)
RETURN properties(a) AS aprops,
       labels(n) AS n_labels, properties(n) AS n_props, elementId(n) AS nid,
       p.record_id AS tp_record_id
"""

def neo_rows(uri, user, pwd, q):
    driver = GraphDatabase.driver(uri, auth=(user,pwd))
    data=[]
    with driver.session() as s:
        for rec in s.run(q):
            data.append(rec.data())
    driver.close()
    return data

# =============== 内存索引：事件 → [人物] ===============
EVENT_PERSONS: Dict[Thing, List[Thing]] = defaultdict(list)

# ================== 构建本体骨架 ==================
def build_onto():
    onto = get_ontology(ONTO_IRI)
    with onto:
        # === Classes ===
        class Text(Thing): pass
        class Person(Thing): pass
        class OfficialPosition(Thing): pass
        class ImperialExam(Thing): pass
        class Place(Thing): pass

        class Event(Thing): pass
        class ParticipationEvent(Event): pass
        class AppointmentEvent(Event): pass
        class SocialRelationEvent(Event): pass
        class BirthEvent(Event): pass
        class TaskEvent(Event): pass
        class OfficeAtPlaceEvent(Event): pass
        class SubordinationEvent(Event): pass

        # 溯源类
        class TextProvenance(Thing): pass
        class PropAssertion(Thing): pass

        # === 对象属性（删除事件→人物：hasPerson / hasOtherPerson） ===
        class hasPlace(ObjectProperty): domain=[Event]; range=[Place]
        class hasSuperPlace(ObjectProperty): domain=[SubordinationEvent]; range=[Place]
        class hasExam(ObjectProperty): domain=[ParticipationEvent]; range=[ImperialExam]
        class hasPosition(ObjectProperty): domain=[AppointmentEvent]; range=[OfficialPosition]

        # === 人物 → 事件（Domain=Person, Range=各事件子类） ===
        class participatesIn(ObjectProperty):
            domain = [Person]; range = [ParticipationEvent]
        class appointedIn(ObjectProperty):
            domain = [Person]; range = [AppointmentEvent]
        class socialRelationEvent(ObjectProperty):
            domain = [Person]; range = [SocialRelationEvent]
        class bornInEvent(ObjectProperty):
            domain = [Person]; range = [BirthEvent]
        class performedTaskIn(ObjectProperty):
            domain = [Person]; range = [TaskEvent]
        class heldOfficeAtEvent(ObjectProperty):
            domain = [Person]; range = [OfficeAtPlaceEvent]
        # 注：不设置 inverse

        # 溯源关系
        class about(ObjectProperty):        domain=[PropAssertion]; range=[Thing]
        class derivedFrom(ObjectProperty):  domain=[PropAssertion]; range=[TextProvenance]
        class contains(ObjectProperty):     domain=[TextProvenance]; range=[Thing]

        # === Data properties ===
        class neo4j_id(DataProperty, FunctionalProperty): domain=[Thing]; range=[str]

        # Person —— 唯一（FunctionalProperty）
        class 姓名(DataProperty, FunctionalProperty):     domain=[Person]; range=[str]
        class 户籍类型(DataProperty, FunctionalProperty): domain=[Person]; range=[str]
        class 户籍地(DataProperty, FunctionalProperty):   domain=[Person]; range=[str]
        class 家庭排行(DataProperty, FunctionalProperty): domain=[Person]; range=[str]

        # Person —— 非唯一
        class 字(DataProperty):     domain=[Person]; range=[str]
        class 生年(DataProperty):   domain=[Person]; range=[str]
        class 学籍(DataProperty):   domain=[Person]; range=[str]
        class 学术专长(DataProperty): domain=[Person]; range=[str]

        # OfficialPosition
        class 官职名称(DataProperty): domain=[OfficialPosition]; range=[str]
        class 官阶(DataProperty):   domain=[OfficialPosition]; range=[str]

        # ImperialExam —— 考试时间唯一
        class 考试等级(DataProperty): domain=[ImperialExam]; range=[str]
        class 考试时间(DataProperty, FunctionalProperty): domain=[ImperialExam]; range=[str]
        class 考试地点(DataProperty): domain=[ImperialExam]; range=[str]
        class 主考官(DataProperty):   domain=[ImperialExam]; range=[str]
        class 科举政策(DataProperty): domain=[ImperialExam]; range=[str]

        # Place
        class 历史名称(DataProperty):   domain=[Place]; range=[str]
        class 现代名称(DataProperty):   domain=[Place]; range=[str]
        class 现代区划层级(DataProperty): domain=[Place]; range=[str]
        class 经纬坐标(DataProperty):   domain=[Place]; range=[str]

        # ParticipationEvent —— 唯一
        class 甲第等级(DataProperty, FunctionalProperty): domain=[ParticipationEvent]; range=[str]
        class 功名(DataProperty, FunctionalProperty):     domain=[ParticipationEvent]; range=[str]
        class 考中年龄(DataProperty, FunctionalProperty): domain=[ParticipationEvent]; range=[str]
        class 名次(DataProperty, FunctionalProperty):     domain=[ParticipationEvent]; range=[str]

        # AppointmentEvent ——（不写“授予类型”）
        class 迁转(DataProperty):     domain=[AppointmentEvent]; range=[str]
        class 授予时间(DataProperty): domain=[AppointmentEvent]; range=[str]

        # SocialRelationEvent
        class 关系类型(DataProperty): domain=[SocialRelationEvent]; range=[str]

        # TaskEvent / OfficeAtPlaceEvent
        class 职事名目(DataProperty): domain=[TaskEvent]; range=[str]
        class 任务时间(DataProperty): domain=[TaskEvent]; range=[str]
        class 任职时间(DataProperty): domain=[OfficeAtPlaceEvent]; range=[str]

        # TextProvenance / PropAssertion 数据属性
        class record_id(DataProperty, FunctionalProperty):        domain=[TextProvenance, PropAssertion, Thing]; range=[str]
        class Text_source(DataProperty):                          domain=[TextProvenance]; range=[str]
        class record_confidence(DataProperty):                    domain=[TextProvenance]; range=[str]
        class Text_body(DataProperty):                            domain=[TextProvenance]; range=[str]
        class prop(DataProperty):                                 domain=[PropAssertion]; range=[str]
        class value(DataProperty):                                domain=[PropAssertion]; range=[str]
        class value_norm(DataProperty):                           domain=[PropAssertion]; range=[str]

        # === 事件类互斥（去掉对人物的 exactly(1) 约束） ===
        AllDisjoint([ParticipationEvent, AppointmentEvent, SocialRelationEvent,
                     BirthEvent, TaskEvent, OfficeAtPlaceEvent, SubordinationEvent])

        # === HasKey 公理（带回退）===
        def _try_add_key(cls, props):
            try:
                from owlready2 import HasKey
                cls.is_a.append(HasKey(props))
                return True
            except Exception as e:
                print(f"!! 环境不支持 HasKey，跳过 {cls.name} 键公理：{[p.name for p in props]} | {e}")
                return False

        _try_add_key(Person, [姓名])
        _try_add_key(Place, [历史名称])
        _try_add_key(OfficialPosition, [官职名称])
        _try_add_key(ImperialExam, [考试等级, 考试时间])

    return onto


# =============== DP / OP 工具 ===============
def set_dp(onto, inst, key, val):
    if val in (None,"","[]"): return
    try:
        cur = list(onto[key][inst]) if onto[key][inst] else []
        sval = str(val)
        if sval not in cur:
            onto[key][inst] = cur + [sval]
    except Exception:
        pass

def list_vals(onto, inst, key):
    try: return list(onto[key][inst]) or []
    except Exception: return []

def get_first(onto, inst, key):
    try:
        vals = onto[key][inst]; return vals[0] if vals else ""
    except Exception:
        return ""

def op_append(op, subj, obj):
    """安全地向对象属性 op 中追加一条 subj -> obj 的边，避免覆盖和重复。"""
    try:
        cur = list(op[subj]) if op[subj] else []
        if obj not in cur:
            op[subj] = cur + [obj]
    except Exception:
        pass

# =============== 个体构造 ===============
def ensure_person(onto, props, neo_id=None):
    name = f"Person_{safe_id(neo_id) if neo_id else hashlib.md5(json.dumps(props,ensure_ascii=False).encode()).hexdigest()[:10]}"
    inst = onto.Person(name); set_dp(onto, inst, "neo4j_id", neo_id)
    for k in ("姓名","字","生年","户籍类型","户籍地","学籍","家庭排行","学术专长","record_id","batch_id","node_uid"):
        if k in props: set_dp(onto, inst, k, props.get(k))
    return inst

def ensure_place(onto, props, neo_id=None):
    name = f"Place_{safe_id(neo_id) if neo_id else hashlib.md5(json.dumps(props,ensure_ascii=False).encode()).hexdigest()[:10]}"
    inst = onto.Place(name); set_dp(onto, inst, "neo4j_id", neo_id)
    for k in ("历史名称","现代名称","现代区划层级","经纬坐标","record_id","batch_id","node_uid"):
        if k in props: set_dp(onto, inst, k, props.get(k))
    return inst

def ensure_position(onto, props, neo_id=None):
    name = f"OfficialPosition_{safe_id(neo_id) if neo_id else hashlib.md5(json.dumps(props,ensure_ascii=False).encode()).hexdigest()[:10]}"
    inst = onto.OfficialPosition(name); set_dp(onto, inst, "neo4j_id", neo_id)
    for k in ("官职名称","官阶","record_id","batch_id","node_uid"):
        if k in props: set_dp(onto, inst, k, props.get(k))
    return inst

def ensure_exam(onto, props, neo_id=None):
    name = f"ImperialExam_{safe_id(neo_id) if neo_id else hashlib.md5(json.dumps(props,ensure_ascii=False).encode()).hexdigest()[:10]}"
    inst = onto.ImperialExam(name); set_dp(onto, inst, "neo4j_id", neo_id)
    for k in ("考试等级","考试时间","考试地点","主考官","科举政策","record_id","batch_id","node_uid"):
        if k in props: set_dp(onto, inst, k, props.get(k))
    return inst

def build_node(onto, labels, props, neo_id):
    lbs = set(labels or [])
    if "Person" in lbs: return ensure_person(onto, props, neo_id)
    if "Place" in lbs: return ensure_place(onto, props, neo_id)
    if "ImperialExam" in lbs: return ensure_exam(onto, props, neo_id)
    if "OfficialPosition" in lbs or "Official Position" in lbs: return ensure_position(onto, props, neo_id)
    # 兜底
    return ensure_exam(onto, props, neo_id)

# =============== 溯源：TextProvenance ===============
def get_or_create_textprov(onto, rid: Optional[str], extra: Dict[str,Any]) -> Optional[Thing]:
    if not rid: return None
    tp = onto.TextProvenance(f"TextProv_{safe_id(rid)}")
    set_dp(onto, tp, "record_id", rid)
    if extra.get("Text_source"): set_dp(onto, tp, "Text_source", extra["Text_source"])
    if extra.get("record_confidence"): set_dp(onto, tp, "record_confidence", extra["record_confidence"])
    if extra.get("Text"): set_dp(onto, tp, "Text_body", extra["Text"])
    return tp

# =============== 从 Neo4j 拉取并写入 OWL（事件化 + 溯源） ===============
APPOINTMENT_EVENT_WRITE_KEYS = {"迁转","授予时间"}  # 明确排除“授予类型”

def _event_add_person(e: Thing, p: Thing):
    """把人物 p 记录到事件 e 的内存索引中（供后续查询/去重/导出使用）。"""
    if p is None: return
    lst = EVENT_PERSONS.get(e, [])
    if p not in lst:
        lst.append(p)
        EVENT_PERSONS[e] = lst

def import_edges_eventify(onto, rows_edges: List[Dict[str,Any]], tp_map: Dict[str,Dict[str,Any]]):
    def add_event_pa(e, rprops):
        rid = rprops.get("record_id")
        bid = rprops.get("batch_id")
        for k,v in (rprops or {}).items():
            if k in {"record_id","batch_id","node_uid"}:
                continue
            if k in PROV_SKIP_KEYS:
                continue
            if v in (None,""): continue
            pa = onto.PropAssertion(f"PA_{hashlib.md5((e.name+k+str(v)+str(rid)).encode()).hexdigest()[:10]}")
            set_dp(onto, pa, "prop", k)
            set_dp(onto, pa, "value", str(v))
            set_dp(onto, pa, "value_norm", canon_text(v))
            if rid: set_dp(onto, pa, "record_id", rid)
            if bid: set_dp(onto, pa, "batch_id", bid)
            try: onto.about[pa] = [e]
            except: pass
            tp_extra = tp_map.get(rid, {}) if rid else {}
            tp = get_or_create_textprov(onto, rid, tp_extra)
            if tp:
                try: onto.derivedFrom[pa] = [tp]
                except: pass
                try:
                    cur = list(onto.contains[tp]) if onto.contains[tp] else []
                    if e not in cur: onto.contains[tp] = cur + [e]
                except: pass

    for row in rows_edges:
        rtype = norm_rel(row["rel_type"])
        if rtype not in REL_TYPES: continue

        s_labels = list(norm_labels(row["s_labels"]))
        t_labels = list(norm_labels(row["t_labels"]))
        s_props  = norm_props(row["s_props"])
        t_props  = norm_props(row["t_props"])
        rprops   = norm_props(row["rprops"])
        sid, tid = row["sid"], row["tid"]

        # 事件溯源兜底：从两端补 record_id/batch_id
        if not rprops.get("record_id"):
            rprops["record_id"] = s_props.get("record_id") or t_props.get("record_id")
        if not rprops.get("batch_id"):
            rprops["batch_id"] = s_props.get("batch_id") or t_props.get("batch_id")

        # 构造两端个体（并挂节点数据属性）
        s_ind = build_node(onto, s_labels, s_props, sid)
        t_ind = build_node(onto, t_labels, t_props, tid)

        # 把实体也纳入 contains(TextProvenance)
        for ent, eprops in [(s_ind, s_props), (t_ind, t_props)]:
            rid = eprops.get("record_id")
            if not rid: continue
            tp = get_or_create_textprov(onto, rid, tp_map.get(rid, {}))
            if tp:
                try:
                    cur = list(onto.contains[tp]) if onto.contains[tp] else []
                    if ent not in cur: onto.contains[tp] = cur + [ent]
                except: pass

        # 事件实例 + 业务属性 + 事件 PropAssertion
        if rtype == "参与":
            e = onto.ParticipationEvent(f"Evt_{hashlib.md5((sid+tid+json.dumps(rprops,ensure_ascii=False)).encode()).hexdigest()[:10]}")
            try: onto.hasExam[e] = [t_ind]
            except: pass
            for k in ("甲第等级","考中年龄","名次","功名"):
                if rprops.get(k): set_dp(onto, e, k, rprops.get(k))
            # 人物→事件
            if isinstance(s_ind, onto.Person):
                op_append(onto.participatesIn, s_ind, e)
                _event_add_person(e, s_ind)
            add_event_pa(e, rprops)

        elif rtype == "担任":
            e = onto.AppointmentEvent(f"Evt_{hashlib.md5((sid+tid+json.dumps(rprops,ensure_ascii=False)).encode()).hexdigest()[:10]}")
            if isinstance(t_ind, onto.OfficialPosition):
                try: onto.hasPosition[e] = [t_ind]
                except: pass
            for k in APPOINTMENT_EVENT_WRITE_KEYS:
                if rprops.get(k): set_dp(onto, e, k, rprops.get(k))
            if isinstance(s_ind, onto.Person):
                op_append(onto.appointedIn, s_ind, e)
                _event_add_person(e, s_ind)
            add_event_pa(e, rprops)

        elif rtype == "社会关系":
            e = onto.SocialRelationEvent(f"Evt_{hashlib.md5((sid+tid+json.dumps(rprops,ensure_ascii=False)).encode()).hexdigest()[:10]}")
            # 关系类型值保留在事件数据属性
            if rprops.get("关系类型"): set_dp(onto, e, "关系类型", rprops.get("关系类型"))
            # 两端如为人物都挂上 人物→事件
            if isinstance(s_ind, onto.Person):
                op_append(onto.socialRelationEvent, s_ind, e)
                _event_add_person(e, s_ind)
            if isinstance(t_ind, onto.Person):
                op_append(onto.socialRelationEvent, t_ind, e)
                _event_add_person(e, t_ind)
            add_event_pa(e, rprops)

        elif rtype == "生":
            e = onto.BirthEvent(f"Evt_{hashlib.md5((sid+tid+json.dumps(rprops,ensure_ascii=False)).encode()).hexdigest()[:10]}")
            try: onto.hasPlace[e] = [t_ind]
            except: pass
            if isinstance(s_ind, onto.Person):
                op_append(onto.bornInEvent, s_ind, e)
                _event_add_person(e, s_ind)
            add_event_pa(e, rprops)

        elif rtype == "任务执行":
            e = onto.TaskEvent(f"Evt_{hashlib.md5((sid+tid+json.dumps(rprops,ensure_ascii=False)).encode()).hexdigest()[:10]}")
            try: onto.hasPlace[e] = [t_ind]
            except: pass
            for k in ("职事名目","任务时间"):
                if rprops.get(k): set_dp(onto, e, k, rprops.get(k))
            if isinstance(s_ind, onto.Person):
                op_append(onto.performedTaskIn, s_ind, e)
                _event_add_person(e, s_ind)
            add_event_pa(e, rprops)

        elif rtype == "职任":
            e = onto.OfficeAtPlaceEvent(f"Evt_{hashlib.md5((sid+tid+json.dumps(rprops,ensure_ascii=False)).encode()).hexdigest()[:10]}")
            try: onto.hasPlace[e] = [t_ind]
            except: pass
            if rprops.get("任职时间"): set_dp(onto, e, "任职时间", rprops.get("任职时间"))
            if isinstance(s_ind, onto.Person):
                op_append(onto.heldOfficeAtEvent, s_ind, e)
                _event_add_person(e, s_ind)
            add_event_pa(e, rprops)

        elif rtype == "隶属":
            e = onto.SubordinationEvent(f"Evt_{hashlib.md5((sid+tid+json.dumps(rprops,ensure_ascii=False)).encode()).hexdigest()[:10]}")
            try:
                onto.hasPlace[e] = [s_ind]       # 下级地
                onto.hasSuperPlace[e] = [t_ind]  # 上级地
            except: pass
            add_event_pa(e, rprops)

# =============== 导入 PropAssertion （Neo4j → OWL） ===============
def import_propassertions(onto, rows_pa: List[Dict[str,Any]], tp_map: Dict[str,Dict[str,Any]]):
    for row in rows_pa:
        aprops = norm_props(row["aprops"] or {})
        prop_key = norm_prop(aprops.get("prop"))
        if prop_key in PROV_SKIP_KEYS:
            continue
        val      = aprops.get("value")
        val_norm = aprops.get("value_norm")
        rid      = aprops.get("record_id") or row.get("tp_record_id")
        bid      = aprops.get("batch_id")
        nid      = row.get("nid")

        n_labels = list(norm_labels(row["n_labels"]))
        n_props  = norm_props(row["n_props"])

        # 先按 neo4j_id 匹配（若已有），否则新建
        target = None
        for inst in onto.individuals():
            try:
                if list_vals(onto, inst, "neo4j_id") and list_vals(onto, inst, "neo4j_id")[0] == nid:
                    target = inst; break
            except: pass
        if target is None:
            target = build_node(onto, n_labels, n_props, nid)

        # 新建 PropAssertion 个体
        pa = onto.PropAssertion(f"PA_{hashlib.md5((target.name+prop_key+str(val)+str(rid)).encode()).hexdigest()[:10]}")
        if prop_key: set_dp(onto, pa, "prop", prop_key)
        if val: set_dp(onto, pa, "value", val)
        if val_norm: set_dp(onto, pa, "value_norm", val_norm)
        if rid: set_dp(onto, pa, "record_id", rid)
        if bid: set_dp(onto, pa, "batch_id", bid)
        if aprops.get("node_uid"): set_dp(onto, pa, "node_uid", aprops.get("node_uid"))

        try: onto.about[pa] = [target]
        except: pass

        # derivedFrom + contains
        tp_extra = tp_map.get(rid, {}) if rid else {}
        tp = get_or_create_textprov(onto, rid, tp_extra)
        if tp:
            try: onto.derivedFrom[pa] = [tp]
            except: pass
            try:
                cur = list(onto.contains[tp]) if onto.contains[tp] else []
                if target not in cur: onto.contains[tp] = cur + [target]
            except: pass

# =============== 合并工具（个体） ===============
def merge_two_individuals(onto, survivor: Thing, victim: Thing):
    # 数据属性合并
    for dp in onto.data_properties():
        try:
            vs = list(dp[survivor]) if dp[survivor] else []
            vv = list(dp[victim]) if dp[victim] else []
            if vv:
                merged = list(dict.fromkeys(vs + vv))
                dp[survivor] = merged
        except: pass
    # 出边合并（注意此处仅剩：hasPlace/hasSuperPlace/hasExam/hasPosition/人物→事件 + 溯源）
    for op in onto.object_properties():
        try:
            outs_s = list(op[survivor]) if op[survivor] else []
            outs_v = list(op[victim]) if op[victim] else []
            if outs_v:
                merged = outs_s[:]
                for x in outs_v:
                    if x not in merged and x is not survivor:
                        merged.append(x)
                op[survivor] = merged
        except: pass
    # 入边改向（包含 about/derivedFrom/contains、以及 人物→事件 等）
    inds = list(onto.individuals())
    for op in onto.object_properties():
        for s in inds:
            try: lst = list(op[s]) if op[s] else []
            except: lst = []
            if victim in lst:
                repl = [survivor if x is victim else x for x in lst]
                uniq=[]; [uniq.append(x) for x in repl if x not in uniq]
                try: op[s] = uniq
                except: pass
    try: destroy_entity(victim)
    except: pass

# =============== 分桶 ===============
def buckets_person(onto):
    m=defaultdict(list)
    for p in onto.Person.instances():
        nm=get_first(onto,p,"姓名")
        if nm: m[canon_text(nm)].append(p)
    return m

def buckets_place(onto):
    m=defaultdict(list)
    for x in onto.Place.instances():
        hn=get_first(onto,x,"历史名称")
        if hn: m[canon_text(hn)].append(x)
    return m

def buckets_position(onto):
    m=defaultdict(list)
    for x in onto.OfficialPosition.instances():
        t=get_first(onto,x,"官职名称")
        if t: m[canon_text(t)].append(x)
    return m

def buckets_exam_full(onto):
    m=defaultdict(list)
    for x in onto.ImperialExam.instances():
        lv=get_first(onto,x,"考试等级"); tm=get_first(onto,x,"考试时间")
        if lv and tm: m[(canon_text(lv),canon_text(tm))].append(x)
    return m

def merge_by_buckets(onto, m):
    cnt=0
    for k,inds in m.items():
        if len(inds)<2: continue
        s,*rest=inds
        for v in rest:
            merge_two_individuals(onto,s,v); cnt+=1
    return cnt

# =============== “人域无时间”科举合并（基于人物→事件） ===============
def persons_of_event(e: Thing)->List[Thing]:
    return EVENT_PERSONS.get(e, [])

def first_person_of_event(e: Thing)->Optional[Thing]:
    lst = EVENT_PERSONS.get(e, [])
    return lst[0] if lst else None

def consolidate_person_exam_no_time(onto)->int:
    merged=0
    for p in list(onto.Person.instances()):
        # 此人的所有“参与”事件
        try:
            evs = list(onto.participatesIn[p]) if onto.participatesIn[p] else []
        except: evs=[]
        if not evs: continue
        concrete=defaultdict(list)  # lvl -> [exam(有时间)]
        shadow=defaultdict(list)    # lvl -> [exam(无时间)]
        ev_by_exam=defaultdict(list)
        for e in evs:
            try: exs = onto.hasExam[e] or []
            except: exs=[]
            if not exs: continue
            ex=exs[0]; ev_by_exam[ex].append(e)
            lv=get_first(onto,ex,"考试等级"); tm=get_first(onto,ex,"考试时间")
            if not lv: continue
            if tm and canon_text(tm): concrete[canon_text(lv)].append(ex)
            else: shadow[canon_text(lv)].append(ex)

        for lvl in set(list(concrete.keys()) + list(shadow.keys())):
            sh_list=shadow.get(lvl,[])
            if not sh_list: continue
            if concrete.get(lvl):
                surv=concrete[lvl][0]
                for sh in list(dict.fromkeys(sh_list)):
                    for e in ev_by_exam.get(sh, []):
                        try: onto.hasExam[e] = [surv]
                        except: pass
                    merge_two_individuals(onto, surv, sh); merged+=1
            else:
                uniq_sh=list(dict.fromkeys(sh_list))
                surv,*rest=uniq_sh
                for sh in rest:
                    for e in ev_by_exam.get(sh, []):
                        try: onto.hasExam[e] = [surv]
                        except: pass
                    merge_two_individuals(onto, surv, sh); merged+=1
    return merged

# =============== 校验：同人同等级多时间（基于人物→事件） ===============
def validate_person_level_uniqueness(onto)->int:
    rows=[]
    for p in onto.Person.instances():
        try:
            evs = list(onto.participatesIn[p]) if onto.participatesIn[p] else []
        except: evs=[]
        if not evs: continue
        lvl2times=defaultdict(set)
        for e in evs:
            try: ex=onto.hasExam[e][0] if onto.hasExam[e] else None
            except: ex=None
            if not ex: continue
            lv=get_first(onto,ex,"考试等级"); tm=get_first(onto,ex,"考试时间")
            if not lv: continue
            lvl2times[canon_text(lv)].add(canon_text(tm) if tm else "")
        for lvl, ts in lvl2times.items():
            non_empty={t for t in ts if t}
            if len(non_empty)>=2:
                rows.append([get_first(onto,p,"姓名") or p.name, lvl, "|".join(sorted(non_empty))])
    path=os.path.join(OUT_DIR,"violations_person_exam_level_multi.csv")
    with open(path,"w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerows([["人物","考试等级","不同时间集"]]+rows)
    return len(rows)

# =============== 事件去重（基于人物→事件） ===============
def event_key(onto, e):
    et = next((c.name for c in e.is_a if isinstance(c, ThingClass) and c.name.endswith("Event")), "Event")

    def one(op):
        try: return op[e][0] if op[e] else None
        except: return None

    persons = persons_of_event(e)
    p  = persons[0] if persons else None
    p2 = persons[1] if len(persons) > 1 else None

    PL = one(onto.hasPlace);  SP = one(onto.hasSuperPlace)
    EX = one(onto.hasExam);   PO = one(onto.hasPosition)

    pid  = p.name  if p  else ""
    p2id = p2.name if p2 else ""
    plid = PL.name if PL else ""
    spid = SP.name if SP else ""
    exid = EX.name if EX else ""
    poid = PO.name if PO else ""

    if et=="ParticipationEvent": return ("ParticipationEvent", pid, exid)
    if et=="AppointmentEvent":   return ("AppointmentEvent", pid, poid)
    if et=="SocialRelationEvent":return ("SocialRelationEvent", pid, p2id, canon_text(get_first(onto,e,"关系类型")))
    if et=="BirthEvent":         return ("BirthEvent", pid, plid)
    if et=="TaskEvent":          return ("TaskEvent", pid, plid, canon_text(get_first(onto,e,"职事名目")), canon_text(get_first(onto,e,"任务时间")))
    if et=="OfficeAtPlaceEvent": return ("OfficeAtPlaceEvent", pid, plid, canon_text(get_first(onto,e,"任职时间")))
    if et=="SubordinationEvent": return ("SubordinationEvent", plid, spid)
    return ("Event", e.name)

def dedup_events(onto)->int:
    m=defaultdict(list); merged=0
    for e in onto.Event.instances(): m[event_key(onto,e)].append(e)
    for k, es in m.items():
        if len(es)<=1: continue
        s,*rest = es
        for v in rest:
            merge_two_individuals(onto, s, v); merged+=1
    return merged

# =============== 导出 ===============
PERSON_DPS = ["姓名","字","生年","户籍类型","户籍地","学籍","家庭排行","学术专长"]
PLACE_DPS  = ["历史名称","现代名称","现代区划层级","经纬坐标"]
EXAM_DPS   = ["考试等级","考试时间","考试地点","主考官","科举政策"]
POSI_DPS   = ["官职名称","官阶"]

def write_csv(path, header, rows):
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

def export_entities(onto, tag):
    rows=[]
    for p in onto.Person.instances():
        rows.append([get_first(onto,p,"姓名") or p.name] + ["|".join(list_vals(onto,p,k)) for k in PERSON_DPS])
    write_csv(os.path.join(OUT_DIR,f"persons_{tag}.csv"), ["显示名"]+PERSON_DPS, rows)

    rows=[]
    for x in onto.Place.instances():
        disp = get_first(onto,x,"历史名称") or get_first(onto,x,"现代名称") or x.name
        rows.append([disp] + ["|".join(list_vals(onto,x,k)) for k in PLACE_DPS])
    write_csv(os.path.join(OUT_DIR,f"places_{tag}.csv"), ["显示名"]+PLACE_DPS, rows)

    rows=[]
    for x in onto.ImperialExam.instances():
        disp = (get_first(onto,x,"考试等级") or "") + (" "+get_first(onto,x,"考试时间") if get_first(onto,x,"考试时间") else "")
        rows.append([disp.strip() or x.name] + ["|".join(list_vals(onto,x,k)) for k in EXAM_DPS])
    write_csv(os.path.join(OUT_DIR,f"exams_{tag}.csv"), ["显示名"]+EXAM_DPS, rows)

    rows=[]
    for x in onto.OfficialPosition.instances():
        disp = get_first(onto,x,"官职名称") or x.name
        rows.append([disp] + ["|".join(list_vals(onto,x,k)) for k in POSI_DPS])
    write_csv(os.path.join(OUT_DIR,f"positions_{tag}.csv"), ["显示名"]+POSI_DPS, rows)

def export_events_table(onto, tag):
    def one(op,e):
        try: return op[e][0] if op[e] else None
        except: return None
    rows=[]
    for e in onto.Event.instances():
        et = next((c.name for c in e.is_a if isinstance(c, ThingClass) and c.name.endswith("Event")), "Event")
        persons = persons_of_event(e)
        p   = persons[0] if persons else None
        p2  = persons[1] if len(persons)>1 else None
        pl  = one(onto.hasPlace,e);  spl = one(onto.hasSuperPlace,e)
        ex  = one(onto.hasExam,e);   op  = one(onto.hasPosition,e)
        dps={}
        for dp in onto.__dict__.values():
            if isinstance(dp, DataPropertyClass):
                try:
                    vals=dp[e]
                    if vals: dps[dp.name]="|".join(map(str,vals))
                except: pass
        rows.append([
            e.name, et,
            (get_first(onto,p,"姓名") if p else ""),
            (get_first(onto,p2,"姓名") if p2 else ""),
            (get_first(onto,pl,"历史名称") or get_first(onto,pl,"现代名称") if pl else ""),
            (get_first(onto,spl,"历史名称") or get_first(onto,spl,"现代名称") if spl else ""),
            ((get_first(onto,ex,"考试等级") or "") + (" "+get_first(onto,ex,"考试时间") if ex and get_first(onto,ex,"考试时间") else "")).strip(),
            (get_first(onto,op,"官职名称") if op else ""),
            json.dumps(dps, ensure_ascii=False)
        ])
    write_csv(os.path.join(OUT_DIR,f"events_{tag}.csv"),
              ["事件ID","事件类型","人","另一人","地点","上级地","科举","官职","事件属性JSON"], rows)

def save_multi_formats(onto, prefix):
    for fmt,fn in [("turtle",f"ontology_{prefix}.ttl"),("rdfxml",f"ontology_{prefix}.rdf"),("ntriples",f"ontology_{prefix}.nt")]:
        try:
            onto.save(file=os.path.join(OUT_DIR,fn), format=fmt)
            print("   保存：", fn)
        except Exception as e:
            print("!! 保存失败：", fmt, repr(e))

def collect_graph_eventful(onto):
    G=nx.DiGraph()
    def add(inst, typ):
        if inst is None: return
        nid=inst.name
        if nid not in G: G.add_node(nid,label=nid,type=typ)
    def link(a,b,label):
        if a is None or b is None: return
        G.add_edge(a.name,b.name,label=label)

    for p in onto.Person.instances(): add(p,"Person")
    for pl in onto.Place.instances(): add(pl,"Place")
    for ex in onto.ImperialExam.instances(): add(ex,"ImperialExam")
    for op in onto.OfficialPosition.instances(): add(op,"OfficialPosition")
    for tp in onto.TextProvenance.instances(): add(tp,"TextProv")
    for pa in onto.PropAssertion.instances(): add(pa,"PropAssertion")

    for e in onto.Event.instances():
        add(e,"Event")
        try:
            for x in onto.hasPlace[e] or []: add(x,"Place"); link(e,x,"hasPlace")
        except: pass
        try:
            for x in onto.hasSuperPlace[e] or []: add(x,"Place"); link(e,x,"hasSuperPlace")
        except: pass
        try:
            for x in onto.hasExam[e] or []: add(x,"ImperialExam"); link(e,x,"hasExam")
        except: pass
        try:
            for x in onto.hasPosition[e] or []: add(x,"OfficialPosition"); link(e,x,"hasPosition")
        except: pass

    # 溯源边
    for pa in onto.PropAssertion.instances():
        try:
            for n in onto.about[pa] or []: add(n,"Any"); link(pa,n,"about")
        except: pass
        try:
            for t in onto.derivedFrom[pa] or []: add(t,"TextProv"); link(pa,t,"derivedFrom")
        except: pass

    for tp in onto.TextProvenance.instances():
        try:
            for n in onto.contains[tp] or []: add(n,"Any"); link(tp,n,"contains")
        except: pass

    # 人物 → 事件边（新增保留）
    for p in onto.Person.instances():
        add(p, "Person")
        try:
            for e in onto.participatesIn[p] or []:
                add(e,"Event"); link(p, e, "participatesIn")
        except: pass
        try:
            for e in onto.appointedIn[p] or []:
                add(e,"Event"); link(p, e, "appointedIn")
        except: pass
        try:
            for e in onto.socialRelationEvent[p] or []:
                add(e,"Event"); link(p, e, "socialRelationEvent")
        except: pass
        try:
            for e in onto.bornInEvent[p] or []:
                add(e,"Event"); link(p, e, "bornInEvent")
        except: pass
        try:
            for e in onto.performedTaskIn[p] or []:
                add(e,"Event"); link(p, e, "performedTaskIn")
        except: pass
        try:
            for e in onto.heldOfficeAtEvent[p] or []:
                add(e,"Event"); link(p, e, "heldOfficeAtEvent")
        except: pass

    return G

def collect_graph_projected(onto):
    G=nx.DiGraph()
    def add(inst, typ):
        if inst is None: return
        nid=inst.name
        if nid not in G: G.add_node(nid,label=nid,type=typ)
    def elink(a,b,label):
        if a is None or b is None: return
        u,v=a.name,b.name
        if G.has_edge(u,v):
            lab=G[u][v].get("label","")
            if label not in lab: G[u][v]["label"]=(lab+"|"+label) if lab else label
        else:
            G.add_edge(u,v,label=label)

    for p in onto.Person.instances(): add(p,"Person")
    for pl in onto.Place.instances(): add(pl,"Place")
    for ex in onto.ImperialExam.instances(): add(ex,"ImperialExam")
    for op in onto.OfficialPosition.instances(): add(op,"OfficialPosition")

    def one(op, e):
        try: return op[e][0] if op[e] else None
        except: return None

    for e in onto.Event.instances():
        et = next((c.name for c in e.is_a if isinstance(c, ThingClass) and c.name.endswith("Event")), "Event")
        persons = persons_of_event(e)
        p   = persons[0] if persons else None
        p2  = persons[1] if len(persons)>1 else None
        pl  = one(onto.hasPlace,e);  spl = one(onto.hasSuperPlace,e)
        xx  = one(onto.hasExam,e);   po  = one(onto.hasPosition,e)
        if et=="ParticipationEvent" and p and xx: elink(p,xx,"参与")
        elif et=="AppointmentEvent" and p and po: elink(p,po,"担任")
        elif et=="SocialRelationEvent" and p and p2: elink(p,p2,"社会关系")
        elif et=="BirthEvent" and p and pl: elink(p,pl,"生")
        elif et=="TaskEvent" and p and pl: elink(p,pl,"任务执行")
        elif et=="OfficeAtPlaceEvent" and p and pl: elink(p,pl,"职任")
        elif et=="SubordinationEvent" and pl and spl: elink(pl,spl,"隶属")
    return G

def draw_png(G, path):
    pos = nx.spring_layout(G, seed=42, k=0.6)
    plt.figure(figsize=(14,10))
    nx.draw(G, pos, with_labels=True, font_size=7, node_size=600, width=0.8, arrows=True)
    edge_labels=nx.get_edge_attributes(G,"label")
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_size=7)
    plt.axis("off"); plt.tight_layout(); plt.savefig(path,dpi=200); plt.close()

def export_graphs(onto, tag="after"):
    Ge = collect_graph_eventful(onto)
    Gp = collect_graph_projected(onto)
    p1 = os.path.join(OUT_DIR, f"graph_eventful_{tag}.png")
    p2 = os.path.join(OUT_DIR, f"graph_projected_{tag}.png")
    draw_png(Ge, p1); draw_png(Gp, p2)
    nx.write_graphml(Ge, os.path.join(OUT_DIR, f"graph_eventful_{tag}.graphml"))
    nx.write_graphml(Gp, os.path.join(OUT_DIR, f"graph_projected_{tag}.graphml"))
    print("   可视化：", p1, p2)

def export_dashboard_html(stats: Dict[str, Any]):
    path=os.path.join(OUT_DIR,"index.html")
    html=f"""<!doctype html><meta charset="utf-8"><title>OWL 消歧 + 事件 + 溯源</title>
<style>body{{font-family:system-ui,Segoe UI,Arial;margin:24px}}.card{{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:12px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}.kv{{display:flex;justify-content:space-between;margin:6px 0}}
.kv span:first-child{{color:#6b7280}}a.file{{display:inline-block;margin-right:12px}}</style>
<h1>OWL 消歧 + 事件合并 + 溯源</h1>
<div class="grid">
<div class="card"><h3>合并前</h3>
<div class="kv"><span>个体总数</span><b>{stats['before_n_individuals']}</b></div>
<div class="kv"><span>事件总数</span><b>{stats['before_n_events']}</b></div>
<div class="kv"><span>人物总数</span><b>{stats['before_n_persons']}</b></div>
</div>
<div class="card"><h3>合并后</h3>
<div class="kv"><span>个体总数</span><b>{stats['after_n_individuals']}</b></div>
<div class="kv"><span>事件总数</span><b>{stats['after_n_events']}</b></div>
<div class="kv"><span>人物总数</span><b>{stats['after_n_persons']}</b></div>
<div class="kv"><span>合并事件数</span><b>{stats['merged_events']}</b></div>
<div class="kv"><span>合并科举(无时间)</span><b>{stats['merged_exam_no_time']}</b></div>
<div class="kv"><span>同等级多场违规</span><b>{stats['violations_person_level_multi']}</b></div>
<div class="kv"><span>TextProvenance 个体</span><b>{stats['n_textprov']}</b></div>
<div class="kv"><span>PropAssertion 个体</span><b>{stats['n_propassert']}</b></div>
</div>
<div class="card"><h3>导出文件</h3>
<div>
  <a class="file" href="./ontology_raw.ttl">ontology_raw.ttl</a>
  <a class="file" href="./ontology_dedup.ttl">ontology_dedup.ttl</a>
  <a class="file" href="./persons_before.csv">persons_before.csv</a>
  <a class="file" href="./persons_after.csv">persons_after.csv</a>
  <a class="file" href="./places_before.csv">places_before.csv</a>
  <a class="file" href="./places_after.csv">places_after.csv</a>
  <a class="file" href="./exams_before.csv">exams_before.csv</a>
  <a class="file" href="./exams_after.csv">exams_after.csv</a>
  <a class="file" href="./positions_before.csv">positions_before.csv</a>
  <a class="file" href="./positions_after.csv">positions_after.csv</a>
  <a class="file" href="./events_before.csv">events_before.csv</a>
  <a class="file" href="./events_after.csv">events_after.csv</a>
  <a class="file" href="./violations_person_exam_level_multi.csv">violations_person_exam_level_multi.csv</a>
  <a class="file" href="./graph_eventful_after.png">graph_eventful_after.png</a>
  <a class="file" href="./graph_projected_after.png">graph_projected_after.png</a>
</div>
</div>
</div>"""
    with open(path,"w",encoding="utf-8") as f: f.write(html)
    return path

# =============== 主流程 ===============
def main():
    print(">> 从 Neo4j 读取：关系/PropAssertion/TextProvenance ...")
    rows_edges = neo_rows(NEO4J_URI, NEO4J_USER, NEO4J_PWD, Q_FETCH_EDGES)
    rows_pa    = neo_rows(NEO4J_URI, NEO4J_USER, NEO4J_PWD, Q_FETCH_PA)
    rows_tp    = neo_rows(NEO4J_URI, NEO4J_USER, NEO4J_PWD, Q_FETCH_TP)

    TP_MAP = {}
    for r in rows_tp:
        rid = r.get("record_id")
        if not rid: continue
        TP_MAP[rid] = {
            "Text_source": r.get("Text_source") or "",
            "record_confidence": r.get("record_confidence") or "",
            "Text": r.get("Text") or "",
        }

    print(f"   关系列：{len(rows_edges)}，PropAssertion：{len(rows_pa)}，TextProvenance：{len(TP_MAP)}")

    print(">> 构建 OWL 本体骨架 ...")
    onto = build_onto()

    print(">> 事件化 + 事件属性 → 事件 PropAssertion + contains ...")
    import_edges_eventify(onto, rows_edges, TP_MAP)

    print(">> 导入节点属性 PropAssertion（来自 Neo4j） ...")
    import_propassertions(onto, rows_pa, TP_MAP)

    if RUN_REASONER_BEFORE:
        print(">> 推理（合并前） ...")
        try:
            sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
            print("   推理完成")
        except Exception as e:
            print("!! 推理失败：", repr(e))

    print(">> 合并前导出 ...")
    export_entities(onto, "before")
    export_events_table(onto, "before")
    save_multi_formats(onto, "raw")

    before_stats = {
        "n_individuals": len(list(onto.individuals())),
        "n_events":      len(list(onto.Event.instances())),
        "n_persons":     len(list(onto.Person.instances())),
    }

    # ===== 实体消歧 =====
    print(">> 消歧：人物（姓名）"); merge_by_buckets(onto, buckets_person(onto))
    print(">> 消歧：地点（历史名称）"); merge_by_buckets(onto, buckets_place(onto))
    print(">> 消歧：官职（官职名称）"); merge_by_buckets(onto, buckets_position(onto))
    print(">> 消歧：科举（等级+时间齐全方合并）"); merge_by_buckets(onto, buckets_exam_full(onto))

    print(">> 科举（无考试时间）人域整合 ...")
    merged_exam_no_time = consolidate_person_exam_no_time(onto)

    print(">> 校验：同人同等级多场不同时间（仅输出清单） ...")
    n_violations = validate_person_level_uniqueness(onto)

    print(">> 事件去重（参与=同人同考；担任=同人同官） ...")
    merged_events = dedup_events(onto)

    if RUN_REASONER_AFTER:
        print(">> 合并后推理 ...")
        try:
            sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
            print("   推理完成")
        except Exception as e:
            print("!! 推理失败：", repr(e))

    print(">> 合并后导出 ...")
    export_entities(onto, "after")
    export_events_table(onto, "after")
    save_multi_formats(onto, "dedup")
    export_graphs(onto, "after")

    stats = {
        "before_n_individuals": before_stats["n_individuals"],
        "before_n_events":      before_stats["n_events"],
        "before_n_persons":     before_stats["n_persons"],
        "after_n_individuals":  len(list(onto.individuals())),
        "after_n_events":       len(list(onto.Event.instances())),
        "after_n_persons":      len(list(onto.Person.instances())),
        "merged_events":        merged_events,
        "merged_exam_no_time":  merged_exam_no_time,
        "violations_person_level_multi": n_violations,
        "n_textprov":           len(list(onto.TextProvenance.instances())),
        "n_propassert":         len(list(onto.PropAssertion.instances())),
    }
    export_dashboard_html(stats)

    print("\n>> 完成。输出目录：", os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()
