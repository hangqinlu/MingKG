# -*- coding: utf-8 -*-
"""
明代职官称谓 → 原子属性 解析器（RTL核心优先版 | 右→左优先识别核心 | 别名安全化）
修复要点：
1) 右→左优先锁定“核心职称”，再回溯识别机构/地名/方位；
2) 两司尾缀优先：(左|右)?(参政|参议|副使|佥事|提学|兵备)$；
3) 别名安全化：移除会造成串内级联替换的别名（如“郎→郎中”“给事→给事中”“都给→都给事中”“员外/外郎→员外郎”），
   改为在命中后由 normalize_core_from_hit 精准归一；
4) 新增命中词：'外郎'、'员外'、'郎'，并在规范化中映射为：外郎/员外→员外郎，郎→郎中；
   彻底解决“员外郎”被误识别为“郎中”的问题。
"""

import re
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

# ========= 机构缩写 =========
INSTITUTION_ABBR = {
    "都察院": "JCY", "吏部": "LIBU", "户部": "HUBU", "礼部": "LIPU",
    "兵部": "BINGBU", "刑部": "XINGBU", "工部": "GONGBU",
    "太常寺": "TAICHANGSI", "光禄寺": "GUANGLUSI", "太仆寺": "TAIPUSI",
    "大理寺": "DALISI", "鸿胪寺": "HONGLUSI", "通政司": "TONGZHENGSI",
    "宗人府": "ZONGRENFU", "詹事府": "ZHANSHIFU", "国子监": "GUOZIJIAN",
    "布政使司": "BUZHENGSI", "按察使司": "ANCHASI",
    "五军都督府": "WJDD", "都督府": "DUDUFU",
    "都指挥使司": "DUZHIHUISI", "卫所": "WEISUO",
    "锦衣卫": "JINYIWEI",
    "南京都察院": "NJCY", "南京吏部": "NJLIBU", "南京户部": "NJHUBU", "南京礼部": "NJLIPU",
    "南京兵部": "NJBINGBU", "南京刑部": "NJXINGBU", "南京工部": "NJGONGBU",
    "南京国子监": "NJGZJ",
    "司礼监": "SILIJIAN", "御马监": "YUMAJIAN", "东厂": "DONGCHANG",
    "西厂": "XICHANG", "内官监": "NEIGUANJIAN",
    "散官": "SANGUAN", "勋官": "XUNGUAN",
    "盐运司": "YANYUNSI", "盐课司": "YANKESI",
    # 六科具体科名
    "吏科": "LIKE", "户科": "HUKE", "礼科": "LIKE_LI",
    "兵科": "BINGKE", "刑科": "XINGKE", "工科": "GONGKE",
}

# ========= 职系映射 =========
FAMILY_MAP = {
    # 六部系
    "尚书": "六部系", "侍郎": "六部系", "郎中": "六部系", "员外郎": "六部系", "主事": "六部系", "司务": "六部系",
    # 都察院系
    "都御史": "都察院系", "左都御史": "都察院系", "右都御史": "都察院系",
    "副都御史": "都察院系", "左副都御史": "都察院系", "右副都御史": "都察院系",
    "佥都御史": "都察院系", "左佥都御史": "都察院系", "右佥都御史": "都察院系",
    "监察御史": "都察院系", "御史": "都察院系", "巡按御史": "都察院系", "巡按": "都察院系",
    # 六科系
    "都给事中": "六科系", "左给事中": "六科系", "右给事中": "六科系",
    "给事中": "六科系", "给事": "六科系",
    # 通政司 / 大理寺
    "通政使": "通政司系", "通政": "通政司系", "左通政": "通政司系", "右通政": "通政司系", "参议": "通政司系",
    "大理寺卿": "大理寺系", "卿": "大理寺系", "少卿": "大理寺系", "寺丞": "大理寺系", "寺正": "大理寺系", "寺副": "大理寺系", "评事": "大理寺系",
    # 翰林院系
    "大学士": "翰林院系", "学士": "翰林院系", "侍读学士": "翰林院系", "侍讲学士": "翰林院系",
    "侍读": "翰林院系", "侍讲": "翰林院系", "修撰": "翰林院系", "编修": "翰林院系", "检讨": "翰林院系", "庶吉士": "翰林院系",
    # 地方文官
    "知府": "府系", "府同知": "府系", "通判": "府系", "推官": "府系",
    "知州": "州系", "州同知": "州系", "州判官": "州系",
    "知县": "县系", "县丞": "县系", "主簿": "县系", "典史": "县系",
    "教授": "县系", "训导": "县系", "学正": "县系", "教谕": "县系", "典籍": "县系", "典簿": "县系",
    # 都督/都指挥/卫所
    "都督": "都督府系", "左都督": "都督府系", "右都督": "都督府系", "都督同知": "都督府系", "都督佥事": "都督府系",
    "都指挥使": "都指挥使司系", "都指挥同知": "都指挥使司系", "都指挥佥事": "都指挥使司系",
    "指挥使": "卫所系", "指挥同知": "卫所系", "指挥佥事": "卫所系",
    "总兵官": "镇戍系", "副总兵": "镇戍系", "参将": "镇戍系", "游击将军": "镇戍系", "游击": "镇戍系", "守备": "镇戍系",
    # 省级两司四属
    "布政使": "承宣布政使司系", "参政": "承宣布政使司系", "参议": "承宣布政使司系",
    "按察使": "提刑按察使司系", "副使": "提刑按察使司系", "佥事": "提刑按察使司系",
    "提学": "提刑按察使司系", "兵备": "提刑按察使司系",
    # 盐务
    "盐运使": "盐运司系", "运同": "盐运司系", "运副": "盐运司系", "运判": "盐运司系",
    # 杂职
    "经历": "杂职系", "都事": "杂职系", "照磨": "杂职系", "检校": "杂职系", "司狱": "杂职系",
    "库大使": "杂职系", "仓大使": "杂职系", "税课大使": "杂职系", "织染大使": "杂职系", "河泊所官": "杂职系",
    "驿丞": "杂职系", "闸官": "杂职系",
}

# ========= 核心词典（右→左搜索池） =========
CORE_HINTS = sorted({
    # 御史完整
    "左都御史","右都御史","都御史",
    "左副都御史","右副都御史","副都御史",
    "左佥都御史","右佥都御史","佥都御史",
    "巡按御史","监察御史","御史","巡按",
    # 强优先
    "寺正","通政",
    # 六科/六部/九寺
    "都给事中","左给事中","右给事中","给事中","给事",
    "尚书","侍郎","郎中","员外郎","外郎","员外","郎","主事","司务",
    "太常寺卿","光禄寺卿","太仆寺卿","鸿胪寺卿","大理寺卿",
    "少卿","卿","寺丞","寺副","评事",
    "通政使","左通政","右通政",
    # 省级两司
    "布政使","参政","参议",
    "按察使","副使","佥事","提学","兵备",
    # 地方文官
    "知府","知州","知县","府同知","通判","县丞","主簿","典史","教谕","训导","典籍","典簿","推官","判官","州同知","州判官",
    # 军制
    "左都督","右都督","都督","都督同知","都督佥事",
    "都指挥使","都指挥同知","都指挥佥事",
    "指挥使","指挥同知","指挥佥事",
    "总兵官","副总兵","参将","游击将军","游击","守备",
    # 学术/翰林
    "大学士","学士","侍读学士","侍讲学士","侍读","侍讲","修撰","编修","检讨","庶吉士",
    "詹事","少詹事","谕德","赞善","洗马",
    "祭酒","司业","博士","助教","学正","学录",
    # 道职
    "兵备道","督粮道","驿传道","分守道","巡海道","提学道",
    # 盐务
    "盐运使","运同","运副","运判",
    # 杂职
    "经历","都事","照磨","检校","司狱","库大使","仓大使","税课大使","织染大使","河泊所官","驿丞","闸官",
    # 别名短语
    "通政","学政",
}, key=len, reverse=True)

# ========= 别名 → 规范（安全：只保留不会串内级联的别名） =========
CORE_ALIASES = {
    # 去掉：'郎'→'郎中' / '给事'→'给事中' / '都给'→'都给事中' / '员外'/'外郎'→'员外郎'
    "主政": "主事",
    "抚台": "巡抚", "抚军": "巡抚",
    "都堂": "都御史", "都爷": "都御史", "都宪": "都御史",
    "中丞": "副都御史", "侍御": "监察御史",
    "柱史": "御史", "绣衣": "御史", "按台": "监察御史", "巡按": "巡按御史",
    "银台": "通政使",
    "学政": "提学",
}

UNSPLIT_CORES = (
    "佥事","副使","少卿",
    "都督","都督同知","都督佥事",
    "都指挥使","都指挥同知","都指挥佥事",
    "指挥使","指挥佥事",
    "总兵官","副总兵","游击将军",
)

PLACE_SUFFIX = ("府","州","县","卫","所","路","司","厅","盐课司","盐运司")

REGION_PREFIXES = [
    "直隶","京师","北京","南京","顺天","宣府","大同","太原",
    "山东","山西","河南","陕西","浙江","江西","福建","湖北","湖南","湖广",
    "广东","广西","四川","云南","贵州","甘肃","两广","两湖",
]
REGION_PREFIX_RE = re.compile(r"^(" + "|".join(map(re.escape, REGION_PREFIXES)) + r")")
DIR_PREFIX_RE = re.compile(r'^(南台|左|右|前|后|中|东|西|南|北|内|外)')

# ========= 小工具 =========
def norm(s: str) -> str:
    if not s: return ""
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("（","(").replace("）",")").replace("　","")
    return s

def premap_alias_whole_string(s: str) -> Tuple[str, bool]:
    """
    仅替换安全别名，避免“给事/郎/员外/外郎”等引发的串内级联。
    """
    replaced = False
    for alias in sorted(CORE_ALIASES.keys(), key=len, reverse=True):
        if alias in s:
            s = s.replace(alias, CORE_ALIASES[alias])
            replaced = True
    return s, replaced

def abbr(inst: Optional[str]) -> Optional[str]:
    if not inst: return None
    return INSTITUTION_ABBR.get(inst, inst)

def mk_place_id(place: Optional[str]) -> Optional[str]:
    return f"Place#{place}" if place else None

def mk_capital_flag(dir_mod: Optional[str], place: Optional[str]) -> Optional[str]:
    if place == "南京" or dir_mod == "南台":
        return "NANJING"
    return None

def mk_g0_core(core_norm: Optional[str]) -> Optional[str]:
    return f"CORE/{core_norm}" if core_norm else None

def mk_g1_inst(inst: Optional[str], core_norm: Optional[str]) -> Optional[str]:
    return f"INST/{abbr(inst)}/{core_norm}" if (inst and core_norm) else None

def mk_g2_tier(inst: Optional[str], core_norm: Optional[str], tier_norm: Optional[str]) -> Optional[str]:
    return f"TIER/{abbr(inst)}/{core_norm}/{tier_norm}" if (inst and core_norm and tier_norm) else None

def mk_g3_loc_core(place: Optional[str], core_norm: Optional[str]) -> Optional[str]:
    return f"LOC_CORE/{mk_place_id(place)}/{core_norm}" if (place and core_norm) else None

def mk_g4_loc_inst(place: Optional[str], inst: Optional[str], core_norm: Optional[str]) -> Optional[str]:
    return f"LOC_INST/{mk_place_id(place)}/{abbr(inst)}/{core_norm}" if (place and inst and core_norm) else None

def mk_g5_loc_full(place: Optional[str], inst: Optional[str], core_norm: Optional[str],
                   tier_norm: Optional[str], dir_mod: Optional[str], dep_mod: Optional[str]) -> Optional[str]:
    if not (place and inst and core_norm and tier_norm): return None
    base = f"LOC_FULL/{mk_place_id(place)}/{abbr(inst)}/{core_norm}/{tier_norm}"
    flags = []
    if dir_mod and dir_mod not in ("南台",):
        flags.append(f"DIR={dir_mod}")
    if dep_mod:
        flags.append(f"DEP={dep_mod}")
    cap = mk_capital_flag(dir_mod, place)
    if cap:
        flags.append(f"CAPITAL={cap}")
    return base + ((";" + ";".join(flags)) if flags else "")

def extract_dir_from_tier(tier_text: str) -> Optional[str]:
    if not tier_text: return None
    m = DIR_PREFIX_RE.match(tier_text)
    return m.group(1) if m else None

# —— 命中词 → 核心规范（含安全归一） —— #
DIR_PREFIXES = ("左","右","前","后","中")
RANK_PREFIXES = ("都","佥")
NINE_TEMPLE = ("太常寺","光禄寺","太仆寺","鸿胪寺","大理寺")

def normalize_core_from_hit(hit: str) -> str:
    if not hit: return hit
    # 精准别名：只在“命中词本身”上归一，不在整串上替换
    if hit in {"外郎","员外"}:
        return "员外郎"
    if hit == "郎":
        return "郎中"

    if hit in UNSPLIT_CORES:
        return hit
    base = hit
    if base[:1] in DIR_PREFIXES:
        base = base[1:]
    if base[:1] in RANK_PREFIXES:
        tail = base[1:]
        if re.match(r"(?:都御史|副都御史|佥都御史|御史|给事中?|督|指挥|通政)", tail):
            base = tail
    if base == "都给事中":
        return "都给事中"
    if base.endswith("给事中") or base.endswith("给事"):
        return "给事中"
    for t in NINE_TEMPLE:
        if base == f"{t}卿":
            return "卿"
    return base

# ========= 右→左核心选择 =========
TWOSI_TAIL_RE = re.compile(r"(左|右)?(参政|参议|副使|佥事|提学|兵备)$")

def find_core_right_to_left(text: str) -> Tuple[Optional[str], Optional[str], Optional[Tuple[int,int]]]:
    """
    返回 (core, dir_mod, span)
    1) 先匹配两司尾缀：(左|右)?(参政|参议|副使|佥事|提学|兵备)$
    2) 否则在 CORE_HINTS 中寻找“最靠右”的命中；若结束位置相同取“更长”的
    """
    m = TWOSI_TAIL_RE.search(text)
    if m:
        dir_mod = m.group(1)
        core = m.group(2)
        return core, dir_mod, (m.start(2), m.end(2))

    best = None  # (end_idx, length, start_idx, word)
    for w in CORE_HINTS:
        idx = text.rfind(w)
        if idx >= 0:
            end_idx = idx + len(w)
            cand = (end_idx, len(w), idx, w)
            if (best is None) or (cand[0] > best[0]) or (cand[0] == best[0] and cand[1] > best[1]):
                best = cand
    if best:
        end_idx, length, start_idx, w = best
        return normalize_core_from_hit(w), None, (start_idx, end_idx)
    return None, None, None

# ========= 层级（通用标签） =========
def infer_universal_tier_label(inst: Optional[str], core_norm: Optional[str], tier_norm: Optional[str]) -> Optional[str]:
    c = core_norm or ""
    t = tier_norm or c
    i = inst or ""

    if c == "巡抚": return "决策"

    if c == "御史" or "御史" in t or i == "都察院":
        if t in {"左都御史","右都御史","都御史"}: return "决策"
        if t in {"左副都御史","右副都御史","副都御史","左佥都御史","右佥都御史","佥都御史"}: return "分管"
        return "执行"

    if (i in {"六科","吏科","户科","礼科","兵科","刑科","工科"}) or ("给事" in c) or ("给事" in t):
        if ("都给事中" in t) or (c == "都给事中"): return "决策"
        if c == "给事中":
            if any(prefix in (t or "") for prefix in ["左给事中","右给事中"]): return "分管"
            return "执行"
        return "执行"

    if i in {"吏部","户部","礼部","兵部","刑部","工部"} or c in {"尚书","侍郎","郎中","员外郎","主事","司务"}:
        if c == "尚书": return "决策"
        if c == "侍郎": return "分管"
        return "执行"

    if i == "通政司" or c in {"通政使","通政","左通政","右通政","参议"}:
        if c == "通政使": return "决策"
        if c in {"左通政","右通政","参议"}: return "分管"
        return "执行"

    if i in {"大理寺","太常寺","光禄寺","太仆寺","鸿胪寺"} or c in {"卿","少卿","寺丞","寺正","寺副","评事","典簿"}:
        if c in {"大理寺卿","太常寺卿","光禄寺卿","太仆寺卿","鸿胪寺卿","卿"}: return "决策"
        if c in {"少卿","寺丞"}: return "分管"
        return "执行"

    if i == "国子监" or c in {"祭酒","司业","博士","助教","学正","学录","典簿"}:
        if c == "祭酒": return "决策"
        if c == "司业": return "分管"
        return "执行"

    if i in {"布政使司","按察使司"} or c in {"布政使","参政","参议","按察使","副使","佥事","提学","兵备"}:
        if c in {"布政使","按察使"}: return "决策"
        if c in {"参政","参议","副使","佥事","提学","兵备"}: return "分管"
        return "执行"

    if c in {"知府","知州","知县","府同知","通判","州同知","州判官","县丞","主簿","推官","判官","经历","知事","吏目","典史","教谕","训导","典籍","典簿"}:
        if c in {"知府","知州","知县"}: return "决策"
        if c in {"府同知","通判","州同知","州判官","县丞","主簿"}: return "分管"
        return "执行"

    if any(s in t for s in ["总","都","正"]) and not any(s in t for s in ["副","佥","同","少"]): return "决策"
    if any(s in t for s in ["副","佥","同","少"]): return "分管"
    return "执行" if c else None

# ========= 地名归一 =========
def normalize_place_for_role(place: Optional[str], core: Optional[str]) -> Optional[str]:
    if not place: return None
    p = place
    if p in REGION_PREFIXES: return p
    suf_set = ("盐课司","盐运司","府","州","县","卫","所","路","司","厅")
    if len(p) == 2 and p[-1] in {"府","州","县"} and (core in {"巡抚","布政使","参政","参议","按察使","副使","佥事","提学","兵备","总督","巡按"}): return p
    if p in suf_set: return None
    for suf in sorted(suf_set, key=len, reverse=True):
        if p.endswith(suf):
            stem = p[:-len(suf)]
            if len(stem) >= 1:
                if len(stem) == 1 and (core in {"巡抚","布政使","参政","参议","按察使","副使","佥事","提学","兵备","总督","巡按"}): return p
                return stem
            return p
    return p

@dataclass
class Parsed:
    原始称谓: str
    机构: Optional[str] = None
    职系: Optional[str] = None
    核心职称: Optional[str] = None
    通用层级: Optional[str] = None
    修饰_方位: Optional[str] = None
    修饰_副: Optional[str] = None
    地名: Optional[str] = None
    对齐码_core: Optional[str] = None
    对齐码_inst: Optional[str] = None
    对齐码_tier: Optional[str] = None
    对齐码_loc_core: Optional[str] = None
    对齐码_loc_inst: Optional[str] = None
    对齐码_loc_full: Optional[str] = None

# ========= 主解析（右→左核心优先） =========
def parse_title(title: str) -> Parsed:
    raw_input = title
    s0 = norm(title)
    s, _ = premap_alias_whole_string(s0)

    inst = None
    place = None
    dir_mod = None
    dep_mod = None
    tier = None
    core = None

    # 1) 省域前缀
    m_region = REGION_PREFIX_RE.match(s)
    rest = s
    if m_region:
        place = m_region.group(1)
        rest = s[len(place):]

    # 2) 右→左锁定核心
    core, dir_from_tail, span = find_core_right_to_left(rest)
    if dir_from_tail:
        dir_mod = dir_from_tail

    # 地方简称 & 地方强模式（仅当核心未定）
    if core is None:
        m_alias = re.match(r"^(?P<p>[\u4e00-\u9fa5]{0,8})(?P<suf>府|州|县)?(?P<a>令|守|同知)$", rest if place is None else (place + rest))
        if m_alias and (place is None):
            p_tok = m_alias.group("p") or ""
            suf   = m_alias.group("suf") or ""
            alias = m_alias.group("a")
            place = (p_tok + suf) or place
            if alias == "令": core = "知县"
            elif alias == "守": core = "知府"
            else:
                if   suf == "府": core = "府同知"
                elif suf == "州": core = "州同知"
                else:             core = "同知"
            tier = core
            span = (len(rest)-(len(alias)), len(rest))

    if core is None:
        pat_local = re.compile(
            rf"^(?P<place>[\u4e00-\u9fa5]{{2,8}}?(?:{'|'.join(map(re.escape, PLACE_SUFFIX))})?)"
            rf"(?P<title>知府|知州|知县|府同知|推官|判官|通判|府丞|县丞|教授|学正)$"
        )
        m_local = pat_local.match(rest if place is None else (place + rest))
        if m_local and (place is None):
            place = m_local.group("place") or place
            core  = m_local.group("title")
            tier  = core

    # 3) 回溯机构（看核心左侧文本）
    left = rest[:span[0]] if (core and span) else rest
    if re.search(r"布政使司", left): inst = "布政使司"
    elif re.search(r"按察使司", left): inst = "按察使司"
    elif re.search(r"通政(司)?", left): inst = "通政司"
    elif re.search(r"太常(寺)?", left): inst = "太常寺"
    elif re.search(r"光禄(寺)?", left): inst = "光禄寺"
    elif re.search(r"太仆(寺)?", left): inst = "太仆寺"
    elif re.search(r"鸿胪(寺)?", left): inst = "鸿胪寺"
    elif re.search(r"大理(寺)?", left): inst = "大理寺"
    elif re.search(r"都察(院)?", left): inst = "都察院"
    elif re.search(r"吏部", left): inst = "吏部"
    elif re.search(r"户部", left): inst = "户部"
    elif re.search(r"礼部", left): inst = "礼部"
    elif re.search(r"兵部", left): inst = "兵部"
    elif re.search(r"刑部", left): inst = "刑部"
    elif re.search(r"工部", left): inst = "工部"
    elif re.search(r"锦衣卫", left): inst = "锦衣卫"
    elif re.search(r"盐运司", left): inst = "盐运司"
    elif re.search(r"盐课司", left): inst = "盐课司"
    if inst is None:
        m_ke = re.search(r"(吏科|户科|礼科|兵科|刑科|工科)", left)
        if m_ke: inst = m_ke.group(1)

    # 4) 机构兜底
    if inst is None and core:
        if core in {"参政","参议"}: inst = "布政使司"
        elif core in {"副使","佥事","提学","兵备"}: inst = "按察使司"
        elif core in {"通政","通政使","左通政","右通政"}: inst = "通政司"
        elif core in {"盐运使","运同","运副","运判"}: inst = "盐运司"
        elif core in {"都督","都督同知","都督佥事"}: inst = "五军都督府"
        elif core in {"都指挥使","都指挥同知","都指挥佥事"}: inst = "都指挥使司"
        elif core in {"指挥使","指挥佥事"}: inst = "卫所"
        elif core == "御史": inst = "都察院"

    # 5) 方位补抽（若尾缀未取到）
    if dir_mod is None and core and span:
        left_tail = rest[max(0, span[0]-2):span[0]]
        m_dir = re.search(r"(南台|左|右|前|后|中|东|西|南|北|内|外)$", left_tail)
        if m_dir:
            dir_mod = m_dir.group(1)

    # 6) 规范核心与层级
    core_norm = normalize_core_from_hit(core) if core else None
    if core_norm == "御史":
        tier = tier or (("监察御史") if re.search(r"监察御史|巡按御史|巡按", rest) else "御史")
    else:
        if tier is None:
            tier = (dir_mod or "") + core_norm if (dir_mod and core_norm) else core_norm

    # 7) 地名归一
    place_norm = normalize_place_for_role(place, core_norm)

    # 8) 职系
    fam = FAMILY_MAP.get(core_norm or "", None)
    if not fam and core_norm:
        if core_norm.endswith("御史"): fam = "都察院系"
        elif core_norm.endswith("给事中") or core_norm.endswith("给事"): fam = "六科系"
        elif core_norm in {"尚书","侍郎","郎中","员外郎","主事"}: fam = "六部系"
        elif core_norm in {"布政使","参政","参议"}: fam = "承宣布政使司系"
        elif core_norm in {"按察使","副使","佥事","提学","兵备"}: fam = "提刑按察使司系"
        elif core_norm in {"通政","通政使","左通政","右通政"}: fam = "通政司系"
        elif core_norm in {"盐运使","运同","运副","运判"}: fam = "盐运司系"

    # 9) 层级标签与对齐码
    uni_label = infer_universal_tier_label(inst, core_norm, tier)
    g0 = mk_g0_core(core_norm)
    g1 = mk_g1_inst(inst, core_norm)
    g2 = mk_g2_tier(inst, core_norm, tier)
    g3 = mk_g3_loc_core(place_norm, core_norm)
    g4 = mk_g4_loc_inst(place_norm, inst, core_norm)
    g5 = mk_g5_loc_full(place_norm, inst, core_norm, tier, dir_mod, dep_mod)

    return Parsed(
        原始称谓=raw_input, 机构=inst, 职系=fam,
        核心职称=core_norm, 通用层级=uni_label,
        修饰_方位=dir_mod, 修饰_副=dep_mod, 地名=place_norm,
        对齐码_core=g0, 对齐码_inst=g1, 对齐码_tier=g2,
        对齐码_loc_core=g3, 对齐码_loc_inst=g4, 对齐码_loc_full=g5
    )

# ========= 自测 =========
if __name__ == "__main__":
    tests = [
        # 关键修复
        "员外郎",          # → 核心=员外郎（非郎中），无“中”方位
        "外郎",            # → 核心=员外郎
        "员外",            # → 核心=员外郎
        "郎",              # → 核心=郎中
        # 右→左核心优先
        "福建布政使司右参政",   # 核心=参政 方位=右 机构=布政使司
        "广东布政使司参政",
        "浙江布政使司左参议",
        "湖广按察使司副使",
        "云南按察使司兵备佥事",
        "临沅兵备副使",
        # 六科/给事
        "吏科给事中","刑科给事中","兵科右给事中","礼科左给事中","都给事中","给事",
        # 御史
        "巡按御史","南台监察御史","左都御史","右副都御史",
        # 其他
        "南京兵部尚书","南京通政司","南台御史",
        "福州知府","辰州知州","安吉州知州","州同知","石首令",
        "锦衣卫指挥使","五军都督府左都督","都指挥佥事",
        "教授","训导","学正","教谕","典籍","典簿",
        "盐运使","运同","运副","运判","河泊所官","驿丞","闸官",
        "大理寺正","通政"
    ]
    for t in tests:
        print("例：", t, "->", asdict(parse_title(t)))

    print("\n【职官解析 · 交互调试】输入官职称谓，回车解析；空行/exit 结束")
    while True:
        s = input("> 官职：").strip()
        if not s or s.lower() == "exit": break
        try:
            print(asdict(parse_title(s)))
        except Exception as e:
            print(f"[ERROR] {e}")
