# apps/knowledge_discovery.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, importlib, importlib.util, inspect, os
from pathlib import Path
import streamlit as st

# ---------------- 路径（apps 与 modules 同级） ----------------
APPS_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT  = APPS_DIR.parent
MODULES_DIR   = PROJECT_ROOT / "modules"
ASSETS_DIR    = PROJECT_ROOT / "assets"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------- 动态导入（仅“找不到模块”时回落到路径导入） ----------------
_FALLBACK_EXC = (ModuleNotFoundError, ImportError)

def _safe_import(fqmn: str):
    """导入完全限定名（如 'modules.ming_yearly_trend'）。成功→(module,None)，失败→(None,error)。"""
    try:
        mod = importlib.import_module(fqmn)
        return mod, None
    except _FALLBACK_EXC as e_pkg:
        if not fqmn.startswith("modules."):
            return None, e_pkg
        rel = fqmn.split("modules.", 1)[1]
        pyfile = MODULES_DIR / (rel.replace(".", "/") + ".py")
        if not pyfile.exists():
            return None, e_pkg
        try:
            spec = importlib.util.spec_from_file_location(fqmn, str(pyfile))
            mod  = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            sys.modules[fqmn] = mod
            return mod, None
        except Exception as e_path:
            return None, e_path
    except Exception as e:
        return None, e

# ---------------- 智能调用 run（支持全局数据路径与多命名） ----------------
def _call_run(mod, data_path: str | None = None, prefer_kwargs: dict | None = None):
    """
    智能调用模块 run：
      - 自动识别 st-like 参数名：st / st_ / streamlit / app / ctx
      - 自动识别数据路径参数名：default_data_path / data_path / path / rdf_path
      - 若存在 configure_page 参数，默认 False（可被 prefer_kwargs 覆盖）
      - 兜底：若模块存在 DEFAULT_DATA 常量则覆盖为 data_path
    """
    if not hasattr(mod, "run"):
        st.error("目标模块缺少 run() 函数。")
        return

    fn = mod.run
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
    except Exception:
        try:
            return fn(st)
        except TypeError:
            return fn()

    st_like_names   = {"st", "st_", "streamlit", "app", "ctx"}
    data_like_names = {"default_data_path", "data_path", "path", "rdf_path"}

    st_param_name   = next((n for n in params if n in st_like_names), None)
    data_param_name = next((n for n in params if n in data_like_names), None)

    kwargs = dict(prefer_kwargs or {})
    if "configure_page" in params and "configure_page" not in kwargs:
        kwargs["configure_page"] = False
    if st_param_name:
        kwargs[st_param_name] = st
    if data_param_name and data_path:
        kwargs[data_param_name] = data_path

    if data_path and hasattr(mod, "DEFAULT_DATA"):
        try:
            setattr(mod, "DEFAULT_DATA", data_path)
        except Exception:
            pass

    try:
        return fn(**kwargs) if kwargs else fn()
    except TypeError:
        pass
    except Exception as e:
        st.error(f"子模块运行异常：{e}")
        return

    if st_param_name:
        try:
            return fn(st)
        except TypeError:
            pass
        except Exception as e:
            st.error(f"子模块运行异常：{e}")
            return

    if "configure_page" in params:
        try:
            return fn(configure_page=False)
        except TypeError:
            pass
        except Exception as e:
            st.error(f"子模块运行异常：{e}")
            return

    try:
        return fn()
    except Exception as e:
        st.error(f"子模块运行异常：{e}")

# ---------------- 统一入口：供主入口调用 ----------------
def run(default_data_path: str | None = None):
    """
    知识发现（容器页）
    - 数据输入优先：在未应用有效数据源前，不显示任何子模块导航与主区内容
    - 一旦数据源有效，再显示九个子模块与其内容
    - 不在主区添加额外标题
    """
    # ======= KD 专属样式 =======
    st.markdown("""
    <style>
      :root{
        --kd-accent: #8B5CF6;         /* violet-500 */
        --kd-accent-weak:#F3E8FF;     /* violet-100 */
        --kd-accent-weak2:#FAF5FF;    /* violet-50  */
        --kd-text-strong:#111827;
      }
      .block-container{
        background: linear-gradient(180deg, #fff 0%, #fff 60%, var(--kd-accent-weak2) 100%);
        border-radius: 12px;
      }
      /* 导航卡 */
      [data-testid="stSidebar"] .kd-nav-card{
        background: linear-gradient(135deg, var(--kd-accent) 0%, #6D28D9 100%);
        color:#fff; border-radius:16px; padding:14px 14px 8px 14px;
        box-shadow: 0 6px 18px rgba(139,92,246,.25);
        border:1px solid rgba(255,255,255,.18);
        margin-bottom:14px;
      }
      [data-testid="stSidebar"] .kd-nav-title{
        font-weight:900; letter-spacing:.3px;
        font-size: clamp(14px, 1.05vw + 10px, 20px);
        margin-bottom:10px;
      }
      [data-testid="stSidebar"] .kd-nav-card .stRadio{
        background: rgba(255,255,255,.09);
        border-radius:12px; padding:8px 8px;
      }
      [data-testid="stSidebar"] .kd-nav-card .stRadio [role="radiogroup"] label p{
        color:#fff !important; font-weight:800;
        font-size: clamp(14px, 1vw + 9px, 18px); line-height:1.25;
      }
      [data-testid="stSidebar"] .kd-nav-card .stRadio [role="radiogroup"] label [data-baseweb="radio"] input:checked + div p{
        text-decoration: underline; text-decoration-thickness: 2px;
      }
      /* 设置卡（数据源与子模块参数区） */
      [data-testid="stSidebar"] .kd-settings-card{
        background:#fff; border:1px solid #E5E7EB; border-radius:14px;
        box-shadow:0 1px 3px rgba(0,0,0,.06); padding:10px 12px; margin: 4px 0 12px 0;
      }
      [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{
        font-weight:800; color: var(--kd-text-strong);
        background: linear-gradient(90deg, var(--kd-accent-weak), transparent);
        padding:6px 10px; border-radius:10px; border-left: 4px solid var(--kd-accent);
      }
      [data-testid="stSidebar"] .kd-settings-hint{ color:#4B5563; font-size:12px; margin: 4px 0 8px 0; }
      .kd-banner{ height:6px; border-radius:999px; margin:4px 0 10px 0;
                  background: linear-gradient(90deg, var(--kd-accent), rgba(139,92,246,0.15)); }
    </style>
    """, unsafe_allow_html=True)

    # 左栏 Logo（保留）
    if (ASSETS_DIR / "logo.png").exists():
        st.sidebar.image(str(ASSETS_DIR / "logo.png"), use_column_width=True)

    # ===== 1) 数据输入（优先、始终可见） =====
    init_path = st.session_state.get("kd_data_path") or default_data_path or ""
    with st.sidebar:
        st.markdown('<div class="kd-settings-card">', unsafe_allow_html=True)
        st.markdown("**数据源**（RDF / TTL / NT / OWL / XML）", unsafe_allow_html=True)
        data_input = st.text_input(
            label="数据文件路径",
            value=init_path,
            key="kd_data_input",
            placeholder="粘贴数据文件的绝对路径…"
        )
        c1, c2 = st.columns(2)
        with c1:
            apply_clicked = st.button("应用数据源", type="primary", use_container_width=True, key="kd_btn_apply")
        with c2:
            clear_clicked = st.button("清空缓存", use_container_width=True, key="kd_btn_clear")
        st.markdown('</div>', unsafe_allow_html=True)

    if clear_clicked:
        st.cache_data.clear(); st.cache_resource.clear()
        try:
            st.experimental_rerun()
        except Exception:
            st.rerun()

    if apply_clicked:
        st.session_state["kd_data_path"] = (data_input or "").strip()
        if st.session_state["kd_data_path"]:
            os.environ["MINGKG_GLOBAL_DATA_PATH"] = st.session_state["kd_data_path"]
        st.cache_data.clear(); st.cache_resource.clear()
        st.rerun()

    data_path = st.session_state.get("kd_data_path")
    path_ok = bool(data_path and Path(data_path).exists())

    # ===== 若数据源未就绪：仅提示，不显示模块导航 & 主区；直接 return =====
    if not path_ok:
        st.markdown('<div class="kd-banner"></div>', unsafe_allow_html=True)
        if data_path and not Path(data_path).exists():
            st.error("未找到指定的数据文件，请检查路径是否正确。")
        else:
            st.info(" 请输入并『应用数据源』后，将显示可用的九个功能模块。")
        return

    # ===== 2) 数据源有效后，才显示：模块导航卡 + 系统操作 + 主区内容 =====
    # 导航卡
    with st.sidebar:
        st.markdown('<div class="kd-nav-card">', unsafe_allow_html=True)
        st.markdown('<div class="kd-nav-title"> 知识发现 · 模块导航</div>', unsafe_allow_html=True)

    page = st.sidebar.radio(
        label="",
        options=(
            "科举人物数据属性检索",
            "科举人物官职信息检索",
            "科举人物迁移路线检索",
            "科举人物社会关系检索",
            "群体结构归纳——人物",
            "群体结构归纳——地点",
            "群体结构归纳——职官",
            "群体结构归纳——科考",
            "时空计量",
        ),
        index=8,
        key="kd_nav_radio",
        label_visibility="collapsed"
    )

    with st.sidebar:
        st.markdown('</div>', unsafe_allow_html=True)  # 关闭导航卡
        # 系统卡
        st.markdown('<div class="kd-settings-card">', unsafe_allow_html=True)
        st.markdown("<div class='kd-settings-hint'>以下为容器级操作；各子模块的设置项将显示在其自身的侧栏分区中。</div>", unsafe_allow_html=True)
        with st.expander("系统", expanded=False):
            if st.button("清缓存并刷新", use_container_width=True, key="kd_btn_refresh"):
                st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # 主区：不再添加额外标题，仅色带分隔
    st.markdown('<div class="kd-banner"></div>', unsafe_allow_html=True)

    ROUTE_MAP = {
        "科举人物数据属性检索":   "modules.person_profile_attrs",
        "科举人物官职信息检索":   "modules.official_positions",
        "科举人物迁移路线检索":   "modules.person_place_events",
        "科举人物社会关系检索":   "modules.social_relations",
        "群体结构归纳——人物":     "modules.people_facets_ranks",
        "群体结构归纳——地点":     "modules.place_level_stats",
        "群体结构归纳——职官":     "modules.official_positions_metrics",
        "群体结构归纳——科考":     "modules.dianshi_timeline",
        "时空计量":               "modules.ming_yearly_trend",
    }

    fqmn = ROUTE_MAP.get(page)
    mod, err = _safe_import(fqmn)
    if mod is None:
        st.error(f"模块加载失败：{err}")
        hint = MODULES_DIR / (fqmn.split(".", 1)[1].replace(".", "/") + ".py")
        st.info(f"请确认文件存在：{hint}")
        return

    # 统一把 data_path 下发给子模块，并禁用其内部 set_page_config
    _call_run(mod, data_path=data_path, prefer_kwargs={"configure_page": False})
