# app.py
# -*- coding: utf-8 -*-
"""
主系统入口：信息组织 / 信息消歧 / 知识发现
运行：streamlit run app.py
"""

import sys, inspect, importlib, importlib.util
from pathlib import Path
import streamlit as st

# ---------- 路径 ----------
ROOT        = Path(__file__).resolve().parent
APPS_DIR    = ROOT / "apps"
MODULES_DIR = ROOT / "modules"
ASSETS_DIR  = ROOT / "assets"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))

# ---------- 稳健导入（仅在真正“找不到模块”时才回落到路径导入） ----------
FALLBACK_EXC = (ModuleNotFoundError, ImportError)

def safe_import(fqmn: str):
    """导入完全限定名（如 'apps.knowledge_discovery'）。成功→(module, None)，失败→(None, error)。"""
    try:
        mod = importlib.import_module(fqmn)
        return mod, None
    except FALLBACK_EXC as e_pkg:
        if fqmn.startswith("apps."):
            base = APPS_DIR; rel = fqmn.split("apps.", 1)[1]
        elif fqmn.startswith("modules."):
            base = MODULES_DIR; rel = fqmn.split("modules.", 1)[1]
        else:
            return None, e_pkg
        pyfile = base / (rel.replace(".", "/") + ".py")
        if not pyfile.exists():
            return None, e_pkg
        try:
            spec = importlib.util.spec_from_file_location(fqmn, str(pyfile))
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            sys.modules[fqmn] = mod
            return mod, None
        except Exception as e_path:
            return None, e_path
    except Exception as e:
        return None, e

def call_run(mod, prefer_kwargs=None):
    """
    智能调用 run()（支持多命名 st 参数；默认关闭子页 set_page_config）
    """
    if not hasattr(mod, "run"):
        st.error("模块缺少 run() 函数。")
        return

    fn = mod.run
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
    except Exception:
        params = {}

    st_like_names = {"st", "st_", "streamlit", "app", "ctx"}
    st_param = next((n for n in params if n in st_like_names), None)

    kwargs = dict(prefer_kwargs or {})
    if "configure_page" in params and "configure_page" not in kwargs:
        kwargs["configure_page"] = False
    if st_param:
        kwargs[st_param] = st

    try:
        return fn(**kwargs) if kwargs else fn()
    except TypeError:
        pass
    except Exception as e:
        st.error(f"运行异常：{e}")
        return
    try:
        return fn(st)
    except TypeError:
        pass
    except Exception as e:
        st.error(f"运行异常：{e}")
        return
    try:
        return fn()
    except Exception as e:
        st.error(f"运行异常：{e}")

# ---------- 页面设置 ----------
st.set_page_config(page_title="明清科举 · 主系统入口", layout="wide")

# ---------- 样式（侧栏卡片 + 垂直粗黑箭头） ----------
st.markdown("""
<style>
  .block-container { max-width: 96vw; padding-left: 8px; padding-right: 8px; }

  /* 侧栏：卡片容器 */
  [data-testid="stSidebar"] .card-base {
    width: 100%;
    border-radius: 18px; padding: 16px 14px;
    border: 1.5px solid #111; background: #fff;
    box-shadow: 6px 6px 0 #000; position: relative;
    text-align: center; margin: 2px 0 8px 0;
  }
  [data-testid="stSidebar"] .card-step {
    position:absolute; top: -12px; left: -12px;
    width: 38px; height: 38px; border-radius: 50%;
    background: #000; color:#fff; font-weight: 900;
    display:flex; align-items:center; justify-content:center;
    border: 2px solid #fff; box-shadow: 2px 2px 0 #000;
    font-size: 14px;
  }
  [data-testid="stSidebar"] .card-icon  { font-size: clamp(24px, 1.6vw + 16px, 36px); line-height:1; }
  [data-testid="stSidebar"] .card-title { margin-top: 6px; line-height: 1.1; }

  /* 三种标题风格，避免“重复字体”观感 */
  [data-testid="stSidebar"] .title-org { font-weight: 900; letter-spacing: .2px; font-size: clamp(16px, 1.2vw + 12px, 22px); }
  [data-testid="stSidebar"] .title-dis { font-weight: 800; font-style: italic; letter-spacing: .2px; font-size: clamp(16px, 1.2vw + 12px, 22px); }
  [data-testid="stSidebar"] .title-kd  { font-weight: 800; font-variant-caps: small-caps; font-size: clamp(16px, 1.2vw + 12px, 22px); }

  [data-testid="stSidebar"] .card-sub { margin-top: 4px; color:#334155; font-size: 12px; }

  /* 粗黑实心向下箭头（居中） */
  [data-testid="stSidebar"] .fat-arrow-vert { width: 100%; display:flex; align-items:center; justify-content:center; padding: 6px 0 10px 0; }
  [data-testid="stSidebar"] .fat-arrow-vert svg { width: 34px; height: 34px; display:block; }
  [data-testid="stSidebar"] .fat-arrow-vert path { fill:#000; }

  /* 侧栏“进入”按钮样式，做成卡片主按钮 */
  [data-testid="stSidebar"] .enter-btn > button {
    width: 100%;
    border: 1.5px solid #111 !important;
    box-shadow: 4px 4px 0 #000 !important;
    background: #111 !important; color: #fff !important;
    font-weight: 800; letter-spacing: .2px;
  }
  [data-testid="stSidebar"] .enter-btn > button:hover {
    filter: brightness(0.95);
  }
</style>
""", unsafe_allow_html=True)

# ---------- 侧栏（卡片选择：从上到下） ----------
if (ASSETS_DIR / "logo.png").exists():
    st.sidebar.image(str(ASSETS_DIR / "logo.png"), use_column_width=True)

MODULES = [
    {"key": "信息组织", "icon": "🧩", "title_cls": "title-org", "subtitle": "清洗/结构化原始资料，定义类与属性", "step": "1"},
    {"key": "信息消歧", "icon": "🔗", "title_cls": "title-dis", "subtitle": "实体共指合并 · 溯源挂接 · 约束校验", "step": "2"},
    {"key": "知识发现", "icon": "📈", "title_cls": "title-kd",  "subtitle": "GIS · 统计 · 模型，面向研究问题", "step": "3"},
]
ROUTE = {
    "信息组织": "apps.info_organization",
    "信息消歧": "apps.disambiguation",
    "知识发现": "apps.knowledge_discovery",
}

def arrow_down_svg():
    return """
    <div class="fat-arrow-vert">
      <svg viewBox="0 0 100 100" role="img" aria-label="arrow-down">
        <path d="M42 12 H58 V60 H74 L50 92 L26 60 H42 Z" />
      </svg>
    </div>
    """

with st.sidebar:
    for i, M in enumerate(MODULES):
        # 卡片头（标题视觉，不可点击）
        st.markdown(
            f"""
            <div class="card-base">
              <div class="card-step">{M['step']}</div>
              <div class="card-icon">{M['icon']}</div>
              <div class="card-title {M['title_cls']}">{M['key']}</div>
              <div class="card-sub">{M['subtitle']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        # 主操作按钮（可点击）
        holder = st.container()
        with holder:
            st.markdown('<div class="enter-btn">', unsafe_allow_html=True)
            if st.button(f"进入 {M['key']}", key=f"enter_{M['key']}", use_container_width=True):
                st.session_state["active_module"] = M["key"]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        # 箭头（最后一个不显示）
        if i < len(MODULES) - 1:
            st.markdown(arrow_down_svg(), unsafe_allow_html=True)

    with st.expander("系统", expanded=False):
        if st.button("清缓存并刷新", use_container_width=True, key="btn_clear_cache_main"):
            st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

# ---------- 主区 ----------
active_page = st.session_state.get("active_module")

if not active_page:
    # 首屏占位与流程说明（主区保持简洁）
    st.markdown("### ")
    st.info("在左侧卡片按流程从上到下选择模块进入。")
else:
    fqmn = ROUTE.get(active_page)
    mod, err = safe_import(fqmn)
    if mod is None:
        st.error(f"页面模块加载失败：{err}")
        hint = (APPS_DIR / (fqmn.split('.',1)[1].replace('.', '/') + ".py")) if fqmn else None
        if hint:
            st.info(f"请确认文件存在：{hint}")
    else:
        call_run(mod, prefer_kwargs={"configure_page": False})

    st.divider()
    if st.button("← 返回模块选择", use_container_width=True, key="btn_back_to_hub"):
        st.session_state["active_module"] = None
        st.rerun()
