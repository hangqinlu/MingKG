import streamlit as st

# --- 导入各功能模块 ---
from modules import module_profile
from modules import module_group_obj
from modules import module_space
from modules import module_social
from modules import module_qa

# --- 页面设置 ---
st.set_page_config(page_title="明清科举知识多维智能平台", layout="wide")

# --- 侧边栏导航 ---
st.sidebar.image("assets/logo.png", use_column_width=True)  # 如有logo
st.sidebar.title("功能导航")
page = st.sidebar.radio(
    "请选择功能模块：",
    (
        "全息画像",
        "群体结构归纳",
        "空间流动分析",
        "社会关系可视化",
        "智能问答"
    ),
    index=0
)

st.markdown(
    """
    <style>
        .sidebar .sidebar-content {background-color:s #F8F9FA;}
        .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 主体区块根据导航动态渲染 ---
if page == "全息画像":
    module_profile.run()
elif page == "群体结构归纳":
    module_group_obj.run()
elif page == "空间流动分析":
    module_space.run()
elif page == "社会关系可视化":
    module_social.run()
elif page == "智能问答":
    module_qa.run()
