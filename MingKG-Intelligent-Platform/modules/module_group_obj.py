import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from py2neo import Graph
import json
from openai import OpenAI

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ LLM配置 =============
llm_config = {
    "api_base_url": "https://tbnx.plus7.plus/v1",
    "api_key": "sk-e8DdamFXsM6jBn1MA5NTyUAvMDdsQLJnKLKfgItEz75GUj1Q"
}
MODEL_NAME = "deepseek-reasoner"
TIMEOUT = 2000
client = OpenAI(base_url=llm_config['api_base_url'], api_key=llm_config['api_key'])

def call_openai(full_prompt: str) -> list:
    full_response = ""
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是中国历史社会科学领域的知识归纳助手，需要对明清科举人物数据库的属性或对象属性分布做分层归纳、统计和注释，"
                        "结果严格输出JSONL格式，每行含“分层名称”“人数统计”“专业注释”。"
                    )
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            stream=True,
            timeout=TIMEOUT
        )
        for chunk in stream:
            if hasattr(chunk, "choices"):
                for choice in chunk.choices:
                    delta = getattr(choice, "delta", None)
                    if delta and getattr(delta, "content", None):
                        content = delta.content
                        if content.strip():
                            full_response += content
    except Exception as e:
        st.error(f"[ERROR] OpenAI 调用失败: {e}")
        return []
    return [line for line in full_response.strip().split('\n') if line.strip()]

def build_prompt(field, value_counts):
    prompt = (
        f"请你根据下列“{field}”分布，结合明清历史社会学知识，进行分层归纳、人数统计与注释。"
        "每个分层输出分层名称、人数统计、专业注释，要求JSONL格式，每行一条。\n"
        "原始分布如下：\n"
    )
    for k, v in value_counts.items():
        prompt += f"{k}: {v}\n"
    return prompt

# ============ Neo4j连接 =============
@st.cache_resource
def get_graph():
    return Graph("bolt://localhost:7687", auth=("neo4j", "lhq18385458795"))
graph = get_graph()

@st.cache_data
def load_profile_data():
    query = '''
    MATCH (p:ImperialPerson)
    RETURN 
      p.户籍类型 AS 户籍类型, 
      p.学籍 AS 学籍, 
      p.家庭排行 AS 家庭排行, 
      p.学术专长 AS 学术专长
    '''
    data = [dict(record) for record in graph.run(query)]
    return pd.DataFrame(data)

obj_fields = {
    "参与": {
        "query": '''
            MATCH (p:ImperialPerson)-[:参与]->(e:ImperialExam)
            WHERE e.考试等级 = "殿试"
            RETURN e.unique_key AS 目标名称, count(p) AS 人数
            ORDER BY e.unique_key
        ''',
        "label": "殿试时间"
    },
    "生": {
        "query": '''
            MATCH (p:ImperialPerson)-[:生]->(pl:Place)
            RETURN pl.历史名称 AS 目标名称, count(p) AS 人数
            ORDER BY 人数 DESC
        ''',
        "label": "出生地"
    },
    "职任": {
        "query": '''
            MATCH (p:ImperialPerson)-[:职任]->(pl:Place)
            RETURN pl.历史名称 AS 目标名称, count(p) AS 人数
            ORDER BY 人数 DESC
        ''',
        "label": "职任地"
    },
    "担任-迁转次数": {
        "query": '''
            MATCH (p:ImperialPerson)-[r:担任]->(:`OfficialPositions`)
            WHERE r.迁转 IS NOT NULL AND r.迁转 <> ""
            RETURN p.姓名 AS 目标名称, count(r) AS 人数
            ORDER BY 人数 DESC
        ''',
        "label": "担任迁转次数（每人）"
    },
    "担任-官职人数": {
        "query": '''
            MATCH (:ImperialPerson)-[:担任]->(op:`OfficialPositions`)
            RETURN op.官职名称 AS 目标名称, count(*) AS 人数
            ORDER BY 人数 DESC
        ''',
        "label": "官职担任人数"
    }
}

def run():
    st.header("群体结构分析与分层归纳可视化")
    mode = st.radio(
        "请选择分析模式",
        ("属性量化", "对象属性量化"),
        horizontal=True
    )
    col1, col2 = st.columns([2, 3])

    df = load_profile_data()
    attr_list = ["户籍类型", "学籍", "家庭排行", "学术专长"]
    obj_list = list(obj_fields.keys())

    # 属性量化模式
    if mode == "属性量化":
        selected = col1.selectbox("选择需要量化/归纳的属性", attr_list)
        if col1.button("可视化分析", key="profile_btn"):
            value_counts = df[selected].fillna("（空值）").replace("", "（空值）").value_counts()
            st.session_state['last_value_counts'] = value_counts
            st.session_state['last_selected_attr'] = selected
            st.session_state['last_mode'] = "属性量化"

    # 对象属性量化模式
    else:
        selected_obj = col1.selectbox("选择对象属性", obj_list)
        if col1.button("可视化分析", key="obj_btn"):
            result = graph.run(obj_fields[selected_obj]["query"])
            data = [dict(record) for record in result]
            value_counts = pd.Series([d['人数'] for d in data], index=[d['目标名称'] for d in data]).sort_values(ascending=False)
            st.session_state['last_obj_value_counts'] = value_counts
            st.session_state['last_selected_obj'] = selected_obj
            st.session_state['last_mode'] = "对象属性量化"

    # 数据展示逻辑（状态安全，页面刷新不丢失）
    if st.session_state.get('last_mode') == "属性量化" and \
       'last_value_counts' in st.session_state and 'last_selected_attr' in st.session_state:
        value_counts = st.session_state['last_value_counts']
        selected = st.session_state['last_selected_attr']
        col2.subheader("数据库分布")
        fig, ax = plt.subplots(figsize=(8, 4))
        value_counts.plot(kind='bar', color="#6495ED", ax=ax)
        ax.set_title(f"{selected} 各取值分布（数据库原始）")
        ax.set_xlabel(selected)
        ax.set_ylabel("人数")
        for i, v in enumerate(value_counts):
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=10)
        col2.pyplot(fig)
        col2.dataframe(value_counts.rename("人数"))

        llm_btn = col2.button("一键归纳优化", key="profile_llm")
        if llm_btn:
            with col2.status("⏳ 正在进行归纳推理...", expanded=True):
                prompt = build_prompt(selected, value_counts)
                llm_result = call_openai(prompt)
                parsed = []
                for line in llm_result:
                    try:
                        obj = json.loads(line)
                        parsed.append(obj)
                    except Exception:
                        continue
                if not parsed:
                    col2.error("未返回有效结构化数据！")
                else:
                    df_llm = pd.DataFrame(parsed)
                    x_col = "分层名称" if "分层名称" in df_llm else df_llm.columns[0]
                    y_col = "人数统计" if "人数统计" in df_llm else df_llm.columns[1]
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    df_llm.set_index(x_col)[y_col].plot(kind='bar', color="#FF9800", ax=ax2)
                    for i, v in enumerate(df_llm[y_col]):
                        ax2.text(i, v, str(v), ha='center', va='bottom', fontsize=10)
                    ax2.set_xlabel(x_col)
                    ax2.set_ylabel("人数")
                    ax2.set_title(f"{selected} 归纳分层分布")
                    col2.pyplot(fig2)
                    note_col = "专业注释" if "专业注释" in df_llm else df_llm.columns[-1]
                    col2.dataframe(df_llm[[x_col, y_col, note_col]])

    elif st.session_state.get('last_mode') == "对象属性量化" and \
         'last_obj_value_counts' in st.session_state and 'last_selected_obj' in st.session_state:
        value_counts = st.session_state['last_obj_value_counts']
        selected_obj = st.session_state['last_selected_obj']
        col2.subheader(f"数据库分布（{obj_fields[selected_obj]['label']}）")
        fig, ax = plt.subplots(figsize=(8, 4))
        value_counts[:20].plot(kind='bar', color="#03a9f4", ax=ax)
        ax.set_title(f"{obj_fields[selected_obj]['label']} 前20项分布（数据库原始）")
        ax.set_xlabel(obj_fields[selected_obj]['label'])
        ax.set_ylabel("人数")
        for i, v in enumerate(value_counts[:20]):
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=10)
        col2.pyplot(fig)
        col2.dataframe(value_counts.rename("人数").reset_index().rename(columns={'index': obj_fields[selected_obj]['label']}))
        llm_btn = col2.button("一键LLM归纳优化", key="obj_llm")
        if llm_btn:
            with col2.status("⏳ 正在进行大模型归纳推理...", expanded=True):
                prompt = build_prompt(obj_fields[selected_obj]['label'], value_counts[:20].to_dict())
                llm_result = call_openai(prompt)
                parsed = []
                for line in llm_result:
                    try:
                        obj = json.loads(line)
                        parsed.append(obj)
                    except Exception:
                        continue
                if not parsed:
                    col2.error("未返回有效结构化数据！")
                else:
                    df_llm = pd.DataFrame(parsed)
                    x_col = "分层名称" if "分层名称" in df_llm else df_llm.columns[0]
                    y_col = "人数统计" if "人数统计" in df_llm else df_llm.columns[1]
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    df_llm.set_index(x_col)[y_col].plot(kind='bar', color="#FF9800", ax=ax2)
                    for i, v in enumerate(df_llm[y_col]):
                        ax2.text(i, v, str(v), ha='center', va='bottom', fontsize=10)
                    ax2.set_xlabel(x_col)
                    ax2.set_ylabel("人数")
                    ax2.set_title(f"{selected_obj} LLM归纳分层分布")
                    col2.pyplot(fig2)
                    note_col = "专业注释" if "专业注释" in df_llm else df_llm.columns[-1]
                    col2.dataframe(df_llm[[x_col, y_col, note_col]])

# run() 由 app.py 调用
