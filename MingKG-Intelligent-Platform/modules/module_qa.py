import streamlit as st
from py2neo import Graph
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import re

llm_config = {
    "api_base_url": "https://tbnx.plus7.plus/v1",
    "api_key": "sk-e8DdamFXsM6jBn1MA5NTyUAvMDdsQLJnKLKfgItEz75GUj1Q"
}
MODEL_NAME = "deepseek-reasoner"
client = OpenAI(base_url=llm_config['api_base_url'], api_key=llm_config['api_key'])

def clean_code_block(text: str) -> str:
    text = text.strip()
    pattern = r"^```[a-zA-Z]*\n([\s\S]+?)\n```$"
    match = re.match(pattern, text)
    if match:
        return match.group(1).strip()
    if text.startswith("```"):
        text = text.lstrip("`").split('\n', 1)[-1]
    if text.endswith("```"):
        text = text.rstrip("`").rsplit('\n', 1)[0]
    return text.strip()

@st.cache_resource
def get_graph():
    return Graph("bolt://localhost:7687", auth=("neo4j", "lhq18385458795"))
graph = get_graph()

def call_openai(user_prompt: str, system_prompt: str = None) -> str:
    full_response = ""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        timeout=1500
    )
    for chunk in stream:
        if hasattr(chunk, "choices"):
            for choice in chunk.choices:
                delta = getattr(choice, "delta", None)
                if delta and getattr(delta, "content", None):
                    full_response += delta.content
    return full_response.strip()

def run():
    st.header("科举知识问答平台")

    if "db_result" not in st.session_state:
        st.session_state["db_result"] = None
    if "last_df" not in st.session_state:
        st.session_state["last_df"] = None

    with st.form("query_form", clear_on_submit=False):
        user_query = st.text_area("请输入你的问题（如：‘张三在哪些地方担任过官职？’）", key="main_ask")
        submit = st.form_submit_button("提问", type="primary", use_container_width=True)
        if submit and user_query.strip():
            cypher_prompt = f"""
你是一名“明清科举知识图谱”Neo4j数据库专家。数据库结构如下：

节点类型，及其中文属性名：
- ImperialPerson/Person（姓名、字、家庭排行、户籍类型、户籍地、学籍、学术专长）
- OfficialPositions（官职名称、官阶）
- ImperialExam（考试等级、考试地点、考试时间、unique_key）
- Place（历史名称、现代名称、区划层级、经纬坐标）

关系类型及其中文属性名：
- 参与（Person→ImperialExam，功名、名次、甲第等级）
- 担任（Person→OfficialPositions，授予类型、迁转、授予时间）
- 生（Person→Place）
- 职任（Person→Place，任职时间）
- 任务执行（Person→Place，职事名目、任务时间）
- 社会关系（Person↔Person，关系类型）
- 隶属（Place→Place）

请你首先对用户输入进行推理分析，精准识别其核心查询需求，理解其实际意图与信息诉求。然后，**只生成一条**严格与需求对应、可直接在 Neo4j 运行的 Cypher 语句。  
- 仅包含所需的单一关系或节点及最相关属性，不返回全部属性。  
- 结果字段必须和节点/关系一一对应，禁止一次性查询多关系或多节点。  
- 禁止OPTIONAL MATCH，不要输出多条语句、拼接、分号、注释或说明。  
- 只输出Cypher语句本体，不输出任何其他内容。  

{user_query}
"""
            cypher_code_raw = call_openai(cypher_prompt, system_prompt="你是知识图谱查询专家。")
            cypher_code = clean_code_block(cypher_code_raw)
            st.markdown("**自动生成的Cypher查询：**")
            st.code(cypher_code, language="cypher")
            try:
                data = list(graph.run(cypher_code))
                df = pd.DataFrame([dict(record) for record in data])
                st.session_state["db_result"] = df
                st.session_state["last_df"] = df
                st.markdown("**数据库返回结果：**")
                st.dataframe(df)
            except Exception as e:
                st.session_state["db_result"] = None
                st.session_state["last_df"] = None
                st.error(f"Cypher执行失败: {e}")

    if st.session_state["db_result"] is not None and not st.session_state["db_result"].empty:
        st.markdown("---")
        st.markdown("##### RAG归纳结果：")
        with st.form("llm_postprocess_form", clear_on_submit=True):
            further_instruction = st.text_area("请输入你希望进一步整理、归纳、分析或美化的需求", key="llm_ask")
            post_submit = st.form_submit_button("分析", type="primary")
            if post_submit and further_instruction.strip():
                answer_prompt = (
                    f"你是明清科举知识专家。请根据用户意图“{further_instruction}”对以下表格数据进行学术归纳与信息重组，提炼关键信息，输出清晰有条理的结果，严禁添加任何注释、说明、解释性语言。\n"
                    f"表格内容：\n{st.session_state['last_df'].head(278).to_json(orient='records', force_ascii=False)}"
                )
                llm_answer = call_openai(answer_prompt, system_prompt="你是明清知识整理专家。")
                st.markdown("**推理归纳整理（仅供参考）：**")
                st.info(llm_answer)
        # 可选：自动可视化
        try:
            df = st.session_state["db_result"]
            cols = df.columns.tolist()
            if "地点" in cols or "历史名称" in cols:
                st.markdown("**自动柱状图（地点分布示例）：**")
                col = "地点" if "地点" in cols else "历史名称"
                value_counts = df[col].value_counts().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 4))
                value_counts[:20].plot(kind='bar', color="#03a9f4", ax=ax)
                st.pyplot(fig)
        except Exception:
            pass
