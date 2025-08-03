import streamlit as st
from py2neo import Graph
import pandas as pd

@st.cache_resource
def get_graph():
    return Graph("bolt://localhost:7687", auth=("neo4j", "lhq18385458795"))
graph = get_graph()

def run():
    st.header("科举人物全息画像查询")
    person_name = st.text_input("请输入科举人物姓名", key="name_input")
    if st.button("查询", key="query_btn") and person_name.strip():
        # 基础信息
        cypher1 = f"""MATCH (p:Person) WHERE p.姓名 = '{person_name}' RETURN p.姓名 AS 姓名, p.字 AS 字, p.家庭排行 AS 家庭排行, p.户籍类型 AS 户籍类型, p.户籍地 AS 户籍地, p.学籍 AS 学籍, p.学术专长 AS 学术专长"""
        data1 = graph.run(cypher1).data()
        if data1:
            st.subheader("基础信息")
            st.table(pd.DataFrame(data1))
        else:
            st.error("未查询到该人物的基础信息，请检查姓名拼写或数据库。")
        # 官职履历
        cypher2 = f"""MATCH (p:Person)-[r:担任]->(o:OfficialPositions) WHERE p.姓名 = '{person_name}' RETURN o.官职名称 AS 官职名称, o.官阶 AS 官阶, r.授予类型 AS 授予类型, r.迁转 AS 迁转, r.授予时间 AS 授予时间 ORDER BY r.授予时间"""
        data2 = graph.run(cypher2).data()
        if data2:
            st.subheader("官职履历")
            st.table(pd.DataFrame(data2))
        # 科举履历
        cypher3 = f"""MATCH (p:Person)-[r:参与]->(ie:ImperialExam) WHERE p.姓名 = '{person_name}' RETURN ie.考试等级 AS 考试等级, ie.考试地点 AS 考试地点, ie.考试时间 AS 考试时间, r.功名 AS 功名, r.名次 AS 名次, r.甲第等级 AS 甲第等级 ORDER BY ie.考试时间"""
        data3 = graph.run(cypher3).data()
        if data3:
            st.subheader("科举履历")
            st.table(pd.DataFrame(data3))
        # 地理/空间履历
        cypher4 = f"""MATCH (p:Person)-[r]->(pl:Place) WHERE p.姓名 = '{person_name}' AND type(r) IN ['生','职任','任务执行'] RETURN type(r) AS 关联类型, pl.历史名称 AS 地点, pl.现代名称 AS 现代地名, pl.区划层级 AS 区划层级, pl.经纬坐标 AS 经纬度, r.任职时间 AS 任职时间, r.任务时间 AS 任务时间, r.职事名目 AS 职事名目"""
        data4 = graph.run(cypher4).data()
        if data4:
            st.subheader("空间/任职履历")
            st.table(pd.DataFrame(data4))
        # 社会关系
        cypher5 = f"""MATCH (p1:Person)-[r:社会关系]-(p2:Person) WHERE p1.姓名 = '{person_name}' RETURN p2.姓名 AS 关联人物, r.关系类型 AS 关系类型"""
        data5 = graph.run(cypher5).data()
        if data5:
            st.subheader("社会关系网络")
            st.table(pd.DataFrame(data5))
