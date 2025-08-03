import streamlit as st
import folium
from streamlit_folium import st_folium
from py2neo import Graph
import re

def parse_coord(coord_str):
    coord_str = coord_str.replace('，', ',').replace(' ', '')
    parts = coord_str.split(',')
    if len(parts) != 2:
        raise ValueError("格式不正确")
    lat_str, lng_str = parts
    def get_num(s):
        num = float(re.findall(r"[-+]?\d*\.\d+|\d+", s)[0])
        if ('S' in s) or ('南' in s):
            num = -num
        if ('W' in s) or ('西' in s):
            num = -num
        return num
    lat = get_num(lat_str)
    lng = get_num(lng_str)
    return lng, lat

@st.cache_resource
def get_graph():
    return Graph("bolt://localhost:7687", auth=("neo4j", "lhq18385458795"))

def get_person_places(name):
    graph = get_graph()
    query = f'''
    MATCH (p:ImperialPerson {{姓名: "{name}"}})-[r]->(place:Place)
    WHERE place.经纬坐标 IS NOT NULL
    RETURN type(r) AS rel_type, place.历史名称 AS place, place.经纬坐标 AS coord
    '''
    result = graph.run(query, parameters={"name": name})
    places = []
    for record in result:
        try:
            lng, lat = parse_coord(record["coord"])
            html_info = f"""
            <div style='font-size:13px;line-height:1.6;min-width:120px;max-width:200px;'>
              <span style='font-weight:bold;color:#205493;'>地点：</span>{record['place']}<br>
              <span style='font-weight:bold;color:#205493;'>关系：</span>{record['rel_type']}<br>
              <span style='color:#888;'>坐标：</span>{lat:.4f}, {lng:.4f}
            </div>
            """
            places.append({
                "关系": record['rel_type'],
                "地点": record['place'],
                "经度": lng,
                "纬度": lat,
                "信息": html_info
            })
        except Exception:
            continue
    return places

def run():
    st.header("科举人物地理迁移地图")
    if 'last_name' not in st.session_state:
        st.session_state['last_name'] = ""
    if 'places' not in st.session_state:
        st.session_state['places'] = []

    name = st.text_input("请输入人名：", value=st.session_state['last_name'])
    if st.button("查询") or (name and name != st.session_state['last_name']):
        st.session_state['places'] = get_person_places(name)
        st.session_state['last_name'] = name

    places = st.session_state['places']

    if not places:
        st.info("请输入人名并点击查询。")
    else:
        center = [places[0]['纬度'], places[0]['经度']]
        m = folium.Map(location=center, zoom_start=6, tiles="OpenStreetMap")
        for p in places:
            folium.Marker(
                [p['纬度'], p['经度']],
                icon=folium.Icon(icon='arrow-up', prefix='fa', color='blue'),
                popup=folium.Popup(p['信息'], max_width=250, min_width=120, show=False)
            ).add_to(m)
        st_folium(m, width=900, height=600)
