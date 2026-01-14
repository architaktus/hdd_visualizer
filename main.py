#python -m venv venv --system-site-packages
#python -m streamlit run main.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# --- 1. 页面配置 ---
st.set_page_config(page_title="HDD Visualizer Final", layout="wide")
st.title("💿 硬盘坏道物理映射工具")

DATA_FILE = "bad_sectors.csv"

# --- 2. 文件 I/O ---
def load_from_csv():
    if not os.path.exists(DATA_FILE):
        return """546699936-546716320|250ms|circle
3000000000-3000500000|3s|circle
100000-200000|1s
7800000000|err"""
    try:
        df = pd.read_csv(DATA_FILE)
        text_lines = []
        for _, row in df.iterrows():
            line = f"{row['range']}"
            if pd.notna(row['tag']) and str(row['tag']).strip() != '':
                line += f"|{row['tag']}"
            if 'count' in row and pd.notna(row['count']) and str(row['count']).strip() != '':
                line += f"|{row['count']}"
            text_lines.append(line)
        return "\n".join(text_lines)
    except:
        return ""

def save_to_csv(text_input):
    lines = text_input.strip().split('\n')
    data_list = []
    for line in lines:
        line = line.strip()
        if not line: continue
        parts = line.split('|')
        rng = parts[0].strip()
        tag = parts[1].strip() if len(parts) > 1 else ''
        cnt = parts[2].strip() if len(parts) > 2 else ''
        data_list.append({'range': rng, 'tag': tag, 'count': cnt})
    pd.DataFrame(data_list).to_csv(DATA_FILE, index=False)

# --- 3. 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 硬盘物理参数")
    preset = st.selectbox("硬盘预设", ["WD 4TB (WD40EFRX)", "Seagate 2TB", "Custom"])
    
    if preset == "WD 4TB (WD40EFRX)":
        lba_max, rpm, spd_out, spd_in = 7814037168, 5400, 175.0, 80.0
    elif preset == "Seagate 2TB":
        lba_max, rpm, spd_out, spd_in = 3907029168, 7200, 210.0, 100.0
    else:
        lba_max = st.number_input("总 LBA", 7814037168)
        rpm = st.number_input("RPM", 7200)
        spd_out = st.number_input("外圈速度 MB/s", 180.0)
        spd_in = st.number_input("内圈速度 MB/s", 80.0)

# --- 4. 核心计算逻辑 (点线分离) ---
def calculate_geometry_and_map(rpm, s_out, s_in, total_lba, input_data):
    # 物理计算
    rps = rpm / 60.0
    spt_out = (s_out * 1_000_000) / (512 * rps)
    spt_in = (s_in * 1_000_000) / (512 * rps)
    avg_spt = (spt_out + spt_in) / 2
    total_tracks = total_lba / avg_spt
    
    A = spt_out
    B = (spt_out - spt_in) / total_tracks
    
    # 两个绘图列表：scatter 用于点，line 用于圆环/弧线
    scatter_points = []
    line_shapes = [] 
    list_entries = []
    
    lines = input_data.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        
        parts = line.split('|')
        rng_str = parts[0].strip()
        tag = parts[1].strip().lower() if len(parts) > 1 else 'default'
        user_param = parts[2].strip().lower() if len(parts) > 2 else None
        
        # 颜色映射
        c = 'gray'
        if '250ms' in tag: c = 'green'
        elif '1s' in tag: c = 'orange'
        elif '3s' in tag: c = 'red'
        elif 'err' in tag: c = 'black'

        try:
            if '-' in rng_str:
                s_lba, e_lba = map(int, rng_str.split('-'))
                lba_mid = (s_lba + e_lba) // 2
                range_len = e_lba - s_lba
                is_range = True
            else:
                s_lba = e_lba = int(rng_str)
                lba_mid = s_lba
                range_len = 0
                is_range = False
        except:
            continue
            
        # 计算半径 (基于中点)
        delta = A**2 - 2 * B * lba_mid
        if delta < 0: delta = 0
        track_index = (A - np.sqrt(delta)) / B if B != 0 else lba_mid / A
        
        current_spt = A - B * track_index
        norm_track = track_index / total_tracks
        if norm_track > 1.0: norm_track = 1.0
        r_inner_ratio = spt_in / spt_out
        radius = 1.0 - norm_track * (1.0 - r_inner_ratio)

        # === 绘图模式判定 ===
        mode = "Point"
        
        # 判定条件 1: 用户显式指定 'circle'
        is_circle_cmd = (user_param == 'circle') #改为0
        
        # 判定条件 2: 范围超过一圈，且用户没有指定具体的数字（如 |5）
        is_auto_circle = (range_len >= current_spt) and (not (user_param and user_param.isdigit()))
        
        # 判定条件 3: 用户指定了具体的点数 (如 |5)
        is_discrete_count = (user_param and user_param.isdigit())
        
        if is_circle_cmd or is_auto_circle:
            # === 模式 A: 实线圆环 (Line Plot) ===
            mode = "Solid Ring"
            # 生成 0 到 2pi 的连续坐标
            thetas = np.linspace(0, 2*np.pi, 200) # 200个点足够平滑
            radii = np.full_like(thetas, radius) # 半径恒定
            
            line_shapes.append({
                'theta': thetas,
                'r': radii,
                'color': c,
                'lw': 2.0 # 线宽
            })
            
        elif is_discrete_count:
            # === 模式 B: 用户强制指定点数 (Scatter) ===
            count = int(user_param)
            mode = f"Discrete ({count} pts)"
            
            if count > 0:
                lbas = np.linspace(s_lba, e_lba, count).astype(int)
                for lba in lbas:
                    offset = lba % current_spt
                    theta = (offset / current_spt) * 2 * np.pi
                    scatter_points.append({'theta': theta, 'r': radius, 'color': c, 'size': 30})
                    
        else:
            # === 模式 C: 默认行为 (小范围弧线或单点) ===
            if is_range:
                mode = "Arc (Auto)"
                # 默认画首尾两点示意范围，或者画一段小弧线
                # 为了简单，这里用散点画首尾，中间连线太复杂涉及跨0度问题
                lbas = [s_lba, e_lba]
                for lba in lbas:
                    offset = lba % current_spt
                    theta = (offset / current_spt) * 2 * np.pi
                    scatter_points.append({'theta': theta, 'r': radius, 'color': c, 'size': 20})
            else:
                mode = "Single Point"
                offset = s_lba % current_spt
                theta = (offset / current_spt) * 2 * np.pi
                scatter_points.append({'theta': theta, 'r': radius, 'color': c, 'size': 40})

        list_entries.append({
            'Range': rng_str,
            'Tag': tag,
            'Mode': mode,
            'Radius': f"{radius:.3f}"
        })
        
    return scatter_points, line_shapes, list_entries, r_inner_ratio

# --- 5. UI 布局 ---
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = load_from_csv()

col_ctrl1, col_ctrl2 = st.columns([1, 6])
with col_ctrl1:
    if st.button("📂 重载 CSV"):
        st.session_state['input_text'] = load_from_csv()
        st.rerun()

col_editor, col_result = st.columns([35, 65])

with col_editor:
    st.subheader("📝 数据录入")
    st.markdown("""
    **显示规则:**
    1. `...|circle` : 强制显示为**实线圆环**。
    2. `...|5` : 强制显示为 **5个离散点**。
    3. 大范围默认显示为圆环。
    """)
    new_text = st.text_area("Input", value=st.session_state['input_text'], height=450, label_visibility="collapsed")
    if new_text != st.session_state['input_text']:
        st.session_state['input_text'] = new_text

    if st.button("💾 保存并更新", type="primary", use_container_width=True):
        save_to_csv(new_text)
        st.rerun()

with col_result:
    scatter_data, line_data, list_data, r_in_ratio = calculate_geometry_and_map(
        rpm, spd_out, spd_in, lba_max, st.session_state['input_text']
    )
    
    st.subheader("📊 物理可视化")
    sub_c1, sub_c2 = st.columns([4, 6])
    
    with sub_c1:
        if list_data:
            st.dataframe(pd.DataFrame(list_data), height=400, use_container_width=True, hide_index=True)

    with sub_c2:
        if scatter_data or line_data:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='polar')
            ax.set_theta_zero_location('N') #type:ignore
            ax.set_theta_direction(-1)      #type:ignore
            
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            # 背景
            ax.fill_between(np.linspace(0, 2*np.pi, 100), r_in_ratio, 1, color='#808080', alpha=0.1)
            ax.plot(np.linspace(0, 2*np.pi, 100), [1]*100, color='#666', lw=0.5)
            ax.plot(np.linspace(0, 2*np.pi, 100), [r_in_ratio]*100, color='#666', lw=0.5)
            
            # --- 绘制实线圆环 ---
            for line in line_data:
                ax.plot(line['theta'], line['r'], color=line['color'], linewidth=line['lw'], alpha=0.8)
            
            # --- 绘制散点 ---
            if scatter_data:
                thetas = [d['theta'] for d in scatter_data]
                radii = [d['r'] for d in scatter_data]
                colors = [d['color'] for d in scatter_data]
                sizes = [d['size'] for d in scatter_data]
                ax.scatter(thetas, radii, c=colors, s=sizes, edgecolors='none', alpha=0.9)
            
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(True, alpha=0.2)
            ax.spines['polar'].set_visible(False)
            
            st.pyplot(fig, use_container_width=True)
            st.caption(
                f"内径/外径比: {r_in_ratio:.2f}\n"
                "🟢<250ms 🟠1s 🔴3s ⚫Bad"
            )