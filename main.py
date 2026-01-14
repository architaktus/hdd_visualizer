#python -m venv venv --system-site-packages
#streamlit run main.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# --- 1. 页面配置 ---
st.set_page_config(page_title="HDD ZBR Pro Visualizer", layout="wide")
st.title("💿 硬盘坏道 ZBR 物理映射工具")

# 定义外置文件名
DATA_FILE = "bad_sectors.csv"

# --- 2. 工具函数：文件 I/O ---

def load_from_csv():
    """从 CSV 读取数据并转换为文本格式"""
    if not os.path.exists(DATA_FILE):
        # 如果文件不存在，创建默认数据
        default_data = pd.DataFrame({
            'range': ['546699936-546716320', '100000-100005', '7800000000', '3000000000'],
            'tag': ['250ms', '1s', 'err', '3s']
        })
        default_data.to_csv(DATA_FILE, index=False)
        return "546699936-546716320|250ms\n100000-100005|1s\n7800000000|err\n3000000000|3s"
    
    try:
        df = pd.read_csv(DATA_FILE)
        # 将 DataFrame 转换为 文本格式
        text_lines = []
        for _, row in df.iterrows():
            line = f"{row['range']}"
            if pd.notna(row['tag']) and str(row['tag']).strip() != '':
                line += f"|{row['tag']}"
            text_lines.append(line)
        return "\n".join(text_lines)
    except Exception as e:
        st.error(f"读取 CSV 失败: {e}")
        return ""

def save_to_csv(text_input):
    """将文本框内容解析并保存回 CSV"""
    lines = text_input.strip().split('\n')
    data_list = []
    for line in lines:
        line = line.strip()
        if not line: continue
        if '|' in line:
            rng, tag = line.split('|', 1)
        else:
            rng, tag = line, ''
        data_list.append({'range': rng.strip(), 'tag': tag.strip()})
    
    df = pd.DataFrame(data_list)
    try:
        df.to_csv(DATA_FILE, index=False)
        st.toast(f"✅ 数据已同步至 {DATA_FILE}", icon="💾")
        return True
    except Exception as e:
        st.error(f"保存 CSV 失败: {e}")
        return False

# --- 3. 侧边栏：物理参数 ---
with st.sidebar:
    st.header("⚙️ 硬盘物理模型")
    
    # 预设配置
    preset = st.selectbox("快速预设", ["Custom", "WD 4TB (WD40EFRX)", "Seagate 2TB (ST2000DM001)"])
    
    if preset == "WD 4TB (WD40EFRX)":
        def_lba = 7814037168
        def_rpm = 5400
        def_spd_out = 175.0
        def_spd_in = 80.0
    elif preset == "Seagate 2TB (ST2000DM001)":
        def_lba = 3907029168
        def_rpm = 7200
        def_spd_out = 210.0
        def_spd_in = 100.0
    else:
        def_lba = 7814037168
        def_rpm = 7200
        def_spd_out = 180.0
        def_spd_in = 80.0

    lba_max = st.number_input("总 LBA 数", value=def_lba, format="%d")
    rpm = st.number_input("转速 (RPM)", value=def_rpm)
    speed_outer = st.number_input("外圈速度 (MB/s)", value=def_spd_out)
    speed_inner = st.number_input("内圈速度 (MB/s)", value=def_spd_in)
    
    st.markdown("---")
    st.caption("视觉参数")
    visual_windings = st.slider("螺旋线密度", 100, 2000, 500, help="仅影响绘图时点的分散程度，不影响物理半径")

# --- 4. 物理计算核心 (ZBR) ---
def calculate_geometry_and_map(rpm: int, s_out:float, s_in:float, total_lba: int, input_data):
    # 1. 物理反推
    rps = rpm / 60.0
    # 扇区大小 512B
    spt_out = (s_out * 1_000_000) / (512 * rps)
    spt_in = (s_in * 1_000_000) / (512 * rps)
    
    avg_spt = (spt_out + spt_in) / 2
    total_tracks = total_lba / avg_spt
    
    # 2. 解析数据点
    mapped_points = []
    
    lines = input_data.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        
        parts = line.split('|')
        rng_str = parts[0].strip()
        tag = parts[1].strip().lower() if len(parts) > 1 else 'default'
        
        # 解析 LBA
        try:
            if '-' in rng_str:
                s, e = map(int, rng_str.split('-'))
                lba = (s + e) // 2
                count = e - s
            else:
                lba = int(rng_str)
                count = 1
        except:
            continue
            
        # 3. ZBR 映射核心公式 (解一元二次方程)
        # SPT(x) = A - Bx
        A = spt_out
        B = (spt_out - spt_in) / total_tracks
        
        # 0.5*B*x^2 - A*x + LBA = 0
        delta = A**2 - 2 * B * lba
        if delta < 0: delta = 0
        track_index = (A - np.sqrt(delta)) / B if B != 0 else lba / A
        
        # 归一化半径 (0=外圈, 1=内圈)
        norm_track = track_index / total_tracks
        if norm_track > 1.0: norm_track = 1.0
        
        # 实际绘图半径 (R_out=1.0, R_in=spt_in/spt_out)
        r_inner_ratio = spt_in / spt_out
        radius = 1.0 - norm_track * (1.0 - r_inner_ratio)
        
        # 角度 (模拟螺旋)
        theta = (lba / total_lba) * visual_windings * 2 * np.pi
        
        # 4. 颜色映射 (用户指定)
        c = 'gray'
        if '250ms' in tag: c = 'green'
        elif '1s' in tag: c = 'orange'
        elif '3s' in tag: c = 'red'
        elif 'err' in tag: c = 'black'
        
        mapped_points.append({
            'lba': lba,
            'range': rng_str,
            'count': count,
            'tag': tag,
            'color': c,
            'theta': theta,
            'r': radius
        })
        
    return mapped_points, r_inner_ratio

# --- 5. UI 主布局 ---

# 初始化 Session State 用于存储文本内容
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = load_from_csv()

# 顶部操作栏
col_ctrl1, col_ctrl2 = st.columns([1, 6])
with col_ctrl1:
    if st.button("📂 重载 CSV"):
        st.session_state['input_text'] = load_from_csv()
        st.rerun()

# 主界面分栏：左侧编辑 (35%)，右侧结果 (65%)
col_editor, col_result = st.columns([35, 65])

# --- 左侧：编辑器 ---
with col_editor:
    st.subheader("📝 数据录入")
    
    # 文本区域绑定到 session_state
    new_text = st.text_area(
        "格式: 起始-结束|标签", 
        value=st.session_state['input_text'],
        height=500,
        key="editor_area"
    )
    
    # 如果文本发生变化，更新 session_state
    if new_text != st.session_state['input_text']:
        st.session_state['input_text'] = new_text

    if st.button("💾 保存并更新图表", type="primary", use_container_width=True):
        save_to_csv(new_text)
        st.rerun()
        
    st.info(f"数据文件位置: `{os.path.abspath(DATA_FILE)}`")

# --- 右侧：结果展示 (嵌套分栏) ---
with col_result:
    # 计算数据
    points, r_in_ratio = calculate_geometry_and_map(
        rpm, speed_outer, speed_inner, lba_max, st.session_state['input_text']
    )
    
    st.subheader("📊 诊断视图")
    
    # 再次拆分：左边是数据列表，右边是图
    sub_c1, sub_c2 = st.columns([4, 6])
    
    with sub_c1:
        st.markdown("**坏道列表解析**")
        if points:
            # 创建一个用于显示的 DataFrame
            display_df = pd.DataFrame([{
                'LBA范围': p['range'], 
                '延迟': p['tag'].upper(), 
                '位置': f"R={p['r']:.2f}"
            } for p in points])
            
            # 使用 dataframe 组件显示，高度限制
            st.dataframe(display_df, height=400, hide_index=True, use_container_width=True)
            
            # 简单统计
            st.markdown("---")
            total_bad = sum(1 for p in points if 'err' in p['tag'])
            slow_sec = sum(1 for p in points if 'ms' in p['tag'] or 's' in p['tag'])
            st.write(f"❌ 坏道区域: **{total_bad}**")
            st.write(f"⚠️ 响应慢区域: **{slow_sec}**")
        else:
            st.warning("暂无有效数据")

    with sub_c2:
        if points:
            thetas = [p['theta'] for p in points]
            radii = [p['r'] for p in points]
            colors = [p['color'] for p in points]
            sizes = [50 if c != 'black' else 80 for c in colors] # Err点画大一点

            # 绘图 - 调整尺寸适应布局
            fig = plt.figure(figsize=(5, 5)) # 缩小尺寸
            ax = fig.add_subplot(111, projection='polar')
            ax.set_theta_zero_location('N') #type: ignore
            ax.set_theta_direction(-1) #type: ignore
            
            # 背景色透明，融入网页
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            # 绘制盘片区域
            ax.fill_between(np.linspace(0, 2*np.pi, 100), r_in_ratio, 1, color='#808080', alpha=0.1)
            # 边界线
            ax.plot(np.linspace(0, 2*np.pi, 100), [1]*100, color='#666', lw=1, alpha=0.5)
            ax.plot(np.linspace(0, 2*np.pi, 100), [r_in_ratio]*100, color='#666', lw=1, alpha=0.5)
            
            # 绘制点
            ax.scatter(thetas, radii, c=colors, s=sizes, edgecolors='white', alpha=0.9, linewidth=0.5)
            
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(False)
            ax.axis('off') # 完全移除坐标轴边框，只留点和背景环
            
            st.pyplot(fig, use_container_width=True)
            
            # 图例
            st.caption(
                "🟢 <250ms | 🟠 >1s | 🔴 >3s | ⚫ ERR (Bad)\n"
                f"盘片内径比: {r_in_ratio:.2f}"
            , unsafe_allow_html=True)