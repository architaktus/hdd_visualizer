#python -m venv venv --system-site-packages
#python -m streamlit run main.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import yaml
import re

# --- 1. 常量与配置初始化 ---
st.set_page_config(page_title="HDD Physical Diagnostic Pro", layout="wide")

PRESETS_FILE = "presets.yaml"
DATA_FILE = "bad_sectors_v3.csv"

# Victoria 风格颜色映射 delay等级
COLOR_MAP = {
    # 增加一个浅灰：正常
    'gray': '#D3D3D3',   # 灰色
    'green': '#7FFF00',  # 黄绿色
    'orange': '#FF4500', # 橙色
    'red': "#FF1E00",    # 红色
    'blue': "#3F20EB",   # 错误 (UNCR/Error)
    'black': '#000000'   # 物理损坏
}

# --- 2. 阈值逻辑函数 ---
def victoria_grade(val, block_size):
    """根据 BlockSize 和 响应/错误 判定等级"""
    if isinstance(val, str): # 错误情况 (UNCR, AMNF 等)
        return 'blue'
    
    bs = int(block_size)
    ms = float(val)
    
    # 阈值表映射
    if bs <= 256: 
        thresholds = [50, 200, 600]
    elif bs == 512: 
        thresholds = [100, 400, 1200]
    elif bs == 1024: 
        thresholds = [150, 600, 1800]
    elif bs == 2048: 
        thresholds = [250, 1000, 3000]
    elif bs == 4096: 
        thresholds = [450, 1800, 5400]
    elif bs == 8192: 
        thresholds = [850, 3400, 10000]
    elif bs == 16384: 
        thresholds = [1700, 6600, 19000]
    elif bs == 32768: 
        thresholds = [3300, 13000, 39000]
    else: # 65535
        thresholds = [6400, 25000, 76000]

    if ms < thresholds[0]: return 'gray'
    if ms < thresholds[1]: return 'green'
    if ms < thresholds[2]: return 'orange'
    return 'red'

# --- 3. 配置管理 ---
def load_presets():
    if not os.path.exists(PRESETS_FILE):
        default = {
            'WD40EFRX': {'lba_max': 7814037168, 'heads': 8, 'rpm': 5400, 'speed_out': 175.0, 'speed_in': 80.0},
            'ST2000DM001': {'lba_max': 3907029168, 'heads': 6, 'rpm': 7200, 'speed_out': 210.0, 'speed_in': 100.0}
        }
        with open(PRESETS_FILE, 'w') as f: yaml.dump(default, f)
        return default
    with open(PRESETS_FILE, 'r') as f: return yaml.safe_load(f)

# --- 4. 物理映射核心 ---
def lba_to_chs(lba, heads, A, B, total_tracks):
    H = heads
    delta = (A*H)**2 - 2*(0.5*B*H)*lba
    if delta < 0: delta = 0
    cylinder = (A*H - np.sqrt(delta)) / (B*H) if B != 0 else lba/(A*H)
    
    current_spt = A - B * cylinder
    lba_start_of_cylinder = H * (A*cylinder - 0.5*B*cylinder**2)
    lba_in_cyl = lba - lba_start_of_cylinder
    
    head = int(lba_in_cyl // current_spt)
    head = min(head, H - 1)
    
    sector_offset = lba_in_cyl % current_spt
    theta = (sector_offset / current_spt) * 2 * np.pi
    
    norm_track = cylinder / total_tracks
    return cylinder, head, theta, norm_track

# --- 5. UI 侧边栏与配置 ---
presets = load_presets()
with st.sidebar:
    st.header("🛠️ 硬盘规格预设")
    model_name = st.selectbox("选择型号", list(presets.keys()) + ["Custom"])
    
    if model_name != "Custom":
        p = presets[model_name]
        lba_max = p['lba_max']; heads = p['heads']; rpm = p['rpm']; s_out = p['speed_out']; s_in = p['speed_in']
    else:
        lba_max = st.number_input("LBA Max", 7814037168)
        heads = st.number_input("Heads", 8)
        rpm = st.number_input("RPM", 5400)
        s_out = st.number_input("Speed Out (MB/s)", 180.0)
        s_in = st.number_input("Speed In (MB/s)", 80.0)

    # 计算 ZBR
    rps = rpm / 60.0
    spt_out = (s_out * 1_000_000) / (512 * rps)
    spt_in = (s_in * 1_000_000) / (512 * rps)
    total_tracks = lba_max / ((spt_out + spt_in)/2 * heads)
    A, B = spt_out, (spt_out - spt_in) / total_tracks
    r_in_ratio = spt_in / spt_out

# --- 6. 数据录入逻辑 ---
@st.dialog("Victoria Log 解析器")
def log_importer():
    st.write("解析 Victoria 日志并自动判定延迟等级")
    bs_choice = st.selectbox("Block Size", [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65535], index=2)
    log_text = st.text_area("粘贴 Log 行", height=200, placeholder="9:03:09 : Block start at 803429456 ... = 62 ms\n10:10:01 : Block start at 103655936 ... Read error: UNCR")
    
    if st.button("开始解析并追加"):
        lines = log_text.split('\n')
        new_entries = []
        # Pattern 1: Time-based | Pattern 2: Error-based
        p1 = r"Block start at (\d+) .* = (\d+) ms"
        p2 = r"Block start at (\d+) .* Read error: (.*)"
        
        for l in lines:
            m1 = re.search(p1, l)
            m2 = re.search(p2, l)
            if m1:
                lba_s = int(m1.group(1))
                grade = victoria_grade(int(m1.group(2)), bs_choice)
                new_entries.append(f"{lba_s}-{lba_s + bs_choice - 1}|{grade}")
            elif m2:
                lba_s = int(m2.group(1))
                grade = victoria_grade("ERR", bs_choice) # 强制蓝紫色
                new_entries.append(f"{lba_s}-{lba_s + bs_choice - 1}|{grade}")
        
        if new_entries:
            st.session_state.raw_data += "\n" + "\n".join(new_entries)
            st.rerun()

# --- 7. 主界面 ---
if 'raw_data' not in st.session_state: st.session_state.raw_data = ""

c1, c2 = st.columns([4, 6])

with c1:
    st.subheader("📝 数据管理")
    st.session_state.raw_data = st.text_area("LBA-Range|Grade|Points", value=st.session_state.raw_data, height=400)
    
    col_btns = st.columns(3)
    with col_btns[0]:
        if st.button("🪄 助手"): log_importer()
    with col_btns[1]:
        if st.button("🚀 渲染"): st.rerun()
    with col_btns[2]:
        # 导出 CSV
        df_export = []
        for line in st.session_state.raw_data.strip().split('\n'):
            if '|' in line:
                p = line.split('|')
                df_export.append({'range': p[0], 'grade': p[1], 'pts': p[2] if len(p)>2 else ''})
        if df_export:
            csv = pd.DataFrame(df_export).to_csv(index=False).encode('utf-8')
            st.download_button("💾 导出", csv, "export.csv", "text/csv")

with c2:
    st.subheader("💿 磁盘物理示意图")
    mode = st.radio("显示模式", ["Surface Individual", "Merge All Surfaces"], horizontal=True)
    
    # 物理映射计算
    plot_objects = []
    for line in st.session_state.raw_data.strip().split('\n'):
        if not line.strip() or '|' not in line: continue
        pts = line.split('|')
        rng = pts[0].strip()
        grade = pts[1].strip().lower()
        cnt_req = int(pts[2]) if len(pts)>2 and pts[2].isdigit() else 0
        
        s_lba, e_lba = map(int, rng.split('-')) if '-' in rng else (int(rng), int(rng))
        color = COLOR_MAP.get(grade, COLOR_MAP['gray'])
        
        # 采样点逻辑
        if s_lba == e_lba or cnt_req > 0:
            num = max(1, cnt_req)
            lbas = np.linspace(s_lba, e_lba, num)
            for l in lbas:
                _, h, th, r_norm = lba_to_chs(l, heads, A, B, total_tracks)
                r_val = 1.0 - r_norm * (1.0 - r_in_ratio)
                plot_objects.append({'type': 'pt', 'h': h, 'r': r_val, 'th': th, 'c': color})
        else:
            # 弧线逻辑：如果跨越磁头，需分段
            _, h1, th1, r_n1 = lba_to_chs(s_lba, heads, A, B, total_tracks)
            _, h2, th2, r_n2 = lba_to_chs(e_lba, heads, A, B, total_tracks)
            r_val = 1.0 - r_n1 * (1.0 - r_in_ratio)
            
            if h1 == h2:
                plot_objects.append({'type': 'arc', 'h': h1, 'r': r_val, 't1': th1, 't2': th2, 'c': color})
            else:
                # 跨磁头处理 (简单首尾弧 + 中间全圆)
                plot_objects.append({'type': 'arc', 'h': h1, 'r': r_val, 't1': th1, 't2': 2*np.pi, 'c': color})
                for mid_h in range(h1+1, h2):
                    plot_objects.append({'type': 'arc', 'h': mid_h, 'r': r_val, 't1': 0, 't2': 2*np.pi, 'c': color})
                plot_objects.append({'type': 'arc', 'h': h2, 'r': r_val, 't1': 0, 't2': th2, 'c': color})

    # 绘图渲染
    if mode == "Merge All Surfaces":
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
        ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
        # 只在盘面区域画辅助虚线
        for rad in np.linspace(r_in_ratio, 1.0, 5):
            ax.plot(np.linspace(0, 2*np.pi, 100), [rad]*100, color='gray', lw=0.5, ls='--', alpha=0.3)
        ax.fill_between(np.linspace(0, 2*np.pi, 50), r_in_ratio, 1.0, color='gray', alpha=0.05)
        
        for p in plot_objects:
            if p['type'] == 'pt': ax.scatter(p['th'], p['r'], c=p['c'], s=15, edgecolors='none')
            else: ax.plot(np.linspace(p['t1'], p['t2'], 50), [p['r']]*50, color=p['c'], lw=1.5)
        
        ax.set_yticklabels([]); ax.set_xticklabels([])
        st.pyplot(fig)
    else:
        # 分磁头平铺 (Individual Surfaces)
        h_to_view = st.columns(4)
        for i in range(int(heads)):
            with h_to_view[i % 4]:
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(3, 3))
                fig.patch.set_alpha(0.0); ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
                ax.set_title(f"Head {i}", fontsize=8)
                ax.fill_between(np.linspace(0, 2*np.pi, 50), r_in_ratio, 1.0, color='gray', alpha=0.05)
                # 盘面虚线
                ax.plot(np.linspace(0, 2*np.pi, 100), [r_in_ratio]*100, color='gray', lw=0.5, ls='--')
                ax.plot(np.linspace(0, 2*np.pi, 100), [1.0]*100, color='gray', lw=0.5, ls='--')
                
                for p in plot_objects:
                    if p['h'] == i:
                        if p['type'] == 'pt': ax.scatter(p['th'], p['r'], c=p['c'], s=10)
                        else: ax.plot(np.linspace(p['t1'], p['t2'], 30), [p['r']]*30, color=p['c'], lw=1)
                ax.set_yticklabels([]); ax.set_xticklabels([])
                st.pyplot(fig)