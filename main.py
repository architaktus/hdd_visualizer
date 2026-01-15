#python -m venv venv --system-site-packages
#python -m streamlit run main.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import yaml
import re

# --- 1. 配置与全局常量 ---
st.set_page_config(page_title="HDD Physical Diagnostic V4", layout="wide")

PRESETS_FILE = "presets.yaml"
# 颜色映射 (Delay Level)
COLOR_MAP = {
    'level1': '#D3D3D3',   # Gray (Normal)
    'level2': '#7FFF00',   # Green (Good)
    'level3': '#FFA500',   # Orange (Warning)
    'level4': '#FF4500',   # Red (Critical)
    'error':  '#4169E1',   # Blue/Purple (Error)
    'black':  '#000000'    # Bad
}

# 延迟等级阈值表 (ms)
DELAY_THRESHOLDS = {
    'small':  [50, 200, 600],       # 1-256
    512:      [100, 400, 1200],
    1024:     [150, 600, 1800],
    2048:     [250, 1000, 3000],
    4096:     [450, 1800, 5400],
    8192:     [850, 3400, 10000],
    16384:    [1700, 6600, 19000],
    32768:    [3300, 13000, 39000],
    65535:    [6400, 25000, 76000]
}

# --- 2. 状态初始化 ---
if 'block_size_idx' not in st.session_state: st.session_state.block_size_idx = 3 # 默认 2048
if 'view_mode' not in st.session_state: st.session_state.view_mode = "Merge All Surfaces"
if 'raw_data' not in st.session_state: st.session_state.raw_data = ""
if 'edit_mode' not in st.session_state: st.session_state.edit_mode = False

# --- 3. 核心物理计算 (修正版) ---

def calculate_zbr_params(lba_max, heads, rpm, s_out, s_in):
    """
    计算 ZBR 物理参数
    假设 SPT (Sectors Per Track) 从外向内线性递减
    """
    rps = rpm / 60.0
    # 物理扇区大小 512B
    spt_out = (s_out * 1_000_000) / (512 * rps)
    spt_in = (s_in * 1_000_000) / (512 * rps)
    
    # 平均 SPT * 磁头数 * 磁道数 = 总 LBA
    avg_spt_per_cyl = (spt_out + spt_in) / 2 * heads
    total_cylinders = lba_max / avg_spt_per_cyl
    
    # 线性方程系数: SPT(cyl) = A - B * cyl
    A = spt_out
    B = (spt_out - spt_in) / total_cylinders
    
    return A, B, total_cylinders, spt_out, spt_in

def lba_to_chs(lba, heads, A, B, total_cyls):
    """
    LBA 转 CHS (Cylinder, Head, Sector) 及 归一化半径 (修正版)
    """
    H = heads
    
    # --- 修正点 1: 判别式系数修正 ---
    # 公式: 0.5*B*H * cyl^2 - A*H * cyl + lba = 0
    # a = 0.5*B*H, b = -A*H, c = lba
    # delta = b^2 - 4ac = (AH)^2 - 4*(0.5BH)*lba = (AH)^2 - 2*B*H*lba
    
    if B == 0: # 恒定速度 (非 ZBR)
        cyl_float = lba / (A * H)
    else:
        delta = (A*H)**2 - 2 * B * H * lba  # <--- 修正了系数 (原代码少乘了2)
        if delta < 0: delta = 0
        cyl_float = (A*H - np.sqrt(delta)) / (B*H)

    # --- 修正点 2: 必须取整 ---
    # 物理柱面是整数。cyl_float 是理论连续值，必须向下取整
    # 才能计算出“当前柱面起始位置”
    cyl_int = int(cyl_float)
    
    # 防止浮点误差导致的越界
    if cyl_int >= total_cyls: cyl_int = int(total_cyls) - 1
    if cyl_int < 0: cyl_int = 0

    # 2. 计算该柱面(整数)的起始 LBA
    # 使用 cyl_int 代入积分公式
    lba_start_cyl = H * (A*cyl_int - 0.5*B*(cyl_int**2))
    
    # 3. 计算在当前柱面内的偏移量
    lba_in_cyl = lba - lba_start_cyl
    
    # 4. 当前柱面的 SPT (使用整数索引计算)
    current_spt = A - B * cyl_int
    
    # 5. 计算磁头 (Head) 和 角度 (Theta)
    # 注意：lba_in_cyl 可能因为浮点误差出现微小的负数或略大于容量，需由于 int() 截断
    if lba_in_cyl < 0: lba_in_cyl = 0
    
    head = int(lba_in_cyl // current_spt)
    if head >= heads: head = heads - 1 # 钳位
    
    sector_offset = lba_in_cyl % current_spt
    
    # 计算角度 (0~2pi)
    # 加上偏移量让它不要总是从0度开始，或者保持原样。这里保持原样。
    theta = (sector_offset / current_spt) * 2 * np.pi
    
    # 6. 归一化半径
    norm_cyl = cyl_int / total_cyls
    if norm_cyl > 1.0: norm_cyl = 1.0
    
    return cyl_int, head, theta, norm_cyl

def capacity_percent_to_radius(percent, A, B, total_cyls, r_in_ratio):
    """
    将容量百分比转换为绘图用的物理半径
    例如 50% 容量对应的不是 0.5 半径，而可能是在外圈 0.6 的位置
    """
    # 1. 找到对应的 Cylinder Index
    # 使用比例求解：Capacity(x) / Total_Capacity = percent
    # 公式简化为: (Ax - 0.5Bx^2) / (A*T - 0.5B*T^2) = p
    
    T = total_cyls
    Total_Cap_Factor = A*T - 0.5*B*T**2
    Target_Cap_Factor = Total_Cap_Factor * percent
    
    # 解 0.5Bx^2 - Ax + Target = 0
    delta = A**2 - 2*B*Target_Cap_Factor
    if delta < 0: delta = 0
    if B == 0:
        target_cyl = Target_Cap_Factor / A
    else:
        target_cyl = (A - np.sqrt(delta)) / B
        
    # 2. 映射到绘图半径
    # 绘图半径: Outer=1.0, Inner=r_in_ratio
    # Cyl 0 -> 1.0, Cyl T -> r_in_ratio
    norm_cyl = target_cyl / total_cyls
    visual_r = 1.0 - norm_cyl * (1.0 - r_in_ratio)
    
    return visual_r

# --- 4. 辅助功能 ---

def load_presets():
    if not os.path.exists(PRESETS_FILE):
        default = {
            'WD40EFRX': {'lba_max': 7814037168, 'heads': 8, 'rpm': 5400, 'speed_out': 175.0, 'speed_in': 80.0},
            'ST2000DM001': {'lba_max': 3907029168, 'heads': 6, 'rpm': 7200, 'speed_out': 210.0, 'speed_in': 100.0}
        }
        with open(PRESETS_FILE, 'w') as f: yaml.dump(default, f)
        return default
    with open(PRESETS_FILE, 'r') as f: return yaml.safe_load(f)

def save_presets(data):
    with open(PRESETS_FILE, 'w') as f: yaml.dump(data, f)

def get_grade(ms_val, block_size_key):
    """Victoria 等级判定"""
    if isinstance(ms_val, str): return 'error' # Error text
    
    thresholds = DELAY_THRESHOLDS.get(block_size_key, DELAY_THRESHOLDS[2048])
    if ms_val < thresholds[0]: return 'level1'
    if ms_val < thresholds[1]: return 'level2'
    if ms_val < thresholds[2]: return 'level3'
    return 'level4'

# --- 5. UI: 侧边栏配置 ---
presets = load_presets()

with st.sidebar:
    st.header("🛠️ 硬盘配置")
    
    # 模式切换
    col_mode, col_edit_btn = st.columns([2, 1])
    with col_mode:
        selected_model = st.selectbox("选择预设", list(presets.keys()) + ["New Profile"], 
                                      index=0 if "New Profile" not in list(presets.keys()) else 0,
                                      disabled=st.session_state.edit_mode)
    with col_edit_btn:
        if st.toggle("解锁", value=st.session_state.edit_mode):
            st.session_state.edit_mode = True
        else:
            st.session_state.edit_mode = False

    # 数据加载
    if selected_model == "New Profile":
        current_data = {'lba_max': 0, 'heads': 1, 'rpm': 7200, 'speed_out': 150.0, 'speed_in': 80.0}
        display_name = "New_HDD"
    else:
        current_data = presets[selected_model]
        display_name = selected_model

    # 表单区域
    with st.container(border=True):
        st.caption("参数详情")
        # 如果是编辑模式，允许修改 Key (Model Name)
        new_model_name = st.text_input("型号名称", value=display_name, disabled=not st.session_state.edit_mode)
        
        c_lba = st.number_input("LBA Max", value=current_data['lba_max'], disabled=not st.session_state.edit_mode)
        c_heads = st.number_input("磁头数 (Heads)", value=current_data['heads'], disabled=not st.session_state.edit_mode)
        c_rpm = st.number_input("转速 (RPM)", value=current_data['rpm'], disabled=not st.session_state.edit_mode)
        c_s_out = st.number_input("外圈速度 (MB/s)", value=current_data['speed_out'], disabled=not st.session_state.edit_mode)
        c_s_in = st.number_input("内圈速度 (MB/s)", value=current_data['speed_in'], disabled=not st.session_state.edit_mode)
        
        if st.session_state.edit_mode:
            if st.button("💾 保存配置到 YAML"):
                # 更新 presets
                new_entry = {
                    'lba_max': int(c_lba), 'heads': int(c_heads), 'rpm': int(c_rpm),
                    'speed_out': float(c_s_out), 'speed_in': float(c_s_in)
                }
                # 如果改了名字，删除旧的
                if new_model_name != selected_model and selected_model != "New Profile":
                    del presets[selected_model]
                
                presets[new_model_name] = new_entry
                save_presets(presets)
                st.toast(f"配置 {new_model_name} 已保存!")
                st.rerun()

    # ZBR 参数计算 (用于后续绘图)
    A, B, Total_Cyls, spt_out, spt_in = calculate_zbr_params(c_lba, c_heads, c_rpm, c_s_out, c_s_in)
    r_in_ratio = spt_in / spt_out

# --- 6. UI: Log 解析助手 (Dialog) ---
@st.dialog("Victoria Log 助手")
def log_helper():
    st.markdown("##### 粘贴扫描日志")
    
    # 选项合并逻辑
    bs_options = ["1/64/128/256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65535"]
    
    # 查找 session 中记忆的 index
    def_idx = st.session_state.block_size_idx
    selected_bs_str = st.selectbox("Block Size", bs_options, index=def_idx, key="bs_selector")
    
    # 更新记忆
    new_idx = bs_options.index(selected_bs_str)
    if new_idx != st.session_state.block_size_idx:
        st.session_state.block_size_idx = new_idx
        st.rerun()

    # 将选项字符串转为 key
    if selected_bs_str == "1/64/128/256": bs_key = 'small'; bs_int = 256
    else: bs_key = int(selected_bs_str); bs_int = int(selected_bs_str)

    log_txt = st.text_area("Log Content", height=200, placeholder="Block start at ... = 20 ms")
    
    if st.button("解析并添加"):
        lines = log_txt.split('\n')
        added = []
        p1 = r"Block start at (\d+) .* = (\d+) ms"
        p2 = r"Block start at (\d+) .* Read error: (.*)"
        
        for l in lines:
            m1 = re.search(p1, l)
            m2 = re.search(p2, l)
            if m1:
                lba_s = int(m1.group(1))
                ms = int(m1.group(2))
                grade = get_grade(ms, bs_key)
                added.append(f"{lba_s}-{lba_s + bs_int - 1}|{grade}")
            elif m2:
                lba_s = int(m2.group(1))
                grade = 'error'
                added.append(f"{lba_s}-{lba_s + bs_int - 1}|{grade}")
        
        if added:
            st.session_state.raw_data += ("\n" if st.session_state.raw_data else "") + "\n".join(added)
            st.rerun()

# --- 7. 主界面布局 ---
col_main_ui, col_viz = st.columns([1, 1.8])

with col_main_ui:
    st.subheader("📝 数据录入")
    
    # 按钮组
    c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 1])
    with c_btn1: 
        if st.button("🪄 Log助手", use_container_width=True): log_helper()
    with c_btn2: 
        if st.button("🚀 更新图表", type="primary", use_container_width=True): pass # Trigger rerun
    with c_btn3:
        # CSV 导出逻辑
        export_data = []
        for line in st.session_state.raw_data.strip().split('\n'):
            if '|' in line:
                p = line.split('|')
                export_data.append({'range': p[0], 'level': p[1]})
        if export_data:
            csv_str = pd.DataFrame(export_data).to_csv(index=False).encode('utf-8')
            st.download_button("💾 导出CSV", csv_str, "hdd_scan.csv", "text/csv", use_container_width=True)

    # 文本框
    st.session_state.raw_data = st.text_area("输入 (LBA范围|Level|点数)", 
                                             value=st.session_state.raw_data, 
                                             height=500)
    
    # 图例表
    st.markdown("---")
    st.caption("Victoria 等级对照表 (Delay Levels)")
    legend_data = {
        "Level": ["Level 1 (Gray)", "Level 2 (Green)", "Level 3 (Orange)", "Level 4 (Red)", "Error (Blue)"],
        "Description": ["Normal", "Good", "Warning", "Critical", "Read Error"],
        "Color": [COLOR_MAP['level1'], COLOR_MAP['level2'], COLOR_MAP['level3'], COLOR_MAP['level4'], COLOR_MAP['error']]
    }
    st.dataframe(pd.DataFrame(legend_data), hide_index=True, use_container_width=True)


with col_viz:
    # 视图控制
    st.subheader("💿 物理视图")
    # 保持视图状态
    view_opt = st.radio("显示模式", ["Merge All Surfaces", "Individual Surfaces"], 
                        index=0 if st.session_state.view_mode == "Merge All Surfaces" else 1,
                        horizontal=True)
    st.session_state.view_mode = view_opt

    # 解析数据
    plot_items = []
    lines = st.session_state.raw_data.strip().split('\n')
    for line in lines:
        if not line.strip() or '|' not in line: continue
        parts = line.split('|')
        rng = parts[0].strip()
        lvl = parts[1].strip().lower()
        cnt = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        
        if '-' in rng: s, e = map(int, rng.split('-'))
        else: s = e = int(rng)
        
        color = COLOR_MAP.get(lvl, COLOR_MAP['level1'])
        
        # 逻辑：单点、指定点数或小范围画散点；大范围画弧线
        if s == e or cnt > 0:
            num = max(1, cnt)
            for lba in np.linspace(s, e, num):
                _, h, th, r_norm = lba_to_chs(lba, c_heads, A, B, Total_Cyls)
                r_vis = 1.0 - r_norm * (1.0 - r_in_ratio)
                plot_items.append({'type': 'pt', 'h': h, 'r': r_vis, 'th': th, 'c': color})
        else:
            # 弧线逻辑
            _, h1, th1, rn1 = lba_to_chs(s, c_heads, A, B, Total_Cyls)
            _, h2, th2, rn2 = lba_to_chs(e, c_heads, A, B, Total_Cyls)
            r_vis = 1.0 - rn1 * (1.0 - r_in_ratio)
            
            # 如果起始和结束不在同一个圆环(radius)或者跨度极大，
            # 为了避免画图混乱，建议降级为画点，或者只画一段
            is_same_cyl = (rn1 == rn2) 
            
            if h1 == h2 and is_same_cyl:
                # 同柱面同磁头：正常画弧
                 plot_items.append({'type': 'arc', 'h': h1, 'r': r_vis, 't1': th1, 't2': th2, 'c': color})
            else:
                # 跨磁头或跨柱面
                # 简化处理：画一段完整的弧代表这个区域繁忙
                # 或者：只画起点到终点的连线可能不准确，这里改为画几个离散点或者一段特定弧
                # 下面是一个简化的“单圈处理”，防止报错：
                
                if not is_same_cyl:
                     # 跨柱面了，简单起见，只画起点所在磁头的剩余部分
                     plot_items.append({'type': 'arc', 'h': h1, 'r': r_vis, 't1': th1, 't2': 2*np.pi, 'c': color})
                else:
                    # 同柱面，跨磁头 (h1 -> h2)
                    if h1 < h2:
                        plot_items.append({'type': 'arc', 'h': h1, 'r': r_vis, 't1': th1, 't2': 2*np.pi, 'c': color})
                        for mh in range(h1+1, h2):
                            plot_items.append({'type': 'arc', 'h': mh, 'r': r_vis, 't1': 0, 't2': 2*np.pi, 'c': color})
                        plot_items.append({'type': 'arc', 'h': h2, 'r': r_vis, 't1': 0, 't2': th2, 'c': color})
                    else:
                        # h1 > h2 这种情况通常不会在同柱面发生(除非数据排序错)，
                        # 但如果是物理柱面一样计算出了误差，就按点画
                        pass
                    
    # 绘图辅助函数
    def draw_background(ax, r_in):
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.0) # 消除边缘缝隙
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['polar'].set_visible(False)
        
        # 背景色：灰色区域 (Ring)
        ax.fill_between(np.linspace(0, 2*np.pi, 100), r_in, 1.0, color='#F0F0F0', alpha=0.5)
        # 内外边界
        ax.plot(np.linspace(0, 2*np.pi, 100), [r_in]*100, color='black', lw=0.8)
        ax.plot(np.linspace(0, 2*np.pi, 100), [1.0]*100, color='black', lw=1.2) # 外圈加粗

        # 辅助线 a: 容量百分比同心圆
        for cap_pct in [0.25, 0.50, 0.75]:
            r_cap = capacity_percent_to_radius(cap_pct, A, B, Total_Cyls, r_in)
            ax.plot(np.linspace(0, 2*np.pi, 100), [r_cap]*100, color='#888', lw=0.5, ls=':')
            # 标注
            ax.text(np.radians(45), r_cap, f"{int(cap_pct*100)}%", fontsize=6, color='#666')

        # 辅助线 a: 轴线 (仅在 Ring 内)
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            ax.plot([angle, angle], [r_in, 1.0], color='#CCC', lw=0.5)

    # 渲染
    if view_opt == "Merge All Surfaces":
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        draw_background(ax, r_in_ratio)
        
        for p in plot_items:
            if p['type'] == 'pt': 
                ax.scatter(p['th'], p['r'], c=p['c'], s=20, edgecolors='none', alpha=0.9)
            elif p['type'] == 'arc':
                # 处理跨0度
                ts = np.linspace(p['t1'], p['t2'], 50)
                ax.plot(ts, [p['r']]*50, color=p['c'], lw=2, alpha=0.9)
        
        st.pyplot(fig)

    else: # Individual Surfaces
        # 使用 cols 布局，每个图是一个单独的 figure，方便单独放大
        cols = st.columns(4) # 两列排布
        for h_idx in range(c_heads):
            with cols[h_idx % 4]:
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
                draw_background(ax, r_in_ratio)
                ax.set_title(f"Head {h_idx}", y=1.05)
                
                # 筛选当前磁头数据
                h_items = [p for p in plot_items if p['h'] == h_idx]
                for p in h_items:
                    if p['type'] == 'pt': 
                        ax.scatter(p['th'], p['r'], c=p['c'], s=15, edgecolors='none')
                    elif p['type'] == 'arc':
                        ts = np.linspace(p['t1'], p['t2'], 50)
                        ax.plot(ts, [p['r']]*50, color=p['c'], lw=1.5)
                
                st.pyplot(fig) # 独立的 pyplot 允许用户 hover 时单独放大