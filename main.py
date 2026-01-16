import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import yaml
import re
from io import StringIO

# --- 1. 配置与全局常量 ---
st.set_page_config(page_title="HDD Physical Diagnostic V4.4", layout="wide")

# --- CSS 样式注入：解决 Padding 过大问题 ---
st.markdown("""
    <style>
        /* 调整主内容区域的上下 Padding */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
        }
        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0);
            }
    </style>
""", unsafe_allow_html=True)

PRESETS_FILE = "hdd_presets.yaml"

# 等级定义/颜色映射 (Delay Level)
DELAY_LEVELS = {
    'L1':  {'label': 'L1 (Gray)',   'color': "#929292", 'desc': 'Slow'}, 
    'L2':  {'label': 'L2 (Green)',  'color': '#32CD32', 'desc': 'Mid'},
    'L3':  {'label': 'L3 (Orange)', 'color': '#FFA500', 'desc': 'Warning'},
    'L4':  {'label': 'L4 (Red)',    'color': '#FF0000', 'desc': 'Critical'},
    'ERR': {'label': 'ERR (Blue)',  'color': '#0000FF', 'desc': 'Read Error'},
    'BAD': {'label': 'BAD (Black)', 'color': '#000000', 'desc': 'Damaged'}
}

# 延迟等级阈值表 (对应victoria不同检测blocksize的延迟阈值(ms))
DELAY_THRESHOLDS = {
    'small':  [50, 200, 600],       # 1/32/64/128/256
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
if 'hdd_sn' not in st.session_state: st.session_state.hdd_sn = ""
if 'target_preset_idx' not in st.session_state: st.session_state.target_preset_idx = 0

# --- 3. 核心物理计算---
def calculate_zbr_params(lba_max, heads, rpm, s_out, s_in):
    """
    计算 ZBR 物理参数
    假设 SPT (Sectors Per Track) 从外向内线性递减
    """
    rps = rpm / 60.0
    # 扇区：物理 4K，逻辑 512B——LBA=Logical Block Addressing
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
    
    # --- 判别式系数 ---
    # 公式: 0.5*B*H * cyl^2 - A*H * cyl + lba = 0
    # a = 0.5*B*H, b = -A*H, c = lba
    # delta = b^2 - 4ac = (AH)^2 - 4*(0.5BH)*lba = (AH)^2 - 2*B*H*lba    
    if B == 0: # 恒定速度 (非 ZBR)
        cyl_float = lba / (A * H)
    else:
        delta = (A*H)**2 - 2 * B * H * lba
        if delta < 0: delta = 0
        cyl_float = (A*H - np.sqrt(delta)) / (B*H)

    # 物理柱面是整数。cyl_float 是理论连续值，必须向下取整 才能计算出“当前柱面起始位置”
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
    """ 判定等级返回 Key """
    if isinstance(ms_val, str): return 'ERR'
    
    thresholds = DELAY_THRESHOLDS.get(block_size_key, DELAY_THRESHOLDS[2048])
    if ms_val < thresholds[0]: return 'L1'
    if ms_val < thresholds[1]: return 'L2'
    if ms_val < thresholds[2]: return 'L3'
    return 'L4'

# --- 5. UI: 侧边栏配置 ---
presets = load_presets()

with st.sidebar:
    st.title("⚙️ 硬盘参数")
    
    # 硬盘基本信息 (独立于 Preset 之外)
    st.markdown("### 🏷️ 识别信息")
    # 序列号输入 (绑定 session_state) 
    st.session_state.hdd_sn = st.text_input("序列号 (S/N)", 
                                                   value=st.session_state.hdd_sn,
                                                   placeholder="如: WD-WCC1E1ARP1XX")    
    st.divider()
    st.markdown("### 🛠️ 物理规格")
    
    preset_keys = list(presets.keys())
    options_list = preset_keys + ["New Profile"]
    # 如果 target_preset_idx 超出范围 (例如删除了预设)，重置为 0
    if st.session_state.target_preset_idx >= len(options_list):
        st.session_state.target_preset_idx = 0    

    col_mode, col_edit_btn = st.columns([2, 1])
    with col_mode:
        selected_model = st.selectbox("选择预设", options_list, 
                                      index=st.session_state.target_preset_idx,
                                      disabled=st.session_state.edit_mode)
    with col_edit_btn:
        if st.toggle("解锁", value=st.session_state.edit_mode):
            st.session_state.edit_mode = True
        else:
            st.session_state.edit_mode = False

    # 根据选择加载数据
    if selected_model == "New Profile":
        # 默认空模板
        current_data = {'lba_max': 0, 'heads': 1, 'rpm': 7200, 'speed_out': 150.0, 'speed_in': 80.0}
        display_name = "New_HDD"
    else:
        current_data = presets[selected_model]
        display_name = selected_model

    # 表单区域
    # 编辑模式允许修改 Key (Model Name)；另，使用 pop 读取临时导入值，实现一次性自动填充
    with st.container(border=True):
        st.caption("参数详情")

        # --- 自动填入逻辑 ---
        # 优先弹出 import 进来的临时数据，如果没有则使用当前 current_data

        # 1. 型号
        val_model = st.session_state.pop('tmp_imported_model', display_name)        
        new_model_name = st.text_input("型号名称", value=val_model, disabled=not st.session_state.edit_mode)
        #Fix Pylance:new_model_name = st.text_input("型号名称", value=str(val_model if val_model else ""), disabled=not st.session_state.edit_mode)
        # 2. 物理参数
        val_lba = st.session_state.pop('tmp_imported_lba', current_data['lba_max'])
        c_lba = st.number_input("LBA Max", value=int(val_lba), disabled=not st.session_state.edit_mode)

        val_heads = st.session_state.pop('tmp_imported_heads', current_data['heads'])
        c_heads = st.number_input("磁头数 (Heads)", value=int(val_heads), disabled=not st.session_state.edit_mode)

        val_rpm = st.session_state.pop('tmp_imported_rpm', current_data['rpm'])
        c_rpm = st.number_input("转速 (RPM)", value=int(val_rpm), disabled=not st.session_state.edit_mode)

        val_sout = st.session_state.pop('tmp_imported_sout', current_data['speed_out'])
        c_s_out = st.number_input("外圈速度 (MB/s)", value=float(val_sout), disabled=not st.session_state.edit_mode)

        val_sin = st.session_state.pop('tmp_imported_sin', current_data['speed_in'])
        c_s_in = st.number_input("内圈速度 (MB/s)", value=float(val_sin), disabled=not st.session_state.edit_mode)

        if st.session_state.edit_mode:
            if st.button("💾 保存配置到 YAML"):
                if not new_model_name:
                    st.error("型号名称不能为空")
                else:
                    new_entry = {
                        'lba_max': int(c_lba), 'heads': int(c_heads), 'rpm': int(c_rpm),
                        'speed_out': float(c_s_out), 'speed_in': float(c_s_in)
                    }
                    if new_model_name != selected_model and selected_model != "New Profile":
                        if selected_model in presets:
                            del presets[selected_model]
                    
                    presets[new_model_name] = new_entry
                    save_presets(presets)
                    
                    # 保存后，更新选中项索引到这个新名字
                    st.session_state.target_preset_idx = list(presets.keys()).index(new_model_name)
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
    
    # 更新blocksize选项记忆
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
                added.append(f"{lba_s}-{lba_s + bs_int - 1}|{grade}|0")
            elif m2:
                lba_s = int(m2.group(1))
                grade = 'ERR'
                added.append(f"{lba_s}-{lba_s + bs_int - 1}|{grade}|0")
        
        if added:
            st.session_state.raw_data += ("\n" if st.session_state.raw_data else "") + "\n".join(added)
            st.rerun()

# --- 7. 主界面布局 ---
col_main_ui, col_viz = st.columns([1, 1.8])

# ================= 左侧：控制与图例 =================
with col_main_ui:
    st.subheader("📝 数据录入")
    
    # 定义导入功能的 Dialog
    @st.dialog("📂 导入扫描数据")
    def import_helper():
        st.markdown("上传带有元数据的 CSV 文件。")
        st.caption("必需列名: `range`, `level`")
        
        uploaded_file = st.file_uploader("选择 CSV 文件", type=["csv"])
        if uploaded_file is not None:
            try:
                # 1. 读取文件
                content = uploaded_file.getvalue().decode("utf-8").splitlines()                
                if not content:
                    st.error("文件为空")
                    return
                
                # 2. 解析第一行 Metadata
                header_line = content[0]
                # Header format: Model: ...; LBA: ...; Heads: ...; RPM: ...; SO: ...; SI: ...; SN: ...
                meta_pattern = r"Model: (.*); SN: (.*); LBA: (\d+); Heads: (\d+); RPM: (\d+); Speed: ([\d\.]+)/([\d\.]+)"
                match = re.search(meta_pattern, header_line)
                
                parsed_meta = {}
                csv_start_line = 0
                # 匹配 Model
                if match:
                    parsed_meta['model'] = match.group(1).strip()
                    parsed_meta['sn'] = match.group(2).strip()
                    parsed_meta['lba'] = int(match.group(3))
                    parsed_meta['heads'] = int(match.group(4))
                    parsed_meta['rpm'] = int(match.group(5))
                    parsed_meta['s_out'] = float(match.group(6))
                    parsed_meta['s_in'] = float(match.group(7))

                    csv_start_line = 1 # 跳过第一行
                    st.success(f"识别到硬盘: {parsed_meta['model']} (SN: {parsed_meta['sn']})")
                else:
                    st.warning("未检测到标准元数据头，将作为普通 CSV 读取。")

                # 3. 解析数据部分 (跳过第一行 Metadata)
                # 将剩余内容重新组合供 pandas 读取
                csv_body = "\n".join(content[csv_start_line:])
                df = pd.read_csv(StringIO(csv_body))

                # 校验
                required_cols = ['range', 'level'] # count 可选
                if not all(col in df.columns for col in required_cols):
                    st.error(f"CSV 格式错误：缺少必要的列 {required_cols}")
                else:
                    # 预览
                    st.dataframe(df.head(3), hide_index=True, use_container_width=True)
                    
                    new_lines = []
                    for _, row in df.iterrows():
                        rng = str(row['range'])
                        lvl = str(row['level'])
                        # 读取 count 列，如果没有则默认为 0
                        cnt = row['count'] if 'count' in df.columns and pd.notna(row['count']) else 0
                        # 格式: range|level|count
                        new_lines.append(f"{rng}|{lvl}|{int(cnt)}")

                    new_data_str = "\n".join(new_lines)

                    col_overwrite, col_append = st.columns(2)

                    with col_overwrite:
                        if st.button("🗑️ 覆盖并应用参数", type="primary", use_container_width=True):
                            # 更新数据
                            st.session_state.raw_data = new_data_str
                            
                            # 如果有元数据，强制更新当前设置
                            if match:
                                imp_model = parsed_meta['model']
                                st.session_state.hdd_sn = parsed_meta['sn']

                                preset_match_key = None
                                if imp_model in presets:
                                    preset_match_key = imp_model
                                
                                # 准备要写入侧边栏输入框的临时数据
                                st.session_state.tmp_imported_model = imp_model
                                st.session_state.tmp_imported_lba = parsed_meta['lba']
                                st.session_state.tmp_imported_heads = parsed_meta['heads']
                                st.session_state.tmp_imported_rpm = parsed_meta['rpm']
                                st.session_state.tmp_imported_sout = parsed_meta['s_out']
                                st.session_state.tmp_imported_sin = parsed_meta['s_in']
                                
                                # 存在同名预设
                                if preset_match_key:
                                    p_data = presets[preset_match_key]
                                    # 检查参数一致性
                                    is_identical = (
                                        p_data['lba_max'] == parsed_meta['lba'] and
                                        p_data['heads'] == parsed_meta['heads'] and
                                        p_data['rpm'] == parsed_meta['rpm'] and
                                        p_data['speed_out'] == parsed_meta['s_out'] and
                                        p_data['speed_in'] == parsed_meta['s_in']
                                    )
                                    # 设置 Selectbox 指向该预设
                                    idx = list(presets.keys()).index(preset_match_key)
                                    print(f'idx={idx}')
                                    st.session_state.target_preset_idx = idx

                                    if is_identical:
                                        # 2.1 内容一致 -> 锁定
                                        st.session_state.edit_mode = False
                                        st.toast(f"参数与预设 '{imp_model}' 完美匹配。")
                                    else:
                                        # 2.2 内容不一致 -> 解锁并提示
                                        st.session_state.edit_mode = True
                                        st.toast(f"预设 '{imp_model}' 存在但参数不一致，已开启编辑模式。", icon="⚠️")
                                else:
                                    # 情况 3: 不存在 -> 指向 New Profile
                                    # New Profile 是列表最后一项
                                    st.session_state.target_preset_idx = len(presets.keys()) 
                                    st.session_state.edit_mode = True
                                    st.toast(f"新检测到型号 '{imp_model}'，已切换至 New Profile。", icon="🆕")

                            st.rerun()
                    
                    with col_append:
                        if st.button("➕ 仅追加数据", use_container_width=True):
                            if st.session_state.raw_data.strip():
                                st.session_state.raw_data = st.session_state.raw_data.strip() + "\n" + new_data_str
                            else:
                                st.session_state.raw_data = new_data_str
                            st.rerun()
            except Exception as e:
                st.error(f"读取失败: {e}")

    # 按钮组
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns([1, 1.1, 1.1, 1.1])
    with c_btn1: 
        if st.button("🪄 Log助手", use_container_width=True): log_helper()

    with c_btn2:
        if st.button("📂 导入CSV", use_container_width=True): import_helper()

    with c_btn4: 
        if st.button("🚀 更新图表", type="primary", use_container_width=True): pass # Trigger rerun
    
    with c_btn3:
        # CSV 导出逻辑
        export_list = []
        lines = st.session_state.raw_data.strip().split('\n')
        for line in lines:
            if not line.strip() or '|' not in line: continue
            p = line.split('|')
            # 清洗数据
            r_val = p[0].strip()
            l_val = p[1].strip()
            # 获取点数，缺省为 0
            c_val = int(p[2]) if len(p) > 2 and p[2].strip().isdigit() else 0

            export_list.append({'range': r_val, 'level': l_val, 'count': c_val})
            
        if export_list:
            current_model_name = new_model_name if 'new_model_name' in locals() else selected_model
            current_model_name = str(current_model_name) if current_model_name else "Unknown" # Pylance guard
            safe_model = re.sub(r'[\\/*?:"<>|]', '_', current_model_name).strip()
            safe_sn = re.sub(r'[\\/*?:"<>|]', '_', st.session_state.hdd_sn).strip()
            if not safe_sn: safe_sn = "NoSN"
            
            filename = f"BadSectors_{safe_model}_{safe_sn}.csv"
            
            # 文件内容
            # Header: Model: ...; Capacity ...; SN: ...
            header_str = (f"Model: {current_model_name}; SN: {st.session_state.hdd_sn}; "
                          f"LBA: {int(c_lba)}; Heads: {int(c_heads)}; RPM: {int(c_rpm)}; "
                          f"Speed: {float(c_s_out)}/{float(c_s_in)}\n")
            # CSV Body
            df = pd.DataFrame(export_list)
            csv_body = df.to_csv(index=False)
            final_csv_content = header_str + csv_body
            
            st.download_button("💾 导出CSV", 
                               final_csv_content, 
                               filename, 
                               "text/csv", 
                               use_container_width=True)
        else:
            st.button("💾 导出CSV", disabled=True, use_container_width=True)

    # 1. 新增功能：等级过滤器
    # 默认全选，获取 LEVELS 的所有 key
    all_levels = list(DELAY_LEVELS.keys())
    selected_levels = st.multiselect(
        "👁️ 视图过滤器 (显示特定等级)",
        options=all_levels,
        default=all_levels
    )

    # 文本框
    st.session_state.raw_data = st.text_area("输入 (LBA范围|Level|点数)", 
                                             value=st.session_state.raw_data, 
                                             height=400,
                                             help="支持格式：\n100-200|L4\n5000|ERR")
    
    # 图例表
    st.markdown("---")
    st.caption("颜色等级对照 (Victoria Delay Levels)")
    cols = st.columns(len(DELAY_LEVELS))
    for i, (k, v) in enumerate(DELAY_LEVELS.items()):
        with cols[i]:
            # HTML 圆点 + 文字居中
            st.markdown(f"""
                <div style='
                    background-color:{v['color']};
                    height:20px;
                    width:20px;
                    border-radius:50%;
                    margin-bottom:5px;
                    border: 1px solid #ccc;'>
                </div>
                """, unsafe_allow_html=True)
            # 显示描述
            st.caption(f"**{k}**")
            st.caption(f"*{v['desc']}*")


with col_viz:
    # 视图控制
    st.subheader("💿 物理视图")

    # c_ctrl1 单选框，c_ctrl2 滑块
    c_ctrl1, c_ctrl2 = st.columns([1, 1], gap="medium")
    with c_ctrl1:
        view_opt = st.radio("显示模式", ["Merge All Surfaces", "Individual Surfaces"], 
                            index=0 if st.session_state.view_mode == "Merge All Surfaces" else 1,
                            horizontal=True)
        # 保持视图状态
        st.session_state.view_mode = view_opt

    cols_per_row = 4
    with c_ctrl2:
        # 仅在分层视图下显示滑块
        if view_opt == "Individual Surfaces":
            slider_max = min(max(1, c_heads), 8)
            slider_default = min(4, slider_max)
            cols_per_row = st.slider("每行图表数", min_value=1, max_value=slider_max, value=slider_default, key="cols_slider")

    # 解析数据
    plot_items = []
    lines = st.session_state.raw_data.strip().split('\n')

    for line in lines:
        if not line.strip() or '|' not in line: continue
        parts = line.split('|')
        rng = parts[0].strip()

        lvl = parts[1].strip().upper()        
        # 过滤：如果不在多选框中，直接跳过
        if lvl not in selected_levels:
            continue

        cnt = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        
        if '-' in rng: s, e = map(int, rng.split('-'))
        else: s = e = int(rng)
        
        color = DELAY_LEVELS.get(lvl, DELAY_LEVELS['L1'])['color']
        
        # 逻辑：单点、指定点数或小范围画散点；大范围画弧线
        if s == e or cnt > 0:
            # 散点模式
            num = max(1, cnt)
            for lba in np.linspace(s, e, num):
                c, h, th, r_norm = lba_to_chs(lba, c_heads, A, B, Total_Cyls)
                r_vis = 1.0 - r_norm * (1.0 - r_in_ratio)
                plot_items.append({'type': 'pt', 'h': h, 'r': r_vis, 'th': th, 'c': color})
        else:
            # 弧线模式 (Range Mode)
            # 获取起点和终点的完整坐标、整数柱面索引 c1, c2
            c1, h1, th1, rn1 = lba_to_chs(s, c_heads, A, B, Total_Cyls)
            c2, h2, th2, rn2 = lba_to_chs(e, c_heads, A, B, Total_Cyls)            
            # 计算各自的可视化半径 (跨柱面时半径不同)
            r_vis1 = 1.0 - rn1 * (1.0 - r_in_ratio)
            r_vis2 = 1.0 - rn2 * (1.0 - r_in_ratio)
            
            if c1 == c2:
                # 情况 A: 完全在同一个柱面、同一个磁头上 -> 画一条简单的弧
                if h1 == h2:
                    plot_items.append({'type': 'arc', 'h': h1, 'r': r_vis1, 't1': th1, 't2': th2, 'c': color})
            
                # 情况 B: 同一柱面，但跨磁头 (例如 Head 0 末尾 -> Head 1 开头)
                else:
                    # 1. 起点磁头：从 th1 画到 2pi (一圈结束)
                    plot_items.append({'type': 'arc', 'h': h1, 'r': r_vis1, 't1': th1, 't2': 2*np.pi, 'c': color})
                    
                    # 2. 中间磁头：画整圈 (如果跨了多个磁头)
                    # 磁头写入顺序 0->1->2...,不应该出现h1 > h2
                    if h1 + 1 < h2:
                        for mh in range(h1 + 1, h2):
                            plot_items.append({'type': 'arc', 'h': mh, 'r': r_vis1, 't1': 0, 't2': 2*np.pi, 'c': color})
                    # 3. 终点磁头：从 0 画到 th2
                    plot_items.append({'type': 'arc', 'h': h2, 'r': r_vis1, 't1': 0, 't2': th2, 'c': color})

            # 情况 C: 跨柱面
            # 如：Cyl 100/Head 1(End) -> Cyl 102/Head 0(Start)
            # 则
            #   若c2-c1=1: c1: h1 ->  h_end, c2: h0 -> h2;
            #   若c2-c1>1: 各head全部画满一圈表达之
            else:
                # 起点 -> 该磁道末尾
                plot_items.append({'type': 'arc', 'h': h1, 'r': r_vis1, 't1': th1, 't2': 2*np.pi, 'c': color})
                if c2 - c1 == 1:
                    # 起点 -> 后续磁头
                    for mh in range(h1 + 1, c_heads):
                        plot_items.append({'type': 'arc', 'h': mh, 'r': r_vis1, 't1': 0, 't2': 2*np.pi, 'c': color})
                    # 首磁头 -> 终点
                    for mh in range(0, h2):
                        plot_items.append({'type': 'arc', 'h': mh, 'r': r_vis2, 't1': 0, 't2': 2*np.pi, 'c': color})
                #全部画一圈，注：【这里半径r用的是vis1
                else:
                    for mh in range(0, c_heads):
                        plot_items.append({'type': 'arc', 'h': mh, 'r': r_vis1, 't1': 0, 't2': 2*np.pi, 'c': color})
                # 终点所在位置 -> 该磁道开头
                plot_items.append({'type': 'arc', 'h': h2, 'r': r_vis2, 't1': 0, 't2': th2, 'c': color})

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
            ax.text(np.radians(45), r_cap, f"{int(cap_pct*100)}%", fontsize=6, color='#666')

        # 辅助线 a: 轴线 (仅在 Ring 内)
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            ax.plot([angle, angle], [r_in, 1.0], color='#CCC', lw=0.5, ls=':')

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
                ax.plot(ts, [p['r']]*50, color=p['c'], lw=1, alpha=0.9)
        st.pyplot(fig)

    else: # Individual Surfaces
        total_rows: int = (c_heads + cols_per_row - 1) // cols_per_row #type: ignore
        for row in range(total_rows):
            cols = st.columns(cols_per_row)
            for i in range(cols_per_row):
                h_idx = row * cols_per_row + i
                if h_idx < c_heads:
                    with cols[i]:
                        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
                        draw_background(ax, r_in_ratio)
                        ax.set_title(f"Head {h_idx}", y=1.05)
                        
                        # 筛选数据
                        h_items = [p for p in plot_items if p['h'] == h_idx]                        
                        for p in h_items:
                            if p['type'] == 'pt': 
                                ax.scatter(p['th'], p['r'], c=p['c'], s=15, edgecolors='none')
                            elif p['type'] == 'arc':
                                ts = np.linspace(p['t1'], p['t2'], 50)
                                ax.plot(ts, [p['r']]*50, color=p['c'], lw=0.6)
                        
                        st.pyplot(fig)# 独立的 pyplot 允许 hover 时单独放大