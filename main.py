import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import yaml
import re
from io import StringIO
from dotenv import load_dotenv, set_key
import glob

# --- 1. 配置与全局常量 ---
st.set_page_config(page_title="HDD Physical Diagnostic V4.6", layout="wide")

if 'pending_toast' in st.session_state and st.session_state.pending_toast:
    st.toast(st.session_state.pending_toast['msg'], duration=st.session_state.pending_toast.get('duration'))
    st.session_state.pending_toast = None # 清空

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
INVENTORY_DIR = "Data"
HIST_DIR = os.path.join(INVENTORY_DIR, "History")
INVENTORY_FILE = os.path.join(INVENTORY_DIR, "hdd_inventory.yaml")
if not os.path.exists(INVENTORY_DIR):
    os.makedirs(INVENTORY_DIR)

# 加载环境变量
ENV_FILE = ".env"
if not os.path.exists(ENV_FILE):
    with open(ENV_FILE, "w") as f: f.write("")
load_dotenv(ENV_FILE)

# 等级定义/颜色映射 (Delay Level)
DELAY_LEVELS = {
    'L1':  {'label': 'L1 (Gray)',   'color': "#929292", 'desc': 'Slow'}, 
    'L2':  {'label': 'L2 (Green)',  'color': '#32CD32', 'desc': 'Mid'},
    'L3':  {'label': 'L3 (Orange)', 'color': '#FFA500', 'desc': 'Warning'},
    'L4':  {'label': 'L4 (Red)',    'color': '#FF0000', 'desc': 'Critical'},
    'ERR': {'label': 'ERR (Blue)',  'color': '#0000FF', 'desc': 'Read Error'},
    'BAD': {'label': 'BAD (Black)', 'color': '#000000', 'desc': 'Damaged'}
}

# victoria 的Block Size 的映射
BLOCK_SIZES = {
    "1/64/128/256": 256, "512": 512, "1024": 1024, "2048": 2048, 
    "4096": 4096, "8192": 8192, "16384": 16384, "32768": 32768, "65535": 65535
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

NEW_PROFILE = "New Profile"

# --- 2. 状态初始化 ---
if 'block_size_key' not in st.session_state: st.session_state.block_size_key = "2048"
if 'view_mode' not in st.session_state: st.session_state.view_mode = "Merge All Surfaces"
if 'raw_data' not in st.session_state: st.session_state.raw_data = ""
if 'profile_edit_mode' not in st.session_state: st.session_state.profile_edit_mode = False
if 'hdd_sn' not in st.session_state: st.session_state.hdd_sn = ""
if 'selected_preset' not in st.session_state: st.session_state.selected_preset = None

# --- 3. 核心物理计算---
def calculate_zbr_params(lba_max, heads, rpm, s_out, s_in):
    """
    计算 ZBR 物理参数
    假设 SPT (Sectors Per Track) 从外向内线性递减
    """
    try:
        lba_max = float(lba_max)
        rps = float(rpm) / 60.0
        if rps <= 0: return 0, 0, 1.0, 0, 0

        # 扇区：物理 4K，逻辑 512B——LBA=Logical Block Addressing
        spt_out = (s_out * 1_000_000) / (512.0 * rps)
        spt_in = (s_in * 1_000_000) / (512.0 * rps)
        
        # 平均 SPT * 磁头数 * 磁道数 = 总 LBA
        avg_spt_per_cyl = (spt_out + spt_in) / 2.0 * float(heads)
        if avg_spt_per_cyl <= 0: avg_spt_per_cyl = 1.0
        total_cylinders = lba_max / avg_spt_per_cyl
        if total_cylinders <= 0: total_cylinders = 1.0
        
        # 线性方程系数: SPT(cyl) = A - B * cyl
        A = spt_out
        B = (spt_out - spt_in) / total_cylinders
        
        return A, B, total_cylinders, spt_out, spt_in
    except Exception:
            return 0, 0, 1.0, 0, 0

def lba_to_chs(lba, heads, A, B, total_cyls):
    """
    LBA 转 CHS (Cylinder, Head, Sector) 及 归一化半径 (修正版)
    """
    try:
        H = float(heads)
        lba = float(lba)
        epsilon = 1e-9 # 浮点容差
        
        # --- 判别式系数 ---
        # 公式: 0.5*B*H * cyl^2 - A*H * cyl + lba = 0
        # a = 0.5*B*H, b = -A*H, c = lba
        # delta = b^2 - 4ac = (AH)^2 - 4*(0.5BH)*lba = (AH)^2 - 2*B*H*lba    
        if B == 0: # 恒定速度 (非 ZBR)
            cyl_float = lba / (A * H) if (A*H) > 0 else 0
        else:
            delta = (A*H)**2 - 2 * B * H * lba
            if delta < 0: delta = 0
            cyl_float = (A*H - np.sqrt(max(0, delta))) / (B*H)

        # 物理柱面是整数。cyl_float 是理论连续值，必须向下取整 才能计算出“当前柱面起始位置”
        cyl_int = int(cyl_float + epsilon)    
        # 防止浮点误差导致的越界
        if cyl_int >= total_cyls: cyl_int = int(total_cyls) - 1
        if cyl_int < 0: cyl_int = 0

        # 计算该柱面(整数)的起始 LBA
        # LBA_start = H * (A*C - 0.5*B*C^2)
        c_val = float(cyl_int)
        lba_start_cyl = H * (A*c_val - 0.5*B*(c_val**2))
        
        #  计算在当前柱面内的偏移量
        lba_in_cyl = lba - lba_start_cyl
        # 计算磁头 (Head) 和 角度 (Theta)
        # 注意：lba_in_cyl 可能因为浮点误差出现微小的负数或略大于容量，需由于 int() 截断
        if lba_in_cyl < 0: lba_in_cyl = 0.0

        # 当前柱面的 SPT (使用整数索引计算)
        current_spt = A - B * c_val    
        if current_spt < 1.0: current_spt = 1.0
        
        # 计算磁头 (Head)
        head = int((lba_in_cyl + epsilon) // current_spt)
        if head >= heads: head = heads - 1 # 钳位
        
        sector_offset = lba_in_cyl % current_spt
        
        # 计算角度 (0~2pi)
        # 加上偏移量让它不要总是从0度开始，或者保持原样。这里保持原样。
        theta = (sector_offset / current_spt) * 2 * np.pi
        
        # 6. 归一化半径
        norm_cyl = cyl_int / total_cyls
        if norm_cyl > 1.0: norm_cyl = 1.0
        
        return cyl_int, head, theta, norm_cyl
    except Exception:
            return 0, 0, 0.0, 0.0

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
def load_inventory():
    if not os.path.exists(INVENTORY_FILE):
        return {}
    with open(INVENTORY_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def save_inventory(data):
    with open(INVENTORY_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

def register_hdd(sn, model, associated_file=None, memo=None):
    """ 注册或更新硬盘信息 """
    if not sn: return False
    inv = load_inventory()
    
    # 如果已存在，保留原有历史，更新模型
    if sn not in inv:
        inv[sn] = {'model': model, 'history': [], 'memo': ''}

    # 更新字段
    inv[sn]['model'] = model
    if memo is not None: # 仅当传入 memo 时才更新，防止覆盖
        inv[sn]['memo'] = memo
        
    # 如果有关联文件，追加到历史记录
    if associated_file:
        if 'history' not in inv[sn]: inv[sn]['history'] = []
        if associated_file not in inv[sn]['history']:
            inv[sn]['history'].append(associated_file)

    save_inventory(inv)
    return True

def delete_hdd(sn):
    """ 删除库存记录 """
    inv = load_inventory()
    if sn in inv:
        del inv[sn]
        save_inventory(inv)
        return True
    return False

def get_inventory_options(inv_data):
    """ 生成下拉菜单的选项列表: SN - Model (Memo) """
    options = []
    for sn, data in inv_data.items():
        mod = data.get('model', 'Unknown')
        mem = data.get('memo', '')
        # 格式化显示：WD-XXX | Model (备注...)
        display_str = f"{sn} | {mod}"
        if mem:
            short_mem = (mem[:10] + '..') if len(mem) > 10 else mem
            display_str += f" ({short_mem})"
        options.append(display_str)
    return options



# --- ENV 管理 ---
def get_log_path():
    return os.getenv("VICTORIA_LOG_PATH", "")
def save_log_path(path):
    # 更新内存环境变量
    os.environ["VICTORIA_LOG_PATH"] = path
    # 写入文件
    set_key(ENV_FILE, "VICTORIA_LOG_PATH", path)

# --- Victoria Log 解析 ---
def parse_victoria_filename(filename):
    """
    从文件名提取 Model 和 SN
    格式示例: bads_WDC WD40EFRX-68WT0N0_WD-WCC4E7ARP4XF.txt
    假设格式为: 前缀_型号_序列号.txt
    """
    basename = os.path.basename(filename)
    # 去除扩展名
    name_body = os.path.splitext(basename)[0]
    
    # 简单的正则尝试： bads_(Model)_(SN)
    if name_body.startswith("bads_"):
        content = name_body[5:] # 去掉 bads_
        if "_" in content:
            # rsplit 限制分割1次，确保 SN 独立，剩余部分归为 Model
            model, sn = content.rsplit("_", 1)
            return model.strip(), sn.strip()
    
    return "Unknown_Model", "Unknown_SN"

def parse_victoria_content(file_content):
    """
    解析 Victoria 日志内容
    目标 Pattern: "103651840, 2048  ;53 GB  Scan bad"
    """
    lines = file_content.splitlines()
    parsed_lines = []
    
    # 正则: 数字, 数字 ;... Scan bad
    # Group 1: Start LBA, Group 2: Block Size
    pattern = re.compile(r"^\s*(\d+),\s*(\d+)\s*;.*Scan bad", re.IGNORECASE)
    
    for line in lines:
        match = pattern.search(line)
        if match:
            lba_start = int(match.group(1))
            block_size = int(match.group(2))
            lba_end = lba_start + block_size - 1
            
            # 格式化为标准输入: Range|Level|Count|GB|Memo
            # Level 强制为 L4
            # Memo 保留原始行信息"Scan bad" 部分供参考            
            # 不计算 [GB] 标签，交给 format_columns 统一处理
            raw_suffix = line.split(";")[-1].strip() if ";" in line else "Scan bad"
            clean_memo = re.sub(r'^[\d\.]+\s*GB\s*', '', raw_suffix, flags=re.IGNORECASE).strip()
            final_memo = clean_memo if clean_memo else "Scan error"

            row_str = f"{lba_start}-{lba_end}|L4|0||{final_memo}"
            parsed_lines.append(row_str)
            
    return parsed_lines

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

def format_columns(raw_text, sort=False):
    """
    5列格式化    Range | Level | Count | [xx.xxGB] | Memo
    """
    if not raw_text: return ""    
    lines = raw_text.split('\n')
    parsed_rows = []
    
    for line in lines:
        if not line.strip():
            if not sort: parsed_rows.append({'type': 'blank', 'content': line})
            continue
        if '|' not in line:
            parsed_rows.append({'type': 'str', 'content': line})
            continue
        
        parts = [p.strip() for p in line.split('|')]
        # 补齐列数到 5 列 (Range, Level, Count, Memo)
        while len(parts) < 5: parts.append("")

        # RANGE, 去空格
        rng = parts[0].strip().replace(" ", "")

        # 计算 GB (忽略输入值，总是重新计算以保证准确)
        #   起始点
        lba_start = 0
        m = re.match(r'^(\d+)', rng)
        if m: lba_start = int(m.group(1))
        gb_val_start = lba_start * 512 / (1000**3)

        #   末端点
        lba_end = None
        if '-' in rng:
            rng_parts = rng.split('-')
            if len(rng_parts) > 1 and rng_parts[1].isdigit():
                    lba_end = int(rng_parts[1])

        #   str
        gb_str = f"[{gb_val_start:.2f}GB]"
        if lba_end:
            gb_val_end = lba_end * 512 / (1000**3)
            if gb_val_end - gb_val_start > 0.01:
                gb_str = f"[{gb_val_start:.2f}-{gb_val_end:.2f}GB]"

        parsed_rows.append({
            'type': 'data',
            'sort_key': lba_start,
            'col1': rng,
            'col2': parts[1].upper() if parts[1] and parts[1].upper() in DELAY_LEVELS else "ERR", # Level 校验
            'col3': parts[2] if parts[2].isdigit() else "0", # Count 校验
            'col4': gb_str,
            'col5': parts[4]
        })

    # 排序,丢弃空行
    if sort:
        data_rows = [r for r in parsed_rows if r['type'] == 'data']
        str_rows = [r for r in parsed_rows if r['type'] == 'str']
        data_rows.sort(key=lambda x: x['sort_key'])
        parsed_rows = data_rows + str_rows

    # 对齐宽度
    w1, w2, w3, w4 = 20, 0, 0, 12   # 最小列宽
    for r in parsed_rows:
        if r['type'] == 'data':
            w1 = max(w1, len(r['col1']))
            w2 = max(w2, len(r['col2']))
            w3 = max(w3, len(r['col3']))
            w4 = max(w4, len(r['col4']))
    
    # 重组
    final_lines = []
    for r in parsed_rows:
        if r['type'] == 'raw':
            final_lines.append(r['content'])
        else:
            line = f"{r['col1'].ljust(w1)} | {r['col2'].ljust(w2)} | {r['col3'].ljust(w3)} | {r['col4'].ljust(w4)} | {r['col5']}"
            final_lines.append(line)

    return "\n".join(final_lines)


# --- 5. UI: 侧边栏配置 ---
presets = load_presets()


# --- 定义库存管理弹窗 ---
@st.dialog("📦 资产列表管理", width="large")
def inventory_manager_dialog():
    inv_data = load_inventory()
    
    if not inv_data:
        st.info("暂无库存记录，请在侧边栏注册新设备。")
        if st.button("关闭"): st.rerun()
    else:
        st.caption("勾选 **Load** 加载配置，双击 **Memo** 修改备注。勾选 **删除** 移除记录")
        
        # --- A. 数据转换 ---
        table_data = []
        sorted_keys = sorted(inv_data.keys())

        for sn_key in sorted_keys:
            info = inv_data[sn_key]
            history_list = info.get('history', [])
            history_str = ", ".join(history_list) if history_list else ""

            table_data.append({
                "加载": False,
                "删除": False, # 新增删除列
                "SN": sn_key,
                "Model": info['model'],
                "Memo": info.get('memo', ''),
                "History": history_str
            })
        
        df = pd.DataFrame(table_data)

        # --- B. 渲染宽屏表格 ---
        edited_df = st.data_editor(
        df,
        key="inventory_editor_dialog",
        hide_index=True,
        width='stretch',
        height=400,
        disabled=["SN", "Model", "History"],
        column_config={
            "加载": st.column_config.CheckboxColumn("Load", width="small"),
            "删除": st.column_config.CheckboxColumn("Del", width="small"), # 删除列配置
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "SN": st.column_config.TextColumn("Serial Number", width="medium"),
            "Memo": st.column_config.TextColumn("Memo (可编辑)", width="medium"),
            "History": st.column_config.TextColumn("History Files", width="large", help="关联的历史CSV文件")
        }
    )

        # --- C. 逻辑处理 ---
        
        # C1. 自动保存 Memo 修改
        # Streamlit 的 dialog 在内部交互时保持开启，不会因为数据刷新而关闭
        is_changed = False
        for index, row in edited_df.iterrows():
            sn_key = row['SN']
            new_memo = row['Memo']
            if inv_data[sn_key].get('memo', '') != new_memo:
                inv_data[sn_key]['memo'] = new_memo
                is_changed = True
        
        if is_changed:
            save_inventory(inv_data)
            st.toast("备注已保存 ✅")
        
        # C2. 处理删除动作
        rows_to_delete = edited_df[edited_df["删除"] == True]
        if not rows_to_delete.empty:
            st.divider()
            with st.container(border=True):
                st.markdown("#### ⚠️ 删除确认")
                
                # 列出即将删除的 SN
                delete_sns = rows_to_delete['SN'].tolist()
                st.warning(f"⚠️ 确定要永久删除设备`{delete_sns}`及其关联记录吗？ {len(delete_sns)} 条记录")
                
                col_del_conf, col_del_cancel = st.columns([1, 4])
                
                # 确认按钮
                with col_del_conf:
                    if st.button("🚨 确认删除", type="primary", width='stretch'):
                        for sn in delete_sns:
                            delete_hdd(sn)
                        st.toast(f"已删除 {len(delete_sns)} 条记录")
                        st.rerun() # 刷新以更新表格
                
                # 提示文本
                with col_del_cancel:
                    st.caption("取消：取消上方表格中的“删除”勾选")

        # C3. 处理加载动作 (分步确认)
        selected_rows = edited_df[edited_df["加载"] == True]
        
        if not selected_rows.empty:
            # 取最后勾选的一个
            target_row = selected_rows.iloc[-1]
            target_sn = target_row['SN']
            target_model = target_row['Model']
            
            # 获取真实的历史文件列表 (从 inv_data 取，因为 df 里是字符串)
            history_files = inv_data[target_sn].get('history', [])
            
            st.divider()
            st.markdown(f"#### 📥 准备加载: `{target_sn}`")
            
            # --- 二级确认区 ---
            c_conf, c_act = st.columns([3, 1])
            
            target_file_path = None
            load_csv_data = False
            
            with c_conf:
                # 如果有历史文件，询问是否加载
                if history_files:
                    # 默认选择最新的一个（假设列表最后是新的）
                    target_file = st.selectbox("是否同步读取历史 CSV 数据？", 
                                            options=["不读取 (仅加载参数)"] + history_files[::-1],
                                            index=1 if history_files else 0)
                    
                    if target_file != "不读取 (仅加载参数)":
                        load_csv_data = True
                        target_file_path = os.path.join(HIST_DIR, target_file)
                else:
                    st.info("此设备无关联的历史 CSV 文件，仅加载物理参数。")

            with c_act:
                st.write("") # Spacer
                if st.button("🚀 确认执行", type="primary", width='stretch'):
                    # 1. 设置 SN
                    st.session_state.hdd_sn = target_sn
                    st.session_state["sn_input_widget"] = target_sn
                    msg_list = []
                    
                    # 2. 设置 Model
                    if target_model in presets:
                        st.session_state.selected_preset = target_model
                        st.session_state.tmp_imported_model = target_model
                        st.session_state.edit_mode = False
                        msg_list.append(f"参数: {target_model}。\r\n")
                    else:
                        msg_list.append(f"预设缺失: {target_model}，仅加载 SN。\r\n")

                    # 3. 读取 CSV (如果选择了)
                    if load_csv_data and target_file_path:
                        # 尝试读取文件
                        if os.path.exists(target_file_path):
                            try:
                                df_csv = pd.read_csv(target_file_path, encoding='utf-8')
                                df_csv = df_csv.fillna("")

                                if 'range' in df_csv.columns and 'level' in df_csv.columns:
                                    new_lines = []
                                    for _, r in df_csv.iterrows():
                                        rng = str(r['range'])
                                        lvl = str(r['level'])
                                        cnt = r['count'] if 'count' in df_csv.columns else 0
                                        memo = r['memo'] if 'memo' in df_csv.columns else ""
                                        new_lines.append(f"{rng}|{lvl}|{cnt}||{memo}")
                                    
                                    # 格式化并更新
                                    raw_str = "\n".join(new_lines)
                                    st.session_state.raw_data = format_columns(raw_str, sort=True)
                                    msg_list.append(f"历史数据已加载: {target_file_path}\r\n")
                                else:
                                    msg_list.append("CSV 格式不兼容")
                            except Exception as e:
                                #print("正在尝试显示提示 ERR! 无法读取文件")
                                st.error(f"无法读取文件: {e}")
                                return
                        else:
                            print("正在尝试显示提示 ERR! 找不到文件")
                            st.toast(f"ERR! 找不到文件: {target_file_path}", duration="long")
                            return
                    
                    # 刷新主界面，关闭弹窗
                    st.session_state.pending_toast = {'msg': " | ".join(msg_list), 'duration': 'long'}
                    st.rerun()

with st.sidebar:
    st.title("⚙️ 硬盘工具箱")

    # [模块 1] LBA 计算器
    with st.container(border=True):
        st.markdown("**🧮 LBA 转换器**")
        c1, c2 = st.columns([2, 1])
        cal_lba = c1.text_input("输入 LBA", placeholder="12345678", label_visibility="collapsed").replace(" ", "")
        if c2.button("📲", width='stretch'):
            if cal_lba.isdigit():
                val = int(cal_lba) * 512
                gb = val / (1000**3)
                gib = val / (1024**3)
                st.info(f"💾 **{gb:.2f} GB**\n\n💻 **{gib:.2f} GiB**")
            else:
                st.error("请输入数字")
    
    # === [模块 0] 资产库存管理 (新功能) ===
    st.markdown("### 🏷️ 资产识别 & 库存")
    
    inv_data = load_inventory()    
    
    # 布局：左侧输入框，右侧库存列表
    col_sn_input, col_sn_btn = st.columns([3, 1], gap="small")

    # value 直接绑定 session_state，不需要 key 也能双向绑定，
    input_sn = col_sn_input.text_input("序列号 (S/N)", 
                                     value=st.session_state.hdd_sn, 
                                     placeholder="输入或右侧选择", 
                                     label_visibility="collapsed",
                                     key="sn_input_widget")
    
    # --- 右侧：库存列表管理器 ---
    with col_sn_btn:        
        if st.button("📂", help="打开库存列表 (宽屏模式)", width='stretch'):
            inventory_manager_dialog()

    # --- 状态同步 ---
    # 将输入框的值同步回 session_state (处理手动输入的情况)
    if input_sn != st.session_state.hdd_sn:
        st.session_state.hdd_sn = input_sn

    # --- 资产信息展示与操作区 ---
    # 获取当前 SN 在库存中的信息
    current_sn_info = inv_data.get(st.session_state.hdd_sn, None)
    
    if st.session_state.hdd_sn:
        # 场景 A: 已在库
        if current_sn_info:
            curr_model = current_sn_info.get('model', 'Unknown')
            curr_memo = current_sn_info.get('memo', '')
            st.caption(f"当前载入: {curr_memo}")

            # 历史文件记录
            history = current_sn_info.get('history', [])
            if history:
                with st.expander(f"📚 关联文件 ({len(history)})"):
                    for h_file in history:
                        st.caption(f"📄 {h_file}")

        # 场景 B: 未入库 (新设备)
        else:
            st.info("🆕 新设备 (未登记)")
            # 注册按钮
            if st.button("💾 注册到库存", width='stretch'):
                current_model = st.session_state.selected_preset
                if current_model == "New Profile" or not current_model:
                    st.error("请先选择有效的物理预设模型！")
                else:
                    register_hdd(st.session_state.hdd_sn, current_model)
                    st.toast(f"已注册: {st.session_state.hdd_sn}")
                    st.rerun()

    # [模块 2] 硬盘参数配置 
    st.markdown("### 🛠️ 物理规格")
    preset_keys = list(presets.keys()) + [NEW_PROFILE]

    # 状态同步：如果当前 session 中的预设不在列表里，重置为第一个
    if st.session_state.selected_preset not in preset_keys:
        st.session_state.selected_preset = preset_keys[0]  

    selected_model = st.selectbox("选择预设", preset_keys, key="selected_preset")

    # 根据选择加载数据
    if selected_model == NEW_PROFILE:
        # 默认空模板
        current_data = {'lba_max': 0, 'heads': 1, 'rpm': 7200, 'speed_out': 150.0, 'speed_in': 80.0}
    else:
        current_data = presets[selected_model]

    # 表单区域
    # 编辑模式允许修改 Key (Model Name)；另，使用 pop 读取临时导入值，实现一次性自动填充
    with st.expander("📝 详细参数编辑", expanded=False): # 默认折叠
        is_edit = st.toggle("解锁编辑", value=st.session_state.profile_edit_mode, key="edit_mode_toggle")
        st.session_state.profile_edit_mode = is_edit

        # 自动填入逻辑 (Pop临时值)
        # 优先弹出 import 进来的临时数据，如果没有则使用当前 current_data
        val_model = st.session_state.pop('tmp_imported_model', selected_model)        
        val_lba = st.session_state.pop('tmp_imported_lba', current_data['lba_max'])
        val_heads = st.session_state.pop('tmp_imported_heads', current_data['heads'])
        val_rpm = st.session_state.pop('tmp_imported_rpm', current_data['rpm'])
        val_sout = st.session_state.pop('tmp_imported_sout', current_data['speed_out'])
        val_sin = st.session_state.pop('tmp_imported_sin', current_data['speed_in'])

        # 输入框
        new_model = st.text_input("型号", value=val_model, disabled=not is_edit)
        c_lba = st.number_input("LBA Max", value=int(val_lba), disabled=not is_edit)
        c_heads = st.number_input("磁头数 (Heads)", value=int(val_heads), disabled=not is_edit)
        c_rpm = st.number_input("转速 (RPM)", value=int(val_rpm), disabled=not is_edit)
        c_s_out = st.number_input("外圈速度 (MB/s)", value=float(val_sout), disabled=not is_edit)
        c_s_in = st.number_input("内圈速度 (MB/s)", value=float(val_sin), disabled=not is_edit)

        if is_edit:
            if st.button("💾 保存预设", width='stretch'):
                if not new_model:
                    st.error("需输入型号名")
                else:
                    new_entry = {
                        'lba_max': int(c_lba), 'heads': int(c_heads), 'rpm': int(c_rpm),
                        'speed_out': float(c_s_out), 'speed_in': float(c_s_in)
                    }
                    if new_model != selected_model and selected_model != NEW_PROFILE:
                        if selected_model in presets: del presets[selected_model]
                    
                    presets[new_model] = new_entry
                    save_presets(presets)                    
                    # 保存后更新选中状态
                    st.session_state.selected_preset = new_model
                    st.toast(f"配置 {new_model} 已保存!")
                    st.rerun()

    # ZBR 参数计算 (供绘图用)
    A, B, Total_Cyls, spt_out, spt_in = calculate_zbr_params(c_lba, c_heads, c_rpm, c_s_out, c_s_in)
    r_in_ratio = spt_in / spt_out if spt_out > 0 else 0.5

# --- 6. UI: Log 解析助手 (Dialog) ---
@st.dialog("Victoria Log 助手")
def log_helper():
    st.markdown("##### 粘贴扫描日志")
    
    # 选项
    bs_keys = list(BLOCK_SIZES.keys())

    sel_bs = st.selectbox("Block Size", bs_keys, key="bs_selector")
    bs_int = BLOCK_SIZES[sel_bs] # "2048" -> 2048
    bs_threshold_key = 256 if sel_bs == "1/64/128/256" else bs_int
    
    log_txt = st.text_area("Log Content", height=200, placeholder="(Block start at) ... = 20 ms")
    
    if st.button("解析并追加"):
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
                grade = get_grade(ms, bs_threshold_key)
                added.append(f"{lba_s}-{lba_s + bs_int - 1}|{grade}|0||")
            elif m2:
                lba_s = int(m2.group(1))
                grade = 'ERR'
                added.append(f"{lba_s}-{lba_s + bs_int - 1}|{grade}|0||")
        
        if added:
            current = st.session_state.raw_data
            new_block = "\n".join(added)
            st.session_state.raw_data = (current + "\n" + new_block).strip()
            st.rerun()

# --- 7. 主界面布局 ---
col_main_ui, col_viz = st.columns([1, 1.8])

# ================= 左侧：控制与图例 =================
with col_main_ui:
    st.subheader("📝 数据录入")
    
    # 定义导入功能的 Dialog
    @st.dialog("📂 导入扫描数据")
    def import_helper():
        tab_csv, tab_vic = st.tabs(["📄 CSV 导入", "🩺 Victoria 日志"])
        
        # === TAB 1: CSV 导入 ===
        with tab_csv:
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
                        st.dataframe(df.head(3), hide_index=True, width='stretch')
                        
                        new_lines = []
                        for _, row in df.iterrows():
                            rng = str(row['range'])
                            lvl = str(row['level'])
                            # 读取 count 列，如果没有则默认为 0
                            cnt = row['count'] if 'count' in df.columns and pd.notna(row['count']) else 0
                            memo = str(row['memo']) if 'memo' in df.columns and pd.notna(row['memo']) else ""

                            # 组合 4 列，格式: range|level|count|memo
                            new_lines.append(f"{rng}|{lvl}|{int(cnt)}||{memo}")

                        new_data_str = "\n".join(new_lines)
                        new_data_str = format_columns(new_data_str)

                        col_overwrite, col_append = st.columns(2)

                        with col_overwrite:
                            if st.button("🗑️ 覆盖并应用参数", type="primary", width='stretch'):
                                # 更新数据
                                st.session_state.raw_data = new_data_str
                                
                                # 如果有元数据，强制更新当前设置
                                if match:
                                    imp_model = parsed_meta['model']

                                    # 要写入侧边栏输入框的临时数据
                                    st.session_state.hdd_sn = parsed_meta['sn']
                                    st.session_state.tmp_imported_model = imp_model
                                    st.session_state.tmp_imported_lba = parsed_meta['lba']
                                    st.session_state.tmp_imported_heads = parsed_meta['heads']
                                    st.session_state.tmp_imported_rpm = parsed_meta['rpm']
                                    st.session_state.tmp_imported_sout = parsed_meta['s_out']
                                    st.session_state.tmp_imported_sin = parsed_meta['s_in']
                                    
                                    target_preset = NEW_PROFILE
                                    # 存在同名预设
                                    if imp_model in presets:
                                        target_preset = imp_model    

                                        # 检查参数一致性
                                        p_data = presets[imp_model]
                                        is_identical = (
                                            p_data['lba_max'] == parsed_meta['lba'] and
                                            p_data['heads'] == parsed_meta['heads'] and
                                            p_data['rpm'] == parsed_meta['rpm'] and
                                            p_data['speed_out'] == parsed_meta['s_out'] and
                                            p_data['speed_in'] == parsed_meta['s_in']
                                        )

                                        if is_identical:
                                            # 2.1 内容一致 -> 锁定
                                            st.session_state.profile_edit_mode = False
                                            st.toast(f"参数与预设 '{imp_model}' 完美匹配。")
                                        else:
                                            # 2.2 内容不一致 -> 解锁并提示
                                            st.session_state.profile_edit_mode = True
                                            st.toast(f"预设 '{imp_model}' 存在但参数不一致，已开启编辑模式。", icon="⚠️")
                                    else:
                                        # 情况 3: 不存在 -> 指向 New Profile
                                        st.session_state.profile_edit_mode = True
                                        st.toast(f"新检测到型号 '{imp_model}'，已切换至 New Profile。", icon="🆕")
                                    
                                    st.session_state.selected_preset = target_preset
                                st.rerun()
                        
                        with col_append:
                            if st.button("➕ 仅追加数据", width='stretch'):
                                if st.session_state.raw_data.strip():
                                    st.session_state.raw_data = st.session_state.raw_data.strip() + "\n" + new_data_str
                                else:
                                    st.session_state.raw_data = new_data_str
                                st.rerun()
                except Exception as e:
                    st.error(f"读取失败: {e}")

        # === TAB 2: Victoria Log 导入 ===
        with tab_vic:
            st.caption("读取本地 Victoria `bads_*.txt` 日志文件。")
            
            # 1. 路径选择
            col_path, col_btn = st.columns([3, 1])
            current_path = get_log_path()
            
            with col_path:
                input_path = st.text_input("Victoria Log 文件夹路径", value=current_path, 
                                         placeholder="C:/Victoria/LOGS",
                                         label_visibility="collapsed")
            with col_btn:
                if st.button("💾 保存"):
                    if os.path.isdir(input_path):
                        save_log_path(input_path)
                        st.success("已保存")
                        st.rerun()
                    else:
                        st.error("路径不存在")
            
            # 打开开关才执行文件扫描
            enable_scan = st.toggle("📂 扫描日志目录", value=False, help="开启后将搜索目录下所有的 bads_*.txt 文件")

            # 2. 文件扫描与选择
            if not enable_scan:
                st.info("打开开关执行文件扫描。")
            elif input_path and os.path.isdir(input_path):
                # 查找 bads_*.txt
                search_pattern = os.path.join(input_path, "**", "bads_*.txt")
                files = glob.glob(search_pattern, recursive=True)
                # 按修改时间倒序排列
                files.sort(key=os.path.getmtime, reverse=True)
                
                if not files:
                    st.warning("该目录下未找到 `bads_*.txt` 文件")
                else:
                    # 显示"子文件夹/文件名"
                    file_options = {}
                    for f in files:
                        rel_path = os.path.relpath(f, input_path)
                        file_options[rel_path] = f
                    
                    selected_rel_path = st.selectbox("选择日志文件", list(file_options.keys()))

                    if selected_rel_path:
                        full_path = file_options[selected_rel_path]
                        filename_only = os.path.basename(full_path)
                        
                        # 显示提示信息
                        model, sn = parse_victoria_filename(filename_only)

                        # 预览解析结果
                        is_model_known = model in presets
                        
                        if is_model_known:
                            st.info(f"📄 **路径**: `{selected_rel_path}`\n\n🏷️ **识别**: Model=`{model}` (匹配预设 ✅), SN=`{sn}`")
                        else:
                            st.warning(f"📄 **路径**: `{selected_rel_path}`\n\n🏷️ **识别**: Model=`{model}` (未匹配预设 ⚠️), SN=`{sn}`\n\n*注意：追加功能仅对已知预设模型开放。*")
                        c_imp, c_app = st.columns(2)

                        # 读取文件内容逻辑
                        def read_and_parse():
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            rows = parse_victoria_content(content)
                            if not rows:
                                st.warning("文件中未找到匹配 `... Scan bad` 的记录。")
                                return None
                            return "\n".join(rows)

                        with c_imp:
                            if st.button("⚡ 覆盖导入", type="primary", width='stretch'):
                                new_data_str = read_and_parse()
                                if new_data_str:
                                    # 格式化
                                    formatted = format_columns(new_data_str, sort=True)
                                    st.session_state.raw_data = formatted
                                    
                                    # 更新元数据
                                    st.session_state.hdd_sn = sn
                                    st.session_state.tmp_imported_model = model
                                    
                                    # 匹配预设
                                    target_preset = "New Profile"
                                    if model in presets:
                                        target_preset = model
                                        st.session_state.edit_mode = False
                                        st.toast(f"匹配预设: {model}")
                                    else:
                                        st.session_state.edit_mode = True
                                        st.toast("新预设", icon="🆕")
                                    
                                    st.session_state.selected_preset = target_preset
                                    st.rerun()

                        with c_app:
                            btn_disabled = not is_model_known
                            help_msg = "仅当日志中的硬盘型号与当前系统预设匹配时，才允许追加数据。" if btn_disabled else "将此日志中的坏道追加到当前视图"
                            
                            if st.button("➕ 追加数据", 
                                         width='stretch', 
                                         disabled=btn_disabled, 
                                         help=help_msg,
                                         key="btn_vic_append"):
                                         
                                new_data_str = read_and_parse()
                                if new_data_str:
                                    combined = (st.session_state.raw_data + "\n" + new_data_str).strip()
                                    st.session_state.raw_data = format_columns(combined, sort=True)
                                    st.rerun()
            else:
                if input_path: # 有输入但无效
                    st.info("路径无效，请输入包含 LOGS 的文件夹路径。")

    # 按钮组
    c_btn1, c_btn2, c_btn3, c_btn4, c_btn5 = st.columns([1, 1, 1, 1, 1], gap="small")
    with c_btn1: 
        if st.button("🪄 Log", width='stretch'): log_helper()

    with c_btn2:
        if st.button("📂 导入", width='stretch'): import_helper()

    with c_btn3:
        if st.button("🔢 排序", width='stretch', help="按 LBA 起始位置排序"):
            st.session_state.raw_data = format_columns(st.session_state.raw_data, sort=True)
            st.rerun()

    with c_btn5:
        if st.button("🚀 更新", type="primary", width='stretch'):
            st.session_state.raw_data = format_columns(st.session_state.raw_data, sort=False)
            st.rerun()

    with c_btn4:
        # CSV 导出逻辑(4列: Range, Level, Count, Memo)
        export_list = []
        lines_raw = st.session_state.raw_data.strip().split('\n')
        for line in lines_raw:
            if not line.strip() or '|' not in line: continue
            parts = [p.strip() for p in line.split('|')]
            r_val = parts[0]
            l_val = parts[1] if len(parts) > 1 else ""
            c_val = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            m_val = parts[4] if len(parts) > 4 else ""

            export_list.append({'range': r_val, 'level': l_val, 'count': c_val, 'memo': m_val})
            
        if export_list:
            current_model_name = new_model if 'new_model_name' in locals() else selected_model
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
            df = df[['range', 'level', 'count', 'memo']]
            csv_body = df.to_csv(index=False)
            final_csv_content = header_str + csv_body
            
            if st.download_button("💾 导出", 
                               final_csv_content, 
                               filename, 
                               "text/csv", 
                               width='stretch'):
                register_hdd(st.session_state.hdd_sn, current_model_name, filename)
                if not os.path.exists(HIST_DIR): os.makedirs(HIST_DIR)
                save_path = os.path.join(HIST_DIR, filename)
                with open(save_path, "w", encoding='utf-8') as f:
                    f.write(final_csv_content)
                # 更新 register_hdd 传入带路径的文件名
                register_hdd(st.session_state.hdd_sn, current_model_name, save_path)

        else:
            st.button("💾 导出", disabled=True, width='stretch')

    # 等级过滤器：默认全选，获取 LEVELS 的所有 key
    all_levels = list(DELAY_LEVELS.keys())
    selected_levels = st.multiselect(
        "👁️ 视图过滤器 (显示特定等级)",
        options=all_levels,
        default=all_levels
    )

    # 文本框
    st.session_state.raw_data = st.text_area("输入 (LBA范围 | Level | Count(显示点数 0即默认描绘圆弧) | GB | Memo)", 
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
        raw_rng = parts[0].strip()
        rng = re.sub(r'\([\d\.]+[Gg][Bb]\)', '', raw_rng) # 剔除显示用的 GB 信息

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
                # 动态计算分辨率：根据弧度跨度决定点数，最小 2 点，每 1 度至少 1 个点
                arc_span = abs(p['t2'] - p['t1'])
                dynamic_res = max(2, int(arc_span * 60)) # *60 约等于每度一个点
                ts = np.linspace(p['t1'], p['t2'], dynamic_res)
                ax.plot(ts, [p['r']]*dynamic_res, color=p['c'], lw=1, alpha=0.9)
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
                                # 动态分辨率
                                arc_span = abs(p['t2'] - p['t1'])
                                dynamic_res = max(2, int(arc_span * 60))                                
                                ts = np.linspace(p['t1'], p['t2'], dynamic_res)
                                ax.plot(ts, [p['r']]*dynamic_res, color=p['c'], lw=0.6)
                        
                        st.pyplot(fig)# 独立的 pyplot 允许 hover 时单独放大