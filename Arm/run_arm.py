import time
import numpy as np
import yaml
import mujoco
import mujoco.viewer

# ---------- 加载 YAML 配置 ----------
def load_config(yaml_path="config.yaml"):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ 加载 YAML 成功")
        return config
    except FileNotFoundError:
        print("⚠️ 未找到 config.yaml，使用默认参数")
        # 默认参数只保留轨迹相关，kp/kd 已在 XML 中定义
        return {
            'amplitude1': 1.5, 
            'amplitude2': 1.0,
            'freq1': 0.8, 
            'freq2': 1.0, 
            'phase2': 0.5,
        }

# ---------- 加载模型 ----------
try:
    model = mujoco.MjModel.from_xml_path("two_link_arm.xml")
    print("✅ 加载 XML 模型成功")
except Exception as e:
    print(f"❌ 加载 XML 失败: {e}")
    exit(1)

data = mujoco.MjData(model)
config = load_config()

# ---------- 仿真主循环 ----------
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    while viewer.is_running():
        elapsed = time.time() - start_time

        # ----- 1. 规划目标轨迹（生成期望角度） -----
        target1 = config['amplitude1'] * np.sin(2 * np.pi * config['freq1'] * elapsed)
        target2 = config['amplitude2'] * np.sin(2 * np.pi * config['freq2'] * elapsed + config['phase2'])

        # ----- 2. 直接下发目标角度（核心改动） -----
        # 因为 XML 中使用了 <position> 执行器，它内部自带 kp 和 kv，
        # 会自动将角度误差转化为力矩，无需我们在 Python 里手动算 PD。
        data.ctrl[0] = target1
        data.ctrl[1] = target2

        # ----- 3. 步进物理引擎 -----
        mujoco.mj_step(model, data)

        # ----- 4. 同步可视化窗口 -----
        viewer.sync()

        # ----- 5. 控制仿真速度（与物理步长对齐） -----
        time.sleep(model.opt.timestep)

    print("🔚 仿真结束")