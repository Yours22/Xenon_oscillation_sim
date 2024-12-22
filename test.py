# %%
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体（根据需要调整路径）
# 确保系统中存在'simsun.ttc'字体文件
font_path = 'C:/Windows/Fonts/simsun.ttc'
font = FontProperties(fname=font_path)

# %% 参数设置

# 物理参数
gamma_I = 6.386e-2        # I-135的产生速率 (s^{-1})
gamma_Xe = 0.228e-2       # Xe-135的产生速率 (s^{-1})
lambda_I = 2.87e-5        # I-135的衰变常数 (s^{-1})
lambda_Xe = 2.09e-5       # Xe-135的衰变常数 (s^{-1})
Sigma_a_Xe = 2.7e-18     # Xe-135吸收截面 (cm²)
Sigma_f = 0.043           # 裂变截面 (cm²)
beta = 0.0065             # 缓发中子产额
Lambda = 0.0001           # 中子寿命 (s)
rho0 = 0.001              # 初始反应性（假设一个小的正反应性）
alpha = 1e-24             # Xe-135引起的反应性变化系数 (cm²)

# 其他参数
nu = 2.73                 # 中子产额
# 设定仿真总时间（秒）和时间步长
total_time = 1e6          # 总时间 (s)
delta_t = 1               # 时间步长 (s)

# 初始条件
Phi_initial = 3e13        # 初始中子通量 (neutrons/cm²/s)
I_initial = 0             # 初始I-135浓度
Xe_initial = 0            # 初始Xe-135浓度

# %% 定义ODE系统

def reactor_dynamics(t, y):
    """
    定义反应堆动力学的微分方程组。
    
    y[0] -> Phi (中子通量)
    y[1] -> I (碘-135浓度)
    y[2] -> Xe135 (氙-135浓度)
    """
    Phi, I, Xe135 = y
    
    # 反应性随Xe-135浓度变化
    rho = rho0 - alpha * Xe135
    
    # 中子通量的变化率
    dPhi_dt = ((rho - beta) / Lambda) * Phi + nu * Sigma_f * Phi - Sigma_a_Xe * Xe135 * Phi
    
    # 碘-135的变化率
    dI_dt = gamma_I * Sigma_f * Phi - lambda_I * I
    
    # Xe-135的变化率
    dXe135_dt = gamma_Xe * Sigma_f * Phi + lambda_I * I - (lambda_Xe + Sigma_a_Xe * Phi) * Xe135
    
    return [dPhi_dt, dI_dt, dXe135_dt]

# %% 设置初始条件和时间范围

initial_conditions = [Phi_initial, I_initial, Xe_initial]
t_span = (0, total_time)
t_eval = np.arange(0, total_time + delta_t, delta_t)

# %% 使用solve_ivp求解ODE

solution = solve_ivp(
    reactor_dynamics,
    t_span,
    initial_conditions,
    method='BDF',             # 使用适合刚性问题的BDF方法
    t_eval=t_eval,
    vectorized=False
)

# 检查求解是否成功
if not solution.success:
    print("ODE求解失败:", solution.message)
else:
    Phi = solution.y[0]
    I = solution.y[1]
    Xe135 = solution.y[2]
    time = solution.t

    # 计算理想浓度（稳态）
    N_I_ideal = gamma_I * Sigma_f * Phi_initial / lambda_I
    N_Xe_ideal = (gamma_I + gamma_Xe) * Sigma_f * Phi_initial / (lambda_Xe + Sigma_a_Xe * Phi_initial)
    
    # %% 绘制结果

    plt.figure(figsize=(12, 8))

    # 绘制中子通量
    plt.subplot(3, 1, 1)
    plt.plot(time / (24 * 3600), Phi, label='中子通量 Φ(t)', color='blue')
    plt.axhline(y=N_I_ideal, color='r', linestyle='--', label='理想I浓度')
    plt.xlabel('时间 (天)', fontproperties=font)
    plt.ylabel('中子通量 Φ(t) (neutrons/cm²/s)', fontproperties=font)
    plt.title('中子通量随时间的变化', fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True)

    # 绘制碘-135浓度
    plt.subplot(3, 1, 2)
    plt.plot(time / (24 * 3600), I, label='碘-135浓度 I(t)', color='orange')
    plt.axhline(y=N_I_ideal, color='r', linestyle='--', label='理想I浓度')
    plt.xlabel('时间 (天)', fontproperties=font)
    plt.ylabel('碘-135浓度 I(t)', fontproperties=font)
    plt.title('碘-135浓度随时间的变化', fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True)

    # 绘制Xe-135浓度
    plt.subplot(3, 1, 3)
    plt.plot(time / (24 * 3600), Xe135, label='氙-135浓度 Xe135(t)', color='green')
    plt.axhline(y=N_Xe_ideal, color='b', linestyle='--', label='理想Xe135浓度')
    plt.xlabel('时间 (天)', fontproperties=font)
    plt.ylabel('氙-135浓度 Xe135(t)', fontproperties=font)
    plt.title('氙-135浓度随时间的变化', fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('Reactor_Dynamics_Oscillations.png')
    plt.show()

    print('模拟完成，结果已保存为"Reactor_Dynamics_Oscillations.png"。')

# %% 进一步分析氙震荡

# 检查是否存在振荡现象
# 通过绘制相空间图或者频谱分析等方法
# 这里简单绘制Xe135随时间的变化，观察是否有周期性波动

plt.figure(figsize=(10, 5))
plt.plot(time / (24 * 3600), Xe135, label='氙-135浓度 Xe135(t)', color='green')
plt.xlabel('时间 (天)', fontproperties=font)
plt.ylabel('氙-135浓度 Xe135(t)', fontproperties=font)
plt.title('氙-135浓度随时间的变化 - 振荡分析', fontproperties=font)
plt.legend(prop=font)
plt.grid(True)
plt.savefig('Xe135_Oscillations.png')
plt.show()
