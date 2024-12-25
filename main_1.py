import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 尝试设置中文字体，如果路径不存在则使用默认字体
import os
font_path = 'C:/Windows/Fonts/simsun.ttc'
if os.path.exists(font_path):
    font_prop = FontProperties(fname=font_path)
else:
    print("指定的字体路径不存在，使用默认字体。")
    font_prop = FontProperties()

# 物理参数
sigma_a_Xe = 2.7e-18   # Xe-135的微观吸收截面，cm²
nu = 2.73              # 中子产额
Sigma_f = 0.043        # 宏观裂变截面，cm⁻¹
gamma_I = 6.386e-2     # I-135的产生率
gamma_Xe = 0.228e-2    # Xe-135的产生率
lambda_I = 2.87e-5     # I-135的衰变常数，s⁻¹
lambda_Xe = 2.09e-5    # Xe-135的衰变常数，s⁻¹
beta = 0.0065          # 缓发中子分数
Lambda = 0.0003        # 中子寿命，s
alpha = sigma_a_Xe / (nu * Sigma_f)  # 修正后的反应性系数

# 中子交换速率
D_exchange = 1e-13       # 单位 s⁻¹，假设值，可根据实际情况调整

# 持续性扰动参数
# 初始扰动
# t_start1 = 500000         # 扰动开始时间，单位秒
# delta_t1 = 10000         # 扰动持续时间，单位秒
# Delta_rho1 = 0.02        # 扰动的反应性变化量（仅堆1）

# 第二阶段扰动
t_start2 = 500000         # 第二个扰动开始时间，单位秒（可以根据需要调整）
delta_t2 = 10000         # 第二个扰动持续时间，单位秒
Delta_rho2 = 0.02        # 堆1的反应性增加量
Delta_rho3 = -0.02       # 堆2的反应性减少量（确保总反应性变化为0）

# 初始条件
P0_1 = 0              # 堆1初始功率（归一化）
C_I0_1 = 0.0            # 堆1初始碘-135浓度
C_Xe0_1 = 1e-6          # 堆1初始氙-135浓度（微小扰动）

P0_2 = 1.0              # 堆2初始功率（归一化）
C_I0_2 = 0.0            # 堆2初始碘-135浓度
C_Xe0_2 = 1e-6          # 堆2初始氙-135浓度（微小扰动）

# 时间范围
t_start_sim = 0
t_end_sim = 1e6        # 总模拟时间，单位秒（约27.8小时）
num_points = 10000     # 时间点数量

# 定义矩形脉冲函数，支持多个扰动阶段
def reactivity_perturbation(t):
    Delta_rho_1 = 0.0
    Delta_rho_2 = 0.0
    
    # 第一阶段扰动：仅堆1增加反应性
    # if t_start1 <= t < t_start1 + delta_t1:
    #     Delta_rho_1 += Delta_rho1
    #     # 堆2不受影响，Delta_rho_2 remains 0.0
    
    # 第二阶段扰动：堆1增加反应性，堆2减少反应性
    if t_start2 <= t < t_start2 + delta_t2:
        Delta_rho_1 += Delta_rho2
        Delta_rho_2 += Delta_rho3
    
    return Delta_rho_1, Delta_rho_2

# 定义动力学方程
def reactor_dynamics(t, y):
    P1, C_I1, C_Xe1, P2, C_I2, C_Xe2 = y
    
    # 获取当前反应性扰动
    perturb1, perturb2 = reactivity_perturbation(t)
    
    # 计算当前反应性，包括持续性扰动
    rho1 = 0.05 - alpha * C_Xe1 + perturb1    # 堆1的反应性
    rho2 = 0.05 - alpha * C_Xe2 + perturb2    # 堆2的反应性
    
    # 动力学方程
    dP1_dt = ((rho1 - beta) / Lambda) * P1 + (beta / Lambda) * C_I1 + D_exchange * (P2 - P1)
    dC_I1_dt = gamma_I * P1 - lambda_I * C_I1
    dC_Xe1_dt = gamma_Xe * P1 + lambda_I * C_I1 - lambda_Xe * C_Xe1
    
    dP2_dt = ((rho2 - beta) / Lambda) * P2 + (beta / Lambda) * C_I2 + D_exchange * (P1 - P2)
    dC_I2_dt = gamma_I * P2 - lambda_I * C_I2
    dC_Xe2_dt = gamma_Xe * P2 + lambda_I * C_I2 - lambda_Xe * C_Xe2
    
    return [dP1_dt, dC_I1_dt, dC_Xe1_dt, dP2_dt, dC_I2_dt, dC_Xe2_dt]

# 初始条件向量
y0 = [P0_1, C_I0_1, C_Xe0_1, P0_2, C_I0_2, C_Xe0_2]

# 时间点
t_eval = np.linspace(t_start_sim, t_end_sim, num_points)

# 求解ODE
solution = solve_ivp(reactor_dynamics, [t_start_sim, t_end_sim], y0, method='BDF', t_eval=t_eval, vectorized=False)

# 检查求解是否成功
if not solution.success:
    print("ODE求解失败：", solution.message)
    exit(1)

# 提取结果
P1 = solution.y[0]
C_I1 = solution.y[1]
C_Xe1 = solution.y[2]
P2 = solution.y[3]
C_I2 = solution.y[4]
C_Xe2 = solution.y[5]
t = solution.t

# 绘制结果
plt.figure(figsize=(16, 12))

# 绘制堆1和堆2的功率
plt.subplot(3, 1, 1)
plt.plot(t / 3600, P1, label='堆1功率 $P_1(t)$')
plt.plot(t / 3600, P2, label='堆2功率 $P_2(t)$')
# 标记第一个扰动期间
# plt.axvspan(t_start1 / 3600, (t_start1 + delta_t1) / 3600, color='red', alpha=0.3, label='初始扰动期间')
# 标记第二个扰动期间
plt.axvspan(t_start2 / 3600, (t_start2 + delta_t2) / 3600, color='orange', alpha=0.3, label='功率调整期间')
plt.xlabel('时间 (小时)', fontproperties=font_prop)
plt.ylabel('功率 (归一化)', fontproperties=font_prop)
plt.yscale('log')
plt.title('两个堆的核反应堆功率随时间变化', fontproperties=font_prop)
plt.grid(True)
plt.legend()

# 绘制堆1和堆2的碘-135浓度
plt.subplot(3, 1, 2)
plt.plot(t / 3600, C_I1, label='堆1碘-135浓度 $C_{I1}(t)$', color='green')
plt.plot(t / 3600, C_I2, label='堆2碘-135浓度 $C_{I2}(t)$', color='blue')
# 标记第一个扰动期间
# plt.axvspan(t_start1 / 3600, (t_start1 + delta_t1) / 3600, color='red', alpha=0.3)
# 标记第二个扰动期间
plt.axvspan(t_start2 / 3600, (t_start2 + delta_t2) / 3600, color='orange', alpha=0.3)
plt.xlabel('时间 (小时)', fontproperties=font_prop)
plt.ylabel('碘-135浓度 (归一化)', fontproperties=font_prop)
plt.title('两个堆的碘-135浓度随时间变化', fontproperties=font_prop)
plt.yscale('log')
plt.grid(True)
plt.legend()

# 绘制堆1和堆2的氙-135浓度
plt.subplot(3, 1, 3)
plt.plot(t / 3600, C_Xe1, label='堆1氙-135浓度 $C_{Xe1}(t)$', color='orange')
plt.plot(t / 3600, C_Xe2, label='堆2氙-135浓度 $C_{Xe2}(t)$', color='purple')
# 标记第一个扰动期间
# plt.axvspan(t_start1 / 3600, (t_start1 + delta_t1) / 3600, color='red', alpha=0.3)
# 标记第二个扰动期间
plt.axvspan(t_start2 / 3600, (t_start2 + delta_t2) / 3600, color='orange', alpha=0.3)
plt.xlabel('时间 (小时)', fontproperties=font_prop)
plt.ylabel('氙-135浓度 (归一化)', fontproperties=font_prop)
plt.title('两个堆的氙-135浓度随时间变化', fontproperties=font_prop)
plt.grid(True)
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()

# 绘制相位空间图
plt.figure(figsize=(8,6))
plt.plot(P1, P2, label='相位空间轨迹')
plt.xlabel('堆1功率 $P_1(t)$', fontproperties=font_prop)
plt.ylabel('堆2功率 $P_2(t)$', fontproperties=font_prop)
plt.title('堆1与堆2功率的相位空间图', fontproperties=font_prop)
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()

# ...existing code...

# 提取500000秒到510000秒之间的时间段
start_time = t_start2
end_time = t_start2 + 100* delta_t2

start_index = np.searchsorted(t, start_time)
end_index = np.searchsorted(t, end_time)

t_zoom = t[start_index:end_index]
P1_zoom = P1[start_index:end_index]
P2_zoom = P2[start_index:end_index]

# 绘制500000秒到510000秒之间的中子通量
plt.figure(figsize=(10, 6))
plt.plot(t_zoom / 3600, P1_zoom, label='堆1中子通量 $P_1(t)$')
plt.plot(t_zoom / 3600, P2_zoom, label='堆2中子通量 $P_2(t)$')
plt.xlabel('时间 (小时)', fontproperties=font_prop)
plt.ylabel('中子通量 (归一化)', fontproperties=font_prop)
plt.title('500000秒到510000秒之间的中子通量', fontproperties=font_prop)
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()

# 计算中子通量差值
flux_difference = P1_zoom - P2_zoom

# 绘制500000秒到510000秒之间的中子通量差值
plt.figure(figsize=(10, 6))
plt.plot(t_zoom / 3600, flux_difference, label='中子通量差值 $P_1(t) - P_2(t)$')
plt.xlabel('时间 (小时)', fontproperties=font_prop)
plt.ylabel('中子通量差值 (归一化)', fontproperties=font_prop)
plt.title('500000秒到510000秒之间的中子通量差值', fontproperties=font_prop)
plt.grid(True)
plt.legend()
plt.show()