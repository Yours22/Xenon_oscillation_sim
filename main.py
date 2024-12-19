# %%
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='C:/Windows/Fonts/simsun.ttc')

# # 参数列表
# data = {
#     '参数': [r'$\gamma_{I}$',r'$\gamma_{Xe}$', r'$\lambda_{I}$',r'$\lambda_{Xe}$',r'$\Sigma_{a,Xe}$',r'$\Sigma_f$',r'$\Phi$'],
#     '单位': ['/', '/', r'$s^{-1}$',r'$s^{-1}$','barn',r'$cm^{-1}$',r'$cm^{-2}s^{-1}$'],
#     '值': ['6.386e-2','0.228e-2','2.87e-5', '2.09e-5','2.7e6','0.043','3e13']
# }

# df = pd.DataFrame(data)
# # 绘制表格
# fig, ax = plt.subplots()
# ax.axis('tight')
# ax.axis('off')
# table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', colColours=['#f5f5f5']*3)
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# table.scale(1.2, 1.2)

# # 设置字体
# for key, cell in table.get_celld().items():
#     cell._text.set_fontproperties(font)
# plt.show()


# %%

# 参数设置
gamma_I = 6.386e-2
gamma_Xe = 0.228e-2
lambda_I = 2.87e-5
lambda_Xe = 2.09e-5
Sigma_a_Xe = 2.7e-18
Sigma_f = 0.043

def iodine_xenon_dynamics (t, y,Phi):
    """
    y[0] -> I (碘浓度)
    y[1] -> [Xe-135] (氙-135浓度)
    """
    I, Xe135 = y

    dIdt = gamma_I * Sigma_f * Phi - lambda_I*I

    dXe135dt = gamma_Xe * Sigma_f * Phi + lambda_I* I - (lambda_Xe+Sigma_a_Xe*Phi)*Xe135

    return [dIdt, dXe135dt]

def euler_method(iodine_xenon_dynamics, t, initial_conditions,Phi):
    steps = int(t/delta_t)
    I_values = np.zeros(steps+1)
    Xe_values = np.zeros(steps+1)
    I_values[0] = initial_conditions[0]
    Xe_values[0] = initial_conditions[1]
    for i in range(1, int(steps)+1):
        I_values[i] = I_values[i-1] + delta_t * iodine_xenon_dynamics(i*delta_t, [I_values[i-1], Xe_values[i-1]],Phi)[0]
        Xe_values[i] = Xe_values[i-1] + delta_t * iodine_xenon_dynamics(i*delta_t, [I_values[i-1], Xe_values[i-1]],Phi)[1]
    return I_values, Xe_values

# %%

## 模拟启动过程，初始条件为0
initial_conditions = [0, 0] # 初始条件
delta_t = 1 # 时间步长
t = 1080000 # 总时间
Phi_start=3e13 # 初始通量

I_values,Xe_values=euler_method(iodine_xenon_dynamics, t, initial_conditions,Phi_start)

N_I_ideal = gamma_I * Sigma_f * Phi_start / lambda_I
N_Xe_ideal = (gamma_I+gamma_Xe)* Sigma_f * Phi_start / (lambda_Xe+Sigma_a_Xe*Phi_start)

plt.figure()
time = np.arange(0, t + delta_t, delta_t) / (24 * 3600)
plt.plot(time, I_values, label='Iodine Concentration')
plt.plot(time, Xe_values, label='Xenon-135 Concentration')
plt.axhline(y=N_I_ideal, color='r', linestyle='--', label='Ideal Iodine Concentration')
plt.axhline(y=N_Xe_ideal, color='b', linestyle='--', label='Ideal Xenon-135 Concentration')
# plt.yscale('log')
plt.xlabel('Time (days)')
plt.ylabel('Concentration')
plt.legend()
plt.savefig('Iodine_Xenon_start.png')
plt.close()

## 模拟停堆过程 
Phi_stop=0
stop_condition = [I_values[-1], Xe_values[-1]]

I_values_final,Xe_values_final=euler_method(iodine_xenon_dynamics, t, stop_condition,Phi_stop)

plt.figure()
time = np.arange(0, t + delta_t, delta_t) / (24 * 3600)
plt.plot(time, I_values_final, label='Iodine Concentration')
plt.plot(time, Xe_values_final, label='Xenon-135 Concentration')
plt.xlabel('Time (days)')
plt.ylabel('Concentration')
plt.legend()
plt.savefig('Iodine_Xenon_stop.png')
plt.close()