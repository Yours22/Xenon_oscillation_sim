import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 参数定义
# 空间域
L = 10.0  # 一维空间长度，单位cm
nx = 100  # 增加空间离散点数以提高空间分辨率
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

# 时间域
total_time = 1000  # 总时间，单位s
delta_t = 0.001      # 时间步长，单位s
nt = int(total_time / delta_t)

# 物理参数
D = 1.0            # 扩散系数，cm²/s
sigma_a_Xe = 2.7e-18  # Xe吸收截面，cm²
nu = 2.73             # 中子产额
Sigma_f = 0.043      # 裂变截面，cm²
gamma_I = 6.386e-2   # I-135的产生速率
gamma_Xe = 0.228e-2  # Xe-135的产生速率
lambda_I = 2.87e-5   # I-135的衰变常数，s⁻¹
lambda_Xe = 2.09e-5  # Xe-135的衰变常数，s⁻¹
beta = 0.0065        # 缓发中子产额
Lambda = 0.0001      # 中子寿命，s⁻¹
rho_0 = 0            # 初始反应性
Delta_rho_Xe = 0.000  # Xe引起的反应性变化
v = 220000               # 中子速度，cm/s

# 推导参数
rho = rho_0 + Delta_rho_Xe  # 反应性

# 初始条件
phi_initial = np.ones(nx) * 3e13    # 初始中子通量
N_I_initial = np.zeros(nx)          # 初始I-135浓度
N_Xe_initial = np.zeros(nx)         # 初始Xe-135浓度

# 初始化
phi = phi_initial.copy()
N_I = N_I_initial.copy()
N_Xe = N_Xe_initial.copy()

# 优化数据存储：预分配数组并按需存储
# 例如，每隔100步存储一次
store_every = 100
stored_steps = nt // store_every + 1

phi_history = np.zeros((stored_steps, nx))
N_I_history = np.zeros((stored_steps, nx))
N_Xe_history = np.zeros((stored_steps, nx))
time_history = np.zeros(stored_steps)

current_store = 0

def compute_phi_derivative(phi, rho, beta, Lambda, N_Xe):
    d2phi_dx2 = np.zeros_like(phi)
    d2phi_dx2[1:-1] = (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / dx**2

    # 边界条件
    d2phi_dx2[0] = 0
    d2phi_dx2[-1] = 0

    # dphi_dt = D * d2phi_dx2 - sigma_a_Xe * N_Xe * phi + nu * Sigma_f * phi
    dphi_dt = D * d2phi_dx2 - sigma_a_Xe * N_Xe * phi + nu * Sigma_f * phi - ((rho - beta) / Lambda) * phi

    return dphi_dt

def iodine_xenon_dynamics(phi, N_I, N_Xe):
    dN_I_dt = gamma_I * Sigma_f * phi - lambda_I * N_I
    dN_Xe_dt = lambda_I * N_I + gamma_Xe * Sigma_f * phi - (lambda_Xe + sigma_a_Xe * phi) * N_Xe
    return dN_I_dt, dN_Xe_dt

# 时间循环
for t_step in range(nt):
    current_time = t_step * delta_t
    dphi_dt = compute_phi_derivative(phi, rho, beta, Lambda, N_Xe)
    dN_I_dt, dN_Xe_dt = iodine_xenon_dynamics(phi, N_I, N_Xe)

    # 更新变量
    phi += delta_t * dphi_dt
    N_I += delta_t * dN_I_dt
    N_Xe += delta_t * dN_Xe_dt

    # 边界条件
    phi[0] = 0
    phi[-1] = 0

    # 数据验证
    if np.any(np.isnan(phi)) or np.any(np.isinf(phi)):
        logging.error(f'Invalid values detected in phi at time step {t_step}, time {current_time:.2f}s')
        print(phi)
        break  # 或者采取其他错误处理措施
    
    if np.any(np.isnan(N_I)) or np.any(np.isinf(N_I)):
        logging.error(f'Invalid values detected in N_I at time step {t_step}, time {current_time:.2f}s')
        break
    
    if np.any(np.isnan(N_Xe)) or np.any(np.isinf(N_Xe)):
        logging.error(f'Invalid values detected in N_Xe at time step {t_step}, time {current_time:.2f}s')
        break
    

    # 按需存储结果
    if t_step % store_every == 0:
        if current_store >= stored_steps:
            logging.warning('Stored_steps exceeded the preallocated size. Increasing storage arrays.')
            # 动态扩展存储数组
            phi_history = np.vstack([phi_history, phi])
            N_I_history = np.vstack([N_I_history, N_I])
            N_Xe_history = np.vstack([N_Xe_history, N_Xe])
            time_history = np.append(time_history, t_step * delta_t)
            stored_steps += 1

        phi_history[current_store] = phi
        N_I_history[current_store] = N_I
        N_Xe_history[current_store] = N_Xe
        time_history[current_store] = t_step * delta_t
        current_store += 1

# 处理最后一步（如果未存储）
if current_store < stored_steps:
    phi_history[current_store] = phi
    N_I_history[current_store] = N_I
    N_Xe_history[current_store] = N_Xe
    time_history[current_store] = nt * delta_t

# 修剪存储数组以去除未使用的预分配部分
phi_history = phi_history[:current_store]
N_I_history = N_I_history[:current_store]
N_Xe_history = N_Xe_history[:current_store]
time_history = time_history[:current_store]

# 在绘图前进行数据完整性检查
def validate_data(history, name):
    if not isinstance(history, np.ndarray):
        logging.error(f'{name} history is not a numpy array.')
        return False
    if np.isnan(history).any():
        logging.error(f'NaN values found in {name} history.')
        return False
    if np.isinf(history).any():
        logging.error(f'Infinite values found in {name} history.')
        return False
    if history.shape[1] != nx:
        logging.error(f'{name} history has incorrect spatial dimension: {history.shape[1]} vs {nx}')
        return False
    return True

if not validate_data(phi_history, 'phi'):
    logging.error('Phi history validation failed. Exiting.')
    exit(1)

if not validate_data(N_I_history, 'N_I'):
    logging.error('Iodine concentration history validation failed. Exiting.')
    exit(1)

if not validate_data(N_Xe_history, 'N_Xe'):
    logging.error('Xenon concentration history validation failed. Exiting.')
    exit(1)

# 验证时间历史
if np.any(np.isnan(time_history)) or np.any(np.isinf(time_history)):
    logging.error('Invalid values detected in time history.')
    exit(1)

# %% 绘图优化

# 选择绘图时使用的时间点
# 例如，绘制初始、中间和结束时刻
plot_times = [0, total_time / 2, total_time]
plot_indices = []
for t in plot_times:
    idx = np.argmin(np.abs(time_history - t))
    if idx < len(time_history):
        plot_indices.append(idx)
    else:
        logging.warning(f'Plot time {t} exceeds simulation time. Using last available index.')
        plot_indices.append(len(time_history) - 1)

output_dir = 'results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 绘制中子通量φ(x, t)的分布
plt.figure(figsize=(10, 6))
for idx, t in zip(plot_indices, plot_times):
    plt.plot(x, phi_history[idx], label=f't={time_history[idx]:.1f}s')
plt.xlabel('Position x (cm)')
plt.ylabel('Neutron Flux φ(x, t) (neutrons/cm²/s)')
plt.title('Neutron Flux Distribution at Selected Times')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'Neutron_Flux_Distribution_Selected_Times.png'))
plt.show()

# 绘制I-135浓度分布
plt.figure(figsize=(10, 6))
for idx, t in zip(plot_indices, plot_times):
    plt.plot(x, N_I_history[idx], label=f't={time_history[idx]:.1f}s')
plt.xlabel('Position x (cm)')
plt.ylabel('Iodine Concentration N_I(x, t)')
plt.title('Iodine Concentration Distribution at Selected Times')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'Iodine_Concentration_Distribution_Selected_Times.png'))
plt.show()

# 绘制Xe-135浓度分布
plt.figure(figsize=(10, 6))
for idx, t in zip(plot_indices, plot_times):
    plt.plot(x, N_Xe_history[idx], label=f't={time_history[idx]:.1f}s')
plt.xlabel('Position x (cm)')
plt.ylabel('Xenon-135 Concentration N_Xe(x, t)')
plt.title('Xenon-135 Concentration Distribution at Selected Times')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'Xenon135_Concentration_Distribution_Selected_Times.png'))
plt.show()

# 可选：使用热图显示随时间和空间变化的分布
# 中子通量热图
plt.figure(figsize=(10, 6))
extent = [x.min(), x.max(), time_history.min(), time_history.max()]
plt.imshow(phi_history, aspect='auto', extent=extent, origin='lower', cmap='viridis')
plt.colorbar(label='Neutron Flux φ(x, t) (neutrons/cm²/s)')
plt.xlabel('Position x (cm)')
plt.ylabel('Time t (s)')
plt.title('Neutron Flux Distribution Over Time')
plt.savefig(os.path.join(output_dir, 'Neutron_Flux_Heatmap.png'))
plt.show()

# I-135浓度热图
plt.figure(figsize=(10, 6))
plt.imshow(N_I_history, aspect='auto', extent=extent, origin='lower', cmap='plasma')
plt.colorbar(label='Iodine Concentration N_I(x, t)')
plt.xlabel('Position x (cm)')
plt.ylabel('Time t (s)')
plt.title('Iodine Concentration Distribution Over Time')
plt.savefig(os.path.join(output_dir, 'Iodine_Concentration_Heatmap.png'))
plt.show()

# Xe-135浓度热图
plt.figure(figsize=(10, 6))
plt.imshow(N_Xe_history, aspect='auto', extent=extent, origin='lower', cmap='inferno')
plt.colorbar(label='Xenon-135 Concentration N_Xe(x, t)')
plt.xlabel('Position x (cm)')
plt.ylabel('Time t (s)')
plt.title('Xenon-135 Concentration Distribution Over Time')
plt.savefig(os.path.join(output_dir, 'Xenon135_Concentration_Heatmap.png'))
plt.show()

# 可选：生成动画以动态展示变化
# 注意：生成动画可能需要更长的时间和更多的存储空间

# def animate(i):
#     plt.clf()
#     plt.plot(x, phi_history[i], label=f'φ at t={time_history[i]:.1f}s')
#     plt.plot(x, N_I_history[i], label=f'I-135 at t={time_history[i]:.1f}s')
#     plt.plot(x, N_Xe_history[i], label=f'Xe-135 at t={time_history[i]:.1f}s')
#     plt.xlabel('Position x (cm)')
#     plt.ylabel('Values')
#     plt.title(f'Distribution at t={time_history[i]:.1f}s')
#     plt.legend()
#     plt.grid(True)

# fig = plt.figure(figsize=(10, 6))
# ani = FuncAnimation(fig, animate, frames=stored_steps, interval=100)
# ani.save(os.path.join(output_dir, 'Distribution_Animation.gif'), writer='imagemagick')
# plt.show()

print('Simulation and plotting completed successfully!')
