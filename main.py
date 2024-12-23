import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 参数定义
# 空间域
L = 100  # 一维空间长度，单位cm
nx = 40  # 增加空间离散点数以提高空间分辨率
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

# 时间域
total_time = 1000  # 总时间，单位s
delta_t = 0.001      # 时间步长，单位s
nt = int(total_time / delta_t)

# 物理参数
D = 1.02            # 扩散系数，cm²/s
sigma_a_Xe = 2.7e-18  # Xe吸收截面，cm²
nu = 2.73             # 中子产额
Sigma_f = 0.043      # 裂变截面，cm²
gamma_I = 6.386e-2   # I-135的产额
gamma_Xe = 0.228e-2  # Xe-135的产额
lambda_I = 2.87e-5   # I-135的衰变常数，s⁻¹
lambda_Xe = 2.09e-5  # Xe-135的衰变常数，s⁻¹
beta = 0.0065        # 缓发中子产额
Lambda = 0.0003      # 中子寿命，s⁻¹
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

# 每隔100步存储一次
store_every = 100
stored_steps = nt // store_every + 1

phi_history = np.zeros((stored_steps, nx))
N_I_history = np.zeros((stored_steps, nx))
N_Xe_history = np.zeros((stored_steps, nx))
time_history = np.zeros(stored_steps)

current_store = 0

def compute_phi_derivative(phi, rho, beta, Lambda, N_Xe):
    # 引入虚拟点
    phi_extended = np.zeros(nx + 2, dtype=np.float64)
    phi_extended[1:-1] = phi

    # 应用第一类边界条件（Dirichlet 边界条件）
    # 假设边界值固定为0
    phi_extended[0] = 0.0          # 左边界
    phi_extended[-1] = 0.0         # 右边界

    # 计算二阶导数
    d2phi_dx2 = (phi_extended[2:] - 2 * phi_extended[1:-1] + phi_extended[:-2]) / dx**2

    # 计算 dphi/dt ，暂时不考虑缓发中子
    dphi_dt = D * d2phi_dx2 - sigma_a_Xe * N_Xe * phi + nu * Sigma_f * phi
    # dphi_dt = (D * d2phi_dx2 - sigma_a_Xe * N_Xe * phi + nu * Sigma_f * phi 
    #            - ((rho - beta) / Lambda) * phi)

    return dphi_dt

def iodine_xenon_dynamics(phi, N_I, N_Xe):
    dN_I_dt = gamma_I * Sigma_f * phi - lambda_I * N_I
    dN_Xe_dt = lambda_I * N_I + gamma_Xe * Sigma_f * phi - (lambda_Xe + sigma_a_Xe * phi) * N_Xe
    return dN_I_dt, dN_Xe_dt

# 收敛参数定义，暂时没有用到，也不知道怎么用
convergence_tol = 1e6  # 定义收敛的容忍度（根据问题规模调整）
convergence_steps = 10  # 连续满足收敛条件的步数
convergence_window = []  # 用于记录最近的变化量

# 最大迭代步数（可根据需要调整）
max_steps = 145000

# 时间循环
for t_step in range(max_steps):
    if t_step >= nt:
        logging.warning('Reached the predefined maximum number of iterations.')
        break

    current_time = t_step * delta_t

    # 计算动态
    dN_I_dt, dN_Xe_dt = iodine_xenon_dynamics(phi, N_I, N_Xe)
    N_I_new = N_I + delta_t * dN_I_dt
    N_Xe_new = N_Xe + delta_t * dN_Xe_dt
    dphi_dt = compute_phi_derivative(phi, rho, beta, Lambda, N_Xe)
    phi_new = phi + delta_t * dphi_dt

    # 数据验证
    if np.any(np.isnan(phi_new)) or np.any(np.isinf(phi_new)):
        logging.error(f'Invalid values detected in phi at time step {t_step}, time {current_time:.2f}s')
        break 

    if np.any(np.isnan(N_I_new)) or np.any(np.isinf(N_I_new)):
        logging.error(f'Invalid values detected in N_I at time step {t_step}, time {current_time:.2f}s')
        break

    if np.any(np.isnan(N_Xe_new)) or np.any(np.isinf(N_Xe_new)):
        logging.error(f'Invalid values detected in N_Xe at time step {t_step}, time {current_time:.2f}s')
        break

    # 更新变量
    phi = phi_new
    N_I = N_I_new
    N_Xe = N_Xe_new

    # 计算变化量（使用绝对差的最大值作为变化指标）
    delta_phi = np.max(np.abs(phi - phi_history[current_store - 1] if current_store > 0 else phi))
    delta_N_I = np.max(np.abs(N_I - (N_I_history[current_store - 1] if current_store > 0 else N_I)))
    delta_N_Xe = np.max(np.abs(N_Xe - (N_Xe_history[current_store - 1] if current_store > 0 else N_Xe)))
    max_delta = max(delta_phi, delta_N_I, delta_N_Xe)

    # 更新收敛窗口
    convergence_window.append(max_delta)
    if len(convergence_window) > convergence_steps:
        convergence_window.pop(0)

    # 检查是否满足收敛条件
    if len(convergence_window) == convergence_steps and all(delta < convergence_tol for delta in convergence_window):
        logging.info(f'Steady-state reached at time step {t_step}, time={current_time:.2f}s')
        # 存储当前状态
        if t_step % store_every == 0:
            if current_store >= stored_steps:
                logging.warning('Stored_steps exceeded the preallocated size. Increasing storage arrays.')
                phi_history = np.vstack([phi_history, phi])
                N_I_history = np.vstack([N_I_history, N_I])
                N_Xe_history = np.vstack([N_Xe_history, N_Xe])
                time_history = np.append(time_history, current_time)
                stored_steps += 1

            phi_history[current_store] = phi
            N_I_history[current_store] = N_I
            N_Xe_history[current_store] = N_Xe
            time_history[current_store] = current_time
            current_store += 1
        break  # 退出时间循环

    # 按需存储结果
    if t_step % store_every == 0:
        logging.info(f'Processing time step {t_step}/{nt}, time={current_time:.2f}s')

        if current_store >= stored_steps:
            logging.warning('Stored_steps exceeded the preallocated size. Increasing storage arrays.')
            # 动态扩展存储数组
            phi_history = np.vstack([phi_history, phi])
            N_I_history = np.vstack([N_I_history, N_I])
            N_Xe_history = np.vstack([N_Xe_history, N_Xe])
            time_history = np.append(time_history, current_time)
            stored_steps += 1

        phi_history[current_store] = phi
        N_I_history[current_store] = N_I
        N_Xe_history[current_store] = N_Xe
        time_history[current_store] = current_time
        current_store += 1

# 处理最后一步（如果未存储）
if current_store < stored_steps:
    phi_history[current_store] = phi
    N_I_history[current_store] = N_I
    N_Xe_history[current_store] = N_Xe
    time_history[current_store] = t_step * delta_t

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

# 选择绘图时使用的时间点
# 例如，绘制初始、中间和结束时刻
plot_times = [0, max_steps * delta_t / 4, max_steps * delta_t / 2, max_steps * delta_t*9/10, (max_steps-1000 )* delta_t,max_steps * delta_t ]  

plot_indices = []
for t in plot_times:
    idx = np.argmin(np.abs(time_history - t))
    if (idx < len(time_history)):
        plot_indices.append(idx)
    else:
        logging.warning(f'Plot time {t} exceeds simulation time. Using last available index.')
        plot_indices.append(len(time_history) - 1)

output_dir = 'results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建子图
fig, axs = plt.subplots(3, 2, figsize=(15, 18))

# 绘制中子通量φ(x, t)的分布
for idx, t in zip(plot_indices, plot_times):
    axs[0, 0].plot(x, phi_history[idx], label=f't={time_history[idx]:.1f}s')
axs[0, 0].set_xlabel('Position x (cm)')
axs[0, 0].set_ylabel('Neutron Flux φ(x, t) (neutrons/cm²/s)')
axs[0, 0].set_yscale('log')
axs[0, 0].set_title('Neutron Flux Distribution at Selected Times')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 绘制I-135浓度分布
for idx, t in zip(plot_indices, plot_times):
    axs[0, 1].plot(x, N_I_history[idx], label=f't={time_history[idx]:.1f}s')
axs[0, 1].set_xlabel('Position x (cm)')
axs[0, 1].set_ylabel('Iodine Concentration N_I(x, t)')
axs[0, 1].set_yscale('log')
axs[0, 1].set_title('Iodine Concentration Distribution at Selected Times')
axs[0, 1].legend()
axs[0, 1].grid(True)

# 绘制Xe-135浓度分布
for idx, t in zip(plot_indices, plot_times):
    axs[1, 0].plot(x, N_Xe_history[idx], label=f't={time_history[idx]:.1f}s')
axs[1, 0].set_xlabel('Position x (cm)')
axs[1, 0].set_ylabel('Xenon-135 Concentration N_Xe(x, t)')
axs[1, 0].set_yscale('log')
axs[1, 0].set_title('Xenon-135 Concentration Distribution at Selected Times')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 中子通量热图
extent = [x.min(), x.max(), time_history.min(), time_history.max()]
im1 = axs[1, 1].imshow(phi_history, aspect='auto', extent=extent, origin='lower', cmap='viridis')
fig.colorbar(im1, ax=axs[1, 1], label='Neutron Flux φ(x, t) (neutrons/cm²/s)')
axs[1, 1].set_xlabel('Position x (cm)')
axs[1, 1].set_ylabel('Time t (s)')
axs[1, 1].set_title('Neutron Flux Distribution Over Time')

# I-135浓度热图
im2 = axs[2, 0].imshow(N_I_history, aspect='auto', extent=extent, origin='lower', cmap='plasma')
fig.colorbar(im2, ax=axs[2, 0], label='Iodine Concentration N_I(x, t)')
axs[2, 0].set_xlabel('Position x (cm)')
axs[2, 0].set_ylabel('Time t (s)')
axs[2, 0].set_title('Iodine Concentration Distribution Over Time')

# Xe-135浓度热图
im3 = axs[2, 1].imshow(N_Xe_history, aspect='auto', extent=extent, origin='lower', cmap='inferno')
fig.colorbar(im3, ax=axs[2, 1], label='Xenon-135 Concentration N_Xe(x, t)')
axs[2, 1].set_xlabel('Position x (cm)')
axs[2, 1].set_ylabel('Time t (s)')
axs[2, 1].set_title('Xenon-135 Concentration Distribution Over Time')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Combined_Plots.png'))
plt.show()

# # 设置全局绘图样式
# plt.style.use('ggplot')

# # 创建中子通量动画
# def create_flux_animation(x, phi_history, time_history, output_path):
#     fig, ax = plt.subplots(figsize=(10, 6))

#     def animate(i):
#         ax.clear()
#         ax.plot(x, phi_history[i], color='blue')
#         ax.set_xlabel('Position x (cm)')
#         ax.set_ylabel('Neutron Flux φ(x, t) (neutrons/cm²/s)')
#         ax.set_yscale('log')
#         ax.set_title(f'Neutron Flux Distribution at t={time_history[i]:.1f} s')
#         ax.grid(True)

#     flux_anim = animation.FuncAnimation(fig, animate, frames=len(time_history), interval=10)
#     flux_anim.save(output_path, writer='imagemagick')
#     plt.close(fig)

# # 创建I-135浓度动画
# def create_I_concentration_animation(x, N_I_history, time_history, output_path):
#     fig, ax = plt.subplots(figsize=(10, 6))

#     def animate(i):
#         ax.clear()
#         ax.plot(x, N_I_history[i], color='green')
#         ax.set_xlabel('Position x (cm)')
#         ax.set_ylabel('Iodine Concentration N_I(x, t)')
#         ax.set_yscale('log')
#         ax.set_title(f'Iodine Concentration Distribution at t={time_history[i]:.1f} s')
#         ax.grid(True)

#     I_anim = animation.FuncAnimation(fig, animate, frames=len(time_history), interval=10)
#     I_anim.save(output_path, writer='imagemagick')
#     plt.close(fig)

# # 创建Xe-135浓度动画
# def create_Xe_concentration_animation(x, N_Xe_history, time_history, output_path):
#     fig, ax = plt.subplots(figsize=(10, 6))

#     def animate(i):
#         ax.clear()
#         ax.plot(x, N_Xe_history[i], color='red')
#         ax.set_xlabel('Position x (cm)')
#         ax.set_ylabel('Xenon-135 Concentration N_Xe(x, t)')
#         ax.set_yscale('log')
#         ax.set_title(f'Xenon-135 Concentration Distribution at t={time_history[i]:.1f} s')
#         ax.grid(True)

#     Xe_anim = animation.FuncAnimation(fig, animate, frames=len(time_history), interval=10)
#     Xe_anim.save(output_path, writer='imagemagick')
#     plt.close(fig)

# # 保存动画的路径
# flux_gif_path = os.path.join(output_dir, 'Neutron_Flux_Distribution.gif')
# I_gif_path = os.path.join(output_dir, 'Iodine_Concentration_Distribution.gif')
# Xe_gif_path = os.path.join(output_dir, 'Xenon135_Concentration_Distribution.gif')

# # 创建并保存动画
# create_flux_animation(x, phi_history, time_history, flux_gif_path)
# create_I_concentration_animation(x, N_I_history, time_history, I_gif_path)
# create_Xe_concentration_animation(x, N_Xe_history, time_history, Xe_gif_path)

# print(f"动画已保存到 {output_dir}")


# print('Simulation and plotting completed successfully!')
