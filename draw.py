import os
import numpy as np
import matplotlib.pyplot as plt
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载数据
data_file = os.path.join('results', 'simulation_data_with_keff_rho.npz')
if not os.path.exists(data_file):
    logging.error(f'Data file {data_file} does not exist. Please run simulation.py first.')
    exit(1)

data = np.load(data_file)
x = data['x']
time_history = data['time_history']
phi_history = data['phi_history']
N_I_history = data['N_I_history']
N_Xe_history = data['N_Xe_history']

keff_history = data['keff_history']
rho_history = data['rho_history']

logging.info('Data loaded successfully.')

# 数据完整性检查
def validate_data(x, time_history, phi_history, N_I_history, N_Xe_history):
    if not isinstance(x, np.ndarray):
        logging.error('x is not a numpy array.')
        return False
    if not isinstance(time_history, np.ndarray):
        logging.error('time_history is not a numpy array.')
        return False
    if not isinstance(phi_history, np.ndarray):
        logging.error('phi_history is not a numpy array.')
        return False
    if not isinstance(N_I_history, np.ndarray):
        logging.error('N_I_history is not a numpy array.')
        return False
    if not isinstance(N_Xe_history, np.ndarray):
        logging.error('N_Xe_history is not a numpy array.')
        return False
    return True

if not validate_data(x, time_history, phi_history, N_I_history, N_Xe_history):
    logging.error('Data validation failed. Exiting.')
    exit(1)

# 选择绘图时使用的时间点
plot_times = [
    0,
    time_history.max() / 4,
    time_history.max() / 2,
    time_history.max() * 9 / 10,
    time_history.max() - 1000 * 0.001,  # Assuming delta_t = 0.001
    time_history.max()
]

plot_indices = []
for t in plot_times:
    idx = np.argmin(np.abs(time_history - t))
    if idx < len(time_history):
        plot_indices.append(idx)
    else:
        logging.warning(f'Plot time {t} exceeds simulation time. Using last available index.')
        plot_indices.append(len(time_history) - 1)

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
output_dir = 'results'
plot_file = os.path.join(output_dir, 'Combined_Plots.png')
plt.savefig(plot_file)
logging.info(f'Plots saved to {plot_file}')
plt.show()

# 可视化 rho 和 k_eff
def plot_keff_rho(time_history, keff_history, rho_history):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('$k_{eff}$', color=color)
    ax1.plot(time_history, keff_history, color=color, label='$k_{eff}$')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # 共享 x 轴
    color = 'tab:red'
    ax2.set_ylabel('$\\rho$', color=color)
    ax2.plot(time_history, rho_history, color=color, label='$\\rho$')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('$k_{eff}$ and $\\rho$ over Time')
    plt.tight_layout()
    plt.show()

plot_keff_rho(time_history, keff_history, rho_history)

# 可视化（可选）
def animate_simulation(x, time_history, phi_history, N_I_history, N_Xe_history):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    line1, = axs[0].plot([], [], 'b-')
    axs[0].set_xlim(0, L)
    axs[0].set_ylim(max(0, np.min(phi_history)), np.max(phi_history)*1.1)
    axs[0].set_title('Neutron Flux (phi)')
    axs[0].set_xlabel('Position (cm)')
    axs[0].set_ylabel('Flux')

    line2, = axs[1].plot([], [], 'r-')
    axs[1].set_xlim(0, L)
    axs[1].set_ylim(max(0, np.min(N_I_history)), np.max(N_I_history)*1.1)
    axs[1].set_title('I-135 Concentration (N_I)')
    axs[1].set_xlabel('Position (cm)')
    axs[1].set_ylabel('Concentration')

    line3, = axs[2].plot([], [], 'g-')
    axs[2].set_xlim(0, L)
    axs[2].set_ylim(max(0, np.min(N_Xe_history)), np.max(N_Xe_history)*1.1)
    axs[2].set_title('Xe-135 Concentration (N_Xe)')
    axs[2].set_xlabel('Position (cm)')
    axs[2].set_ylabel('Concentration')

    plt.tight_layout()

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line1, line2, line3

    def animate(i):
        line1.set_data(x, phi_history[i])
        line2.set_data(x, N_I_history[i])
        line3.set_data(x, N_Xe_history[i])
        return line1, line2, line3

    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(time_history), interval=50, blit=True
    )

    plt.show()

# 取消注释以下行以查看动画
# animate_simulation(x, time_history, phi_history, N_I_history, N_Xe_history)


# 可选：创建动画（取消注释以下代码以生成动画）

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

# # 创建并保存动画
# flux_gif_path = os.path.join(output_dir, 'Neutron_Flux_Distribution.gif')
# I_gif_path = os.path.join(output_dir, 'Iodine_Concentration_Distribution.gif')
# Xe_gif_path = os.path.join(output_dir, 'Xenon135_Concentration_Distribution.gif')

# create_flux_animation(x, phi_history, time_history, flux_gif_path)
# create_I_concentration_animation(x, N_I_history, time_history, I_gif_path)
# create_Xe_concentration_animation(x, N_Xe_history, time_history, Xe_gif_path)

# logging.info(f"Animations saved to {output_dir}")

logging.info('Plotting completed successfully!')
