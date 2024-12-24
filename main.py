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
    # 假设左边界和右边界的phi值为固定值，可以根据具体问题设定
    # 例如，将边界值固定为0
    phi_extended[0] = 0.0          # 左边界
    phi_extended[-1] = 0.0         # 右边界

    # 计算二阶导数
    d2phi_dx2 = (phi_extended[2:] - 2 * phi_extended[1:-1] + phi_extended[:-2]) / dx**2

    # 计算 dphi/dt
    dphi_dt = D * d2phi_dx2 - sigma_a_Xe * N_Xe * phi + nu * Sigma_f * phi
    # dphi_dt = (D * d2phi_dx2 - sigma_a_Xe * N_Xe * phi + nu * Sigma_f * phi 
    #            - ((rho - beta) / Lambda) * phi)

    return dphi_dt

def iodine_xenon_dynamics(phi, N_I, N_Xe):
    dN_I_dt = gamma_I * Sigma_f * phi - lambda_I * N_I
    dN_Xe_dt = lambda_I * N_I + gamma_Xe * Sigma_f * phi - (lambda_Xe + sigma_a_Xe * phi) * N_Xe
    return dN_I_dt, dN_Xe_dt

max_steps = 145000

# 时间循环
for t_step in range(max_steps):
    print(f'Processing time step {t_step}/{nt}', end='\r')
    current_time = t_step * delta_t

    dN_I_dt, dN_Xe_dt = iodine_xenon_dynamics(phi, N_I, N_Xe)
    N_I += delta_t * dN_I_dt
    N_Xe += delta_t * dN_Xe_dt
    dphi_dt = compute_phi_derivative(phi, rho, beta, Lambda, N_Xe)
    phi += delta_t * dphi_dt

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
        logging.info(f'Processing time step {t_step}/{nt}, time={t_step * delta_t:.2f}s')

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

# 保存数据
output_dir = 'results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_file = os.path.join(output_dir, 'simulation_data.npz')
np.savez_compressed(data_file,
                    x=x,
                    time_history=time_history,
                    phi_history=phi_history,
                    N_I_history=N_I_history,
                    N_Xe_history=N_Xe_history)

logging.info(f'Data successfully saved to {data_file}')

print('Simulation completed and data saved successfully!')
