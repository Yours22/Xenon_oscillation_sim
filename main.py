import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging
from scipy.integrate import solve_ivp

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 参数定义
# 空间域
L = 100  # 一维空间长度，单位cm
nx = 50  # 增加空间离散点数以提高空间分辨率
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

# 时间域
total_time = 10000  # 总时间，单位s
# 在隐式方法中，时间步长由求解器自动选择，因此不需要手动定义delta_t
# 但我们需要定义存储结果的时间点
store_every = 100  # 每隔100步存储一次
estimated_steps = int(total_time / 0.001) // store_every + 1
t_eval = np.linspace(0, total_time, estimated_steps)

# 物理参数
D = 1.02             # 扩散系数，cm²/s
sigma_a_Xe = 2.7e-18 # Xe微观吸收截面，cm²
nu = 2.73            # 中子产额
Sigma_f = 0.043      # 宏观裂变截面，cm-1
gamma_I = 6.386e-2   # I-135的产额
gamma_Xe = 0.228e-2  # Xe-135的产额
lambda_I = 2.87e-5   # I-135的衰变常数，s⁻¹
lambda_Xe = 2.09e-5  # Xe-135的衰变常数，s⁻¹
beta = 0.0065        # 缓发中子产额
Lambda = 0.0003      # 中子寿命，s
rho_0 = 0.05          # 初始反应性
Delta_rho_Xe_initial = 0.0   # Xe引起的反应性变化
v = 220000           # 中子速度，cm/s

# 初始条件
phi_initial = np.ones(nx) * 3e13    # 初始中子通量
N_I_initial = np.zeros(nx)          # 初始I-135浓度
N_Xe_initial = np.zeros(nx)         # 初始Xe-135浓度

# 合并初始条件为单个向量
y0 = np.concatenate([phi_initial, N_I_initial, N_Xe_initial])

def calculate_delta_rho_xe(sigma_a_Xe, N_Xe,phi):
    """
    计算由于 Xe-135 吸收导致的反应性变化。
    
    参数：
    - sigma_a_Xe: Xe-135 的吸收截面
    - N_Xe: Xe-135 的浓度（数组）
    - Lambda: 中子寿命
    
    返回：
    - Delta_rho_Xe: 反应性变化（数组）
    """
    # Delta_rho_Xe = - sigma_a_Xe * N_Xe / Lambda
    Delta_rho_Xe = - sigma_a_Xe * N_Xe /Sigma_f

    return Delta_rho_Xe

def compute_phi_derivative(phi, rho, beta, Lambda, N_I, N_Xe):
    """
    计算中子通量phi的时间导数。
    使用有限差分法计算空间上的扩散项，并考虑反应项。
    """
    # 引入虚拟点以应用边界条件
    phi_extended = np.zeros(nx + 2, dtype=np.float64)
    phi_extended[1:-1] = phi

    # 应用第一类边界条件（Dirichlet 边界条件）
    phi_extended[0] = 0.0          # 左边界
    phi_extended[-1] = 0.0         # 右边界

    # 计算二阶导数
    d2phi_dx2 = (phi_extended[2:] - 2 * phi_extended[1:-1] + phi_extended[:-2]) / dx**2

    # # 计算 dphi/dt
    # dphi_dt = D * d2phi_dx2 + nu * Sigma_f * phi +v*(lambda_I+lambda_Xe)*phi
    # # 添加反应性变化项
    # dphi_dt += ((rho - beta) / Lambda) * phi

    dphi_dt = D * d2phi_dx2 + ((rho - beta) / Lambda) * phi + lambda_Xe * N_Xe + lambda_I * N_I

    return dphi_dt


def iodine_xenon_dynamics(phi, N_I, N_Xe):
    """
    计算I-135和Xe-135的时间导数。
    """
    dN_I_dt = gamma_I * Sigma_f * phi - lambda_I * N_I
    dN_Xe_dt = lambda_I * N_I + gamma_Xe * Sigma_f * phi - lambda_Xe * N_Xe
    return dN_I_dt, dN_Xe_dt

# def xenon_dynamics(phi, C):
#     """
#     计算 Xe-135 的时间导数。
    
#     参数：
#     - phi: 中子通量（数组）
#     - C: Xe-135 浓度（数组）
    
#     返回：
#     - dC_dt: Xe-135 的时间导数（数组）
#     """
#     # 根据耦合方程：
#     # dC/dt = (beta / Lambda) * (phi / v) - lambda_Xe * C
#     dC_dt = (beta / Lambda) * (phi / v) - lambda_Xe * C
#     return dC_dt


def ode_system(t, y):
    """
    定义整个系统的微分方程。
    y 包含 [phi, N_I, N_Xe] 的所有空间点。
    """
    phi = y[:nx]
    N_I = y[nx:2*nx]
    N_Xe = y[2*nx:]

    # 计算反应性变化
    Delta_rho_Xe = calculate_delta_rho_xe(sigma_a_Xe, N_Xe,phi)
    rho = rho_0 + Delta_rho_Xe

    # 计算phi的导数
    dphi_dt = compute_phi_derivative(phi, rho, beta, Lambda,N_I, N_Xe)

    # 计算N_I和N_Xe的导数
    dN_I_dt, dN_Xe_dt = iodine_xenon_dynamics(phi, N_I, N_Xe)

    # 计算 C 的导数
    # dC_dt = xenon_dynamics(phi, N_Xe)

    # 合并所有导数为一个向量
    dydt = np.concatenate([dphi_dt, dN_I_dt, dN_Xe_dt])
    # dydt = np.concatenate([dphi_dt, dC_dt])


    return dydt

# 使用 solve_ivp 进行隐式时间积分
logging.info('Starting simulation using implicit BDF method...')
solution = solve_ivp(
    ode_system,
    [0, total_time],
    y0,
    method='BDF',
    t_eval=t_eval,
    vectorized=False,
    rtol=1e-6,
    atol=1e-9
)

# 检查求解是否成功
if not solution.success:
    logging.error(f'ODE solver failed: {solution.message}')
    exit(1)

logging.info('Simulation completed successfully.')

# 提取结果
phi_history = solution.y[:nx].T
N_I_history = solution.y[nx:2*nx].T
N_Xe_history = solution.y[2*nx:].T
time_history = solution.t

# 数据完整性检查
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
np.savez_compressed(
    data_file,
    x=x,
    time_history=time_history,
    phi_history=phi_history,
    N_I_history=N_I_history,
    N_Xe_history=N_Xe_history
)

logging.info(f'Data successfully saved to {data_file}')

print('Simulation completed and data saved successfully!')

# 计算 k_eff 和 rho 的后处理
def compute_keff_rho(time_history, phi_history, N_I_history, N_Xe_history):
    """
    根据中子通量的变化计算 k_eff 和 rho。
    
    参数：
    - time_history: 时间数组
    - phi_history: 中子通量历史数组，形状为 (时间步数, 空间点数)
    - N_I_history: I-135 历史浓度数组
    - N_Xe_history: Xe-135 历史浓度数组
    
    返回：
    - keff_history: k_eff 随时间变化的数组
    - rho_history: rho 随时间变化的数组
    """
    dt = np.gradient(time_history)  # 计算时间步长
    # 计算空间平均的中子通量
    phi_avg = np.mean(phi_history, axis=1)
    # 计算中子通量的时间导数
    dphi_dt = np.gradient(phi_avg, time_history)
    # 计算反应性 rho
    rho = (Lambda * dphi_dt) / phi_avg + beta
    # 计算 k_eff
    keff = 1 / (1 - rho)
    return keff, rho

keff_history, rho_history = compute_keff_rho(time_history, phi_history, N_I_history, N_Xe_history)

# 将 k_eff 和 rho 保存到文件中
data_with_keff = os.path.join(output_dir, 'simulation_data_with_keff_rho.npz')
np.savez_compressed(
    data_with_keff,
    x=x,
    time_history=time_history,
    phi_history=phi_history,
    N_I_history=N_I_history,
    N_Xe_history=N_Xe_history,
    keff_history=keff_history,
    rho_history=rho_history
)

logging.info(f'k_eff and rho successfully saved to {data_with_keff}')

