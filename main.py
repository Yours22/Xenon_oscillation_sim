# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# 参数定义
# 空间域
L = 10.0  # 一维空间长度，单位cm
nx = 100  # 离散空间点数
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

# 时间域
total_time = 1000  # 总时间，单位s
delta_t = 0.1      # 时间步长，单位s
nt = int(total_time / delta_t)

# 物理参数
D = 1.0            # 扩散系数，cm^2/s
sigma_a_Xe = 2.7e-18  # Xe吸收截面，cm^2
sigma_a_I = 1.0e-20   # I吸收截面，cm^2
nu = 2.73             # 中子产额
Sigma_f = 0.043      # 裂变截面，cm^2
gamma_I = 6.386e-2   # I的产生速率
gamma_Xe = 0.228e-2  # Xe的产生速率
lambda_I = 2.87e-5   # I的衰变常数，s^{-1}
lambda_Xe = 2.09e-5  # Xe的衰变常数，s^{-1}
beta = 0.0065        # 缓发中子产额
Lambda = 0.1         # 中子寿命，s^{-1}
rho_0 = 0            # 初始反应性
Delta_rho_Xe = 0.000  # Xe引起的反应性变化
v=220000               # 中子速度，cm/s

# Derived parameters
rho = rho_0 + Delta_rho_Xe  # Reactivity

# Initial conditions
phi_initial = np.ones(nx) * 1e14    # 初始中子通量
N_I_initial = np.zeros(nx)          # 初始I-135浓度
N_Xe_initial = np.zeros(nx)         # 初始Xe-135浓度

# %%
# 初始化
phi = phi_initial.copy()
N_I = N_I_initial.copy()
N_Xe = N_Xe_initial.copy()

# 存储结果
phi_history = []
N_I_history = []
N_Xe_history = []

# %%
def compute_phi_derivative(phi, D, sigma_a_Xe, nu, Sigma_f, rho, beta, Lambda, x, dx):
    """
    Compute the time derivative of phi using finite differences for spatial derivatives.
    """
    dphi_dt = np.zeros_like(phi)
    
    # Compute second spatial derivative using central differences
    d2phi_dx2 = np.zeros_like(phi)
    d2phi_dx2[1:-1] = (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / dx**2
    
    # Boundary conditions (e.g., phi=0 at boundaries)
    d2phi_dx2[0] = 0
    d2phi_dx2[-1] = 0
    
    # Compute dn/dt and rho (if necessary)
    # Assuming n = phi / v, but v is not defined. If v=1, then n=phi
    # Adjust as per actual definition
    
    # Example: v=1 for simplicity
    v = 1.0
    n = phi / v
    dn_dt = (rho * Lambda)  # Assuming dn/dt = (k_eff -1)/Lambda, where rho = (k_eff -1)/k_eff
    
    # Update dphi/dt based on the given PDE
    dphi_dt = D * d2phi_dx2 - sigma_a_Xe * phi + nu * Sigma_f * phi - ((rho - beta) / Lambda) * phi
    
    return dphi_dt

def iodine_xenon_dynamics(phi, N_I, N_Xe, gamma_I, gamma_Xe, Sigma_f_I, lambda_I, lambda_Xe, sigma_a_Xe):
    """
    Compute the time derivatives of I and Xe-135 concentrations.
    """
    dN_I_dt = gamma_I * Sigma_f_I * phi - lambda_I * N_I
    dN_Xe_dt = lambda_I * N_I + gamma_Xe * Sigma_f_I * phi - (lambda_Xe + sigma_a_Xe * phi) * N_Xe
    return dN_I_dt, dN_Xe_dt

# %% [markdown]
# ## Simulation Loop

# %%
for t_step in range(nt):
    # Compute derivatives
    dphi_dt = compute_phi_derivative(phi, D, sigma_a_Xe, nu, Sigma_f, rho, beta, Lambda, x, dx)
    dN_I_dt, dN_Xe_dt = iodine_xenon_dynamics(phi, N_I, N_Xe, gamma_I, gamma_Xe, Sigma_f, lambda_I, lambda_Xe, sigma_a_Xe)
    
    # Update concentrations using Euler method
    phi += delta_t * dphi_dt
    N_I += delta_t * dN_I_dt
    N_Xe += delta_t * dN_Xe_dt
    
    # Apply boundary conditions (e.g., phi=0 at boundaries)
    phi[0] = 0
    phi[-1] = 0
    
    # Store results for visualization every 1000 steps
    if t_step % 1000 == 0:
        phi_history.append(phi.copy())
        N_I_history.append(N_I.copy())
        N_Xe_history.append(N_Xe.copy())

# %% [markdown]
# ## Visualization

# %%
# Time points to visualize
time_points = np.linspace(0, total_time, len(phi_history))

# Plot neutron flux phi at different times
plt.figure(figsize=(10, 6))
for i, phi_snapshot in enumerate(phi_history):
    plt.plot(x, phi_snapshot, label=f't={time_points[i]:.1f}s')
plt.xlabel('Position x (cm)')
plt.ylabel('Neutron Flux φ(x, t) (neutrons/cm²/s)')
plt.title('Neutron Flux Distribution Over Time')
plt.legend()
plt.grid(True)
plt.savefig('Neutron_Flux_Distribution.png')
plt.show()

# Plot Iodine Concentration N_I at different times
plt.figure(figsize=(10, 6))
for i, N_I_snapshot in enumerate(N_I_history):
    plt.plot(x, N_I_snapshot, label=f't={time_points[i]:.1f}s')
plt.xlabel('Position x (cm)')
plt.ylabel('Iodine Concentration N_I(x, t)')
plt.title('Iodine Concentration Distribution Over Time')
plt.legend()
plt.grid(True)
plt.savefig('Iodine_Concentration_Distribution.png')
plt.show()

# Plot Xenon-135 Concentration N_Xe at different times
plt.figure(figsize=(10, 6))
for i, N_Xe_snapshot in enumerate(N_Xe_history):
    plt.plot(x, N_Xe_snapshot, label=f't={time_points[i]:.1f}s')
plt.xlabel('Position x (cm)')
plt.ylabel('Xenon-135 Concentration N_Xe(x, t)')
plt.title('Xenon-135 Concentration Distribution Over Time')
plt.legend()
plt.grid(True)
plt.savefig('Xenon135_Concentration_Distribution.png')
plt.show()

