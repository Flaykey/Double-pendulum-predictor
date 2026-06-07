import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sm
import joblib
from scipy.integrate import solve_ivp

print("Initializing chaotic pendulum mathematics...")
t = sm.symbols('t')
m1, m2, g = sm.symbols('m1 m2 g', positive=True)
theta1_f, theta2_f = sm.symbols('theta1 theta2', cls=sm.Function)

theta1 = theta1_f(t)
theta2 = theta2_f(t)

l1, l2 = 1, 1

x1 = l1 * sm.sin(theta1)
y1 = -l1 * sm.cos(theta1)
x2 = x1 + l2 * sm.sin(theta2)
y2 = y1 - l2 * sm.cos(theta2)

theta1_d = sm.diff(theta1, t)
theta2_d = sm.diff(theta2, t)
theta1_dd = sm.diff(theta1_d, t)
theta2_dd = sm.diff(theta2_d, t)

x1_d = sm.diff(x1, t)
y1_d = sm.diff(y1, t)
x2_d = sm.diff(x2, t)
y2_d = sm.diff(y2, t)

T1 = 0.5 * m1 * (x1_d**2 + y1_d**2)
T2 = 0.5 * m2 * (x2_d**2 + y2_d**2)
V1 = m1 * g * y1
V2 = m2 * g * y2

L = T1 + T2 - (V1 + V2)

LE1 = sm.diff(sm.diff(L, theta1_d), t) - sm.diff(L, theta1)
LE2 = sm.diff(sm.diff(L, theta2_d), t) - sm.diff(L, theta2)

sol = sm.solve([LE1, LE2], [theta1_dd, theta2_dd])

ACC = sm.lambdify(
    (theta1, theta2, theta1_d, theta2_d, t, m1, m2, g),
    [sol[theta1_dd], sol[theta2_dd]],
    modules="numpy"
)

def system(t, y, m1v, m2v, gv):
    th1, w1, th2, w2 = y
    a1, a2 = ACC(th1, th2, w1, w2, t, m1v, m2v, gv)
    return [w1, a1, w2, a2]

m1v, m2v, gv = 5, 1, 9.81
dt = 0.02
test_steps = 150 

try:
    mlp = joblib.load("trained_mlp_model.pkl")
    scaler = joblib.load("fitted_scaler.pkl")
    print("MLP Model and Scaler loaded successfully!")
except FileNotFoundError:
    print("Error: Make sure 'trained_mlp_model.pkl' and 'fitted_scaler.pkl' are in this directory.")

initial_state = [1.5, 0.0, 1.5, 0.0] 

t_eval = np.linspace(0, dt * test_steps, test_steps)
sol_true = solve_ivp(system, (0, dt * test_steps), initial_state, t_eval=t_eval, args=(m1v, m2v, gv), method="RK45")
true_traj = sol_true.y.T

mlp_traj = [initial_state]
current_state = np.array(initial_state)
for _ in range(test_steps - 1):
    state_scaled = scaler.transform(current_state.reshape(1, -1))
    next_state = mlp.predict(state_scaled)[0]
    next_state[0] = np.arctan2(np.sin(next_state[0]), np.cos(next_state[0]))
    next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))
    mlp_traj.append(next_state)
    current_state = next_state
mlp_traj = np.array(mlp_traj)

def get_xy(trajectory):
    th1 = trajectory[:, 0]
    th2 = trajectory[:, 2]
    x1 = l1 * np.sin(th1)
    y1 = -l1 * np.cos(th1)
    x2 = x1 + l2 * np.sin(th2)
    y2 = y1 - l2 * np.cos(th2)
    return x1, y1, x2, y2

true_x1, true_y1, true_x2, true_y2 = get_xy(true_traj)
mlp_x1, mlp_y1, mlp_x2, mlp_y2 = get_xy(mlp_traj)
time_axis = np.arange(test_steps) * dt

fig = plt.figure(figsize=(14, 8))


ax_true = fig.add_subplot(2, 2, 1)
ax_true.set_title("Ground Truth (RK45 Physics)")
ax_true.set_xlim(-2.2, 2.2)
ax_true.set_ylim(-2.2, 2.2)
ax_true.set_aspect('equal')
ax_true.grid(True)
line_true, = ax_true.plot([], [], 'o-', lw=3, color='green', markersize=8)


ax_mlp = fig.add_subplot(2, 2, 2)
ax_mlp.set_title("MLP Neural Network Prediction")
ax_mlp.set_xlim(-2.2, 2.2)
ax_mlp.set_ylim(-2.2, 2.2)
ax_mlp.set_aspect('equal')
ax_mlp.grid(True)
line_mlp, = ax_mlp.plot([], [], 'o--', lw=3, color='red', markersize=8)


ax_graph1 = fig.add_subplot(2, 2, 3)
ax_graph1.set_xlim(0, dt * test_steps)
ax_graph1.set_ylim(-np.pi, np.pi)
ax_graph1.set_ylabel("Theta 1 (rad)")
ax_graph1.set_xlabel("Time (s)")
ax_graph1.grid(True)
graph_true_th1, = ax_graph1.plot([], [], 'g-', label='True')
graph_mlp_th1, = ax_graph1.plot([], [], 'r--', label='MLP')
ax_graph1.legend(loc="upper right")


ax_graph2 = fig.add_subplot(2, 2, 4)
ax_graph2.set_xlim(0, dt * test_steps)
ax_graph2.set_ylim(-np.pi, np.pi)
ax_graph2.set_ylabel("Theta 2 (rad)")
ax_graph2.set_xlabel("Time (s)")
ax_graph2.grid(True)
graph_true_th2, = ax_graph2.plot([], [], 'b-', label='True')
graph_mlp_th2, = ax_graph2.plot([], [], 'r--', label='MLP')
ax_graph2.legend(loc="upper right")

plt.tight_layout()

def update(frame):
    line_true.set_data([0, true_x1[frame], true_x2[frame]], [0, true_y1[frame], true_y2[frame]])
    line_mlp.set_data([0, mlp_x1[frame], mlp_x2[frame]], [0, mlp_y1[frame], mlp_y2[frame]])

    graph_true_th1.set_data(time_axis[:frame], true_traj[:frame, 0])
    graph_mlp_th1.set_data(time_axis[:frame], mlp_traj[:frame, 0])
    
    graph_true_th2.set_data(time_axis[:frame], true_traj[:frame, 2])
    graph_mlp_th2.set_data(time_axis[:frame], mlp_traj[:frame, 2])
    
    return line_true, line_mlp, graph_true_th1, graph_mlp_th1, graph_true_th2, graph_mlp_th2

ani = animation.FuncAnimation(
    fig, update, frames=test_steps, interval=dt*1000, blit=True, repeat=True
)

plt.show()