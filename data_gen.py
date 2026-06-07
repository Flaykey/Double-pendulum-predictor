import numpy as np
from scipy.integrate import solve_ivp
import sympy as sm
import pandas as pd
import random

t = sm.symbols('t')
m1, m2, g = sm.symbols('m1 m2 g', positive=True)

theta1_f, theta2_f = sm.symbols('theta1 theta2', cls=sm.Function)

theta1 = theta1_f(t)
theta2 = theta2_f(t)

l1 = 1
l2 = 1

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
steps = 25
rollouts = 1000

data = []

state = [1.0, 0.0, 1.0, 0.0]


for r in range(rollouts):

    if r % 10 == 0:
        state = [
            random.uniform(-np.pi, np.pi),
            random.uniform(-2, 2),
            random.uniform(-np.pi, np.pi),
            random.uniform(-2, 2)
        ]

    t_eval = np.linspace(0, dt * steps, steps)

    sol = solve_ivp(
        system,
        (0, dt * steps),
        state,
        t_eval=t_eval,
        args=(m1v, m2v, gv),
        method="RK45"
    )

    traj = sol.y.T


    for i in range(len(traj) - 1):

        s_t = traj[i]
        s_next = traj[i + 1]

        data.append({
            "theta1": s_t[0],
            "omega1": s_t[1],
            "theta2": s_t[2],
            "omega2": s_t[3],

            "next_theta1": s_next[0],
            "next_omega1": s_next[1],
            "next_theta2": s_next[2],
            "next_omega2": s_next[3],

            "rollout": r
        })

    state = traj[-1]

df = pd.DataFrame(data)
df.to_csv("double_pendulum_dataset.csv", index=False)

print(df.head())
print("Saved:", len(df), "samples")