import os
import shutil
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
import casadi as ca
import math
import matplotlib.pyplot as plt

# Register the custom environment
register(
    id="my-highway-v0",
    entry_point="my_envs.custom_highway:MyHighwayEnv",
)

# Global parameters
DT = 0.05  # MPC time step in seconds
N = 30  # Increased MPC horizon for better lookahead (1.5 seconds)
EPISODE_MAX_STEPS = 40  # number of policy steps

V_MIN, V_MAX = 15.0, 40.0
Y_MIN, Y_MAX = -2.0, 6.0
A_MAX = 2 * math.sqrt(2)  # maximum acceleration
STEER_MAX = 0.2  # maximum steering

# Adjusted tracking weights
Q_ACC = 20  # Reduced to make acceleration less costly
Q_STEER = 500  # Reduced to allow more freedom in steering

# Vehicle dimensions and safety margins
VEHICLE_LENGTH = 5
VEHICLE_WIDTH = 2
# Increased collision threshold for more safety margin
COLLISION_THRESHOLD = 5

# Wheelbase
L_model = 2.5


###############################################################################
# RL Action Interpretation
###############################################################################
def interpret_discrete_action(rl_action_idx):
    try:
        idx = int(rl_action_idx)
    except Exception as e:
        print("Error converting RL action to int:", e)
        idx = rl_action_idx
    a_rl = 0.0
    delta_rl = 0.0
    if idx == 0:  # LEFT
        delta_rl = -0.2
    elif idx == 2:  # RIGHT
        delta_rl = 0.2
    elif idx == 3:  # FASTER
        a_rl = 1.0
    elif idx == 4:  # SLOWER
        a_rl = -1.0
    return a_rl, delta_rl


###############################################################################
# Obstacle Trajectories - Improved for more realistic predictions
###############################################################################
def build_obstacle_trajectories(env, ego_x, ego_y, ego_v, dt, N):
    """
    Build more realistic obstacle trajectories based on current environment state
    """
    obs1_x = []
    obs1_y = []
    obs2_x = []
    obs2_y = []

    # Get vehicles from environment
    vehicles = env.unwrapped.road.vehicles
    obstacles = []

    for veh in vehicles:
        if veh is not env.unwrapped.vehicle:  # If not ego vehicle
            obstacles.append(veh)

    obs1 = obstacles[0]
    obs2 = obstacles[1]

    obs1_x_init, obs1_y_init = obs1.position
    obs1_v = obs1.speed
    obs1_x = [obs1_x_init + obs1_v * (t * dt) for t in range(N + 1)]
    obs1_y = [obs1_y_init for t in range(N + 1)]

    obs2_x_init, obs2_y_init = obs2.position
    obs2_v = obs2.speed
    obs2_x = [obs2_x_init + obs2_v * (t * dt) for t in range(N + 1)]
    obs2_y = [obs2_y_init for t in range(N + 1)]

    return obs1_x, obs1_y, obs2_x, obs2_y


###############################################################################
# MPC Setup using CasADi
###############################################################################
def build_casadi_mpc(N, dt):
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    v = ca.SX.sym('v')
    th = ca.SX.sym('th')
    state = ca.vertcat(x, y, v, th)

    a = ca.SX.sym('a')
    delta = ca.SX.sym('delta')
    control = ca.vertcat(a, delta)

    beta = ca.atan(0.5 * ca.tan(delta))
    x_next = x + v * ca.cos(th + beta) * dt
    y_next = y + v * ca.sin(th + beta) * dt
    v_next = v + a * dt
    th_next = th + (v * ca.sin(beta)) / (L_model / 2) * dt
    rhs = ca.vertcat(x_next, y_next, v_next, th_next)
    f_step = ca.Function('f_step', [state, control], [rhs])

    X = ca.SX.sym('X', N + 1, 4)
    U = ca.SX.sym('U', N, 2)

    P = ca.SX.sym('P', 8 + 6 * N)

    cost_expr = 0
    g_expr = []

    # Initial state constraint
    for i in range(4):
        g_expr.append(X[0, i] - P[i])

    # Control tracking costs and dynamics constraints
    for t in range(N):
        a_ref = P[4 + t]
        delta_ref = P[4 + N + t]

        # Tracking costs
        cost_expr += Q_ACC * (U[t, 0] - a_ref) ** 2 + Q_STEER * (U[t, 1] - delta_ref) ** 2

        # Smoother control changes by penalizing rate of change
        if t > 0:
            cost_expr += 10.0 * (U[t, 0] - U[t - 1, 0]) ** 2  # Penalize acceleration changes
            cost_expr += 10.0 * (U[t, 1] - U[t - 1, 1]) ** 2  # Penalize steering changes

        # Dynamics constraints
        X_next = f_step(X[t, :].T, U[t, :].T)
        for i in range(4):
            g_expr.append(X[t + 1, i] - X_next[i])

    # Lane centering soft cost: prefer to be in the center of lanes
    lane_centers = [0.0, 4.0]  # Centers of the two lanes
    for t in range(N + 1):
        y_pos = X[t, 1]
        lane_cost = ca.fmin(
            (y_pos - lane_centers[0]) ** 2,
            (y_pos - lane_centers[1]) ** 2
        )
        cost_expr += 5 * lane_cost  # Soft cost for lane centering

    # Extra constraints
    g_extra = []
    lbg_extra = []
    ubg_extra = []

    # State bounds constraints
    for t in range(N + 1):
        # Speed constraints
        g_extra.append(X[t, 2])
        lbg_extra.append(V_MIN)
        ubg_extra.append(V_MAX)

        # Lateral position constraints
        g_extra.append(X[t, 1])
        lbg_extra.append(Y_MIN)
        ubg_extra.append(Y_MAX)

    # Collision avoidance constraints
    for t in range(N + 1):
        # Scale collision checking by time step to be less conservative at later time steps
        time_scale = min(1.0 + t * 0.05, 1.5)  # Gradually increases threshold with time
        scaled_threshold = COLLISION_THRESHOLD * time_scale

        obs1_x = P[4 + 2 * N + t]
        obs1_y = P[4 + 2 * N + (N + 1) + t]
        ell1 = ((X[t, 0] - obs1_x) / (VEHICLE_LENGTH / 2)) ** 2 + ((X[t, 1] - obs1_y) / (VEHICLE_WIDTH / 2)) ** 2
        g_extra.append(ell1)
        lbg_extra.append(scaled_threshold)
        ubg_extra.append(1e12)

        obs2_x = P[4 + 2 * N + 2 * (N + 1) + t]
        obs2_y = P[4 + 2 * N + 3 * (N + 1) + t]
        ell2 = ((X[t, 0] - obs2_x) / (VEHICLE_LENGTH / 2)) ** 2 + ((X[t, 1] - obs2_y) / (VEHICLE_WIDTH / 2)) ** 2
        g_extra.append(ell2)
        lbg_extra.append(scaled_threshold)
        ubg_extra.append(1e12)

    g_total = ca.vertcat(*(g_expr + g_extra))
    dec_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    nlp_prob = {'f': cost_expr, 'x': dec_vars, 'p': P, 'g': g_total}

    solver_opts = {
        'ipopt.print_level': 0,  # Reduced verbosity
        'print_time': 0,
        'ipopt.max_iter': 1000,
        'ipopt.tol': 1e-3,  # More relaxed tolerance
        'ipopt.acceptable_tol': 1e-2,  # More relaxed acceptable tolerance
        'ipopt.acceptable_obj_change_tol': 1e-2  # Accept solutions that improve the objective by this amount
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, solver_opts)

    eq_count = 4 * (N + 1)
    lbg_list = [0] * eq_count
    ubg_list = [0] * eq_count
    lbg_list.extend(lbg_extra)
    ubg_list.extend(ubg_extra)

    nx = (N + 1) * 4
    lbx_list = [-1e12] * nx
    ubx_list = [1e12] * nx
    for t in range(N):
        lbx_list.append(-A_MAX)
        ubx_list.append(A_MAX)
        lbx_list.append(-STEER_MAX)
        ubx_list.append(STEER_MAX)

    return solver, lbg_list, ubg_list, lbx_list, ubx_list


###############################################################################
# Shift previous solution to use as a warm start for next iteration.
###############################################################################
def shift_solution(prev_sol, x0, y0, v0, th0, N, dt):
    n_state = (N + 1) * 4
    n_control = N * 2
    # Reshape state and control parts.
    states = prev_sol[:n_state].reshape((N + 1, 4))
    controls = prev_sol[n_state:].reshape((N, 2))

    # Build new state trajectory: set new initial state and shift the previous states.
    new_states = np.zeros_like(states)
    new_states[0] = np.array([x0, y0, v0, th0])
    new_states[1:-1] = states[2:]

    # For the last state, extrapolate from the second-to-last state
    last_x = new_states[-2, 0] + new_states[-2, 2] * np.cos(new_states[-2, 3]) * dt
    last_y = new_states[-2, 1] + new_states[-2, 2] * np.sin(new_states[-2, 3]) * dt
    last_v = new_states[-2, 2]  # Maintain velocity
    last_th = new_states[-2, 3]  # Maintain heading
    new_states[-1] = np.array([last_x, last_y, last_v, last_th])

    # Shift controls: drop the first control and repeat the last one.
    new_controls = np.zeros_like(controls)
    new_controls[:-1] = controls[1:]
    new_controls[-1] = controls[-1]  # Alternatively, use the latest RL command [a_rl, delta_rl].

    warm_start = np.concatenate([new_states.flatten(), new_controls.flatten()])
    return warm_start


###############################################################################
# Solve MPC (modified to accept an optional initial guess for warm start)
###############################################################################
def solve_mpc(solver, lbg, ubg, lbx, ubx,
              x0, y0, v0, th0,
              a_rl, delta_rl,
              obs1_x_list, obs1_y_list, obs2_x_list, obs2_y_list,
              N, dt, dec_init=None):
    # Construct parameter vector P.
    P_vals = [x0, y0, v0, th0]
    P_vals += [a_rl] * N
    P_vals += [delta_rl] * N
    P_vals += obs1_x_list + obs1_y_list + obs2_x_list + obs2_y_list
    P_np = np.array(P_vals)

    n_X = (N + 1) * 4
    n_U = N * 2
    if dec_init is None:
        dec_init = np.zeros(n_X + n_U)
        for t in range(N + 1):
            idx = t * 4
            if t == 0:
                dec_init[idx:idx + 4] = [x0, y0, v0, th0]
            else:
                # More sophisticated initialization: account for lane changes
                estimated_y = y0
                if delta_rl != 0:
                    # Estimate lateral position change based on steering
                    estimated_y = y0 + min(0.5, abs(delta_rl * t * dt * v0)) * np.sign(delta_rl)

                dec_init[idx:idx + 4] = [x0 + v0 * dt * t, estimated_y, v0, th0]

        # Initialize controls with gradual changes
        for t in range(N):
            idx = n_X + t * 2
            # Smoothly transition to RL controls
            smooth_a = a_rl * min(1.0, (t + 1) / 5.0)
            smooth_delta = delta_rl * min(1.0, (t + 1) / 5.0)
            dec_init[idx:idx + 2] = [smooth_a, smooth_delta]

    # First try normal solve
    try:
        res = solver(x0=dec_init, p=P_np, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
        sol_status = solver.stats()['return_status']
        if 'Infeasible' in sol_status or 'Failure' in sol_status:
            raise Exception("Initial solve failed")

        cost_val = float(res['f'].full().flatten()[0])
        sol = res['x'].full().flatten()
    except Exception as e:
        print(f"[MPC] Initial attempt failed: {e}, trying with relaxed constraints")

        # Try with temporarily relaxed collision constraints
        temp_lbg = lbg.copy()
        for i in range(len(lbg) - 2 * (N + 1), len(lbg)):
            if temp_lbg[i] > 0:  # Only modify collision constraints
                temp_lbg[i] *= 0.8  # Relax by 20%

        try:
            res = solver(x0=dec_init, p=P_np, lbg=temp_lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            sol_status = solver.stats()['return_status']
            if 'Infeasible' in sol_status or 'Failure' in sol_status:
                print("[MPC] Relaxed attempt also failed:", sol_status)
                return None, None

            cost_val = float(res['f'].full().flatten()[0])
            sol = res['x'].full().flatten()
            print("[MPC] Successfully solved with relaxed constraints")
        except Exception as e:
            print(f"[MPC] Relaxed attempt also failed: {e}")
            return None, None

    idx_u0 = (N + 1) * 4
    a0 = sol[idx_u0]
    delta0 = sol[idx_u0 + 1]
    return (a0, delta0, cost_val), sol


###############################################################################
# Helper: Compute lane index from lateral position.
###############################################################################
def get_lane_index(y):
    """
    Simplified lane identification:
      - If y is between -2 and 2, return index 0.
      - If y is between 2 and 6, return index 1.
    Values outside these ranges are saturated.
    """
    if -2 <= y < 2:
        return 0
    elif 2 <= y <= 6:
        return 1


###############################################################################
# Main Loop: Run MPC Safety Filter with Gym and DQN
###############################################################################
def run_mpc_safety_filter():
    video_folder = "video_mpc"
    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
    os.makedirs(video_folder, exist_ok=True)

    env_config = {
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "collision_termination": False,
        "offroad_terminal": False,
        "duration": 1000,
    }
    env = gym.make("my-highway-v0", render_mode="rgb_array", config=env_config)

    env = RecordVideo(env, video_folder=video_folder, name_prefix="casadi_mpc_run",
                      episode_trigger=lambda ep: True)

    model = DQN.load("custom_highway_dqn/model.zip")

    rl_acc_list = []
    rl_steer_list = []
    mpc_acc_list = []
    mpc_steer_list = []
    cost_list = []

    # New lists for tracking ego vehicle speed and trajectory
    ego_speed_list = []
    ego_x_list = []
    ego_y_list = []
    timestamp_list = []

    obs, info = env.reset()
    step_count = 0
    done = False
    truncated = False

    # Warm start variable: holds the previous MPC solution.
    warm_start = None

    # Build MPC solver once (more efficient)
    solver, lbg, ubg, lbx, ubx = build_casadi_mpc(N, DT)

    # Keep track of whether we're in an overtaking maneuver
    overtaking = False
    lane_change_cooldown = 0

    while step_count < EPISODE_MAX_STEPS and not done and not truncated:
        step_count += 1

        rl_action_idx, _ = model.predict(obs, deterministic=True)
        print(f"[STEP {step_count}] RL action={rl_action_idx}")
        a_rl, delta_rl = interpret_discrete_action(rl_action_idx)
        print(f"RL control: a={a_rl}, delta={delta_rl}")
        rl_acc_list.append(a_rl)
        rl_steer_list.append(delta_rl)

        ego = env.unwrapped.vehicle
        if ego is None:
            break
        x_ego, y_ego = ego.position
        v_ego = ego.speed
        th_ego = getattr(ego, 'heading', 0.0)

        # Store ego vehicle data for plotting
        ego_speed_list.append(v_ego)
        ego_x_list.append(x_ego)
        ego_y_list.append(y_ego)
        timestamp_list.append(step_count * DT)

        # Log vehicle positions
        ego_lane = get_lane_index(y_ego)
        print(f"  Ego vehicle: x = {x_ego:.2f}, y = {y_ego:.2f}, v = {v_ego:.2f}, lane = {ego_lane}")

        vehicles_ahead = []
        for i, veh in enumerate(env.unwrapped.road.vehicles):
            if veh is not ego:
                x, y = veh.position
                v = veh.speed
                veh_lane = get_lane_index(y)

                # Check if vehicle is ahead in same lane
                if x > x_ego and abs(y - y_ego) < 2.0 and v < v_ego:
                    vehicles_ahead.append((veh, x - x_ego))  # Store vehicle and distance

                print(f"  Vehicle {i}: x = {x:.2f}, y = {y:.2f}, v = {v:.2f}, lane = {veh_lane}")

        # Sort vehicles ahead by distance
        vehicles_ahead.sort(key=lambda x: x[1])

        # Get updated obstacle trajectories based on current environment state
        obs1_x_list, obs1_y_list, obs2_x_list, obs2_y_list = build_obstacle_trajectories(
            env, x_ego, y_ego, v_ego, DT, N)

        # If a warm start is available, update it with the current state
        if warm_start is not None:
            # Update the first state with the current state
            warm_start[:4] = np.array([x_ego, y_ego, v_ego, th_ego])
            dec_init = warm_start
        else:
            dec_init = None

        sol_data, sol = solve_mpc(solver, lbg, ubg, lbx, ubx,
                                  x_ego, y_ego, v_ego, th_ego,
                                  a_rl, delta_rl,
                                  obs1_x_list, obs1_y_list, obs2_x_list, obs2_y_list,
                                  N, DT, dec_init=dec_init)

        if sol_data is None:
            print("   => MPC infeasible, fallback to safe action")
            # If we were turning, keep turning but slow down
            if abs(delta_rl) > 0.1:
                final_action_idx = 0 if delta_rl < 0 else 2  # Continue lane change
                mpc_acc_list.append(0.0)
                mpc_steer_list.append(delta_rl)
            else:
                final_action_idx = 1  # IDLE
                mpc_acc_list.append(0.0)
                mpc_steer_list.append(0.0)

            cost_list.append(np.nan)
            warm_start = None  # Reset warm start if no solution is found
        else:
            a0, delta0, cost_val = sol_data

            # Check lane boundaries using the simplified lane index
            lane_index = get_lane_index(y_ego)
            steer_threshold = 0.12  # Threshold to consider a lane change

            # If overtaking in progress, check if we should return to the original lane
            if overtaking and lane_index == 0:
                # Check if we've passed all the vehicles we were overtaking
                all_passed = True
                for veh, _ in vehicles_ahead:
                    x, _ = veh.position
                    if x > x_ego:  # Vehicle still ahead
                        all_passed = False
                        break

                if all_passed and (y_ego > -2):
                    print("  Completing overtaking, returning to right lane")
                    overtaking = False
                    lane_change_cooldown = 3  # Prevent immediate lane changes
                    delta0 = 0.2

            # If MPC suggests left lane change but ego is already in leftmost lane
            if delta0 < -steer_threshold and lane_index == 0 and not overtaking:
                print("   => At leftmost lane; limiting left steering")
                delta0 = 0  # Stay in lane

            # If MPC suggests right lane change but ego is already in rightmost lane
            elif delta0 > steer_threshold and lane_index == 1 and not overtaking:
                print("   => At rightmost lane; limiting right steering")
                delta0 = 0  # Stay in lane

            print(f"   => MPC solution: a0={a0:.2f}, delta0={delta0:.2f}, cost={cost_val:.2f}")
            mpc_acc_list.append(a0)
            mpc_steer_list.append(delta0)
            cost_list.append(cost_val)

            # Map continuous MPC outputs to discrete actions
            if a0 >= 1:
                final_action_idx = 3  # FASTER
            elif a0 <= -1:
                final_action_idx = 4  # SLOWER
            else:
                if delta0 > 0.12:
                    final_action_idx = 2  # RIGHT
                elif delta0 < -0.12:
                    final_action_idx = 0  # LEFT
                else:
                    final_action_idx = 1  # IDLE

            # Shift the solution to use as a warm start for the next iteration
            if sol is not None:
                warm_start = shift_solution(sol, x_ego, y_ego, v_ego, th_ego, N, DT)

            print(f"   => Final discrete action from MPC: {final_action_idx}")

        # Decrement lane change cooldown
        if lane_change_cooldown > 0:
            lane_change_cooldown -= 1

        obs, reward, done, truncated, info = env.step(final_action_idx)

    env.close()

    # Create new plots as specified
    # Step values for x-axis in all plots
    steps = np.arange(1, len(rl_acc_list) + 1)

    # ============ PLOT 1: Control variables (acceleration and steering) ============
    plt.figure(figsize=(12, 6))

    # Subplot 1: Acceleration
    plt.subplot(2, 1, 1)
    plt.plot(steps, rl_acc_list, 'bo-', label='RL Acceleration')
    plt.plot(steps, mpc_acc_list, 'ro-', label='MPC Acceleration')
    plt.xlabel("Policy Step")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.title("Acceleration Commands")
    plt.grid(True, alpha=0.3)

    # Subplot 2: Steering
    plt.subplot(2, 1, 2)
    plt.plot(steps, rl_steer_list, 'bo-', label='RL Steering')
    plt.plot(steps, mpc_steer_list, 'ro-', label='MPC Steering')
    plt.xlabel("Policy Step")
    plt.ylabel("Steering")
    plt.legend()
    plt.title("Steering Commands")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("control_variables.png")

    # ============ PLOT 2: MPC Cost (single plot) ============
    plt.figure(figsize=(10, 6))
    plt.plot(steps, cost_list, 'ko-', label='MPC Cost')
    plt.xlabel("Policy Step")
    plt.ylabel("Cost")
    plt.title("MPC Cost over Policy Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mpc_cost.png")

    # ============ PLOT 3: Speed and track visualization ============
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})

    # Subplot 1: Speed
    ax1.plot(steps, ego_speed_list, 'go-', label='Ego Speed')
    ax1.set_xlabel("Policy Step")
    ax1.set_ylabel("Speed (m/s)")
    ax1.set_title("Ego Vehicle Speed over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Track visualization
    # Invert Y-axis so higher y values are at the top (lane 1/left lane at top)
    # This makes it consistent with the description
    ax2.invert_yaxis()

    # Plot lane boundaries
    lane_bottom = -2.0
    lane_center = 2.0
    lane_top = 6.0

    # Draw lane boundaries
    ax2.axhline(y=lane_bottom, color='k', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.axhline(y=lane_center, color='k', linestyle='-', alpha=0.7, linewidth=1.5)
    ax2.axhline(y=lane_top, color='k', linestyle='--', alpha=0.7, linewidth=1.5)

    # Label lanes (note that with inverted y-axis, lane 0 is now at the bottom)
    ax2.text(np.min(ego_x_list), (lane_bottom + lane_center) / 3, "Lane 0 (LEFT)",
             ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7))
    ax2.text(np.min(ego_x_list), (lane_center + lane_top) / 3, "Lane 1 (RIGHT)",
             ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7))

    # Plot vehicle path with color indicating speed
    scatter = ax2.scatter(ego_x_list, ego_y_list, c=ego_speed_list, cmap='viridis',
                          s=100, marker='o', edgecolors='k', linewidths=0.5)

    # Add arrows to show direction
    for i in range(1, len(ego_x_list), max(1, len(ego_x_list) // 10)):  # Add arrows at intervals
        dx = ego_x_list[i] - ego_x_list[i - 1]
        dy = ego_y_list[i] - ego_y_list[i - 1]
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length > 0:
            ax2.arrow(ego_x_list[i - 1], ego_y_list[i - 1], dx * 0.8, dy * 0.8,
                      head_width=0.3, head_length=0.5, fc='k', ec='k', alpha=0.7)

    # Connect points with a line to show the path
    ax2.plot(ego_x_list, ego_y_list, 'k-', alpha=0.5)

    # Add color bar for speed
    cbar = fig.colorbar(scatter, ax=ax2)
    cbar.set_label('Speed (m/s)')

    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Y Position (m)")
    ax2.set_title("Ego Vehicle Track")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(max(ego_y_list) + 2.5, min(ego_y_list) - 2.5)

    # Draw lane boundaries (assumed lanes: bottom lane between 2 and 6, top lane between -2 and 2)
    ax2.axhline(y=Y_MIN, color='k', linestyle='--', linewidth=1)
    ax2.axhline(y=Y_MAX, color='k', linestyle='--', linewidth=1)
    fig.tight_layout()
    plt.show()

    print("Video recorded in:", video_folder)


if __name__ == "__main__":
    run_mpc_safety_filter()