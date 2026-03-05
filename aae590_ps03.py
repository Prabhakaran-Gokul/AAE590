import numpy as np
import matplotlib.pyplot as plt

from aae590_ps01 import so2_exp, so2_vee, so2_wedge
from aae590_ps02 import (
    se2_wedge,
    so2_log,
    lie_group_integration,
)

GRAVITATIONAL_ACCELERATION = 9.81
RACE_TRACK_TRAJECTORY = {
    "segment_0": {"v_x": 20, "v_y": 0.0, "omega": 0.0, "time": 10.0},
    "segment_1": {"v_x": 15, "v_y": 0.0, "omega": 0.3, "time": 5.24},
    "segment_2": {"v_x": 20, "v_y": 0.0, "omega": 0.0, "time": 5.0},
    "segment_3": {"v_x": 15, "v_y": 0.0, "omega": 0.3, "time": 5.24},
    "segment_4": {"v_x": 20, "v_y": 0.0, "omega": 0.0, "time": 10.0},
    "segment_5": {"v_x": 15, "v_y": 0.0, "omega": 0.3, "time": 5.24},
    "segment_6": {"v_x": 20, "v_y": 0.0, "omega": 0.0, "time": 5.0},
    "segment_7": {"v_x": 15, "v_y": 0.0, "omega": 0.3, "time": 5.24},
}

TEST_TRACK_TRAJECTORY_1 = {
    "segment_0": {"v_x": 20, "v_y": 0.0, "omega": 0.0, "time": 10.0},
}

TEST_TRACK_TRAJECTORY_2 = {
    "segment_0": {"v_x": 15, "v_y": 0.0, "omega": 0.3, "time": 10.0},
}

TWO_D_INS_TRAJECTORY = {
    "acceleration": {"a_x": 2.0, "a_y": 0.0, "omega": 0.0, "time": 5.0},
    "cruise": {"a_x": 0.0, "a_y": 0.0, "omega": 0.0, "time": 5.0},
    "banked_turn": {"a_x": 0.0, "a_y": 0.0, "omega": 0.3, "time": np.pi / 0.3},
    "deceleration": {"a_x": -1.0, "a_y": 0.0, "omega": 0.0, "time": 5.0},
}

SPIRAL_TRAJECTORY = {
    "spiral": {"a_x": 1.0, "a_y": 0.0, "omega": 0.5, "time": 10.0},
}


def get_turning_radius(velocity, phi):
    turning_radius = (velocity**2) / (GRAVITATIONAL_ACCELERATION * np.tan(phi))
    return turning_radius


def get_position_from_SE2(X):
    assert X.shape == (3, 3), "Input matrix must be 3x3."
    return X[0:2, 2]


def get_orientation_from_SE2(X):
    assert X.shape == (3, 3), "Input matrix must be 3x3."
    return np.arctan2(X[1, 0], X[0, 0])


def propagate_trajectory(race_track_trajectory, X_init, dt=0.01):
    X = X_init
    trajectory = {}
    for segment, data in race_track_trajectory.items():
        v_x = data["v_x"]
        v_y = data["v_y"]
        omega = data["omega"]
        time = data["time"]
        print(
            f"{segment}: v_x = {v_x} m/s, v_y = {v_y} m/s, Omega = {omega} rad/s, Time = {time} s"
        )

        trajectory[segment] = [X]  # Store initial pose for the segment
        # Update pose
        xi = np.array([v_x, v_y, omega])
        num_time_steps = int(time / dt)
        for _ in range(num_time_steps):
            X = lie_group_integration(xi=xi, X=X, dt=dt)
            trajectory[segment].append(X)

    return trajectory


def get_bank_angle(velocity, omega):
    phi = np.arctan((omega * velocity) / GRAVITATIONAL_ACCELERATION)
    return phi


def check_bank_angle_feasibility(race_track_trajectory):
    for segment, data in race_track_trajectory.items():
        v_x = data["v_x"]
        v_y = data["v_y"]
        omega = data["omega"]
        if omega != 0.0:  # Only check for segments with non-zero angular velocity
            phi = get_bank_angle(v_x, omega)
            if abs(phi) > np.pi / 4:  # Check if bank angle exceeds 45 degrees
                print(
                    f"Segment {segment} has an infeasible bank angle of {np.degrees(phi)} degrees."
                )
                return False
    return True


def check_group_affine_property(X1, X2, twist):
    left_side = X1 @ X2 @ se2_wedge(twist)
    right_side = (
        X1 @ se2_wedge(twist) @ X2
        + X1 @ X2 @ se2_wedge(twist)
        - X1 @ se2_wedge(twist) @ X2
    )
    return np.allclose(left_side, right_side)


def get_components_from_se22(X):
    assert X.shape == (4, 4), "Input matrix must be 4x4."
    R = X[0:2, 0:2]
    v = X[0:2, 2]
    p = X[0:2, 3]
    return R, v, p


def se22_compose(X1, X2):
    assert X1.shape == (4, 4) and X2.shape == (4, 4), "Input matrices must be 4x4."
    R1, v1, p1 = get_components_from_se22(X1)
    R2, v2, p2 = get_components_from_se22(X2)
    R_composed = R1 @ R2
    v_composed = R1 @ v2 + v1
    p_composed = R1 @ p2 + p1

    X_composed = np.eye(4)
    X_composed[0:2, 0:2] = R_composed
    X_composed[0:2, 2] = v_composed
    X_composed[0:2, 3] = p_composed

    return X_composed


def se22_inverse(X):
    assert X.shape == (4, 4), "Input matrix must be 4x4."
    R, v, p = get_components_from_se22(X)
    R_inv = R.T
    v_inv = -R_inv @ v
    p_inv = -R_inv @ p

    X_inv = np.eye(4)
    X_inv[0:2, 0:2] = R_inv
    X_inv[0:2, 2] = v_inv
    X_inv[0:2, 3] = p_inv

    return X_inv


def se22_wedge(xi):
    assert xi.shape == (5,), "Input vector must be of shape (5,)."
    a1, a2, b2, b3, omega = xi
    omega_wedge = so2_wedge(omega)
    xi_wedge = np.zeros((4, 4))
    xi_wedge[0:2, 0:2] = omega_wedge
    xi_wedge[0:2, 2] = np.array([a1, a2])
    xi_wedge[0:2, 3] = np.array([b2, b3])
    return xi_wedge


def se22_vee(xi_wedge):
    assert xi_wedge.shape == (4, 4), "Input matrix must be 4x4."
    omega = so2_vee(xi_wedge[0:2, 0:2])
    a1, a2 = xi_wedge[0:2, 2]
    b2, b3 = xi_wedge[0:2, 3]
    xi = np.array([a1, a2, b2, b3, omega])
    return xi


def se22_exp(xi):
    assert xi.shape == (5,), "Input vector must be of shape (5,)."
    omega = xi[4]
    a = xi[0:2]
    b = xi[2:4]

    # Use the first 5 terms of the Taylor series expansion for small angles
    # sin x / x
    sinx_over_x = (
        1 - omega**2 / 6 + omega**4 / 120 - omega**6 / 5040 + omega**8 / 362880
    )
    # (1 - cos x) / x
    one_minus_cosx_over_x = (
        omega / 2
        - omega**3 / 24
        + omega**5 / 720
        - omega**7 / 40320
        + omega**9 / 3628800
    )

    if np.isclose(omega, 0):
        V = (
            np.eye(2) * sinx_over_x
            + np.array([[0, -1], [1, 0]]) * one_minus_cosx_over_x
        )
    else:
        V = (
            np.eye(2) * np.sin(omega) / omega
            + np.array([[0, -1], [1, 0]]) * (1 - np.cos(omega)) / omega
        )

    R = so2_exp(omega)
    v = V @ a
    p = V @ b

    X_exp = np.eye(4)
    X_exp[0:2, 0:2] = R
    X_exp[0:2, 2] = v
    X_exp[0:2, 3] = p

    return X_exp


def se22_log(X):
    assert X.shape == (4, 4), "Input matrix must be 4x4."
    R, v, p = get_components_from_se22(X)

    omega = so2_log(R)

    # Use the first 5 terms of the Taylor series expansion for small angles
    # (x / 2) cot(x / 2) = 1 - x^2 / 12 - x^4 / 720 - x^6 / 30240 - x^8 / 1209600
    x_over_2_cot_x_over_2 = (
        1 - omega**2 / 12 - omega**4 / 720 - omega**6 / 30240 - omega**8 / 1209600
    )

    if np.isclose(omega, 0):
        V_inv = np.eye(2) * x_over_2_cot_x_over_2 + np.array([[0, 1], [-1, 0]]) * (
            omega / 2
        )
    else:
        V_inv = np.eye(2) * (omega / 2) * (1 / np.tan(omega / 2)) + np.array(
            [[0, 1], [-1, 0]]
        ) * (omega / 2)

    a = V_inv @ v
    b = V_inv @ p

    xi_log = np.array([a[0], a[1], b[0], b[1], omega])

    return xi_log


def se22_adjoint(X):
    assert X.shape == (4, 4), "Input matrix must be 4x4."
    R, v, p = get_components_from_se22(X)

    Ad_X = np.eye(5)
    Ad_X[0:2, 0:2] = R
    Ad_X[0:2, 2:4] = np.zeros((2, 2))
    Ad_X[0:2, 4] = np.array([v[1], -v[0]])
    Ad_X[2:4, 0:2] = np.zeros((2, 2))
    Ad_X[2:4, 2:4] = R
    Ad_X[2:4, 4] = np.array([p[1], -p[0]])
    Ad_X[4, 0:2] = np.zeros(2)
    Ad_X[4, 2:4] = np.zeros(2)
    Ad_X[4, 4] = 1

    return Ad_X


def se22_small_adjoint(xi):
    assert xi.shape == (5,), "Input vector must be of shape (5,)."
    a1, a2, b1, b2, omega = xi

    ad_X = np.eye(5)
    ad_X[0:2, 0:2] = np.eye(2) * omega
    ad_X[0:2, 2:4] = np.zeros((2, 2))
    ad_X[0:2, 4] = np.array([-a1, -a2])
    ad_X[2:4, 0:2] = np.zeros((2, 2))
    ad_X[2:4, 2:4] = np.eye(2) * omega
    ad_X[2:4, 4] = np.array([-b1, -b2])
    ad_X[4, 0:2] = np.zeros(2)
    ad_X[4, 2:4] = np.zeros(2)
    ad_X[4, 4] = omega

    return ad_X


def lie_bracket(xi1, xi2):
    assert xi1.shape == (5,) and xi2.shape == (5,), (
        "Input vectors must be of shape (5,)."
    )
    a1 = xi1[0:2]
    b1 = xi1[2:4]
    omega1 = xi1[4]
    a2 = xi2[0:2]
    b2 = xi2[2:4]
    omega2 = xi2[4]

    bracket_a = omega1 * a2 - omega2 * a1
    bracket_b = omega1 * b2 - omega2 * b1
    bracket_omega = omega1 * omega2
    return np.array(
        [bracket_a[0], bracket_a[1], bracket_b[0], bracket_b[1], bracket_omega]
    )


def se22_propagation(xi, X, dt):
    assert xi.shape == (5,), "Input vector must be of shape (5,)."
    assert X.shape == (4, 4), "Input matrix must be 4x4."

    R_old, v_old, p_old = get_components_from_se22(X)

    # Extract components from xi
    a = xi[0:2]  # body-frame acceleration
    omega = xi[4]  # angular velocity

    # Update orientation (body frame rotates)
    R_new = R_old @ so2_exp(omega * dt)

    # Update body-frame velocity (stays in body frame, no rotation needed)
    v_new = v_old + a * dt

    # Update world position (integrate world-frame velocity)
    p_new = p_old + R_old @ v_old * dt

    # Construct output matrix
    out = np.eye(4)
    out[0:2, 0:2] = R_new
    out[0:2, 2] = v_new
    out[0:2, 3] = p_new

    return out


def two_d_ins_simulation(ins_trajectory, X_init, dt=0.01):
    X = X_init
    trajectory_output = {}
    for segment, data in ins_trajectory.items():
        a_x = data["a_x"]
        a_y = data["a_y"]
        omega = data["omega"]
        time = data["time"]
        print(
            f"{segment}: a_x = {a_x} m/s^2, a_y = {a_y} m/s^2, Omega = {omega} rad/s, Time = {time} s"
        )
        trajectory_output[segment] = [X]  # Store initial pose for the segment
        # Update pose
        num_time_steps = int(time / dt)
        for _ in range(num_time_steps):
            xi = np.array(
                [a_x, a_y, 0.0, 0.0, omega]
            )  # b components are zero; position updates via coupling
            X = se22_propagation(xi=xi, X=X, dt=dt)
            trajectory_output[segment].append(X)

    return trajectory_output


# Plotting functions
def plot_turning_radius(
    velocity_low, velocity_high, phi_degrees, filename="turning_radius.png"
):
    phi = np.radians(phi_degrees)  # Convert phi from degrees to radians
    velocity_range = np.linspace(velocity_low, velocity_high, 100)
    turning_radii = [get_turning_radius(v, phi) for v in velocity_range]
    plt.plot(velocity_range, turning_radii)
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Turning Radius (m)")
    plt.title(f"Turning Radius vs Velocity (phi = {phi_degrees} degrees)")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    plt.close()


def plot_trajectory(trajectory, trajectory_name="Trajectory", trajectory_type="SE2"):
    assert trajectory_type in ["SE2", "SE22"], (
        "trajectory_type must be either 'SE2' or 'SE22'."
    )

    plt.figure(figsize=(10, 5))
    for segment, poses in trajectory.items():
        if trajectory_type == "SE2":
            positions = np.array([get_position_from_SE2(pose) for pose in poses])
            orientations = np.array([get_orientation_from_SE2(pose) for pose in poses])
        else:  # SE22
            positions = np.array(
                [get_components_from_se22(pose)[2] for pose in poses]
            )  # Extract p component for position
            orientations = np.array(
                [so2_log(get_components_from_se22(pose)[0]) for pose in poses]
            )  # Extract R component for orientation and compute angle using so2_log

        # Plot the positions and orientations as arrows
        plt.plot(positions[:, 0], positions[:, 1], label=segment)
        heading_intervals = 100  # Plot heading every 10 poses
        plt.quiver(
            positions[::heading_intervals, 0],
            positions[::heading_intervals, 1],
            np.cos(orientations[::heading_intervals]),
            np.sin(orientations[::heading_intervals]),
            scale=100,
        )

    plt.title(f"Race Track Trajectory: {trajectory_name}")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.savefig(f"{trajectory_name}.png")
    plt.show()


def plot_velocity(trajectory, trajectory_name="Velocity_Trajectory", dt=0.01):
    plt.figure(figsize=(10, 5))
    t = 0.0
    for segment, poses in trajectory.items():
        velocities = []
        for i in range(1, len(poses)):
            R_prev, v_prev, p_prev = get_components_from_se22(poses[i - 1])
            R_curr, v_curr, p_curr = get_components_from_se22(poses[i])
            velocity = np.linalg.norm(v_curr)  # Compute velocity magnitude
            velocities.append(velocity)

        time_steps = np.arange(len(velocities)) * dt + t  # Add segment start time
        t += len(velocities) * dt  # Update time for next segment
        plt.plot(time_steps, velocities, label=segment)

    plt.title(f"Velocity Trajectory: {trajectory_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid()
    plt.savefig(f"{trajectory_name}.png")
    plt.show()


def plot_heading(trajectory, trajectory_name="Heading_Trajectory", dt=0.01):
    plt.figure(figsize=(10, 5))
    t = 0.0
    for segment, poses in trajectory.items():
        headings = []
        for pose in poses:
            R, v, p = get_components_from_se22(pose)
            heading = so2_log(R)  # Compute heading angle from rotation matrix
            headings.append(heading)

        time_steps = np.arange(len(headings)) * dt + t  # Add segment start time
        t += len(headings) * dt  # Update time for next segment
        plt.plot(time_steps, np.degrees(headings), label=segment)  # Convert to degrees

    plt.title(f"Heading Trajectory: {trajectory_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (degrees)")
    plt.legend()
    plt.grid()
    plt.savefig(f"{trajectory_name}.png")
    plt.show()


###### Test cases


def generate_random_se22(num_samples=5):
    se22_matrices = []
    for _ in range(num_samples):
        omega = np.random.uniform(-np.pi, np.pi)
        a = np.random.uniform(-10, 10, size=2)
        b = np.random.uniform(-10, 10, size=2)
        xi = np.array([a[0], a[1], b[0], b[1], omega])
        se22_matrix = se22_exp(xi)
        se22_matrices.append(se22_matrix)
    return se22_matrices


def test_se22_exp_log(se22_matrix):
    xi_log = se22_log(se22_matrix)
    se22_recovered = se22_exp(xi_log)
    assert np.allclose(se22_matrix, se22_recovered), (
        "se22_exp and se22_log test failed!"
    )


def test_se22_adjointX1X2_equals_adjX1_adjX2(X1, X2):
    Ad_X1 = se22_adjoint(X1)
    Ad_X2 = se22_adjoint(X2)
    assert np.allclose(Ad_X1 @ Ad_X2, se22_adjoint(X1 @ X2)), (
        "se22_adjoint test failed!"
    )


def test_small_adjointX1X2_equals_lie_bracketX1X2(xi1, xi2):
    ad_xi1 = se22_small_adjoint(xi1)
    assert np.allclose(ad_xi1 @ xi2, lie_bracket(xi1, xi2)), "Lie bracket test failed!"


def test_group_affine_property(X1, X2, omega, a, wind: None | np.ndarray = None):
    assert X1.shape == (4, 4) and X2.shape == (4, 4), "Input matrices must be 4x4."
    assert isinstance(omega, (int, float)), "Omega must be a scalar."
    assert a.shape == (2,), "a must be a 2D vector."
    if wind is not None:
        assert wind.shape == (2,), "wind must be a 2D vector."

    twist_l = np.zeros((4, 4))
    twist_l[0:2, 0:2] = so2_wedge(omega)
    twist_l[0:2, 2] = a
    twist_l[2, 3] = 1

    twist_r = np.zeros((4, 4))
    twist_r[2, 3] = -1
    if wind is not None:
        twist_r[0:2, 2] = wind

    f_X1X2 = twist_r @ X1 @ X2 + X1 @ X2 @ twist_l
    f_X1 = twist_r @ X1 + X1 @ twist_l
    f_X2 = twist_r @ X2 + X2 @ twist_l
    f_I = twist_r + twist_l

    left_side = f_X1X2
    right_side = f_X1 @ X2 + X1 @ f_X2 - X1 @ f_I @ X2

    assert np.allclose(left_side, right_side), "Group affine property test failed!"


def run_question_1():
    plot_turning_radius(velocity_low=10, velocity_high=40, phi_degrees=30)

    trajectory = propagate_trajectory(RACE_TRACK_TRAJECTORY, np.eye(3), dt=0.01)
    plot_trajectory(
        trajectory, trajectory_name="Race_Track_Trajectory", trajectory_type="SE2"
    )

    trajectory = propagate_trajectory(TEST_TRACK_TRAJECTORY_1, np.eye(3), dt=0.01)
    plot_trajectory(trajectory, trajectory_name="Test_Track_Trajectory_1")

    trajectory = propagate_trajectory(TEST_TRACK_TRAJECTORY_2, np.eye(3), dt=0.01)
    plot_trajectory(trajectory, trajectory_name="Test_Track_Trajectory_2")


def run_question_3():
    se22_matrices = generate_random_se22(num_samples=5)
    for se22_matrix in se22_matrices:
        test_se22_exp_log(se22_matrix)
    print("All se22_exp and se22_log tests passed!")

    for i in range(len(se22_matrices)):
        for j in range(len(se22_matrices)):
            test_se22_adjointX1X2_equals_adjX1_adjX2(se22_matrices[i], se22_matrices[j])
    print("All se22_adjoint tests passed!")

    for i in range(len(se22_matrices)):
        for j in range(len(se22_matrices)):
            xi1 = se22_log(se22_matrices[i])
            xi2 = se22_log(se22_matrices[j])
            test_small_adjointX1X2_equals_lie_bracketX1X2(xi1, xi2)
    print("All Lie bracket tests passed!")


def run_question_4():
    se22_matrices = generate_random_se22(num_samples=5)
    wind = np.array([3, -2])
    for i in range(len(se22_matrices)):
        for j in range(len(se22_matrices)):
            omega = np.random.uniform(-np.pi, np.pi)
            a = np.random.uniform(-10, 10, size=2)
            test_group_affine_property(se22_matrices[i], se22_matrices[j], omega, a)
            test_group_affine_property(
                se22_matrices[i], se22_matrices[j], omega, a, wind
            )
    print("All group affine property tests passed!")

    trajectory_output = two_d_ins_simulation(TWO_D_INS_TRAJECTORY, np.eye(4), dt=0.01)
    print(trajectory_output)
    plot_trajectory(
        trajectory_output,
        trajectory_name="Two_D_INS_Trajectory",
        trajectory_type="SE22",
    )

    plot_velocity(trajectory_output, trajectory_name="Two_D_INS_Velocity_Trajectory")
    plot_heading(trajectory_output, trajectory_name="Two_D_INS_Heading_Trajectory")

    spiral_trajectory_output = two_d_ins_simulation(
        SPIRAL_TRAJECTORY, np.eye(4), dt=0.01
    )
    plot_trajectory(
        spiral_trajectory_output,
        trajectory_name="Spiral_Trajectory",
        trajectory_type="SE22",
    )


if __name__ == "__main__":
    run_question_1()
