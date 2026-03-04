import numpy as np
from aae590_ps01 import so2_wedge, so2_vee, so2_exp, so2_log, generate_random_theta_R

def se2_compose(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    assert X1.shape == (3, 3) and X2.shape == (3, 3), "Input matrices must be 3x3."
    R1 = X1[0:2, 0:2]
    t1 = X1[0:2, 2]
    R2 = X2[0:2, 0:2]
    t2 = X2[0:2, 2]

    out_R = R1 @ R2
    out_t = R1 @ t2 + t1
    out = np.eye(3)
    out[0:2, 0:2] = out_R
    out[0:2, 2] = out_t
    
    return out

def se2_inverse(X: np.ndarray) -> np.ndarray:
    assert X.shape == (3, 3), "Input matrix must be 3x3."
    R = X[0:2, 0:2]
    t = X[0:2, 2]

    R_inv = R.T
    t_inv = -R_inv @ t
    X_inv = np.eye(3)
    X_inv[0:2, 0:2] = R_inv
    X_inv[0:2, 2] = t_inv
    
    return X_inv

def se2_wedge(xi: np.ndarray) -> np.ndarray:
    """xi is a 3D vector [v_x, v_y, theta]"""
    assert xi.shape == (3,), "Input vector must be of shape (3,)."
    theta = xi[2]
    theta_wedge = so2_wedge(theta)
    xi_wedge = np.zeros((3, 3))
    xi_wedge[0:2, 0:2] = theta_wedge
    xi_wedge[0:2, 2] = xi[0:2]
    
    return xi_wedge

def se2_vee(xi_wedge: np.ndarray) -> np.ndarray:
    assert xi_wedge.shape == (3, 3), "Input matrix must be 3x3."
    v = xi_wedge[0:2, 2]
    theta = so2_vee(xi_wedge[0:2, 0:2])
    xi = np.array([v[0], v[1], theta])
    
    return xi

def se2_exp(xi: np.ndarray) -> np.ndarray:
    assert xi.shape == (3,), "Input vector must be of shape (3,)."
    theta = xi[2]
    v = xi[0:2]
    
    # Use the first 5 terms of the Taylor series expansion for small angles
    # sin x / x
    sinx_over_x = 1 - theta**2 / 6 + theta**4 / 120 - theta**6 / 5040 + theta**8 / 362880
    # (1 - cos x) / x
    one_minus_cosx_over_x = theta / 2 - theta**3 / 24 + theta**5 / 720 - theta**7 / 40320 + theta**9 / 3628800

    if np.isclose(theta, 0):
        V = np.eye(2) * sinx_over_x + np.array([[0, -1], [1, 0]]) * one_minus_cosx_over_x
    else:
        V = np.eye(2) * np.sin(theta) / theta + np.array([[0, -1], [1, 0]]) * (1 - np.cos(theta)) / theta

    R = so2_exp(theta)
    t = V @ v
    X = np.eye(3)
    X[0:2, 0:2] = R
    X[0:2, 2] = t

    return X
 
def se2_log(X: np.ndarray) -> np.ndarray:
    assert X.shape == (3, 3), "Input matrix must be 3x3."
    R = X[0:2, 0:2]
    t = X[0:2, 2]
    
    theta = so2_log(R)
    
    # Use the first 5 terms of the Taylor series expansion for small angles
    # (x / 2) cot(x / 2) = 1 - x^2 / 12 - x^4 / 720 - x^6 / 30240 - x^8 / 1209600
    x_over_2_cot_x_over_2 = 1 - theta**2 / 12 - theta**4 / 720 - theta**6 / 30240 - theta**8 / 1209600

    if np.isclose(theta, 0):
        V_inv = np.eye(2) * x_over_2_cot_x_over_2 + np.array([[0, 1], [-1, 0]]) * (theta / 2)
    else:
        V_inv = np.eye(2) * (theta / 2) * (1 / np.tan(theta / 2)) + np.array([[0, 1], [-1, 0]]) * (theta / 2)

    v = V_inv @ t
    xi = np.array([v[0], v[1], theta])
    
    return xi

def se2_adjoint(X: np.ndarray) -> np.ndarray:
    assert X.shape == (3, 3), "Input matrix must be 3x3."
    R = X[0:2, 0:2]
    t = X[0:2, 2]

    adj_X = np.eye(3)
    adj_X[0:2, 0:2] = R
    adj_X[0:2, 2] = [t[1], -t[0]]

    return adj_X

def euler_integration(xi: np.ndarray, X: np.ndarray, dt: float) -> np.ndarray:
    assert xi.shape == (3,), "Input vector must be of shape (3,)."
    assert X.shape == (3, 3), "Input matrix must be 3x3."
    out = X @ (np.eye(3) + dt * se2_wedge(xi))
    return out

def lie_group_integration(xi: np.ndarray, X: np.ndarray, dt: float) -> np.ndarray:
    assert xi.shape == (3,), "Input vector must be of shape (3,)."
    assert X.shape == (3, 3), "Input matrix must be 3x3."
    out = X @ se2_exp(xi * dt)
    return out

def simulate_motion(xi, X0, t_final, dt):
    num_steps = int(t_final / dt)
    X_euler = np.zeros((num_steps, 3, 3))
    X_lie_group = np.zeros((num_steps, 3, 3))

    X_euler[0] = X0
    X_lie_group[0] = X0

    for i in range(1, num_steps):
        X_euler[i] = euler_integration(xi, X_euler[i-1], dt)
        X_lie_group[i] = lie_group_integration(xi, X_lie_group[i-1], dt)

    return X_euler, X_lie_group

def plot_trajectories(X_euler, X_lie_group):
    import matplotlib.pyplot as plt

    euler_positions = X_euler[:, 0:2, 2]
    lie_group_positions = X_lie_group[:, 0:2, 2]

    plt.figure(figsize=(10, 5))
    plt.plot(euler_positions[:, 0], euler_positions[:, 1], label='Euler Integration', marker='o')
    plt.plot(lie_group_positions[:, 0], lie_group_positions[:, 1], label='Lie Group Integration', marker='x')
    plt.title('Trajectory Comparison')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.savefig('trajectory_comparison.png')
    plt.show()
    plt.close()

def plot_frobenius_error(X_euler, X_lie_group):
    import matplotlib.pyplot as plt

    num_steps = X_euler.shape[0]
    lie_group_errors = np.zeros(num_steps)
    euler_errors = np.zeros(num_steps)

    for i in range(num_steps):
        R_lie_group = X_lie_group[i][0:2, 0:2]
        R_euler = X_euler[i][0:2, 0:2]
        lie_group_errors[i] = np.linalg.norm(R_lie_group.T @ R_lie_group - np.eye(2), ord='fro')
        euler_errors[i] = np.linalg.norm(R_euler.T @ R_euler - np.eye(2), ord='fro')

    plt.figure(figsize=(10, 5))
    plt.plot(euler_errors, label='Euler Error', marker='o')
    plt.plot(lie_group_errors, label='Lie Group Error', marker='x')
    plt.title('Frobenius Norm Error Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Frobenius Norm Error')
    plt.legend()
    plt.grid()
    plt.savefig('frobenius_error.png')
    plt.show()
    plt.close()

###### Test cases

def generate_random_translation(num_samples=5):
    translations = np.random.uniform(-10, 10, size=(num_samples, 2))
    return translations

def generate_random_se2(num_samples=5):
    X_list = []
    thetas, Rs = generate_random_theta_R(num_samples)
    translations = generate_random_translation(num_samples)

    for i in range(num_samples):
        X = np.eye(3)
        X[0:2, 0:2] = Rs[i]
        X[0:2, 2] = translations[i]
        X_list.append((X))

    return X_list

def test_se2_exp_log(X):
    log_X = se2_log(X)
    X_recovered = se2_exp(log_X)
    assert np.allclose(X, X_recovered), "se2_exp and se2_log test failed!"

def test_adjointX1X2_equals_adjX1_adjX2(X1, X2):
    adj_X1X2 = se2_adjoint(se2_compose(X1, X2))
    adj_X1_adj_X2 = se2_adjoint(X1) @ se2_adjoint(X2)
    assert np.allclose(adj_X1X2, adj_X1_adj_X2), "Adjoint property test failed!"

def test_adjointX_inv_equals_adjX_inv(X):
    adj_X_inv = se2_adjoint(se2_inverse(X))
    adj_X_inv_direct = np.linalg.inv(se2_adjoint(X))
    assert np.allclose(adj_X_inv, adj_X_inv_direct), "Adjoint inverse property test failed!"

def test_exact_integration_equals_lie_group_integration(xi, X, t, dt):
    exact_X = X @ se2_exp(xi * t)
    X_euler, X_lie_group = simulate_motion(xi, X, t, dt)
    print("Exact final pose:\n", exact_X)
    print("Lie group final pose:\n", X_lie_group[-1])
    assert np.allclose(exact_X, X_lie_group[-1], rtol=1e-1, atol=1e-1), "Exact integration and Lie group integration test failed!"

def run_all_tests(num_samples=5):
    Xs = generate_random_se2(num_samples)
    for i in range(num_samples):
        test_se2_exp_log(Xs[i])
        test_adjointX_inv_equals_adjX_inv(Xs[i])
    
    for i in range(num_samples):
        for j in range(num_samples):
            test_adjointX1X2_equals_adjX1_adjX2(Xs[i], Xs[j])
    
    # Test exact integration vs Lie group integration
    xi = np.array([1.0, 0.0, 0.5])  # Example twist (v_x, v_y, omega)
    X0 = np.eye(3)  # Initial pose
    t_final = 20.0  # Total simulation time
    dt = 0.1  # Time step
    test_exact_integration_equals_lie_group_integration(xi, X0, t_final, dt)

    print("All tests passed!")

if __name__ == "__main__":
    num_samples = 100
    run_all_tests(num_samples=num_samples)
    
    # Simulate motion and plot trajectories
    xi = np.array([1.0, 0.0, 0.5])  # Example twist (v_x, v_y, omega)
    X0 = np.eye(3)  # Initial pose
    t_final = 20.0  # Total simulation time
    dt = 0.1  # Time step
    X_euler, X_lie_group = simulate_motion(xi, X0, t_final, dt)
    plot_trajectories(X_euler, X_lie_group)
    plot_frobenius_error(X_euler, X_lie_group)
