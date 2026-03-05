import numpy as np


def so2_wedge(theta):
    theta_wedge = np.array([[0, -theta], [theta, 0]])
    return theta_wedge


def so2_vee(omega):
    theta = omega[1, 0]
    return theta


def so2_exp(theta):
    if np.isclose(theta, 0):
        return np.eye(2)
    else:
        omega = so2_wedge(theta)
        R = np.eye(2) * np.cos(theta) + omega * np.sin(theta) / theta
        return R


def so2_log(R):
    theta = np.arctan2(R[1, 0], R[0, 0])
    return theta


### Test cases


def generate_random_theta_R(num_samples=5):
    thetas = np.random.uniform(-np.pi, np.pi, size=num_samples)
    Rs = []
    for theta in thetas:
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        Rs.append(R)
    return thetas, Rs


def test_so2_exp_log(R):
    log_R = so2_log(R)
    R_recovered = so2_exp(log_R)
    assert np.allclose(R, R_recovered), "so2_exp and so2_log test failed!"


def test_so2_log_exp(theta):
    exp_theta = so2_exp(theta)
    log_exp_theta = so2_log(exp_theta)
    assert np.isclose(theta, log_exp_theta), "so2_log and so2_exp test failed!"


def test_R_multiplication(R1, R2):
    theta1 = so2_log(R1)
    theta2 = so2_log(R2)
    R_product = R1 @ R2
    theta_sum = theta1 + theta2
    R_product_recovered = so2_exp(theta_sum)
    assert np.allclose(R_product, R_product_recovered), "R multiplication test failed!"


def run_all_tests(num_samples=5):
    thetas, Rs = generate_random_theta_R(num_samples)
    for i in range(num_samples):
        test_so2_exp_log(Rs[i])
        test_so2_log_exp(thetas[i])
    for i in range(num_samples):
        for j in range(num_samples):
            test_R_multiplication(Rs[i], Rs[j])
    print("All tests passed!")


if __name__ == "__main__":
    num_samples = 100
    run_all_tests(num_samples)
