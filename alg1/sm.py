import numpy as np

def simpsons_rule(y, dt):
    n = len(y)
    if n % 2 == 0:
        raise ValueError("Number of samples must be odd.")

    integral = 0.0
    for i in range(0, n-2, 2):
        integral += (y[i] + 4*y[i+1] + y[i+2])

    integral *= dt / 3.0

    return integral

def simpsons_rule_with_error_correction(y, dt):
    n = len(y)
    if n % 2 == 0:
        raise ValueError("Number of samples must be odd.")

    I1 = simpsons_rule(y, dt)
    print(I1)
    # I2 = simpsons_rule(y[::2], dt*2)

    # I = I2 + (I2 - I1) / 15

    # return I

# 示例加速度信号
acceleration = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]
sampling_rate = 1.0  # 采样频率，单位为Hz
dt = 1.0 / sampling_rate  # 采样间隔，单位为秒

displacement = simpsons_rule_with_error_correction(acceleration, dt)
print("Displacement:", displacement)
