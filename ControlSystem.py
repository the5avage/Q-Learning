import collections
import torch

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PT1:
    def __init__(self, K, T, delta_t, delay = 0, device=default_device):
        """
        Initializes the PT1 system with delay.

        :param K: Gain factor of the PT1 system.
        :param T: Time constant of the PT1 system.
        :param delta_t: Time step for discretization.
        :param delay: Time delay (dead time) in seconds.
        """
        self.K = K
        self.T = T
        self.delta_t = delta_t
        self.delay_steps = int(delay / delta_t) + 1
        self.y_prev = torch.tensor([0.0], device=device)
        self.input_history = collections.deque([torch.tensor([0.0], device=device)] * self.delay_steps, maxlen=self.delay_steps)

    def calculate(self, u):
        """
        Calculates the output value y for the next time step.

        :param u: Current input/control value.
        :return: Calculated output value y for the next time step.
        """

        delayed_u = 0
        # Shift the input into the history (implements delay)
        self.input_history.append(u)
        delayed_u = self.input_history[0]
        # PT1 difference equation
        y_next = self.y_prev + (self.delta_t / self.T) * (-self.y_prev + self.K * delayed_u)

        self.y_prev = y_next
        return y_next

# PID Controller with windup compensation
class PID:
    def __init__(self, Kp, Ki, Kd, delta_t, min, max):
        """
        Initializes the PID controller.

        :param Kp: Proportional gain.
        :param Ki: Integral gain.
        :param Kd: Derivative gain.
        :param delta_t: Time step for discretization.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.delta_t = delta_t
        self.integral = 0
        self.prev_error = 0
        self.min = min
        self.max = max

    def calculate(self, setpoint, measured_value):
        """
        Calculates the control output u for the next time step.

        :param setpoint: Desired setpoint value.
        :param measured_value: Current measured value (feedback).
        :return: Calculated control output u.
        """
        error = setpoint - measured_value

        P = self.Kp * error

        new_integral = error * self.delta_t + self.integral
        I = self.Ki * new_integral

        # when the measurement of the process variable is noisy
        # more than one previous error must be used to calculate the derivative
        derivative = (error - self.prev_error) / self.delta_t
        D = self.Kd * derivative

        u = P + I + D

        self.prev_error = error

        if u > self.max:
            u = self.max
        elif u < self.min:
            u = self.min
        else:
            self.integral = new_integral

        return u
