from dataclasses import dataclass
from periphery import PWM
import multiprocessing as mp


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

@dataclass
class Motor(metaclass=SingletonMeta):
    _current_pos: int = 90
    _current_pwm: float = 0.0
    _scan: bool = False
    _pwm0: object = None
    sub_scan: object = None
    MIN_DEGREE = 45
    MAX_DEGREE = 135


    def __post_init__(self):
        self.MIN_PWM = Motor._degrees_to_pwm(self.MIN_DEGREE)
        self.MAX_PWM = Motor._degrees_to_pwm(self.MAX_DEGREE)
        self._pwm0 = PWM(0, 0)
        self._pwm0.frequency = 50
        self._pwm0.enable()
        self.pos = 90

        
    @staticmethod
    def _degree_to_pwm(degree):
        return 0.03 + 0.0725 * degree / 180

    def _set_pwm(self, pwm):
        pwm = min(self.MAX_PWM, pwm)
        pwm = max(self.MIN_PWM, pwm)
        self._pwm0.duty_cycle = pwm
        if self.sub_scan:
            self.stop_scan()
        self._current_pwm = pwm
        print("PWM = ", pwm)


    @property
    def pos(self):
        return self._current_pos

    @pos.setter
    def pos(self, degree):
        degree = min(self.MAX_DEGREE, degree)
        degree = max(self.MIN_DEGREE, degree)
        self._set_pwm(self._degree_to_pwm(degree))
        self._current_pos = degree
        print("Set to ", self.pos)

    def rotate(self, value):
        print ("Rotate", value)
        self.pos = self._current_pos + value

    @staticmethod
    def _scan(pwm0,
              stop_event,
              current_pwm,
              MIN_PWM,
              MAX_PWM):
        print("pwm0", pwm0)
        print("Escaneando", end="", flush=True)
        factor = 1
        TIME_SCAN = 3
        WAIT_TIME = 0.05
        SPEED = TIME_SCAN / WAIT_TIME
        step = (MAX_PWM - MIN_PWM) / SPEED

        while not stop_event.wait(WAIT_TIME):
            current_pwm += factor * step
            print(".", end="", flush=True)

            if current_pwm >= MAX_PWM:
                current_pwm = MAX_PWM
                factor = -1
                print("pwm tope max")
            elif current_pwm <= MIN_PWM:
                current_pwm = MIN_PWM
                factor = 1
                print("pwm tope min")
            pwm0.duty_cycle = current_pwm


    def scan(self):
        if not self.sub_scan:
            self.stop_event = mp.Event()
            print("Main pwm0", self._pwm0)
            self.sub_scan = mp.Process(
                target=Motor._scan, 
                args=(self._pwm0,
                      self.stop_event,
                      self._current_pwm,
                      self.MIN_PWM,
                      self.MAX_PWM)
            )
            self.sub_scan.start()

    def stop_scan(self):
        if self.sub_scan:
            self.stop_event.set()
            self.sub_scan.terminate()
            self.sub_scan.join()
            self.sub_scan = None

if __name__ == "__main__":
    motor = Motor()
    command = input("Command: ")
    while command != "exit":
        if command == "scan":
            motor.scan()
        elif command == "stop":
            motor.stop_scan()
        elif command == "rotate":
            value = int(input("Value: "))
            motor.rotate(value)
        elif command == "pos":
            print("Current pos:", motor.pos)
        command = input("Command: ")
    