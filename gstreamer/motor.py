import time
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
    MIN_PWM = 0.03
    MAX_PWM = 0.1025

    def __post_init__(self):
        self._pwm0 = PWM(0, 0)
        self._pwm0.frequency = 50
        self._pwm0.enable()
        self.pos = 90
        
    @staticmethod
    def _degree_to_pwm(degree):
        return 0.03 + 0.0725 * degree / 180

    def _set_pwm(self, pwm):
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
                factor = -1
                print("pwm tope max")
            elif current_pwm <= MIN_PWM:
                factor = 1
                print("pwm tope min")


    def scan(self):
        if not self.sub_scan:
            self.stop_event = mp.Event()
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
            self.sub_scan.join()


        


m1 = Motor()
m1.scan()
m2 = Motor()
time.sleep(6)
m2.pos=75
m2.scan()
time.sleep(1)
m2.stop_scan()
    