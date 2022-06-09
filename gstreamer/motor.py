import argparse
import rich

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

    min_degree:int = 50
    max_degree:int = 130
    degree_to_move:int = 5
    inverted:bool = True


    def __post_init__(self):
        self.__MIN_PWM = Motor._degree_to_pwm(self.min_degree)
        self.__MAX_PWM = Motor._degree_to_pwm(self.max_degree)
        self.__pwm0 = PWM(0, 0)
        self.__pwm0.frequency = 50
        self.__pwm0.enable()
        self.__range= self.max_degree - self.min_degree
        self.__current_pos: int = 90
        self.__current_pwm: float = 0.0
        self.__scan: bool = False
        self.pos = 90

        
    @staticmethod
    def _degree_to_pwm(degree):
        return 0.03 + 0.0725 * degree / 180

    def _set_pwm(self, pwm):
        pwm = min(self.__MAX_PWM, pwm)
        pwm = max(self.__MIN_PWM, pwm)
        self.__pwm0.duty_cycle = pwm
        if self.__sub_scan:
            self.stop_scan()
        self.__current_pwm = pwm
        #print("PWM = ", pwm)



    @property
    def range(self):
        return self.__range

    @property
    def pos(self):
        return self.__current_pos

    @pos.setter
    def pos(self, degree):
        degree = min(self.max_degree, degree)
        degree = max(self.min_degree, degree)
        self._set_pwm(self._degree_to_pwm(degree))
        self.__current_pos = degree
        print ("Set to ", self.pos)

    def rotate(self, value):
        # if self.inverted:
        #     value = -value
        if abs(value) > self.degree_to_move:
            self.pos = self.__current_pos + value

    @staticmethod
    def __scan(pwm0,
              stop_event,
              current_pwm,
              MIN_PWM,
              MAX_PWM):
        #print("pwm0", pwm0)
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
        if not self.__sub_scan:
            self.stop_event = mp.Event()
            #print("Main pwm0", self._pwm0)
            self.__sub_scan = mp.Process(
                target=Motor.__scan, 
                args=(self.__pwm0,
                      self.stop_event,
                      self.__current_pwm,
                      self.__MIN_PWM,
                      self.__MAX_PWM),
                daemon=True
            )
            self.__sub_scan.start()

    def stop_scan(self):
        if self.__sub_scan:
            self.stop_event.set()
            self.__sub_scan.terminate()
            self.__sub_scan.join()
            self.__sub_scan = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor")
    parser.add_argument("--inverted", action="store_true", default=False)  
    parser.add_argument("--degree_to_move", type=int, default=5)
    args = parser.parse_args()

    motor = Motor(inverted=args.inverted, degree_to_move=args.degree_to_move)
    
    def get_angle(d, screen = (320,320)):
        return int(d * motor.range / screen[0])

    command = input("Command: ")
    while command != "exit":
        if command == "scan":
            motor.scan()
        elif command == "stop":
            motor.stop_scan()
        elif command.startswith("rot"):
            value = int(command.split()[1])
            motor.rotate(value)
        elif command.startswith("pos"):
            value = int(command.split()[1])
            motor.rotate(value-motor.pos)
        elif command.startswith("obj"):
            value = int(command.split()[1])
            motor.pos = get_angle(d=value)
            motor.rotate(value-motor.pos)
        command = input("Command: ")
    