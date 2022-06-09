import argparse
import rich

from dataclasses import dataclass
from periphery import PWM
import multiprocessing as mp


# @dataclass
# class PWM:
#     canal: int
#     pin: int


#     def enable(self):
#         print("PWM enabled")

#     def __post_init__(self):
#         print("PWM created")
#         self._frequency = 50
#         self._duty_cycle = 0

#     @property
#     def frequency(self):
#         return self._frequency

#     @frequency.setter
#     def frequency(self, value):
#         self._frequency = value

#     @property
#     def duty_cycle(self):
#         return self._duty_cycle

#     @duty_cycle.setter
#     def duty_cycle(self, value):
#         self._duty_cycle = value
#         print("PWM duty cycle set to {}".format(value))

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
        self.__sub_scan: mp.Process = None
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
    def PWM(self):
        return self.__current_pwm

    @property
    def pos(self):
        return self.__current_pos

    @pos.setter
    def pos(self, degree):
        degree = min(self.max_degree, degree)
        degree = max(self.min_degree, degree)
        if self.inverted:
            self._set_pwm(self._degree_to_pwm(180-degree))
        else:
            self._set_pwm(self._degree_to_pwm(degree))
        self.__current_pos = degree
        print ("Set to ", self.pos)

    def rotate(self, value):
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
    parser.add_argument("--min_degree", type=int, default=40)
    parser.add_argument("--max_degree", type=int, default=140)
    args = parser.parse_args()

    motor = Motor(inverted=args.inverted, 
                  degree_to_move=args.degree_to_move,
                  min_degree=args.min_degree,
                  max_degree=args.max_degree)
    
    def get_angle(d, screen = (320,320)):
        return int(d * motor.range / screen[0])+ motor.min_degree

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
            angle = get_angle(d=value)
            motor.pos = angle
            print("distance: ", value, "angle: ", angle, "PWM: ", motor.PWM)
        else:
            print("Commands: scan, stop, rot <value>, pos <value>, obj <value>")
        command = input("Command: ")
    