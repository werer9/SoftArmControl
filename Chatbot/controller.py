import requests


class Controller:

    def __init__(self, url: str):
        """
        Constructor
        :param url: URL for raspberry pi server
        """
        self.url = url
        self.base = 0.0
        self.green = 0.0
        self.yellow = 0.0
        self.rigid = 1.0
        self.params = {"base": f"{self.base}", "green": f"{self.green}", "yellow": f"{self.yellow}",
                       "rigid": f"{self.rigid}", "misc": "0.0"}

    def verify_data(self):
        """
        Verify control data is within limits, if not, set to zero
        :return:
        """
        if self.base > 1.0:
            self.base = 0.0
        if self.green > 1.0:
            self.green = 0.0
        if self.yellow > 1.0:
            self.yellow = 0.0
        if self.rigid > 1.0:
            self.rigid = 0.0

    def send_packet(self):
        """
        Send data to control server on raspberry pi
        :return:
        """
        self.verify_data()
        self.params = {"base": f"{self.base}", "green": f"{self.green}", "yellow": f"{self.yellow}",
                       "rigid": f"{self.rigid}", "misc": "0.0"}
        requests.get(self.url, params=self.params)

    def stop_movement(self):
        """
        Stop all movement of the arm
        :return:
        """
        self.base = 0.0
        self.green = 0.0
        self.yellow = 0.0
        self.rigid = 1.0
        self.send_packet()

    def set_base(self, speed):
        """
        Control rotation of the base
        :param speed: Speed of rotation <1.0
        :return:
        """
        self.base = speed
        self.send_packet()

    def set_green(self, speed):
        """
        Control rotation of the lowest joint
        :param speed: Speed of rotation <1.0
        :return:
        """
        self.green = speed
        self.send_packet()

    def set_yellow(self, speed):
        """
        Control rotation of the highest joint
        :param speed: Speed of rotation <1.0
        :return:
        """
        self.yellow = speed
        self.send_packet()


