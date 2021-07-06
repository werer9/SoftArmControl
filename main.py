from time import sleep

from chatbot import *
from controller import *

if __name__ == '__main__':
    detect_intent_texts("robotarm-315611", "123", ["Who are you?"], "en")
    # Create controller object, connecting to IP address and port: http://172.22.0.75:5000
    controller = Controller("http://172.22.0.75:5000")
    while True:
        # Change control parameters
        controller.set_green(0.12)
        controller.set_yellow(-0.11)
        controller.set_base(0.16)
        sleep(2)
        controller.set_green(-0.12)
        controller.set_yellow(0.11)
        controller.set_base(-0.16)
        sleep(2)
