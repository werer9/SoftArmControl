from Chatbot.chatbot import *
import tkinter as tk
from tkinter import ttk
import pyaudio
import threading
from yolov5.depth import *
import queue
from intent import intent
class App(tk.Frame):
    FORMAT = pyaudio.paInt16
    CHUNK = 1024
    RATE = 16000
    CHANNELS = 1

    def __init__(self, master=None, chatbot: Chatbot = Chatbot("robotarm-315611", "123", "en")):
        """
        Initialise the chatbot window
        :param master: Tkinter session
        :param chatbot: chatbot instance
        """
        # Create the window and start a new thread to listen to audio
        super().__init__(master)
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.thread = threading.Thread(target=self.record_audio, args=[self.frames])
      
        self.new_intent = intent()
        self.yolo_thread = threading.Thread(target=depth, args=[self.new_intent])
        self.yolo_thread.start()
        self.record = False

        self.master = master
        self.chatbot = chatbot
        self.pack()

        # Create the main window frame
        self.main_frame = ttk.Frame(self.master, padding="3 3 12 12")
        self.top_frame = ttk.Frame(self.main_frame)
        self.label_display = ttk.Label(self.top_frame, text="Chatbot Text Output")
        self.text_display = tk.Text(self.top_frame, width=40, height=20)
        self.scroll_bar = ttk.Scrollbar(self.top_frame, orient=tk.VERTICAL, command=self.text_display.yview())

        self.bottom_frame = ttk.Frame(self.main_frame)
        self.input_label = ttk.Label(self.bottom_frame, text="Input: ")
        self.text_input = ttk.Entry(self.bottom_frame, width=40)
        self.text_button = ttk.Button(self.bottom_frame, text="Send", command=self.sendText)
        self.mic_button = ttk.Button(self.bottom_frame, text="Microphone")
        self.mic_button.bind('<ButtonPress-1>', self.start_recording)
        self.mic_button.bind('<ButtonRelease-1>', self.stop_recording)

        self.create_widgets()

    def create_widgets(self):
        """
        Create widgets and position them
        :return:
        """
        self.master.title("Chatbot Controller")
        self.main_frame.pack()

        self.text_display.configure(yscrollcommand=self.scroll_bar.set)
        self.text_display['state'] = 'disabled'

        self.label_display.grid(column=0, row=0, columnspan=4)
        self.text_display.grid(column=0, row=1, columnspan=3)
        self.scroll_bar.grid(column=3, row=1, sticky=(tk.N, tk.S))

        self.input_label.grid(column=0, row=0, sticky=tk.W)
        self.text_input.grid(column=0, row=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.text_button.grid(column=0, row=2, sticky=(tk.W, tk.E), padx=(0, 5))
        self.mic_button.grid(column=1, row=2, sticky=(tk.W, tk.E), padx=(5, 0))

        self.top_frame.grid(column=0, row=0)
        self.bottom_frame.grid(column=0, row=1)

    def sendText(self):
        """
        Send text input to the chatbot and display the input and response in the window
        :return:
        """
        # Get string from input box
        inputData = self.text_input.get()
        # Write input text to the text display
        self.text_display['state'] = 'normal'
        self.text_input.delete(0, 'end')
        self.text_display.insert('end', "User: " + inputData)
        self.text_display['state'] = 'disabled'
        
        # Get the text response from google and write to text display
        [self.outputData, intent, _] = self.chatbot.get_user_intent_text([inputData])   
        self.new_intent.intent = intent
        self.text_display['state'] = 'normal'
        self.text_display.insert('end', "\nChatbot: " + self.outputData + "\n")
        self.text_display['state'] = 'disabled'

    def start_recording(self, event):
        """
        Start new thread to listen to microphone input
        :param event:
        :return:
        """
        self.thread.start()
        return

    def stop_recording(self, event):
        """
        Stop listening to microphone input
        :param event:
        :return:
        """
        # Tell recording thread to stop, wait for it to stop and then store audio input data
        self.record = False
        self.thread.join()
        data = b''.join(self.frames)
        self.frames = []
        # Send audio input data to google cloud and get the response
        [inputData, outputData, intent, _] = self.chatbot.get_user_intent_audio(data)
        # Write the audio input and output transcript to the text display
        self.text_display['state'] = 'normal'
        self.text_input.delete(0, 'end')
        self.text_display.insert('end', "User: " + inputData)
        self.text_display.insert('end', "\nChatbot: " + outputData + "\n")
        self.new_intent.intent = intent
        print("intent",intent)
        self.thread = threading.Thread(target=self.record_audio, args=[self.frames])
        
    def record_audio(self, frames: list):
        """
        Record microphone input
        :param frames: List of buffers
        :return:
        """
        # Tell thread to loop and open input stream
        self.record = True
        stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True,
                             frames_per_buffer=self.CHUNK)
        # Keep looping (reading audio input) until stop recording function is called in main thread
        while self.record:
            data = stream.read(self.CHUNK)
            frames.append(data)
        stream.stop_stream()
        return


if __name__ == '__main__':
    # Create new tkinter app and run it
    root = tk.Tk()
    app = App(root)
    record = False
    app.mainloop()
    

