from Chatbot.chatbot import *
import tkinter as tk
from tkinter import ttk


class App(tk.Frame):
    def __init__(self, master=None, chatbot: Chatbot = Chatbot("robotarm-315611", "123", "en")):
        super().__init__(master)
        self.master = master
        self.chatbot = chatbot
        self.pack()

        self.output = tk.StringVar()
        self.input = tk.StringVar()

        self.main_frame = ttk.Frame(self.master, padding="3 3 12 12")
        self.top_frame = ttk.Frame(self.main_frame)
        self.label_display = ttk.Label(self.top_frame, text="Chatbot Text Output")
        self.text_display = tk.Text(self.top_frame, width=40, height=20)
        self.scroll_bar = ttk.Scrollbar(self.top_frame, orient=tk.VERTICAL, command=self.text_display.yview())

        self.bottom_frame = ttk.Frame(self.main_frame)
        self.input_label = ttk.Label(self.bottom_frame, text="Input: ")
        self.text_input = ttk.Entry(self.bottom_frame, width=40, textvariable=self.input)
        self.text_button = ttk.Button(self.bottom_frame, text="Send", command=self.sendText)
        self.mic_button = ttk.Button(self.bottom_frame, text="Microphone")

        self.create_widgets()

    def create_widgets(self):
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

        # print(self.chatbot.get_user_intent([input("Hello, what do you want to do?\n")]))

    def sendText(self):
        inputData = self.text_input.get()
        self.text_display['state'] = 'normal'
        self.text_input.delete(0, 'end')
        self.text_display.insert('end', "User: " + inputData)
        self.text_display['state'] = 'disabled'
        [outputData, _, _] = self.chatbot.get_user_intent([inputData])
        self.text_display['state'] = 'normal'
        self.text_display.insert('end', "\nChatbot: " + outputData + "\n")
        self.text_display['state'] = 'disabled'


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    app.mainloop()

