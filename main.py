from Chatbot.chatbot import *

if __name__ == '__main__':
    chatbot = Chatbot("robotarm-315611", "123", "en")
    print(chatbot.get_user_intent([input("Hello, what do you want to do?\n")]))

