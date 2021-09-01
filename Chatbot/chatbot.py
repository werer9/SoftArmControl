import os

from google.cloud import dialogflow
import simpleaudio as sa


def detect_intent_texts(project_id, session_id, texts, language_code):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""

    session_client = dialogflow.SessionsClient.from_service_account_json("robotarm-315611-becfd47d7bf3.json")

    session = session_client.session_path(project_id, session_id)
    print("Session path: {}\n".format(session))

    for text in texts:
        text_input = dialogflow.TextInput(text=text, language_code=language_code)

        query_input = dialogflow.QueryInput({"text": text_input})

        request = dialogflow.DetectIntentRequest({"session": session, "query_input": query_input})

        response = session_client.detect_intent(
            request=request
        )

        print("=" * 20)
        print("Query text: {}".format(response.query_result.query_text))
        print(
            "Detected intent: {} (confidence: {})\n".format(
                response.query_result.intent.display_name,
                response.query_result.intent_detection_confidence,
            )
        )
        print("Fulfillment text: {}\n".format(response.query_result.fulfillment_text))


class Chatbot(object):

    def __init__(self, project_id, session_id, language_code):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
            "C:\\Users\\Caelan\\PycharmProjects\\SoftArmControl\\Chatbot\\robotarm-315611-becfd47d7bf3.json"
        self.project_id = project_id
        self.session_id = session_id
        self.language_code = language_code
        self.session_client = dialogflow.SessionsClient()
        self.session = self.session_client.session_path(self.project_id, self.session_id)

    def get_user_intent(self, texts):
        for text in texts:
            text_input = dialogflow.TextInput(text=text, language_code=self.language_code)

            query_input = dialogflow.QueryInput({"text": text_input})

            output_audio_config = dialogflow.OutputAudioConfig(
                audio_encoding=dialogflow.OutputAudioEncoding.OUTPUT_AUDIO_ENCODING_LINEAR_16
            )

            request = dialogflow.DetectIntentRequest({"session": self.session, "query_input": query_input,
                                                      "output_audio_config": output_audio_config})

            response = self.session_client.detect_intent(
                request=request
            )

            print("{}\n".format(response.query_result.fulfillment_text))
            with open("output.wav", "wb") as out:
                out.write(response.output_audio)
                print('Audio content written to file "output.wav"')
            wave_file = sa.WaveObject.from_wave_file("output.wav")
            wave_file = wave_file.play()
            wave_file.wait_done()

            return [response.query_result.fulfillment_text, response.query_result.intent.display_name,
                    response.query_result.intent_detection_confidence]

