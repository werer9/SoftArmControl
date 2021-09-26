import os

from google.cloud import dialogflow
import pyaudio


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
    CHUNK = 4096
    RATE = 16000

    def __init__(self, project_id, session_id, language_code):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
            "C:\\Users\\Caelan\\PycharmProjects\\SoftArmControl\\Chatbot\\robotarm-315611-becfd47d7bf3.json"
        self.project_id = project_id
        self.session_id = session_id
        self.language_code = language_code
        self.encoding = dialogflow.OutputAudioEncoding.OUTPUT_AUDIO_ENCODING_LINEAR_16
        self.session_client = dialogflow.SessionsClient()
        self.session = self.session_client.session_path(self.project_id, self.session_id)
        self.p = pyaudio.PyAudio()
        self.frames = []

    def get_requests(self):
        audio_config = dialogflow.InputAudioConfig(
            audio_encoding=self.encoding,
            language_code=self.language_code,
            sample_rate_hertz=self.RATE,
        )

        output_audio_config = dialogflow.OutputAudioConfig(
            audio_encoding=self.encoding,
            sample_rate_hertz=self.RATE
        )

        query_input = dialogflow.QueryInput({"audio_config": audio_config})

        # The first request contains the configuration.
        yield dialogflow.StreamingDetectIntentRequest(
            {"session": self.session, "query_input": query_input, "output_audio_config": output_audio_config}
        )

        yield dialogflow.StreamingDetectIntentRequest({"input_audio": self.frames})

    def get_user_intent_audio(self, frames):
        self.frames = frames
        responses = self.session_client.streaming_detect_intent(self.get_requests())
        query_result = None
        for response in responses:
            if response.query_result.fulfillment_text is not "":
                query_result = response.query_result

        self.play_response(response)

        return [query_result.query_text, query_result.fulfillment_text, query_result.intent.display_name,
                query_result.intent_detection_confidence]

    def get_user_intent_text(self, texts):
        for text in texts:
            text_input = dialogflow.TextInput(text=text, language_code=self.language_code)

            query_input = dialogflow.QueryInput({"text": text_input})

            output_audio_config = dialogflow.OutputAudioConfig(
                audio_encoding=self.encoding,
                sample_rate_hertz=self.RATE
            )

            request = dialogflow.DetectIntentRequest({"session": self.session, "query_input": query_input,
                                                      "output_audio_config": output_audio_config})

            response = self.session_client.detect_intent(
                request=request
            )

            self.play_response(response)

            return [response.query_result.fulfillment_text, response.query_result.intent.display_name,
                    response.query_result.intent_detection_confidence]

    def play_response(self, response):
        stream = self.p.open(format=pyaudio.paInt16,
                             channels=1,
                             rate=self.RATE,
                             output=True)

        stream.write(response.output_audio)
