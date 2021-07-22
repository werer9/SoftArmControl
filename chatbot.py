from google.cloud import dialogflow


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
        self.project_id = project_id
        self.session_id = session_id
        self.language_code = language_code
        self.session_client = dialogflow.SessionsClient.from_service_account_json("robotarm-315611-becfd47d7bf3.json")
        self.session = self.session_client.session_path(self.project_id, self.session_id)

    def get_user_intent(self, texts):
        for text in texts:
            text_input = dialogflow.TextInput(text=text, language_code=self.language_code)

            query_input = dialogflow.QueryInput({"text": text_input})

            request = dialogflow.DetectIntentRequest({"session": self.session, "query_input": query_input})

            response = self.session_client.detect_intent(
                request=request
            )

            print("{}\n".format(response.query_result.fulfillment_text))

            return [response.query_result.intent.display_name, response.query_result.intent_detection_confidence]

