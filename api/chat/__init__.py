import azure.functions as func
from openai import OpenAI
from huggingface_hub import InferenceClient
import os
import logging
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
        user_text = req_body.get('text')

        if not user_text:
            return func.HttpResponse(
                "Please provide 'text' in request body",
                status_code=400
            )

        logging.info(f'User text: {user_text}')

        # Get response from xAI Grok
        xai_client = OpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )

        chat_response = xai_client.chat.completions.create(
            model="grok-4-1-fast",
            messages=[
                {
                    "role": "system",
                    "content": "You are an LLM designed to create dialog for a text to speech model. You create conversational tone as though it were coming straight out of their mouth, and you create it for a loving teacher named lara who will help the user in any capacity to learn. The dialog is human, natural and sweet. Additionally you add ... to create pauses or moments of interest and use parentheses like so to create emotional moments (laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles). Try to answer in a loving short and slow manner if possible."
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ],
            temperature=0.7,
            max_tokens=550,
        )

        response_text = chat_response.choices[0].message.content
        logging.info(f'AI response: {response_text}')

        # Convert to speech using Hugging Face Nari Labs Dia
        # NOTE: InferenceClient doesn't support the advanced parameters through the API
        hf_client = InferenceClient(
            provider="fal-ai",
            api_key=os.environ.get("HF_TOKEN")
        )

        audio_bytes = hf_client.text_to_speech(
            response_text,
            model="nari-labs/Dia-1.6B"
            # No additional parameters - they're not supported via InferenceClient
        )

        # Return audio as response
        return func.HttpResponse(
            audio_bytes,
            mimetype="audio/wav",
            status_code=200
        )

    except Exception as e:
        logging.error(f'Error: {str(e)}')
        import traceback
        logging.error(traceback.format_exc())
        return func.HttpResponse(
            f"Error: {str(e)}",
            status_code=500
        )