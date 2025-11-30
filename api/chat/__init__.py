import azure.functions as func
from openai import OpenAI
import os
import logging

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    try:
        # Log environment variables (without exposing keys)
        logging.info(f'XAI_API_KEY exists: {bool(os.environ.get("XAI_API_KEY"))}')
        logging.info(f'OPENAI_API_KEY exists: {bool(os.environ.get("OPENAI_API_KEY"))}')
        
        # Get request body
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
        
        logging.info('Calling xAI API...')
        chat_response = xai_client.chat.completions.create(
            model="grok-4-1-fast",
            messages=[{"role": "user", "content": user_text}]
        )
        
        response_text = chat_response.choices[0].message.content
        logging.info(f'AI response: {response_text[:100]}...')
        
        # Convert response to speech using OpenAI TTS
        logging.info('Calling OpenAI TTS...')
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        speech_response = openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=response_text
        )
        
        logging.info('Returning audio response')
        # Return audio as response
        return func.HttpResponse(
            speech_response.content,
            mimetype="audio/mpeg",
            status_code=200
        )
        
    except Exception as e:
        logging.error(f'Error: {str(e)}')
        logging.error(f'Error type: {type(e).__name__}')
        import traceback
        logging.error(f'Traceback: {traceback.format_exc()}')
        return func.HttpResponse(
            f"Error: {str(e)}",
            status_code=500
        )