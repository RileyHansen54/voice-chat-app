import azure.functions as func
from openai import OpenAI
import fal_client
import requests
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
                    "content": "You are a helpful, friendly AI assistant. Keep responses concise and conversational."
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ],
            temperature=0.7,
            max_tokens=150,
        )
        
        response_text = chat_response.choices[0].message.content
        logging.info(f'AI response: {response_text}')
        
        # Set FAL_KEY for the client
        os.environ["FAL_KEY"] = os.environ.get("FAL_KEY")
        
        # Use fal.ai with speed control
        result = fal_client.subscribe(
            "fal-ai/nari-labs/Dia-1.6B",
            arguments={
                "text": response_text,
                "speed_factor": 0.7,  # ⬅️ EDIT THIS: 0.5-1.0 (lower = slower)
                "max_new_tokens": 3072,
                "cfg_scale": 1.9,
                "temperature": 1.6,
                "top_p": 0.9,
                "cfg_filter_top_k": 45
            },
            with_logs=True
        )
        
        # Download audio from URL
        audio_url = result.get("audio_url", {}).get("url")
        logging.info(f'Audio URL: {audio_url}')
        
        audio_response = requests.get(audio_url)
        audio_response.raise_for_status()
        audio_bytes = audio_response.content
        
        # Return audio as response
        return func.HttpResponse(
            audio_bytes,
            mimetype="audio/wav",
            status_code=200
        )
        
    except Exception as e:
        logging.error(f'Error: {str(e)}')
        import traceback
        tb = traceback.format_exc()
        logging.error(f'Full traceback: {tb}')
        return func.HttpResponse(
            f"Error: {str(e)}\n\nTraceback:\n{tb}",
            status_code=500
        )