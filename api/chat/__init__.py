import azure.functions as func
from openai import OpenAI
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
        
        # ============================================
        # EDIT GROK PROMPT HERE
        # ============================================
        xai_client = OpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        
        chat_response = xai_client.chat.completions.create(
            model="grok-4-1-fast",
            messages=[
                {
                    "role": "system",
                    # ⬇️ EDIT THIS LINE
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
        
        # ============================================
        # CALL FAL.AI DIRECTLY FOR CUSTOM PARAMETERS
        # ============================================
        fal_api_url = "https://queue.fal.run/fal-ai/nari-labs/Dia-1.6B"
        
        headers = {
            "Authorization": f"Key {os.environ.get('HF_TOKEN')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": response_text,
            # ⬇️ EDIT THESE PARAMETERS
            "max_new_tokens": 3072,
            "cfg_scale": 1.9,
            "temperature": 1.6,
            "top_p": 0.9,
            "cfg_filter_top_k": 45,
            "speed_factor": 0.7  # ⬅️ SLOWER (0.7 = 70% speed, lower = slower)
        }
        
        logging.info(f'Calling fal.ai with speed_factor: {payload["speed_factor"]}')
        
        response = requests.post(fal_api_url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Get the audio URL from response
        result = response.json()
        audio_url = result.get("audio_url") or result.get("url")
        
        if audio_url:
            # Download the audio
            audio_response = requests.get(audio_url)
            audio_response.raise_for_status()
            audio_bytes = audio_response.content
        else:
            # Audio might be directly in response
            audio_bytes = result.get("audio", b"")
        
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