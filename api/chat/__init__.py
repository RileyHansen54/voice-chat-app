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
                    # ⬇️ EDIT THIS LINE - Change Grok's personality/behavior
                    "content": "You are a helpful, friendly online Tutor. Keep responses conversational and upbeat. You laugh, pause, and are reactive to responses but only when the moment is right and in a normal casual way."
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ],
            # Optional Grok parameters:
            temperature=0.7,  # 0.0-2.0 (higher = more creative/random)
            max_tokens=150,   # Maximum response length in tokens
        )
        
        response_text = chat_response.choices[0].message.content
        logging.info(f'AI response: {response_text}')
        
        # ============================================
        # EDIT TTS PARAMETERS HERE
        # ============================================
        hf_client = InferenceClient(
            provider="fal-ai",
            api_key=os.environ.get("HF_TOKEN")
        )
        
        audio_bytes = hf_client.text_to_speech(
            response_text,
            model="nari-labs/Dia-1.6B",
            
            # ⬇️ EDIT THESE PARAMETERS
            max_new_tokens=3072,      # Audio length: 860-3072 (higher = can generate longer audio)
            cfg_scale=1.9,            # Text adherence: 1.0-5.0 (higher = follows text more closely)
            temperature=1.6,          # Randomness: 1.0-2.5 (higher = more varied pronunciation)
            top_p=0.7,                # Vocabulary diversity: 0.7-1.0 (higher = more diverse)
            cfg_filter_top_k=25,      # CFG filtering: 15-100
            speed_factor=0.80        # Speech speed: 0.8-1.0 (1.0 = normal, lower = slower/clearer)
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