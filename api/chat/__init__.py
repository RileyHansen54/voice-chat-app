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
        # GROK PROMPT - EDIT HERE
        # ============================================
        xai_client = OpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        
        # You can add a system message to control Grok's behavior
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
            # Optional Grok parameters:
            temperature=0.7,  # 0.0-2.0, higher = more creative
            max_tokens=150,   # Limit response length
            # top_p=0.9,      # Nucleus sampling
        )
        
        response_text = chat_response.choices[0].message.content
        logging.info(f'AI response: {response_text}')
        
        # ============================================
        # TTS PARAMETERS - EDIT HERE
        # ============================================
        hf_client = InferenceClient(
            provider="fal-ai",
            api_key=os.environ.get("HF_TOKEN")
        )
        
        audio_bytes = hf_client.text_to_speech(
            response_text,
            model="nari-labs/Dia-1.6B",
            
            # Audio length (higher = can generate longer audio)
            max_new_tokens=3072,  # 860-3072
            
            # How closely to follow the text (higher = more accurate to prompt)
            cfg_scale=1.9,  # 1.0-5.0
            
            # Randomness (higher = more varied pronunciation)
            temperature=1.6,  # 1.0-2.5
            
            # Vocabulary filtering (higher = more diverse words)
            top_p=0.9,  # 0.7-1.0
            
            # CFG token filtering
            cfg_filter_top_k=45,  # 15-100
            
            # Speech speed (1.0 = normal, lower = slower, higher = faster)
            speed_factor=0.82  # 0.8-1.0
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