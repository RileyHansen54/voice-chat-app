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
                    # ⬇️ EDIT GROK'S PERSONALITY HERE
                    "content": "You are a helpful, friendly AI assistant. Keep responses concise and conversational."
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ],
            temperature=0.7,  # 0.0-2.0 (higher = more creative)
            max_tokens=150,   # Max response length
        )
        
        response_text = chat_response.choices[0].message.content
        logging.info(f'AI response: {response_text}')
        
        # ============================================
        # TTS - USING KOKORO MODEL
        # ============================================
        hf_client = InferenceClient(
            provider="fal-ai",
            api_key=os.environ.get("HF_TOKEN")
        )
        
        audio_bytes = hf_client.text_to_speech(
            response_text,
            model="hexgrad/Kokoro-82M"  # Changed to Kokoro
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