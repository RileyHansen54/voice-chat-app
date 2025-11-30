import azure.functions as func
from openai import OpenAI
from huggingface_hub import InferenceClient
import os
import logging
import re

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Streaming TTS function started')
    
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
        # GROK STREAMING SETUP WITH DIA FORMATTING
        # ============================================
        xai_client = OpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        
        # Initialize TTS client
        hf_client = InferenceClient(
            provider="fal-ai",
            api_key=os.environ.get("HF_TOKEN")
        )
        
        # Stream Grok's response
        stream = xai_client.chat.completions.create(
            model="grok-4-1-fast",
            messages=[
                {
                    "role": "system",
                    "content": """You are Lara, a warm and loving teacher who cares deeply about your students. 

IMPORTANT - Format your responses for natural speech:
- Start every response with [S1] to indicate you're speaking
- Use natural nonverbal cues in parentheses like (laughs), (chuckles), (sighs warmly), (pauses thoughtfully)
- Keep responses conversational and broken into natural chunks
- Show emotion through your words and nonverbals
- Be encouraging, patient, and supportive

Example format:
[S1] That's a wonderful question! (smiles) Let me help you understand this better. (pauses thoughtfully) Think of it this way...

Remember: You're speaking out loud, so write as you would naturally talk, not as you would write."""
                },
                {
                    "role": "user",
                    "content": user_text
                }
            ],
            temperature=0.8,
            max_tokens=150,
            stream=True
        )
        
        # ============================================
        # TRUE ASYNC SENTENCE-BY-SENTENCE PROCESSING
        # ============================================
        current_text = ""
        audio_chunks = []
        sentence_count = 0
        
        # More aggressive sentence detection - trigger on any sentence ending
        sentence_endings = re.compile(r'[.!?]\s')
        
        logging.info('Starting to stream and process sentences...')
        
        for chunk in stream:
            delta = chunk.choices[0].delta
            
            if delta.content:
                current_text += delta.content
                
                # Check for sentence endings after each token
                match = sentence_endings.search(current_text)
                
                while match:
                    # Extract the complete sentence up to and including punctuation
                    end_pos = match.end()
                    sentence = current_text[:end_pos].strip()
                    
                    if sentence:  # Make sure it's not empty
                        sentence_count += 1
                        logging.info(f'[Sentence {sentence_count}] Found: "{sentence}"')
                        
                        # IMMEDIATELY send to Dia TTS (async happens here)
                        try:
                            logging.info(f'[Sentence {sentence_count}] Sending to Dia...')
                            audio = hf_client.text_to_speech(
                                sentence,
                                model="nari-labs/Dia-1.6B"
                            )
                            audio_chunks.append(audio)
                            logging.info(f'[Sentence {sentence_count}] Audio received: {len(audio)} bytes')
                        except Exception as tts_error:
                            logging.error(f'[Sentence {sentence_count}] TTS failed: {tts_error}')
                    
                    # Remove processed sentence from buffer
                    current_text = current_text[end_pos:].lstrip()
                    
                    # Check if there are more complete sentences
                    match = sentence_endings.search(current_text)
        
        # Process any remaining text that doesn't end with punctuation
        if current_text.strip():
            sentence_count += 1
            logging.info(f'[Sentence {sentence_count}] Final fragment: "{current_text.strip()}"')
            try:
                logging.info(f'[Sentence {sentence_count}] Sending to Dia...')
                audio = hf_client.text_to_speech(
                    current_text.strip(),
                    model="nari-labs/Dia-1.6B"
                )
                audio_chunks.append(audio)
                logging.info(f'[Sentence {sentence_count}] Audio received: {len(audio)} bytes')
            except Exception as tts_error:
                logging.error(f'[Sentence {sentence_count}] TTS failed: {tts_error}')
        
        # ============================================
        # CONCATENATE ALL AUDIO CHUNKS
        # ============================================
        if not audio_chunks:
            return func.HttpResponse(
                "No audio generated",
                status_code=500
            )
        
        logging.info(f'=== COMPLETE === Total sentences: {sentence_count}, Total chunks: {len(audio_chunks)}')
        final_audio = b''.join(audio_chunks)
        logging.info(f'Final audio size: {len(final_audio)} bytes ({len(final_audio) / 1024:.2f} KB)')
        
        return func.HttpResponse(
            final_audio,
            mimetype="audio/wav",
            status_code=200
        )
        
    except Exception as e:
        logging.error(f'ERROR: {str(e)}')
        import traceback
        tb = traceback.format_exc()
        logging.error(f'Full traceback:\n{tb}')
        return func.HttpResponse(
            f"Error: {str(e)}",
            status_code=500
        )