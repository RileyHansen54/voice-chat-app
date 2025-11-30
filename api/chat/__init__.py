import azure.functions as func
from openai import OpenAI
from huggingface_hub import InferenceClient
import os
import logging
import re
import io
import wave
from concurrent.futures import ThreadPoolExecutor

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
        
        # Initialize clients
        xai_client = OpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        
        hf_client = InferenceClient(
            provider="fal-ai",
            api_key=os.environ.get("HF_TOKEN")
        )
        
        # Stream Grok response and collect sentences
        stream = xai_client.chat.completions.create(
            model="grok-4-1-fast",
            messages=[{"role": "user", "content": user_text}],
            stream=True
        )
        
        current_text = ""
        sentences = []
        sentence_endings = re.compile(r'[.!?](?:\s+|$)')
        
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                current_text += delta.content
                
                match = sentence_endings.search(current_text)
                while match:
                    end_pos = match.end()
                    sentence = current_text[:end_pos].strip()
                    if sentence:
                        sentences.append(sentence)
                        logging.info(f'Sentence {len(sentences)}: "{sentence[:50]}..."')
                    current_text = current_text[end_pos:].lstrip()
                    match = sentence_endings.search(current_text)
        
        # Don't forget remaining text
        if current_text.strip():
            sentences.append(current_text.strip())
        
        if not sentences:
            return func.HttpResponse("No text to synthesize", status_code=400)
        
        logging.info(f'Total sentences: {len(sentences)}')
        
        # Parallel TTS generation
        def generate_tts(sentence_data):
            index, sentence = sentence_data
            try:
                audio = hf_client.text_to_speech(
                    sentence,
                    model="nari-labs/Dia-1.6B"
                )
                logging.info(f'[TTS {index+1}] ✓ {len(audio)} bytes')
                return (index, audio)
            except Exception as e:
                logging.error(f'[TTS {index+1}] ✗ {e}')
                return (index, None)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(generate_tts, enumerate(sentences)))
        
        # Sort by order and filter failures
        results.sort(key=lambda x: x[0])
        audio_chunks = [audio for _, audio in results if audio is not None]
        
        if not audio_chunks:
            return func.HttpResponse("TTS failed", status_code=500)
        
        # Concatenate WAV files
        final_audio = concatenate_wav(audio_chunks)
        
        logging.info(f'✓ Final: {len(final_audio)} bytes')
        
        return func.HttpResponse(
            final_audio,
            mimetype="audio/wav",
            status_code=200
        )
        
    except Exception as e:
        logging.error(f'Error: {str(e)}')
        import traceback
        logging.error(traceback.format_exc())
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)


def concatenate_wav(audio_chunks):
    """Combine multiple WAV files into one"""
    first_wav = wave.open(io.BytesIO(audio_chunks[0]), 'rb')
    params = first_wav.getparams()
    first_wav.close()
    
    all_audio_data = []
    for chunk in audio_chunks:
        wav_file = wave.open(io.BytesIO(chunk), 'rb')
        all_audio_data.append(wav_file.readframes(wav_file.getnframes()))
        wav_file.close()
    
    output = io.BytesIO()
    output_wav = wave.open(output, 'wb')
    output_wav.setparams(params)
    output_wav.writeframes(b''.join(all_audio_data))
    output_wav.close()
    
    output.seek(0)
    return output.read()