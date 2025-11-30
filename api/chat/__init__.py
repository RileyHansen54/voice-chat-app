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
        
        # ============================================
        # INITIALIZE CLIENTS (CORRECTED SYNTAX)
        # ============================================
        xai_client = OpenAI(
            api_key=os.environ["XAI_API_KEY"],
            base_url="https://api.x.ai/v1"
        )
        
        # FIXED: Correct HuggingFace InferenceClient syntax
        hf_client = InferenceClient(
            provider="fal-ai",
            api_key=os.environ["HF_TOKEN"]
        )
        
        # ============================================
        # STREAM GROK RESPONSE
        # ============================================
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
        # COLLECT SENTENCES FROM STREAM
        # ============================================
        current_text = ""
        sentences = []
        sentence_endings = re.compile(r'[.!?](?:\s+|$)')
        
        logging.info('Collecting sentences from Grok stream...')
        
        for chunk in stream:
            delta = chunk.choices[0].delta
            
            if delta.content:
                current_text += delta.content
                
                # Extract complete sentences
                match = sentence_endings.search(current_text)
                while match:
                    end_pos = match.end()
                    sentence = current_text[:end_pos].strip()
                    
                    if sentence:
                        sentences.append(sentence)
                        logging.info(f'Sentence {len(sentences)}: "{sentence[:60]}..."')
                    
                    current_text = current_text[end_pos:].lstrip()
                    match = sentence_endings.search(current_text)
        
        # Add any remaining text
        if current_text.strip():
            sentences.append(current_text.strip())
            logging.info(f'Final text: "{current_text.strip()[:60]}..."')
        
        if not sentences:
            return func.HttpResponse(
                "No text to synthesize",
                status_code=400
            )
        
        logging.info(f'Total sentences collected: {len(sentences)}')
        
        # ============================================
        # PARALLEL TTS GENERATION
        # ============================================
        def generate_tts(sentence_data):
            """Generate TTS for a single sentence"""
            index, sentence = sentence_data
            try:
                logging.info(f'[TTS {index+1}/{len(sentences)}] Processing...')
                
                # Use the exact syntax from HuggingFace docs
                audio = hf_client.text_to_speech(
                    sentence,
                    model="nari-labs/Dia-1.6B"
                )
                
                logging.info(f'[TTS {index+1}/{len(sentences)}] ✓ Generated {len(audio)} bytes')
                return (index, audio)
            except Exception as e:
                logging.error(f'[TTS {index+1}/{len(sentences)}] ✗ Failed: {e}')
                return (index, None)
        
        # Process up to 5 sentences in parallel
        logging.info('Starting parallel TTS generation...')
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(generate_tts, enumerate(sentences)))
        
        # Sort by original order and filter failures
        results.sort(key=lambda x: x[0])
        audio_chunks = [audio for _, audio in results if audio is not None]
        
        if not audio_chunks:
            return func.HttpResponse(
                "TTS generation failed for all sentences",
                status_code=500
            )
        
        success_rate = len(audio_chunks) / len(sentences) * 100
        logging.info(f'TTS Success: {len(audio_chunks)}/{len(sentences)} ({success_rate:.0f}%)')
        
        # ============================================
        # CONCATENATE WAV FILES PROPERLY
        # ============================================
        logging.info('Concatenating audio chunks...')
        final_audio = concatenate_wav_files(audio_chunks)
        
        logging.info(f'✓ Final audio: {len(final_audio)} bytes ({len(final_audio) / 1024:.2f} KB)')
        
        return func.HttpResponse(
            final_audio,
            mimetype="audio/wav",
            status_code=200
        )
        
    except KeyError as e:
        # Missing environment variable
        error_msg = f"Missing environment variable: {str(e)}"
        logging.error(error_msg)
        return func.HttpResponse(
            error_msg,
            status_code=500
        )
    except Exception as e:
        logging.error(f'ERROR: {str(e)}')
        import traceback
        logging.error(traceback.format_exc())
        return func.HttpResponse(
            f"Error: {str(e)}",
            status_code=500
        )


def concatenate_wav_files(audio_chunks):
    """
    Properly concatenate WAV files by:
    1. Reading parameters from first file
    2. Extracting raw audio data from each chunk
    3. Combining audio data
    4. Creating new WAV file with single header
    """
    if not audio_chunks:
        raise ValueError("No audio chunks to concatenate")
    
    logging.info(f'Concatenating {len(audio_chunks)} WAV files...')
    
    try:
        # Read first WAV to get audio parameters
        first_wav = wave.open(io.BytesIO(audio_chunks[0]), 'rb')
        params = first_wav.getparams()
        logging.info(f'WAV params: {params.nchannels} ch, {params.framerate} Hz, {params.sampwidth*8} bit')
        first_wav.close()
        
        # Extract raw audio data from all chunks
        all_audio_data = []
        total_frames = 0
        
        for i, chunk in enumerate(audio_chunks):
            wav_file = wave.open(io.BytesIO(chunk), 'rb')
            frames = wav_file.getnframes()
            audio_data = wav_file.readframes(frames)
            all_audio_data.append(audio_data)
            total_frames += frames
            wav_file.close()
            logging.info(f'Chunk {i+1}: {frames} frames, {len(audio_data)} bytes')
        
        # Combine all audio data
        combined_audio = b''.join(all_audio_data)
        logging.info(f'Combined audio: {total_frames} frames, {len(combined_audio)} bytes')
        
        # Create new WAV file with combined data
        output = io.BytesIO()
        output_wav = wave.open(output, 'wb')
        output_wav.setparams(params)
        output_wav.writeframes(combined_audio)
        output_wav.close()
        
        # Get final WAV bytes
        output.seek(0)
        final_wav = output.read()
        
        logging.info(f'Final WAV size: {len(final_wav)} bytes')
        return final_wav
        
    except Exception as e:
        logging.error(f'WAV concatenation failed: {e}')
        raise