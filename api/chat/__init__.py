import azure.functions as func
import logging
import traceback

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    SUPER DEBUG VERSION - Shows exactly where it breaks
    """
    error_log = []
    
    try:
        error_log.append("Step 1: Starting function")
        logging.info("Step 1: Function started")
        
        # Test request parsing
        error_log.append("Step 2: Parsing request")
        try:
            req_body = req.get_json()
            user_text = req_body.get('text', 'test')
            error_log.append(f"Step 2 OK: Got text '{user_text}'")
        except Exception as e:
            error_log.append(f"Step 2 FAILED: {str(e)}")
            raise
        
        # Test os import
        error_log.append("Step 3: Importing os")
        try:
            import os
            error_log.append("Step 3 OK: os imported")
        except Exception as e:
            error_log.append(f"Step 3 FAILED: {str(e)}")
            raise
        
        # Test environment variables
        error_log.append("Step 4: Checking environment variables")
        try:
            hf_token = os.environ.get("HF_TOKEN")
            xai_key = os.environ.get("XAI_API_KEY")
            error_log.append(f"Step 4: HF_TOKEN exists: {hf_token is not None}")
            error_log.append(f"Step 4: XAI_API_KEY exists: {xai_key is not None}")
            
            if not hf_token:
                error_log.append("Step 4 ERROR: HF_TOKEN is missing!")
                return func.HttpResponse(
                    "\n".join(error_log) + "\n\nERROR: HF_TOKEN not found in environment",
                    status_code=500
                )
            if not xai_key:
                error_log.append("Step 4 ERROR: XAI_API_KEY is missing!")
                return func.HttpResponse(
                    "\n".join(error_log) + "\n\nERROR: XAI_API_KEY not found in environment",
                    status_code=500
                )
            error_log.append("Step 4 OK: Both env vars exist")
        except Exception as e:
            error_log.append(f"Step 4 FAILED: {str(e)}")
            raise
        
        # Test huggingface_hub import
        error_log.append("Step 5: Importing huggingface_hub")
        try:
            from huggingface_hub import InferenceClient
            error_log.append("Step 5 OK: InferenceClient imported")
        except Exception as e:
            error_log.append(f"Step 5 FAILED: {str(e)}")
            error_log.append("Step 5 ERROR: huggingface_hub not installed or wrong version")
            return func.HttpResponse(
                "\n".join(error_log) + f"\n\nERROR: Cannot import InferenceClient: {str(e)}",
                status_code=500
            )
        
        # Test InferenceClient initialization
        error_log.append("Step 6: Creating InferenceClient")
        try:
            client = InferenceClient(
                provider="fal-ai",
                api_key=hf_token
            )
            error_log.append("Step 6 OK: InferenceClient created")
        except Exception as e:
            error_log.append(f"Step 6 FAILED: {str(e)}")
            error_log.append(f"Step 6 ERROR: {traceback.format_exc()}")
            return func.HttpResponse(
                "\n".join(error_log) + f"\n\nERROR: Cannot create InferenceClient: {str(e)}",
                status_code=500
            )
        
        # Test TTS
        error_log.append("Step 7: Calling text_to_speech")
        try:
            audio = client.text_to_speech(
                "test",
                model="nari-labs/Dia-1.6B"
            )
            error_log.append(f"Step 7 OK: TTS returned {len(audio)} bytes")
        except Exception as e:
            error_log.append(f"Step 7 FAILED: {str(e)}")
            error_log.append(f"Step 7 ERROR: {traceback.format_exc()}")
            return func.HttpResponse(
                "\n".join(error_log) + f"\n\nERROR: TTS failed: {str(e)}",
                status_code=500
            )
        
        # Success!
        error_log.append("SUCCESS: All steps passed!")
        
        return func.HttpResponse(
            "\n".join(error_log) + f"\n\n✅ SUCCESS! Audio size: {len(audio)} bytes",
            status_code=200
        )
        
    except Exception as e:
        error_log.append(f"\nFATAL ERROR: {str(e)}")
        error_log.append(f"\nTraceback:\n{traceback.format_exc()}")
        
        return func.HttpResponse(
            "\n".join(error_log),
            status_code=500
        )