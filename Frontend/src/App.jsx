import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  const recognitionRef = useRef(null);

  useEffect(() => {
    // Initialize speech recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      setError('Speech recognition is not supported in your browser. Please use Chrome or Edge.');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setIsListening(true);
      setError('');
    };

    recognition.onresult = async (event) => {
      const userText = event.results[0][0].transcript;
      setTranscript(userText);
      setIsListening(false);
      
      // Send to backend
      await sendToBackend(userText);
    };

    recognition.onerror = (event) => {
      setIsListening(false);
      setError(`Speech recognition error: ${event.error}`);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;
  }, []);

  const sendToBackend = async (text) => {
    setIsLoading(true);
    setError('');
    
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      // Get audio blob
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // Play audio
      const audio = new Audio(audioUrl);
      audio.play();
      
      setResponse('Response received and playing...');
      
      // Clean up URL after playing
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
      };
      
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const startListening = () => {
    if (recognitionRef.current) {
      setTranscript('');
      setResponse('');
      recognitionRef.current.start();
    }
  };

  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>TutorAI</h1>
        <p>Click the button and speak to chat with AI</p>
        
        <div className="controls">
          {!isListening ? (
            <button 
              className="mic-button" 
              onClick={startListening}
              disabled={isLoading}
            >
              {isLoading ? 'Processing...' : 'Start Speaking'}
            </button>
          ) : (
            <button 
              className="mic-button listening" 
              onClick={stopListening}
            >
              Listening... (Click to stop)
            </button>
          )}
        </div>

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        {transcript && (
          <div className="transcript">
            <h3>You said:</h3>
            <p>{transcript}</p>
          </div>
        )}

        {response && (
          <div className="response">
            <h3>AI Response:</h3>
            <p>{response}</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;