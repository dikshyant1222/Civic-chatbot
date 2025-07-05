# Legal Assistant Chatbot

A chatbot for answering questions about Nepalese law, with text and voice capabilities.

## Features

- Answers legal questions based on a corpus of Nepalese law documents
- Maintains conversation history for context-aware responses
- Adaptive response length (automatically detects if you want shorter or detailed responses)
- Voice input and output capabilities
- Clean, modern UI

## Setup

### Installation

1. Clone this repository
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

### API Keys

To use the voice features, you'll need to set up API keys:

1. **ElevenLabs API Key**: For text-to-speech capabilities
   - Get a free API key from [ElevenLabs](https://elevenlabs.io)
   
2. **AssemblyAI API Key**: For speech-to-text capabilities  
   - Get a free API key from [AssemblyAI](https://www.assemblyai.com)

Set your API keys using the provided script:

```
# Set ElevenLabs API key
python set_api_keys.py --elevenlabs YOUR_ELEVENLABS_API_KEY

# Set AssemblyAI API key
python set_api_keys.py --assemblyai YOUR_ASSEMBLYAI_API_KEY

# Check current API key status
python set_api_keys.py
```

## Usage

Start the application:

```
python app.py
```

Then navigate to `http://localhost:5000` in your web browser.

### Text Interaction

Simply type your message in the text box and press Enter or click the send button.

### Voice Commands

1. Click the microphone button to start recording
2. Speak your question or command
3. Click the microphone button again to stop recording
4. The system will transcribe your speech and send it as a message

### Listening to Responses

Each bot response includes a speaker icon. Click it to have the response read aloud.

### Response Length

The chatbot automatically detects if you want brief or detailed responses:

- For brief responses, include words like "short", "brief", or "concise" in your query
- For detailed responses, include words like "detailed", "explain", or "comprehensive" in your query

## License

This project is licensed under the MIT License.
