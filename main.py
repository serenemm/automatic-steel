import os
from fastapi import FastAPI, WebSocket
from deepgram import DeepgramClient, PrerecordedOptions
from openai import OpenAI
import requests

app = FastAPI()

# Clients using Environment Variables
dg_client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key=os.getenv("OPENROUTER_API_KEY")
)

@app.websocket("/voice-chat")
async def voice_chat(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # 1. Receive Audio from Browser
            audio_data = await websocket.receive_bytes()

            # 2. STT: Convert Voice to Text (Deepgram)
            # Use 'nova-2' for the fastest free-tier speed
            options = PrerecordedOptions(model="nova-2", smart_format=True)
            response = dg_client.listen.prerecorded.v("1").transcribe_indata(audio_data, options)
            user_text = response.results.channels[0].alternatives[0].transcript

            if not user_text: continue

            # 3. LLM: Get AI Response (OpenRouter)
            # Use the ':free' models to avoid charges
            completion = ai_client.chat.completions.create(
                model="google/gemini-2.0-flash-lite-preview-02-05:free",
                messages=[{"role": "user", "content": user_text}]
            )
            ai_text = completion.choices[0].message.content

            # 4. TTS: Convert Text to Voice (ElevenLabs)
            tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{os.getenv('VOICE_ID')}"
            headers = {"xi-api-key": os.getenv("ELEVENLABS_API_KEY")}
            tts_resp = requests.post(tts_url, json={"text": ai_text}, headers=headers)

            # 5. Send Audio back to Browser
            await websocket.send_bytes(tts_resp.content)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
