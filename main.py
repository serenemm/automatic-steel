import os
from fastapi import FastAPI, WebSocket
from deepgram import DeepgramClient, PrerecordedOptions
from openai import OpenAI

app = FastAPI()

# Clients
dg_client = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
ai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # 1. Receive audio bytes from your browser/app
        data = await websocket.receive_bytes()
        
        # 2. STT (Deepgram)
        response = dg_client.listen.prerecorded.v("1").transcribe_phonetic(data, options)
        user_text = response.results.channels[0].alternatives[0].transcript

        # 3. LLM (OpenRouter)
        ai_resp = ai_client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-preview-02-05:free",
            messages=[{"role": "user", "content": user_text}]
        )
        ai_text = ai_resp.choices[0].message.content

        # 4. TTS (ElevenLabs)
        # Convert ai_text to audio bytes and send back to websocket
        await websocket.send_bytes(audio_bytes)
