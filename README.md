# Parakeet AI Assistant

A Windows desktop AI conversation assistant that captures microphone and system audio, transcribes speech in real time, detects questions, and generates automatic AI-powered answer suggestions.

## Features

- Microphone audio capture
- System audio capture using WASAPI loopback
- Speech-to-text using Faster-Whisper
- Automatic question detection
- AI-generated answer suggestions
- Streaming responses
- Local LLM inference using Ollama
- Desktop interface using CustomTkinter

## Technologies

- Python
- CustomTkinter
- Faster-Whisper
- PyAudioWPatch
- SoundDevice
- NumPy
- Ollama
- Llama 3.2 3B

## Requirements

- Python 3.12
- Ollama
- Windows operating system
- Microphone and audio output device

## Installation

1. Clone the repository.
2. Create a virtual environment:

   python -m venv venv

3. Activate it:

   .\venv\Scripts\Activate.ps1

4. Install dependencies:

   pip install -r requirements.txt

5. Install Ollama and download the model:

   ollama pull llama3.2:3b

6. Start Ollama.

7. Run the application:

   python main.py

## How It Works

1. The application captures microphone and system audio.
2. Faster-Whisper converts audio into text.
3. The question detection module identifies questions.
4. Ollama processes detected questions using Llama 3.2 3B.
5. Suggested answers are displayed in the desktop interface.

## Project Status

Functional prototype. Model evaluation, fine-tuning, and further performance optimization are potential future improvements.
