# Gemma4_VisionBeyondSight
## Project Overview
This project is an offline AI assistant designed to support visually impaired users in daily activities.  
The system uses Gemma 4 multimodal models running locally with CUDA acceleration for real-time performance.

Main features:
- OCR text reading (English & Vietnamese)
- Vietnamese banknote recognition
- Object and surrounding scene understanding
- Voice response assistance
- Offline local inference using GGUF models

## Models: Gemma 4 with Currency Recognition Model: https://huggingface.co/Amber1122/gemma4_E2B_IT_Q4_currency/tree/main

Download the models and place them inside the `models/` directory before running the project.

## Requirements
- NVIDIA GPU with CUDA support
- Python 3.10+
- Webcam / OAK Camera
- Speaker or earphones for audio output

## Notes
- The system is optimized for embedded/edge AI devices
- OCR supports both printed and handwritten English/Vietnamese text
- All inference runs locally without cloud processing
