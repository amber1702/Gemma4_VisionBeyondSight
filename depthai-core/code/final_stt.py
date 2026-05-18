#!/usr/bin/env python3

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["ORT_LOGGING_LEVEL"] = "3"

import cv2
import depthai as dai
import requests
import base64
import Jetson.GPIO as GPIO
import time
import subprocess
import sounddevice as sd
import numpy as np
import pygame
import re 
from scipy.io.wavfile import write

# =========================================
# GPIO BUTTON CONFIG
# =========================================

BUTTON_SCENE = 31
BUTTON_OCR = 29
BUTTON_MONEY = 32
BUTTON_LANGUAGE = 33

GPIO.setmode(GPIO.BOARD)

GPIO.setup(BUTTON_SCENE, GPIO.IN)
GPIO.setup(BUTTON_OCR, GPIO.IN)
GPIO.setup(BUTTON_MONEY, GPIO.IN)
GPIO.setup(BUTTON_LANGUAGE, GPIO.IN)

# =========================================
# HOLD CONFIG
# =========================================

HOLD_TIME = 1.0

# =========================================
# LANGUAGE STATE
# =========================================

current_language = "en"

# =========================================
# GEMMA SERVER
# =========================================

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

# =========================================
# PIPER MODELS
# =========================================

EN_MODEL = "/home/amber/gemma/depthai-core/code/piper_models/en_US-lessac-low.onnx"

VI_MODEL = "/home/amber/gemma/depthai-core/code/piper_models/vi_VN-vais1000-medium.onnx"

# =========================================
# AUDIO INIT
# =========================================

pygame.mixer.init()

# =========================================
# PROMPTS
# =========================================

SCENE_PROMPT_EN = (
    "Describe shortly the image "
    "(danger, people, environment, obstacles, objects) "
    "in one short sentence only."
)

SCENE_PROMPT_VI = (
    "Hãy mô tả ngắn gọn hình ảnh (nguy hiểm, chướng ngại vật, người, nếu có) chỉ trong 1 câu."
)

OCR_PROMPT_EN = (
    "Read the most visible text exactly as written. "
    "Do not summarize. "
    "Translate to English correctly. "
    "Return text only."
)

OCR_PROMPT_VI = (
"Hãy đọc đoạn văn bản dễ nhìn nhất chính xác."
    "Không tóm tắt. "
    "Giữ đúng dấu tiếng Việt. "
    "Chỉ trả về văn bản."
)



MONEY_PROMPT_EN = (
        "List Vietnamese banknote denominations.\n"
        "Only 1 or 2 banknotes in the images.\n"
        "Output the numbers with unit VND sperated by space.\n"
	    "Possible values are:\n"
	    "1000 VND,\n"
	    "2000 VND,\n"
	    "5000 VND,\n"
	    "10000 VND,\n"
	    "20000 VND,\n"
	    "50000 VND,\n"
	    "100000 VND,\n"
	    "200000 VND,\n"
	    "500000 VND.\n\n"
	    "Return only the denomination values ( maximum 2 banknotes). If not return no VietNamese banknote recognition."
)

MONEY_PROMPT_VI = (
        "List Vietnamese banknote denominations.\n"
        "Only 1 or 2 banknotes in the images.\n"
        "Output the numbers with unit VND sperated by space.\n"
	    "Possible values are:\n"
	    "1000 VND,\n"
	    "2000 VND,\n"
	    "5000 VND,\n"
	    "10000 VND,\n"
	    "20000 VND,\n"
	    "50000 VND,\n"
	    "100000 VND,\n"
	    "200000 VND,\n"
	    "500000 VND.\n\n"
	    "Return only the denomination values ( maximum 2 banknotes). If not return no VietNamese banknote recognition."
)
# =========================================
# IMAGE ENCODER
# =========================================

def encode_image(image_path):

    with open(image_path, "rb") as f:

        return base64.b64encode(
            f.read()
        ).decode("utf-8")

# =========================================
# GEMMA VLM
# =========================================

def ask_gemma(image_path, prompt):

    image_base64 = encode_image(
        image_path
    )

    # =====================================
    # DYNAMIC MAX TOKENS
    # =====================================

    if "money" in prompt.lower():

        max_tokens = 15

    elif "Read visible text" in prompt:

        max_tokens = 35

    else:

        max_tokens = 25

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Short assistive responses only."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":
                            f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],

        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_k": 20,
        "top_p": 0.8
    }

    response = requests.post(
        SERVER_URL,
        json=payload
    )

    result = response.json()

    return result["choices"][0]["message"]["content"]

# =========================================
# GEMMA VOICE + VISION
# =========================================

def ask_gemma_voice_vision(
    image_path,
    user_text
):

    image_base64 = encode_image(
        image_path
    )

    if current_language == "vi":

        system_prompt = (
            "Bạn là trợ lý AI hỗ trợ "
            "người khiếm thị. "
            "Trả lời ngắn gọn."
        )

    else:

        system_prompt = (
            "You are an assistive AI "
            "for blind users. "
            "Answer shortly."
        )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":
                            f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],

        "max_tokens": 30,
        "temperature": 0.2,
        "top_k": 20,
        "top_p": 0.8
    }

    response = requests.post(
        SERVER_URL,
        json=payload
    )

    result = response.json()

    return result["choices"][0]["message"]["content"]

# =========================================
# PHOWHISPER STT
# =========================================

def phowhisper_stt(audio_path):

    response = requests.post(
        "http://127.0.0.1:5001/transcribe",
        json={
            "audio_path": audio_path
        }
    )

    result = response.json()

    return result["text"]

# =========================================
# WHISPER SERVER STT
# =========================================

def whisper_stt(audio_path):

    response = requests.post(
        "http://127.0.0.1:5002/transcribe",
        json={
            "audio_path": audio_path
        }
    )

    result = response.json()

    return result["text"]

# =========================================
# PIPER TTS
# =========================================

def speak_text(text):

    global current_language

    try:

        print("\n========== SPEAK ==========\n")
        print(text)
        text = clean_tts_text(text)

        # =================================================
        # NATURAL VIETNAMESE MONEY SPEECH
        # =================================================
        if current_language == "vi":

            if GPIO.input(BUTTON_MONEY) == GPIO.LOW:

                text = currency_to_speech_vi(text)

        print("\n========== TTS TEXT ==========\n")
        print(text)

        temp_wav = "/tmp/output.wav"

        # =================================================
        # MODEL
        # =================================================
        if current_language == "vi":

            model_path = VI_MODEL

        else:

            model_path = EN_MODEL

        # =================================================
        # PIPER
        # =================================================
        cmd = f'''
        echo "{text}" | \
        piper \
        --model "{model_path}" \
        --length_scale 1.15 \
        --sentence_silence 0.4 \
        --output_file "{temp_wav}"
        '''

        subprocess.run(

            cmd,

            shell=True
        )

        # =================================================
        # PLAY
        # =================================================
        pygame.mixer.music.load(
            temp_wav
        )

        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():

            time.sleep(0.05)

    except Exception as e:

        print("TTS Error:", e)




# =========================================================
# NUMBER TO NATURAL VIETNAMESE
# =========================================================
def number_to_vietnamese(n):

    units = [

        "",
        "một",
        "hai",
        "ba",
        "bốn",
        "năm",
        "sáu",
        "bảy",
        "tám",
        "chín",
    ]

    # =====================================================
    # READ 3 DIGITS
    # =====================================================
    def read3(x, full=False):

        hundred = x // 100
        ten = (x % 100) // 10
        unit = x % 10

        result = ""

        # -------------------------------------------------
        # HUNDRED
        # -------------------------------------------------
        if hundred > 0 or full:

            result += units[hundred] + " trăm"

        # -------------------------------------------------
        # TEN > 1
        # -------------------------------------------------
        if ten > 1:

            result += " " + units[ten] + " mươi"

            if unit == 1:

                result += " mốt"

            elif unit == 5:

                result += " lăm"

            elif unit > 0:

                result += " " + units[unit]

        # -------------------------------------------------
        # TEN == 1
        # -------------------------------------------------
        elif ten == 1:

            result += " mười"

            if unit == 5:

                result += " lăm"

            elif unit > 0:

                result += " " + units[unit]

        # -------------------------------------------------
        # TEN == 0
        # -------------------------------------------------
        elif unit > 0:

            if hundred > 0:

                result += " lẻ"

            result += " " + units[unit]

        return result.strip()

    # =====================================================
    # UNDER 1000
    # =====================================================
    if n < 1000:

        return read3(n)

    # =====================================================
    # THOUSAND
    # =====================================================
    elif n < 1_000_000:

        thousand = n // 1000
        remain = n % 1000

        result = read3(thousand) + " ngàn"

        if remain > 0:

            result += " " + read3(remain, True)

        return result.strip()


    return str(n)

# =========================================================
# NUMBER TO ENGLISH
# =========================================================
def number_to_english(n):

    under_20 = [

        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen"
    ]

    tens = [

        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety"
    ]

    def words(num):

        if num < 20:

            return under_20[num]

        elif num < 100:

            return (

                tens[num // 10]

                +

                (
                    " " + under_20[num % 10]
                    if num % 10 != 0
                    else ""
                )
            )

        elif num < 1000:

            return (

                under_20[num // 100]
                + " hundred"

                +

                (
                    " " + words(num % 100)
                    if num % 100 != 0
                    else ""
                )
            )

        elif num < 1_000_000:

            return (

                words(num // 1000)
                + " thousand"

                +

                (
                    " " + words(num % 1000)
                    if num % 1000 != 0
                    else ""
                )
            )

        elif num < 1_000_000_000:

            return (

                words(num // 1_000_000)
                + " million"

                +

                (
                    " " + words(num % 1_000_000)
                    if num % 1_000_000 != 0
                    else ""
                )
            )

        return str(num)

    return words(n)

# =========================================================
# NATURAL VIETNAMESE CURRENCY SPEECH
# =========================================================
def currency_to_speech_vi(text):

    nums = re.findall(r"\d+", text)

    results = []

    for n in nums:

        num = int(n)

        speech = number_to_vietnamese(num)

        results.append(speech)

    # =====================================================
    # NATURAL JOIN
    # =====================================================
    if len(results) == 0:

        return "không phát hiện tiền"

    elif len(results) == 1:

        return results[0]

    else:

        return " và ".join(results)

# =========================================================
# ENGLISH CURRENCY SPEECH
# =========================================================
def currency_to_speech_en(text):

    nums = re.findall(r"\d+", text)

    results = []

    for n in nums:

        num = int(n)

        speech = (

            number_to_english(num)
            + " Vietnamese dong"
        )

        results.append(speech)

    # =====================================================
    # NATURAL JOIN
    # =====================================================
    if len(results) == 0:

        return "No Vietnamese banknotes detected"

    elif len(results) == 1:

        return results[0]

    else:

        return " and ".join(results)

# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":

    tests = [

        "1000 VND",

        "10000 VND",

        "50000 VND",

        "100000 VND",

        "200000 VND",

        "500000 VND",

        "10000 VND 200000 VND",

        "5000 VND 500000 VND",
    ]

    print("\n==============================")
    print("VIETNAMESE")
    print("==============================")

    for t in tests:

        print(f"{t}")
        print(currency_to_speech_vi(t))
        print()

    print("\n==============================")
    print("ENGLISH")
    print("==============================")

    for t in tests:

        print(f"{t}")
        print(currency_to_speech_en(t))
        print()
    

def clean_tts_text(text):
    # Remove markdown bold/italic/code markers
    text = re.sub(r'[*_`#]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
    
# =========================================
# START CAMERA
# =========================================

print("Starting camera...")

device = dai.Device()

with dai.Pipeline(device) as pipeline:

    outputQueues = {}

    sockets = device.getConnectedCameras()

    for socket in sockets:

        cam = pipeline.create(
            dai.node.Camera
        )

        cam.initialControl.setAutoFocusMode(
            dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO
        )

        cam = cam.build(socket)

        outputQueues[str(socket)] = (
            cam.requestOutput(
                (640, 480),
                dai.ImgFrame.Type.BGR888p,
                dai.ImgResizeMode.LETTERBOX,
                30
            ).createOutputQueue()
        )

    pipeline.start()

    print("\n===================================")
    print("Assistive AI System Ready!")
    print("===================================")
    print("Short Press Button 1 -> Scene")
    print("Hold Button 1 -> Voice Assistant")
    print("Button 2 -> OCR")
    print("Button 3 -> Money")
    print("Button 4 -> Language")
    print("===================================\n")

    while pipeline.isRunning():

        frames = {}

        for name, queue in outputQueues.items():

            videoIn = queue.get()

            frame = videoIn.getCvFrame()

            frames[name] = frame

            cv2.imshow(name, frame)

        first_cam = list(frames.keys())[0]

        frame = frames[first_cam]

        # =====================================
        # BUTTON 1
        # =====================================

        if GPIO.input(BUTTON_SCENE) == GPIO.LOW:

            press_time = time.time()

            # =================================
            # WAIT PRESS / HOLD
            # =================================

            while GPIO.input(BUTTON_SCENE) == GPIO.LOW:

                held_time = (
                    time.time() - press_time
                )

                # =============================
                # HOLD -> VOICE MODE
                # =============================

                if held_time >= HOLD_TIME:

                    print("\n[VOICE MODE]")

                    fs = 16000

                    audio_chunks = []

                    stream = sd.InputStream(
                        samplerate=fs,
                        channels=1,
                        dtype='int16',
                        blocksize=1024
                    )

                    stream.start()

                    print("Recording...")

                    # =========================
                    # RECORD WHILE HOLDING
                    # =========================

                    while GPIO.input(BUTTON_SCENE) == GPIO.LOW:

                        data, overflowed = (
                            stream.read(1024)
                        )

                        audio_chunks.append(data)

                    stream.stop()
                    stream.close()

                    print("Recording finished")

                    if len(audio_chunks) == 0:

                        print("No audio recorded")

                        break

                    audio = np.concatenate(
                        audio_chunks,
                        axis=0
                    )

                    audio_path = "/tmp/voice.wav"

                    write(
                        audio_path,
                        fs,
                        audio
                    )

                    # =========================
                    # SAVE SMALL IMAGE
                    # =========================

                    voice_image = (
                        "/tmp/voice_scene.jpg"
                    )

                    small_frame = cv2.resize(
                        frame,
                        (320, 240)
                    )

                    cv2.imwrite(
                        voice_image,
                        small_frame
                    )

                    try:

                        if current_language == "vi":

                            user_text = (
                                phowhisper_stt(
                                    audio_path
                                )
                            )

                        else:

                            user_text = (
                                whisper_stt(
                                    audio_path
                                )
                            )

                        print(
                            "\n========== USER ==========\n"
                        )

                        print(user_text)

                        if len(
                            user_text.strip()
                        ) == 0:

                            print(
                                "Empty transcription"
                            )

                            break

                        answer = (
                            ask_gemma_voice_vision(
                                voice_image,
                                user_text
                            )
                        )

                        print(
                            "\n========== GEMMA ==========\n"
                        )

                        print(answer)

                        speak_text(answer)

                    except Exception as e:

                        print("Error:", e)

                    # =========================
                    # DEBOUNCE
                    # =========================

                    time.sleep(0.5)

                    break

                time.sleep(0.01)

            # =================================
            # SHORT PRESS -> SCENE
            # =================================

            release_time = (
                time.time() - press_time
            )

            if release_time < HOLD_TIME:

                print("\n[SCENE BUTTON PRESSED]")

                image_path = "scene.jpg"

                small_frame = cv2.resize(
                    frame,
                    (320, 240)
                )

                cv2.imwrite(
                    image_path,
                    small_frame
                )

                if current_language == "vi":

                    prompt = SCENE_PROMPT_VI

                else:

                    prompt = SCENE_PROMPT_EN

                try:

                    output = ask_gemma(
                        image_path,
                        prompt
                    )

                    print(
                        "\n========== SCENE OUTPUT ==========\n"
                    )

                    print(output)

                    speak_text(output)

                except Exception as e:

                    print("Gemma Error:", e)

            time.sleep(0.3)

        # =====================================
        # BUTTON 2 - OCR
        # =====================================

        elif GPIO.input(BUTTON_OCR) == GPIO.LOW:

            print("\n[OCR BUTTON PRESSED]")

            image_path = "ocr.jpg"

            # =================================
            # WAIT AUTOFOCUS
            # =================================

            time.sleep(0.5)

            # =================================
            # HIGHER RESOLUTION FOR OCR
            # =================================

            small_frame = cv2.resize(
                frame,
                (640, 480)
            )

            # =================================
            # CONVERT TO GRAYSCALE
            # =================================

            gray = cv2.cvtColor(
                small_frame,
                cv2.COLOR_BGR2GRAY
            )

            # =================================
            # SHARPEN FILTER
            # =================================

            sharpen_kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ])

            sharpen = cv2.filter2D(
                gray,
                -1,
                sharpen_kernel
            )

            # =================================
            # SAVE OCR IMAGE
            # =================================

            cv2.imwrite(
                image_path,
                sharpen
            )

            # =================================
            # LANGUAGE PROMPT
            # =================================

            if current_language == "vi":

                prompt = OCR_PROMPT_VI

            else:

                prompt = OCR_PROMPT_EN

            try:

                output = ask_gemma(
                    image_path,
                    prompt
                )

                print(
                    "\n========== OCR OUTPUT ==========\n"
                )

                print(output)

                speak_text(output)

            except Exception as e:

                print("Gemma Error:", e)

            time.sleep(1)


        # =====================================
        # BUTTON 3 - MONEY
        # =====================================

        elif GPIO.input(BUTTON_MONEY) == GPIO.LOW:

            print("\n[MONEY BUTTON PRESSED]")

            image_path = "money.jpg"

            small_frame = cv2.resize(
                frame,
                (640, 480)
            )

            cv2.imwrite(
                image_path,
                small_frame
            )

            if current_language == "vi":

                prompt = MONEY_PROMPT_VI

            else:

                prompt = MONEY_PROMPT_EN

            try:

                output = ask_gemma(
                    image_path,
                    prompt
                )

                print(
                    "\n========== MONEY OUTPUT ==========\n"
                )

                print(output)

                speak_text(output)

            except Exception as e:

                print("Gemma Error:", e)

            time.sleep(1)

        # =====================================
        # BUTTON 4 - LANGUAGE
        # =====================================

        elif GPIO.input(BUTTON_LANGUAGE) == GPIO.LOW:

            if current_language == "en":

                current_language = "vi"

                print("\nSwitched to Vietnamese")

                speak_text(
                    "Đã chuyển sang tiếng Việt"
                )

            else:

                current_language = "en"

                print("\nSwitched to English")

                speak_text(
                    "Switched to English"
                )

            time.sleep(1)

        key = cv2.waitKey(1)

        if key == ord("q"):

            break

GPIO.cleanup()

cv2.destroyAllWindows()


