
#!/usr/bin/env python3

import cv2
import depthai as dai
import requests
import base64
import Jetson.GPIO as GPIO
import time
import subprocess

# =========================================
# GPIO BUTTON CONFIG
# =========================================

BUTTON_SCENE = 31
BUTTON_OCR = 29
BUTTON_MONEY = 32
BUTTON_LANGUAGE = 33

GPIO.setmode(GPIO.BOARD)

GPIO.setup(BUTTON_SCENE, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_OCR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_MONEY, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_LANGUAGE, GPIO.IN, pull_up_down=GPIO.PUD_UP)

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
# PROMPTS
# =========================================

SCENE_PROMPT_EN = (
    "Describe shortly the image "
    "(danger, people, environment, obstacles, objects) "
    "in one short sentence only."
)

SCENE_PROMPT_VI = (
    "Mô tả ngắn gọn hình ảnh "
    "(nguy hiểm, con người, môi trường, chướng ngại vật, đồ vật)"
    "chỉ trong một câu ngắn gọn."
)

OCR_PROMPT_EN = (
    "Read visible text only."
)

OCR_PROMPT_VI = (
    "đọc phần văn bản hiển thị. Nếu là tiếng Anh, vui lòng dịch sang tiếng Việt."
)

MONEY_PROMPT_EN = (
    "Identify Vietnamese money denomination only."
)

MONEY_PROMPT_VI = (
    "Nhận diện mệnh giá tiền Việt Nam."
)

# =========================================
# IMAGE ENCODER
# =========================================

def encode_image(image_path):

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# =========================================
# GEMMA REQUEST
# =========================================

def ask_gemma(image_path, prompt):

    image_base64 = encode_image(image_path)

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistive AI for blind users. "
                    "Answer very short. "
                    "Maximum 8 words. "
                    "No explanation."
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
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 20,
        "temperature": 0.1
    }

    response = requests.post(
        SERVER_URL,
        json=payload
    )

    result = response.json()

    return result["choices"][0]["message"]["content"]

# =========================================
# PIPER TTS
# =========================================

def speak_text(text):

    global current_language

    try:

        print("\n========== FINAL OUTPUT ==========\n")
        print(text)

        temp_wav = "/tmp/output.wav"

        # =====================================
        # SELECT MODEL
        # =====================================

        if current_language == "vi":
            model_path = VI_MODEL
        else:
            model_path = EN_MODEL

        # =====================================
        # PIPER COMMAND
        # =====================================

        cmd = f'''
        echo "{text}" | \
        piper \
        --model "{model_path}" \
        --output_file "{temp_wav}"
        '''

        subprocess.run(
            cmd,
            shell=True
        )

        subprocess.run(
            f"aplay {temp_wav}",
            shell=True
        )

    except Exception as e:

        print("TTS Error:", e)

# =========================================
# START CAMERA
# =========================================

print("Starting camera...")

device = dai.Device()

with dai.Pipeline(device) as pipeline:

    outputQueues = {}

    sockets = device.getConnectedCameras()

    for socket in sockets:

        cam = pipeline.create(dai.node.Camera)

        # =====================================
        # AUTOFOCUS
        # =====================================

        cam.initialControl.setAutoFocusMode(
            dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO
        )

        cam = cam.build(socket)

        outputQueues[str(socket)] = cam.requestOutput(
            (640, 480),
            dai.ImgFrame.Type.BGR888p,
            dai.ImgResizeMode.LETTERBOX,
            30
        ).createOutputQueue()

    pipeline.start()

    print("\n===================================")
    print("Assistive AI System Ready!")
    print("===================================")
    print("Button 1 -> Scene Description")
    print("Button 2 -> OCR")
    print("Button 3 -> Money")
    print("Button 4 -> Language Switch")
    print("Default Language -> English")
    print("Press Q to quit")
    print("===================================\n")

    while pipeline.isRunning():

        frames = {}

        # =====================================
        # GET CAMERA FRAMES
        # =====================================

        for name, queue in outputQueues.items():

            videoIn = queue.get()

            frame = videoIn.getCvFrame()

            frames[name] = frame

            cv2.imshow(name, frame)

        # =====================================
        # USE FIRST CAMERA
        # =====================================

        first_cam = list(frames.keys())[0]

        frame = frames[first_cam]

        # =====================================
        # BUTTON 1 - SCENE
        # =====================================

        if GPIO.input(BUTTON_SCENE) == GPIO.LOW:

            print("\n[SCENE BUTTON PRESSED]")

            image_path = "scene.jpg"

            cv2.imwrite(image_path, frame)

            if current_language == "vi":
                prompt = SCENE_PROMPT_VI
            else:
                prompt = SCENE_PROMPT_EN

            try:

                output = ask_gemma(
                    image_path,
                    prompt
                )

                print("\n========== SCENE OUTPUT ==========\n")
                print(output)

                speak_text(output)

            except Exception as e:

                print("Gemma Error:", e)

            time.sleep(1)

        # =====================================
        # BUTTON 2 - OCR
        # =====================================

        elif GPIO.input(BUTTON_OCR) == GPIO.LOW:

            print("\n[OCR BUTTON PRESSED]")

            image_path = "ocr.jpg"

            cv2.imwrite(image_path, frame)

            if current_language == "vi":
                prompt = OCR_PROMPT_VI
            else:
                prompt = OCR_PROMPT_EN

            try:

                output = ask_gemma(
                    image_path,
                    prompt
                )

                print("\n========== OCR OUTPUT ==========\n")
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

            cv2.imwrite(image_path, frame)

            if current_language == "vi":
                prompt = MONEY_PROMPT_VI
            else:
                prompt = MONEY_PROMPT_EN

            try:

                output = ask_gemma(
                    image_path,
                    prompt
                )

                print("\n========== MONEY OUTPUT ==========\n")
                print(output)

                speak_text(output)

            except Exception as e:

                print("Gemma Error:", e)

            time.sleep(1)

        # =====================================
        # BUTTON 4 - LANGUAGE SWITCH
        # =====================================

        elif GPIO.input(BUTTON_LANGUAGE) == GPIO.LOW:

            if current_language == "en":

                current_language = "vi"

                print("\nSwitched to Vietnamese")

                speak_text("Đã chuyển sang tiếng Việt")

            else:

                current_language = "en"

                print("\nSwitched to English")

                speak_text("Switched to English")

            time.sleep(1)

        # =====================================
        # QUIT
        # =====================================

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

# =========================================
# CLEANUP
# =========================================

GPIO.cleanup()

cv2.destroyAllWindows()


