import subprocess

MODEL = "/home/amber/gemma/depthai-core/code/models/gemma4_currency_Q4.gguf"
MMPROJ = "/home/amber/gemma/depthai-core/code/models/mmproj-F16.gguf"

LLAMA_BIN = "/home/amber/gemma/depthai-core/code/llama.cpp/build/bin/llama-mtmd-cli"

while True:

    image_path = input("Image path: ")

    if image_path.lower() == "exit":
        break

    prompt = input("Prompt: ")

    command = [
        LLAMA_BIN,

        "--jinja",

        "--no-warmup",

        "-m", MODEL,
        "--mmproj", MMPROJ,

        "--image", image_path,

        "-ngl", "50",

        "-c", "512",

        "-n", "100",

        "-p", prompt
    ]

    print("\nRunning Gemma4...\n")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    print("\n========== OUTPUT ==========\n")

    print(result.stdout)
