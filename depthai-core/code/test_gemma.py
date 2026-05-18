import subprocess

MODEL = "/home/amber/gemma/depthai-core/code/models/gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ = "/home/amber/gemma/depthai-core/code/models/mmproj-F16.gguf"
IMAGE = "/home/amber/gemma/depthai-core/code/images.jpeg"

command = [
    "/home/amber/gemma/depthai-core/code/llama.cpp/build/bin/llama-mtmd-cli",

    "--jinja",

    "-m", MODEL,
    "--mmproj", MMPROJ,
    "--image", IMAGE,

    "-ngl", "5",

    "-c", "512",
  
    "-n", "100",

    "-p", "Read all text in this images with its language"
]

result = subprocess.run(
    command,
    capture_output=True,
    text=True
)

print(result.stdout)
print(result.stderr)
