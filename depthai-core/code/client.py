import requests
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

while True:

    image_path = input("Image path: ")

    if image_path.lower() == "exit":
        break

    prompt = input("Prompt: ")

    image_base64 = encode_image(image_path)

    payload = {
        "messages": [
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
        "max_tokens": 50,
        "temperature": 0.2
    }

    response = requests.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json=payload
    )

    result = response.json()

    print("\n========== OUTPUT ==========\n")

    print(result["choices"][0]["message"]["content"])
