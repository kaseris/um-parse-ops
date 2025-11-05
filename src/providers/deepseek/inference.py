from fitz import extra
from openai import OpenAI

from PIL import Image
from src.settings import settings
from src.providers.dots_ocr.image_utils import PILimage_to_base64


def inference(image: Image):
    client = OpenAI(
        api_key=settings.RUNPOD_API_KEY,
        base_url=f"https://api.runpod.ai/v2/{settings.DEEPSEEK_OCR_ENDPOINT_ID}/openai/v1",
        timeout=3600
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": PILimage_to_base64(image)}
                }
            ]
        },
        {
            "type": "text",
            "text": "Free OCR."
        }
    ]

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-OCR",
        messages=messages,
        max_tokens=32768,
        temperature=0.0,
        extra_body={
        "skip_special_tokens": False,
        "vllm_xargs": {
            "ngram_size": 30,
            "window_size": 90,
            # whitelist: <td>, </td>
            "whitelist_token_ids": [128821, 128822],
        },
    },
    )
    return response

if __name__ == "__main__":
    img = Image.open("/Users/michaliskaseris/dev/um-parse-ops/55-02-135-00-31_ΑΚΝ32911764.jpg")
    response = inference(img)
    print(response)