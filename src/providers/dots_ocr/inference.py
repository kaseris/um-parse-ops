import requests
from src.providers.dots_ocr.image_utils import PILimage_to_base64
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

from PIL import Image

def inference(
        image,
        prompt, 
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=32768,
        model_name='rednote-hilab/dots.ocr',
        ):
    ENDPOINT_ID = os.getenv("VLLM_ENDPOINT_ID")
    DOTS_OCR_API_KEY = os.getenv("DOTS_OCR_TEST_KEY")
    addr = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1"
    
    client = OpenAI(api_key=DOTS_OCR_API_KEY, base_url=addr)
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url":  PILimage_to_base64(image)},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}  # if no "<|img|><|imgpad|><|endofimg|>" here,vllm v1 will add "\n" here
            ],
        }
    )
    try:
        response = client.chat.completions.create(
            messages=messages, 
            model=model_name, 
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p)
        response = response.choices[0].message.content
        return response
    except requests.exceptions.RequestException as e:
        print(f"request error: {e}")
        return None

if __name__ == "__main__":
    dict_promptmode_to_prompt = {
    # prompt_layout_all_en: parse all layout info in json format.
    "prompt_layout_all_en": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

    1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
""",

    # prompt_layout_only_en: layout detection
    "prompt_layout_only_en": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",

    # prompt_layout_only_en: parse ocr text except the Page-header and Page-footer
    "prompt_ocr": """Extract the text content from this image.""",

    # prompt_grounding_ocr: extract text content in the given bounding box
    "prompt_grounding_ocr": """Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).\nBounding Box:\n""",

    # "prompt_table_html": """Convert the table in this image to HTML.""",
    # "prompt_table_latex": """Convert the table in this image to LaTeX.""",
    # "prompt_formula_latex": """Convert the formula in this image to LaTeX.""",
}
    img = Image.open("fd3be268-5996-4d42-8156-41ad23399b1a%2FPresentation_speech_(draft).jpg")
    prompt = dict_promptmode_to_prompt["prompt_layout_all_en"]
    output = inference(
        image=img,
        prompt=prompt,
    )
    print(output)