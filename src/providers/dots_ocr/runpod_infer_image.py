import os
from dotenv import load_dotenv
from PIL import Image

from src.providers.dots_ocr.runpod_client import runpod_infer_image


load_dotenv()


def main():
    # Hardcoded demo image path relative to repo root
    image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                              "55-02-135-00-31_ΑΚΝ32911764.jpg")

    # Prompt copied from inference.py example (layout_all)
    prompt = (
        "Please output the layout information from the PDF image, including each layout "
        "element's bbox, its category, and the corresponding text content within the bbox.\n\n"
        "1. Bbox format: [x1, y1, x2, y2]\n\n"
        "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', "
        "'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n\n"
        "3. Text Extraction & Formatting Rules:\n"
        "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
        "    - Formula: Format its text as LaTeX.\n"
        "    - Table: Format its text as HTML.\n"
        "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
        "4. Constraints:\n"
        "    - The output text must be the original text from the image, with no translation.\n"
        "    - All layout elements must be sorted according to human reading order.\n\n"
        "5. Final Output: The entire output must be a single JSON object."
    )

    # Open image
    img = Image.open(image_path)

    # endpoint id from your request
    endpoint_id = "zd3u5itgf0mplf"

    # If your Runpod handler does NOT accept an 'image' field in input,
    # set include_image_key=False to only send the text prompt.
    include_image_key = True

    try:
        resp = runpod_infer_image(
            image=img,
            prompt=prompt,
            endpoint_id=endpoint_id,
            include_image_key=include_image_key,
        )
        print(resp)
    except Exception as e:
        print(f"Runpod request failed: {e}")


if __name__ == "__main__":
    main()

