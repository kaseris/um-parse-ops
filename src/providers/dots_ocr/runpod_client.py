import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from src.providers.dots_ocr.image_utils import PILimage_to_base64



load_dotenv()


def _get_api_key(explicit_key: Optional[str] = None) -> str:
    if explicit_key:
        return explicit_key
    # Prefer the key already used in inference.py, fall back to a common name
    key = os.getenv("DOTS_OCR_TEST_KEY") or os.getenv("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing API key. Set DOTS_OCR_TEST_KEY or RUNPOD_API_KEY in your environment/.env."
        )
    return key


def _build_url(endpoint_id: str, protocol: str = "https", host: str = "api.runpod.ai") -> str:
    return f"{protocol}://{host}/v2/{endpoint_id}/run"


def runpod_infer_prompt(
    prompt: str,
    *,
    endpoint_id: str = "zd3u5itgf0mplf",
    api_key: Optional[str] = None,
    timeout: int = 600,
) -> Dict[str, Any]:
    """
    Send a Runpod-compatible request with only a text prompt.

    This matches the documented format: { "input": { "prompt": "..." } }.
    Returns the parsed JSON response (raises for non-2xx).
    """
    url = _build_url(endpoint_id)
    key = _get_api_key(api_key)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    payload = {"input": {"prompt": prompt}}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def runpod_infer_image(
    image: "Image.Image",
    prompt: str,
    *,
    endpoint_id: str = "zd3u5itgf0mplf",
    api_key: Optional[str] = None,
    include_image_key: bool = True,
    timeout: int = 600,
) -> Dict[str, Any]:
    """
    Send a Runpod-compatible request with a prompt and optional base64 image.

    - By default, sends: {"input": {"prompt": full_prompt, "image": data_url}}
    - If your Runpod handler ignores/doesn't need the image, set include_image_key=False
      to only send the text prompt in the documented format.
    """
    if Image is None:
        raise RuntimeError("Pillow is required for image handling but is not available.")

    url = _build_url(endpoint_id)
    key = _get_api_key(api_key)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    image_data_url = PILimage_to_base64(image)
    full_prompt = f"<|img|><|imgpad|><|endofimg|>{prompt}"

    input_payload: Dict[str, Any] = {"prompt": full_prompt}
    if include_image_key:
        # Many serverless handlers expect a separate image field; if your handler
        # only wants the prompt, disable this via include_image_key=False.
        input_payload["image"] = image_data_url

    payload = {"input": input_payload}

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()



