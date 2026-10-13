"""
Image processing module for GenieBot
Handles image captioning and tag extraction via an OpenRouter vision model
(no local model weights, keeps the process lightweight for free-tier hosting).
"""

import base64
import io
from typing import Dict, List

import requests
from PIL import Image

from utils.logger import setup_logger

logger = setup_logger(__name__)

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_PIXELS = 25_000_000  # guards against decompression bombs


class ImageProcessor:
    """Process images for captioning and tagging using a hosted vision LLM."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str = "z-ai/glm-4.6v",
        site_url: str = "https://github.com/iamvisheshsrivastava/geniebot",
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.timeout = timeout

    def _is_model_available(self) -> bool:
        return bool(self.api_key)

    def _load_image(self, image_input) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input
        if isinstance(image_input, (bytes, bytearray)):
            if len(image_input) > MAX_IMAGE_BYTES:
                raise ValueError("Image is too large (max 10 MB)")
            image = Image.open(io.BytesIO(bytes(image_input)))
        elif hasattr(image_input, "read"):
            data = image_input.read()
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("Image is too large (max 10 MB)")
            image = Image.open(io.BytesIO(data))
        else:
            raise ValueError("Invalid image input")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Image dimensions are too large")
        return image

    def _to_data_url(self, image: Image.Image) -> str:
        max_size = 768
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def generate_caption(self, image_input) -> str:
        if not self._is_model_available():
            logger.error("Vision model not available (missing OPENROUTER_API_KEY)")
            return "Image processing model not available"

        try:
            image = self._load_image(image_input)
            data_url = self._to_data_url(image)

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in one concise sentence, suitable as a caption.",
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                # Reasoning vision models (e.g. GLM-4.6v) spend part of this
                # budget on hidden reasoning before the visible caption.
                "max_tokens": 400,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "Content-Type": "application/json",
            }

            response = requests.post(self.API_URL, headers=headers, json=payload, timeout=self.timeout)
            if response.status_code != 200:
                logger.error(f"Vision API error: {response.status_code} - {response.text[:300]}")
                return "Error: Could not process image"

            result = response.json()
            caption = (result["choices"][0]["message"].get("content") or "").strip()
            if not caption:
                logger.warning("Vision model returned empty/null content (likely ran out of max_tokens on reasoning)")
                return "Error: Could not generate a caption for this image"
            logger.debug(f"Generated caption: {caption}")
            return caption

        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            if isinstance(e, ValueError) and "too large" in str(e):
                return f"Error: {e}"
            return "Error: Could not process image"

    def extract_tags(self, caption: str, num_tags: int = 3) -> List[str]:
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'has',
            'have', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they'
        }

        words = caption.lower().split()
        tags = []

        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word and clean_word not in stop_words and len(clean_word) > 2:
                if clean_word not in tags:
                    tags.append(clean_word)
                    if len(tags) == num_tags:
                        break

        return tags

    def process_image(self, image_input) -> Dict[str, any]:
        logger.info("Starting image processing")
        try:
            caption = self.generate_caption(image_input)
            if caption.startswith("Error:") or caption == "Image processing model not available":
                return {"caption": caption, "tags": [], "success": False, "error": caption}
            tags = self.extract_tags(caption, num_tags=3)
            result = {"caption": caption, "tags": tags, "success": True}
            logger.info(f"Image processing complete: {caption[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return {
                "caption": "Error processing image",
                "tags": [],
                "success": False,
                "error": str(e),
            }

    def is_available(self) -> bool:
        return self._is_model_available()
