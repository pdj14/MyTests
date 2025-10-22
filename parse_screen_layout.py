"""Utility script to parse screen layouts using Pix2Struct."""

from typing import Dict

import torch
from PIL import Image
from transformers import (
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
)

# Shared identifier for both the processor and model so we only configure it once.
MODEL_ID = "google/pix2struct-screen-parsing-base"


def parse_screen_layout(image_path: str) -> str:
    """Parse the layout of a screen image using the Pix2Struct model."""
    # Load the Pix2Struct processor and model optimized for screen parsing.
    processor = Pix2StructProcessor.from_pretrained(MODEL_ID)
    model = Pix2StructForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    # Move the model to GPU if available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model = model.to(device)
    else:
        # CPU execution does not support bfloat16 well, so fall back to float32.
        model = model.to(device=device, dtype=torch.float32)
    model.eval()

    # Open the target image using PIL.
    with Image.open(image_path) as pil_image:
        image = pil_image.convert("RGB")

    # Prepare the image for the model using the processor.
    model_inputs: Dict[str, torch.Tensor] = processor(
        images=image,
        return_tensors="pt",
    )
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

    # Generate the parsed layout tokens using the model.
    with torch.inference_mode():
        generated_ids = model.generate(**model_inputs)

    # Decode the generated tokens into a human-readable layout description.
    parsed_layout = processor.decode(generated_ids[0], skip_special_tokens=True)

    return parsed_layout


if __name__ == "__main__":
    # Replace "path/to/your/screenshot.png" with the actual path to your screenshot.
    sample_image_path = "path/to/your/screenshot.png"

    # Run the parser and print the resulting layout description.
    print(parse_screen_layout(sample_image_path))
