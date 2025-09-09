import base64
import io
import traceback
from PIL import Image
import runpod

# Global variables for caching model and processor
model = None
processor = None

# Your prompt
prompt = """
You are a highly accurate document parser for Jordanian National IDs and Electricity Bills. 

Your task is to analyze the provided document image and output a valid JSON containing the extracted fields.

Instructions:

1. Determine the document type: "Jordanian National ID", "Jordanian Electricity Bill", or "None".

2. Extract fields only relevant to the detected document type.

For Jordanian National ID:
- Name_Arabic: الاسم in Arabic
- English_Name: full name in English
- National_Number: الرقم الوطني
- Gender: الجنس (Male/Female)
- DateOfBirth: تاريخ الولادة (format YYYY-MM-DD)
- PlaceOfBirth: مكان الولادة
- Mother_Name: الأم اسم (optional, only if present)

For Electricity Bill:
- Name_Subscriber: اسم المشترك
- Reading_Date: تاريخ القراءة (format YYYY-MM-DD)
- Meter_Number: رقم العداد
- Required_Amount: المطلوب (optional, only if present)
- Total_Debts: مجموع قيمة الذمم (optional, only if present)

Rules:
- Dates must use ISO 8601 format: YYYY-MM-DD.
- Optional fields must only appear if present in the document.
- If the document is neither an ID nor a bill, output:
  {
      "Document_Type": "None",
      "Reason": "Explain why the document could not be processed (e.g., unreadable, back side, or foreign ID)"
  }
- Output must be valid JSON and nothing else.
"""

def handler(event):
    """
    RunPod serverless handler
    Expected JSON input:
    {
        "image_base64": "<base64 string of document image>"
    }
    """
    global model, processor

    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        # Lazy-load the model and processor
        if model is None or processor is None:
            print("Loading model and processor... (this may take a few minutes for 32B model)")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-32B-Instruct",
                torch_dtype="auto",
                device_map="auto",           # Automatically place layers on available devices                
            )
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")
            print("Model and processor loaded successfully.")

        # Decode base64 image
        image_data = base64.b64decode(event["input"]["image_base64"])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Prepare messages for processor
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare model inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=512)

        # Trim input ids from generated ids
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode generated text
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        result = {"status": "success", "result": output_text[0]}
        print("Handler Output:", result)  # <-- prints output to Python console

        return result

    except Exception as e:
        error_info = {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }
        print("Handler Error:", error_info)  # <-- prints error to console
        return error_info

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})