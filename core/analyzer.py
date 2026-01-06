import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel
import json

image_path = input("Enter image file path: ") 
image = Image.open(image_path).convert("RGB")

user_text = input("Describe your design intent: ")

MODEL_ID = "google/siglip2-large-patch16-512"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

DESIGN_AXES = {
    "youthful": "a youthful and energetic design",
    "mature": "a mature and sophisticated design",
    "energetic": "an energetic and dynamic design",
    "modern": "a modern and clean design",
    "classic": "a classic and traditional design",
    "minimal": "a minimal and simple design",
    "complex": "a complex and detailed design",
    "friendly": "a friendly and approachable brand design",
    "luxurious": "a luxurious and premium brand design",
    "playful": "a playful and fun design",
    "serious": "a serious and professional design",
    "soft": "a soft and gentle visual style",
    "bold": "a bold and strong visual style",
    "warm": "a warm color palette",
    "cool": "a cool color palette"
}

axis_texts = list(DESIGN_AXES.values())

inputs = processor(
    images=image,
    text=axis_texts,
    return_tensors="pt",
    padding=True
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits_per_image.squeeze(0)

scores = F.softmax(logits, dim=0)

alignment_scores = {
    axis: float(score)
    for axis, score in zip(DESIGN_AXES.keys(), scores)
}

alignment_scores = dict(
    sorted(alignment_scores.items(), key=lambda x: x[1], reverse=True)
)

print("\nAlignment Scores (0~1):")
for k, v in alignment_scores.items():
    print(f"{k:10s}: {v:.3f}")


alignment_scores = {
    axis: round(float(score), 3)
    for axis, score in zip(DESIGN_AXES.keys(), scores)
}

output = {
    "user_intent": user_text,
    "image_analysis": alignment_scores
}

with open("design_analysis.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
