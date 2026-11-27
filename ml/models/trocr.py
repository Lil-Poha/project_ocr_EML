import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class TrOCRModel:
    """Обёртка над TrOCR из transformers со встроенным препроцессингом."""

    def __init__(self, model_name="microsoft/trocr-base-printed", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _load_image(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    @torch.no_grad()
    def predict(self, image, max_new_tokens=64):
        return self.predict_batch([image], batch_size=1, max_new_tokens=max_new_tokens)[0]

    @torch.no_grad()
    def predict_batch(self, images, batch_size=16, max_new_tokens=64):
        preds = []
        for i in range(0, len(images), batch_size):
            chunk = [self._load_image(im) for im in images[i : i + batch_size]]
            pixel_values = self.processor(
                images=chunk, return_tensors="pt"
            ).pixel_values.to(self.device)
            generated_ids = self.model.generate(
                pixel_values, max_new_tokens=max_new_tokens
            )
            texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            preds.extend(t.strip() for t in texts)
        return preds


if __name__ == "__main__":
    # Тестовая картинка в памяти, чтобы проверить запуск модели без датасета
    img = Image.new("RGB", (520, 120), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=48)
    draw.text((30, 35), "Hello TrOCR 2026", fill="black", font=font)

    print("Загружаю TrOCR...")
    ocr = TrOCRModel("microsoft/trocr-base-printed")
    print("Устройство:", ocr.device)
    print("Распознанный текст:", ocr.predict(img))