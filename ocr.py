import os
from PIL import Image
from mistralai import Mistral
from mistralai.models import OCRResponse
from pathlib import Path
import os
import base64

# 設定圖片資料夾路徑
img_dir = './pre-error-log-img'
API_KEY = "XXXXX"

def parse_ocr_results(ocr_response: OCRResponse) -> str:
    markdown_str = ocr_response.pages[0].markdown
    clean_str = markdown_str.lstrip('#').replace(' ', '')
    return clean_str


def process_pdf(jpg_path: str, api_key: str):
    client = Mistral(api_key=api_key)

    file = Path(jpg_path)
    if not file.is_file():
        raise FileNotFoundError(f"jpg 文件不存在: {jpg_path}")

    base64_image = base64.b64encode(file.read_bytes()).decode('utf-8')

    client = Mistral(api_key=api_key)

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}"
        },
        include_image_base64=True
    )

    return parse_ocr_results(ocr_response)


# 遍歷資料夾中的所有 .jpg 檔案
for filename in os.listdir(img_dir):
    if filename.lower().endswith('.jpg') and not filename.startswith('ZS'):
        file_path = os.path.join(img_dir, filename)

        # 讀取圖片
        image = Image.open(file_path)

        code = process_pdf(file_path, API_KEY)

        new_name = f'ZS-{code}-{filename}'
        new_path = os.path.join(img_dir, new_name)

        # 檢查是否為四位數字
        if code is not None and len(code) == 4:
        # 重新命名圖片
            os.rename(file_path, new_path)
            print(f'✔️ 已重新命名：{filename} → {new_name}')