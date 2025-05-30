#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台灣高鐵驗證碼識別 CLI 工具

使用方式:
    python predict_captcha.py <圖片路徑>

範例:
    python predict_captcha.py ./tmp_code.jpg
    python predict_captcha.py captcha/123.jpg
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image
import argparse
import tempfile

try:
    from keras.models import load_model
    from preprocessBatch import preprocessing
except ImportError as e:
    print(f"錯誤: 缺少必要的套件 - {e}")
    print("請執行: uv pip install -r requirements.txt")
    sys.exit(1)

# 設定參數
MODEL_PATH = "09-0.09-0.10.hdf5"

class CaptchaPredictor:
    def __init__(self, model_path=MODEL_PATH):
        """初始化驗證碼預測器"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型檔案: {model_path}")

        print("載入模型中...")
        self.model = load_model(model_path)
        print("模型載入完成")

    def predict(self, image_path):
        """
        預測驗證碼

        Args:
            image_path (str): 圖片檔案路徑

        Returns:
            str: 預測的驗證碼結果
        """
        allowedChars = '234579ACFHKMNPQRTYZ'
        WIDTH = 140
        HEIGHT = 48


        img = Image.open(image_path)
        img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
        # convert rgba to rgb
        img = img.convert('RGB')

        # Save processed image to temp path
        temp_dir = tempfile.gettempdir()
        temp_path_processed = os.path.join(temp_dir, 'captcha.jpg')
        img.save(temp_path_processed, "JPEG")

        preprocessing(temp_path_processed, temp_path_processed)

        train_data = np.stack([np.array(cv2.imread(temp_path_processed))/255.0])
        model = load_model("thsrc_cnn_model.hdf5")
        prediction = model.predict(train_data)

        predict_captcha = ''
        for predict in prediction:
            value = np.argmax(predict[0])
            predict_captcha += allowedChars[value]

        # Remove temp files
        os.remove(temp_path_processed)

        # 將結果儲存到檔案.txt, 要存在與圖片同一目錄下
        result_file = os.path.splitext(image_path)[0] + '_result.txt'
        with open(result_file, 'w') as f:
            f.write(predict_captcha)

        return predict_captcha

def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="台灣高鐵驗證碼識別工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  %(prog)s ./tmp_code.jpg
  %(prog)s captcha/123.jpg
  %(prog)s /path/to/captcha.png
        """
    )

    parser.add_argument(
        'image_path',
        help='驗證碼圖片檔案路徑'
    )

    parser.add_argument(
        '-m', '--model',
        default=MODEL_PATH,
        help=f'模型檔案路徑 (預設: {MODEL_PATH})'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='顯示詳細資訊'
    )

    args = parser.parse_args()

    try:
        # 初始化預測器
        predictor = CaptchaPredictor(args.model)

        if args.verbose:
            print(f"處理圖片: {args.image_path}")

        # 進行預測
        result = predictor.predict(args.image_path)

        if args.verbose:
            print(f"預測結果: {result}")
        else:
            print(result)

    except FileNotFoundError as e:
        print(f"檔案錯誤: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"執行錯誤: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()