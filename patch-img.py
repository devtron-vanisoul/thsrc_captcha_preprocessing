import os
import shutil
from preprocess import preprocessing

# 設定來源與目的資料夾路徑
source_folder = './error-log-img'
destination_folder = './pre-error-log-img'

# 先刪除 destination_folder 資料夾, 不存在則忽略
if os.path.exists(destination_folder):
    shutil.rmtree(destination_folder)

# 確保目的資料夾存在
os.makedirs(destination_folder, exist_ok=True)

# 遍歷來源資料夾中的所有檔案
for filename in os.listdir(source_folder):
    if filename.lower().endswith('.jpg'):

        # 建立新的檔案名稱
        new_filename = f"{filename}"
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, new_filename)

        # 複製檔案
        # shutil.copy2(src_path, dst_path)
        preprocessing(src_path, dst_path)
        print(f"已複製：{src_path} -> {dst_path}")
