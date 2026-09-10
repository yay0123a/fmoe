import os
import csv
import numpy as np
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
import warnings
from Metric import *

warnings.filterwarnings("ignore")

# ==================== 配置路径 ====================
#root = '..'
fused_dir = os.path.join('OUTPUT', 'test10_min')
#data/refine_SPGFusion_main/OUTPUT/Time_test2
ir_dir = os.path.join('assets', 'data', 'MSRS', 'Test_ir1')
vi_dir = os.path.join('assets', 'data', 'MSRS', 'Test_vi1')

csv_name = 'test_time1_metric.csv'
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def evaluation_one(ir_name, vi_name, f_name):
    f_img = Image.open(f_name).convert('L')
    ir_img = Image.open(ir_name).convert('L')
    vi_img = Image.open(vi_name).convert('L')

    f_img_int = np.array(f_img).astype(np.int32)
    f_img_double = np.array(f_img).astype(np.float32)
    ir_img_int = np.array(ir_img).astype(np.int32)
    ir_img_double = np.array(ir_img).astype(np.float32)
    vi_img_int = np.array(vi_img).astype(np.int32)
    vi_img_double = np.array(vi_img).astype(np.float32)

    EN = EN_function(f_img_int)
    MI = MI_function(ir_img_int, vi_img_int, f_img_int, gray_level=256)
    SF = SF_function(f_img_double)
    SD = SD_function(f_img_double)
    AG = AG_function(f_img_double)
    PSNR = PSNR_function(ir_img_double, vi_img_double, f_img_double)
    MSE = MSE_function(ir_img_double, vi_img_double, f_img_double)
    VIF = VIF_function(ir_img_double, vi_img_double, f_img_double)
    CC = CC_function(ir_img_double, vi_img_double, f_img_double)
    SCD = SCD_function(ir_img_double, vi_img_double, f_img_double)
    Qabf = Qabf_function(ir_img_double, vi_img_double, f_img_double)
    Nabf = Nabf_function(ir_img_double, vi_img_double, f_img_double)
    SSIM = SSIM_function(ir_img_double, vi_img_double, f_img_double)
    MS_SSIM = MS_SSIM_function(ir_img_double, vi_img_double, f_img_double)
    return EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM


def main():
    fused_files = [f for f in natsorted(os.listdir(fused_dir))
                   if f.lower().endswith(IMAGE_EXTS)]

    if not fused_files:
        print(f"未在 {fused_dir} 中找到图像文件")
        return

    results = []
    skipped = 0
    for item in tqdm(fused_files, desc="计算指标"):
        f_name = os.path.join(fused_dir, item)
        ir_name = os.path.join(ir_dir, item)
        vi_name = os.path.join(vi_dir, item)

        if not os.path.exists(ir_name):
            print(f"  跳过 {item}: IR 源图像缺失 ({ir_name})")
            skipped += 1
            continue
        if not os.path.exists(vi_name):
            print(f"  跳过 {item}: VI 源图像缺失 ({vi_name})")
            skipped += 1
            continue

        EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = \
            evaluation_one(ir_name, vi_name, f_name)

        results.append({
            'Image': item,
            'EN': round(EN, 4),
            'MI': round(MI, 4),
            'SF': round(SF, 4),
            'AG': round(AG, 4),
            'SD': round(SD, 4),
            'CC': round(CC, 4),
            'SCD': round(SCD, 4),
            'VIF': round(VIF, 4),
            'MSE': round(MSE, 4),
            'PSNR': round(PSNR, 4),
            'Qabf': round(Qabf, 4),
            'Nabf': round(Nabf, 4),
            'SSIM': round(SSIM, 4),
            'MS_SSIM': round(MS_SSIM, 4),
        })

    if not results:
        print("没有成功计算任何图像")
        return

    # 添加均值与标准差行
    metric_keys = ['EN', 'MI', 'SF', 'AG', 'SD', 'CC', 'SCD', 'VIF',
                   'MSE', 'PSNR', 'Qabf', 'Nabf', 'SSIM', 'MS_SSIM']

    mean_row = {'Image': 'mean'}
    std_row = {'Image': 'std'}
    for key in metric_keys:
        vals = [r[key] for r in results]
        mean_row[key] = round(np.mean(vals), 4)
        std_row[key] = round(np.std(vals), 4)
    results.append(mean_row)
    results.append(std_row)

    # 写入 CSV
    #output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_name)
    output_path = os.path.join(os.path.abspath(fused_dir), csv_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ['Image'] + metric_keys
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"完成: {len(results) - 2} 张图像, 跳过 {skipped} 张")
    print(f"结果已保存至 {output_path}")


if __name__ == '__main__':
    main()
