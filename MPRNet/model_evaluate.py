import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from glob import glob
import pandas as pd
from tqdm import tqdm


def calculate_metrics(gt_img, pred_img):
    # 确保图像是float类型
    gt_img = gt_img.astype(np.float32) / 255.0
    pred_img = pred_img.astype(np.float32) / 255.0

    # 计算PSNR
    psnr_value = psnr(gt_img, pred_img, data_range=1.0)

    # 计算SSIM
    # 对于彩色图像，使用multichannel=True
    if len(gt_img.shape) == 3 and gt_img.shape[2] == 3:
        ssim_value = ssim(gt_img, pred_img, channel_axis=2, data_range=1.0)
    else:
        ssim_value = ssim(gt_img, pred_img, data_range=1.0)

    return psnr_value, ssim_value


def evaluate_dataset(dataset_name, gt_path, pred_path, result_dir="result/evaluation"):
    print(f"正在评估数据集: {dataset_name}")

    # 根据数据集类型获取文件列表
    if dataset_name == 'Test2800':
        # Test2800的文件名格式为"数字_数字"
        gt_files = sorted(glob(os.path.join(gt_path, "*.png")))
        if not gt_files:
            gt_files = sorted(glob(os.path.join(gt_path, "*.jpg")))
    else:
        # 其他数据集的文件名为数字
        gt_files = []
        for ext in ['*.png', '*.jpg', '*.bmp', '*.jpeg']:
            files = sorted(glob(os.path.join(gt_path, ext)),
                           key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            gt_files.extend(files)

    if not gt_files:
        print(f"警告: 在 {gt_path} 中未找到ground truth图像")
        return None, 0, 0

    results = []
    total_psnr = 0
    total_ssim = 0
    valid_count = 0

    for gt_file in tqdm(gt_files, desc=f"处理{dataset_name}"):
        # 获取文件名
        filename = os.path.basename(gt_file)
        name_without_ext = os.path.splitext(filename)[0]

        # 构建预测文件路径
        pred_file = os.path.join(pred_path, filename)

        # 检查预测文件是否存在
        if not os.path.exists(pred_file):
            # 尝试不同的扩展名
            found = False
            for ext in ['.png', '.jpg', '.bmp', '.jpeg']:
                alt_file = os.path.join(pred_path, name_without_ext + ext)
                if os.path.exists(alt_file):
                    pred_file = alt_file
                    found = True
                    break

            if not found:
                print(f"警告: 预测文件 {filename} 不存在，跳过")
                continue

        # 读取图像
        gt_img = cv2.imread(gt_file)
        pred_img = cv2.imread(pred_file)

        if gt_img is None or pred_img is None:
            print(f"警告: 无法读取图像 {filename}，跳过")
            continue

        # 确保图像大小相同
        if gt_img.shape != pred_img.shape:
            print(f"警告: 图像 {filename} 尺寸不匹配，调整预测图像尺寸")
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

        # 计算指标
        psnr_value, ssim_value = calculate_metrics(gt_img, pred_img)

        # 添加到结果
        results.append({
            'filename': filename,
            'psnr': psnr_value,
            'ssim': ssim_value
        })

        total_psnr += psnr_value
        total_ssim += ssim_value
        valid_count += 1

    if valid_count == 0:
        print(f"错误: 数据集 {dataset_name} 没有有效的图像对")
        return None, 0, 0

    # 创建DataFrame
    results_df = pd.DataFrame(results)

    # 计算平均值
    avg_psnr = total_psnr / valid_count
    avg_ssim = total_ssim / valid_count

    print(f"{dataset_name} 评估完成:")
    print(f"  处理图像数量: {valid_count}")
    print(f"  平均PSNR: {avg_psnr:.4f}")
    print(f"  平均SSIM: {avg_ssim:.6f}")
    print()

    return results_df, avg_psnr, avg_ssim


def evaluate_all_datasets():
    # 数据集列表
    datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800', 'RealRain-1k-L', 'RealRain-1k-H']

    # 结果保存目录
    result_dir = "../evaluate_result/"

    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)

    print(f"评估结果将保存到: {result_dir}/")
    print()

    all_results = {}
    summary = []

    for dataset in datasets:
        # 构建路径
        gt_path = f"dataset/test/{dataset}/norain"
        pred_path = f"test_result/{dataset}"

        # 检查路径是否存在
        if not os.path.exists(gt_path):
            print(f"警告: ground truth路径不存在: {gt_path}")
            continue

        if not os.path.exists(pred_path):
            print(f"警告: 预测路径不存在: {pred_path}")
            continue

        # 评估数据集
        results_df, avg_psnr, avg_ssim = evaluate_dataset(dataset, gt_path, pred_path, result_dir)

        if results_df is not None:
            # 保存详细结果到指定目录
            detail_file = os.path.join(result_dir, f"evaluation_results_{dataset}_detailed.csv")
            results_df.to_csv(detail_file, index=False, encoding='utf-8-sig')
            print(f"详细结果已保存到: {detail_file}")

            # 保存每个数据集的汇总到指定目录
            dataset_file = os.path.join(result_dir, f"evaluation_results_{dataset}.csv")
            dataset_summary = pd.DataFrame([{
                'dataset': dataset,
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim,
                'num_images': len(results_df)
            }])
            dataset_summary.to_csv(dataset_file, index=False, encoding='utf-8-sig')
            print(f"数据集汇总已保存到: {dataset_file}")

            # 添加到总结果
            all_results[dataset] = results_df
            summary.append({
                'dataset': dataset,
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim,
                'num_images': len(results_df)
            })

    # 保存总汇总到指定目录
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_file = os.path.join(result_dir, "evaluation_summary.csv")
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 50)
        print("评估汇总:")
        print("=" * 50)
        print(summary_df.to_string(index=False))
        print("=" * 50)
        print(f"\n汇总结果已保存到: {summary_file}")

        # 同时保存为Markdown格式，便于查看
        markdown_file = os.path.join(result_dir, "evaluation_summary.md")
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write("# 去雨模型评估结果\n\n")
            f.write("| 数据集 | 平均PSNR | 平均SSIM | 图像数量 |\n")
            f.write("|--------|----------|----------|----------|\n")
            for row in summary:
                f.write(f"| {row['dataset']} | {row['avg_psnr']:.4f} | {row['avg_ssim']:.6f} | {row['num_images']} |\n")
        print(f"Markdown格式汇总已保存到: {markdown_file}")

        # 保存为文本格式
        txt_file = os.path.join(result_dir, "evaluation_summary.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("去雨模型评估结果\n")
            f.write("=" * 50 + "\n")
            for row in summary:
                f.write(f"数据集: {row['dataset']}\n")
                f.write(f"  平均PSNR: {row['avg_psnr']:.4f}\n")
                f.write(f"  平均SSIM: {row['avg_ssim']:.6f}\n")
                f.write(f"  图像数量: {row['num_images']}\n")
                f.write("-" * 30 + "\n")
        print(f"文本格式汇总已保存到: {txt_file}")

    return all_results, summary


if __name__ == "__main__":
    print("开始评估去雨模型效果...")
    print("=" * 50)

    # 检查所需库是否安装
    try:
        import cv2
        import numpy as np
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        import pandas as pd
        from tqdm import tqdm
    except ImportError as e:
        print(f"错误: 缺少必要的库: {e}")
        print("请安装以下库:")
        print("pip install opencv-python numpy scikit-image pandas tqdm")
        exit(1)

    # 运行评估
    all_results, summary = evaluate_all_datasets()

    print("\n评估完成！")