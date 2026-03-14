import os


def calculate_mae(gt_file, pred_file):
    """
    计算两个文件之间的平均绝对误差（MAE）
    参数：
        gt_file: 真实值文件路径（val.txt格式：filename\tage）
        pred_file: 预测值文件路径（pred_result.txt格式：filename\tage）
    返回：
        mae: 平均绝对误差
        matched_count: 成功匹配的样本数
    """
    # 读取真实值
    gt_dict = {}
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split('\t')
                if len(parts) < 2:
                    print(f"[WARN] {gt_file} 第{line_idx}行格式错误: {line}")
                    continue

                filename = os.path.basename(parts[0].replace("*", "_"))  # 统一文件名格式
                true_age = float(parts[1])
                gt_dict[filename] = true_age
            except Exception as e:
                print(f"[ERROR] 解析{gt_file}第{line_idx}行失败: {str(e)}")
                continue

    # 读取预测值
    pred_dict = {}
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split('\t')
                if len(parts) < 2:
                    print(f"[WARN] {pred_file} 第{line_idx}行格式错误: {line}")
                    continue

                filename = os.path.basename(parts[0].replace("*", "_"))  # 统一文件名格式
                pred_age = float(parts[1])
                pred_dict[filename] = pred_age
            except Exception as e:
                print(f"[ERROR] 解析{pred_file}第{line_idx}行失败: {str(e)}")
                continue

    # 计算MAE
    total_error = 0.0
    matched_count = 0
    missing_files = []

    for filename, pred_age in pred_dict.items():
        if filename in gt_dict:
            true_age = gt_dict[filename]
            total_error += abs(pred_age - true_age)
            matched_count += 1
        else:
            missing_files.append(filename)

    # 输出统计信息
    print(f"\n{'统计信息':-^30}")
    print(f"真实值样本数: {len(gt_dict)}")
    print(f"预测值样本数: {len(pred_dict)}")
    print(f"成功匹配样本数: {matched_count}")
    print(f"未匹配预测样本数: {len(pred_dict) - matched_count}")

    if len(missing_files) > 0:
        print(f"\n前5个未匹配文件示例:")
        for f in missing_files[:5]:
            print(f"  {f}")

    if matched_count == 0:
        raise ValueError("没有匹配的样本，无法计算MAE")

    return total_error / matched_count, matched_count


if __name__ == "__main__":
    # 文件路径配置
    val_file = "./annotations/val.txt"  # 修改为实际路径
    pred_file = "./predict/pred_result.txt"  # 修改为实际路径

    try:
        mae, count = calculate_mae(val_file, pred_file)
        print(f"\nMAE计算结果: {mae:.4f} (基于{count}个样本)")
    except Exception as e:
        print(f"\n[ERROR] 计算失败: {str(e)}")