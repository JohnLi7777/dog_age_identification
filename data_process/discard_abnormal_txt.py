with open('../annotations/train.txt', 'r') as infile, open('../annotations/train_correct.txt', 'w') as outfile:
    for line in infile:
        # 分割文件名和数字
        parts = line.strip().split('\t')

        # 确保每行格式正确
        if len(parts) == 2:
            try:
                number = int(parts[1])
                # 保留数字小于等于192的行
                if number <= 192:
                    outfile.write(line)
            except ValueError:
                # 处理无法转换为数字的情况（可选）
                pass