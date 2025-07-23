import glob
from pathlib import Path

import pandas as pd

# 配置参数
output_root = Path(__file__).parent.parent.parent.parent / r"1-com-chem/gzy-ecust/work/W-V-O/da/1/ga_standard_results"  # 替换为你的输出目录路径
log_pattern = "generation_*/gen*.log"  # log文件路径模式
output_csv = output_root / "summary" / "log_energies.csv"  # 输出CSV文件路径


def extract_final_energy(log_path):
    """从log文件中提取最终能量值"""
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 反向查找最后有效的BFGS行
    for line in reversed(lines):
        if line.startswith("BFGS:"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    return float(parts[3])
                except (ValueError, IndexError):
                    continue
    return None


def process_logs():
    """处理所有log文件并保存结果到CSV"""
    # 获取所有log文件路径
    log_files = glob.glob(str(output_root / log_pattern), recursive=True)
    print(f"找到 {len(log_files)} 个log文件")

    results = []
    for log_path in log_files:
        log_file = Path(log_path)
        energy = extract_final_energy(log_path)

        if energy is not None:
            # 获取不带后缀的文件名
            filename = log_file.stem
            results.append({"filename": filename, "energy": energy})
        else:
            print(f"警告: 无法从 {log_file.name} 提取能量值")

    # 创建DataFrame并保存到CSV
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="energy", ascending=True)  # 按能量升序排序
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"成功提取 {len(df)} 个能量值，已保存到 {output_csv}")
        return df
    else:
        print("未找到有效能量值")
        return None


if __name__ == "__main__":
    df = process_logs()
    if df is not None:
        print("\n能量最低的5个结构:")
        print(df.head())
