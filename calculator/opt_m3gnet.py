import os
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ase.optimize import BFGS
from m3gnet.models import M3GNet, M3GNetCalculator, Potential
from ase.io import read, write

model_path = "/mnt/d/project/m3gnet/pretrained/MP-2021.2.8-EFS/"
pretrained_model = M3GNet.from_dir(model_path)  #

# 初始化计算器（自动下载预训练模型）
calc = M3GNetCalculator(potential=Potential(pretrained_model))  # 可选参数 device="cuda" 启用GPU


def process_cif_files(input_dir: Path, output_dir: Path):
    """处理指定目录中的所有cif文件"""
    # 确保输入目录存在
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_path}")
    if not output_path.exists():
        os.makedirs(output_path)
    # 遍历所有cif文件
    for cif_file in input_path.glob("*.cif"):
        base_name = cif_file.stem
        outcar_path = Path(output_dir / f"{base_name}.log")
        # 读取文件
        atoms = read(str(cif_file), format="cif")

        # 设置计算器
        atoms.calc = calc
        opt = BFGS(atoms, trajectory=None, logfile=str(outcar_path))
        opt.run(fmax=0.05)  # 设置力收敛阈值

        output_file = output_dir / base_name
        write(str(output_file), atoms, format="vasp")


def read_energy(log_dir: Path):
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {log_path}")
    with open(str(log_path), "r") as f:
        lines = f.readlines()
        energy = lines[-1].split()[-2]
    return energy


if __name__ == "__main__":
    input_dir = Path(r"/mnt/d/project/catalyst-select/ase-GA/generate")
    output_dir = Path(r"/mnt/d/project/catalyst-select/ase-GA/generate/opt")
    process_cif_files(input_dir, output_dir)
