import os
from pathlib import Path

import numpy as np
from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator, get_all_atom_types
from ase.io import read
from ase.spacegroup import Spacegroup


def generate_structure(work_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        pass
    # 读取CIF文件
    cif_file = Path(work_dir)
    slab = read(str(cif_file))  # 推荐用ase.io.read

    # 设置周期性边界条件 (表面在xy平面)
    slab.pbc = [True, True, False]

    # 定义要添加的原子: 21个O(原子序数8), 1个W(原子序数74), 1个V(原子序数23)
    atom_numbers = [8] * 21 + [74] * 1 + [23] * 1

    # 定义放置原子的区域 (在表面上方)
    pos = slab.get_positions()
    cell = slab.get_cell()
    p0 = np.array([0.0, 0.0, np.max(pos[:, 2]) + 0.8])  # 表面最高点上方0.8Å
    v1 = cell[0, :] * 0.9  # 使用90%的晶胞x方向
    v2 = cell[1, :] * 0.9  # 使用90%的晶胞y方向
    v3 = np.array([0.0, 0.0, 4.0])  # 在z方向给出4Å的空间


    # 定义原子间最短距离
    unique_atom_types = get_all_atom_types(slab, atom_numbers)
    blmin = closest_distances_generator(
        atom_numbers=unique_atom_types,
        ratio_of_covalent_radii=0.7
    )

    # 创建起始生成器
    sg = StartGenerator(
        slab,
        atom_numbers,
        blmin,
        box_to_place_in=[p0, [v1, v2, v3]]
    )

    #设置每一代有多少结构
    population_size = 20

    #撒点
    for i in range(population_size):
        starting_population = sg.get_new_candidate()
        path = Path(f"{output_dir}/{i}-generate_structure.cif")
        starting_population.write(str(path),format='cif')
    print("成功撒点并保存到 generate 路径")

if __name__ == "__main__":
    generate_structure(work_dir=Path('Slab/substrate.cif'), output_dir=Path('./generate/'))