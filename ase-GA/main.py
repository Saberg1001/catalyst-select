import sys
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.constraints import FixAtoms
from ase.io import read
import numpy as np
import argparse


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用遗传算法优化指定高度上的原子位置')
    parser.add_argument('--slab_file', type=str, required=True,
                        help='基底结构文件路径 (支持POSCAR, CIF, XYZ等格式)')
    parser.add_argument('--z_threshold', type=float, default=10.0,
                        help='Z轴高度阈值(埃)')
    parser.add_argument('--population', type=int, default=10,
                        help='遗传算法种群大小')
    parser.add_argument('--generations', type=int, default=20,
                        help='遗传算法迭代次数')
    args = parser.parse_args()

    # 从文件加载基底结构
    slab = read(args.slab_file)
    print(f"已加载基底结构: {args.slab_file}")
    print(f"基底包含 {len(slab)} 个原子")
    print(f"晶胞尺寸: {slab.cell.cellpar()}")

    # 待添加的元素组成
    element_counts = {'O': 16, 'W': 1, 'V': 1}

    # 创建初始结构 (包含基底和随机位置的添加原子)
    atoms = create_initial_structure(slab, args.z_threshold, element_counts)

    # 设置约束: 固定z坐标低于阈值的基底原子
    constraint = FixAtoms(indices=get_fixed_indices(atoms, args.z_threshold))
    atoms.set_constraint(constraint)

    # 设置最小原子间距规则
    blmin = closest_distances_generator(element_counts.keys(),
                                        bond_factor=0.8,
                                        respect_valency=False)

    # 创建遗传算法起始生成器
    box = get_optimization_region(slab, args.z_threshold)
    generator = StartGenerator(atoms,
                               lambda a: get_idx_to_optimize(a, args.z_threshold, element_counts),
                               blmin,
                               box_volume=box,
                               number_of_variable_cells=0,
                               test_dist_to_slab=False)

    # 初始化种群
    population = [generator.get_new_individual() for _ in range(args.population)]

    # 遗传算法主循环
    print(f"\n开始遗传算法优化 (种群={args.population}, 代数={args.generations})")
    for gen in range(args.generations):
        new_population = []

        # 1. 评估当前种群
        for individual in population:
            energy = evaluate(individual)
            individual.info['energy'] = energy
            individual.info['key_value_pairs'] = {'raw_score': -energy}

        # 2. 选择父代
        parents = select_parents(population)

        # 3. 生成新后代
        for _ in range(args.population):
            offspring = generator.get_new_individual(parents)
            new_population.append(offspring)

        # 4. 替换旧种群
        population = new_population

        # 报告当前最佳能量
        best_energy = min(atom.info['energy'] for atom in population)
        print(f"代数 {gen + 1}/{args.generations}: 最低能量 = {best_energy:.6f}")

    # 输出最优结构
    best_atoms = min(population, key=lambda a: a.info['energy'])
    print("\n优化完成!")
    print(f"最低能量结构: {best_atoms.info['energy']:.6f}")
    best_atoms.write('optimized_structure.xyz')
    print("已保存最优结构到 optimized_structure.xyz")


def create_initial_structure(slab, z_threshold, element_counts):
    """创建包含基底和随机位置添加原子的结构"""
    atoms = slab.copy()

    # 添加新原子
    symbols = []
    for sym, count in element_counts.items():
        symbols.extend([sym] * count)
    num_to_add = sum(element_counts.values())

    # 在基底上方随机生成位置
    cell = atoms.cell.cellpar()
    if cell[2] < z_threshold + 5:
        print(f"警告: 晶胞Z高度({cell[2]} Å)小于阈值+5({z_threshold + 5} Å)，可能需要增大晶胞")

    # 在晶胞XY平面内随机，Z在阈值以上
    positions = np.zeros((num_to_add, 3))
    positions[:, 0] = np.random.uniform(0, cell[0], num_to_add)
    positions[:, 1] = np.random.uniform(0, cell[1], num_to_add)
    positions[:, 2] = np.random.uniform(z_threshold, min(cell[2], z_threshold + 10), num_to_add)

    # 添加新原子
    atoms += Atoms(symbols=symbols, positions=positions)

    print(f"添加了 {num_to_add} 个原子在Z > {z_threshold} Å区域")
    return atoms


def get_fixed_indices(atoms, z_threshold):
    """获取固定原子的索引 (Z < 阈值)"""
    return [i for i, atom in enumerate(atoms) if atom.position[2] < z_threshold]


def get_idx_to_optimize(atoms, z_threshold, element_counts):
    """获取优化原子的索引 (指定元素且Z >= 阈值)"""
    return [i for i, atom in enumerate(atoms)
            if atom.symbol in element_counts and atom.position[2] >= z_threshold]


def get_optimization_region(slab, z_threshold):
    """获取优化区域 (XY为基底尺寸, Z从阈值开始)"""
    cell = slab.cell.cellpar()
    return np.array([[0, 0, z_threshold],
                     [cell[0], cell[1], max(cell[2], z_threshold + 10)]])


def select_parents(population):
    """锦标赛选择父代"""
    tournament_size = min(5, len(population))
    participants = np.random.choice(population, size=tournament_size, replace=False)
    return [min(participants, key=lambda a: a.info['energy'])]


def evaluate(atoms):
    """评估函数 - 实际应用中应替换为DFT计算"""
    # 此处仅为示例 - 实际应用中应使用DFT或ML势能进行能量计算
    return np.random.rand() * 100  # 替换为实际计算


if __name__ == "__main__":
    main()