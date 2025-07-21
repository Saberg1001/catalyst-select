import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from ase.ga import set_raw_score
from ase.ga.convergence import GenerationRepetitionConvergence
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.data import DataConnection, PrepareDB
from ase.ga.population import Population
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.standardmutations import RattleMutation
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator, get_all_atom_types
from ase.io import read, write
from ase.optimize import BFGS
from m3gnet.models import M3GNet, M3GNetCalculator, Potential

from config import Config
from constant import DBFileName


def init_dirs():
    Config.output_root.mkdir(exist_ok=True)
    (Config.output_root / "all_structures").mkdir(exist_ok=True)
    (Config.output_root / "summary").mkdir(exist_ok=True)


def sort_atoms_reference(atoms):
    """使 atoms 的 get_chemical_symbols() 顺序与 target_symbols_order 保持一致"""
    target_symbols_order = ["Ce", "O", "W", "V"]
    symbol_indices = []
    symbols = atoms.get_chemical_symbols()
    for sym in target_symbols_order:
        idx = [i for i, s in enumerate(symbols) if s == sym]
        symbol_indices.extend(idx)
    return atoms[symbol_indices]


def generate_structure(slab):
    atom_numbers = [8] * 21 + [74] * 1 + [23] * 1  # 撒21个O，1个W，1个V
    pos = slab.get_positions()
    cell = slab.get_cell()
    p0 = np.array([0.0, 0.0, np.max(pos[:, 2]) + 0.8])  # 表面最高点上方0.8Å
    v1 = cell[0, :] * 0.9  # 使用90%的晶胞x方向
    v2 = cell[1, :] * 0.9  # 使用90%的晶胞y方向
    v3 = np.array([0.0, 0.0, 4.0])  # 在z方向给出4Å的空间

    unique_atom_types = get_all_atom_types(slab, atom_numbers)
    blmin = closest_distances_generator(atom_numbers=unique_atom_types,
                                        ratio_of_covalent_radii=Config.ratio_of_covalent_radii)

    sg = StartGenerator(slab, atom_numbers, blmin, box_to_place_in=[p0, [v1, v2, v3]], test_too_far=False)
    return sg, atom_numbers


def generate_initial_db(slab, adsorbate_stoichiometry) -> PrepareDB:
    # 读取CIF文件
    seed_Path = Config.seed_path  # 替换为你的CIF文件夹路径
    seed_files = [f for f in os.listdir(seed_Path) if f.endswith('.cif')]
    # 确保不超过种群大小
    seed_files = seed_files[:Config.pop_size]
    # 读取并处理CIF文件
    starting_population = []
    for seed_file in seed_files:
        atoms = read(os.path.join(seed_Path, seed_file))
        atoms = sort_atoms_reference(atoms)
        formula = atoms.get_chemical_formula()
        print(formula)
        # 这里可能需要根据你的需求对atoms进行一些处理
        starting_population.append(atoms)

    # 正确的方式：使用指定的、仅包含吸附原子的化学计量比来初始化数据库
    db = PrepareDB(
        db_file_name=DBFileName,
        simulation_cell=slab,
        stoichiometry=adsorbate_stoichiometry
    )

    for i, a in enumerate(starting_population):
        a.info['confid'] = f"seed_{i}"  # 初始种群使用独立的命名
        db.add_unrelaxed_candidate(a)
    return db


def relax_structure(atoms, calc, gen, indiv_id):
    atoms = atoms.copy()
    atoms.calc = calc

    gen_dir = Config.output_root / f"generation_{gen}"
    gen_dir.mkdir(exist_ok=True)

    logfile = gen_dir / f"gen{gen}_id{indiv_id}.log"

    opt = BFGS(atoms, logfile=str(logfile), trajectory=None)
    opt.run(fmax=Config.fmax)

    # 保存优化后结构
    final_structure = gen_dir / f"gen{gen}_id{indiv_id}_final.cif"
    write(str(final_structure), atoms, format="cif")

    # 保存到总结构库
    shutil.copy(final_structure,
                Config.output_root / "all_structures" / f"gen{gen}_id{indiv_id}.cif")
    return atoms.get_potential_energy(), atoms


def record_results(generation, population, db):
    # 当前代记录
    gen_dir = Config.output_root / f"generation_{generation}"
    gen_dir.mkdir(exist_ok=True)

    # 写入JSON格式的完整信息
    generation_data = []
    for indiv in population:
        gen_data = {
            "structure_id": indiv.info['confid'],
            "energy": indiv.get_potential_energy(),
            "origin": indiv.info.get('origin', 'unknown'),
            "parents": indiv.info.get('parents', [])
        }
        generation_data.append(gen_data)

    with open(gen_dir / "generation_info.json", 'w') as f:
        json.dump(generation_data, f, indent=2)

    # 全局记录
    summary_file = Config.output_root / "summary" / "all_energies.csv"
    with open(summary_file, "a") as f:
        for indiv in population:
            f.write(f"{generation},{indiv.info['confid']},{indiv.get_potential_energy():.6f}\n")

    # 按能量排序的记录
    sorted_pop = sorted(population, key=lambda x: x.get_potential_energy())
    sorted_file = Config.output_root / "summary" / "sorted_energies.csv"
    with open(sorted_file, "w") as f:
        f.write("rank,generation,structure_id,energy,origin\n")
        for rank, indiv in enumerate(sorted_pop):
            f.write(
                f"{rank + 1},{generation},{indiv.info['confid']},{indiv.get_potential_energy():.6f},"
                f"{indiv.info.get('origin', 'unknown')}\n")


# ---Step4: GA主循环 ---
def main():
    init_dirs()
    slab = read(Config.slab_file)
    slab.pbc = [True, True, False]
    adsorbate_atom_numbers = [8] * 21 + [74] * 1 + [23] * 1  # 21个O, 1个W, 1个V
    n_top = len(adsorbate_atom_numbers)  # = 23
    adsorbate_stoichiometry = dict(Counter(adsorbate_atom_numbers))

    calc = M3GNetCalculator(
        potential=Potential(M3GNet.from_dir(Config.model_path)),
        device="cuda"
    )
    if not Path(DBFileName).exists():
        db = generate_initial_db(slab, adsorbate_stoichiometry)
    else:
        dc = DataConnection(DBFileName)
        if dc.get_number_of_unrelaxed_candidates() == 0:
            # 如果数据库存在且没有未松弛结构，则需要清空数据库
            Path(DBFileName).unlink()
            db = generate_initial_db(slab, adsorbate_stoichiometry)

    # 4. 设置GA算子

    unique_atom_types = get_all_atom_types(slab, adsorbate_atom_numbers)
    blmin = closest_distances_generator(atom_numbers=unique_atom_types,
                                        ratio_of_covalent_radii=Config.ratio_of_covalent_radii)
    # 交叉算子
    pairing = CutAndSplicePairing(slab, n_top,
                                  blmin=blmin,
                                  p1=Config.crossover_prob,
                                  p2=0)

    # 变异算子
    rattle = RattleMutation(blmin=blmin,
                            n_top=n_top,
                            rattle_strength=0.3,
                            rattle_prop=Config.rattle_mutation_prob)

    # 比较器
    comp = InteratomicDistanceComparator(n_top=n_top,
                                         pair_cor_cum_diff=0.015,
                                         pair_cor_max=0.7,
                                         dE=0.02)

    # 5. 主循环
    dc = DataConnection(db_file_name=DBFileName)
    pop = Population(data_connection=dc, population_size=Config.pop_size, comparator=comp)
    conv = GenerationRepetitionConvergence(population_instance=pop,  # 必须传入Population实例
                                           number_of_generations=5,  # 检查间隔代数
                                           number_of_individuals=3)  # 需要重复的个体数

    # 评估未松弛的候选结构
    while dc.get_number_of_unrelaxed_candidates() > 0:
        a = dc.get_an_unrelaxed_candidate()
        a.pbc = [True, True, False]
        a.calc = calc
        energy, a = relax_structure(a, calc, 0, a.info['confid'])
        set_raw_score(a, -float(energy))  # 最小化能量
        a.calc.results['forces'] = a.calc.results['forces'].astype(np.float64)
        a.calc.results['stress'] = a.calc.results['stress'].astype(np.float64)
        dc.add_relaxed_step(a)
    pop.update()

    global_id_counter = Config.pop_size

    for gen in range(Config.n_generations):
        print(f"\n--- Generation {gen + 1}/{Config.n_generations} ---")
        if conv.converged():
            print("GA已收敛!")
            break
        print("Generating new candidates...")

        # 生成交叉个体
        for i in range(Config.n_crossovers):
            # for i in range(Config.pop_size):
            # 选择
            a1, a2 = pop.get_two_candidates()
            a3, desc = pairing.get_new_individual([a1, a2])
            a3.info['data'] = {'parents': [a1.info['confid'], a2.info['confid']]}
            a3.info['origin'] = 'crossover'
            a3.info['confid'] = f"gen{gen + 1}_id{i}"
            dc.add_unrelaxed_candidate(a3, description=desc)
        for i in range(Config.n_crossovers, Config.n_crossovers + Config.n_mutations):
            a1, _ = pop.get_two_candidates()
            a3, desc = rattle.get_new_individual([a1])
            a3.info['data'] = {'parents': [a1.info['confid']]}
            a3.info['origin'] = 'mutation'
            a3.info['confid'] = f"gen{gen + 1}_id{i}"
            dc.add_unrelaxed_candidate(a3, description=desc)
            global_id_counter += 1
        for i in range(Config.n_crossovers + Config.n_mutations, Config.pop_size):
            sg, _ = generate_structure(slab)
            a3 = sg.get_new_candidate()
            a3.info['confid'] = f"gen{gen + 1}_id{global_id_counter}"
            a3.info['origin'] = 'random'
            desc = "random:generate"
            db.add_unrelaxed_candidate(a3, description=desc)
            global_id_counter += 1
        while dc.get_number_of_unrelaxed_candidates() > 0:
            a = dc.get_an_unrelaxed_candidate()
            a.calc = calc
            energy, a = relax_structure(a, calc, gen + 1, a.info['confid'])
            set_raw_score(a, -float(energy))  # 最小化能量
            a.calc.results['forces'] = a.calc.results['forces'].astype(np.float64)
            a.calc.results['stress'] = a.calc.results['stress'].astype(np.float64)
            dc.add_relaxed_step(a)
        pop.update()

    # 保存最佳结构
    best = pop.get_current_population()[0]
    best.calc = calc
    energy = best.get_potential_energy()
    best_structure_file = Config.output_root / "summary" / "best_structure.cif"
    write(str(best_structure_file), best, format='cif')

    print("\nGA优化完成！结果保存在:", Config.output_root)
    print(f"最佳结构能量: {energy:.4f} eV")


if __name__ == "__main__":
    main()
