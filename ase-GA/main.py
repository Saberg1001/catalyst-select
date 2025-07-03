import json
import shutil
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


def generate_initial_db(slab) -> PrepareDB:
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

    db = PrepareDB(db_file_name=DBFileName, simulation_cell=slab)
    for i in range(Config.pop_size):
        candidate = sg.get_new_candidate()
        candidate.info['confid'] = f"gen0_id{i}"
        db.add_unrelaxed_candidate(candidate, description="x:x")
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
    slab = read(Config.cif_file)
    slab.pbc = [True, True, False]
    calc = M3GNetCalculator(
        potential=Potential(M3GNet.from_dir(Config.model_path)),
        device="cuda"
    )
    if not Path(DBFileName).exists():
        db = generate_initial_db(slab)
    else:
        dc = DataConnection(DBFileName)
        if dc.get_number_of_unrelaxed_candidates() == 0:
            # 如果数据库存在且没有未松弛结构，则需要清空数据库
            Path(DBFileName).unlink()
            db = generate_initial_db(slab)

    # 4. 设置GA算子
    n_top = len(slab)  # 基底原子数

    # 交叉算子
    pairing = CutAndSplicePairing(slab, n_top,
                                  blmin=0.7,
                                  p1=Config.crossover_prob,
                                  p2=0)

    # 变异算子
    rattle = RattleMutation(blmin=0.7,
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

    for gen in range(Config.n_generations):
        print(f"\n--- Generation {gen + 1}/{Config.n_generations} ---")

        # 评估未松弛的候选结构
        while dc.get_number_of_unrelaxed_candidates() > 0:
            a = dc.get_an_unrelaxed_candidate()
            energy = float(relax_structure(a, calc, gen + 1, a.info['confid'])[0])
            set_raw_score(a, -energy)  # 最小化能量
            dc.add_relaxed_step(a)

        # 获取当前种群
        population = pop.get_current_population()

        # 记录结果
        record_results(gen + 1, population, db)

        # 检查收敛
        if conv.converged():
            print("GA已收敛!")
            break

        # 产生新一代
        print("Generating new candidates...")
        for i in range(Config.pop_size):
            # 选择
            a1, a2 = pop.get_two_candidates()

            # 变异或交叉
            if np.random.random() < Config.crossover_prob:
                a3, desc = pairing.get_new_individual([a1, a2])
                a3.info['data'] = {'parents': [a1.info['confid'], a2.info['confid']]}
                a3.info['origin'] = 'crossover'
            else:
                a3, desc = rattle.get_new_individual([a1])
                a3.info['data'] = {'parents': [a1.info['confid']]}
                a3.info['origin'] = 'mutation'
            a3.info['confid'] = f"gen{gen + 2}_id{i}"
            db.add_unrelaxed_candidate(a3)

    # 保存最佳结构
    best = pop.get_current_population()[0]
    best_structure_file = Config.output_root / "summary" / "best_structure.cif"
    write(str(best_structure_file), best, format='cif')

    print("\nGA优化完成！结果保存在:", Config.output_root)
    print(f"最佳结构能量: {best.get_potential_energy():.4f} eV")


if __name__ == "__main__":
    main()
