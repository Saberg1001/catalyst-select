from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp import Kpoints

# 读取 POSCAR
structure = Structure.from_file("/mnt/d/desktop/POSCAR")

# 生成 k点网格（密度 1000 k点/原子）
kpoints = Kpoints.automatic_density(structure, kppa=1000)
print("k点网格尺寸:", kpoints.kpts[0])
