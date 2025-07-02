import os
from pathlib import Path

import ase.io
from lam_optimize.main import relax_run, single_point
from lam_optimize.relaxer import Relaxer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def relax_mace(unrelax_path: Path, outpath: Path):
    df_out = relax_run(fpth=unrelax_path,
                       relaxer=Relaxer("mace"),
                       fmax=0.05,
                       steps=300,
                       traj_file=outpath,
                       )
    df_out.to_csv(Path(rf"{outpath}/result.csv"))


def single_point_mace(relax_path: Path):
    for file_path in relax_path.glob("*"):
        if file_path.is_file():
            atoms = ase.io.read(file_path, format='vasp')
            cif_path = file_path.with_suffix('.cif')
            ase.io.write(cif_path,atoms,format='cif')
    df_out = single_point(fpth=relax_path,
                          relaxer=Relaxer("mace"))
    df_out.to_csv(Path(rf"{relax_path}/result.csv"))


def calculate_m3gnet():
    pass


if __name__ == "__main__":
     unrelax_path=Path(r"/mnt/d/1-com-chem/gzy-ecust/work/W-V-O/mace/O2")
     outpath=Path(r"/mnt/d/1-com-chem/gzy-ecust/work/W-V-O/mace/relaxed-mace")
     relax_mace(unrelax_path, outpath)
