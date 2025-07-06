from constant import CifFile, ModelPath, OutputRoot


class Config(object):
    pop_size = 2
    n_generations = 3
    ratio_of_covalent_radii = 0.7
    fmax = 0.05
    output_root = OutputRoot
    tournament_size = 3
    rattle_mutation_prob = 0.3
    crossover_prob = 0.7
    Z = 151
    cif_file = CifFile
    model_path = ModelPath



for i in range(Config.pop_size):
    pass
