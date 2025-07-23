from constant import SlabFile, ModelPath, OutputRoot, SeedPath


class Config(object):
    pop_size = 10
    n_generations = 1000
    ratio_of_covalent_radii = 0.7
    fmax = 0.05
    output_root = OutputRoot
    tournament_size = 3
    rattle_mutation_prob = 0.3
    crossover_prob = 0.7
    Z = 151
    slab_file = SlabFile
    model_path = ModelPath
    n_crossovers = 3
    n_mutations = 2
    seed_path = SeedPath


for i in range(Config.pop_size):
    pass
