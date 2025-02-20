from Run_Game import *
from random import randint

def calculating_population_fitness(population):
    
    fitness = []
    scores = []
    for i in range(population.shape[0]):
        fit, in_game_score = run_game_with_ML(screen,clock,population[i])
        print('fitness value of chromosome '+ str(i) +' :  ', fit)
        fitness.append(fit)
        scores.append(in_game_score)
    return np.array(fitness), np.array(scores)

def selection_by_roulette(population, fitness, num_parents):

    parents = np.empty((num_parents,population.shape[1]))
    fit = [fitness[i] + np.abs(np.min(fitness)) for i in range(len(fitness))]
    relative_fitness = [fit[i]/np.sum(fit) for i in range(len(fit))]
    cumulative = [np.sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]
    for parent_num in range(num_parents):
        r = random.random()
        for i,prob in enumerate(cumulative):
            if r<=prob:
                parents[parent_num, :] = population[i, :]
    return parents

def selecting_best(population, fitness, num_parents):
    
    parents = np.empty((num_parents, population.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = population[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999
    return parents

def crossover(parents, offspring_size):

    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]): 
  
        while True:
            parent1_idx = random.randint(0, parents.shape[0] - 1)
            parent2_idx = random.randint(0, parents.shape[0] - 1)
            if parent1_idx != parent2_idx:
                for j in range(offspring_size[1]):
                    if random.uniform(0, 1) < 0.5:
                        offspring[k, j] = parents[parent1_idx, j]
                    else:
                        offspring[k, j] = parents[parent2_idx, j]
                break
    return offspring

def mutation(offspring):

    for idx in range(offspring.shape[0]):
        for _ in range(25):
            i = randint(0,offspring.shape[1]-1)

        random_value = np.random.choice(np.arange(-1,1,step=0.001),size=(1),replace=False)
        offspring[idx, i] = offspring[idx, i] + random_value

    return offspring