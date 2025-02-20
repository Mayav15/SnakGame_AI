from Genetic_Algorithm import *
from SnakeGame import *
import matplotlib.pyplot as plt

individuals_per_generation = 30
num_generations = 15
num_weights = n_x*n_h + n_h*n_h2 + n_h2*n_y                     # THESE VALUES ARE SPECIFIED IN Feed_Forward_Neural_Network.py
population_size = (individuals_per_generation,num_weights)
new_population = np.random.choice(np.arange(-1,1,step=0.01),size=population_size,replace=True)

num_parents = 10
avg_fitness = []
fitnesses = []
high_scores = []

for generation in range(num_generations):
    print('GENERATION ',str(generation+1) )
    
    fitness,scores = calculating_population_fitness(new_population)
    print(f'Fittest chromosome in gneneration {str(generation+1)} has fitness value:  ', np.max(fitness))

    avg = np.average(fitness)
    print(f"Average fitness of Generation {generation+1} is: ", avg)

    avg_fitness.append(avg)

    print(f"Highest score in the generation {generation+1} is: ",np.max(scores))

    high_scores.append(np.max(scores))

    parents = selecting_best(new_population, fitness, num_parents)
    # parents = selection_by_roulette(new_population, fitness, num_parents)

    offspring_crossover = crossover(parents, offspring_size=(population_size[0] - parents.shape[0], num_weights))

    offspring_mutation = mutation(offspring_crossover)

    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

print(f"Best generation in terms of average fitness is Generation {np.argmax(avg_fitness)+1}")

plt.figure(figsize=(15,8))

plt.subplot(1,2,2)
x = [i for i in range(1,len(avg_fitness)+1)]
plt.plot(x,avg_fitness)
plt.xlabel("Generation")
plt.ylabel("Average Fitness")
plt.title("Average fitness over generations")

plt.subplot(1,2,1)
plt.plot(x,high_scores)
plt.xlabel("Generation")
plt.ylabel("Highest scores in game")
plt.title("High scores over generations")

plt.show()
