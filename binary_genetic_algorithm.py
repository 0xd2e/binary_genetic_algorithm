#!/usr/bin/python3

'''Binary genetic algorithm engine'''


import numpy as np


def generate(pop_size, chrom_length):
    '''
    Inputs:

    pop_size -- positive integer number
             -- total number of chromosomes in a generation

    chrom_length -- positive integer number
                 -- number of bits in a chromosome
                 -- length of a chromosome


    Each chromosome is represented as a fixed length 1D numpy array
    with random sequence of Boolean values.

    Population size must be smaller than the total number of unique
    chromosome sequences (binary patterns) that can be generated from
    a given number of bits.


    Returns nested (2D) numpy boolean array, the entire population
    of chromosomes (solution candidates).
    '''

    assert pop_size < (2 ** chrom_length)

    return np.random.randint(0, 2, size=(pop_size, chrom_length), dtype=np.bool)


def select(population, fit_values):
    '''
    Inputs:

    population -- array of chromosomes
               -- each chromosome must be represented as 1D numpy boolean array

    fit_values -- array of fitness values corresponding to chromosomes
               -- each fitness value must be a non-negative real number


    Selection is based on the roulette wheel method (fitness proportionate selection)
    where probability of a chromosome being selected is related to its fitness value.

    This does not guarantee that the best chromosome sequence (binary pattern)
    will be selected but helps to avoid local optima.


    Returns nested (2D) numpy boolean array, the entire next generation of chromosomes
    (solution candidates) chosen with repetition from a given population.
    '''

    # Enumerate chromosomes in the population
    indexes = np.arange(population.shape[0], dtype=np.uint)

    indexes = np.random.choice(indexes, size=indexes.size, replace=True, p=fit_values)

    np.random.shuffle(indexes)

    return population[indexes]


def mutate(population, mut_prob):
    '''
    Inputs:

    population -- array of chromosomes
               -- each chromosome must be represented as 1D numpy boolean array

    mut_prob -- positive real number
             -- mutation rate
             -- probability that a bit will be inverted


    Mutation can occur independently at every bit along each chromosome
    with uniform probability.


    Returns nested (2D) numpy boolean array, the entire population of chromosomes
    (solution candidates) with randomly altered bits.
    '''

    bits_to_mutate = np.random.uniform(size=population.shape) < mut_prob

    # Change only specific bits in chromosomes, using XOR
    return population ^ bits_to_mutate


def crossover(population, crs_prob):
    '''
    Inputs:

    population -- array of chromosomes
               -- each chromosome must be represented as 1D numpy boolean array
               -- number of chromosomes must be even

    crs_prob -- positive real number
             -- crossover (recombination) probability
             -- probability that a pair of chromosomes will exchange
                part of bit sequences


    Commute part of binary sequences between paired chromosomes.


    Returns nested (2D) numpy boolean array, the entire population of chromosomes
    (solution candidates) where random chromosome pairs swapped their binary pattern.
    '''

    # Each row represents a pair of chromosomes
    # Each column represents specific bit

    # Get the number of pairs and the length of sequences
    rows, cols = population[0::2].shape

    # Select pairs of chromosomes for which sequences of bits will be exchanged
    pairs = np.random.uniform(size=(rows, 1)) < crs_prob

    # Each chromosome must contribute at least one bit

    # Set a single crossover bit for each pair of chromosomes
    breakpoints = np.random.randint(1, cols, size=(rows, 1), dtype=np.uint)

    # Enumerate bits
    bits = np.arange(cols, dtype=np.uint)

    # Divide each sequence of bits into two parts
    positions = bits < breakpoints

    # Keep information of bit positions only for selected pairs
    positions &= pairs

    return np.concatenate((
        (population[0::2] &  positions) | (population[1::2] & ~positions),
        (population[0::2] & ~positions) | (population[1::2] &  positions)
    ))


def run(fitness,
        crs_prob,
        mut_prob,
        chrom_length,
        pop_size=100,
        iterations=200,
        threshold=1):
    '''
    Inputs:

    fitness -- cost function
            -- must take one positional argument: nested (2D) numpy boolean array
            -- must return 1D numpy array of real numbers between
               0 (invalid sequence) and 1 (perfect fitness)

    crs_prob -- positive real number
             -- crossover (recombination) probability
             -- probability that a pair of chromosomes will exchange
                part of bit sequences

    mut_prob -- positive real number
             -- mutation rate
             -- probability that a bit will be inverted

    chrom_length -- positive integer number
                 -- number of bits in a chromosome
                 -- length of a chromosome

    pop_size -- positive integer number
             -- must be even
             -- total number of chromosomes in a generation
             -- default is 100

    iterations -- positive integer number
               -- maximum number of generations
               -- default is 200

    threshold -- positive number less than or equal to 1
              -- minimum satisfactory fitness level
              -- higher value is more strict
              -- default is 1, which produces the highest fitness value after
                 a given number of iterations (preferably a perfect fitness
                 or an exact match)


    The fitness function operates on numpy arrays: for a given population
    of chromosomes it must return corresponding fitness values.

    This function always returns a chromosome with the largest fitness value.
    However, depending on the fitness function, it can solve both minimization
    and maximization problems.

    The highest fitted chromosome is returned even if the minimum fitness threshold
    condition is not satisfied within a given number of generations.


    Returns a tuple with two elements: chromosome (1D numpy boolean array)
    and its fitness value (positive real number).
    '''

    assert (pop_size % 2) is 0
    assert (crs_prob > 0) and (crs_prob < 1)
    assert (mut_prob > 0) and (mut_prob < 1)
    assert (threshold >= 0) and (threshold <= 1)

    # Create initial population and calculate corresponding fitness values
    population = generate(pop_size, chrom_length)
    fit_values = fitness(population)

    # Find the best candidate from a current generation
    j = np.argmax(fit_values)
    chromosome = population[j]
    score = fit_values[j]

    # Create successive generations
    for _ in range(iterations):

        if score >= threshold:
            break

        # Recreate population
        population = select(population, fit_values)

        population = crossover(population, crs_prob)

        population = mutate(population, mut_prob)

        # Recalculate fitness values
        fit_values = fitness(population)

        j = np.argmax(fit_values)

        if score < fit_values[j]:
            chromosome = population[j]
            score = fit_values[j]

    return (chromosome, score)
