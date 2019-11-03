#!/usr/bin/python3

"""Binary genetic algorithm engine"""


from typing import Any, Callable, List, Union, Tuple

import numpy as np


_DEFAULT_RAND_GEN = np.random.Generator(np.random.pcg64.PCG64(None))


def generate(
        pop_size: int,
        chrom_length: int,
        threshold: float = 0.5,
        rand_generator: np.random.Generator = _DEFAULT_RAND_GEN
) -> np.ndarray:
    """
    Inputs:

    pop_size
    -- positive integer number
    -- total number of chromosomes in a generation

    chrom_length
    -- positive integer number
    -- number of bits in a chromosome
    -- length of a chromosome

    threshold
    -- real number between 0.0 and 1.0 (default is 0.5)
    -- values (from uniform distribution) lower than this number
       translate to True values in chromosomes

    rand_generator
    -- instance of Numpy Random Generator
    -- Generator with Numpy default BitGenerator (PCG64)
       and None as a seed value is used as default


    Each chromosome is represented as a fixed length 1D numpy array
    with random sequence of Boolean values.

    Population size must be smaller than the total number of unique
    chromosome sequences (binary patterns) that can be generated from
    a given number of bits.


    Returns nested (2D) numpy boolean array, the entire population
    of chromosomes (solution candidates).
    """

    if pop_size >= (1 << chrom_length):
        raise ValueError('Population must be smaller than overall unique chromosome sequences.')

    return rand_generator.uniform(low=0.0, high=1.0, size=(pop_size, chrom_length)) < threshold


def select(
        population: np.ndarray,
        scores: np.ndarray,
        indexes: np.ndarray,
        rand_generator: np.random.Generator = _DEFAULT_RAND_GEN
) -> np.ndarray:
    """
    Inputs:

    population
    -- array of chromosomes
    -- each chromosome must be represented as 1D numpy boolean array

    scores
    -- array of fitness values corresponding to chromosomes
    -- each fitness value must be a non-negative real number

    indexes
    -- 1D numpy integer array
    -- indexes of chromosomes in the population (row index)

    rand_generator
    -- instance of Numpy Random Generator
    -- Generator with Numpy default BitGenerator (PCG64)
       and None as a seed value is used as default


    Selection is based on the roulette wheel method (fitness proportionate selection)
    where probability of a chromosome being selected is related to its fitness value.

    This does not guarantee that the best chromosome sequence (binary pattern)
    will be selected but helps to avoid local optima.


    Returns nested (2D) numpy boolean array, the entire next generation of chromosomes
    (solution candidates) chosen with repetition from a given population.
    """

    probabilities = scores / np.sum(scores)

    indexes = rand_generator.choice(indexes, size=indexes.size, replace=True, p=probabilities)

    rand_generator.shuffle(indexes)

    return population[indexes]


def mutate(
        population: np.ndarray,
        mut_prob: float,
        rand_generator: np.random.Generator = _DEFAULT_RAND_GEN
) -> np.ndarray:
    """
    Inputs:

    population
    -- array of chromosomes
    -- each chromosome must be represented as 1D numpy boolean array

    mut_prob
    -- positive real number
    -- mutation rate
    -- probability that a bit will be inverted

    rand_generator
    -- instance of Numpy Random Generator
    -- Generator with Numpy default BitGenerator (PCG64)
       and None as a seed value is used as default


    Mutation can occur independently at every bit along each chromosome
    with uniform probability.


    Returns nested (2D) numpy boolean array, the entire population of chromosomes
    (solution candidates) with randomly altered bits.
    """

    bits_to_mutate = rand_generator.uniform(low=0.0, high=1.0, size=population.shape) < mut_prob

    # Change only specific bits in chromosomes, using XOR
    return population ^ bits_to_mutate


def crossover(
        population: np.ndarray,
        crs_prob: float,
        bits: np.ndarray,
        rand_generator: np.random.Generator = _DEFAULT_RAND_GEN
) -> np.ndarray:
    """
    Inputs:

    population
    -- array of chromosomes
    -- each chromosome must be represented as 1D numpy boolean array
    -- number of chromosomes must be even

    crs_prob
    -- positive real number
    -- crossover (recombination) probability
    -- probability that a pair of chromosomes will exchange
       part of bit sequences

    bits
    -- 1D numpy integer array
    -- indexes of bits in a chromosome (column index)

    rand_generator
    -- instance of Numpy Random Generator
    -- Generator with Numpy default BitGenerator (PCG64)
       and None as a seed value is used as default


    Commute part of binary sequences between paired chromosomes.


    Returns nested (2D) numpy boolean array, the entire population of chromosomes
    (solution candidates) where random chromosome pairs swapped their binary pattern.
    """

    # Each row represents a pair of chromosomes
    # Each column represents specific bit

    # Get the number of pairs and the length of sequences
    rows, cols = population.shape

    rows >>= 1  # rows //= 2

    # Select pairs of chromosomes for which sequences of bits will be exchanged
    pairs = rand_generator.uniform(low=0.0, high=1.0, size=(rows, 1)) < crs_prob

    # Each chromosome must contribute at least one bit

    # Set a single crossover bit for each pair of chromosomes
    breakpoints = rand_generator.integers(
        low=1,
        high=cols,
        size=(rows, 1),
        dtype=bits.dtype,
        endpoint=False
    )

    # Divide each sequence of bits into two parts
    positions = bits < breakpoints

    # Keep information of bit positions only for selected pairs
    positions &= pairs

    return np.concatenate((
        (population[0::2] &  positions) | (population[1::2] & ~positions),
        (population[0::2] & ~positions) | (population[1::2] &  positions)
    ), axis=0)


def run(
        fit_func: Callable[..., np.ndarray],
        crs_prob: float,
        mut_prob: float,
        chrom_length: int,
        pop_size: int,
        iterations: int,
        fit_args: Union[List[Any], Tuple, None] = None,
        threshold: float = 1.0
) -> np.ndarray:
    """
    Inputs:

    fit_func
    -- fitness function
    -- first positional argument must be a nested (2D) numpy boolean array
    -- must return 1D numpy array with real numbers between
       0 (invalid sequence) and 1 (perfect fitness) and size
       equal to the number of rows of the given numpy array

    crs_prob
    -- positive real number
    -- crossover (recombination) probability
    -- probability that a pair of chromosomes will exchange
       part of bit sequences

    mut_prob
    -- positive real number
    -- mutation rate
    -- probability that a bit will be inverted

    chrom_length
    -- positive integer number
    -- number of bits in a chromosome
    -- length of a chromosome

    fit_args
    -- list, tuple, or None (default)
    -- additional argument(s) require by fitness function

    pop_size
    -- positive integer number (default is 100)
    -- must be even
    -- total number of chromosomes in a generation

    iterations
    -- positive integer number (default is 200)
    -- maximum number of generations

    threshold
    -- positive number less than or equal to 1
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


    Returns a chromosome (1D numpy boolean array).
    """

    if pop_size & 1:
        raise ValueError('Total amount of chromosomes in a population must be an even number.')
    if (crs_prob <= 0) or (crs_prob >= 1):
        raise ValueError('Crossover probability must be a number between 0.0 and 1.0 (exclusive).')
    if (mut_prob <= 0) or (mut_prob >= 1):
        raise ValueError('Mutation probability must be a number between 0.0 and 1.0 (exclusive).')
    if 1 < threshold < 0:
        raise ValueError('Threshold must be a number between 0.0 and 1.0 (inclusive).')

    if fit_args is None:
        fit_args = []
    elif not isinstance(fit_args, (list, tuple)):
        raise TypeError('Additional fitness argument(s) must be placed in a list or tuple.')

    # Create initial population and calculate corresponding fitness values
    population = generate(pop_size, chrom_length)
    scores = fit_func(population, *fit_args)

    # Find the best candidate from current generation
    alpha = np.argmax(scores)
    alpha_chromosome = population[alpha]
    alpha_score = scores[alpha]

    # Enumerate chromosomes in the population (create rows index)
    indexes = np.arange(pop_size, dtype=np.uint)

    # Enumerate bits in a chromosome (create columns index)
    bits = np.arange(chrom_length, dtype=np.uint)

    # Create successive generations
    for _ in range(iterations):

        if alpha_score >= threshold:
            break

        # Recreate population
        population = select(population, scores, indexes)
        population = crossover(population, crs_prob, bits)
        population = mutate(population, mut_prob)

        # Recalculate fitness values
        scores = fit_func(population, *fit_args)

        alpha = np.argmax(scores)

        if alpha_score < scores[alpha]:
            alpha_chromosome = population[alpha]
            alpha_score = scores[alpha]

    return alpha_chromosome
