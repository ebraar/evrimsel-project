import numpy as np
import random

# Problem Parametreleri
n = 5  # Problem boyutu
bit_length = 15  # Her karar değişkeninin bit uzunluğu
chromosome_length = n * bit_length  # Birey uzunluğu
population_size = 20  # Popülasyon büyüklüğü
max_function_evaluations = 10000  # Maksimum amaç fonksiyonu hesaplama sayısı
crossover_probability = 0.7  # Çaprazlama olasılığı
mutation_probability = 0.001  # Mutasyon olasılığı
lower_bound, upper_bound = -100, 100  # Karar değişkenleri sınırı

# Amaç fonksiyonu
def objective_function(x):
    return sum(xi**2 for xi in x)

# Binary'den Reel Sayıya Çevirme
def binary_to_real(binary_chromosome):
    real_values = []
    for i in range(0, len(binary_chromosome), bit_length):
        binary_segment = binary_chromosome[i:i + bit_length]
        decimal = int(''.join(map(str, binary_segment)), 2)
        real_value = lower_bound + (decimal / (2**bit_length - 1)) * (upper_bound - lower_bound)
        real_values.append(real_value)
    return real_values

# Popülasyon Oluşturma
def initialize_population():
    return np.random.randint(2, size=(population_size, chromosome_length))

# Amaç Fonksiyonu Değerlerini Hesaplama
def evaluate_population(population):
    fitness = []
    for individual in population:
        real_values = binary_to_real(individual)
        fitness.append(objective_function(real_values))
    return np.array(fitness)

# Rulet Tekerleği Seçimi
def roulette_wheel_selection(population, fitness):
    total_fitness = sum(1 / (1 + fit) for fit in fitness)
    probabilities = [(1 / (1 + fit)) / total_fitness for fit in fitness]
    selected_indices = np.random.choice(range(population_size), size=population_size, p=probabilities)
    return population[selected_indices]

# İki Noktalı Çaprazlama
def crossover(parent1, parent2):
    if random.random() < crossover_probability:
        point1 = random.randint(0, chromosome_length - 1)
        point2 = random.randint(point1, chromosome_length - 1)
        child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
        child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
        return child1, child2
    return parent1, parent2

# Mutasyon
def mutate(individual):
    for i in range(chromosome_length):
        if random.random() < mutation_probability:
            individual[i] = 1 - individual[i]  # Bit'i tersine çevir
    return individual

# Genetik Algoritma
def genetic_algorithm():
    population = initialize_population()
    best_solution = None
    best_fitness = float('inf')
    function_evaluations = 0

    while function_evaluations < max_function_evaluations:
        fitness = evaluate_population(population)
        function_evaluations += population_size

        # En iyi bireyi güncelle
        current_best_index = np.argmin(fitness)
        current_best_fitness = fitness[current_best_index]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = binary_to_real(population[current_best_index])

        # Yeni popülasyon oluştur
        selected_population = roulette_wheel_selection(population, fitness)
        new_population = []

        for i in range(0, population_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = np.array(new_population)

        # Iterasyon Sonuçlarını Göster
        print(f"Function Evaluations: {function_evaluations}, Best Fitness: {best_fitness}, Best Solution: {best_solution}")

    return best_solution, best_fitness

# Algoritmayı Çalıştır
best_solution, best_fitness = genetic_algorithm()
print("\nFinal Best Solution:", best_solution)
print("Final Best Fitness:", best_fitness)