"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from tqdm import tqdm

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt',
    filemode='w'
)

def train_networks(networks, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        pbar.update(1)

    pbar.close()



def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def get_average_loss(networks):
    """Get the average loss for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average loss of a population of networks.
    """

    total_loss = 0
    for network in networks:
        total_loss += network.loss

    return total_loss / len(networks)

def generate(generations, population_size, parameters_to_optimize, dataset):
    optimizer = Optimizer(parameters_to_optimize)
    networks = optimizer.create_population(population_size)

    for i in range(generations):
        # Train and get accuracy for networks.
        logging.info("Generation %d" % (i+1))
        train_networks(networks, dataset)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)
        average_loss = get_average_loss(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average accuracy: %.2f%%" % (average_accuracy * 100))
        logging.info("Generation average loss: %.2f%%" % (average_loss))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)
	
	# Print out the top network.
	print_best_networks(networks)

def print_best_networks(networks, n = 1):
    """Print the best N networks.

    Args:
        networks (list): The population of networks

        n (int): number of networks to print

    """
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)
    networks = networks[:n]

    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main():
    generations = 20  # Number of times to evole the population.
    population_size = 20   # Number of networks in each generation.
    dataset = 'mnist'

    parameters_to_optimize = {
        'neurons_per_layer': [64, 128, 256, 512, 768, 1024],
        'number_of_layers': [1, 2, 3, 4, 5],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'selu'],
        'dropout_per_layer': [0.2, 0.4, 0.6],
        'optimizer': ['adam']
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population_size))

    generate(generations, population_size, parameters_to_optimize, dataset)

if __name__ == '__main__':
    main()
