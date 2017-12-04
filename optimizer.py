from functools import reduce
from operator import add
import random
from network import Network
from utils import *

class Optimizer():

    def __init__(self, parameters_to_optimize, retain=0.7,
                 random_select=0.1, mutate_chance=0.4):

        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.parameters_to_optimize = parameters_to_optimize

    def create_population(self, size):

        population = []
        for _ in range(0, size):
            # Create a random network.
            network = Network(self.parameters_to_optimize)
            network.create_random()

            # Add the network to our population.
            population.append(network)

        return population

    @staticmethod
    def fitness(network):
        return network.accuracy

    def grade(self, pop):

        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):

        children = []
        for _ in range(2):

            child = {}

            # Loop through the parameters and pick params for the kid.
            for param in self.parameters_to_optimize:
                child[param] = random.choice([mother.network[param], father.network[param]])

            if type(child['neurons_per_layer']) is int:
                child['neurons_per_layer'] = [child['neurons_per_layer']]

            if type(child['dropout_per_layer']) is float:
                child['dropout_per_layer'] = [child['dropout_per_layer']]

            # Now create a network object.
            network = Network(self.parameters_to_optimize)
            network.create_set(child)

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                network = self.mutate(network)


            if network.is_consistent():
                children.append(network)

        return children

    def mutate(self, network):

        # Choose a random key.
        parameter = random.choice(list(self.parameters_to_optimize.keys()))
        #print("mutating key: " + str(parameter))

        # Mutate one of the params.
        #print("previous value: " + str(network.get_parameter(parameter)))
        new_values = []

        if type(network.get_parameter(parameter)) is list:
            for i in range(len(network.get_parameter(parameter))):
                new_value = random.choice(self.parameters_to_optimize[parameter])
                new_values.append(new_value)

            network.set_parameter(parameter, new_values)
        else:
            new_value = random.choice(self.parameters_to_optimize[parameter])
            network.set_parameter(parameter, new_value)

        #print("mutated value: " + str(network.get_parameter(parameter)))

        if network.get_parameter('number_of_layers') != len(network.get_parameter('neurons_per_layer')):
            network.network['neurons_per_layer'] = correct_neurons_per_layer(network.network)
            #network.correct_parameter('neurons_per_layer')

        if network.get_parameter('number_of_layers') != len(network.get_parameter('dropout_per_layer')):
            network.network['dropout_per_layer'] = correct_dropout_per_layer(network.network)
            #network.correct_parameter('dropout_per_layer')


        return network

    def evolve(self, pop):

        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded)*self.retain)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)


        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []


        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:
            #print(len(children), desired_length)

            # Get a random mom and dad.
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            print(male, female)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                try:
                    babies = self.breed(male, female)
                except Exception as e:
                    print("exception " + str(e))

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
