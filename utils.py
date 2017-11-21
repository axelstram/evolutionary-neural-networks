def correct_neurons_per_layer(child):
    print("neurons per layer start: " + str(child['neurons_per_layer']))
    correct_parameter = child['neurons_per_layer']

    while child['number_of_layers'] > len(correct_parameter):
        #print("adding " + str(child['neurons_per_layer'][-1]) + " to " + str(child['neurons_per_layer']))
        correct_parameter.append(correct_parameter[-1])   #Fill neurons_per_layer with it's last value

    if child['number_of_layers'] < len(correct_parameter):
        correct_parameter = correct_parameter[:child['number_of_layers']]     #Discard the extra values

    print("neurons per layer end: " + str(correct_parameter))

    return correct_parameter


def correct_dropout_per_layer(child):
    print("dropout per layer start: " + str(child['dropout_per_layer']))
    correct_parameter = child['dropout_per_layer']

    while child['number_of_layers'] > len(correct_parameter):
        #print("adding " + str(child['dropout_per_layer'][-1]) + " to " + str(child['dropout_per_layer']))
        correct_parameter.append(correct_parameter[-1])   #Fill dropout_per_layer with it's last value

    if child['number_of_layers'] < len(correct_parameter):
        correct_parameter = correct_parameter[:child['number_of_layers']]     #Discard the extra values

    print("dropout per layer end: " + str(correct_parameter))

    return correct_parameter
