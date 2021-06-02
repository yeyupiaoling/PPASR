
def labels_to_string(label, vocabulary, blank_index=0):
    labels = []
    for l in label:
        index_list = [index for index in l if index != blank_index]
        labels.append(''.join([vocabulary[index] for index in index_list]))
    return labels
