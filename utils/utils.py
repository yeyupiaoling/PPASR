import distutils.util


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)


def labels_to_string(label, vocabulary, blank_index=0):
    labels = []
    for l in label:
        index_list = [index for index in l if index != blank_index]
        labels.append(''.join([vocabulary[index] for index in index_list]))
    return labels
