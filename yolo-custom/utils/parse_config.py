def parse_model_config(path):
    with open(path, 'r') as file:
        lines = file.read().split('\n')
        lines = [x.strip() for x in lines if x and not x.startswith('#')]
        module_defs = []
        for line in lines:
            if line.startswith('['):
                module_defs.append({})
                module_defs[-1]['type'] = line[1:-1].rstrip()
                if module_defs[-1]['type'] == 'convolutional':
                    module_defs[-1]['batch_normalize'] = 0
            else:
                key, value = line.split("=")
                value = value.strip()
                module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


def parse_data_config(path):
    options = {}
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue
        k, v = line.split("=")
        options[k.strip()] = v.strip()
    return options
    

def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')[:-1]
    return names
