import yaml 

categories_map = {
    134533: {"description": "face", "id": 0},
    90020: {"description": "logo", "id": 1},
}
print(categories_map)

with open('config.yaml', 'w') as yaml_file:
    yaml.dump(categories_map, yaml_file)