import yaml

def get_config():
    with open('./config.yml') as file:
        yaml_data= yaml.safe_load(file)
        return yaml_data