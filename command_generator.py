def generate_command(entity_name, project_name, dataset_name, parameters):
    """
    Generates a command to run a training script with the specified parameters.

    Args:
        entity_name (str): The WandB entity.
        project_name (str): The project name.
        dataset_name (str): The name of dataset
        parameters (str): Multi-line string containing model parameters.

    Returns:
        str: The generated command string.
    """
    base_command = f"python train.py --wandb_project {project_name} --wandb_entity {entity_name} --dataset {dataset_name}"
    param_command = ""

    # Splitting the multi-line string and parse each line
    for line in parameters.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()  # Use underscore as is
            value = value.strip().strip('"')
            param_command += f" --{key} {value}"

    # Combining base command and parameter command
    full_command = f"{base_command}{param_command}"
    return full_command


# Change any of these according to requirement
entity_name = "amar74384-iit-madras"
project_name = "DA6401_assign_1"
dataset_name= "mnist"

# Paste the config parameters of run inside """__________"""
parameters = """
activation:"relu"
batch_size:16
epochs:8
hidden_layers:2
hidden_size:128
learning_rate:0.001
optimizer:"nadam"
weight_decay:0
weight_init:"xavier"
"""

command = generate_command(entity_name, project_name, dataset_name, parameters)
print(command)
