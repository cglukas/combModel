"""File for training based on configuration files."""
from pathlib import Path

import click
import torch
import wandb

from development.model.comb_model import CombModel
from development.trainer.configured_training.configuration import (
    TrainingConfig,
    yml_to_config,
)
from development.trainer.configured_training.load_from_config import (
    ConfigError,
    get_optimizer,
    init_model_and_optimizer,
    load_datasets,
    load_level_manager,
    load_logger,
)
from development.trainer.trainer import Trainer
from development.trainer.training_file_io import TrainingIO


def run_training_for_single_config(config: TrainingConfig) -> None:
    """Run the training based on the config."""
    if config.resume_checkpoint and config.pretraining_checkpoint:
        msg = (
            "Resuming with pretrained checkpoint does not work. Only provide one value."
        )
        raise ConfigError(msg)

    dataset_manager = load_datasets(config)
    level_manager = load_level_manager(config)
    if config.pretraining_checkpoint:
        model, optimizer = init_model_and_optimizer(config)
    else:
        model = CombModel(persons=len(config.datasets))
        optimizer = get_optimizer(config, model)

    training_io = TrainingIO(model, optimizer, level_manager)
    output_folder = Path(config.trainings_folder)
    if not output_folder.exists():
        output_folder.mkdir()
    training_io.set_folder(output_folder)

    if config.resume_checkpoint:
        training_io.load(Path(config.resume_checkpoint))

    logger = load_logger(config)
    device = torch.device(config.device)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataset_manager=dataset_manager,
        file_io=training_io,
        device=device,
        logger=logger,
        level_manager=level_manager,
    )

    print(f"Run training for {config.name}")
    trainer.train()
    wandb.finish()


@click.command("Run Training")
@click.option(
    "--raise-error",
    help="Will raise configuration errors and won't skip invalid configs.",
    default=False,
    is_flag=True,
)
@click.argument("config_files", nargs=-1, type=str)
def run_training(config_files: list[str], raise_error: bool):
    """Run the training loop for the training configs the config file."""
    all_configs: list[TrainingConfig] = []

    for c_file in config_files:
        with open(c_file, encoding="utf-8") as file:
            all_configs.extend(yml_to_config(file.read()))

    for single_config in all_configs:
        try:
            run_training_for_single_config(single_config)
        except ConfigError as e:
            if raise_error:
                raise e
            print(f"Skipping config: '{single_config.name}' because of error: \n {e}")


if __name__ == "__main__":
    run_training()  # pylint: disable=no-value-for-parameter
