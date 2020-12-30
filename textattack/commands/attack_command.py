from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from textattack import Attacker, CommandLineAttackArgs, DatasetArgs, ModelArgs
from textattack.commands import TextAttackCommand


class AttackCommand(TextAttackCommand):
    """The TextAttack attack module:

    A command line parser to run an attack from user specifications.
    """

    def run(self, args):
        attack_args = CommandLineAttackArgs(**vars(args))
        model_wrapper = ModelArgs.create_model_from_args(attack_args)
        dataset = DatasetArgs.create_dataset_from_args(attack_args)

        attack = CommandLineAttackArgs.create_attack_from_args(
            attack_args, model_wrapper
        )
        if attack_args.interactive:
            Attacker.attack_interactive(attack)
        else:
            attacker = Attacker(attack, dataset, attack_args)
            attacker.attack_dataset()

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "attack",
            help="run an attack on an NLP model",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser = CommandLineAttackArgs.add_parser_args(parser)
        parser.set_defaults(func=AttackCommand())
