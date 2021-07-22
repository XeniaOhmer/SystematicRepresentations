from egg.zoo.systematicity.metrics.tre import Objective, LinearComposition, CompositionFunction, MultipleCrossEntropyLoss
from typing import Iterable, Type
import torch
import os
import pickle
from abc import ABC
import argparse


def get_protocol(interaction, vocab_size):

    sender_in = interaction.sender_input.cpu()
    n_atts = int(sum(sender_in[0]))
    n_vals = int(len(sender_in[0]) // n_atts)
    messages = interaction.message[:, :-1].cpu() - 1

    k_hot_messages = []
    for m in messages:
        k_hot_messages.append(torch.nn.functional.one_hot(
            m, num_classes=vocab_size).reshape(-1))
    k_hot_messages = torch.stack(k_hot_messages, dim=0)

    derivations = []
    for att in range(n_atts):
        derivations.append(torch.argmax(sender_in[:, att * n_vals:(att + 1) * n_vals], dim=1))
    derivations = torch.stack(derivations, dim=1)

    protocol = {}
    for i, derivation in enumerate(derivations):
        protocol[tuple([torch.unsqueeze(elem, dim=0) for elem in derivation])] = k_hot_messages[i]

    return protocol


def get_name(atts, vals, vs, ml, seed):
    name = ('atts' + str(atts) + '_vals' + str(vals) + '_vs' + str(vs) + '_len' + str(ml) + '/seed' +
            str(seed) + '/')
    return name


class TreeReconstructionError(ABC):

    def __init__(
            self,
            num_concepts: int,
            message_length: int,
            vocab_size: int,
            composition_fn: Type[CompositionFunction],
            weight_decay=1e-5,
    ):
        self.num_concepts = num_concepts
        self.message_length = message_length
        self.composition_fn = composition_fn
        self.weight_decay = weight_decay
        self.vocab_size = vocab_size

    def measure(self, interaction) -> (float, float):
        protocol = get_protocol(interaction, self.vocab_size)
        objective = Objective(
            num_concepts=self.num_concepts,
            vocab_size=self.vocab_size,
            message_length=self.message_length,
            composition_fn=self.composition_fn(representation_size=self.message_length * self.vocab_size),
            loss_fn=MultipleCrossEntropyLoss(representation_size=self.message_length * self.vocab_size,
                                             message_length=self.message_length)
        )
        reconstruction_error_sum, reconstruction_error_mean = self._train_model(
            messages=protocol.values(),
            derivations=protocol.keys(),
            objective=objective,
            optimizer=torch.optim.Adam(objective.parameters(), lr=1e-1, weight_decay=self.weight_decay),
            n_epochs=1000
        )
        return reconstruction_error_sum, reconstruction_error_mean, objective

    def evaluate(self, interaction, trained_objective) -> (float, float):
        protocol = get_protocol(interaction, self.vocab_size)
        messages = protocol.values()
        derivations = protocol.keys()
        errors = [trained_objective(message, derivation) for message, derivation in zip(messages, derivations)]
        return sum(errors).item(), torch.mean(torch.tensor(errors)).item()

    def _train_model(
            self,
            messages: Iterable[torch.Tensor],
            derivations: Iterable[torch.Tensor],
            objective: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            n_epochs: int,
            quiet: bool = False
    ) -> (float, float):

        for t in range(n_epochs):
            optimizer.zero_grad()
            errors = [objective(message, derivation) for message, derivation in zip(messages, derivations)]
            loss = sum(errors)
            loss.backward()
            if not quiet and t % 100 == 0:
                print(f'Training loss at epoch {t} is {loss.item():.4f}')
            optimizer.step()
        return loss.item(), torch.mean(torch.tensor(errors)).item()


def main(n_atts, n_vals, prefix):

    modes = ['test', 'generalization_hold_out', 'uniform_holdout']

    for message_length in [3, 4, 6, 8]:
        for vocab_size in [10, 50, 100]:
            for seed_orig in range(5):

                print(vocab_size, message_length, seed_orig)

                path = (prefix + 'egg/zoo/systematicity/results/' +
                        get_name(n_atts, n_vals, vocab_size, message_length, seed_orig))

                interaction_paths = {}
                for mode in modes:
                    interaction_paths[mode] = path + 'interactions/' + mode + '/'
                interactions = {}
                for mode in modes:
                    for filename in os.listdir(interaction_paths[mode]):
                        interactions[mode] = torch.load(interaction_paths[mode] + filename + '/interaction_gpu0')

                NUM_SEEDS = 3
                tre_errors = {}
                for seed in range(NUM_SEEDS):
                    tre_errors['seed' + str(seed)] = {}
                    TRE = TreeReconstructionError(n_atts * n_vals, message_length, vocab_size, LinearComposition)
                    value_sum, value_mean, objective = TRE.measure(interactions['test'])
                    tre_errors['seed' + str(seed)]['_training_sum'] = value_sum
                    tre_errors['seed' + str(seed)]['_training_mean'] = value_mean
                    value_sum, value_mean = TRE.evaluate(interactions['generalization_hold_out'], objective)
                    tre_errors['seed' + str(seed)]['_generalization_holdout_sum'] = value_sum
                    tre_errors['seed' + str(seed)]['_generalization_holdout_mean'] = value_mean
                    value_sum, value_mean = TRE.evaluate(interactions['uniform_holdout'], objective)
                    tre_errors['seed' + str(seed)]['_uniform_holdout_sum'] = value_sum
                    tre_errors['seed' + str(seed)]['_uniform_holdout_mean'] = value_mean

                pickle.dump(tre_errors, open(path + 'tre.pkl', 'wb'))
                torch.save(objective, open(path + 'tre_objective.pt', 'wb'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_attributes", type=int, default=2)
    parser.add_argument("--n_values", type=int, default=16)
    parser.add_argument("--prefix", type=str, default='C:/Users/Xenia/PycharmProjects/SystematicRepresentations/')
    args = parser.parse_args()
    main(args.n_attributes, args.n_values, args.prefix)
