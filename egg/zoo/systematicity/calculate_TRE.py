from egg.zoo.systematicity.metrics.tre import *
from typing import Iterable, Type
import torch
import os
from abc import ABC
import argparse
import copy
import pickle


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
            weight_decay=1e-1,
            lr=1e-3,
            early_stopping=True
    ):
        self.num_concepts = num_concepts
        self.message_length = message_length
        self.composition_fn = composition_fn
        self.weight_decay = weight_decay
        self.vocab_size = vocab_size
        self.learning_rate = lr
        if early_stopping:
            self.patience = 50
        else:
            self.patience = 1000

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
        error_train, error_val, objective_final, objective_es, epoch_es = self._train_model(
            messages=list(protocol.values()),
            derivations=list(protocol.keys()),
            objective=objective,
            optimizer=torch.optim.Adam(objective.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay),
            n_epochs=1000
        )
        return error_train, error_val, objective_final, objective_es, epoch_es

    def evaluate(self, interaction, trained_objective) -> (float, float):
        protocol = get_protocol(interaction, self.vocab_size)
        messages = protocol.values()
        derivations = protocol.keys()
        with torch.no_grad():
            errors = [trained_objective(message, derivation) for message, derivation in zip(messages, derivations)]
        return torch.mean(torch.tensor(errors)).item()

    def _train_model(
            self,
            messages: Iterable[torch.Tensor],
            derivations: Iterable[torch.Tensor],
            objective: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            n_epochs: int,
            quiet: bool = False
    ) -> (float, float):

        collect_error_train = []
        collect_error_val = []

        n_samples = len(messages)
        n_train = int(round(n_samples * 0.9))
        messages_train = messages[:n_train]
        messages_val = messages[n_train:]
        derivations_train = derivations[:n_train]
        derivations_val = derivations[n_train:]

        patience_count = 0
        min_val_error = 1e10
        early_stopping_flag = False

        for t in range(n_epochs):

            if patience_count == self.patience:
                early_stopping_flag = True

            optimizer.zero_grad()
            errors = [objective(message, derivation) for message, derivation in zip(messages_train, derivations_train)]
            loss = sum(errors)
            loss.backward()
            optimizer.step()
            mean_train_loss = torch.mean(torch.tensor(errors)).item()
            collect_error_train.append(mean_train_loss)

            with torch.no_grad():
                errors_val = [objective(message, derivation) for message, derivation
                              in zip(messages_val, derivations_val)]
                mean_val_loss = torch.mean(torch.tensor(errors_val)).item()
                collect_error_val.append(mean_val_loss)
                if (mean_val_loss < min_val_error) and (early_stopping_flag is False):
                    min_val_error = mean_val_loss
                    patience_count = 0
                    min_val_objective = copy.deepcopy(objective)
                    min_val_epoch = t
                elif early_stopping_flag is False:
                    patience_count += 1

            if (t == n_epochs - 1) and (early_stopping_flag is False):
                min_val_epoch = t-1
                min_val_objective = copy.deepcopy(objective)

            if not quiet and t % 50 == 0:
                print(f'Training loss at epoch {t} is {mean_train_loss:.4f}',
                      f'Validation loss at epoch {t} is {mean_val_loss:.4f}')

        return collect_error_train, collect_error_val, objective, min_val_objective, min_val_epoch


def main(n_atts, n_vals, prefix, composition_fn):
    modes = ['test', 'generalization_hold_out', 'uniform_holdout']

    try:
        if composition_fn == 'linear':
            composition_function = LinearComposition
        elif composition_fn == 'mlp':
            composition_function = MLPComposition
    except UnboundLocalError:
        print('Invalid composition function provided')

    for message_length in [3, 4, 6, 8]:
        for vocab_size in [10, 50, 100]:
            for seed_orig in range(3):

                print(composition_fn, "values", n_vals, "vs", vocab_size, "ml", message_length, seed_orig)

                path = (prefix + 'egg/zoo/systematicity/results/' +
                        get_name(n_atts, n_vals, vocab_size, message_length, seed_orig))

                try:
                    interaction_paths = {}
                    for mode in modes:
                        interaction_paths[mode] = path + 'interactions/' + mode + '/'
                    interactions = {}
                    for mode in modes:
                        for filename in os.listdir(interaction_paths[mode]):
                            interactions[mode] = torch.load(
                                interaction_paths[mode] + filename + '/interaction_gpu0')
                except FileNotFoundError:
                    continue

                NUM_SEEDS = 1
                tre_errors = {}
                for seed in range(NUM_SEEDS):
                    tre_errors['seed' + str(seed)] = {}
                    TRE = TreeReconstructionError(n_atts * n_vals, message_length, vocab_size,
                                                  composition_function)
                    error_train, error_val, objective, ES_objective, ES_epoch = TRE.measure(interactions['test'])
                    print('mean error train', error_train[-1], 'mean_error val', error_val[-1])
                    tre_errors['seed' + str(seed)]['training_mean'] = error_train
                    tre_errors['seed' + str(seed)]['validation_mean'] = error_val
                    tre_errors['seed' + str(seed)]['early_stopping_epoch'] = ES_epoch
                    error_gen_holdout = TRE.evaluate(interactions['generalization_hold_out'], objective)
                    tre_errors['seed' + str(seed)]['generalization_holdout_mean'] = error_gen_holdout
                    error_gen_holdout = TRE.evaluate(interactions['generalization_hold_out'], ES_objective)
                    tre_errors['seed' + str(seed)]['generalization_holdout_mean_es'] = error_gen_holdout
                    error_uniform_holdout = TRE.evaluate(interactions['uniform_holdout'], objective)
                    tre_errors['seed' + str(seed)]['uniform_holdout_mean'] = error_uniform_holdout
                    print('generalization error', error_uniform_holdout)
                    error_uniform_holdout = TRE.evaluate(interactions['uniform_holdout'], ES_objective)
                    tre_errors['seed' + str(seed)]['uniform_holdout_mean_es'] = error_uniform_holdout
                    print('generalization error es', error_uniform_holdout)

                if not os.path.exists(path + 'tre/'):
                    os.makedirs(path + 'tre/')
                pickle.dump(tre_errors, open(path + 'tre/tre_' + composition_fn + '.pkl', 'wb'))
                torch.save(objective, open(path + 'tre/tre_objective_' + composition_fn + '.pt', 'wb'))
                torch.save(objective, open(path + 'tre/tre_objective_es_' + composition_fn + '.pt', 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_attributes", type=int, default=2)
    parser.add_argument("--n_values", type=int, default=50)
    parser.add_argument("--composition_fn", type=str, default='mlp')
    parser.add_argument("--prefix", type=str, default='C:/Users/Xenia/PycharmProjects/SystematicRepresentations/')
    args = parser.parse_args()
    main(args.n_attributes, args.n_values, args.prefix, args.composition_fn)
