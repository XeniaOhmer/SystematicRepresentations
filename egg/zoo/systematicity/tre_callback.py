from typing import Iterable, Type
from egg.core.callbacks import Callback
import os
import json
import torch
import pickle
from egg.core.interaction import Interaction
from egg.core.batch import Batch
import copy


class CompositionFunction(torch.nn.Module):

    def __init__(self, representation_size: int):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AdditiveComposition(CompositionFunction):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class LinearComposition(CompositionFunction):
    def __init__(self, representation_size: int):
        super().__init__(representation_size)
        self.linear = torch.nn.Linear(representation_size * 2, representation_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat((x, y), dim=1))


class MLPComposition(CompositionFunction):
    def __init__(self, representation_size: int):
        super().__init__(representation_size)
        self.linear_1 = torch.nn.Linear(representation_size * 2, 50)
        self.linear_2 = torch.nn.Linear(50, representation_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.linear_2(torch.tanh(self.linear_1(torch.cat((x, y), dim=1))))


class LinearMultiplicationComposition(CompositionFunction):
    def __init__(self, representation_size: int):
        super().__init__(representation_size)
        self.linear_1 = torch.nn.Linear(representation_size, representation_size)
        self.linear_2 = torch.nn.Linear(representation_size, representation_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.linear_1(x) * self.linear_2(y)


class MultiplicativeComposition(CompositionFunction):
    def __init__(self, representation_size: int):
        super().__init__(representation_size)
        self.bilinear = torch.nn.Bilinear(
            in1_features=representation_size,
            in2_features=representation_size,
            out_features=representation_size
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bilinear(x, y)


class MultipleCrossEntropyLoss(torch.nn.Module):

    def __init__(self, representation_size: int, message_length: int):
        super().__init__()
        self.representation_size = representation_size
        self.message_length = message_length

    def forward(self, reconstruction, message):
        assert self.representation_size % self.message_length == 0
        width_of_single_symbol = self.representation_size//self.message_length
        loss = 0
        for i in range(self.message_length):
            start = width_of_single_symbol * i
            end = width_of_single_symbol * (i+1)
            loss += torch.nn.functional.cross_entropy(
                reconstruction[:, start:end],
                message[start:end].argmax(dim=0).reshape(1)
            )
        return loss


class Objective(torch.nn.Module):
    def __init__(
            self,
            num_concepts: int,
            vocab_size: int,
            message_length: int,
            composition_fn: torch.nn.Module,
            loss_fn: torch.nn.Module,
            zero_init=False
    ):
        super().__init__()
        self.composition_fn = composition_fn
        self.loss_fn = loss_fn
        self.emb = torch.nn.Embedding(num_concepts, message_length * vocab_size)
        if zero_init:
            self.emb.weight.data.zero_()

    def compose(self, derivations):
        if isinstance(derivations, tuple):
            args = (self.compose(node) for node in derivations)
            return self.composition_fn(*args)
        else:
            return self.emb(derivations)

    def forward(self, messages, derivations):
        return self.loss_fn(self.compose(derivations), messages)


def get_protocol(interaction, vocab_size):
    sender_in = interaction.sender_input
    n_atts = int(sum(sender_in[0]))
    n_vals = int(len(sender_in[0]) // n_atts)
    messages = interaction.message[:, :-1] - 1
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
    name = ('atts' + str(atts) + '_vals' + str(vals) + '_vs' + str(vs) + '_len' + str(ml) + '/seed' + str(seed) + '/')
    return name


class TreeReconstructionError(Callback):

    def __init__(
            self,
            num_concepts: int,
            message_length: int,
            vocab_size: int,
            composition_fn: Type[CompositionFunction],
            weight_decay=1e-5,
            n_epochs: int = None,
            save_path: str = './',
            num_seeds: int = 5,
            train_epochs: int = 1000,
            loaders_metrics=None,
    ):
        if loaders_metrics is None:
            loaders_metric = []
        self.num_concepts = num_concepts
        self.message_length = message_length
        self.composition_fn = composition_fn
        self.weight_decay = weight_decay
        self.vocab_size = vocab_size
        self.n_epochs = n_epochs
        self.save_path = save_path
        self.num_seeds = num_seeds
        self.n_epochs = n_epochs
        self.train_epochs = train_epochs
        self.objective = Objective(
            num_concepts=self.num_concepts,
            vocab_size=self.vocab_size,
            message_length=self.message_length,
            composition_fn=self.composition_fn(representation_size=self.message_length * self.vocab_size),
            loss_fn=MultipleCrossEntropyLoss(representation_size=self.message_length * self.vocab_size,
                                             message_length=self.message_length)
        )
        self.loaders_metrics = loaders_metrics

    def measure(self, interaction):
        protocol = get_protocol(interaction, self.vocab_size)
        reconstruction_error_sum, reconstruction_error_mean = self._train_model(
            messages=protocol.values(),
            derivations=protocol.keys(),
            # objective=self.objective,
            optimizer=torch.optim.Adam(self.objective.parameters(), lr=1e-1, weight_decay=self.weight_decay),
            epochs=self.train_epochs
        )
        return reconstruction_error_sum, reconstruction_error_mean

    def _train_model(
            self,
            messages: Iterable[torch.Tensor],
            derivations: Iterable[torch.Tensor],
            # objective: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            quiet: bool = False
    ):
        for t in range(epochs):
            optimizer.zero_grad()
            errors = [self.objective(message, derivation) for message, derivation in zip(messages, derivations)]
            loss = sum(errors)
            loss.backward()
            if not quiet and t % 1000 == 0:
                print(f'Training loss at epoch {t} is {loss.item():.4f}')
            optimizer.step()
        return loss.item(), torch.mean(torch.tensor(errors)).item()

    def on_early_stopping(
            self,
            train_loss: float,
            train_logs: Interaction,
            epoch: int,
            test_loss: float = None,
            test_logs: Interaction = None,
    ):
        tre_sum, tre_mean = self.measure(train_logs)
        tre = {'sum': tre_sum,
               'mean': tre_mean}
        pickle.dump(tre, open(self.save_path + 'tre.pkl', 'wb'))
        torch.save(self.objective, open(self.save_path + 'tre_objective.pt', 'wb'))
        self.evaluate_generalization()

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if epoch == self.n_epochs:
            tre_sum, tre_mean = self.measure(logs)
            tre = {'sum': tre_sum,
                   'mean': tre_mean}
            pickle.dump(tre, open(self.save_path + 'tre.pkl', 'wb'))
            torch.save(self.objective, open(self.save_path + 'tre_objective.pt', 'wb'))
            self.evaluate_generalization()

    def evaluate_generalization(self):

        holdout_results = {}
        sender = copy.deepcopy(self.trainer.game.sender).cpu()

        for loader_name, loader, metric in self.loaders_metrics:
            input_data = torch.stack(loader.dataset.examples, dim=0)
            new_interaction = Interaction(input_data, None, None,  None,
                                          sender.forward(input_data), None, None, None)
            protocol = get_protocol(new_interaction, self.vocab_size)
            messages = protocol.values()
            derivations = protocol.keys()
            errors = [self.objective(message, derivation) for message, derivation in zip(messages, derivations)]
            errors = torch.tensor([err.item() for err in errors])

            holdout_results[loader_name] = {
                "sum": torch.sum(errors).item(),
                "mean": torch.mean(errors).item()
            }

            output_json = json.dumps(holdout_results)
            print(output_json, flush=True)

            pickle.dump(holdout_results, open(self.save_path + 'tre_holdout.pkl', 'wb'))


# def main(n_atts, n_vals, prefix):
#
#     for message_length in [3, 4, 6, 8]:
#         for vocab_size in [10, 50, 100]:
#             for seed_orig in range(5):
#
#                 print(vocab_size, message_length, seed_orig)
#
#                 path_to_interaction = (prefix + 'results/' +
#                                        get_name(n_atts, n_vals, vocab_size, message_length, seed_orig) +
#                                        'interactions/test/')
#                 try:
#                     for filename in os.listdir(path_to_interaction):
#                         interaction = torch.load(path_to_interaction + filename + '/interaction_gpu0')
#                 except:
#                     continue
#                 NUM_SEEDS = 5
#                 tre_errors = {}
#                 # tre_objectives = {}
#                 for seed in range(NUM_SEEDS):
#                     TRE = TreeReconstructionError(n_atts * n_vals, message_length, vocab_size, LinearComposition)
#                     value = TRE.measure(interaction)
#                     tre_errors['seed' + str(seed)] = value
#                     # tre_objectives['seed' + str(seed)] = value
#                 print(tre_errors)

                #pickle.dump(tre_errors, open(
                #    'results/' + get_name(n_atts, n_vals, vocab_size, message_length, seed_orig) + 'tre_error.pkl', 'wb'))
                # pickle.dump(tre_objectives, open(
                #    'results/' + get_name(n_atts, n_vals, vocab_size, message_length, seed_orig) + 'tre_objective.pkl', 'wb'))


# if __name__ == "__main__":
#
#     import os
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--n_attributes", type=int, default=2)
#     parser.add_argument("--n_values", type=int, default=16)
#     parser.add_argument("--prefix", type=str,
#                         default='C:/Users/Xenia/PycharmProjects/SystematicRepresentations/egg/zoo/systematicity/')
#     args = parser.parse_args()
#     main(args.n_attributes, args.n_values, args.prefix)
