# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from egg.core.language_analysis import *
from egg.core.callbacks import *
import pickle
from egg.core.batch import Batch


def get_attributes(sender_inputs):
    n_attributes = int(torch.sum(sender_inputs[0, :]))
    n_values = int(len(sender_inputs[0, :]) / n_attributes)
    attribute_parts = []
    for i in range(n_attributes):
        attribute_parts.append(torch.argmax(sender_inputs[:, i*n_values:(i+1)*n_values], dim=1) + 1)
    attributes = torch.stack(attribute_parts, dim=1)
    return attributes


def get_unique_attributes_and_messages(attributes, messages):
    _, indices = np.unique(attributes, return_index=True, axis=0)
    unique_attributes = attributes[indices]
    unique_messages = messages[indices]
    return unique_attributes, unique_messages


class TopographicSimilarityCustom(Callback):
    """
    """
    def __init__(
        self,
        is_gumbel: False,
        sender_input_distance_fn: str = "cosine",
        message_distance_fn: str = "edit",
        compute_topsim_train_set: bool = False,
        compute_topsim_test_set: bool = False,
        save_path: str = './',
        n_epochs: int = None
    ):
        self.sender_input_distance_fn = sender_input_distance_fn
        self.message_distance_fn = message_distance_fn
        self.compute_topsim_train_set = compute_topsim_train_set
        self.compute_topsim_test_set = compute_topsim_test_set
        self.is_gumbel = is_gumbel
        self.save_path = save_path
        self.n_epochs = n_epochs

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.compute_topsim_train_set:
            self.print_message(logs, "train", epoch)
        if epoch == self.n_epochs:
            messages = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
            unique_inputs, unique_messages = get_unique_attributes_and_messages(logs.sender_input, messages)
            topsim = TopographicSimilarity.compute_topsim(unique_inputs, unique_messages,
                                                          self.sender_input_distance_fn,
                                                          self.message_distance_fn)
            pickle.dump(topsim, open(self.save_path + 'topsim.pkl', 'wb'))

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        if self.compute_topsim_test_set:
            self.print_message(logs, "test", epoch)

    def print_message(self, logs: Interaction, mode: str, epoch: int) -> None:
        messages = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        unique_inputs, unique_messages = get_unique_attributes_and_messages(logs.sender_input, messages)

        topsim = TopographicSimilarity.compute_topsim(unique_inputs, unique_messages,
                                                      self.sender_input_distance_fn,
                                                      self.message_distance_fn)

        output = json.dumps(dict(topsim=topsim, mode=mode, epoch=epoch))
        print(output, flush=True)

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs: Interaction,
        epoch: int,
        test_loss: float = None,
        test_logs: Interaction = None,
    ):
        messages = train_logs.message.argmax(dim=-1) if self.is_gumbel else train_logs.message
        unique_inputs, unique_messages = get_unique_attributes_and_messages(train_logs.sender_input, messages)
        topsim = TopographicSimilarity.compute_topsim(unique_inputs, unique_messages,
                                                      self.sender_input_distance_fn,
                                                      self.message_distance_fn)
        pickle.dump(topsim, open(self.save_path + 'topsim.pkl', 'wb'))


class DisentCustom(Callback):
    """
    Callback to compute positional and bago of symbols disentanglement metrics.

    Metrics introduced in "Compositionality and Generalization in Emergent Languages", Chaabouni et al., ACL 2020.

    Two-symbol messages representing two-attribute world. One symbol encodes one attribute:
    in this case, the metric should be maximized:
    >>> samples = 1_000
    >>> _ = torch.manual_seed(0)
    >>> attribute1 = torch.randint(low=0, high=10, size=(samples, 1))
    >>> attribute2 = torch.randint(low=0, high=10, size=(samples, 1))
    >>> attributes = torch.cat([attribute1, attribute2], dim=1)
    >>> messages = attributes
    >>> round(Disent.posdis(attributes, messages), 6)
    0.978656
    >>> messages = torch.cat([messages, torch.zeros_like(messages)], dim=1)
    >>> round(Disent.posdis(attributes, messages), 6)
    0.978656

    Miniature language with perfect (=1) bosdis. Taken from Chaabouni et al. 2020, Appendix section 8.2.
    >>> attributes = torch.Tensor(
    ... [[0, 0], [0, 1], [0, 2], [0, 3],
    ... [1, 0], [1, 1], [1, 2], [1, 3],
    ... [2, 0], [2, 1], [2, 2], [2, 3],
    ... [3, 0], [3, 1], [3, 2], [3, 3]]
    ... )
    >>> messages = torch.Tensor(
    ... [[0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7],
    ... [1, 4, 1], [1, 5, 1], [1, 6, 1], [1, 7, 1],
    ... [2, 4, 2], [2, 5, 2], [2, 6, 2], [2, 7, 2],
    ... [3, 4, 3], [3, 3, 5], [3, 3, 6], [3, 3, 7]]
    ... )
    >>> Disent.bosdis(attributes, messages, vocab_size=3)
    1.0

    """

    def __init__(
        self,
        is_gumbel: bool,
        compute_posdis: bool = True,
        compute_bosdis: bool = False,
        vocab_size: int = 0,
        print_train: bool = False,
        print_test: bool = True,
        save_path: str = './',
        n_epochs: int = None
    ):
        super().__init__()
        # assert (
        #     print_train or print_test
        # ), "At least one of `print_train` and `print_train` must be set"
        # assert (
        #     compute_posdis or compute_bosdis
        # ), "At least one of `compute_posdis` and `compute_bosdis` must be set"
        assert (
            not compute_bosdis or vocab_size > 0
        ), "To compute a positive vocab_size must be specifed"

        self.vocab_size = vocab_size
        self.is_gumbel = is_gumbel
        self.compute_posdis = compute_posdis
        self.compute_bosdis = compute_bosdis
        self.print_train = print_train
        self.print_test = print_test
        self.save_path = save_path
        self.n_epochs = n_epochs

    def print_message(self, logs: Interaction, tag: str, epoch: int):
        messages = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        attributes = get_attributes(logs.sender_input)
        unique_attributes, unique_messages = get_unique_attributes_and_messages(attributes, messages)

        posdis = (
            Disent.posdis(unique_attributes, unique_messages) if self.compute_posdis else None
        )
        bosdis = (
            Disent.bosdis(unique_attributes, unique_messages, self.vocab_size)
            if self.compute_bosdis
            else None
        )

        output = json.dumps(dict(posdis=posdis, bosdis=bosdis, mode=tag, epoch=epoch))
        print(output, flush=True)

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if self.print_train:
            self.print_message(logs, "train", epoch)
        if epoch == self.n_epochs:
            messages = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
            attributes = get_attributes(logs.sender_input)
            unique_attributes, unique_messages = get_unique_attributes_and_messages(attributes, messages)
            posdis = Disent.posdis(unique_attributes, unique_messages)
            bosdis = Disent.bosdis(unique_attributes, unique_messages, self.vocab_size)
            pickle.dump({'posdis': posdis, 'bosdis': bosdis}, open(self.save_path + 'disent.pkl', 'wb'))

    def on_validation_end(self, loss, logs, epoch):
        if self.print_test:
            self.print_message(logs, "test", epoch)

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs: Interaction,
        epoch: int,
        test_loss: float = None,
        test_logs: Interaction = None,
    ):
        messages = train_logs.message.argmax(dim=-1) if self.is_gumbel else train_logs.message
        attributes = get_attributes(train_logs.sender_input)
        unique_attributes, unique_messages = get_unique_attributes_and_messages(attributes, messages)

        posdis = Disent.posdis(unique_attributes, unique_messages)
        bosdis = Disent.bosdis(unique_attributes, unique_messages, self.vocab_size)
        pickle.dump({'posdis': posdis, 'bosdis': bosdis}, open(self.save_path + 'disent.pkl', 'wb'))


class ConsoleLoggerCustom(Callback):
    def __init__(self, save_path='./', n_epochs=None):
        self.save_path = save_path
        self.metrics = {'train_loss': [], 'test_loss': [],
                        'train_acc': [], 'test_acc': [],
                        'train_length': [], 'test_length': [],
                        'train_acc_or': [], 'test_acc_or': [],
                        'train_sender_entropy': [], 'test_sender_entropy': [],
                        'train_receiver_entropy': [], 'test_receiver_entropy': []}
        self.n_epochs = n_epochs

    def aggregate_save(self, loss: float, logs: Interaction, mode: str, epoch: int):
        self.metrics[mode + '_loss'].append(loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        for key in aggregated_metrics.keys():
            if key != mode:
                self.metrics[mode + '_' + key].append(aggregated_metrics[key])

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        self.aggregate_save(loss, logs, "test", epoch)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        self.aggregate_save(loss, logs, "train", epoch)
        if epoch == self.n_epochs:
            pickle.dump(self.metrics, open(self.save_path + 'train_val_metrics.pkl', 'wb'))

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs: Interaction,
        epoch: int,
        test_loss: float = None,
        test_logs: Interaction = None,
    ):
        pickle.dump(self.metrics, open(self.save_path + 'train_val_metrics.pkl', 'wb'))


class CheckpointSaverCustom(CheckpointSaver):
    def __init__(self,
                 checkpoint_path: Union[str, pathlib.Path],
                 checkpoint_freq: int = 1, prefix: str = "",
                 max_checkpoints: int = sys.maxsize):
        """Saves a checkpoint file for training.
        :param checkpoint_path:  path to checkpoint directory, will be created if not present
        :param checkpoint_freq:  Number of epochs for checkpoint saving
        :param prefix: Name of checkpoint file, will be {prefix}{current_epoch}.tar
        :param max_checkpoints: Max number of concurrent checkpoint files in the directory.
        """
        super().__init__(checkpoint_path, checkpoint_freq, prefix, max_checkpoints)

    def on_early_stopping(
        self,
        train_loss: float,
        train_logs: Interaction,
        epoch: int,
        test_loss: float = None,
        test_logs: Interaction = None,
    ):
        self.save_checkpoint(
            filename=f"{self.prefix}_final" if self.prefix else "final"
        )


class InteractionSaverCustom(InteractionSaver):
    def __init__(
        self,
        train_epochs: Optional[List[int]] = None,
        test_epochs: Optional[List[int]] = None,
        checkpoint_dir: str = "",
        aggregated_interaction: bool = True,
        n_epochs: int = None
    ):
        super().__init__(train_epochs, test_epochs, checkpoint_dir, aggregated_interaction)
        self.n_epochs = n_epochs

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if epoch in self.test_epochs or epoch == self.n_epochs:
            if (
                not self.aggregated_interaction
                or self.trainer.distributed_context.is_leader
            ):
                rank = self.trainer.distributed_context.rank
                self.dump_interactions(logs, "train", epoch, rank, self.checkpoint_dir)

    def on_early_stopping(
            self,
            train_loss: float,
            train_logs: Interaction,
            epoch: int,
            test_loss: float = None,
            test_logs: Interaction = None,
    ):
        self.dump_interactions(train_logs, mode='train', epoch=epoch, rank=0, dump_dir=self.checkpoint_dir)
        self.dump_interactions(test_logs, mode='test', epoch=epoch, rank=0, dump_dir=self.checkpoint_dir)


class Evaluator(Callback):
    def __init__(self, loaders_metrics, device, freq=1, save_path='./'):
        self.loaders_metrics = loaders_metrics
        self.device = device
        self.epoch = 0
        self.freq = freq
        self.results = {}
        self.save_path = save_path

    def evaluate(self, train_end=False):
        game = self.trainer.game
        game.eval()
        old_loss = game.loss

        for loader_name, loader, metric in self.loaders_metrics:

            acc_or, acc = 0.0, 0.0
            n_batches = 0
            game.loss = metric

            for batch in loader:
                n_batches += 1
                if not isinstance(batch, Batch):
                    batch = Batch(*batch)
                batch = batch.to(self.device)
                with torch.no_grad():
                    _, interaction = game(*batch)
                acc += interaction.aux["acc"].mean().item()

                acc_or += interaction.aux["acc_or"].mean().item()

            self.results[loader_name] = {
                "acc": acc / n_batches,
                "acc_or": acc_or / n_batches,
            }

            if train_end:
                InteractionSaver.dump_interactions(logs=interaction,
                                                   mode=loader_name,
                                                   epoch=self.epoch,
                                                   rank=0,
                                                   dump_dir=self.save_path+'interactions')
        self.results["epoch"] = self.epoch
        output_json = json.dumps(self.results)
        print(output_json, flush=True)

        game.loss = old_loss
        game.train()

    def on_train_end(self):
        self.evaluate(train_end=True)

    def on_epoch_end(self, *stuff):
        self.epoch += 1

        if self.freq <= 0 or self.epoch % self.freq != 0:
            return
        self.evaluate()
