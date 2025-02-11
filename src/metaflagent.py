import torch
import models
import utils
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn


class MetaFlAgent():

    def __init__(self, id, args, train_dataset=None, data_idxs=None):
        self.id = id
        self.args = args
        self.data_idxs = data_idxs
        self.train_dataset_init = train_dataset
        # get datasets, fedemnist is handled differently as it doesn't come with pytorch
        if train_dataset is None:
            self.train_dataset = torch.load(f'../data/Fed_EMNIST/user_trainsets/user_{id}_trainset.pt')
        else:
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)

        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=False)

        # size of local dataset
        self.n_data = len(self.train_dataset)

    def turn_malicious(self):
        if self.args.data == 'fedemnist':
            utils.poison_dataset(self.train_dataset, self.args, self.data_idxs, agent_idx=self.id)
        else:
            utils.poison_dataset(self.train_dataset_init, self.args, self.data_idxs, agent_idx=self.id)

    def local_train(self, global_model, criterion):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr,
                                    momentum=self.args.client_moment)

        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                 labels.to(device=self.args.device, non_blocking=True)

                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward()
                # to prevent exploding gradients
                nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                optimizer.step()

                # doing projected gradient descent to ensure the update is within the norm bounds
                if self.args.clip > 0:
                    with torch.no_grad():
                        local_model_params = parameters_to_vector(global_model.parameters())
                        update = local_model_params - initial_global_model_params
                        clip_denom = max(1, torch.norm(update, p=2) / self.args.clip)
                        update.div_(clip_denom)
                        vector_to_parameters(initial_global_model_params + update, global_model.parameters())

        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
            return update, torch.sign(update)

#    def get_update(self):
#        return self.update

#    def get_sign(self):
#        return torch.sign(self.update)
