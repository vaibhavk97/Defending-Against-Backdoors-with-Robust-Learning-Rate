import torch
import models
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
from collections import *

class Aggregation():
    def __init__(self, agent_data_sizes, n_params, poisoned_val_loader, args, writer):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.poisoned_val_loader = poisoned_val_loader
        self.cum_net_mov = 0

    def aggregate_updates(self, global_model, agent_updates_dict, agent_updates_sign=None):
        lr_vector = torch.Tensor([self.server_lr] * self.n_params).to(self.args.device)
        if self.args.robustLR_threshold > 0:
            if not self.args.cohort == 'true':
                lr_vector = self.compute_robustLR(agent_updates_dict)
            else:
                lr_vector = self.compute_robustLR_fromsgn(agent_updates_sign.values())

        aggregated_updates = 0
        if not self.args.cohort == 'true':
            if self.args.aggr == 'avg':
                aggregated_updates = self.agg_avg(agent_updates_dict)
            elif self.args.aggr == 'comed':
                aggregated_updates = self.agg_comed(agent_updates_dict)
            elif self.args.aggr == 'sign':
                aggregated_updates = self.agg_sign(agent_updates_dict)
        else:
            if self.args.aggr == 'avg':
                aggregated_updates = self.cohort_agg_avg(agent_updates_dict)
            elif self.args.aggr == 'comed':
                aggregated_updates = self.cohort_agg_comed(agent_updates_dict)
            elif self.args.aggr == 'krum':
                aggregated_updates = self.cohort_agg_avg_krum(agent_updates_dict)
            elif self.args.aggr == 'trimmed':
                aggregated_updates = self.cohort_agg_avg_trimmed_mean(agent_updates_dict)

        if self.args.noise > 0:
            aggregated_updates.add_(
                torch.normal(mean=0, std=self.args.noise * self.args.clip, size=(self.n_params,)).to(self.args.device))

        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params = (cur_global_params + lr_vector * aggregated_updates).float()
        vector_to_parameters(new_global_params, global_model.parameters())

        # some plotting stuff if desired
        # self.plot_sign_agreement(lr_vector, cur_global_params, new_global_params, cur_round)
        # self.plot_norms(agent_updates_dict, cur_round)
        return

    def compute_robustLR(self, agent_updates_dict):
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_of_signs = torch.abs(sum(agent_updates_sign))

        sm_of_signs[sm_of_signs < self.args.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.robustLR_threshold] = self.server_lr
        return sm_of_signs.to(self.args.device)

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        return sm_updates / total_data

    def compute_robustLR_fromsgn(self, agent_updates_sign):
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        sm_of_signs[sm_of_signs < self.args.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.robustLR_threshold] = self.server_lr
        return sm_of_signs.to(self.args.device)

    def cohort_agg_avg(self, cohort_updates_dict):
        """ classic fed avg for cohorts"""
        sm_updates, total_cohorts = 0, 0 #take simple average of the items
        for _id, update in cohort_updates_dict.items():
            sm_updates += update
            total_cohorts += 1
        aggregated_updates = sm_updates / total_cohorts
        return aggregated_updates

    def krum_create_distances(self, users_grads):
        distances = defaultdict(dict)
        for i in range(len(users_grads)):
            for j in range(i):
                distances[i][j] = distances[j][i] = torch.linalg.norm(users_grads[i] - users_grads[j])
        return distances

    def cohort_agg_avg_krum(self, cohort_users_grads, distances=None):
        """ krum for cohorts"""
        users_grads = cohort_users_grads[0].to(self.args.device)
        for i in range(1, len(cohort_users_grads)):
            users_grads = torch.vstack((users_grads, cohort_users_grads[i].to(self.args.device)))

        users_count = args.num_cohorts #need to fix this
        corrupted_count = 6 #need to fix this
        non_malicious_count = users_count - corrupted_count
        minimal_error = 1e20
        minimal_error_index = -1

        if distances is None:
            distances = self.krum_create_distances(users_grads)

        for user in distances.keys():
            errors = sorted(distances[user].values())
            current_error = sum(errors[:non_malicious_count])
            if current_error < minimal_error:
                minimal_error = current_error
                minimal_error_index = user

        return users_grads[minimal_error_index]

    def cohort_agg_avg_trimmed_mean(self, cohort_users_grads):
        users_grads = cohort_users_grads[0].to(self.args.device)
        for i in range(1, len(cohort_users_grads)):
            users_grads = torch.vstack((users_grads, cohort_users_grads[i].to(self.args.device)))
        beta = 0.25
        skip = int(beta*users_grads.shape[0])
        current_grads = torch.empty((users_grads.shape[1],), dtype=users_grads.dtype).to(self.args.device)
        for i, param_across_users in enumerate(users_grads.T):
            good_vals = sorted(param_across_users)[skip:-skip]
            current_grads[i] = np.mean(good_vals)
        return current_grads

    def cohort_agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        aggregated_updates = torch.median(concat_col_vectors, dim=1).values
        if self.args.noise > 0:
            aggregated_updates.add_(
                torch.normal(mean=0, std=self.args.noise * self.args.clip, size=(self.n_params,)).to(self.args.device))
        return aggregated_updates

    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values

    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)

    def clip_updates(self, agent_updates_dict):
        for update in agent_updates_dict.values():
            l2_update = torch.norm(update, p=2)
            update.div_(max(1, l2_update / self.args.clip))
        return

    def plot_norms(self, agent_updates_dict, cur_round, norm=2):
        """ Plotting average norm information for honest/corrupt updates """
        honest_updates, corrupt_updates = [], []
        for key in agent_updates_dict.keys():
            if key < self.args.num_corrupt:
                corrupt_updates.append(agent_updates_dict[key])
            else:
                honest_updates.append(agent_updates_dict[key])

        l2_honest_updates = [torch.norm(update, p=norm) for update in honest_updates]
        avg_l2_honest_updates = sum(l2_honest_updates) / len(l2_honest_updates)
        self.writer.add_scalar(f'Norms/Avg_Honest_L{norm}', avg_l2_honest_updates, cur_round)

        if len(corrupt_updates) > 0:
            l2_corrupt_updates = [torch.norm(update, p=norm) for update in corrupt_updates]
            avg_l2_corrupt_updates = sum(l2_corrupt_updates) / len(l2_corrupt_updates)
            self.writer.add_scalar(f'Norms/Avg_Corrupt_L{norm}', avg_l2_corrupt_updates, cur_round)
        return

    def comp_diag_fisher(self, model_params, data_loader, adv=True):

        model = models.get_model(self.args.data)
        vector_to_parameters(model_params, model.parameters())
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        precision_matrices = {}
        for n, p in deepcopy(params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        model.eval()
        for _, (inputs, labels) in enumerate(data_loader):
            model.zero_grad()
            inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                             labels.to(device=self.args.device, non_blocking=True).view(-1, 1)
            if not adv:
                labels.fill_(self.args.base_class)

            outputs = model(inputs)
            log_all_probs = F.log_softmax(outputs, dim=1)
            target_log_probs = outputs.gather(1, labels)
            batch_target_log_probs = target_log_probs.sum()
            batch_target_log_probs.backward()

            for n, p in model.named_parameters():
                precision_matrices[n].data += (p.grad.data ** 2) / len(data_loader.dataset)

        return parameters_to_vector(precision_matrices.values()).detach()

    def plot_sign_agreement(self, robustLR, cur_global_params, new_global_params, cur_round):
        """ Getting sign agreement of updates between honest and corrupt agents """
        # total update for this round
        update = new_global_params - cur_global_params

        # compute FIM to quantify these parameters: (i) parameters which induces adversarial mapping on trojaned, (ii) parameters which induces correct mapping on trojaned
        fisher_adv = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader)
        fisher_hon = self.comp_diag_fisher(cur_global_params, self.poisoned_val_loader, adv=False)
        _, adv_idxs = fisher_adv.sort()
        _, hon_idxs = fisher_hon.sort()

        # get most important n_idxs params
        n_idxs = self.args.top_frac  # math.floor(self.n_params*self.args.top_frac)
        adv_top_idxs = adv_idxs[-n_idxs:].cpu().detach().numpy()
        hon_top_idxs = hon_idxs[-n_idxs:].cpu().detach().numpy()

        # minimized and maximized indexes
        min_idxs = (robustLR == -self.args.server_lr).nonzero().cpu().detach().numpy()
        max_idxs = (robustLR == self.args.server_lr).nonzero().cpu().detach().numpy()

        # get minimized and maximized idxs for adversary and honest
        max_adv_idxs = np.intersect1d(adv_top_idxs, max_idxs)
        max_hon_idxs = np.intersect1d(hon_top_idxs, max_idxs)
        min_adv_idxs = np.intersect1d(adv_top_idxs, min_idxs)
        min_hon_idxs = np.intersect1d(hon_top_idxs, min_idxs)

        # get differences
        max_adv_only_idxs = np.setdiff1d(max_adv_idxs, max_hon_idxs)
        max_hon_only_idxs = np.setdiff1d(max_hon_idxs, max_adv_idxs)
        min_adv_only_idxs = np.setdiff1d(min_adv_idxs, min_hon_idxs)
        min_hon_only_idxs = np.setdiff1d(min_hon_idxs, min_adv_idxs)

        # get actual update values and compute L2 norm
        max_adv_only_upd = update[max_adv_only_idxs]  # S1
        max_hon_only_upd = update[max_hon_only_idxs]  # S2

        min_adv_only_upd = update[min_adv_only_idxs]  # S3
        min_hon_only_upd = update[min_hon_only_idxs]  # S4

        # log l2 of updates
        max_adv_only_upd_l2 = torch.norm(max_adv_only_upd).item()
        max_hon_only_upd_l2 = torch.norm(max_hon_only_upd).item()
        min_adv_only_upd_l2 = torch.norm(min_adv_only_upd).item()
        min_hon_only_upd_l2 = torch.norm(min_hon_only_upd).item()

        self.writer.add_scalar(f'Sign/Hon_Maxim_L2', max_hon_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Adv_Maxim_L2', max_adv_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Adv_Minim_L2', min_adv_only_upd_l2, cur_round)
        self.writer.add_scalar(f'Sign/Hon_Minim_L2', min_hon_only_upd_l2, cur_round)

        net_adv = max_adv_only_upd_l2 - min_adv_only_upd_l2
        net_hon = max_hon_only_upd_l2 - min_hon_only_upd_l2
        self.writer.add_scalar(f'Sign/Adv_Net_L2', net_adv, cur_round)
        self.writer.add_scalar(f'Sign/Hon_Net_L2', net_hon, cur_round)

        self.cum_net_mov += (net_hon - net_adv)
        self.writer.add_scalar(f'Sign/Model_Net_L2_Cumulative', self.cum_net_mov, cur_round)
        return
