import torch
import utils
import models
import copy
import numpy as np
from metaflagent import MetaFlAgent
from tqdm import tqdm
from options import args_parser
from aggregation import Aggregation
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
from time import ctime
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from SecAggSimul import SecAggSimul
from utils import H5Dataset, generate_saliency_map

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = args_parser()
    secAggUnit = SecAggSimul()
    secAggUnit.reset_values()
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0
    utils.print_exp_details(args)

    # data recorders
    file_name = f"""time:{ctime()}-clip_val:{args.clip}-noise_std:{args.noise}""" \
                + f"""-aggr:{args.aggr}-s_lr:{args.server_lr}-num_cor:{args.num_corrupt}""" \
                + f"""thrs_robustLR:{args.robustLR_threshold}""" \
                + f"""-num_corrupt:{args.num_corrupt}-pttrn:{args.pattern_type}"""
    writer = SummaryWriter('logs/' + file_name)
    cum_poison_acc_mean = 0

    # load dataset and user groups (i.e., user to data mapping)
    train_dataset, val_dataset = utils.get_datasets(args.data)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)
    # fedemnist is handled differently as it doesn't come with pytorch
    if args.data != 'fedemnist':
        user_groups = utils.distribute_data(train_dataset, args)

    # poison the validation dataset
    idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=False)

    # initialize a model, and the agents
    global_model = models.get_model(args.data).to(args.device)
    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        if args.data == 'fedemnist':
            agent = MetaFlAgent(_id, args)
        else:
            agent = MetaFlAgent(_id, args, train_dataset, user_groups[_id])
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

    # aggregation server and the loss function
    n_model_params = len(parameters_to_vector(global_model.parameters()))
    aggregator = Aggregation(agent_data_sizes, n_model_params, poisoned_val_loader, args, writer)
    criterion = nn.CrossEntropyLoss().to(args.device)
    # Let's `configure these parameters
    total_cohort = args.num_cohorts
    client_in_cohort = args.client_per_cohort
    pois_cohort = args.num_p_cohorts
    pois_client = args.num_p_cohorts_clients

    #randomly_sampled_client = np.random.choice(total_cohort * client_in_cohort, total_cohort * client_in_cohort,
    #                                           replace=False).reshape(total_cohort, client_in_cohort)
    randomly_sampled_client = np.random.choice(args.num_agents, args.num_agents,replace=False).reshape(int(args.num_agents/(args.num_corrupt)), args.num_corrupt)

    for i in randomly_sampled_client[0,:]:
        agents[i].turn_malicious()
    
    # Meta FL training loop
    for rnd in tqdm(range(1, args.rounds + 1)):
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        cohort_agent_updates_dict = {}
        sign_updates_dict = {}
        sign_cohorts_dict = {}
        poisinous_client = np.random.choice(randomly_sampled_client[0,:],pois_cohort*pois_client ,replace=False)
        non_poisinous_client = np.random.choice(randomly_sampled_client[1:,:].flatten(),args.num_agents - args.num_corrupt,replace=False)
        cohort_counter = 0
        non_poisinous_client_counter = 0
        posinous_client_counter = 0
        for cohort in range(total_cohort):
            secAggUnit.reset_values()
            client_quota_counter = 0
            for client in range(client_in_cohort):
                if(client_quota_counter < pois_client and cohort_counter < pois_cohort):
                    agent_id = poisinous_client[posinous_client_counter]
                    posinous_client_counter = posinous_client_counter + 1
                    client_quota_counter = client_quota_counter + 1
                else:
                    agent_id = non_poisinous_client[non_poisinous_client_counter]
                    non_poisinous_client_counter = non_poisinous_client_counter + 1
                #agent_id = randomly_sampled_client[cohort][client]
                update, sign_update = agents[agent_id].local_train(global_model, criterion)
                sign_updates_dict[agent_id] = sign_update
                secAggUnit.submit_grad_ndata_prod(update * agent_data_sizes[agent_id])
                secAggUnit.submit_ndata_points(agent_data_sizes[agent_id])
                vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())
            cohort_counter = cohort_counter + 1
            cohort_agent_updates_dict[cohort] = secAggUnit.get_average_values()
            sign_cohorts_dict[cohort] = torch.sign(cohort_agent_updates_dict[cohort])
        aggregator.aggregate_updates(global_model, cohort_agent_updates_dict,sign_updates_dict,sign_cohorts_dict)

        # inference in every args.snap rounds
        if rnd % args.snap == 0:
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader,
                                                                                   args)
                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)
                print(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                print(f'| Val_Per_Class_Acc: {val_per_class_acc} ')

                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader,
                                                                         args)
                cum_poison_acc_mean += poison_acc
                writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean / rnd, rnd)
                print(f'| Poison Loss/Poison Acc: {poison_loss:.3f} / {poison_acc:.3f} |')


    if args.saliency_map == 'true':
        generate_saliency_map(args, global_model)
    print('Training has finished!')