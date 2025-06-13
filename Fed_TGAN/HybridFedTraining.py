import argparse
import os
import sys


# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))



from MyProjects.CausalFed.FedDCM.Fed_TGAN.dcm_conf import args

from conf import conf
import torch

from fedtgan.server import Server, StatisticalAggregation
from fedtgan.client import Client
from fedtgan.data_transformer import DataTransformer

from fedtgan.model import Discriminator, Generator,weights_init_normal
from utils import get_data
import copy
from similarity_test import table_similarity
import numpy as np





def init_global_models(conf):
    global_gens=[]
    global_discs=[]


    for iter, model_conf in enumerate(conf['gm_conf']):

        # in_dim = sum([dim for lb,dim in conf['lb_dim'] if lb in cond['in_indx']])
        # out_dim = sum([dim for lb,dim in conf['lb_dim'] if lb in cond['out_indx']])

        # in_indx= model_conf['in_indx']
        # out_indx= model_conf['out_indx']

        in_dim=0
        for key in model_conf['in_indx']:
            model_conf['in_indx'][key]= conf['lb_indx'][key]
            in_dim+=conf['lb_dim'][key]

        out_dim=0
        for key in model_conf['out_indx']:
            model_conf['out_indx'][key]= conf['lb_indx'][key]
            out_dim+=conf['lb_dim'][key]

        # in_dim = sum([slot[1] - slot[0] for slot in in_indx.values()])
        # out_dim = sum([slot[1] - slot[0] for slot in out_indx.values()])
        generator = Generator(latent_dim=conf['latent_size'], input_dim=in_dim, output_dim=out_dim,
                              generator_dim=conf['generator_dim'])
        discriminator = Discriminator(input_dim=in_dim + out_dim, discriminator_dim=conf['discriminator_dim'])
        # generator.apply(weights_init_normal)
        # discriminator.apply(weights_init_normal)

        if torch.cuda.is_available():
            generator.to(args.device)
            discriminator.to(args.device)


        global_gens.append({**model_conf, **{'model': generator}})
        global_discs.append({**model_conf, **{'model': discriminator}})

    return global_gens, global_discs


def synthesize(n_sample):

    train_datasets, test_dataset, columns= get_data(conf)

    print("data partitiorn done !")

    clients = {}
    clients_num = {}

    for key in train_datasets.keys():
        clients[key] = Client(conf, args, train_datasets[key], key)
        clients_num[key] = len(train_datasets[key])

    print("clients initialization done !")

    # federated feature encoding
    clients_categorical = {}
    clients_gmm = {}
    for key in clients.keys():
        cate_frequency, con_gmm = clients[key].compute_local_statistics()
        clients_categorical[key] = copy.deepcopy(cate_frequency)
        clients_gmm[key] = copy.deepcopy(con_gmm)

    print("local statistics aggregating ...")
    sa = StatisticalAggregation(conf, clients_categorical, clients_gmm, clients_num)
    vir_data = sa.construct_vir_data()
    #order the column
    vir_data = vir_data[columns]

    #global data transformer
    # transformer = DataTransformer(conf['parallel_transform'], conf['in_indx'], conf['out_indx'])
    transformer = DataTransformer(conf['parallel_transform'], args.dcm_dim)
    transformer.fit(vir_data, conf['discrete_columns'][conf['data_name']])

    #input dimension
    data_dim = transformer.output_dimensions

    conf['lb_indx'] = transformer.lb_indx
    conf['lb_dim'] = transformer.lb_dim

    print('data dimensions :{}'.format(data_dim))

    print("local data encoding ....")
    for key in clients.keys():
        clients[key].data_encoding(transformer)
        clients[key].lb_indx = transformer.lb_indx
        clients[key].lb_dim = transformer.lb_dim

    ##aggregate weight
    #compute the table-wise similarity weights
    client_weight = {}
    if conf["is_init_avg"]:
        ##fedavg
        for key in train_datasets.keys():
            client_weight[key] = 1 / len(train_datasets)
    else:
        jsd = sa.compute_jsd_matrix()
        wd = sa.compute_wd_matrix(vir_data)

        new_weight = sa.compute_new_weight(jsd,wd)
        for key in train_datasets.keys():
            client_weight[key] = new_weight[key]

        print("new weight = {}".format(new_weight))
    print('weight init done !')



    # init global models
    global_gens, global_discs =  init_global_models(conf)

    # init local DCM
    for key in train_datasets.keys():
        clients[key].init_dcm(conf, args)


    #init server
    server = Server(args, global_discs,global_gens)


    #init client model
    for key in train_datasets.keys():
        clients[key].init_model(copy.deepcopy(server.global_discriminator), copy.deepcopy(server.global_generator))



    transformed_testdata = transformer.transform(test_dataset)


    #federated training
    clients_dis={}
    clients_gen={}
    min_tvd= 100

    results={k:[] for k in clients.keys()}
    for e in range(conf['global_epochs']):

        for key in clients.keys():
            print("client {0} training in epoch {1}".format(key,e))
            discriminator_k, generator_k = clients[key].local_train(server.global_discriminator, server.global_generator)

            # saving individual model weights based on their individual tvd
            clients[key].evaluate_global_model(transformed_testdata, e)


            clients_dis[key] = copy.deepcopy(discriminator_k)
            clients_gen[key] = copy.deepcopy(generator_k)

            if conf['isDCM']:
                # generate interventional data from dcms. Specific to the causal graph.
                syn_intv_data = clients[key].intv_sample(len(clients[key].train_data), conf)

                # Train local DCM on both empirical obs data and intv syn_data generated from global models.
                real_obs_data = clients[key].sample_data(n_sample)
                real_obs_data = torch.from_numpy(real_obs_data.astype('float32')).to(args.device)
                results[key] = clients[key].local_dcm_train(args , real_obs_data, syn_intv_data, results[key], test_data= transformed_testdata)
            else:
                print('Caution No DCM happening')

        #weight aggregate of individually trained models (might be trained in FL manner or independently)
        print('Model aggregrating')
        server.model_aggregate(clients_dis,clients_gen,client_weight)


        # data similarity tests by generating synthetic data(was outside the loop)
        for i in range(len(server.global_generator)):

            cur_model= server.global_generator[i]

            test_input = []
            for inp in cur_model['in_indx']:
                test_input.append(transformed_testdata[:, cur_model['in_indx'][inp][0]: cur_model['in_indx'][inp][1]])


            test_outputs = []
            for inp in cur_model['out_indx']:
                test_outputs.append(transformed_testdata[:, cur_model['out_indx'][inp][0]: cur_model['out_indx'][inp][1]])

            cur_testdata = np.concatenate(test_input+test_outputs , axis=1)

            allowed_vars = {**cur_model['in_indx'], **cur_model['out_indx']}
            cur_testdata= transformer.inverse_transform(cur_testdata, allowed_vars)

            syn_data = server.sample(n_sample,transformer,conf, np.concatenate(test_input , axis=1), cur_model, allowed_vars)

            avg_jsd, avg_wd, obs_tvd, obs_kl = table_similarity(syn_data,cur_testdata,conf["discrete_columns"][conf["data_name"]])
            print('---------')
            print(f"### Model:{i} epoch {e}, avg_jsd:{avg_jsd}, avg_wd:{avg_wd}, obs_tvd:{obs_tvd:.03f}, obs_kl:{obs_kl:.03f},  cur_best:{min_tvd}")


            # saving global model weights based on global model's tvd
            if obs_tvd<=min_tvd: #Atleast after 5 epochs
                min_tvd= obs_tvd
                save_dir = f"{args.saved_path}/{conf['exp_name']}/{conf['gm_conf'][i]['expr']}_global"
                os.makedirs(save_dir, exist_ok=True)
                filename = f"cond_model_epoch{e}_tvd{min_tvd:.4f}.pt"
                save_path = os.path.join(save_dir, filename)
                torch.save(cur_model['model'].state_dict(), save_path)
                print(f"Model saved at: {save_path}")


    return syn_data


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Parse arguments for the experiment.")
    parser.add_argument('--exp_name', type=str, default='fedcm', help="name of the experiment")
    parser.add_argument('--isFL', type=str, default='True', help="How your federated models should be. Options: ['True' (regular FL), 'Pre-trained' (uses pre-trained global model),"
                                                                 " 'Individual' (each client train model independently)]")
    parser.add_argument('--isDCM', type=lambda v: v.lower() in ('true', '1', 'yes'), default=True, help="Do we want DCM training")

    parser.add_argument('--local_epochs', type=int, default=100, help="Local epoch of federated models.")
    parser.add_argument('--dcm_epochs', type=int, default=200, help="Number of epochs for training DCM.")
    parser.add_argument('--num_parties', type=int, default=5, help="Number of clients.")

    parser.add_argument('--per_client_samples', type=int, default=1000, help="Number of clients.")

    parser.add_argument('--pre_trained_global', type=str, default='', help="location of pre-trained models")
    parser.add_argument("--gpu",type=int,default=0,  help="Specify GPU index (e.g., --gpu 0 for first GPU, --gpu 1 for second GPU)")

    #for causal effect maximization
    parser.add_argument('--max_intv', nargs='+', default=[], help="'maximize', 'Z', 1, 0.2: maximize do(Z=1) with 0.5 weight")
    parser.add_argument('--mediator_dim', type=int, default=2, help="Dimension of mediator")


    targs = parser.parse_args()



    #terminal commands
    # python3 HybridFedTraining.py --exp_name=dFL_noDCM_128x5x1000 --local_epochs=200 --dcm_epochs=200 --num_parties=5 --isFL=True --isDCM=False --per_client_samples=1000 --gpu=0
    # python3 HybridFedTraining.py --exp_name=dnoFL_noDCM_128x5x1000 --local_epochs=200 --dcm_epochs=200 --num_parties=5 --isFL=Individual --isDCM=False --per_client_samples=1000 --gpu=1

    # python3 HybridFedTraining.py --exp_name=fedcm_maximize --local_epochs=200 --dcm_epochs=300 --num_parties=3 --isFL=True --isDCM=True --per_client_samples=1000 --max_intv maximize X 1 0.2 --gpu=0
    # python3 HybridFedTraining.py --exp_name=fedcm_minimize --local_epochs=200 --dcm_epochs=300 --num_parties=3 --isFL=True --isDCM=True --per_client_samples=1000 --max_intv minimize X 1 0.2 --gpu=1
    #

    # python3 HybridFedTraining.py --exp_name=nonid_max --local_epochs=200 --dcm_epochs=300 --num_parties=3 --isFL=Pre-trained --pre_trained_global=./save/nonid/ --isDCM=True --per_client_samples=1000 --max_intv maximize Z 1 0.3 --gpu=0
    #


    # For run from debugger
    # args.dcm_dim = {}
    # conf['data_name'] = "adult"
    # targs.local_epochs= 100
    # targs.per_client_samples=1000
    # targs.isFL= 'Pre-trained'
    # targs.isFL= 'True'
    # targs.isDCM= True
    # targs.pre_trained_global= './save/nonid/'
    # targs.exp_name='nonid_min'
    # targs.max_intv=['minimize', 'Z', 1, 0.3]
    # targs.num_parties=3


    args.dcm_dim = {'Z': 2, 'W':2, 'X': targs.mediator_dim, 'Y': 2}
    # args.dcm_dim = {'Z': 2, 'X': targs.mediator_dim, 'Y': 2}


    # for maximizing causal effects
    args.max_intv = [
        int(str(x)) if str(x).isdigit()
        else float(str(x)) if str(x).replace('.', '', 1).isdigit()
        else x
        for x in targs.max_intv
    ]





    # for frontdoor graph.
    dn= conf['data_name']
    loc= conf['train_dataset'][dn]
    conf['train_dataset'][dn]= f"{loc}{args.dcm_dim['X']}"
    conf['test_dataset'][dn]= f"{loc}{args.dcm_dim['X']}/test_data.csv"

    conf['exp_name'] = targs.exp_name
    conf['isFL'] = targs.isFL
    conf['isDCM'] = targs.isDCM

    conf['local_epochs'] = targs.local_epochs
    conf['num_parties'] = targs.num_parties
    conf['per_client_samples'] = targs.per_client_samples
    conf['pre_trained_global'] = targs.pre_trained_global

    args.dcm_epochs = targs.dcm_epochs
    args.device = torch.device(f"cuda:{targs.gpu}" if torch.cuda.is_available() else "cpu")


    print('Conf--->', conf)
    print('Args--->', args)

    syn_data = synthesize(1000)


    save_file= conf['syn_data'][conf['data_name']]
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    syn_data.to_csv(save_file, index=False)