import argparse
import os
import pickle
import sys


# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from MyProjects.CausalFed.FedDCM.Fed_TGAN.ihdp.ihdp_client import ihdp_Client

from MyProjects.CausalFed.FedDCM.Fed_TGAN.ihdp.dcm_conf import args
from MyProjects.CausalFed.FedDCM.Fed_TGAN.utils import get_data


from conf import conf
import torch

from MyProjects.CausalFed.FedDCM.Fed_TGAN.fedtgan.server import Server, StatisticalAggregation
from MyProjects.CausalFed.FedDCM.Fed_TGAN.fedtgan.data_transformer import DataTransformer

from MyProjects.CausalFed.FedDCM.Fed_TGAN.fedtgan.model import Discriminator, Generator,weights_init_normal
# from utils import get_data
import copy
from MyProjects.CausalFed.FedDCM.Fed_TGAN.similarity_test import table_similarity
import numpy as np



def init_global_models(conf):
    global_gens=[]
    global_discs=[]


    for iter, model_conf in enumerate(conf['gm_conf']):


        in_dim=0
        for key in model_conf['in_indx']:
            model_conf['in_indx'][key]= conf['lb_indx'][key]
            in_dim+=conf['lb_dim'][key]

        out_dim=0
        for key in model_conf['out_indx']:
            model_conf['out_indx'][key]= conf['lb_indx'][key]
            out_dim+=conf['lb_dim'][key]


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
        clients[key] = ihdp_Client(conf, args, train_datasets[key], key)
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
    min_metric= [100 for j in range(len(server.global_generator))]
    # y_est1_list=[]
    # y_est0_list = []



    ate_list={k:[] for k in clients.keys()}
    results={k:[] for k in clients.keys()}
    for e in range(conf['global_epochs']):

        for key in clients.keys():
            print("client {0} training in epoch {1}".format(key,e))

            #
            ##
            if conf['data_name'] == 'ihdp':  # perform intervention directly
                print('Cov generated from', conf['cov_from'])

                if conf['cov_from']=="model":
                    # Estimate ATE with model generated X~P(X)
                    cur_model = clients[key].local_generator[0]  # P(X)
                    x_output = clients[key].conditional_sample(cur_model, conf, inputs=[], n=len(clients[key].train_data))
                    cur_model = clients[key].local_generator[1]  # model for P(y_factual|X, treatment)
                elif conf['cov_from']=="train_data":
                    #Estimate ATE with own X~P(X)
                    x_output = clients[key].train_data[:, 0:-12]  #only covariates
                    cur_model = server.global_generator[0]  # model for P(y_factual|X, treatment)
                elif conf['cov_from']=="test_data":
                    # Estimate ATE with test X~P(X)
                    x_output = transformed_testdata[:, 0:-12]  #only covariates
                    cur_model = server.global_generator[0]  # model for P(y_factual|X, treatment)



                # --- do(T=1)
                treatment = torch.ones(x_output.shape[0], ).to(torch.int64)
                treatment = torch.nn.functional.one_hot(treatment, num_classes=2).numpy()

                xt_inputs = np.concatenate((x_output, treatment), axis=1)
                y_xt_output = clients[key].conditional_sample(cur_model, conf, inputs=xt_inputs, n=len(xt_inputs))
                allowed_vars = {**cur_model['in_indx'], **cur_model['out_indx']}
                syn_data = clients[key].transformer.inverse_transform(y_xt_output, allowed_vars)
                y_val = syn_data['y_factual']
                y_est1 = np.mean(y_val)

                # --- do(T=0)
                treatment = torch.zeros(x_output.shape[0], ).to(torch.int64)
                treatment = torch.nn.functional.one_hot(treatment, num_classes=2).numpy()
                xt_inputs = np.concatenate((x_output, treatment), axis=1)
                y_xt_output = clients[key].conditional_sample(cur_model, conf, inputs=xt_inputs,
                                                              n=len(xt_inputs))
                allowed_vars = {**cur_model['in_indx'], **cur_model['out_indx']}
                syn_data = clients[key].transformer.inverse_transform(y_xt_output, allowed_vars)
                y_val = syn_data['y_factual']
                y_est0 = np.mean(y_val)

                ate_list[key].append(y_est1 - y_est0)
                print(f'y_est|do(T=1): {y_est1}, y_est|do(T=0): {y_est0}, ATE: {y_est1 - y_est0}')
                print(f'Client:{key}, ate_list:{ate_list[key]}, mean: {np.mean(ate_list[key][-10:])}, std:{np.std(ate_list[key][-10:])}')
            #

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



        # Evaluation: data similarity tests by generating synthetic data(was outside the loop)
        ##
        print('ATE_dict', ate_list)
        # Save to pickle file
        filename= f"{args.saved_path}/{conf['exp_name']}/ate_dict.pkl"
        with open(filename, "wb") as f:
            pickle.dump(ate_list, f)

        print(f'Saved {filename}')



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

            if len(test_input)>0:
                test_input= np.concatenate(test_input, axis=1)
            syn_data = server.sample(n_sample,transformer,conf, test_input , cur_model, allowed_vars)

            avg_jsd, avg_wd, obs_tvd, obs_kl = table_similarity(syn_data,cur_testdata,conf["discrete_columns"][conf["data_name"]])
            print('---------')
            print(f"### Model:{i} epoch {e}, avg_jsd:{avg_jsd}, avg_wd:{avg_wd}, obs_tvd:{obs_tvd:.03f}, obs_kl:{obs_kl:.03f},  cur_best:{min_metric}")


            if obs_tvd==0:
                cur_metric= avg_wd
            else:
                cur_metric= obs_tvd

            # saving global model weights based on global model's tvd
            if cur_metric<=min_metric[i]: #Atleast after 5 epochs
                min_metric[i]= cur_metric
                save_dir = f"{args.saved_path}/{conf['exp_name']}/{conf['gm_conf'][i]['expr']}_global"
                os.makedirs(save_dir, exist_ok=True)
                filename = f"cond_model_epoch{e}_tvd{min_metric[i]:.4f}.pt"
                save_path = os.path.join(save_dir, filename)
                torch.save(cur_model['model'].state_dict(), save_path)
                print(f"Model saved at: {save_path}")





    return syn_data


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Parse arguments for the experiment.")
    parser.add_argument('--exp_name', type=str, default='ihdp_fedcm', help="name of the experiment")
    parser.add_argument('--isFL', type=str, default='True', help="How your federated models should be. Options: ['True' (regular FL), 'Pre-trained' (uses pre-trained global model),"
                                                                 " 'Individual' (each client train model independently)]")
    parser.add_argument('--isDCM', type=lambda v: v.lower() in ('true', '1', 'yes'), default=True, help="Do we want DCM training")
    parser.add_argument('--cov_from', type=str, default='model', help="where are the covariate values sampled from")

    parser.add_argument('--local_epochs', type=int, default=100, help="Local epoch of federated models.")
    parser.add_argument('--dcm_epochs', type=int, default=200, help="Number of epochs for training DCM.")
    parser.add_argument('--num_parties', type=int, default=3, help="Number of clients.")

    parser.add_argument('--per_client_samples', type=int, default=1000, help="Number of clients.")

    parser.add_argument('--pre_trained_global', type=str, default='', help="location of pre-trained models")

    parser.add_argument("--gpu",type=int,default=0,  help="Specify GPU index (e.g., --gpu 0 for first GPU, --gpu 1 for second GPU)")

    targs = parser.parse_args()



    #terminal commands
    # python3 ihdpFedTraining.py --local_epochs=1000 --isDCM=False --num_parties=3 --per_client_samples=50 --gpu=0
    # python3 ihdpFedTraining.py --local_epochs=1000 --isDCM=False --num_parties=3 --per_client_samples=200 --gpu=0
    # python3 ihdpFedTraining.py --local_epochs=1000 --isDCM=False --num_parties=12 --per_client_samples=50 --gpu=1

    targs.exp_name= f"{targs.exp_name}_{targs.num_parties}_{targs.per_client_samples}"
    dn= conf['data_name']
    loc= conf['train_dataset'][dn]
    conf['train_dataset'][dn]= f"{loc}{targs.num_parties}_{targs.per_client_samples}"
    conf['test_dataset'][dn]= f"{loc}{targs.num_parties}_{targs.per_client_samples}/test_data.csv"
    conf['cov_from']= targs.cov_from

    # python3 ihdpFedTraining.py --exp_name=ihdp_v2 --local_epochs=1000 --isDCM=False --cov_from=model --num_parties=3 --per_client_samples=50 --gpu=0
    # python3 ihdpFedTraining.py --exp_name=ihdp_shuffle1 --local_epochs=1000 --isDCM=False --cov_from=model --num_parties=3 --per_client_samples=50 --gpu=1
    # python3 ihdpFedTraining.py --exp_name=ihdp_shuffle2 --local_epochs=1000 --isDCM=False --cov_from=model --num_parties=3 --per_client_samples=50 --gpu=1

    # For run from debugger
    # args.dcm_dim = {}
    # conf['data_name'] = "adult"
    # targs.per_client_samples=1000


    # targs.pre_trained_global='./save/ihdp_fedcm'
    # targs.isDCM= False
    # targs.local_epochs= 1000
    # targs.num_parties=3
    # targs.per_client_samples = 50



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