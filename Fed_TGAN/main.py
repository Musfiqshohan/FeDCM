import os

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


def synthesize(n_sample):

    train_datasets, test_dataset, columns= get_data(conf)
    print("data partitiorn done !")

    clients = {}
    clients_num = {}

    for key in train_datasets.keys():
        clients[key] = Client(conf,train_datasets[key])
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
    transformer = DataTransformer(conf['parallel_transform'], conf['in_indx'], conf['out_indx'])
    transformer.fit(vir_data, conf['discrete_columns'][conf['data_name']])

    #input dimension
    data_dim = transformer.output_dimensions

    conf['in_indx'] = transformer.in_indx
    conf['out_indx'] = transformer.out_indx

    print('data dimensions :{}'.format(data_dim))

    print("local data encoding ....")
    for key in clients.keys():
        clients[key].data_encoding(transformer)
        clients[key].in_indx = conf['in_indx']
        clients[key].out_indx = conf['out_indx']

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


    clients_dis = {}
    clients_gen = {}

    #init models

    in_dim = sum([slot[1]-slot[0]  for slot in conf['in_indx'].values()])
    out_dim = sum([slot[1]-slot[0]  for slot in conf['out_indx'].values()])
    generator = Generator(latent_dim= conf['latent_size'], input_dim= in_dim, output_dim= out_dim, generator_dim= conf['generator_dim'])
    discriminator = Discriminator(input_dim= in_dim+ out_dim, discriminator_dim= conf['discriminator_dim'])
    # generator.apply(weights_init_normal)
    # discriminator.apply(weights_init_normal)


    # load model weights
    saved_models = conf['saved_models'][conf['data_name']]
    if os.path.exists(saved_models[0]):
        last_model = torch.load(saved_models[0], map_location="cuda")
        generator.load_state_dict(last_model)
        last_model = torch.load(saved_models[1], map_location="cuda")
        discriminator.load_state_dict(last_model)



    # if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

    #init server
    server = Server(discriminator,generator)

    #init client model
    for key in train_datasets.keys():
        clients[key].init_model(copy.deepcopy(server.global_discriminator), copy.deepcopy(server.global_generator))

    #federated training


    minL= [100]

    for e in range(conf['global_epochs']):

        for key in clients.keys():
            print("client {0} training in epoch {1}".format(key,e))
            discriminator_k, generator_k = clients[key].local_train(server.global_discriminator, server.global_generator)
            clients_dis[key] = copy.deepcopy(discriminator_k)
            clients_gen[key] = copy.deepcopy(generator_k)


        #weight aggregate
        server.model_aggregate(clients_dis,clients_gen,client_weight)


        # data similarity tests (was outside the loop)

        transformed_testdata=  transformer.transform(test_dataset)

        syn_data = server.sample(n_sample,transformer,conf, transformed_testdata)
        avg_jsd, avg_wd, obs_tvd, obs_kl = table_similarity(syn_data,test_dataset,conf["discrete_columns"][conf["data_name"]])
        # print("### epoch {0}, avg_jsd:{1}, avg_wd:{2}, obs_tvd:{.03f}, obs_kl:{.03f}".format(e,avg_jsd,avg_wd, obs_tvd, obs_kL))
        print(f"### epoch {e}, avg_jsd:{avg_jsd}, avg_wd:{avg_wd}, obs_tvd:{obs_tvd:.03f}, obs_kl:{obs_kl:.03f},  cur_best:{minL}")

        L= [obs_tvd]
        if sum([a < b for a, b in zip(L, minL)])==len(L): #all elements are smaller
            print('Found a minimum')
            minL= L
            save_path= f"./checkpoints/{conf['data_name']}/{conf['exp_name']}"
            os.makedirs(save_path, exist_ok=True)
            torch.save(generator.state_dict(), save_path + f"/gen_epoch{e:03}_tvd{obs_tvd:.3f}_kl{obs_kl:.3f}.pth")
            torch.save(discriminator.state_dict(), save_path + f"/disc_epoch{e:03}_tvd{obs_tvd:.3f}_kl{obs_kl:.3f}.pth")

    return syn_data


if __name__ == "__main__":

    syn_data = synthesize(1000)


    save_file= conf['syn_data'][conf['data_name']]
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    syn_data.to_csv(save_file, index=False)