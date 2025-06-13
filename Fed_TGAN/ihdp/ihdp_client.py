import os
import pickle

import torch
from torch import optim
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from MyProjects.CausalFed.FedDCM.Fed_TGAN.similarity_test import table_similarity
from MyProjects.PosID.trainDCM.noisyDCM import get_generators
from MyProjects.modularUtils.evaluate_distributions import get_fake_distribution, get_joint_distributions_from_samples, \
    map_to_discrete
from MyProjects.modularUtils.models import ControllerDiscriminator
import os
import torch
import re

class ihdp_Client(object):

    def __init__(self, conf, args, train_df, clno, cuda=True):
        """
        client side
        """

        self.client_no =clno
        self.conf= conf
        self.train_df = train_df

        self._batch_size = conf['batch_size']

        self.local_epoch = conf['local_epochs']
        self.local_discriminator_steps = conf['local_discriminator_steps']

        self.gen_lr = conf['gen_lr']
        self.gen_weight_decay = conf['gen_weight_decay']
        self.dis_lr = conf['dis_lr']
        self.dis_weight_decay = conf['dis_weight_decay']

        self.discrete_columns = conf['discrete_columns'][conf['data_name']]
        self.max_clusters = conf['max_clusters']
        self.embedding_dim = conf['latent_size']



        # new
        self.device = args.device

        self.lb_indx = {}
        self.lb_dim = {}

        self.args= args   # for dcm
        self.min_tvd=100


    def init_model(self,discriminator, generator ):
        """
        :param discriminator:
        :param generator:
        :return:
        """
        self.local_discriminator = discriminator
        self.local_generator = generator


    def apply_activate(self,data, out_indx):

        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for key, column_info in self.transformer.output_info_dict.items():

            if key not in out_indx:
                continue

            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = F.gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)



    def train_global_model(self, disc, gen):

        optimizerG = optim.Adam(
            gen['model'].parameters(), lr=self.gen_lr, betas=(0.5, 0.9),
            weight_decay=self.gen_weight_decay
        )

        optimizerD = optim.Adam(
            disc['model'].parameters(), lr=self.dis_lr,
            betas=(0.5, 0.9), weight_decay=self.dis_weight_decay
        )

        mean = torch.zeros(self._batch_size, self.embedding_dim, device=self.device)

        std = mean + 1

        training_step_per_epoch = max(len(self.train_data) // self._batch_size, 1)

        in_indx = gen['in_indx']
        out_indx = gen['out_indx']

        for i in range(self.local_epoch):

            # self.local_generator.train()
            # self.local_discriminator.train()
            for j in range(training_step_per_epoch):
                # taining discriminator
                for n_d in range(self.local_discriminator_steps):

                    # real data
                    real = self.sample_data(self._batch_size)
                    real = torch.from_numpy(real.astype('float32')).to(self.device)



                    inputs = []
                    for inp in in_indx:
                        inputs.append(real[:, in_indx[inp][0]: in_indx[inp][1]])
                    outputs = []
                    for inp in out_indx:
                        outputs.append(real[:, out_indx[inp][0]: out_indx[inp][1]])

                    real = inputs + outputs
                    real = torch.cat(real, dim=1)

                    real_critic = disc['model'](real)

                    # fake data generation
                    noise = torch.normal(mean=mean, std=std)
                    fake = gen['model']([noise], inputs)
                    fakeact = self.apply_activate(fake, out_indx)
                    fakeact = inputs + [fakeact]
                    fakeact = torch.cat(fakeact, dim=1)

                    fake_critic = disc['model'](fakeact)

                    pen = disc['model'].calc_gradient_penalty(real, fakeact, self.device)
                    loss_d = -(torch.mean(real_critic) - torch.mean(fake_critic))

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                # # training generator
                # if index % self.local_discriminator_steps == 0:
                noise = torch.normal(mean=mean, std=std)
                fake = gen['model']([noise], inputs)  # same input as discriminator
                fakeact = self.apply_activate(fake, out_indx)
                fakeact = inputs + [fakeact]
                fakeact = torch.cat(fakeact, dim=1)  # concatenating inputs with fake outputs.

                fake_critic = disc['model'](fakeact)
                loss_g = -torch.mean(fake_critic)
                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            print("Epoch {0}, loss G: {1}, loss D: {2}".format(i + 1, loss_g.detach().cpu(),
                                                               loss_d.detach().cpu()))

        return disc['model'].state_dict(), gen['model'].state_dict()



    def load_best_models(self, client_id, base_folder, distribution):

        #load the global model to each client. Optionally: the previous local model to each client.
        # client_folder = os.path.join(base_folder, f"{distribution}_client{client_id}")
        client_folder = os.path.join(base_folder, f"{distribution}_global")
        files = os.listdir(client_folder)

        def get_best_model(pattern):
            matches = [re.match(pattern, f) for f in files]
            best_file = min(
                (m for m in matches if m),
                key=lambda m: float(m.group(1)),
                default=None
            )
            return os.path.join(client_folder, best_file.group(0)) if best_file else None

        gen_path = get_best_model(r"cond_model_epoch\d+_tvd([\d.]+)\.pt")     # global model saved as "model"; Just naming issue.
        # gen_path = get_best_model(r"cond_generator_epoch\d+_tvd([\d.]+)\.pt")  #local model saved as "generator"
        # critic_path = get_best_model(r"cond_critic_epoch\d+_tvd([\d.]+)\.pt")


        # return torch.load(gen_path, map_location=torch.device(self.device)) if gen_path else None, torch.load(critic_path, map_location=torch.device(self.device)) if critic_path else None
        return torch.load(gen_path, map_location=torch.device(self.device)) if gen_path else None


    def local_train(self, discriminators, generators):
        """
        :param discriminator:
        :param generator:
        :return:
        """

        # update the local generator and discriminator

        d_dict = []
        g_dict = []
        for iter in range(len(generators)):

            disc = discriminators[iter]
            gen= generators[iter]

            print('Training', gen['expr'])


            if self.conf['isFL']=='True': # global models will be used as local models
                print('Copying global models to local models')

                for name, param in disc['model'].state_dict().items():
                    self.local_discriminator[iter]['model'].state_dict()[name].copy_(param.clone())

                for name, param in gen['model'].state_dict().items():
                    self.local_generator[iter]['model'].state_dict()[name].copy_(param.clone())

                d, g = self.train_global_model(self.local_discriminator[iter], self.local_generator[iter])

            elif self.conf['isFL'] == 'Pre-trained':
                #load models

                best_gen_model = self.load_best_models(self.client_no, self.conf['pre_trained_global'], self.conf['gm_conf'][iter]['expr'])
                print("Loaded best generator and critic models.")

                for name, param in best_gen_model.items():
                    self.local_generator[iter]['model'].state_dict()[name].copy_(param.clone())
                g = self.local_generator[iter]['model'].state_dict()

                #keep the same disriminator as before as it is not useful
                d= self.local_discriminator[iter]['model'].state_dict()

            elif self.conf['isFL']=='Individual':
                print('Caution! No federated learning is being performed! Clients training global models independently')
                d, g = self.train_global_model(self.local_discriminator[iter], self.local_generator[iter])

            else:
                raise Exception("Unknown input for isFL")



            d_dict.append(d)
            g_dict.append(g)

        return d_dict, g_dict



    def init_dcm(self, conf, args):


        latent_conf = {var:[] for var in args.Observed_DAG}

        for cnf in args.confounders:
            for var in args.confounders[cnf]:
                latent_conf[var].append(cnf)

        args.latent_conf = latent_conf

        # label_slots = {**conf['in_indx'], **conf['out_indx']}
        args.label_dim = conf['lb_dim']

        args.label_names= list(args.Observed_DAG.keys())

        self.dcm, self.dcm_gopt = get_generators(args)

        critic = {}
        dopt = {}
        critic['obs'] = ControllerDiscriminator(input_dim=sum(list(args.label_dim.values())),
                                                discriminator_dim=args.hid_dims).to(self.device)
        dopt['obs'] = torch.optim.Adam(critic['obs'].parameters(), lr=args.learning_rate, betas=(0.5, 0.9),
                                       weight_decay=1e-5)

        # dims = 0
        # for lb in args.Observed_DAG:
        #     if args.highdim_var in args.Observed_DAG[lb]:
        #         dims += args.label_dim[lb]


        ith_model=0
        dims=[args.label_dim[key] for key in conf['gm_conf'][ith_model]['in_indx']]
        dims +=[args.label_dim[key] for key in conf['gm_conf'][ith_model]['out_indx']]
        dims = sum(dims)


        iter = 0
        each_intv = args.intv_list[iter]
        dims=0      #dimension of c-factor.
        for key in each_intv['inter_vars'] + each_intv['obs']:
            dims+= self.conf['lb_dim'][key]

        critic['intv'] = ControllerDiscriminator(input_dim=dims, discriminator_dim=args.hid_dims).to(self.device)
        dopt['intv'] = torch.optim.Adam(critic['intv'].parameters(), lr=args.learning_rate, betas=(0.5, 0.9),
                                        weight_decay=1e-5)

        self.dcm_critic = critic
        self.dcm_dopt= dopt

    def evaluate_global_model(self, transformed_testdata, epoch):

        for i in range(len(self.local_generator)):

            cur_model = self.local_generator[i]

            test_input = []
            for inp in cur_model['in_indx']:
                test_input.append(transformed_testdata[:, cur_model['in_indx'][inp][0]: cur_model['in_indx'][inp][1]])

            test_outputs = []
            for inp in cur_model['out_indx']:
                test_outputs.append(
                    transformed_testdata[:, cur_model['out_indx'][inp][0]: cur_model['out_indx'][inp][1]])

            cur_testdata = np.concatenate(test_input + test_outputs, axis=1)

            # Which order? Y is at the end.

            allowed_vars = {**cur_model['in_indx'], **cur_model['out_indx']}
            cur_testdata = self.transformer.inverse_transform(cur_testdata, allowed_vars)


            if len(test_input)>0:
                test_input= np.concatenate(test_input, axis=1)
            syn_data= self.conditional_sample(cur_model, self.conf, test_input , cur_testdata.shape[0])

            allowed_vars = {**cur_model['in_indx'], **cur_model['out_indx']}
            syn_data= self.transformer.inverse_transform(syn_data, allowed_vars)

            avg_jsd, avg_wd, obs_tvd, obs_kl = table_similarity(syn_data, cur_testdata,self.conf["discrete_columns"][self.conf["data_name"]])
            print('---------')
            print(
                f"###Client{self.client_no} cond model:{i}, avg_jsd:{avg_jsd}, avg_wd:{avg_wd}, obs_tvd:{obs_tvd:.03f}, obs_kl:{obs_kl:.03f}, cur_best:{self.min_tvd}")



            #
            if obs_tvd<=self.min_tvd: #Atleast after 5 epochs:
                self.min_tvd= obs_tvd
                save_dir = f"{self.args.saved_path}/{self.conf['exp_name']}/{self.conf['gm_conf'][i]['expr']}_client{self.client_no}"
                os.makedirs(save_dir, exist_ok=True)
                print(f"Model saved at: {save_dir}")

                filename = f"cond_generator_epoch{epoch}_tvd{self.min_tvd:.4f}.pt"
                save_path = os.path.join(save_dir, filename)
                torch.save(self.local_generator[i]['model'].state_dict(), save_path)

                filename = f"cond_critic_epoch{epoch}_tvd{self.min_tvd:.4f}.pt"
                save_path = os.path.join(save_dir, filename)
                torch.save(self.local_discriminator[i]['model'].state_dict(), save_path)



    def evaluate_dcm(self, args, label_generators, current_real_label, results, sample_size):
        for gen in label_generators:
            label_generators[gen].eval()

        with torch.no_grad():

            intv_key={}
            compare_Var= args.label_names
            generated_labels_dict = self.get_generated_labels(args, label_generators, dict(intv_key), compare_Var, sample_size)

            #
            generated_labels_full = torch.cat([generated_labels_dict[lb] for lb in compare_Var], dim=1).cpu().numpy()
            syn_data= self.transformer.inverse_transform(generated_labels_full, self.conf['lb_indx'])
            avg_jsd, avg_wd, obs_tvd, obs_kl = table_similarity(syn_data, current_real_label,self.conf["discrete_columns"][self.conf["data_name"]])
            #


            # generated_labels_full = map_to_discrete(args, generated_labels_dict, compare_Var)
            # fake = get_joint_distributions_from_samples(args.label_dim, generated_labels_full)
            # real = get_joint_distributions_from_samples(args.label_dim,current_real_label.astype(int))
            # print(f' fake: {fake}')
            # print(f' Real: {real}')
            # obs_tvd = 0.5 * sum([abs(fake[key] - real[key]) for key in fake])
            # obs_kl = sum([(real[key]) * np.log(real[key] / (fake[key])) for key in fake])


            cur_res= {'tvd': round(obs_tvd, 4),
                            'kl': round(obs_kl, 4),
                            'epoch': args.cur_epoch
                            }


            # do intervention

            for iter, cur_intv in enumerate(args.intv):


                if self.conf['data_name']=='ihdp':
                    gen_intv = \
                    self.get_generated_labels(args, label_generators, intervened={cur_intv['key']: cur_intv['val']},
                                              chosen_labels=['y_factual'],
                                              mini_batch=sample_size)['y_factual']

                    result=  np.mean(self.transformer.inverse_transform(gen_intv.cpu().numpy(), {'y_factual': self.conf['lb_indx']['y_factual']}))


                else:
                    gen_intv = self.get_generated_labels(args, label_generators, intervened={cur_intv['key']: cur_intv['val']},
                                              chosen_labels=['Y'],
                                              mini_batch=sample_size)['Y']
                    gen_intv = torch.argmax(gen_intv, dim=1).view(-1, 1)

                    result = get_joint_distributions_from_samples({'Y': 2}, gen_intv.cpu())

                expr = f"P(Y | do({cur_intv['key']}={cur_intv['val']}))"
                cur_res[iter] = (expr, result)


            results.append(cur_res)


        for gen in label_generators:
            label_generators[gen].train()

        ll = -min(10, len(results))


        # printing loss
        print("### P(V)", " TVD:", [round(val['tvd'], 4) for val in results[ll:]])
        print("### P(V)", " KL:", [round(val['kl'], 4) for val in results[ll:]])
        print(f'### {results[-1][0][0]}={results[-1][0][1]}')
        print(f'### {results[-1][1][0]}={results[-1][1][1]}')

        # print(f'### P(y|do({args.intv["key"]}={args.intv["val"]})): {fidict}')
        # print(f'### P(y|do({args.intv["key"]}={args.intv["val"]})): {fidict}')



        save_loc = f"{args.saved_path}/{self.conf['exp_name']}"
        os.makedirs(save_loc, exist_ok=True)
        save_loc = f"{save_loc}/dcm_res_cl{self.client_no}.pkl"
        with open(save_loc, 'wb') as ff:
            pickle.dump(results, ff)

        # if args.cur_epoch % 10 == 1:
        print('Saving labels at', save_loc)


        return results

    def get_generated_labels(self, args, label_generators, intervened, chosen_labels, mini_batch):
            label_noises = {}
            conf_noises = {}
            if not label_noises:
                for name in args.label_names:
                    label_noises[name] = torch.randn(mini_batch, args.NOISE_DIM).to(self.device)  # white noise. no bias

            if not conf_noises:
                for label in args.label_names:
                    confounders = args.latent_conf[label]
                    for confx in confounders:  # no confounder name, only their sequence matters here.
                        conf_noises[confx] = torch.randn(mini_batch, args.NOISE_DIM).to(self.device)  # white noise. no bias

            gen_labels = {}
            for lbid, label in enumerate(args.Observed_DAG):

                # first adding exogenous noise
                Noises = []
                Noises.append(label_noises[label])

                # secondly, adding confounding noise
                for confx in args.latent_conf[label]:
                    Noises.append(conf_noises[confx])

                # getting observed parent values
                parent_gen_labels = []
                for parent in args.Observed_DAG[label]:
                    parent_gen_labels.append(gen_labels[parent])

                if label in intervened.keys():
                    if torch.is_tensor(intervened[label]):
                        gen_labels[label] = intervened[label]
                    else:
                        gen_labels[label] = torch.ones(mini_batch, args.label_dim[label]).to(self.device) * 0.00001
                        gen_labels[label][:, intervened[label]] = 0.99999

                else:
                    gen_labels[label] = label_generators[label](Noises, parent_gen_labels, args.Temperature,
                                                                gumbel_noise=None,
                                                                hard=False)

            return_labels = {}
            for label in chosen_labels:
                return_labels[label] = gen_labels[label]

            return return_labels



    # def train_batch(self, args, generators, G_optimizers, critic, D_optimizer, batch_data, optimize=None):

    def train_batch(self, args, dcm, dcm_gopt, dcm_critic, dcm_dopt, obs_batch, intv_batch, optimize=None):

        ########### obs critic #######
        real_labels = obs_batch
        real_critic = dcm_critic['obs'](real_labels)
        gen_output = self.get_generated_labels(args, dcm, intervened={}, chosen_labels=args.label_names, mini_batch=args.batch_size)
        gen_labels = torch.cat(list(gen_output.values()), dim=1)  # Real Z, Fake X, Fake Y
        fake_critic = dcm_critic['obs'](gen_labels)
        pen = dcm_critic['obs'].calc_gradient_penalty(real_labels, gen_labels, device=self.device)
        loss_d = -(torch.mean(real_critic) - torch.mean(fake_critic))
        dcm_dopt['obs'].zero_grad()
        pen.backward(retain_graph=True)
        loss_d.backward()
        dcm_dopt['obs'].step()

        ########### intv critic #######
        real_labels = intv_batch
        real_critic = dcm_critic['intv'](real_labels)


        # getting intervened values for c-factors; same values are used in generator training.
        cc_parents={}
        # for iter, each_intv in enumerate(args.intv_list):
        iter= 0
        each_intv = args.intv_list[iter]
        for key in each_intv['inter_vars']:
            indices= slice(*self.conf['lb_indx'][key])
            cc_parents[key]= intv_batch[:, indices]
        cc_variables = each_intv['inter_vars']+ each_intv['obs']  #c-component and their parents

        gen_output = self.get_generated_labels(args, dcm, intervened=cc_parents, chosen_labels=cc_variables, mini_batch=args.batch_size)
        gen_labels = torch.cat(list(gen_output.values()), dim=1)  # fake samples from [parents, c-factor]
        fake_critic = dcm_critic['intv'](gen_labels)
        pen = dcm_critic['intv'].calc_gradient_penalty(real_labels, gen_labels, device=self.device)
        loss_d = -(torch.mean(real_critic) - torch.mean(fake_critic))
        dcm_dopt['intv'].zero_grad()
        pen.backward(retain_graph=True)
        loss_d.backward()
        dcm_dopt['intv'].step()


        ############ generator  training  ############
        # fake obs generation
        gen_output = self.get_generated_labels(args, dcm, intervened={}, chosen_labels=args.label_names, mini_batch=args.batch_size)
        gen_labels = torch.cat(list(gen_output.values()), dim=1)  # Real Z, Fake X, Fake Y
        fake_critic = dcm_critic['obs'](gen_labels)
        obs_loss_g = -torch.mean(fake_critic)

        # fake intv generation
        # graph-specific; need generalization
        gen_output = self.get_generated_labels(args, dcm, intervened=cc_parents, chosen_labels= cc_variables, mini_batch=args.batch_size)
        gen_labels = torch.cat(list(gen_output.values()), dim=1)  # Real Z, Fake X, Fake Y
        fake_critic = dcm_critic['intv'](gen_labels)
        intv_loss_g = -torch.mean(fake_critic)

        loss_g = obs_loss_g + intv_loss_g

        for lb in dcm_gopt:  # do we need one optimizer?
            dcm_gopt[lb].zero_grad()

        loss_g.backward()

        for lb in dcm_gopt:
            dcm_gopt[lb].step()

        return loss_g.data, loss_d.data, gen_labels

    def local_dcm_train(self, args, real_obs_data, syn_intv_data, results, test_data):

        # dataset = TensorDataset(syn_real_data)

        # Create datasets
        dataset_real = TensorDataset(real_obs_data)
        dataset_syn = TensorDataset(syn_intv_data)

        dataloader_real = DataLoader(dataset_real, batch_size=args.batch_size, shuffle=True, drop_last=True)
        dataloader_syn = DataLoader(dataset_syn, batch_size=args.batch_size, shuffle=True, drop_last=True)

        iteration = 0
        for epoch in tqdm(range(args.dcm_epochs)):

            # Iterate over both datasets in parallel
            for (batch_real, batch_syn) in zip(dataloader_real, dataloader_syn):
                real_batch_data = batch_real[0]  # Extract data from TensorDataset
                syn_batch_data = batch_syn[0]  # Extract data from TensorDataset

                g_loss, d_loss, gen_labels = self.train_batch(args, self.dcm, self.dcm_gopt, self.dcm_critic, self.dcm_dopt, real_batch_data, syn_batch_data)

            if (epoch + 1) % 100 == 0:
                # print("Turn on caffeinate or these results are gone!")
                print('Epoch:', epoch)
                args.cur_epoch= epoch

                # cur_obs= self.transformer.inverse_transform(real_obs_data.detach().cpu(), args.label_names)
                cur_obs= self.transformer.inverse_transform(test_data, args.label_names)

                results = self.evaluate_dcm(args, self.dcm, cur_obs, results, sample_size=5000)

        return results


    def compute_local_statistics(self):
        """
        :return: compute the frequency of categorical columns and the gmm for continuous columns
        """
        columns = self.train_df.columns

        categorical = {}
        continuous = {}

        for c in columns:

            if c in self.discrete_columns:
                categorical[c] = self.categorical_frequency(c)
            else:
                continuous[c] = self.continuous_gmm(c)

        return categorical, continuous


    def categorical_frequency(self,column):
        """
        :param column: categorical column name
        :return: frequency of categories
        """

        fre = self.train_df[column].value_counts()
        categorical_frequency = {}
        for cat, value in zip(fre.index, fre.values):
            categorical_frequency[cat] = value
        return categorical_frequency

    def continuous_gmm(self,column):
        """
        :param column: continuous column
        :return: GMM of the continuous column
        """

        data = self.train_df[column].values

        gmm = BayesianGaussianMixture(
            n_components=self.max_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )
        gmm.fit(data.reshape(-1,1))

        return gmm

    def data_encoding(self,transformer):
        """
        :param transformer:
        :return: encode the local data by the global transformer
        """

        self.transformer = transformer

        self.train_data = self.transformer.transform(self.train_df)


    def sample_data(self, size):
        """
        :return: sample data for training
        """

        data_size = len(self.train_data)


        index = np.random.randint(data_size, size=size)

        return self.train_data[index]



    def intv_sample(self, n, conf): # graph-specific; need generalization

        inputs = []

        #
        if self.conf['data_name']=='ihdp':
            cur_model = self.local_generator[0]  # P(X)
            output = self.conditional_sample(cur_model, conf, inputs, n)



        if self.conf['data_name']=='frontdoor':
            cur_model=  self.local_generator[0]  #P(X|Z)
            real = self.sample_data(n)
            inp = 'Z'
            inputs.append(real[:, cur_model['in_indx'][inp][0]: cur_model['in_indx'][inp][1]])
            inputs = np.concatenate(inputs, axis=1)
            output= self.conditional_sample(cur_model, conf, inputs, n)
        #

        # cur_model=  self.local_generator[1]  #P(Y|Z,X)
        # #Sample Z.
        # real = self.sample_data(n)
        # # real = torch.from_numpy(real.astype('float32')).to(self.device)
        # inp= 'Z'
        # inputs.append(real[:, cur_model['in_indx'][inp][0]: cur_model['in_indx'][inp][1]])
        # #independently sample X
        # real = self.sample_data(n)
        # # real = torch.from_numpy(real.astype('float32')).to(self.device)
        # inp = 'X'
        # inputs.append(real[:, cur_model['in_indx'][inp][0]: cur_model['in_indx'][inp][1]])
        # inputs = np.concatenate(inputs, axis=1)
        # output= self.conditional_sample(cur_model, conf, inputs, n)  #first model for generation of X. 2nd model for Y.


        return torch.from_numpy(output.astype('float32')).to(self.device)




    def conditional_sample(self, model, conf, inputs,  n ):
        """
        :param n:
        :param transformer:
        :param conf:
        :param inputs:
        :return: generate synthetic data
        """
        model['model'].eval()

        if len(inputs)>0:
            inputs = [torch.from_numpy(inputs.astype('float32')).to(self.device)]

        mean = torch.zeros(n, conf['latent_size'])
        std = mean + 1
        fakez = torch.normal(mean=mean, std=std)
        if torch.cuda.is_available():
            fakez = fakez.to(self.device)

        fake = model['model']([fakez], inputs)
        fakeact = self.apply_activate(fake, model['out_indx'])
        fakeact = inputs + [fakeact]
        fakeact = torch.cat(fakeact, dim=1)

        data=fakeact.detach().cpu().numpy()


        # allowed_vars = {**model['in_indx'], **model['out_indx']}
        # return self.transformer.inverse_transform(data, allowed_vars)

        return data
