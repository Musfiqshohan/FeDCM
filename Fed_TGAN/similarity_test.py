import itertools

import numpy as np
import pandas as pd

from conf import conf
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import  MinMaxScaler
from ctgan import CTGAN


def get_joint_distributions_from_samples(label_dim, corrensponding_samples):
    dim_list = list(label_dim.values())
    observe_perms = np.array(list(itertools.product(*[range(dim) for dim in dim_list])))

    combinations, count = np.unique(corrensponding_samples, axis=0, return_counts=True)

    upd_dist = {}
    for comb in observe_perms:
        upd_dist[tuple(list(comb))] = 1e-6

    total = corrensponding_samples.shape[0]
    for comb, cnt in zip(combinations, count):
        upd_dist[tuple(list(comb))] = cnt / total

    return upd_dist



def get_tvd_kl(real_data, syn_data):

    support = {col: real_data[col].nunique() for col in real_data.columns}

    real = get_joint_distributions_from_samples(support, real_data.to_numpy().astype(int))
    fake = get_joint_distributions_from_samples(support, syn_data.to_numpy().astype(int))
    #
    # print(f' fake: {fake}')
    # print(f' Real: {real}')

    epsilon = 1e-10  # Small constant to prevent division by zero
    for key in real:
        if key not in fake:
            fake[key] = 0

    obs_tvd = 0.5 * sum([abs(fake[key] - real[key]) for key in real])
    obs_kl = sum([(real[key]) * np.log(real[key] / (fake[key] + epsilon)) for key in real])

    return obs_tvd, obs_kl

def table_similarity(syn_data, real_data, dis_columns):
    """

    :param syn_data:
    :param real_data:
    :param dis_columns:
    :return: compute the average Jensen-Shannon Divergence (JSD) and the average Wasserstein Distance (WD)
    """
    real_data= real_data[syn_data.columns]
    columns = real_data.columns

    jsd = []
    wd =[]
    for c in columns:

        if c in dis_columns:
            jsd.append(cal_jsd(syn_data[c], real_data[c]))
        else:
            wd.append(cal_wd(syn_data[c], real_data[c]))

    avg_jsd = sum(jsd) / len(jsd)
    if len(wd)>0:
        avg_wd = sum(wd) / len(wd)
    else:
        avg_wd=-1



    # # Getting similarity for only discrete

    cur_discrete_cols = [col for col in dis_columns if col in real_data.columns]
    real_data = real_data[cur_discrete_cols]
    syn_data = syn_data[cur_discrete_cols]

    if len(cur_discrete_cols)<=5:
        tvd, kl= get_tvd_kl(real_data, syn_data)
    else:
        t_list=[]
        k_list=[]
        for col in cur_discrete_cols:
            t, k = get_tvd_kl(real_data[[col]], syn_data[[col]])
            t_list.append(t)
            k_list.append(k)

        tvd= np.max(t_list)
        kl= np.max(k_list)

    #

    return avg_jsd, avg_wd, tvd, kl

def get_fre(data):
    cf = data.value_counts()
    cate = []
    fre = []
    for c, f in zip(cf.index,cf.values):
        cate.append(c)
        fre.append(f)
    cate_fre = pd.DataFrame({'cate':cate, 'fre':fre})
    return cate_fre


def cal_jsd(syn,real):
    """
    :param syn:
    :param real:
    :return: compute the js distance for discrete columns between synthetic data and real data
    """
    syn_cf = get_fre(syn)
    real_cf = get_fre(real)

    if len(syn_cf) > len(real_cf):
        cate = syn_cf['cate'].tolist()
    else:
        cate = real_cf['cate'].tolist()

    syn_f = []
    real_f =[]
    for c in cate:
        s = syn_cf[syn_cf['cate']==c]['fre'].values
        if len(s) >0:
            syn_f.append(s[0])
        else:
            syn_f.append(0)

        f = real_cf[real_cf['cate']==c]['fre'].values
        if len(f)>0:
            real_f.append(f[0])
        else:
            real_f.append(0)

    return distance.jensenshannon(syn_f,real_f,base=2)


def cal_wd(syn,real):
    """
    :param syn:
    :param real:
    :return: the Wasserstein Distance for each continuous column
    """
    min_max_enc = MinMaxScaler(feature_range=(0, 1))
    syn = min_max_enc.fit_transform(syn.values.reshape(-1,1))
    real = min_max_enc.fit_transform(real.values.reshape(-1,1))

    return wasserstein_distance(syn.ravel(),real.ravel())

def ctgan_syn(real_data, dis_columns, num):


    ctgan = CTGAN(epochs=10,verbose=True)
    ctgan.fit(real_data, dis_columns)

    return ctgan.sample(num)


if __name__ == "__main__":

    dis_columns = conf["discrete_columns"][conf["data_name"]]
    test_data = pd.read_csv(conf['test_dataset'][conf['data_name']])

    real_data = pd.read_csv(conf['train_dataset'][conf['data_name']])

    avg_jsd, avg_wd = table_similarity(real_data,test_data,dis_columns)
    print("training data")
    print("avg_jsd:{}".format(avg_jsd))
    print("avg_wd:{}".format(avg_wd))

    syn_data = ctgan_syn(real_data, dis_columns, 1000)
    avg_jsd, avg_wd = table_similarity(syn_data, test_data, dis_columns)
    print("synthetic data")
    print("avg_jsd:{}".format(avg_jsd))
    print("avg_wd:{}".format(avg_wd))