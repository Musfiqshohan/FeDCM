
conf = {
	#export PYTHONPATH="${PYTHONPATH}:/local/scratch/a/rahman89/PycharmProjects/GenUnseen/MyProjects/CausalFed/FedDCM"
	##################################### The most commonly used parameters
    #data name
    # "data_name":"adult",
    # "data_name": "frontdoor",
    "data_name": "nonid",
    # "data_name":"covtype",


	#global epochs
	"global_epochs" :200,

	#local epoch
	"local_epochs" : 100,

	"parallel_transform":False,

	"batch_size": 128,

	"per_client_samples":1000,

	# number of clients
	"num_parties": 5,


	"isFL": True,
	"exp_name": "fedcm",

	"gm_conf": [
	{"expr": "P(X|Z)",
		 "out_indx": {'X': []},
		 "in_indx": {'Z': []}
		 },

    # {"expr": "P(Y|Z,X)",
    # "out_indx": {'Y': []} ,
    # "in_indx": {'Z': [], 'X': []}
    #  }

	],


	# trained generator and discriminator
	"saved_models": {
		# "frontdoor": ["./checkpoints/frontdoor/gen_epochxxx.pth", "./checkpoints/frontdoor/disc_epochxxx.pth"]
		"nonid": ["./checkpoints/nonid/gen_epochxxx.pth", "./checkpoints/nonid/disc_epochxxx.pth"]
	},


	#parameter of Dirichlet distribution
	#beta is used to control the level of noniid
	#the lower the beta, the unbalance the global data
	"beta" : 100,

	#number of labels
	"num_classes": {
		"clinical":2,
		"adult":2,
        "frontdoor":2,
        "nonid":2,
		"intrusion":10,
		"covtype":7,
		"tb":2,
		"credit":2
	},



    #label column
	"label_column":{
        "clinical": "label",
        "credit": "Class",
        "covtype": "Cover_Type",
        "intrusion": "label",
        "tb": "Condition",
        "IRIS": "species",
        "body": "class",
		"adult":"income_bracket",
        "frontdoor":"Y",
        "nonid":"Y"
    },

    #test data file
	"test_dataset": {
		"clinical":"./data/clinical/clinical_test.csv",
		"intrusion":"./data/intrusion/intrusion_test.csv",
		"adult":"./data/adult/adult_test.csv",
        "frontdoor":"/local/scratch/a/rahman89/PycharmProjects/GenUnseen/MyProjects/CausalFed/FedDCM/data/frontdoor/test_data.csv",
        "nonid":"/local/scratch/a/rahman89/PycharmProjects/GenUnseen/MyProjects/CausalFed/FedDCM/data/nonid/test_data.csv",
		"covtype":"./data/covtype/covtype_test.csv"
	},

    #training data file
	"train_dataset" : {
		"clinical":"./data/clinical/clinical_train.csv",
		"intrusion":"./data/intrusion/intrusion_train.csv",
		"adult":"./data/adult/adult_train.csv",
		"frontdoor":"/local/scratch/a/rahman89/PycharmProjects/GenUnseen/MyProjects/CausalFed/FedDCM/data/frontdoor",
		"nonid":"/local/scratch/a/rahman89/PycharmProjects/GenUnseen/MyProjects/CausalFed/FedDCM/data/nonid",
		"covtype":"./data/covtype/covtype_train.csv"
	},



	# model generated synthetic data
	"syn_data":{
		"clinical":"./data/clinical/clinical_syn.csv",
		"intrusion":"./data/intrusion/intrusion_syn.csv",
		"adult":"./data/adult/adult_syn.csv",
		"frontdoor":"./data/frontdoor/frontdoor_syn.csv",
		"nonid":"./data/nonid/nonid_syn.csv",
		"covtype":"./data/covtype/covtype_syn.csv",
	},

    #discrete columns
    "discrete_columns": {
    "adult":[
    'workclass',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'native_country',
	'income_bracket'
],
    "frontdoor":['Z', 'X', 'Y'],
    "nonid":['Z', 'W', 'X', 'Y'],
    # "frontdoor":['Z', 'X'],
    # "frontdoor":['X'],

    "intrusion":['protocol_type', 'service', 'flag','label'],
    "credit":["Class"],
    "covtype":['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39','Soil_Type40','Cover_Type'],
    # "covtype": ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Soil_Type1', 'Soil_Type2',
		# 			'Soil_Type3', 'Soil_Type4', 'Soil_Type6', 'Soil_Type8', 'Soil_Type9',
		# 			'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
		# 			'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',
		# 			'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
		# 			'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33',
		# 			'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
		# 			'Soil_Type40', 'Cover_Type'],

		"clinical":["anaemia","diabetes","high_blood_pressure","sex","smoking","label"],
    "tb":["Condition"]
},
	##########################################################

	#max clusters for GMM
	"max_clusters":10,



    #is fedavg
	"is_init_avg": False,

	"gen_weight_decay":1e-5,

    #generator learning rate
	"gen_lr" :2e-4,

	#generator
	"generator_dim":(256,256),

	"local_discriminator_steps":3,

	"dis_weight_decay":1e-5,

    #discriminator learning rate
	"dis_lr" :2e-4,

	"discriminator_dim":(256,256),

	#generator input dimension
	"latent_size":128,
}

##config

