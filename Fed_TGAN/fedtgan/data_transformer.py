"""
DataTransformer module.

This file comes from CTGAN: https://github.com/sdv-dev/CTGAN/blob/master/ctgan/data_transformer.py
"""

from collections import namedtuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdt.transformers import ClusterBasedNormalizer
from sklearn.preprocessing import OneHotEncoder


SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'
    ]
)


class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self, parallel_transform, dcm_dim=None,  max_clusters=10, weight_threshold=0.005):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
            parallel_transform:
                If transform the data parallely.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

        #new
        self.dcm_dim=dcm_dim
        self.parallel_transform=parallel_transform
        self.lb_indx= {}
        self.lb_dim= {}


    def _fit_continuous(self, data):
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]

        if self.dcm_dim== {}:
            gm = ClusterBasedNormalizer(model_missing_values=True, max_clusters=min(len(data), 10))
            gm.fit(data, column_name)
            num_components = sum(gm.valid_component_indicator)
        else:
            num_components= self.dcm_dim[column_name]-1     # 1 for tanh
            gm = ClusterBasedNormalizer(model_missing_values=True, max_clusters=num_components)
            gm.fit(data, column_name)



        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components)

    def _fit_discrete(self, data):
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]

        if self.dcm_dim== {}:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        else:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=[[i for i in range(self.dcm_dim[column_name])]])
        input= np.array(data[[column_name]])
        ohe.fit(input, column_name)
        num_categories = len(ohe.categories[0])


        return ColumnTransformInfo(
            column_name=column_name, column_type='discrete', transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories)

    def fit(self, raw_data,  discrete_columns=()):
        """Fit the ``DataTransformer``.

        Fits a ``ClusterBasedNormalizer`` for continuous columns and a
        ``OneHotEncoder`` for discrete columns.

        This step also counts the #columns in matrix data and span information.
        """
        self.output_info_dict = {}
        self.output_dimensions = 0
        self.dataframe = True


        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            # work around for RDT issue #328 Fitting with numerical column names fails
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_dict = {}


        cur_ind=0
        for column_name in raw_data.columns:
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])


            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_dict[column_name] = column_transform_info

            #new
            self.lb_indx[column_name]=[cur_ind, cur_ind+column_transform_info.output_dimensions]
            self.lb_dim[column_name] = column_transform_info.output_dimensions

            if column_name in self.lb_indx: # only for outputs. Specifically useful for conditional distributions ex: P(Y|X) when we apply this only on Y not on X.
                self.output_info_dict[column_name]= column_transform_info.output_info
            # if column_name in self.in_indx.keys():
            #     self.in_indx[column_name]=[cur_ind, cur_ind+column_transform_info.output_dimensions]
            # elif column_name in self.out_indx.keys():
            #     self.out_indx[column_name]=[cur_ind, cur_ind+column_transform_info.output_dimensions]

            cur_ind+= column_transform_info.output_dimensions


    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        data[column_name] = data[column_name].to_numpy().flatten()
        gm = column_transform_info.transform
        transformed = gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data)

    def _synchronous_transform(self, raw_data, column_transform_info_dict):
        """Take a Pandas DataFrame and transform columns synchronous.

        Outputs a list with Numpy arrays.
        """
        column_data_list = []
        for key, column_transform_info in column_transform_info_dict.items():
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data))

        return column_data_list

    def _parallel_transform(self, raw_data, column_transform_info_dict):
        """Take a Pandas DataFrame and transform columns in parallel.

        Outputs a list with Numpy arrays.
        """
        processes = []
        for key, column_transform_info in column_transform_info_dict.items():
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            process = None
            if column_transform_info.column_type == 'continuous':
                process = delayed(self._transform_continuous)(column_transform_info, data)
            else:
                process = delayed(self._transform_discrete)(column_transform_info, data)
            processes.append(process)

        return Parallel(n_jobs=-1)(processes)

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower (NB: pycharm debugger cant be used then).

        if self.parallel_transform and raw_data.shape[0] >= 500:
            column_data_list = self._parallel_transform(
                raw_data,
                self._column_transform_info_dict
            )
        else:
            # if raw_data.shape[0] < 500:
            column_data_list = self._synchronous_transform(
                raw_data,
                self._column_transform_info_dict
            )



        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes()))
        data.iloc[:, 1] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        columns = list(ohe.get_feature_names_out())
        data = pd.DataFrame(column_data, columns=columns )
        return ohe.inverse_transform(data)

    def inverse_transform(self, data, allowed_vars,  sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []

        # allowed_vars={**in_indx, **out_indx}


        for key, column_transform_info in self._column_transform_info_dict.items():

            #might be issue if not consecutive variables
            if column_transform_info.column_name not in allowed_vars:
                continue

            dim = column_transform_info.output_dimensions

            column_data = data[:, st:st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st)
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes.loc[list(allowed_vars)]))
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for key, column_transform_info in self._column_transform_info_dict.items():
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            'discrete_column_id': discrete_counter,
            'column_id': column_id,
            'value_id': np.argmax(one_hot)
        }
