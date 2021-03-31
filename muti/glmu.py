from modeling.glm import glm
import muti.general as gen
import numpy as np
import math


def build_model_formula(features_dict, target):
    """
    Builds the model formula for glm from modeling based on the features_dict specification.
    Does not included embedded features
    
    :param features_dict: features dictionary
    :type features_dict: dict
    :param target: dependent variable
    :type target: str
    :return: model formula
    :rtype str
    """
    ms = target + '~'
    extra = ''
    for feature in features_dict:
        if features_dict[feature][0] == 'cts':
            ms += extra + feature
        elif features_dict[feature][0] == 'spl':
            ms += extra + 'h(' + feature + ',' + features_dict[feature][1] + ',0)'
        elif features_dict[feature][0] == 'cat':
            ms += extra + 'c(' + feature + ',' + features_dict[feature][2] + ')'
        extra = ' + '
    return ms


def incr_build(model, target_var, start_list, add_list, get_data_fn, sample_size, client, global_valid_df_in,
               family='normal'):

    """
    This function builds a sequence of GLM models. The get_data_fn takes a list of values as contained in
    start_list and add_list and returns data subset to those values. The initial model is built on the
    values of start_list and then evaluated on the data subset to the first value of add_list.

    At the next step, the data in the first element of add_list is added to the start_list data, the model
    is updated and the evaluation is conducted on the second element of add_list.

    This function is the GLM counterpart to incr_build

    :param model: model specification for glm
    :param target_var: response variable we're modeling
    :param start_list: list of (general) time periods for model build for the first model build
    :param add_list: list of out-of-time periods to evaluate
    :param get_data_fn: function to get a pandas DataFrame of data to work on
    :param sample_size: size of pandas DataFrames to get
    :param client: clickhouse_driver.Client
    :param family: family of the model ('normal' or 'binomial')
    :return: lists of out-of-sample values:
             add_list
             rmse  root mean squared error
             corr  correlation
    """
    
    build_list = start_list
    global_valid_df = global_valid_df_in.copy()
    global_valid_df['model_glm_inc'] = np.full((global_valid_df.shape[0]), 0.0)
    rmse_valid = []
    corr_valid = []
    segs = []
    for j, valid in enumerate(add_list):
        segs += [valid]
        model_df = get_data_fn(build_list, sample_size, client)
        valid_df = get_data_fn([valid], sample_size, client)
        print('Data sizes for out-of-sample value {0}: build {1}, validate {2}'.format(valid, model_df.shape[0],
                                                                                       valid_df.shape[0]))
        # print('Build list: {0}'.format(build_list))
        
        glm_model = glm(model, model_df, family=family)
        build_list += [valid]

        gyh = glm_model.predict(global_valid_df)
        i = global_valid_df['vintage'] == valid
        global_valid_df.loc[i, 'model_glm_inc'] = gyh[i]

        yh = glm_model.predict(valid_df)
        res = valid_df[target_var] - np.array(yh).flatten()
        rmse_valid += [math.sqrt(np.square(res).mean())]
        valid_df['yh'] = yh
        cor = gen.r_square(valid_df['yh'], valid_df[target_var])
        corr_valid += [cor]
    
    return segs, rmse_valid, corr_valid, global_valid_df
