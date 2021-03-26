import modeling.glm as glm
import numpy as np
import math

def incr_build(model, target_var, start_list, add_list, get_data_fn, sample_size, client, family='normal'):
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
    
    rmse_valid = []
    corr_valid = []
    segs = []
    for j, valid in enumerate(add_list):
        segs += [valid]
        model_df = get_data_fn(build_list, sample_size, client)
        valid_df = get_data_fn([valid], sample_size, client)
        print('Data sizes for out-of-sample value {0}: build {1}, validate {2}'.format(valid, model_df.shape[0],
                                                                                       valid_df.shape[0]))
        print('Build list: {0}'.format(build_list))
        
        glm_model = glm(model, model_df, family=family)
        build_list += [valid]
        
        yh = glm_model.predict(valid_df)
        res = valid_df[target_var] - np.array(yh).flatten()
        rmse_valid += [math.sqrt(np.square(res).mean())]
        valid_df['yh'] = yh
        cor = valid_df[[target_var, 'yh']].corr()
        cor = float(cor.iloc[0]['yh'])
        corr_valid += [cor]
    
    return segs, rmse_valid, corr_valid
