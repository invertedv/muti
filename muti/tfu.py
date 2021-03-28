"""
Utilities that help with the building of tensorflow keras models
"""
import tensorflow as tf
import os
import math
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots


def plot_history(history, groups=['loss'], metric='loss', first_epoch=0, title=None, plot_file=None):
    """
    plot the history of metrics from a keras model tf build
    :param history: history returned from keras fit
    :param groups: groups to plot
    :param metric: metric to plot
    :param first_epoch: first element to plot
    :param title: title for plot
    :param plot_file: file name to plot to
    :return:
    """
    pio.renderers.default = 'browser'
    fig = []
    for g in groups:
        x = np.arange(first_epoch, len(history.history[g]) - first_epoch)
        y = history.history[g][first_epoch:len(history.history[metric])]
        fig += [go.Scatter(x=x, y=y, name=g)]
    if title is None:
        title = 'TensorFlow Model Build<br>' + metric
    layout = go.Layout(title=title,
                       xaxis=dict(title='Epoch'),
                       yaxis=dict(title=metric))
    figx = go.Figure(fig, layout=layout)
    figx.show()
    if plot_file is not None:
        fig.write_image(plot_file)


def build_column(feature_name, feature_params, out_path=None):
    """
    Returns a tensorflow feature columns and, optionally, the vocabulary for categorical and
    embedded features. Optionally creates files of the vocabularies for use in TensorBoard.
  
    :param feature_name: name of the feature
    :type str
    :param feature_params:
        Element 0: type of feature ('cts', 'cat', 'emb').
        Element 1: ('cat', 'emb') vocabulary list (list of levels)
        Element 2: ('cat', 'emb') default index. If None, 0 is used
        Element 3: ('emb') embedding dimension
    :type list
    :param out_path: path to write files containing levels of 'cat' and 'emb' variables
    :return: tf feature column and (for 'cat' and 'emb') a list of levels (vocabulary)
    """
    
    if feature_params[0] == 'cts':
        print('col {0} is numeric'.format(feature_name))
        return tf.feature_column.numeric_column(feature_name)
    # categorical and embedded features
    if feature_params[0] in ['cat', 'emb']:
        vocab = feature_params[1]
        
      # save vocabulary for TensorBoard
        if out_path is not None:
            if out_path[-1] != '/': out_path += '/'
            if not os.path.isdir(out_path):
                os.makedirs(out_path)
            f = open(out_path + feature_name + '.txt', 'w')
            f.write('label\tId\n')
            for j, s in enumerate(vocab):
                f.write(s + '\t' + str(j) + '\n')
            f.close()
        dv = [j for j in range(len(vocab)) if vocab[j] == feature_params[2]][0]
        col_cat = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab,
                                                                          default_value=dv)
      
      # go with 1-hot encoding
        if feature_params[0] == 'cat':
            col_ind = tf.feature_column.indicator_column(col_cat)
            print('col {0} is categorical with {1} levels'.format(feature_name, len(vocab)))
            return col_ind
      
        # for embedded features, the third element of feature_params input is the dimension of the
        # embedding
        levels = feature_params[3]
        col_emb = tf.feature_column.embedding_column(col_cat, levels)
        print('col {0} is embedded with {1} levels'.format(feature_name, levels))
        return col_emb


def build_model_cols(feature_dict, out_vocab_dir=None):
    """
    Builds inputs needed to specify a tf.keras.Model. The tf_cols_* are TensorFlow feature_columns. The
    inputs_* are dictionaries of tf.keras.Inputs.  The tf_cols_* are used to specify keras.DenseFeatures methods and
    the inputs_* are the inputs to those layers.
    
    :param feature_dict: dictionary of features to build columns for. The key is the feature name. The entry is a list:
                feature type (str)  'cts', 'cat', 'emb'
                list of unique levels for 'cat' and 'emb'
                embedding dimension for 'emb'
    :param out_vocab_dir: directory to write out unique levels
    :return: 4 lists:
               - tf_cols_cts: tf.feature_column defining each continuous feature
               - inputs_cts: list of tf.keras.Inputs for each continuous column
               - tf_cols_cat: tf.feature_column defining each categorical ('cat','emb') feature
               - inputs_cat: list of tf.keras.Inputs for each categorical ('cat', 'emb') column
               
               The tf_cols_* are used in tf.keras.layers.DenseFeatures
               the inputs_* are used to define the inputs to those tensors
    """
    tf_cols_cts = []
    tf_cols_cat = []
    inputs_cts = {}
    inputs_cat = {}
    for feature in feature_dict.keys():
        if feature_dict[feature][0] == 'cts':
            feat = build_column(feature, feature_dict[feature])
            tf_cols_cts += [feat]
            inputs_cts[feature] = tf.keras.Input(shape=(1,), name=feature)
        else:
            feat = build_column(feature, feature_dict[feature], out_vocab_dir)
            tf_cols_cat += [feat]
            inputs_cat[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.string)
    return tf_cols_cts, inputs_cts, tf_cols_cat, inputs_cat


def get_tf_dataset(feature_dict, target, df, batch_size, repeats=None):
    """
    build a tf dataset from a pandas DataFrame
    
    :param feature_dict: dictionary whose keys are the features
    :param target: target var
    :param df: pandas DataFrame to work on
    :param batch_size: Batch size
    :param repeats: how many repeats of the dataset (None = infinite)
    :return: tf dataset
    """
    tf_ds = tf.data.Dataset.from_tensor_slices((dict(df[feature_dict.keys()]), df[target]))
    if repeats is None:
        tf_ds = tf_ds.shuffle(df.shape[0]).batch(batch_size).repeat()
    else:
        tf_ds = tf_ds.batch(batch_size).repeat(repeats)
    return tf_ds


def incr_build(model, start_list, add_list, get_data_fn, sample_size, feature_dict, target_var,
               client, batch_size, epochs_list, steps_per_epoch, model_dir=None, plot=False, verbose=0):
    """
    This function builds a sequence of models. The get_data_fn takes a list of values as contained in
    start_list and add_list and returns data subset to those values. The initial model is built on the
    values of start_list and then evaluated on the data subset to the first value of add_list.
    
    At the next step, the data in the first element of add_list is added to the start_list data, the model
    is updated and the evaluation is conducted on the second element of add_list
    
    :param model: tf keras model
    :param start_list: list of (general) time periods for model build for the first model build
    :param add_list: list of out-of-time periods to evaluate
    :param get_data_fn: function to get a pandas DataFrame of data to work on
    :param sample_size: size of pandas DataFrames to get
    :param feature_dict: dictionary of features in the model
    :param target_var: target variable of model build
    :param client: clickhouse_driver.Client
    :param batch_size: size of batches for model build
    :param epochs_list: list (length 2) of epochs for model fit; entry 0 is initial model, entry 1 is subsequent
                        models
    :param steps_per_epoch: steps_per_epoch for keras model fit
    :param model_dir: directory to save models
    :param plot: if True, plot history
    :param verbose: print verobisity for keras.fit (0 = quiet, 1 = normal level, 2=talkative)
    :return: lists of out-of-sample values:
             add_list
             rmse  root mean squared error
             corr  correlation
    """

    if model_dir is not None:
        if model_dir[-1] != '/': model_dir += '/'
        if os.path.isdir(model_dir):
          os.system('rm -r ' + model_dir)
        os.makedirs(model_dir)

    build_list = start_list
    epochs = epochs_list[0]
    mse_valid = []
    corr_valid = []
    segs = []
    
    for j, valid in enumerate(add_list):
        segs += [valid]
        model_df = get_data_fn(build_list, sample_size, client)
        print(model_df['vintage'].value_counts())
        model_ds = get_tf_dataset(feature_dict, target_var, model_df, batch_size=batch_size)
        
        valid_df = get_data_fn([valid], sample_size, client)
        print(valid_df['vintage'].value_counts())
        valid_ds = get_tf_dataset(feature_dict, target_var, valid_df, batch_size=batch_size, repeats=1)
        
        print('Data sizes for out-of-sample value {0}: build {1}, validate {2}'.format(valid, model_df.shape[0],
                                                                                       valid_df.shape[0]))
        print('Build list: {0}'.format(build_list))

        history = model.fit(model_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
                            validation_data=valid_ds, verbose=verbose)

        build_list += [valid]  # NOTE Accumulates
        if model_dir is not None:
            out_m = model_dir + "before_" + valid + '.h5'
            model.save(out_m, overwrite=True, save_format='h5')
        
        if plot:
            title = 'model loss\n' + 'Training up to ' + valid
            plot_history(history, ['loss', 'val_loss'], 'loss', title=title)

        yh = model.predict(valid_ds)
        res = valid_df[target_var] - yh.flatten()
        mse_valid += [math.sqrt(np.square(res).mean())]
        valid_df['yh'] = yh
        cor = valid_df[[target_var, 'yh']].corr()
        cor = float(cor.iloc[0]['yh'])
        corr_valid += [cor]
        epochs = epochs_list[1]

    return segs, mse_valid, corr_valid


def marginal(model, features_target, features_dict, sample_df_in, plot_dir=None, num_sample=100, cat_top=10):
    """
    Generate plots to illustrate the marginal effects of the model 'model'. Live plots are output to the default
    browser and, optionally, png's are written to plot_dir

    The process is:

    - The model output is found on sample_df:
        - Six groups based on the quantiles [0, .1,. .25, .5, .75, .9, 1] are found from sample_df.
    - for each feature in features_target:
        - Six graphs are contructed: one for each group defined above.
            - The graph covers
            -  A random sample of size num_sample is taken from this group
            -  The target feature value is replace by an array that has values of its
               [0.01, .1, .2, .3, .4, .5, .6, .7, .8, 0.9, 0.99] quantiles in this group, if it is continuous or
               is the top cat_top [None means all] most frequent levels if categorical
            - The model output is found for all these
            - Boxplots are formed.  These plots have a common y-axis with limits from the .01 to .99 quantile
              of the model output on sample_df

    Features:
        - Since the x-values are sampled from sample_df, any correlation within the features not plotted on the
          x-axis are preserved.
        - The values of the target feature are observed within the group (of the six) being plotted, so extrapolation
          into unobserved space is reduced. The

    Returns a metric that rates the importance of the feature to the model (e.g. sloping). It is calculated as:
    
    - Within each model output segment,
        - calculate the median model output for each x value.
        - Then calculate the range of these medians.
    - We then have a range for each model output segment. Now find the maximum across the segments. This is the
      impportance value.

    :param model: A keras tf model with a 'predict' method that takes a tf dataset as input
    :type tf.keras.Mode
    :param features_target: features to generate plots for.
    :type list of str
    :param features_dict: dictionary whose keys are the features in the model
    :type dict
    :param sample_df: DataFrame from which to take samples and calculate distributions
    :type pandas DataFrame
    :param plot_dir: directory to write plots out to
    :type str
    :param num_sample: number of samples to base box plots on
    :type int
    :param cat_top: maximum number of levels of categorical variables to plot
    :type int
    :return: for each target, the range of the median across the target levels for each model output group
    :rtype dict
    """
    if plot_dir is not None:
        if plot_dir[-1] != '/': plot_dir += '/'
        if os.path.isdir(plot_dir):
            os.system('rm -r ' + plot_dir)
        os.makedirs(plot_dir)
    pio.renderers.default = 'browser'
    
    sample_df = sample_df_in.copy()
    
    sample_df['target'] = np.full(sample_df.shape[0], 0.0)
    score_ds = get_tf_dataset(features_dict, 'target', sample_df, sample_df.shape[0], 1)
    
    sample_df['yh'] = np.array(model.predict(score_ds)).flatten()
    rangey = sample_df['yh'].quantile([.01, .99])
    miny = float(rangey.iloc[0])
    maxy = float(rangey.iloc[1])
    target_qs = [0, .1, .25, .5, .75, .9, 1]
    quantiles = sample_df['yh'].quantile(target_qs)
    quantiles.iloc[0] -= 1.0
    num_grp = quantiles.shape[0] - 1
    sample_df['grp'] = pd.cut(sample_df['yh'], quantiles, labels=['grp' + str(j) for j in range(num_grp)], right=True)
    sub_titles = []
    importance = {}
    
    for j in range(num_grp):
        sub_title = 'Model Output in {0} to {1}'.format(round(quantiles.iloc[j], 2), round(quantiles.iloc[j + 1], 2))
        sub_title += '<br>'
        sub_title += 'Quantile {0} to {1}'.format(target_qs[j], target_qs[j + 1])
        sub_titles += [sub_title]
    for target in features_target:
        # the specs list gives some padding between the top of the plots and the overall title
        fig = make_subplots(rows=1, cols=num_grp, subplot_titles=sub_titles,
                            specs=[[{'t': 0.05}, {'t': 0.05}, {'t': 0.05}, {'t': 0.05}, {'t': 0.05}, {'t': 0.05}]])
        
        median_ranges = []
        # go across the model output groups
        for j in range(num_grp):
            yhall = None
            i = sample_df['grp'] == 'grp' + str(j)
            
            if features_dict[target][0] == 'cts':
                qs = sample_df.loc[i, target].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]).unique()
                nobs = qs.shape[0]
                xval = np.array(qs).flatten()
                xlab = 'Values at deciles within each model-value group'
            else:
                cats = list(sample_df.loc[i][target].value_counts().sort_values(ascending=False).index)
                if cat_top is None or len(cats) < cat_top:
                    nobs = len(cats)
                    xval = cats
                else:
                    nobs = cat_top
                    xval = cats[0:nobs]
                xlab = 'Top values by frequency within each model-value group'
            # score_df is just the values of the feature we going to score
            score_df = pd.DataFrame({target: xval})
            
            # pick a random sample within the model output group
            vals = sample_df.loc[i].sample(num_sample)
            
            # go across the number of samples to draw
            for k in range(num_sample):
                # load up the rest of the features moving through our random sample
                for feature in features_dict.keys():
                    if feature != target:
                        xval = np.full(nobs, vals.iloc[k][feature])
                        score_df[feature] = xval
                
                # placeholder
                score_df['target'] = np.full(nobs, 0.0)
                score_ds = get_tf_dataset(features_dict, 'target', score_df, nobs, 1)
                
                yh = np.array(model.predict(score_ds)).flatten()
                
                # stack up the model outputs
                if yhall is None:
                    yhall = yh
                    xall = np.array(score_df[target]).flatten()
                else:
                    yhall = np.append(yhall, yh)
                    xall = np.append(xall, np.array(score_df[target]).flatten())
            
            # create grouped boxplots based on the values of the target feature
            if features_dict[target][0] == 'cts':
                xv = pd.DataFrame({'x': [str(round(x, 2)) for x in xall], 'yh': yhall})
                fig.add_trace(go.Box(x=xv['x'], y=xv['yh']), row=1, col=j + 1)
            else:
                xv = pd.DataFrame({'x': [str(x) for x in xall], 'yh': yhall})
                fig.add_trace(go.Box(x=xv['x'], y=xv['yh']), row=1, col=j + 1)
            # give the figure a title
            fig.update_annotations(sub_title='Group ' + str(j), row=1, col=j + 1)
            fig.update_traces(name='grp ' + str(j), row=1, col=j + 1)
            score_df['yh'] = yh
            medians = xv.groupby('x')['yh'].median()
            median_ranges += [medians.max() - medians.min()]
        
        importance[target] = max(median_ranges)
        # overall title
        fig.update_layout(
            title=dict(text='Marginal Response for ' + target, font=dict(size=24), xanchor='center', xref='paper',
                       x=0.5), showlegend=False)
        # add label at bottom of graphs
        fig.add_annotation(text=target, font=dict(size=16), x=0.5, xanchor='center', xref='paper', y=0,
                           yanchor='top', yref='paper', yshift=-40, showarrow=False)
        fig.add_annotation(text=xlab, font=dict(size=10), x=0.5, xanchor='center', xref='paper', y=0,
                           yanchor='top', yref='paper', yshift=-60, showarrow=False)
        for jj in range(num_grp):
            fig['layout']['yaxis' + str(jj + 1)]['range'] = [miny, maxy]
        for jj in range(1, num_grp):
            fig['layout']['yaxis' + str(jj + 1)]['showticklabels'] = False
        fig.show()
        if plot_dir is not None:
            # needed for png to look decent
            fig.update_layout(width=1800, height=600)
            fname = plot_dir + target + '.png'
            fig.write_image(fname)
    
    return pd.DataFrame(importance, index=['max median range'])
