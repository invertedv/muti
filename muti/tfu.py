"""
Utilities that help with the building of tensorflow keras models

"""

import muti.genu as gen
import tensorflow as tf
import os
import math
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots


def get_pred(yh, column=None):
    """
    Returns an array of predicted values from a keras predict method. If column is None, then this
    assumes the output has one column and it returns a flattened array.
    If column is an int, it returns that column from the prediction matrix.
    If column is a list of int, it returns the column sums
    
    :param yh: keras model prediction
    :type yh: keras.model.predict() output
    :param column: which column(s) to return
    :type column: int or list of int
    :return: prediction array
    :rtype ndarray
    """
    if column is None:
        return np.array(yh).flatten()
    if not isinstance(column, list):
        return yh[:, column]
    # sum up columns
    return np.sum(yh[:, column], axis=1)


def plot_history(history, groups=['loss'], metric='loss', first_epoch=0, title=None, plot_dir=None, in_browser=False):
    """
    plot the history of metrics from a keras model tf build
    :param history: history returned from keras fit
    :type history: tf.keras.History class
    :param groups: groups to plot
    :type groups: list
    :param metric: metric to plot
    :type metric: str
    :param first_epoch: first element to plot
    :type first_epoch: int
    :param title: title for plot
    :type title: str
    :param plot_dir: directory to plot to
    :type plot_dir: str
    :param in_browser: if True display in browser
    :type in_browser: bool
    :return:
    """
    pio.renderers.default = 'browser'
    os.makedirs(plot_dir, exist_ok=True)
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
    if in_browser:
        figx.show()
    if plot_dir is not None:
        plot_file = plot_dir + metric + '.png'
        figx.write_image(plot_file)

        plot_file = plot_dir + metric + '.html'
        figx.write_html(plot_file)


def build_column(feature_name, feature_params, out_path=None):
    """
    Returns a tensorflow feature columns and, optionally, the vocabulary for categorical and
    embedded features. Optionally creates files of the vocabularies for use in TensorBoard.
  
    :param feature_name: name of the feature
    :type feature_name: str
    :param feature_params:
        Element 0: type of feature ('cts'/'spl', 'cat', 'emb').
        Element 1: ('cat', 'emb') vocabulary list (list of levels)
        Element 2: ('cat', 'emb') default index. If None, 0 is used
        Element 3: ('emb') embedding dimension
    :type feature_params: list
    :param out_path: path to write files containing levels of 'cat' and 'emb' variables
    :type out_path: str
    :return: tf feature column and (for 'cat' and 'emb') a list of levels (vocabulary)
    """
    
    if feature_params[0] == 'cts' or feature_params[0] == 'spl':
        print('col {0} is numeric'.format(feature_name))
        return tf.feature_column.numeric_column(feature_name)
    # categorical and embedded features
    if feature_params[0] in ['cat', 'emb']:
        vocab = feature_params[1]
        
        # save vocabulary for TensorBoard
        if out_path is not None:
            if out_path[-1] != '/':
                out_path += '/'
            if not os.path.isdir(out_path):
                os.makedirs(out_path)
            f = open(out_path + feature_name + '.txt', 'w')
            f.write('label\tId\n')
            for j, s in enumerate(vocab):
                f.write(str(s) + '\t' + str(j) + '\n')
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
                feature type (str)  'cts'/'spl', 'cat', 'emb'
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
        if feature_dict[feature][0] == 'cts' or feature_dict[feature][0] == 'spl':
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


def incr_build(model,  by_var, start_list, add_list, get_data_fn, sample_size, feature_dict, target_var,
               batch_size, epochs_list, global_valid_df_in,
               model_dir=None, plot=False, verbose=0, output_size = 1, **kwargs):
    """
    This function builds a sequence of models. The get_data_fn takes a list of values as contained in
    start_list and add_list and returns data subset to those values. The initial model is built on the
    values of start_list and then evaluated on the data subset to the first value of add_list.
    
    At the next step, the data in the first element of add_list is added to the start_list data, the model
    is updated and the evaluation is conducted on the second element of add_list
    
    :param model: input model structure
    :type model: tf keras model
    :param start_list: list of (general) time periods for model build for the first model build
    :type start_list: list
    :param add_list: list of out-of-time periods to evaluate
    :type add_list: list
    :param get_data_fn: function to get a pandas DataFrame of data to work on
    :type get_data_fn: function
    :param sample_size: size of pandas DataFrames to get
    :type sample_size: int
    :param feature_dict: dictionary of features in the model
    :type feature_dict: dict
    :param target_var: target variable of model build
    :type target_var: str
    :param batch_size: size of batches for model build
    :type batch_size: int
    :param epochs_list: list (length 2) of epochs for model fit; entry 0 is initial model, entry 1 is subsequent
                        models
    :type epochs_list: list
    :param global_valid_df_in: DataFrame that includes all the segments in add_list -- for validation
    :type global_valid_df_in: pandas DataFrame
    :param model_dir: directory to save models
    :type model_dir: str
    :param plot: if True, plot history
    :type plot: bool
    :param verbose: print verobisity for keras.fit (0 = quiet, 1 = normal level, 2=talkative)
    :type verbose int
    :param output_size: the number of columns returned by keras model predict
    :type output_size: int
    :return: lists of out-of-sample values:
             add_list
             rmse  root mean squared error
             corr  correlation
    """

    if model_dir is not None:
        if model_dir[-1] != '/':
            model_dir += '/'
        if os.path.isdir(model_dir):
            os.system('rm -r ' + model_dir)
        os.makedirs(model_dir)

    build_list = start_list
    epochs = epochs_list[0]
    segs = []

    global_valid_df = global_valid_df_in.copy()
    # validation data
    if output_size == 1:
        global_valid_df['model_dnn_inc'] = np.full((global_valid_df.shape[0]), 0.0)
    else:
        for c in range(output_size):
          global_valid_df['model_dnn_inc' + str(c)] = np.full((global_valid_df.shape[0]), 0.0)
    global_valid_ds = get_tf_dataset(feature_dict, target_var, global_valid_df, 10000, 1)

    for j, valid in enumerate(add_list):
        segs += [valid]
        model_df = get_data_fn(build_list, sample_size, **kwargs)
        steps_per_epoch = int(model_df.shape[0] / batch_size)
        model_ds = get_tf_dataset(feature_dict, target_var, model_df, batch_size=batch_size)
        
        valid_df = get_data_fn([valid], sample_size, **kwargs)
        valid_ds = get_tf_dataset(feature_dict, target_var, valid_df, batch_size=batch_size, repeats=1)
        
        print('Data sizes for out-of-sample value {0}: build {1}, validate {2}'.format(valid, model_df.shape[0],
                                                                                       valid_df.shape[0]))

        history = model.fit(model_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
                            validation_data=valid_ds, verbose=verbose)
        
        gyh = model.predict(global_valid_ds)
        
        i = global_valid_df[by_var] == valid
        if output_size == 1:
            global_valid_df.loc[i, 'model_dnn_inc'] = gyh[i]
        else:
            for c in range(output_size):
                global_valid_df.loc[i, 'model_dnn_inc' + str(c)] = gyh[i][:,c]

        build_list += [valid]  # NOTE Accumulates
#        build_list = [valid]   # NOTE Accumulates NOT
        if model_dir is not None:
            out_m = model_dir + "before_" + valid + '.h5'
            model.save(out_m, overwrite=True, save_format='h5')
        
        if plot:
            title = 'model loss\n' + 'Training up to ' + valid
            plot_history(history, ['loss', 'val_loss'], 'loss', title=title)

        epochs = epochs_list[1]

    return segs, global_valid_df


def marginal(model, features_target, features_dict, sample_df_in, plot_dir=None, num_sample=100, cat_top=10,
             in_browser=False, column=None, title=None):
    """
    Generate plots to illustrate the marginal effects of the model 'model'. Live plots are output to the default
    browser and, optionally, png's are written to plot_dir

    The process is:

    - The model output is found on sample_df:
        - Six groups based on the quantiles [0, .1,. .25, .5, .75, .9, 1] are found from sample_df.
    - for each feature in features:
        - Six graphs are contructed: one for each group defined above.
            - The graph covers
            -  A random sample of size num_sample is taken from this group
            -  The target feature value is replace by an array that has values of its
               [0.01, .1, .2, .3, .4, .5, .6, .7, .8, 0.9, 0.99] quantiles in this group, if it is continuous or
               is the top cat_top [None means all] most frequent levels if categorical
            - The model output is found for all these
            - Boxplots are formed.  These plots have a common y-axis with limits from the .01 to .99 quantile
              of the model output on sample_df
        - A seventh graph show the distribution of the feature across the six groups.
            - For continuous features, these are box plots of the feature distribution *within* each model output group.
            - For discrete features, these are histograms of each feature level *across* the model output groups.
              These ARE NOT the feature distribution within each model group -- which is influenced the the
              prevelance of feature levels (e.g. there are a lot of loans in CA). This latter info can be gleaned
              from the first 6 plots since the feature levels are in descending order of prevelance within the
              model group.

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
    :type model: tf.keras.Mode
    :param features_target: features to generate plots for.
    :type features_target: list of str
    :param features_dict: dictionary whose keys are the features in the model
    :type features_dict: dict
    :param sample_df_in: DataFrame from which to take samples and calculate distributions
    :type sample_df_in:  pandas DataFrame
    :param plot_dir: directory to write plots out to
    :type plot_dir: str
    :param num_sample: number of samples to base box plots on
    :type num_sample: int
    :param cat_top: maximum number of levels of categorical variables to plot
    :type cat_top: int
    :param in_browser: if True, plot in browser
    :type in_browser: bool
    :param column: column or list of columns to use from keras model .predict
    :type column: int or list of ints
    :param title: optional additional title for graphs
    :type title: str
    :return: for each target, the range of the median across the target levels for each model output group
    :rtype dict
    """
    
    if plot_dir is not None:
        if plot_dir[-1] != '/':
            plot_dir += '/'
    pio.renderers.default = 'browser'
    
    sample_df = sample_df_in.copy()
    
    sample_df['target'] = np.full(sample_df.shape[0], 0.0)
    score_ds = tfu.get_tf_dataset(features_dict, 'target', sample_df, sample_df.shape[0], 1)
    
    sample_df['yh'] = tfu.get_pred(model.predict(score_ds), column)  # np.array(model.predict(score_ds)).flatten()
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
    cols = ['violet', 'blue', 'green', 'yellow', 'orange', 'red']
    
    for j in range(num_grp):
        sub_title = 'Model Output in {0} to {1}'.format(round(quantiles.iloc[j], 2), round(quantiles.iloc[j + 1], 2))
        sub_title += '<br>'
        sub_title += 'Quantile {0} to {1}'.format(target_qs[j], target_qs[j + 1])
        sub_titles += [sub_title]
    sub_titles += ['Feature Distribution Over<br>Model Output Groups']
    for target in features_target:
        # the specs list gives some padding between the top of the plots and the overall title
        fig = make_subplots(rows=1, cols=num_grp + 1, subplot_titles=sub_titles,
                            specs=[[{'t': 0.05}, {'t': 0.05}, {'t': 0.05},
                                    {'t': 0.05}, {'t': 0.05}, {'t': 0.05},
                                    {'t': 0.05}]])
        
        median_ranges = []
        # go across the model output groups
        for j in range(num_grp):
            yhall = None
            i = sample_df['grp'] == 'grp' + str(j)
            
            if features_dict[target][0] == 'cts' or features_dict[target][0] == 'spl':
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
                score_ds = tfu.get_tf_dataset(features_dict, 'target', score_df, nobs, 1)
                
                yh = tfu.get_pred(model.predict(score_ds), column)  # np.array(model.predict(score_ds)).flatten()
                
                # stack up the model outputs
                if yhall is None:
                    yhall = yh
                    xall = np.array(score_df[target]).flatten()
                else:
                    yhall = np.append(yhall, yh)
                    xall = np.append(xall, np.array(score_df[target]).flatten())
            
            # create grouped boxplots based on the values of the target feature
            if features_dict[target][0] == 'cts' or features_dict[target][0] == 'spl':
                xv = pd.DataFrame({'x': [str(round(x, 2)) for x in xall], 'yh': yhall})
                fig.add_trace(go.Box(x=xv['x'], y=xv['yh'], marker=dict(color=cols[j])), row=1, col=j + 1)
            else:
                xv = pd.DataFrame({'x': [str(x) for x in xall], 'yh': yhall})
                fig.add_trace(go.Box(x=xv['x'], y=xv['yh'], marker=dict(color=cols[j])), row=1, col=j + 1)
            # give the figure a title
            fig.update_annotations(sub_title='Group ' + str(j), row=1, col=j + 1)
            fig.update_traces(name='grp ' + str(j), row=1, col=j + 1)
            medians = xv.groupby('x')['yh'].median()
            median_ranges += [medians.max() - medians.min()]
        ##
        if features_dict[target][0] == 'cts' or features_dict[target][0] == 'spl':
            for j, g in enumerate(['grp' + str(j) for j in range(num_grp)]):
                i = sample_df['grp'] == g
                if j == 0:
                    nm = 'Lowest'
                elif j == num_grp - 1:
                    nm = 'Highest'
                else:
                    nm = 'G' + str(j)
                fig.add_trace(go.Box(y=sample_df.loc[i][target], name=nm, marker=dict(color=cols[j]), ),
                              row=1, col=num_grp + 1)
        else:
            cts = sample_df[target].value_counts().sort_values(ascending=False)
            cat = cts.index
            if cts.shape[0] > 5:
                cat = cat[0:5]
            for c in cat:
                i = sample_df[target] == c
                cts = sample_df.loc[i]['grp'].value_counts().sort_index()
                probs = 100.0 * cts / cts.sum()
                fig.add_trace(go.Bar(x=probs.index, y=probs, text=c, textposition='outside',
                                     marker=dict(color=cols)),
                              row=1, col=num_grp + 1)
        
        #            for j, g in enumerate(['grp' + str(j) for j in range(num_grp)]):
        #                i =sample_df['grp'] == g
        #                cts = sample_df.loc[i][target].value_counts().sort_values(ascending=False)
        #                probs = 100.0 * cts / cts.sum()
        #                if cts.shape[0] > 5:
        #                    probs = probs.iloc[0:5]
        #                fig.add_trace(go.Bar(x=probs.index, y=probs, marker=dict(color=cols[j]), name='G' + str(j)),
        #                              row=1, col=num_grp+1)
        
        #                fig.add_trace(go.Histogram(x=sample_df.loc[i][target].sort_values(), name='G' + str(j),
        #                                           marker = dict(color=cols[j]),
        #                                           histnorm='probability density'),
        #                              row=1, col=num_grp+1)
        
        importance[target] = max(median_ranges)
        # overall title
        titl = ''
        if title is not None:
            titl = title + '<br>' + titl
        titl += 'Marginal Response for ' + target
        fig.update_layout(
            title=dict(text=titl, font=dict(size=24), xanchor='center', xref='paper',
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
        if in_browser:
            fig.show()
        if plot_dir is not None:
            fname = plot_dir + 'html/' + target + '.html'
            fig.write_html(fname)
            
            # needed for png to look decent
            fig.update_layout(width=1800, height=600)
            fname = plot_dir + 'png/' + target + '.png'
            fig.write_image(fname)
    
    imp_df = pd.DataFrame(importance, index=['max median range']).transpose()
    imp_df = imp_df.sort_values('max median range', ascending=False)
    return imp_df
