"""
Utilities that help with the building of tensorflow keras models

"""
import gc

import tensorflow as tf
import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import warnings

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
        x = np.arange(first_epoch, len(history[g]) - first_epoch)
        y = history[g][first_epoch:len(history[metric])]
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


def build_column(feature_name, feature_params, out_path=None, print_details=False):
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
    :param print_details: print info about each feature
    :return: tf feature column and (for 'cat' and 'emb') a list of levels (vocabulary)
    """
    
    if feature_params[0] == 'cts' or feature_params[0] == 'spl':
        if print_details:
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
            if print_details:
                print('col {0} is categorical with {1} levels'.format(feature_name, len(vocab)))
            return col_ind
      
        # for embedded features, the third element of feature_params input is the dimension of the
        # embedding
        levels = feature_params[3]
        col_emb = tf.feature_column.embedding_column(col_cat, levels)
        if print_details:
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


def _marginal_cts(model, column, features_dict, sample_df, target, num_grp, num_sample, title, sub_titles, cols):
    """
    Build a Marginal Effects plot for a continuous feature
    
    :param model: model
    :param column: column of model output
    :param features_dict: features in the model
    :param sample_df: DataFrame operating on
    :param target: target feature
    :param num_grp: # of groups model output is sliced into
    :param num_sample: # of obs to take from sample_df to build graph
    :param title: title for graph
    :param sub_titles: titles for subplots
    :param cols: colors
    :return: plotly_fig and importance metric
    """
    
    sub_titles[6] = 'Box Plots'
    # 't' is top spacing, 'b' is bottom, 'None' means there is no graph in that cell. We make
    # 2 x 7 -- eliminating the (2,7) graph and putting the RHS graph in the (1,7) position
    fig = make_subplots(rows=2, cols=num_grp + 1, subplot_titles=sub_titles,
                        row_heights=[1, .5],
                        specs=[[{'t': 0.07, 'b': -.1}, {'t': 0.07, 'b': -.10}, {'t': 0.07, 'b': -.10},
                                {'t': 0.07, 'b': -.10}, {'t': 0.07, 'b': -.10}, {'t': 0.07, 'b': -.10},
                                {'t': 0.35, 'b': -0.35}],
                               [{'t': -0.07}, {'t': -.07}, {'t': -.07}, {'t': -0.07}, {'t': -.07},
                                {'t': -.07}, None]])
    
    # start with top row graphs
    # find ranges by MOG and merge
    lows = sample_df.groupby('grp')[target].quantile(.01)
    highs = sample_df.groupby('grp')[target].quantile(.99)
    both = pd.merge(left=lows, right=highs, left_index=True, right_index=True)
    both.rename(columns={target + '_x': 'low', target + '_y': 'high'}, inplace=True)
    
    # repeat these to accomodate the range of the feature we're going to build next
    to_join = pd.concat([both] * 11).sort_index()
    
    # range of the feature
    xval = np.arange(11) / 10
    xval = np.concatenate([xval] * num_grp)
    to_join['steps'] = xval
    to_join[target] = to_join['low'] + (to_join['high'] - to_join['low']) * to_join['steps']
    
    # now sample the DataFrame
    samps = sample_df.groupby('grp').sample(num_sample, replace=True)
    samps['samp_num'] = np.arange(samps.shape[0])
    # drop the target column -- we're going to replace it with our grid of values
    samps.pop(target)
    # join in our grid
    score_df = pd.merge(samps, to_join[target], on='grp')
    nobs = score_df.shape[0]
    score_df['target'] = np.full(nobs, 0.0)  # noop value
    score_ds = get_tf_dataset(features_dict, 'target', score_df, nobs, 1)
    
    # get model output
    score_df['yh'] = get_pred(model.predict(score_ds), column)
    
    # need to convert our feature values to string so rounding doesn't make graph look stupid
    xplot_name = target + '_str'
    score_df[xplot_name] = np.round(score_df[target], 2).astype(str)
    
    # add the boxplots. I don't see a way to do grouped boxplots in a single pass in a subplot.
    for j in range(num_grp):
        i = score_df['grp'] == 'grp' + str(j)
        fig.add_trace(go.Box(x=score_df.loc[i][xplot_name], y=score_df.loc[i]['yh'], marker=dict(color=cols[j])),
                      row=1, col=j + 1)
    
    # now let's do bottom row
    maxyr2 = 0.0
    for j in range(num_grp):
        i = sample_df['grp'] == 'grp' + str(j)
        fig.add_trace(go.Histogram(x=sample_df.loc[i][target], marker=dict(color=cols[j]),
                                   histnorm='probability density'),
                      row=2, col=j + 1)
        # this appears to be the only way to get the bin heights of the histogram -- the histogrm is built
        # by javascript. All that's stored in the
        ym = fig.full_figure_for_development(warn=False)['layout']['yaxis' + str(num_grp + 2 + j)]['range'][1]
        if ym > maxyr2:
            maxyr2 = ym
    xlab = '(Top row values span 1%ile-99%ile within each model output group)'
    
    # set a common x range for the bottom row
    xrng = sample_df[target].quantile([0.01, 0.99])
    xmin2 = float(xrng.iloc[0])
    xmax2 = float(xrng.iloc[1])
    xmin2 -= 0.01 * (xmax2 - xmin2)
    xmax2 += 0.01 * (xmax2 - xmin2)
    for j in range(num_grp):
        fig['layout']['yaxis' + str(num_grp + 2 + j)]['range'] = [0, maxyr2]
        fig['layout']['xaxis' + str(num_grp + 2 + j)]['range'] = [xmin2, xmax2]
    
    # Now do RHS graph
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
    mm = sample_df[target].quantile([0.01, 0.99])
    minys = float(mm.iloc[0])
    maxys = float(mm.iloc[1])
    fig['layout']['yaxis' + str(num_grp + 1)]['range'] = [minys, maxys]
    
    # Make it pretty
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
    fig.add_annotation(text='Within Group Distribution', font=dict(size=20), x=0.45, xanchor='center', xref='paper',
                       y=.4, yanchor='top', yref='paper', yshift=-40, showarrow=False)
    maxy = score_df['yh'].max()
    miny = score_df['yh'].min()
    for jj in range(num_grp):
        fig['layout']['yaxis' + str(jj + 1)]['range'] = [miny, maxy]
    for jj in range(1, num_grp):
        fig['layout']['yaxis' + str(jj + 1)]['showticklabels'] = False
        fig['layout']['yaxis' + str(num_grp + 2 + jj)]['showticklabels'] = False
    
    meds = score_df.groupby(['grp', xplot_name], observed=True)['yh'].median()
    medr = meds.groupby('grp').max() - meds.groupby('grp').min()
    imp_within = medr.max()
    
    return fig, imp_within


def _marginal_cat(model, column, features_dict, sample_df, target, num_grp, num_sample, title, sub_titles, cols):
    """
    Build a Marginal Effects plot for a categorical feature

    :param model: model
    :param column: column of model output
    :param features_dict: features in the model
    :param sample_df: DataFrame operating on
    :param target: target feature
    :param num_grp: # of groups model output is sliced into
    :param num_sample: # of obs to take from sample_df to build graph
    :param title: title for graph
    :param sub_titles: titles for subplots -- must be 7 long
    :param cols: colors
    :return: plotly_fig and importance metric
    """

    if len(sub_titles) != 7:
        warnings.warn('Incorrect # of sub_titles')
        return
    sub_titles[6] = 'Across Group<br>Distribution'
    # 't' is top spacing, 'b' is bottom, 'None' means there is no graph in that cell. We make
    # 2 x 7 -- eliminating the (2,7) graph and putting the RHS graph in the (1,7) position
    fig = make_subplots(rows=2, cols=num_grp + 1, subplot_titles=sub_titles,
                        row_heights=[1, .5],
                        specs=[[{'t': 0.07, 'b': -.1}, {'t': 0.07, 'b': -.10}, {'t': 0.07, 'b': -.10},
                                {'t': 0.07, 'b': -.10}, {'t': 0.07, 'b': -.10}, {'t': 0.07, 'b': -.10},
                                {'t': 0.35, 'b': -0.35}],
                               [{'t': -0.07}, {'t': -.07}, {'t': -.07}, {'t': -0.07}, {'t': -.07},
                                {'t': -.07}, None]])
    
    # row 1 graphs
    # get levels & counts of the feature within each MOG. Note these will be desc by count within MOG
    vcts = sample_df.groupby('grp')[target].value_counts().rename('cts', inplace=True).reset_index()
    # keep at most the top 10 within each MOG
    to_join = vcts.groupby('grp').head(10).reset_index()
    to_join.set_index('grp', inplace=True)
    # totals within each MOG
    vcts_tot = vcts.groupby(target)['cts'].sum().rename('tots').reset_index()
    # normalize to probabilities
    probs = pd.merge(vcts, vcts_tot, on=target)
    probs['prob'] = probs['cts'] / probs['tots']
    # keep those levels that are in any of the MOG graphs
    itop = probs[target].isin(to_join[target].unique())
    probs = probs.loc[itop]
    
    # get sample
    samps = sample_df.groupby('grp').sample(num_sample, replace=True)
    samps['samp_num'] = np.arange(samps.shape[0])
    # drop target -- we're going to replace these
    samps.pop(target)
    # join to sample
    score_df = pd.merge(samps, to_join[target], on='grp')
    nobs = score_df.shape[0]
    score_df['target'] = np.full(nobs, 0.0)  # noop value
    score_ds = get_tf_dataset(features_dict, 'target', score_df, nobs, 1)
    
    # get model output
    score_df['yh'] = get_pred(model.predict(score_ds), column)
    
    xplot_name = target + '_str'
    score_df[xplot_name] = score_df[target].astype(str)
    for j in range(num_grp):
        i = score_df['grp'] == 'grp' + str(j)
        fig.add_trace(go.Box(x=score_df.loc[i][xplot_name], y=score_df.loc[i]['yh'], marker=dict(color=cols[j])),
                      row=1, col=j + 1)
    
    # generate row 2 graphs
    maxp = 0.0
    for j in range(num_grp):
        g = 'grp' + str(j)
        grp_vals = to_join.loc[g][target]
        i = (probs['grp'] == g) & (probs[target].isin(list(grp_vals)))
        pi = probs.loc[i].sort_values('cts', ascending=False)
        p = pi['cts'] / pi['cts'].sum()
        if p.max() > maxp:
            maxp = p.max()
        fig.add_trace(go.Bar(x=pi[target], y=p, marker=dict(color=cols[j]), name='Group ' + str(j)),
                      row=2, col=j + 1)
    for jj in range(num_grp):
        fig['layout']['yaxis' + str(num_grp + 2 + jj)]['range'] = [0.0, maxp]
    xlab = '(Up to top 10 levels within group)'
    
    # Generate RHS graph. If we have too many categories, the graph is very difficult to read
    # But we don't want just keep the top k because we may lose interesting changes across the MOGs,
    # so let's keep the top K per MOG and keep lowering k until we get a value <= 6
    cats = probs[target].unique()
    nhead = 3
    while cats.shape[0] > 6:
        cats = probs.sort_values(['grp', 'prob'], ascending=[True, False]).groupby('grp').head(nhead)[
            target].unique()
        nhead -= 1
    for cat in cats:
        i = probs[target] == cat
        fig.add_trace(go.Bar(x=probs.loc[i]['grp'], y=probs.loc[i]['prob'], text=cat,
                             textposition='outside', marker=dict(color=cols), name=cat),
                      row=1, col=num_grp + 1)
    
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
    fig.add_annotation(text='Within Group Distribution', font=dict(size=20), x=0.45, xanchor='center', xref='paper',
                       y=.4, yanchor='top', yref='paper', yshift=-40, showarrow=False)
    maxy = score_df['yh'].max()
    miny = score_df['yh'].min()
    for jj in range(num_grp):
        fig['layout']['yaxis' + str(jj + 1)]['range'] = [miny, maxy]
    for jj in range(1, num_grp):
        fig['layout']['yaxis' + str(jj + 1)]['showticklabels'] = False
        fig['layout']['yaxis' + str(num_grp + 2 + jj)]['showticklabels'] = False
    
    meds = score_df.groupby(['grp', xplot_name], observed=True)['yh'].median()
    medr = meds.groupby('grp').max() - meds.groupby('grp').min()
    imp_within = medr.max()
    
    return fig, imp_within


def marginal(model, features_target, features_dict, sample_df, model_col, plot_dir=None, num_sample=100, in_browser=False, column=None, title=None, slices=dict()):
    """
    Generate plots to illustrate the marginal effects of the model 'model'. Live plots are output to the default
    browser and, optionally, png's are written to plot_dir

    The process is:

    - Define six model output groups (MOG) by the quantiles of the model output:
        - Q(0) to Q(.1)
        - Q(.1) to Q(.25)
        - Q(.25) to Q(.5)
        - Q(.5) to Q(.75)
        - Q(.75) to Q(.9)
        - Q(.9) to Q(1)

    - for each feature in features:
        - **top row**: Six graphs are contructed: one for each group defined above.
            -  For each MOG
                1. A random sample of size num_sample is taken from the MOG
                2. The target feature is replaced by:
                    - Continuous Feature: an equally spaced array from Q(0.01) to Q(0.99) where the quantiles are
                      found on the feature values within the MOG
                    - Categorical feature: the levels of the feature within the MOG arranged in descending frequency
                      within the MOG
                3. The model output is found for all these
                4. Grouped boxplots are formed.
            - These plots have a common y-axis range

        - **bottom row**: Six graphs are constructed. These graphs show the distribution of the feature *within* the
          MOG.
        - RHS (right-hand side) graph show the distribution of the feature.
            - For continuous features, these are box plots of the feature distribution *within* each model output group.
              They are the boxpolot equivalent of the bottom row.
            - For discrete features, these are bar charts of each feature level *across* the model output groups.
              These ARE NOT the feature distribution within each model group (bottom row).

    Features:
        - Since the x-values are sampled from sample_df, any correlation within the features not plotted on the
          x-axis are preserved.
        - The values of the target feature used are those observed within each MOG, so extrapolation
          into unobserved space is reduced. The potential correlation between this feature and the others, however,
          is lost.

    **Return** a metric that rates the importance of the feature to the model (e.g. sloping). It is calculated as:

    - For each MOG top row graph, calculate the range of the median from the boxplots.
    - Find the maximum range across the six MOGs.
    - This value is termed 'importance' and is returned.

    Currently the MOG groups are defined once -- not separately for each slice

    :param model: A keras tf model with a 'predict' method that takes a tf dataset as input
    :type model: tf.keras.Mode
    :param features_target: features to generate plots for.
    :type features_target: list of str
    :param features_dict: dictionary whose keys are the features in the model
    :type features_dict: dict
    :param sample_df_in: DataFrame from which to take samples and calculate distributions
    :type sample_df_in:  pandas DataFrame
    :param model_col: name of model output in sample_df
    :type model_col: str
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
    :param slices: optional slices of sample_df_in to also make graphs for. key to dict is name of slice, entry is
                   boolean array for .loc access to sample_df_in
    :type slices: dict
    :return: for each target, the range of the median across the target levels for each model output group
    :rtype dict
    """
    
    pio.renderers.default = 'browser'
    
    target_qs = [0, .1, .25, .5, .75, .9, 1]
    quantiles = sample_df[model_col].quantile(target_qs)
    quantiles.iloc[0] -= 1.0
    num_grp = quantiles.shape[0] - 1
    if num_grp != 6:
        warnings.warn('Did not get 6 MOG groups')
        return
    # now we have the six MOG groups that we will base the graphs on
    sample_df['grp'] = pd.cut(sample_df[model_col], quantiles, labels=['grp' + str(j) for j in range(num_grp)], right=True)

    sub_titles = []
    importance = {}
    # reverse(ROYGBIV)
    cols = ['#7d459c', '#2871a7', '#056916', '#dbac1a', '#dd7419', '#bd0d0d']

    # graph titles. The title of the RHS graph depends on the feature type -- so it's assigned later
    for j in range(num_grp):
        sub_title = 'Model Output in {0} to {1}'.format(round(quantiles.iloc[j], 2), round(quantiles.iloc[j + 1], 2))
        sub_title += '<br>'
        sub_title += 'Quantile {0} to {1}'.format(target_qs[j], target_qs[j + 1])
        sub_titles += [sub_title]
    sub_titles += ['Place Holder']
    slices['Overall'] = np.full((sample_df.shape[0]), True)
    
    # go through the features
    for target in features_target:
        # run through the slices
        for slice in slices.keys():
            i = slices[slice]
            if i.sum() > 100:
                title_aug = title + '<br>Slice: ' + slice
                if features_dict[target][0] == 'cts' or features_dict[target][0] == 'spl':
                    fig, imp_in = _marginal_cts(model, column, features_dict, sample_df.loc[i], target, num_grp, num_sample, title_aug,
                                                sub_titles, cols)
                else:
                    fig, imp_in = _marginal_cat(model, column, features_dict, sample_df.loc[i], target, num_grp, num_sample, title_aug,
                                                sub_titles, cols)
                importance[target + '_' + slice] = imp_in
                if in_browser:
                    fig.show()
                if plot_dir is not None:
                    if plot_dir[-1] != '/':
                        plot_dir += '/'
                    pdir = plot_dir + slice + '/'
                    os.makedirs(pdir + 'html/', exist_ok=True)
                    os.makedirs(pdir + 'png/', exist_ok=True)
                    
                    fname = pdir + 'html/Marginal_'  + target + '.html'
                    fig.write_html(fname)
                    
                    # needed for png to look decent
                    fig.update_layout(width=1800, height=1150)
                    fname = pdir + 'png/Marginal_' + target + '.png'
                    fig.write_image(fname)
            else:
                print('No marginal graph for {0} and slice {1}'.format(target, slice))

    imp_df = pd.DataFrame(importance, index=['importance']).transpose()
    imp_df = imp_df.sort_values('importance', ascending=False)
    return imp_df
