"""
utilities that are helpful in general model building

"""
from muti import chu, tfu
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import scipy.stats as stats
import math
import os


def r_square(yh, y):
    """
    find the r-square for the model implied by yh
    
    :param yh: model output
    :type yh: pd.Series or nd.array
    :param y: actual values
    :type y: pd.Series or nd.array
    :return: r-squared
    :rtype float
    """
    res_full = y - yh
    res_reduced = y - y.mean()
    r2 = 100.0 * (1.0 - np.square(res_full).sum() / np.square(res_reduced).sum())
    return float(r2)


def get_unique_levels(feature, client, db, table, cnt_min=None):
    """
    Retrieves the unique levels of the column 'feature' in the table 'table' of database db.
    At most 1000 are returned.

    :param feature: column name in db.table to get unique levels
    :type feature: str
    :param client: clickhouse client connector
    :type client: clickhouse_driver.Client
    :param db: database name
    :type db: str
    :param table: table name
    :type table: str
    :param cnt_min: minimum count for a level to be returned
    :type cnt_min: int
    :return: list of unique levels and the most frequent level
    :rtype list, <value>
    """
    qry = 'SELECT ' + feature + ' AS grp, count(*) AS nl FROM ' + db + '.' + table + ' GROUP BY grp'
    if cnt_min is not None:
        qry += ' HAVING nl > ' + str(cnt_min)
    qry += ' ORDER BY nl DESC LIMIT 1000'
    df = chu.run_query(qry, client, return_df=True)
    most_freq_level = df.iloc[0]['grp']
    df.sort_values('grp')
    u = list(df['grp'])
    return u, most_freq_level


def get_closest(ul, field, target, db, table, client):
    """
    This function is designed to select the out-of-list default value for an embedding. It selects this value
    as the in-list value which has target mean closest to the average value of all out-of-list values
    
    :param ul: in-list values
    :type ul: list
    :param field: name of field we're working on
    :type field: str
    :param target: target field used for assessing 'close'
    :type target: str
    :param db: database to use
    :type db: str
    :param table: table to use
    :type table: str
    :param client: clickhouse client
    :type client: clickhouse_driver.client
    :return: value of in-list elements with average closest to out-of-list averages
    """
    qry = """
        /*
          we have a feature that has lots of levels. Some levels are part of the embedding. For those that aren't
          we want to find the default value -- that level which has the closest mean of a target to them.
        
          TTTT  list of values called out in embedding
          XXXX  field we're working with
          YYYY  target variable
          ZZZZ  db.table to query
         */
        SELECT
          XXXX AS grp,
          avg(arrayAvg(YYYY)) AS in_avg,
          (SELECT
            avg(arrayAvg(YYYY))
          FROM
              ZZZZ
          WHERE
            XXXX not in (TTTT)) AS out_avg,
          abs(in_avg - out_avg) AS mad
        FROM
          ZZZZ
        WHERE
          grp in (TTTT)
          GROUP BY grp
          ORDER BY mad
        LIMIT 1
    """
    repl = ''
    for j, u in enumerate(ul):
        if j != 0:
            repl += ', '
        repl += "'" + u + "'"
    
    df = chu.run_query(qry, client, return_df=True,
                       replace_source=['TTTT', 'XXXX', 'YYYY', 'ZZZZ'],
                       replace_dest=[repl, field, target, db + '.' + table])
    print('Out-of-list element selection for field {0} using target {1}'.format(field, target))
    print(df)
    print('\n')
    return df.iloc[0]['grp']


def cont_hist(yh, y, title='2D Contour Histogram', xlab='Model Output', ylab='Y', subtitle=None, plot_dir=None,
              in_browser=False):
    """
    Make a 2D contour histogram plot of y vs yh.
    The plot is produced in the browser and optionally written to a file.
    
    :param yh: Model outputs
    :type yh: nd.array or pd.Series
    :param y: Target value
    :type y: nd.array or pd.Series
    :param title: Title for plot
    :type title: str
    :param xlab: x-axis label
    :type xlab: str
    :param ylab: y-axis label
    :type ylab: str
    :param subtitle: optional subtitle
    :type subtitle: str
    :param plot_dir: optional file to write graph to
    :type plot_dir: str
    :param in_browser: if True plot to browser
    :type in_browser: bool
    :return:
    """
    
    fig = [go.Histogram2dContour(x=yh, y=y)]
    min_value = min([yh.min(), y.quantile(.01)])
    max_value = max([yh.max(), y.quantile(.99)])
    fig += [go.Scatter(x=[min_value, max_value], y=[min_value, max_value],
                       mode='lines', line=dict(color='red'))]
    
    if subtitle is not None:
        title += title + '<br>' + subtitle
    layout = go.Layout(title=dict(text=title, x=0.5),
                       height=800, width=800,
                       xaxis=dict(title=xlab),
                       yaxis=dict(title=ylab))
    figx = go.Figure(fig, layout=layout)
    if in_browser:
        figx.show()
    if plot_dir is not None:
        figx.write_image(plot_dir + 'png/model_fit.png')
        figx.write_html(plot_dir + 'html/model_fit.html')


def ks_calculate(score_variable, binary_variable, plot=False, xlab='Score', ylab='CDF', title='KS Plot',
                 subtitle=None, plot_dir=None, out_file=None, in_browser=True):
    """
    Calculates the KS (Kolmogorov Smirnov) distance between two cdfs.  The KS statistic is 100 times the
    maximum vertical difference between the two cdfs

    The single input score_variable contains values from the two populations.  The two populations are distinguished
    by the value of binvar (0 means population A, 1 means population B).

    Optionally, the plot of the CDF of score variable for the two values of binary_variable may be plotted.

    :param score_variable: continuous variable from the logistic regression
    :type score_variable: pandas series, numpy array or numpy vector
    :param binary_variable: binary outcome (dependent) variable from the logistic regression
    :type binary_variable: numpy array or numpy vector
    :param plot: creates a graph if True
    :type plot: bool
    :param xlab: label for the x-axis (score variable), optional
    :type xlab: str
    :param ylab: label for the y-axis (binary variable), optional
    :type ylab: str
    :param title: title for the plot, optional
    :type title: str
    :param subtitle: subtitle for the plot, optional (default=None)
    :type subtitle: str
    :param plot_dir: directory to write plot to
    :type plot_dir str
    :param out_file file name for writing out the plot
    :type out_file: str
    :param in_browser: if True, plots to browser
    :type in_browser: bool
    :return: KS statistic (0 to 100),
    :rtype: float
    """
    
    if isinstance(score_variable, np.ndarray):
        score_variable = pd.Series(score_variable)
    
    if isinstance(binary_variable, np.ndarray):
        binary_variable = pd.Series(binary_variable)
    
    # divide the score_variable array by whether the binary variable is 0 or 1
    index0 = binary_variable == 0
    index1 = binary_variable == 1
    
    # sort the scores
    score0 = score_variable[index0]
    score0 = score0.sort_values()
    score0.index = np.arange(0, score0.shape[0])
    u0 = (np.arange(0, score0.shape[0]) + 1 - 0.5) / score0.shape[0]
    
    score1 = score_variable[index1]
    score1 = score1.sort_values()
    score1.index = np.arange(0, score1.shape[0])
    u1 = (np.arange(0, score1.shape[0]) + 1 - 0.5) / score1.shape[0]
    
    # interpolate these at common values
    delta = (score_variable.max() - score_variable.min()) / 100
    sc = np.arange(score_variable.min(), score_variable.max(), delta)
    
    ind0 = score0.searchsorted(sc)
    # it's possible that ind0 may have a value = score0.shape[0] (e.g. sc bigger than biggest score0)
    ind0[ind0 >= score0.shape[0]] = score0.shape[0] - 1
    uu0 = u0[ind0]
    
    ind1 = score1.searchsorted(sc)
    ind1[ind1 >= score1.shape[0]] = score1.shape[0] - 1
    uu1 = u1[ind1]
    
    ks = round(float(100.0 * max(abs(uu1 - uu0))), 1)
    
    if plot:
        pio.renderers.default = 'browser'
        fig = [go.Scatter(x=score0, y=u0, line=dict(color='black'))]
        fig += [go.Scatter(x=score1, y=u1, line=dict(color='black'))]
        
        maxx = score_variable.max()
        if subtitle is None:
            sub_title = "KS: " + str(ks)
        else:
            sub_title = subtitle + "\nKS: " + str(ks)
        layout = go.Layout(title=dict(text=title + '<br>' + sub_title, x=0.5),
                           showlegend=False,
                           xaxis=dict(title=xlab, range=[0, maxx]),
                           yaxis=dict(title=ylab, range=[0, 1]))
        figx = go.Figure(fig, layout=layout)
        if in_browser:
            pio.renderers.default = 'browser'
            figx.show()
        if out_file is not None:
            figx.write_image(plot_dir + 'png/' + out_file + '.png')
            figx.write_html(plot_dir + 'html/' + out_file + '.html')
    return ks


def decile_plot(score_variable, binary_variable, xlab='Score', ylab='Actual', title='Decile Plot',
                plot_maximum=None, plot_minimum=None, confidence_level=0.95, correlation=0, subtitle=None,
                plot_dir=None, out_file=None, in_browser=True):
    """
    This function creates the so-called decile plot.  The input data (score_variable, binary_variable) is
    divided into 10 equal groups based on the deciles of score_variable.  Within each decile, the values of the
    two are averaged.  These 10 pairs are plotted.  A reference line is ploted.  Within each group a confidence
    interval is plotted as a vertical line.  The user may specify the confidence level and also the pair-wise
    correltion between the points (binary variable) within a decile.



    :param score_variable: continuous variable from the logistic regression
    :type score_variable: pandas series, numpy array or numpy column vector
    :param binary_variable: binary outcome (dependent) variable from the logistic regression
    :type binary_variable: pandas series, numpy array or numpy column vector
    :param xlab: label for the x-axis (score variable), optional
    :type xlab: str
    :param ylab: label for the y-axis (binary variable), optional
    :type ylab: str
    :param title: title for the plot, optional
    :type title: str
    :param plot_maximum: maximum value for the plot, optional
    :type plot_maximum: float
    :param plot_minimum: minimum value for the plot, optional
    :type plot_minimum: float
    :param confidence_level: confidence level for confidence intervals around each decile, optional (default = 0.95)
    :type confidence_level: float
    :param correlation: pair-wise correlation between data within each decile, optional (default=0)
    :type correlation: float
    :param subtitle: subtitle for the plot, optional (default=None)
    :type subtitle: str
    :param plot_dir: directory to write plot to
    :type plot_dir: str
    :param out_file file name for writing out the plot
    :type out_file: str
    :param in_browser: if True, plots to browser
    :type in_browser: bool
    :return: plot
    :rtype: N/A
    """
    
    if isinstance(score_variable, np.ndarray):
        score_variable = pd.Series(score_variable)
    
    if isinstance(binary_variable, np.ndarray):
        binary_variable = pd.Series(binary_variable)
    
    bins = pd.qcut(score_variable, 10, labels=False, duplicates='drop')
    
    mscore = score_variable.groupby(bins).mean()
    counts = score_variable.groupby(bins).count()
    mbinary = binary_variable.groupby(bins).mean()
    vbinary = binary_variable.groupby(bins).var()
    
    # is the response binary?
    vals = np.unique(binary_variable)
    binary = (len(vals) == 2) and min(vals) == 0 and max(vals) == 1
    
    # Critical N(0,1) value for confidence intervals
    zcrit = stats.norm.isf((1.0 - confidence_level) / 2.0)
    
    if plot_maximum is None:
        # Want the x & y axes to have same range
        max_limit = max(max(mscore), max(mbinary))
        if binary:
            max_limit = round(max_limit + 0.05, 1)
        else:
            max_limit = round(max_limit * 1.05, 1)
    else:
        max_limit = plot_maximum  # User-supplied value
    
    if plot_minimum is None:
        # Want the x & y axes to have same range
        min_limit = min(mscore[0], mbinary[0])
        if binary:
            min_limit = round(min_limit - 0.05, 1)
        else:
            min_limit = round(min_limit * 0.95, 1)
    else:
        min_limit = plot_minimum  # User-supplied value
    
    # Reference line
    rxy = [min_limit, max_limit]
    # plot--deciles
    fig = [go.Scatter(x=rxy, y=rxy, line=dict(color='red'))]
    fig += [go.Scatter(x=mscore, y=mbinary, mode='markers',
                       line=dict(color='black'))]
    
    # Do confidence intervals
    for k in range(0, mbinary.shape[0]):
        if binary:
            variance = mbinary[k] * (1.0 - mbinary[k])
        else:
            variance = vbinary[k]
        width = zcrit * math.sqrt(variance * (1 + (1 + counts[k]) * correlation) / counts[k])
        fig += [go.Scatter(x=[mscore[k], mscore[k]], y=[mbinary[k] - width, mbinary[k] + width],
                           line=dict(color='black'), mode='lines')]
    
    # Make pretty
    if subtitle is not None:
        title += '<br>' + subtitle
    layout = go.Layout(title=dict(text=title, x=0.5),
                       showlegend=False,
                       xaxis=dict(title=xlab, range=[min_limit, max_limit]),
                       yaxis=dict(title=ylab, range=[min_limit, max_limit]))
    
    figx = go.Figure(fig, layout=layout)

    # Add # of obs
    annot = np.str(score_variable.shape[0]) + ' Obs'
    annot += '\nconfidence: ' + str(confidence_level)
    annot += '\ncorrelation: ' + str(correlation)
    rangexy = max_limit - min_limit

    figx.add_annotation(x=min_limit + 0.6*rangexy, y=min_limit+0.2*rangexy,
                        text=annot,
                        showarrow=False)
    # Add mean of binvar and score
    means_title = 'Actual ' + np.str(round(binary_variable.mean(), 3))
    means_title = means_title + '\nScore ' + np.str(round(score_variable.mean(), 3))
    figx.add_annotation(x=min_limit + 0.1 * rangexy, y=max_limit - 0.1 * rangexy,
                        text=means_title,
                        showarrow=False,
                        xshift=1, xref='paper')
    if in_browser:
        pio.renderers.default = 'browser'
        figx.show()
    if out_file is not None:
        figx.write_image(plot_dir + 'png/' + out_file + '.png')
        figx.write_html(plot_dir + 'html/' + out_file + '.html')
    return


def fit_by_feature(features, targets, sample_df_in, plot_dir=None, num_quantiles=10,
                   boot_samples=1000, boot_coverage=0.95, extra_title=None, in_browser=False):
    """
    Generates two plots to assess model fit.

    The first is a set of paired boxplots of the model output and target 'y' grouped by values of the feature.
    The second is a plot of the mean model output versus mean target 'y' grouped by values of the feature.

    :param features: features to generate plots for, key is feature name, value is 'cts'/'spl', 'cat', 'emb'.
    :type features: dict
    :param targets: dict with keys 'model_output' and 'target' that point to columns in sample_df_in
    :type targets dict
    :param sample_df_in: DataFrame from which to take samples and calculate distributions
    :type sample_df_in: pandas DataFrame
    :param plot_dir: directory to write plots out to
    :type plot_dir: str
    :param num_quantiles: number of quantiles at which to discretize continuous variables
    :type num_quantiles: int
    :param boot_samples: # of bootstrap samples to take
    :type boot_samples: int
    :param extra_title: optional second title line
    :type extra_title: str
    :param in_browser: True means also show in browser
    :type in_browser: bool
    :param boot_coverage: coverage of bootstrap CI
    :type boot_coverage: float

    """
    
    def boot_mean(y_in, num_samples, coverage=0.95):
        """
        Bootstraps the mean of y_in to form a CI with coverage 'coverage'.
        Assumes there's enough correlation in the data to reduce the effective sample size to 1/4 the length
        of y_in.

        :param y_in: data to form a CI for
        :type y_in: pandas Series
        :param num_samples: # of bootstrap samples to run
        :type num_samples: int
        :param coverage: CI coverage level (as a decimal)
        :type coverage: float
        :return: bootstrap CI
        :rtype list
        """
        means = []
        n = int(y_in.shape[0] / 4)
        alpha2 = (1.0 - coverage) / 2.0
        for j in range(num_samples):
            ys = y_in.sample(n, replace=True)
            means += [ys.mean()]
        med_df = pd.DataFrame({'means': means})
        ci_boot = med_df.quantile([alpha2, 1.0 - alpha2])
        return list(ci_boot['means'])
    
    pio.renderers.default = 'browser'
    
    sample_df = sample_df_in.copy()
    
    y = targets['target']
    yh = targets['model_output']
    
    for feature in features.keys():
        if features[feature][0] == 'cts' or features[feature][0] == 'spl':
            us = np.arange(num_quantiles + 1) / num_quantiles
            quantiles = sample_df[feature].quantile(us).unique()
            quantiles[0] -= 1.0
            decimals = 5
            while np.unique(np.round(quantiles, decimals)).shape[0] == quantiles.shape[0]:
                decimals -= 1
                if decimals < 0:
                    break
            quantiles = np.round(quantiles, decimals + 1)
            if decimals < 0:
                quantiles = quantiles.astype(int)
            sample_df[feature] = pd.cut(sample_df[feature], quantiles,
                                        labels=[feature + ' ' + str(quantiles[j + 1]) for j in
                                                range(quantiles.shape[0] - 1)], right=True)
        
        fig = [go.Box(x=sample_df[feature], y=sample_df[yh], name='model')]
        fig += [go.Box(x=sample_df[feature], y=sample_df[y], name='actual')]
        co = sample_df.groupby(feature)[yh].mean().sort_values(ascending=False).index
        
        layout = go.Layout(title='Model and Actual values by ' + feature,
                           xaxis=dict(title=feature, categoryorder='array', categoryarray=co),
                           yaxis=dict(title=y))
        
        figx = go.Figure(fig, layout=layout)
        figx.update_layout(boxmode='group')
        if in_browser:
            figx.show()
        
        co = sample_df.groupby(feature)[[y, yh]].mean()
        fig1 = [go.Scatter(x=co[yh], y=co[y], mode='markers', name='',
                           customdata=co.index, marker=dict(color='black'),
                           hovertemplate='%{customdata}<br>Model %{x}<br>Actual %{y}')]
        for indx in co.index:
            i = sample_df[feature] == indx
            ci = boot_mean(sample_df.loc[i][y], boot_samples, coverage=boot_coverage)
            x = [co.loc[indx][yh], co.loc[indx][yh]]
            fig1 += [go.Scatter(x=x, y=ci, mode='lines', line=dict(color='black'), name='')]
        minv = min([co[y].min(), co[yh].min()])
        maxv = max([co[y].max(), co[yh].max()])
        fig1 += [go.Scatter(x=[minv, maxv], y=[minv, maxv], mode='lines', line=dict(color='red'), name='')]
        title = 'mean Model vs Actual Grouped by ' + feature
        if extra_title is not None:
            title += '<br>' + extra_title
        layout1 = go.Layout(title=dict(text=title, x=0.5, xref='paper',
                                       font=dict(size=24)),
                            xaxis=dict(title='Model Output'),
                            yaxis=dict(title=y),
                            height=800,
                            width=800,
                            showlegend=False)
        figx1 = go.Figure(fig1, layout=layout1)
        xlab = 'Bootstrap CI at {0:.0f}% coverage'.format(100 * boot_coverage)
        figx1.add_annotation(text=xlab, font=dict(size=10), x=0.5, xanchor='center', xref='paper', y=0,
                             yanchor='top', yref='paper', yshift=-50, showarrow=False)
        if in_browser:
            figx1.show()
        if plot_dir is not None:
            if plot_dir[-1] != '/':
                plot_dir += '/'
            if co.shape[0] > 10:
                figx.update_layout(width=1800, height=600)
                
            fname = plot_dir + 'png/BoxPlotModelFit' + feature + '.png'
            figx.write_image(fname)

            fname = plot_dir + 'html/BoxPlotModelFit' + feature + '.html'
            figx.write_html(fname)

            fname = plot_dir + 'png/CrossMeanModelFit' + feature + '.png'
            figx1.write_image(fname)
            
            fname = plot_dir + 'html/CrossMeanModelFit' + feature + '.html'
            figx1.write_html(fname)


def make_dir_tree(base_path, dirs, rename_to=None):
    """
    Create a directory structure.
    The directory structure is created under base_path.  If base_path already exists, it is renamed to
    rename_to if that exists, otherwise it is **deleted**.

    dirs is a list that contains end-point directories (there is no need to specify directories higher
    up the tree).

    :param base_path: base path to (and *including*) the top-level directory of the structure
    :type base_path: str
    :param dirs: structure to build: list of 'leaf' directories
    :type dirs: list
    :param rename_to: if an existing structure is found, what to rename it to.
    :type rename_to: str
    :return: <none>
    """
    
    def check_path(path, new_name):
        if os.path.isdir(path):
            if rename_to is not None:
                if os.path.isdir(new_name):
                    raise IsADirectoryError(new_name + ' already exists')
                os.system('mv ' + path + ' ' + new_name)
            else:
                os.system('rm -r ' + path)
    
    if base_path[-1] != '/':
        base_path += '/'
    check_path(base_path, rename_to)
    os.makedirs(base_path)
    for p in dirs:
        os.makedirs(base_path + p)


def fit_eval(mod, df, yh, target_var, features, plot_dir, title, isin=None,
             in_browser=False, slice_dict=None, incl_ks = True):
    """
    driver function for fit_by_feature and ks_calculate & decile_plot. The function:
        - makes the necessary directories
        - Pulls out the required levels of the target value (if isin != None)
        - Calls fit_by_feature and ks_calculate, decile_plot
        - Cycles through the subsets specified by ilist, creating KS and Decile plots

    :param mod: keras model to evaluate
    :type mod: tf.keras.Model
    :param df: data frame with features in 'features' and target_var
    :type df: pandas DataFrame
    :param yh: output of keras.Model.Predict
    :type yh: tf.keras.Model.Predict output
    :param target_var: the target variable of the model. Must be a column of df.
    :type title: str
    :param features: dict returned by build_feature_dict
    :type features: dict
    :param plot_dir: directory to write graphs to. It will be created, plus subdirectories png & html
    :type plot_dir: str
    :param title: Title for plots
    :type title: str
    :param isin: list of values of target_var to base the plots on
    :type isin: list of ints
    :param in_browser: if True, also put plots in browser
    :type in_browser: bool
    :param slice_dict: dictionary of slices for ks/decile plots. key is the title and entry is a boolean array into df
    :type slice_dict: dict
    :param incl_ks: if True, generate KS plot, too
    :type incl_ks: bool
    :return: dictionary with the same keys as slice_dict (+ 'all') of importance values
    :rtype dict
    """
    if plot_dir[-1] != '/':
        plot_dir += '/'
    os.makedirs(plot_dir + 'effects/png', exist_ok=True)
    os.makedirs(plot_dir + 'effects/html', exist_ok=True)
    os.makedirs(plot_dir + 'marginal/png', exist_ok=True)
    os.makedirs(plot_dir + 'marginal/html', exist_ok=True)
    os.makedirs(plot_dir + 'ks_decile/png', exist_ok=True)
    os.makedirs(plot_dir + 'ks_decile/html', exist_ok=True)
    feats = features.copy()
    feats['model_output'] = ['cts']
    if isin is not None:
        df['model'] = tfu.get_pred(yh, isin)
        df['actual'] = df[target_var].isin(isin).astype(int)
    else:
        df['model'] = tfu.get_pred(yh)
        df['actual'] = df[target_var]

    df['model_output'] = df['model']
    targs = dict(model_output='model', target='actual')
    fit_by_feature(feats, targs, df, plot_dir + 'effects/', in_browser=in_browser,
                   boot_samples=100, extra_title=title)
    importance = tfu.marginal(mod, features, features, df, plot_dir + 'marginal/', in_browser=in_browser, column=isin,
                 title=title)
    ks_calculate(df['model'], df['actual'], plot=True, title=title, plot_dir=plot_dir + 'ks_decile/',
                 out_file='all_ks', in_browser=in_browser)
    decile_plot(df['model'], df['actual'], title=title, plot_dir=plot_dir + 'ks_decile/',
                out_file='all_decile', in_browser=in_browser)
    out_dict = dict()
    out_dict['all'] = importance
    if slice_dict is not None:
        for k in slice_dict.keys():
            i = slice_dict[k]
            if incl_ks:
                ks_calculate(df.loc[i]['model'], df.loc[i]['actual'], plot=True,
                             title=title + ' ' + k, plot_dir=plot_dir + 'ks_decile/',
                             out_file=k + '_ks', in_browser=in_browser)
            decile_plot(df.loc[i]['model'], df.loc[i]['actual'], title=title + ' ' + k, plot_dir=plot_dir + 'ks_decile/',
                        out_file=k + '_decile', in_browser=in_browser)
            out_dir = plot_dir + 'marginal/' + k + '/'
            os.makedirs(out_dir, exist_ok=True)
            importance = tfu.marginal(mod, features, features, df, out_dir, in_browser=in_browser, column=isin,
                         title=title)
            out_dict[k] = importance
    return out_dict