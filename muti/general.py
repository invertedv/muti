"""
utilities that are helpful in general model building

"""
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import scipy.stats as stats
import math

def get_unique_levels(feature, client, db, table):
    """
    Retrieves the unique levels of the column 'feature' in the table 'table' of database db.

    :param feature: column name in db.table to get unique levels
    :type str
    :param client: clickhouse client connector
    :type clickhouse_driver.Client
    :param db: database name
    :type str
    :param table: table name
    :type str
    :return: list of unique levels
    """
    qry = 'SELECT DISTINCT ' + feature + ' FROM ' + db + '.' + table + ' ORDER BY ' + feature
    uf = client.execute(qry)
    return [u[0] for u in uf]


def cont_hist(yh, y, title='2D Contour Histogram', xlab='Model Output', ylab='Y', subtitle=None, out_file=None):
    """
    Make a 2D contour histogram plot of y vs yh.
    The plot is produced in the browser and optionally written to a file.
    
    :param yh: Model outputs
    :param y: Target value
    :param title: Title for plot
    :param xlab: x-axis label
    :param ylab: y-axis label
    :param subtitle: optional subtitle
    :param out_file: optional file to write graph to
    :return:
    """
    
    fig = [go.Histogram2dContour(x=x, y=y)]
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
    figx.show()
    if out_file is not None:
        figx.write_image(out_file)

def ks_calculate(score_variable, binary_variable, plot=False, xlab='Score', ylab='CDF', title='KS Plot',
                 subtitle=None, out_file=None):
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
    :param xlab: label for the x-axis (score variable), optional
    :type xlab: str
    :param ylab: label for the y-axis (binary variable), optional
    :type ylab: str
    :param title: title for the plot, optional
    :type title: str
    :param subtitle: subtitle for the plot, optional (default=None)
    :type subtitle: str
    :param out_file file name for writing out the plot
    :type str
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
        fig = [go.Scatter(x=score0, y=u0, line=dict(color='black') )]
        fig += [go.Scatter(x=score1, y=u1, line=dict(color='black') )]
        
#        fig += [go.Scatter(x=sc, y=uu0, line=dict(color='red'))]
#        fig += [go.Scatter(x=sc, y=uu1, line=dict(color='red'))]
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
        figx.show()
    if out_file is not None:
        figx.write_image(out_file)

    return ks


def decile_plot(score_variable, binary_variable, xlab='Score', ylab='Actual', title='Decile Plot',
                plot_maximum=None, plot_minimum=None, confidence_level=0.95, correlation=0, subtitle=None,
                out_file=None):
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
    :param out_file file name for writing out the plot
    :type str
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
#    plt.plot(rxy, rxy, mscore, mbinary, 'ro')
    
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
    if subtitle is not None: title += '<br>' + subtitle
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
    MeansTitle = 'Actual ' + np.str(round(binary_variable.mean(), 3))
    MeansTitle = MeansTitle + '\nScore ' + np.str(round(score_variable.mean(), 3))
#    plt.annotate(MeansTitle, xy=[min_limit + 0.1 * rangexy, max_limit - 0.1 * rangexy])
    figx.add_annotation(x=min_limit + 0.1 * rangexy, y=max_limit - 0.1 * rangexy,
                        text=MeansTitle,
                        showarrow=False,
                        xshift=1, xref='paper')
    pio.renderers.default = 'browser'
    figx.show()
    if out_file is not None:
        figx.write_image(out_file)
    
    return
