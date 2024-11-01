import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score, auc
from os.path import splitext, join, exists
from sklearn.tree import export_graphviz
import seaborn as sns
import pandas as pd
from Tools.IOData import IOData
from Tools.ToolsModels import is_tf_model, is_rulefit_model, is_binary
from Common.Config.ConfigHolder import ATTR, FEATURE, MAX_SIZE_FEATURES, MAX_IMPORTANCES, CORR_CUTOFF, STD
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
import shap
from sklearn.inspection import plot_partial_dependence
from alibi.explainers import plot_ale
from glob import glob


class Graphics:
    """ Draws the ANN model. Only valid for TensorFlow models """
    def draw_model(self, model, file_out):
        if isinstance(model, tf.keras.Model):
            tf.keras.utils.plot_model(model,
                                      to_file='{}_model.png'.format(file_out),
                                      show_shapes=True,
                                      expand_nested=True,
                                      dpi=300)

    """ Plots the correlation between true and predicted values """
    def plot_correlation(self, y, yhat, file_out, xlabel='Predictions', ylabel='True values'):
        fig = plt.figure(clear=True)
        ax = plt.gca()
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', c='black')

        plt.scatter(yhat, y, alpha=0.6)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        self.save_fig(file_out)
        plt.close(fig)

    """
    Plots the evolution of loss and accuracy while training. 
    """
    def plot_metrics_evolution(self, epochs, metrics, file_out):
        steps = [i + 1 for i in range(epochs)]

        fig, ax = plt.subplots()
        for m in metrics.keys():
            ax.plot(steps, metrics[m], label=m)

        ax.set(xlabel='Epoch', title=f'Training Loss vs Metrics')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(file_out)
        plt.close()

    def _generate_graph_roc(self, x, y, auc_value, file_out, title):
        plt.clf()
        plt.cla()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(x, y, label='(AUC = {:.3f})'.format(auc_value))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(title)
        plt.legend(loc='best')
        self.save_fig(file_out)
        plt.close()

    def plot_roc_curve(self, model, xts, yts, ypr, file_out, cfg):
        """
            https://github.com/dataprofessor/code/blob/master/python/ROC_curve.ipynb
        """
        yppr = ypr

        if is_tf_model(model) and file_out.find("RNN") >= 0:
            xts = tf.expand_dims(xts, -1)
        elif not is_tf_model(model):
            # predict_proba() in sklearn produces returns two columns (N,K ) N number of datapoits, k number of classes
            yppr = model.predict_proba(xts)
            yppr = yppr[:, 1]

        self.plot_class(yppr, yts, file_out)

        classes = np.unique(yts)
        for clazz in classes:            
            if is_tf_model(model):
                yppr = model(xts).numpy()[:, clazz]

            fpr, tpr, thr = roc_curve(yts.ravel(), yppr.ravel(), pos_label=clazz)
            auc_value = auc(fpr, tpr)
            proba_out = splitext(file_out)[0] + "_proba_{}.png".format(clazz)
            self._generate_graph_roc(fpr, tpr, auc_value, proba_out, 'ROC curve prob class {}'.format(str(clazz)))

            if is_binary(cfg):
                ypr_class = tf.round(yppr).numpy().astype(int)
                fpr, tpr, thr = roc_curve(yts, ypr_class, pos_label=clazz)
                auc_value = roc_auc_score(yts, ypr_class)
                roc_out = splitext(file_out)[0] + "_{}.png".format(clazz)
                self._generate_graph_roc(fpr, tpr, auc_value, roc_out, 'ROC curve class {}'.format(str(clazz)))

    def plot_class(self, yppr, yts, file_out):
        plt.clf()
        plt.cla()
        lst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in yts:
            if i != 0 and i != 1:
                return

        for i in yppr:
            lst[int(i*9)] += 1
        x = np.arange(12)
        plt.xlabel('Range')
        plt.ylabel('Number of Samples')
        plt.title('Class Probability')
        proba_out = splitext(file_out)[0] + "_proba_class.png"
        lst.insert(0, 0)
        lst.insert(12, 0)
        sns.barplot(x, lst)
        #plt.yticks(np.arange(0, max(lst) * 1.10))
        plt.xticks(x, ('', '[0 - 0.09]', '[0.1 - 0.19]', '[0.2 - 0.29]', '[0.3 - 0.39]', '[0.4 - 0.49]', '[0.5 - 0.59]', '[0.6 - 0.69]', '[0.7 - 0.79]', '[0.8 - 0.89]', '[0.9 - 1]', ''))
        plt.tick_params(rotation=90)
        self.save_fig(proba_out)

    def graph_tree(self, model, id_list, claasses, prefix_out):
        """
              leaves_parallel--> draw all leaf nodes at the bottom of the tree.                            
          """
        dot_file = prefix_out + "_file.dot"
        png_file = prefix_out + "_file.png"
        _ = export_graphviz(model,
                            out_file=dot_file,
                            feature_names=id_list,
                            class_names=claasses,
                            rounded=True,
                            proportion=False,
                            precision=1,
                            leaves_parallel=False,
                            rotate=False)
        from subprocess import call
        call(['dot', '-Tpng', dot_file, '-o', png_file, '-Gdpi=600'])

    def save_fig(self, file):
        plt.tight_layout()
        plt.savefig(file, dpi=300)
        plt.close()

    def plot_attributions(self, df, title, out_file, sample_id=None, errors=None):
        if sample_id is not None:
            title += '. Sample #{}'.format(sample_id)

        # Plot the chart
        labels = df[FEATURE].to_numpy()
        weights = df[ATTR].to_numpy()
        colors = ['r' if v < 0 else 'tab:blue' for v in weights]
        colors = ['g' if 'Sum' in labels[i] else colors[i] for i, l in enumerate(labels)]

        plt.clf()
        plt.cla()
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title(title)
        plt.xlabel('Average Attribution')
        plt.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        #plt.barh(labels, np.abs(weights), color=colors, xerr=errors, error_kw=dict(lw=0.75, capsize=1))
        plt.barh(labels, np.abs(weights), color=colors)
        plt.gca().invert_yaxis()

        for i in range(len(weights)):
            v = weights[i]
            s = ' {:+.2e}'.format(v)
            plt.text(abs(v), i-0.05, s, fontsize=6, color=colors[i])

        self.save_fig(out_file)

    def graph_dataset(self, x, y, feature_list, filed_taget, prefix):
        df = pd.DataFrame(x, columns=feature_list)
        df.insert(len(df.columns), column=filed_taget, value=y)
        self.correlation_df_matrix(df, prefix + "/Df_correlation.png")

        if len(df.columns) <= MAX_SIZE_FEATURES:
            self.box_plot(df, prefix + "/Df_group_data.png")
            try:
                self.dispersion(df, filed_taget, prefix + "/Df_dispersion.png")
            except:
                pass

    def graphic_pie(self, df, file_out, title):
        try:

            plt.clf()
            plt.cla()

            pie = plt.pie(x=df[ATTR].abs(),
                          autopct='%1.1f%%',
                          startangle=90,
                          normalize=True,
                          textprops=dict(size=5),
                          pctdistance=0.8)
            plt.legend(pie[0],
                       df[FEATURE],
                       bbox_to_anchor=(1, 0.5),
                       loc="center right",
                       fontsize=5,
                       bbox_transform=plt.gcf().transFigure)
            plt.title(title)
            plt.savefig(file_out, dpi=300, bbox_inches="tight")
            plt.close()
        except:
            pass

    def graph_hist(self, df, file_out, title):
        plt.clf()
        plt.cla()

        if STD in df:
            plt.errorbar(df[FEATURE], df[ATTR], df[STD], linestyle='None', marker='o')
        else:
            df.plot(x=FEATURE, y=ATTR, alpha=0.5, kind='bar')

        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right', fontsize=6)
        plt.yticks(fontsize=6)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(file_out, dpi=300)
        plt.close()

    def correlation_df_matrix(self, df, file_out):
        corr = df.corr()
        annot = True

        if len(df.columns) > MAX_IMPORTANCES:
            # find the top N highest correlation values
            au_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool)).stack().dropna()
            order = au_corr.abs().sort_values(ascending=False)
            au_corr = au_corr[order.index][:MAX_IMPORTANCES]

            # make a new dataset only with columns involved
            columns = list(set([x[0] for x in au_corr.keys()] + [x[1] for x in au_corr.keys()]))
            df2 = df.filter(items=columns)

            # calculate the correlation matrix of the new temporary dataset
            corr = df2.corr()
            sns.set_context("paper", font_scale=0.4)
        else:
            plt.xticks(rotation=45)

        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr,
                    cmap="rainbow",
                    annot=annot,
                    square=True,
                    mask=mask,
                    linewidths=.5,
                    cbar_kws={"shrink": .5},
                    vmin=-1,
                    vmax=1,
                    fmt=".2f")

        self.save_fig(file_out)
        sns.set_context("notebook")

    def box_plot(self, df, file_out):
        #
        #   https://towardsdatascience.com/keras-101-a-simple-and-interpretable-neural-network-model-for-house-pricing-regression-31b1a77f05ae
        #
        total_items = len(df.columns)
        items_per_row = 3
        total_rows = math.ceil(total_items / items_per_row)
        fig = make_subplots(rows=total_rows, cols=items_per_row)
        cur_row = 1
        cur_col = 1
        for index, column in enumerate(df.columns):
            fig.add_trace(go.Box(y=df[column], name=column), row=cur_row, col=cur_col)

            if cur_col % items_per_row == 0:
                cur_col = 1
                cur_row = cur_row + 1
            else:
                cur_col = cur_col + 1

        fig.update_layout(height=1000, width=550, showlegend=False)
        with open(file_out, 'wb') as f:
            f.write(fig.to_image(format='png', engine='kaleido'))

    def dispersion(self, df, filed_taget, file_out):
        #
        #   https://towardsdatascience.com/keras-101-a-simple-and-interpretable-neural-network-model-for-house-pricing-regression-31b1a77f05ae
        #

        total_items = len(df.columns)
        items_per_row = 3
        total_rows = math.ceil(total_items / items_per_row)
        fig = make_subplots(rows=total_rows, cols=items_per_row, subplot_titles=df.columns)
        cur_row = 1
        cur_col = 1
        for index, column in enumerate(df.columns):
            fig.add_trace(go.Scattergl(x=df[column], y=df[filed_taget], mode="markers", marker=dict(size=3)),
                          row=cur_row,
                          col=cur_col)

            intercept = np.poly1d(np.polyfit(df[column], df[filed_taget], 1))(np.unique(df[column]))

            fig.add_trace(go.Scatter(x=np.unique(df[column]), y=intercept, line=dict(color='red', width=1)),
                          row=cur_row,
                          col=cur_col)

            if cur_col % items_per_row == 0:
                cur_col = 1
                cur_row = cur_row + 1
            else:
                cur_col = cur_col + 1

        fig.update_layout(height=1000, width=550, showlegend=False)
        with open(file_out, 'wb') as f:
            f.write(fig.to_image(format='png', engine='kaleido'))

    def plot_interpretability_times(self, dct_times, out_graph, name_model):
        plt.clf()
        plt.cla()

        methods = list(dct_times.keys())
        times = [x[0] for x in dct_times.values()]

        y_pos = np.arange(len(methods))
        fig, ax = plt.subplots()
        barlist = ax.barh(y_pos, times, align='center', height=0.3)

        dct_colors = {
            'Training': 'tab:red',
            'Load data': 'tab:orange',
            'Analysis': 'tab:green'
        }
        for i, b in enumerate(barlist):
            if methods[i] in dct_colors.keys():
                b.set_color(dct_colors[methods[i]])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        ax.set_xlabel('Execution time (seconds)')
        ax.set_title('Partial running times {}'.format(name_model))
        ax.set_xscale('log')
        self.save_fig(out_graph)

    def plot_lime_html(self, html, file_out):
        with open(file_out, 'w', encoding='utf-8') as f:
            f.write(html)

    def plot_shapley(self, xts, feature_names, shap_values, prefix):
        shap.summary_plot(shap_values, features=xts, feature_names=feature_names, show=False)

        plt.yticks(fontsize=8)  # make font smaller for a better reading
        plt.xticks(fontsize=8)
        plt.title('Global Interpretation using SHAP')
        self.save_fig(prefix + "_Shapley.png")

        plt.rcParams.update(plt.rcParamsDefault)  # restore default fontsize and parameters

    def plot_shapley_local(self, shapley, feature_names, file_out, sample_id):
        shap.plots.bar(shap.Explanation(values=shapley.values,
                                        base_values=shapley.base_values,
                                        data=shapley.data,
                                        feature_names=feature_names),
                       max_display=10,
                       show=False)

        plt.yticks(fontsize=6)  # make font smaller for a better reading
        plt.xticks(fontsize=6)
        plt.title('Local Interpretation using SHAP. Sample #{}'.format(sample_id))
        self.save_fig(file_out)

        plt.rcParams.update(plt.rcParamsDefault)  # restore default fontsize and parameters

    def plot_pdp_ice(self, model, xtr_df, feature, file_out, jobs=None, seed=0):
        display = plot_partial_dependence(model, xtr_df, [feature], kind='both', random_state=seed, n_jobs=jobs)
        for i in range(display.lines_.shape[1]):
            display.lines_[0, i, -1].set_color('gold')
            display.axes_[0, i].legend()

        self.save_fig(file_out)

    """ Plots the Accumulated Local Effects plot for a given feature """
    def plot_ale(self, explainer, feature, file_out):
        plot_ale(explainer, n_cols=1, sharey='row', features=[feature])
        self.save_fig(file_out)

    """ Plots the execution times of every phase of the pipeline """
    def plot_times(self, dir_name, prefix, name_model):
        dct_times = {}
        lst_files = glob('{}*_time.txt'.format(prefix)) + [join(dir_name, 'load_time.txt')]

        for file in lst_files:
            with open(file) as f:
                first_line = f.readline()
            method = first_line.split(":")[0]
            time = round(float(first_line.split(":")[1].strip()), 3)
            position = round(float(first_line.split(":")[2].strip()))

            dct_times[method] = (time, position)
        # sort entries by position: entry=(time, position)
        dct_times = dict(sorted(dct_times.items(), key=lambda x: x[1][1]))

        self.plot_interpretability_times(dct_times, prefix + "_times.png", name_model)

    """ Plots the time taken by code executed on GPU """
    def plot_gpu_usage(self, logger_fname, file_out):
        if not exists(logger_fname):
            return
        logger_df = pd.read_csv(logger_fname)
        if 'Timestamp (s)' not in logger_df.columns:
            return

        t = pd.to_datetime(logger_df['Timestamp (s)'], unit='s')
        cols = [col for col in logger_df.columns if 'time' not in col.lower() and 'temp' not in col.lower()]
        plt.figure(figsize=(15, 9))
        plt.plot(t, logger_df[cols])
        plt.legend(cols)
        plt.xlabel('Time')
        plt.ylabel('Utilisation (%)')
        self.save_fig(file_out)
        
    """ Plots KNN cluster distribution """
    def graph_knn_points(self, model, xtr, ytr, id_list, file_out):
        sns.scatterplot(x=xtr[:,0], y=ytr, palette=plt.cm.Paired, alpha=1.0, edgecolor="black")
        self.save_fig(file_out)

    """ Plot scopes rules """
    def plot_anchors(self, df, file_out):
        df['feature_wrapped'] = df['feature'].apply(lambda x: self.split_text(x, max_length=20))
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(df))
        ax.barh(y_pos, df['precision'], xerr=df['std'], align='center', color='skyblue', label='Precision', capsize=5)
        ax.barh(y_pos, df['coverage'], align='center', color='salmon', alpha=0.5, label='Coverage')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['feature_wrapped'], fontsize=6, rotation=30, rotation_mode='anchor')
        ax.invert_yaxis()
        ax.set_xlabel('Values')
        ax.set_title('Global Precision and Coverage of Anchors')
        ax.legend()
        plt.tight_layout()
        self.save_fig(file_out)

    def split_text(self, text, max_length=20):
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + word) <= max_length:
                current_line += (word + " ")
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)
        return "\n".join(lines)

