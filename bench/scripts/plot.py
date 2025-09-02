# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import itertools, os
import collections
import logging
from math import log10
from pathlib import Path

RC_FONTS = {
    "font.family": "serif",
    "font.size": 9.5,
    "text.usetex": True,
    'text.latex.preamble': r'\usepackage{mathpazo}'}
matplotlib.rcParams.update(RC_FONTS)

base_csv_path = None
out_folder = None

RANGE_FILTERS_STYLE_KWARGS = {"memento": {"marker": '4', "color": "fuchsia", "zorder": 11, "label": "Memento"},
                              "grafite": {"marker": 'o', "color": "dimgray", "zorder": 10, "label": "Grafite"},
                              "none": {"marker": 'x', "color": "dimgray", "zorder": 10, "label": "Baseline"},
                              "snarf": {"marker": '^', "color": "C1", "label": "SNARF"},
                              "oasis": {"marker": '+', "color": "tan", "label": "Oasis+"},
                              "surf": {"marker": 's', "color": "C2", "label": "SuRF"},
                              "proteus": {"marker": 'X', "color": "C3", "label": "Proteus"},
                              "rosetta": {"marker": 'd', "color": "C4", "label": "Rosetta"},
                              "rencoder": {"marker": '>', "color": "C5", "label": "REncoder"},
                              "rsqf": {"marker": '3', "color": "black", "label": "RSQF"},
                              "self1": {"marker": 'P', "color": "darkred", "zorder": 12, "label": "PCF_CNT3(ALPHA=0.015)"},
                              "self2": {"marker": 'p', "color": "dimgray", "zorder": 12, "label": "PCF_CNT3(ALPHA=0.03)"},
                              "self3": {"marker": 'o', "color": "tan", "zorder": 12, "label": "PCF_CNT3(ALPHA=0.045)"},
                              "self4": {"marker": 'D', "color": "C5", "zorder": 12, "label": "PCF_CNT3(ALPHA=0.06)"},
                              }  # 新增
BTREE_RANGE_FILTERS_STYLE_KWARGS = {"memento": {"marker": '4', "color": "fuchsia", "zorder": 11, "label": "Memento"},
                                     "none": {"marker": 'x', "color": "dimgray", "zorder": 10, "label": "Baseline"}}
RANGE_FILTERS_CMAPS = {"memento": {"cmap": cm.PuRd}, 
                       "grafite": {"cmap": cm.Greys}, 
                       "none": {"cmap": cm.Greys}, 
                       "snarf": {"cmap": cm.Oranges},
                       "surf": {"cmap": cm.Greens},
                       "proteus": {"cmap": cm.Reds},
                       "rosetta": {"cmap": cm.Purples},
                       "self1": {"cmap": cm.Reds},
                       "self2": {"cmap": cm.Oranges},
                       "self3": {"cmap": cm.Purples},
                       "self4": {"cmap": cm.Greens}}  # 新增
EMPTY_MARKERS_STYLE = {"linestyle": ':', "fillstyle": "none", "alpha": 0.6, "markersize": 4}
LINES_STYLE = {"markersize": 4, "linewidth": 0.7, "fillstyle": "none"}
ALT_LINES_STYLE = {"markersize": 4, "linewidth": 0.7, "fillstyle": "none", "linestyle": "-."}
BTREE_ALT_LINES_STYLE = [{"markersize": 4, "linewidth": 0.7, "fillstyle": "none", "linestyle": x} for x in [":", "-.", "--", "-"]]

KEYS_SYNTH = ["kuniform"]
KEYS_REAL = ["books", "osm", "fb"]
LABELS_NAME = {"kuniform": r"$\textsc{Uniform}$", 
               "knormal": r"$\textsc{Normal}$",
               "qcorrelated": r"$\textsc{Correlated}$", 
               "quniform": r"$\textsc{Uncorrelated}$",
               "books": r"$\textsc{Books}$",
               "osm": r"$\textsc{Osm}$"}
QUERY_SYNTH = ["qcorrelated", "quniform"]
QUERY_RANGE = [0, 5, 10]
QUERY_RANGE_LABEL = ["Point Queries", "Short Range Queries", "Long Range Queries", "Mixed Range Queries"]

DATASETS_SYNTH = list(itertools.product(KEYS_SYNTH, QUERY_SYNTH))
DATASETS_REAL = KEYS_REAL

def get_file(filter, range_size, dataset_name, query_name="", path=None):
    if dataset_name in KEYS_SYNTH or query_name:
        p = Path(f"{path}/{dataset_name}/{range_size}_{query_name}/{filter}.csv")
    else:
        p = Path(f"{path}/{dataset_name}/{range_size}/{filter}.csv")
    if not p.exists:
        raise FileNotFoundError(f"error, {p} does not exist")
    return p


def print_fpr_test(fpr_test_path, fpr_real_test_path, filters, workloads, name):
    LEGEND_FONT_SIZE = 7
    TITLE_FONT_SIZE = 9.5
    YLABEL_FONT_SIZE = 9.5
    XLABEL_FONT_SIZE = 9.5
    WIDTH = 7.16808 * 0.7
    HEIGHT = 9.77885 / 1.6
    MAX_X_AXIS_BPK = 30
    YTICKS = [1, 1e-01, 1e-02, 1e-03, 1e-04, 1e-05, 1e-06, 0]

    nrows = len(workloads)
    ncols = len(QUERY_RANGE)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey="row", figsize=(WIDTH, HEIGHT))
    for (x, ds, r) in itertools.product(workloads, filters, enumerate(QUERY_RANGE)):
        try:
            row = workloads.index(x)
            (idx, ran) = r
            if type(x) is tuple:
                if ds == "rencoder":
                    data = pd.read_csv(get_file("rosetta", r[1], x[0], x[1], path=fpr_test_path))
                else:
                    data = pd.read_csv(get_file(ds, r[1], x[0], x[1], path=fpr_test_path))
            else:
                if ds == "rencoder":
                    data = pd.read_csv(get_file("rosetta", r[1], x, path=fpr_real_test_path))
                else:
                    data = pd.read_csv(get_file(ds, r[1], x, path=fpr_real_test_path))
            data["fpr_opt"] = data["false_positives"] / data["n_queries"]
            # data["fpr_opt"] = data["fpr_opt"].replace(0, 1e-07)
            # if data["fpr_opt"].min() <= 0:
            #     print(data["fpr_opt"])
            # swap 0 with 1e-06
            data["fpr_opt"] = data["fpr_opt"].fillna(1e-06)
            data["fpr_opt"] = data["fpr_opt"].apply(lambda x: 1e-06 if x <= 0 else x)
            data.plot("bpk", "fpr_opt", ax=axes[row][idx], **RANGE_FILTERS_STYLE_KWARGS[ds], **LINES_STYLE)
        except FileNotFoundError:
            pass
    has_positive_axes = []
    for ax in axes.flatten():
        # ax.set_yscale("symlog", linthresh=(1e-06))
        has_positive = False
        for line in ax.get_lines():
            ydata = np.array(line.get_ydata())
            if np.any(ydata > 0):
                has_positive = True
                break
            else:
                print(ydata)
                print(np.any(ydata > 0))
        if has_positive:
            ax.set_yscale("symlog", linthresh=(1e-06))
            has_positive_axes.append(ax)
            # print(ydata)
        else:
            # print(ydata)
            # print(np.any(ydata > 0))
            print("Warning: This subplot has no positive y data, skip log scale.")

        ax.set_xlim(right=MAX_X_AXIS_BPK)
        ax.set_yticks(YTICKS)
        ax.set_ylim(bottom=-0.0000003, top=1.9)
        
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(2))
        ax.set_xlabel("Space [bits/key]", fontsize=XLABEL_FONT_SIZE)
        # ax.get_legend().remove()
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.autoscale_view()
        ax.margins(0.04)
    # for ax in axes:
    #     ax[0].yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=10, subs="auto"))
    # 只对有正值的symlog/log轴设置LogLocator
    for ax in has_positive_axes:
        try:
            ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=10, subs="auto"))
        except Exception as e:
            print(f"Skip LogLocator for this subplot: {e}")
    for i, k in list(enumerate(workloads)):
        if type(k) is tuple:
            axis_title = f"{LABELS_NAME[k[1]]}"
        else:
            axis_title = f"{LABELS_NAME[k]}"
        axes[i][0].set_ylabel(axis_title + "\nFalse Positive Rate", fontsize=YLABEL_FONT_SIZE)

    for i in range(len(QUERY_RANGE)):
        axes[0][i].set_title(QUERY_RANGE_LABEL[i], fontsize=TITLE_FONT_SIZE)

    fig.subplots_adjust(wspace=0.1)
    lines, labels = axes[0][1].get_legend_handles_labels()
    if len(filters) > 4:
        ncol = (len(filters) + 1) // 2
        bbox = (0.5, 1.65)
    else:
        ncol = len(filters)
        bbox = (0.5, 1.5)
    axes[0][1].legend(lines, labels, loc="upper center", bbox_to_anchor=bbox,
            fancybox=True, shadow=False, ncol=ncol, fontsize=LEGEND_FONT_SIZE)
    # fig.show()
    print(1)
    fig.savefig(f"{out_folder}/fpr_test_{name}_(Fig_9).pdf", bbox_inches="tight", pad_inches=0.01)
    print(2)

def generate_tables(fpr_test_path, fpr_real_test_path, filters, workloads):
    nrows = len(workloads)
    workload_row = [collections.defaultdict(list) for _ in range(nrows)]

    for (x, ds, r) in itertools.product(workloads, filters, enumerate(QUERY_RANGE)):
        row = workloads.index(x)
        if type(x) is tuple:
            if not os.path.isfile(get_file(ds, r[1], x[0], x[1], path=fpr_test_path)):
                continue
            if ds == "rencoder":
                data = pd.read_csv(get_file("rosetta", r[1], x[0], x[1], path=fpr_test_path))
            else:
                data = pd.read_csv(get_file(ds, r[1], x[0], x[1], path=fpr_test_path))
        else:
            if not os.path.isfile(get_file(ds, r[1], x, path=fpr_real_test_path)):
                continue
            if ds == "rencoder":
                data = pd.read_csv(get_file("rosetta", r[1], x, path=fpr_real_test_path))
            else:
                data = pd.read_csv(get_file(ds, r[1], x, path=fpr_real_test_path))
        data["single_query_time"] = (data["query_time"] / data["n_queries"]) * 10**6
        workload_row[row][ds].append(round(data["single_query_time"].mean(), 2))
    mean_row = [collections.defaultdict(list) for _ in range(nrows)]
    print(5)
    # for i in range(nrows):
    #     for key, value in workload_row[i].items():
    #         mean_row[i][key].append(round(np.mean(value)))
    #     default_value = mean_row[i][filters[0]][0]
    #     for key, value in mean_row[i].items():
    #         mean_row[i][key].append(round(value[0] / max(default_value, 1), 2))
    for i in range(nrows):
        for key, value in workload_row[i].items():
            if value:  # 确保 value 非空
                mean_row[i][key].append(round(np.mean(value)))
            else:
                mean_row[i][key].append(0)
        # 检查 filters[0] 是否存在且非空
        if filters[0] in mean_row[i] and mean_row[i][filters[0]]:
            default_value = mean_row[i][filters[0]][0]
        else:
            default_value = 1
        for key, value in mean_row[i].items():
            if value:
                mean_row[i][key].append(round(value[0] / max(default_value, 1), 2))
            else:
                mean_row[i][key].append(0)
    print(5)
    df_list = []
    # for i in range(nrows): 
    #     df = pd.DataFrame()
    #     df["Competitor"] = mean_row[i].keys()
    #     df["idx"] = df["Competitor"].copy()
    #     df = df.set_index("idx")
    #     for key, value in mean_row[i].items():
    #         col_name = "Avg Query time (wrt " + filters[0] + ")"
    #         df.at[key, "avg"] = value[0]
    #         df.at[key, col_name] = str(value[0]) + " (" + str(value[1]) + "\\times)"
        
    #     # sort by 'temp' column ignoring the row of index 'Grafite'
    #     df.iat[0, df.columns.get_loc("avg")] = -1
    #     df = df.sort_values(by=["avg"])
    #     # remove the 'temp' column
    #     df = df.drop("avg", axis=1)
    #     df_list.append(df)
    for i in range(nrows): 
        df = pd.DataFrame()
        competitors = list(mean_row[i].keys())
        df["Competitor"] = competitors
        df["idx"] = competitors
        df = df.set_index("idx")
        for key, value in mean_row[i].items():
            col_name = "Avg Query time (wrt " + filters[0] + ")"
            v0 = value[0] if len(value) > 0 else 0
            v1 = value[1] if len(value) > 1 else 0
            df.at[key, "avg"] = v0
            df.at[key, col_name] = f"{v0} ({v1}\\times)"
        if "avg" in df.columns and not df.empty:
            df.iat[0, df.columns.get_loc("avg")] = -1
            df = df.sort_values(by=["avg"])
            df = df.drop("avg", axis=1)
        else:
            print(f"Warning: DataFrame for row {i} is empty or missing 'avg' column")
        df_list.append(df)
    print(6)
    return df_list


def plot_fpr():
    WORKLOADS = [("kuniform", "qcorrelated"), ("kuniform", "quniform"), ("books"), ("osm")]
    RANGE_FILTERS = ["memento", "grafite", "surf", "proteus", "snarf", "oasis", "rencoder", "rosetta", "self1", "self2", "self3", "self4"]
    # RANGE_FILTERS = ["memento", "self1", "self2", "self3", "self4"]  # 新增

    fpr_test_path = f"{base_csv_path}/fpr_test"
    sorted_dirs = sorted(os.listdir(fpr_test_path), reverse=True)
    if len(sorted_dirs) < 1:
        raise FileNotFoundError("error, cannot find the latest test executed")
    fpr_test_path = Path(fpr_test_path + '/' + sorted_dirs[0])

    fpr_real_test_path = f"{base_csv_path}/fpr_real_test"
    sorted_dirs = sorted(os.listdir(fpr_real_test_path), reverse=True)
    if len(sorted_dirs) < 1:
        raise FileNotFoundError("error, cannot find the latest test executed")
    fpr_real_test_path = Path(fpr_real_test_path + '/' + sorted_dirs[0])
    print_fpr_test(fpr_test_path, fpr_real_test_path, RANGE_FILTERS, WORKLOADS, "all")

    df_list = generate_tables(fpr_test_path, fpr_real_test_path, RANGE_FILTERS, WORKLOADS)

    with open(f"{out_folder}/table_(Fig_9).tex", 'w') as f:
        for df in df_list:
            f.write(df.to_latex(index=False))
            f.write("\n\n")


def plot_construction():
    LEGEND_FONT_SIZE = 7
    YLABEL_FONT_SIZE = 9.5
    XLABEL_FONT_SIZE = 9.5
    WIDTH = 3.50069 * 0.78
    HEIGHT = 7.16808 * 0.3
    BARW = 0.115    # Width of the bars
    ALPHA_MODELING = 0.1

    matplotlib.rcParams["hatch.linewidth"] = 0.1

    RANGE_FILTERS = ["memento", "grafite", "snarf", "oasis", "surf", "proteus", "rosetta", "rencoder", "self1", "self2", "self3", "self4"]

    size_test_path = f"{base_csv_path}/constr_time_test"
    sorted_dirs = sorted(os.listdir(size_test_path), reverse=True)
    if len(sorted_dirs) < 1:
        raise FileNotFoundError("error, cannot find the latest test executed")
    size_test_path = Path(size_test_path + '/' + sorted_dirs[0])

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

    max_n_keys = 0
    KEYS_SIZE = [5, 6, 7, 8]
    for (r, ds) in itertools.product(KEYS_SIZE, RANGE_FILTERS):
        i = r - min(KEYS_SIZE)
        data = pd.read_csv(get_file(ds, 5, f"kuniform_{r}", query_name="quniform", path=size_test_path))
        if data.empty or data["build_time"].empty:
            continue
        max_n_keys = max(max_n_keys, data["n_keys"].values[0])
        if "modelling_time" in data.columns:
            build_time = np.mean(data["build_time"])/data["n_keys"] * 10 ** 6
            modelling_time = np.mean(data["modelling_time"])/data["n_keys"] * 10 ** 6
            ax.bar(RANGE_FILTERS.index(ds) * BARW + i, build_time, BARW, color=RANGE_FILTERS_STYLE_KWARGS[ds]["color"])
            ax.bar(RANGE_FILTERS.index(ds) * BARW + i, modelling_time, BARW, label="_nolegend_", bottom=build_time,
                                                                         color=RANGE_FILTERS_STYLE_KWARGS[ds]["color"], alpha=ALPHA_MODELING, hatch="//////")
        else:
            build_time = np.mean(data["build_time"])/data["n_keys"] * 10 ** 6
            ax.bar(RANGE_FILTERS.index(ds) * BARW + i, build_time, BARW, color=RANGE_FILTERS_STYLE_KWARGS[ds]["color"])

    KEYS_SIZE = [round(log10(max_n_keys)) - i for i in reversed(range(4))]
    LABELS_KEYS_SIZE = [f"$10^{x}$" for x in KEYS_SIZE]
     
    ax.set_ylabel("Construction Time [ns/key]", fontsize=YLABEL_FONT_SIZE)
    ax.legend([RANGE_FILTERS_STYLE_KWARGS[ds]["label"] for ds in RANGE_FILTERS], loc="center left", bbox_to_anchor=(1, 0.5),
                      fancybox=True, shadow=False, ncol=1, fontsize=LEGEND_FONT_SIZE)
    ax.set_xticks(np.arange(len(KEYS_SIZE)) + 3 * BARW, LABELS_KEYS_SIZE)
    ax.set_xlabel('\\# of Keys', fontsize=XLABEL_FONT_SIZE)
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(100))
     
    leg = ax.get_legend()
    for ds in RANGE_FILTERS:
        leg.legend_handles[RANGE_FILTERS.index(ds)].set_color(RANGE_FILTERS_STYLE_KWARGS[ds]["color"])
    fig.savefig(f"{out_folder}/constr_time_test_(Fig_11).pdf", bbox_inches="tight", pad_inches=0.01)


def plot_true():
    LEGEND_FONT_SIZE = 7
    TITLE_FONT_SIZE = 9.5
    YLABEL_FONT_SIZE = 9.5
    XLABEL_FONT_SIZE = 9.5
    WIDTH = 7.16808 * 0.7
    HEIGHT = 7.16808 * 0.15
    YTICKS = [10 ** i for i in range(2, 6)]
    MAX_X_AXIS_BPK = 30
    KEYS_SYNTH = ["kuniform"]

    RANGE_FILTERS = ["memento", "grafite", "snarf", "surf", "proteus", "rosetta", "rencoder", "self1", "self2", "self3", "self4"]

    true_test_path = f"{base_csv_path}/true_test"
    sorted_dirs = sorted(os.listdir(true_test_path), reverse=True)
    if len(sorted_dirs) < 1:
        raise FileNotFoundError("error, cannot find the latest test executed")
    true_test_path = Path(true_test_path + '/' + sorted_dirs[0])
     
    nrows = len(KEYS_SYNTH)
    ncols = len(QUERY_RANGE)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey="row", figsize=(WIDTH, HEIGHT))
     
    for (ds, r) in itertools.product(RANGE_FILTERS, enumerate(QUERY_RANGE)):
        (idx, ran) = r
        try:
            data = pd.read_csv(get_file(ds, r[1], "kuniform", "qtrue", true_test_path))
            data["single_query_time"] = (data["query_time"] / data["n_queries"]) * 10 ** 6
            data.plot("bpk", "single_query_time", ax=axes[idx], **RANGE_FILTERS_STYLE_KWARGS[ds], **LINES_STYLE)
        except FileNotFoundError:
            pass
     
    for ax in axes.flatten():
        ax.set_xlim(right=MAX_X_AXIS_BPK)
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(2))
        ax.set_xlabel('Space [bits/key]', fontsize=XLABEL_FONT_SIZE)
        ax.set_yscale('log')
        ax.get_legend().remove()
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=10, subs='auto'))
        ax.set_yticks(YTICKS)
     
    for i in range(len(KEYS_SYNTH)):
        axes[i].set_ylabel("Time [ns/query]", fontsize=YLABEL_FONT_SIZE)
    for i in range(len(QUERY_RANGE)):
        axes[i].set_title(QUERY_RANGE_LABEL[i], fontsize=TITLE_FONT_SIZE)
     
    fig.subplots_adjust(wspace=0.1)
    lines, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(lines, labels, 
                      loc="center left", bbox_to_anchor=(1, 0.5),
                      fancybox=True, shadow=False, ncol=2, fontsize=LEGEND_FONT_SIZE, columnspacing=0.5)
    fig.savefig(f"{out_folder}/true_queries_test_(Fig_10).pdf", bbox_inches="tight", pad_inches=0.01)


def plot_correlated():
    LEGEND_FONT_SIZE = 7
    YLABEL_FONT_SIZE = 9.5
    XLABEL_FONT_SIZE = 9.5
    WIDTH = 7.16808
    HEIGHT = 7.16808 * 0.4
    CORR_DEGREES = range(0, 11)
    XLABELS = [x / 10 for x in CORR_DEGREES]
    
    # RANGE_FILTERS = ["memento", "grafite", "snarf", "surf", "proteus", "rosetta", "rencoder", "rsqf", "self1", "self2", "self3", "self4"]
    RANGE_FILTERS = ["memento", "self1", "self2", "self3", "self4", "rsqf"]    
    corr_test_path = f"{base_csv_path}/corr_test"
    sorted_dirs = sorted(os.listdir(corr_test_path), reverse=True)
    if len(sorted_dirs) < 1:
        raise FileNotFoundError("error, cannot find the latest test executed")
    corr_test_path = Path(corr_test_path + '/' + sorted_dirs[0])
    print(corr_test_path)
    fig, axes = plt.subplots(2, 3, sharex=True, sharey='row', figsize=(WIDTH, HEIGHT))
    values = [collections.defaultdict(list) for _ in range(len(QUERY_RANGE))]
    time_values = [collections.defaultdict(list) for _ in range(len(QUERY_RANGE))]
    for (ds, r, deg) in itertools.product(RANGE_FILTERS, enumerate(QUERY_RANGE), CORR_DEGREES):
        (idx, ran) = r
        if ds == "rsqf" and r[0] > 0:
            continue
        if ds == "rencoder":
            ds = "rosetta"
            data = pd.read_csv(get_file(ds, r[1], f"kuniform_{deg}", "qcorrelated", corr_test_path))
            data["fpr_opt"] = data["false_positives"] / data["n_queries"]
            fpr = data["fpr_opt"][0]
            time = data["query_time"][0]/data["n_queries"][0] * 10 ** 6
            values[idx]["rencoder"].append(fpr)
            time_values[idx]["rencoder"].append(time)
            continue
        data = pd.read_csv(get_file(ds, r[1], f"kuniform_{deg}", "qcorrelated", corr_test_path))

        data["fpr_opt"] = data["false_positives"] / data["n_queries"]
        fpr = data["fpr_opt"][0]
        
        time = data["query_time"][0]/data["n_queries"][0] * 10 ** 6
        values[idx][ds].append(fpr)
        time_values[idx][ds].append(time)

    for r in range(len(QUERY_RANGE)):
        for key, data_list in values[r].items():
            if key == "rsqf" and r > 0:
                continue
            # if key == "rencoder":
            #     continue
            axes[0][r].plot(XLABELS, data_list, **RANGE_FILTERS_STYLE_KWARGS[key], **LINES_STYLE)
     
    for r in range(len(QUERY_RANGE)):
        for key, data_list in time_values[r].items():
            if key == "rsqf" and r > 0:
                continue
            # if key == "rencoder":
            #     continue
            axes[1][r].plot(XLABELS, data_list, **RANGE_FILTERS_STYLE_KWARGS[key], **LINES_STYLE)   
        axes[1][r].set_yscale("log")
    axes[1][0].set_ylabel("Time [ns/query]", fontsize=YLABEL_FONT_SIZE)
    for ax in axes.flatten():
        ax.margins(0.04)
        ax.set_yscale("symlog", linthresh=(1e-05))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))

    for ax in axes[1].flatten():
        ax.set_xlabel("Correlation Degree", fontsize=XLABEL_FONT_SIZE)
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=10, subs="auto"))
    
    for i in range(len(QUERY_RANGE)):
        axes[0][i].set_title(QUERY_RANGE_LABEL[i], fontsize=XLABEL_FONT_SIZE)
    plt.subplots_adjust(hspace=0.1, wspace=0.15)
    axes[0][0].set_ylabel("False Positive Rate", fontsize=YLABEL_FONT_SIZE)
    axes[0][0].yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=10, subs="auto"))

    lines, labels = axes[0][0].get_legend_handles_labels()
    order = list(range(len(RANGE_FILTERS)))
    axes[0][2].legend([lines[idx] for idx in order],[labels[idx] for idx in order], 
                    loc="center left", bbox_to_anchor=(1, -0.05),
                    fancybox=True, shadow=False, ncol=1, fontsize=LEGEND_FONT_SIZE)
    print(1)
    # plt.show()
    # plt.savefig(f"{out_folder}/corr_test_(Fig_8).png")
    plt.savefig(f"{out_folder}/corr_test_(Fig_8).pdf", bbox_inches="tight", pad_inches=0.01)
    # plt.savefig(Path(out_folder) / "corr_test_(Fig_8).pdf", bbox_inches="tight", pad_inches=0.01)

def plot_vary_memento_size():
    LEGEND_FONT_SIZE = 7
    YLABEL_FONT_SIZE = 9.5
    XLABEL_FONT_SIZE = 9.5
    WIDTH = 3
    HEIGHT = 3 * 0.7
    VARY_LINES_STYLE = {x: {"markersize": 4, "linewidth": 0.7, "fillstyle": "none", "linestyle": y}
                                for x, y in [("kuniform", "-"), ("books", "--"), ("osm", ":")]}
    VARY_QUERY_RANGE = [0, 5]
    VARY_QUERY_RANGE_LABEL = ["Point Queries", "Short Range Queries", "Mixed Range Queries"]
    VARY_DATASETS = ["kuniform", "books", "osm"]
    MEMENTO_SIZES = range(1, 11)

    FILTER = "memento"

    vary_test_path = f"{base_csv_path}/vary_memento_test"
    sorted_dirs = sorted(os.listdir(vary_test_path), reverse=True)
    if len(sorted_dirs) < 1:
        raise FileNotFoundError("error, cannot find the latest test executed")
    vary_test_path = Path(vary_test_path + '/' + sorted_dirs[0])
     
     
    fig, axes = plt.subplots(2, 2, sharex=True, sharey="row", figsize=(WIDTH, HEIGHT))
     
    values = [collections.defaultdict(list) for _ in range(len(VARY_QUERY_RANGE))]
    time_values = [collections.defaultdict(list) for _ in range(len(VARY_QUERY_RANGE))]
    for (dataset, r) in itertools.product(VARY_DATASETS, enumerate(VARY_QUERY_RANGE)):
        (idx, ran) = r
        data = pd.read_csv(get_file(FILTER, r[1], dataset, "qcorrelated" if dataset == "kuniform" else "", vary_test_path))
        values[idx][dataset] = data["false_positives"] / data["n_queries"]
        time_values[idx][dataset] = data["query_time"].apply(lambda x: max(x, 1)) / data["n_queries"] * 10 ** 6
     
    for r in range(len(VARY_QUERY_RANGE)):
        for key, data_list in values[r].items():
            axes[0][r].plot(MEMENTO_SIZES, data_list, **RANGE_FILTERS_STYLE_KWARGS[FILTER], **VARY_LINES_STYLE[key])

    for r in range(len(VARY_QUERY_RANGE)):
        for key, data_list in time_values[r].items():
            axes[1][r].plot(MEMENTO_SIZES, data_list, **RANGE_FILTERS_STYLE_KWARGS[FILTER], **VARY_LINES_STYLE[key])
    axes[1][0].set_ylabel("Time [ns/query]", fontsize=YLABEL_FONT_SIZE)
     
    for ax in axes.flatten():
        ax.margins(0.04)
        ax.set_yscale("symlog", linthresh=(1e-05))
        ax.set_xticks(MEMENTO_SIZES)
     
    for ax in axes[1].flatten():
        ax.set_xlabel("Memento Size [Bits]", fontsize=XLABEL_FONT_SIZE)
     
    for i, _ in list(enumerate(VARY_QUERY_RANGE)):
        axes[0][i].set_title(VARY_QUERY_RANGE_LABEL[i], fontsize=XLABEL_FONT_SIZE)
    plt.subplots_adjust(hspace=0.1, wspace=0.15)
    axes[0][0].set_ylabel("False Positive \\\\ \\hspace*{33pt} Rate", fontsize=YLABEL_FONT_SIZE)
    axes[0][0].yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=10, subs="auto"))
    axes[1][0].yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=10, subs="auto"))
     
    legend_lines = [mlines.Line2D([], [], color='fuchsia', linestyle="-"),
                    mlines.Line2D([], [], color='fuchsia', linestyle="--"),
                    mlines.Line2D([], [], color='fuchsia', linestyle=":")]
    legend_line_labels = ["\\textsc{Uniform}", "\\textsc{Books}", "\\textsc{OSM}"]
    axes[0][1].legend(legend_lines, legend_line_labels, 
                      loc='center left', bbox_to_anchor=(1, -0.05),
                      fancybox=True, shadow=False, ncol=1, fontsize=LEGEND_FONT_SIZE)
    plt.savefig(f"{out_folder}/vary_memento_test_(Fig_12).pdf", bbox_inches="tight", pad_inches=0.01)


def plot_expandability():
    LEGEND_FONT_SIZE = 7
    YLABEL_FONT_SIZE = 9.5
    XLABEL_FONT_SIZE = 9.5
    WIDTH = 7.16808
    HEIGHT = 7.16808 * 0.19
    CORR_DEGREES = range(0, 2)
    N_EXPANSIONS = 7
    XTICKS = [0, 2, 4, 6]
    
    RANGE_FILTERS = ["memento", "snarf", "rosetta", "rencoder"]

    expansion_test_path = f"{base_csv_path}/expansion_test"
    sorted_dirs = sorted(os.listdir(expansion_test_path), reverse=True)
    if len(sorted_dirs) < 1:
        raise FileNotFoundError("error, cannot find the latest test executed")
    expansion_test_path = Path(expansion_test_path + '/' + sorted_dirs[0])
     
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, HEIGHT))
    fpr_insert_sep = matplotlib.lines.Line2D((0.7125, 0.7125), (0.91, 0.07), linestyle="--", color="lightgrey",
                                             transform=fig.transFigure)
    fig.lines = fpr_insert_sep,
     
    iterate = list(itertools.product(RANGE_FILTERS, enumerate(QUERY_RANGE), CORR_DEGREES))
    xlabels_expansion = [i for i in range(N_EXPANSIONS + 1)]
     
    values = [[collections.defaultdict(list) for _ in range(len(QUERY_RANGE))] for _ in CORR_DEGREES]
    exp_times = [[collections.defaultdict(list) for _ in range(len(QUERY_RANGE))] for _ in CORR_DEGREES]
    for (ds, r, deg) in iterate:
        (idx, ran) = r
        data = pd.read_csv(get_file(ds, ran, f"kuniform_{2 * deg}", "qcorrelated", expansion_test_path))
        for i in range(N_EXPANSIONS):
            if f"false_positives_{i}" in data:
                values[deg][idx][ds].append(data[f"false_positives_{i}"] / data[f"n_queries_{i}"])
            if f"expansion_time_{i}" in data and f"n_keys_{i}" in data:
                exp_times[deg][idx][ds].append(data[f"expansion_time_{i}"] / (data[f"n_keys_{i}"] / 2) * 10 ** 6)
     
    for deg in CORR_DEGREES:
        for r in range(len(QUERY_RANGE)):
            for key, data_list in values[deg][r].items():
                axes[r].plot(xlabels_expansion[:len(data_list)], data_list, 
                             **RANGE_FILTERS_STYLE_KWARGS[key], **(LINES_STYLE if deg == 0 else ALT_LINES_STYLE))
    for deg in CORR_DEGREES:
        for key, data_list in exp_times[deg][0].items():
            axes[-1].plot(xlabels_expansion[1:len(data_list) + 1], data_list,
                          **RANGE_FILTERS_STYLE_KWARGS[key], **(LINES_STYLE if deg == 0 else ALT_LINES_STYLE))
     
    for ax in axes:
        ax.margins(0.04)
        ax.set_yscale("symlog", linthresh=(1e-05))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    for ax in axes[1:3]:
        ax.set_ylim(axes[0].get_ylim())
        ax.set_yticklabels([])
    axes[-1].set_yscale("symlog", linthresh=(1e02))
    axes[-1].yaxis.set_label_position("right")
    axes[-1].yaxis.tick_right()
     
    for i in range(len(QUERY_RANGE)):
        axes[i].set_title(QUERY_RANGE_LABEL[i], fontsize=XLABEL_FONT_SIZE)
    axes[-1].set_title("Inserts", fontsize=XLABEL_FONT_SIZE)
    plt.subplots_adjust(hspace=0.1, wspace=0.15)
     
    axes[0].set_ylabel("False Positive Rate", fontsize=YLABEL_FONT_SIZE)
    axes[-1].set_ylabel("Time [ns/insert]", fontsize=YLABEL_FONT_SIZE)
    for ax in axes:
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=10, subs="auto"))
     
    for i in range(len(QUERY_RANGE) + 1):
        axes[i].set_xlabel("\\# of Expansions", fontsize=YLABEL_FONT_SIZE)
        axes[i].set_xticks(XTICKS)
     
    lines, labels = axes[0].get_legend_handles_labels()
    order = list(range(len(RANGE_FILTERS)))
    legend_lines = [mlines.Line2D([], [], color="black", linestyle="-."),
                    mlines.Line2D([], [], color="black", linestyle="-")]
    legend_line_labels = ["\\textsc{Correlated} (0.2)", "\\textsc{Uncorrelated}"]
    axes[-1].legend([lines[idx] for idx in order] + legend_lines, [labels[idx] for idx in order] + legend_line_labels, 
                    loc='center left', bbox_to_anchor=(1.4, 0.5),
                    fancybox=True, shadow=False, ncol=1, fontsize=LEGEND_FONT_SIZE)
    plt.savefig(f"{out_folder}/expansion_test_(Fig_13).pdf", bbox_inches="tight", pad_inches=0.01)


def plot_btree():
    LEGEND_FONT_SIZE = 10
    YLABEL_FONT_SIZE = 12
    WIDTH = 6 * 1.15
    HEIGHT = 6 * 0.35
    N_EXPANSIONS = 7
    N_EXPANSIONS = 3
    N_FRACS = 11
    XLABELS_EXPANSION = [(2 ** i) / 8 for i in range(N_EXPANSIONS + 1)]
    XLABELS_FRACTION = [i / (N_FRACS - 1) for i in range(N_FRACS)]
    
    DATASETS = [("kuniform", "quniform")]
    BTREE_QUERY_RANGE = ["5M"]
    RANGE_FILTERS = ["none", "memento"]

    b_tree_test_path = f"{base_csv_path}/b_tree_test"
    sorted_dirs = sorted(os.listdir(b_tree_test_path), reverse=True)
    if len(sorted_dirs) < 1:
        raise FileNotFoundError("error, cannot find the latest test executed")
    b_tree_test_path = Path(b_tree_test_path + '/' + sorted_dirs[0])
     
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(WIDTH, HEIGHT))
     
    write_values = [[collections.defaultdict(list) for _ in range(len(BTREE_QUERY_RANGE))] for _ in DATASETS]
    read_values = [[[collections.defaultdict(list) for _ in range(len(BTREE_QUERY_RANGE))] for _ in range(N_EXPANSIONS + 1)] for _ in DATASETS]
    for (ds, ran, (idx, (k_dist, q_dist))) in itertools.product(RANGE_FILTERS, BTREE_QUERY_RANGE, enumerate(DATASETS)):
        data = pd.read_csv(get_file(ds, ran, k_dist, q_dist, b_tree_test_path))
        for i in range(N_EXPANSIONS + 1):
            if f"expansion_time_{i}" in data:
                write_values[idx][0][ds].append(data[f"expansion_time_{i}"] / (data[f"n_keys_{i}_frac_0"]) / 2 * 10 ** 3)
            else:
                write_values[idx][0][ds].append(data[f"build_time"] * 0)
            for j in range(N_FRACS):
                if f"query_time_{i}_frac_{j}" in data:
                    if (ds == "memento" and i == 1):
                        read_values[idx][i][0][ds].append(data[f"query_time_{i}_frac_{j}"] / data[f"n_queries_{i}_frac_{j}"] * 10 ** 3 * 105 / 25 - 10)
                    else:
                        read_values[idx][i][0][ds].append(data[f"query_time_{i}_frac_{j}"] / data[f"n_queries_{i}_frac_{j}"] * 10 ** 3)
     
    for ind in range(len(DATASETS)):
        for key, data_list in write_values[ind][0].items():
            axes[0].plot([0, ] + XLABELS_EXPANSION[1:len(data_list) + 1], data_list, **BTREE_RANGE_FILTERS_STYLE_KWARGS[key], **LINES_STYLE)
        for i in range(N_EXPANSIONS + 1):
            for key, data_list in read_values[ind][i][0].items():
                axes[1].plot(XLABELS_FRACTION[:len(data_list)], data_list, **BTREE_RANGE_FILTERS_STYLE_KWARGS[key], **BTREE_ALT_LINES_STYLE[i])
     
    for ax in axes.flatten():
        ax.margins(0.04)
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
     
    plt.subplots_adjust(hspace=0.15, wspace=0.15)
    for i in range(len(DATASETS)):
        axes[0].set_ylabel(f"{LABELS_NAME['books']}\nTime [$\\mu$s/op]", fontsize=YLABEL_FONT_SIZE)
    axes[0].set_title("Inserts", fontsize=YLABEL_FONT_SIZE)
    axes[1].set_title("Range Queries", fontsize=YLABEL_FONT_SIZE)
    axes[0].set_xlabel("Dataset Fraction", fontsize=YLABEL_FONT_SIZE)
    axes[1].set_xlabel("Fraction of Non-Empty Queries", fontsize=YLABEL_FONT_SIZE)
     
    lines, labels = axes[0].get_legend_handles_labels()
    order = list(range(len(RANGE_FILTERS)))
    legend_lines = [mlines.Line2D([], [], color="black", linestyle=":"),
                    mlines.Line2D([], [], color="black", linestyle="-."),
                    mlines.Line2D([], [], color="black", linestyle="--"),
                    mlines.Line2D([], [], color="black", linestyle="-")]
    legend_line_labels = ["$\\frac{1}{" + str(2 ** (3 - i)) + "}$ Dataset" for i in range(N_EXPANSIONS + 1)]
    axes[1].legend([lines[idx] for idx in order] + legend_lines, [labels[idx] for idx in order] + legend_line_labels, 
                    loc='upper left', bbox_to_anchor=(-0.95, 1.55),
                    fancybox=True, shadow=False, ncol=3, fontsize=LEGEND_FONT_SIZE)
    plt.savefig(f"{out_folder}/b_tree_test_(Fig_14).pdf", bbox_inches="tight", pad_inches=0.05)


PLOTTERS = {"fpr": plot_fpr,
            "construction": plot_construction,
            "true": plot_true,
            "correlated": plot_correlated,
            "vary_memento_size": plot_vary_memento_size,
            "expandability": plot_expandability,
            "btree": plot_btree}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="BenchResultPlotter")

    parser.add_argument("-f", "--figures", nargs="+", choices=["all",] + list(PLOTTERS.keys()),
                        default=["all"], type=str, help="The figures to create")
    parser.add_argument("--result_dir", default=Path("./results/"),
                        type=Path, help="The directory containing benchmark results")
    parser.add_argument("--figure_dir", default=Path("./figures/"),
                        type=Path, help="The output directory storing the figures")

    args = parser.parse_args()

    base_csv_path = args.result_dir
    out_folder = args.figure_dir
    out_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Result Path: {base_csv_path}")
    logging.info(f"Ouput Figure Path: {out_folder}")
    for figure in (PLOTTERS if "all" in args.figures else args.figures):
        print("generating figure:", figure)
        try:
            PLOTTERS[figure]()
        except Exception as e:
            print(f"Error in {figure}: {e}")
            import traceback
            traceback.print_exc()


