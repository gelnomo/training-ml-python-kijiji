import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class GeneralGraphs:
    def __init__(self):
        self.path = "reports/figures"
        self._title = None
        self._x_rotation = None
        self._x_fontsize = None
        self._bar_rotation = None
        self._bar_fontsize = None
        self._column_huge_name = None
        self._legend_loc = "upper left"

    def __del__(self):
        plt.close("all")

    @property
    def legend_loc(self):
        return self._legend_loc

    @legend_loc.setter
    def legend_loc(self, new_legend_loc):
        self._legend_loc = new_legend_loc

    @property
    def column_huge_name(self):
        return self._column_huge_name

    @column_huge_name.setter
    def column_huge_name(self, new_column_huge_name):
        self._column_huge_name = new_column_huge_name

    @property
    def bar_fontsize(self):
        return self._bar_fontsize

    @bar_fontsize.setter
    def bar_fontsize(self, new_bar_fontsize):
        self._bar_fontsize = new_bar_fontsize

    @property
    def bar_rotation(self):
        return self._bar_rotation

    @bar_rotation.setter
    def bar_rotation(self, new_bar_rotation):
        self._bar_rotation = new_bar_rotation

    @property
    def x_fontsize(self):
        return self._x_fontsize

    @x_fontsize.setter
    def x_fontsize(self, new_x_fontsize):
        self._x_fontsize = new_x_fontsize

    @property
    def x_rotation(self):
        return self._x_rotation

    @x_rotation.setter
    def x_rotation(self, new_x_rotation):
        self._x_rotation = new_x_rotation

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, new_title):
        self._title = new_title

    def count_bar_plot(self, column_df, column_x_name, column_y_name, order=None):
        plt.close("all")
        fig, ax = plt.subplots(figsize=(20, 9))
        sns.countplot(x=column_df, ax=ax, order=order)
        [
            ax.bar_label(cont, padding=10, fontsize=15, rotation=self.bar_rotation)
            for cont in ax.containers
        ]
        plt.xlabel(column_x_name)
        plt.ylabel(column_y_name)
        plt.xticks(rotation=self.x_rotation, fontsize=self.x_fontsize)
        plt.title(self.title, fontsize=15).set_position([0.5, 1.05])
        plt.savefig(
            f"{self.path}/count_bar_plot_{self.title.lower().replace(' ', '_')}.png",
            bbox_inches="tight",
        )

    def bar_plot(
        self,
        column_x,
        column_y,
        column_x_name,
        column_y_name,
        column_hue=None,
        order_x=None,
    ):
        plt.close("all")
        fig, ax = plt.subplots(figsize=(20, 9))
        sns.barplot(
            x=column_x, y=column_y, hue=column_hue, order=order_x, ax=ax, errorbar=None
        )
        [
            ax.bar_label(
                cont,
                padding=10,
                fmt="%d",
                fontsize=self.bar_fontsize,
                rotation=self.bar_rotation,
            )
            for cont in ax.containers
        ]
        plt.xlabel(column_x_name)
        plt.ylabel(column_y_name)
        if column_hue is not None and self.column_huge_name is not None:
            plt.legend(title=self.column_huge_name, loc=self.legend_loc)
        plt.title(self.title, fontsize=15).set_position([0.5, 1.05])
        plt.xticks(rotation=self.x_rotation, fontsize=15)
        plt.savefig(
            f"{self.path}/bar_plot_{self.title.lower().replace(' ', '_')}.png",
            bbox_inches="tight",
        )

    def line_plot(self, column_x, column_y, column_hue, column_x_name, column_y_name):
        plt.close("all")
        fig, ax = plt.subplots(figsize=(20, 9))
        sns.lineplot(
            x=column_x,
            y=column_y,
            hue=column_hue,
            palette=sns.color_palette("crest", as_cmap=True),
            estimator=None,
            lw=1,
            markers=True,
            ax=ax,
        )
        plt.xlabel(column_x_name)
        plt.ylabel(column_y_name)
        plt.title(self.title, fontsize=15).set_position([0.5, 1.05])

        for x, y in zip(column_x, column_y):
            plt.text(x=x, y=y - 150, s="{:.0f}".format(y), color="purple")

        if column_hue is not None and self.column_huge_name is not None:
            plt.legend(title=self.column_huge_name, loc=self.legend_loc)

        plt.rcParams["figure.autolayout"] = True
        plt.xticks(np.arange(1, 13))
        plt.savefig(
            f"{self.path}/line_plot_{self.title.lower().replace(' ', '_')}.png",
            bbox_inches="tight",
        )

    def count_pie_plot(self, data, column_y, explode, colors, sub_title):
        plt.close("all")
        fig, ax = plt.subplots()
        data.plot.pie(
            y=column_y,
            figsize=(5, 5),
            explode=explode,
            labels=None,
            autopct="%.2f%%",
            pctdistance=0.85,
            colors=colors,
            startangle=90,
            ax=ax,
        )

        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        plt.title(self.title, fontsize=15).set_position([0.5, 1.05])
        ax.text(
            0.0,
            0.0,
            sub_title,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=24,
        )
        ax.axis("equal")
        plt.tight_layout()
        plt.savefig(
            f"{self.path}/count_pie_plot_{self.title.lower().replace(' ', '_')}.png",
            bbox_inches="tight",
        )

    def distribution_kdeplot(self, data_filter, column_x_name, column_y_name):
        plt.close("all")
        sns.kdeplot(data=data_filter, color="Red", fill=True)
        plt.xlabel(column_x_name)
        plt.ylabel(column_y_name)
        plt.title(self.title, fontsize=15).set_position([0.5, 1.05])
        plt.savefig(
            f"{self.path}/distribution_kdeplot_{self.title.lower().replace(' ', '_')}.png",
            bbox_inches="tight",
        )

    def heat_map_corr(self, data):
        plt.close("all")
        plt.figure(figsize=(15, 8))
        plt.title(self.title, y=1.05, size=15)
        sns.heatmap(
            data,
            square=True,
            cmap="RdYlBu",
            linecolor="white",
            annot=True,
        )
        plt.tight_layout()
        plt.savefig(
            f"{self.path}/heat_map_corr_{self.title.lower().replace(' ', '_')}.png",
            bbox_inches="tight",
        )

    def count_scatter(self, column_x, column_y, column_x_name, column_y_name):
        plt.close("all")
        plt.figure(figsize=(10, 6))
        plt.scatter(column_x, column_y, c="m")
        plt.xlabel(column_x_name)
        plt.ylabel(column_y_name)
        plt.title(self.title)
        plt.savefig(
            f"{self.path}/count_scatter_{self.title.lower().replace(' ', '_')}.png",
            bbox_inches="tight",
        )

    def count_lmplot(self, data, column_x, column_y, column_x_name, column_y_name):
        plt.close("all")
        sns.lmplot(
            x=column_x,
            y=column_y,
            hue=self.column_huge_name,
            data=data,
            height=6,
            aspect=1.5,
        )
        plt.title(self.title)
        plt.xlabel(column_x_name)
        plt.ylabel(column_y_name)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=6)
        plt.savefig(
            f"{self.path}/count_lmplot_{self.title.lower().replace(' ', '_')}.png",
            bbox_inches="tight",
        )

    def heatmap_nulls(self, data):
        plt.close("all")
        ax = plt.axes()
        ax.xaxis.tick_top()
        sns.heatmap(data.isnull(), cbar=False, ax=ax)
        plt.title(self.title)
        plt.xlabel("Column")
        plt.ylabel("Quantity")
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=6)
        plt.savefig(
            f"{self.path}/heatmap_nulls_{self.title.lower().replace(' ', '_')}.png",
            bbox_inches="tight",
        )

    def facet_grid(self, data, col, x):
        g = sns.FacetGrid(data, col=col, height=3, aspect=0.5, col_wrap=5)
        g.map(sns.kdeplot, x)
        plt.title(self.title)
        plt.savefig(
            f"{self.path}/facet_grid_{self.title.lower().replace(' ', '_')}.png",
            bbox_inches="tight",
        )

    def comparison_plot(self, data, x, y_real, y_predicted):
        ax = data.plot(x=x, y=y_real)
        plt.xlabel(x)
        plt.ylabel(f"mean {y_real}")
        plt.title(self.title, fontsize=15).set_position([0.5, 1.05])
        data.plot(x=x, y=y_predicted, color="Red", ax=ax)
        plt.savefig(
            f"{self.path}/comparison_plot_{self.title.lower().replace(' ', '_')}.png",
            bbox_inches="tight",
        )
