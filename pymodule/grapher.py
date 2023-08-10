
import seaborn as sns
import matplotlib.pyplot as plt


def graph(df, cols, kind, figsize, title, xlabel, ylabel, lw):
    """
    Generate a graph based on the input data.

    Params:
        df (DataFrame): The DataFrame containing the data.
        cols (List[str]): List of column names to plot.
        kind (matplotlib.pyplot func): The type of plot to create (e.g., 'line', 'bar', 'scatter').
        figsize (Tuple[float, float]): The size of the figure (width, height) in inches.
        title (str): The title of the graph.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        lw (float): The linewidth of the plot.

    Returns:
        None: Displays the generated graph.
    """
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=10)
    plt.xlabel(xlabel, fontsize=8)
    plt.ylabel(ylabel, fontsize=8)
    for col in cols:
        kind(df.index, df[col], label=col, lw=lw)
    plt.legend(bbox_to_anchor=(1, 1.1))
    sns.despine()
    plt.show()

def pairplot(df, title):
    """
    Create a pairplot to visualize relationships between variable distributions in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        title (str): The title for the pairplot.

    Returns:
        None: Displays the generated pairplot.
    """
    g = sns.PairGrid(df)  
    g.map_lower(sns.scatterplot)  
    g.map_diag(sns.histplot)  
    g.map_upper(lambda *args, **kwargs: plt.axis("off"))
    g.fig.suptitle(title)
    plt.show()
        