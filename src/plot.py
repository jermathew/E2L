import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def compare_boxplots(data_list, labels, use_plotly, figsize=(16,8)):
    if use_plotly:
        fig = go.Figure()
        for i, data in enumerate(data_list):
            fig.add_trace(go.Box(y=data, name=labels[i]))
        return fig.show()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(data_list, labels=labels)
        plt.show()
        
        
def compare_lineplots(x_data, y_data_list, labels, x_label, y_label, use_plotly, figsize=(12,8)):
    if use_plotly:
        fig = go.Figure()
        for i, y_data in enumerate(y_data_list):
            fig.add_trace(go.Scatter(x=x_data, 
                                     y=y_data, 
                                     mode='markers+lines', 
                                     name=labels[i]))
        fig.update_layout(xaxis_title=x_label,yaxis_title=y_label)
        fig.show()
    
    else:
        fig, ax = plt.subplots(figsize=figsize)
        for i, y_data in enumerate(y_data_list):
            ax.plot(x_data, y_data, label=labels[i])
        plt.legend(loc="lower right")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        fig.show()


def show_barplot(data_x, data_y, x_label, y_label, use_plotly, figsize=(16,8)):
    sorted_data = sorted(zip(data_x, data_y), key=lambda x: x[1])
    sorted_x = [entry[0] for entry in sorted_data]
    sorted_y = [entry[1] for entry in sorted_data]
    if use_plotly:
        fig = go.Figure([go.Bar(x=sorted_x, y=sorted_y)])
        fig.update_xaxes(title_text=x_label, type='category')
        fig.update_yaxes(title_text=y_label)
        fig.show()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(sorted_x, sorted_y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=90)
        fig.show()
        
    