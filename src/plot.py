import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def compare_boxplots(data_a, data_b, label_a, label_b, use_plotly, figsize=(16,8)):
    if use_plotly:
        fig = go.Figure()
        fig.add_trace(go.Box(y=data_a, name=label_a))
        fig.add_trace(go.Box(y=data_b, name=label_b))
        return fig.show()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot([data_a, data_b], labels=[label_a, label_b])
        plt.show()
        
        
def compare_lineplots(data_x, data_ya, data_yb, label_a, label_b, x_label, y_label, use_plotly, figsize=(12,8)):
    if use_plotly:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_x,
                                 y=data_ya,
                                 mode='markers+lines',
                                 name=label_a))
        fig.add_trace(go.Scatter(x=data_x,
                                 y=data_yb,
                                 mode='markers+lines',
                                 name=label_b))
        fig.update_layout(xaxis_title=x_label,yaxis_title=y_label)
        fig.show()
    
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(data_x, data_ya, label=label_a)
        ax.plot(data_x, data_yb, label=label_b)
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
        
    