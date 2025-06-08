import pandas as pd
import matplotlib.pyplot as plt


def scatter_chart(data: pd.DataFrame, centroids, fig_name: str):
        plt.scatter(data['x'], data['y'], c=data['z'], cmap='viridis')
        marker=['o', 'x', '^', 'v', '*']
        colors=['red', 'blue', 'green']
        for i in range(len(centroids)):
                plt.scatter(*centroids[i], 
                        marker=marker[i], s=100, c=colors[i], label='Centroids')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Scatter Plot Colored by Cluster')
        plt.colorbar(label='Cluster')  
        plt.savefig(fig_name)