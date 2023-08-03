import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_accuracy_scores(algorithms, datasets, scores):
    num_algorithms = len(algorithms)
    num_datasets = len(datasets)
    bar_width = 0.3  # Width of the bars
    opacity = 0.8  # Opacity of the bars

    index = np.arange(num_algorithms)
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = sns.color_palette('husl', num_datasets)  # Generate a color palette

    for i in range(num_datasets):
        ax.bar(index + i * bar_width, scores[i, :], bar_width, alpha=opacity, label=datasets[i])

    # Set seaborn style
    sns.set_style('white')

    ax.set_xlabel('users')
    ax.set_ylabel('Accuracy Score')
    ax.set_title('Accuracy Scores von ML-Algorithmen auf verschiedenen Datensätzen')
    ax.set_xticks(index + bar_width * (num_datasets - 1) / 2)
    ax.set_xticklabels(algorithms)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 0))

    # Begrenzen der y-Achsenansicht auf den Bereich zwischen 0.6 und 0.9
    plt.ylim(0, 1.0)

    plt.tight_layout()
    plt.show()

users = ['HI_user', 'MI_user', 'LI_user']
algorithms = ['RandomForest', 'SVM']
accuracy_scores = np.array([[0.672, 0.636, 0.89], [0.582, 0.709, 0.89]])

plot_accuracy_scores(users, algorithms, accuracy_scores)