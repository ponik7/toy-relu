import matplotlib.pyplot as plt
import torch

def get_fractional_dimensions(h):
    h_norms = torch.norm(h, dim=0)
    h_unit = h / h_norms
    projections = (h_unit.T @ h) ** 2
    numerators = h_norms ** 2
    denominators = projections.sum(dim=1)
    return (numerators / denominators).tolist()

def plot_vectors(vectors, T, color, prefix):
    vectors = vectors.detach().cpu() if torch.is_tensor(vectors) else vectors
    x, y = vectors[0], vectors[1]

    plt.figure(figsize=(10, 10))
    limit = max(abs(x).max(), abs(y).max())
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)

    plt.plot(x, y, 'o', color=color)
    for xi, yi in zip(x, y):
        plt.plot([0, xi], [0, yi], color=color)

    plt.savefig(f'assets/{prefix}_{T}.png')
    plt.show()
    plt.close()

def plot_model_features(model, T):
    W = model.w
    plot_vectors(W, T, color='blue', prefix='feats')
    return get_fractional_dimensions(W)

def plot_hidden_states(data, model, T):
    with torch.no_grad():
        h = model(data, return_hidden_state=True)
    plot_vectors(h, T, color='red', prefix='sample')
    return get_fractional_dimensions(h)
