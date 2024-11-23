import torch
import os


def save_model(model, winrate, folder="saved_models"):
    """
    Save the model with a given winrate.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    model_path = os.path.join(folder, f"model_{winrate:.2f}.pth")
    torch.save(model.state_dict(), model_path)
    return model_path


def load_model(model, path):
    """
    Load a model from a given path.
    """
    model.load_state_dict(torch.load(path))
    return model


def update_best_models(winrate, model_path, best_models, top_k=5):
    """
    Update the top K best models list based on winrate.
    """
    best_models.append((winrate, model_path))
    best_models = sorted(best_models, key=lambda x: x[0], reverse=True)[:top_k]

    # Remove files for models outside the top K
    for _, path in best_models[top_k:]:
        if os.path.exists(path):
            os.remove(path)

    return best_models
