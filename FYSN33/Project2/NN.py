import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from eventGenerator import dphi


class FeatureBuilder:
    def __init__(self, events):
        self.events = events

    def build_feature(self, event):
        """
        for a given event e_i, construct x_i = (pt,...) 10 component vector
        """
        j = event.leading_jet()
        l = event.leading_lepton()

        if (j is None) or (l is None):
            return None
        
        # extract vector components from Event
        pt_j = j.pt
        eta_j = j.eta
        phi_j = j.phi

        pt_l = l.pt
        eta_l = l.eta
        phi_l = l.phi

        delta_phi = dphi(phi_j, phi_l)
        delta_eta = eta_j - eta_l
        delta_R = np.sqrt(delta_eta**2 + delta_phi**2)  # slides lect 2

        # build feature vector
        x = np.array([pt_j, 
                    eta_j,
                    phi_j, 
                    pt_l, 
                    eta_l, 
                    phi_l, 
                    delta_phi, 
                    delta_R,
                    pt_j/pt_l,
                    (pt_j-pt_l)/(pt_j+pt_l)])
        
        return x, int(j.truth)
        
    def build_dataset(self, with_labels=True):
        """
        builds feauture vectors x_i in X
        - each row = one event
        - y in {0,1}^N if truth labels
        """
        X, y = [], []

        for e in self.events:
            x, truth = self.build_feature(e)
            if x is None:
                continue

            X.append(x)

            if with_labels:
                y.append(truth)

        if with_labels:
            return np.array(X), np.array(y)
        else:
            return np.array(X)
        
    def standardize(self, Xtrain, Xval):
        """
        standardize all features so that
        - X.mean == 0
        - X.std == 1
        """

        mu = Xtrain.mean(axis=0)
        sigma = Xtrain.std(axis=0)
        sigma[sigma==0] = 1.0

        Xtrain_std = (Xtrain - mu) / sigma
        Xval_std = (Xval - mu) / sigma
        
        return Xtrain_std, Xval_std, mu, sigma
    
class JetData(torch.utils.data.Dataset):
    """
    convert dataset to torch
    - identical to class MoonData
    """
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class JetClassifier(nn.Module):
    """
    Defines model 10->32->16->1 model
    
    return: logits z in (-inf,inf), NOT probability
    - use 
    out = model(in)
    p = sigmoid(out)
    """
    def __init__(self, input_dim=10, h1=32, h2=16, p=0.1, print_model=False):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1) # logits (-inf,inf)
        )

        if print_model:
            print(self.model)
            print("Number of trainable parameters:", 
                  sum(p.numel() for p in self.model.parameters() if p.requires_grad))


    def forward(self, x):
        return self.model(x)
    
class Trainer:
    """
    Trainer class:
    initialized with a model, loss function, optimizer

    contains the main training (epoch) loop

    return: dict [training/validation loss; training/validation acc]
    """
    def __init__(self, model, lr=1e-3, patience=20):
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.patience = patience

    def train(self, train_loader, val_loader, epochs=200):

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):

            # -------- TRAIN --------
            self.model.train()
            running_loss = 0
            num_correct = 0
            total_samples = 0

            for X, y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.loss_fn(outputs, y.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                preds = (outputs.squeeze(1) > 0).float()
                num_correct += (preds == y).sum().item() # .item -> floats instead of tensors
                total_samples += y.size(0)

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)
            accuracy = num_correct / total_samples
            train_accuracies.append(accuracy)

            # -------- VALIDATE --------
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X, y in val_loader:
                    outputs = self.model(X)
                    loss = self.loss_fn(outputs, y.unsqueeze(1))
                    val_loss += loss.item()

                    preds = (outputs.squeeze(1) > 0).float()
                    val_correct += (preds == y).sum().item()
                    val_total += y.size(0)

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            val_accuracy = val_correct / val_total
            val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch+1:03d}\t | Train Loss: {train_loss:.4f}\t | Val Loss: {val_loss:.4f}\t | Train Acc: {accuracy:.3f}\t | Val Acc: {val_accuracy:.3f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping triggered.")
                break

        self.model.load_state_dict(best_state)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }

class NNselector:
    """
    NN selection
    - behaves like pass_cuts(event)
    - arb default threshold, input working point t*
    """
    def __init__(self, model, feature_builder, mu, sigma, threshold=0.5):
        self.model = model
        self.feature_builder = feature_builder
        self.mu = mu
        self.sigma = sigma
        self.threshold = threshold

    def __call__(self, event):
        x, _ = self.feature_builder.build_feature(event)

        if x is None:
            return False

        x = (x - self.mu) / self.sigma
        x = torch.tensor(x.reshape(1,-1), dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            score = torch.sigmoid(self.model(x)).item()

        return score >= self.threshold 



# ================
# Helpers
# ================

def split_by_id(events, frac=0.6, seed=42):
    """
    split data in train/val partition
    frac == frac training
    """
    np.random.seed(seed)

    event_ids = np.array([e.event_id for e in events])
    unique_ids = np.unique(event_ids)

    np.random.shuffle(unique_ids)
    n_train = int(len(unique_ids) * frac)

    train_ids = set(unique_ids[:n_train])

    train_events = [e for e in events if e.event_id in train_ids]
    val_events   = [e for e in events if e.event_id not in train_ids]

    return train_events, val_events

def get_scores(model, X):
    """
    get scores as logits output = model(input)
    return probability scores sigma(z)
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float()
        outputs = model(X_tensor).squeeze()
        probs = torch.sigmoid(outputs)
    return probs.numpy()

def roc_curve_manual(y, s):
    order = np.argsort(-s)
    y = y[order]
    s = s[order]

    tp = np.cumsum(y == 1).astype(float)
    fp = np.cumsum(y == 0).astype(float)

    P = max(1, (y == 1).sum())
    N = max(1, (y == 0).sum())

    tpr = tp / P
    fpr = fp / N

    fpr = np.r_[0.0, fpr, 1.0]
    tpr = np.r_[0.0, tpr, 1.0]

    auc = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc

def plot_training(history, save=False):
    """
    Helper to plot the train and validation losses vs epoch
    param: history = returned dict from Trainer -> train()
    """
    plt.figure()
    plt.plot(history["train_losses"], label="Train")
    plt.plot(history["val_losses"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    if save:
        plt.savefig('./results/val_train_loss.pdf')

def plot_score_distribution(y, s, save=False):
    """
    Plot area normalized (density = True) 
    probability density for signal and background scores
    - probability density (y-axis) for a given NN score (x-axis)
    """
    signal_scores = s[y==1]
    background_scores = s[y==0]

    plt.figure()
    plt.hist(background_scores, bins=50, alpha=0.5, density=True, label=f'background N={len(background_scores)}')
    plt.hist(signal_scores,bins=50, alpha=0.5, density=True, label=f'Signal N={len(signal_scores)}')
    
    plt.xlabel('NN Score')
    plt.ylabel('Density')
    plt.title('Score distribution')
    plt.legend() 
    plt.grid()
    plt.show()

    if save:
        plt.savefig('./results/score_distrib.pdf')


def plot_purity_vs_threshold(y, s, save=False):
    """
    
    """
    order = np.argsort(-s)

    y_sorted = y[order]
    s_sorted = s[order]

    # Cumulative TP/FP as we include items one by one
    tp = np.cumsum(y_sorted == 1).astype(float)
    fp = np.cumsum(y_sorted == 0).astype(float)

    # Purity at each step after including that item
    purity = tp / np.maximum(1.0, tp + fp)

    # Threshold used at each step is t = score
    t = s_sorted

    # Class prevalence = expected purity for random scores (baseline)
    prevalence = y.mean()

    plt.figure(figsize=(5,4))
    plt.step(t, purity, where="pre", label="purity S/(S+B)")
    plt.scatter(t, purity, s=18)

    # Random-guess baseline (horizontal line at prevalence)
    plt.plot([0,1], [prevalence, prevalence], "--", lw=1, color="gray",
            label=f"random baseline (prevalence={prevalence:.2f})")

    plt.xlim(0,1); plt.ylim(0,1.05)
    plt.xlabel(r"$t$  (predict positive if score $>= t$)")
    plt.ylabel(r"Purity  $S/(S+B)=\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$")
    plt.title("Purity vs threshold")
    plt.grid(True)
    plt.legend()
    plt.show()

    if save:
        plt.savefig('./results/plot_purity_v_thresh.pdf')


def find_working_point(y, s, min_eps_S = 0.3):
    """
    Find working point t* by maximizing purity subject to eps_S >= 0.3

    return t*, purity, eps_S, eps_B
    """

    order = np.argsort(-s)

    y_sorted = y[order]
    s_sorted = s[order]

    S0 = (y == 1).sum()
    B0 = (y == 0).sum()

    tp = np.cumsum(y_sorted == 1).astype(float)
    fp = np.cumsum(y_sorted == 0).astype(float)

    eps_S = tp/max(1, S0)
    eps_B = fp/max(1, B0)

    purity = tp / np.maximum(1, tp + fp)
    filtered = np.where(eps_S >= min_eps_S)[0] # first occurance where eps_S >= min_eps_S

    best_id = filtered[np.argmax(purity[filtered])]

    best_threshold = s_sorted[best_id]
    best_purity = purity[best_id]
    best_eps_S = eps_S[best_id]
    best_eps_B = eps_B[best_id]

    return best_threshold, best_purity, best_eps_S, best_eps_B




