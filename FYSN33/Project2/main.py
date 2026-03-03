from eventGenerator import load_events_csv
from analysis import compute_mass_spectrum, pass_cuts, evaluate_selection_mc
from plotter import plot_density_hist
from NN import *


# =================
#       Main
# =================
if __name__ == "__main__":
    
    DATA_FILE = "./datasets/jets.csv"
    MC_FILE   = "./datasets/pythia.csv"

    # ----------------------------------
    # 1. Load events
    # ----------------------------------
    data_events = load_events_csv(DATA_FILE)
    mc_events = load_events_csv(MC_FILE)

    # ---------------------------------
    # 2. Split MC by event id (80/20)
    # ---------------------------------
    train_events, val_events = split_by_id(mc_events, frac=0.8)

    # ---------------------------------
    # 3. Build features
    # ---------------------------------
    train_builder = FeatureBuilder(train_events)
    val_builder = FeatureBuilder(val_events)

    X_train, y_train = train_builder.build_dataset()
    X_val, y_val = val_builder.build_dataset()

    # ---------------------------------
    # 4. Standardize
    # ---------------------------------
    X_train_std, X_val_std, mu, sigma = train_builder.standardize(X_train, X_val)

    # ---------------------------------
    # 5. Dataloaders (to torch)
    # ---------------------------------
    train_dataset = JetData(X_train_std, y_train)
    val_dataset = JetData(X_val_std, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128)

    # --------------------------------
    # 6. Model setup and training
    # --------------------------------

    model = JetClassifier(print_model=True)
    trainer = Trainer(model, lr=1e-3, patience=30)

    history = trainer.train(train_loader, val_loader, epochs=500) # dict with val/train loss and acc

    plot_training(history) # train and val loss v epoch

    # ----------------------------------
    # 7. plot loss and acc for train/val
    # ----------------------------------
    losses = history["train_losses"]
    val_losses = history["val_losses"]
    acc = history["train_accuracies"]
    val_acc = history["val_accuracies"]

    epochs = len(losses)

    plt.figure()
    plt.plot(range(epochs), acc, label="Train Accuracy")
    plt.plot(range(epochs), val_acc, label="Val Accuracy")
    plt.legend()
    plt.show()

    # -------------------------------
    # 8. Validation, ROC, AUC
    # -------------------------------
    val_scores = get_scores(model, X_val_std)
    fpr, tpr, auc = roc_curve_manual(y_val, val_scores)

    print("Validation AUC:", auc)
    print()

    fpr_m, tpr_m, auc_m = roc_curve_manual(y_val, val_scores)

    plt.figure()
    plt.plot(fpr_m, tpr_m, label=f"AUC = {auc_m:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Jet Classifier)")
    plt.legend()
    plt.grid()
    plt.show()

    plot_score_distribution(y_val, val_scores)
    plot_purity_vs_threshold(y_val, val_scores) # P(t)

    # ----------------------------------------
    # 9. Choose working point t*
    # ----------------------------------------
    t_star, best_purity, best_eps_S, best_eps_B = find_working_point(y_val, val_scores)
    print(f"Chosen working point (optimal threshold), t* = {t_star:.4f}")
    print(f"Purity P(t*) = {best_purity:.4f}")
    print(f"Signal efficiency epsilon_S = {best_eps_S:.4f}")
    print(f"Background efficiency epsilon_B = {best_eps_B:.4f}")
    print()

    # ---------------------------------------
    # 10. Create nn selection function
    # ---------------------------------------
    nn_selection = NNselector(
        model=model,
        feature_builder=FeatureBuilder(mc_events),
        mu=mu,
        sigma=sigma,
        threshold=t_star
    )

    # ----------------------------------------
    # 11. Compare purity on mc_data
    # ----------------------------------------
    results_cuts = evaluate_selection_mc(mc_events, pass_cuts)
    results_nn = evaluate_selection_mc(mc_events, nn_selection)

    print(f"Baseline purity (no cuts) = {results_cuts['purity0']:.3f}  "
          f"with S0={results_cuts['S0']}, B0={results_cuts['B0']}")
    print(f"MC (cuts): purity S/(S+B) = {results_cuts['purity_mc']:.3f}  "
          f"with S={results_cuts['S']}, B={results_cuts['B']}, N={results_cuts['N_sel']}  |  eps_S={results_cuts['eps_S']:.3f}, eps_B={results_cuts['eps_B']:.3f}")
    print(f"MC (NN): purity S/(S+B) = {results_nn['purity_mc']:.3f}  "
          f"with S={results_nn['S']}, B={results_nn['B']}, N={results_nn['N_sel']}  |  eps_S={results_nn['eps_S']:.3f}, eps_B={results_nn['eps_B']:.3f}")
    print()

    # ------------------------------------------------------------
    # 12. Apply NN to DATA mass spectrum (compare to pass_cuts(event))
    # ------------------------------------------------------------
    all_masses = compute_mass_spectrum(data_events)
    sel_masses_cuts = compute_mass_spectrum(data_events, pass_cuts)
    sel_masses_nn = compute_mass_spectrum(data_events, nn_selection)

    print(f"Data (cuts): slected = {len(sel_masses_cuts)/len(all_masses) * 100:.3f} %")
    print(f"Data (NN): slected = {len(sel_masses_nn)/len(all_masses) * 100:.3f} %") # this looks terrible?? (13 %)

    plot_density_hist(all_masses, sel_masses_cuts, sel_masses_nn, labels=["All data", "Cuts", "NN"])
    












