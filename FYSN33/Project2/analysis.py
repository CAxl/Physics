from eventGenerator import Event, dphi

# Cuts
min_pt_j  = 250.0 # min pT for jet
min_pt_l  = 50.0 # min pT for lepton
min_dphi  = 2.4  # radians
eta_j_max = 2.0  # max eta for jet

def pass_cuts(e: Event):
    """
    provided cut-based selection
    """
    
    j = e.leading_jet()
    l = e.leading_lepton()
    if (j is None) or (l is None):
        return False
    return (
        (j.pt >= min_pt_j) and
        (l.pt >= min_pt_l) and
        (dphi(j.phi, l.phi) >= min_dphi) and
        (abs(j.eta) <= eta_j_max)
    )

# mass spectra, pass_function -> mass spectra after cuts
def compute_mass_spectrum(events, pass_function=None):
    """
    Collect:
    - masses (pass_func == None)
    - sel_masses (pass_func)
    """
    
    masses = []

    for event in events:
        j = event.leading_jet()
        if j is None:
            continue

        # keep event if no selection, OR if it passes the selection
        if pass_function is None or pass_function(event):
            masses.append(j.m)

    return masses

def mc_totals(events):
    """
    Collects without selection:
    - total signal (S0)
    - total background (B0)
    (for efficiencies)
    """

    S0 = 0
    B0 = 0

    for e in events:
        j = e.leading_jet()
        if j is None:
            continue
        if j.truth:
            S0 += 1
        else:
            B0 += 1

    return S0, B0

def evaluate_selection_mc(events, pass_function):
    """
    Compute for selection:
    - purity_mc = S/(S+B)
    - signal efficiency eps_S = S / S0
    - background efficiency eps_B = B / B0
    """
    S0, B0 = mc_totals(events)
    S = 0
    B = 0

    for e in events:
        j = e.leading_jet()
        l = e.leading_lepton()

        if (j is None) or (l is None):
            continue

        if pass_function(e):
            if j.truth:
                S += 1
            else:
                B += 1

        N_sel = S + B
        purity_mc = (S / N_sel) if N_sel > 0 else float('nan')
        eps_S = S / S0 if S0 > 0 else float('nan')
        eps_B = B / B0 if B0 > 0 else float('nan')
        purity0 = S0 / max(1, (S0 + B0)) #(?)

    # return dict
    return {"S": S, 
            "B": B, 
            "S0": S0, 
            "B0": B0,
            "purity_mc": purity_mc,
            "eps_S": eps_S,
            "eps_B": eps_B,
            "purity0": purity0,
            "N_sel": N_sel
            }




