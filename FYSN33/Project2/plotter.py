import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["axes.grid"] = True

def plot_density_hist(data1, data2 = None, data3 = None, bins = 40, rng = (60,140), labels = None):
    plt.figure(figsize=(5.8,4.2))
    
    plt.hist(data1, bins=bins, range=rng, density=True, histtype="step", label=labels[0] if labels else None)

    # if plot one more histograms in same fig
    if data2 is not None:
        plt.hist(data2, bins=bins, range=rng, density=True, histtype="step", label=labels[1] if labels else None)

    # if plot one more histograms in same fig
    if data3 is not None:
        plt.hist(data3, bins=bins, range=rng, density=True, histtype="step", label=labels[2] if labels else None)


    
    plt.xlabel("Large-R jet mass [GeV]")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    #plt.title("Data: jet mass before/after cuts") # in original template (include?)
    plt.show()

