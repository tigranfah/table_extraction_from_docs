import matplotlib.pyplot as plt


def imshow(image, fac=50, mi=0, ma=1):
    fig = plt.figure(figsize=(image.shape[0]/fac, image.shape[1]/fac))
    ax = fig.add_subplot()
    ax.imshow(image, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])


def anshow(image, mask, fac=15, mi=0, ma=1):
    fig, ax = plt.subplots(1, 2, figsize=(image.shape[0]/fac, image.shape[1]/fac))
    ax[0].imshow(image, cmap="gray", vmin=mi, vmax=ma)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(mask, cmap="gray", vmin=mi, vmax=ma)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.show()