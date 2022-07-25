import matplotlib.pyplot as plt
import cv2
import os


def imshow(image, fac=50):
    fig = plt.figure(figsize=(image.shape[0]/fac, image.shape[1]/fac))
    ax = fig.add_subplot()
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])


def anshow(image, mask, fac=15, mi=0, ma=1):
    fig, ax = plt.subplots(1, 2, figsize=(image.shape[0]/fac, image.shape[1]/fac))
    ax[0].imshow(image, cmap="gray", vmin=mi, vmax=1)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(mask, cmap="gray", vmin=mi, vmax=1)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.show()


def colanshow(image, mask, fac=15, mi=0, ma=1, alpha=0.7, gamma=0.3):
    fig, ax = plt.subplots(1, 2, figsize=(image.shape[0]/fac, image.shape[1]/fac))
    ax[0].imshow(image, cmap="gray", vmin=mi, vmax=1)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    col_image = cv2.addWeighted(image, alpha, mask, gamma, 0)
    # col_image = image + mask * alpha
    # print(np.unique(col_image))
    ax[1].imshow(col_image, cmap="gray", vmin=mi, vmax=1)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.show()


def save_pred(image, gt, pred, name, directory, fac=15, mi=0, ma=1):
    fig, ax = plt.subplots(1, 3, figsize=(image.shape[0]/fac, image.shape[1]/fac))
    ax[0].imshow(image, cmap="gray", vmin=mi, vmax=1)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].imshow(gt, cmap="gray", vmin=mi, vmax=1)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].imshow(pred, cmap="gray", vmin=mi, vmax=1)
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    fig.savefig(f"{directory}/" + name + ".png")