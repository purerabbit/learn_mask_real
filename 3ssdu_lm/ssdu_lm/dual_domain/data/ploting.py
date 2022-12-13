import math
import numpy as np
import torch
from matplotlib import pyplot as plt


def imsshow(imgs, titles=None, ncols=5, dpi=100, cmap=None, is_colorbar=False, is_ticks=False,
            col_width=3, row_width=3, margin_ratio=0.1, n_images_max=50, filename2save=None, **imshow_kwargs):
    '''
    assume imgs is Sequence[ndarray[Nx, Ny]]
    '''
    num_imgs = len(imgs)

    if num_imgs > n_images_max:
        print(
            f"[WARNING] Too many images ({num_imgs}), clip to argument n_images_max({n_images_max}) for performance reasons.")
        imgs = imgs[:n_images_max]
        num_imgs = n_images_max

    if isinstance(cmap, list):
        assert len(cmap) == len(imgs)
    else:
        cmap = [cmap, ] * num_imgs

    nrows = math.ceil(num_imgs / ncols)

    # compute the figure size, compute necessary size first, then add margin
    figsize = (ncols * col_width, nrows * row_width)
    figsize = (figsize[0] * (1 + margin_ratio), figsize[1] * (1 + margin_ratio))
    fig = plt.figure(dpi=dpi, figsize=figsize)
    for i in range(num_imgs):
        ax = plt.subplot(nrows, ncols, i + 1)
        im = ax.imshow(imgs[i], cmap=cmap[i], **imshow_kwargs)
        if titles:
            plt.title(titles[i])
        if is_colorbar:
            cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.01, ax.get_position().height])
            plt.colorbar(im, cax=cax)
        if not is_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
    if filename2save is not None:
        fig.savefig(filename2save)
    else:
        plt.show()
    plt.close('all')

#测试此单元 
#/home/liuchun/dataset_dual/dual_domain/save_results/257.png

if __name__=="__main__":
    a=np.random.randn(1,256,256)
    b=np.random.randn(1,256,256)
    c=np.concatenate((a,b),0)
    imsshow(imgs=c,titles='test',is_colorbar=True)