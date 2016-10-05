import matplotlib.pyplot as plt


def save_figure_as_image(fileName, fig=None, **kwargs):
    ''' Save a Matplotlib figure as an image without borders or frames.
        Args:
        fileName (str): String that ends in .png etc.

        fig (Matplotlib figure instance): figure you want to save as the image
        Keyword Args:
        orig_size (tuple): width, height of the original image used to maintain
        aspect ratio.
    '''
    fig_size = fig.get_size_inches()
    w, h = fig_size[0], fig_size[1]
    fig.patch.set_alpha(0)
    if kwargs.has_key('orig_size'):  # Aspect ratio scaling if required
        print('New')
        w, h = kwargs['orig_size']

        DPI = fig.get_dpi()
        fig.set_size_inches(w/float(DPI),h/float(DPI))

        #w2, h2 = fig_size[0], fig_size[1]
        #fig.set_size_inches([(w2 / w) * w, (w2 / w) * h])
        #fig.set_dpi((w2 / w) * fig.get_dpi())
    a = fig.gca()
    a.set_frame_on(False)
    a.set_xticks([])
    a.set_yticks([])
    plt.axis('off')
    plt.xlim(0, h)
    plt.ylim(w, 0)
    fig.savefig(fileName, transparent=True, bbox_inches='tight', pad_inches=0)
