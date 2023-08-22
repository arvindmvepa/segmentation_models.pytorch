import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io


def draw_results(inputs, targets, segs, aucroc, dice, save_dir, image_width=1024, image_height=1024,
                 n_examples_to_plot=4, decision_thresh=.999):
    fig, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10), squeeze=False)
    fig.suptitle("AUCROC: {}, DICE: {}".format(aucroc, dice), fontsize=20)
    for example_i in range(n_examples_to_plot):
        axs[0][example_i].imshow(inputs[example_i], cmap='gray')
        axs[0][example_i].axis('off')
        axs[1][example_i].imshow(targets[example_i].astype(np.float32), cmap='gray')
        axs[1][example_i].axis('off')
        axs[2][example_i].imshow(
            np.reshape(segs[example_i, :, :], [image_width, image_height]),
            cmap='gray')
        axs[2][example_i].axis('off')
        test_image_thresholded = np.array(
            [0 if x < decision_thresh else 255 for x in segs[example_i, :, :].flatten()])
        axs[3][example_i].imshow(
            np.reshape(test_image_thresholded, [image_width, image_height]),
            cmap='gray')
        axs[3][example_i].axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('{}/figure{}.jpg'.format(save_dir, epoch_num))
    return buf
