from PIL import ImageDraw
import matplotlib.pyplot as plt


def draw_boxes(boxes: list, image):
    for i in range(len(boxes)):
        x1, y1 = boxes[i][0], boxes[i][1]
        x2, y2 = boxes[i][2], boxes[i][3]
        draw = ImageDraw.Draw(image)
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)
    return image


def viszualize_images(imgs: list, titles: list, save: str = "../monitoring_old/before_after.png"):
    fig, axs = plt.subplots(1, len(imgs), figsize=(10, 5))
    for i in range(len(imgs)):
        axs[i].imshow(imgs[i])
        axs[i].set_title(titles[i])
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.savefig(save)


def draw_losses(losses: list, save: str = "../monitoring_old/losses.png"):
    x_values = range(len(losses))
    fig, ax = plt.subplots()
    ax.plot(x_values, losses)
    ax.set_xlabel('iteration')
    ax.set_ylabel('combined loss')
    plt.tight_layout()
    plt.savefig(save)