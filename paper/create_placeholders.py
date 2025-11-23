# Script untuk membuat placeholder images
import matplotlib.pyplot as plt
import numpy as np

def create_placeholder_image(filename, title="Placeholder Image", figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color='lightgray'))

    ax.text(0.5, 0.5, title, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=16, fontweight='bold', color='darkgray')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

create_placeholder_image('images/placeholder.png', 'Generic Placeholder Image', (10, 6))

print("Generic placeholder image created successfully!")