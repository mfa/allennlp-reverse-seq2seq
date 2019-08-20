import matplotlib
matplotlib.use('Agg')
import numpy as np

import matplotlib.pyplot as plt
import json


with open("output/examples.output") as fp:
    for index, line in enumerate(fp.readlines()):
        data = json.loads(line)
        source = data["meta"]["source_tokens"]
        target = data["meta"]["target_tokens"]
        matrix = np.asarray(data["attentions"], dtype=np.float32)
        fig, ax = plt.subplots()
        im = ax.matshow(matrix)

        ax.set_xticks(np.arange(len(source)))
        ax.set_yticks(np.arange(len(target)))
        ax.set_xticklabels(source, rotation=90)
        ax.set_yticklabels(target)

        fig.tight_layout()
        plt.savefig(f"output/attention_{index}.png")
