import matplotlib.pyplot as plt
from typing import List, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')

colors=['#7f6d5f','#557f2d','#2d7f5e']

def group_bar_plot(groups_data: List[List], vars: List[str], groups_keys: List, bar_width: float = .25, file_name: str = ''):
    """
    data: [[group1], [group2], ..., [groupN]] \n
    key: [key1, key2, ..., keyN]
    """
    assert len(groups_data) > 1
    assert len(groups_data) == len(vars)

    # Set positions
    for i, group_data in enumerate(groups_data):
        r = np.arange(len(group_data)) + i*bar_width
        plt.bar(r, group_data, color=colors[i],
                width=bar_width, edgecolor='white', label=vars[i])

    # Add xticks on the middle of the group bars
    plt.xlabel('group', fontweight='bold')
    plt.xticks([r+bar_width/2 for r in range(len(groups_data[0]))], groups_keys)

    plt.legend()
    plt.savefig('test.png'.format(file_name))
