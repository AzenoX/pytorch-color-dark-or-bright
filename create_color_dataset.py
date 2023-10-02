"""
This script generates a dataset of unique random colors, checks if each color is considered dark
based on its luminance, and saves the resulting dataset to an Excel file named 'colors.xlsx'.
"""

import random
import pandas as pd
import progressbar


def is_dark_color(red: object, green: object, blue: object) -> int:
    """
    :rtype: int
    :param red:
    :param green:
    :param blue:
    :return:
    """
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    if luminance < 128:  # This threshold can be adjusted
        return 1
    return 0


# Generate the colors
data = []
colors = set()

for i in progressbar.progressbar(range(1000000)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    if (r, g, b) not in colors:  # Ensure unique colors
        colors.add((r, g, b))
        IS_DARK = is_dark_color(r, g, b)
        data.append([IS_DARK, r, g, b])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['is_dark', 'red', 'green', 'blue'])

# Save to Excel
print('Saving...')
df.to_excel('colors.xlsx', index=False, header=False)
print('Saved !')
