import pandas as pd
import random
import progressbar


# Function to determine if a color is dark
def is_dark_color(r, g, b):
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
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
        is_dark = is_dark_color(r, g, b)
        data.append([is_dark, r, g, b])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['is_dark', 'red', 'green', 'blue'])

# Save to Excel
print('Saving...')
df.to_excel('colors.xlsx', index=False, header=False)
print('Saved !')
