import matplotlib.pyplot as plt
import seaborn as sns

#  Vintage-inspired color palette
vintage_palette = [
    '#8B7355',  # Warm bronze (primary)
    '#6B8E23',  # Olive green (secondary)
    '#CD5C5C',  # Terra cotta (accent)
    '#4682B4',  # Steel blue (accent)
    '#D2B48C',  # Tan (neutral)
    '#696969',  # Dim gray (text/dark)
    '#F5F5DC',  # Beige (background)
    '#8FBC8F',  # Sage green (accent)
    '#B8860B',  # Dark goldenrod (accent)
    '#778899'   # Light slate gray (neutral)
]

#  Extended palette for various purposes
vintage_palette_extended = {
    'primary': '#8B7355',      # Warm bronze
    'secondary': '#6B8E23',    # Olive green
    'accent1': '#CD5C5C',      # Terra cotta
    'accent2': '#4682B4',      # Steel blue
    'accent3': '#D2B48C',      # Tan
    'accent4': '#8FBC8F',      # Sage green
    'accent5': '#B8860B',      # Dark goldenrod
    'dark': '#696969',         # Dim gray
    'light': '#F5F5DC',        # Beige
    'neutral': '#778899',      # Light slate gray
    'success': '#6B8E23',      # Olive green
    'warning': '#B8860B',      # Dark goldenrod
    'alert': '#CD5C5C'         # Terra cotta
}

#  Apply complete vintage visual theme
def set_vintage_theme():
    """
    Apply a vintage-inspired aesthetic to all matplotlib & seaborn plots.
    """
    sns.set_palette(vintage_palette)
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({
        'figure.facecolor': '#FDF5E6',  # Old lace background
        'axes.facecolor': '#FAF0E6',    # Linen axes background
        'grid.color': '#D2B48C',        # Tan grid lines
        'grid.alpha': 0.3,
        'text.color': '#696969',        # Dim gray text
        'axes.labelcolor': '#696969',
        'xtick.color': '#696969',
        'ytick.color': '#696969',
        'axes.edgecolor': '#8B7355',    # Bronze edges
        'axes.linewidth': 1.2,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Georgia', 'Palatino']
    })

    return vintage_palette_extended


#  Get colors for specific plot types
def get_vintage_colors(n_colors=5, color_type='main'):
    """
    Return a subset of vintage colors depending on the theme type.
    
    Parameters:
        n_colors (int): number of colors to return
        color_type (str): ['main', 'warm', 'cool', 'sequential']
    """
    if color_type == 'main':
        return vintage_palette[:n_colors]
    elif color_type == 'warm':
        return ['#8B7355', '#CD5C5C', '#D2B48C', '#B8860B', '#8B4513'][:n_colors]
    elif color_type == 'cool':
        return ['#4682B4', '#6B8E23', '#8FBC8F', '#778899', '#2F4F4F'][:n_colors]
    elif color_type == 'sequential':
        return ['#F5F5DC', '#D2B48C', '#8B7355', '#696969', '#2F2F2F'][:n_colors]
    else:
        return vintage_palette[:n_colors]
