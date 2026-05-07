import numpy as np

def matprint(mat, fmt=".4g"):

    """
    Print a 2D array in a formatted way, with each column aligned according to the maximum width of the elements in that column.
    The format for each element can be specified using the fmt argument, which defaults to ".4g" for general format with 4 significant digits.

    Args:
        mat (2D array-like): The 2D array to print.
        fmt (str, optional): The format string for each element. Defaults to ".4g".
    """

    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")      

def scale_point(x, lo, hi):
    """
    Scale a point x from the unit interval [0, 1] to the interval [lo, hi].
    The function takes a point x in the unit interval and maps it to the corresponding point in the interval [lo, hi] using a linear transformation.

    Args:
        x (0 <= x <= 1): The point to scale.
        lo (float): The lower bound of the target interval.
        hi (float): The upper bound of the target interval.

    Returns:
        float: The scaled point in the interval [lo, hi].
    """

    return np.array(x) * (hi - lo) + lo
