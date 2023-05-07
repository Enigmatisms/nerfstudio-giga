"""
    Visualize PFM depth from IGEV output
"""

import re

import matplotlib.pyplot as plt
import numpy as np


def read_pfm(filename):
    """Read PFM file, from https://gist.github.com/aminzabardast/cdddae35c367c611b6fd5efd5d63a326"""
    # rb: binary file and read only
    file = open(filename, "rb")
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf":  # depth is Pf
        color = False
    else:
        raise ValueError("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))  # re is used for matching
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise ValueError("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width, 1)
    # depth: H*W
    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

if __name__ == "__main__":
    import sys
    out_data, out_scale = read_pfm(sys.argv[1])
    out_data = np.where(out_data > 100, 0, out_data)
    print(out_data.shape)

    plt.imshow(out_data, cmap = 'rainbow')
    plt.colorbar()
    # plt.show()
    plt.savefig("result.png")
