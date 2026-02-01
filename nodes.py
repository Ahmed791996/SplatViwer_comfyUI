import struct
import json
import os
import numpy as np


def parse_ply(filepath):
    """Parse a 3D Gaussian Splatting PLY file."""
    with open(filepath, "rb") as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Parse header for element count and properties
        vertex_count = 0
        properties = []
        for line in header_lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))

        prop_names = [p[1] for p in properties]

        # Build struct format from property types
        type_map = {
            "float": "f",
            "double": "d",
            "uchar": "B",
            "uint8": "B",
            "int": "i",
            "uint": "I",
            "short": "h",
            "ushort": "H",
        }
        fmt = "<" + "".join(type_map.get(p[0], "f") for p in properties)
        stride = struct.calcsize(fmt)

        # Read binary data
        data = f.read(vertex_count * stride)

    positions = []
    colors = []
    scales = []
    rotations = []
    opacities = []

    # Find property indices
    def idx(name):
        try:
            return prop_names.index(name)
        except ValueError:
            return -1

    ix, iy, iz = idx("x"), idx("y"), idx("z")

    # Color: try f_dc_0/1/2 (SH DC component) first, then red/green/blue
    i_dc0, i_dc1, i_dc2 = idx("f_dc_0"), idx("f_dc_1"), idx("f_dc_2")
    i_red, i_green, i_blue = idx("red"), idx("green"), idx("blue")

    # Scale
    i_s0, i_s1, i_s2 = idx("scale_0"), idx("scale_1"), idx("scale_2")

    # Rotation quaternion
    i_r0, i_r1, i_r2, i_r3 = idx("rot_0"), idx("rot_1"), idx("rot_2"), idx("rot_3")

    # Opacity
    i_opacity = idx("opacity")

    for i in range(vertex_count):
        vals = struct.unpack_from(fmt, data, i * stride)

        positions.append([vals[ix], vals[iy], vals[iz]])

        # Color from SH DC coefficients (convert SH0 to RGB)
        if i_dc0 >= 0:
            SH_C0 = 0.2820947917738781
            r = 0.5 + SH_C0 * vals[i_dc0]
            g = 0.5 + SH_C0 * vals[i_dc1]
            b = 0.5 + SH_C0 * vals[i_dc2]
            colors.append([
                max(0.0, min(1.0, r)),
                max(0.0, min(1.0, g)),
                max(0.0, min(1.0, b)),
            ])
        elif i_red >= 0:
            colors.append([vals[i_red] / 255.0, vals[i_green] / 255.0, vals[i_blue] / 255.0])
        else:
            colors.append([1.0, 1.0, 1.0])

        if i_s0 >= 0:
            scales.append([
                float(np.exp(vals[i_s0])),
                float(np.exp(vals[i_s1])),
                float(np.exp(vals[i_s2])),
            ])
        else:
            scales.append([0.01, 0.01, 0.01])

        if i_r0 >= 0:
            q = [vals[i_r0], vals[i_r1], vals[i_r2], vals[i_r3]]
            norm = max((q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2) ** 0.5, 1e-8)
            rotations.append([q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm])
        else:
            rotations.append([1.0, 0.0, 0.0, 0.0])

        if i_opacity >= 0:
            # Sigmoid activation on raw opacity
            raw = vals[i_opacity]
            sig = 1.0 / (1.0 + float(np.exp(-raw)))
            opacities.append(sig)
        else:
            opacities.append(1.0)

    return {
        "positions": positions,
        "colors": colors,
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities,
        "count": vertex_count,
    }


def parse_splat(filepath):
    """Parse a .splat binary file (antimatter15 format).
    Each splat is 32 bytes:
      3x float32 position (12 bytes)
      3x float32 scale (12 bytes)
      4x uint8 color RGBA (4 bytes)
      4x uint8 rotation quaternion (4 bytes) — mapped from [0,255] to [-1,1]
    """
    with open(filepath, "rb") as f:
        data = f.read()

    stride = 32
    count = len(data) // stride

    positions = []
    colors = []
    scales = []
    rotations = []
    opacities = []

    for i in range(count):
        off = i * stride
        x, y, z = struct.unpack_from("<fff", data, off)
        sx, sy, sz = struct.unpack_from("<fff", data, off + 12)
        r, g, b, a = struct.unpack_from("<BBBB", data, off + 24)
        qr, qi, qj, qk = struct.unpack_from("<BBBB", data, off + 28)

        positions.append([x, y, z])
        scales.append([sx, sy, sz])
        colors.append([r / 255.0, g / 255.0, b / 255.0])
        opacities.append(a / 255.0)

        # Map [0,255] -> [-1,1] and normalize
        q = [(qr - 128) / 128.0, (qi - 128) / 128.0, (qj - 128) / 128.0, (qk - 128) / 128.0]
        norm = max((q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2) ** 0.5, 1e-8)
        rotations.append([q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm])

    return {
        "positions": positions,
        "colors": colors,
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities,
        "count": count,
    }


class GaussianSplatViewer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "view"
    CATEGORY = "3D/Gaussian Splatting"

    def view(self, file_path):
        if not file_path or not os.path.isfile(file_path):
            return {"ui": {"splat_data": [json.dumps({"error": "File not found", "count": 0})]}}

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".ply":
            splat_data = parse_ply(file_path)
        elif ext == ".splat":
            splat_data = parse_splat(file_path)
        else:
            return {"ui": {"splat_data": [json.dumps({"error": f"Unsupported format: {ext}", "count": 0})]}}

        return {"ui": {"splat_data": [json.dumps(splat_data)]}}


NODE_CLASS_MAPPINGS = {
    "GaussianSplatViewer": GaussianSplatViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GaussianSplatViewer": "Gaussian Splat Viewer",
}
