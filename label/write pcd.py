
def save_pcd(pc: np.ndarray, file, binary=True):
    pc = pc.astype(np.float32)
    num_points = len(pc)

    with open(file, 'wb' if binary else 'w') as f:
        # heads
        headers = [
            '# .PCD v0.7 - Point Cloud Data file format',
            'VERSION 0.7',
            'FIELDS x y z i',
            'SIZE 4 4 4 4',
            'TYPE F F F F',
            'COUNT 1 1 1 1',
            f'WIDTH {num_points}',
            'HEIGHT 1',
            'VIEWPOINT 0 0 0 1 0 0 0',
            f'POINTS {num_points}',
            f'DATA {"binary" if binary else "ascii"}'
        ]
        header = '\n'.join(headers) + '\n'
        if binary:
            header = bytes(header, 'ascii')
        f.write(header)

        # points
        if binary:
            f.write(pc.tobytes())
        else:
            for num in range(num_points):
                x, y, z, i = pc[num]
                f.write(f"{x:.3f} {y:.3f} {z:.3f}  {i:.3f}\n")
