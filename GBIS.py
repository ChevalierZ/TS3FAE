import numpy as np

def sam(p_1, p_2):
    r = p_1.reshape((-1, 1))
    d = p_2.reshape((-1, 1))
    rr = np.sum(r ** 2, 0) ** 0.5
    dd = np.sum(d ** 2, 0) ** 0.5
    rd = np.sum(r * d, 0)
    cos_value = rd / (rr * dd)
    eps = 1e-6
    if 1.0 < cos_value < 1.0 + eps:
        cos_value = 1.0
    elif -1.0 - eps < cos_value < -1.0:
        cos_value = -1.0
    result = 1 / abs(cos_value)
    return result


def ed(p_1, p_2):
    result = 2 * np.sin(sam(p_1, p_2)) / 2
    return result


def sid(p_1, p_2):
    r = p_2
    d = p_1
    m = r / np.sum(r, 0)
    n = d / np.sum(d, 0)
    drd = np.sum(m * np.log((m / n)), 0)
    ddr = np.sum(n * np.log((n / m)), 0)
    result = drd + ddr
    return result


def segment_graph(height_width, num, edges, c=1.0, min_size=5):
    u_array = np.zeros((height_width, 3), dtype=np.int32)
    u_array[:, 1] = np.array(range(height_width), dtype=np.int32)
    u_array[:, 2] = np.ones(height_width, dtype=np.int32)
    thresholds_copy = np.full(height_width, c, dtype=np.float32)
    loop_range = range(num)

    for i in loop_range:
        edge = edges[i]
        a = edge['a']
        while a != u_array[a, 1]:
            a = edge['a'] = u_array[a, 1]
        b = edge['b']
        while b != u_array[b, 1]:
            b = edge['b'] = u_array[b, 1]
        if a != b:
            if edge['w'] <= thresholds_copy[a] and edge['w'] <= thresholds_copy[b]:
                if (u_array[a, 0] > u_array[b, 0]):
                    u_array[b, 1] = a
                    u_array[a, 2] += u_array[b, 2]
                else:
                    u_array[a, 1] = b
                    u_array[b, 2] += u_array[a, 2]
                    if u_array[a, 0] == u_array[b, 0]:
                        u_array[b, 0] += 1
                while a != u_array[edge['a'], 1]:
                    a = edge['a'] = u_array[edge['a'], 1]
                thresholds_copy[a] = edge['w'] + c / u_array[a, 2]
    for i in loop_range:
        while (edges[i]['a'] != u_array[edges[i]['a'], 1]):
            edges[i]['a'] = u_array[edges[i]['a'], 1]
        while (edges[i]['b'] != u_array[edges[i]['b'], 1]):
            edges[i]['b'] = u_array[edges[i]['b'], 1]
        if ((edges[i]['a'] != edges[i]['b']) and (
                (u_array[edges[i]['a'], 2] < min_size) or (u_array[edges[i]['b'], 2] < min_size))):
            if (u_array[edges[i]['a'], 0] > u_array[edges[i]['b'], 0]):
                u_array[edges[i]['b'], 1] = edges[i]['a']
                u_array[edges[i]['a'], 2] += u_array[edges[i]['b'], 2]
            else:
                u_array[edges[i]['a'], 1] = edges[i]['b']
                u_array[edges[i]['b'], 2] += u_array[edges[i]['a'], 2]
                if u_array[edges[i]['a'], 0] == u_array[edges[i]['b'], 0]:
                    u_array[edges[i]['b'], 0] += 1
    return u_array


# ===========================SegmentImage==========================================
# 像素间的差异度量
def diff(img3f, x1, y1, x2, y2, phase):
    r = img3f[..., y1, x1]
    d = img3f[..., y2, x2]
    if phase == 'sam':
        vertex_value = sam(r, d)
    elif phase == 'ed':
        vertex_value = ed(r, d)
    elif phase == 'sid':
        vertex_value = sid(r, d)
    return vertex_value


def SegmentImage(smImg3f, c=1.0, min_size=5):
    height, width = smImg3f.shape[1], smImg3f.shape[2]
    # (a,b)边连接的顶点，w为权重 获取完全图
    edges = np.zeros((height - 1) * (width - 1) * 4 + (height - 1) + (width - 1),
                     dtype={'names': ['a', 'b', 'w'], 'formats': ['i4', 'i4', 'f4']})
    num = 0
    width_range = range(width)
    height_range = range(height)
    for y in height_range:
        for x in width_range:
            if x < width - 1:
                edges[num]['a'] = y * width + x
                edges[num]['b'] = y * width + (x + 1)
                edges[num]['w'] = diff(smImg3f, x, y, x + 1, y, 'sid')
                num += 1
            if y < height - 1:
                edges[num]['a'] = y * width + x
                edges[num]['b'] = (y + 1) * width + x
                edges[num]['w'] = diff(smImg3f, x, y, x, y + 1, 'sid')
                num += 1
            if (x < (width - 1)) and (y < (height - 1)):
                edges[num]['a'] = y * width + x
                edges[num]['b'] = (y + 1) * width + (x + 1)
                edges[num]['w'] = diff(smImg3f, x, y, x + 1, y + 1, 'sid')
                num += 1
            if (x < (width - 1)) and y > 0:
                edges[num]['a'] = y * width + x
                edges[num]['b'] = (y - 1) * width + (x + 1)
                edges[num]['w'] = diff(smImg3f, x, y, x + 1, y - 1, 'sid')
                num += 1
    edges = np.sort(edges, order='w')
    u_array = segment_graph(width * height, num, edges, c, min_size)
    marker = {}
    imgIdx = np.zeros((smImg3f.shape[1], smImg3f.shape[2]), np.int32)
    idxNum = 0
    for y in height_range:
        for x in width_range:
            comp = y * width + x
            while (comp != u_array[comp, 1]):
                comp = u_array[comp, 1]
            if comp not in marker.keys():
                marker[comp] = idxNum
                idxNum += 1
            idx = marker[comp]
            imgIdx[y, x] = idx
    return idxNum, imgIdx