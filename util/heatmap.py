import numpy as np

class GeneratePoseTarget:
    def __init__(self, sigma=2.0, eps=1e-4):
        self.sigma = sigma
        self.eps = eps

    def generate_a_heatmap(self, img_h, img_w, center, sigma, max_value):
        heatmap = np.zeros((img_h, img_w), dtype=np.float32)
        mu_x, mu_y = center
        if max_value < self.eps:
            return heatmap

        st_x = max(int(mu_x - 3 * sigma), 0)
        ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
        st_y = max(int(mu_y - 3 * sigma), 0)
        ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
        x = np.arange(st_x, ed_x, 1, np.float32)
        y = np.arange(st_y, ed_y, 1, np.float32)

        if not (len(x) and len(y)):
            return heatmap
        y = y[:, None]

        patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
        patch = patch * max_value
        heatmap[st_y:ed_y, st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x], patch)

        return heatmap

    def generate_a_limb_heatmap(self, img_h, img_w, start, end, sigma, start_value, end_value):
        try:
            heatmap = np.zeros((img_h, img_w), dtype=np.float32)
        except:
            print(img_h, img_w)
        value_coeff = min(start_value, end_value)
        if value_coeff < self.eps:
            return heatmap

        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

        min_x = max(int(min_x - 3 * sigma), 0)
        max_x = min(int(max_x + 3 * sigma) + 1, img_w)
        min_y = max(int(min_y - 3 * sigma), 0)
        max_y = min(int(max_y + 3 * sigma) + 1, img_h)

        x = np.arange(min_x, max_x, 1, np.float32)
        y = np.arange(min_y, max_y, 1, np.float32)

        if not (len(x) and len(y)):
            return heatmap

        y = y[:, None]
        x_0 = np.zeros_like(x)
        y_0 = np.zeros_like(y)

        d2_start = ((x - start[0]) ** 2 + (y - start[1]) ** 2)
        d2_end = ((x - end[0]) ** 2 + (y - end[1]) ** 2)
        d2_ab = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

        if d2_ab < 1:
            full_map = self.generate_a_heatmap(img_h, img_w, start, sigma, start_value)
            heatmap = np.maximum(heatmap, full_map)
            return heatmap

        coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab
        a_dominate = coeff <= 0
        b_dominate = coeff >= 1
        seg_dominate = 1 - a_dominate - b_dominate

        position = np.stack([x + y_0, y + x_0], axis=-1)
        projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
        d2_line = position - projection
        d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
        d2_seg = (a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line)

        patch = np.exp(-d2_seg / 2. / sigma ** 2)
        patch = patch * value_coeff

        heatmap[min_y:max_y, min_x:max_x] = np.maximum(heatmap[min_y:max_y, min_x:max_x], patch)

        return heatmap