import torch


MASK_DEFAULT = ["0", "0", "0", "0", "1", "0"]


def parse_mask_strategy(mask_strategy):
    mask_batch = []
    if mask_strategy == "" or mask_strategy is None:
        return mask_batch

    mask_strategy = mask_strategy.split(";")
    for mask in mask_strategy:
        mask_group = mask.split(",")
        num_group = len(mask_group)
        assert num_group >= 1 and num_group <= 6, f"Invalid mask strategy: {mask}"
        mask_group.extend(MASK_DEFAULT[num_group:])
        for i in range(5):
            mask_group[i] = int(mask_group[i])
        mask_group[5] = float(mask_group[5])
        mask_batch.append(mask_group)
    return mask_batch


def find_nearest_point(value, point, max_value):
    t = value // point
    if value % point > point / 2 and t < max_value // point - 1:
        t += 1
    return t * point


def apply_mask_strategy(z, refs_x, mask_strategys, loop_i, align=None):
    masks = []
    no_mask = True
    for i, mask_strategy in enumerate(mask_strategys):
        no_mask = False
        mask = torch.ones(z.shape[2], dtype=torch.float, device=z.device)
        mask_strategy = parse_mask_strategy(mask_strategy)
        for mst in mask_strategy:
            loop_id, m_id, m_ref_start, m_target_start, m_length, edit_ratio = mst
            if loop_id != loop_i:
                continue
            ref = refs_x[i][m_id]

            if m_ref_start < 0:
                # ref: [C, T, H, W]
                m_ref_start = ref.shape[1] + m_ref_start
            if m_target_start < 0:
                # z: [B, C, T, H, W]
                m_target_start = z.shape[2] + m_target_start
            if align is not None:
                m_ref_start = find_nearest_point(m_ref_start, align, ref.shape[1])
                m_target_start = find_nearest_point(m_target_start, align, z.shape[2])
            m_length = min(m_length, z.shape[2] - m_target_start, ref.shape[1] - m_ref_start)
            z[i, :, m_target_start : m_target_start + m_length] = ref[:, m_ref_start : m_ref_start + m_length]
            mask[m_target_start : m_target_start + m_length] = edit_ratio
        masks.append(mask)
    if no_mask:
        return None
    masks = torch.stack(masks)
    return masks