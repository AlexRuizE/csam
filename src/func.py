"""Auxiliary functions."""

def get_k_img(k, img_dataset):
    """Get k images from a pytorch <dataset> object."""
    l = len(img_dataset.imgs)
    indices = torch._np.random.choice(l,
                                      size=k,
                                      replace=False)
    img_list = [img_dataset.__getitem__(i)[0] for i in indices]
    img_tensors = torch.stack(img_list)
    return img_tensors


