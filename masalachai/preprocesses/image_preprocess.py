# -*- coding: utf-8 -*-

import numpy

import scipy.ndimage


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = numpy.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = numpy.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = numpy.dot(
        numpy.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_index=0,
                    fill_mode='nearest',
                    cval=0.):
    x = numpy.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [scipy.ndimage.interpolation.affine_transform(
        x_channel, final_affine_matrix, final_offset,
        order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = numpy.stack(channel_images, axis=0)
    x = numpy.rollaxis(x, 0, channel_index + 1)
    return x


def rotation(x,
             g,
             row_index=1,
             col_index=2,
             channel_index=0,
             fill_mode='nearest',
             cval=0.):
    theta = numpy.pi / 180 * g
    rotation_matrix = numpy.array([[numpy.cos(theta), -numpy.sin(theta), 0],
                                   [numpy.sin(theta), numpy.cos(theta), 0],
                                   [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_rotation(x,
                    rg,
                    row_index=1,
                    col_index=2,
                    channel_index=0,
                    fill_mode='nearest',
                    cval=0.):
    g = numpy.random.uniform(-rg, rg)
    return rotation(x, g, row_index=row_index, col_index=col_index,
                    channel_index=channel_index,
                    fill_mode=fill_mode, cval=cval)


def shift(x,
          wg,
          hg,
          row_index=1,
          col_index=2,
          channel_index=0,
          fill_mode='nearest',
          cval=0.):
    h, w = x.shape[row_index], x.shape[col_index]
    tx = hg * h
    ty = wg * w
    translation_matrix = numpy.array([[1, 0, tx],
                                      [0, 1, ty],
                                      [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shift(x,
                 wrg,
                 hrg,
                 row_index=1,
                 col_index=2,
                 channel_index=0,
                 fill_mode='nearest',
                 cval=0.):
    wg = numpy.random.uniform(-hrg, hrg)
    hg = numpy.random.uniform(-wrg, wrg)
    return shift(x, wg, hg, row_index=row_index, col_index=col_index,
                 channel_index=channel_index, fill_mode=fill_mode, cval=cval)


def shear(x,
          s,
          row_index=1,
          col_index=2,
          channel_index=0,
          fill_mode='nearest',
          cval=0.):
    shear_matrix = numpy.array([[1, -numpy.sin(s), 0],
                                [0, numpy.cos(s), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shear(x,
                 intensity,
                 row_index=1,
                 col_index=2,
                 channel_index=0,
                 fill_mode='nearest',
                 cval=0.):
    s = numpy.random.uniform(-intensity, intensity)
    return shear(x, s, row_index=row_index, col_index=col_index,
                 channel_index=channel_index, fill_mode=fill_mode, cval=cval)


def zoom(x,
         zoom_rate,
         row_index=1,
         col_index=2,
         channel_index=0,
         fill_mode='nearest',
         cval=0.):
    if len(zoom_rate) != 2:
        raise Exception('zoom_rate should be a tuple or list of two floats. '
                        'Received arg: ', zoom_rate)

    zx = zoom_rate[0]
    zy = zoom_rate[1]
    zoom_matrix = numpy.array([[zx, 0, 0],
                               [0, zy, 0],
                               [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_zoom(x,
                zoom_range,
                row_index=1,
                col_index=2,
                channel_index=0,
                fill_mode='nearest',
                cval=0.):
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = numpy.random.uniform(zoom_range[0], zoom_range[1], 2)
    return zoom(x, (zx, zy), row_index=row_index, col_index=col_index,
                channel_index=channel_index, fill_mode=fill_mode, cval=cval)


def load_image(path,
               grayscale=False,
               target_size=None):
    from PIL import Image
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')

    if target_size:
        img = img.resize(target_size)

    img_ary = numpy.asarray(img, dtype=numpy.float32)

    if grayscale:
        img_ary = numpy.expand_dims(img_ary, -1)

    img_ary = img_ary.transpose(2, 0, 1)

    return img_ary
