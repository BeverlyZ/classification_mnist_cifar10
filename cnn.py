from typing import Union, Tuple


def get_cnn_filtered_size(
        input_shape: Union[int, Tuple[int, int]],
        kernel: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0
) -> Tuple[int, int]:
    """
    :param input_shape: 输入尺寸
    :param kernel: 过滤器尺寸
    :param stride: 过滤器步长
    :param padding: 边缘填充
    :return: 过滤后每层尺寸
    """
    if isinstance(input_shape, int):
        input_shape = (input_shape, input_shape)
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    calculate = (lambda i: (input_shape[i] - kernel[i] + 2 * padding[i]) // (stride[i]) + 1)
    return calculate(0), calculate(1)


def get_flatten_size(input_shape: Tuple[int, ...], depth: int) -> int:
    """
    :param input_shape:
    :param depth: CNN最后一层的输出深度
    :return:
    """
    result = 1
    for num in input_shape:
        result *= num
    return result * depth


if __name__ == '__main__':
    # input_shape = (28, 28)
    # input_shape = get_cnn_filtered_size(input_shape, kernel=5)
    # input_shape = get_cnn_filtered_size(input_shape, kernel=2, stride=2)
    # input_shape = get_cnn_filtered_size(input_shape, kernel=3)
    # output_size = get_flatten_size(input_shape, 20)
    # print(output_size)

    input_shape = (32, 32)
    input_shape = get_cnn_filtered_size(input_shape, kernel=5)
    input_shape = get_cnn_filtered_size(input_shape, kernel=2, stride=2)
    input_shape = get_cnn_filtered_size(input_shape, kernel=3)
    output_size = get_flatten_size(input_shape, 20)
    print(output_size)
