import random


def net_wise(stage_wise_length, stage_wise_range):
    """生成搜索的配置

    Args:
        stage_wise_length (int): 指我们需要返回的setting的长度是多少，
        stage_wise_range (list of list): 采样的范围

    Returns:
        search_settings (list): 采样后的结果, len(search_settings) equals sample_length
    """
    search_settings = [random.choice(stage_wise_range[0])] * sum(stage_wise_length)
    return search_settings


def stage_wise(stage_wise_length, stage_wise_range):
    """生成搜索的配置

    Args:
        stage_wise_length (list): 每一个stage的长度
        stage_wise_range (list of list): 每一个stage的sample的范围

    Returns:
        search_settings (list): 采样后的结果，len(search_settings) equals sum(stage_wise_length)
    """
    search_settings = []
    for depth, curr_range in zip(stage_wise_length, stage_wise_range):
        curr = random.choice(curr_range)
        search_settings += [curr] * depth
    return search_settings


def ordered_stage_wise(stage_wise_length, stage_wise_range):
    """生成搜索的配置

    Args:
        stage_wise_length (list): 每一个stage的长度
        stage_wise_range (list of list): 每一个stage的sample的范围

    Returns:
        search_settings (list): 采样后的结果，len(search_settings) equals sum(stage_wise_length)
    """
    search_settings = []
    for depth, curr_range in zip(stage_wise_length, stage_wise_range):
        curr = random.choice(curr_range)
        if search_settings:
            while curr < search_settings[-1]:
                curr = random.choice(curr_range)
            search_settings += [curr] * depth
        else:
            search_settings += [curr] * depth
    return search_settings


def stage_wise_depth(stage_wise_length, stage_wise_range):
    """生成搜索的配置

    Args:
        stage_wise_length (list): 每一个stage的长度
        stage_wise_range (list of list): 每一个stage的sample的范围

    Returns:
        search_settings (list): 采样后的结果，len(search_settings) equals len(stage_wise_length)
    """
    search_settings = []
    for _, curr_range in zip(stage_wise_length, stage_wise_range):
        curr = random.choice(curr_range)
        search_settings += [curr]
    return search_settings


def ordered_stage_wise_depth(stage_wise_length, stage_wise_range):
    """生成搜索的配置

    Args:
        stage_wise_length (list): 每一个stage的长度
        stage_wise_range (list of list): 每一个stage的sample的范围

    Returns:
        search_settings (list): 采样后的结果，len(search_settings) equals len(stage_wise_length)
    """
    search_settings = []
    for _, curr_range in zip(stage_wise_length, stage_wise_range):
        curr = random.choice(curr_range)
        if search_settings:
            while curr < search_settings[-1]:
                curr = random.choice(curr_range)
            search_settings += [curr]
        else:
            search_settings += [curr]
    return search_settings


def block_wise(stage_wise_length, stage_wise_range):
    """生成搜索的配置

    Args:
        stage_wise_length (list): 每一个stage的长度
        stage_wise_range (list of list): 每一个stage的sample的范围

    Returns:
        search_settings (list): 采样后的结果，len(search_settings) equals sum(stage_wise_length)
    """
    search_settings = []
    for depth, curr_range in zip(stage_wise_length, stage_wise_range):
        for _ in range(depth):
            curr = random.choice(curr_range)
            search_settings.append(curr)
    return search_settings


def ordered_block_wise(stage_wise_length, stage_wise_range):
    """生成搜索的配置

    Args:
        stage_wise_length (list): 每一个stage的长度
        stage_wise_range (list of list): 每一个stage的sample的范围

    Returns:
        search_settings (list): 采样后的结果，len(search_settings) equals sum(stage_wise_length)
    """
    search_settings = []
    for depth, curr_range in zip(stage_wise_length, stage_wise_range):
        for i in range(depth):
            curr = random.choice(curr_range)
            if search_settings:
                while curr < search_settings[-1]:
                    curr = random.choice(curr_range)
                search_settings.append(curr)
            else:
                search_settings.append(curr)
    return search_settings


def sample_search_settings(stage_wise_length, stage_wise_range, sample_strategy):
    """生成搜索的配置

    Args:
        stage_wise_length (list): 每一个stage的长度
        stage_wise_range (list of list): 每一个stage的sample的范围
        sample_strategy (str): 采样的方式

    Returns:
        search_settings (list): 采样后的结果
    """
    if sample_strategy == 'net_wise':
        return net_wise(stage_wise_length, stage_wise_range)
    elif sample_strategy == 'stage_wise':
        return stage_wise(stage_wise_length, stage_wise_range)
    elif sample_strategy == 'ordered_stage_wise':
        return ordered_stage_wise(stage_wise_length, stage_wise_range)
    elif sample_strategy == 'stage_wise_depth':
        return stage_wise_depth(stage_wise_length, stage_wise_range)
    elif sample_strategy == 'ordered_stage_wise_depth':
        return ordered_stage_wise_depth(stage_wise_length, stage_wise_range)
    elif sample_strategy == 'block_wise':
        return block_wise(stage_wise_length, stage_wise_range)
    elif sample_strategy == 'ordered_block_wise':
        return ordered_block_wise(stage_wise_length, stage_wise_range)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    stage_wise_length = [1, 2, 3, 4]
    stage_wise_range = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    for method in ['net_wise', 'stage_wise', 'ordered_stage_wise',
                   'stage_wise_depth', 'ordered_stage_wise_depth', 'block_wise', 'ordered_block_wise']:
        print(method, sample_search_settings(stage_wise_length, stage_wise_range, method))
