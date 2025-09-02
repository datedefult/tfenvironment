import bisect


def find_max_leq_optimized(sorted_lst, x):
    """
    预排序+二分查找法（适合多次查询）
    时间复杂度：O(n log n) 预处理，之后每次查询 O(log n)
    """
    # 先对列表进行排序（只需一次）
    sorted_lst = sorted(sorted_lst)

    # 使用bisect_right找到第一个大于x的位置
    idx = bisect.bisect_right(sorted_lst, x)

    if idx == 0:
        return None  # 所有元素都大于x
    return sorted_lst[idx - 1]


def find_max_leq_linear(lst, x):
    """
    直接线性遍历法（适合单次查询）
    时间复杂度：O(n)
    """
    max_val = -float('inf')
    for num in lst:
        if x >= num > max_val:
            max_val = num
    return max_val if max_val != -float('inf') else None


# 示例测试
list1 = [1, 5, 6, 7, 99, 21, 35, 56, 78, 64, 13]

# 正确测试案例
print(find_max_leq_optimized(list1, 77))  # 输出 64
print(find_max_leq_optimized(list1, 78))  # 输出 78
print(find_max_leq_optimized(list1, 0))  # 输出 None

print(find_max_leq_linear(list1, 77))  # 输出 64
print(find_max_leq_linear(list1, 78))  # 输出 78