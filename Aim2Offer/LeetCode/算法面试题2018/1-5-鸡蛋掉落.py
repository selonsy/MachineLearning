import functools
@functools.lru_cache(maxsize=None)
def f(K, N):
    if K == 0:
        return 0
    if N == 1:
        return K

    ans = min([max([f(i - 1, N - 1), f(K - i, N)]) for i in range(1, K + 1)]) + 1
    return ans

print(f(100, 2))	# 14
print(f(200, 2))
print(f(1, 2))