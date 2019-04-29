'''

给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。

说明:

初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。
示例:

输入:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

输出: [1,2,2,3,5,6]
'''
class Solution:
    #def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    #    """
    #    Do not return anything, modify nums1 in-place instead.
    #    """
    def merge1(self, nums1, m, nums2, n):
        """
        Do not return anything, modify nums1 in-place instead.
        """            
        nums1[:m].extend(nums2[:n])  # 此方法改变了nums1的指向,无效
        # extend没改变指向，但是切片改了        
        nums1[:m] + nums2[:n]        # 此方法改变了nums1的指向,无效
        # +号改变了指向

        nums1.sort()

    def merge2(self, nums1, m, nums2, n):
        """
        Do not return anything, modify nums1 in-place instead.
        """     
        # 从nums1的第m个数字开始，用nums2的值来替换
        for i in range(n):
            nums1[m+i] = nums2[i]
        nums1.sort()

s = Solution()
nums1 = [1,2,3,0,0,0]
x = id(nums1)
s.merge2(nums1,3,[2,5,6],3)
print(nums1)