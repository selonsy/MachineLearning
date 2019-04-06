#include<malloc.h>

/*
NO 1. Accepted Runtime: 93 ms
Given an array of integers, return indices of the two numbers such that they add up to a specific target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

Note: The returned array must be malloced, assume caller calls free().
返回的数组必须被分配，假设调用者负责释放

调用示例:
int nums[] = {2,7,11,15};
int *p  = twoSum(nums, 4, 9);
free(p);
*/
int * twoSum(int * nums, int numsSize, int target)
{
	int i, j;
	int* arr = (int*)malloc(2 * sizeof(int));;
	int index1, index2;
	int flag = 0;
	for (i = 0; i < numsSize; i++)
	{
		for (j = i + 1; j < numsSize; j++)
		{
			if (nums[i] + nums[j] == target)
			{
				arr[0] = i;
				arr[1] = j;
				return arr;
			}
		}
	}
	return NULL;
}

/*
NO 2. Accepted Runtime:37ms
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.

* Definition for singly-linked list.
* struct ListNode {
*     int val;
*     struct ListNode *next;
* };
*/

struct ListNode {
	int data;
	struct ListNode *next;	
};
struct ListNode* addTwoNumbers(struct ListNode* l1, struct ListNode* l2) {
	struct ListNode* p = (struct ListNode*)malloc(sizeof(struct ListNode));
	struct ListNode* q = p;
	p->next = NULL;
	int flag = 0, first = 0;;
	while (l1 != NULL || l2 != NULL)
	{
		int _val = (l1 != NULL ? l1->data : 0) + (l2 != NULL ? l2->data : 0) + flag;
		int val = _val % 10;
		if (_val >= 10)flag = 1;
		else flag = 0;
		struct ListNode* node = (struct ListNode*)malloc(sizeof(struct ListNode));

		if (p->next == NULL&&first == 0) {
			p->data = val; first = 1;
		}
		else {
			node->data = val;
			q->next = node;
			q = node;
		}
		if (l1 != NULL)l1 = l1->next;
		if (l2 != NULL)l2 = l2->next;
	}
	if (flag == 0)q->next = NULL;
	else {
		struct ListNode* end = (struct ListNode*)malloc(sizeof(struct ListNode));
		end->data = 1;
		end->next = NULL;
		q->next = end;
	}
	return p;
}

/*
NO 3. Accepted Runtime:36 ms
Given a string, find the length of the longest substring without repeating characters.
Examples:
Given "abcabcbb", the answer is "abc", which the length is 3.
Given "bbbbb", the answer is "b", with the length of 1.
Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
*/
int lengthOfLongestSubstring(char* s) {
	int exist[256];
	int i, j, k;
	for (k = 0; k < 256; k++)
	{
		exist[k] = 0;
	}
	int max = 0;

	for (i = 0; s[i] != NULL; i++)
	{
		exist[s[i]] = 1;
		int count = 1;
		for (j = i + 1; s[j] != NULL; j++)
		{
			if (exist[s[j]] != 1)
			{
				exist[s[j]] = 1;
				count++;
			}
			else
			{
				for (k = 0; k < 256; k++)
				{
					exist[k] = 0;
				}
				break;
			}
		}
		max = count > max ? count : max;
	}
	return max;
}

/*
NO.4 Accepted Runtime:36 ms
There are two sorted arrays nums1 and nums2 of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

Example 1:
nums1 = [1, 3]
nums2 = [2]

The median is 2.0


Example 2:
nums1 = [1, 2]
nums2 = [3, 4]

The median is (2 + 3)/2 = 2.5
*/
double findMedianSortedArrays(int* nums1, int nums1Size, int* nums2, int nums2Size) {
	return 0.0;
}
