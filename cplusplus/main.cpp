/*
* leetcode c++版本刷题集合
* author：selonsy
* time：2019_4_17_15_13_26
*/
#define _CRT_SECURE_NO_WARNINGS
#include<vector>
#include<iostream>
#include<assert.h>
#include<unordered_map>
#include<map>
#include<string>
#include<algorithm>
using namespace std;

struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}	
};

class Solution {
public:
	// 1.两数之和-Two Sum
	vector<int> twoSum_1(vector<int>& nums, int target) 
	{
		/*
			给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
			你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。
			示例:
			给定 nums = [2, 7, 11, 15], target = 9
			因为 nums[0] + nums[1] = 2 + 7 = 9
			所以返回 [0, 1]
		*/
		// 暴力解法：耗时550ms
		vector<int> res;
		for (int i = 0; i < nums.size()-1; i++)
		{
			for (int j = i+1; j < nums.size(); j++)
			{
				if (nums[i] + nums[j] == target) 
				{
					res = { i,j };					
				}
			}
		}
		return res;
	}
	vector<int> twoSum_2(vector<int>& nums, int target)
	{
		// 空间换时间：16ms
		unordered_map <int, int> ans;
		for (int i = 0; i < nums.size(); i++) 
		{
			int res = target - nums[i];
			if (ans.count(res))
				return vector<int>({ ans[res],i });
			else
				ans[nums[i]] = i;
		}
		return vector<int>({});
	}

	// 2.两数相加-Add Two Numbers
	ListNode* addTwoNumbers_1(ListNode* l1, ListNode* l2) {
		/*
		给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
		如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
		您可以假设除了数字 0 之外，这两个数都不会以 0 开头。
		示例：
		输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
		输出：7 -> 0 -> 8
		原因：342 + 465 = 807
		*/
		// 40ms 10.2MB 68%
		ListNode * res = NULL,* current = NULL;
		int jin_wei = 0;
		while (l1 != NULL || l2 != NULL)
		{
			if (l1 != NULL && l2 != NULL)
			{
				int s = l1->val + l2->val + jin_wei;
				if (s >= 10)
				{
					jin_wei = 1;
					s = s - 10;
				}
				else
				{
					jin_wei = 0;
				}
				ListNode * p = new ListNode(s);
				if (res == NULL)
				{
					res = p;
					current = p;
				}
				else
				{
					current->next = p;
					current = current->next;
				}
				l1 = l1->next;
				l2 = l2->next;
			}
			else if (l1 == NULL && l2 != NULL)
			{
				while (l2 != NULL)
				{
					int s = l2->val + jin_wei;
					if (s >= 10)
					{
						jin_wei = 1;
						s = s - 10;
					}
					else
					{
						jin_wei = 0;
					}
					ListNode * p = new ListNode(s);
					if (res == NULL)
					{
						res = p;
						current = p;
					}
					else
					{
						current->next = p;
						current = current->next;
					}
					l2 = l2->next;
				}
			}
			else if (l1 != NULL && l2 == NULL)
			{
				while (l1 != NULL)
				{
					int s = l1->val + jin_wei;
					if (s >= 10)
					{
						jin_wei = 1;
						s = s - 10;
					}
					else
					{
						jin_wei = 0;
					}
					ListNode * p = new ListNode(s);
					if (res == NULL)
					{
						res = p;
						current = p;
					}
					else
					{
						current->next = p;
						current = current->next;
					}
					l1 = l1->next;
				}
			}
		}
		if (jin_wei != 0)
		{
			ListNode * p = new ListNode(jin_wei);
			current->next = p;
			current = current->next;
		}

		return res;
	}
	ListNode* addTwoNumbers_2(ListNode* l1, ListNode* l2) {		
		// addTwoNumbers_1的精简版本
		ListNode * res = NULL, *current = NULL;
		int jin_wei = 0;
		while (l1 != NULL || l2 != NULL)
		{
			int s = 0;
			if (l1 != NULL && l2 != NULL)
			{
				s = l1->val + l2->val + jin_wei;				
				l1 = l1->next;
				l2 = l2->next;
			}
			else if (l1 == NULL && l2 != NULL)
			{
				s = l2->val + jin_wei;
				l2 = l2->next;
			}
			else if (l1 != NULL && l2 == NULL)
			{
				s = l1->val + jin_wei;
				l1 = l1->next;				
			}
			if (s >= 10)
			{
				jin_wei = 1;
				s = s - 10;
			}
			else
			{
				jin_wei = 0;
			}
			ListNode * p = new ListNode(s);
			if (res == NULL)
			{
				res = p;
				current = p;
			}
			else
			{
				current->next = p;
				current = current->next;
			}
		}
		if (jin_wei != 0)
		{
			ListNode * p = new ListNode(jin_wei);
			current->next = p;
			current = current->next;
		}
		return res;
	}

	// 3.无重复字符的最长子串-Longest Substring Without Repeating Characters
	int lengthOfLongestSubstring_1(string s) {
		/*
		给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

		示例 1:
		输入: "abcabcbb"
		输出: 3 
		解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

		示例 2:
		输入: "bbbbb"
		输出: 1
		解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

		示例 3:
		输入: "pwwkew"
		输出: 3
		解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
			 请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
		*/
		// 1000ms,150MB
		int len = 0,t = 0;
		unordered_map<char, int> map;
		for (int i = 0; i < s.length(); i++)
		{			
			if (map.count(s[i]))
			{
				i = i - t;
				t = 0;								
				map.clear();
			}
			else
			{
				map[s[i]] = 1;
				t++;
			}
			len = t > len ? t : len;
		}
		return len;
	}
	int lengthOfLongestSubstring_2(string s) {
		// 使用ascii编码2^8=256
		// 28ms,10.3MB
		vector<int> dict(256, -1);
		int maxLen = 0, start = -1;
		for (int i = 0; i != s.length(); i++) {
			if (dict[s[i]] > start)
				start = dict[s[i]];
			dict[s[i]] = i;
			maxLen = max(maxLen, i - start);
		}
		return maxLen;
	}

#pragma region 动态规划-Dynamic Programming
	// 395.至少有K个重复字符的最长子串
	int longestSubstring_1(string s, int k) {
		/*
		找到给定字符串（由小写字符组成）中的最长子串 T ， 要求 T 中的每一字符出现次数都不少于 k 。输出 T 的长度。

		示例 1:

		输入:
		s = "aaabb", k = 3

		输出:
		3

		最长子串为 "aaa" ，其中 'a' 重复了 3 次。
		示例 2:

		输入:
		s = "ababbc", k = 2

		输出:
		5

		最长子串为 "ababb" ，其中 'a' 重复了 2 次， 'b' 重复了 3 次。
		*/
		int res = 0, i = 0, n = s.size();
		while (i + k <= n) {
			int m[26] = { 0 }, mask = 0, max_idx = i;
			for (int j = i; j < n; ++j) {
				int t = s[j] - 'a';
				++m[t];
				if (m[t] < k) mask |= (1 << t);  // '|=' 按位或后赋值; 1 << t 左移t位
				else mask &= (~(1 << t));
				if (mask == 0) {
					res = max(res, j - i + 1);
					max_idx = j;
				}
			}
			i = max_idx + 1;
		}
		return res;
	}
	int longestSubstring_2(string s, int k) {
		/*
		1、in the first pass I record counts of every character in a hashmap
		2、in the second pass I locate the first character that appear less than k times in the string. 
			this character is definitely not included in the result, and that separates the string into two parts.
		3、keep doing this recursively and the maximum of the left/right part is the answer.
		*/
		// 超时了

		if (s.size() == 0 || k > s.size())   return 0;
		if (k == 0)  return s.size();

		unordered_map<char, int> Map;
		for (int i = 0; i < s.size(); i++) {
			Map[s[i]]++;
		}

		int idx = 0;
		while (idx < s.size() && Map[s[idx]] >= k)    
			idx++;
		if (idx == s.size()) return s.size();

		int left = longestSubstring_2(s.substr(0, idx), k);
		int right = longestSubstring_2(s.substr(idx + 1), k);

		return max(left, right);

	}
	int longestSubstring_3(string s, int k) {
		/*
		longestSubstring_2 TLE,change the type of Map to int Maps[26], the solution will be accepted.
		*/
		// 380ms 54.6MB
		if (s.size() == 0 || k > s.size())   return 0;
		if (k == 0)  return s.size();

		int Map[26] = { 0 };
		for (int i = 0; i < s.size(); i++) {
			Map[s[i] - 'a']++;
		}

		int idx = 0;
		while (idx < s.size() && Map[s[idx] - 'a'] >= k)    idx++;
		if (idx == s.size()) return s.size();

		int left = longestSubstring_3(s.substr(0, idx), k);
		int right = longestSubstring_3(s.substr(idx + 1), k);

		return max(left, right);

	}
#pragma endregion

};

int main()
{
	Solution solution;
	printf("Hello World!");


	// 3.无重复字符的最长子串-Longest Substring Without Repeating Characters
	/*string s = "pwwkew";
	int res = solution.lengthOfLongestSubstring_1(s);
	cout << res << endl;;
	assert(res == 3);*/

	// 2.两数相加-Add Two Numbers
	/*vector<int> arr1 = { 2,4,3 };
	vector<int> arr2 = { 5,6,4 };
	ListNode * l1_head = new ListNode(arr1[0]), *l2_head = new ListNode(arr2[0]);
	ListNode * l1_p = l1_head, *l2_p = l2_head;
	for (int i = 1; i < arr1.size() ; i++)
	{		
		ListNode * p = new ListNode(arr1[i]);
		l1_p->next = p;
		l1_p = l1_p->next;
	}
	l1_p = l1_head;
	while (l1_p != NULL)
	{
		cout << l1_p->val << " ";
		l1_p = l1_p->next;
	}
	for (int i = 1; i < arr2.size(); i++)
	{		
		ListNode * p = new ListNode(arr2[i]);
		l2_p->next = p;
		l2_p = l2_p->next;
	}
	l2_p = l2_head;
	while (l2_p != NULL)
	{
		cout << l2_p->val << " ";
		l2_p = l2_p->next;
	}
	ListNode * res = solution.addTwoNumbers_2(l1_head, l2_head);
	while (res != NULL) 
	{
		cout << res->val << " ";
		res = res->next;
	}*/

	// 1.两数之和-Two Sum
	/*vector<int> nums_1 = { 2, 7, 11, 15 };
	vector<int> res_1 = solution.twoSum_2(nums_1,9);
	vector<int> case_1 = { 0,1 };
	assert(res_1 == case_1);*/

	cout << "ok" << endl;
	system("pause");
	return 0;
}