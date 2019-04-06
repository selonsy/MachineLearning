
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
	// 1.����֮��-Two Sum
	vector<int> twoSum_1(vector<int>& nums, int target) 
	{
		/*
			����һ���������� nums ��һ��Ŀ��ֵ target�������ڸ��������ҳ���ΪĿ��ֵ���� ���� ���������������ǵ������±ꡣ
			����Լ���ÿ������ֻ���Ӧһ���𰸡����ǣ��㲻���ظ��������������ͬ����Ԫ�ء�
			ʾ��:
			���� nums = [2, 7, 11, 15], target = 9
			��Ϊ nums[0] + nums[1] = 2 + 7 = 9
			���Է��� [0, 1]
		*/
		// �����ⷨ����ʱ550ms
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
		// �ռ任ʱ�䣺16ms
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

	// 2.�������-Add Two Numbers
	ListNode* addTwoNumbers_1(ListNode* l1, ListNode* l2) {
		/*
		�������� �ǿ� ������������ʾ�����Ǹ������������У����Ǹ��Ե�λ���ǰ��� ���� �ķ�ʽ�洢�ģ��������ǵ�ÿ���ڵ�ֻ�ܴ洢 һλ ���֡�
		��������ǽ��������������������᷵��һ���µ���������ʾ���ǵĺ͡�
		�����Լ���������� 0 ֮�⣬���������������� 0 ��ͷ��
		ʾ����
		���룺(2 -> 4 -> 3) + (5 -> 6 -> 4)
		�����7 -> 0 -> 8
		ԭ��342 + 465 = 807
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
		// addTwoNumbers_1�ľ���汾
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

	// 3.���ظ��ַ�����Ӵ�-Longest Substring Without Repeating Characters
	int lengthOfLongestSubstring_1(string s) {
		/*
		����һ���ַ����������ҳ����в������ظ��ַ��� ��Ӵ� �ĳ��ȡ�

		ʾ�� 1:
		����: "abcabcbb"
		���: 3 
		����: ��Ϊ���ظ��ַ�����Ӵ��� "abc"�������䳤��Ϊ 3��

		ʾ�� 2:
		����: "bbbbb"
		���: 1
		����: ��Ϊ���ظ��ַ�����Ӵ��� "b"�������䳤��Ϊ 1��

		ʾ�� 3:
		����: "pwwkew"
		���: 3
		����: ��Ϊ���ظ��ַ�����Ӵ��� "wke"�������䳤��Ϊ 3��
			 ��ע�⣬��Ĵ𰸱����� �Ӵ� �ĳ��ȣ�"pwke" ��һ�������У������Ӵ���
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
		// ʹ��ascii����2^8=256
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

#pragma region ��̬�滮-Dynamic Programming
	// 395.������K���ظ��ַ�����Ӵ�
	int longestSubstring_1(string s, int k) {
		/*
		�ҵ������ַ�������Сд�ַ���ɣ��е���Ӵ� T �� Ҫ�� T �е�ÿһ�ַ����ִ����������� k ����� T �ĳ��ȡ�

		ʾ�� 1:

		����:
		s = "aaabb", k = 3

		���:
		3

		��Ӵ�Ϊ "aaa" ������ 'a' �ظ��� 3 �Ρ�
		ʾ�� 2:

		����:
		s = "ababbc", k = 2

		���:
		5

		��Ӵ�Ϊ "ababb" ������ 'a' �ظ��� 2 �Σ� 'b' �ظ��� 3 �Ρ�
		*/
		int res = 0, i = 0, n = s.size();
		while (i + k <= n) {
			int m[26] = { 0 }, mask = 0, max_idx = i;
			for (int j = i; j < n; ++j) {
				int t = s[j] - 'a';
				++m[t];
				if (m[t] < k) mask |= (1 << t);  // '|=' ��λ���ֵ; 1 << t ����tλ
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
		1��in the first pass I record counts of every character in a hashmap
		2��in the second pass I locate the first character that appear less than k times in the string. 
			this character is definitely not included in the result, and that separates the string into two parts.
		3��keep doing this recursively and the maximum of the left/right part is the answer.
		*/
		// ��ʱ��

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



	// 3.���ظ��ַ�����Ӵ�-Longest Substring Without Repeating Characters
	/*string s = "pwwkew";
	int res = solution.lengthOfLongestSubstring_1(s);
	cout << res << endl;;
	assert(res == 3);*/

	// 2.�������-Add Two Numbers
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

	// 1.����֮��-Two Sum
	/*vector<int> nums_1 = { 2, 7, 11, 15 };
	vector<int> res_1 = solution.twoSum_2(nums_1,9);
	vector<int> case_1 = { 0,1 };
	assert(res_1 == case_1);*/

	cout << "ok" << endl;
	system("pause");
	return 0;
}