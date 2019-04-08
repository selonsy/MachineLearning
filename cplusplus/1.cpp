
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

	int func()
	{
				
		return 0;
	}
	
};

int main()
{
	//ios::sync_with_stdio(false);

	Solution solution;

	char s[100];
	scanf("%[^\n]", &s);
	printf(s);

	int res = solution.func();
	
	assert(res == 0);

	cout << "ok" << endl;
	system("pause");
	return 0;
}