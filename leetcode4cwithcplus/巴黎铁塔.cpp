//#define _CRT_SECURE_NO_WARNINGS
#include "pch.h"
#include<iostream>
#include<algorithm>
#include<vector>
#include<ctime>
#include<climits>
#include<set>
#include<string>
#include<queue>
#include<list>
using namespace std;

static string s;

int mainA()
{
	ios::sync_with_stdio(false);
	cin >> s;
	list<char> li;
	for (int i = s.length() - 1; i >= 1; i--)
	{
		li.push_front(s[i]);
		li.push_front(li.back()); li.pop_back();
		if (li.size() % 2 == 0) reverse(li.begin(), li.end());
	
	}
	li.push_front(s[0]);
	for (auto& c : li) cout << c;
	cout << endl;
	system("pause");
	return 0;
}