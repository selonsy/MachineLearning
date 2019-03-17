#define _CRT_SECURE_NO_WARNINGS
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

static int N, S;
static vector<int> ve;
static vector<int> isUsed;
static vector<int> indices;
int main()
{
	ios::sync_with_stdio(false);
	cin >> N >> S;
	ve.clear(); ve.resize(N);
	isUsed.clear(); isUsed.resize(N, 1);
	indices.clear(); indices.resize(N);
	for (int i = 0; i < N; i++) cin >> ve[i];
	sort(ve.begin(), ve.end());
	int tsum = 0; int tr = 0;
	for (int i = 0; i < N; i++)
	{
		if (tsum + ve[i] > ve.back() || isUsed[N - 1] == S)
		{
			for (int j = N - 1; j >= i + 1; j--)
			{
				if (tsum <= ve[j] && isUsed[j] < S)
				{
					++isUsed[j]; indices.push_back();
					break;
				}
			}
			tsum = ve[i]; tr = i;
		}
		else
		{
			tsum += ve[i]; tr = i;
		}
	}
	vector<int> newlist;
	int maxd = -1;
	for (int i = N - 1; i >= 0; i--)
	{
		for (int j = 0; j < indices[i].size(); j++) maxd = max(maxd, indices[i][j].second);
	}
	if (maxd >= 0) ve.erase(ve.begin(), ve.begin() + maxd + 1);
	
	system("pause");
	return 0;
}