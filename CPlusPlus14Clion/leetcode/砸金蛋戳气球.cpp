//
// Created by devin on 2019/5/23.
//
//#define _CRT_SECURE_NO_WARNINGS  /*可以使用scanf，必须放在最顶上*/
#define gets(str) gets_s(str)
#include<iostream>
#include<algorithm>
using namespace std;
#include<string>
/*
有n个金蛋，每砸破一个可获得 a[left] * a[i] * a[right] 的金钱，默认a[0]和a[n + 1] 等于1.
样例：
输入：
3 1 5 8
输出：
167
3 1 5 8 --> 3 5 8 --> 3 8 --> 8
+15 +120 +24 +8 =167
dp[l][r]代表砸完区间(l,r)内所有蛋蛋只剩下两个边界l和r时获得的最大金钱，在输入的数据前面前后加入一个1记忆化搜索即可
*/


const int N = 510;
int dp[N][N];
static int a[N];
int dfs(int l, int r) {
    if (dp[l][r] != -1)return dp[l][r];
    if (r - l == 1) {
        return dp[l][r] = 0;
    }
    int ans = 0;
    for (int i = l + 1; i < r; i++) {
        ans = max(ans, dfs(l, i) + dfs(i, r) + a[l] * a[i] * a[r]);
    }
    return dp[l][r] = ans;
}
char s[N * 4];
int main2019_4_5_21_00_58() {
    int n = 0, res = 0;
    a[0] = 1;
    gets(s);
    int len = strlen(s);
    for (int i = 0; i < len; i++) {
        if (s[i] >= '0' && s[i] <= '9') {
            res = res * 10 + s[i] - '0';
        }
        else {
            a[++n] = res;
            res = 0;
        }
    }
    a[++n] = res;
    a[++n] = 1;
    memset(dp, -1, sizeof(dp));
    printf("%d\n", dfs(0, n));

    return 0;
}
