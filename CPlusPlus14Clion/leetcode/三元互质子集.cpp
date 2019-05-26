//
// Created by devin on 2019/5/23.
//
#define _CRT_SECURE_NO_WARNINGS  /*可以使用scanf，必须放在最顶上*/
#define gets(str) gets_s(str)
#include<iostream>
#include<algorithm>
#include<string>

/*
作者：lienus
链接：https://www.nowcoder.com/discuss/172765
来源：牛客网

第二题意：给定n个数，求三元组(ai,aj,ak)最大公约数等于1的个数。
其中n<=99999,0 <= a[i]<=99999
样例：
输入：
8
1 2 3 4 5 6 7 8
输出：
52
如果不知道莫比乌斯反演，第二题难度感觉比前面两题加起来还要大。。。
其实莫比乌斯反演到底怎么回事，别问我，百度去。。。
最终NlogN求解
*/

using namespace std;
const int N = 100000;
bool vis[N];
int mu[N], prime[N];
//求解莫比乌斯函数
void Moblus()
{
    memset(vis, false, sizeof(vis));
    mu[1] = 1;
    int tot = 0;
    for (int i = 2; i <= N; i++) {
        if (!vis[i]) {
            prime[tot++] = i;
            mu[i] = -1;
        }
        for (int j = 0; j < tot; j++) {
            if (i * prime[j] > N)break;
            vis[i * prime[j]] = true;
            if (i % prime[j] == 0) {
                mu[i * prime[j]] = 0;
                break;
            }
            else {
                mu[i * prime[j]] = -mu[i];
            }
        }
    }
}
long long num[N], cnt[N];
static int a[N];
int gcd(int x, int y) {
    return x % y == 0 ? y : gcd(y, x % y);
}
//暴力检查一下结果
int check(int n) {
    int res = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            for (int k = j + 1; k < n; k++) {
                if (gcd(a[i], gcd(a[j], a[k])) == 1)res++;
            }
        }
    }
    return res;
}
int main2019_4_4_15_16_32() {
    int n, x;
    Moblus();
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &x);
        a[i] = x;
        num[x]++;
    }
    for (int i = 1; i < N; i++) {
        for (int j = i; j < N; j += i) {
            cnt[i] += num[j];
        }
    }
    long long ans = 0;
    for (int i = 1; i < N; i++) {
        if (cnt[i] < 3)continue;
        ans += mu[i] * cnt[i] * (cnt[i] - 1) * (cnt[i] - 2) / 6;
    }
    printf("%lld\n", ans);
    printf("%d\n", check(n));

    return 0;
}
