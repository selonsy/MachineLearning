#define _CRT_SECURE_NO_WARNINGS  /*����ʹ��scanf������������*/
#define gets(str) gets_s(str)
#include<iostream>
#include<algorithm>
#include<string>

/*
���ߣ�lienus
���ӣ�https://www.nowcoder.com/discuss/172765
��Դ��ţ����

�ڶ����⣺����n����������Ԫ��(ai,aj,ak)���Լ������1�ĸ�����
����n<=99999,0 <= a[i]<=99999
������
���룺
8
1 2 3 4 5 6 7 8
�����
52
�����֪��Ī����˹���ݣ��ڶ����Ѷȸо���ǰ�������������Ҫ�󡣡���
��ʵĪ����˹���ݵ�����ô���£������ң��ٶ�ȥ������
����NlogN���
*/

using namespace std;
const int N = 100000;
bool vis[N];
int mu[N], prime[N];
//���Ī����˹����
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
//�������һ�½��
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