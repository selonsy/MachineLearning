#define _CRT_SECURE_NO_WARNINGS  /*����ʹ��scanf������������*/
#include<iostream>
#include<algorithm>
using namespace std;
/*
�������⣺��n���������ݣ������ܳ�������w������n���˵�����������ô�����õ���ʹ�����Ρ�
������
���룺
6 11
1 2 4 7 9 10
�����
5
��˳��Ϊ2 10 4 9 7 1ʱ����Ҫʹ��5�ε���
�ⷨ��
̰��ÿ��ö�ٴ���w��һ��(l, r)����ǰ�棬�ض�Ҫ�ֳ�����������
*/
const int N = 100010;
static int a[N];
//��
int main2019_4_3_21_01_26() {
	//ջ
	int n, w;
	scanf("%d%d", &n, &w);
	for (int i = 0; i < n; i++) {
		scanf("%d", &a[i]);
	}
	sort(a, a + n);
	int l = 0, r = n - 1, ans = 0;
	while (l < r) {
		if (a[l] + a[r] > w) {
			ans += 2;
			l++; r--;
		}
		else {
			l++;
		}
	}
	ans += (n - ans) / 2 + (n - ans) % 2;
	printf("%d\n", ans);
	system("pause");
	return 0;
}
