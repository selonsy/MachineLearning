//
// Created by devin on 2019/5/23.
//
#define _CRT_SECURE_NO_WARNINGS  /*可以使用scanf，必须放在最顶上*/
#include<iostream>
#include<algorithm>
using namespace std;
/*
第三题意：有n个人坐电梯，电梯能承受最重w，给定n个人的重量，求怎么排列让电梯使用最多次。
样例：
输入：
6 11
1 2 4 7 9 10
输出：
5
当顺序为2 10 4 9 7 1时，需要使用5次电梯
解法：
贪心每次枚举大于w的一对(l, r)排在前面，必定要分成两次坐电梯
*/
const int N = 100010;
static int a[N];
//堆
int main2019_4_3_21_01_26() {
    //栈
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

