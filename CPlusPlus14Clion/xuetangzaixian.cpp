//
// Created by shenjl on 2019/5/16.
//
#include<iostream>
// #include<limits.h>
#include<climits>
#include<string>
#include<cmath>

using namespace std;

void coding_c_2_1() {
    /*
    题目描述
    任意给定 n 个整数，求这 n 个整数序列的和、最小值、最大值

    输入描述
    输入一个整数n，代表接下来输入整数个数，0 < n <= 100，接着输入n个整数，整数用int表示即可。

    输出描述
    输出整数序列的和、最小值、最大值。用空格隔开，占一行

    样例输入
    2
    1 2

    样例输出
    3 1 2
     */
    int n;
    cin >> n;
    int sum = 0;
    int max = INT_MIN;    // -2147483648
    int min = INT_MAX;    // 2147483647
    int m;
    for (int i = 0; i < n; i++) {
        cin >> m;
        sum += m;
        if (m > max)max = m;
        if (m < min)min = m;
    }
    cout << sum << " " << min << " " << max << endl;
}

void coding_c_2_2() {
    /*
    题目描述
    已知一个只包含 0 和 1 的二进制数，长度不大于 10 ，将其转换为十进制并输出。

    输入描述
    输入一个二进制整数n，其长度大于0且不大于10
    输出描述
    输出转换后的十进制数， 占一行

    样例输入
    110
    样例输出
    6
     */
    string str = "110";
//    getline(cin,str);
    int res = 0;
    int length = str.length();
    for (int i = 0; i < length; ++i) {
//        cout<<str[i]-'0'<<endl;
//        cout<<pow(2,length-i-1)<<endl;
        res += (str[i] - '0') * pow(2, length - i - 1);
    }
    cout << res << endl;
}

void coding_c_2_3() {
    /*
    题目描述
    打印 n 阶实心菱形

    输入描述
    输入一个整数n，0 < n <= 10

    输出描述
    输出 n 阶实心菱形 ， 占 2*n-1 行

    样例输入
    3
    样例输出
      *
     ***
    *****
     ***
      *
    */
    // 好吧，需要分成上下两个部分分别打印，煞笔了
    int n;
    cin >> n;
    if (n > 10 || n <= 0) {
        cout << "" << endl;
        return;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            cout << " ";
        }
        for (int k = 0; k < 2 * (i + 1) - 1; ++k) {
            cout << "*";
        }
        cout << endl;
    }
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < i + 1; ++j) {
            cout << " ";
        }
        for (int k = 0; k < 2 * (n - 1) - (2 * i + 1); ++k) {
            cout << "*";
        }
        cout << endl;
    }
}

void coding_c_6_1() {
    /*

    最大子数组和

    题目描述
    给定一个数组a[0,...,n-1],求其最大子数组(长度>=1)和

    输入描述
    第一行一个整数n(1<=n<=5000)，然后依次输入n个整数（每个整数范围[-5000, 5000]）

    输出描述
    输出一个整数表示最大子数组和

    样例输入
    5
    1 -1 1 1 -1

    样例输出
    2
     */
    int n;
    cin >> n;
    int num[n];
    for (int i = 0; i < n; i++) {
        cin >> num[i];
    }
    int max = num[0], sum = num[0];
    for (int j = 1; j < n; ++j) {
        sum += num[j];
        max = sum > max ? sum : max;
        if (sum < 0) {
            sum = 0;
        }
    }
    cout << max << endl;
}

void coding_c_6_2() {
    /*

    字符串的回文子序列个数

    题目描述
    求一个长度不超过15的字符串的回文子序列个数（子序列长度>=1）。

    输入描述
    输入一个长度不超过15的字符串,字符串均由小写字母表示

    输出描述
    输出其回文子序列个数

    样例输入
    abaa

    样例输出
    10

    注释:本例中其所有回文子序列为：
    a,b,a,a,aba,aba,aa,aa,aa,aaa
    一个字符串的子序列是指在原字符串上去除某些字符但不破坏余下元素的相对位置（在前或在后）而形成的新字符串。
     */
    string str;
    getline(cin,str);
    int res = 0;

    cout << res << endl;
}

int main() {
//    cout<<"Hello World!"<<endl;
    coding_c_6_2();
//    system("pause");
    return 0;
}
