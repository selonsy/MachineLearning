//
// Created by shenjl on 2019/5/16.
//
#include<iostream>
// #include<limits.h>
#include<climits>
#include<string>
#include<cmath>
using namespace std;

void coding_c_2_1()
{
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
    cin>>n;
    int sum = 0;
    int max = INT_MIN;    // -2147483648
    int min = INT_MAX;    // 2147483647
    int m;
    for (int i = 0; i < n; i++)
    {
        cin>>m;
        sum += m;
        if(m>max)max = m;
        if(m<min)min = m;
    }
    cout<<sum<<" "<<min<<" "<<max<<endl;
}
void coding_c_2_2()
{
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
    string str;
    getline(cin,str);
    int res = 0;
    int length = str.length();
    for (int i = 0; i < length ; ++i)
    {
//        cout<<str[i]-'0'<<endl;
//        cout<<pow(2,length-i-1)<<endl;
        res += (str[i]-'0') * pow(2,length-i-1);
    }
    cout<<res<<endl;
}
int main()
{
//    cout<<"Hello World!"<<endl;
    coding_c_2_2();
//    system("pause");
    return 0;
}
