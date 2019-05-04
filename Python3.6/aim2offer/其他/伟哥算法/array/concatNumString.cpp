//#include<malloc.h>
#include<vector>
#include<iostream>
#include<string>

/*
    问题：设有n个正整数，将他们连接成一排，组成一个最大的多位整数。如:n=3时，3个整数13,312,343,连成的最大整数为34331213。
    思路：
      1.以字符串数组形式保存要拼接的数字数组；
      2.按照冒泡排序的思想对数组中的字符串进行按前后拼接方式两两比较，每一轮确定“最大”的数字，如"13"+"312"< "312"+"13"，则
        将"312"与"13"置换。
 */

using namespace std;

int main()
{
    int num;
    string temp;
    bool flag;
    
    while(cin >> num){
        //arr = (string *)malloc(sizeof(string *)*num);
        vector<string> arr(num);
        for(int i=0; i<num; i++){
            //scanf("%s", arr[i]);
            cin >> arr[i];
        }
        for(int i=0; i<num; i++){
            flag = false;
            for(int j=num-2; j>=i; j--){
                if(arr[j]+arr[j+1]<arr[j+1]+arr[j]){
                    flag = true;
                    temp = arr[j];
                    arr[j] = arr[j+1];
                    arr[j+1] = temp;
                }
            }
            if(flag==false)
                break;
        }
        for(int i=0; i<num; i++){
            cout << arr[i];
        }
        cout << endl;
    }
    return 0;
}
