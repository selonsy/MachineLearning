#include <stdio.h>
/* 
  题目：长度为n的可重复数组输出全排列
  思路：
    1.n个元素的全排列=（1个元素作为前缀）+（n-1元素进行全排列）；
    2.出口：只有一个元素时，输出整个数组；
    3.将n个元素的素组中的每个元素与第一个元素进行置换，剩余n-1个元素进行全排列；
*/

void swap(int list[], int i, int j)
{
	int temp = list[i];
	list[i] = list[j];
	list[j] = temp;
}
void perm(int list[], int i, int len)
{
	if(i==len-1){
		for(int j=0; j<len; j++)
			printf("%d", list[j]);
		printf("\n");
		return ;
	}
	for(int j=i; j<len; j++){
		if(list[i]!=list[j] || i==j){
			// 若不是同一位值上的值相等，则为数组中重复的值，无需置换再排列
			swap(list, i, j);
			perm(list, i+1, len);
			swap(list, i, j);
		}
	}
}
int main(void)
{
	int A[] = {1,2,3,2};
	int len = sizeof(A)/sizeof(int);
	perm(A,0,len);

	return 1;
}

