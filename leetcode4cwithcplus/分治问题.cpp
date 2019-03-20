#include<stdio.h>

//分治思想
void maxmin(int A[], int n, int *max, int *min)
{
	int i;
	*min = *max = A[0];
	for (i = 0; i < n; i++)
	{
		if (A[i] > *max) *max = A[i];
		if (A[i] < *min) *min = A[i];
	}
}
void maxmin_fenzhi(int A[], int i, int j, int *max, int *min)
{
	/*
	用分治策略比较的算法如下：
	void maxmin2(int A[],int i,int j,int *max,int *min)
	A存放输入的数据，i，j存放数据的范围，初值为0，n-1，*max,*min 存放最大和最小值
	{ int mid, max1, max2, min1, min2;
	if (j == i) { 最大和最小值为同一个数; return; }
	if (j - 1 == i) { 将两个数直接比较，求得最大和最小值；return； }
	mid = (i + j) / 2;
	求i~mid之间的最大最小值分别为max1，min1;
	求mid + 1~j之间的最大最小值分别为max2，min2;
	比较max1和max2，大的就是最大值;
	比较min1和min2，小的就是最小值;
	}
	*/
	if (j == i) { *max = A[i]; *min = A[i]; return; }
	if (j - 1 == i) { *max = A[i] > A[j] ? A[i] : A[j]; *min = A[i] < A[j] ? A[i] : A[j]; return; }
	int mid = (i + j) / 2;
	int max1 = A[i], min1 = A[i], max2 = A[mid + 1], min2 = A[mid + 1];
	maxmin_fenzhi(A, i, mid, &max1, &min1);
	maxmin_fenzhi(A, mid + 1, j, &max2, &min2);

	*max = max1 > max2 ? max1 : max2;
	*min = min1 < min2 ? min1 : min2;
}
void fenzhi()
{
	/*
	分治算法的基本思想是将一个规模为N的问题分解为K个规模较小的子问题，这些子问题相互独立且与原问题性质相同。
	求出子问题的解，就可得到原问题的解。即一种分目标完成程序算法，简单问题可用二分法完成。

	应用举例:
	1.求数组的最大最小值
	*/

	int A[8] = { -13,13,9, -5,7,23,0,15 };

	//传统方式,比较次数为2n次.
	int max = A[0], min = A[0];
	maxmin(A, 8, &max, &min);
	printf("the max is %d,and the min is %d\n", max, min);

	//分治方式,比较次数为次.
	maxmin_fenzhi(A, 0, 7, &max, &min);
	printf("the max is %d,and the min is %d\n", max, min);

}
