#include<stdio.h>

//����˼��
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
	�÷��β��ԱȽϵ��㷨���£�
	void maxmin2(int A[],int i,int j,int *max,int *min)
	A�����������ݣ�i��j������ݵķ�Χ����ֵΪ0��n-1��*max,*min ���������Сֵ
	{ int mid, max1, max2, min1, min2;
	if (j == i) { ������СֵΪͬһ����; return; }
	if (j - 1 == i) { ��������ֱ�ӱȽϣ����������Сֵ��return�� }
	mid = (i + j) / 2;
	��i~mid֮��������Сֵ�ֱ�Ϊmax1��min1;
	��mid + 1~j֮��������Сֵ�ֱ�Ϊmax2��min2;
	�Ƚ�max1��max2����ľ������ֵ;
	�Ƚ�min1��min2��С�ľ�����Сֵ;
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
	�����㷨�Ļ���˼���ǽ�һ����ģΪN������ֽ�ΪK����ģ��С�������⣬��Щ�������໥��������ԭ����������ͬ��
	���������Ľ⣬�Ϳɵõ�ԭ����Ľ⡣��һ�ַ�Ŀ����ɳ����㷨����������ö��ַ���ɡ�

	Ӧ�þ���:
	1.������������Сֵ
	*/

	int A[8] = { -13,13,9, -5,7,23,0,15 };

	//��ͳ��ʽ,�Ƚϴ���Ϊ2n��.
	int max = A[0], min = A[0];
	maxmin(A, 8, &max, &min);
	printf("the max is %d,and the min is %d\n", max, min);

	//���η�ʽ,�Ƚϴ���Ϊ��.
	maxmin_fenzhi(A, 0, 7, &max, &min);
	printf("the max is %d,and the min is %d\n", max, min);

}
