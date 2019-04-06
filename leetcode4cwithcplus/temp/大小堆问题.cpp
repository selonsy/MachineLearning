#include<stdio.h>
/*
请判断所给序列是不是堆，如果是，请输出是大顶堆还是小顶堆.
*/
//判断一个序列是否是堆,是什么类型的堆
void judge_heap(int arr[], int size)
{
	int i;
	int flag_b = 1, flag_s = 1;//默认同时是大小顶堆,当然这不可能实现
	for (i = 0; i < size / 2; i++)
	{
		if (((2 * i + 1 < size) && arr[i] < arr[2 * i + 1]) || ((2 * i + 2) < size&&arr[i] < arr[2 * i + 2]))
		{
			flag_b = 0;//不是大顶堆
		}
		if (((2 * i + 1 < size) && arr[i] > arr[2 * i + 1]) || ((2 * i + 2) < size&&arr[i] > arr[2 * i + 2]))
		{
			flag_s = 0;//不是小顶堆
		}
	}
	if (flag_b == 1 && flag_s == 0)printf("big heap");
	else if (flag_b == 0 && flag_s == 1)printf("small heap");
	else printf("no heap");
}
//建立大顶堆
void adjustDown(int arr[], int k, int size)
{
	int i;
	arr[0] = arr[k];
	for (i = 2 * k; i <= size; i *= 2)
	{
		if (i < size&&arr[i] < arr[i + 1])i++;
		if (arr[0] >= arr[i])break;
		else
		{
			arr[k] = arr[i];
			k = i;
		}
	}
	arr[k] = arr[0];
}
//建立小顶堆
void adjustUp(int arr[], int k, int size)
{

}
void build_heap(int arr[], int size)
{
	int i;
	for (i = size / 2; i > 0; i--)
	{
		adjustDown(arr, i, size);
	}
}
void build_heap_test()
{
	int arr[5] = { 0,9,5,2,7 };
	int arr1[8] = { 87,45,78,32,17,65,53,9 };
	build_heap(arr, 4);
	judge_heap(arr1, 8);
}
