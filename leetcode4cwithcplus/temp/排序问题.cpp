#include<stdio.h>

//拓扑排序
void Topological_Sort()
{

}

//冒泡排序
int* sort_by_bubble(int arr[])
{
	int sz = sizeof(arr) / sizeof(arr[0]);//arr为首指针,此计算出的大小不对
	int i = 0, j = 0;
	for (i = 0; i<sz - 1; i++) {
		for (j = 0; j<sz - i - 1; j++) {
			if (arr[j]>arr[j + 1]) {
				int tmp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = tmp;
			}
		}
	}
	return arr;
}
void sort_by_bubble1(int *arr, int sz) {
	int i = 0, j = 0, temp;
	for (i = 0; i < sz - 1; i++) {
		for (j = 0; j < sz - i - 1; j++) {
			if (arr[j] > arr[j + 1]) {
				temp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = temp;
			}
		}
	}
}
void sort_by_bubble_test()
{
	int arr[8] = { 1,2,3,4,5,9,78,6 };
	int sz = sizeof(arr) / sizeof(arr[0]);
	//int * p = sort_by_bubble(arr);
	sort_by_bubble1(arr, sz);
	int i;
	for (i = 0; i < sz; i++)
	{
		printf("%d ",/**(p+i)*/arr[i]);
	}
}
