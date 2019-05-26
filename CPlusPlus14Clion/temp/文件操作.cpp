#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<string.h>

//文件读取写入练习
/*
设计一个程序，从in.txt读入数据，对每一行的数字都单独按从大到小的顺序排序，将结果输出到out.txt。每一行的数字第一个字符是数字标志，每个数字之间用空格隔开。如：
读入：5 10 -20 2000 36 -100
输出：2000 36 10 -20 -100（注意5只是个开始标志）。
*/
void sortFileNumAndOutput()
{
	char src[100] = { '\0' };
	int i, j, t, num, nums[100];
	FILE* in, *out;
	in = fopen("in.txt", "r");
	out = fopen("out.txt", "w+");
	if (in == NULL)
	{
		printf("file does not exist");
		fprintf(stderr, "file does not exist");
	}
	while (!feof(in))
	{
		fscanf(in, "%d", &num);
		for (i = 0; i < num; i++)
		{
			fscanf(in, "%d", &nums[i]);
		}
		for (i = 0; i < num - 1; i++)
		{
			for (j = 0; j < num - i - 1; j++)
			{
				if (nums[j] < nums[j + 1])
				{
					t = nums[j];
					nums[j] = nums[j + 1];
					nums[j + 1] = t;
				}
			}
		}
		int k = 0;
		for (; k < num; k++)
		{
			char temp[10];
			sprintf(temp, "%d ", nums[k]);//格式化字符串
			strcat(src, temp);
		}
		strcat(src, "\n");
	}
	fwrite(src, sizeof(src), 1, out);
}
