#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<string.h>

//�ļ���ȡд����ϰ
/*
���һ�����򣬴�in.txt�������ݣ���ÿһ�е����ֶ��������Ӵ�С��˳�����򣬽���������out.txt��ÿһ�е����ֵ�һ���ַ������ֱ�־��ÿ������֮���ÿո�������磺
���룺5 10 -20 2000 36 -100
�����2000 36 10 -20 -100��ע��5ֻ�Ǹ���ʼ��־����
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
			sprintf(temp, "%d ", nums[k]);//��ʽ���ַ���
			strcat(src, temp);
		}
		strcat(src, "\n");
	}
	fwrite(src, sizeof(src), 1, out);
}
