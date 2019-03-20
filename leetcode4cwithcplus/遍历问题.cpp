#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<string.h>

/*
4.��֪��������ǰ������������������εõ����ĺ������
N->���ڵ�
L->������
R->������
ǰ�����    N��>L��>R
�������    L��>N��>R
�������    L��>R��>N

Ҫ������������������Ҫ�������¼������ԣ�
����A������ǰ���������һ���϶��Ǹ��ڵ㣻
����B�����ں�����������һ���϶��Ǹ��ڵ㣻
����C������ǰ�����������ȷ�����ڵ㣬����������У����ڵ�����߾Ϳ��Էֳ�����������������
����D�������������������ֱ���ǰ��3��ķ����Ͳ�֣��൱�����ݹ飬���ǾͿ����ؽ��������Ķ�������

������һ��������һ��������̣����裺
ǰ�������˳����: CABGHEDF
���������˳����: GHBACDEF
��һ�������Ǹ�������A�����Ե�֪���ڵ���C��Ȼ�󣬸�������C������֪���������ǣ�GHBA���������ǣ�DEF��
C
/ \
GHBA  DEF
�ڶ�����ȡ������������������ǰ������ǣ�ABGH����������ǣ�GHBA����������A��C���ó��������ĸ��ڵ���A������Aû����������
C
/ \
A   DEF
/
GBH
��������ʹ��ͬ���ķ�����ǰ����BGH��������GHB���ó����ڵ���B��GH��B�ڵ��������������Bû����������
���Ĳ���ʹ��ͬ���ķ�����ǰ����GH��������GH���ó����ڵ���G��HΪG��������������û����������
C
/ \
A   DEF
/
B
/
G
\
H
���岽���ص�������������ǰ����EDF��������DEF����Ȼ��������A��C���ó����ڵ���E�����ҽڵ���D��F��
C
/ \
A   E
/   / \
B   D   F
/
G
\
H
���ˣ����ǵõ�����������Ķ���������ˣ����ĺ���������ǣ�HGBADFEC��
*/

#define N 100
char pre[N], in[N], res[N];
void Find_Last(int p1, int p2, int q1, int q2, int root)
{
	if (p1 > p2) return;
	for (root = q1; in[root] != pre[p1]; ++root);
	Find_Last(p1 + 1, p1 + root - q1, q1, root - 1, 0);
	Find_Last(p1 + root + 1 - q1, p2, root + 1, q2, 0);
	printf("%c", in[root]);
}
void Find_Last1(int n, char* pre, char* in, char* res)//�ݹ鹹��,������������������������������  
{
	if (n <= 0) return;
	int p = strchr(in, pre[0]) - in;//�ҵ����ڵ������������λ��  
	Find_Last1(p, pre + 1, in, res);//�ݹ鹹���������ĺ������  
	Find_Last1(n - p - 1, pre + p + 1, in + p + 1, res + p);//�ݹ鹹���������ĺ������  
	res[n - 1] = pre[0];//��Ӹ��ڵ㵽�����  
}
void nlr()
{
	while (scanf("%s%s", pre, in) == 2)
	{
		int len = strlen(pre) - 1;
		Find_Last(0, len, 0, len, 0);
		puts("");
	}
	return;
}
void nlr1()//this one is better.
{
	while (scanf("%s%s", pre, in) == 2)
	{
		int n = strlen(in);
		Find_Last1(n, pre, in, res);
		res[n] = '\0';	//�ַ�����\0��β
		printf("%s\n", res);
	} 
	return;
}

