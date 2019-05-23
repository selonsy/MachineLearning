#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<string.h>

/*
4.已知二叉树的前序遍历和中序遍历，如何得到它的后序遍历
N->根节点
L->左子树
R->右子树
前序遍历    N－>L－>R
中序遍历    L－>N－>R
后序遍历    L－>R－>N

要完成这个任务，我们首先要利用以下几个特性：
特性A，对于前序遍历，第一个肯定是根节点；
特性B，对于后序遍历，最后一个肯定是根节点；
特性C，利用前序或后序遍历，确定根节点，在中序遍历中，根节点的两边就可以分出左子树和右子树；
特性D，对左子树和右子树分别做前面3点的分析和拆分，相当于做递归，我们就可以重建出完整的二叉树；

我们以一个例子做一下这个过程，假设：
前序遍历的顺序是: CABGHEDF
中序遍历的顺序是: GHBACDEF
第一步，我们根据特性A，可以得知根节点是C，然后，根据特性C，我们知道左子树是：GHBA，右子树是：DEF。
C
/ \
GHBA  DEF
第二步，取出左子树，左子树的前序遍历是：ABGH，中序遍历是：GHBA，根据特性A和C，得出左子树的父节点是A，并且A没有右子树。
C
/ \
A   DEF
/
GBH
第三步，使用同样的方法，前序是BGH，中序是GHB，得出父节点是B，GH是B节点的左子树，并且B没有右子树。
第四步，使用同样的方法，前序是GH，中序是GH，得出父节点是G，H为G的右子树，并且没有左子树。
C
/ \
A   DEF
/
B
/
G
\
H
第五步，回到右子树，它的前序是EDF，中序是DEF，依然根据特性A和C，得出父节点是E，左右节点是D和F。
C
/ \
A   E
/   / \
B   D   F
/
G
\
H
到此，我们得到了这棵完整的二叉树，因此，它的后序遍历就是：HGBADFEC。
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
void Find_Last1(int n, char* pre, char* in, char* res)//递归构造,输入先序遍历，中序遍历，求后序遍历  
{
	if (n <= 0) return;
	int p = strchr(in, pre[0]) - in;//找到根节点在中序遍历的位置  
	Find_Last1(p, pre + 1, in, res);//递归构造左子树的后序遍历  
	Find_Last1(n - p - 1, pre + p + 1, in + p + 1, res + p);//递归构造右子树的后序遍历  
	res[n - 1] = pre[0];//添加根节点到最后面  
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
		res[n] = '\0';	//字符串以\0结尾
		printf("%s\n", res);
	} 
	return;
}

