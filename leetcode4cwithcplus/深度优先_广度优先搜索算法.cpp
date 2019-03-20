//#define _CRT_SECURE_NO_WARNINGS
//using namespace std;
//
//#include<stdio.h>
//#include<malloc.h>
//
//#include<iostream>
//#include<string>
//#include<queue>
//
//#define MaxVerNum 100 /*定义最大节点数*/
//int visited[MaxVerNum];
//typedef char VertexType;
//typedef struct node
//{
//	int adjvex;
//	struct node *next; //指向下一个邻接节点域
//} EdgeNode;
//typedef struct vnode
//{
//	VertexType vertex[3]; //顶点域
//	EdgeNode *firstedge; //边表头指针
//} VertexNode;
//typedef VertexNode AdjList[MaxVerNum];
///*
//*定义以邻接边为存储类型的图
//*/
//typedef struct
//{
//	AdjList adjList; //邻接表
//	int n, e; //顶点数与边数
//} ALGraph;
///*
//*建立队列的数据结构方便进行广度优先遍历
//*/
//typedef struct
//{
//	int data[MaxVerNum];
//	int head, tail; //队头与队尾
//} Quene;
///*
//*建立无向图的邻接表存储
//*/
//void CreateALGraph(ALGraph *G)
//{
//	int i, j, k;
//	EdgeNode *s;
//	printf("请输入顶点数与边数（输入格式为：顶点数，边数）： ");
//	scanf("%d,%d", &G->n, &G->e);
//	printf("请输入顶点信息（输入格式为：顶点号<CR>）：\n");
//	for (i = 0; i < G->n; i++)
//	{
//		scanf("%s", G->adjList[i].vertex);
//		G->adjList[i].firstedge = NULL; //将顶点的边表头指针设置为空
//	}
//	printf("请输入边的信息（输入格式为：i,j）：\n");
//	for (k = 0; k < G->e; k++)
//	{
//		scanf("%d,%d", &i, &j);
//		s = (VertexNode*)malloc(sizeof(VertexNode));
//		//边上的第一个节点
//		s->adjvex = j;
//		s->next = G->adjList[i].firstedge;
//		G->adjList[i].firstedge = s;
//		//边上的第二个节点
//		s = (VertexNode*)malloc(sizeof(VertexNode));
//		s->adjvex = i;
//		s->next = G->adjList[j].firstedge;
//		G->adjList[j].firstedge = s;
//	}
//}
///*
//*进行图的深度优先搜索遍历
//*/
//void DFSAL(ALGraph *G, int i)
//{
//	//以Vi为出发点对图进行遍历
//	EdgeNode *p;
//	printf("visit vertex : %s \n", G->adjList[i].vertex);
//	visited[i] = 1;
//	p = G->adjList[i].firstedge;
//	while (p)
//	{
//		if (!visited[p->adjvex])
//		{
//			DFSAL(G, p->adjvex);
//		}
//		p = p->next;
//	}
//}
//void DFSTraverseAL(ALGraph *G)
//{
//	int i;
//	for (i = 0; i < G->n; i++)
//	{
//		visited[i] = 0;
//	}
//	for (i = 0; i < G->n; i++)
//	{
//		if (!visited[i])
//		{
//			DFSAL(G, i);
//		}
//	}
//}
///*
//*进行广度优先搜索遍历
//*/
//void BFSG(ALGraph *G, int k)
//{
//	int i, j;
//	Quene q;
//	EdgeNode *p;
//	q.head = 0;
//	q.tail = 0; //进行队列的初始化
//	printf("visit vertex : %s \n", G->adjList[k].vertex);
//	visited[k] = 1;
//	q.data[q.tail++] = k;
//	while (q.head % (MaxVerNum - 1) != q.tail % (MaxVerNum - 1))
//	{
//		i = q.data[q.head++];
//		p = G->adjList[i].firstedge;
//		while (p)
//		{
//			if (!visited[p->adjvex])
//			{
//				printf("visit vertex : %s \n", G->adjList[p->adjvex].vertex);
//				visited[p->adjvex] = 1;
//				q.data[q.tail++] = p->adjvex;
//			}
//			p = p->next;
//		}
//	}
//}
//void BFSTraverseAL(ALGraph *G)
//{
//	int i;
//	for (i = 0; i < G->n; i++)
//	{
//		visited[i] = 0;
//	}
//	for (i = 0; i < G->n; i++)
//	{
//		if (!visited[i])
//		{
//			BFSG(G, i);
//		}
//	}
//}
///*
//*进行图的测试
//*/
//void BFS_DFS_Test()
//{
//	ALGraph *G;
//	EdgeNode *p = NULL;
//	int i;
//	G = (ALGraph*)malloc(sizeof(ALGraph));
//	CreateALGraph(G);
//	printf("进行深度优先遍历\n");
//	DFSTraverseAL(G);
//	printf("进行广度优先遍历\n");
//	BFSTraverseAL(G);
//}
//
//
///*
//Description
//对于二叉树T，可以递归定义它的先序遍历、中序遍历和后序遍历如下： PreOrder(T)=T的根节点+PreOrder(T的左子树)+PreOrder(T的右子树) InOrder(T)=InOrder(T的左子树)+T的根节点+InOrder(T的右子树) PostOrder(T)=PostOrder(T的左子树)+PostOrder(T的右子树)+T的根节点 其中加号表示字符串连接运算。例如，对下图所示的二叉树，先序遍历为DBACEGF，中序遍历为ABCDEFG。
//输入一棵二叉树的先序遍历序列和中序遍历序列，输出它的广度优先遍历序列。
//
//Input
//第一行为一个整数t（0<t<10），表示测试用例个数。 以下t行，每行输入一个测试用例，包含两个字符序列s1和s2，其中s1为一棵二叉树的先序遍历序列，s2为中序遍历序列。s1和s2之间用一个空格分隔。序列只包含大写字母，并且每个字母最多只会出现一次。
//
//Output
//为每个测试用例单独一行输出广度优先遍历序列。
//
//Sample Input
//Copy sample input to clipboard
//2
//DBACEGF ABCDEFG
//BCAD CBAD
//Sample Output
//DBEACGF
//BCAD
//*/
//struct BitNode
//{
//	char c;
//	BitNode* lchild;
//	BitNode* rchild;
//};
////先重建二叉树
//BitNode* rebuild(string pre, string in)
//{
//	BitNode* T = NULL;
//	if (pre.length()>0)
//	{
//		//前序遍历首元素为根结点
//		T = new BitNode();
//		T->c = pre[0];
//		T->lchild = NULL;
//		T->rchild = NULL;
//	}
//	if (pre.length()>1)
//	{
//		//find the position of root in inorder
//		int root = 0;
//		for (; in[root] != pre[0]; root++);
//
//		//recrusive
//		T->lchild = rebuild(pre.substr(1, root), in.substr(0, root));
//		T->rchild = rebuild(pre.substr(root + 1, pre.length() - 1 - root), in.substr(root + 1, in.length() - 1 - root));
//	}
//
//	return T;
//}
////访问函数
//void visit(BitNode* T)
//{
//	cout << T->c;
//}
////广度优先遍历
//void BFS(BitNode* T)
//{
//	//用一个队列储存已访问结点
//	queue<BitNode*> q;
//	q.push(T);
//
//	while (!q.empty())
//	{
//		BitNode* t1 = q.front();
//		q.pop();
//		visit(t1);
//		if (t1->lchild)
//			q.push(t1->lchild);
//		if (t1->rchild)
//			q.push(t1->rchild);
//	}
//	cout << endl;
//}
//int BFE_TEST()
//{
//	string pre, in;
//	cin >> pre >> in;
//
//	BFS(rebuild(pre, in));
//
//	return 0;
//}