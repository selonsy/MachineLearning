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
//#define MaxVerNum 100 /*�������ڵ���*/
//int visited[MaxVerNum];
//typedef char VertexType;
//typedef struct node
//{
//	int adjvex;
//	struct node *next; //ָ����һ���ڽӽڵ���
//} EdgeNode;
//typedef struct vnode
//{
//	VertexType vertex[3]; //������
//	EdgeNode *firstedge; //�߱�ͷָ��
//} VertexNode;
//typedef VertexNode AdjList[MaxVerNum];
///*
//*�������ڽӱ�Ϊ�洢���͵�ͼ
//*/
//typedef struct
//{
//	AdjList adjList; //�ڽӱ�
//	int n, e; //�����������
//} ALGraph;
///*
//*�������е����ݽṹ������й�����ȱ���
//*/
//typedef struct
//{
//	int data[MaxVerNum];
//	int head, tail; //��ͷ���β
//} Quene;
///*
//*��������ͼ���ڽӱ�洢
//*/
//void CreateALGraph(ALGraph *G)
//{
//	int i, j, k;
//	EdgeNode *s;
//	printf("�����붥����������������ʽΪ������������������ ");
//	scanf("%d,%d", &G->n, &G->e);
//	printf("�����붥����Ϣ�������ʽΪ�������<CR>����\n");
//	for (i = 0; i < G->n; i++)
//	{
//		scanf("%s", G->adjList[i].vertex);
//		G->adjList[i].firstedge = NULL; //������ı߱�ͷָ������Ϊ��
//	}
//	printf("������ߵ���Ϣ�������ʽΪ��i,j����\n");
//	for (k = 0; k < G->e; k++)
//	{
//		scanf("%d,%d", &i, &j);
//		s = (VertexNode*)malloc(sizeof(VertexNode));
//		//���ϵĵ�һ���ڵ�
//		s->adjvex = j;
//		s->next = G->adjList[i].firstedge;
//		G->adjList[i].firstedge = s;
//		//���ϵĵڶ����ڵ�
//		s = (VertexNode*)malloc(sizeof(VertexNode));
//		s->adjvex = i;
//		s->next = G->adjList[j].firstedge;
//		G->adjList[j].firstedge = s;
//	}
//}
///*
//*����ͼ�����������������
//*/
//void DFSAL(ALGraph *G, int i)
//{
//	//��ViΪ�������ͼ���б���
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
//*���й��������������
//*/
//void BFSG(ALGraph *G, int k)
//{
//	int i, j;
//	Quene q;
//	EdgeNode *p;
//	q.head = 0;
//	q.tail = 0; //���ж��еĳ�ʼ��
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
//*����ͼ�Ĳ���
//*/
//void BFS_DFS_Test()
//{
//	ALGraph *G;
//	EdgeNode *p = NULL;
//	int i;
//	G = (ALGraph*)malloc(sizeof(ALGraph));
//	CreateALGraph(G);
//	printf("����������ȱ���\n");
//	DFSTraverseAL(G);
//	printf("���й�����ȱ���\n");
//	BFSTraverseAL(G);
//}
//
//
///*
//Description
//���ڶ�����T�����Եݹ鶨�����������������������ͺ���������£� PreOrder(T)=T�ĸ��ڵ�+PreOrder(T��������)+PreOrder(T��������) InOrder(T)=InOrder(T��������)+T�ĸ��ڵ�+InOrder(T��������) PostOrder(T)=PostOrder(T��������)+PostOrder(T��������)+T�ĸ��ڵ� ���мӺű�ʾ�ַ����������㡣���磬����ͼ��ʾ�Ķ��������������ΪDBACEGF���������ΪABCDEFG��
//����һ�ö�����������������к�����������У�������Ĺ�����ȱ������С�
//
//Input
//��һ��Ϊһ������t��0<t<10������ʾ�������������� ����t�У�ÿ������һ���������������������ַ�����s1��s2������s1Ϊһ�ö�����������������У�s2Ϊ����������С�s1��s2֮����һ���ո�ָ�������ֻ������д��ĸ������ÿ����ĸ���ֻ�����һ�Ρ�
//
//Output
//Ϊÿ��������������һ�����������ȱ������С�
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
////���ؽ�������
//BitNode* rebuild(string pre, string in)
//{
//	BitNode* T = NULL;
//	if (pre.length()>0)
//	{
//		//ǰ�������Ԫ��Ϊ�����
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
////���ʺ���
//void visit(BitNode* T)
//{
//	cout << T->c;
//}
////������ȱ���
//void BFS(BitNode* T)
//{
//	//��һ�����д����ѷ��ʽ��
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