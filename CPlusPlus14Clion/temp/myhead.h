#pragma once
#include<stdio.h>
#include<malloc.h>
#include<stdlib.h>

//��������,����ͷָ��,�������
typedef int ElementType;
typedef struct node
{
	ElementType data;
	struct node *next;
}Node, *PNode;
PNode create_head_1(ElementType data)
{
	PNode node = (PNode)malloc(sizeof(Node));
	if (node == NULL)
	{
		fprintf(stderr, "����ͷ���ʧ��");
		exit(1);		
	}
	node->data = data;
	node->next = NULL;
	return node;
}
PNode create_list_1(PNode head, int arr[], int type)
{
	//type==1��ʾ˳���,==2��ʾ�����
	//headͷ����е�������������������ĳ���
	int i;
	PNode p = head;
	for (i = 0; i < head->data; i++)
	{
		PNode tmp = (PNode)malloc(sizeof(Node));
		tmp->data = arr[i];
		tmp->next = NULL;
		if (type == 1)
		{
			p->next = tmp;
			p = tmp;
		}
		else if (type == 2)
		{
			tmp->next = p->next;
			p->next = tmp;
		}
	}
	return head;
}
void output_list_1(PNode head)
{
	int i;
	PNode p = head->next;
	for (i = 0; i < head->data&&p; i++)
	{
		printf("%d ", p->data);
		p = p->next;
	}
}
void create_list_test()
{
	int arr[5] = { 9,5,2,7,4 };
	PNode head = create_head_1(5);
	PNode p = create_list_1(head, arr, 1);
	output_list_1(p);
	PNode q = create_list_1(head, arr, 2);
	output_list_1(q);
}