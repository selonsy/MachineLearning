#include "lib.h"
//��������(����)
int Create_List_Head(PNode h, ElementType data)
{
	if (h == NULL)
	{
		return ERROR;
	}

	PNode node = (PNode)malloc(sizeof(Node) / sizeof(char));
	if (node == NULL)
	{
		return MALLOC_ERROR;
	}
	node->data = data;
	node->next = h->next;
	h->next = node;

	return OK;
}

//չʾ����
void DisPlay(PNode h)
{
	if (h == NULL)
	{
		return;
	}
	PNode temp = h->next;  // �����һ�����ָ��  
	while (temp)
	{
		printf("%4d", temp->data);
		temp = temp->next;
	}

	printf("\n");
}

//��ȡ���鳤��
int Get_Length_Of_Array(int array[])
{
	int count = 0;
	for (int i = 0; array[i]!=NULL; i++)
	{
		count++;
	}
	return count;
	//return (sizeof(array) / sizeof(array[0]));
}
