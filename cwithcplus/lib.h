#pragma once
#include "head.h"

//��������
int Create_List_Head(PNode h, ElementType data);

//չʾ����
void DisPlay(PNode h);

//��ȡ���鳤��
int Get_Length_Of_Array(int array[]);

#define GET_ARRAY_LEN(array,len){len = (sizeof(array) / sizeof(array[0]));}