#pragma once
#include "head.h"

//创建链表
int Create_List_Head(PNode h, ElementType data);

//展示链表
void DisPlay(PNode h);

//获取数组长度
int Get_Length_Of_Array(int array[]);

#define GET_ARRAY_LEN(array,len){len = (sizeof(array) / sizeof(array[0]));}