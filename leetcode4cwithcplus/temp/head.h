#pragma once
#include "stdio.h" 
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <memory.h>
#include <math.h>
#include <stdlib.h>
#ifndef __HEAD_H__
#define __HEAD_H__
#define OK            0  
#define ERROR        -1  
#define MALLOC_ERROR -2   
#define GET_ARRAY_LEN(array,len){len = (sizeof(array) / sizeof(array[0]));}

typedef int ElementType;
typedef struct ListNode
{
	ElementType data;
	struct ListNode *next;
}Node;
typedef Node *PNode;

#pragma region Definition

int* twoSum(int* nums, int numsSize, int target);
struct ListNode* addTwoNumbers(struct ListNode* l1, struct ListNode* l2);
int lengthOfLongestSubstring(char* s);
void spiralArray(int n);
void personalRate();
void horseChess();
void horseChess1();
void nlr();
void nlr1();
void bracketsMatch();
void primeNumber();
#pragma endregion


#pragma region Lib

//��������
int Create_List_Head(PNode h, ElementType data);
//չʾ����
void DisPlay(PNode h);
//��ȡ���鳤��
int Get_Length_Of_Array(int array[]);

#pragma endregion

#endif