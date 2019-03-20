#include "stdio.h" 

#include <malloc.h>
#include <time.h>
#ifndef __HEAD_H__
#define __HEAD_H__
#define OK            0  
#define ERROR        -1  
#define MALLOC_ERROR -2   

typedef int ElementType;
typedef struct ListNode
{
	ElementType data;
	struct ListNode *next;
}Node;
typedef Node *PNode;

#endif