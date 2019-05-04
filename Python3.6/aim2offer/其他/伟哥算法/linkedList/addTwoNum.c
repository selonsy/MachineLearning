#include<stdio.h>
#include<malloc.h>

/*
  问题: 使用链表做两个数的加法，如：342+465=807用链表进行计算时为 2->4->3 + 5->6->4 = 7->0->8
  思路: 1.对链表进行按位加，相加结果余10为当前位值，除10为进位值，每位加得结果直接覆盖第一个链表的对应位；
        2.按照L2长度同时遍历两条链表，链表L1比L2长时，需要对L1剩余的部分进行进位处理；
        3.L1比L2短时，每读取一位L2的数值，则为L1创建新的尾节点，并赋值为0，继续往下按位叠加；
 */

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    if(l1==NULL || l2==NULL){
        return NULL;
    }
    ListNode *p, *q;
    p = l1;
    q = l2;
    int temp = 0;
    while(q){
        temp += p->val + q->val;
        p->val = temp%10;
        temp /= 10;
        q = q->next;
        if((q||temp>0)&&p->next == NULL){
            p->next = (ListNode *)malloc(sizeof(ListNode *));
            p->next->val = 0;
            p->next->next = NULL;
        }
        p = p->next;
    }
    while(temp>0){
        temp += p->val;
        p->val = temp%10;
        temp /= 10;
        if(temp>0&&p->next==NULL){
            p->next = (ListNode *)malloc(sizeof(ListNode *));
            p->next->val = 0;
            p->next->next = NULL;
        }
        p = p->next;
    }
    return l1;
}
ListNode* createList(int *arr, int len)
{
    // 头插法

    ListNode *head, *p;
    head = NULL;
    for(int i=len-1; i>=0; i--){
        p = (ListNode *)malloc(sizeof(ListNode*));
        p->val = arr[i];
        p->next = head;
        head = p;
    }
    return head;
}
void traverseLink(ListNode *list)
{
    ListNode *p;
    p = list;
    if(p==NULL)
        return ;

    printf("%d",p->val);
    p = p->next;
    while(p){
        printf("->%d",p->val);
        p = p->next;
    }
    printf("\n");
}

int main()
{
    const int N=6, M=3;
    int l1_arr[N] = {2,4,9,9,9,9}; 
    int l2_arr[M] = {5,6,4};
    ListNode *l1, *l2;    

    l1 = createList(l1_arr, N);
    l2 = createList(l2_arr, M);

    traverseLink(l1);
    printf("+\n");
    traverseLink(l2);

    l1 = addTwoNumbers(l1, l2);
    printf("=\n");    
    traverseLink(l1);

    return 1;
}
