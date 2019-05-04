#include<stdio.h>
#include<malloc.h>

/*
    问题：从一个数中删除n位，使删除后的数字是最大结果，如325，删除1位，结果为35
    思路：
      1.数字高位是影响数字整体大小的关键，从数字高位往低位遍历，删除第一个比后面一位数小的数字；
      2.对该数遍历删除n次，若没有比后面小的数则删除最后一位。
 */

struct Node{
	char val;
	Node *next;
};

int main()
{
	int del_num;
	bool flag;
	char *num_str;
	Node *num_List, *p, *q;

	while(scanf("%s\n%d", num_str, &del_num)!=EOF){
		for(int j=0; num_str[j]!='\0'; j++){
			p = (Node *)malloc(sizeof(Node *));
			p->val = num_str[j];
			p->next = NULL;
			if(j==0){
				num_List = p;
				q = p;
			}else{
				q->next = p;
				q = q->next;
			}
		}
		
		for(int i=0; i<del_num && num_List->next; i++){
			p = num_List;
			q = p->next;
			if(p->val < q->val){
				// 1st node smaller than the 2nd one
				free(num_List);
				num_List = q;
				continue;
			}
			flag = true;
			while(q->next){
				if(q->val < q->next->val){
					flag = false;
					p->next = q->next;
					free(q);
					break;
				}
				p = p->next;
				q = q->next;
			}
			if(flag){
				p->next = NULL;
				free(q);
			}
				
		}
		for(p=num_List; p!=NULL; ){
			printf("%c", p->val);
			q = p;
			p = p->next;
			free(q);
		}
		printf("\n");
	}
}
